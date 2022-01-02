import os
import sys
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from smarts.env.wrappers.parallel_env import ParallelEnv
from torch import nn
from torch.distributions import Normal
from tqdm import tqdm

from prepare_all_in_one_86 import *

if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"

CORE_NUM = 12
FEATURE_VECTOR_LENGTH = 86

"""
模型定义部分
"""


class NetCritic(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(NetCritic, self).__init__()
        layers = []
        last_size = input_size
        for i in range(len(hidden_size)):
            layers.append(torch.nn.utils.spectral_norm(torch.nn.Linear(last_size, hidden_size[i])))
            layers.append(torch.nn.Tanh())
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)
    
    def forward(self, inputs):
        # inputs += torch.normal(0, self.noise, size=inputs.shape, device=device)
        res = self._net(inputs)
        res = torch.sigmoid(res)
        return res


class NetAgent(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(NetAgent, self).__init__()
        layers = []
        last_size = input_size
        for i in range(len(hidden_size)):
            layers.append(torch.nn.Linear(last_size, hidden_size[i]))
            layers.append(torch.nn.Tanh())
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)
    
    def forward(self, inputs):
        res = self._net(inputs)
        return res


class MLPNet(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(MLPNet, self).__init__()
        layers = []
        last_size = input_size
        for i in range(len(hidden_size)):
            layers.append(torch.nn.utils.spectral_norm(torch.nn.Linear(last_size, hidden_size[i])))
            layers.append(torch.nn.LeakyReLU())
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)
    
    def forward(self, inputs):
        res = self._net(inputs)
        return res


class GRUNet(torch.nn.Module):
    def __init__(self, hidden_size, input_size, output_size):
        super(GRUNet, self).__init__()
        layers = []
        last_size = input_size
        for i in range(len(hidden_size)):
            layers.append(torch.nn.GRU(last_size, hidden_size[i]))
            last_size = hidden_size[i]
        layers.append(torch.nn.Linear(last_size, output_size))
        
        self._net = torch.nn.Sequential(*layers)
        self._net.to(device)
    
    def forward(self, inputs):
        inputs = inputs.view(len(inputs), 1, -1)
        res = self._net(inputs)
        return res


class PSGAIL():
    def __init__(self,
                 discriminator_lr=1e-4,
                 policy_lr=1e-4,
                 value_lr=1e-4,
                 hidden_size=[128, 256, 512, 256, 128],  # [64,128,64],
                 state_action_space=FEATURE_VECTOR_LENGTH + 2,
                 state_space=FEATURE_VECTOR_LENGTH,
                 action_space=4,
                 ):
        self._tau = 0.01
        self._clip_range = 0.1
        self.kl_target = 0.01
        self.beta = 0.5
        self.v_clip_range = 0.2
        self.klmax = 0.05
        self.dis_crit = nn.BCELoss()
        self.discriminator = NetCritic(hidden_size, input_size=state_action_space, output_size=1)
        self.discriminator = self.discriminator.to(device)
        self.value = NetAgent(hidden_size, input_size=state_space, output_size=1)
        self.value = self.value.to(device)
        self.policy = NetAgent(hidden_size, input_size=state_space, output_size=action_space)
        self.policy = self.policy.to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, weight_decay=0.001)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr, weight_decay=0.001)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr, weight_decay=0.001)
    
    def get_action(self, obs, action=None):
        policy_out = self.policy(obs)
        mean1, var1, mean2, var2 = torch.chunk(policy_out, 4, dim=1)
        mean1 = 3 * torch.tanh(mean1)
        mean2 = 0.3 * torch.tanh(mean2)
        var1 = torch.nn.functional.softplus(var1)
        var2 = torch.nn.functional.softplus(var2)
        act1 = Normal(mean1, var1)
        act2 = Normal(mean2, var2)
        if action is None:
            action1 = act1.sample()
            action2 = act2.sample()
            log_prob1 = act1.log_prob(action1)
            log_prob2 = act2.log_prob(action2)
            log_prob = log_prob1 + log_prob2
            action = torch.cat((action1, action2), dim=1)
        else:
            log_prob1 = act1.log_prob(action[:, 0].unsqueeze(1))
            log_prob2 = act2.log_prob(action[:, 1].unsqueeze(1))
            log_prob = log_prob1 + log_prob2
        return action, log_prob.reshape(-1, 1)
    
    def compute_adv(self, batch, gamma):
        s = batch["state"]
        s1 = batch["next_state"]
        reward = batch['agents_rew']
        done = batch["done"].reshape(-1, 1)
        batch['old_v'] = self.value(s)
        with torch.no_grad():
            td_target = reward + gamma * self.value(s1) * (1 - done)
            adv = td_target - batch['old_v']
        return adv, td_target
    
    def update_discriminator(self, sap_agents, sap_experts):
        sap_experts = torch.tensor(sap_experts, device=device, dtype=torch.float32)
        sap_agents = sap_agents.detach()
        disc_expert = self.discriminator(sap_experts)
        disc_agents = self.discriminator(sap_agents)
        experts_score = -torch.log(disc_expert)
        agents_score = -torch.log(1 - disc_agents)
        discriminator_loss = (agents_score + experts_score).mean()
        
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        
        return float(agents_score.mean().data), float(experts_score.mean().data), float(discriminator_loss)
    
    def update_generator(self, batch):
        state = batch["state"]
        action = batch["action"]
        old_log_prob = batch["log_prob"]
        adv = batch["adv"]
        td_target = batch['td_target']
        
        act, log_prob = self.get_action(state, action)
        old_log_prob = old_log_prob.detach()
        old_log_prob = old_log_prob.unsqueeze(1)
        ip_sp = torch.exp(log_prob - old_log_prob)
        ip_sp_clip = torch.clamp(ip_sp, 1 - self._clip_range, 1 + self._clip_range)
        cur_prob = torch.exp(log_prob)
        
        policy_loss = -torch.mean(torch.min(ip_sp * adv.detach(), ip_sp_clip * adv.detach()))
        value_loss = torch.mean(F.mse_loss(self.value(state), td_target.detach()))
        kl_div = torch.nn.functional.kl_div(old_log_prob, cur_prob)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return float(policy_loss.data), float(value_loss.data), float(kl_div.data)


"""
工具函数部分
"""


class Logger(object):
    def __init__(self, file_name='default.log', stream=sys.stdout):
        self.terminal = stream
        self.file_name = file_name
        self.log = open(self.file_name, 'a')
        self.log.close()
    
    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.file_name, 'a')
        self.log.write(message)
        self.log.close()
    
    def flush(self):
        pass


def plot_infos(output_dir, stage, infos):
    for key in tqdm(infos):
        if not key in ['epochs', 'survival_time_list', 'final_position_list']:
            plt.ylabel(key)
            plt.xlabel('epoch')
            x, y = infos['epochs'], infos[key]
            plt.plot(x, y)
            plt.savefig(output_dir + 'figures/' + stage + '_' + key + '.pdf')
            plt.close()
        elif key in ['survival_time_list', 'final_position_list']:
            plt.ylabel(key.replace('_list', ''))
            plt.xlabel('epoch')
            min_list = []
            mean_list = []
            max_list = []
            point_x_list = []
            point_y_list = []
            for i, x in tqdm(enumerate(infos['epochs'])):
                ys = infos[key][i]
                for y in ys:
                    point_x_list.append(x)
                    point_y_list.append(y)
                min_list.append(np.min(ys))
                mean_list.append(np.mean(ys))
                max_list.append(np.max(ys))
            
            plt.scatter(point_x_list, point_y_list, s=0.1, c='grey')
            plt.plot(infos['epochs'], min_list, linewidth=0.5, label='minimum survival time')
            plt.plot(infos['epochs'], mean_list, linewidth=0.5, label='average survival time')
            plt.plot(infos['epochs'], max_list, linewidth=0.5, label='maximum survivial time')
            plt.legend()
            plt.savefig(output_dir + 'figures/' + stage + '_' + key + '.pdf')
            plt.close()


def getlist(list_, i):
    if i < 0 or i >= len(list_) or len(list_) == 0:
        return None
    else:
        return list_[i]


class trajectory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []
        self.steps = 0


class samples_agents():
    def __init__(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.probs = []


def dump_trajectory(expert_trajectory, agent_id, batch_samples, ends, counter, done_agents_steps, done_agents_steps_list, final_xs, final_xs_list):
    batch_samples.states += expert_trajectory[agent_id].states
    batch_samples.probs += expert_trajectory[agent_id].probs
    batch_samples.actions += expert_trajectory[agent_id].actions
    batch_samples.next_states += expert_trajectory[agent_id].next_states
    batch_samples.rewards += expert_trajectory[agent_id].rewards
    batch_samples.dones += expert_trajectory[agent_id].dones
    counter += len(expert_trajectory[agent_id].states)
    done_agents_steps += (len(expert_trajectory[agent_id].states) + expert_trajectory[agent_id].steps)
    done_agents_steps_list.append(len(expert_trajectory[agent_id].states) + expert_trajectory[agent_id].steps)
    if expert_trajectory[agent_id].states[-1][0] > 300:
        ends += 1
    final_xs += expert_trajectory[agent_id].states[-1][0]
    final_xs_list.append(expert_trajectory[agent_id].states[-1][0])
    return ends, counter, done_agents_steps, done_agents_steps_list, final_xs, final_xs_list


def dump_all(expert_trajectory, agent_traj):
    for env in expert_trajectory.keys():
        for agent_id in expert_trajectory[env].keys():
            agent_traj.states += expert_trajectory[env][agent_id].states
            agent_traj.probs += expert_trajectory[env][agent_id].probs
            agent_traj.actions += expert_trajectory[env][agent_id].actions
            agent_traj.next_states += expert_trajectory[env][agent_id].next_states
            agent_traj.rewards += expert_trajectory[env][agent_id].rewards
            agent_traj.dones += expert_trajectory[env][agent_id].dones
            steps = len(expert_trajectory[env][agent_id].states) + expert_trajectory[env][agent_id].steps
            expert_trajectory[env][agent_id] = trajectory()
            expert_trajectory[env][agent_id].steps = steps


def trans2tensor(batch):
    for k in batch:
        batch[k] = torch.tensor(batch[k], device=device, dtype=torch.float32)
    return batch


def sampling(psgail, vector_env, batch_size, vec_obs, vec_done, expert_trajectory):
    total_agent_num = 0
    agent_traj = samples_agents()
    counter = 0
    ends = 0
    final_xs = 0
    done_agents_steps = 0
    done_agents_steps_list = []
    final_xs_list = []
    while True:
        vec_act = []
        obs_vectors_list = np.zeros((1, FEATURE_VECTOR_LENGTH))
        
        for i, obs in enumerate(vec_obs):
            for agent_id in obs.keys():
                if getlist(vec_done, i) is None or not vec_done[i].get(agent_id):
                    if agent_id not in expert_trajectory[i]:
                        expert_trajectory[i][agent_id] = trajectory()
                    
                    obs_vectors_list = np.vstack((obs_vectors_list, obs[agent_id]['feature_vector']))
        
        obs_vectors = torch.tensor(obs_vectors_list[1:, :], device=device, dtype=torch.float32)
        acts, prob = psgail.get_action(obs_vectors)
        act_i = 0
        
        for i, obs in enumerate(vec_obs):
            act_n = {}
            for agent_id in obs.keys():
                if getlist(vec_done, i) is None or not vec_done[i].get(agent_id):
                    act_tmp = acts[act_i].cpu()
                    act_n[agent_id] = act_tmp.numpy()
                    expert_trajectory[i][agent_id].states.append(obs_vectors_list[act_i + 1])
                    expert_trajectory[i][agent_id].probs.append(prob[act_i])
                    expert_trajectory[i][agent_id].actions.append(act_n[agent_id])
                    act_i += 1
            
            vec_act.append(act_n)
        
        vec_obs, vec_rew, vec_done, vec_info = vector_env.step(vec_act)
        
        for i, act_n in enumerate(vec_act):
            for agent_id in act_n.keys():
                expert_trajectory[i][agent_id].rewards.append(vec_rew[i].get(agent_id))
                expert_trajectory[i][agent_id].dones.append(vec_done[i].get(agent_id))
                if vec_done[i].get(agent_id):
                    expert_trajectory[i][agent_id].next_states.append(np.zeros(FEATURE_VECTOR_LENGTH))
                    ends, counter, done_agents_steps, done_agents_steps_list, final_xs, final_xs_list = dump_trajectory(expert_trajectory[i], agent_id, agent_traj,
                                                                                                                        ends, counter, done_agents_steps, done_agents_steps_list, final_xs, final_xs_list)
                    total_agent_num += 1
                    
                    del expert_trajectory[i][agent_id]
                else:
                    expert_trajectory[i][agent_id].next_states.append(vec_obs[i][agent_id]['feature_vector'].squeeze())
        
        if counter >= batch_size:
            dump_all(expert_trajectory, agent_traj)
            break
    
    return ends, total_agent_num, done_agents_steps, done_agents_steps_list, final_xs, final_xs_list, agent_traj.states, agent_traj.next_states, agent_traj.actions, agent_traj.probs, agent_traj.dones, agent_traj.rewards, vec_obs, vec_done, expert_trajectory


def _cal_angle(vec):
    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])
    
    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        if abs(rel_pos_vec[0]) > 60 or abs(rel_pos_vec[1]) > 15:
            continue
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    return groups


def load_model(model2test):
    with open(model2test, "rb") as f:
        models = pk.load(f)
    return models


"""
模型训练部分
"""


def train(psgail, output_dir, experts, method='GAIL', stage='train', num_epoch=1000, print_every=1, add_every=100, cal_every=10, save_every=100, gamma=0.95, batch_size=10000, agent_num=2, add_agent_num=2, mini_epoch=10,
          process_num=12):
    rewards_log = []
    avg_step_log = []
    step_log = []
    epochs_log = []
    kl_log = []
    dis_ag_rew = []
    dis_ex_rew = []
    dis_total_losses = []
    pol_losses = []
    val_losses = []
    done_rate = []
    avg_finals = []
    final_log = []
    
    cal_range = min(cal_every, print_every)
    env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
    vector_env = ParallelEnv([env_creator] * process_num, auto_reset=True)
    vec_obs = vector_env.reset()
    vec_done = []
    expert_trajectory = {}
    for i in range(process_num):
        expert_trajectory[i] = {}
    
    with tqdm(range(num_epoch), ascii=True) as pbar:
        for epoch in pbar:
            
            pbar.set_description('Method {} Stage {} Epoch {}'.format(method, stage, epoch))
            
            if epoch % add_every == 0 and epoch != 0:
                agent_num += add_agent_num
                vector_env.close()
                env_creator = lambda: MATrafficSim(["./ngsim"], agent_number=agent_num)
                vector_env = ParallelEnv([env_creator] * process_num, auto_reset=True)
                vec_obs = vector_env.reset()
                expert_trajectory = {}
                vec_done = []
                for i in range(process_num):
                    expert_trajectory[i] = {}
            
            dis_agent_buffer = []
            dis_expert_buffer = []
            dis_total_buffer = []
            pol_buffer = []
            val_buffer = []
            kl_buffer = []
            
            ends, total_agent_num, done_agents_steps, done_agents_steps_list, final_xs, final_xs_list, states, next_states, actions, log_probs, dones, rewards, vec_obs, vec_done, expert_trajectory = sampling(psgail, vector_env, batch_size,
                                                                                                                                                                                                                vec_obs,
                                                                                                                                                                                                                vec_done, expert_trajectory)
            
            rewards_log.append(np.sum(rewards) / total_agent_num)
            avg_step_log.append(done_agents_steps / total_agent_num)
            step_log.append(done_agents_steps_list)
            final_log.append(final_xs_list)
            done_rate.append(ends / total_agent_num)
            avg_finals.append(final_xs / total_agent_num)
            epochs_log.append(epoch)
            batch = trans2tensor({"state": states,
                                  "action": actions,
                                  "log_prob": log_probs,
                                  "next_state": next_states,
                                  "done": dones})
            sap_agents = torch.cat((batch["state"], batch["action"]), dim=1)
            sap_agents = sap_agents.detach()
            
            for i in range(mini_epoch):
                dis_rd_sample = np.random.randint(0, high=len(experts), size=len(sap_agents))
                cur_experts = experts[dis_rd_sample]
                dis_agent_tmp, dis_expert_tmp, dis_total_tmp = psgail.update_discriminator(sap_agents, cur_experts)
                dis_agent_buffer.append(dis_agent_tmp)
                dis_expert_buffer.append(dis_expert_tmp)
                dis_total_buffer.append(dis_total_tmp)
            dis_ag_rew.append(np.mean(dis_agent_buffer))
            dis_ex_rew.append(np.mean(dis_expert_buffer))
            dis_total_losses.append(np.mean(dis_total_buffer))
            disc_agents = psgail.discriminator(sap_agents)
            batch["agents_rew"] = -torch.log(1 - disc_agents.detach())
            batch['adv'], batch['td_target'] = psgail.compute_adv(batch, gamma)
            
            for i in range(mini_epoch):
                policy_tmp, value_tmp, kl_div = psgail.update_generator(batch)
                pol_buffer.append(policy_tmp)
                val_buffer.append(value_tmp)
                kl_buffer.append(kl_div)
                if kl_div > psgail.klmax:
                    break
            pol_losses.append(np.mean(pol_buffer))
            val_losses.append(np.mean(val_buffer))
            kl_log.append(np.mean(kl_buffer))
            
            if (epoch + 1) % save_every == 0 or epoch + 1 == num_epoch:
                if epoch + 1 == num_epoch:
                    checkpoint_path = output_dir + 'models/final_checkpoint_' + method + '_' + str(int(epoch + 1)) + '_' + stage + '.model'
                else:
                    checkpoint_path = output_dir + 'models/checkpoint_' + method + '_' + str(int(epoch + 1)) + '_' + stage + '.model'
                with open(checkpoint_path, "wb") as f:
                    pk.dump({
                        'model': psgail,
                        'epoch': epoch,
                        'rewards_log': rewards_log,
                        'epochs_log': epochs_log,
                        'agent_num': agent_num,
                        'stage': stage,
                        'dis_ag_rew': dis_ag_rew,
                        'dis_ex_rew': dis_ex_rew,
                        'pol_losses': pol_losses,
                        'val_losses': val_losses,
                        'avg_finals': avg_finals,
                        'done_rate': done_rate,
                    }, f, )
                print('\ncheckpoints saved in ' + checkpoint_path)
            
            if (epoch + 1) % print_every == 0 or epoch + 1 == num_epoch:
                print('\nSurvival List', done_agents_steps_list)
                print('\nFinal List', final_xs_list)
                pbar.set_postfix({'Agents Num': '{}'.format(agent_num),
                                  'Reward': '{0:1.4f}'.format(np.mean(rewards_log[-cal_range:])),
                                  'Survival Time': '{0:1.4f}'.format(np.mean(avg_step_log[-cal_range:])),
                                  'Final Position': '{0:1.4f}'.format(np.mean(avg_finals[-cal_range:])),
                                  'Policy Loss': '{0:1.4f}'.format(np.mean(pol_losses[--cal_range:])),
                                  'Value Loss': '{0:1.4f}'.format(np.mean(val_losses[--cal_range:])),
                                  'Discriminator Loss': '{0:1.4f}'.format(np.mean(dis_total_losses[-cal_range:]))
                                  })
    
    infos = {
        "rewards": rewards_log,
        "epochs": epochs_log,
        'pol_loss': pol_losses,
        'val_loss': val_losses,
        'kl_div': kl_log,
        'dis_ag_rew': dis_ag_rew,
        'dis_ex_rew': dis_ex_rew,
        'avg_survival_time': avg_step_log,
        'dis_loss': dis_total_losses,
        'avg_finals': avg_finals,
        'done_rate': done_rate,
        'survival_time_list': step_log,
        'final_position_list': final_log,
    }
    vector_env.close()
    return infos


if __name__ == "__main__":
    dt = datetime.utcnow()
    dt = dt.replace(tzinfo=timezone.utc)
    tzutc_8 = timezone(timedelta(hours=8))
    local_dt = dt.astimezone(tzutc_8)
    date_and_time = local_dt.strftime("%Y%m%d-%H%M%S")
    
    train_epochs = 1000
    finetune_epochs = 200
    discriminator_lr = 2e-5
    policy_lr = 5e-5
    value_lr = 1e-4
    
    train_flag = True
    plot_flag = False
    continue_flag = False
    
    exp_name = 'EXP15_Final-t-ft'
    env_name = 'NGSIM SMARTS'
    
    if train_flag:
        output_dir = './output/' + date_and_time + '-' + exp_name + '-' + str(train_epochs) + '-' + str(finetune_epochs) + '-' + str(discriminator_lr) + '-' + str(policy_lr) + '-' + str(value_lr) + '/'
        os.mkdir(output_dir)
        os.mkdir(output_dir + 'models')
        os.mkdir(output_dir + 'figures')
        
        log_file_path = output_dir + 'log.txt'
        sys.stdout = Logger(file_name=log_file_path, stream=sys.stdout)
        sys.stderr = Logger(file_name=log_file_path, stream=sys.stderr)
        
        psgail = PSGAIL(discriminator_lr=discriminator_lr, policy_lr=policy_lr, value_lr=value_lr)
        experts = np.load('experts_{}.npy'.format(FEATURE_VECTOR_LENGTH))
        
        stage = 'train'
        train_infos = train(psgail, output_dir=output_dir, experts=experts, stage=stage, num_epoch=train_epochs, print_every=1, add_every=50, save_every=25, gamma=0.95, batch_size=2000, agent_num=2, add_agent_num=2, mini_epoch=10,
                            process_num=CORE_NUM)
        
        with open(output_dir + 'infos_' + stage + '.pkl', "wb") as f:
            pk.dump(train_infos, f)
        
        plot_infos(output_dir, stage, train_infos)
        
        stage = 'finetune'
        finetune_infos = train(psgail, output_dir=output_dir, experts=experts, stage=stage, num_epoch=finetune_epochs, print_every=1, add_every=50, save_every=25, gamma=0.99, batch_size=10000, agent_num=100, add_agent_num=50,
                               mini_epoch=10,
                               process_num=CORE_NUM)
        
        with open(output_dir + 'infos_' + stage + '.pkl', "wb") as f:
            pk.dump(finetune_infos, f)
        
        plot_infos(output_dir, stage, finetune_infos)
    
    elif plot_flag:
        output_dir = './output/20211226-025434-EXP4_fixed-t-ft-1000-200-5e-05-0.0001-5e-05/'
        for stage in ['train']:
            with open(output_dir + 'infos_' + stage + '.pkl', "rb") as f:
                infos = pk.load(f)
            plot_infos(output_dir, stage, infos)
    
    elif continue_flag:
        output_dir = './output/20211226-025434-EXP4_fixed-t-ft-1000-200-5e-05-0.0001-5e-05/'
        checkpoint_path = output_dir + 'models/final_checkpoint_GAIL_1000_train.model'
        log_file_path = output_dir + 'log_continue.txt'
        sys.stdout = Logger(file_name=log_file_path, stream=sys.stdout)
        sys.stderr = Logger(file_name=log_file_path, stream=sys.stderr)
        with open(checkpoint_path, "rb") as f:
            checkpoint = pk.load(f)
        psgail = checkpoint['model']
        experts = np.load('experts_{}.npy'.format(FEATURE_VECTOR_LENGTH))
        
        stage = 'finetune'
        finetune_infos = train(psgail, output_dir=output_dir, experts=experts, stage=stage, num_epoch=finetune_epochs, print_every=1, add_every=25, save_every=25, gamma=0.99, batch_size=10000, agent_num=100, add_agent_num=50,
                               mini_epoch=5, process_num=CORE_NUM)
        
        with open(output_dir + 'infos_' + stage + '.pkl', "wb") as f:
            pk.dump(finetune_infos, f)
        
        plot_infos(output_dir, stage, finetune_infos)

