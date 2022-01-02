import subprocess

from envision.client import Client as Envision
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.traffic_history_provider import TrafficHistoryProvider

from train_all_in_one_86 import *

if torch.cuda.is_available():
    device = "cuda:1"
else:
    device = "cpu"


class TrafficSimV(gym.Env):
    def __init__(self, scenarios):
        super(TrafficSimV, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._next_scenario()
        self.obs_stacked_size = 1
        self.agent_spec = AgentSpec(
            interface=AgentInterface(
                max_episode_steps=None,
                waypoints=False,
                neighborhood_vehicles=True,
                ogm=False,
                rgb=False,
                lidar=False,
                action=ActionSpaceType.Imitation,
            ),
            action_adapter=get_action_adapter(),
            observation_adapter=get_observation_adapter(self.obs_stacked_size),
        )
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64
        )
        
        envision_client = Envision(
            endpoint="ws://localhost:8081",
            sim_name="NGSIM_MAGAIL",
            output_dir='./visual',
            headless=None,
        )
        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=envision_client,
        )
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def _convert_obs(self, raw_observations):
        observation = self.agent_spec.observation_adapter(raw_observations[self.vehicle_id])
        ego_state = []
        other_info = []
        for feature in observation:
            if feature in ["ego_pos", "speed", "heading"]:
                ego_state.append(observation[feature])
            else:
                other_info.append(observation[feature])
        ego_state = np.concatenate(ego_state, axis=1).reshape(-1)
        other_info = np.concatenate(other_info, axis=1).reshape(-1)
        full_obs = np.concatenate((ego_state, other_info))
        return full_obs
    
    def step(self, action):
        raw_observations, rewards, dones, _ = self.smarts.step({self.vehicle_id: self.agent_spec.action_adapter(action)})
        
        info = {}
        info["reached_goal"] = raw_observations[self.vehicle_id].events.reached_goal
        info["collision"] = len(raw_observations[self.vehicle_id].events.collisions) > 0
        obs = self.agent_spec.observation_adapter(raw_observations[self.vehicle_id])
        
        return (
            obs,
            rewards[self.vehicle_id],
            dones[self.vehicle_id],
            info,
        )
    
    def reset(self):
        if self.vehicle_itr >= len(self.vehicle_ids):
            self._next_scenario()
        
        self.vehicle_id = self.vehicle_ids[self.vehicle_itr]
        vehicle_mission = self.vehicle_missions[self.vehicle_id]
        
        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        traffic_history_provider.start_time = vehicle_mission.start_time
        
        modified_mission = replace(vehicle_mission, start_time=0.0)
        self.scenario.set_ego_missions({self.vehicle_id: modified_mission})
        self.smarts.switch_ego_agents({self.vehicle_id: self.agent_spec.interface})
        
        observations = self.smarts.reset(self.scenario)
        obs = self.agent_spec.observation_adapter(observations[self.vehicle_id])
        self.vehicle_itr += 1
        return obs
    
    def _next_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.vehicle_ids = list(self.vehicle_missions.keys())
        np.random.shuffle(self.vehicle_ids)
        self.vehicle_itr = 0
    
    def destroy(self):
        if self.smarts is not None:
            self.smarts.destroy()


class MATrafficSimV(gym.Env):
    def __init__(self, scenarios, agent_number=10, obs_stacked_size=1):
        super(MATrafficSimV, self).__init__()
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self.n_agents = agent_number
        self.obs_stacked_size = obs_stacked_size
        self._init_scenario()
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_spec = AgentSpec(
            interface=AgentInterface(
                max_episode_steps=None,
                waypoints=False,
                neighborhood_vehicles=True,
                ogm=False,
                rgb=False,
                lidar=False,
                action=ActionSpaceType.Imitation,
            ),
            action_adapter=get_action_adapter(),
            observation_adapter=get_observation_adapter(obs_stacked_size),
        )
        
        envision_client = Envision(
            endpoint="ws://localhost:8081",
            sim_name="NGSIM_MAGAIL",
            output_dir='./visual',
            headless=None,
        )
        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=envision_client,
        )
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def step(self, action):
        # raw_observations, rewards, dones, _ = self.smarts.step({self.vehicle_id: self.agent_spec.action_adapter(action)})
        #
        # info = {}
        # info["reached_goal"] = raw_observations[self.vehicle_id].events.reached_goal
        # info["collision"] = len(raw_observations[self.vehicle_id].events.collisions) > 0
        # observations = self.agent_spec.observation_adapter(raw_observations[self.vehicle_id])
        
        for agent_id in self.agent_ids:
            if agent_id not in action.keys():
                continue
            agent_action = action[agent_id]
            action[agent_id] = self.agent_spec.action_adapter(agent_action)
        
        observations, rewards, dones, _ = self.smarts.step(action)
        info = {}
        
        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(
                observations[k]
            )
        
        dones["__all__"] = all(dones.values())
        
        return (observations,
                rewards,
                dones,
                info,
                )
    
    def reset(self):
        if self.vehicle_itr + self.n_agents >= (len(self.vehicle_ids) - 1):
            self.vehicle_itr = 0
        
        self.vehicle_id = self.vehicle_ids[self.vehicle_itr: self.vehicle_itr + self.n_agents]
        
        traffic_history_provider = self.smarts.get_provider_by_type(TrafficHistoryProvider)
        assert traffic_history_provider
        
        for i in range(self.n_agents):
            self.agentid_to_vehid[f"agent_{i}"] = self.vehicle_id[i]
        
        agent_interfaces = {}
        history_start_time = self.vehicle_missions[self.vehicle_id[0]].start_time
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            agent_interfaces[agent_id] = self.agent_spec.interface
            if history_start_time > self.vehicle_missions[vehicle].start_time:
                history_start_time = self.vehicle_missions[vehicle].start_time
        
        traffic_history_provider.start_time = history_start_time
        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle = self.agentid_to_vehid[agent_id]
            ego_missions[agent_id] = replace(self.vehicle_missions[vehicle], start_time=self.vehicle_missions[vehicle].start_time - history_start_time, )
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)
        
        observations = self.smarts.reset(self.scenario)
        
        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(observations[k])
        self.vehicle_itr += np.random.choice(len(self.vehicle_ids) - self.n_agents)
        
        return observations
    
    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.veh_start_times = {}
        for v_id, mission in self.vehicle_missions.items():
            self.veh_start_times[v_id] = mission.start_time
        self.vehicle_ids = list(self.vehicle_missions.keys())
        vlist = []
        for vehicle_id, start_time in self.veh_start_times.items():
            vlist.append((vehicle_id, start_time))
        dtype = [("id", int), ("start_time", float)]
        vlist = np.array(vlist, dtype=dtype)
        vlist = np.sort(vlist, order="start_time")
        self.vehicle_ids = list(self.vehicle_missions.keys())
        for id in range(len(self.vehicle_ids)):
            self.vehicle_ids[id] = f"{vlist[id][0]}"
        self.vehicle_itr = np.random.choice(len(self.vehicle_ids) - self.n_agents)
    
    def close(self):
        if self.smarts is not None:
            self.smarts.destroy()


class MATrafficSimNewV:
    def __init__(self, scenarios, agent_number, obs_stacked_size=1):
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self._init_scenario()
        self.obs_stacked_size = obs_stacked_size
        self.n_agents = agent_number
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_spec = AgentSpec(
            interface=AgentInterface(
                max_episode_steps=None,
                waypoints=False,
                neighborhood_vehicles=True,
                ogm=False,
                rgb=False,
                lidar=False,
                action=ActionSpaceType.Imitation,
            ),
            action_adapter=get_action_adapter(),
            observation_adapter=get_observation_adapter(obs_stacked_size),
        )
        
        envision_client = Envision(
            endpoint="ws://localhost:8081",
            sim_name="NGSIM_MAGAIL",
            output_dir='./visual',
            headless=None,
        )
        self.smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            envision=envision_client,
        )
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def step(self, action):
        for agent_id in self.agent_ids:
            if agent_id not in action.keys():
                continue
            agent_action = action[agent_id]
            action[agent_id] = self.agent_spec.action_adapter(agent_action)
        observations, rewards, dones, _ = self.smarts.step(action)
        info = {}
        
        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(observations[k])
        
        dones["__all__"] = all(dones.values())
        
        return (
            observations,
            rewards,
            dones,
            info,
        )
    
    def reset(self, internal_replacement=True, min_successor_time=5.0):
        if self.vehicle_itr + self.n_agents >= (len(self.vehicle_ids) - 1):
            self.vehicle_itr = 0
        
        self.vehicle_id = self.vehicle_ids[
                          self.vehicle_itr: self.vehicle_itr + self.n_agents
                          ]
        
        traffic_history_provider = self.smarts.get_provider_by_type(
            TrafficHistoryProvider
        )
        assert traffic_history_provider
        
        for i in range(self.n_agents):
            self.agentid_to_vehid[f"agent_{i}"] = self.vehicle_id[i]
        
        history_start_time = self.vehicle_missions[self.vehicle_id[0]].start_time
        agent_interfaces = {a_id: self.agent_spec.interface for a_id in self.agent_ids}
        
        if internal_replacement:
            history_end_time = min(
                [
                    self.scenario.traffic_history.vehicle_final_exit_time(v_id)
                    for v_id in self.vehicle_id
                ]
            )
            alive_time = history_end_time - history_start_time
            traffic_history_provider.start_time = (
                    history_start_time
                    + np.random.choice(
                max(0, round(alive_time * 10) - round(min_successor_time * 10))
            )
                    / 10
            )
        else:
            traffic_history_provider.start_time = history_start_time
        
        ego_missions = {}
        for agent_id in self.agent_ids:
            vehicle_id = self.agentid_to_vehid[agent_id]
            start_time = max(
                0,
                self.vehicle_missions[vehicle_id].start_time
                - traffic_history_provider.start_time,
            )
            ego_missions[agent_id] = replace(
                self.vehicle_missions[vehicle_id],
                start_time=start_time,
                start=get_vehicle_start_at_time(
                    vehicle_id,
                    round(
                        max(
                            traffic_history_provider.start_time,
                            self.vehicle_missions[vehicle_id].start_time,
                        ),
                        1,
                    ),
                    self.scenario.traffic_history,
                ),
            )
        self.scenario.set_ego_missions(ego_missions)
        self.smarts.switch_ego_agents(agent_interfaces)
        
        observations = self.smarts.reset(self.scenario)
        for k in observations.keys():
            observations[k] = self.agent_spec.observation_adapter(observations[k])
        self.vehicle_itr += self.n_agents
        
        return observations
    
    def _init_scenario(self):
        self.scenario = next(self.scenarios_iterator)
        self.vehicle_missions = self.scenario.discover_missions_of_traffic_histories()
        self.veh_start_times = {}
        for v_id, mission in self.vehicle_missions.items():
            self.veh_start_times[v_id] = mission.start_time
        self.vehicle_ids = list(self.vehicle_missions.keys())
        vlist = []
        for vehicle_id, start_time in self.veh_start_times.items():
            vlist.append((vehicle_id, start_time))
        dtype = [("id", int), ("start_time", float)]
        vlist = np.array(vlist, dtype=dtype)
        vlist = np.sort(vlist, order="start_time")
        self.vehicle_ids = list(self.vehicle_missions.keys())
        for id in range(len(self.vehicle_ids)):
            self.vehicle_ids[id] = f"{vlist[id][0]}"
        self.vehicle_itr = np.random.choice(len(self.vehicle_ids))
    
    def close(self):
        if self.smarts is not None:
            self.smarts.destroy()


if __name__ == "__main__":
    envision_proc = subprocess.Popen("scl envision start -s ./ngsim", shell=True)
    
    agent_number = 10
    random_flag = False
    eval_flag = True
    
    output_dir = './output/20211230-002334-EXP15_Final-t-ft-1000-200-2e-05-5e-05-0.0001/'
    checkpoint_path = output_dir + 'models/final_checkpoint_GAIL_1000_train.model'
    log_file_path = output_dir + 'log_eval.txt'
    sys.stdout = Logger(file_name=log_file_path, stream=sys.stdout)
    sys.stderr = Logger(file_name=log_file_path, stream=sys.stderr)
    
    with open(checkpoint_path, "rb") as f:
        checkpoint = pk.load(f)
    psgail = checkpoint['model']
    
    if random_flag:
        env = MATrafficSimV(["./ngsim"], agent_number=agent_number)
        for epoch in range(100):
            obs = env.reset()
            done = {a_id: False for a_id in obs.keys()}
            n_steps = 500
            for step in tqdm(range(n_steps)):
                act_n = {}
                for agent_id in obs.keys():
                    if step and done[agent_id]:
                        break
                    act_n[agent_id] = np.random.normal(0, 1, size=(2,))
                obs, rew, done, info = env.step(act_n)
            env.close()
    else:
        env = MATrafficSimV(scenarios=["./ngsim"], agent_number=agent_number)
        for epoch in range(100):
            obs = env.reset()
            done = {a_id: False for a_id in obs.keys()}
            n_steps = 250
            for step in tqdm(range(n_steps)):
                act_n = {}
                for agent_id in obs.keys():
                    if step and done[agent_id]:
                        continue
                    obs_vectors = obs[agent_id]['feature_vector']
                    obs_vectors = torch.tensor(obs_vectors, device=device, dtype=torch.float32)
                    act_tmp, _ = psgail.get_action(obs_vectors)
                    act_tmp = act_tmp.cpu()
                    act_n[agent_id] = act_tmp.numpy().squeeze()
                obs, rew, done, _ = env.step(act_n)
        env.close()
    
    print("finished")
