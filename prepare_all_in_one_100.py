import argparse
import copy
import math
import pickle as pk
from collections import namedtuple
from dataclasses import replace

import cv2
import gym
import numpy as np
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.core.utils.math import vec_2d

CORE_NUM = 12
FEATURE_VECTOR_LENGTH = 100
EGO_FEATURE_VECTOR_LENGTH = 20
V_FEATURE_VECTOR_LENGTH = 10

Config = namedtuple("Config", "name, agent, interface, policy, learning, other, trainer")
FeatureMetaInfo = namedtuple("FeatureMetaInfo", "space, data")

SPACE_LIB = dict(feature_vector=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
                 distance_to_center=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
                 heading_errors=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
                 speed=lambda shape: gym.spaces.Box(low=-330.0, high=330.0, shape=shape),
                 steering=lambda shape: gym.spaces.Box(low=-1.0, high=1.0, shape=shape),
                 neighbor=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
                 ego_pos=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
                 heading=lambda shape: gym.spaces.Box(low=-1e3, high=1e3, shape=shape),
                 ego_lane_dist_and_speed=lambda shape: gym.spaces.Box(low=-1e2, high=1e2, shape=shape),
                 img_gray=lambda shape: gym.spaces.Box(low=0.0, high=1.0, shape=shape),
                 )


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
    
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        if abs(rel_pos_vec[0]) > 60 or abs(rel_pos_vec[1]) > 15:
            continue
        
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)
    
    return groups


class CalObs:
    @staticmethod
    def cal_feature_vector(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        # pdb.set_trace()
        # print(ego)
        # EgoVehicleObservation(id='history-vehicle-36',
        #                       position=array([ 5.92964208, 16.70202832,  0.        ]),
        #                       bounding_box=Dimensions(length=7.55904, width=2.5908, height=1.89),
        #                       heading=Heading(-1.577958366496052),
        #                       speed=4.511155698948702,
        #                       steering=None,
        #                       yaw_rate=-7.993605777301127e-14,
        #                       road_id='gneE01',
        #                       lane_id='gneE01_2',
        #                       lane_index=2,
        #                       mission=Mission(start=Start(position=Point(x=128.5617740793885, y=1.2430420441580357, z=0), heading=Heading(-1.5297118970629224), from_front_bumper=True), goal=EndlessGoal(), route_vias=(), start_time=0.1, entry_tactic=None, via=(), vehicle_spec=None),
        #                       linear_velocity=array([ 4.51104  , -0.0323088,  0.       ]),
        #                       angular_velocity=array([ 0.00000000e+00, -7.99273842e-14,  0.00000000e+00]),
        #                       linear_acceleration=array([ 8.88178420e-15, -3.60614316e-13,  0.00000000e+00]),
        #                       angular_acceleration=array([3.2158080e-05, 4.5041601e-03, 0.0000000e+00]),
        #                       linear_jerk=array([1.77635684e-14, 2.03200000e-03, 0.00000000e+00]),
        #                       angular_jerk=array([6.47725235e-05, 9.07243928e-03, 0.00000000e+00]))
    
        roads_idx = {'gneE01': 1, 'gneE05a': 2, 'gneE05b': 3, 'gneE51': 4}
        ego_pos = ego.position[:2]
        ego_heading = ego.heading
        ego_speed = ego.speed
        ego_vL = ego[2].length
        ego_vW = ego[2].width
        ego_vl = ego_vL / 2
        ego_vw = ego_vW / 2
        ego_corners = ego_pos + [[ego_vl, ego_vw], [ego_vl, -ego_vw], [-ego_vl, ego_vw], [-ego_vl, -ego_vw]]
    
        if ego.lane_id is None or ego.road_id is None:
            ego_road_idx = 0
            ego_lane_idx = 0
            print(ego.id, ego.road_id, ego.lane_id)
        else:
            ego_road_id, ego_lane_id = ego.lane_id.split('_')
            ego_road_idx = roads_idx.get(ego_road_id)
            ego_lane_idx = int(ego_lane_id)
    
        ego_feature = np.asarray(
            [*ego_pos, ego_vL, ego_vW, ego_heading, ego_speed, *ego.linear_velocity[:2], *ego.angular_velocity[:2], *ego.linear_acceleration[:2], *ego.angular_acceleration[:2], *ego.linear_jerk[:2], *ego.angular_jerk[:2], ego_road_idx,
             ego_lane_idx])
    
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        v_features = np.zeros((closest_neighbor_num, V_FEATURE_VECTOR_LENGTH))
        surrounding_vehicles = _get_closest_vehicles(ego, neighbor_vehicle_states, n=closest_neighbor_num)
    
        for i, v in surrounding_vehicles.items():
            # VehicleObservation(id='history-vehicle-36',
            #                    position=array([21.20865648, 16.57539408, 0.]),
            #                    bounding_box=Dimensions(length=7.55904, width=2.5908, height=1.89),
            #                    heading=Heading(-1.5462927484570121),
            #                    speed=0.7547081616201112,
            #                    road_id='gneE01',
            #                    lane_id='gneE01_2',
            #                    lane_index=2)
        
            if v[0] is None:
                v_features[i, :] = np.zeros(V_FEATURE_VECTOR_LENGTH)
                continue
            else:
                v = v[0]
        
            v_pos = v.position[:2]  # 取坐标
            v_heading = np.asarray(float(v.heading))  # 取运动方向
            v_speed = np.asarray(v.speed)  # 取速率
            rel_pos = v_pos - ego_pos  # 取中心相对位置
            v_L = v[2].length
            v_W = v[2].width
            v_l = v_L / 2
            v_w = v_W / 2
            v_corners = v_pos + [[v_l, v_w], [v_l, -v_w], [-v_l, v_w], [-v_l, -v_w]]
            distance = 1e10
            for v_corner in v_corners:
                for ego_corner in ego_corners:
                    distance = min(distance, np.linalg.norm(v_corner - ego_corner))
            distance = min(distance, abs(rel_pos[0]), abs(rel_pos[1]))  # 近似计算最近距离
        
            if v.lane_id is None or v.road_id is None:
                v_lane_idx = 0
                print(v.id, v.road_id, v.lane_id)
            else:
                v_road_id, v_lane_id = v.lane_id.split('_')
                v_road_idx = roads_idx.get(v_road_id)
                v_lane_idx = int(v_lane_id)
        
            if -ego_vw < rel_pos[1] < ego_vw:
                dv = v_speed - ego_speed
                v_ttc = rel_pos[0] / (np.sign(dv) * abs(dv) + 1e-8)
                if v_ttc < 0:
                    v_ttc = 1e8
                a = ego.linear_acceleration[0]
                if a == 0:
                    v_ttc_with_a = v_ttc
                else:
                    delta = 4 * dv * dv / (a * a) - 8 * rel_pos[0] / a
                    if delta < 0:
                        v_ttc_with_a = 1e8
                    else:
                        v_ttc_with_a = dv / a + np.sqrt(delta) / 2
            else:
                v_ttc = 1e4
                v_ttc_with_a = 1e4
            v_features[i, :] = np.asarray([v_L, v_W, v_heading, v_speed, *rel_pos, distance, v_lane_idx, v_ttc, v_ttc_with_a])
        v_features = v_features.reshape((-1,))
    
        vecs = np.concatenate((ego_feature, v_features), axis=0)
        return vecs
    
    @staticmethod
    def cal_ego_pos(env_obs: Observation, **kwargs):
        return env_obs.ego_vehicle_state.position[:2]
    
    @staticmethod
    def cal_heading(env_obs: Observation, **kwargs):
        return np.asarray(float(env_obs.ego_vehicle_state.heading))
    
    @staticmethod
    def cal_distance_to_center(env_obs: Observation, **kwargs):
        """Calculate the signed distance to the center of the current lane.
        Return a FeatureMetaInfo(space, data) instance
        """
        
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_to_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        # TODO(ming): for the case of overwhilm, it will throw error
        norm_dist_from_center = signed_dist_to_center / lane_hwidth
        
        dist = np.asarray([norm_dist_from_center])
        return dist
    
    @staticmethod
    def cal_heading_errors(env_obs: Observation, **kwargs):
        look_ahead = kwargs["look_ahead"]
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        closest_path = waypoint_paths[closest_wp.lane_index][:look_ahead]
        
        heading_errors = [
            math.sin(math.radians(wp.relative_heading(ego.heading)))
            for wp in closest_path
        ]
        
        if len(heading_errors) < look_ahead:
            last_error = heading_errors[-1]
            heading_errors = heading_errors + [last_error] * (
                    look_ahead - len(heading_errors)
            )
        
        # assert len(heading_errors) == look_ahead
        return np.asarray(heading_errors)
    
    @staticmethod
    def cal_speed(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        res = np.asarray([ego.speed])
        return res
    
    @staticmethod
    def cal_steering(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        return np.asarray([ego.steering / 45.0])
    
    @staticmethod
    def cal_neighbor(env_obs: Observation, **kwargs):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_neighbor_num = kwargs.get("closest_neighbor_num", 8)
        # dist, speed, ttc, pos
        features = np.zeros((closest_neighbor_num, 4))
        # fill neighbor vehicles into closest_neighboor_num areas
        surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )
        ego_pos = ego.position[:2]
        ego_heading = np.asarray(float(ego.heading))
        ego_speed = np.asarray(ego.speed)
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                v = ego
            else:
                v = v[0]
            
            pos = v.position[:2]
            heading = np.asarray(float(v.heading))
            speed = np.asarray(v.speed)
            
            features[i, :] = np.asarray([pos[0] - ego_pos[0], pos[1] - ego_pos[1], heading - ego_heading, speed - ego_speed])
        return features.reshape((-1,))
        # return None
    
    @staticmethod
    def cal_ego_lane_dist_and_speed(env_obs: Observation, **kwargs):
        """Calculate the distance from ego vehicle to its front vehicles (if have) at observed lanes,
        also the relative speed of the front vehicle which positioned at the same lane.
        """
        observe_lane_num = kwargs["observe_lane_num"]
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        
        wps_with_lane_dist = []
        for path_idx, path in enumerate(waypoint_paths):
            lane_dist = 0.0
            for w1, w2 in zip(path, path[1:]):
                wps_with_lane_dist.append((w1, path_idx, lane_dist))
                lane_dist += np.linalg.norm(w2.pos - w1.pos)
            wps_with_lane_dist.append((path[-1], path_idx, lane_dist))
        
        # TTC calculation along each path
        ego_closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        
        wps_on_lane = [
            (wp, path_idx, dist)
            for wp, path_idx, dist in wps_with_lane_dist
            # if wp.lane_id == v.lane_id
        ]
        
        ego_lane_index = closest_wp.lane_index
        lane_dist_by_path = [1] * len(waypoint_paths)
        ego_lane_dist = [0] * observe_lane_num
        speed_of_closest = 0.0
        
        for v in env_obs.neighborhood_vehicle_states:
            nearest_wp, path_idx, lane_dist = min(
                wps_on_lane,
                key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position)),
            )
            if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                # this vehicle is not close enough to the path, this can happen
                # if the vehicle is behind the ego, or ahead past the end of
                # the waypoints
                continue
            
            # relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
            # relative_speed_m_per_s = max(abs(relative_speed_m_per_s), 1e-5)
            dist_wp_vehicle_vector = vec_2d(v.position) - vec_2d(nearest_wp.pos)
            direction_vector = np.array(
                [
                    math.cos(math.radians(nearest_wp.heading)),
                    math.sin(math.radians(nearest_wp.heading)),
                ]
            ).dot(dist_wp_vehicle_vector)
            
            dist_to_vehicle = lane_dist + np.sign(direction_vector) * (
                np.linalg.norm(vec_2d(nearest_wp.pos) - vec_2d(v.position))
            )
            lane_dist = dist_to_vehicle / 100.0
            
            if lane_dist_by_path[path_idx] > lane_dist:
                if ego_closest_wp.lane_index == v.lane_index:
                    speed_of_closest = (v.speed - ego.speed) / 120.0
            
            lane_dist_by_path[path_idx] = min(lane_dist_by_path[path_idx], lane_dist)
        
        # current lane is centre
        flag = observe_lane_num // 2
        ego_lane_dist[flag] = lane_dist_by_path[ego_lane_index]
        
        max_lane_index = len(lane_dist_by_path) - 1
        
        if max_lane_index == 0:
            right_sign, left_sign = 0, 0
        else:
            right_sign = -1 if ego_lane_index + 1 > max_lane_index else 1
            left_sign = -1 if ego_lane_index - 1 >= 0 else 1
        
        ego_lane_dist[flag + right_sign] = lane_dist_by_path[
            ego_lane_index + right_sign
            ]
        ego_lane_dist[flag + left_sign] = lane_dist_by_path[ego_lane_index + left_sign]
        
        res = np.asarray(ego_lane_dist + [speed_of_closest])
        return res
        # space = SPACE_LIB["goal_relative_pos"](res.shape)
        # return (res - space.low) / (space.high - space.low)
    
    @staticmethod
    def cal_img_gray(env_obs: Observation, **kwargs):
        resize = kwargs["resize"]
        
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        
        rgb_ndarray = env_obs.top_down_rgb
        gray_scale = (
                cv2.resize(
                    rgb2gray(rgb_ndarray), dsize=resize, interpolation=cv2.INTER_CUBIC
                )
                / 255.0
        )
        return gray_scale


def _update_obs_by_item(ith, obs_placeholder: dict, tuned_obs: dict, space_dict: gym.spaces.Dict):
    for key, value in tuned_obs.items():
        if obs_placeholder.get(key, None) is None:
            obs_placeholder[key] = np.zeros(space_dict[key].shape)
        obs_placeholder[key][ith] = value


def _cal_obs(env_obs: Observation, space, **kwargs):
    obs = dict()
    for name in space.spaces:
        if hasattr(CalObs, f"cal_{name}"):
            obs[name] = getattr(CalObs, f"cal_{name}")(env_obs, **kwargs)
    return obs


def subscribe_features(**kwargs):
    res = dict()
    
    for k, config in kwargs.items():
        if bool(config):
            res[k] = SPACE_LIB[k](config)
    
    return res


def get_observation_adapter_(observation_space, **kwargs):
    def observation_adapter(env_obs):
        obs = dict()
        if isinstance(env_obs, list) or isinstance(env_obs, tuple):
            for i, e in enumerate(env_obs):
                temp = _cal_obs(e, observation_space, **kwargs)
                _update_obs_by_item(i, obs, temp, observation_space)
        else:
            temp = _cal_obs(env_obs, observation_space, **kwargs)
            _update_obs_by_item(0, obs, temp, observation_space)
        return obs
    
    return observation_adapter


def get_observation_adapter(obs_stack_size):
    stack_size = obs_stack_size
    closest_neighbor_num = 8
    img_resolution = 40
    observe_lane_num = 3
    
    subscribed_features = dict(feature_vector=(stack_size, FEATURE_VECTOR_LENGTH))
    
    observation_space = gym.spaces.Dict(subscribe_features(**subscribed_features))
    
    observation_adapter = get_observation_adapter_(
        observation_space,
        observe_lane_num=observe_lane_num,
        resize=(img_resolution, img_resolution),
        closest_neighbor_num=closest_neighbor_num,
    )
    
    return observation_adapter


def acceleration_count(obs, obs_next, acc_dict, ang_v_dict, avg_dis_dict):
    acc_dict = {}
    for car in obs.keys():
        car_state = obs[car].ego_vehicle_state
        angular_velocity = car_state.yaw_rate
        ang_v_dict.append(angular_velocity)
        dis_cal = car_state.speed * 0.1
        if car in avg_dis_dict:
            avg_dis_dict[car] += dis_cal
        else:
            avg_dis_dict[car] = dis_cal
        if car not in obs_next.keys():
            continue
        car_next_state = obs_next[car].ego_vehicle_state
        acc_cal = (car_next_state.speed - car_state.speed) / 0.1
        acc_dict.append(acc_cal)


def cal_action(obs, obs_next, dt=0.1):
    act = {}
    for car in obs.keys():
        if car not in obs_next.keys():
            continue
        car_state = obs[car].ego_vehicle_state
        car_next_state = obs_next[car].ego_vehicle_state
        acceleration = (car_next_state.speed - car_state.speed) / dt
        angular_velocity = car_state.yaw_rate
        act[car] = np.array([acceleration, angular_velocity])
    return act


def prepare(scenario, obs_stack_size=1):
    agent_spec = AgentSpec(
        interface=AgentInterface(
            max_episode_steps=None,
            waypoints=False,
            neighborhood_vehicles=True,
            ogm=False,
            rgb=False,
            lidar=False,
            action=ActionSpaceType.Imitation,
        ),
        observation_adapter=get_observation_adapter(obs_stack_size),
    )
    
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
    )
    scenarios_iterator = Scenario.scenario_variations(
        [scenario],
        list([]),
    )
    
    smarts.reset(next(scenarios_iterator))
    
    expert_obs = []
    expert_acts = []
    expert_obs_next = []
    expert_terminals = []
    cars_obs = {}
    cars_act = {}
    cars_obs_next = {}
    cars_terminals = {}
    
    prev_vehicles = set()
    done_vehicles = set()
    prev_obs = None
    while True:
        smarts.step({})
        
        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        done_vehicles = prev_vehicles - current_vehicles
        prev_vehicles = current_vehicles
        
        if len(current_vehicles) == 0:
            break
        
        smarts.attach_sensors_to_vehicles(
            agent_spec, smarts.vehicle_index.social_vehicle_ids()
        )
        obs, _, _, dones = smarts.observe_from(
            smarts.vehicle_index.social_vehicle_ids()
        )
        
        for v in done_vehicles:
            cars_terminals[f"Agent-{v}"][-1] = True
            print(f"Agent-{v} Ended")
        
        # handle actions
        if prev_obs is not None:
            act = cal_action(prev_obs, obs)
            for car in act.keys():
                if cars_act.__contains__(car):
                    cars_act[car].append(act[car])
                else:
                    cars_act[car] = [act[car]]
        prev_obs = copy.copy(obs)
        
        # handle observations
        cars = obs.keys()
        for car in cars:
            _obs = agent_spec.observation_adapter(obs[car])
            obs_vec = _obs['feature_vector'].squeeze()
            if cars_obs.__contains__(car):
                cars_obs[car].append(obs_vec)
                cars_terminals[car].append(dones[car])
            else:
                cars_obs[car] = [obs_vec]
                cars_terminals[car] = [dones[car]]
    
    for car in cars_obs:
        cars_obs_next[car] = cars_obs[car][1:]
        cars_obs[car] = cars_obs[car][:-1]
        cars_act[car] = np.array(cars_act[car])
        cars_terminals[car] = np.array(cars_terminals[car][:-1])
        expert_obs.append(cars_obs[car])
        expert_acts.append(cars_act[car])
        expert_obs_next.append(cars_obs_next[car])
        expert_terminals.append(cars_terminals[car])
    
    # with open("experts_{}.pkl".format(FEATURE_VECTOR_LENGTH), "wb") as f:
    #     pk.dump(
    #         {
    #             "observations": expert_obs,
    #             "actions": expert_acts,
    #         },
    #         f,
    #     )
    
    observations = []
    
    for observation in expert_obs:
        observations.append(np.array(observation))
    
    experts_observations = np.vstack(observations)
    experts_actions = np.vstack(expert_acts)
    experts = np.hstack((experts_observations, experts_actions))
    np.random.seed(42)
    np.random.shuffle(experts)
    np.save('experts_{}.npy'.format(FEATURE_VECTOR_LENGTH), experts)
    
    print('Generation Finished')
    
    smarts.destroy()


def get_action_adapter():
    def action_adapter(model_action):
        assert len(model_action) == 2
        return model_action[0], model_action[1]
    
    return action_adapter


class MATrafficSim:
    def __init__(self, scenarios, agent_number, obs_stacked_size=1):
        self.scenarios_iterator = Scenario.scenario_variations(scenarios, [])
        self.obs_stacked_size = obs_stacked_size
        self.n_agents = agent_number
        self._init_scenario()
        self.agentid_to_vehid = {}
        self.agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_spec = AgentSpec(interface=AgentInterface(max_episode_steps=None,
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
        
        self.smarts = SMARTS(agent_interfaces={},
                             traffic_sim=None,
                             envision=None,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="./ngsim",
    )
    args = parser.parse_args()
    prepare(scenario=args.scenario)
