import csv
import datetime
import logging
import time
from statistics import mean

import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from gymnasium.utils import seeding
from datetime import datetime

from .deployment import (
    get_max_cpu,
    get_max_mem,
    get_max_traffic,
    get_redis_deployment_list,
    get_kiali_service_graph,
)
from .util import save_to_csv, get_cost_reward, get_latency_reward_redis, get_num_pods
from .dynamic_graph_space import DynamicGraphSpace, DynamicGraphConfig

# MIN and MAX Replication
MIN_REPLICATION = 1
MAX_REPLICATION = 7  # 實際測試最大可用值

MAX_STEPS = 25  # MAX Number of steps per episode

# Possible Actions (Discrete)
ACTION_DO_NOTHING = 0
ACTION_ADD_1_REPLICA = 1
ACTION_ADD_2_REPLICA = 2
ACTION_ADD_3_REPLICA = 3
ACTION_ADD_4_REPLICA = 4
ACTION_ADD_5_REPLICA = 5
ACTION_ADD_6_REPLICA = 6
ACTION_ADD_7_REPLICA = 7
ACTION_TERMINATE_1_REPLICA = 8
ACTION_TERMINATE_2_REPLICA = 9
ACTION_TERMINATE_3_REPLICA = 10
ACTION_TERMINATE_4_REPLICA = 11
ACTION_TERMINATE_5_REPLICA = 12
ACTION_TERMINATE_6_REPLICA = 13
ACTION_TERMINATE_7_REPLICA = 14

# Deployments
DEPLOYMENTS = ["redis-master", "redis-slave"]

# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_MASTER = 0
ID_SLAVE = 1

# Reward objectives
LATENCY = 'latency'
COST = 'cost'


class Redis(gym.Env):
    """Horizontal Scaling for Redis in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward=COST, waiting_period=5.0, use_graph=False, dataset_path=None):
        # Define action and observation space
        # They must be gym.spaces objects

        super(Redis, self).__init__()

        self.k8s = k8s
        self.name = "redis_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.use_graph = use_graph
        self.waiting_period = waiting_period  # seconds to wait after action
        
        # Initialize dynamic graph space for Redis services
        if self.use_graph:
            config = DynamicGraphConfig(
                max_nodes=8,  # Redis可能有額外的服務節點
                max_edges=16,  # 對應的邊數上限
                node_feat_dim=6,  # Redis節點特徵維度
                edge_feat_dim=7,  # 邊特徵維度 
                global_feat_dim=4  # 全局特徵維度
            )
            self.dynamic_graph = DynamicGraphSpace(config)

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(self.name, self.k8s, self.__version__))

        # Current Step
        self.current_step = 0

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15

        # Multi-Discrete - output action for each service simultaneously
        if self.use_graph:
            # 動態圖模式：使用最大節點數
            num_services = self.dynamic_graph.config.max_nodes
        else:
            num_services = len(DEPLOYMENTS)
        self.action_space = spaces.MultiDiscrete([self.num_actions] * num_services)

        # Observations: 22 Metrics! -> 2 * 11 = 22
        # "number_pods"                     -> Number of deployed Pods
        # "cpu_usage_aggregated"            -> via metrics-server
        # "mem_usage_aggregated"            -> via metrics-server
        # "cpu_requests"                    -> via metrics-server/pod
        # "mem_requests"                    -> via metrics-server/pod
        # "cpu_limits"                      -> via metrics-server
        # "mem_limits"                      -> via metrics-server
        # "lstm_cpu_prediction_1_step"      -> via pod annotation
        # "lstm_cpu_prediction_5_step"      -> via pod annotation
        # "average_number of requests"      -> Prometheus metric: sum(rate(http_server_requests_seconds_count[5m]))

        self.min_pods = MIN_REPLICATION
        self.max_pods = MAX_REPLICATION
        self.num_apps = 2

        # Deployment Data
        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        if self.use_graph:
            # 使用動態圖觀察空間
            self.observation_space = self.dynamic_graph.create_observation_space()
        else:
            self.observation_space = self.get_observation_space()

        # Action and Observation Space
        # logging.info("[Init] Action Spaces: " + str(self.action_space))
        # logging.info("[Init] Observation Spaces: " + str(self.observation_space))

        # Info
        self.total_reward = None
        self.avg_pods = []
        self.avg_latency = []

        # episode over
        self.episode_over = False
        self.info = {}

        # Keywords for Reward calculation
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False
        self.cost_weight = 0  # add here a value to consider cost in the reward function

        self.time_start = 0
        self.execution_time = 0
        self.episode_count = 0
        self.file_results = "results.csv"
        self.obs_csv = self.name + "_observation.csv"

        if dataset_path is None:
            dataset_path = (
                Path(__file__).resolve().parents[2]
                / "datasets"
                / "real"
                / self.deploymentList[0].namespace
                / "v1"
                / f"{self.name}_observation.csv"
            )

        if not self.k8s:
            print(f"[INFO] Loading dataset from: {dataset_path}")
            self.df = pd.read_csv(dataset_path)
        else:
            self.df = pd.DataFrame()

    def _fetch_service_graph(self):
        # In simulation mode, return default graph structure
        if not self.k8s:
            num = len(DEPLOYMENTS)
            adj = np.zeros((num, num), dtype=np.float32)
            # Create simple connection: master -> slave
            adj[0, 1] = 1.0
            
            max_edges = num * num
            edges = []
            # Add default edge: master -> slave with default metrics
            edges.append([0, 1, 1.0, 100.0, 1.0, 0.01, 0])  # src, dst, connection, qps, p95, err_rate, unused
            
            while len(edges) < max_edges:
                edges.append([0, 0, 0, 0, 0, 0, 0])
            edges = np.array(edges[:max_edges], dtype=np.float32)
            
            return adj, edges
        
        # K8s mode: fetch real service graph
        from gnnrl.core.utils.kiali_client import fetch_service_graph

        nodes, edge_df = fetch_service_graph(namespace="redis")
        index = {name: DEPLOYMENTS.index(name) for name in DEPLOYMENTS if name in nodes}
        num = len(DEPLOYMENTS)
        adj = np.zeros((num, num), dtype=np.float32)
        edges = []

        for _, row in edge_df.iterrows():
            if len(nodes) > row["src"] and len(nodes) > row["dst"]:
                src_name = nodes[row["src"]]
                dst_name = nodes[row["dst"]]
                if src_name in index and dst_name in index:
                    s = index[src_name]
                    d = index[dst_name]
                    adj[s, d] = 1.0
                    edges.append([s, d, 1.0, row["qps"], row["p95"], row["err_rate"], 0])

        max_edges = num * num
        while len(edges) < max_edges:
            edges.append([0, 0, 0, 0, 0, 0, 0])
        edges = np.array(edges[:max_edges], dtype=np.float32)

        return adj, edges

    def _fetch_service_graph_dynamic(self):
        """獲取動態服務圖特徵，支援可變節點數量"""
        if not self.k8s:
            # 模擬模式：返回基本的Redis主從結構
            active_services = ["redis-master", "redis-slave"]
            
            # 更新節點映射
            node_mapping = self.dynamic_graph.update_node_mapping(active_services)
            
            # 建構動態邊列表
            edges = []
            # 添加 master -> slave 邊
            if "redis-master" in node_mapping and "redis-slave" in node_mapping:
                master_idx = node_mapping["redis-master"]
                slave_idx = node_mapping["redis-slave"]
                edges.append([master_idx, slave_idx, 1.0, 100.0, 1.0, 0.01, 0])
            
            # 使用動態圖空間填充邊特徵
            edge_array = np.array(edges, dtype=np.float32) if edges else np.zeros((0, 7), dtype=np.float32)
            
            return edge_array, len(active_services)
        
        # K8s模式：從Kiali獲取實際服務圖
        from gnnrl.core.utils.kiali_client import fetch_service_graph
        
        nodes, edge_df = fetch_service_graph(namespace="redis")
        
        # 確保包含基本的Redis服務
        active_services = list(nodes)
        for service in DEPLOYMENTS:
            if service not in active_services:
                active_services.append(service)
        
        # 更新節點映射
        node_mapping = self.dynamic_graph.update_node_mapping(active_services)
        
        # 建構動態邊列表
        edges = []
        for _, row in edge_df.iterrows():
            if row["src"] < len(nodes) and row["dst"] < len(nodes):
                src_name = nodes[row["src"]]
                dst_name = nodes[row["dst"]]
                
                if src_name in node_mapping and dst_name in node_mapping:
                    src_idx = node_mapping[src_name]
                    dst_idx = node_mapping[dst_name]
                    edges.append([
                        src_idx, dst_idx, 1.0,
                        row.get("qps", 0.0),
                        row.get("p95", 0.0),
                        row.get("error_rate", 0.0),
                        0  # 未使用的特徵
                    ])
        
        # 使用動態圖空間填充邊特徵
        edge_array = np.array(edges, dtype=np.float32) if edges else np.zeros((0, 7), dtype=np.float32)
        
        return edge_array, len(active_services)

    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()

        # Get first action: deployment
        if action[ID_DEPLOYMENTS] == 0:  # master
            n = ID_MASTER  # master
        else:
            n = ID_SLAVE  # slave

        # Execute one time step within the environment
        self.take_action(action[ID_MOVES], n)

        # Wait a few seconds if on real k8s cluster
        if self.k8s:
            if action[ID_MOVES] != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_replicas is False \
                    and self.constraint_max_pod_replicas is False:
                # logging.info('[Step {}] | Waiting {} seconds for enabling action ...'
                # .format(self.current_step, self.waiting_period))
                time.sleep(self.waiting_period)  # Wait a few seconds...

        # Update observation before reward calculation:
        if self.k8s:  # k8s cluster
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:  # simulation
            self.simulation_update()

        # Get reward
        reward = self.get_reward

        # Update Infos
        self.total_reward += reward
        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency)

        # Print Step and Total Reward
        # if self.current_step == MAX_STEPS:
        # Log action for each deployment safely
        action_desc = []
        for i, act in enumerate(action):
            if i < len(DEPLOYMENTS) and act < len(MOVES):
                action_desc.append(f"{DEPLOYMENTS[i]}:{MOVES[act]}")
            else:
                action_desc.append(f"Service{i}:Action{act}")
        
        logging.info('[Step {}] | Actions: {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, ', '.join(action_desc), reward, self.total_reward))

        ob = self.get_state()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.use_graph:
            self.save_obs_to_csv(self.obs_csv, np.array(ob), date, self.deploymentList[0].latency)

        self.info = dict(
            total_reward=self.total_reward,
        )

        # Update Reward Keywords
        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        if self.current_step == MAX_STEPS:
            self.episode_count += 1
            self.execution_time = time.time() - self.time_start
            save_to_csv(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)

        # return ob, reward, terminated, truncated, self.info
        terminated = self.episode_over
        truncated = False  # Add support for truncation if needed later
        if self.use_graph:
            return ob, reward, terminated, truncated, self.info
        return np.array(ob), reward, terminated, truncated, self.info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        info (dict): auxiliary information
        """
        if seed is not None:
            self.seed(seed)
            
        self.current_step = 0
        self.episode_over = False
        self.total_reward = 0
        self.avg_pods = []
        self.avg_latency = []

        self.constraint_max_pod_replicas = False
        self.constraint_min_pod_replicas = False

        # Deployment Data
        self.deploymentList = get_redis_deployment_list(self.k8s, self.min_pods, self.max_pods)

        state = self.get_state()
        if self.use_graph:
            return state, {}
        return np.array(state), {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    def take_action(self, action, id):
        self.current_step += 1

        # Stop if MAX_STEPS
        if self.current_step == MAX_STEPS:
            # logging.info('[Take Action] MAX STEPS achieved, ending ...')
            self.episode_over = True

        # ACTIONS
        if action == ACTION_DO_NOTHING:
            # logging.info("[Take Action] SELECTED ACTION: DO NOTHING ...")
            pass

        elif action == ACTION_ADD_1_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 1 Replica ...")
            self.deploymentList[id].deploy_pod_replicas(1, self)

        elif action == ACTION_ADD_2_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 2 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(2, self)

        elif action == ACTION_ADD_3_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 3 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(3, self)

        elif action == ACTION_ADD_4_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 4 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(4, self)

        elif action == ACTION_ADD_5_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 5 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(5, self)

        elif action == ACTION_ADD_6_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 6 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(6, self)

        elif action == ACTION_ADD_7_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: ADD 7 Replicas ...")
            self.deploymentList[id].deploy_pod_replicas(7, self)

        elif action == ACTION_TERMINATE_1_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 1 Replica ...")
            self.deploymentList[id].terminate_pod_replicas(1, self)

        elif action == ACTION_TERMINATE_2_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 2 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(2, self)

        elif action == ACTION_TERMINATE_3_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 3 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(3, self)

        elif action == ACTION_TERMINATE_4_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 4 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(4, self)

        elif action == ACTION_TERMINATE_5_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 5 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(5, self)

        elif action == ACTION_TERMINATE_6_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 6 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(6, self)

        elif action == ACTION_TERMINATE_7_REPLICA:
            # logging.info("[Take Action] SELECTED ACTION: TERMINATE 7 Replicas ...")
            self.deploymentList[id].terminate_pod_replicas(7, self)

        else:
            logging.info('[Take Action] Unrecognized Action: ' + str(action))

    @property
    def get_reward(self):
        """ Calculate Rewards """
        '''
        ob = self.get_state()
        logging.info('[Reward] | Master Pods: {} | CPU Usage: {} | MEM Usage: {} | Requests: {} | Response Time: {} | '
                     'Slave Pods: {} | CPU Usage: {} | MEM Usage: {} | Requests: {} | Response Time: {} |'.format(
            ob.__getitem__(0), ob.__getitem__(1), ob.__getitem__(2), ob.__getitem__(9), ob.__getitem__(10),
            ob.__getitem__(11), ob.__getitem__(12), ob.__getitem__(13), ob.__getitem__(20), ob.__getitem__(21), ))
        '''
        # Reward based on Keyword!
        if self.constraint_max_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -250  # penalty

        if self.constraint_min_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -250  # penalty

        # Reward Calculation
        reward = self.calculate_reward()
        # logging.info('[Get Reward] Reward: {} | Ob: {} |'.format(reward, ob))
        # logging.info('[Get Reward] Acc. Reward: {} |'.format(self.total_reward))

        return reward

    def get_state(self):
        # Observations: metrics - 3 Metrics!!
        # "number_pods"
        # "cpu"
        # "mem"
        # "requests"

        features = []
        for d in self.deploymentList:
            features.append([
                d.num_pods,
                d.desired_replicas,
                d.cpu_usage,
                d.mem_usage,
                d.received_traffic,
                d.transmit_traffic,
            ])
        if self.use_graph:
            # 使用動態圖特徵處理
            features_array = np.array(features, dtype=np.float32)
            num_active_nodes = len(features)
            
            # 填充節點特徵到最大節點數
            padded_nodes, node_mask = self.dynamic_graph.pad_node_features(features_array, num_active_nodes)
            
            # 獲取動態邊特徵
            edges, num_active_nodes = self._fetch_service_graph_dynamic()
            padded_edges, edge_mask = self.dynamic_graph.pad_edge_features(edges, len(edges))
            
            # 計算全局特徵
            total_pods = sum(d.num_pods for d in self.deploymentList)
            avg_cpu = np.mean([d.cpu_usage for d in self.deploymentList])
            avg_mem = np.mean([d.mem_usage for d in self.deploymentList])
            total_traffic = sum(d.received_traffic + d.transmit_traffic for d in self.deploymentList)
            global_features = np.array([total_pods, avg_cpu, avg_mem, total_traffic], dtype=np.float32)
            
            # 填充全局特徵
            padded_global = self.dynamic_graph.pad_global_features(global_features)
            
            # 動作遮罩
            mask = self._invalid_action_mask()

            return {
                'svc_df': padded_nodes,
                'node_df': np.zeros((1, 4), dtype=np.float32),  # 保留兼容性
                'edge_df': padded_edges,
                'flat_feats': padded_global,
                'node_mask': node_mask,
                'edge_mask': edge_mask,
                'invalid_action_mask': mask,
            }
        return tuple(np.array(features).flatten())

    def _invalid_action_mask(self):
        if self.use_graph:
            # 動態圖模式：填充到最大節點數
            max_nodes = self.dynamic_graph.config.max_nodes
            mask = []
            
            # 為活躍服務生成遮罩
            for i, d in enumerate(self.deploymentList):
                if i < max_nodes:
                    for action in range(self.num_actions):
                        if action == ACTION_DO_NOTHING:
                            mask.append(1)
                        elif ACTION_ADD_1_REPLICA <= action <= ACTION_ADD_7_REPLICA:
                            n = action
                            mask.append(1 if d.num_pods + n <= d.max_pods else 0)
                        else:
                            n = action - 7
                            mask.append(1 if d.num_pods - n >= d.min_pods else 0)
            
            # 為非活躍節點填充遮罩
            inactive_nodes = max_nodes - len(self.deploymentList)
            for i in range(inactive_nodes):
                for action in range(self.num_actions):
                    if action == ACTION_DO_NOTHING:
                        mask.append(1)
                    else:
                        mask.append(0)  # 非活躍節點不能執行其他動作
                        
            return np.array(mask, dtype=np.float32)
        else:
            # 原始模式
            mask = []
            for d in self.deploymentList:
                for action in range(self.num_actions):
                    if action == ACTION_DO_NOTHING:
                        mask.append(1)
                    elif ACTION_ADD_1_REPLICA <= action <= ACTION_ADD_7_REPLICA:
                        n = action
                        mask.append(1 if d.num_pods + n <= d.max_pods else 0)
                    else:
                        n = action - 7
                        mask.append(1 if d.num_pods - n >= d.min_pods else 0)
            return np.array(mask, dtype=np.float32)

    def get_observation_space(self):
        return spaces.Box(
                low=np.array([
                    self.min_pods,  # Number of Pods  -- master metrics
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- slave metrics
                    self.min_pods,  # Number of Pods -- slave metrics
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                ]), high=np.array([
                    self.max_pods,  # Number of Pods -- master metrics
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- slave metrics
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                ]),
                dtype=np.float32
            )

    # calculates the reward based on the objective
    def calculate_reward(self):
        reward = 0
        if self.goal_reward == COST:
            reward = get_cost_reward(self.deploymentList)
        elif self.goal_reward == LATENCY:
            reward = get_latency_reward_redis(ID_MASTER, self.deploymentList)

        return reward

    def simulation_update(self):
        if self.current_step == 1:
            # Get a random sample!
            if len(self.df) == 0:
                # If DataFrame is empty, use default values
                self.deploymentList[0].num_pods = 1
                self.deploymentList[0].num_previous_pods = 1
                self.deploymentList[1].num_pods = 1
                self.deploymentList[1].num_previous_pods = 1
                return
            
            sample = self.df.sample()
            # print(sample)

            if len(sample) > 0:
                self.deploymentList[0].num_pods = int(sample['redis-leader_num_pods'].values[0])
                self.deploymentList[0].num_previous_pods = int(sample['redis-leader_num_pods'].values[0])
                self.deploymentList[1].num_pods = int(sample['redis-follower_num_pods'].values[0])
                self.deploymentList[1].num_previous_pods = int(sample['redis-follower_num_pods'].values[0])
            else:
                # Fallback to default values
                self.deploymentList[0].num_pods = 1
                self.deploymentList[0].num_previous_pods = 1
                self.deploymentList[1].num_pods = 1
                self.deploymentList[1].num_previous_pods = 1

        else:
            leader_pods = self.deploymentList[0].num_pods
            leader_previous_pods = self.deploymentList[0].num_previous_pods
            follower_pods = self.deploymentList[1].num_pods
            follower_previous_pods = self.deploymentList[1].num_previous_pods

            diff_leader = leader_pods - leader_previous_pods
            diff_follower = follower_pods - follower_previous_pods

            self.df['diff-leader'] = self.df['redis-leader_num_pods'].diff()
            self.df['diff-follower'] = self.df['redis-follower_num_pods'].diff()

            data = self.df.loc[self.df['redis-leader_num_pods'] == leader_pods]

            data = data.loc[data['diff-leader'] == diff_leader]

            if data.size == 0:
                data = self.df.loc[self.df['redis-leader_num_pods'] == leader_pods]

            data = self.df.loc[self.df['redis-follower_num_pods'] == follower_pods]

            data = data.loc[data['diff-follower'] == diff_follower]

            if data.size == 0:
                data = self.df.loc[self.df['redis-follower_num_pods'] == follower_pods]

            if len(data) > 0:
                sample = data.sample()
            else:
                # Fallback: use random sample from full dataset
                sample = self.df.sample() if len(self.df) > 0 else None
            # print(sample)

        # Update deployment metrics with bounds checking
        if sample is not None and len(sample) > 0:
            self.deploymentList[0].cpu_usage = int(sample['redis-leader_cpu_usage'].values[0])
            self.deploymentList[0].mem_usage = int(sample['redis-leader_mem_usage'].values[0])
            self.deploymentList[0].received_traffic = int(sample['redis-leader_traffic_in'].values[0])
            self.deploymentList[0].transmit_traffic = int(sample['redis-leader_traffic_out'].values[0])
            self.deploymentList[0].latency = float(sample['redis-leader_latency'].values[0])

            self.deploymentList[1].cpu_usage = int(sample['redis-follower_cpu_usage'].values[0])
            self.deploymentList[1].mem_usage = int(sample['redis-follower_mem_usage'].values[0])
            self.deploymentList[1].received_traffic = int(sample['redis-follower_traffic_in'].values[0])
            self.deploymentList[1].transmit_traffic = int(sample['redis-follower_traffic_out'].values[0])
            self.deploymentList[1].latency = float(sample['redis-follower_latency'].values[0])
        else:
            # Use default values when no data is available
            for i in range(2):
                self.deploymentList[i].cpu_usage = 50
                self.deploymentList[i].mem_usage = 50
                self.deploymentList[i].received_traffic = 100
                self.deploymentList[i].transmit_traffic = 100
                self.deploymentList[i].latency = 1.0

        for d in self.deploymentList:
            # Update Desired replicas
            d.update_replicas()
        return

    def save_obs_to_csv(self, obs_file, obs, date, latency):
        file = open(obs_file, 'a+', newline='')  # append
        # file = open(file_name, 'w', newline='') # new
        fields = []
        with file:
            fields.append('date')
            for d in self.deploymentList:
                fields.append(d.name + '_num_pods')
                fields.append(d.name + '_desired_replicas')
                fields.append(d.name + '_cpu_usage')
                fields.append(d.name + '_mem_usage')
                fields.append(d.name + '_traffic_in')
                fields.append(d.name + '_traffic_out')
                fields.append(d.name + '_latency')

            '''
            fields = ['date', 'redis-leader_num_pods', 'redis-leader_desired_replicas', 'redis-leader_cpu_usage', 'redis-leader_mem_usage',
                      'redis-leader_cpu_request', 'redis-leader_mem_request', 'redis-leader_cpu_limit', 'redis-leader_mem_limit',
                      'redis-leader_traffic_in', 'redis-leader_traffic_out',
                      'redis-follower_num_pods', 'redis-follower_desired_replicas', 'redis-follower_cpu_usage',
                      'redis-follower_mem_usage', 'redis-follower_cpu_request', 'redis-follower_mem_request', 'redis-follower_cpu_limit',
                      'redis-follower_mem_limit', 'redis-follower_traffic_in', 'redis-follower_traffic_out']
            '''
            writer = csv.DictWriter(file, fieldnames=fields)
            # writer.writeheader() # write header

            writer.writerow(
                {'date': date,
                 'redis-leader_num_pods': int("{}".format(obs[0])),
                 'redis-leader_desired_replicas': int("{}".format(obs[1])),
                 'redis-leader_cpu_usage': int("{}".format(obs[2])),
                 'redis-leader_mem_usage': int("{}".format(obs[3])),
                 'redis-leader_traffic_in': int("{}".format(obs[4])),
                 'redis-leader_traffic_out': int("{}".format(obs[5])),
                 'redis-leader_latency': float("{:.3f}".format(latency)),
                 'redis-follower_num_pods': int("{}".format(obs[6])),
                 'redis-follower_desired_replicas': int("{}".format(obs[7])),
                 'redis-follower_cpu_usage': int("{}".format(obs[8])),
                 'redis-follower_mem_usage': int("{}".format(obs[9])),
                 'redis-follower_traffic_in': int("{}".format(obs[10])),
                 'redis-follower_traffic_out': int("{}".format(obs[11])),
                 'redis-follower_latency': float("{:.3f}".format(latency))
                 }
            )
        return
