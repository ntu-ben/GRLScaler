import csv
import datetime
from datetime import datetime
import logging
import os
import time
from pathlib import Path
from statistics import mean

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

# Number of Requests - Discrete Event
from .deployment import (
    get_max_cpu,
    get_max_mem,
    get_max_traffic,
    get_online_boutique_deployment_list,
    get_kiali_service_graph,
)
from .util import save_to_csv, get_num_pods, get_cost_reward, \
    get_latency_reward_online_boutique
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
DEPLOYMENTS = ["recommendationservice", "productcatalogservice", "cartservice", "adservice",
               "paymentservice", "shippingservice", "currencyservice", "redis-cart",
               "checkoutservice", "frontend", "emailservice"]

# Action Moves
MOVES = ["None", "Add-1", "Add-2", "Add-3", "Add-4", "Add-5", "Add-6", "Add-7",
         "Stop-1", "Stop-2", "Stop-3", "Stop-4", "Stop-5", "Stop-6", "Stop-7"]

# IDs
ID_DEPLOYMENTS = 0
ID_MOVES = 1

ID_recommendation = 0
ID_product_catalog = 1
ID_cart_service = 2
ID_ad_service = 3
ID_payment_service = 4
ID_shipping_service = 5
ID_currency_service = 6
ID_redis_cart = 7
ID_checkout_service = 8
ID_frontend = 9
ID_email = 10

# Reward objectives
LATENCY = 'latency'
COST = 'cost'


class OnlineBoutique(gym.Env):
    """Horizontal Scaling for Online Boutique in Kubernetes - an OpenAI gym environment"""

    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, k8s=False, goal_reward="cost", waiting_period=5.0, use_graph=False, dataset_path=None):
        # Define action and observation space
        # They must be gym.spaces objects

        super(OnlineBoutique, self).__init__()

        self.k8s = k8s
        self.name = "online_boutique_gym"
        self.__version__ = "0.0.1"
        self.seed()
        self.goal_reward = goal_reward
        self.use_graph = use_graph
        self.waiting_period = waiting_period  # seconds to wait after action

        # 初始化動態圖空間管理器
        if self.use_graph:
            config = DynamicGraphConfig(
                max_nodes=20,  # 支援擴展到20個節點
                max_edges=400,  # 20*20
                node_feat_dim=6,
                edge_feat_dim=7,
                global_feat_dim=4
            )
            self.dynamic_graph = DynamicGraphSpace(config)

        logging.info("[Init] Env: {} | K8s: {} | Version {} |".format(self.name, self.k8s, self.__version__))

        # Current Step
        self.current_step = 0
        
        # Action history tracking for diagnostics
        self.action_history = []
        self.action_log_file = None
        self._init_action_logging()

        # Actions identified by integers 0-n -> 15 actions!
        self.num_actions = 15

        # Multi-Discrete - output an action for each service simultaneously
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
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        # Logging Deployment
        for d in self.deploymentList:
            d.print_deployment()

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
        import os

        if dataset_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            dataset_path = os.path.join(
                base_dir,
                "datasets",
                "real",
                self.deploymentList[0].namespace,
                "v1",
                "online_boutique_gym_observation.csv",
            )

        if not self.k8s:
            print(f"[INFO] Loading dataset from: {dataset_path}")
            self.df = pd.read_csv(dataset_path)
        else:
            self.df = pd.DataFrame()


    def _fetch_service_graph(self):
        from gnnrl.core.utils.kiali_client import fetch_service_graph

        nodes, edge_df = fetch_service_graph(namespace="onlineboutique")
        
        if not self.use_graph:
            # 兼容原本的固定圖模式
            index = {name: DEPLOYMENTS.index(name) for name in DEPLOYMENTS if name in nodes}
            num = len(DEPLOYMENTS)
            adj = np.zeros((num, num), dtype=np.float32)
            edges = []

            for _, row in edge_df.iterrows():
                src_name = nodes[row["src"]]
                dst_name = nodes[row["dst"]]
                if src_name in index and dst_name in index:
                    s = index[src_name]
                    d = index[dst_name]
                    adj[s, d] = 1.0
                    edges.append([s, d, 1.0, row["qps"], row["p95"], row["err_rate"], row.get("mtls", 0)])

            max_edges = num * num
            while len(edges) < max_edges:
                edges.append([0, 0, 0, 0, 0, 0, 0])
            edges = np.array(edges[:max_edges], dtype=np.float32)
            return adj, edges
        
        # 動態圖模式：支援可變節點數量
        # 更新節點映射
        active_services = [name for name in nodes if name in DEPLOYMENTS or name.startswith('additional-')]
        node_mapping = self.dynamic_graph.update_node_mapping(active_services)
        
        # 建構動態邊列表
        edges = []
        for _, row in edge_df.iterrows():
            src_name = nodes[row["src"]]
            dst_name = nodes[row["dst"]]
            if src_name in node_mapping and dst_name in node_mapping:
                s = node_mapping[src_name]
                d = node_mapping[dst_name]
                # [src, dst, active, qps, p95, err_rate, mtls]
                edges.append([s, d, 1.0, row["qps"], row["p95"], row["err_rate"], row.get("mtls", 0)])
        
        # 使用動態圖空間填充邊特徵
        edge_array = np.array(edges, dtype=np.float32) if edges else np.zeros((0, 7), dtype=np.float32)
        padded_edges, edge_mask = self.dynamic_graph.pad_edge_features(edge_array, len(edges))
        
        return padded_edges, edge_mask, len(active_services)



    # revision here!
    def step(self, action):
        if self.current_step == 1:
            if not self.k8s:
                self.simulation_update()

            self.time_start = time.time()

        # Handle multi-discrete action space
        if hasattr(action, '__len__') and len(action) > 1:
            # Multi-discrete action: each service gets its own action
            service_actions = action
        else:
            # Single action: convert to multi-discrete format
            service_actions = [action] * len(DEPLOYMENTS)
        
        # Process each service action
        for service_idx, service_action in enumerate(service_actions):
            if service_idx >= len(DEPLOYMENTS):
                break
                
            n = service_idx
            move_action = service_action

        # For simplicity, only execute the first non-zero action
        executed_action = None
        executed_service = None
        
        for service_idx, service_action in enumerate(service_actions):
            if service_idx >= len(DEPLOYMENTS):
                break
            if service_action != 0:  # Non-zero action
                executed_action = service_action
                executed_service = service_idx
                break
        
        # If no non-zero action, use the first action
        if executed_action is None:
            executed_action = service_actions[0] if service_actions else 0
            executed_service = 0
        
        # Record old replicas for logging
        old_replicas = self.deploymentList[executed_service].desired_replicas
        
        # Execute one time step within the environment
        self.take_action(executed_action, executed_service)
        
        # Record new replicas for logging
        new_replicas = self.deploymentList[executed_service].desired_replicas

        # Wait a few seconds if on real k8s cluster
        if self.k8s:
            if executed_action != ACTION_DO_NOTHING \
                    and self.constraint_min_pod_replicas is False \
                    and self.constraint_max_pod_replicas is False:
                # logging.info('[Step {}] | Waiting {} seconds for enabling action ...'
                # .format(self.current_step, self.waiting_period))
                time.sleep(self.waiting_period)  # Wait a few seconds...

        # Update observation before reward calculation:
        if self.k8s:  # k8s cluster
            for d in self.deploymentList:
                d.update_obs_k8s()
        else:
            self.simulation_update()

        # Get reward
        reward = self.get_reward
        self.total_reward += reward

        # Log action for diagnostics
        deployment_name = DEPLOYMENTS[executed_service] if executed_service < len(DEPLOYMENTS) else f"deployment_{executed_service}"
        obs_for_logging = self.get_state()
        self._log_action(self.current_step, [executed_service, executed_action], executed_service, deployment_name, 
                        old_replicas, new_replicas, reward, obs_for_logging)

        self.avg_pods.append(get_num_pods(self.deploymentList))
        self.avg_latency.append(self.deploymentList[0].latency)

        # Print Step and Total Reward
        # if self.current_step == MAX_STEPS:
        logging.info('[Step {}] | Action (Deployment): {} | Action (Move): {} | Reward: {} | Total Reward: {}'.format(
            self.current_step, DEPLOYMENTS[executed_service], MOVES[executed_action], reward, self.total_reward))

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

            # logging.info('Avg. latency : {} ', float("{:.3f}".format(mean(self.avg_latency))))
            save_to_csv(self.file_results, self.episode_count, mean(self.avg_pods), mean(self.avg_latency),
                        self.total_reward, self.execution_time)

        # return ob, reward, terminated, truncated, self.info (gymnasium format)
        terminated = self.episode_over
        truncated = False  # We don't use truncation in this environment
        if self.use_graph:
            return ob, reward, terminated, truncated, self.info
        return np.array(ob), reward, terminated, truncated, self.info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _init_action_logging(self):
        """Initialize CSV logging for action history"""
        # Create logs directory if it doesn't exist
        log_root = Path(os.getenv('LOG_ROOT', str(Path(__file__).resolve().parents[2] / 'logs')))
        log_dir = log_root / 'gnnrl' / 'actions'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create action history CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"action_history_{timestamp}.csv"
        self.action_log_file = str(log_dir / csv_filename)
        
        # Write CSV header
        with open(self.action_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'timestamp', 'deployment_id', 'deployment_name', 
                'action_type', 'action_value', 'old_replicas', 'new_replicas',
                'reward', 'avg_latency', 'total_pods', 'cpu_usage', 'mem_usage'
            ])
        
        logging.info(f"Action history logging initialized: {self.action_log_file}")
    
    def _log_action(self, step, action, deployment_id, deployment_name, 
                   old_replicas, new_replicas, reward, obs):
        """Log action details to CSV file"""
        if self.action_log_file is None:
            return
            
        # Extract observation metrics for logging
        avg_latency = getattr(self.deploymentList[9], 'latency', 0) if len(self.deploymentList) > 9 else 0
        total_pods = sum(d.num_pods for d in self.deploymentList)
        avg_cpu = mean([d.cpu_usage for d in self.deploymentList]) if self.deploymentList else 0
        avg_mem = mean([d.mem_usage for d in self.deploymentList]) if self.deploymentList else 0
        
        # Write to CSV
        try:
            with open(self.action_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, datetime.now().isoformat(), deployment_id, deployment_name,
                    action[1], action[1], old_replicas, new_replicas,
                    reward, avg_latency, total_pods, avg_cpu, avg_mem
                ])
        except Exception as e:
            logging.warning(f"Failed to log action: {e}")

    def reset(self, *, seed=None, options=None):
        """
        Reset the state of the environment and returns an initial observation.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional options
            
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
        self.deploymentList = get_online_boutique_deployment_list(self.k8s, self.min_pods, self.max_pods)

        state = self.get_state()
        info = {}
        if self.use_graph:
            return state, info
        return np.array(state), info

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
        # Reward based on Keyword!
        if self.constraint_max_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -3000  # penalty

        if self.constraint_min_pod_replicas:
            if self.goal_reward == COST:
                return -1  # penalty
            elif self.goal_reward == LATENCY:
                return -3000  # penalty

        # Reward Calculation
        reward = self.calculate_reward()
        return reward

    def get_state(self):
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
            # 使用動態圖系統
            padded_edges, edge_mask, num_active_nodes = self._fetch_service_graph()
            features_array = np.array(features, dtype=np.float32)
            
            # 填充節點特徵到最大節點數
            padded_nodes, node_mask = self.dynamic_graph.pad_node_features(features_array, num_active_nodes)
            
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
                'node_df': np.zeros((1, 4), dtype=np.float32),  # 添加兼容的 node_df
                'edge_df': padded_edges,
                'flat_feats': padded_global,  # 改名為 flat_feats 以匹配觀察空間
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
            
            # 填充到最大節點數（非活躍節點的所有動作都無效）
            current_size = len(self.deploymentList) * self.num_actions
            padding_size = max_nodes * self.num_actions - current_size
            mask.extend([0] * padding_size)  # 非活躍節點的動作無效
            
            return np.array(mask, dtype=np.float32)
        else:
            # 非圖模式：原有邏輯
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
                    self.min_pods,  # Number of Pods  -- 1) recommendationservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 2) productcatalogservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 3) cartservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 4) adservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 5) paymentservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 6) shippingservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 7) currencyservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 8) redis-cart
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 9) checkoutservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 10) frontend
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                    self.min_pods,  # Number of Pods -- 11) emailservice
                    self.min_pods,  # Desired Replicas
                    0,  # CPU Usage (in m)
                    0,  # MEM Usage (in MiB)
                    0,  # Average Number of received traffic
                    0,  # Average Number of transmit traffic
                ]), high=np.array([
                    self.max_pods,  # Number of Pods -- 1)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 2)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 3)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 4)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 5)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 6)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 7)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 8)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 9)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 10)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                    self.max_pods,  # Number of Pods -- 11)
                    self.max_pods,  # Desired Replicas
                    get_max_cpu(),  # CPU Usage (in m)
                    get_max_mem(),  # MEM Usage (in MiB)
                    get_max_traffic(),  # Average Number of received traffic
                    get_max_traffic(),  # Average Number of transmit traffic
                ]),
                dtype=np.float32
            )

    # calculates the desired replica count based on a target metric utilization
    def calculate_reward(self):
        # Calculate Number of desired Replicas
        reward = 0
        if self.goal_reward == COST:
            reward = get_cost_reward(self.deploymentList)
        elif self.goal_reward == LATENCY:
            reward = get_latency_reward_online_boutique(ID_recommendation, self.deploymentList)

        return reward

    def simulation_update(self):
        if self.current_step == 1:
            # Get a random sample!
            sample = self.df.sample()
            # print(sample)

            for i in range(len(DEPLOYMENTS)):
                self.deploymentList[i].num_pods = int(sample[DEPLOYMENTS[i] + '_num_pods'].values[0])
                self.deploymentList[i].num_previous_pods = int(sample[DEPLOYMENTS[i] + '_num_pods'].values[0])

        else:
            pods = []
            previous_pods = []
            diff = []
            for i in range(len(DEPLOYMENTS)):
                pods.append(self.deploymentList[i].num_pods)
                previous_pods.append(self.deploymentList[i].num_previous_pods)
                aux = pods[i] - previous_pods[i]
                diff.append(aux)
                self.df['diff-' + DEPLOYMENTS[i]] = self.df[DEPLOYMENTS[i] + '_num_pods'].diff()

            # print(pods)
            # print(previous_pods)
            # print(diff)
            # print(self.df_aggr)

            data = 0
            for i in range(len(DEPLOYMENTS)):
                data = self.df.loc[self.df[DEPLOYMENTS[i] + '_num_pods'] == pods[i]]
                data = data.loc[data['diff-' + DEPLOYMENTS[i]] == diff[i]]
                if data.size == 0:
                    data = self.df.loc[self.df[DEPLOYMENTS[i] + '_num_pods'] == pods[i]]

            sample = data.sample()
            # print(sample)

        for i in range(len(DEPLOYMENTS)):
            self.deploymentList[i].cpu_usage = int(sample[DEPLOYMENTS[i] + '_cpu_usage'].values[0])
            self.deploymentList[i].mem_usage = int(sample[DEPLOYMENTS[i] + '_mem_usage'].values[0])
            self.deploymentList[i].received_traffic = int(sample[DEPLOYMENTS[i] + '_traffic_in'].values[0])
            self.deploymentList[i].transmit_traffic = int(sample[DEPLOYMENTS[i] + '_traffic_out'].values[0])
            self.deploymentList[i].latency = float("{:.3f}".format(sample[DEPLOYMENTS[i] + '_latency'].values[0]))

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

            # TO ALTER!
            # DEPLOYMENTS = ["recommendationservice", "productcatalogservice", "cartservice",
            # "adservice", "paymentservice", "shippingservice", "currencyservice",
            # "redis-cart", "checkoutservice", "frontend", "emailservice"]

            writer.writerow(
                {'date': date,
                 'recommendationservice_num_pods': int("{}".format(obs[0])),
                 'recommendationservice_desired_replicas': int("{}".format(obs[1])),
                 'recommendationservice_cpu_usage': int("{}".format(obs[2])),
                 'recommendationservice_mem_usage': int("{}".format(obs[3])),
                 'recommendationservice_traffic_in': int("{}".format(obs[4])),
                 'recommendationservice_traffic_out': int("{}".format(obs[5])),
                 'recommendationservice_latency': float("{:.3f}".format(latency)),

                 'productcatalogservice_num_pods': int("{}".format(obs[6])),
                 'productcatalogservice_desired_replicas': int("{}".format(obs[7])),
                 'productcatalogservice_cpu_usage': int("{}".format(obs[8])),
                 'productcatalogservice_mem_usage': int("{}".format(obs[9])),
                 'productcatalogservice_traffic_in': int("{}".format(obs[10])),
                 'productcatalogservice_traffic_out': int("{}".format(obs[11])),
                 'productcatalogservice_latency': float("{:.3f}".format(latency)),

                 'cartservice_num_pods': int("{}".format(obs[12])),
                 'cartservice_desired_replicas': int("{}".format(obs[13])),
                 'cartservice_cpu_usage': int("{}".format(obs[14])),
                 'cartservice_mem_usage': int("{}".format(obs[15])),
                 'cartservice_traffic_in': int("{}".format(obs[16])),
                 'cartservice_traffic_out': int("{}".format(obs[17])),
                 'cartservice_latency': float("{:.3f}".format(latency)),

                 'adservice_num_pods': int("{}".format(obs[18])),
                 'adservice_desired_replicas': int("{}".format(obs[19])),
                 'adservice_cpu_usage': int("{}".format(obs[20])),
                 'adservice_mem_usage': int("{}".format(obs[21])),
                 'adservice_traffic_in': int("{}".format(obs[22])),
                 'adservice_traffic_out': int("{}".format(obs[23])),
                 'adservice_latency': float("{:.3f}".format(latency)),

                 'paymentservice_num_pods': int("{}".format(obs[24])),
                 'paymentservice_desired_replicas': int("{}".format(obs[25])),
                 'paymentservice_cpu_usage': int("{}".format(obs[26])),
                 'paymentservice_mem_usage': int("{}".format(obs[27])),
                 'paymentservice_traffic_in': int("{}".format(obs[28])),
                 'paymentservice_traffic_out': int("{}".format(obs[29])),
                 'paymentservice_latency': float("{:.3f}".format(latency)),

                 'shippingservice_num_pods': int("{}".format(obs[30])),
                 'shippingservice_desired_replicas': int("{}".format(obs[31])),
                 'shippingservice_cpu_usage': int("{}".format(obs[32])),
                 'shippingservice_mem_usage': int("{}".format(obs[33])),
                 'shippingservice_traffic_in': int("{}".format(obs[34])),
                 'shippingservice_traffic_out': int("{}".format(obs[35])),
                 'shippingservice_latency': float("{:.3f}".format(latency)),

                 'currencyservice_num_pods': int("{}".format(obs[36])),
                 'currencyservice_desired_replicas': int("{}".format(obs[37])),
                 'currencyservice_cpu_usage': int("{}".format(obs[38])),
                 'currencyservice_mem_usage': int("{}".format(obs[39])),
                 'currencyservice_traffic_in': int("{}".format(obs[40])),
                 'currencyservice_traffic_out': int("{}".format(obs[41])),
                 'currencyservice_latency': float("{:.3f}".format(latency)),

                 'redis-cart_num_pods': int("{}".format(obs[42])),
                 'redis-cart_desired_replicas': int("{}".format(obs[43])),
                 'redis-cart_cpu_usage': int("{}".format(obs[44])),
                 'redis-cart_mem_usage': int("{}".format(obs[45])),
                 'redis-cart_traffic_in': int("{}".format(obs[46])),
                 'redis-cart_traffic_out': int("{}".format(obs[47])),
                 'redis-cart_latency': float("{:.3f}".format(latency)),

                 'checkoutservice_num_pods': int("{}".format(obs[48])),
                 'checkoutservice_desired_replicas': int("{}".format(obs[49])),
                 'checkoutservice_cpu_usage': int("{}".format(obs[50])),
                 'checkoutservice_mem_usage': int("{}".format(obs[51])),
                 'checkoutservice_traffic_in': int("{}".format(obs[52])),
                 'checkoutservice_traffic_out': int("{}".format(obs[53])),
                 'checkoutservice_latency': float("{:.3f}".format(latency)),

                 'frontend_num_pods': int("{}".format(obs[54])),
                 'frontend_desired_replicas': int("{}".format(obs[55])),
                 'frontend_cpu_usage': int("{}".format(obs[56])),
                 'frontend_mem_usage': int("{}".format(obs[57])),
                 'frontend_traffic_in': int("{}".format(obs[58])),
                 'frontend_traffic_out': int("{}".format(obs[59])),
                 'frontend_latency': float("{:.3f}".format(latency)),

                 'emailservice_num_pods': int("{}".format(obs[60])),
                 'emailservice_desired_replicas': int("{}".format(obs[61])),
                 'emailservice_cpu_usage': int("{}".format(obs[62])),
                 'emailservice_mem_usage': int("{}".format(obs[63])),
                 'emailservice_traffic_in': int("{}".format(obs[64])),
                 'emailservice_traffic_out': int("{}".format(obs[65])),
                 'emailservice_latency': float("{:.3f}".format(latency))
                 }
            )
        return
