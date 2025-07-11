# GNNRL 完整運作過程指南

## 系統架構概述

GNNRL (Graph Neural Network Reinforcement Learning) 是一個基於圖神經網路的強化學習系統，專門用於Kubernetes環境中的動態資源調度。系統支援真正的動態圖結構，能夠處理節點增減的情況。

## 🏗️ 核心組件架構

### 1. 動態圖管理層 (`DynamicGraphSpace`)
**檔案**: `gnnrl/core/envs/dynamic_graph_space.py`

```python
class DynamicGraphSpace:
    """動態圖空間管理器，支援可變節點和邊數量"""
    
    def __init__(self, config: DynamicGraphConfig):
        self.config = config
        self.node_mapping: Dict[str, int] = {}
        self.edge_mapping: Dict[tuple, int] = {}
        
    def update_node_mapping(self, service_names: list) -> Dict[str, int]:
        """更新節點映射，處理服務增減"""
        # 實現節點ID的動態分配
        
    def pad_node_features(self, features: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """填充節點特徵到最大維度，返回特徵和遮罩"""
        
    def pad_edge_features(self, edges: np.ndarray, num_edges: int) -> Tuple[np.ndarray, np.ndarray]:
        """填充邊特徵到最大維度，返回特徵和遮罩"""
```

**關鍵特性**:
- 支援最大20個節點的動態擴展
- 自動處理節點映射變化
- 提供padding和masking機制
- 邊特徵包含7個維度：[src, dst, active, qps, p95, err_rate, mtls_percent]

### 2. 環境層 (`OnlineBoutique`)
**檔案**: `gnnrl/core/envs/online_boutique.py`

```python
class OnlineBoutique(gym.Env):
    """OnlineBoutique Kubernetes環境"""
    
    def __init__(self, k8s=False, goal_reward="cost", use_graph=False):
        # 初始化動態圖空間
        if self.use_graph:
            config = DynamicGraphConfig(max_nodes=20, max_edges=400, ...)
            self.dynamic_graph = DynamicGraphSpace(config)
    
    def _fetch_service_graph(self):
        """從Kiali獲取服務圖數據"""
        # 支援動態節點數量
        active_services = [name for name in nodes if name in DEPLOYMENTS]
        node_mapping = self.dynamic_graph.update_node_mapping(active_services)
        return padded_edges, edge_mask, len(active_services)
    
    def get_state(self):
        """獲取環境狀態"""
        if self.use_graph:
            return {
                'svc_df': padded_nodes,      # 節點特徵 (max_nodes, 6)
                'edge_df': padded_edges,     # 邊特徵 (max_edges, 7)
                'global_feats': padded_global, # 全局特徵 (4,)
                'node_mask': node_mask,      # 節點遮罩
                'edge_mask': edge_mask,      # 邊遮罩
                'num_nodes': num_active_nodes,
                'invalid_action_mask': mask,
            }
```

**關鍵特性**:
- 支援11個固定服務 + 動態擴展服務
- 動作空間：15個動作 × 11個服務 = 165維
- 真實Kiali數據獲取和處理
- 動態觀察空間適配

### 3. 時序圖編碼器 (`DynamicTGNEncoder`)
**檔案**: `gnnrl/encoders/tgn_encoder.py`

```python
class DynamicTGNEncoder(nn.Module):
    """支援動態節點映射的TGN編碼器"""
    
    def __init__(self, max_nodes: int, in_dim: int, memory_dim: int = 32):
        self.memory = TGNMemory(num_nodes=max_nodes, ...)
        self.conv = TransformerConv(in_dim, memory_dim, heads=2)
        self.node_mapping: Dict[str, int] = {}
        
    def update_node_mapping(self, service_names: list) -> Dict[str, int]:
        """更新節點映射並重新分配記憶體"""
        if old_mapping != new_mapping:
            self._remap_memory(old_mapping, new_mapping)
            
    def _remap_memory(self, old_mapping: Dict[str, int], new_mapping: Dict[str, int]):
        """重新映射TGN記憶體狀態"""
        # 保存現有服務的記憶體狀態
        # 重置記憶體並恢復保存的狀態
        
    def forward(self, edge_data, node_features, edge_mask, node_mask):
        """動態圖前向傳播"""
        # 更新時序記憶體
        # 應用Transformer卷積
        # 返回節點表示
```

**關鍵特性**:
- 支援節點映射變化時的記憶體重新分配
- 時序記憶體機制保持服務間的歷史依賴
- Transformer注意力機制捕捉服務間關係

### 4. 策略網路 (`PPO_GNN`)
**檔案**: `gnnrl/core/agents/ppo_gnn.py`

```python
class PPO_GNN:
    """基於GNN的PPO代理"""
    
    def __init__(self, obs_space, action_space, use_tgn=False):
        if use_tgn:
            self.tgn_encoder = DynamicTGNEncoder(max_nodes=20, in_dim=6)
        self.gnn_encoder = GNNEncoder(node_dim=6, edge_dim=7)
        self.policy_head = PolicyHead(hidden_dim=128, action_dim=action_space.nvec)
        
    def forward(self, obs):
        """前向傳播"""
        # 1. TGN編碼時序信息
        if self.use_tgn:
            temporal_features = self.tgn_encoder(
                obs['edge_df'], obs['svc_df'], 
                obs['edge_mask'], obs['node_mask']
            )
        
        # 2. GNN編碼圖結構
        graph_features = self.gnn_encoder(
            obs['svc_df'], obs['edge_df'], obs['node_mask']
        )
        
        # 3. 策略和價值預測
        actions, values = self.policy_head(graph_features)
        return actions, values
```

## 🔄 完整運作流程

### Phase 1: 環境初始化
```python
# 1. 創建環境實例
env = OnlineBoutique(k8s=True, use_graph=True, goal_reward="latency")

# 2. 初始化動態圖空間
config = DynamicGraphConfig(max_nodes=20, max_edges=400)
env.dynamic_graph = DynamicGraphSpace(config)

# 3. 初始化PPO代理
agent = PPO_GNN(env.observation_space, env.action_space, use_tgn=True)
```

### Phase 2: 數據獲取與預處理
```python
def step_data_flow():
    # 1. 從Kiali獲取服務圖
    nodes, edge_df = fetch_service_graph(namespace="onlineboutique")
    
    # 2. 更新節點映射
    active_services = [name for name in nodes if name in DEPLOYMENTS]
    node_mapping = env.dynamic_graph.update_node_mapping(active_services)
    
    # 3. 建構邊特徵
    edges = []
    for _, row in edge_df.iterrows():
        src_name, dst_name = nodes[row["src"]], nodes[row["dst"]]
        if src_name in node_mapping and dst_name in node_mapping:
            s, d = node_mapping[src_name], node_mapping[dst_name]
            edges.append([s, d, 1.0, row["qps"], row["p95"], row["err_rate"], row.get("mtls", 0)])
    
    # 4. 填充到最大維度
    padded_edges, edge_mask = env.dynamic_graph.pad_edge_features(edges, len(edges))
    
    # 5. 獲取節點特徵
    node_features = []
    for d in env.deploymentList:
        node_features.append([d.num_pods, d.desired_replicas, d.cpu_usage, 
                            d.mem_usage, d.received_traffic, d.transmit_traffic])
    
    padded_nodes, node_mask = env.dynamic_graph.pad_node_features(node_features, len(active_services))
```

### Phase 3: 圖神經網路處理
```python
def gnn_processing_flow(obs):
    # 1. TGN時序編碼
    if agent.use_tgn:
        # 更新節點映射
        agent.tgn_encoder.update_node_mapping(active_services)
        
        # 時序特徵編碼
        temporal_features = agent.tgn_encoder.forward(
            obs['edge_df'],    # 邊特徵 (max_edges, 7)
            obs['svc_df'],     # 節點特徵 (max_nodes, 6)
            obs['edge_mask'],  # 邊遮罩
            obs['node_mask']   # 節點遮罩
        )
        
        # 更新TGN記憶體
        agent.tgn_encoder.update_memory(src, dst, timestamps, messages)
    
    # 2. GNN圖結構編碼
    graph_features = agent.gnn_encoder.forward(
        obs['svc_df'],      # 節點特徵
        obs['edge_df'],     # 邊特徵  
        obs['node_mask']    # 節點遮罩
    )
    
    # 3. 特徵融合
    if agent.use_tgn:
        combined_features = temporal_features + graph_features
    else:
        combined_features = graph_features
    
    return combined_features
```

### Phase 4: 策略決策
```python
def policy_decision_flow(combined_features, obs):
    # 1. 策略網路預測
    action_logits, value = agent.policy_head(combined_features)
    
    # 2. 應用動作遮罩
    masked_logits = action_logits.masked_fill(
        obs['invalid_action_mask'].bool(), float('-inf')
    )
    
    # 3. 採樣動作
    action_dist = torch.distributions.Categorical(logits=masked_logits)
    action = action_dist.sample()
    
    # 4. 執行動作
    obs, reward, done, info = env.step(action)
    
    return action, reward, obs
```

### Phase 5: 動作執行與環境更新
```python
def action_execution_flow(action):
    # 1. 解析動作
    deployment_id = action[0]  # 選擇的服務
    move_id = action[1]        # 選擇的動作類型
    
    # 2. 執行Kubernetes操作
    if move_id == ACTION_ADD_1_REPLICA:
        env.deploymentList[deployment_id].deploy_pod_replicas(1, env)
    elif move_id == ACTION_TERMINATE_1_REPLICA:
        env.deploymentList[deployment_id].terminate_pod_replicas(1, env)
    
    # 3. 等待Kubernetes更新
    if env.k8s and move_id != ACTION_DO_NOTHING:
        time.sleep(env.waiting_period)
    
    # 4. 更新觀察值
    for d in env.deploymentList:
        d.update_obs_k8s()  # 從Kubernetes獲取最新狀態
    
    # 5. 計算獎勵
    reward = env.get_reward  # 基於延遲或成本的獎勵
```

### Phase 6: 學習更新
```python
def learning_update_flow():
    # 1. 收集軌跡數據
    trajectories = []
    for step in range(max_steps):
        action, reward, next_obs = policy_decision_flow(features, obs)
        trajectories.append((obs, action, reward, next_obs))
        obs = next_obs
    
    # 2. 計算優勢函數
    advantages = compute_gae(trajectories, gamma=0.99, lambda_=0.95)
    
    # 3. PPO更新
    for epoch in range(ppo_epochs):
        # 策略損失
        policy_loss = compute_policy_loss(trajectories, advantages)
        
        # 價值損失
        value_loss = compute_value_loss(trajectories)
        
        # 總損失
        total_loss = policy_loss + value_loss
        
        # 反向傳播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

## 📊 關鍵數據流

### 觀察空間結構
```python
observation = {
    'svc_df': np.array(shape=(20, 6)),      # 節點特徵 [pods, replicas, cpu, mem, traffic_in, traffic_out]
    'edge_df': np.array(shape=(400, 7)),    # 邊特徵 [src, dst, active, qps, p95, err_rate, mtls]
    'global_feats': np.array(shape=(4,)),   # 全局特徵 [total_pods, avg_cpu, avg_mem, total_traffic]
    'node_mask': np.array(shape=(20,)),     # 節點有效性遮罩
    'edge_mask': np.array(shape=(400,)),    # 邊有效性遮罩
    'num_nodes': int,                       # 當前活躍節點數
    'invalid_action_mask': np.array(shape=(165,))  # 無效動作遮罩
}
```

### 動作空間結構
```python
action_space = MultiDiscrete([15] * 11)  # 11個服務，每個服務15種動作
# 動作類型：
# 0: DO_NOTHING
# 1-7: ADD_1_REPLICA to ADD_7_REPLICA  
# 8-14: TERMINATE_1_REPLICA to TERMINATE_7_REPLICA
```

## 🔧 系統配置

### 動態圖配置
```python
config = DynamicGraphConfig(
    max_nodes=20,           # 最大節點數
    max_edges=400,          # 最大邊數 (20*20)
    node_feat_dim=6,        # 節點特徵維度
    edge_feat_dim=7,        # 邊特徵維度
    global_feat_dim=4       # 全局特徵維度
)
```

### TGN配置
```python
tgn_config = {
    'max_nodes': 20,
    'in_dim': 6,
    'memory_dim': 32,
    'msg_dim': 32,
    'heads': 2
}
```

### PPO配置
```python
ppo_config = {
    'lr': 3e-4,
    'gamma': 0.99,
    'lambda_': 0.95,
    'clip_ratio': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5
}
```

## 🚀 運行範例

### 完整訓練流程
```python
# 1. 環境初始化
env = OnlineBoutique(k8s=True, use_graph=True, goal_reward="latency")
agent = PPO_GNN(env.observation_space, env.action_space, use_tgn=True)

# 2. 訓練循環
for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # 獲取動作
        action = agent.act(obs)
        
        # 執行動作
        next_obs, reward, done, info = env.step(action)
        
        # 存儲經驗
        agent.store_transition(obs, action, reward, next_obs, done)
        
        obs = next_obs
        total_reward += reward
        
        if done:
            break
    
    # 更新策略
    agent.update()
    
    print(f"Episode {episode}, Total Reward: {total_reward}")
```

## 🎯 性能優化

### 記憶體管理
- TGN記憶體狀態的有效重用
- 動態padding避免不必要的計算
- 遮罩機制減少無效特徵處理

### 計算效率
- 批量處理多個環境實例
- GPU並行化圖神經網路計算
- 異步Kubernetes API調用

### 收斂穩定性
- 動作遮罩避免無效動作
- 獎勵標準化改善訓練穩定性
- 梯度裁剪防止梯度爆炸

## 📈 評估指標

### 系統性能
- 平均延遲降低百分比
- 資源利用率提升
- 成本節約效果

### 學習效率
- 收斂速度（episodes to convergence）
- 樣本效率（sample efficiency）
- 策略穩定性

### 動態適應性
- 節點變化適應速度
- 記憶體重用效果
- 拓撲變化魯棒性

這個完整的運作過程展示了GNNRL系統如何從數據獲取、圖神經網路處理、策略決策到動作執行的全流程，實現了真正的動態圖結構支援和高效的強化學習訓練。