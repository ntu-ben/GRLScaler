# GRLScaler

本專案由 **國立台灣大學電機工程學研究所 NMLAB** 研究團隊維護，旨在研究於 Kubernetes 平台上使用強化學習進行自動擴縮。部分程式碼來自 [gym-hpa](https://github.com/jpedro1992/gym-hpa)，並在此基礎上加入圖神經網路與真實叢集測試工具。

建議使用 **Python 3.10** 執行本專案。

## 必要的 Kubernetes 設定

1. 安裝 Istio 與 [Kiali](https://kiali.io/)，確保 Prometheus 能存取 `/metrics`。
2. 為 `onlineboutique` 命名空間啟用 sidecar injection，使服務可由 Istio 監控。
3. 依需要安裝 Prometheus，其 Helm `values` 皆收錄於 [`macK8S/`](macK8S/)。

## 在一台或兩台主機上重現實驗

執行 `rl_batch_loadtest.py` 即可啟動訓練並串接 Locust 載入測試。

```bash
python rl_batch_loadtest.py --model grl --tag myrun
```

若採用兩台電腦進行分散式負載測試，將 `M1_HOST` 環境變數設為遠端 agent 的位址，腳本會自動呼叫該節點執行 Locust。

若想以手動方式啟動長時間壓測，可執行 `loadtest/locust_agent_manual.py`，
預設持續 24 小時。

所有測試紀錄會輸出至 `logs/<method>/<tag>/` 目錄，其中 `method` 可能為
`grl`、`gym`、`gwydion` 或 `hpa`。HPA baseline 的結果將存放在
`logs/hpa/<hpa-name>/`。

### 主要環境變數

在專案根目錄建立 `.env` 檔，內容可參考 `\.envTemplate`。所有腳本會從
`.env` 讀取設定，下表列出常用變數：

| 變數 | 用途 |
|------|------|
| `TARGET_HOST` | Locust 測試目標服務 URL |
| `M1_HOST` | 遠端 Locust agent 位址（選用） |
| `PROMETHEUS_URL` | gnn_rl 查詢 Prometheus 用 |
| `KUBE_HOST` | gnn_rl 連線至 Kubernetes proxy |
| `KIALI_URL` | 取得服務拓撲 |
| `LOCUST_RUN_TIME` | 每次 Locust 執行的持續時間 |
| `NAMESPACE_REDIS`、`NAMESPACE_ONLINEBOUTIQUE` | 各範例對應的命名空間 |

## 專案結構

```
gnn_rl/        # 強化學習策略與訓練程式
gnn_rl/envs/   # Gym 環境實作（原 gnn_rl_env）
loadtest/      # Locust 測試腳本與遠端 agent
macK8S/        # Kubernetes 設定檔（Istio、Kiali、Prometheus、HPA 等）
```

## 使用說明

以下範例展示如何在本機快速啟用 GNN + RL 自動擴縮器。

1. 建議使用 **Python 3.10**，先安裝相依套件（已改用 `gymnasium`）：

   ```bash
   pip install -e .
   pip install -r requirements.txt
   # 若已加入打包設定，亦可透過
   # pip install -e .
   # 安裝為可編輯模式，否則請手動設定
   # export PYTHONPATH=$(pwd)
   ```

2. 啟動資料收集器（需先設定 `PROMETHEUS_URL` 與 `KIALI_URL`）：

   ```bash
   python -m data_collector.kiali_prom --graph-url $KIALI_URL/api/namespaces/onlineboutique/graph \
       --metrics-url $PROMETHEUS_URL/api/v1/query
   ```

3. 另開終端執行訓練（連線至 K8s 叢集請加上 `--k8s`）：

   ```bash
   # Redis
   python scripts/train_gnnppo.py \
       --use-case redis \
       --dataset-path datasets/real/redis/v1/redis_gym_observation.csv \
       --model gat --steps 100000 --k8s

   # Online Boutique
   python scripts/train_gnnppo.py \
       --use-case online_boutique \
       --dataset-path datasets/real/onlineboutique/v1/online_boutique_gym_observation.csv \
       --model gat --steps 100000 --k8s
   ```

4. 訓練完成後可執行基準測試：

   ```bash
   python scripts/benchmark.py --steps 10000 --seeds 3
   ```

更多使用情境與真實叢集設定，請參考 [docs/Operating_Guide.md](docs/Operating_Guide.md)。

## 🚀 快速開始：GNN 模式實驗

本節提供簡化的 GNN 模式實驗指南，讓用戶能快速上手圖神經網路自動擴縮實驗。

### 前置需求

1. **Kubernetes 叢集**：確保已部署 OnlineBoutique 微服務
2. **Python 環境**：建議 Python 3.9+ 
3. **依賴套件**：
   ```bash
   pip install -e .
   pip install -r requirements.txt
   pip install sb3-contrib  # 必要的額外套件
   ```

### 環境設定

1. 複製環境變數模板：
   ```bash
   cp .envTemplate .env
   ```

2. 編輯 `.env` 檔，設定必要變數：
   ```bash
   # Kubernetes API endpoint (使用 kubectl proxy)
   KUBE_HOST=http://localhost:8001
   
   # Prometheus endpoint  
   PROMETHEUS_URL=http://localhost:9090/
   
   # Kiali endpoint (可選，用於服務拓撲)
   KIALI_URL=http://localhost:20001/kiali/
   
   # OnlineBoutique 命名空間
   NAMESPACE_ONLINEBOUTIQUE=onlineboutique
   ```

3. 啟動 kubectl proxy (在背景執行)：
   ```bash
   kubectl proxy --port=8001 &
   ```

### 實驗模式

#### 模式 1：模擬模式（推薦入門）
使用預存的資料集進行訓練，無需連接真實 K8s 叢集：

```bash
# 使用 OnlineBoutique 資料集進行 GNN 訓練
python scripts/train_gnnppo.py \
    --use-case online_boutique \
    --dataset-path datasets/real/onlineboutique/v1/online_boutique_gym_observation.csv \
    --model gat \
    --steps 10000

# 使用 Redis 資料集進行 GNN 訓練  
python scripts/train_gnnppo.py \
    --use-case redis \
    --dataset-path datasets/real/redis/v1/redis_gym_observation.csv \
    --model gcn \
    --steps 10000
```

#### 模式 2：即時 K8s 叢集模式
連接真實 Kubernetes 叢集進行即時訓練：

```bash
# OnlineBoutique + GNN + 真實 K8s 叢集
python scripts/train_gnnppo.py \
    --use-case online_boutique \
    --model gat \
    --steps 5000 \
    --k8s

# Redis + GNN + 真實 K8s 叢集
python scripts/train_gnnppo.py \
    --use-case redis \
    --model gcn \
    --steps 5000 \
    --k8s
```

#### 模式 3：簡化的實驗腳本
使用預建的實驗腳本進行快速測試：

```bash
# 基本實驗（模擬模式）
python run_onlineboutique_gnn.py

# 真實 K8s 叢集實驗
python run_onlineboutique_gnn.py --k8s

# 自訂參數實驗
python run_onlineboutique_gnn.py --k8s --steps 5000 --goal cost

# 檢視訓練日誌
ls runs/gnnppo/
```

### GNN 模型選項

| 模型類型 | 參數值 | 說明 |
|---------|--------|------|
| Graph Attention Network | `--model gat` | 使用注意力機制的圖神經網路 |
| Graph Convolutional Network | `--model gcn` | 標準圖卷積網路 |
| Dynamic Self-Attention | `--model dysat` | 動態自注意力網路 |

### 實驗參數調整

```bash
# 調整訓練步數
--steps 50000

# 調整 GNN 嵌入維度
# 需修改 scripts/train_gnnppo.py 中的 policy_kwargs

# 選擇不同的微服務應用
--use-case online_boutique  # 或 redis

# 啟用/停用 K8s 即時模式
--k8s  # 加上此參數連接真實叢集
```

### 實驗結果查看

1. **TensorBoard 日誌**：
   ```bash
   tensorboard --logdir runs/gnnppo/
   ```

2. **模型檔案**：
   - 訓練完成的模型存放在當前目錄
   - 檔名格式：`ppo_env_<app>_gym_goal_<goal>_k8s_<mode>_totalSteps_<steps>.zip`

3. **實驗日誌**：
   - 控制台輸出包含每步的獎勵、動作資訊
   - 即時 K8s 模式會顯示真實的容器指標

### 常見問題排解

1. **Kiali 連線錯誤**：
   ```
   ERROR:root:Kiali request failed: 404 Client Error
   ```
   - 這是正常現象，不影響 GNN 訓練
   - 僅影響服務拓撲圖的獲取

2. **觀測空間錯誤**：
   - 確保使用 `scripts/train_gnnppo.py` 而非舊版 `gnn_rl/run/run.py`
   - GNN 模式需要 Dict 類型的觀測空間

3. **依賴套件問題**：
   ```bash
   pip install torch torch-geometric stable-baselines3 sb3-contrib
   ```

### 效能基準測試

完成 GNN 訓練後，可執行基準測試比較不同方法：

```bash
# 比較 GNN vs 標準 RL vs HPA
python scripts/benchmark.py --steps 10000 --seeds 3
```

此測試會輸出包含 SLO 違反率、資源使用效率等指標的比較表格。

### 🔧 快速參考表

| 實驗目標 | 推薦命令 | 說明 |
|---------|----------|------|
| 初次體驗 GNN | `python run_onlineboutique_gnn.py` | 使用預建腳本快速測試 |
| 真實叢集快速測試 | `python run_onlineboutique_gnn.py --k8s` | 一鍵啟動 K8s 叢集實驗 |
| 進階 GNN 訓練 | `python scripts/train_gnnppo.py --use-case online_boutique --model gat --steps 10000 --k8s` | 使用完整 GNN 架構 |
| 效能比較 | `python scripts/benchmark.py --steps 10000 --seeds 3` | 比較不同自動擴縮方法 |

---

## 🚀 Quick Start: GNN Mode Experiments (English)

This section provides a simplified guide for GNN mode experiments, enabling users to quickly get started with graph neural network-based autoscaling experiments.

### Prerequisites

1. **Kubernetes Cluster**: Ensure OnlineBoutique microservices are deployed
2. **Python Environment**: Recommended Python 3.9+
3. **Dependencies**:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   pip install sb3-contrib  # Required additional package
   ```

### Environment Setup

1. Copy environment template:
   ```bash
   cp .envTemplate .env
   ```

2. Edit `.env` file with necessary variables:
   ```bash
   # Kubernetes API endpoint (using kubectl proxy)
   KUBE_HOST=http://localhost:8001
   
   # Prometheus endpoint  
   PROMETHEUS_URL=http://localhost:9090/
   
   # Kiali endpoint (optional, for service topology)
   KIALI_URL=http://localhost:20001/kiali/
   
   # OnlineBoutique namespace
   NAMESPACE_ONLINEBOUTIQUE=onlineboutique
   ```

3. Start kubectl proxy (run in background):
   ```bash
   kubectl proxy --port=8001 &
   ```

### Experiment Modes

#### Mode 1: Simulation Mode (Recommended for Beginners)
Train using pre-stored datasets without connecting to real K8s cluster:

```bash
# GNN training with OnlineBoutique dataset
python scripts/train_gnnppo.py \
    --use-case online_boutique \
    --dataset-path datasets/real/onlineboutique/v1/online_boutique_gym_observation.csv \
    --model gat \
    --steps 10000

# GNN training with Redis dataset  
python scripts/train_gnnppo.py \
    --use-case redis \
    --dataset-path datasets/real/redis/v1/redis_gym_observation.csv \
    --model gcn \
    --steps 10000
```

#### Mode 2: Live K8s Cluster Mode
Connect to real Kubernetes cluster for live training:

```bash
# OnlineBoutique + GNN + Live K8s cluster
python scripts/train_gnnppo.py \
    --use-case online_boutique \
    --model gat \
    --steps 5000 \
    --k8s

# Redis + GNN + Live K8s cluster
python scripts/train_gnnppo.py \
    --use-case redis \
    --model gcn \
    --steps 5000 \
    --k8s
```

#### Mode 3: Simplified Experiment Script
Use pre-built experiment scripts for quick testing:

```bash
# Basic experiment (simulation mode)
python run_onlineboutique_gnn.py

# Live K8s cluster experiment
python run_onlineboutique_gnn.py --k8s

# Custom parameters experiment
python run_onlineboutique_gnn.py --k8s --steps 5000 --goal cost

# View training logs
ls runs/gnnppo/
```

### Quick Reference

| Experiment Goal | Recommended Command | Description |
|----------------|-------------------|-------------|
| First GNN Experience | `python run_onlineboutique_gnn.py` | Quick test with pre-built script |
| Live Cluster Quick Test | `python run_onlineboutique_gnn.py --k8s` | One-click K8s cluster experiment |
| Advanced GNN Training | `python scripts/train_gnnppo.py --use-case online_boutique --model gat --steps 10000 --k8s` | Full GNN architecture |
| Performance Comparison | `python scripts/benchmark.py --steps 10000 --seeds 3` | Compare different autoscaling methods |

### Troubleshooting

1. **Kiali Connection Error**:
   ```
   ERROR:root:Kiali request failed: 404 Client Error
   ```
   - This is expected and doesn't affect GNN training
   - Only impacts service topology graph retrieval

2. **Observation Space Error**:
   - Use `scripts/train_gnnppo.py` instead of legacy `gnn_rl/run/run.py`
   - GNN mode requires Dict observation space

3. **Dependency Issues**:
   ```bash
   pip install torch torch-geometric stable-baselines3 sb3-contrib
   ```

4. **Reset Method Compatibility**:
   - If encountering gymnasium/stable-baselines3 compatibility issues
   - Use the simplified `run_onlineboutique_gnn.py` script

### Performance Benchmarking

After completing GNN training, run benchmark tests to compare different methods:

```bash
# Compare GNN vs Standard RL vs HPA
python scripts/benchmark.py --steps 10000 --seeds 3
```

This outputs comparison tables with metrics including SLO violation rates and resource efficiency.

## GNN + RL Autoscaler 架構指引

專案已將原 `gnn_rl_env` 環境整合至 `gnn_rl.envs`，可依照下列流程建置 GNN + RL 自動擴縮器。
詳細步驟與需求收錄於 [docs/GNN_RL_Autoscaler.md](docs/GNN_RL_Autoscaler.md)。

若需擴充資料拉取或特徵處理，可額外建立 `data_collector/`、`feature_builder/` 等
子模組，並在 `scripts/` 內撰寫訓練與評測腳本。

### 主要流程

1. **資料收集**：`data_collector` 每 30 秒抓取 edges、容器指標與節點資源。
2. **特徵轉換**：`feature_builder` 將呼叫關係與資源數據轉成 PyG `HeteroData`，同時擷取全域指標供 RL 使用。
3. **GNN 編碼器**：在 `gnn_rl/models` 中實作 `HeteroGAT` 或其他變體，核心程式位於
   [`models/gnn_encoder.py`](gnn_rl/models/gnn_encoder.py)，輸出服務與節點嵌入。
4. **RL 策略**：`gnn_rl/agents` 的 `GNNPPOPolicy` 將 GNN 向量與 scalar 特徵拼接，決定 `svc_id`、`node_type`、`Δreplicas` 與 `Δquota` 等動作。
5. **訓練與測試**：執行 `scripts/train_gnnppo.py` 或 `scripts/benchmark.py` 進行比較，結果輸出於 `results/`。

### Baseline 與評估

| 編碼器   | RL 演算法 | cfg 名稱 |
|---------|-----------|----------|
| None    | PPO       | `mlp_ppo`|
| HeteroGAT | PPO     | `gat_ppo`|
| HeteroGCN | PPO     | `gcn_ppo`|
| HeteroGAT | Discrete-SAC | `gat_sac`|

評估指標包含 `SLO_violate%`、`Resource Slack%`、`Scaling Lag` 與 CAF。`benchmark.py` 會重播四種載入情境並輸出統計表，以檢視不同模型的自動擴縮效果。

如需自行調整 Online Boutique 或 HPA 範例，可參考 `macK8S/HPA/README.md`。

更多真實叢集接入與操作步驟，請見 [docs/Operating_Guide.md](docs/Operating_Guide.md)。

---
本倉庫僅供研究與教學用途，歡迎提出 issue 與貢獻。
