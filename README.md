# GRLScaler - Graph Reinforcement Learning for Kubernetes Autoscaling

**GRLScaler** is a Kubernetes autoscaling system based on Graph Neural Network Reinforcement Learning, supporting intelligent scaling for microservices and distributed applications.

**GRLScaler** 是一個基於圖神經網絡強化學習的 Kubernetes 自動擴展系統，支援微服務和分散式應用的智能擴展。

## 📋 Table of Contents | 目錄

- [System Overview | 系統概述](#system-overview--系統概述)
- [Environment Setup | 環境準備](#environment-setup--環境準備)
- [Installation Guide | 安裝指南](#installation-guide--安裝指南)
- [Dataset Information | 數據集說明](#dataset-information--數據集說明)
- [Configuration | 配置](#configuration--配置)
- [Experiment Reproduction | 實驗復現](#experiment-reproduction--實驗復現)
- [Results Analysis | 結果分析](#results-analysis--結果分析)
- [Model Management | 模型管理](#model-management--模型管理)
- [Troubleshooting | 故障排除](#troubleshooting--故障排除)
- [Advanced Usage | 進階使用](#advanced-usage--進階使用)

## 📊 System Overview | 系統概述

GRLScaler provides comparative research on three autoscaling methods:

GRLScaler 提供三種自動擴展方法的比較研究：

1. **GNNRL** - Graph Neural Network Reinforcement Learning, considering inter-service dependencies | 圖神經網絡強化學習，考慮服務間依賴關係
2. **Gym-HPA** - Basic reinforcement learning using MLP policy | 基礎強化學習，使用 MLP 策略  
3. **K8s-HPA** - Native Kubernetes HPA baseline testing | Kubernetes 原生 HPA 基準測試

### Supported Test Environments | 支援的測試環境

- **OnlineBoutique** - Google microservices e-commerce platform (11 microservices) | Google 微服務電商平台 (11個微服務)
- **Redis** - Master-Slave in-memory database cluster | Master-Slave 內存數據庫集群

## 🛠️ Environment Setup | 環境準備

### System Requirements | 系統需求

#### Tested Hardware Configuration | 測試硬體配置

本實驗在以下配置上進行測試：
- **主機 1**: MacBook Pro M4 Pro - 運行 Kubernetes 集群
- **主機 2**: MacBook Pro M4 Pro - 執行負載測試

#### Software Requirements | 軟體需求

- **Kubernetes Cluster** (v1.20+ recommended) | **Kubernetes 集群** (建議 v1.20+)
- **Python 3.8+** (3.9-3.11 recommended) | **Python 3.8+** (建議 3.9-3.11)
- **Docker** 
- **Istio** (optional, for service mesh monitoring) | **Istio** (可選，用於服務網格監控)
- **Kiali** (optional, for graph topology visualization) | **Kiali** (可選，用於圖拓撲視覺化)

### Minimum Hardware Requirements | 最低硬體需求

- **CPU**: 4+ cores (8+ cores recommended for live K8s experiments) | **CPU**: 4+ 核心 (實時 K8s 實驗建議 8+ 核心)
- **RAM**: 8GB+ (16GB+ recommended for K8s cluster) | **RAM**: 8GB+ (K8s 集群建議 16GB+)
- **Storage**: 20GB+ available space | **儲存空間**: 20GB+ 可用空間
- **Network**: Stable connection between K8s cluster and load testing machines | **網路**: K8s 集群與負載測試機器間的穩定連接

#### Recommended Setup | 建議配置

對於最佳實驗效果，建議使用雙機配置：
- **K8s 主機**: 專門運行 Kubernetes 集群和微服務應用
- **負載測試主機**: 專門執行 Locust 負載測試，避免資源競爭

### Essential Components | 必要組件

```bash
# Check Kubernetes cluster | 檢查 Kubernetes 集群
kubectl cluster-info

# Check Python version | 檢查 Python 版本
python3 --version

# Check Docker | 檢查 Docker 
docker --version
```

## 📦 Installation Guide | 安裝指南

### 1. Clone Project | 克隆專案

```bash
git clone <repository-url>
cd GRLScaler
```

### 2. Install Dependencies | 安裝依賴

```bash
# Install core Python dependencies | 安裝核心 Python 依賴
pip install -r requirements.txt

# Install additional dependencies for GNNRL | 安裝 GNNRL 額外依賴
pip install torch-geometric httpx locust

# Install gym-hpa environment | 安裝 gym-hpa 環境
cd gym-hpa && pip install -e . && cd ..

# Install gnnrl environment modules | 安裝 gnnrl 環境模組
cd gnnrl/environments && pip install -e . && cd ../..

# Install main gnnrl modules | 安裝主要 gnnrl 模組
pip install -e .
```

### 3. Deploy Test Applications | 部署測試應用

#### Deploy OnlineBoutique | 部署 OnlineBoutique

```bash
# Deploy microservices e-commerce platform | 部署微服務電商平台
kubectl apply -f MicroServiceBenchmark/microservices-demo/kubernetes-manifests/

# Check deployment status | 檢查部署狀態
kubectl get pods -n onlineboutique
```

#### Deploy Redis Cluster | 部署 Redis 集群

```bash
# Deploy Redis Master-Slave | 部署 Redis Master-Slave
kubectl apply -f MicroServiceBenchmark/redis-cluster/redis-cluster.yaml

# Check Redis status | 檢查 Redis 狀態
kubectl get pods -n redis
```

### 4. Configure Monitoring (Optional) | 配置監控 (可選)

```bash
# Deploy Kiali (if using Istio) | 部署 Kiali (如果使用 Istio)
kubectl apply -f macK8S/istio/

# Deploy Prometheus monitoring | 部署 Prometheus 監控
kubectl apply -f macK8S/prometheus/
```

## 📂 Dataset Information | 數據集說明

### Pre-collected Datasets | 預收集數據集

本項目包含預收集的實驗數據集，用於離線訓練和測試：

- **OnlineBoutique Dataset**: 
  - 位置: `gnnrl/data/datasets/real/onlineboutique/`
  - 包含真實 K8s 環境收集的指標數據
  - 主要文件: `online_boutique_gym_observation.csv`
  - 大小: ~500MB, 包含 10000+ 樣本
  - 包含 11 個微服務的性能指標、拓撲關係和擴展動作

- **Redis Dataset**:
  - 位置: `gnnrl/data/datasets/real/redis/`
  - Redis 集群性能指標數據
  - 主要文件: `redis_gym_observation.csv`
  - 大小: ~200MB, 包含 5000+ 樣本
  - 包含 Master-Slave 配置的性能數據

### Dataset Structure | 數據集結構

```
gnnrl/data/
├── datasets/
│   └── real/
│       ├── onlineboutique/
│       │   └── online_boutique_gym_observation.csv
│       └── redis/
│           └── redis_gym_observation.csv
├── edges.json          # 服務拓撲邊信息
└── nodes_stat.json     # 節點統計信息
```

## ⚙️ Configuration | 配置

### Environment Configuration | 環境配置

創建 `.env` 文件在項目根目錄：

```bash
# Kubernetes Configuration
KUBE_HOST=http://localhost:8001
NAMESPACE_ONLINEBOUTIQUE=onlineboutique
NAMESPACE_REDIS=redis

# Monitoring URLs
KIALI_URL=http://localhost:20001/kiali
PROMETHEUS_URL=http://localhost:9090

# Load Testing
LOADTEST_SERVER=192.168.1.100  # 分散式測試主機
TARGET_HOST=http://k8s.orb.local

# Training Configuration
```

### Version Compatibility | 版本相容性

#### Tested Environments | 測試環境
- **Kubernetes**: v1.20+ to v1.28
- **Python**: 3.8, 3.9, 3.10, 3.11
- **PyTorch**: 1.11.0+ to 2.0.0

#### Known Issues | 已知問題
- Python 3.12: 部分依賴尚未支援 
- Kubernetes 1.29+: 需要更新 API 版本
- macOS M1/M2: 需要使用 conda 安裝 torch-geometric

## 🔬 Experiment Reproduction | 實驗復現

### Quick Start | 快速開始

Use the unified experiment manager for complete experiments:

使用統一實驗管理器進行完整實驗：

```bash
# Run OnlineBoutique experiments with all methods | 執行所有方法的 OnlineBoutique 實驗
python run_autoscaling_experiment.py onlineboutique --all-methods --scenarios all

# Run Redis experiments with all methods | 執行所有方法的 Redis 實驗  
python run_autoscaling_experiment.py redis --all-methods --scenarios all
```

### Detailed Experiment Configuration | 詳細實驗配置

#### 1. GNNRL Experiments | GNNRL 實驗

```bash
# OnlineBoutique GNNRL experiment (GAT model) | OnlineBoutique GNNRL 實驗 (GAT 模型)
python run_autoscaling_experiment.py onlineboutique \
    --method gnnrl \
    --algorithm gat \
    --steps 5000 \
    --scenarios offpeak,peak,rushsale,fluctuating

# Redis GNNRL experiment (TGN model) | Redis GNNRL 實驗 (TGN 模型)  
python run_autoscaling_experiment.py redis \
    --method gnnrl \
    --algorithm tgn \
    --steps 5000 \
    --scenarios offpeak,peak,rushsale,fluctuating
```

#### 2. Gym-HPA Experiments | Gym-HPA 實驗

```bash
# OnlineBoutique Gym-HPA experiment (PPO algorithm) | OnlineBoutique Gym-HPA 實驗 (PPO 算法)
python run_autoscaling_experiment.py onlineboutique \
    --method gym_hpa \
    --algorithm ppo \
    --total-steps 5000 \
    --scenarios offpeak,peak,rushsale,fluctuating

# Redis Gym-HPA experiment (A2C algorithm) | Redis Gym-HPA 實驗 (A2C 算法)
python run_autoscaling_experiment.py redis \
    --method gym_hpa \
    --algorithm a2c \
    --total-steps 5000 \
    --scenarios offpeak,peak,rushsale,fluctuating
```

#### 3. K8s-HPA Baseline Testing | K8s-HPA 基準測試

```bash
# OnlineBoutique K8s-HPA baseline test | OnlineBoutique K8s-HPA 基準測試
python run_autoscaling_experiment.py onlineboutique \
    --method k8s_hpa \
    --hpa-cpu-threshold 40,60,80 \
    --scenarios offpeak,peak,rushsale,fluctuating

# Redis K8s-HPA baseline test | Redis K8s-HPA 基準測試
python run_autoscaling_experiment.py redis \
    --method k8s_hpa \
    --hpa-cpu-threshold 20,40,60,80 \
    --scenarios offpeak,peak,rushsale,fluctuating
```

### Individual Method Execution | 個別方法執行

#### Using GNNRL | 使用 GNNRL

```bash
# Train GNNRL model | 訓練 GNNRL 模型
cd gnnrl
python training/train_gnnppo.py --use-case online_boutique --model gat --steps 5000

# Test trained model | 測試訓練好的模型
python core/run/run.py --testing --test-path logs/models/gnnrl_gat_latency_k8s_True_steps_5000.zip
```

#### Using Gym-HPA | 使用 Gym-HPA

```bash
# Train Gym-HPA model | 訓練 Gym-HPA 模型
cd gym-hpa  
python policies/run/run.py --alg ppo --use-case online_boutique --training --total-steps 5000

# Test trained model | 測試訓練好的模型
python policies/run/run.py --testing --test-path logs/models/ppo_env_online_boutique_gym_goal_latency_k8s_True_totalSteps_5000.zip
```

#### Using K8s-HPA | 使用 K8s-HPA

```bash
# Run K8s-HPA baseline test | 執行 K8s-HPA 基準測試
python k8s_hpa/HPABaseLineTest.py --use-case online_boutique --cpu-threshold 60
```

### Traffic Pattern Description | 流量模式說明

The system supports four traffic testing patterns:

系統支援四種流量測試模式：

- **offpeak** - Low traffic baseline testing (50 RPS) | 低流量基準測試 (50 RPS)
- **peak** - High sustained traffic (300 RPS) | 高峰持續流量 (300 RPS) 
- **rushsale** - Rush hour impact traffic (500 RPS) | 搶購衝擊流量 (500 RPS)
- **fluctuating** - Fluctuating traffic pattern (150-400 RPS) | 波動流量模式 (150-400 RPS)

### Distributed Testing (Optional) | 分散式測試 (可選)

If you have multiple machines, you can configure distributed load testing:

如果有多台機器，可以配置分散式負載測試：

```bash
# Set distributed host environment variable | 設定分散式主機環境變量
export LOADTEST_SERVER=192.168.1.100

# Run distributed testing | 執行分散式測試
python run_autoscaling_experiment.py onlineboutique \
    --distributed \
    --scenarios offpeak,peak
```

## 📈 Results Analysis | 結果分析

### View Experiment Results | 查看實驗結果

Experiment results are stored in the following paths:

實驗結果存儲在以下路徑：

```
logs/
├── gnnrl/                    # GNNRL experiment results | GNNRL 實驗結果
│   ├── actions/             # Scaling action records | 擴展動作記錄
│   ├── tensorboard/         # TensorBoard logs | TensorBoard 日誌  
│   └── models/              # Trained models | 訓練好的模型
├── gym-hpa/                 # Gym-HPA experiment results | Gym-HPA 實驗結果
│   ├── models/              # Trained models | 訓練模型
│   └── tensorboard/         # TensorBoard logs | TensorBoard 日誌
├── k8s-hpa/                 # K8s-HPA baseline results | K8s-HPA 基準結果
├── comparisons/             # Method comparison results | 方法比較結果
└── runtime/                 # Execution logs | 執行日誌
```

### Generate Comparison Reports | 生成比較報告

```bash
# Generate scenario comparison report | 生成場景比較報告
python generate_scenario_comparison.py

# View experimental results | 查看實驗結果
ls logs/comparisons/
```

### View Training Process with TensorBoard | 使用 TensorBoard 查看訓練過程

```bash
# View GNNRL training process | 查看 GNNRL 訓練過程
tensorboard --logdir=logs/gnnrl/tensorboard --port=6006

# View Gym-HPA training process | 查看 Gym-HPA 訓練過程  
tensorboard --logdir=logs/gym-hpa/tensorboard --port=6007
```

### Analyze Key Metrics | 分析關鍵指標

Experiments automatically record the following metrics:

實驗會自動記錄以下指標：

- **RPS (Requests Per Second)** - System throughput | **RPS (每秒請求數)** - 系統吞吐量
- **Latency (P95)** - 95% request response time | **延遲 (P95)** - 95%的請求響應時間 
- **Pod Count** - Autoscaling effectiveness | **Pod 數量** - 自動擴展效果
- **Resource Utilization** - CPU/Memory usage | **資源使用率** - CPU/內存使用情況
- **Convergence Time** - Training convergence speed | **收斂時間** - 訓練收斂速度

## 🗂️ Model Management | 模型管理

### Trained Models Location | 訓練模型位置

```
logs/models/
├── gnnrl_gat_online_boutique_latency_k8s_True_steps_5000.zip
├── gnnrl_tgn_redis_latency_k8s_True_steps_5000.zip  
├── ppo_env_online_boutique_goal_latency_k8s_True_totalSteps_5000.zip
└── [other trained models...]
```

### Model Naming Convention | 模型命名規則

- **GNNRL**: `gnnrl_{model}_{env}_{goal}_k8s_{mode}_steps_{steps}.zip`
- **Gym-HPA**: `{alg}_env_{env}_goal_{goal}_k8s_{mode}_totalSteps_{steps}.zip`

Where | 其中：
- `{model}`: gat, gcn, tgn
- `{env}`: online_boutique, redis  
- `{goal}`: latency, cost
- `{mode}`: True (live K8s), False (simulation)
- `{alg}`: ppo, a2c

### Model Loading Example | 模型載入範例

```python
from stable_baselines3 import PPO
from gnnrl.core.envs import OnlineBoutique

# Load environment
env = OnlineBoutique(k8s=True, use_graph=True)

# Load trained model
model = PPO.load("logs/models/gnnrl_gat_online_boutique_latency_k8s_True_steps_5000")

# Use model for prediction
obs, info = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

## 🔧 配置說明

### 實驗配置文件

主要配置文件：`experiment_config.yaml`

```yaml
experiments:
  gnnrl:
    default_args:
      k8s: true
      steps: 5000
      model: "gat"
      goal: "latency"
      
loadtest:
  scenarios:
    peak:
      duration: "15m"
      expected_rps: 300
```

### Environment Variables | 環境變量

```bash
# Kiali monitoring URL | Kiali 監控 URL
export KIALI_URL=http://kiali.istio-system:20001

# Prometheus monitoring URL | Prometheus 監控 URL  
export PROMETHEUS_URL=http://prometheus:9090

# Distributed host IP | 分散式主機 IP
export LOADTEST_SERVER=192.168.1.100
```

## 🐛 Troubleshooting | 故障排除

### Common Issues | 常見問題

#### 1. Pods Cannot Start | Pod 無法啟動

```bash
# Check if resources are sufficient | 檢查資源是否充足
kubectl describe nodes

# Check image pulling | 檢查鏡像拉取
kubectl describe pod <pod-name> -n <namespace>
```

#### 2. Load Testing Failure | 負載測試失敗

```bash
# Check if target service is reachable | 檢查目標服務是否可達
curl http://k8s.orb.local

# Check load testing configuration | 檢查負載測試配置
cat loadtest/redis/locust_redis_peak.py
```

#### 3. GNNRL Training Failure | GNNRL 訓練失敗

```bash
# Check GPU availability (if using) | 檢查 GPU 可用性 (如果使用)
nvidia-smi

# Check Python dependencies | 檢查 Python 依賴
pip list | grep torch
```

#### 4. Kiali Monitoring Issues | Kiali 監控問題

```bash
# Check Istio status | 檢查 Istio 狀態
kubectl get pods -n istio-system

# Check Kiali service | 檢查 Kiali 服務
kubectl get svc -n istio-system
```

#### 5. Python Environment Issues | Python 環境問題

```bash
# 如果遇到 gymnasium/gym 版本衝突
pip uninstall gym gymnasium
pip install gymnasium>=0.29

# 如果遇到 torch-geometric 安裝問題  
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.11.0+cpu.html

# macOS M1/M2 特殊安裝方式
conda install pytorch torchvision torchaudio -c pytorch
conda install pyg -c pyg
```

#### 6. Model Loading Errors | 模型載入錯誤

```bash
# 檢查模型文件是否存在
ls -la logs/models/

# 檢查模型格式是否正確
python -c "from stable_baselines3 import PPO; model = PPO.load('logs/models/model_name')"

# 檢查模型相容性
python -c "import torch; print(torch.__version__)"
```

#### 7. Permission Issues | 權限問題

```bash
# 檢查 kubectl 權限
kubectl auth can-i create pods --namespace=onlineboutique

# 檢查文件權限
chmod +x run_autoscaling_experiment.py

# 檢查 Docker 權限 (Linux)
sudo usermod -aG docker $USER
```

#### 8. Network and Port Issues | 網路和端口問題

```bash
# 檢查端口占用
lsof -i :8001  # kubectl proxy
lsof -i :9090  # Prometheus
lsof -i :20001 # Kiali

# 檢查服務可達性
curl http://localhost:8001/api/v1/namespaces/onlineboutique/services/frontend/proxy/

# 檢查負載測試連接
curl http://k8s.orb.local/cart
```

### Log Viewing | 日誌查看

```bash
# View experiment logs | 查看實驗日誌
tail -f logs/runtime/unified_experiment_$(date +%Y%m%d_\H%M%S).log

# View Pod logs | 查看 Pod 日誌
kubectl logs -f deployment/frontend -n onlineboutique
```

### Reset Experiment Environment | 重置實驗環境

```bash
# Clean all deployments | 清理所有部署
kubectl delete namespace onlineboutique redis

# Redeploy | 重新部署
kubectl apply -f MicroServiceBenchmark/microservices-demo/kubernetes-manifests/
kubectl apply -f MicroServiceBenchmark/redis-cluster/redis-cluster.yaml
```

## 📚 Advanced Usage | 進階使用

### Custom Models | 自定義模型

```python
# Modify GNNRL model architecture | 修改 GNNRL 模型架構
# Edit: gnnrl/models/gnn_encoder.py | 編輯: gnnrl/models/gnn_encoder.py

# Modify Gym-HPA policy | 修改 Gym-HPA 策略
# Edit: gym-hpa/policies/ppo_policy.py | 編輯: gym-hpa/policies/ppo_policy.py
```

### Add New Test Scenarios | 添加新的測試場景

```python
# Create new Locust test script | 創建新的 Locust 測試腳本
# Reference: loadtest/redis/locust_redis_custom.py | 參考: loadtest/redis/locust_redis_custom.py

# Register new scenario in configuration | 在配置中註冊新場景
# Edit: experiment_config.yaml | 編輯: experiment_config.yaml
```

### Integrate New Monitoring Systems | 整合新的監控系統

```python
# Extend monitoring integration | 擴展監控整合
# Edit: unified_experiment_manager.py | 編輯: unified_experiment_manager.py
```

## 📄 License | 授權條款

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

本專案採用 MIT 授權條款，詳見 [LICENSE](LICENSE) 文件。

## 📧 Contact | 聯繫方式

If you have questions or suggestions, please contact us through:

如有問題或建議，請通過以下方式聯繫：

- GitHub Issues
- Email: [f11942184@ntu.edu.tw]

---

**Note**: Please ensure that the Kubernetes cluster and related dependencies are properly configured before running experiments. It is recommended to perform small-scale validation in a test environment first, then execute full experiments.

**注意**：請確保在運行實驗前已正確配置 Kubernetes 集群和相關依賴。建議在測試環境中先進行小規模驗證，再執行完整實驗。

---

Ho, P. H., Chen, H. Y., & Lin, T. N.(2025, December) "Graphpilot: A Temporal Graph Actor-Critic Autoscaler Reducing Degradation of Resource Oscillation in Microservice" Proceedings of the IEEE/ACM 18th International Conference on Utility and Cloud Computing.
