# GRLScaler 重構計劃

## 目標
1. 移除重複和無用的檔案/資料夾
2. 統一整合所有實驗方法
3. 創建單一實驗介面
4. 統一日誌和結果格式

## 當前問題
1. **重複的資料集**: datasets 出現在 3 個位置
2. **重複的環境實作**: OnlineBoutique 和 Redis 環境在 3 個模組中重複
3. **無用的轉發模組**: gnn_rl_env 只做轉發
4. **分散的實驗腳本**: 多個不同的 run.py
5. **不一致的日誌格式**: 不同方法使用不同的日誌結構

## 重構後的新結構

```
GRLScaler/
├── core/                           # 核心功能模組
│   ├── environments/              # 統一的環境實作
│   │   ├── __init__.py
│   │   ├── online_boutique.py     # 整合版OnlineBoutique
│   │   ├── redis.py               # 整合版Redis
│   │   └── base.py               # 基礎環境類
│   ├── policies/                  # 所有RL策略
│   │   ├── __init__.py
│   │   ├── gnn_policy.py         # GNN-based policies
│   │   ├── mlp_policy.py         # Standard MLP policies
│   │   └── base_policy.py        # 基礎策略類
│   ├── models/                    # GNN模型
│   │   ├── __init__.py
│   │   ├── gnn_encoder.py
│   │   └── variants/
│   └── utils/                     # 共用工具
│       ├── __init__.py
│       ├── k8s_utils.py          # K8s相關工具
│       ├── metrics.py            # 指標計算
│       └── deployment.py         # 部署管理
│
├── methods/                       # 各種自動擴縮方法
│   ├── __init__.py
│   ├── gnn_rl.py                 # GNN-based RL
│   ├── traditional_rl.py         # 傳統RL (原gym-hpa)
│   ├── gwydion.py                # Gwydion方法
│   └── hpa_baseline.py           # HPA基準
│
├── experiments/                   # 實驗管理
│   ├── __init__.py
│   ├── unified_runner.py         # 統一實驗介面
│   ├── config/                   # 實驗配置
│   │   ├── default.yaml
│   │   ├── gnn_config.yaml
│   │   └── hpa_config.yaml
│   └── benchmarks/               # 基準測試
│       ├── __init__.py
│       └── compare_methods.py
│
├── infrastructure/               # 基礎設施
│   ├── loadtest/                # 載入測試 (從現有搬移)
│   ├── k8s/                     # K8s配置 (原macK8S)
│   └── monitoring/              # 監控配置
│
├── datasets/                    # 統一的資料集
│   ├── real/
│   │   ├── onlineboutique/
│   │   └── redis/
│   └── synthetic/              # 可選的合成資料
│
├── results/                    # 統一的結果輸出
│   ├── experiments/           # 實驗結果
│   │   ├── {experiment_id}/
│   │   │   ├── config.yaml    # 實驗配置
│   │   │   ├── metrics.json   # 統一指標格式
│   │   │   ├── logs/          # 詳細日誌
│   │   │   └── models/        # 訓練的模型
│   └── comparisons/           # 比較結果
│       └── benchmark_report.html
│
├── scripts/                   # 便利腳本
│   ├── setup_environment.py  # 環境設定
│   ├── run_experiment.py     # 主要實驗腳本
│   └── generate_report.py    # 報告生成
│
├── docs/                     # 文檔
├── tests/                    # 測試
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 移除計劃

### 完全移除
- `gnn_rl_env/` - 只有轉發功能
- `gwydion/gwydion/datasets/` - 重複資料集
- `gym-hpa/datasets/` - 重複資料集
- `MicroServiceBenchmark/` - 未使用的Google範例 (48MB)

### 整合移除
- `gym-hpa/gym_hpa/envs/` -> `core/environments/`
- `gwydion/gwydion/envs/` -> `core/environments/`
- `gnn_rl/envs/` -> `core/environments/`
- `macK8S/` -> `infrastructure/k8s/`
- `k8s_hpa/` -> `methods/hpa_baseline.py`

## 統一日誌格式

```yaml
experiment:
  id: "exp_20250621_143000_gnn_onlineboutique"
  method: "gnn_rl"  # gnn_rl, traditional_rl, gwydion, hpa
  target: "onlineboutique"  # onlineboutique, redis
  mode: "k8s"  # k8s, simulation
  config:
    steps: 10000
    goal: "latency"
    
metrics:
  training:
    episode_rewards: [...]
    training_time: 1200.5
  performance:
    slo_violation_rate: 0.05
    resource_efficiency: 0.82
    scaling_lag: 2.3
    
logs:
  console_output: "logs/console.log"
  detailed_metrics: "logs/metrics.csv"
  model_checkpoints: "models/"
```

## 實作步驟
1. 創建新的核心結構
2. 整合環境實作
3. 統一實驗介面
4. 更新所有引用
5. 測試並驗證