# GNNRL 場景測試功能

本功能允許你選擇特定場景來測試已訓練的GNNRL模型，而不需要執行所有4個場景。

## 🎯 使用方式

### 1. 直接使用統一實驗管理器

```bash
# 測試peak和rushsale場景
python unified_experiment_manager.py \
  --experiment gnnrl \
  --use-case online_boutique \
  --testing \
  --load-path logs/models/your_model.zip \
  --test-scenarios peak rushsale \
  --k8s

# 使用TGN模型和A2C算法測試
python unified_experiment_manager.py \
  --experiment gnnrl \
  --use-case online_boutique \
  --model tgn \
  --alg a2c \
  --testing \
  --load-path logs/models/your_model.zip \
  --test-scenarios peak rushsale \
  --k8s
```

### 2. 使用便捷腳本

```bash
# 測試peak場景
python test_gnnrl_scenarios.py peak --k8s

# 測試peak和rushsale場景
python test_gnnrl_scenarios.py peak rushsale --k8s

# 使用TGN模型和A2C算法
python test_gnnrl_scenarios.py peak rushsale --model tgn --alg a2c --k8s

# 使用特定模型路徑
python test_gnnrl_scenarios.py peak --model-path logs/models/your_model.zip --k8s
```

## 📋 可用場景

- `offpeak`: 低峰時段
- `peak`: 高峰時段  
- `rushsale`: 搶購活動
- `fluctuating`: 波動負載

## 🧠 支援的模型

- `gat`: Graph Attention Network (默認)
- `gcn`: Graph Convolutional Network
- `tgn`: Temporal Graph Network (時間序列圖神經網路)

## 🎯 支援的算法

- `ppo`: Proximal Policy Optimization (默認)
- `a2c`: Advantage Actor-Critic

## 📁 模型自動發現

如果不指定 `--model-path` 或 `--load-path`，系統會自動查找最新的模型檔案：

- OnlineBoutique: `gnnrl_*latency_k8s_True_steps_*.zip`
- Redis: `gnnrl_*redis*_k8s_True_steps_*.zip`

## 🔧 使用範例

### 重跑peak和rushsale場景

```bash
# 方法1：使用便捷腳本（推薦）
python test_gnnrl_scenarios.py peak rushsale --k8s

# 方法2：直接使用統一實驗管理器
python unified_experiment_manager.py \
  --experiment gnnrl \
  --use-case online_boutique \
  --testing \
  --load-path auto \
  --test-scenarios peak rushsale \
  --k8s
```

### 使用TGN模型測試

```bash
# 使用TGN模型和A2C算法測試peak場景
python test_gnnrl_scenarios.py peak --model tgn --alg a2c --k8s
```

### Redis環境測試

```bash
# 測試Redis環境的peak場景
python test_gnnrl_scenarios.py peak --use-case redis --k8s
```

## 📊 輸出結果

測試結果將保存在以下目錄：
```
logs/gnnrl/gnnrl_test_seed42_TIMESTAMP/
├── peak_001/
│   ├── peak_stats.csv
│   ├── peak_stats_history.csv
│   └── pod_metrics/
└── rushsale_002/
    ├── rushsale_stats.csv
    ├── rushsale_stats_history.csv
    └── pod_metrics/
```

## 🎲 隨機種子

使用 `--seed` 參數控制場景執行順序：
- 相同的種子會產生相同的執行順序
- 不同的種子會打亂場景順序，但仍然只執行選定的場景

## ✅ 驗證功能

測試完成後，你可以使用現有的可視化工具來查看結果：

```bash
# 生成場景對比圖
python generate_scenario_comparison.py onlineboutique

# 查看結果
python view_results.py
```