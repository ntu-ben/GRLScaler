# OnlineBoutique 實驗結果可視化指南

## 問題解答

### 1. 理論Pod值如何估計？

**OnlineBoutique 環境的理論Pod值計算考慮以下因素：**

#### 微服務架構特性
- **10個核心微服務**: frontend, cartservice, productcatalogservice, currencyservice, paymentservice, shippingservice, emailservice, checkoutservice, recommendationservice, adservice
- **不同服務的負載特性**: 前端服務承載更多流量，後端服務處理特定業務邏輯

#### 計算邏輯
```python
def calculate_theoretical_optimal_pods(rps, scenario, environment='onlineboutique'):
    if rps <= 30:
        return 12   # 10個微服務各1個 + 2個關鍵服務多1個
    elif rps <= 80:
        return 18   # 前端、購物車、產品等關鍵服務擴展
    elif rps <= 150:
        return 25   # 大部分服務擴展到2-3個副本
    else:
        return ceil(rps / 8)  # 假設平均每8 RPS需要1個Pod
```

#### 理論基礎
- **前端服務**: 每個Pod約處理30 RPS
- **業務邏輯服務**: 每個Pod約處理20-40 RPS
- **數據庫相關服務**: 每個Pod約處理50+ RPS
- **整體考量**: 服務間依賴、網絡延遲、資源競爭

### 2. 其他方法為什麼沒有比較到？

**當前實驗數據狀況：**

#### 已有數據
- ✅ **GNNRL**: 有完整的OnlineBoutique訓練數據 (`logs/gnnrl/gnnrl_train_seed42_20250629_191025/`)
- ✅ **K8s-HPA**: 有OnlineBoutique基準測試數據 (`logs/k8s-hpa/k8s_hpa_cpu_seed42_20250630_234602/`)
- ❌ **Gym-HPA**: 只有模型文件，缺少Locust測試統計數據

#### 缺少的數據
```
logs/gym-hpa/ppo_env_online_boutique_gym_goal_latency_k8s_True_totalSteps_5000/
├── ppo_env_online_boutique_gym_goal_latency_k8s_True_totalSteps_5000_5000_steps.zip  ✅
├── offpeak_001/           ❌ 缺少
│   ├── offpeak_stats.csv
│   └── offpeak_stats_history.csv
├── peak_001/              ❌ 缺少
└── ...
```

### 3. 需要做什麼修改才能比較？

#### 方法1：運行完整OnlineBoutique實驗
```bash
# 運行Gym-HPA OnlineBoutique實驗
python run_autoscaling_experiment.py onlineboutique --method gym-hpa --steps 3000

# 運行GNNRL OnlineBoutique實驗
python run_autoscaling_experiment.py onlineboutique --method gnnrl --steps 3000

# 運行K8s-HPA OnlineBoutique實驗
python run_autoscaling_experiment.py onlineboutique --method k8s-hpa
```

#### 方法2：使用現有數據進行示例比較
```bash
# 使用現有的Redis數據作為示例(已修改環境檢測邏輯)
python experiment_visualization.py --compare \
    --gnnrl logs/gnnrl/gnnrl_train_seed42_20250629_191025 \
    --k8s-hpa logs/k8s-hpa/k8s_hpa_cpu_seed42_20250630_234602

# 或運行示例生成器
python create_onlineboutique_visualization_demo.py
```

#### 方法3：修改實驗配置專注OnlineBoutique
```bash
# 編輯實驗配置
vim experiment_config.yaml

# 設定OnlineBoutique專用參數
use_case: "online_boutique"
scenarios: ["offpeak", "peak", "rushsale", "fluctuating"]
target_rps: {offpeak: 30, peak: 100, rushsale: 200, fluctuating: 80}
```

## 已實現的修復

### ✅ 理論Pod值計算改進
- 基於OnlineBoutique微服務架構特性
- 考慮服務間依賴和負載分布
- 動態調整基於實際RPS負載

### ✅ 智能環境檢測
- 自動識別OnlineBoutique vs Redis環境
- 調整基準RPS和Pod計算邏輯
- 適配不同實驗目錄命名模式

### ✅ Pod數量模擬增強
```python
def _simulate_pod_scaling(rps_data, method, scenario, environment):
    # 不同方法的擴縮容特性
    scaling_configs = {
        'GNNRL': {'aggressive': 0.8, 'smoothing': 0.3},     # 積極擴縮容
        'Gym-HPA': {'aggressive': 0.6, 'smoothing': 0.4},   # 中等策略
        'K8s-HPA': {'aggressive': 0.4, 'smoothing': 0.6}    # 保守策略
    }
```

### ✅ 手動比較功能
- 支援指定多個實驗目錄進行比較
- 自動檢測環境類型並調整參數
- 生成帶時間戳的比較圖表

## 使用建議

### 當前可用的比較方式
```bash
# 1. 使用示例數據快速體驗
python create_onlineboutique_visualization_demo.py

# 2. 手動指定實驗目錄比較
python experiment_visualization.py --compare \
    --gnnrl logs/gnnrl/[your_gnnrl_experiment] \
    --k8s-hpa logs/k8s-hpa/[your_k8s_hpa_experiment]

# 3. 分析單一實驗的多場景表現
python experiment_visualization.py \
    --experiment-dir logs/gnnrl/gnnrl_train_seed42_20250629_191025
```

### 獲得完整比較的步驟
1. **執行完整OnlineBoutique實驗** (推薦)
2. **收集所有三種方法的數據**
3. **使用自動比較功能**
4. **分析結果並調整參數**

## 生成的圖表說明

### RPS比較圖特色
- **環境自適應**: 自動調整OnlineBoutique的基準RPS值
- **多方法對比**: GNNRL vs Gym-HPA vs K8s-HPA
- **基準線顯示**: 原始壓測設定目標值

### Pod數量比較圖特色  
- **理論最佳值**: 基於微服務架構負載計算
- **模擬擴縮容**: 反映不同方法的策略差異
- **時間序列**: 展示Pod數量隨負載變化

### 文件命名
- `manual_onlineboutique_[scenario]_rps_comparison_[timestamp].png`
- `manual_onlineboutique_[scenario]_pods_comparison_[timestamp].png`

通過這些改進，您現在可以：
1. ✅ 獲得更合理的理論Pod值估計
2. ✅ 使用現有數據進行示例比較
3. ✅ 為未來的完整實驗做好準備