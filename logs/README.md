# GRLScaler 實驗日誌目錄結構

## 目錄說明

### 📊 實驗結果目錄
- `experiments/` - 實驗執行記錄
  - `onlineboutique/` - OnlineBoutique 實驗
  - `redis/` - Redis 實驗
  - `比較實驗/` - 跨方法比較實驗

### 🧠 方法特定日誌
- `gnnrl/` - 圖神經網路強化學習日誌
  - `models/` - 訓練模型文件
  - `tensorboard/` - TensorBoard 日誌
  - `onlineboutique/` - OnlineBoutique 實驗記錄
  - `redis/` - Redis 實驗記錄

- `gym-hpa/` - 基礎強化學習日誌
  - `models/` - 訓練模型文件  
  - `tensorboard/` - TensorBoard 日誌
  - `onlineboutique/` - OnlineBoutique 實驗記錄
  - `redis/` - Redis 實驗記錄

- `k8s-hpa/` - Kubernetes HPA 日誌
  - `onlineboutique/` - OnlineBoutique 實驗記錄
  - `redis/` - Redis 實驗記錄

### 📈 分析和比較
- `comparisons/` - 方法比較結果
  - `method_comparison.csv` - 方法性能比較
  - `scenario_comparison.csv` - 場景比較

### 🔧 運行時日誌
- `runtime/` - 實驗執行日誌
  - `unified_experiment.log` - 統一實驗日誌
  - `error.log` - 錯誤日誌

### 🌐 Kiali 服務圖
- `kiali/` - Kiali 服務圖記錄
  - `kiali_start.json` - 實驗開始時服務圖
  - `kiali_mid.json` - 實驗中期服務圖  
  - `kiali_end.json` - 實驗結束時服務圖

## 命名規範

### 實驗批次命名
格式: `{method}_{environment}_{type}_{timestamp}`

範例:
- `gnnrl_redis_train_20250707_031500/` - GNNRL Redis 訓練 2025年7月7日 3:15
- `gymhpa_onlineboutique_test_20250707_031500/` - Gym-HPA OnlineBoutique 測試
- `k8shpa_redis_comparison_20250707_031500/` - K8s-HPA Redis 比較實驗

### 場景命名
- `offpeak_001/` - 低峰場景第1次
- `peak_001/` - 高峰場景第1次  
- `fluctuating_001/` - 波動場景第1次
- `rushsale_001/` - 搶購場景第1次

### 模型文件命名
格式: `{method}_{model}_{goal}_k8s_{k8s_mode}_steps_{steps}.zip`

範例:
- `gnnrl_gat_latency_k8s_False_steps_5000.zip`
- `gymhpa_ppo_cost_k8s_True_steps_3000.zip`

## 查找實驗記錄

### 按時間查找
```bash
# 查找今天的實驗
find logs/ -name "*$(date +%Y%m%d)*" -type d

# 查找最近的實驗  
ls -lt logs/*/
```

### 按方法查找
```bash
# GNNRL 實驗
find logs/gnnrl/ -name "*train*" -type d

# Gym-HPA 實驗  
find logs/gym-hpa/ -name "*train*" -type d

# K8s-HPA 實驗
find logs/k8s-hpa/ -name "*" -type d
```

### 按環境查找
```bash
# Redis 相關實驗
find logs/ -name "*redis*" -type d

# OnlineBoutique 相關實驗
find logs/ -name "*onlineboutique*" -type d
```