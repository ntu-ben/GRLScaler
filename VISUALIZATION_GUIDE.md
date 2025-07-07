# GRLScaler 實驗結果可視化指南

## 概述

`experiment_visualization.py` 是為 GRLScaler 自動擴展實驗設計的可視化工具，能夠生成兩種類型的時間序列圖表：

1. **RPS 表現比較圖** - 比較不同方法 (GNNRL, Gym-HPA, K8s-HPA) 與原始壓測設定的 RPS 表現
2. **Pod 數量比較圖** - 比較不同方法的 Pod 擴展行為與理論最佳值

## 功能特色

- ✅ 自動解析 Locust 測試統計數據
- ✅ 支援多種實驗場景 (offpeak, peak, rushsale, fluctuating)
- ✅ 智能色彩配置和圖表樣式
- ✅ 自動保存高解析度圖表
- ✅ 支援 Redis 和 OnlineBoutique 環境
- ✅ 理論最佳值計算
- ✅ 數據平滑化處理

## 使用方式

### 1. 自動比較所有可用實驗 (推薦)

```bash
# 自動比較 Redis 環境的所有實驗
python experiment_visualization.py --auto-compare --environment redis

# 自動比較 OnlineBoutique 環境的所有實驗
python experiment_visualization.py --auto-compare --environment onlineboutique
```

這是最簡單的使用方式，會自動：
- 查找最新的 GNNRL、Gym-HPA、K8s-HPA 實驗
- 為每個場景生成 RPS 和 Pod 數量比較圖
- 將圖表保存到 `logs/visualizations/` 目錄

### 2. 分析單一實驗

```bash
# 分析特定實驗目錄
python experiment_visualization.py --experiment-dir logs/gnnrl/gnnrl_redis_train_seed42_20250706_190527
```

### 3. 手動指定多個實驗比較

```bash
python experiment_visualization.py --compare \
    --gnnrl logs/gnnrl/gnnrl_redis_train_seed42_20250706_190527 \
    --gym-hpa logs/gym-hpa/gym_hpa_redis_train_seed42_20250706_122635 \
    --k8s-hpa logs/k8s_hpa_redis/redis_hpa_cpu-40_20250706_125639
```

### 4. 自定義輸出目錄

```bash
python experiment_visualization.py --auto-compare --environment redis --output-dir custom_charts/
```

## 圖表說明

### RPS 表現比較圖

- **橫軸**: 時間軸 (實驗執行時間)
- **縱軸**: 每秒請求數 (RPS)
- **基準線**: 原始壓測設定的目標 RPS
- **實線**: 各方法的實際 RPS 表現
- **顏色編碼**:
  - 🔵 GNNRL (藍色)
  - 🔴 Gym-HPA (紅色)  
  - 🟡 K8s-HPA (橙色)
  - ⚫ 原始壓測設定 (黑色虛線)

### Pod 數量比較圖

- **橫軸**: 時間 (分鐘)
- **縱軸**: Pod 數量
- **理論最佳值**: 基於負載計算的理論最佳 Pod 數量
- **實際表現**: 各方法的實際 Pod 擴展行為

## 數據源說明

### 自動發現的數據路徑

工具會自動查找以下路徑的實驗數據：

```
logs/
├── gnnrl/                    # GNNRL 實驗
│   └── gnnrl_*_redis_*/
├── gym-hpa/                  # Gym-HPA 實驗  
│   └── *redis*/
└── k8s_hpa_redis/           # K8s-HPA 實驗
    └── redis_hpa_*/
```

### 數據格式要求

工具期望每個實驗目錄包含：
- 場景子目錄 (如 `offpeak_001/`, `peak_001/`)
- 每個場景包含 `*_stats_history.csv` 文件 (Locust 生成)

## 理論最佳值計算

### Redis 環境
- **Redis Master**: 每個 Pod 處理 100 RPS
- **Redis Slave**: 每個 Pod 處理 150 RPS

### OnlineBoutique 環境  
- **微服務**: 每個 Pod 處理 50 RPS

計算公式: `最佳Pod數 = ceil(目標RPS / 每Pod處理能力)`

## 輸出文件

生成的圖表文件命名格式：
```
{environment}_{scenario}_{type}_comparison_{timestamp}.png
```

範例：
- `redis_offpeak_rps_comparison_20250707_150436.png`
- `redis_peak_pods_comparison_20250707_150436.png`

## 場景類型

### 標準場景
- **offpeak**: 低峰時段 (50 RPS)
- **peak**: 高峰時段 (200 RPS)  
- **rushsale**: 搶購時段 (500 RPS)
- **fluctuating**: 波動負載 (平均 150 RPS)

### Redis 專用場景
- **redis_offpeak**: Redis 低峰 (100 RPS)
- **redis_peak**: Redis 高峰 (300 RPS)

## 故障排除

### 常見問題

1. **無法找到實驗數據**
   ```
   ❌ 未找到 redis 環境的實驗數據
   ```
   - 檢查 `logs/` 目錄結構
   - 確認實驗名稱包含環境關鍵字

2. **數據解析失敗**
   ```
   ❌ 未找到有效的實驗數據
   ```
   - 檢查 CSV 文件是否完整
   - 確認 Locust 統計文件格式正確

3. **依賴套件問題**
   ```
   ModuleNotFoundError: No module named 'pandas'
   ```
   - 安裝所需套件: `pip install pandas matplotlib numpy`

### 調試模式

如需詳細調試信息，可查看生成過程的輸出信息，工具會顯示：
- 找到的實驗路徑
- 數據提取狀態
- 圖表保存位置

## 高級使用

### 批次處理多個環境

```bash
# 連續處理多個環境
python experiment_visualization.py --auto-compare --environment redis
python experiment_visualization.py --auto-compare --environment onlineboutique
```

### 結合其他分析工具

生成的圖表可配合其他分析腳本使用：
- `analyze_results.py` - 數值分析
- `experiment_planner.py` - 實驗規劃
- `run_autoscaling_experiment.py` - 實驗執行

## 依賴套件

- Python 3.7+
- pandas >= 1.0.0
- matplotlib >= 3.0.0  
- numpy >= 1.18.0

## 許可證

與 GRLScaler 項目相同的許可證