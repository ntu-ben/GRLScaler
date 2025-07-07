# 🚀 實驗結果可視化工具使用指南

## 🎯 一分鐘快速上手

### 最簡單的使用方式

```bash
# 生成所有環境的比較圖表
python generate_experiment_charts.py
```

**就這麼簡單！** 這個命令會：
- 自動找到所有可用的實驗數據
- 生成 RPS 和 Pod 數量比較圖
- 保存到 `logs/visualizations/` 目錄

---

## 📊 具體使用場景

### 1. 自動比較所有方法

```bash
# Redis 環境比較
python experiment_visualization.py --auto-compare --environment redis

# OnlineBoutique 環境比較
python experiment_visualization.py --auto-compare --environment onlineboutique
```

**輸出結果：**
- 找到 GNNRL、Gym-HPA、K8s-HPA 的最新實驗
- 為每個場景 (offpeak, peak, rushsale, fluctuating) 生成兩種圖表

### 2. 手動指定實驗進行比較

```bash
python experiment_visualization.py --compare \
    --gnnrl logs/gnnrl/gnnrl_train_seed42_20250629_191025 \
    --gym-hpa logs/gym-hpa/gym_hpa_train_seed42_20250629_024235 \
    --k8s-hpa logs/k8s-hpa/k8s_hpa_cpu_seed42_20250630_234602
```

**適用場景：**
- 想比較特定日期的實驗
- 需要精確控制比較對象

### 3. 分析單一實驗

```bash
python experiment_visualization.py --experiment-dir logs/gnnrl/gnnrl_train_seed42_20250629_191025
```

**輸出結果：**
- 為該實驗的每個場景生成 RPS 分析圖

---

## 📈 生成的圖表類型

### 🔵 RPS 時間序列比較圖
- **橫軸**: 實驗執行時間
- **縱軸**: 每秒請求數 (RPS)
- **內容**: 
  - 黑色虛線：原始壓測設定目標值
  - 藍色實線：GNNRL 實際表現
  - 紅色虛線：Gym-HPA 實際表現
  - 橙色點線：K8s-HPA 實際表現

### 🔴 Pod 數量時間序列比較圖
- **橫軸**: 時間 (分鐘)
- **縱軸**: Pod 數量
- **內容**:
  - 綠色虛線：理論最佳值
  - 各方法的實際 Pod 擴縮容行為

---

## 📁 查看結果

### 圖表保存位置
```
logs/visualizations/
├── redis_offpeak_rps_comparison_20250707_155613.png      # Redis低峰RPS比較
├── redis_offpeak_pods_comparison_20250707_155613.png     # Redis低峰Pod比較
├── redis_peak_rps_comparison_20250707_155613.png         # Redis高峰RPS比較
└── ...
```

### 快速查看
```bash
# 列出最新生成的圖表
ls -lt logs/visualizations/*.png | head -10

# 在當前目錄打開圖表文件夾 (macOS)
open logs/visualizations/
```

---

## 🔧 常見使用場景

### 場景1：我想看看最新實驗的表現
```bash
python generate_experiment_charts.py
```

### 場景2：我想比較特定的實驗
```bash
# 查看可用的實驗目錄
ls logs/gnnrl/ | grep train
ls logs/gym-hpa/ | grep train  
ls logs/k8s-hpa/ | grep cpu

# 手動比較
python experiment_visualization.py --compare \
    --gnnrl logs/gnnrl/[你選的目錄] \
    --gym-hpa logs/gym-hpa/[你選的目錄] \
    --k8s-hpa logs/k8s-hpa/[你選的目錄]
```

### 場景3：我只想看某一種方法的表現
```bash
python experiment_visualization.py --experiment-dir logs/gnnrl/gnnrl_train_seed42_20250629_191025
```

### 場景4：我想為論文準備圖表
```bash
# 使用手動比較確保使用相同時期的實驗
python experiment_visualization.py --compare \
    --gnnrl logs/gnnrl/gnnrl_train_seed42_20250629_191025 \
    --gym-hpa logs/gym-hpa/gym_hpa_train_seed42_20250629_024235 \
    --k8s-hpa logs/k8s-hpa/k8s_hpa_cpu_seed42_20250630_234602
```

---

## ⚠️ 常見問題

### Q: 為什麼沒有找到實驗數據？
A: 檢查實驗目錄是否包含 Locust 統計數據：
```bash
ls logs/gym-hpa/[實驗目錄]/*/  # 應該看到 *_stats_history.csv 文件
```

### Q: 圖表顯示不正確？
A: 確保實驗目錄結構正確：
```
實驗目錄/
├── offpeak_001/
│   ├── offpeak_stats.csv
│   └── offpeak_stats_history.csv
├── peak_001/
└── ...
```

### Q: 想要自定義圖表樣式？
A: 編輯 `experiment_visualization.py` 中的顏色和樣式設定：
```python
colors = {'GNNRL': '#2E86AB', 'Gym-HPA': '#A23B72', 'K8s-HPA': '#F18F01'}
```

---

## 🎯 推薦工作流程

1. **運行實驗** → 使用 `run_autoscaling_experiment.py`
2. **生成圖表** → 運行 `python generate_experiment_charts.py`
3. **查看結果** → 打開 `logs/visualizations/` 目錄
4. **論文撰寫** → 使用手動比較生成精確的對比圖

---

## 📞 需要幫助？

- 檢查 `VISUALIZATION_GUIDE.md` 了解詳細功能
- 檢查 `ONLINEBOUTIQUE_VISUALIZATION_README.md` 了解 OnlineBoutique 特定問題
- 運行 `python experiment_visualization.py --help` 查看所有選項