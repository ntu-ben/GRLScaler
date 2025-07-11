# GNNRL 圖形數據可視化指南

## 🎯 概述

GNNRL 訓練過程現在支持每500步自動輸出圖形數據，包括：
- 🌐 **網絡拓撲圖**：服務間連接關係
- 📊 **節點特徵**：CPU、記憶體、Pod數量等
- 🔗 **邊特徵**：服務間流量、延遲、錯誤率
- 📈 **訓練指標**：獎勵趨勢、模型表現

## 🚀 快速開始

### 1. 啟動帶圖形可視化的訓練

```bash
# 啟動GNNRL訓練（自動每500步輸出圖形數據）
python unified_experiment_manager.py --experiment gnnrl --steps 5000 --use-case online_boutique --model gat --goal latency --seed 42
```

### 2. 查看實時圖形數據

```bash
# 檢查圖形數據輸出目錄
ls logs/gnnrl/gnnrl_train_seed42_*/graph_visualizations/

# 每個步驟目錄包含：
# - network_topology_step_*.png        # 網絡拓撲圖
# - node_features_step_*.png           # 節點特徵圖
# - edge_features_step_*.png           # 邊特徵圖
# - training_metrics_step_*.png        # 訓練指標圖
# - raw_data_step_*.json               # 原始數據
# - node_features_step_*.csv           # 節點特徵CSV
```

### 3. 生成動態儀表板

```bash
# 生成交互式HTML儀表板
python gnnrl/training/graph_visualization_dashboard.py --log-dir logs/gnnrl/gnnrl_train_seed42_20250711_120622/graph_visualizations

# 生成網絡演化動畫GIF
python gnnrl/training/graph_visualization_dashboard.py --log-dir logs/gnnrl/gnnrl_train_seed42_20250711_120622/graph_visualizations --gif-only

# 生成訓練報告
python gnnrl/training/graph_visualization_dashboard.py --log-dir logs/gnnrl/gnnrl_train_seed42_20250711_120622/graph_visualizations --report-only
```

## 📊 圖形數據內容

### 🌐 網絡拓撲圖 (Network Topology)
- **節點**：微服務（顏色表示CPU使用率）
  - 🟢 綠色：CPU < 40%
  - 🟡 黃色：CPU 40-60%
  - 🟠 橙色：CPU 60-80%
  - 🔴 紅色：CPU > 80%
- **節點大小**：Pod數量（越大表示Pod越多）
- **邊**：服務間通信連接

### 📊 節點特徵圖 (Node Features)
- **Pod Count**：當前Pod數量
- **Desired Replicas**：期望副本數
- **CPU Usage (%)**：CPU使用率
- **Memory Usage (MB)**：記憶體使用量
- **RX Traffic**：接收流量
- **TX Traffic**：傳送流量

### 🔗 邊特徵圖 (Edge Features)
- **QPS Distribution**：每秒查詢數分佈
- **P95 Latency Distribution**：95分位數延遲分佈
- **Error Rate Distribution**：錯誤率分佈
- **QPS vs P95 Latency**：QPS與延遲關係散點圖

### 📈 訓練指標圖 (Training Metrics)
- **Reward Trend**：獎勵趨勢線
- **Trend Line**：線性趨勢分析
- **Training Progress**：訓練進度可視化

## 🎨 動態儀表板功能

### 交互式HTML儀表板
- **實時圖表**：使用Plotly生成的交互式圖表
- **多視角分析**：同時顯示多個指標
- **縮放與篩選**：可以縮放和篩選數據
- **數據導出**：可以導出圖表和數據

### 網絡演化動畫
- **GIF動畫**：顯示網絡拓撲隨時間的變化
- **節點變化**：觀察Pod數量和CPU使用率變化
- **顏色編碼**：直觀顯示服務健康狀態

### 訓練報告
- **統計摘要**：訓練步數、時間範圍、獎勵統計
- **服務分析**：每個服務的資源使用變化
- **JSON格式**：結構化數據便於後續分析

## 🔧 配置選項

### 修改圖形輸出頻率
```python
# 在 run_gnnrl_experiment.py 中修改
graph_viz_callback = GraphVisualizationCallback(
    save_freq=500,  # 改為其他值，如250（更頻繁）或1000（較少）
    output_dir=str(graph_viz_dir),
    verbose=1
)
```

### 自定義圖形樣式
```python
# 在 graph_visualization_callback.py 中修改
plt.rcParams['font.size'] = 10        # 字體大小
plt.rcParams['figure.dpi'] = 100      # 圖片解析度
plt.rcParams['figure.figsize'] = (12, 8)  # 圖片大小
```

## 📂 文件結構

```
logs/gnnrl/gnnrl_train_seed42_*/
├── graph_visualizations/           # 圖形可視化輸出
│   ├── step_00000500/             # 每500步的數據
│   │   ├── network_topology_step_00000500.png
│   │   ├── node_features_step_00000500.png
│   │   ├── edge_features_step_00000500.png
│   │   ├── training_metrics_step_00000500.png
│   │   ├── raw_data_step_00000500.json
│   │   └── node_features_step_00000500.csv
│   ├── step_00001000/
│   │   └── ...
│   └── dashboard/                 # 儀表板輸出
│       ├── interactive_dashboard.html
│       ├── network_evolution.gif
│       └── training_report.json
├── tensorboard/                   # TensorBoard日誌
└── checkpoints/                   # 模型檢查點
```

## 📋 依賴套件

### 必需套件
```bash
pip install matplotlib numpy pandas
```

### 可選套件（增強功能）
```bash
# 交互式儀表板
pip install plotly

# 網絡圖處理
pip install networkx

# 動畫生成
pip install pillow

# 圖表美化
pip install seaborn
```

## 🎯 使用場景

### 1. 訓練監控
- 實時觀察網絡拓撲變化
- 監控服務資源使用情況
- 追蹤訓練進度和獎勵趨勢

### 2. 性能分析
- 分析不同訓練階段的系統行為
- 識別資源瓶頸和異常模式
- 評估縮放決策的效果

### 3. 研究與開發
- 比較不同模型的學習行為
- 分析圖神經網絡的特徵演化
- 生成論文和報告的可視化材料

### 4. 調試與診斷
- 定位訓練中的問題
- 驗證環境設置是否正確
- 檢查數據流和連接狀態

## 🔍 故障排除

### 常見問題

1. **沒有生成圖形數據**
   - 確保使用 `--use-graph` 參數
   - 檢查是否安裝了必要的依賴套件
   - 查看訓練日誌中的錯誤信息

2. **圖形顯示異常**
   - 確保環境能正確訪問Kiali服務圖
   - 檢查K8s集群連接狀態
   - 驗證服務名稱和特徵數據

3. **儀表板無法打開**
   - 確保安裝了plotly: `pip install plotly`
   - 檢查HTML文件是否生成成功
   - 嘗試手動在瀏覽器中打開文件

4. **GIF動畫生成失敗**
   - 安裝pillow: `pip install pillow`
   - 確保有足夠的磁盤空間
   - 檢查數據是否完整

## 🎓 進階使用

### 自定義分析腳本
```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

# 讀取所有步驟數據
log_dir = Path("logs/gnnrl/gnnrl_train_seed42_*/graph_visualizations")
data = []
for step_dir in sorted(log_dir.glob("step_*")):
    json_file = list(step_dir.glob("raw_data_*.json"))[0]
    with open(json_file) as f:
        data.append(json.load(f))

# 自定義分析
steps = [d['step'] for d in data]
rewards = [d['reward'] for d in data]
plt.plot(steps, rewards)
plt.title('Custom Analysis')
plt.show()
```

### 批量處理多個實驗
```bash
# 批量生成多個實驗的儀表板
for exp_dir in logs/gnnrl/gnnrl_train_*; do
    if [ -d "$exp_dir/graph_visualizations" ]; then
        echo "Processing $exp_dir"
        python gnnrl/training/graph_visualization_dashboard.py --log-dir "$exp_dir/graph_visualizations"
    fi
done
```

## 🚀 最佳實踐

1. **訓練前確認**：確保所有依賴套件已安裝
2. **磁盤空間**：為圖形文件預留足夠空間
3. **定期清理**：清理舊的圖形數據避免累積
4. **並行分析**：訓練期間可以並行分析已生成的圖形數據
5. **數據備份**：重要的可視化結果應該備份保存

## 📞 支援與反饋

如果遇到問題或有改進建議，請：
1. 檢查日誌文件中的詳細錯誤信息
2. 確認環境配置和依賴套件
3. 提供具體的錯誤復現步驟
4. 附上相關的配置和日誌文件

---

🎉 現在你已經掌握了GNNRL圖形數據可視化系統的完整用法！開始探索你的訓練數據，發現隱藏的模式和洞察吧！