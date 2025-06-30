# 🔬 實驗結果分析指南

## 📋 實驗概述

本次實驗比較三種 Kubernetes 自動縮放方法：
- **Gym-HPA**: 基礎強化學習 (PPO + MLP)
- **GNNRL**: 圖神經網路強化學習 (PPO + GAT)  
- **K8s-HPA**: 原生 HPA 基準測試 (CPU閾值)

所有實驗使用 **Seed 42** 確保可重現性，在 OnlineBoutique 微服務上進行測試。

## 📁 結果檔案結構

```
logs/
├── experiment_comparison.csv         # 跨方法比較結果
├── hpa_scenario_sequence.txt        # 測試場景序列
├── models/                          # 訓練模型
│   ├── ppo_env_*_gym_*.zip         # Gym-HPA 模型
│   └── gnnrl_gat_*.zip             # GNNRL 模型
├── gym-hpa/
│   ├── tensorboard/                # 訓練過程可視化
│   ├── gym_hpa_train_*/           # 訓練期間負載測試
│   └── gym_hpa_test_*/            # 測試期間負載測試
├── gnnrl/
│   ├── tensorboard/                # 訓練過程可視化
│   ├── gnnrl_train_*/             # 訓練期間負載測試
│   └── gnnrl_test_*/              # 測試期間負載測試
└── k8s-hpa/
    └── k8s_hpa_cpu_*/             # HPA配置測試結果
        ├── cpu-20/                # CPU 20% 閾值
        ├── cpu-40/                # CPU 40% 閾值
        ├── cpu-60/                # CPU 60% 閾值
        └── cpu-80/                # CPU 80% 閾值
```

## 🔍 分析步驟

### 1. 快速總覽

```bash
# 查看整體比較結果
cat logs/experiment_comparison.csv

# 查看測試場景序列 
cat logs/hpa_scenario_sequence.txt

# 生成詳細分析報告
python analyze_results.py
```

### 2. TensorBoard 可視化

```bash
# 查看所有方法的訓練過程
tensorboard --logdir logs

# 只查看特定方法
tensorboard --logdir logs/gym-hpa/tensorboard    # Gym-HPA
tensorboard --logdir logs/gnnrl/tensorboard      # GNNRL
```

**TensorBoard 關鍵指標**：
- `episode_reward_mean`: 平均回報
- `episode_length_mean`: 平均步數
- `learning_rate`: 學習率衰減
- `policy_loss`: 策略損失
- `value_loss`: 價值函數損失

### 3. 性能指標比較

#### 📊 主要比較維度

| 指標 | 說明 | 期望表現 |
|------|------|----------|
| **Average Response Time** | 平均響應時間 | 越低越好 |
| **95%ile Response Time** | 95分位響應時間 | 越低越好 |
| **Requests/Second (RPS)** | 每秒請求數 | 越高越好 |
| **Total Requests** | 總請求數 | 反映測試強度 |
| **Failure Rate** | 失敗率 | 越低越好 |

#### 🎯 分析重點

1. **響應時間穩定性**
   - 比較各方法的響應時間分佈
   - 關注 95%ile 和 99%ile 指標

2. **縮放效率** 
   - 觀察不同負載下的自動縮放行為
   - 分析是否過度縮放或縮放不足

3. **學習效果**
   - Gym-HPA vs GNNRL 的學習曲線
   - 訓練過程中的收斂性

4. **HPA 閾值影響**
   - 不同 CPU 閾值對性能的影響
   - 找出最佳閾值設定

### 4. 詳細數據分析

#### 查看單個實驗結果

```bash
# 查看 Gym-HPA 訓練結果
ls -la logs/gym-hpa/gym_hpa_train_*/

# 查看 GNNRL 測試結果  
ls -la logs/gnnrl/gnnrl_test_*/

# 查看特定 HPA 配置結果
ls -la logs/k8s-hpa/k8s_hpa_cpu_*/cpu-40/
```

#### 分析 Locust 測試數據

```bash
# 查看詳細統計
head -5 logs/gym-hpa/*/offpeak_001/*_stats.csv
head -5 logs/gnnrl/*/peak_001/*_stats.csv
head -5 logs/k8s-hpa/*/cpu-40/offpeak_001/*_stats.csv
```

#### Kiali 服務網格分析

```bash
# 查看服務間流量圖表
ls logs/*/kiali_*.json

# 分析服務調用關係和響應時間
jq '.elements.edges[].data.responseTime' logs/gnnrl/*/kiali_start.json
```

### 5. 統計分析腳本

創建自定義分析腳本：

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 讀取比較結果
df = pd.read_csv('logs/experiment_comparison.csv')

# 繪製性能比較圖
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 響應時間比較
axes[0,0].bar(df['Method'], df['Avg Response Time (ms)'])
axes[0,0].set_title('Average Response Time')
axes[0,0].set_ylabel('ms')

# RPS 比較
axes[0,1].bar(df['Method'], df['Avg RPS'])
axes[0,1].set_title('Requests per Second')
axes[0,1].set_ylabel('RPS')

# 95%ile 比較
axes[1,0].bar(df['Method'], df['Avg P95 (ms)'])
axes[1,0].set_title('95th Percentile Response Time')
axes[1,0].set_ylabel('ms')

# 總請求數比較
axes[1,1].bar(df['Method'], df['Total Requests'])
axes[1,1].set_title('Total Requests')
axes[1,1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('logs/performance_comparison.png')
plt.show()
```

## 📈 期望結果解讀

### 🏆 性能排名預期

1. **GNNRL** (圖神經網路)
   - 應該有最佳的整體性能
   - 能理解服務間依賴關係
   - 更智能的縮放決策

2. **Gym-HPA** (基礎強化學習)
   - 中等性能，比 HPA 更靈活
   - 學習能力有限於 MLP 架構
   - 比傳統方法更適應性強

3. **K8s-HPA** (原生 HPA)
   - 基準性能，相對穩定
   - 反應可能較慢或過度縮放
   - 不同閾值會有明顯差異

### ⚠️ 需要關注的異常

- **訓練不收斂**: 學習曲線持續振盪
- **響應時間異常**: 某方法明顯較差
- **失敗率過高**: 系統過載或配置錯誤
- **縮放異常**: Pod 數量異常變化

## 🔧 故障排除

### 常見問題

1. **TensorBoard 無法載入**
   ```bash
   # 檢查檔案權限
   ls -la logs/*/tensorboard/
   
   # 重新啟動 TensorBoard
   pkill tensorboard
   tensorboard --logdir logs --reload_interval 1
   ```

2. **測試結果不完整**
   ```bash
   # 檢查測試是否完成
   find logs/ -name "*_stats.csv" | wc -l
   
   # 查看錯誤日誌
   grep -r "ERROR\|Failed" logs/
   ```

3. **模型載入失敗**
   ```bash
   # 檢查模型檔案
   ls -la logs/models/
   file logs/models/*.zip
   ```

## 📋 報告模板

### 實驗結果摘要

```markdown
# 三方法自動縮放性能比較

## 實驗設定
- 種子: 42
- 訓練步數: 5,000
- 測試場景: OnlineBoutique
- 測試時間: [填入]

## 結果摘要

| 方法 | 平均響應時間 | 95%ile | RPS | 總請求數 |
|------|-------------|--------|-----|----------|
| GNNRL | X ms | X ms | X | X |
| Gym-HPA | X ms | X ms | X | X |
| K8s-HPA | X ms | X ms | X | X |

## 關鍵發現
1. [填入最佳方法及原因]
2. [填入學習效果觀察]
3. [填入HPA閾值建議]

## 建議
1. [填入生產環境建議]
2. [填入進一步實驗方向]
```

## 🚀 下一步

1. **深入分析**: 使用 Jupyter Notebook 進行更詳細的數據分析
2. **參數調優**: 基於結果調整超參數重新實驗
3. **擴展實驗**: 測試不同工作負載模式
4. **生產部署**: 選擇最佳方法進行生產環境驗證