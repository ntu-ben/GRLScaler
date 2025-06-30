# 實驗管理系統使用指南

## 🚀 快速開始

### 完整實驗流程
```bash
# 執行完整三方法實驗 (推薦)
python run_complete_experiment.py

# 自定義參數
python run_complete_experiment.py --steps 3000 --goal cost --model gcn
```

### 階段性執行 (新功能!)

#### 1. 只執行特定階段
```bash
# 只進行實驗規劃
python run_complete_experiment.py --stage plan

# 只執行 Gym-HPA 實驗
python run_complete_experiment.py --stage gym-hpa

# 只執行 GNNRL 實驗
python run_complete_experiment.py --stage gnnrl

# 只執行 K8s-HPA 基準測試
python run_complete_experiment.py --stage k8s-hpa

# 只進行結果分析
python run_complete_experiment.py --stage analysis
```

#### 2. 跳過特定階段
```bash
# 跳過規劃，使用現有計劃執行實驗
python run_complete_experiment.py --skip-stages plan

# 跳過 Gym-HPA 和 GNNRL，只做 K8s-HPA 基準測試
python run_complete_experiment.py --skip-stages gym-hpa gnnrl

# 跳過分析階段
python run_complete_experiment.py --skip-stages analysis
```

## 🛠️ 常見使用場景

### 場景 1: 第一次運行實驗
```bash
# 完整流程，會自動檢測現有模型並詢問是否使用
python run_complete_experiment.py
```

### 場景 2: 只想測試 K8s-HPA (修復版本)
```bash
# 直接執行 K8s-HPA 測試
python run_complete_experiment.py --stage k8s-hpa
```

### 場景 3: 已有模型，只想重新測試
```bash
# 先規劃 (選擇使用現有模型)
python run_complete_experiment.py --stage plan

# 然後執行測試
python run_complete_experiment.py --skip-stages plan
```

### 場景 4: 開發調試模式
```bash
# 只規劃，查看會執行什麼
python experiment_planner.py

# 單獨測試某個實驗
python run_complete_experiment.py --stage gym-hpa
```

## 🔧 問題解決

### K8s-HPA 錯誤修復
**問題**: `run_distributed_locust() missing 1 required positional argument: 'out_dir'`
**解決**: 已修復 `unified_experiment_manager.py` 中的函數調用錯誤

### 模型路徑問題修復  
**問題**: 找不到模型或路徑錯誤
**解決**: 重寫為 Python 版本，使用正確的模型檢測模式：
- Gym-HPA: `*online_boutique_gym*{steps}*.zip`
- GNNRL: `gnnrl*{steps}*.zip`

## 📁 檔案結構

```
├── run_complete_experiment.py    # 主實驗執行器 (Python 版本)
├── experiment_planner.py         # 實驗規劃器
├── unified_experiment_manager.py # 底層實驗管理器 (已修復)
├── experiment_plan.json          # 實驗計劃檔案 (自動生成)
└── logs/
    ├── models/                   # 訓練好的模型
    ├── gym-hpa/                  # Gym-HPA 實驗結果
    ├── gnnrl/                    # GNNRL 實驗結果
    └── k8s_hpa/                  # K8s-HPA 實驗結果
```

## 🎯 參數說明

| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--steps` | 訓練步數 | 5000 | `--steps 3000` |
| `--goal` | 優化目標 | latency | `--goal cost` |
| `--model` | GNNRL 模型類型 | gat | `--model gcn` |
| `--stage` | 只執行指定階段 | - | `--stage k8s-hpa` |
| `--skip-stages` | 跳過指定階段 | - | `--skip-stages plan analysis` |

## 💡 提示

1. **首次運行**: 建議使用完整流程 `python run_complete_experiment.py`
2. **偵錯模式**: 使用 `--stage` 參數單獨測試各階段
3. **重複實驗**: 規劃一次後可使用 `--skip-stages plan` 重複執行
4. **模型管理**: 系統會自動檢測現有模型並提供選擇
5. **錯誤恢復**: 可以從任何階段重新開始執行