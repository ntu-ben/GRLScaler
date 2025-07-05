# GRLScaler - Kubernetes 自動擴展三方法比較平台

🚀 **完整的 Kubernetes 微服務自動擴展解決方案**，比較三種先進的自動擴展方法：

- **🧠 GNNRL**: 圖神經網路強化學習 (Graph Neural Network + Reinforcement Learning)
- **🎯 Gym-HPA**: 基礎強化學習 (Gymnasium + PPO)  
- **⚖️ K8s-HPA**: 原生 Kubernetes HPA (Horizontal Pod Autoscaler)

## 📋 目錄

- [功能特色](#功能特色)
- [系統架構](#系統架構)
- [快速開始](#快速開始)
- [實驗方法](#實驗方法)
- [GNNRL 測試模式](#gnnrl-測試模式)
- [結果分析](#結果分析)
- [使用指南](#使用指南)
- [開發文檔](#開發文檔)

## ✨ 功能特色

### 🔬 三種自動擴展方法
- **GNNRL**: 利用服務依賴圖進行智能擴展決策
- **Gym-HPA**: 基於 PPO 算法的強化學習擴展  
- **K8s-HPA**: 基於 CPU/Memory 閾值的傳統擴展

### 📊 完整實驗平台
- 統一的實驗管理器
- 多負載模式測試（peak、off-peak、fluctuating、rush-sale）
- 分散式 Locust 負載測試
- 自動化實驗結果收集與分析

### 🎯 智能實驗規劃
- 自動檢測現有訓練模型
- 用戶友好的選擇界面
- 支援跳過特定實驗
- 階段式執行（訓練、測試、分析）

## 🏗️ 系統架構

```
GRLScaler/
├── 🧠 gnnrl/                     # GNNRL 圖神經網路強化學習
│   ├── core/envs/                # K8s 環境接口
│   └── training/                 # 訓練與測試腳本
├── 🎯 gym-hpa/                   # Gym-HPA 基礎強化學習  
│   ├── gym_hpa/envs/            # Gymnasium 環境
│   └── policies/                 # PPO 策略實現
├── ⚖️ k8s_hpa/                   # K8s-HPA 原生擴展
├── 🧪 loadtest/                  # 負載測試腳本
├── 📊 logs/                      # 實驗數據與模型
│   ├── models/                   # 訓練好的模型
│   ├── gnnrl/                    # GNNRL 實驗結果
│   ├── gym-hpa/                  # Gym-HPA 實驗結果
│   └── k8s-hpa/                  # K8s-HPA 實驗結果
└── 🔧 統一管理工具               # 實驗執行與分析
```

## 🚀 快速開始

### 1. 環境準備
```bash
# 確保 Kubernetes 集群運行
kubectl get nodes

# 部署 OnlineBoutique 微服務
kubectl apply -f k8s-manifests/

# 安裝依賴
pip install -r requirements.txt
```

### 2. 一鍵運行完整實驗
```bash
python run_complete_experiment.py
```

### 3. 分析結果
```bash
python analyze_comprehensive.py
```

## 🧪 實驗方法

### 實驗階段
1. **🎯 訓練階段**: 訓練 ML 模型（GNNRL、Gym-HPA）
2. **🧪 測試階段**: 使用訓練好的模型進行性能評估
3. **📊 分析階段**: 比較三種方法的性能指標

### 負載模式
- **Off-peak** (低負載): 50-100 用戶
- **Peak** (高負載): 500 用戶  
- **Fluctuating** (波動負載): 動態變化
- **Rush Sale** (突發負載): 快速增長到高峰

### 評估指標
- **響應時間**: 平均、P95、P99
- **吞吐量**: RPS (Requests Per Second)  
- **穩定性**: 失敗率、抖動率
- **資源效率**: CPU/Memory 利用率
- **成本效益**: Pod-時間、資源浪費率

## 🧠 GNNRL 測試模式

### 📋 背景
原始實現中 GNNRL 只有訓練數據，缺少測試階段數據，導致與其他方法的比較不公平。

### ✅ 解決方案
我們實現了完整的 GNNRL 測試模式：

#### 1. 新增功能
- ✅ `--testing` 模式支援
- ✅ `--load-path` 模型載入  
- ✅ 模型評估流程
- ✅ 測試數據生成

#### 2. 使用方式

**方式 1: 直接測試**
```bash
python run_gnnrl_test_mode.py
```

**方式 2: 完整實驗流程**
```bash
python run_complete_experiment.py
# 選擇使用現有 GNNRL 模型
```

**方式 3: 手動執行**
```bash
python unified_experiment_manager.py \
  --experiment gnnrl \
  --testing \
  --load-path logs/models/gnnrl_gat_latency_k8s_True_steps_5000.zip \
  --k8s --goal latency --model gat --alg ppo
```

#### 3. 效果
現在可以進行公平比較：
- **GNNRL**: 訓練數據 + 測試數據 ✅
- **Gym-HPA**: 訓練數據 + 測試數據 ✅  
- **K8s-HPA**: 測試數據 ✅

## 📊 結果分析

### 分析工具
```bash
# 全面分析
python analyze_comprehensive.py

# 基礎分析  
python analyze_results.py

# 額外指標分析
python ADDITIONAL_METRICS_ANALYSIS.md
```

### 主要發現
基於最新實驗數據（僅測試階段）：

| 方法 | 平均響應時間 | 失敗率 | 平均 RPS | P95 延遲 |
|-----|-------------|-------|----------|----------|
| **K8s-HPA** | **1,087.80ms** | **0.95%** | 127.93 | **1,776ms** |
| **Gym-HPA** | 1,403.07ms | 2.50% | 74.12 | 2,268ms |
| **GNNRL** | 📊 待測試 | 📊 待測試 | 📊 待測試 | 📊 待測試 |

> 📝 GNNRL 測試數據將在運行測試模式後可用

### 性能洞察
1. **K8s-HPA** 在延遲一致性方面表現最佳
2. **Gym-HPA** 在某些場景中響應時間優異
3. **GNNRL** 在吞吐量方面具有潛力（基於訓練數據）

## 📚 使用指南

### 單一實驗執行
```bash
# GNNRL 訓練
python unified_experiment_manager.py --experiment gnnrl --steps 5000

# GNNRL 測試  
python unified_experiment_manager.py --experiment gnnrl --testing \
  --load-path logs/models/gnnrl_gat_latency_k8s_True_steps_5000.zip

# Gym-HPA 實驗
python unified_experiment_manager.py --experiment gym_hpa --steps 5000

# K8s-HPA 實驗
python unified_experiment_manager.py --experiment k8s_hpa
```

### 階段式執行
```bash
# 只執行訓練階段
python run_complete_experiment.py --stage training

# 只執行測試階段  
python run_complete_experiment.py --stage testing

# 只執行分析階段
python run_complete_experiment.py --stage analysis
```

### 自定義配置
```bash
# 自定義步數和目標
python run_complete_experiment.py --steps 10000 --goal cost

# 跳過特定實驗
python run_complete_experiment.py --skip-stages gnnrl,gym_hpa
```

## 📖 開發文檔

### 核心檔案
- **`run_complete_experiment.py`**: 主要實驗執行器
- **`unified_experiment_manager.py`**: 統一實驗管理器  
- **`experiment_planner.py`**: 智能實驗規劃器
- **`analyze_comprehensive.py`**: 全面結果分析器

### 實驗數據結構
```
logs/
├── models/                           # 訓練好的模型
│   ├── gnnrl_gat_latency_k8s_True_steps_5000.zip
│   └── ppo_env_online_boutique_gym_goal_latency_k8s_True_totalSteps_5000.zip
├── gnnrl/
│   ├── gnnrl_train_seed42_*/        # 訓練數據
│   └── gnnrl_test_seed42_*/         # 測試數據 🆕
├── gym-hpa/
│   ├── gym_hpa_train_seed42_*/      # 訓練數據  
│   └── gym_hpa_test_seed42_*/       # 測試數據
└── k8s-hpa/
    └── k8s_hpa_cpu_seed42_*/        # 測試數據
```

### 配置文件
- **`experiment_config.yaml`**: 實驗參數配置
- **`CLAUDE.md`**: Claude AI 使用說明
- **各種分析報告**: `*_ANALYSIS_*.md`

## 🤝 貢獻指南

1. Fork 此專案
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)  
5. 開啟 Pull Request

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 🙏 致謝

- **Kubernetes 社群** - 提供強大的容器編排平台
- **Stable Baselines3** - 優秀的強化學習框架
- **Locust** - 靈活的負載測試工具
- **OnlineBoutique** - Google 提供的微服務範例應用

---

📊 **準備好開始你的 Kubernetes 自動擴展實驗了嗎？**

```bash
git clone <repository-url>
cd GRLScaler  
python run_complete_experiment.py
```

🎯 **讓數據說話，找出最適合你的自動擴展策略！**