# 🚀 GRLScaler - 圖神經網路增強的 Kubernetes 自動擴展平台

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-1.20+-blue.svg)](https://kubernetes.io/)

**GRLScaler** 是一個基於圖神經網路和強化學習的 Kubernetes 自動擴展研究平台，支援多種環境和自動擴展方法的性能比較。

## 🎯 核心功能

### 📊 支援的環境
- **OnlineBoutique** - Google 微服務電商平台 (10個微服務)
- **Redis** - 高性能內存數據庫 (Master-Slave 架構)

### 🧠 支援的自動擴展方法
1. **GNNRL** - 圖神經網路強化學習 (Graph Neural Network + Reinforcement Learning)
2. **Gym-HPA** - 基礎強化學習 (Proximal Policy Optimization)
3. **K8s-HPA** - Kubernetes 原生 Horizontal Pod Autoscaler

### 🔬 實驗特性
- ✅ **標準化場景** - 8個固定種子場景確保公平比較
- ✅ **多維度評估** - 吞吐量、響應時間、失敗率、資源效率
- ✅ **實時監控** - Kiali、Prometheus、Istio 集成
- ✅ **結果分析** - 自動生成性能報告和比較圖表

## 🚀 快速開始

### 前置需求

```bash
# Kubernetes 集群 (推薦 1.20+)
kubectl version

# Python 環境
python --version  # 3.8+

# 必要的 Python 套件
pip install -r requirements.txt
```

### 一鍵啟動實驗

```bash
# 1. 克隆項目
git clone <repository-url>
cd GRLScaler

# 2. 部署測試環境
kubectl apply -f MicroServiceBenchmark/  # OnlineBoutique
kubectl apply -f MicroServiceBenchmark/redis-cluster/  # Redis

# 3. 驗證環境
python run_autoscaling_experiment.py onlineboutique --verify
python run_autoscaling_experiment.py redis --verify

# 4. 執行實驗
python run_autoscaling_experiment.py onlineboutique --steps 5000
```

## 📋 詳細使用指南

### OnlineBoutique 實驗

```bash
# 完整三方法比較實驗 (推薦)
python run_autoscaling_experiment.py onlineboutique --standardized --steps 5000

# 只測試特定方法
python run_autoscaling_experiment.py onlineboutique --method gnnrl --steps 3000
python run_autoscaling_experiment.py onlineboutique --method gym-hpa --steps 3000
python run_autoscaling_experiment.py onlineboutique --method k8s-hpa

# 跳過特定階段
python run_autoscaling_experiment.py onlineboutique --skip plan analysis --steps 3000
```

### Redis 實驗

```bash
# 完整 Redis 自動擴展實驗
python run_autoscaling_experiment.py redis --steps 5000

# 快速驗證環境
python run_autoscaling_experiment.py redis --verify

# 測試 HPA 配置
python redis_hpa_test.py
```

### 進階選項

```bash
# 不同優化目標
python run_autoscaling_experiment.py onlineboutique --goal latency  # 延遲優先
python run_autoscaling_experiment.py onlineboutique --goal cost     # 成本優先

# 不同 GNNRL 模型
python run_autoscaling_experiment.py onlineboutique --model gat   # Graph Attention Network
python run_autoscaling_experiment.py onlineboutique --model gcn   # Graph Convolutional Network
python run_autoscaling_experiment.py onlineboutique --model sage  # GraphSAGE

# 查看可用配置
python run_autoscaling_experiment.py onlineboutique --list-configs
python run_autoscaling_experiment.py redis --list-configs
```

## 📊 實驗結果分析

### 自動分析報告

實驗完成後，系統會自動生成：

```bash
# OnlineBoutique 結果
logs/standardized_method_comparison.csv     # 三方法整體比較
logs/standardized_scenario_comparison.csv  # 場景級別詳細比較
STANDARDIZED_COMPARISON_REPORT.md          # 完整分析報告

# Redis 結果
logs/redis_hpa_comparison.csv              # Redis HPA 配置比較
logs/redis_method_comparison.csv           # Redis 三方法比較
```

### 手動分析

```bash
# 分析 OnlineBoutique 結果
python analyze_onlineboutique_results.py

# 分析一般結果
python analyze_results.py

# 啟動 TensorBoard
tensorboard --logdir logs/
```

## 🗂️ 項目結構

```
GRLScaler/
├── 📁 gnnrl/                           # GNNRL 圖神經網路強化學習
│   ├── core/envs/                      # 環境實現 (OnlineBoutique, Redis)
│   ├── training/                       # 訓練腳本
│   └── data/                          # 數據集和圖結構
├── 📁 gym-hpa/                        # Gym-HPA 基礎強化學習
│   ├── gym_hpa/envs/                  # Gym 環境
│   └── policies/                      # 策略實現
├── 📁 macK8S/HPA/                     # K8s-HPA 配置
│   ├── onlineboutique/                # OnlineBoutique HPA 配置
│   └── redis/                         # Redis HPA 配置
├── 📁 loadtest/                       # 負載測試腳本
│   ├── onlineboutique/                # OnlineBoutique 測試場景
│   └── redis/                         # Redis 測試場景
├── 📁 logs/                           # 實驗結果和模型
└── 📁 scripts/                        # 工具腳本
    ├── run_autoscaling_experiment.py  # 🚀 主要入口腳本
    ├── run_onlineboutique_experiment.py # OnlineBoutique 專用
    ├── run_redis_experiment.py        # Redis 專用
    ├── analyze_onlineboutique_results.py # 結果分析
    ├── redis_hpa_test.py              # Redis HPA 測試
    └── redis_environment_check.py     # Redis 環境檢查
```

## 🎯 核心腳本說明

| 腳本 | 用途 | 範例 |
|------|------|------|
| `run_autoscaling_experiment.py` | **統一入口** - 所有實驗的主要入口 | `python run_autoscaling_experiment.py onlineboutique --steps 5000` |
| `run_onlineboutique_experiment.py` | OnlineBoutique 微服務實驗 | `python run_onlineboutique_experiment.py --standardized --steps 5000` |
| `run_redis_experiment.py` | Redis 數據庫實驗 | `python run_redis_experiment.py --steps 5000` |
| `analyze_onlineboutique_results.py` | OnlineBoutique 結果分析 | `python analyze_onlineboutique_results.py` |
| `redis_hpa_test.py` | Redis HPA 配置測試 | `python redis_hpa_test.py` |
| `redis_environment_check.py` | Redis 環境驗證 | `python redis_environment_check.py` |

## 📈 性能基準

### OnlineBoutique 實驗結果 (基於 8 個標準化場景)

| 方法 | 平均 RPS | 平均響應時間 | 失敗率 | 綜合評分 |
|------|----------|--------------|--------|----------|
| **GNNRL** | **197.07** | **384.89ms** | **0.05%** | ⭐⭐⭐⭐⭐ |
| **K8s-HPA (CPU-20%)** | 274.28 | 808.93ms | 0.81% | ⭐⭐⭐⭐ |
| **Gym-HPA** | 179.93 | 514.94ms | 0.13% | ⭐⭐⭐ |

### Redis 實驗結果

| HPA 配置 | 場景支援 | 建議用途 |
|----------|----------|----------|
| **CPU-20%** | 高敏感度擴展 | 延遲敏感應用 |
| **CPU-40%** | 平衡性能 | 一般生產環境 |
| **CPU-80%** | 資源節約 | 成本敏感環境 |
| **CPU+Memory** | 複合指標 | 複雜工作負載 |

## 🔧 進階配置

### 自定義 HPA 配置

```bash
# 生成新的 HPA 配置
python macK8S/HPA/redis/generate_redis_hpa.py

# 測試自定義配置
python redis_hpa_test.py --config custom-cpu-30
```

### 自定義負載場景

```python
# 在 loadtest/ 目錄下創建新場景
# 參考現有的 locust_*.py 文件
```

### 環境變數配置

```bash
export M1_HOST="http://your-loadtest-agent:8000"  # 分散式測試
export KIALI_URL="http://your-kiali:20001"        # Kiali 監控
export PROMETHEUS_URL="http://your-prometheus:9090" # Prometheus 監控
```

## 🐛 問題排除

### 常見問題

1. **Kubernetes 連接失敗**
   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```

2. **服務未就緒**
   ```bash
   kubectl get pods -n onlineboutique
   kubectl get pods -n redis
   ```

3. **分散式測試失敗**
   ```bash
   # 檢查測試代理
   curl $M1_HOST
   ```

4. **HPA 不生效**
   ```bash
   kubectl get hpa -A
   kubectl describe hpa -n <namespace>
   ```

### 日誌檢查

```bash
# 檢查實驗日誌
tail -f logs/*/latest_experiment.log

# 檢查 Pod 日誌
kubectl logs -n onlineboutique deployment/frontend
kubectl logs -n redis deployment/redis-master
```

## 🤝 貢獻指南

1. Fork 項目
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 創建 Pull Request

## 📜 授權條款

本項目採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

- **Google Cloud** - OnlineBoutique 微服務範例
- **Kubernetes** - 容器編排平台
- **Istio** - 服務網格
- **PyTorch Geometric** - 圖神經網路庫
- **OpenAI Gym** - 強化學習環境

## 📧 聯絡資訊

如有問題或建議，請透過以下方式聯絡：
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/GRLScaler/issues)
- 📖 文檔: [項目 Wiki](https://github.com/your-repo/GRLScaler/wiki)

---

**⭐ 如果這個項目對你有幫助，請給我們一個 Star！**