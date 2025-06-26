# GRLScaler

本專案由 **國立台灣大學電機工程學研究所 NMLAB** 研究團隊維護，旨在研究於 Kubernetes 平台上使用強化學習進行自動擴縮。部分程式碼來自 [gym-hpa](https://github.com/jpedro1992/gym-hpa)，並在此基礎上加入圖神經網路與真實叢集測試工具。

建議使用 **Python 3.10** 執行本專案。

## 📋 目錄

- [系統需求](#系統需求)
- [快速開始](#快速開始)
- [實驗類型](#實驗類型)
- [環境配置](#環境配置)
- [分散式測試](#分散式測試)
- [專案結構](#專案結構)
- [進階使用](#進階使用)
- [常見問題](#常見問題)

## 🔧 系統需求

### 基本環境
- Python 3.10+
- kubectl (Kubernetes 命令行工具)
- 可選：python-dotenv (環境變數管理)

### Kubernetes 設定
1. 安裝 Istio 與 [Kiali](https://kiali.io/)，確保 Prometheus 能存取 `/metrics`
2. 為 `onlineboutique` 命名空間啟用 sidecar injection，使服務可由 Istio 監控
3. 依需要安裝 Prometheus，其 Helm `values` 皆收錄於 [`macK8S/`](macK8S/)

## 🚀 快速開始

### 1. 安裝依賴
```bash
pip install -e .
pip install -r requirements.txt
```

### 2. 配置環境
```bash
# 複製環境變數範本
cp .envTemplate .env

# 編輯環境變數 (可選)
vim .env
```

### 3. 啟動 Kubernetes 代理 (K8s 模式)
```bash
kubectl proxy --port=8001 &
```

### 4. 執行實驗

#### 模擬模式 (使用預存數據)
```bash
# GNNRL 實驗
python unified_experiment_manager.py --experiment gnnrl --steps 5000

# Gym-HPA 實驗  
python unified_experiment_manager.py --experiment gym_hpa --steps 3000
```

#### 真實 K8s 集群模式
```bash
# GNNRL 實驗
python unified_experiment_manager.py --experiment gnnrl --k8s --steps 5000

# Gym-HPA 實驗
python unified_experiment_manager.py --experiment gym_hpa --k8s --steps 3000

# HPA 基準測試
python unified_experiment_manager.py --experiment k8s_hpa --k8s
```

#### 批次實驗
```bash
# 執行所有實驗
python unified_experiment_manager.py --batch-all --k8s --steps 5000

# 環境驗證
python unified_experiment_manager.py --validate-only
```

## 🧪 實驗類型

| 實驗名稱 | 描述 | 支援場景 | 模式 |
|---------|------|----------|------|
| **gnnrl** | 圖神經網路強化學習 | OnlineBoutique (主要), Redis (有限) | 模擬 + K8s |
| **gym_hpa** | 基礎強化學習 (MLP) | OnlineBoutique, Redis | 模擬 + K8s |
| **k8s_hpa** | Kubernetes HPA 基準 | OnlineBoutique, Redis | 僅 K8s |

### 應用場景選擇

**OnlineBoutique (預設)**：Google 的微服務電商平台
```bash
python unified_experiment_manager.py --experiment gnnrl --k8s --use-case online_boutique --steps 5000
```

**Redis**：分散式緩存系統
```bash
python unified_experiment_manager.py --experiment gym_hpa --k8s --use-case redis --steps 5000
```

## ⚙️ 環境配置

在專案根目錄建立 `.env` 檔，內容可參考 `.envTemplate`：

| 變數 | 用途 | 預設值 |
|------|------|--------|
| `TARGET_HOST` | Locust 測試目標服務 URL | `http://k8s.orb.local:8080` |
| `M1_HOST` | 遠端 Locust agent 位址 | 無 (使用本地測試) |
| `PROMETHEUS_URL` | Prometheus 服務位址 | `http://localhost:9090/` |
| `KUBE_HOST` | Kubernetes proxy 位址 | `http://localhost:8001` |
| `KIALI_URL` | Kiali 服務位址 | `http://localhost:20001/kiali/` |
| `LOCUST_RUN_TIME` | Locust 執行時間 | `15m` |
| `NAMESPACE_ONLINEBOUTIQUE` | OnlineBoutique 命名空間 | `onlineboutique` |
| `NAMESPACE_REDIS` | Redis 命名空間 | `redis` |
| `DEFAULT_STEPS` | 預設訓練步數 | `5000` |
| `DEFAULT_GOAL` | 預設優化目標 | `latency` |

## 🌐 分散式測試

系統支援分散式 Locust 負載測試，可在遠端機器上部署測試代理。

### 設置遠端代理

#### 1. 在遠端機器上準備環境
```bash
# 方法一：複製 loadtest 目錄
scp -r loadtest/ user@remote-machine:/path/to/project/

# 方法二：Clone 整個專案
git clone https://github.com/your-repo/GRLScaler.git
cd GRLScaler
```

#### 2. 安裝依賴套件
```bash
pip install fastapi uvicorn locust python-dotenv
```

#### 3. 啟動代理服務
```bash
cd loadtest

# 前景執行 (開發測試)
uvicorn locust_agent:app --host 0.0.0.0 --port 8000

# 背景執行 (生產環境)
nohup uvicorn locust_agent:app --host 0.0.0.0 --port 8000 > agent.log 2>&1 &
```

#### 4. 在主機器配置環境變數
```bash
# 在 .env 檔案中設定遠端代理
echo "M1_HOST=http://REMOTE_MACHINE_IP:8000" >> .env
```

#### 5. 驗證連接
```bash
# 測試代理狀態
curl http://REMOTE_MACHINE_IP:8000/docs

# 查看 API 文檔
curl http://REMOTE_MACHINE_IP:8000/openapi.json
```

**注意事項**：
- 確保防火牆開放 8000 端口
- 代理機器需要完整的 `loadtest/onlineboutique/` 測試腳本
- 連接失敗時自動回退到本地測試

## 📁 專案結構

### 基本專案結構
```
├── gnnrl/                    # 圖神經網路強化學習
│   ├── core/                 # 核心模組
│   │   ├── envs/            # 環境實作 (OnlineBoutique, Redis)
│   │   ├── models/          # GNN 模型 (GAT, GCN)
│   │   └── agents/          # RL 代理 (PPO)
│   └── training/            # 訓練腳本
├── gym-hpa/                 # 基礎強化學習
│   ├── gym_hpa/envs/        # 環境定義
│   └── policies/run/        # 訓練腳本
├── k8s_hpa/                 # HPA 基準測試
├── loadtest/                # Locust 負載測試
│   ├── locust_agent.py      # 遠端代理服務
│   └── onlineboutique/      # 測試場景腳本
├── macK8S/                  # Kubernetes 設定檔
├── unified_experiment_manager.py  # 統一實驗管理器
├── experiment_path_manager.py     # 實驗路徑管理器
├── test_integration.py      # 整合測試
├── experiment_config.yaml   # 實驗配置
└── .envTemplate             # 環境變數範本
```

### 🗂️ 實驗結果統一結構 (新版)

所有實驗結果現在統一存放在 `experiments/` 目錄下，使用標準化的路徑結構：

```
experiments/
├── YYYYMMDD_HHMMSS_{type}_{alg}_{model}_{goal}_{steps}/  # 統一實驗目錄格式
│   ├── experiment_info.json                             # 實驗配置和元信息
│   ├── experiment_summary.json                          # 完整實驗摘要
│   ├── loadtest_summary.csv                            # 壓測指標摘要 (橫向比較用)
│   ├── action_history.csv                              # RL 動作歷史記錄
│   ├── training_log.txt                                # 訓練詳細日誌
│   ├── loadtest_scenarios/                             # 所有壓測場景結果
│   │   ├── scenario_001_peak_143055/                   # 場景結果目錄
│   │   │   ├── peak_stats.csv                         # Locust 統計結果
│   │   │   ├── peak_stats_history.csv                 # 時序統計
│   │   │   ├── peak.html                              # 結果報告
│   │   │   └── scenario_info.json                     # 場景元信息
│   │   ├── scenario_002_offpeak_144125/
│   │   └── scenario_N_fluctuating_HHMMSS/
│   ├── performance_charts/                             # 性能圖表 (未來功能)
│   │   ├── rps_timeline.png
│   │   ├── latency_distribution.png
│   │   └── replica_changes.png
│   └── models/                                         # 訓練好的模型
│       ├── final_model.zip
│       └── checkpoints/
├── comparison_reports/                                  # 橫向比較報告
│   ├── all_experiments_summary.csv                    # 所有實驗對比表
│   ├── performance_comparison.html                    # 可視化比較 (未來功能)
│   └── latest_comparison_YYYYMMDD.json                # 最新比較數據
└── archive/                                           # 歷史實驗存檔
    └── YYYY/MM/
```

#### 實驗 ID 命名規範

**格式**: `YYYYMMDD_HHMMSS_{experiment_type}_{algorithm}_{model}_{goal}_{steps}`

**範例**:
- `20250626_143022_gnnrl_a2c_gat_latency_2000` - GNNRL 實驗，A2C 算法，GAT 模型
- `20250626_150000_gym_hpa_ppo_mlp_cost_5000` - Gym-HPA 實驗，PPO 算法
- `20250626_160000_hpa_baseline_cpu80_latency_NA` - HPA 基準測試

#### 橫向比較報告

系統會自動生成標準化的比較報告，方便不同實驗間進行性能對比：

- **loadtest_summary.csv**: 每個實驗的詳細壓測指標
- **experiment_summary.json**: 完整的實驗配置、訓練結果和擴縮行為分析
- **all_experiments_summary.csv**: 所有實驗的關鍵指標對比表

#### 向後兼容

舊版的 `logs/` 目錄結構仍然保留，新系統會通過符號連接確保向後兼容性。

## 🔬 進階使用

### 直接使用實驗腳本
```bash
# GNNRL 實驗
python gnnrl/training/run_gnnrl_experiment.py --k8s --steps 5000

# Gym-HPA 實驗
python gym-hpa/policies/run/run.py --k8s --training --total-steps 5000 --use_case online_boutique

# HPA 基準測試
python k8s_hpa/HPABaseLineTest.py
```

### 查看實驗結果

#### 新版統一結構 (推薦)
```bash
# 查看所有實驗結果
ls experiments/

# 查看特定實驗詳情
ls experiments/20250626_143022_gnnrl_a2c_gat_latency_2000/

# 查看壓測摘要 (橫向比較用)
cat experiments/20250626_143022_gnnrl_a2c_gat_latency_2000/loadtest_summary.csv

# 查看完整實驗摘要
cat experiments/20250626_143022_gnnrl_a2c_gat_latency_2000/experiment_summary.json

# 查看橫向比較報告
cat experiments/comparison_reports/all_experiments_summary.csv
```

#### 舊版結構 (向後兼容)
```bash
# 日誌目錄 (符號連接到新結構)
ls logs/{experiment}/{run-tag}/

# TensorBoard 可視化
tensorboard --logdir=results/

# 比較實驗結果
python unified_experiment_manager.py --compare logs/gnnrl/run1 logs/gym_hpa/run2
```

### GNN + RL 架構

專案實現圖神經網路與強化學習的結合：

1. **資料收集**：每 30 秒抓取服務拓撲、容器指標與節點資源
2. **特徵轉換**：將服務關係轉成 PyG `HeteroData` 格式
3. **GNN 編碼**：使用 HeteroGAT/GCN 生成服務與節點嵌入
4. **RL 策略**：PPO 結合 GNN 特徵決定擴縮動作
5. **評估比較**：支援多種基線方法效能比較

## ❓ 常見問題

### Kiali 連線錯誤
```
ERROR:root:Kiali request failed: 404 Client Error
```
- 這是正常現象，不影響實驗執行
- 僅影響服務拓撲圖的獲取

### 分散式測試失敗
- 檢查 `M1_HOST` 網路連通性
- 驗證遠端代理服務狀態
- 系統會自動回退到本地測試

### K8s 環境問題
- 確認 `onlineboutique` namespace 存在
- 檢查 Pod 運行狀態：`kubectl get pods -n onlineboutique`
- 驗證服務健康檢查

### TensorBoard 日誌
- 日誌位置：`results/{use_case}/{scenario}/{goal}/`
- 啟動 TensorBoard：`tensorboard --logdir=results/`
- 瀏覽器開啟：`http://localhost:6006`

### 實驗結果管理
- **新版**: 所有結果存放在 `experiments/` 目錄
- **舊版**: 通過符號連接保持 `logs/` 兼容性
- **比較**: 使用 `experiments/comparison_reports/` 進行橫向分析
- **存檔**: 舊實驗自動移至 `experiments/archive/YYYY/MM/`

### 權限問題
- 確保對 `experiments/`, `logs/` 和 `results/` 目錄有寫入權限
- 檢查 kubectl 對 K8s 集群的存取權限

---

## 📄 相關文檔

- [操作指南](docs/Operating_Guide.md) - 詳細的操作說明
- [GNN+RL 架構](docs/GNN_RL_Autoscaler.md) - 技術架構詳解
- [實驗指南](docs/EXPERIMENT_GUIDE.md) - 實驗設計與執行

---

本倉庫僅供研究與教學用途，歡迎提出 issue 與貢獻。