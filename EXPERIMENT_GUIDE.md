# GRLScaler 統一實驗指南

## 概述

這個統一實驗系統整合了三種自動擴展實驗，支援分散式 Locust 負載測試環境：

- **gym_hpa**: 基礎強化學習 (MLP 策略)
- **k8s_hpa**: Kubernetes HPA 基準測試  
- **gnnrl**: 圖神經網路強化學習

## 快速開始

### 1. 環境準備

確保以下環境已設置：

```bash
# Kubernetes 環境
kubectl get pods -n onlineboutique  # 確認服務運行

# 環境變數 (.env 檔案)
M1_HOST=http://10.0.0.2              # 分散式測試代理
TARGET_HOST=http://k8s.orb.local    # 目標服務
LOCUST_RUN_TIME=15m                  # 測試時長
```

### 2. 執行單一實驗

```bash
# 執行 GNNRL 實驗
python unified_experiment_manager.py --experiment gnnrl --steps 5000

# 執行 Gym-HPA 實驗  
python unified_experiment_manager.py --experiment gym_hpa --steps 3000 --goal latency

# 執行 HPA 基準測試
python unified_experiment_manager.py --experiment k8s_hpa
```

### 3. 批次實驗

```bash
# 執行所有實驗
python unified_experiment_manager.py --batch-all --steps 5000

# 執行指定實驗組合
python unified_experiment_manager.py --experiments gym_hpa gnnrl --steps 3000
```

### 4. 僅環境驗證

```bash
python unified_experiment_manager.py --validate-only
```

## 分散式測試環境

### 遠端 Locust 代理

設置 `M1_HOST` 環境變數指向 Locust 代理服務器：

```bash
# .env 檔案
M1_HOST=http://10.0.0.2:8099
```

代理服務器需要運行 `loadtest/locust_agent.py`：

```bash
# 在代理服務器上
cd loadtest
uvicorn locust_agent:app --host 0.0.0.0 --port 8099
```

### 本地 Fallback

如果遠端代理不可用，系統會自動回退到本地測試。

## 實驗配置

### 參數說明

- `--steps`: 訓練步數 (預設: 5000)
- `--goal`: 優化目標 (`latency` 或 `cost`)
- `--use-case`: 應用場景 (`online_boutique` 或 `redis`)
- `--run-tag`: 自定義運行標籤

### 負載測試情境

系統包含 4 種測試情境：

1. **offpeak**: 低流量基準 (~50 RPS)
2. **rushsale**: 搶購衝擊 (~500 RPS)  
3. **peak**: 高峰持續 (~300 RPS)
4. **fluctuating**: 波動流量 (150-400 RPS)

## 結果分析

### 日誌結構

```
logs/
├── gym_hpa/
│   └── run_20250623_143022/
│       ├── batch.log
│       ├── summary.csv
│       ├── aggregate.html
│       ├── offpeak/
│       ├── rushsale/
│       ├── peak/
│       └── fluctuating/
├── gnnrl/
└── hpa/
```

### 關鍵指標

- **Requests**: 總請求數
- **Failures**: 失敗請求數  
- **Avg RPS**: 平均每秒請求數
- **P95 ms**: 95% 延遲
- **Kiali RPS**: Kiali 監控的 RPS

## 進階使用

### 自定義實驗

修改 `experiment_config.yaml` 來調整實驗參數：

```yaml
experiments:
  gnnrl:
    default_args:
      k8s: true
      steps: 10000        # 增加訓練步數
      model: "gcn"        # 使用 GCN 模型
      embed_dim: 64       # 增加嵌入維度
```

### 監控整合

系統支援多種監控工具：

- **Kiali**: 服務圖快照
- **Prometheus**: 指標收集
- **Linkerd**: 服務網格統計

### 實驗同步

訓練過程與負載測試自動同步：

1. 訓練開始後，等待環境穩定
2. 依序執行 5 種負載情境
3. 情境間有冷卻期 (60秒)
4. 監控訓練進程狀態

## 故障排除

### 常見問題

1. **分散式測試失敗**
   - 檢查 M1_HOST 連接
   - 驗證代理服務運行狀態
   - 系統會自動 fallback 到本地

2. **K8s 環境問題**
   - 確認 onlineboutique namespace 
   - 檢查 Pod 運行狀態
   - 驗證服務健康檢查

3. **實驗腳本錯誤**
   - 檢查路徑配置
   - 驗證依賴安裝
   - 查看詳細日誌

### 日誌檢查

```bash
# 查看統一管理器日誌
tail -f unified_experiment.log

# 查看實驗日誌
tail -f logs/gnnrl/run_*/batch.log
```

## 實驗比較

使用內建比較功能分析不同實驗結果：

```bash
python unified_experiment_manager.py --compare \
  logs/gym_hpa/run1 \
  logs/gnnrl/run2 \
  logs/hpa/baseline
```

這將生成比較報告，包含性能指標對比和可視化圖表。