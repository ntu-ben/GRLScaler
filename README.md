# GRLScaler

本專案由 **國立台灣大學電機工程學研究所 NMLAB** 研究團隊維護，旨在研究於 Kubernetes 平台上使用強化學習進行自動擴縮。部分程式碼來自 [gym-hpa](https://github.com/jpedro1992/gym-hpa)，並在此基礎上加入圖神經網路與真實叢集測試工具。

## 必要的 Kubernetes 設定

1. 安裝 [Linkerd](https://linkerd.io/)。建議在 `values.yaml` 中設定
   `proxy.defaultInboundPolicy: "cluster-unauthenticated"`，以便 Prometheus 能不經 mTLS 擷取 `/metrics`。
2. 為 `onlineboutique` 命名空間加註 `linkerd.io/inject: enabled`，讓所有部署自動注入 Linkerd sidecar。
3. 依需要安裝 Prometheus 與 Istio，相關 Helm `values` 皆收錄於 [`macK8S/`](macK8S/)。

## 在一台或兩台主機上重現實驗

執行 `rl_batch_loadtest.py` 即可啟動訓練並串接 Locust 載入測試。

```bash
python rl_batch_loadtest.py --model grl --tag myrun
```

若採用兩台電腦進行分散式負載測試，將 `M1_HOST` 環境變數設為遠端 agent 的位址，腳本會自動呼叫該節點執行 Locust。

若想以手動方式啟動長時間壓測，可執行 `loadtest/locust_agent_manual.py`，
預設持續 24 小時。

所有測試紀錄會輸出至 `logs/<method>/<tag>/` 目錄，其中 `method` 可能為
`grl`、`gym`、`gwydion` 或 `hpa`。HPA baseline 的結果將存放在
`logs/hpa/<hpa-name>/`。

### 主要環境變數

所有腳本可從 `.env` 讀取設定，下表列出常用變數：

| 變數 | 用途 |
|------|------|
| `TARGET_HOST` | Locust 測試目標服務 URL |
| `M1_HOST` | 遠端 Locust agent 位址（選用） |
| `PROMETHEUS_URL` | gnn_rl 查詢 Prometheus 用 |
| `KUBE_HOST` | gnn_rl 連線至 Kubernetes proxy |
| `LINKERD_VIZ_API_URL` | 取得 RPS 等指標 |
| `LOCUST_RUN_TIME` | 每次 Locust 執行的持續時間 |
| `NAMESPACE_REDIS`、`NAMESPACE_ONLINEBOUTIQUE` | 各範例對應的命名空間 |

## 專案結構

```
gnn_rl/        # 強化學習策略與訓練程式
gnn_rl/envs/   # Gym 環境實作（原 gnn_rl_env）
loadtest/      # Locust 測試腳本與遠端 agent
macK8S/        # Kubernetes 設定檔（Linkerd、Istio、Prometheus、HPA 等）
```

## 使用說明

以下範例展示如何在本機快速啟用 GNN + RL 自動擴縮器。

1. 安裝相依套件：

   ```bash
   pip install -r requirements.txt
   ```

2. 啟動資料收集器（需先設定 `PROMETHEUS_URL` 與 `LINKERD_VIZ_API_URL`）：

   ```bash
   python -m data_collector.linkerd_prom --edges-url $LINKERD_VIZ_API_URL/api/edges \
       --metrics-url $PROMETHEUS_URL/api/v1/query
   ```

3. 另開終端執行訓練：

   ```bash
   python scripts/train_gnnppo.py --model gat --steps 100000
   ```

4. 訓練完成後可執行基準測試：

   ```bash
   python scripts/benchmark.py --steps 10000 --seeds 3
   ```

更多使用情境與真實叢集設定，請參考 [docs/Operating_Guide.md](docs/Operating_Guide.md)。

## GNN + RL Autoscaler 架構指引

專案已將原 `gnn_rl_env` 環境整合至 `gnn_rl.envs`，可依照下列流程建置 GNN + RL 自動擴縮器。
詳細步驟與需求收錄於 [docs/GNN_RL_Autoscaler.md](docs/GNN_RL_Autoscaler.md)。

若需擴充資料拉取或特徵處理，可額外建立 `data_collector/`、`feature_builder/` 等
子模組，並在 `scripts/` 內撰寫訓練與評測腳本。

### 主要流程

1. **資料收集**：`data_collector` 每 30 秒抓取 edges、容器指標與節點資源。
2. **特徵轉換**：`feature_builder` 將呼叫關係與資源數據轉成 PyG `HeteroData`，同時擷取全域指標供 RL 使用。
3. **GNN 編碼器**：在 `gnn_rl/models` 中實作 `HeteroGAT` 或其他變體，核心程式位於
   [`models/gnn_encoder.py`](gnn_rl/models/gnn_encoder.py)，輸出服務與節點嵌入。
4. **RL 策略**：`gnn_rl/agents` 的 `GNNPPOPolicy` 將 GNN 向量與 scalar 特徵拼接，決定 `svc_id`、`node_type`、`Δreplicas` 與 `Δquota` 等動作。
5. **訓練與測試**：執行 `scripts/train_gnnppo.py` 或 `scripts/benchmark.py` 進行比較，結果輸出於 `results/`。

### Baseline 與評估

| 編碼器   | RL 演算法 | cfg 名稱 |
|---------|-----------|----------|
| None    | PPO       | `mlp_ppo`|
| HeteroGAT | PPO     | `gat_ppo`|
| HeteroGCN | PPO     | `gcn_ppo`|
| HeteroGAT | Discrete-SAC | `gat_sac`|

評估指標包含 `SLO_violate%`、`Resource Slack%`、`Scaling Lag` 與 CAF。`benchmark.py` 會重播四種載入情境並輸出統計表，以檢視不同模型的自動擴縮效果。

如需自行調整 Online Boutique 或 HPA 範例，可參考 `macK8S/HPA/README.md`。

更多真實叢集接入與操作步驟，請見 [docs/Operating_Guide.md](docs/Operating_Guide.md)。

---
本倉庫僅供研究與教學用途，歡迎提出 issue 與貢獻。
