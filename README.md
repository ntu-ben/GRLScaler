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
| `PROMETHEUS_URL` | gnn_rl_env 查詢 Prometheus 用 |
| `KUBE_HOST` | gnn_rl_env 連線至 Kubernetes proxy |
| `LINKERD_VIZ_API_URL` | 取得 RPS 等指標 |
| `LOCUST_RUN_TIME` | 每次 Locust 執行的持續時間 |
| `NAMESPACE_REDIS`、`NAMESPACE_ONLINEBOUTIQUE` | 各範例對應的命名空間 |

## 專案結構

```
gnn_rl/        # 強化學習策略與訓練程式
gnn_rl_env/    # 取自 gym-hpa 的環境實作
loadtest/      # Locust 測試腳本與遠端 agent
macK8S/        # Kubernetes 設定檔（Linkerd、Istio、Prometheus、HPA 等）
```

如需自行調整 Online Boutique 或 HPA 範例，可參考 `macK8S/HPA/README.md`。

---
本倉庫僅供研究與教學用途，歡迎提出 issue 與貢獻。
