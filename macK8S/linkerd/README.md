# Linkerd 設定

包含 ServiceMonitor 等監控元件。`metrics.yaml` 併入控制平面與 proxy 指標收集。

若僅需啟用 Linkerd 自帶的 Prometheus，可在 `macK8S/prometheus`
目錄下找到 `linkerd-prometheus.yaml`，並以 `kubectl apply -f` 套用：

```bash
kubectl apply -f ../prometheus/linkerd-prometheus.yaml
```

此檔案同樣是預先渲染的 manifests，並非 Helm values。

## linkerd-viz 安裝腳本

`install_viz_nodeport.sh` 透過 Helm 在 `linkerd-viz` 命名空間安裝 Linkerd-Viz，
設定 dashboard 為 NodePort 並讓 Prometheus Stack 只收集 `onlineboutique` 與 `redis`
命名空間中 sidecar 的指標。
