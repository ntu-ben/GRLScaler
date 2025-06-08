# Linkerd 設定

包含 ServiceMonitor 等監控元件。`metrics.yaml` 併入控制平面與 proxy 指標收集。

## linkerd-viz 安裝腳本

`install_viz_nodeport.sh` 透過 Helm 在 `linkerd-viz` 命名空間安裝 Linkerd-Viz，
設定 dashboard 為 NodePort 並讓 Prometheus Stack 只收集 `onlineboutique` 與 `redis`
命名空間中 sidecar 的指標。
