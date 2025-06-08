# Prometheus 設定

`prometheus.yaml` 提供額外的 scrape configs 以及 Helm 安裝用 values。

若希望啟用 Linkerd 提供的獨立 Prometheus，
可直接套用 `linkerd-prometheus.yaml`：

```bash
kubectl apply -f linkerd-prometheus.yaml
```

此檔案已從 Helm Chart 預先渲染，**並不是** Helm 的 values 檔。
