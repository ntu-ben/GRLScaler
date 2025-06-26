# Prometheus 查詢指南 - GNNRL 診斷

## 📊 基本連接資訊
- **Prometheus URL**: `http://localhost:9090`
- **時間範圍**: 訓練期間 (建議至少 2 小時覆蓋 2000 步)

## 🔍 關鍵指標查詢

### **1. P99 延遲 (P99 Latency)**
```promql
# OnlineBoutique Frontend P99 延遲
histogram_quantile(0.99, 
  sum(rate(istio_request_duration_milliseconds_bucket{
    destination_service_name="frontend",
    destination_service_namespace="onlineboutique"
  }[5m])) by (le)
)

# 或者使用 Locust 延遲指標
histogram_quantile(0.99,
  sum(rate(locust_requests_response_time_bucket{
    method="GET",
    name="/"
  }[5m])) by (le)
)
```

### **2. Replica Count (副本數)**
```promql
# 各服務當前副本數
kube_deployment_status_replicas{
  namespace="onlineboutique"
}

# 期望副本數
kube_deployment_spec_replicas{
  namespace="onlineboutique"  
}

# 可用副本數
kube_deployment_status_replicas_available{
  namespace="onlineboutique"
}
```

### **3. 按服務分組的副本數**
```promql
# 所有 OnlineBoutique 服務的副本數時序
sum by (deployment) (
  kube_deployment_status_replicas{
    namespace="onlineboutique",
    deployment=~"recommendationservice|productcatalogservice|cartservice|adservice|paymentservice|shippingservice|currencyservice|checkoutservice|frontend|emailservice"
  }
)
```

### **4. CPU 使用率**
```promql
# 容器 CPU 使用率
sum by (pod) (
  rate(container_cpu_usage_seconds_total{
    namespace="onlineboutique",
    container!="POD",
    container!=""
  }[5m])
) * 1000  # 轉換為 millicores
```

### **5. 記憶體使用量**
```promql
# 容器記憶體使用量
sum by (pod) (
  container_memory_working_set_bytes{
    namespace="onlineboutique",
    container!="POD",
    container!=""
  }
) / 1024 / 1024  # 轉換為 MiB
```

## 📈 Grafana Dashboard 查詢

### **Panel 1: P99 延遲趨勢**
- **查詢**: 上述 P99 延遲查詢
- **圖表類型**: 時序圖 (Time series)
- **Y軸**: 毫秒 (ms)

### **Panel 2: 副本數熱力圖**
- **查詢**: 按服務分組的副本數
- **圖表類型**: 堆疊時序圖 (Stacked time series)
- **Y軸**: 副本數量

### **Panel 3: 總 Pod 數**
```promql
sum(
  kube_deployment_status_replicas{
    namespace="onlineboutique"
  }
)
```

## 🔗 直接 URL 範例

### **Prometheus Web UI 查詢**
```
http://localhost:9090/graph?g0.expr=histogram_quantile(0.99%2C%20sum(rate(istio_request_duration_milliseconds_bucket%7Bdestination_service_name%3D%22frontend%22%2Cdestination_service_namespace%3D%22onlineboutique%22%7D%5B5m%5D))%20by%20(le))&g0.tab=0&g0.stacked=0&g0.range_input=2h
```

### **Grafana Dashboard**
- 通常位於: `http://localhost:3000`
- 預設登入: admin/admin
- 建議建立自訂 dashboard 包含上述指標

## 📅 時間同步要點

1. **記錄訓練開始時間**: 從 `run_gnnrl_experiment.py` 日誌中找到
2. **記錄訓練結束時間**: 查看 2000 步完成時間
3. **在 Prometheus/Grafana 中設定相同時間範圍**
4. **匯出 CSV 數據** (如果需要離線分析)