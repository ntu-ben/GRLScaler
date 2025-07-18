# Redis Locust Agent 設置指南

## 問題描述
Locust Agent 在 M1 MacBook (10.0.0.2) 上運行，但需要連接到 Kubernetes 集群中的 Redis 服務。

## 解決方案

### 1. 在 M1 MacBook 上設置 Redis Port-Forward

在 M1 MacBook 上運行以下命令：

```bash
# 設置 Redis port-forward（後台運行）
kubectl port-forward -n redis svc/redis-master 6379:6379 &

# 驗證 port-forward 是否成功
telnet localhost 6379
```

### 2. 設置環境變數（可選）

如果需要使用不同的 Redis 主機，可以設置環境變數：

```bash
# 在 M1 MacBook 上設置
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### 3. 驗證連接

```bash
# 測試 Redis 連接
redis-cli -h localhost -p 6379 ping
```

應該返回 `PONG`。

## 自動化設置

系統會自動：
1. 檢測 Redis 環境
2. 使用 `localhost:6379` 作為預設連接
3. 設置適當的環境變數給 Locust 腳本

## 故障排除

### 如果連接失敗：
1. 確保 port-forward 正在運行：`ps aux | grep port-forward`
2. 檢查 Redis 服務狀態：`kubectl get pods -n redis`
3. 測試本地連接：`telnet localhost 6379`

### 如果需要使用不同的 Redis 主機：
設置 `REDIS_HOST` 環境變數，例如：
```bash
export REDIS_HOST=10.0.0.1
```

## 修改內容

1. **locust_agent.py**：
   - 修改 `_get_redis_target_host()` 優先使用 `localhost`
   - 自動設置 `REDIS_HOST` 和 `REDIS_PORT` 環境變數

2. **unified_experiment_manager.py**：
   - 自動檢測 Redis 環境
   - 調整 `target_host` 為 `redis://localhost:6379`

3. **Redis Locust 腳本**：
   - 使用環境變數 `REDIS_HOST` 和 `REDIS_PORT`
   - 預設連接到 `localhost:6379`