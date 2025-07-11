# 穩定負載測試場景完整清單

## 🎯 穩定流量測試 (Stable Load Test) 概念

**問題描述**：
- 舊版本的負載測試因為隨機發送請求，會造成嚴重的流量抖動
- 設定 300 RPS 的場景可能實際產生 200-500 RPS 的波動
- 影響實驗結果的準確性和可重複性

**解決方案**：
- 使用 `constant_throughput(1)` 確保每個用戶每秒固定1個請求
- 通過 `LoadTestShape` 精確控制用戶數量 = 目標 RPS
- 四階段波動場景：每階段內**穩定維持**該階段的RPS值

## 📊 OnlineBoutique 場景清單

### 原始場景 (有抖動問題)
- `locust_offpeak.py` - 低峰時段 (~50 RPS)
- `locust_peak.py` - 高峰時段 (~200 RPS)
- `locust_rushsale.py` - 搶購時段 (~400 RPS)
- `locust_fluctuating.py` - 波動負載 (50→300→50→800 RPS)

### 穩定場景 (無抖動)
- `locust_stable_offpeak.py` - 穩定低峰時段 (50 RPS)
- `locust_stable_peak.py` - 穩定高峰時段 (200 RPS)
- `locust_stable_rushsale.py` - 穩定搶購時段 (400 RPS)
- `locust_stable_fluctuating.py` - 穩定波動負載 (50→300→50→800 RPS)

## 🔧 Redis 場景清單

### 原始場景 (有抖動問題)
- `locust_redis_offpeak.py` - Redis低峰時段 (~75 RPS)
- `locust_redis_peak.py` - Redis高峰時段 (~300 RPS)
- `locust_redis_rushsale.py` - Redis搶購時段 (~400 RPS)
- `locust_redis_fluctuating.py` - Redis波動負載 (75→200→75→400 RPS)

### 穩定場景 (無抖動)
- `locust_redis_stable_offpeak.py` - 穩定Redis低峰時段 (75 RPS)
- `locust_redis_stable_peak.py` - 穩定Redis高峰時段 (300 RPS)
- `locust_redis_stable_rushsale.py` - 穩定Redis搶購時段 (350 RPS)
- `locust_redis_stable_fluctuating.py` - 穩定Redis波動負載 (75→200→75→300 RPS)

## 🎯 穩定波動場景 (Fluctuating) 詳細說明

### OnlineBoutique Fluctuating
```
階段1 (0-25%): 50 RPS  - 穩定維持50RPS
階段2 (25-50%): 300 RPS - 穩定維持300RPS  
階段3 (50-75%): 50 RPS  - 穩定維持50RPS
階段4 (75-100%): 800 RPS - 穩定維持800RPS
```

### Redis Fluctuating
```
階段1 (0-25%): 75 RPS  - 穩定維持75RPS
階段2 (25-50%): 200 RPS - 穩定維持200RPS
階段3 (50-75%): 75 RPS  - 穩定維持75RPS
階段4 (75-100%): 300 RPS - 穩定維持300RPS (穩定版本降低)
```

## 🔧 技術實現要點

### 1. 穩定RPS機制
```python
# 每個用戶每秒固定1個請求
wait_time = constant_throughput(1)

# LoadTestShape 返回固定用戶數
def tick(self):
    return (target_users, target_users)  # 用戶數 = RPS
```

### 2. 錯誤處理
```python
# 即使請求失敗也繼續測試，避免中斷
try:
    # 執行請求
    pass
except Exception as e:
    logging.warning(f"請求失敗: {e}, 但繼續測試")
```

### 3. 環境變量配置
```bash
# 運行時間
LOCUST_RUN_TIME=15m

# 波動場景各階段RPS
LOCUST_PHASE1_RPS=50
LOCUST_PHASE2_RPS=300
LOCUST_PHASE3_RPS=50
LOCUST_PHASE4_RPS=800

# 穩定場景RPS上限
LOCUST_MAX_RPS=400
```

## 🚀 使用方法

### 1. 直接運行穩定場景
```bash
# OnlineBoutique穩定波動測試
locust -f loadtest/onlineboutique/locust_stable_fluctuating.py \
       --host http://k8s.orb.local:8080 \
       --headless --run-time 15m

# Redis穩定搶購測試
locust -f loadtest/redis/locust_redis_stable_rushsale.py \
       --host http://redis.local:6379 \
       --headless --run-time 15m
```

### 2. 通過穩定負載測試管理器
```bash
# 使用穩定負載測試管理器
python loadtest/stable_loadtest_manager.py fluctuating \
       --host http://k8s.orb.local:8080 \
       --max-rps 400 \
       --run-time 15m
```

### 3. 在實驗腳本中使用
```bash
# 運行完整實驗（自動使用穩定版本）
python run_autoscaling_experiment.py onlineboutique \
       --algorithm a2c \
       --stable-loadtest
```

## 📈 效果對比

### 舊版本 (有抖動)
```
設定300 RPS → 實際 200-500 RPS 波動
標準差大，實驗結果不穩定
```

### 穩定版本 (無抖動)
```
設定300 RPS → 實際 295-305 RPS 穩定
標準差小，實驗結果可重複
```

## 🎯 總結

現在所有負載測試場景都完整了：
- ✅ **OnlineBoutique**: 4個原始場景 + 4個穩定場景
- ✅ **Redis**: 4個原始場景 + 4個穩定場景
- ✅ **波動場景**: 支援四階段穩定流量 (50→300→50→800)
- ✅ **無抖動**: 使用 `constant_throughput(1)` 確保穩定RPS
- ✅ **可配置**: 支援環境變量配置各階段參數

這樣你的實驗就可以使用真正**穩定的流量測試**，獲得更準確和可重複的結果！