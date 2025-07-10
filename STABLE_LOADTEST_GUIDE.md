# 穩定Loadtest使用指南

## 概述

新的穩定loadtest系統提供以下功能：
1. **限定最高RPS** - 避免系統過載
2. **失敗時維持測試** - 即使遇到錯誤也繼續測試
3. **穩定的流量模式** - 使用`constant_throughput`確保一致的RPS

## 主要改進

### 1. 穩定流量控制
- 使用`constant_throughput(1)`確保每個用戶每秒1個請求
- 相比原來的`between(1,1)`更加穩定

### 2. 錯誤處理機制
```python
try:
    with self.client.get("/cart", timeout=30, catch_response=True) as response:
        if response.status_code >= 400:
            # 記錄失敗但繼續測試
            response.failure("HTTP error")
        else:
            response.success()
except Exception as e:
    # 捕獲異常但不中斷測試
    logging.warning(f"Request exception: {e}, but continuing test")
```

### 3. RPS限制機制
- 通過環境變數`LOCUST_MAX_RPS`控制最高RPS
- 預設限制：offpeak=50, peak=200, rushsale=400

## 使用方式

### 1. 使用run_autoscaling_experiment.py (推薦)

```bash
# OnlineBoutique實驗，使用A2C算法，限制最高RPS為150
python run_autoscaling_experiment.py onlineboutique \
    --method gym-hpa \
    --algorithm a2c \
    --stable-loadtest \
    --max-rps 150 \
    --steps 5000

# 完整實驗，使用穩定loadtest
python run_autoscaling_experiment.py onlineboutique \
    --algorithm a2c \
    --stable-loadtest \
    --max-rps 200 \
    --standardized
```

### 2. 直接使用StableLoadTestManager

```bash
# 單一場景測試
python loadtest/stable_loadtest_manager.py peak \
    --host http://k8s.orb.local \
    --max-rps 150 \
    --run-time 15m

# 自定義配置
python loadtest/stable_loadtest_manager.py rushsale \
    --host http://k8s.orb.local \
    --max-rps 300 \
    --timeout 60 \
    --run-time 20m
```

### 3. 直接使用Locust腳本

```bash
# 設置環境變數控制行為
export LOCUST_MAX_RPS=200
export LOCUST_RUN_TIME=15m
export LOCUST_TIMEOUT=30

# 執行穩定版本
locust -f loadtest/onlineboutique/locust_stable_peak.py \
    --host http://k8s.orb.local \
    --headless \
    --run-time 15m \
    --csv ./logs/stable_peak
```

## 可用的穩定腳本

1. `locust_stable_offpeak.py` - 穩定低峰測試 (預設50 RPS上限)
2. `locust_stable_peak.py` - 穩定峰值測試 (預設200 RPS上限)  
3. `locust_stable_rushsale.py` - 穩定搶購測試 (預設400 RPS上限)

## 環境變數配置

| 變數名 | 說明 | 預設值 |
|--------|------|--------|
| `LOCUST_MAX_RPS` | 最高RPS限制 | 依場景而定 |
| `LOCUST_RUN_TIME` | 運行時間 | 15m |
| `LOCUST_TIMEOUT` | 請求超時時間 | 30秒 |

## 與原系統的兼容性

- 如果穩定版本腳本不存在，會自動回退到原版本
- 輸出格式與原系統完全兼容
- 不影響現有的數據分析和視覺化流程

## 優勢

1. **穩定性** - 即使系統過載也能維持測試
2. **可控性** - 可以精確控制最高負載
3. **一致性** - 提供更一致的基準測試
4. **容錯性** - 網絡問題或服務暫時不可用時仍能繼續

## 建議配置

### OnlineBoutique環境
- offpeak: 50 RPS上限
- peak: 150-200 RPS上限  
- rushsale: 300-400 RPS上限

### Redis環境  
- offpeak: 75 RPS上限
- peak: 400-500 RPS上限

這些配置能在測試系統性能的同時避免完全壓垮服務。