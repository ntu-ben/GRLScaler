# Redis 實驗修正摘要

## 修正的問題

### 1. ✅ GNNRL Redis 環境觀察空間錯誤
**問題**: `Expected: (4, 6), actual shape: (4, 7)`
**修正**: 更新 `gnnrl/core/envs/redis.py:123` 中的 edge_df 觀察空間從 6 維改為 7 維
```python
'edge_df': spaces.Box(-np.inf, np.inf, shape=(num_nodes * num_nodes, 7), dtype=np.float32)
```

### 2. ✅ 缺失依賴套件
**問題**: 
- `No module named locust`
- `No module named redis` 
- `No module named sb3_contrib`

**修正**: 安裝所有缺失的套件
```bash
pip install locust redis sb3_contrib
```

### 3. ✅ K8s-HPA 沒有真正執行負載測試
**問題**: K8s-HPA 只是跳過了 locust 測試
**修正**: 
- 修正腳本路徑查找邏輯
- 添加錯誤處理和日誌輸出
- 使用正確的 Python 環境執行 locust

### 4. ✅ 添加用戶詢問機制
**功能**: 新增 `ask_user_confirmation()` 方法
- 支援 y/n/skip 選項
- 允許用戶選擇要執行的實驗方法
- 中文和英文輸入都支援

### 5. ✅ 自動測試功能
**功能**: 訓練完成後自動進行測試
- Gym-HPA: 訓練 → 測試
- GNNRL: 訓練 → 測試  
- K8s-HPA: 直接測試

### 6. ✅ Redis Pod 重置功能
**功能**: 每次測試前重置 Redis Pod 數量為 1
```python
def reset_redis_pods(self):
    deployments = ['redis-master', 'redis-slave']
    for deployment in deployments:
        kubectl scale deployment {deployment} --replicas=1 -n redis
```

### 7. ✅ 統一實驗邏輯
**改進**: 
- 三種方法都支援 A2C 算法
- 統一的錯誤處理
- 一致的日誌格式
- 統一的結果報告

## 新的使用方式

### 基本執行
```bash
python run_autoscaling_experiment.py redis --algorithm a2c --steps 5000
```

### 完整參數
```bash
python run_autoscaling_experiment.py redis \
  --algorithm a2c \
  --steps 5000 \
  --goal latency \
  --model gat \
  --stable-loadtest \
  --max-rps 300
```

## 修正後的執行流程

1. **環境檢查**: 驗證 Redis 集群狀態
2. **用戶詢問**: 選擇要執行的方法 (Gym-HPA/GNNRL/K8s-HPA)
3. **逐一執行**:
   - 重置 Redis Pod 數量為 1
   - 執行訓練 (如適用)
   - 執行測試
4. **結果報告**: 顯示每個方法的成功/失敗狀態

## 驗證

所有修正都已通過測試:
```bash
python test_redis_fixes.py
```

測試結果:
- ✅ Redis 環境觀察空間: 通過
- ✅ 依賴套件: 通過  
- ✅ Redis 實驗執行器: 通過

## 注意事項

1. 確保 Redis 集群已部署在 `redis` namespace
2. 確保可以訪問 K8s 集群 (kubectl 正常工作)
3. 每個實驗方法都會在測試前重置 Pod 數量
4. 使用 A2C 算法時，所有三種方法都會正確使用 A2C
5. 實驗過程中會有互動式詢問，無人值守運行請用腳本自動回答