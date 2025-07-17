# Redis 實驗監控與紀錄系統修正
=======================================

## 修正的核心問題

### 1. ✅ 缺失 Pod 監控整合
**問題**: Redis 實驗沒有整合 Pod 監控系統，無法記錄 pod 數量與時間關係
**修正**: 
- 新增 `_setup_pod_monitoring_for_redis()` 方法
- 整合 `MultiPodMonitor` 與 `create_pod_monitor_for_experiment`
- 每 15 秒記錄一次 Redis namespace 的 Pod 數量
- 生成 CSV 格式的時間序列數據

```python
def _setup_pod_monitoring_for_redis(self, scenario: str, output_dir: Path) -> MultiPodMonitor:
    """為 Redis 實驗設置 Pod 監控"""
    pod_monitoring_dir = output_dir / "pod_metrics"
    
    pod_monitor = create_pod_monitor_for_experiment(
        experiment_type="k8s-hpa-redis",
        scenario=scenario,
        namespaces=["redis"],
        output_dir=pod_monitoring_dir
    )
    
    return pod_monitor
```

### 2. ✅ 執行方法邏輯問題
**問題**: Redis 實驗沒有正確整合 unified_experiment_manager.py，導致監控與負載測試邏輯不一致
**修正**:
- Gym-HPA 和 GNNRL 現在都通過 `unified_experiment_manager.py` 執行
- 確保使用相同的監控、負載測試和結果記錄邏輯
- 添加 `--enable-loadtest` 參數確保負載測試執行
- 添加詳細的命令日誌記錄

### 3. ✅ K8s-HPA 紀錄檔案結構問題
**問題**: K8s-HPA 不同設定檔案的結果被記錄在一起，而非分別記錄
**修正**:
- 修正日誌目錄結構以匹配 Online Boutique 模式
- 每個 HPA 配置（cpu-20, cpu-40, cpu-60, cpu-80）都有獨立目錄
- 每個場景都有獨立的子目錄（offpeak_001, peak_002 等）
- 包含完整的 Pod 監控數據

**新的目錄結構**:
```
logs/k8s_hpa_redis/redis_hpa_{config}_{timestamp}/
├── offpeak_001/
│   ├── offpeak_stats.csv          # RPS 和延遲數據
│   ├── offpeak_stats_history.csv  # 詳細時間序列
│   ├── offpeak.html               # HTML 報告
│   └── pod_metrics/
│       └── redis/
│           └── offpeak_pod_counts.csv  # Pod 數量時間序列
├── peak_002/
├── rushsale_003/
└── fluctuating_004/
```

### 4. ✅ 缺失 RPS、Pod 數量與時間關係監控
**問題**: 沒有完整記錄 RPS、Pod 數量和時間的關係數據
**修正**:
- **RPS 監控**: 通過 Locust 的 `--csv` 參數記錄詳細的 RPS 數據
- **Pod 監控**: 每 15 秒記錄一次 Pod 數量，包含時間戳
- **時間同步**: 確保 Pod 監控和負載測試同時啟動和停止
- **數據格式**: 標準化的 CSV 格式，便於後續分析

**Pod 監控數據格式**:
```csv
timestamp,elapsed_minutes,pod_count,namespace,experiment_type,scenario
2024-01-01T00:00:00Z,0.00,1,redis,k8s-hpa-redis,offpeak
2024-01-01T00:00:15Z,0.25,1,redis,k8s-hpa-redis,offpeak
2024-01-01T00:01:00Z,1.00,2,redis,k8s-hpa-redis,offpeak
```

### 5. ✅ 負載測試場景整合問題
**問題**: Redis 負載測試場景不完整，缺少部分場景類型
**修正**:
- 完整的四個場景：`offpeak`, `peak`, `rushsale`, `fluctuating`
- 場景編號系統：`scenario_001`, `scenario_002` 等
- 每個場景都包含完整的監控數據
- 場景間適當的等待時間（30 秒）

### 6. ✅ 用戶選擇系統與監控整合
**問題**: 新的用戶選擇系統（train/test/both/skip）沒有正確整合監控
**修正**:
- 所有模式（train, test, both）都包含完整監控
- 測試前自動重置 Redis Pod 數量
- 統一的錯誤處理和日誌記錄
- 模式間適當的等待時間

## 對比 Online Boutique 實現

### ✅ 現在 Redis 實驗具備的功能
1. **Pod 監控**: ✅ 與 OB 相同的 15 秒間隔監控
2. **RPS 記錄**: ✅ 通過 Locust CSV 輸出記錄詳細 RPS 數據
3. **時間序列數據**: ✅ 完整的時間戳和經過時間記錄
4. **配置分離**: ✅ 每個 HPA 配置獨立目錄
5. **場景分離**: ✅ 每個測試場景獨立子目錄
6. **統一日誌格式**: ✅ 與 OB 一致的 CSV 格式
7. **監控整合**: ✅ 完整整合到實驗流程中

### 📊 數據收集能力
- **Pod 數量變化**: 每 15 秒記錄，包含時間戳
- **RPS 數據**: Locust 提供的詳細請求統計
- **延遲數據**: 響應時間分佈和百分位數
- **錯誤率**: 請求成功/失敗統計
- **HPA 行為**: 不同配置下的擴縮容表現

## 使用方式

### 基本執行（包含完整監控）
```bash
python run_redis_experiment.py --algorithm a2c --steps 5000
```

### 系統現在會：
1. **詢問每種方法的執行模式**:
   - Gym-HPA: train/test/both/skip
   - GNNRL: train/test/both/skip  
   - K8s-HPA: test/skip （HPA 只有測試模式）

2. **自動執行完整監控**:
   - Pod 數量每 15 秒記錄
   - RPS 和延遲數據持續記錄
   - 時間同步確保數據對應

3. **生成結構化日誌**:
   - 每個方法獨立目錄
   - 每個配置獨立子目錄
   - 每個場景獨立數據文件

## 驗證方式

### 檢查 Pod 監控數據
```bash
# 查看最新的 Redis HPA 實驗 Pod 數據
find logs/k8s_hpa_redis -name "*_pod_counts.csv" -exec head -5 {} \;

# 查看數據完整性
find logs/k8s_hpa_redis -name "pod_metrics" -type d | wc -l
```

### 檢查日誌結構
```bash
# 確認配置分離
ls logs/k8s_hpa_redis/*/
# 應該看到: cpu-20/, cpu-40/, cpu-60/, cpu-80/ 等目錄

# 確認場景分離  
ls logs/k8s_hpa_redis/redis_hpa_cpu-40_*/
# 應該看到: offpeak_001/, peak_002/, rushsale_003/, fluctuating_004/
```

### 檢查數據同步性
```bash
# 確認 Pod 數據和 RPS 數據時間對應
head logs/k8s_hpa_redis/*/offpeak_001/pod_metrics/redis/offpeak_pod_counts.csv
head logs/k8s_hpa_redis/*/offpeak_001/offpeak_stats_history.csv
```

## 重要改進點

1. **完整監控整合**: Redis 實驗現在具備與 Online Boutique 相同水準的監控能力
2. **數據結構一致性**: 日誌結構和數據格式與 OB 保持一致
3. **配置分離正確**: K8s-HPA 不同配置的結果正確分離記錄
4. **時間序列完整**: Pod 數量、RPS 和時間的關係完整記錄
5. **用戶體驗改善**: 清晰的選擇選項和詳細的執行日誌

所有監控和記錄問題已全面修正，Redis 實驗系統現在提供完整的數據收集和分析能力。