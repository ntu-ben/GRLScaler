# 實驗補充指令 (Experiment Supplement Commands)
==================================================

## 🎯 概述

提供針對特定場景和方法的實驗補充指令，特別適用於重新執行表現不佳的場景或驗證特定條件下的性能。

## 📋 Redis 實驗補充指令

### 1. 重新執行 Peak 場景測試
基於分析發現 GNNRL 和 Gym-HPA 在 peak 場景表現較差，以下指令可重新執行：

```bash
# 只測試 GNNRL 的 Peak 場景 (A2C 算法)
python run_redis_experiment.py --algorithm a2c --steps 5000

# 執行時選擇：
# GNNRL: 2 (test)
# GNNRL 場景: peak
# Gym-HPA: 4 (skip)  
# K8s-HPA: 4 (skip)
```

```bash
# 只測試 Gym-HPA 的 Peak 場景
python run_redis_experiment.py --algorithm a2c --steps 5000

# 執行時選擇：
# Gym-HPA: 2 (test)
# Gym-HPA 場景: peak
# GNNRL: 4 (skip)
# K8s-HPA: 4 (skip)
```

### 2. 對比高負載場景 (Peak + Rush Sale)
```bash
python run_redis_experiment.py --algorithm a2c --steps 5000

# 執行時選擇：
# Gym-HPA: 2 (test)
# Gym-HPA 場景: peak,rushsale
# GNNRL: 2 (test)
# GNNRL 場景: peak,rushsale
# K8s-HPA: 4 (skip)
```

### 3. 完整重測所有方法的特定場景
```bash
python run_redis_experiment.py --algorithm a2c --steps 5000

# 執行時選擇：
# Gym-HPA: 2 (test)
# Gym-HPA 場景: peak
# GNNRL: 2 (test)  
# GNNRL 場景: peak
# K8s-HPA: 2 (test)
# K8s-HPA 場景: peak
```

### 4. 快速驗證單一場景
```bash
# 只測試 off-peak 場景驗證基準性能
python run_redis_experiment.py --algorithm a2c --steps 5000
# 選擇場景: offpeak

# 只測試 fluctuating 場景驗證動態負載處理
python run_redis_experiment.py --algorithm a2c --steps 5000  
# 選擇場景: fluctuating
```

## 📋 Online Boutique 實驗補充指令

### 1. 重新執行 Peak 場景測試
```bash
# 只測試 GNNRL 的 Peak 場景
python run_onlineboutique_experiment.py --steps 5000 --goal latency --model gat

# 執行時選擇：
# Gym-HPA: 4 (skip)
# GNNRL: 2 (test)
# GNNRL 場景: peak
# K8s-HPA: 4 (skip)
```

```bash
# 只測試 Gym-HPA 的 Peak 場景  
python run_onlineboutique_experiment.py --steps 5000 --goal latency --model gat

# 執行時選擇：
# Gym-HPA: 2 (test)
# Gym-HPA 場景: peak
# GNNRL: 4 (skip)
# K8s-HPA: 4 (skip)
```

### 2. 對比高負載場景測試
```bash
python run_onlineboutique_experiment.py --steps 5000 --goal latency --model gat

# 執行時選擇：
# Gym-HPA: 2 (test)
# Gym-HPA 場景: peak,rushsale
# GNNRL: 2 (test)
# GNNRL 場景: peak,rushsale
# K8s-HPA: 4 (skip)
```

### 3. 驗證不同 RPS 設定的 Peak 場景
```bash
# 測試標準 Peak 場景 (RPS 200-400)
python run_onlineboutique_experiment.py --steps 5000 --goal latency --model gat
# 選擇場景: peak

# 測試 Rush Sale 場景 (RPS 300-800) 
python run_onlineboutique_experiment.py --steps 5000 --goal latency --model gat
# 選擇場景: rushsale
```

### 4. 測試不同算法組合
```bash
# 測試 A2C 算法在 Peak 場景的表現
python run_onlineboutique_experiment.py --steps 5000 --goal latency --model gat --algorithm a2c
# 選擇場景: peak

# 測試 PPO 算法在 Peak 場景的表現  
python run_onlineboutique_experiment.py --steps 5000 --goal latency --model gat --algorithm ppo
# 選擇場景: peak
```

## 🔧 透過統一實驗管理器執行

### 直接指定場景參數
```bash
# Redis Peak 場景補充實驗
python unified_experiment_manager.py --experiment gnnrl --scenarios peak --use_case redis --algorithm a2c

# Online Boutique Peak 場景補充實驗
python unified_experiment_manager.py --experiment gnnrl --scenarios peak --use_case online_boutique --goal latency
```

### 多場景批量補充
```bash
# 補充高負載場景測試
python unified_experiment_manager.py --experiment gym_hpa --scenarios peak,rushsale --use_case redis

# 補充所有問題場景
python unified_experiment_manager.py --experiment gnnrl --scenarios peak,fluctuating --use_case online_boutique
```

## 📊 補充實驗結果位置

### Redis 補充實驗結果
```
logs/
├── gym_hpa_redis/              # Gym-HPA Redis 補充結果
│   └── redis_hpa_YYYYMMDD_HHMMSS/
│       └── peak_001/           # 只有 Peak 場景結果
├── gnnrl/                      # GNNRL 補充結果  
│   └── redis_test_YYYYMMDD_HHMMSS/
│       └── peak_001/
└── k8s_hpa_redis/              # K8s-HPA 補充結果
    └── redis_hpa_cpu-XX_YYYYMMDD_HHMMSS/
        └── peak_001/
```

### Online Boutique 補充實驗結果
```
logs/
├── gym_hpa/                    # Gym-HPA OB 補充結果
│   └── ob_test_YYYYMMDD_HHMMSS/
│       └── peak_001/
├── gnnrl/                      # GNNRL OB 補充結果
│   └── gnnrl_test_YYYYMMDD_HHMMSS/  
│       └── peak_001/
└── k8s_hpa/                    # K8s-HPA OB 補充結果
    └── ob_hpa_YYYYMMDD_HHMMSS/
        └── peak_001/
```

## 💡 補充實驗最佳實踐

### 1. 針對性問題解決
```bash
# 問題：GNNRL 在 Redis Peak 場景表現差
# 解決：專門重測該場景
python run_redis_experiment.py --algorithm a2c
# 選擇: GNNRL -> test -> peak
```

### 2. 對比分析
```bash
# 對比同一場景不同方法的表現
python run_redis_experiment.py --algorithm a2c  
# 選擇: 所有方法 -> test -> peak
```

### 3. 驗證修改效果
```bash
# 在調整參數後驗證改善效果
python run_onlineboutique_experiment.py --steps 10000 --goal latency
# 選擇特定場景驗證
```

### 4. 快速驗證
```bash
# 縮短實驗時間進行快速驗證
python run_redis_experiment.py --algorithm a2c --steps 2000
# 選擇問題場景進行快速測試
```

## 🎯 特殊情況處理

### 當需要重新訓練模型時
```bash
# 重新訓練並測試
python run_redis_experiment.py --algorithm a2c --steps 5000
# 選擇: method -> both -> specific_scenarios
```

### 當需要調整測試參數時
```bash
# 使用自定義 RPS 限制
python run_onlineboutique_experiment.py --max_rps 300 --steps 5000
# 針對 Peak 場景進行受控測試
```

### 批量補充實驗
```bash
# 創建批量執行腳本
for scenario in peak rushsale fluctuating; do
    python unified_experiment_manager.py --experiment gnnrl --scenarios $scenario --use_case redis --algorithm a2c
    sleep 300  # 5分鐘間隔
done
```

## ✅ 驗證補充實驗成功

### 檢查日誌文件
```bash
# 檢查最新的實驗日誌
ls -la logs/runtime/unified_experiment_*.log | tail -1

# 檢查特定場景的監控數據
ls -la logs/*/pod_metrics/*/peak_pod_counts.csv
```

### 驗證結果完整性
```bash
# 確認 Pod 監控數據
find logs -name "*peak_pod_counts.csv" -exec wc -l {} \;

# 確認負載測試結果
find logs -name "*peak_stats.csv" -exec head -3 {} \;
```

## 🚀 總結

這些補充指令允許您：
1. **針對性重測** - 只重新執行有問題的場景
2. **快速驗證** - 縮短實驗時間驗證修改效果  
3. **對比分析** - 同場景不同方法對比
4. **批量處理** - 自動化執行多個補充實驗
5. **資源節約** - 避免重複執行正常場景

現在您可以精確地補充任何需要的實驗場景，特別是解決 Peak 場景的性能問題！