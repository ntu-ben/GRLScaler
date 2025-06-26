# GRLScaler 實驗結果統一結構設計

## 🎯 新的統一路徑結構

```
experiments/
├── YYYYMMDD_HHMMSS_{experiment_type}_{algorithm}_{model}/  # 實驗主目錄
│   ├── experiment_info.json                               # 實驗配置和元信息
│   ├── action_history.csv                                 # RL 動作歷史
│   ├── training_log.txt                                  # 訓練詳細日誌
│   ├── loadtest_scenarios/                               # 壓測場景結果
│   │   ├── scenario_001_peak/
│   │   │   ├── locust_stats.csv
│   │   │   ├── locust_history.csv
│   │   │   └── scenario_info.json
│   │   ├── scenario_002_offpeak/
│   │   └── scenario_N_fluctuating/
│   ├── loadtest_summary.csv                              # 壓測指標摘要
│   ├── experiment_summary.json                           # 完整實驗摘要
│   ├── performance_charts/                               # 性能圖表
│   │   ├── rps_timeline.png
│   │   ├── latency_distribution.png
│   │   └── replica_changes.png
│   └── models/                                           # 訓練好的模型
│       ├── final_model.zip
│       └── checkpoints/
├── comparison_reports/                                   # 橫向比較報告
│   ├── all_experiments_summary.csv                      # 所有實驗對比
│   ├── performance_comparison.html                      # 可視化比較
│   └── latest_comparison_YYYYMMDD.json                  # 最新比較數據
└── archive/                                             # 歷史實驗存檔
    └── YYYY/MM/
```

## 📊 統一壓測報告格式

### **loadtest_summary.csv** (主要壓測摘要)
```csv
experiment_id,scenario_name,scenario_order,start_time,end_time,duration_sec,
total_requests,successful_requests,failed_requests,failure_rate,
avg_rps,max_rps,min_rps,
p50_latency,p95_latency,p99_latency,
total_replicas_start,total_replicas_end,replica_changes,
avg_cpu_usage,max_cpu_usage,avg_memory_usage,max_memory_usage,
kiali_rps,prometheus_p99_latency
```

### **experiment_summary.json** (完整實驗摘要)
```json
{
  "experiment_info": {
    "id": "20250626_143022_gnnrl_a2c_gat",
    "type": "gnnrl",
    "algorithm": "a2c", 
    "model": "gat",
    "goal": "latency",
    "use_case": "online_boutique",
    "start_time": "2025-06-26T14:30:22Z",
    "end_time": "2025-06-26T16:45:33Z",
    "duration_hours": 2.25,
    "total_training_steps": 2000,
    "seed": 42
  },
  "training_metrics": {
    "final_reward": 1250.5,
    "avg_episode_reward": 890.2,
    "convergence_step": 1456,
    "total_episodes": 80,
    "model_checkpoints": ["checkpoint_1000.zip", "final_model.zip"]
  },
  "loadtest_aggregation": {
    "total_scenarios_executed": 12,
    "scenario_distribution": {
      "peak": 4,
      "offpeak": 3, 
      "fluctuating": 3,
      "rushsale": 2
    },
    "overall_performance": {
      "avg_rps": 245.6,
      "overall_p95_latency": 156.7,
      "overall_p99_latency": 245.3,
      "total_requests": 2850000,
      "overall_failure_rate": 0.023
    }
  },
  "scaling_behavior": {
    "total_scaling_actions": 156,
    "scaling_distribution": {
      "scale_up": 89,
      "scale_down": 45,
      "no_action": 22
    },
    "most_scaled_service": "frontend",
    "avg_replicas_per_service": 2.8,
    "max_replicas_reached": 6
  },
  "comparison_baseline": {
    "hpa_comparison": {
      "improvement_p95": "+12.5%",
      "improvement_p99": "+8.3%", 
      "rps_difference": "+45.2 RPS"
    }
  }
}
```

## 🔄 橫向比較格式

### **all_experiments_summary.csv** (橫向比較主表)
```csv
experiment_id,timestamp,experiment_type,algorithm,model,goal,total_steps,duration_hours,
avg_p95_latency,avg_p99_latency,avg_rps,total_requests,overall_failure_rate,
total_scaling_actions,final_reward,convergence_step,
improvement_vs_hpa_p95,improvement_vs_hpa_p99,improvement_vs_hpa_rps,
cost_efficiency_score,stability_score,notes
```

## 🎨 可視化報告

### **performance_comparison.html** 結構:
1. **實驗概覽表格** - 所有實驗的關鍵指標對比
2. **性能趨勢圖** - P95/P99 延遲、RPS 趨勢
3. **擴縮行為分析** - 各算法的擴縮決策模式
4. **成本效益分析** - 資源使用 vs 性能提升
5. **穩定性評估** - 波動性和收斂性分析

## 🏷️ 實驗標識規範

### **實驗 ID 格式:**
`YYYYMMDD_HHMMSS_{experiment_type}_{algorithm}_{model}_{goal}_{steps}`

**範例:**
- `20250626_143022_gnnrl_a2c_gat_latency_2000`
- `20250626_150000_gym_hpa_ppo_mlp_cost_5000`
- `20250626_160000_hpa_baseline_cpu80_mem80_latency_NA`

### **場景標識格式:**
`scenario_{3位序號}_{場景名稱}_{開始時間戳}`

**範例:**
- `scenario_001_peak_143055`
- `scenario_002_offpeak_144125`
- `scenario_003_fluctuating_145205`