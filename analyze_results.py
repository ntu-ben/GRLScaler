#!/usr/bin/env python3
"""
實驗結果分析工具
================

分析三種方法的負載測試結果：
- GNNRL (圖神經網路強化學習)
- Gym-HPA (基礎強化學習)  
- K8s-HPA (原生 HPA 基準)
"""

import pandas as pd
import json
from pathlib import Path
import sys

def analyze_locust_results(stats_file):
    """分析 Locust 測試統計檔案"""
    if not Path(stats_file).exists():
        return None
    
    df = pd.read_csv(stats_file)
    aggregated_row = df[df['Type'] == ''].iloc[0] if any(df['Type'] == '') else df.iloc[-1]
    
    return {
        'total_requests': int(aggregated_row['Request Count']),
        'failure_rate': float(aggregated_row['Failure Count']) / float(aggregated_row['Request Count']) * 100,
        'avg_rps': float(aggregated_row['Requests/s']),
        'avg_response_time': float(aggregated_row['Average Response Time']),
        'median_response_time': float(aggregated_row['Median Response Time']),
        'p95_response_time': float(aggregated_row['95%']),
        'p99_response_time': float(aggregated_row['99%'])
    }

def analyze_kiali_graph(kiali_file):
    """分析 Kiali 服務圖檔案"""
    if not Path(kiali_file).exists():
        return None
        
    with open(kiali_file) as f:
        data = json.load(f)
    
    nodes = data.get('elements', {}).get('nodes', [])
    edges = data.get('elements', {}).get('edges', [])
    
    # 統計服務
    services = []
    total_traffic = 0
    
    for node in nodes:
        workload = node['data'].get('workload', 'unknown')
        services.append(workload)
        
        # 統計流量
        traffic = node['data'].get('traffic', [])
        for t in traffic:
            rates = t.get('rates', {})
            for rate_type, rate_value in rates.items():
                if rate_value and rate_value != '0':
                    try:
                        total_traffic += float(rate_value)
                    except:
                        pass
    
    return {
        'service_count': len(services),
        'edge_count': len(edges),
        'services': services,
        'total_traffic_rate': total_traffic
    }

def find_experiment_results(experiment_type):
    """找到實驗結果目錄，分別處理訓練和測試數據"""
    logs_dir = Path(__file__).parent / 'logs' / experiment_type
    if not logs_dir.exists():
        return [], "unknown"
    
    # 優先尋找測試目錄
    test_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and 'test' in d.name]
    if test_dirs:
        latest_dir = max(test_dirs, key=lambda d: d.stat().st_mtime)
        data_type = "test"
    else:
        # 如果沒有test目錄，使用train目錄
        train_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and ('train' in d.name or 'cpu' in d.name)]
        if train_dirs:
            latest_dir = max(train_dirs, key=lambda d: d.stat().st_mtime)
            data_type = "train" if 'train' in latest_dir.name else "test"
        else:
            return [], "unknown"
    
    # 對於k8s-hpa，需要進一步查找cpu配置目錄
    if experiment_type == 'k8s-hpa':
        cpu_dirs = [d for d in latest_dir.iterdir() if d.is_dir() and 'cpu-' in d.name]
        if cpu_dirs:
            all_scenarios = []
            for cpu_dir in cpu_dirs:
                all_scenarios.extend(list(cpu_dir.glob('*/')))
            return all_scenarios, data_type
    
    return list(latest_dir.glob('*/')), data_type

def generate_comparison_report():
    """生成三種方法的比較報告"""
    print("🔍 實驗結果分析報告")
    print("=" * 60)
    
    experiments = {
        'GNNRL': 'gnnrl',
        'Gym-HPA': 'gym-hpa', 
        'K8s-HPA': 'k8s-hpa'
    }
    
    all_results = {}
    
    for exp_name, exp_type in experiments.items():
        scenario_dirs, data_type = find_experiment_results(exp_type)
        
        print(f"\n📊 {exp_name} 結果分析")
        if data_type == "train":
            print(f"⚠️  使用訓練階段數據 (未找到測試數據)")
        elif data_type == "test":
            print(f"✅ 使用測試階段數據")
        print("-" * 40)
        
        if not scenario_dirs:
            print(f"❌ 未找到 {exp_name} 的結果數據")
            continue
        
        exp_results = []
        
        for scenario_dir in scenario_dirs:
            stats_file = scenario_dir / f"{scenario_dir.name.split('_')[0]}_stats.csv"
            
            if stats_file.exists():
                result = analyze_locust_results(stats_file)
                if result:
                    result['scenario'] = scenario_dir.name
                    exp_results.append(result)
                    
                    print(f"  📈 {scenario_dir.name}:")
                    print(f"    請求數: {result['total_requests']:,}")
                    print(f"    失敗率: {result['failure_rate']:.2f}%")
                    print(f"    平均 RPS: {result['avg_rps']:.2f}")
                    print(f"    平均響應時間: {result['avg_response_time']:.2f} ms")
                    print(f"    95%ile: {result['p95_response_time']:.0f} ms")
        
        all_results[exp_name] = exp_results
        
        if exp_results:
            # 計算總體統計
            total_requests = sum(r['total_requests'] for r in exp_results)
            avg_response_time = sum(r['avg_response_time'] * r['total_requests'] for r in exp_results) / total_requests if total_requests > 0 else 0
            avg_p95 = sum(r['p95_response_time'] for r in exp_results) / len(exp_results)
            
            data_note = " (訓練數據)" if data_type == "train" else " (測試數據)"
            print(f"  📋 {exp_name} 總計{data_note}:")
            print(f"    場景數: {len(exp_results)}")
            print(f"    總請求數: {total_requests:,}")
            print(f"    加權平均響應時間: {avg_response_time:.2f} ms")
            print(f"    平均 95%ile: {avg_p95:.0f} ms")
    
    # 生成比較表格
    if len(all_results) > 1:
        print(f"\n🏆 方法比較摘要")
        print("-" * 50)
        
        comparison_data = []
        for exp_name, results in all_results.items():
            if results:
                total_requests = sum(r['total_requests'] for r in results)
                avg_response_time = sum(r['avg_response_time'] * r['total_requests'] for r in results) / total_requests if total_requests > 0 else 0
                avg_p95 = sum(r['p95_response_time'] for r in results) / len(results)
                avg_rps = sum(r['avg_rps'] for r in results) / len(results)
                
                comparison_data.append({
                    'Method': exp_name,
                    'Scenarios': len(results),
                    'Total Requests': total_requests,
                    'Avg Response Time (ms)': round(avg_response_time, 2),
                    'Avg P95 (ms)': round(avg_p95, 0),
                    'Avg RPS': round(avg_rps, 2)
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            print(df_comparison.to_string(index=False))
            
            # 保存比較結果
            comparison_file = Path(__file__).parent / 'logs' / 'experiment_comparison.csv'
            df_comparison.to_csv(comparison_file, index=False)
            print(f"\n💾 比較結果已保存到: {comparison_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--experiment":
        if len(sys.argv) > 2:
            exp_type = sys.argv[2]
            scenario_dirs = find_experiment_results(exp_type)
            print(f"Found {len(scenario_dirs)} scenarios for {exp_type}")
        else:
            print("Usage: python analyze_results.py --experiment <type>")
    else:
        generate_comparison_report()