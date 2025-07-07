#!/usr/bin/env python3
"""
標準化實驗結果分析腳本 (簡化版)
==========================================

針對使用相同8個場景的三方法比較實驗，提供基礎性能分析。
不依賴 pandas/numpy 等外部庫。
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class SimpleStandardizedAnalyzer:
    """簡化版標準化結果分析器"""
    
    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path(__file__).parent
        self.logs_dir = self.repo_root / "logs"
        self.scenario_config = self._load_scenario_config()
        
    def _load_scenario_config(self) -> Dict:
        """載入標準化場景配置"""
        config_file = self.repo_root / "standardized_test_scenarios.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def find_standardized_experiments(self) -> Dict[str, List[Path]]:
        """查找標準化實驗結果目錄"""
        experiments = {
            'gym_hpa': [],
            'gnnrl': [],
            'k8s_hpa': []
        }
        
        # 查找標準化實驗目錄 - 更新為實際的目錄命名模式
        patterns = {
            'gym_hpa': "*_test_seed42_*",
            'gnnrl': "*_test_seed42_*", 
            'k8s_hpa': "*_cpu_seed42_*"
        }
        
        for method in experiments.keys():
            method_dir = self.logs_dir / method.replace('_', '-')
            if method_dir.exists():
                dirs = list(method_dir.glob(patterns[method]))
                experiments[method] = dirs
                print(f"找到 {method} 標準化實驗: {len(dirs)} 個")
                for dir_path in dirs:
                    print(f"  - {dir_path.name}")
            
        return experiments
    
    def analyze_single_scenario(self, scenario_dir: Path) -> Dict:
        """分析單個場景的結果"""
        # 查找 CSV 統計文件
        stats_files = list(scenario_dir.glob("*_stats.csv"))
        if not stats_files:
            return {}
            
        stats_file = stats_files[0]
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if not rows:
                return {}
                
            # 找到 Aggregated 行或使用最後一行
            agg_row = None
            for row in rows:
                if row.get('Name') == 'Aggregated':
                    agg_row = row
                    break
            
            if agg_row is None:
                agg_row = rows[-1]
            
            return {
                'requests': int(agg_row.get('Request Count', 0)),
                'failures': int(agg_row.get('Failure Count', 0)),
                'failure_rate': float(agg_row.get('Failure Count', 0)) / max(int(agg_row.get('Request Count', 1)), 1) * 100,
                'avg_response_time': float(agg_row.get('Average Response Time', 0)),
                'median_response_time': float(agg_row.get('Median Response Time', 0)),
                'p95_response_time': float(agg_row.get('95%', 0)),
                'rps': float(agg_row.get('Requests/s', 0))
            }
            
        except Exception as e:
            print(f"⚠️ 無法分析場景 {scenario_dir.name}: {e}")
            return {}
    
    def analyze_experiment_results(self, exp_dir: Path) -> Dict:
        """分析單個實驗目錄的所有場景"""
        scenario_results = {}
        
        # 預期的8個標準場景
        expected_scenarios = [s['id'] for s in self.scenario_config.get('scenarios', [])]
        
        for scenario_id in expected_scenarios:
            scenario_dir = exp_dir / scenario_id
            if scenario_dir.exists():
                result = self.analyze_single_scenario(scenario_dir)
                if result:
                    result['scenario_type'] = self._get_scenario_type(scenario_id)
                    scenario_results[scenario_id] = result
                    
        return scenario_results
    
    def _get_scenario_type(self, scenario_id: str) -> str:
        """從場景ID獲取類型"""
        for scenario in self.scenario_config.get('scenarios', []):
            if scenario['id'] == scenario_id:
                return scenario['type']
        return scenario_id.split('_')[0]  # fallback
    
    def calculate_method_summary(self, scenario_results: Dict) -> Dict:
        """計算方法的總體指標"""
        if not scenario_results:
            return {}
            
        total_requests = sum(r['requests'] for r in scenario_results.values())
        total_failures = sum(r['failures'] for r in scenario_results.values())
        
        # 加權平均響應時間
        weighted_avg_response = 0
        if total_requests > 0:
            weighted_avg_response = sum(
                r['avg_response_time'] * r['requests'] for r in scenario_results.values()
            ) / total_requests
        
        # 平均指標
        avg_failure_rate = sum(r['failure_rate'] for r in scenario_results.values()) / len(scenario_results)
        avg_p95 = sum(r['p95_response_time'] for r in scenario_results.values()) / len(scenario_results)
        avg_rps = sum(r['rps'] for r in scenario_results.values()) / len(scenario_results)
        
        return {
            'scenarios_tested': len(scenario_results),
            'total_requests': total_requests,
            'total_failures': total_failures,
            'overall_failure_rate': (total_failures / max(total_requests, 1)) * 100,
            'avg_failure_rate': avg_failure_rate,
            'weighted_avg_response_time': weighted_avg_response,
            'avg_p95': avg_p95,
            'avg_rps': avg_rps
        }
    
    def generate_comparison_report(self):
        """生成比較報告"""
        print("🔍 開始標準化實驗結果分析...")
        
        # 查找實驗
        experiments = self.find_standardized_experiments()
        
        if not any(experiments.values()):
            print("❌ 未找到任何標準化實驗結果")
            return
        
        # 分析每個方法
        method_summaries = {}
        scenario_details = {}
        
        for method, exp_dirs in experiments.items():
            if not exp_dirs:
                continue
                
            # 取最新的實驗目錄
            latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
            print(f"📊 分析 {method} 實驗: {latest_exp.name}")
            
            scenario_results = self.analyze_experiment_results(latest_exp)
            method_summaries[method] = self.calculate_method_summary(scenario_results)
            scenario_details[method] = scenario_results
        
        # 生成報告
        self._print_comparison_summary(method_summaries)
        self._print_scenario_details(scenario_details)
        self._save_csv_reports(method_summaries, scenario_details)
        self._generate_markdown_report(method_summaries, scenario_details)
        
        print("✅ 標準化實驗分析完成！")
    
    def _print_comparison_summary(self, summaries: Dict):
        """打印比較摘要"""
        print("\n" + "="*60)
        print("📊 標準化方法比較摘要")
        print("="*60)
        
        # 表頭
        print(f"{'Method':<12} {'Scenarios':<9} {'Requests':<10} {'Avg RT(ms)':<11} {'Failure%':<9} {'P95(ms)':<9} {'RPS':<7}")
        print("-" * 70)
        
        # 數據行
        for method, summary in summaries.items():
            method_name = method.upper().replace('_', '-')
            print(f"{method_name:<12} "
                  f"{summary.get('scenarios_tested', 0):<9} "
                  f"{summary.get('total_requests', 0):<10,} "
                  f"{summary.get('weighted_avg_response_time', 0):<11.1f} "
                  f"{summary.get('avg_failure_rate', 0):<9.2f} "
                  f"{summary.get('avg_p95', 0):<9.0f} "
                  f"{summary.get('avg_rps', 0):<7.1f}")
    
    def _print_scenario_details(self, details: Dict):
        """打印場景詳細信息"""
        print("\n" + "="*60)
        print("📋 場景級別詳細比較")
        print("="*60)
        
        # 獲取所有場景
        all_scenarios = set()
        for method_scenarios in details.values():
            all_scenarios.update(method_scenarios.keys())
        
        for scenario_id in sorted(all_scenarios):
            scenario_type = self._get_scenario_type(scenario_id)
            print(f"\n🎯 {scenario_id} ({scenario_type})")
            print(f"{'Method':<12} {'Requests':<10} {'RT(ms)':<8} {'Failure%':<9} {'P95(ms)':<9} {'RPS':<7}")
            print("-" * 56)
            
            for method, scenarios in details.items():
                if scenario_id in scenarios:
                    result = scenarios[scenario_id]
                    method_name = method.upper().replace('_', '-')
                    print(f"{method_name:<12} "
                          f"{result['requests']:<10,} "
                          f"{result['avg_response_time']:<8.1f} "
                          f"{result['failure_rate']:<9.2f} "
                          f"{result['p95_response_time']:<9.0f} "
                          f"{result['rps']:<7.1f}")
                else:
                    method_name = method.upper().replace('_', '-')
                    print(f"{method_name:<12} {'N/A':<10} {'N/A':<8} {'N/A':<9} {'N/A':<9} {'N/A':<7}")
    
    def _save_csv_reports(self, summaries: Dict, details: Dict):
        """保存 CSV 報告"""
        output_dir = self.logs_dir
        
        # 方法比較 CSV
        method_file = output_dir / "standardized_method_comparison.csv"
        with open(method_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Scenarios', 'Total Requests', 'Avg Response Time (ms)', 
                           'Avg Failure Rate (%)', 'Avg P95 (ms)', 'Avg RPS'])
            
            for method, summary in summaries.items():
                writer.writerow([
                    method.upper().replace('_', '-'),
                    summary.get('scenarios_tested', 0),
                    summary.get('total_requests', 0),
                    round(summary.get('weighted_avg_response_time', 0), 2),
                    round(summary.get('avg_failure_rate', 0), 2),
                    round(summary.get('avg_p95', 0), 2),
                    round(summary.get('avg_rps', 0), 2)
                ])
        
        print(f"✅ 方法比較結果: {method_file}")
        
        # 場景比較 CSV
        scenario_file = output_dir / "standardized_scenario_comparison.csv"
        with open(scenario_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Scenario ID', 'Scenario Type', 'Method', 'Requests', 
                           'Failure Rate (%)', 'Avg Response Time (ms)', 'P95 Response Time (ms)', 'RPS'])
            
            all_scenarios = set()
            for method_scenarios in details.values():
                all_scenarios.update(method_scenarios.keys())
            
            for scenario_id in sorted(all_scenarios):
                scenario_type = self._get_scenario_type(scenario_id)
                for method, scenarios in details.items():
                    if scenario_id in scenarios:
                        result = scenarios[scenario_id]
                        writer.writerow([
                            scenario_id,
                            scenario_type,
                            method.upper().replace('_', '-'),
                            result['requests'],
                            round(result['failure_rate'], 2),
                            round(result['avg_response_time'], 2),
                            round(result['p95_response_time'], 2),
                            round(result['rps'], 2)
                        ])
        
        print(f"✅ 場景比較結果: {scenario_file}")
    
    def _generate_markdown_report(self, summaries: Dict, details: Dict):
        """生成 Markdown 報告"""
        report_content = f"""# 標準化三方法自動縮放比較報告

## 📋 實驗概述

**實驗時間**: {datetime.now().strftime('%Y年%m月%d日 %H時%M分')}
**比較方法**: Gym-HPA, GNNRL, K8s-HPA
**測試場景**: 8個標準化場景 (基於固定種子生成)

## 🎯 標準化測試場景

"""
        
        # 添加場景配置
        if self.scenario_config.get('scenarios'):
            report_content += "| 序號 | 場景ID | 類型 | 描述 |\n"
            report_content += "|-----|--------|------|------|\n"
            for i, scenario in enumerate(self.scenario_config['scenarios'], 1):
                report_content += f"| {i} | {scenario['id']} | {scenario['type']} | {scenario['description']} |\n"
        
        # 添加方法比較
        if summaries:
            report_content += "\n## 📊 方法總體性能比較\n\n"
            report_content += "| 方法 | 場景數 | 總請求數 | 平均響應時間(ms) | 平均失敗率(%) | 平均P95(ms) | 平均RPS |\n"
            report_content += "|-----|--------|----------|-----------------|--------------|-------------|--------|\n"
            
            for method, summary in summaries.items():
                method_name = method.upper().replace('_', '-')
                report_content += f"| {method_name} | {summary.get('scenarios_tested', 0)} | {summary.get('total_requests', 0):,} | {summary.get('weighted_avg_response_time', 0):.1f} | {summary.get('avg_failure_rate', 0):.2f} | {summary.get('avg_p95', 0):.0f} | {summary.get('avg_rps', 0):.1f} |\n"
        
        # 添加結論
        report_content += """

## 💡 關鍵發現

### 1. 整體性能
基於標準化的8個場景測試，各方法在不同負載模式下表現出不同特性。

### 2. 建議

**生產環境選擇**:
- 需要詳細分析上述數據來確定最適合的方法

---
*報告由標準化實驗分析器自動生成*
"""
        
        # 保存報告
        report_file = self.repo_root / "STANDARDIZED_COMPARISON_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 標準化比較報告: {report_file}")


def main():
    """主函數"""
    analyzer = SimpleStandardizedAnalyzer()
    analyzer.generate_comparison_report()


if __name__ == "__main__":
    main()