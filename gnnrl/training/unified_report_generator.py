#!/usr/bin/env python3
"""
統一實驗報告生成器 (Unified Experiment Report Generator)
====================================================

整合所有實驗結果，生成標準化的性能報告，支援橫向比較。

功能：
- 統一的壓測指標聚合
- 標準化的實驗結果格式
- 橫向比較報告
- 可視化性能分析
"""

import os
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from statistics import mean, median

class UnifiedReportGenerator:
    """統一實驗報告生成器"""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
        # 創建子目錄
        (self.experiments_dir / "comparison_reports").mkdir(exist_ok=True)
        (self.experiments_dir / "archive").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def create_experiment_directory(self, experiment_type: str, algorithm: str, 
                                  model: str, goal: str, steps: int, seed: int = 42) -> Path:
        """創建實驗目錄"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{timestamp}_{experiment_type}_{algorithm}_{model}_{goal}_{steps}"
        
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # 創建子目錄
        (exp_dir / "loadtest_scenarios").mkdir(exist_ok=True)
        (exp_dir / "performance_charts").mkdir(exist_ok=True)
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "models" / "checkpoints").mkdir(exist_ok=True)
        
        # 創建實驗配置文件
        experiment_info = {
            "id": exp_id,
            "type": experiment_type,
            "algorithm": algorithm,
            "model": model,
            "goal": goal,
            "steps": steps,
            "seed": seed,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        with open(exp_dir / "experiment_info.json", 'w') as f:
            json.dump(experiment_info, f, indent=2)
            
        self.logger.info(f"Created experiment directory: {exp_id}")
        return exp_dir
    
    def process_loadtest_scenario(self, scenario_dir: Path, scenario_order: int, 
                                scenario_name: str, start_time: datetime) -> Dict:
        """處理單個壓測場景結果"""
        try:
            # 查找統計文件
            stats_files = list(scenario_dir.glob("*_stats.csv"))
            if not stats_files:
                self.logger.warning(f"No stats file found in {scenario_dir}")
                return None
                
            stats_file = stats_files[0]
            df = pd.read_csv(stats_file)
            
            # 查找 Total 行
            total_row = df[df["Name"] == "Total"]
            if total_row.empty:
                total_row = df[df["Type"] == "Total"]
            if total_row.empty:
                self.logger.warning(f"No 'Total' row in {stats_file}")
                return None
                
            total = total_row.iloc[0]
            
            # 計算持續時間
            history_files = list(scenario_dir.glob("*_history.csv"))
            duration_sec = 900  # 預設 15 分鐘
            
            if history_files:
                try:
                    hist_df = pd.read_csv(history_files[0])
                    if not hist_df.empty:
                        duration_sec = len(hist_df)  # 每秒一個記錄
                except Exception as e:
                    self.logger.warning(f"Failed to parse history file: {e}")
            
            # 提取關鍵指標
            scenario_data = {
                "scenario_name": scenario_name,
                "scenario_order": scenario_order,
                "start_time": start_time.isoformat(),
                "end_time": (start_time.timestamp() + duration_sec),
                "duration_sec": duration_sec,
                "total_requests": total.get("Request Count", total.get("Requests", 0)),
                "successful_requests": total.get("Request Count", 0) - total.get("Failure Count", total.get("Failures", 0)),
                "failed_requests": total.get("Failure Count", total.get("Failures", 0)),
                "failure_rate": total.get("Failure Count", 0) / max(total.get("Request Count", 1), 1),
                "avg_rps": total.get("Requests/s", total.get("RPS", 0)),
                "max_rps": total.get("Max RPS", total.get("Requests/s", 0)),
                "min_rps": total.get("Min RPS", total.get("Requests/s", 0)),
                "p50_latency": total.get("50%", total.get("Median", 0)),
                "p95_latency": total.get("95%", total.get("95th percentile", 0)),
                "p99_latency": total.get("99%", total.get("99th percentile", 0)),
                "avg_latency": total.get("Average", total.get("Average Response Time", 0)),
            }
            
            # 創建場景信息文件
            scenario_info_file = scenario_dir / "scenario_info.json"
            with open(scenario_info_file, 'w') as f:
                json.dump(scenario_data, f, indent=2)
                
            return scenario_data
            
        except Exception as e:
            self.logger.error(f"Failed to process scenario {scenario_dir}: {e}")
            return None
    
    def generate_loadtest_summary(self, exp_dir: Path, scenario_dirs: List[Path], 
                                experiment_info: Dict) -> pd.DataFrame:
        """生成壓測摘要報告"""
        summary_data = []
        start_time = datetime.fromisoformat(experiment_info["start_time"])
        
        for i, scenario_dir in enumerate(scenario_dirs, 1):
            scenario_name = scenario_dir.name.split('_')[0] if '_' in scenario_dir.name else scenario_dir.name
            scenario_time = start_time.timestamp() + (i - 1) * 960  # 每個場景間隔16分鐘
            
            scenario_data = self.process_loadtest_scenario(
                scenario_dir, i, scenario_name, 
                datetime.fromtimestamp(scenario_time)
            )
            
            if scenario_data:
                scenario_data["experiment_id"] = experiment_info["id"]
                summary_data.append(scenario_data)
        
        # 創建 DataFrame 並保存
        df = pd.DataFrame(summary_data)
        summary_file = exp_dir / "loadtest_summary.csv"
        df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Generated loadtest summary: {summary_file}")
        return df
    
    def generate_experiment_summary(self, exp_dir: Path, loadtest_df: pd.DataFrame, 
                                  training_metrics: Dict = None) -> Dict:
        """生成完整實驗摘要"""
        
        # 讀取實驗配置
        with open(exp_dir / "experiment_info.json", 'r') as f:
            experiment_info = json.load(f)
        
        # 計算壓測聚合指標
        if not loadtest_df.empty:
            loadtest_aggregation = {
                "total_scenarios_executed": len(loadtest_df),
                "scenario_distribution": loadtest_df["scenario_name"].value_counts().to_dict(),
                "overall_performance": {
                    "avg_rps": loadtest_df["avg_rps"].mean(),
                    "overall_p95_latency": loadtest_df["p95_latency"].mean(),
                    "overall_p99_latency": loadtest_df["p99_latency"].mean(),
                    "total_requests": loadtest_df["total_requests"].sum(),
                    "overall_failure_rate": loadtest_df["failure_rate"].mean()
                }
            }
        else:
            loadtest_aggregation = {"total_scenarios_executed": 0}
        
        # 分析動作歷史（如果存在）
        scaling_behavior = {}
        action_file = exp_dir / "action_history.csv"
        if action_file.exists():
            try:
                action_df = pd.read_csv(action_file)
                scaling_behavior = {
                    "total_scaling_actions": len(action_df),
                    "scaling_distribution": {
                        "scale_up": len(action_df[action_df["action_value"] > 0]),
                        "scale_down": len(action_df[action_df["action_value"] < 0]),
                        "no_action": len(action_df[action_df["action_value"] == 0])
                    },
                    "most_scaled_service": action_df["deployment_name"].value_counts().index[0] if not action_df.empty else "none",
                    "avg_replicas_per_service": action_df["new_replicas"].mean() if "new_replicas" in action_df.columns else 0
                }
            except Exception as e:
                self.logger.warning(f"Failed to analyze action history: {e}")
        
        # 組合完整摘要
        experiment_summary = {
            "experiment_info": experiment_info,
            "training_metrics": training_metrics or {},
            "loadtest_aggregation": loadtest_aggregation,
            "scaling_behavior": scaling_behavior,
            "comparison_baseline": {}  # 後續比較時填入
        }
        
        # 更新結束時間和狀態
        experiment_summary["experiment_info"]["end_time"] = datetime.now().isoformat()
        experiment_summary["experiment_info"]["status"] = "completed"
        
        # 保存摘要
        summary_file = exp_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(experiment_summary, f, indent=2)
            
        # 同時更新 experiment_info.json
        with open(exp_dir / "experiment_info.json", 'w') as f:
            json.dump(experiment_summary["experiment_info"], f, indent=2)
            
        self.logger.info(f"Generated experiment summary: {summary_file}")
        return experiment_summary
    
    def update_comparison_reports(self):
        """更新橫向比較報告"""
        comparison_dir = self.experiments_dir / "comparison_reports"
        
        # 收集所有實驗摘要
        all_summaries = []
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.') and exp_dir.name not in ['comparison_reports', 'archive']:
                summary_file = exp_dir / "experiment_summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            all_summaries.append(summary)
                    except Exception as e:
                        self.logger.warning(f"Failed to load summary from {exp_dir}: {e}")
        
        if not all_summaries:
            self.logger.warning("No experiment summaries found for comparison")
            return
        
        # 生成比較表格
        comparison_data = []
        for summary in all_summaries:
            exp_info = summary["experiment_info"]
            loadtest = summary.get("loadtest_aggregation", {})
            scaling = summary.get("scaling_behavior", {})
            training = summary.get("training_metrics", {})
            
            row = {
                "experiment_id": exp_info["id"],
                "timestamp": exp_info["start_time"],
                "experiment_type": exp_info["type"],
                "algorithm": exp_info["algorithm"],
                "model": exp_info["model"],
                "goal": exp_info["goal"],
                "total_steps": exp_info["steps"],
                "duration_hours": self._calculate_duration_hours(exp_info),
                "avg_p95_latency": loadtest.get("overall_performance", {}).get("overall_p95_latency", 0),
                "avg_p99_latency": loadtest.get("overall_performance", {}).get("overall_p99_latency", 0),
                "avg_rps": loadtest.get("overall_performance", {}).get("avg_rps", 0),
                "total_requests": loadtest.get("overall_performance", {}).get("total_requests", 0),
                "overall_failure_rate": loadtest.get("overall_performance", {}).get("overall_failure_rate", 0),
                "total_scaling_actions": scaling.get("total_scaling_actions", 0),
                "final_reward": training.get("final_reward", 0),
                "convergence_step": training.get("convergence_step", 0)
            }
            comparison_data.append(row)
        
        # 保存比較表格
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = comparison_dir / "all_experiments_summary.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        # 保存最新比較數據
        latest_file = comparison_dir / f"latest_comparison_{datetime.now().strftime('%Y%m%d')}.json"
        with open(latest_file, 'w') as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "total_experiments": len(all_summaries),
                "comparison_data": comparison_data
            }, f, indent=2)
        
        self.logger.info(f"Updated comparison reports: {comparison_file}")
        
    def _calculate_duration_hours(self, exp_info: Dict) -> float:
        """計算實驗持續時間（小時）"""
        try:
            start = datetime.fromisoformat(exp_info["start_time"])
            end = datetime.fromisoformat(exp_info.get("end_time", datetime.now().isoformat()))
            return (end - start).total_seconds() / 3600
        except:
            return 0.0

# 便利函數
def create_report_generator() -> UnifiedReportGenerator:
    """創建報告生成器實例"""
    return UnifiedReportGenerator()

def process_experiment_results(exp_dir: Path, scenario_dirs: List[Path], 
                             experiment_config: Dict, training_metrics: Dict = None):
    """處理實驗結果並生成報告"""
    generator = create_report_generator()
    
    # 生成壓測摘要
    loadtest_df = generator.generate_loadtest_summary(exp_dir, scenario_dirs, experiment_config)
    
    # 生成實驗摘要
    experiment_summary = generator.generate_experiment_summary(exp_dir, loadtest_df, training_metrics)
    
    # 更新比較報告
    generator.update_comparison_reports()
    
    return experiment_summary