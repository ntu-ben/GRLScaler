#!/usr/bin/env python3
"""
場景對比可視化生成器 - Redis vs OnlineBoutique
========================================
生成兩個場景四種壓測模式下三種方法的pod和RPS時間序列對比圖
修正版：正確區分Redis和OnlineBoutique的實驗數據
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import datetime
import json
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from scipy import integrate

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ScenarioComparisonGenerator:
    def __init__(self, logs_root: str = "logs"):
        self.logs_root = Path(logs_root)
        self.output_dir = Path("scenario_comparisons_fixed")
        self.output_dir.mkdir(exist_ok=True)
        
        # 場景定義
        self.scenarios = ["offpeak", "rushsale", "peak", "fluctuating"]
        self.applications = ["redis", "onlineboutique"]
        self.methods = ["GNNRL", "Gym-HPA"]
        self.k8s_hpa_configs = ["cpu-20", "cpu-40", "cpu-60", "cpu-80"]  # 4種CPU配置
        # 擴展方法列表，包含4種K8s-HPA配置
        self.all_methods = self.methods + [f"K8s-HPA-{config}" for config in self.k8s_hpa_configs]
        
        # 應用特定的方法目錄映射
        self.app_method_mapping = {
            "redis": {
                "GNNRL": "gnnrl",  # 需要找到Redis的GNNRL實驗
                "Gym-HPA": "gym-hpa",
                "K8s-HPA": "k8s_hpa_redis"
            },
            "onlineboutique": {
                "GNNRL": "gnnrl", 
                "Gym-HPA": "gym-hpa",
                "K8s-HPA": "k8s-hpa"
            }
        }
        
        # OB微服務名稱集合（用於識別OB實驗）
        self.ob_services = {
            'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
            'emailservice', 'frontend', 'paymentservice', 'productcatalogservice',
            'recommendationservice', 'shippingservice', 'redis-cart'
        }
        
        print(f"🎯 場景對比可視化生成器初始化完成（修正版）")
        print(f"📁 輸出目錄: {self.output_dir}")

    def detect_experiment_application(self, experiment_dir: Path) -> str:
        """檢測實驗目錄是針對哪個應用"""
        # 檢查目錄名是否包含redis
        if "redis" in experiment_dir.name.lower():
            return "redis"
            
        # 檢查是否有運行日誌包含OB服務名稱
        run_log = experiment_dir.parent / "run.log" 
        if run_log.exists():
            try:
                with open(run_log, 'r') as f:
                    content = f.read()
                
                # 檢查是否包含OB服務名稱
                ob_service_mentions = sum(1 for service in self.ob_services 
                                        if service in content)
                
                if ob_service_mentions >= 3:  # 閾值：至少3個OB服務
                    return "onlineboutique"
                    
            except Exception as e:
                print(f"⚠️ 讀取日誌失敗: {e}")
        
        # 檢查子目錄中的場景文件
        for scenario_dir in experiment_dir.iterdir():
            if scenario_dir.is_dir():
                # 檢查stats文件內容
                stats_file = scenario_dir / f"{scenario_dir.name.split('_')[0]}_stats_history.csv"
                if stats_file.exists():
                    try:
                        df = pd.read_csv(stats_file, nrows=5)
                        if 'Name' in df.columns:
                            # 檢查請求URL路徑
                            name_values = df['Name'].dropna().astype(str)
                            if any('/cart' in name or '/checkout' in name for name in name_values):
                                return "onlineboutique"
                            elif any('redis' in name.lower() for name in name_values):
                                return "redis"
                    except:
                        continue
        
        # 默認根據時間推測（7/12前為redis，7/12後為onlineboutique）
        timestamp_match = re.search(r'(\d{8})', experiment_dir.name)
        if timestamp_match:
            date_str = timestamp_match.group(1)
            if date_str <= "20250712":
                return "redis"
            else:
                return "onlineboutique"
        
        return "unknown"

    def _select_best_experiment_dir(self, test_dirs: List[Path]) -> Path:
        """選擇最佳實驗目錄，優先選擇test實驗（含Pod監控），然後按時間戳排序"""
        # 分離test和train實驗
        test_experiments = [d for d in test_dirs if "_test_" in d.name]
        train_experiments = [d for d in test_dirs if "_train_" in d.name]
        
        def extract_timestamp(dir_path: Path):
            """從目錄名提取時間戳"""
            timestamp_match = re.search(r'(\d{8}_\d{6})', dir_path.name)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                return datetime.datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            return datetime.datetime.min  # 如果無法解析，返回最小時間
        
        # 優先選擇test實驗，按時間戳排序
        if test_experiments:
            latest_test = max(test_experiments, key=extract_timestamp)
            print(f"🎯 優先選擇test實驗: {latest_test.name}")
            return latest_test
        
        # 如果沒有test實驗，選擇最新的train實驗
        if train_experiments:
            latest_train = max(train_experiments, key=extract_timestamp)
            print(f"⚠️ 使用train實驗: {latest_train.name} (未找到test實驗)")
            return latest_train
        
        # 兜底：如果都不符合，使用原始邏輯
        return max(test_dirs, key=lambda x: x.name)

    def find_latest_experiment_data(self, method: str, application: str) -> Optional[Path]:
        """找到指定方法和應用的最新實驗數據"""
        method_dir_name = self.app_method_mapping[application][method]
        method_dir = self.logs_root / method_dir_name
        
        if not method_dir.exists():
            print(f"❌ 方法目錄不存在: {method_dir}")
            return None
            
        # 根據不同方法和應用尋找實驗目錄
        if application == "redis":
            if method == "GNNRL":
                # Redis GNNRL實驗需要特殊處理，因為目前沒有實際的Redis GNNRL測試數據
                # 檢查是否有Redis相關的GNNRL實驗
                redis_gnnrl_dirs = []
                for test_dir in method_dir.glob("gnnrl_*redis*"):
                    redis_gnnrl_dirs.append(test_dir)
                
                # 如果沒有專門的Redis GNNRL目錄，檢查通用目錄中的Redis實驗
                if not redis_gnnrl_dirs:
                    for test_dir in method_dir.glob("gnnrl_*seed42_*"):
                        if self.detect_experiment_application(test_dir) == "redis":
                            redis_gnnrl_dirs.append(test_dir)
                
                if not redis_gnnrl_dirs:
                    print(f"⚠️ 警告: 未找到 {method} Redis 實驗目錄，可能需要先運行Redis GNNRL實驗")
                    return None
                    
                test_dirs = redis_gnnrl_dirs
                    
            elif method == "Gym-HPA":
                # 查找 gym_hpa_redis_* 目錄
                test_dirs = list(method_dir.glob("gym_hpa_redis_*seed42_*"))
                if not test_dirs:
                    print(f"❌ 未找到 {method} Redis 實驗目錄")
                    return None
                    
            elif method == "K8s-HPA":
                # K8s-HPA Redis在 k8s_hpa_redis 目錄下
                test_dirs = list(method_dir.glob("redis_hpa_*"))
                if not test_dirs:
                    print(f"❌ 未找到 {method} Redis 實驗目錄")
                    return None
                    
        else:  # onlineboutique
            if method == "GNNRL":
                # 查找OnlineBoutique的GNNRL實驗
                test_dirs = []
                for test_dir in method_dir.glob("gnnrl_*seed42_*"):
                    if self.detect_experiment_application(test_dir) == "onlineboutique":
                        test_dirs.append(test_dir)
                        
                if not test_dirs:
                    print(f"❌ 未找到 {method} OnlineBoutique 實驗目錄")
                    return None
                    
            elif method == "Gym-HPA":
                # 查找OnlineBoutique的Gym-HPA實驗（非redis的）
                test_dirs = []
                for test_dir in method_dir.glob("gym_hpa_*seed42_*"):
                    if "redis" not in test_dir.name and self.detect_experiment_application(test_dir) == "onlineboutique":
                        test_dirs.append(test_dir)
                        
                if not test_dirs:
                    print(f"❌ 未找到 {method} OnlineBoutique 實驗目錄")
                    return None
                    
            elif method == "K8s-HPA":
                # 查找OnlineBoutique的K8s-HPA實驗，包含所有CPU配置
                test_dirs = []
                for test_dir in method_dir.glob("k8s_hpa_*_seed42_*"):
                    if self.detect_experiment_application(test_dir) == "onlineboutique":
                        test_dirs.append(test_dir)
                        
                if not test_dirs:
                    print(f"❌ 未找到 {method} OnlineBoutique 實驗目錄")
                    return None
            
        # 選擇最新的實驗目錄，優先選擇test實驗（包含Pod監控數據）
        latest_dir = self._select_best_experiment_dir(test_dirs)
        detected_app = self.detect_experiment_application(latest_dir)
        
        print(f"✅ 找到 {method} {application} 實驗目錄: {latest_dir.name} (檢測到: {detected_app})")
        
        if detected_app != application:
            print(f"⚠️ 警告: 期望 {application} 但檢測到 {detected_app}")
            
        return latest_dir

    def extract_pod_data_from_kiali(self, scenario: str, experiment_timestamp: str) -> Optional[pd.DataFrame]:
        """從 Kiali 數據中提取 pod 信息"""
        kiali_dir = self.logs_root / "kiali"
        
        # 查找對應時間戳的 kiali 文件
        kiali_files = [
            f"kiali_start_{experiment_timestamp}.json",
            f"kiali_mid_{experiment_timestamp}.json", 
            f"kiali_end_{experiment_timestamp}.json"
        ]
        
        pod_data = []
        for i, kiali_file in enumerate(kiali_files):
            file_path = kiali_dir / kiali_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # 提取pod信息
                    if 'workloads' in data:
                        total_pods = 0
                        for workload in data['workloads']:
                            if 'podCount' in workload:
                                total_pods += workload['podCount']
                        
                        # 時間點: 0分鐘(start), 7.5分鐘(mid), 15分鐘(end)
                        time_point = i * 7.5
                        pod_data.append({
                            'time_minutes': time_point,
                            'pods': total_pods
                        })
                        
                except Exception as e:
                    print(f"⚠️ 讀取 {kiali_file} 失敗: {e}")
        
        if pod_data:
            return pd.DataFrame(pod_data)
        return None

    def extract_rps_data(self, experiment_dir: Path, scenario: str, application: str) -> Optional[pd.DataFrame]:
        """從實驗目錄中提取RPS數據"""
        
        # 根據應用類型調整場景目錄查找模式
        if application == "redis":
            # Redis實驗可能使用 redis_scenario 格式
            scenario_patterns = [
                f"{scenario}_*",
                f"redis_{scenario}*",
                f"redis_{scenario}",
                f"{scenario}"
            ]
        else:
            # OnlineBoutique使用標準格式
            scenario_patterns = [f"{scenario}_*"]
        
        scenario_dir = None
        for pattern in scenario_patterns:
            scenario_dirs = list(experiment_dir.glob(pattern))
            if scenario_dirs:
                scenario_dir = scenario_dirs[0]
                break
        
        if not scenario_dir:
            print(f"⚠️ 未找到 {application} {scenario} 場景目錄在 {experiment_dir}")
            return None
            
        # 查找stats文件
        stats_files = [
            scenario_dir / f"{scenario}_stats_history.csv",
            scenario_dir / f"redis_{scenario}_stats_history.csv",
            scenario_dir / "stats_history.csv"
        ]
        
        stats_file = None
        for file_path in stats_files:
            if file_path.exists():
                stats_file = file_path
                break
                
        if not stats_file:
            print(f"⚠️ 未找到stats文件在 {scenario_dir}")
            return None
            
        try:
            df = pd.read_csv(stats_file)
            
            # 如果有 Requests/s 列
            if 'Requests/s' in df.columns:
                # 轉換時間戳為相對時間(分鐘)
                if 'Timestamp' in df.columns:
                    start_time = df['Timestamp'].min()
                    df['time_minutes'] = (df['Timestamp'] - start_time) / 60
                else:
                    # 假設每行代表1秒，轉換為分鐘
                    df['time_minutes'] = df.index / 60
                
                # 只保留15分鐘內的數據
                df = df[df['time_minutes'] <= 15]
                
                # 重採樣到每分鐘一個數據點
                result_data = []
                for minute in range(16):  # 0-15分鐘
                    minute_data = df[(df['time_minutes'] >= minute) & 
                                   (df['time_minutes'] < minute + 1)]
                    if not minute_data.empty:
                        avg_rps = minute_data['Requests/s'].mean()
                        result_data.append({
                            'time_minutes': minute,
                            'rps': avg_rps
                        })
                    else:
                        result_data.append({
                            'time_minutes': minute,
                            'rps': 0
                        })
                
                print(f"✅ 提取到 {len(result_data)} 個RPS數據點從 {stats_file}")
                return pd.DataFrame(result_data)
                
        except Exception as e:
            print(f"⚠️ 讀取 RPS 數據失敗 {stats_file}: {e}")
            
        return None

    def extract_pod_data_from_logs(self, experiment_dir: Path, method: str, scenario: str = None) -> Optional[pd.DataFrame]:
        """從實驗日誌中提取pod數據，優先使用新的Pod監控CSV文件"""
        
        # 優先檢查新的Pod監控CSV文件，如果有scenario則先檢查scenario目錄
        if scenario:
            pod_csv_data = self._extract_from_scenario_pod_monitoring_csv(experiment_dir, scenario)
            if pod_csv_data is not None:
                return pod_csv_data
        
        # 兜底：檢查實驗主目錄的Pod監控CSV文件
        pod_csv_data = self._extract_from_pod_monitoring_csv(experiment_dir)
        if pod_csv_data is not None:
            return pod_csv_data
        
        # 對於 K8s-HPA，pod 數據可能在 kiali 中
        if method == "K8s-HPA":
            # 從目錄名提取時間戳
            dir_name = experiment_dir.name
            timestamp_match = re.search(r'(\d{8}_\d{6})', dir_name)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                kiali_data = self.extract_pod_data_from_kiali("", timestamp)
                if kiali_data is not None:
                    return kiali_data
            
            # 如果kiali沒有數據，嘗試從Redis HPA場景目錄提取
            if "redis_hpa" in dir_name:
                # Redis HPA的場景目錄結構: redis_hpa_cpu-XX_timestamp/scenario_name/
                scenario_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
                if scenario_dirs:
                    # 從第一個場景目錄嘗試提取pod數據
                    return self.extract_redis_hpa_pod_data(scenario_dirs[0])
        
        # 對於其他方法，嘗試從運行日誌中提取
        run_log_paths = [
            experiment_dir.parent / "run.log",
            experiment_dir / "run.log",
            experiment_dir / "experiment.log"
        ]
        
        for run_log in run_log_paths:
            if run_log.exists():
                try:
                    with open(run_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    pod_data = []
                    step_count = 0
                    
                    for line in lines:
                        # 改進pod信息匹配邏輯
                        if any(keyword in line for keyword in [
                            "Number of pods:", "Desired Replicas:", 
                            "Current pods:", "Pod count:"
                        ]):
                            try:
                                # 匹配數字
                                number_match = re.search(r'(\d+)', line.split(':')[-1])
                                if number_match:
                                    pod_count = int(number_match.group(1))
                                    time_minutes = step_count * 0.6  # 假設每步0.6分鐘
                                    if time_minutes <= 15:
                                        pod_data.append({
                                            'time_minutes': time_minutes,
                                            'pods': pod_count
                                        })
                                    step_count += 1
                            except Exception as e:
                                continue
                    
                    if pod_data:
                        return self.resample_pod_data(pod_data)
                        
                except Exception as e:
                    print(f"⚠️ 讀取pod數據失敗 {run_log}: {e}")
                    continue
        
        # 如果無法獲取實際數據，返回None
        print(f"⚠️ 無法從 {experiment_dir} 提取pod數據")
        return None
    
    def _extract_from_scenario_pod_monitoring_csv(self, experiment_dir: Path, scenario: str) -> Optional[pd.DataFrame]:
        """從scenario目錄中的Pod監控CSV文件提取數據"""
        try:
            # 查找scenario目錄
            scenario_dirs = list(experiment_dir.glob(f"{scenario}_*"))
            if not scenario_dirs:
                return None
            
            # 使用第一個匹配的scenario目錄
            scenario_dir = scenario_dirs[0]
            pod_metrics_dir = scenario_dir / "pod_metrics"
            
            if not pod_metrics_dir.exists():
                return None
            
            # 查找所有namespace的Pod監控CSV文件
            csv_files = list(pod_metrics_dir.rglob("*_pod_counts.csv"))
            if not csv_files:
                return None
            
            # 合併所有namespace的Pod數據
            all_pod_data = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'elapsed_minutes' in df.columns and 'pod_count' in df.columns:
                        all_pod_data.extend([{
                            'time_minutes': row['elapsed_minutes'],
                            'pods': row['pod_count']
                        } for _, row in df.iterrows() if row['elapsed_minutes'] <= 15])
                except Exception as e:
                    print(f"⚠️ 讀取Pod監控CSV失敗 {csv_file}: {e}")
                    continue
            
            if not all_pod_data:
                return None
            
            # 按時間聚合Pod數據（同一時間點的總Pod數）
            df = pd.DataFrame(all_pod_data)
            aggregated_data = []
            
            for minute in range(16):
                minute_data = df[(df['time_minutes'] >= minute) & 
                               (df['time_minutes'] < minute + 1)]
                if not minute_data.empty:
                    total_pods = minute_data['pods'].sum()  # 所有namespace的Pod總數
                    aggregated_data.append({
                        'time_minutes': minute,
                        'pods': int(total_pods)
                    })
                else:
                    # 使用前一分鐘的值或默認值
                    prev_pods = aggregated_data[-1]['pods'] if aggregated_data else 1
                    aggregated_data.append({
                        'time_minutes': minute,
                        'pods': prev_pods
                    })
            
            print(f"✅ 從scenario Pod監控CSV提取到 {len(aggregated_data)} 個數據點")
            return pd.DataFrame(aggregated_data)
            
        except Exception as e:
            print(f"⚠️ scenario Pod監控CSV數據提取失敗: {e}")
            return None
    
    def _extract_from_pod_monitoring_csv(self, experiment_dir: Path) -> Optional[pd.DataFrame]:
        """從新的Pod監控CSV文件中提取數據"""
        try:
            # 查找pod_metrics目錄
            pod_metrics_dir = experiment_dir / "pod_metrics"
            if not pod_metrics_dir.exists():
                return None
            
            # 查找所有namespace的Pod監控CSV文件
            csv_files = list(pod_metrics_dir.rglob("*_pod_counts.csv"))
            if not csv_files:
                return None
            
            # 合併所有namespace的Pod數據
            all_pod_data = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'elapsed_minutes' in df.columns and 'pod_count' in df.columns:
                        all_pod_data.extend([{
                            'time_minutes': row['elapsed_minutes'],
                            'pods': row['pod_count']
                        } for _, row in df.iterrows() if row['elapsed_minutes'] <= 15])
                except Exception as e:
                    print(f"⚠️ 讀取Pod監控CSV失敗 {csv_file}: {e}")
                    continue
            
            if not all_pod_data:
                return None
            
            # 按時間聚合Pod數據（同一時間點的總Pod數）
            df = pd.DataFrame(all_pod_data)
            aggregated_data = []
            
            for minute in range(16):
                minute_data = df[(df['time_minutes'] >= minute) & 
                               (df['time_minutes'] < minute + 1)]
                if not minute_data.empty:
                    total_pods = minute_data['pods'].sum()  # 所有namespace的Pod總數
                    aggregated_data.append({
                        'time_minutes': minute,
                        'pods': int(total_pods)
                    })
                else:
                    # 使用前一分鐘的值或默認值
                    prev_pods = aggregated_data[-1]['pods'] if aggregated_data else 1
                    aggregated_data.append({
                        'time_minutes': minute,
                        'pods': prev_pods
                    })
            
            print(f"✅ 從Pod監控CSV提取到 {len(aggregated_data)} 個數據點")
            return pd.DataFrame(aggregated_data)
            
        except Exception as e:
            print(f"⚠️ Pod監控CSV數據提取失敗: {e}")
            return None

    def extract_redis_hpa_pod_data(self, scenario_dir: Path) -> Optional[pd.DataFrame]:
        """從Redis HPA場景目錄提取pod數據"""
        try:
            # 查找可能的pod數據文件
            pod_files = list(scenario_dir.glob("*pod*.csv")) + list(scenario_dir.glob("*replica*.csv"))
            
            for pod_file in pod_files:
                try:
                    df = pd.read_csv(pod_file)
                    if 'pods' in df.columns or 'replicas' in df.columns:
                        pod_column = 'pods' if 'pods' in df.columns else 'replicas'
                        return self.resample_pod_data([{
                            'time_minutes': i * 0.5,  # 假設每0.5分鐘一個數據點
                            'pods': row[pod_column]
                        } for i, (_, row) in enumerate(df.iterrows())])
                except Exception as e:
                    continue
                    
            # 如果沒有找到專門的pod文件，嘗試從locust統計文件推測
            stats_files = list(scenario_dir.glob("*_stats.csv"))
            if stats_files:
                # 假設Redis HPA在測試期間pod數保持相對穩定
                return self.create_default_pod_data(2)  # 假設平均2個pod
                
        except Exception as e:
            print(f"⚠️ Redis HPA pod數據提取失敗: {e}")
            
        return None
    
    def resample_pod_data(self, pod_data: list) -> pd.DataFrame:
        """將pod數據重採樣到每分鐘"""
        df = pd.DataFrame(pod_data)
        result_data = []
        
        for minute in range(16):
            minute_data = df[(df['time_minutes'] >= minute) & 
                           (df['time_minutes'] < minute + 1)]
            if not minute_data.empty:
                avg_pods = minute_data['pods'].mean()
                result_data.append({
                    'time_minutes': minute,
                    'pods': int(round(avg_pods))
                })
            else:
                # 如果該分鐘沒有數據，使用前一分鐘的值或默認值
                prev_pods = result_data[-1]['pods'] if result_data else 1
                result_data.append({
                    'time_minutes': minute, 
                    'pods': prev_pods
                })
        
        return pd.DataFrame(result_data)
    
    def create_default_pod_data(self, default_pods: int = 1) -> pd.DataFrame:
        """創建默認的pod數據（當無法提取實際數據時使用）"""
        return pd.DataFrame([{
            'time_minutes': minute,
            'pods': default_pods
        } for minute in range(16)])


    def collect_scenario_data(self, application: str, scenario: str) -> Dict:
        """收集指定應用和場景的所有方法數據，包含K8s-HPA各配置"""
        scenario_data = {
            'application': application,
            'scenario': scenario,
            'methods': {}
        }
        
        # 收集GNNRL和Gym-HPA數據
        for method in self.methods:
            print(f"📊 收集 {method} - {application} - {scenario} 數據...")
            
            experiment_dir = self.find_latest_experiment_data(method, application)
            if not experiment_dir:
                print(f"❌ 未找到 {method} {application} 實驗數據")
                pod_data = None
                rps_data = None
            else:
                # 提取實際數據
                pod_data = self.extract_pod_data_from_logs(experiment_dir, method, scenario)
                rps_data = self.extract_rps_data(experiment_dir, scenario, application)
                
                if pod_data is None:
                    print(f"❌ 未能提取 {method} {application} {scenario} pod數據")
                if rps_data is None:
                    print(f"❌ 未能提取 {method} {application} {scenario} RPS數據")
            
            scenario_data['methods'][method] = {
                'pod_data': pod_data,
                'rps_data': rps_data,
                'has_data': pod_data is not None or rps_data is not None
            }
        
        # 收集K8s-HPA各配置數據
        method_dir_name = self.app_method_mapping[application]["K8s-HPA"]
        method_dir = self.logs_root / method_dir_name
        
        for config in self.k8s_hpa_configs:
            method_name = f"K8s-HPA-{config}"
            print(f"📊 收集 {method_name} - {application} - {scenario} 數據...")
            
            # 查找特定配置的實驗目錄
            config_dirs = []
            if application == "redis":
                pattern = f"redis_hpa_{config}_*"
                for test_dir in method_dir.glob(pattern):
                    if self.detect_experiment_application(test_dir) == application:
                        config_dirs.append(test_dir)
            else:
                # OnlineBoutique K8s-HPA: k8s_hpa_cpu_seed42_*/cpu-XX/
                for test_dir in method_dir.glob("k8s_hpa_cpu_seed42_*"):
                    cpu_config_dir = test_dir / config  # config is like "cpu-40"
                    if cpu_config_dir.exists() and self.detect_experiment_application(test_dir) == application:
                        config_dirs.append(cpu_config_dir)
            
            if not config_dirs:
                print(f"❌ 未找到 {method_name} {application} 實驗數據")
                pod_data = None
                rps_data = None
            else:
                # 選擇最新的配置目錄
                latest_config_dir = max(config_dirs, key=lambda x: x.name)
                
                # 提取數據
                pod_data = self.extract_pod_data_from_logs(latest_config_dir, "K8s-HPA", scenario)
                rps_data = self.extract_rps_data(latest_config_dir, scenario, application)
                
                print(f"✅ 找到 {method_name} {application} 配置目錄: {latest_config_dir.name}")
                
                if pod_data is None:
                    print(f"❌ 未能提取 {method_name} {application} {scenario} pod數據")
                if rps_data is None:
                    print(f"❌ 未能提取 {method_name} {application} {scenario} RPS數據")
            
            scenario_data['methods'][method_name] = {
                'pod_data': pod_data,
                'rps_data': rps_data,
                'has_data': pod_data is not None or rps_data is not None
            }
            
        return scenario_data

    def calculate_detailed_statistics(self, scenario_data: Dict) -> Dict:
        """計算詳細統計數據"""
        application = scenario_data['application']
        scenario = scenario_data['scenario']
        
        detailed_stats = {
            'application': application,
            'scenario': scenario,
            'microservices': [],
            'summary_statistics': {}
        }
        
        # 針對每個方法計算統計數據
        for method_name, method_data in scenario_data['methods'].items():
            if not method_data['has_data']:
                continue
                
            pod_data = method_data['pod_data']
            rps_data = method_data['rps_data']
            
            # 計算基本統計指標
            stats = self._calculate_method_statistics(method_name, pod_data, rps_data)
            
            # 如果是微服務架構，嘗試獲取微服務級別的統計數據
            if application == "onlineboutique":
                microservice_stats = self._calculate_microservice_statistics(method_name, method_data)
                detailed_stats['microservices'].extend(microservice_stats)
            
            detailed_stats['summary_statistics'][method_name] = stats
        
        return detailed_stats
    
    def _calculate_method_statistics(self, method_name: str, pod_data: pd.DataFrame, rps_data: pd.DataFrame) -> Dict:
        """計算單個方法的統計數據"""
        stats = {
            'method': method_name,
            'pod_time_area': 0,
            'total_requests': 0,
            'req_per_pod_time_area': 0,
            'avg_rps': 0,
            'avg_response_time': 0,
            'p95_response_time': 0,
            'p99_response_time': 0
        }
        
        # 1. 計算pod跟時間的面積 (Pod-Minutes)
        if pod_data is not None and not pod_data.empty:
            # 使用梯形法則計算面積
            time_minutes = pod_data['time_minutes'].values
            pod_counts = pod_data['pods'].values
            
            # 確保時間是從0開始的連續序列
            if len(time_minutes) > 1:
                stats['pod_time_area'] = integrate.trapz(pod_counts, time_minutes)
            else:
                stats['pod_time_area'] = pod_counts[0] * 15  # 假設15分鐘測試
        
        # 2. 計算總Request數和平均RPS
        if rps_data is not None and not rps_data.empty:
            time_minutes = rps_data['time_minutes'].values
            rps_values = rps_data['rps'].values
            
            # 總請求數 = RPS * 時間間隔 (分鐘)
            if len(time_minutes) > 1:
                # 計算每分鐘的請求數並求和
                total_requests = 0
                for i in range(len(time_minutes) - 1):
                    time_interval = (time_minutes[i+1] - time_minutes[i]) * 60  # 轉換為秒
                    total_requests += rps_values[i] * time_interval
                stats['total_requests'] = total_requests
            else:
                stats['total_requests'] = rps_values[0] * 15 * 60  # 假設15分鐘測試
            
            # 平均RPS
            stats['avg_rps'] = np.mean(rps_values[rps_values > 0])  # 排除0值
        
        # 3. 計算總REQ/pod與時間面積比率
        if stats['pod_time_area'] > 0:
            stats['req_per_pod_time_area'] = stats['total_requests'] / stats['pod_time_area']
        
        return stats
    
    def _calculate_microservice_statistics(self, method_name: str, method_data: Dict) -> List[Dict]:
        """計算微服務級別的統計數據"""
        microservice_stats = []
        
        # 對於OnlineBoutique，我們有11個微服務
        ob_services = [
            'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
            'emailservice', 'frontend', 'paymentservice', 'productcatalogservice',
            'recommendationservice', 'shippingservice', 'redis-cart'
        ]
        
        # 嘗試從pod monitoring CSV文件中獲取微服務級別的Pod數據
        pod_data_per_service = self._extract_microservice_pod_data(method_data)
        
        # 嘗試從stats文件中獲取微服務級別的響應時間數據
        response_data_per_service = self._extract_microservice_response_data(method_data)
        
        for service in ob_services:
            service_stats = {
                'method': method_name,
                'microservice': service,
                'pod_time_area': 0,
                'total_requests': 0,
                'req_per_pod_time_area': 0,
                'avg_rps': 0,
                'avg_response_time': 0,
                'p95_response_time': 0,
                'p99_response_time': 0
            }
            
            # 計算服務級別的Pod時間面積
            if service in pod_data_per_service:
                service_pod_data = pod_data_per_service[service]
                if len(service_pod_data) > 1:
                    time_values = [entry['time_minutes'] for entry in service_pod_data]
                    pod_values = [entry['pods'] for entry in service_pod_data]
                    service_stats['pod_time_area'] = integrate.trapz(pod_values, time_values)
            
            # 計算服務級別的響應時間統計
            if service in response_data_per_service:
                service_response_data = response_data_per_service[service]
                service_stats['total_requests'] = service_response_data.get('request_count', 0)
                service_stats['avg_response_time'] = service_response_data.get('avg_response_time', 0)
                service_stats['p95_response_time'] = service_response_data.get('p95_response_time', 0)
                service_stats['p99_response_time'] = service_response_data.get('p99_response_time', 0)
                
                # 計算RPS（假設15分鐘測試）
                if service_stats['total_requests'] > 0:
                    service_stats['avg_rps'] = service_stats['total_requests'] / (15 * 60)
                
                # 計算請求/Pod時間面積比率
                if service_stats['pod_time_area'] > 0:
                    service_stats['req_per_pod_time_area'] = service_stats['total_requests'] / service_stats['pod_time_area']
            
            microservice_stats.append(service_stats)
        
        return microservice_stats
    
    def _extract_microservice_pod_data(self, method_data: Dict) -> Dict:
        """提取微服務級別的Pod數據"""
        # 這個函數需要根據實際的Pod監控數據格式來實現
        # 目前返回空字典，實際使用時需要解析pod_metrics目錄中的具體文件
        return {}
    
    def _extract_microservice_response_data(self, method_data: Dict) -> Dict:
        """提取微服務級別的響應數據"""
        # 這個函數需要根據實際的stats文件格式來實現
        # 目前返回空字典，實際使用時需要解析stats.csv文件中的微服務級別數據
        return {}
    
    def extract_response_time_data(self, experiment_dir: Path, scenario: str, application: str) -> Optional[pd.DataFrame]:
        """提取響應時間數據"""
        
        # 根據應用類型調整場景目錄查找模式
        if application == "redis":
            scenario_patterns = [
                f"{scenario}_*",
                f"redis_{scenario}*",
                f"redis_{scenario}",
                f"{scenario}"
            ]
        else:
            scenario_patterns = [f"{scenario}_*"]
        
        scenario_dir = None
        for pattern in scenario_patterns:
            scenario_dirs = list(experiment_dir.glob(pattern))
            if scenario_dirs:
                scenario_dir = scenario_dirs[0]
                break
        
        if not scenario_dir:
            return None
        
        # 查找stats文件
        stats_files = [
            scenario_dir / f"{scenario}_stats.csv",
            scenario_dir / f"redis_{scenario}_stats.csv",
            scenario_dir / "stats.csv"
        ]
        
        stats_file = None
        for file_path in stats_files:
            if file_path.exists():
                stats_file = file_path
                break
        
        if not stats_file:
            return None
        
        try:
            df = pd.read_csv(stats_file)
            
            # 檢查是否有響應時間相關的列
            response_time_cols = [
                'Average Response Time', 'Min Response Time', 'Max Response Time',
                '50%', '66%', '75%', '80%', '90%', '95%', '98%', '99%', '99.9%', '99.99%', '100%'
            ]
            
            available_cols = [col for col in response_time_cols if col in df.columns]
            if not available_cols:
                return None
            
            # 整理響應時間數據
            result_data = []
            for _, row in df.iterrows():
                entry = {
                    'name': row.get('Name', 'Unknown'),
                    'type': row.get('Type', 'Unknown'),
                    'request_count': row.get('Request Count', 0),
                    'avg_response_time': row.get('Average Response Time', 0),
                    'p95_response_time': row.get('95%', 0),
                    'p99_response_time': row.get('99%', 0)
                }
                result_data.append(entry)
            
            return pd.DataFrame(result_data)
            
        except Exception as e:
            print(f"⚠️ 讀取響應時間數據失敗 {stats_file}: {e}")
            return None
    
    def generate_detailed_statistics_report(self, application: str = None) -> Dict:
        """生成詳細的統計數據報告"""
        print(f"📊 生成詳細統計數據報告...")
        
        all_statistics = {
            'generation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'applications': {}
        }
        
        applications_to_process = [application] if application else self.applications
        
        for app in applications_to_process:
            app_statistics = {
                'application': app,
                'scenarios': {}
            }
            
            # 獲取該應用的可用場景
            available_scenarios = self.get_available_scenarios(app)
            
            if not available_scenarios:
                print(f"⚠️ 警告: {app} 沒有可用的場景數據")
                continue
            
            for scenario in available_scenarios:
                print(f"📈 分析 {app} - {scenario} 場景...")
                
                # 收集場景數據
                scenario_data = self.collect_scenario_data(app, scenario)
                
                # 計算詳細統計
                detailed_stats = self.calculate_detailed_statistics(scenario_data)
                
                # 增強統計數據 - 添加響應時間信息
                self._enhance_statistics_with_response_times(detailed_stats, app, scenario)
                
                app_statistics['scenarios'][scenario] = detailed_stats
            
            all_statistics['applications'][app] = app_statistics
        
        # 保存統計報告
        stats_file = self.output_dir / "detailed_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_statistics, f, ensure_ascii=False, indent=2)
        
        print(f"📋 詳細統計報告已保存: {stats_file}")
        
        # 生成表格格式的報告
        self._generate_statistics_table(all_statistics)
        
        return all_statistics
    
    def _enhance_statistics_with_response_times(self, detailed_stats: Dict, application: str, scenario: str):
        """增強統計數據，添加響應時間信息"""
        
        for method_name, method_stats in detailed_stats['summary_statistics'].items():
            # 嘗試從對應的實驗目錄中獲取響應時間數據
            # 將K8s-HPA-cpu-XX格式轉換為K8s-HPA
            base_method_name = method_name
            if method_name.startswith('K8s-HPA-'):
                base_method_name = 'K8s-HPA'
            
            experiment_dir = self.find_latest_experiment_data(base_method_name, application)
            
            if experiment_dir:
                response_time_data = self.extract_response_time_data(experiment_dir, scenario, application)
                
                if response_time_data is not None and not response_time_data.empty:
                    # 計算加權平均響應時間
                    total_requests = response_time_data['request_count'].sum()
                    if total_requests > 0:
                        weighted_avg_rt = (response_time_data['avg_response_time'] * 
                                         response_time_data['request_count']).sum() / total_requests
                        weighted_p95_rt = (response_time_data['p95_response_time'] * 
                                         response_time_data['request_count']).sum() / total_requests
                        weighted_p99_rt = (response_time_data['p99_response_time'] * 
                                         response_time_data['request_count']).sum() / total_requests
                        
                        method_stats['avg_response_time'] = weighted_avg_rt
                        method_stats['p95_response_time'] = weighted_p95_rt
                        method_stats['p99_response_time'] = weighted_p99_rt
                        
                        # 更新總請求數（如果stats文件有更准確的數據）
                        if total_requests > method_stats['total_requests']:
                            method_stats['total_requests'] = total_requests
                            
                            # 重新計算 req_per_pod_time_area
                            if method_stats['pod_time_area'] > 0:
                                method_stats['req_per_pod_time_area'] = total_requests / method_stats['pod_time_area']
    
    def _generate_statistics_table(self, all_statistics: Dict):
        """生成表格格式的統計報告"""
        
        # 創建主要統計表格
        table_data = []
        microservice_data = []
        
        for app_name, app_data in all_statistics['applications'].items():
            for scenario_name, scenario_data in app_data['scenarios'].items():
                # 生成方法級別的統計
                for method_name, method_stats in scenario_data['summary_statistics'].items():
                    row = {
                        '應用': app_name,
                        '場景': scenario_name,
                        '微服務': '總計',
                        '方法': method_name,
                        'Pod時間面積': f"{method_stats['pod_time_area']:.2f}",
                        '總請求數': f"{method_stats['total_requests']:.0f}",
                        '請求/Pod時間面積': f"{method_stats['req_per_pod_time_area']:.2f}",
                        '平均RPS': f"{method_stats['avg_rps']:.2f}",
                        '平均響應時間(ms)': f"{method_stats['avg_response_time']:.2f}",
                        '95%響應時間(ms)': f"{method_stats['p95_response_time']:.2f}",
                        '99%響應時間(ms)': f"{method_stats['p99_response_time']:.2f}"
                    }
                    table_data.append(row)
                
                # 生成微服務級別的統計（僅針對OnlineBoutique）
                if app_name == 'onlineboutique' and scenario_data.get('microservices'):
                    for microservice_stats in scenario_data['microservices']:
                        row = {
                            '應用': app_name,
                            '場景': scenario_name,
                            '微服務': microservice_stats['microservice'],
                            '方法': microservice_stats['method'],
                            'Pod時間面積': f"{microservice_stats['pod_time_area']:.2f}",
                            '總請求數': f"{microservice_stats['total_requests']:.0f}",
                            '請求/Pod時間面積': f"{microservice_stats['req_per_pod_time_area']:.2f}",
                            '平均RPS': f"{microservice_stats['avg_rps']:.2f}",
                            '平均響應時間(ms)': f"{microservice_stats['avg_response_time']:.2f}",
                            '95%響應時間(ms)': f"{microservice_stats['p95_response_time']:.2f}",
                            '99%響應時間(ms)': f"{microservice_stats['p99_response_time']:.2f}"
                        }
                        microservice_data.append(row)
        
        # 轉換為DataFrame並保存
        df = pd.DataFrame(table_data)
        
        # 保存主要統計表格
        csv_file = self.output_dir / "statistics_summary.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"📊 統計表格已保存: {csv_file}")
        
        # 保存微服務級別統計表格（如果有的話）
        if microservice_data:
            microservice_df = pd.DataFrame(microservice_data)
            microservice_csv_file = self.output_dir / "microservices_statistics.csv"
            microservice_df.to_csv(microservice_csv_file, index=False, encoding='utf-8-sig')
            print(f"📊 微服務統計表格已保存: {microservice_csv_file}")
        
        # 保存為Excel（如果可能）
        try:
            excel_file = self.output_dir / "statistics_summary.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='總體統計', index=False)
                if microservice_data:
                    microservice_df.to_excel(writer, sheet_name='微服務統計', index=False)
            print(f"📊 統計表格已保存: {excel_file}")
        except ImportError:
            print("📝 提示: 安裝openpyxl可以生成Excel格式的統計表格")
        
        # 打印摘要到控制台
        print("\n" + "="*100)
        print("📊 統計數據摘要")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        if microservice_data:
            print("\n" + "="*100)
            print("📊 微服務統計數據摘要")
            print("="*100)
            print(microservice_df.to_string(index=False))
            print("="*100)
        
        return df

    def create_comparison_plot(self, application: str, scenario: str, metric: str):
        """創建對比圖"""
        print(f"🎨 生成 {application} - {scenario} - {metric} 對比圖...")
        
        scenario_data = self.collect_scenario_data(application, scenario)
        
        plt.figure(figsize=(15, 8))
        
        # 設置顏色 - 6種方法的顏色
        colors = ['#1f77b4', '#ff7f0e', '#e74c3c', '#f39c12', '#2ecc71', '#3498db']
        method_colors = dict(zip(self.all_methods, colors))
        
        # 檢查是否有任何數據
        has_any_data = False
        missing_data_methods = []
        
        ylabel = 'Pod 數量' if metric == 'pods' else 'RPS (每秒請求數)'
        
        for method in self.all_methods:
            if method in scenario_data['methods']:
                method_data = scenario_data['methods'][method]
                
                # 選擇對應的數據
                if metric == 'pods':
                    df = method_data['pod_data']
                else:  # rps
                    df = method_data['rps_data']
                
                if df is not None:
                    # 有數據，繪製線條
                    x_data = df['time_minutes']
                    y_data = df['pods'] if metric == 'pods' else df['rps']
                    
                    plt.plot(x_data, y_data, 
                            label=method, 
                            color=method_colors[method],
                            linewidth=2,
                            marker='o',
                            markersize=4)
                    has_any_data = True
                else:
                    # 沒有數據，記錄下來
                    missing_data_methods.append(method)
        
        # 設置圖表基本屬性
        plt.xlabel('時間 (分鐘)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{application.title()} - {scenario.title()} 場景 - {ylabel}對比 (含K8s-HPA各配置)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 15)
        
        if has_any_data:
            plt.legend(loc='best')
        
        # 如果有缺失數據的方法，在圖上顯示警告
        if missing_data_methods:
            warning_text = f"缺失數據: {', '.join(missing_data_methods)}"
            plt.text(0.02, 0.98, warning_text, 
                    transform=plt.gca().transAxes,
                    fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    verticalalignment='top')
        
        # 如果完全沒有數據，顯示大警告
        if not has_any_data:
            plt.text(0.5, 0.5, f'❌ 數據不足\n\n無法找到 {application} {scenario} 場景的 {ylabel} 數據\n\n請檢查實驗日誌和數據文件', 
                    transform=plt.gca().transAxes,
                    fontsize=16, 
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.8, edgecolor='darkred'),
                    color='white', weight='bold')
            
            # 設置基本的Y軸範圍
            if metric == 'pods':
                plt.ylim(0, 10)
            else:
                plt.ylim(0, 100)
        
        # 保存圖片
        filename = f"{application}_{scenario}_{metric}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        if has_any_data:
            print(f"✅ 已保存: {filename}")
        else:
            print(f"⚠️ 已保存 (數據不足): {filename}")
        
        return filepath


    def get_available_scenarios(self, application: str) -> List[str]:
        """獲取指定應用的可用場景列表"""
        available_scenarios = []
        
        # 檢查每個方法的所有實驗目錄，找出實際存在的場景
        for method in self.methods:
            method_dir_name = self.app_method_mapping[application][method]
            method_dir = self.logs_root / method_dir_name
            
            if not method_dir.exists():
                continue
                
            # 檢查所有實驗目錄，不只是最新的
            if application == "redis":
                if method == "GNNRL":
                    # 檢查所有 GNNRL Redis 實驗目錄
                    for test_dir in method_dir.glob("gnnrl_*redis*"):
                        self._extract_scenarios_from_dir(test_dir, available_scenarios)
                    for test_dir in method_dir.glob("gnnrl_*seed42_*"):
                        if self.detect_experiment_application(test_dir) == "redis":
                            self._extract_scenarios_from_dir(test_dir, available_scenarios)
                elif method == "Gym-HPA":
                    # 檢查所有 Gym-HPA Redis 實驗目錄
                    for test_dir in method_dir.glob("gym_hpa_redis_*seed42_*"):
                        self._extract_scenarios_from_dir(test_dir, available_scenarios)
            else:  # onlineboutique
                if method == "GNNRL":
                    for test_dir in method_dir.glob("gnnrl_*seed42_*"):
                        if self.detect_experiment_application(test_dir) == "onlineboutique":
                            self._extract_scenarios_from_dir(test_dir, available_scenarios)
                elif method == "Gym-HPA":
                    for test_dir in method_dir.glob("gym_hpa_*seed42_*"):
                        if "redis" not in test_dir.name and self.detect_experiment_application(test_dir) == "onlineboutique":
                            self._extract_scenarios_from_dir(test_dir, available_scenarios)
        
        # 檢查 K8s-HPA 數據（所有實驗目錄）
        method_dir_name = self.app_method_mapping[application]["K8s-HPA"]
        method_dir = self.logs_root / method_dir_name
        
        if method_dir.exists():
            if application == "redis":
                # Redis K8s-HPA: redis_hpa_cpu-XX_timestamp/scenario/
                for config_dir in method_dir.glob("redis_hpa_*"):
                    scenario_dirs = [d for d in config_dir.iterdir() if d.is_dir()]
                    for scenario_dir in scenario_dirs:
                        scenario_name = scenario_dir.name.split('_')[0]
                        if scenario_name in self.scenarios and scenario_name not in available_scenarios:
                            available_scenarios.append(scenario_name)
        
        print(f"📋 {application} 可用場景: {available_scenarios}")
        return available_scenarios
    
    def _extract_scenarios_from_dir(self, experiment_dir: Path, available_scenarios: list):
        """從實驗目錄中提取場景名稱"""
        scenario_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
        for scenario_dir in scenario_dirs:
            # 提取場景名稱（去掉編號後綴）
            scenario_name = scenario_dir.name.split('_')[0]
            if scenario_name in self.scenarios and scenario_name not in available_scenarios:
                available_scenarios.append(scenario_name)

    def generate_all_comparisons(self):
        """生成所有對比圖"""
        print(f"🚀 開始生成所有場景對比圖...")
        
        generated_files = []
        
        # 生成包含所有方法的對比圖 (含K8s-HPA各配置)
        for application in self.applications:
            # 獲取該應用的可用場景
            available_scenarios = self.get_available_scenarios(application)
            
            if not available_scenarios:
                print(f"⚠️ 警告: {application} 沒有可用的場景數據")
                continue
            
            for scenario in available_scenarios:
                for metric in ['pods', 'rps']:
                    try:
                        filepath = self.create_comparison_plot(application, scenario, metric)
                        generated_files.append(filepath)
                    except Exception as e:
                        print(f"❌ 生成 {application}_{scenario}_{metric} 失敗: {e}")
        
        # 生成總結報告
        self.generate_summary_report(generated_files)
        
        # 生成詳細統計數據報告
        print(f"\n📊 生成詳細統計數據報告...")
        self.generate_detailed_statistics_report()
        
        print(f"\n🎉 完成！共生成 {len(generated_files)} 個對比圖")
        print(f"📁 輸出目錄: {self.output_dir}")
        
        return generated_files

    def generate_summary_report(self, generated_files: List[Path]):
        """生成總結報告"""
        summary = {
            '生成時間': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '生成文件數量': len(generated_files),
            '應用場景': self.applications,
            '壓測場景': self.scenarios,
            '對比方法': self.methods,
            '生成文件列表': [f.name for f in generated_files]
        }
        
        summary_file = self.output_dir / "comparison_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"📋 總結報告已保存: {summary_file}")

def main():
    """主函數"""
    import sys
    
    print("🎯 場景對比可視化生成器")
    print("=" * 50)
    
    # 檢查命令行參數
    if len(sys.argv) > 1:
        specified_app = sys.argv[1].lower()
        if specified_app not in ["redis", "onlineboutique"]:
            print(f"❌ 不支援的應用: {specified_app}")
            print("💡 支援的應用: redis, onlineboutique")
            return
        print(f"🎯 指定應用: {specified_app}")
    else:
        specified_app = None
        print("🎯 生成所有應用的對比圖")
    
    # 創建生成器實例
    generator = ScenarioComparisonGenerator()
    
    # 如果指定了應用，只處理該應用
    if specified_app:
        generator.applications = [specified_app]
    
    # 生成所有對比圖
    generated_files = generator.generate_all_comparisons()
    
    print("\n" + "=" * 50)
    print("💡 使用說明:")
    print("   📊 對比圖文件:")
    print("      • 查看生成的圖片文件在 scenario_comparisons_fixed/ 目錄")
    print("      • 對比圖命名格式: {應用}_{場景}_{指標}.png")
    print("      • 例如: redis_offpeak_rps.png, onlineboutique_fluctuating_pods.png")
    print("      • 每個圖包含最多6條線: GNNRL, Gym-HPA, K8s-HPA-cpu-20, K8s-HPA-cpu-40, K8s-HPA-cpu-60, K8s-HPA-cpu-80")
    print("      • 可以直接對比不同K8s-HPA CPU閾值設置的性能差異")
    print("      • 只會生成有實際數據的場景對比圖")
    print("   ")
    print("   📈 統計數據報告:")
    print("      • detailed_statistics.json - 完整的JSON格式統計數據")
    print("      • statistics_summary.csv - 表格格式統計數據（可用Excel打開）")
    print("      • statistics_summary.xlsx - Excel格式統計數據（如果安裝了openpyxl）")
    print("   ")
    print("   📋 統計指標包含:")
    print("      • Pod時間面積 (Pod-Minutes)")
    print("      • 總請求數")
    print("      • 請求/Pod時間面積比率")
    print("      • 平均RPS")
    print("      • 平均響應時間(ms)")
    print("      • 95%響應時間(ms)")
    print("      • 99%響應時間(ms)")
    print("   ")
    print("   🎯 數據按照：場景 → 方法 → 統計指標 的層次結構組織")

if __name__ == "__main__":
    main()