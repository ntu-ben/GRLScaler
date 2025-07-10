#!/usr/bin/env python3
"""
實驗結果可視化工具
==================

為GRLScaler自動擴展實驗生成時間序列圖表：
1. RPS表現比較 (GNNRL vs Gym-HPA vs K8s-HPA vs 原始壓測設定)
2. Pod數量比較 (包含理論最佳值)

使用方式：
    # 分析單一實驗批次
    python experiment_visualization.py --experiment-dir logs/gnnrl/gnnrl_redis_train_seed42_20250706_190527
    
    # 比較多個方法
    python experiment_visualization.py --compare \
        --gnnrl logs/gnnrl/gnnrl_redis_train_seed42_20250706_190527 \
        --gym-hpa logs/gym-hpa/gym_hpa_redis_train_seed42_20250706_122635 \
        --k8s-hpa logs/k8s_hpa_redis/redis_hpa_cpu-40_20250706_125639
        
    # 生成所有可用實驗的比較圖
    python experiment_visualization.py --auto-compare --environment redis
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 設定圖表樣式
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')
    
# 設定中文字體支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ExperimentDataExtractor:
    """實驗數據提取器"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        
    def extract_locust_rps_data(self, stats_history_file: Path) -> pd.DataFrame:
        """從Locust stats_history.csv提取RPS時間序列數據"""
        if not stats_history_file.exists():
            return pd.DataFrame()
            
        df = pd.read_csv(stats_history_file)
        
        # 過濾出聚合數據行 (Type為NaN、空字符串或"Aggregated")
        aggregated_df = df[(df['Type'].isna()) | (df['Type'] == '') | (df['Type'] == 'Aggregated')]
        
        if aggregated_df.empty:
            return pd.DataFrame()
            
        # 轉換時間戳
        aggregated_df = aggregated_df.copy()
        aggregated_df['DateTime'] = pd.to_datetime(aggregated_df['Timestamp'], unit='s')
        
        # 提取RPS數據
        result_df = aggregated_df[['DateTime', 'Requests/s', 'User Count']].copy()
        result_df.columns = ['DateTime', 'RPS', 'UserCount']
        
        return result_df.sort_values('DateTime')
    
    def extract_pod_count_from_kiali(self, kiali_file: Path) -> Optional[Dict]:
        """從Kiali文件提取Pod數量信息"""
        if not kiali_file.exists():
            return None
            
        try:
            with open(kiali_file) as f:
                data = json.load(f)
            
            nodes = data.get('elements', {}).get('nodes', [])
            pod_counts = {}
            
            for node in nodes:
                node_data = node.get('data', {})
                workload = node_data.get('workload', '')
                namespace = node_data.get('namespace', '')
                
                if workload and namespace:
                    # 嘗試從節點數據中提取Pod數量
                    # Kiali可能在traffic或其他字段中包含這些信息
                    traffic = node_data.get('traffic', [])
                    if traffic:
                        # 這裡可能需要根據實際Kiali數據結構調整
                        pod_counts[f"{namespace}:{workload}"] = len(traffic)
                    else:
                        pod_counts[f"{namespace}:{workload}"] = 1
            
            return {
                'timestamp': data.get('timestamp'),
                'pod_counts': pod_counts
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"警告: 無法解析Kiali文件 {kiali_file}: {e}")
            return None
    
    def get_baseline_rps(self, scenario: str, environment: str = 'redis') -> float:
        """獲取原始壓測設定的基準RPS (根據loadtest實際配置)"""
        if environment == 'onlineboutique':
            # OnlineBoutique 環境的基準RPS (根據實際loadtest配置)
            baseline_rps = {
                'offpeak': 50.0,      # locust_offpeak.py: 50 users → ~50 RPS
                'peak': 300.0,        # locust_peak.py: 300 users → ~300 RPS  
                'rushsale': 500.0,    # locust_rushsale.py: 峰值800 users，平均~500 RPS
                'fluctuating': 275.0, # locust_fluctuating.py: [50,300,50,800] 平均275 RPS
            }
        else:
            # Redis 環境的基準RPS (根據實際loadtest配置)
            baseline_rps = {
                'offpeak': 75.0,      # redis_offpeak: 10-30 users, 實測50-100 RPS，平均75
                'peak': 650.0,        # redis_peak: 100-200 users, 實測500-800 RPS，平均650
                'rushsale': 500.0,    # 搶購模式
                'fluctuating': 350.0, # 波動模式平均
                'redis_offpeak': 75.0,
                'redis_peak': 650.0
            }
        return baseline_rps.get(scenario, 100.0)
    
    def calculate_theoretical_optimal_pods(self, rps: float, scenario: str, environment: str = 'redis') -> int:
        """計算理論最佳Pod數量"""
        if environment == 'onlineboutique':
            # OnlineBoutique環境的理論計算
            # 基於實際微服務架構，考慮10個微服務的總和
            # 假設前端服務能處理約30 RPS，其他服務能處理約20-40 RPS
            if rps <= 30:
                # 低負載時，大部分服務保持最小副本數
                return 12  # 10個微服務各1個 + 2個前端/關鍵服務多1個
            elif rps <= 80:
                # 中等負載時，關鍵服務擴展
                return 18  # 前端、購物車、產品等關鍵服務擴展
            elif rps <= 150:
                # 高負載時，多數服務需要擴展
                return 25  # 大部分服務擴展到2-3個副本
            else:
                # 極高負載時，所有服務都需要擴展
                return int(np.ceil(rps / 8))  # 假設平均每8 RPS需要1個Pod
        else:
            # Redis環境的理論計算
            if 'master' in scenario:
                # Redis Master: 通常不橫向擴展，但考慮高可用性
                return min(3, max(1, int(np.ceil(rps / 200))))
            else:
                # Redis Slave: 可以橫向擴展
                return max(1, int(np.ceil(rps / 100)))


class ExperimentVisualizer:
    """實驗結果可視化器"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("logs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = ExperimentDataExtractor(Path("."))
    
    def plot_rps_comparison(self, experiment_data: Dict[str, pd.DataFrame], 
                           scenario: str, environment: str = 'redis', save_path: Path = None) -> None:
        """繪製RPS比較圖表"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 獲取基準RPS（更新後的穩定數值）
        baseline_rps = self.extractor.get_baseline_rps(scenario, environment)
        
        # 設定顏色和標記（加入穩定模式識別）
        colors = {'GNNRL': '#2E86AB', 'Gym-HPA': '#A23B72', 'K8s-HPA': '#F18F01', 'Baseline': '#C73E1D', 'Stable-Baseline': '#27AE60'}
        linestyles = {'GNNRL': '-', 'Gym-HPA': '--', 'K8s-HPA': '-.', 'Baseline': ':', 'Stable-Baseline': '--'}
        
        # 標準化時間軸為0-15分鐘
        max_duration_minutes = 15
        
        # 繪製基準線 (0-15分鐘) - 區分穩定模式和原始模式
        baseline_times = list(range(max_duration_minutes + 1))
        baseline_data = [baseline_rps] * len(baseline_times)
        
        # 檢測是否有穩定測試數據
        has_stable_data = any('stable' in str(df.columns).lower() if not df.empty else False for df in experiment_data.values())
        
        if has_stable_data:
            # 使用穩定基準線
            ax.plot(baseline_times, baseline_data, 
                   color=colors['Stable-Baseline'], linestyle=linestyles['Stable-Baseline'], 
                   linewidth=2, alpha=0.8, label='穩定壓測基準 (有RPS限制)')
        else:
            # 使用原始基準線
            ax.plot(baseline_times, baseline_data, 
                   color=colors['Baseline'], linestyle=linestyles['Baseline'], 
                   linewidth=2, alpha=0.8, label='原始壓測設定')
        
        # 繪製各方法的RPS數據
        for method, df in experiment_data.items():
            if not df.empty:
                # 將時間轉換為從0開始的分鐘數
                df_copy = df.copy()
                start_time = df_copy['DateTime'].min()
                df_copy['Minutes'] = (df_copy['DateTime'] - start_time).dt.total_seconds() / 60
                
                # 只顯示前15分鐘的數據
                df_filtered = df_copy[df_copy['Minutes'] <= max_duration_minutes]
                
                if not df_filtered.empty:
                    # 平滑化數據以減少噪聲
                    window_size = min(5, len(df_filtered))
                    if window_size > 1:
                        smoothed_rps = df_filtered['RPS'].rolling(window=window_size, center=True).mean()
                    else:
                        smoothed_rps = df_filtered['RPS']
                    
                    ax.plot(df_filtered['Minutes'], smoothed_rps, 
                           color=colors.get(method, '#333333'), 
                           linestyle=linestyles.get(method, '-'),
                           linewidth=2.5, alpha=0.9, label=method, marker='o', markersize=3)
        
        # 設定圖表樣式
        ax.set_xlabel('時間 (分鐘)', fontsize=12, fontweight='bold')
        ax.set_ylabel('每秒請求數 (RPS)', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario.title()} 場景 - RPS 表現比較 ({environment.title()})', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 設定時間軸範圍
        ax.set_xlim(0, max_duration_minutes)
        ax.set_xticks(range(0, max_duration_minutes + 1, 3))
        
        # 添加網格和圖例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # 調整佈局
        plt.tight_layout()
        
        # 保存圖表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ RPS比較圖已保存: {save_path}")
        else:
            # 如果沒有指定保存路徑，自動生成一個
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_path = self.output_dir / f"rps_comparison_{scenario}_{timestamp}.png"
            plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
            print(f"✅ RPS比較圖已保存: {auto_save_path}")
        
        plt.close()  # 關閉圖表以釋放內存
    
    def plot_pod_count_comparison(self, experiment_data: Dict[str, List], 
                                 scenario: str, environment: str = 'redis', 
                                 save_path: Path = None) -> None:
        """繪製Pod數量比較圖表"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 設定顏色
        colors = {'GNNRL': '#2E86AB', 'Gym-HPA': '#A23B72', 'K8s-HPA': '#F18F01', 'Theoretical': '#27AE60'}
        
        # 計算理論最佳值
        baseline_rps = self.extractor.get_baseline_rps(scenario, environment)
        theoretical_pods = self.extractor.calculate_theoretical_optimal_pods(
            baseline_rps, scenario, environment)
        
        # 繪製理論最佳值線
        if experiment_data:
            # 使用第一個實驗的時間範圍（模擬）
            time_points = list(range(0, 15))  # 假設15分鐘的實驗
            theoretical_data = [theoretical_pods] * len(time_points)
            ax.plot(time_points, theoretical_data, 
                   color=colors['Theoretical'], linestyle=':', 
                   linewidth=3, alpha=0.8, label='理論最佳值')
        
        # 繪製各方法的Pod數量（這裡需要實際的Pod監控數據）
        for method, pod_data in experiment_data.items():
            if pod_data:
                # 這裡應該是實際的Pod數量時間序列數據
                # 目前創建模擬數據作為示例
                time_points = list(range(len(pod_data)))
                ax.plot(time_points, pod_data, 
                       color=colors.get(method, '#333333'), 
                       linewidth=2.5, alpha=0.9, label=method, 
                       marker='s', markersize=4)
        
        # 設定圖表樣式
        ax.set_xlabel('時間 (分鐘)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pod 數量', fontsize=12, fontweight='bold')
        ax.set_title(f'{scenario.title()} 場景 - Pod 數量比較', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 設定Y軸為整數
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 添加網格和圖例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        # 調整佈局
        plt.tight_layout()
        
        # 保存圖表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Pod數量比較圖已保存: {save_path}")
        else:
            # 如果沒有指定保存路徑，自動生成一個
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_path = self.output_dir / f"pod_comparison_{scenario}_{timestamp}.png"
            plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Pod數量比較圖已保存: {auto_save_path}")
        
        plt.close()  # 關閉圖表以釋放內存
    
    def analyze_experiment_directory(self, experiment_dir: Path) -> Dict[str, pd.DataFrame]:
        """分析實驗目錄，提取所有場景的數據"""
        results = {}
        
        if not experiment_dir.exists():
            print(f"❌ 實驗目錄不存在: {experiment_dir}")
            return results
        
        def find_scenario_data(directory: Path, depth: int = 0) -> Dict[str, pd.DataFrame]:
            """遞歸查找場景數據"""
            local_results = {}
            
            if depth > 3:  # 限制搜索深度
                return local_results
            
            for item in directory.iterdir():
                if not item.is_dir():
                    continue
                
                # 檢查當前目錄是否包含場景數據
                stats_files = list(item.glob("*_stats_history.csv"))
                if stats_files:
                    # 從目錄名或文件名提取場景名稱
                    scenario_name = None
                    
                    # 嘗試從目錄名提取
                    for part in item.name.split('_'):
                        if part.lower() in ['peak', 'offpeak', 'rushsale', 'fluctuating']:
                            scenario_name = part.lower()
                            break
                    
                    # 嘗試從文件名提取
                    if not scenario_name:
                        for stats_file in stats_files:
                            for part in stats_file.stem.split('_'):
                                if part.lower() in ['peak', 'offpeak', 'rushsale', 'fluctuating']:
                                    scenario_name = part.lower()
                                    break
                            if scenario_name:
                                break
                    
                    if scenario_name:
                        stats_file = stats_files[0]
                        rps_data = self.extractor.extract_locust_rps_data(stats_file)
                        if not rps_data.empty:
                            # 如果已經有這個場景的數據，選擇更新的
                            if scenario_name not in local_results or len(rps_data) > len(local_results[scenario_name]):
                                local_results[scenario_name] = rps_data
                                print(f"  ✅ 找到 {scenario_name} 場景數據: {stats_file}")
                else:
                    # 遞歸搜索子目錄
                    sub_results = find_scenario_data(item, depth + 1)
                    # 合併結果，優先保留更完整的數據
                    for scenario, data in sub_results.items():
                        if scenario not in local_results or len(data) > len(local_results[scenario]):
                            local_results[scenario] = data
            
            return local_results
        
        results = find_scenario_data(experiment_dir)
        
        if results:
            print(f"  📊 成功提取 {len(results)} 個場景的數據")
        else:
            print(f"  ⚠️  未找到有效的場景數據")
        
        return results
    
    def auto_compare_experiments(self, environment: str = 'redis') -> None:
        """自動比較所有可用的實驗"""
        base_logs_dir = Path("logs")
        
        # 查找各種方法的最新實驗
        if environment == 'onlineboutique':
            experiment_paths = {
                'GNNRL': self._find_latest_experiment(base_logs_dir / "gnnrl", environment),
                'Gym-HPA': self._find_latest_experiment(base_logs_dir / "gym-hpa", environment),
                'K8s-HPA': self._find_latest_experiment(base_logs_dir / "k8s-hpa", environment)
            }
        else:
            experiment_paths = {
                'GNNRL': self._find_latest_experiment(base_logs_dir / "gnnrl", environment),
                'Gym-HPA': self._find_latest_experiment(base_logs_dir / "gym-hpa", environment),
                'K8s-HPA': self._find_latest_experiment(base_logs_dir / f"k8s_hpa_{environment}", environment)
            }
        
        # 移除未找到的實驗
        experiment_paths = {k: v for k, v in experiment_paths.items() if v}
        
        if not experiment_paths:
            print(f"❌ 未找到 {environment} 環境的實驗數據")
            return
        
        print(f"🔍 找到以下實驗進行比較:")
        for method, path in experiment_paths.items():
            print(f"   {method}: {path}")
        
        # 提取所有實驗數據
        all_scenarios_data = {}
        for method, exp_path in experiment_paths.items():
            scenarios = self.analyze_experiment_directory(exp_path)
            for scenario, data in scenarios.items():
                if scenario not in all_scenarios_data:
                    all_scenarios_data[scenario] = {}
                all_scenarios_data[scenario][method] = data
        
        # 為每個場景生成比較圖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for scenario, methods_data in all_scenarios_data.items():
            print(f"\n📊 生成 {scenario} 場景的比較圖表...")
            
            # RPS比較圖
            rps_save_path = self.output_dir / f"{environment}_{scenario}_rps_comparison_{timestamp}.png"
            self.plot_rps_comparison(methods_data, scenario, rps_save_path)
            
            # Pod數量比較圖（基於RPS數據模擬）
            pod_data = {}
            for method, rps_data in methods_data.items():
                # 基於RPS數據模擬Pod數量變化
                pod_counts = self._simulate_pod_scaling(rps_data, method, scenario, environment)
                if pod_counts:
                    pod_data[method] = pod_counts
            
            pod_save_path = self.output_dir / f"{environment}_{scenario}_pods_comparison_{timestamp}.png"
            self.plot_pod_count_comparison(pod_data, scenario, environment, pod_save_path)
    
    def _find_latest_experiment(self, method_dir: Path, environment: str) -> Optional[Path]:
        """查找指定方法和環境的最新實驗"""
        if not method_dir.exists():
            return None
        
        # 根據環境調整搜索策略
        if environment == 'onlineboutique':
            # OnlineBoutique 的命名模式
            search_patterns = ['online_boutique', 'onlineboutique', 'boutique']
            experiment_dirs = []
            for pattern in search_patterns:
                dirs = [d for d in method_dir.iterdir() 
                       if d.is_dir() and pattern in d.name.lower()]
                experiment_dirs.extend(dirs)
            
            # 如果沒找到OnlineBoutique特定實驗，查找有Locust數據的通用實驗
            if not experiment_dirs:
                experiment_dirs = [d for d in method_dir.iterdir() 
                                  if d.is_dir() and self._has_locust_data(d)]
        else:
            # Redis 的命名模式
            experiment_dirs = [d for d in method_dir.iterdir() 
                              if d.is_dir() and environment in d.name.lower()]
        
        # 過濾出有實際Locust數據的目錄
        valid_experiment_dirs = [d for d in experiment_dirs if self._has_locust_data(d)]
        
        if not valid_experiment_dirs:
            return None
        
        # 返回最新的實驗目錄（按名稱排序，通常包含時間戳）
        return sorted(valid_experiment_dirs)[-1]
    
    def _has_locust_data(self, experiment_dir: Path) -> bool:
        """檢查實驗目錄是否包含Locust數據"""
        if not experiment_dir.is_dir():
            return False
        
        # 查找場景子目錄中的stats_history.csv文件（支持多層目錄）
        def search_stats_files(directory: Path, depth: int = 0) -> bool:
            if depth > 3:  # 限制搜索深度
                return False
            
            for item in directory.iterdir():
                if item.is_dir():
                    # 檢查當前目錄
                    stats_files = list(item.glob("*_stats_history.csv"))
                    if stats_files:
                        return True
                    # 遞歸搜索子目錄
                    if search_stats_files(item, depth + 1):
                        return True
            return False
        
        return search_stats_files(experiment_dir)
    
    def _simulate_pod_scaling(self, rps_data: pd.DataFrame, method: str, scenario: str, environment: str) -> List[int]:
        """基於RPS數據模擬Pod擴縮容行為"""
        if rps_data.empty:
            return []
        
        # 基礎配置
        if environment == 'onlineboutique':
            min_pods = 10  # OnlineBoutique 最少10個微服務
            max_pods = 50  # 最大Pod數
        else:
            min_pods = 2   # Redis 最少2個Pod (master + slave)
            max_pods = 20  # 最大Pod數
        
        pod_counts = []
        current_pods = min_pods
        
        # 不同方法的擴縮容特性
        scaling_configs = {
            'GNNRL': {
                'aggressive': 0.8,    # 積極度
                'smoothing': 0.3,     # 平滑度
                'threshold_up': 0.7,  # 擴容閾值
                'threshold_down': 0.3 # 縮容閾值
            },
            'Gym-HPA': {
                'aggressive': 0.6,
                'smoothing': 0.4,
                'threshold_up': 0.75,
                'threshold_down': 0.25
            },
            'K8s-HPA': {
                'aggressive': 0.4,    # HPA相對保守
                'smoothing': 0.6,     # 更平滑
                'threshold_up': 0.8,  # CPU 80%才擴容
                'threshold_down': 0.2
            }
        }
        
        config = scaling_configs.get(method, scaling_configs['K8s-HPA'])
        baseline_rps = self.extractor.get_baseline_rps(scenario, environment)
        
        for _, row in rps_data.iterrows():
            current_rps = row['RPS']
            
            # 計算負載比例
            load_ratio = current_rps / baseline_rps if baseline_rps > 0 else 0
            
            # 根據負載比例決定目標Pod數
            if environment == 'onlineboutique':
                # OnlineBoutique 複雜計算
                if load_ratio <= 0.5:
                    target_pods = min_pods
                elif load_ratio <= 1.0:
                    target_pods = min_pods + int((load_ratio - 0.5) * 20 * config['aggressive'])
                else:
                    target_pods = min_pods + int(load_ratio * 15 * config['aggressive'])
            else:
                # Redis 簡單計算
                target_pods = min_pods + int(load_ratio * 8 * config['aggressive'])
            
            target_pods = max(min_pods, min(max_pods, target_pods))
            
            # 應用平滑化
            if abs(target_pods - current_pods) > 1:
                if target_pods > current_pods and load_ratio > config['threshold_up']:
                    # 擴容
                    change = max(1, int((target_pods - current_pods) * (1 - config['smoothing'])))
                    current_pods = min(target_pods, current_pods + change)
                elif target_pods < current_pods and load_ratio < config['threshold_down']:
                    # 縮容
                    change = max(1, int((current_pods - target_pods) * (1 - config['smoothing'])))
                    current_pods = max(target_pods, current_pods - change)
            
            pod_counts.append(current_pods)
        
        # 采樣以減少數據點（每分鐘一個點）
        if len(pod_counts) > 15:
            step = len(pod_counts) // 15
            pod_counts = pod_counts[::step][:15]
        
        return pod_counts


def main():
    parser = argparse.ArgumentParser(
        description='GRLScaler 實驗結果可視化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--experiment-dir', type=Path,
                       help='單一實驗目錄路徑')
    
    parser.add_argument('--compare', action='store_true',
                       help='比較多個實驗')
    
    parser.add_argument('--gnnrl', type=Path,
                       help='GNNRL 實驗目錄')
    parser.add_argument('--gym-hpa', type=Path,
                       help='Gym-HPA 實驗目錄')
    parser.add_argument('--k8s-hpa', type=Path,
                       help='K8s-HPA 實驗目錄')
    
    parser.add_argument('--auto-compare', action='store_true',
                       help='自動比較所有可用實驗')
    
    parser.add_argument('--environment', default='redis',
                       choices=['redis', 'onlineboutique'],
                       help='實驗環境')
    
    parser.add_argument('--output-dir', type=Path,
                       help='輸出目錄 (預設: logs/visualizations)')
    
    args = parser.parse_args()
    
    # 創建可視化器
    visualizer = ExperimentVisualizer(args.output_dir)
    
    if args.auto_compare:
        print(f"🚀 自動比較 {args.environment} 環境的所有實驗...")
        visualizer.auto_compare_experiments(args.environment)
        
    elif args.compare:
        # 手動比較指定實驗
        experiment_paths = {}
        if args.gnnrl:
            experiment_paths['GNNRL'] = args.gnnrl
        if args.gym_hpa:
            experiment_paths['Gym-HPA'] = args.gym_hpa
        if args.k8s_hpa:
            experiment_paths['K8s-HPA'] = args.k8s_hpa
        
        if not experiment_paths:
            print("❌ 請指定至少一個實驗目錄進行比較")
            return
        
        print("🔍 比較指定的實驗...")
        for method, path in experiment_paths.items():
            print(f"   {method}: {path}")
        
        # 檢測環境類型
        environment = 'onlineboutique' if any('boutique' in str(path) for path in experiment_paths.values()) else 'redis'
        print(f"🌍 檢測到環境: {environment}")
        
        # 提取所有實驗數據
        all_scenarios_data = {}
        for method, exp_path in experiment_paths.items():
            scenarios = visualizer.analyze_experiment_directory(exp_path)
            for scenario, data in scenarios.items():
                if scenario not in all_scenarios_data:
                    all_scenarios_data[scenario] = {}
                all_scenarios_data[scenario][method] = data
        
        # 為每個場景生成比較圖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for scenario, methods_data in all_scenarios_data.items():
            print(f"\n📊 生成 {scenario} 場景的比較圖表...")
            
            # RPS比較圖
            rps_save_path = visualizer.output_dir / f"manual_{environment}_{scenario}_rps_comparison_{timestamp}.png"
            visualizer.plot_rps_comparison(methods_data, scenario, environment, rps_save_path)
            
            # Pod數量比較圖（基於RPS數據模擬）
            pod_data = {}
            for method, rps_data in methods_data.items():
                pod_counts = visualizer._simulate_pod_scaling(rps_data, method, scenario, environment)
                if pod_counts:
                    pod_data[method] = pod_counts
            
            pod_save_path = visualizer.output_dir / f"manual_{environment}_{scenario}_pods_comparison_{timestamp}.png"
            visualizer.plot_pod_count_comparison(pod_data, scenario, environment, pod_save_path)
        
    elif args.experiment_dir:
        # 分析單一實驗
        print(f"📊 分析實驗: {args.experiment_dir}")
        scenarios_data = visualizer.analyze_experiment_directory(args.experiment_dir)
        
        if scenarios_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for scenario, data in scenarios_data.items():
                # 檢測環境類型
                environment = 'onlineboutique' if 'boutique' in str(args.experiment_dir) else 'redis'
                save_path = visualizer.output_dir / f"single_{scenario}_analysis_{timestamp}.png"
                visualizer.plot_rps_comparison({args.experiment_dir.name: data}, scenario, environment, save_path)
        else:
            print("❌ 未找到有效的實驗數據")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()