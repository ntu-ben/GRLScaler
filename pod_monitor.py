#!/usr/bin/env python3
"""
Pod 監控模組
===========

專門用於記錄 Kubernetes Pod 數量變化的監控系統
每15秒記錄一次指定 namespace 的 Pod 總數量
"""

import csv
import json
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging


class PodMonitor:
    """Pod 數量監控器"""
    
    def __init__(self, namespace: str, experiment_type: str, scenario: str, output_dir: Path):
        self.namespace = namespace
        self.experiment_type = experiment_type
        self.scenario = scenario
        self.output_dir = output_dir
        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(f'PodMonitor-{namespace}')
        
        # 確保輸出目錄存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設置CSV文件路徑
        self.csv_file = self.output_dir / f"{scenario}_pod_counts.csv"
        
    def start_monitoring(self, duration_minutes: int = 15):
        """開始監控Pod數量變化
        
        Args:
            duration_minutes: 監控持續時間（分鐘）
        """
        if self.monitoring:
            self.logger.warning("⚠️ Pod監控已在運行中")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(duration_minutes,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"🔄 開始監控 {self.namespace} namespace 的 Pod 數量 ({duration_minutes}分鐘)")
        
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info(f"⏹️ {self.namespace} Pod監控已停止")
        
    def _monitor_loop(self, duration_minutes: int):
        """監控循環（每15秒記錄一次）"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_log_minute = -1
        
        # 初始化CSV文件
        self._init_csv_file()
        
        while self.monitoring and time.time() < end_time:
            try:
                # 獲取Pod數量
                pod_count = self._get_pod_count()
                current_time = time.time()
                elapsed_minutes = (current_time - start_time) / 60
                
                # 每分鐘顯示一次當前Pod數量
                current_minute = int(elapsed_minutes)
                if current_minute != last_log_minute and current_minute >= 0:
                    print(f"📊 [{self.namespace}] 第 {current_minute} 分鐘 - 當前 Pod 數量: {pod_count}")
                    last_log_minute = current_minute
                
                # 記錄到CSV
                self._record_to_csv(elapsed_minutes, pod_count)
                
                self.logger.debug(f"📊 {self.namespace} Pod數量: {pod_count} (第{elapsed_minutes:.1f}分鐘)")
                
                # 等待15秒
                time.sleep(15)
                
            except Exception as e:
                self.logger.error(f"❌ Pod監控錯誤: {e}")
                time.sleep(5)  # 錯誤時短暫等待後繼續
                
        self.monitoring = False
        self.logger.info(f"✅ {self.namespace} Pod監控完成，數據已保存到 {self.csv_file}")
        
    def _get_pod_count(self) -> int:
        """獲取指定namespace的Pod總數量"""
        try:
            # 使用kubectl獲取Pod列表
            result = subprocess.run([
                'kubectl', 'get', 'pods', '-n', self.namespace,
                '--no-headers', '--field-selector=status.phase=Running'
            ], capture_output=True, text=True, check=True)
            
            # 計算運行中的Pod數量
            pod_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return len(pod_lines)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ kubectl命令失敗: {e}")
            return 0
        except Exception as e:
            self.logger.error(f"❌ 獲取Pod數量失敗: {e}")
            return 0
            
    def _init_csv_file(self):
        """初始化CSV文件"""
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp',
                'elapsed_minutes', 
                'pod_count',
                'namespace',
                'experiment_type',
                'scenario'
            ])
            
    def _record_to_csv(self, elapsed_minutes: float, pod_count: int):
        """記錄數據到CSV文件"""
        timestamp = datetime.now().isoformat()
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                round(elapsed_minutes, 2),
                pod_count,
                self.namespace,
                self.experiment_type,
                self.scenario
            ])


class MultiPodMonitor:
    """多namespace Pod監控管理器"""
    
    def __init__(self, experiment_type: str, scenario: str, output_base_dir: Path):
        self.experiment_type = experiment_type
        self.scenario = scenario
        self.output_base_dir = output_base_dir
        self.monitors: Dict[str, PodMonitor] = {}
        self.logger = logging.getLogger('MultiPodMonitor')
        
    def add_namespace(self, namespace: str):
        """添加要監控的namespace"""
        output_dir = self.output_base_dir / namespace
        monitor = PodMonitor(namespace, self.experiment_type, self.scenario, output_dir)
        self.monitors[namespace] = monitor
        self.logger.info(f"✅ 已添加 {namespace} namespace 監控")
        
    def start_all_monitoring(self, duration_minutes: int = 15):
        """開始所有namespace的監控"""
        for namespace, monitor in self.monitors.items():
            monitor.start_monitoring(duration_minutes)
        self.logger.info(f"🚀 已啟動 {len(self.monitors)} 個namespace的Pod監控")
        
    def stop_all_monitoring(self):
        """停止所有監控"""
        for namespace, monitor in self.monitors.items():
            monitor.stop_monitoring()
        self.logger.info(f"⏹️ 已停止所有Pod監控")
        
    def wait_for_completion(self, timeout_minutes: int = 20):
        """等待所有監控完成"""
        timeout_seconds = timeout_minutes * 60
        for namespace, monitor in self.monitors.items():
            if monitor.monitor_thread:
                monitor.monitor_thread.join(timeout=timeout_seconds)
        self.logger.info("✅ 所有Pod監控已完成")


# 便利函數
def create_pod_monitor_for_experiment(experiment_type: str, scenario: str, 
                                    namespaces: List[str], output_dir: Path) -> MultiPodMonitor:
    """為實驗創建Pod監控器
    
    Args:
        experiment_type: 實驗類型 (gnnrl, gym-hpa, k8s-hpa)
        scenario: 場景名稱 (offpeak, peak, rushsale, fluctuating)
        namespaces: 要監控的namespace列表
        output_dir: 輸出目錄
        
    Returns:
        配置好的多namespace Pod監控器
    """
    monitor = MultiPodMonitor(experiment_type, scenario, output_dir)
    
    for namespace in namespaces:
        monitor.add_namespace(namespace)
        
    return monitor