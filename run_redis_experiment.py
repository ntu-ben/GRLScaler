#!/usr/bin/env python3
"""
Redis 自動擴展實驗執行腳本
========================

專門用於執行 Redis 環境的三種自動擴展方法比較實驗。
支援 GNNRL、Gym-HPA 和 K8s-HPA 在 Redis 環境下的性能測試。
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from run_onlineboutique_experiment import ExperimentRunner
from pod_monitor import MultiPodMonitor, create_pod_monitor_for_experiment

class RedisExperimentRunner(ExperimentRunner):
    """Redis 實驗執行器"""
    
    def __init__(self, use_standardized_scenarios: bool = False, 
                 algorithm: str = 'ppo', stable_loadtest: bool = False, 
                 max_rps: int = None, loadtest_timeout: int = 30):
        super().__init__(
            use_standardized_scenarios=use_standardized_scenarios,
            algorithm=algorithm,
            stable_loadtest=stable_loadtest,
            max_rps=max_rps,
            loadtest_timeout=loadtest_timeout
        )
        
        # Redis 專用配置
        self.config.update({
            'use_case': 'redis',
            'namespace': 'redis'
        })
        
        # 設置 Redis HPA 配置
        # 簡化為只測試 CPU 配置
        self.redis_hpa_configs = {
            'cpu': ['cpu-20', 'cpu-40', 'cpu-60', 'cpu-80']
        }
        
    def check_redis_environment(self) -> bool:
        """檢查 Redis 環境"""
        self.log_section("🔍 檢查 Redis 實驗環境")
        
        try:
            # 檢查 Redis namespace
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', 'redis', '--no-headers'],
                capture_output=True, text=True, check=True
            )
            
            running_pods = [p for p in result.stdout.strip().split('\n') if 'Running' in p]
            if len(running_pods) < 2:
                self.log_error(f"❌ Redis 環境不完整，僅 {len(running_pods)} 個 Pod 運行")
                self.log_info("💡 請先部署 Redis 集群:")
                self.log_info("   kubectl apply -f MicroServiceBenchmark/redis-cluster/redis-cluster.yaml")
                return False
            
            self.log_success(f"✅ Redis 環境正常，{len(running_pods)} 個服務運行中")
            
            # 檢查 Redis 連接
            self._test_redis_connectivity()
            
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.log_error(f"❌ Redis 環境檢查失敗: {e}")
            return False
    
    def _test_redis_connectivity(self):
        """測試 Redis 連接"""
        try:
            # 使用 kubectl 測試 Redis 連接
            test_cmd = [
                'kubectl', 'run', 'redis-test', '--rm', '-i', '--restart=Never',
                '--image=redis:7.2-alpine', '-n', 'redis',
                '--', 'redis-cli', '-h', 'redis-master', 'ping'
            ]
            
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            
            if 'PONG' in result.stdout:
                self.log_success("✅ Redis 連接測試通過")
            else:
                self.log_error("❌ Redis 連接測試失敗")
                
        except Exception as e:
            self.log_error(f"❌ Redis 連接測試失敗: {e}")
    
    def run_gym_hpa_redis_experiment(self, plan: dict) -> bool:
        """執行 Gym-HPA Redis 實驗（訓練 + 測試）"""
        self.log_section("🎯 實驗 1/3: Gym-HPA (Redis 環境)")
        
        gym_plan = plan.get('gym_hpa', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 選擇實驗管理器
        manager_script = "standardized_experiment_manager.py" if self.use_standardized_scenarios else "unified_experiment_manager.py"
        
        # 第一階段：訓練
        if not gym_plan.get('skip_training', False):
            self.log_info("📚 第一階段：Gym-HPA 訓練")
            train_cmd = [
                sys.executable, manager_script,
                "--experiment", "gym_hpa",
                "--k8s",
                "--use-case", "redis",
                "--goal", self.config['goal'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", str(self.config['steps']),
                "--run-tag", f"gym_hpa_redis_train_seed{self.config['seed']}_{timestamp}"
            ]
            
            try:
                self.log_info("🧪 開始 Gym-HPA Redis 訓練...")
                result = subprocess.run(train_cmd, cwd=self.repo_root)
                
                if result.returncode != 0:
                    self.log_error("Gym-HPA Redis 訓練失敗")
                    return False
                
                self.log_success("✅ Gym-HPA Redis 訓練完成")
                
            except Exception as e:
                self.log_error(f"Gym-HPA Redis 訓練執行錯誤: {e}")
                return False
        
        # 第二階段：測試
        self.log_info("🧪 第二階段：Gym-HPA 測試")
        self.reset_redis_pods()  # 測試前重置
        
        test_cmd = [
            sys.executable, manager_script,
            "--experiment", "gym_hpa",
            "--k8s",
            "--use-case", "redis",
            "--goal", self.config['goal'],
            "--alg", self.config['alg'],
            "--seed", str(self.config['seed']),
            "--steps", "0",
            "--testing",
            "--run-tag", f"gym_hpa_redis_test_seed{self.config['seed']}_{timestamp}"
        ]
        
        # 如果有指定模型路徑，添加載入參數
        if gym_plan.get('model_path'):
            test_cmd.extend(["--load-path", gym_plan['model_path']])
        
        try:
            self.log_info("🧪 開始 Gym-HPA Redis 測試...")
            result = subprocess.run(test_cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("✅ Gym-HPA Redis 實驗完成")
                return True
            else:
                self.log_error("❌ Gym-HPA Redis 測試失敗")
                return False
                
        except Exception as e:
            self.log_error(f"❌ Gym-HPA Redis 測試執行錯誤: {e}")
            return False
    
    def run_gnnrl_redis_experiment(self, plan: dict) -> bool:
        """執行 GNNRL Redis 實驗（訓練 + 測試）"""
        self.log_section("🧠 實驗 2/3: GNNRL (Redis 環境)")
        
        gnnrl_plan = plan.get('gnnrl', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 選擇實驗管理器
        manager_script = "standardized_experiment_manager.py" if self.use_standardized_scenarios else "unified_experiment_manager.py"
        
        # 第一階段：訓練
        if not gnnrl_plan.get('skip_training', False):
            self.log_info("📚 第一階段：GNNRL 訓練")
            train_cmd = [
                sys.executable, manager_script,
                "--experiment", "gnnrl",
                "--k8s",
                "--use-case", "redis",
                "--goal", self.config['goal'],
                "--model", self.config['model'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", str(self.config['steps']),
                "--run-tag", f"gnnrl_redis_train_seed{self.config['seed']}_{timestamp}"
            ]
            
            try:
                self.log_info("🧪 開始 GNNRL Redis 訓練...")
                result = subprocess.run(train_cmd, cwd=self.repo_root)
                
                if result.returncode != 0:
                    self.log_error("❌ GNNRL Redis 訓練失敗")
                    return False
                
                self.log_success("✅ GNNRL Redis 訓練完成")
                
            except Exception as e:
                self.log_error(f"❌ GNNRL Redis 訓練執行錯誤: {e}")
                return False
        
        # 第二階段：測試
        self.log_info("🧪 第二階段：GNNRL 測試")
        self.reset_redis_pods()  # 測試前重置
        
        test_cmd = [
            sys.executable, manager_script,
            "--experiment", "gnnrl",
            "--k8s",
            "--use-case", "redis",
            "--goal", self.config['goal'],
            "--model", self.config['model'],
            "--alg", self.config['alg'],
            "--seed", str(self.config['seed']),
            "--steps", "0",
            "--testing",
            "--run-tag", f"gnnrl_redis_test_seed{self.config['seed']}_{timestamp}"
        ]
        
        # 如果有指定模型路徑，添加載入參數
        if gnnrl_plan.get('model_path'):
            test_cmd.extend(["--load-path", gnnrl_plan['model_path']])
        
        try:
            self.log_info("🧪 開始 GNNRL Redis 測試...")
            result = subprocess.run(test_cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("✅ GNNRL Redis 實驗完成")
                return True
            else:
                self.log_error("❌ GNNRL Redis 測試失敗")
                return False
                
        except Exception as e:
            self.log_error(f"❌ GNNRL Redis 測試執行錯誤: {e}")
            return False
    
    def run_k8s_hpa_redis_experiment(self) -> bool:
        """執行 K8s-HPA Redis 實驗"""
        self.log_section("⚖️ 實驗 3/3: K8s-HPA (Redis 環境)")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 扁平化所有配置
        all_configs = []
        for config_type, configs in self.redis_hpa_configs.items():
            all_configs.extend(configs)
            
        for hpa_config in all_configs:
            self.log_info(f"🔧 測試 Redis HPA 配置: {hpa_config}")
            
            # 應用 HPA 配置
            config_dir = self.repo_root / "macK8S" / "HPA" / "redis" / hpa_config
            if not config_dir.exists():
                self.log_error(f"❌ HPA 配置目錄不存在: {config_dir}")
                continue
            
            # 清除現有 HPA
            subprocess.run(["kubectl", "delete", "hpa", "--all", "-n", "redis"], 
                         capture_output=True)
            
            # 應用新配置
            for hpa_file in config_dir.glob("*.yaml"):
                subprocess.run(["kubectl", "apply", "-f", str(hpa_file)])
            
            # 等待 HPA 生效
            import time
            time.sleep(30)
            
            # 執行負載測試
            self._run_redis_loadtest(hpa_config, timestamp)
        
        return True
    
    def _run_redis_loadtest(self, hpa_config: str, timestamp: str):
        """執行 Redis 負載測試"""
        scenarios = ['offpeak', 'peak', 'rushsale', 'fluctuating']  # 所有 Redis 場景
        
        for scenario in scenarios:
            self.log_info(f"📊 執行 Redis 負載測試: {scenario}")
            
            # 重置 Redis Pod 數量
            self.reset_redis_pods()
            
            # 構建輸出目錄
            output_dir = self.repo_root / "logs" / "k8s_hpa_redis" / f"redis_hpa_{hpa_config}_{timestamp}" / scenario
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 選擇正確的腳本路徑
            script_path = self.repo_root / "loadtest" / "redis" / f"locust_redis_stable_{scenario}.py"
            
            # 檢查腳本是否存在
            if not script_path.exists():
                script_path = self.repo_root / "loadtest" / "redis" / f"locust_redis_{scenario}.py"
                
            if not script_path.exists():
                self.log_error(f"❌ 找不到負載測試腳本: {script_path}")
                continue
            
            # 使用當前 Python 環境的 locust 來避免模組問題
            import sys
            python_path = sys.executable
            cmd = [
                python_path, "-m", "locust", "-f", str(script_path), "--headless", 
                "--run-time", "15m",
                "--users", "50", "--spawn-rate", "5",
                "--csv", str(output_dir / scenario),
                "--html", str(output_dir / f"{scenario}.html"),
                "--host", "redis-master.redis.svc.cluster.local"
            ]
            
            try:
                self.log_info(f"🚀 開始執行 {scenario} 負載測試 (15分鐘)")
                result = subprocess.run(cmd, timeout=1200, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_success(f"✅ {scenario} 測試完成")
                else:
                    self.log_error(f"❌ {scenario} 測試失敗 (退出碼: {result.returncode})")
                    if result.stderr:
                        # 過濾掉 Locust 的警告信息，只顯示真正的錯誤
                        stderr_lines = result.stderr.strip().split('\n')
                        real_errors = []
                        for line in stderr_lines:
                            # 跳過常見的 Locust 警告信息
                            if ('Python 3.9 support is deprecated' in line or 
                                'have no impact on LoadShapes' in line or
                                'Starting Locust' in line):
                                continue
                            real_errors.append(line)
                        
                        if real_errors:
                            self.log_error(f"錯誤信息: {chr(10).join(real_errors)}")
                        else:
                            self.log_info("📋 只有 Locust 警告信息，無實際錯誤")
                    
                    if result.stdout:
                        self.log_info(f"輸出信息: {result.stdout[-500:]}")  # 只顯示最後500字符
                        
            except subprocess.TimeoutExpired:
                self.log_error(f"❌ {scenario} 測試超時")
            except Exception as e:
                self.log_error(f"❌ {scenario} 測試失敗: {e}")
    
    def run_k8s_hpa_redis_experiment_with_scenarios(self, selected_scenarios: list) -> bool:
        """執行 K8s-HPA Redis 實驗（支援場景選擇）"""
        self.log_section("⚖️ 實驗 3/3: K8s-HPA (Redis 環境) - 場景選擇模式")
        
        if 'all' in selected_scenarios:
            self.log_info("📊 執行所有場景的 K8s-HPA 測試")
            return self.run_k8s_hpa_redis_experiment()
        else:
            self.log_info(f"📊 執行選定場景的 K8s-HPA 測試: {', '.join(selected_scenarios)}")
            return self._run_k8s_hpa_redis_experiment_filtered(selected_scenarios)
    
    def _run_k8s_hpa_redis_experiment_filtered(self, selected_scenarios: list) -> bool:
        """執行篩選場景的 K8s-HPA Redis 實驗"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 扁平化所有配置
        all_configs = []
        for config_type, configs in self.redis_hpa_configs.items():
            all_configs.extend(configs)
        
        total_configs = len(all_configs)
        self.log_info(f"🗂️ 將測試 {total_configs} 個 HPA 配置，每個配置執行 {len(selected_scenarios)} 個場景")
        
        for i, hpa_config in enumerate(all_configs, 1):
            self.log_info(f"🔧 [{i}/{total_configs}] 測試 Redis HPA 配置: {hpa_config}")
            
            # 應用 HPA 配置
            config_dir = self.repo_root / "macK8S" / "HPA" / "redis" / hpa_config
            if not config_dir.exists():
                self.log_error(f"❌ HPA 配置目錄不存在: {config_dir}")
                continue
            
            # 清除現有 HPA
            self.log_info("🗑️ 清除現有 HPA 配置...")
            subprocess.run(["kubectl", "delete", "hpa", "--all", "-n", "redis"], 
                         capture_output=True)
            
            # 應用新配置
            self.log_info(f"📝 應用 {hpa_config} HPA 配置...")
            for hpa_file in config_dir.glob("*.yaml"):
                result = subprocess.run(["kubectl", "apply", "-f", str(hpa_file)], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    self.log_error(f"❌ HPA 配置應用失敗: {result.stderr}")
                    continue
            
            # 等待 HPA 生效
            self.log_info("⏱️ 等待 HPA 配置生效 (30秒)...")
            time.sleep(30)
            
            # 執行選定場景的負載測試
            self._run_redis_loadtest_filtered(hpa_config, timestamp, selected_scenarios)
            
            # 配置間等待
            if i < total_configs:
                self.log_info("⏱️ 配置間等待 60 秒...")
                time.sleep(60)
        
        self.log_success(f"✅ 所有 {total_configs} 個 Redis HPA 配置的選定場景測試完成")
        return True
    
    def _run_redis_loadtest_filtered(self, hpa_config: str, timestamp: str, selected_scenarios: list):
        """執行選定場景的 Redis 負載測試"""
        scenario_counter = 1
        
        for scenario in selected_scenarios:
            scenario_tag = f"{scenario}_{scenario_counter:03d}"
            self.log_info(f"📊 執行選定場景 Redis 負載測試: {scenario_tag}")
            
            # 重置 Redis Pod 數量
            self.reset_redis_pods()
            
            # 構建輸出目錄
            output_dir = self.repo_root / "logs" / "k8s_hpa_redis" / f"redis_hpa_{hpa_config}_{timestamp}" / scenario_tag
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 設置 Pod 監控
            pod_monitor = self._setup_pod_monitoring_for_redis(scenario, output_dir)
            
            # 選擇正確的腳本路徑
            script_path = self.repo_root / "loadtest" / "redis" / f"locust_redis_stable_{scenario}.py"
            
            # 檢查腳本是否存在
            if not script_path.exists():
                script_path = self.repo_root / "loadtest" / "redis" / f"locust_redis_{scenario}.py"
                
            if not script_path.exists():
                self.log_error(f"❌ 找不到負載測試腳本: {script_path}")
                scenario_counter += 1
                continue
            
            # 使用當前 Python 環境的 locust
            python_path = sys.executable
            cmd = [
                python_path, "-m", "locust", "-f", str(script_path), "--headless", 
                "--run-time", "15m",
                "--users", "50", "--spawn-rate", "5",
                "--csv", str(output_dir / scenario),
                "--html", str(output_dir / f"{scenario}.html"),
                "--host", "redis-master.redis.svc.cluster.local"
            ]
            
            try:
                self.log_info(f"🚀 開始執行 {scenario} 負載測試 (15分鐘)")
                
                # 啟動 Pod 監控
                pod_monitor.start_all_monitoring(15)  # 15分鐘監控
                
                # 等待 Pod 穩定
                time.sleep(30)
                
                # 執行負載測試
                start_time = time.time()
                result = subprocess.run(cmd, timeout=1200, capture_output=True, text=True)
                end_time = time.time()
                
                duration = int(end_time - start_time)
                
                # 停止 Pod 監控
                pod_monitor.stop_all_monitoring()
                
                if result.returncode == 0:
                    self.log_success(f"✅ {scenario} 測試完成 (耗時: {duration}秒)")
                    self.log_info(f"📊 數據已保存到: {output_dir}")
                    self.log_info(f"📈 Pod 監控數據: {output_dir / 'pod_metrics'}")
                else:
                    self.log_error(f"❌ {scenario} 測試失敗 (退出碼: {result.returncode})")
                    if result.stderr:
                        # 過濾掉 Locust 的警告信息，只顯示真正的錯誤
                        stderr_lines = result.stderr.strip().split('\n')
                        real_errors = []
                        for line in stderr_lines:
                            # 跳過常見的 Locust 警告信息
                            if ('Python 3.9 support is deprecated' in line or 
                                'have no impact on LoadShapes' in line or
                                'Starting Locust' in line):
                                continue
                            real_errors.append(line)
                        
                        if real_errors:
                            self.log_error(f"錯誤信息: {chr(10).join(real_errors)}")
                        else:
                            self.log_info("📋 只有 Locust 警告信息，無實際錯誤")
                    
                    if result.stdout:
                        self.log_info(f"輸出信息: {result.stdout[-500:]}")  # 只顯示最後500字符
                        
            except subprocess.TimeoutExpired:
                self.log_error(f"❌ {scenario} 測試超時")
                pod_monitor.stop_all_monitoring()
            except Exception as e:
                self.log_error(f"❌ {scenario} 測試失敗: {e}")
                pod_monitor.stop_all_monitoring()
            
            # 場景間等待
            if scenario_counter < len(selected_scenarios):
                self.log_info("⏱️ 場景間等待 30 秒...")
                time.sleep(30)
            
            scenario_counter += 1
    
    def ask_user_experiment_choice(self, method_name: str) -> tuple:
        """詢問用戶對特定方法要執行訓練、測試還是跳過"""
        while True:
            print(f"\n{method_name} 實驗選項:")
            print("  1. train - 只執行訓練")
            print("  2. test - 只執行測試 (需要現有模型)")
            print("  3. both - 執行訓練後接著測試")
            print("  4. skip - 跳過此方法")
            
            response = input(f"請選擇 {method_name} 的執行模式 (1/2/3/4): ").strip()
            
            if response in ['1', 'train', '訓練']:
                return ('train', True)
            elif response in ['2', 'test', '測試']:
                return ('test', True)
            elif response in ['3', 'both', '兩者', '全部']:
                return ('both', True)
            elif response in ['4', 'skip', '跳過']:
                self.log_info(f"⏭️ 跳過 {method_name} 實驗")
                return ('skip', False)
            else:
                print("請輸入 1(train)/2(test)/3(both)/4(skip)")
    
    def ask_model_path_if_needed(self, method_name: str) -> str:
        """如果需要測試，詢問模型路徑"""
        while True:
            print(f"\n{method_name} 測試需要指定模型路徑:")
            print("  1. auto - 自動尋找最新的訓練模型")
            print("  2. path - 手動輸入模型路徑")
            
            choice = input("請選擇 (1/2): ").strip()
            
            if choice in ['1', 'auto', '自動']:
                return 'auto'
            elif choice in ['2', 'path', '手動']:
                path = input("請輸入模型路徑: ").strip()
                if path:
                    return path
                else:
                    print("模型路徑不能為空")
            else:
                print("請輸入 1(auto) 或 2(path)")
    
    def ask_scenario_selection(self, method_name: str, mode: str) -> list:
        """詢問要執行哪些場景（可多選）"""
        if mode not in ['test', 'both']:  # 只有測試相關模式才詢問場景
            return ['all']  # 訓練模式執行所有場景
        
        available_scenarios = ['offpeak', 'peak', 'rushsale', 'fluctuating']
        
        while True:
            print(f"\n{method_name} 要執行哪些場景？")
            print("可用場景:")
            for i, scenario in enumerate(available_scenarios, 1):
                print(f"  {i}. {scenario}")
            print("  a. all - 所有場景")
            
            choice = input("請選擇場景 (可用逗號分隔多選，如: 1,2 或 peak,rushsale): ").strip()
            
            if not choice:
                print("請選擇至少一個場景")
                continue
            
            # 處理 'all' 或 'a' 選項
            if choice.lower() in ['a', 'all', '全部']:
                self.log_info(f"✅ {method_name} 將執行所有場景: {', '.join(available_scenarios)}")
                return ['all']
            
            # 解析用戶輸入
            selected_scenarios = []
            choices = [c.strip() for c in choice.split(',')]
            
            for c in choices:
                # 數字選擇
                if c.isdigit():
                    idx = int(c) - 1
                    if 0 <= idx < len(available_scenarios):
                        scenario = available_scenarios[idx]
                        if scenario not in selected_scenarios:
                            selected_scenarios.append(scenario)
                    else:
                        print(f"無效的數字選擇: {c}")
                        selected_scenarios = []  # 重置，重新選擇
                        break
                # 場景名稱直接選擇
                elif c.lower() in [s.lower() for s in available_scenarios]:
                    # 找到對應的場景名稱（忽略大小寫）
                    scenario = next(s for s in available_scenarios if s.lower() == c.lower())
                    if scenario not in selected_scenarios:
                        selected_scenarios.append(scenario)
                else:
                    print(f"無效的場景選擇: {c}")
                    selected_scenarios = []  # 重置，重新選擇
                    break
            
            if selected_scenarios:
                self.log_info(f"✅ {method_name} 將執行選定場景: {', '.join(selected_scenarios)}")
                return selected_scenarios
            else:
                print("請重新選擇有效的場景")
    
    def reset_redis_pods(self):
        """重置 Redis 所有 Pod 數量為 1"""
        self.log_info("🔄 重置 Redis namespace 所有 Pod 數量到 1")
        try:
            deployments = ['redis-master', 'redis-slave']
            for deployment in deployments:
                cmd = ['kubectl', 'scale', 'deployment', deployment, '--replicas=1', '-n', 'redis']
                subprocess.run(cmd, check=True, capture_output=True)
                self.log_success(f"✅ {deployment} 已重置為 1 replica")
            
            # 等待 Pod 穩定
            import time
            time.sleep(30)
            
        except Exception as e:
            self.log_error(f"❌ 重置 Redis Pod 失敗: {e}")
    
    def _setup_pod_monitoring_for_redis(self, scenario: str, output_dir: Path) -> MultiPodMonitor:
        """為 Redis 實驗設置 Pod 監控"""
        pod_monitoring_dir = output_dir / "pod_metrics"
        
        # 創建多namespace Pod監控器
        pod_monitor = create_pod_monitor_for_experiment(
            experiment_type="k8s-hpa-redis",
            scenario=scenario,
            namespaces=["redis"],  # Redis 只監控 redis namespace
            output_dir=pod_monitoring_dir
        )
        
        return pod_monitor
    
    def find_latest_model(self, method: str) -> str:
        """尋找最新的訓練模型"""
        models_dir = self.repo_root / "logs" / "models"
        if not models_dir.exists():
            return None
        
        # 根據方法匹配正確的檔名模式
        if method == 'gym_hpa':
            # Gym-HPA 模型檔名格式: a2c_env_redis_gym_goal_latency_k8s_True_totalSteps_5000.zip
            # 或 ppo_env_redis_gym_goal_latency_k8s_True_totalSteps_5000.zip
            patterns = ["*_env_redis_gym_*.zip"]
        elif method == 'gnnrl':
            # GNNRL 模型檔名格式: gnnrl_gat_redis_latency_k8s_True_steps_5.zip
            patterns = ["gnnrl_*redis*.zip"]
        else:
            # 通用搜索
            patterns = [f"*{method}*redis*.zip"]
        
        models = []
        for pattern in patterns:
            models.extend(list(models_dir.glob(pattern)))
        
        # 如果沒找到，嘗試舊格式 (不含redis但包含k8s_True)
        if not models:
            if method == 'gym_hpa':
                # 對 Gym-HPA，搜索所有 a2c/ppo 環境模型並檢查是否為 redis
                all_models = list(models_dir.glob("*_env_*_gym_*.zip"))
                models = [m for m in all_models if 'redis' in str(m).lower()]
            else:
                pattern_old = f"*{method}*.zip"
                all_models = list(models_dir.glob(pattern_old))
                # 過濾出可能的redis模型 (檢查檔名是否包含相關關鍵字)
                models = [m for m in all_models if 'redis' in str(m).lower() or 'k8s_True' in str(m)]
        
        if not models:
            return None
        
        # 按修改時間排序，返回最新的
        latest_model = max(models, key=lambda x: x.stat().st_mtime)
        return str(latest_model)
    
    def run_gym_hpa_redis_experiment_with_mode(self, config: dict) -> bool:
        """根據模式執行 Gym-HPA Redis 實驗（使用統一實驗管理器）"""
        mode = config['mode']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if mode == 'train':
            self.log_section("🎯 Gym-HPA 訓練模式 (Redis)")
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gym_hpa",
                "--k8s",
                "--use-case", "redis",
                "--goal", self.config['goal'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", str(self.config['steps']),
                "--run-tag", f"gym_hpa_redis_train_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保執行負載測試
            ]
            
            try:
                self.log_info("🧪 開始 Gym-HPA Redis 訓練...")
                self.log_info(f"🗋 命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=self.repo_root)
                
                if result.returncode == 0:
                    self.log_success("✅ Gym-HPA Redis 訓練完成")
                    return True
                else:
                    self.log_error(f"❌ Gym-HPA Redis 訓練失敗 (退出碼: {result.returncode})")
                    return False
                    
            except Exception as e:
                self.log_error(f"❌ Gym-HPA Redis 訓練執行錯誤: {e}")
                return False
        
        elif mode == 'test':
            self.log_section("🎯 Gym-HPA 測試模式 (Redis)")
            
            model_path = config.get('model_path')
            if model_path == 'auto':
                model_path = self.find_latest_model('gym_hpa')
                if not model_path:
                    self.log_error("❌ 找不到 Gym-HPA 模型進行測試")
                    return False
                self.log_info(f"🔍 自動找到模型: {Path(model_path).name}")
            
            self.reset_redis_pods()
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gym_hpa",
                "--k8s",
                "--use-case", "redis",
                "--goal", self.config['goal'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", "0",
                "--testing",
                "--load-path", model_path,
                "--run-tag", f"gym_hpa_redis_test_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保測試模式也執行負載測試
            ]
            
            try:
                self.log_info("🧪 開始 Gym-HPA Redis 測試...")
                self.log_info(f"🗋 命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=self.repo_root)
                
                if result.returncode == 0:
                    self.log_success("✅ Gym-HPA Redis 測試完成")
                    return True
                else:
                    self.log_error(f"❌ Gym-HPA Redis 測試失敗 (退出碼: {result.returncode})")
                    return False
                    
            except Exception as e:
                self.log_error(f"❌ Gym-HPA Redis 測試執行錯誤: {e}")
                return False
        
        elif mode == 'both':
            self.log_section("🎯 Gym-HPA 訓練+測試模式 (Redis)")
            
            # 先執行訓練
            train_success = self.run_gym_hpa_redis_experiment_with_mode({'mode': 'train'})
            if not train_success:
                return False
            
            # 等待一段時間再執行測試
            self.log_info("⏱️ 訓練完成，等待 30 秒後開始測試...")
            time.sleep(30)
            
            # 再執行測試
            test_config = {'mode': 'test', 'model_path': 'auto'}
            return self.run_gym_hpa_redis_experiment_with_mode(test_config)
        
        else:
            self.log_error(f"❌ 未知的 Gym-HPA 模式: {mode}")
            return False
    
    def run_gnnrl_redis_experiment_with_mode(self, config: dict) -> bool:
        """根據模式執行 GNNRL Redis 實驗（使用統一實驗管理器）"""
        mode = config['mode']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if mode == 'train':
            self.log_section("🧠 GNNRL 訓練模式 (Redis)")
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gnnrl",
                "--k8s",
                "--use-case", "redis",
                "--goal", self.config['goal'],
                "--model", self.config['model'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", str(self.config['steps']),
                "--run-tag", f"gnnrl_redis_train_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保執行負載測試
            ]
            
            try:
                self.log_info("🧪 開始 GNNRL Redis 訓練...")
                self.log_info(f"🗋 命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=self.repo_root)
                
                if result.returncode == 0:
                    self.log_success("✅ GNNRL Redis 訓練完成")
                    return True
                else:
                    self.log_error(f"❌ GNNRL Redis 訓練失敗 (退出碼: {result.returncode})")
                    return False
                    
            except Exception as e:
                self.log_error(f"❌ GNNRL Redis 訓練執行錯誤: {e}")
                return False
        
        elif mode == 'test':
            self.log_section("🧠 GNNRL 測試模式 (Redis)")
            
            model_path = config.get('model_path')
            if model_path == 'auto':
                model_path = self.find_latest_model('gnnrl')
                if not model_path:
                    self.log_error("❌ 找不到 GNNRL 模型進行測試")
                    return False
                self.log_info(f"🔍 自動找到模型: {Path(model_path).name}")
            
            self.reset_redis_pods()
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gnnrl",
                "--k8s",
                "--use-case", "redis",
                "--goal", self.config['goal'],
                "--model", self.config['model'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", "0",
                "--testing",
                "--load-path", model_path,
                "--run-tag", f"gnnrl_redis_test_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保測試模式也執行負載測試
            ]
            
            try:
                self.log_info("🧪 開始 GNNRL Redis 測試...")
                self.log_info(f"🗋 命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=self.repo_root)
                
                if result.returncode == 0:
                    self.log_success("✅ GNNRL Redis 測試完成")
                    return True
                else:
                    self.log_error(f"❌ GNNRL Redis 測試失敗 (退出碼: {result.returncode})")
                    return False
                    
            except Exception as e:
                self.log_error(f"❌ GNNRL Redis 測試執行錯誤: {e}")
                return False
        
        elif mode == 'both':
            self.log_section("🧠 GNNRL 訓練+測試模式 (Redis)")
            
            # 先執行訓練
            train_success = self.run_gnnrl_redis_experiment_with_mode({'mode': 'train'})
            if not train_success:
                return False
            
            # 等待一段時間再執行測試
            self.log_info("⏱️ 訓練完成，等待 30 秒後開始測試...")
            time.sleep(30)
            
            # 再執行測試
            test_config = {'mode': 'test', 'model_path': 'auto'}
            return self.run_gnnrl_redis_experiment_with_mode(test_config)
        
        else:
            self.log_error(f"❌ 未知的 GNNRL 模式: {mode}")
            return False
    
    def run_complete_redis_experiment(self, steps: int = 5000, goal: str = "latency", model: str = "gat") -> bool:
        """執行完整 Redis 實驗流程"""
        
        # 更新配置
        self.config.update({
            'steps': steps,
            'goal': goal,
            'model': model
        })
        
        print("\\033[0;34m")
        print("🚀 開始 Redis 自動擴展實驗")
        print(f"📅 時間: {datetime.now().strftime('%Y年 %m月%d日 %H時%M分%S秒')}")
        print(f"🎲 種子: {self.config['seed']}")
        print(f"📊 步數: {self.config['steps']}")
        print(f"🎯 目標: {self.config['goal']}")
        print(f"🗄️ 環境: Redis")
        print("\\033[0m")
        
        try:
            # 1. 檢查 Redis 環境
            if not self.check_redis_environment():
                return False
            
            # 2. 詢問用戶要執行哪些實驗及模式
            experiment_plan = {}
            
            # Gym-HPA 選擇
            gym_hpa_mode, should_run = self.ask_user_experiment_choice("Gym-HPA")
            if should_run:
                experiment_plan['gym_hpa'] = {'mode': gym_hpa_mode}
                
                # 詢問場景選擇
                scenarios = self.ask_scenario_selection("Gym-HPA", gym_hpa_mode)
                experiment_plan['gym_hpa']['scenarios'] = scenarios
                
                if gym_hpa_mode in ['test', 'both'] and gym_hpa_mode != 'both':
                    model_path = self.ask_model_path_if_needed("Gym-HPA")
                    experiment_plan['gym_hpa']['model_path'] = model_path
            
            # GNNRL 選擇
            gnnrl_mode, should_run = self.ask_user_experiment_choice("GNNRL")
            if should_run:
                experiment_plan['gnnrl'] = {'mode': gnnrl_mode}
                
                # 詢問場景選擇
                scenarios = self.ask_scenario_selection("GNNRL", gnnrl_mode)
                experiment_plan['gnnrl']['scenarios'] = scenarios
                
                if gnnrl_mode in ['test', 'both'] and gnnrl_mode != 'both':
                    model_path = self.ask_model_path_if_needed("GNNRL")
                    experiment_plan['gnnrl']['model_path'] = model_path
            
            # K8s-HPA 選擇 (只有測試模式)
            k8s_hpa_mode, should_run = self.ask_user_experiment_choice("K8s-HPA")
            if should_run:
                experiment_plan['k8s_hpa'] = {'mode': k8s_hpa_mode}
                
                # 詢問場景選擇
                scenarios = self.ask_scenario_selection("K8s-HPA", k8s_hpa_mode)
                experiment_plan['k8s_hpa']['scenarios'] = scenarios
            
            if not experiment_plan:
                self.log_info("⚠️ 沒有選擇任何實驗方法，退出")
                return False
            
            # 顯示執行計劃
            self.log_info("📋 實驗執行計劃:")
            for method, config in experiment_plan.items():
                mode_desc = {
                    'train': '訓練',
                    'test': '測試',
                    'both': '訓練+測試',
                    'skip': '跳過'
                }
                scenarios_desc = config.get('scenarios', ['all'])
                scenario_text = '所有場景' if 'all' in scenarios_desc else ', '.join(scenarios_desc)
                self.log_info(f"   {method.upper()}: {mode_desc.get(config['mode'], config['mode'])} - 場景: {scenario_text}")
            
            # 3. 執行實驗
            experiment_results = {}
            
            # Gym-HPA Redis 實驗
            if 'gym_hpa' in experiment_plan:
                self.reset_redis_pods()
                experiment_results['gym_hpa'] = self.run_gym_hpa_redis_experiment_with_mode(experiment_plan['gym_hpa'])
            
            # GNNRL Redis 實驗  
            if 'gnnrl' in experiment_plan:
                self.reset_redis_pods()
                experiment_results['gnnrl'] = self.run_gnnrl_redis_experiment_with_mode(experiment_plan['gnnrl'])
            
            # K8s-HPA Redis 實驗
            if 'k8s_hpa' in experiment_plan:
                self.reset_redis_pods()
                experiment_results['k8s_hpa'] = self.run_k8s_hpa_redis_experiment_with_scenarios(experiment_plan['k8s_hpa'].get('scenarios', ['all']))
            
            self.log_section("🎉 Redis 實驗完成!")
            
            # 顯示結果摘要
            self.log_info("📊 實驗結果摘要:")
            for method, success in experiment_results.items():
                status = "✅ 成功" if success else "❌ 失敗"
                self.log_info(f"   {method.upper()}: {status}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Redis 實驗執行失敗: {e}")
            return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='Redis 自動擴展實驗')
    parser.add_argument('--steps', type=int, default=5000, help='訓練步數')
    parser.add_argument('--goal', default='latency', help='目標 (latency/cost)')
    parser.add_argument('--model', default='gat', help='GNNRL 模型類型')
    parser.add_argument('--algorithm', '--alg', default='ppo', 
                       choices=['ppo', 'a2c'], help='RL算法選擇 (ppo/a2c)')
    parser.add_argument('--standardized', action='store_true', 
                       help='使用標準化場景')
    parser.add_argument('--stable-loadtest', action='store_true',
                       help='使用穩定loadtest模式')
    parser.add_argument('--max-rps', type=int, default=None,
                       help='限定最高RPS數值')
    
    args = parser.parse_args()
    
    print("🗄️ Redis 自動擴展實驗模式")
    print(f"🧠 算法: {args.algorithm.upper()}")
    print("✅ 將測試 GNNRL、Gym-HPA、K8s-HPA 在 Redis 環境下的性能")
    print()
    
    runner = RedisExperimentRunner(
        use_standardized_scenarios=args.standardized,
        algorithm=args.algorithm,
        stable_loadtest=args.stable_loadtest,
        max_rps=args.max_rps
    )
    success = runner.run_complete_redis_experiment(args.steps, args.goal, args.model)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()