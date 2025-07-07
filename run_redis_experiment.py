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
from pathlib import Path
from datetime import datetime
from run_onlineboutique_experiment import ExperimentRunner

class RedisExperimentRunner(ExperimentRunner):
    """Redis 實驗執行器"""
    
    def __init__(self, use_standardized_scenarios: bool = False):
        super().__init__(use_standardized_scenarios=use_standardized_scenarios)
        
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
        """執行 Gym-HPA Redis 實驗"""
        self.log_section("🎯 實驗 1/3: Gym-HPA (Redis 環境)")
        
        gym_plan = plan.get('gym_hpa', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 選擇實驗管理器
        manager_script = "standardized_experiment_manager.py" if self.use_standardized_scenarios else "unified_experiment_manager.py"
        
        # 構建命令
        cmd = [
            sys.executable, manager_script,
            "--experiment", "gym_hpa",
            "--k8s",
            "--use-case", "redis",
            "--goal", self.config['goal'],
            "--alg", self.config['alg'],
            "--seed", str(self.config['seed'])
        ]
        
        if gym_plan.get('skip_training', False) and gym_plan.get('model_path'):
            cmd.extend([
                "--steps", "0",
                "--testing",
                "--load-path", gym_plan['model_path'],
                "--run-tag", f"gym_hpa_redis_test_seed{self.config['seed']}_{timestamp}"
            ])
        else:
            cmd.extend([
                "--steps", str(self.config['steps']),
                "--run-tag", f"gym_hpa_redis_train_seed{self.config['seed']}_{timestamp}"
            ])
        
        try:
            self.log_info("🧪 開始 Gym-HPA Redis 實驗...")
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("Gym-HPA Redis 實驗完成")
                return True
            else:
                self.log_error("Gym-HPA Redis 實驗失敗")
                return False
                
        except Exception as e:
            self.log_error(f"Gym-HPA Redis 實驗執行錯誤: {e}")
            return False
    
    def run_gnnrl_redis_experiment(self, plan: dict) -> bool:
        """執行 GNNRL Redis 實驗"""
        self.log_section("🧠 實驗 2/3: GNNRL (Redis 環境)")
        
        gnnrl_plan = plan.get('gnnrl', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 選擇實驗管理器
        manager_script = "standardized_experiment_manager.py" if self.use_standardized_scenarios else "unified_experiment_manager.py"
        
        # 構建命令
        cmd = [
            sys.executable, manager_script,
            "--experiment", "gnnrl",
            "--k8s",
            "--use-case", "redis",
            "--goal", self.config['goal'],
            "--model", self.config['model'],
            "--alg", self.config['alg'],
            "--seed", str(self.config['seed'])
        ]
        
        if gnnrl_plan.get('skip_training', False) and gnnrl_plan.get('model_path'):
            cmd.extend([
                "--steps", "0",
                "--testing",
                "--load-path", gnnrl_plan['model_path'],
                "--run-tag", f"gnnrl_redis_test_seed{self.config['seed']}_{timestamp}"
            ])
        else:
            cmd.extend([
                "--steps", str(self.config['steps']),
                "--run-tag", f"gnnrl_redis_train_seed{self.config['seed']}_{timestamp}"
            ])
        
        try:
            self.log_info("🧪 開始 GNNRL Redis 實驗...")
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("GNNRL Redis 實驗完成")
                return True
            else:
                self.log_error("GNNRL Redis 實驗失敗")
                return False
                
        except Exception as e:
            self.log_error(f"GNNRL Redis 實驗執行錯誤: {e}")
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
        scenarios = ['redis_peak', 'redis_offpeak']
        
        for scenario in scenarios:
            self.log_info(f"📊 執行 Redis 負載測試: {scenario}")
            
            # 構建輸出目錄
            output_dir = self.repo_root / "logs" / "k8s_hpa_redis" / f"redis_hpa_{hpa_config}_{timestamp}" / scenario
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 執行 Locust 測試
            script_path = self.repo_root / "loadtest" / "redis" / f"locust_{scenario}.py"
            
            # 使用當前 Python 環境的 locust 來避免模組問題
            import sys
            python_path = sys.executable
            cmd = [
                python_path, "-m", "locust", "-f", str(script_path), "--headless", 
                "--run-time", "15m",
                "--users", "50", "--spawn-rate", "5",
                "--csv", str(output_dir / scenario),
                "--html", str(output_dir / f"{scenario}.html")
            ]
            
            try:
                subprocess.run(cmd, timeout=1200)  # 20分鐘超時
                self.log_success(f"✅ {scenario} 測試完成")
            except Exception as e:
                self.log_error(f"❌ {scenario} 測試失敗: {e}")
    
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
            
            # 2. 執行實驗 (簡化版，不使用複雜的規劃)
            plan = {
                'gym_hpa': {'skip_training': False, 'model_path': None},
                'gnnrl': {'skip_training': False, 'model_path': None}
            }
            
            # Gym-HPA Redis 實驗
            self.run_gym_hpa_redis_experiment(plan)
            
            # GNNRL Redis 實驗  
            self.run_gnnrl_redis_experiment(plan)
            
            # K8s-HPA Redis 實驗
            self.run_k8s_hpa_redis_experiment()
            
            self.log_section("🎉 Redis 實驗完成!")
            
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
    parser.add_argument('--standardized', action='store_true', 
                       help='使用標準化場景')
    
    args = parser.parse_args()
    
    print("🗄️ Redis 自動擴展實驗模式")
    print("✅ 將測試 GNNRL、Gym-HPA、K8s-HPA 在 Redis 環境下的性能")
    print()
    
    runner = RedisExperimentRunner(use_standardized_scenarios=args.standardized)
    success = runner.run_complete_redis_experiment(args.steps, args.goal, args.model)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()