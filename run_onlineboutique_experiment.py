#!/usr/bin/env python3
"""
完整三方法自動縮放實驗 Python 版本
====================================

替代 bash 腳本，提供更穩定的實驗執行流程。
支持標準化場景確保公平比較。
"""

import os
import sys
import subprocess
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from experiment_planner import ExperimentPlanner

class ExperimentRunner:
    def __init__(self, repo_root: Path = None, use_standardized_scenarios: bool = False):
        self.repo_root = repo_root or Path(__file__).parent
        self.planner = ExperimentPlanner(repo_root)
        self.use_standardized_scenarios = use_standardized_scenarios
        
        # 預設配置
        self.config = {
            'seed': 42,
            'steps': 5000,
            'goal': 'latency',
            'use_case': 'online_boutique',
            'model': 'gat',
            'alg': 'ppo'
        }
        
        # 如果使用標準化場景，載入配置
        if self.use_standardized_scenarios:
            self._ensure_standardized_config()
    
    def _ensure_standardized_config(self):
        """確保標準化配置文件存在"""
        config_file = self.repo_root / "standardized_test_scenarios.json"
        
        if not config_file.exists():
            self.log_info("🔧 生成標準化場景配置...")
            subprocess.run([sys.executable, "standardized_test_config.py"], 
                         cwd=self.repo_root, check=True)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.scenario_config = json.load(f)
            
        self.log_info(f"✅ 載入標準化配置：{len(self.scenario_config['scenarios'])} 個場景")
        
    def log_info(self, message: str):
        """資訊日誌"""
        print(f"\033[0;36m[INFO]\033[0m {message}")
        
    def log_success(self, message: str):
        """成功日誌"""
        print(f"\033[0;32m[SUCCESS]\033[0m {message}")
        
    def log_error(self, message: str):
        """錯誤日誌"""
        print(f"\033[0;31m[ERROR]\033[0m {message}")
        
    def log_section(self, title: str):
        """區段標題"""
        print(f"\n\033[0;35m{'=' * 50}\033[0m")
        print(f"\033[0;35m{title}\033[0m")
        print(f"\033[0;35m{'=' * 50}\033[0m")
        
    def check_prerequisites(self) -> bool:
        """檢查前置條件"""
        self.log_section("🔍 檢查實驗環境")
        
        try:
            # 使用 unified_experiment_manager.py 驗證環境
            cmd = [
                sys.executable, "unified_experiment_manager.py", 
                "--validate-only"
            ]
            
            result = subprocess.run(cmd, cwd=self.repo_root, capture_output=False)
            
            if result.returncode == 0:
                self.log_success("環境驗證通過")
                return True
            else:
                self.log_error("環境驗證失敗，請檢查 K8s 集群和分散式測試代理")
                return False
                
        except Exception as e:
            self.log_error(f"環境檢查失敗: {e}")
            return False
    
    def run_gym_hpa_experiment(self, plan: dict) -> bool:
        """執行 Gym-HPA 實驗"""
        if self.use_standardized_scenarios:
            self.log_section("🎯 實驗 1/3: Gym-HPA (標準化場景)")
            self.log_info(f"📊 將執行 {len(self.scenario_config['scenarios'])} 個標準化場景")
        else:
            self.log_section("🎯 實驗 1/3: Gym-HPA (基礎強化學習)")
        
        gym_plan = plan.get('gym_hpa', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 選擇實驗管理器
        manager_script = "standardized_experiment_manager.py" if self.use_standardized_scenarios else "unified_experiment_manager.py"
        
        # 構建命令 - 確保包含負載測試
        cmd = [
            sys.executable, manager_script,
            "--experiment", "gym_hpa",
            "--k8s",
            "--use-case", self.config['use_case'],
            "--goal", self.config['goal'],
            "--alg", self.config['alg'],
            "--seed", str(self.config['seed'])
        ]
        
        if gym_plan.get('skip_training', False) and gym_plan.get('model_path'):
            # 使用現有模型進行測試
            self.log_success(f"使用現有模型: {Path(gym_plan['model_path']).name}")
            self.log_info("⏭️  跳過訓練階段")
            
            cmd.extend([
                "--steps", "0",
                "--testing",
                "--load-path", gym_plan['model_path'],
                "--run-tag", f"gym_hpa_test_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保測試模式也執行負載測試
            ])
        else:
            # 進行完整訓練和測試
            self.log_info(f"🚀 開始 Gym-HPA 訓練 ({self.config['steps']} steps)...")
            
            cmd.extend([
                "--steps", str(self.config['steps']),
                "--run-tag", f"gym_hpa_train_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保訓練模式也執行負載測試
            ])
        
        try:
            self.log_info("🧪 開始 Gym-HPA 實驗...")
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("Gym-HPA 實驗完成")
                return True
            else:
                self.log_error("Gym-HPA 實驗失敗")
                return False
                
        except Exception as e:
            self.log_error(f"Gym-HPA 實驗執行錯誤: {e}")
            return False
    
    def run_gnnrl_experiment(self, plan: dict) -> bool:
        """執行 GNNRL 實驗"""
        if self.use_standardized_scenarios:
            self.log_section("🧠 實驗 2/3: GNNRL (標準化場景)")
            self.log_info(f"📊 將執行 {len(self.scenario_config['scenarios'])} 個標準化場景")
        else:
            self.log_section("🧠 實驗 2/3: GNNRL (圖神經網路強化學習)")
        
        gnnrl_plan = plan.get('gnnrl', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 選擇實驗管理器
        manager_script = "standardized_experiment_manager.py" if self.use_standardized_scenarios else "unified_experiment_manager.py"
        
        # 構建命令 - 添加use-case參數
        cmd = [
            sys.executable, manager_script,
            "--experiment", "gnnrl",
            "--k8s",
            "--use-case", self.config['use_case'],
            "--goal", self.config['goal'],
            "--model", self.config['model'],
            "--alg", self.config['alg'],
            "--seed", str(self.config['seed'])
        ]
        
        if gnnrl_plan.get('skip_training', False) and gnnrl_plan.get('model_path'):
            # 使用現有模型進行測試
            self.log_success(f"使用現有模型: {Path(gnnrl_plan['model_path']).name}")
            self.log_info("⏭️  跳過訓練階段")
            
            cmd.extend([
                "--steps", "0",
                "--testing",
                "--load-path", gnnrl_plan['model_path'],
                "--run-tag", f"gnnrl_test_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保測試模式也執行負載測試
            ])
        else:
            # 進行完整訓練和測試
            self.log_info(f"🚀 開始 GNNRL 訓練 ({self.config['steps']} steps)...")
            
            cmd.extend([
                "--steps", str(self.config['steps']),
                "--run-tag", f"gnnrl_train_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"  # 確保訓練模式也執行負載測試
            ])
        
        try:
            self.log_info("🧪 開始 GNNRL 實驗...")
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("GNNRL 實驗完成")
                return True
            else:
                self.log_error("GNNRL 實驗失敗")
                return False
                
        except Exception as e:
            self.log_error(f"GNNRL 實驗執行錯誤: {e}")
            return False
    
    def run_k8s_hpa_experiment(self) -> bool:
        """執行 K8s-HPA 實驗"""
        if self.use_standardized_scenarios:
            self.log_section("⚖️ 實驗 3/3: K8s-HPA (標準化場景)")
            self.log_info(f"📊 將對每個HPA配置執行 {len(self.scenario_config['scenarios'])} 個標準化場景")
        else:
            self.log_section("⚖️ 實驗 3/3: K8s-HPA (原生HPA基準測試)")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 選擇實驗管理器
        manager_script = "standardized_experiment_manager.py" if self.use_standardized_scenarios else "unified_experiment_manager.py"
        
        cmd = [
            sys.executable, manager_script,
            "--experiment", "k8s_hpa",
            "--hpa-type", "cpu",
            "--seed", str(self.config['seed']),
            "--run-tag", f"k8s_hpa_cpu_seed{self.config['seed']}_{timestamp}"
        ]
        
        try:
            if self.use_standardized_scenarios:
                self.log_info("🧪 開始 K8s-HPA 標準化測試...")
                self.log_info(f"📊 使用 {len(self.scenario_config['scenarios'])} 個標準化場景進行測試")
            else:
                self.log_info("🧪 開始 K8s-HPA CPU配置測試...")
                self.log_info("📋 將測試 4 種 CPU 配置: cpu-20, cpu-40, cpu-60, cpu-80")
                self.log_info("📊 每種配置運行 4 個場景，共 16 個測試")
            
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("K8s-HPA 實驗完成")
                return True
            else:
                self.log_error("K8s-HPA 實驗失敗")
                return False
                
        except Exception as e:
            self.log_error(f"K8s-HPA 實驗執行錯誤: {e}")
            return False
    
    def generate_analysis(self) -> bool:
        """生成實驗結果分析"""
        self.log_section("📊 生成實驗結果分析")
        
        try:
            self.log_info("🔍 分析所有實驗結果...")
            
            # 選擇適當的分析腳本
            if self.use_standardized_scenarios:
                analysis_script = self.repo_root / "analyze_onlineboutique_results.py"
                script_name = "analyze_onlineboutique_results.py"
                self.log_info("🎯 使用標準化結果分析器")
            else:
                analysis_script = self.repo_root / "analyze_results.py"
                script_name = "analyze_results.py"
                self.log_info("📊 使用一般結果分析器")
            
            if analysis_script.exists():
                result = subprocess.run([sys.executable, script_name], cwd=self.repo_root)
                if result.returncode != 0:
                    self.log_error("結果分析失敗")
                    return False
            else:
                self.log_info("分析腳本不存在，跳過自動分析")
            
            # 顯示重要檔案位置
            self.log_info("📋 重要結果檔案:")
            print("• 模型檔案: logs/models/")
            
            models_dir = self.repo_root / "logs" / "models"
            if models_dir.exists():
                models = list(models_dir.glob("*.zip"))
                for model in models[:5]:  # 只顯示前5個
                    print(f"  - {model.name}")
            
            print("• TensorBoard: logs/*/tensorboard/")
            print("• 測試結果: logs/*/")
            
            return True
            
        except Exception as e:
            self.log_error(f"結果分析失敗: {e}")
            return False
    
    def run_single_stage(self, stage: str, steps: int = 5000, goal: str = "latency", model: str = "gat") -> bool:
        """執行單一階段"""
        # 更新配置
        self.config.update({
            'steps': steps,
            'goal': goal,
            'model': model
        })
        
        print("\033[0;34m")
        print(f"🎯 執行單一階段: {stage}")
        print(f"📅 時間: {datetime.now().strftime('%Y年 %m月%d日 %H時%M分%S秒')}")
        print(f"📊 步數: {self.config['steps']}")
        print(f"🎯 目標: {self.config['goal']}")
        print("\033[0m")
        
        try:
            if stage == 'plan':
                self.log_section("📋 實驗規劃階段")
                if not self.check_prerequisites():
                    return False
                plan = self.planner.plan_experiments(steps, goal, model, [])
                self.planner.save_plan()
                self.log_success("實驗規劃完成，已保存到 experiment_plan.json")
                return True
                
            elif stage == 'gym-hpa':
                self.log_section("🎯 只執行 Gym-HPA 實驗")
                if not self.check_prerequisites():
                    return False
                    
                # 嘗試載入現有計劃，否則進行快速規劃
                plan = self.planner.load_plan()
                if not plan:
                    self.log_info("未找到現有計劃，進行快速 Gym-HPA 模型檢查...")
                    gym_models = self.planner.find_models('gym_hpa', steps, goal)
                    action, selected_model = self.planner.prompt_user_choice('gym_hpa', gym_models)
                    
                    if action == 'exit':
                        return False
                    elif action == 'skip':
                        self.log_info("⏭️  用戶選擇跳過 Gym-HPA 實驗")
                        return True
                    elif action == 'use_existing':
                        plan = {
                            'gym_hpa': {
                                'skip_experiment': False,
                                'skip_training': True,
                                'model_path': str(selected_model) if selected_model else None
                            }
                        }
                    elif action == 'retrain':
                        plan = {
                            'gym_hpa': {
                                'skip_experiment': False,
                                'skip_training': False,
                                'model_path': None
                            }
                        }
                
                return self.run_gym_hpa_experiment(plan)
                
            elif stage == 'gnnrl':
                self.log_section("🧠 只執行 GNNRL 實驗")
                if not self.check_prerequisites():
                    return False
                    
                # 嘗試載入現有計劃，否則進行快速規劃
                plan = self.planner.load_plan()
                if not plan:
                    self.log_info("未找到現有計劃，進行快速 GNNRL 模型檢查...")
                    gnnrl_models = self.planner.find_models('gnnrl', steps, goal, model)
                    action, selected_model = self.planner.prompt_user_choice('gnnrl', gnnrl_models)
                    
                    if action == 'exit':
                        return False
                    elif action == 'skip':
                        self.log_info("⏭️  用戶選擇跳過 GNNRL 實驗")
                        return True
                    elif action == 'use_existing':
                        plan = {
                            'gnnrl': {
                                'skip_experiment': False,
                                'skip_training': True,
                                'model_path': str(selected_model) if selected_model else None
                            }
                        }
                    elif action == 'retrain':
                        plan = {
                            'gnnrl': {
                                'skip_experiment': False,
                                'skip_training': False,
                                'model_path': None
                            }
                        }
                
                return self.run_gnnrl_experiment(plan)
                
            elif stage == 'k8s-hpa':
                self.log_section("⚖️ 只執行 K8s-HPA 實驗")
                if not self.check_prerequisites():
                    return False
                return self.run_k8s_hpa_experiment()
                
            elif stage == 'analysis':
                self.log_section("📊 只執行結果分析")
                return self.generate_analysis()
                
            else:
                self.log_error(f"未知階段: {stage}")
                return False
                
        except Exception as e:
            self.log_error(f"階段 {stage} 執行失敗: {e}")
            return False
    
    def run_complete_experiment(self, steps: int = 5000, goal: str = "latency", model: str = "gat", skip_stages: set = None) -> bool:
        """執行完整實驗流程"""
        skip_stages = skip_stages or set()
        
        # 更新配置
        self.config.update({
            'steps': steps,
            'goal': goal,
            'model': model
        })
        
        print("\033[0;34m")
        print("🚀 開始完整三方法自動縮放實驗 (Python 版本)")
        print(f"📅 時間: {datetime.now().strftime('%Y年 %m月%d日 %H時%M分%S秒')}")
        print(f"🎲 種子: {self.config['seed']}")
        print(f"📊 步數: {self.config['steps']}")
        print(f"🎯 目標: {self.config['goal']}")
        print(f"🏢 場景: {self.config['use_case']}")
        if skip_stages:
            print(f"⏭️  跳過階段: {', '.join(skip_stages)}")
        print("\033[0m")
        
        start_time = time.time()
        plan = {}
        
        try:
            # 1. 檢查前置條件
            if not self.check_prerequisites():
                return False
            
            # 2. 實驗規劃
            if 'plan' not in skip_stages:
                plan = self.planner.plan_experiments(steps, goal, model, skip_stages)
            else:
                self.log_info("⏭️  跳過實驗規劃階段")
                # 嘗試載入現有計劃
                plan = self.planner.load_plan()
                if not plan:
                    self.log_error("跳過規劃但找不到現有計劃檔案，請先執行規劃階段")
                    return False
            
            # 3. 執行實驗
            if 'gym-hpa' not in skip_stages:
                gym_plan = plan.get('gym_hpa', {})
                if gym_plan.get('skip_experiment', False):
                    self.log_info("⏭️  根據規劃跳過 Gym-HPA 實驗")
                else:
                    if not self.run_gym_hpa_experiment(plan):
                        return False
            else:
                self.log_info("⏭️  跳過 Gym-HPA 實驗")
                
            if 'gnnrl' not in skip_stages:
                gnnrl_plan = plan.get('gnnrl', {})
                if gnnrl_plan.get('skip_experiment', False):
                    self.log_info("⏭️  根據規劃跳過 GNNRL 實驗")
                else:
                    if not self.run_gnnrl_experiment(plan):
                        return False
            else:
                self.log_info("⏭️  跳過 GNNRL 實驗")
                
            if 'k8s-hpa' not in skip_stages:
                k8s_plan = plan.get('k8s_hpa', {})
                if k8s_plan.get('skip_experiment', False):
                    self.log_info("⏭️  根據規劃跳過 K8s-HPA 實驗")
                else:
                    if not self.run_k8s_hpa_experiment():
                        return False
            else:
                self.log_info("⏭️  跳過 K8s-HPA 實驗")
            
            # 4. 生成分析
            if 'analysis' not in skip_stages:
                self.generate_analysis()
            else:
                self.log_info("⏭️  跳過結果分析")
            
            # 計算總時間
            end_time = time.time()
            duration = int(end_time - start_time)
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            seconds = duration % 60
            
            self.log_section("🎉 所有實驗完成!")
            print(f"\033[0;32m總耗時: {hours}時{minutes}分{seconds}秒\033[0m")
            print()
            print("\033[0;36m📈 下一步分析:\033[0m")
            
            if self.use_standardized_scenarios:
                print("1. 查看標準化比較: cat logs/standardized_method_comparison.csv")
                print("2. 查看場景比較: cat logs/standardized_scenario_comparison.csv")
                print("3. 查看負載分析: cat logs/standardized_load_type_analysis.csv")
                print("4. 詳細分析: python analyze_standardized_results.py")
                print("5. 啟動 TensorBoard: tensorboard --logdir logs")
                print("6. 查看場景序列: cat standardized_scenario_sequence.txt")
                print("7. 查看分析報告: cat STANDARDIZED_COMPARISON_REPORT.md")
            else:
                print("1. 查看比較結果: cat logs/experiment_comparison.csv")
                print("2. 啟動 TensorBoard: tensorboard --logdir logs")
                print("3. 詳細分析: python analyze_results.py")
                print("4. 查看測試序列: cat logs/hpa_scenario_sequence.txt")
            
            return True
            
        except KeyboardInterrupt:
            print("\n👋 用戶中斷實驗")
            return False
        except Exception as e:
            self.log_error(f"實驗執行失敗: {e}")
            return False

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='完整三方法自動縮放實驗 (Python 版本)')
    parser.add_argument('--steps', type=int, default=5000, help='訓練步數')
    parser.add_argument('--goal', default='latency', help='目標 (latency/cost)')
    parser.add_argument('--model', default='gat', help='GNNRL 模型類型')
    
    # 標準化場景選項
    parser.add_argument('--standardized', action='store_true', 
                       help='使用標準化的8個場景確保公平比較 (推薦用於方法對比)')
    
    # 階段選擇功能
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument('--stage', choices=['plan', 'gym-hpa', 'gnnrl', 'k8s-hpa', 'analysis'], 
                           help='只執行特定階段 (plan=規劃, gym-hpa=Gym-HPA實驗, gnnrl=GNNRL實驗, k8s-hpa=K8s-HPA實驗, analysis=結果分析)')
    stage_group.add_argument('--skip-stages', nargs='+', 
                           choices=['plan', 'gym-hpa', 'gnnrl', 'k8s-hpa', 'analysis'],
                           help='跳過指定階段')
    
    args = parser.parse_args()
    
    # 如果使用標準化場景，顯示說明
    if args.standardized:
        print("🎯 使用標準化場景模式")
        print("✅ 確保三種方法測試相同的8個場景，提供公平比較")
        print("📊 場景分佈: 2個offpeak + 2個peak + 2個rushsale + 2個fluctuating")
        print("🎲 基於固定種子生成，結果可重現")
        print()
    
    runner = ExperimentRunner(use_standardized_scenarios=args.standardized)
    
    # 處理階段選擇
    if args.stage:
        success = runner.run_single_stage(args.stage, args.steps, args.goal, args.model)
    else:
        skip_stages = set(args.skip_stages) if args.skip_stages else set()
        success = runner.run_complete_experiment(args.steps, args.goal, args.model, skip_stages)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()