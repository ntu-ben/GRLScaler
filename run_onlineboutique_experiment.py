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
    def __init__(self, repo_root: Path = None, use_standardized_scenarios: bool = False,
                 algorithm: str = 'ppo', stable_loadtest: bool = False, 
                 max_rps: int = None, loadtest_timeout: int = 30):
        self.repo_root = repo_root or Path(__file__).parent
        self.planner = ExperimentPlanner(repo_root)
        self.use_standardized_scenarios = use_standardized_scenarios
        self.stable_loadtest = stable_loadtest
        self.max_rps = max_rps
        self.loadtest_timeout = loadtest_timeout
        
        # 預設配置（支援A2C）
        self.config = {
            'seed': 42,
            'steps': 5000,
            'goal': 'latency',
            'use_case': 'online_boutique',
            'model': 'gat',
            'alg': algorithm or 'ppo'  # 支援傳入的算法
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
        
        # 添加 stable loadtest 參數
        if self.stable_loadtest:
            cmd.append("--stable-loadtest")
        if self.max_rps:
            cmd.extend(["--target-rps", str(self.max_rps)])
        if self.loadtest_timeout:
            cmd.extend(["--loadtest-timeout", str(self.loadtest_timeout)])
        
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
        
        # 添加 stable loadtest 參數
        if self.stable_loadtest:
            cmd.append("--stable-loadtest")
        if self.max_rps:
            cmd.extend(["--target-rps", str(self.max_rps)])
        if self.loadtest_timeout:
            cmd.extend(["--loadtest-timeout", str(self.loadtest_timeout)])
        
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
        
        # 添加 stable loadtest 參數
        if self.stable_loadtest:
            cmd.append("--stable-loadtest")
        if self.max_rps:
            cmd.extend(["--target-rps", str(self.max_rps)])
        if self.loadtest_timeout:
            cmd.extend(["--loadtest-timeout", str(self.loadtest_timeout)])
        
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

    def run_complete_experiment_with_scenario_selection(self, steps: int = 5000, goal: str = "latency", model: str = "gat") -> bool:
        """執行完整實驗流程（支援場景選擇）"""
        
        # 更新配置
        self.config.update({
            'steps': steps,
            'goal': goal,
            'model': model
        })
        
        print("\\033[0;34m")
        print("🚀 開始 Online Boutique 自動擴展實驗 (場景選擇模式)")
        print(f"📅 時間: {datetime.now().strftime('%Y年 %m月%d日 %H時%M分%S秒')}")
        print(f"🎲 種子: {self.config['seed']}")
        print(f"📊 步數: {self.config['steps']}")
        print(f"🎯 目標: {self.config['goal']}")
        print(f"🏢 場景: {self.config['use_case']}")
        print("\\033[0m")
        
        try:
            # 1. 檢查前置條件
            if not self.check_prerequisites():
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
            
            # K8s-HPA 選擇
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
            
            # 3. 執行實驗（使用統一實驗管理器）
            experiment_results = {}
            
            # Gym-HPA 實驗
            if 'gym_hpa' in experiment_plan:
                experiment_results['gym_hpa'] = self.run_gym_hpa_experiment_with_mode(experiment_plan['gym_hpa'])
            
            # GNNRL 實驗  
            if 'gnnrl' in experiment_plan:
                experiment_results['gnnrl'] = self.run_gnnrl_experiment_with_mode(experiment_plan['gnnrl'])
            
            # K8s-HPA 實驗
            if 'k8s_hpa' in experiment_plan:
                experiment_results['k8s_hpa'] = self.run_k8s_hpa_experiment_with_scenarios(experiment_plan['k8s_hpa'].get('scenarios', ['all']))
            
            self.log_section("🎉 Online Boutique 實驗完成!")
            
            # 顯示結果摘要
            self.log_info("📊 實驗結果摘要:")
            for method, success in experiment_results.items():
                status = "✅ 成功" if success else "❌ 失敗"
                self.log_info(f"   {method.upper()}: {status}")
            
            return True
            
        except Exception as e:
            self.log_error(f"Online Boutique 實驗執行失敗: {e}")
            return False
    
    def run_gym_hpa_experiment_with_mode(self, config: dict) -> bool:
        """根據模式執行 Gym-HPA 實驗（使用統一實驗管理器）"""
        mode = config['mode']
        scenarios = config.get('scenarios', ['all'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if mode == 'train':
            self.log_section("🎯 Gym-HPA 訓練模式 (Online Boutique)")
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gym_hpa",
                "--k8s",
                "--use-case", "online_boutique",
                "--goal", self.config['goal'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", str(self.config['steps']),
                "--run-tag", f"gym_hpa_ob_train_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"
            ]
            
            # 如果有特定場景，添加場景參數
            if 'all' not in scenarios:
                cmd.extend(["--scenarios", ",".join(scenarios)])
            
        elif mode == 'test':
            self.log_section("🎯 Gym-HPA 測試模式 (Online Boutique)")
            
            model_path = config.get('model_path', 'auto')
            if model_path == 'auto':
                model_path = self._find_latest_model('gym_hpa')
                if not model_path:
                    self.log_error("❌ 找不到 Gym-HPA 模型進行測試")
                    return False
                self.log_info(f"🔍 自動找到模型: {Path(model_path).name}")
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gym_hpa",
                "--k8s",
                "--use-case", "online_boutique",
                "--goal", self.config['goal'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", "0",
                "--testing",
                "--load-path", model_path,
                "--run-tag", f"gym_hpa_ob_test_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"
            ]
            
            # 如果有特定場景，添加場景參數
            if 'all' not in scenarios:
                cmd.extend(["--scenarios", ",".join(scenarios)])
            
        elif mode == 'both':
            self.log_section("🎯 Gym-HPA 訓練+測試模式 (Online Boutique)")
            
            # 先執行訓練
            train_success = self.run_gym_hpa_experiment_with_mode({'mode': 'train', 'scenarios': scenarios})
            if not train_success:
                return False
            
            # 等待一段時間再執行測試
            self.log_info("⏱️ 訓練完成，等待 30 秒後開始測試...")
            time.sleep(30)
            
            # 再執行測試
            test_config = {'mode': 'test', 'model_path': 'auto', 'scenarios': scenarios}
            return self.run_gym_hpa_experiment_with_mode(test_config)
        
        else:
            self.log_error(f"❌ 未知的 Gym-HPA 模式: {mode}")
            return False
        
        try:
            self.log_info("🧪 開始 Gym-HPA Online Boutique 實驗...")
            self.log_info(f"📋 命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("✅ Gym-HPA Online Boutique 實驗完成")
                return True
            else:
                self.log_error(f"❌ Gym-HPA Online Boutique 實驗失敗 (退出碼: {result.returncode})")
                return False
                
        except Exception as e:
            self.log_error(f"❌ Gym-HPA Online Boutique 實驗執行錯誤: {e}")
            return False
    
    def run_gnnrl_experiment_with_mode(self, config: dict) -> bool:
        """根據模式執行 GNNRL 實驗（使用統一實驗管理器）"""
        mode = config['mode']
        scenarios = config.get('scenarios', ['all'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if mode == 'train':
            self.log_section("🧠 GNNRL 訓練模式 (Online Boutique)")
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gnnrl",
                "--k8s",
                "--use-case", "online_boutique",
                "--goal", self.config['goal'],
                "--model", self.config['model'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", str(self.config['steps']),
                "--run-tag", f"gnnrl_ob_train_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"
            ]
            
            # 如果有特定場景，添加場景參數
            if 'all' not in scenarios:
                cmd.extend(["--scenarios", ",".join(scenarios)])
            
        elif mode == 'test':
            self.log_section("🧠 GNNRL 測試模式 (Online Boutique)")
            
            model_path = config.get('model_path', 'auto')
            if model_path == 'auto':
                model_path = self._find_latest_model('gnnrl')
                if not model_path:
                    self.log_error("❌ 找不到 GNNRL 模型進行測試")
                    return False
                self.log_info(f"🔍 自動找到模型: {Path(model_path).name}")
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "gnnrl",
                "--k8s",
                "--use-case", "online_boutique",
                "--goal", self.config['goal'],
                "--model", self.config['model'],
                "--alg", self.config['alg'],
                "--seed", str(self.config['seed']),
                "--steps", "0",
                "--testing",
                "--load-path", model_path,
                "--run-tag", f"gnnrl_ob_test_seed{self.config['seed']}_{timestamp}",
                "--enable-loadtest"
            ]
            
            # 如果有特定場景，添加場景參數
            if 'all' not in scenarios:
                cmd.extend(["--scenarios", ",".join(scenarios)])
            
        elif mode == 'both':
            self.log_section("🧠 GNNRL 訓練+測試模式 (Online Boutique)")
            
            # 先執行訓練
            train_success = self.run_gnnrl_experiment_with_mode({'mode': 'train', 'scenarios': scenarios})
            if not train_success:
                return False
            
            # 等待一段時間再執行測試
            self.log_info("⏱️ 訓練完成，等待 30 秒後開始測試...")
            time.sleep(30)
            
            # 再執行測試
            test_config = {'mode': 'test', 'model_path': 'auto', 'scenarios': scenarios}
            return self.run_gnnrl_experiment_with_mode(test_config)
        
        else:
            self.log_error(f"❌ 未知的 GNNRL 模式: {mode}")
            return False
        
        try:
            self.log_info("🧪 開始 GNNRL Online Boutique 實驗...")
            self.log_info(f"📋 命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.repo_root)
            
            if result.returncode == 0:
                self.log_success("✅ GNNRL Online Boutique 實驗完成")
                return True
            else:
                self.log_error(f"❌ GNNRL Online Boutique 實驗失敗 (退出碼: {result.returncode})")
                return False
                
        except Exception as e:
            self.log_error(f"❌ GNNRL Online Boutique 實驗執行錯誤: {e}")
            return False
    
    def run_k8s_hpa_experiment_with_scenarios(self, selected_scenarios: list) -> bool:
        """執行 K8s-HPA 實驗（支援場景選擇）"""
        self.log_section("⚖️ K8s-HPA (Online Boutique) - 場景選擇模式")
        
        if 'all' in selected_scenarios:
            self.log_info("📊 執行所有場景的 K8s-HPA 測試")
            return self.run_k8s_hpa_experiment()
        else:
            self.log_info(f"📊 執行選定場景的 K8s-HPA 測試: {', '.join(selected_scenarios)}")
            # 使用統一實驗管理器執行選定場景
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            cmd = [
                sys.executable, "unified_experiment_manager.py",
                "--experiment", "k8s_hpa",
                "--hpa-type", "cpu",
                "--seed", str(self.config['seed']),
                "--run-tag", f"k8s_hpa_cpu_seed{self.config['seed']}_{timestamp}",
                "--scenarios", ",".join(selected_scenarios)
            ]
            
            try:
                self.log_info("🧪 開始 K8s-HPA Online Boutique 場景測試...")
                self.log_info(f"📋 命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=self.repo_root)
                
                if result.returncode == 0:
                    self.log_success("✅ K8s-HPA Online Boutique 實驗完成")
                    return True
                else:
                    self.log_error("❌ K8s-HPA Online Boutique 實驗失敗")
                    return False
                    
            except Exception as e:
                self.log_error(f"❌ K8s-HPA Online Boutique 實驗執行錯誤: {e}")
                return False
    
    def _find_latest_model(self, method: str) -> str:
        """尋找最新的訓練模型"""
        models_dir = self.repo_root / "logs" / "models"
        if not models_dir.exists():
            return None
        
        # 尋找對應方法的模型
        pattern = f"*{method}*online_boutique*.zip"
        models = list(models_dir.glob(pattern))
        
        if not models:
            # 嘗試其他模式
            pattern = f"*{method}*.zip"
            models = list(models_dir.glob(pattern))
        
        if not models:
            return None
        
        # 按修改時間排序，返回最新的
        latest_model = max(models, key=lambda x: x.stat().st_mtime)
        return str(latest_model)

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