#!/usr/bin/env python3
"""
實驗規劃器 - 檢查現有模型並規劃實驗執行
==================================================

替代原本的 bash 腳本邏輯，提供更穩定和用戶友好的模型檢查和實驗規劃功能。
"""

import os
import sys
import glob
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ExperimentPlanner:
    def __init__(self, repo_root: Path = None):
        self.repo_root = repo_root or Path(__file__).parent
        self.models_dir = self.repo_root / "logs" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 實驗配置
        self.experiments = {
            'gym_hpa': {
                'name': 'Gym-HPA (基礎強化學習)',
                'pattern': 'ppo_env_online_boutique_gym_goal_{goal}_k8s_True_totalSteps_{steps}.zip',
                'search_pattern': '*online_boutique_gym*{steps}*.zip'
            },
            'gnnrl': {
                'name': 'GNNRL (圖神經網路強化學習)', 
                'pattern': 'gnnrl_{model}_{goal}_k8s_*_steps_{steps}.zip',
                'search_pattern': 'gnnrl_{model}_*{steps}*.zip'
            },
            'k8s_hpa': {
                'name': 'K8s-HPA (原生HPA基準測試)',
                'pattern': None,  # K8s-HPA 不需要模型檔案
                'search_pattern': None
            }
        }
        
        # 實驗決策結果
        self.plan = {}
        
    def find_models(self, experiment: str, steps: int, goal: str = "latency", model: str = "gat") -> List[Path]:
        """尋找指定實驗的現有模型"""
        if experiment not in self.experiments:
            return []
        
        # K8s-HPA 不需要模型檔案
        if experiment == 'k8s_hpa':
            return []
            
        search_pattern = self.experiments[experiment]['search_pattern']
        if not search_pattern:
            return []
            
        # 使用 search_pattern 來尋找模型
        search_pattern = search_pattern.format(
            steps=steps, goal=goal, model=model
        )
        
        pattern_path = self.models_dir / search_pattern
        found_files = glob.glob(str(pattern_path))
        
        # 轉換為 Path 對象並按修改時間排序（最新的在前）
        models = [Path(f) for f in found_files]
        models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return models
    
    def format_file_info(self, model_path: Path) -> Dict[str, str]:
        """格式化檔案資訊"""
        stat = model_path.stat()
        
        # 檔案大小
        size_bytes = stat.st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes // 1024}K"
        else:
            size_str = f"{size_bytes // (1024 * 1024)}M"
            
        # 修改時間
        mtime = datetime.fromtimestamp(stat.st_mtime)
        time_str = mtime.strftime("%m月%d日 %H:%M")
        
        return {
            'size': size_str,
            'time': time_str,
            'path': str(model_path)
        }
    
    def prompt_user_choice(self, experiment: str, models: List[Path]) -> Tuple[str, Optional[Path]]:
        """提示用戶選擇
        
        Returns:
            Tuple[str, Optional[Path]]: (action, model_path)
            action 可能的值: 'use_existing', 'retrain', 'skip', 'exit'
        """
        exp_name = self.experiments[experiment]['name']
        
        # K8s-HPA 特殊處理
        if experiment == 'k8s_hpa':
            print(f"📋 {exp_name} 不需要訓練模型，將直接進行基準測試")
            print(f"請選擇操作:")
            print(f"  1) 進行 K8s-HPA 基準測試")
            print(f"  2) 跳過此實驗")
            print(f"  3) 退出實驗")
            
            while True:
                try:
                    choice = input("請輸入選擇 [1-3]: ").strip()
                    
                    if choice == '1':
                        print(f"🔄 將進行 {exp_name} 基準測試")
                        return 'retrain', None  # 對K8s-HPA來說，這意味著運行測試
                    elif choice == '2':
                        print(f"⏭️  將跳過 {exp_name} 實驗")
                        return 'skip', None
                    elif choice == '3':
                        print("👋 用戶選擇退出實驗")
                        return 'exit', None
                    else:
                        print("❌ 無效選擇，請輸入 1-3")
                        
                except KeyboardInterrupt:
                    print("\n👋 用戶中斷實驗")
                    return 'exit', None
        
        if not models:
            print(f"❌ 未找到現有的 {exp_name} 模型")
            print(f"請選擇操作:")
            print(f"  1) 進行新訓練")
            print(f"  2) 跳過此實驗")
            print(f"  3) 退出實驗")
            
            while True:
                try:
                    choice = input("請輸入選擇 [1-3]: ").strip()
                    
                    if choice == '1':
                        print(f"🔄 將進行 {exp_name} 新訓練")
                        return 'retrain', None
                    elif choice == '2':
                        print(f"⏭️  將跳過 {exp_name} 實驗")
                        return 'skip', None
                    elif choice == '3':
                        print("👋 用戶選擇退出實驗")
                        return 'exit', None
                    else:
                        print("❌ 無效選擇，請輸入 1、2 或 3")
                        
                except KeyboardInterrupt:
                    print("\n👋 用戶中斷，退出實驗")
                    return 'exit', None
                except EOFError:
                    print("\n👋 輸入結束，退出實驗")
                    return 'exit', None
            
        print(f"\n🔍 發現現有的 {exp_name} 模型:")
        for i, model in enumerate(models, 1):
            info = self.format_file_info(model)
            print(f"  [{i}] {model.name}")
            print(f"      大小: {info['size']}")
            print(f"      時間: {info['time']}")
        
        print(f"\n請選擇操作:")
        print(f"  1) 使用現有模型 (跳過訓練)")
        print(f"  2) 重新訓練新模型")
        print(f"  3) 跳過此實驗")
        print(f"  4) 退出實驗")
        
        while True:
            try:
                choice = input("請輸入選擇 [1-4]: ").strip()
                
                if choice == '1':
                    # 使用現有模型
                    if len(models) == 1:
                        selected_model = models[0]
                    else:
                        # 多個模型，讓用戶選擇
                        while True:
                            try:
                                model_choice = input(f"請選擇模型編號 [1-{len(models)}]: ").strip()
                                model_idx = int(model_choice) - 1
                                if 0 <= model_idx < len(models):
                                    selected_model = models[model_idx]
                                    break
                                else:
                                    print(f"❌ 無效選擇，請輸入 1-{len(models)}")
                            except ValueError:
                                print(f"❌ 請輸入數字 1-{len(models)}")
                    
                    # 驗證模型檔案存在
                    if selected_model.exists():
                        print(f"✅ 將使用模型: {selected_model.name}")
                        return 'use_existing', selected_model
                    else:
                        print(f"❌ 模型檔案不存在: {selected_model}")
                        print(f"🔄 自動切換為重新訓練模式")
                        return 'retrain', None
                        
                elif choice == '2':
                    print(f"🔄 將重新訓練 {exp_name} 模型")
                    return 'retrain', None
                    
                elif choice == '3':
                    print(f"⏭️  將跳過 {exp_name} 實驗")
                    return 'skip', None
                    
                elif choice == '4':
                    print("👋 用戶選擇退出實驗")
                    return 'exit', None
                    
                else:
                    print("❌ 無效選擇，請輸入 1、2、3 或 4")
                    
            except KeyboardInterrupt:
                print("\n👋 用戶中斷，退出實驗")
                return 'exit', None
            except EOFError:
                print("\n👋 輸入結束，退出實驗")
                return 'exit', None
    
    def plan_experiments(self, steps: int = 5000, goal: str = "latency", model: str = "gat", skip_stages: List[str] = None) -> Dict:
        """規劃所有實驗"""
        print("=" * 50)
        print("📋 實驗規劃和模型檢查")
        print("=" * 50)
        print("檢查現有模型並規劃實驗...")
        
        if skip_stages is None:
            skip_stages = []
        
        plan = {}
        
        # 檢查每個實驗
        for exp_key, exp_config in self.experiments.items():
            # 如果實驗在跳過列表中，自動跳過
            exp_key_with_dash = exp_key.replace('_', '-')
            if exp_key_with_dash in skip_stages:
                print(f"\n{'=' * 20} {exp_config['name']} {'=' * 20}")
                print(f"⏭️  根據命令行參數跳過 {exp_config['name']}")
                plan[exp_key] = {
                    'skip_experiment': True,
                    'skip_training': False,
                    'model_path': None,
                    'experiment_name': exp_config['name']
                }
                continue
                
            print(f"\n{'=' * 20} {exp_config['name']} {'=' * 20}")
            
            models = self.find_models(exp_key, steps, goal, model)
            action, selected_model = self.prompt_user_choice(exp_key, models)
            
            if action == 'exit':
                print("👋 退出實驗規劃")
                sys.exit(0)
            elif action == 'skip':
                plan[exp_key] = {
                    'skip_experiment': True,
                    'skip_training': False,
                    'model_path': None,
                    'experiment_name': exp_config['name']
                }
            elif action == 'use_existing':
                plan[exp_key] = {
                    'skip_experiment': False,
                    'skip_training': True,
                    'model_path': str(selected_model) if selected_model else None,
                    'experiment_name': exp_config['name']
                }
            elif action == 'retrain':
                plan[exp_key] = {
                    'skip_experiment': False,
                    'skip_training': False,
                    'model_path': None,
                    'experiment_name': exp_config['name']
                }
        
        # 顯示執行計劃摘要
        self.show_plan_summary(plan)
        
        # 確認執行
        input("\n按 Enter 繼續執行實驗，或 Ctrl+C 取消...")
        
        self.plan = plan
        return plan
    
    def show_plan_summary(self, plan: Dict):
        """顯示實驗執行計劃摘要"""
        print(f"\n📊 實驗執行計劃摘要:")
        print("┌─────────────────────────────────────────────────────────┐")
        print("│ 實驗項目    │ 模型來源      │ 狀態                    │")
        print("├─────────────────────────────────────────────────────────┤")
        
        for exp_key, exp_plan in plan.items():
            exp_name = exp_plan['experiment_name']
            
            if exp_plan.get('skip_experiment', False):
                print(f"│ {exp_name:11} │ 跳過實驗      │ ⏭️  完全跳過              │")
            elif exp_plan.get('skip_training', False):
                model_name = Path(exp_plan['model_path']).name if exp_plan['model_path'] else "未知"
                print(f"│ {exp_name:11} │ 使用現有模型  │ 跳過訓練，直接測試      │")
                print(f"│             │ {model_name[:13]:13} │                         │")
            else:
                # K8s-HPA 特殊處理
                if exp_key == 'k8s_hpa':
                    print(f"│ {exp_name:11} │ 無需模型      │ 直接基準測試            │")
                else:
                    print(f"│ {exp_name:11} │ 新訓練模型    │ 完整訓練 + 測試         │")
        print("└─────────────────────────────────────────────────────────┘")
    
    def save_plan(self, output_file: Path = None):
        """保存實驗計劃到檔案"""
        if output_file is None:
            output_file = self.repo_root / "experiment_plan.json"
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.plan, f, ensure_ascii=False, indent=2)
            
        print(f"📁 實驗計劃已保存到: {output_file}")
    
    def load_plan(self, input_file: Path = None) -> Dict:
        """從檔案載入實驗計劃"""
        if input_file is None:
            input_file = self.repo_root / "experiment_plan.json"
            
        if not input_file.exists():
            return {}
            
        with open(input_file, 'r', encoding='utf-8') as f:
            self.plan = json.load(f)
            
        return self.plan

def main():
    """主函數 - 可以獨立運行進行實驗規劃"""
    import argparse
    
    parser = argparse.ArgumentParser(description='實驗規劃器')
    parser.add_argument('--steps', type=int, default=5000, help='訓練步數')
    parser.add_argument('--goal', default='latency', help='目標 (latency/cost)')
    parser.add_argument('--model', default='gat', help='GNNRL 模型類型')
    parser.add_argument('--save-plan', action='store_true', help='保存實驗計劃')
    
    args = parser.parse_args()
    
    planner = ExperimentPlanner()
    plan = planner.plan_experiments(args.steps, args.goal, args.model)
    
    if args.save_plan:
        planner.save_plan()
    
    print("\n✅ 實驗規劃完成！")
    print("現在可以執行對應的實驗腳本。")

if __name__ == "__main__":
    main()