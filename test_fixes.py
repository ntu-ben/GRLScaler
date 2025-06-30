#!/usr/bin/env python3
"""
測試修復後的功能
"""

import subprocess
import sys
from pathlib import Path

def test_stage_functionality():
    """測試階段選擇功能"""
    print("🧪 測試階段選擇功能")
    print("=" * 50)
    
    # 測試 help
    print("\n1. 測試 help 功能:")
    result = subprocess.run([sys.executable, "run_complete_experiment.py", "--help"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Help 功能正常")
    else:
        print("❌ Help 功能異常")
        print(result.stderr)
    
    # 測試 stage 參數驗證
    print("\n2. 測試 stage 參數驗證:")
    result = subprocess.run([sys.executable, "run_complete_experiment.py", "--stage", "invalid"], 
                          capture_output=True, text=True)
    if result.returncode != 0 and "invalid choice" in result.stderr:
        print("✅ Stage 參數驗證正常")
    else:
        print("❌ Stage 參數驗證異常")
    
    print("\n3. 測試模型檢測:")
    try:
        from experiment_planner import ExperimentPlanner
        planner = ExperimentPlanner()
        
        # 測試 Gym-HPA 模型檢測
        gym_models = planner.find_models('gym_hpa', 5000, 'latency')
        print(f"✅ Gym-HPA 模型檢測: 找到 {len(gym_models)} 個模型")
        
        # 測試 GNNRL 模型檢測
        gnnrl_models = planner.find_models('gnnrl', 5000, 'latency', 'gat')
        print(f"✅ GNNRL 模型檢測: 找到 {len(gnnrl_models)} 個模型")
        
    except Exception as e:
        print(f"❌ 模型檢測異常: {e}")

def check_unified_manager_fix():
    """檢查 unified_experiment_manager.py 中的修復"""
    print("\n🔧 檢查 K8s-HPA 修復")
    print("=" * 30)
    
    try:
        # 檢查修復是否存在
        with open("unified_experiment_manager.py", 'r') as f:
            content = f.read()
            
        # 尋找修復後的程式碼
        if "remote_tag = f\"{experiment_type}/{run_tag}/{config_name}\"" in content:
            print("✅ K8s-HPA run_distributed_locust 修復已套用")
        else:
            print("❌ 未找到 K8s-HPA 修復")
            
        # 檢查函數定義
        if "def run_distributed_locust(self, scenario: str, tag: str, out_dir: Path)" in content:
            print("✅ run_distributed_locust 函數定義正確")
        else:
            print("❌ run_distributed_locust 函數定義可能有問題")
            
    except Exception as e:
        print(f"❌ 檢查修復時發生錯誤: {e}")

def show_usage_examples():
    """顯示使用範例"""
    print("\n📋 使用範例")
    print("=" * 20)
    
    examples = [
        ("只執行 K8s-HPA", "python run_complete_experiment.py --stage k8s-hpa"),
        ("只進行規劃", "python run_complete_experiment.py --stage plan"),
        ("跳過規劃執行", "python run_complete_experiment.py --skip-stages plan"),
        ("只做分析", "python run_complete_experiment.py --stage analysis"),
        ("完整流程", "python run_complete_experiment.py")
    ]
    
    for desc, cmd in examples:
        print(f"• {desc}:")
        print(f"  {cmd}")
    
    print(f"\n📖 詳細說明請參考: USAGE_GUIDE.md")

if __name__ == "__main__":
    print("🔍 測試修復後的實驗管理系統")
    print("=" * 60)
    
    test_stage_functionality()
    check_unified_manager_fix()
    show_usage_examples()
    
    print("\n✅ 測試完成！")
    print("\n🚀 現在可以嘗試:")
    print("python run_complete_experiment.py --stage k8s-hpa")