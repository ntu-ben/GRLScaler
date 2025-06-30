#!/usr/bin/env python3
"""
測試新增的跳過實驗功能
"""

from experiment_planner import ExperimentPlanner

def demo_new_choice_options():
    """演示新的選擇選項"""
    print("🎯 新功能演示: 跳過特定實驗")
    print("=" * 50)
    
    print("現在在實驗規劃時，你有更多選擇:")
    print()
    
    print("📋 如果有現有模型:")
    print("  1) 使用現有模型 (跳過訓練)")
    print("  2) 重新訓練新模型")
    print("  3) 跳過此實驗  ← 🆕 新功能!")
    print("  4) 退出實驗")
    print()
    
    print("📋 如果沒有現有模型:")
    print("  1) 進行新訓練")
    print("  2) 跳過此實驗  ← 🆕 新功能!")
    print("  3) 退出實驗")
    print()
    
    print("🎯 使用場景:")
    print("• 只想測試 K8s-HPA → 跳過 Gym-HPA 和 GNNRL")
    print("• 只想測試 Gym-HPA → 跳過 GNNRL")
    print("• 只想測試 GNNRL → 跳過 Gym-HPA")
    print("• 快速基準測試 → 跳過所有 ML 實驗")
    print()
    
    print("📊 執行計劃摘要會顯示:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 實驗項目    │ 模型來源      │ 狀態                    │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ Gym-HPA     │ 跳過實驗      │ ⏭️  完全跳過              │")
    print("│ GNNRL       │ 使用現有模型  │ 跳過訓練，直接測試      │")
    print("│ K8s-HPA     │ 無需模型      │ 直接基準測試            │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    print("🚀 現在試試:")
    print("python run_complete_experiment.py")

if __name__ == "__main__":
    demo_new_choice_options()