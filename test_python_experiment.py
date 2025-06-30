#!/usr/bin/env python3
"""
測試 Python 版本的實驗管理器
"""

from experiment_planner import ExperimentPlanner
from pathlib import Path

def test_model_detection():
    """測試模型檢測功能"""
    print("🧪 測試 Python 版本的實驗規劃器")
    print("=" * 50)
    
    planner = ExperimentPlanner()
    
    # 測試 Gym-HPA 模型檢測
    print("\n📦 測試 Gym-HPA 模型檢測:")
    gym_models = planner.find_models('gym_hpa', 5000, 'latency')
    print(f"找到 {len(gym_models)} 個 Gym-HPA 模型:")
    for model in gym_models:
        info = planner.format_file_info(model)
        print(f"  ✅ {model.name}")
        print(f"     大小: {info['size']}, 時間: {info['time']}")
    
    # 測試 GNNRL 模型檢測
    print("\n🧠 測試 GNNRL 模型檢測:")
    gnnrl_models = planner.find_models('gnnrl', 5000, 'latency', 'gat')
    print(f"找到 {len(gnnrl_models)} 個 GNNRL 模型:")
    for model in gnnrl_models:
        info = planner.format_file_info(model)
        print(f"  ✅ {model.name}")
        print(f"     大小: {info['size']}, 時間: {info['time']}")
    
    if not gnnrl_models:
        print("  ❌ 未找到 GNNRL 模型 - 將需要新訓練")
    
    print("\n🔍 模型搜尋模式測試:")
    print("Gym-HPA 搜尋模式:", planner.experiments['gym_hpa']['search_pattern'].format(steps=5000))
    print("GNNRL 搜尋模式:", planner.experiments['gnnrl']['search_pattern'].format(steps=5000))
    
    print("\n✅ 測試完成！")
    print("\n📋 使用方法:")
    print("1. 🚀 運行完整實驗: python run_complete_experiment.py")
    print("2. 🔧 只規劃實驗: python experiment_planner.py")
    print("3. 📊 自定義參數: python run_complete_experiment.py --steps 3000 --goal cost")

if __name__ == "__main__":
    test_model_detection()