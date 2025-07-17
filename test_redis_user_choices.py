#!/usr/bin/env python3
"""
測試 Redis 實驗用戶選擇功能
==========================

測試新的用戶選擇系統，包括訓練/測試選項。
"""

import sys
from unittest.mock import patch
from run_redis_experiment import RedisExperimentRunner

def test_user_experiment_choice():
    """測試用戶實驗選擇功能"""
    print("🧪 測試用戶實驗選擇功能...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試各種輸入
    test_cases = [
        ('1', ('train', True)),
        ('train', ('train', True)),
        ('訓練', ('train', True)),
        ('2', ('test', True)),
        ('test', ('test', True)),
        ('測試', ('test', True)),
        ('3', ('both', True)),
        ('both', ('both', True)),
        ('兩者', ('both', True)),
        ('4', ('skip', False)),
        ('skip', ('skip', False)),
        ('跳過', ('skip', False))
    ]
    
    for input_val, expected in test_cases:
        with patch('builtins.input', return_value=input_val):
            with patch('builtins.print'):  # 抑制輸出
                result = runner.ask_user_experiment_choice("TestMethod")
                if result == expected:
                    print(f"✅ 輸入 '{input_val}' -> {expected}")
                else:
                    print(f"❌ 輸入 '{input_val}' -> 期望 {expected}, 得到 {result}")
                    return False
    
    return True

def test_model_path_choice():
    """測試模型路徑選擇功能"""
    print("🧪 測試模型路徑選擇功能...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試自動模式
    with patch('builtins.input', return_value='1'):
        with patch('builtins.print'):
            result = runner.ask_model_path_if_needed("TestMethod")
            if result == 'auto':
                print("✅ 自動模式選擇正確")
            else:
                print(f"❌ 自動模式失敗: {result}")
                return False
    
    # 測試手動模式
    test_path = "/path/to/model.zip"
    with patch('builtins.input', side_effect=['2', test_path]):
        with patch('builtins.print'):
            result = runner.ask_model_path_if_needed("TestMethod")
            if result == test_path:
                print("✅ 手動模式選擇正確")
            else:
                print(f"❌ 手動模式失敗: {result}")
                return False
    
    return True

def test_find_latest_model():
    """測試模型查找功能"""
    print("🧪 測試模型查找功能...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試查找 gym_hpa 模型（可能不存在）
    gym_model = runner.find_latest_model('gym_hpa')
    print(f"🔍 Gym-HPA 模型: {gym_model if gym_model else '未找到'}")
    
    # 測試查找 gnnrl 模型（可能不存在）
    gnnrl_model = runner.find_latest_model('gnnrl')
    print(f"🔍 GNNRL 模型: {gnnrl_model if gnnrl_model else '未找到'}")
    
    print("✅ 模型查找功能正常")
    return True

def main():
    """執行所有測試"""
    print("🚀 開始 Redis 用戶選擇功能測試")
    print("=" * 50)
    
    tests = [
        ("用戶實驗選擇", test_user_experiment_choice),
        ("模型路徑選擇", test_model_path_choice),
        ("模型查找功能", test_find_latest_model),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 測試 {test_name}...")
        results[test_name] = test_func()
        print()
    
    # 顯示結果摘要
    print("=" * 50)
    print("📊 測試結果摘要:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有用戶選擇功能測試都通過！")
        print("\n💡 新的 Redis 實驗運行方式:")
        print("   python run_redis_experiment.py --algorithm a2c --steps 5000")
        print("   系統會詢問每種方法要執行訓練、測試還是跳過")
    else:
        print("⚠️ 某些測試失敗，請檢查上述錯誤信息。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)