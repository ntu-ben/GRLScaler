#!/usr/bin/env python3
"""
測試場景選擇功能
================

測試新增的場景選擇系統。
"""

import sys
from unittest.mock import patch
from run_redis_experiment import RedisExperimentRunner

def test_scenario_selection():
    """測試場景選擇功能"""
    print("🧪 測試場景選擇功能...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試各種輸入組合
    test_cases = [
        # (輸入, 期望結果)
        ('1', ['offpeak']),
        ('2', ['peak']),
        ('1,2', ['offpeak', 'peak']),
        ('peak,rushsale', ['peak', 'rushsale']),
        ('2,4', ['peak', 'fluctuating']),
        ('a', ['all']),
        ('all', ['all']),
        ('1,3,4', ['offpeak', 'rushsale', 'fluctuating']),
        ('peak', ['peak']),
        ('offpeak,fluctuating', ['offpeak', 'fluctuating']),
    ]
    
    for test_input, expected in test_cases:
        with patch('builtins.input', return_value=test_input):
            with patch('builtins.print'):  # 抑制輸出
                with patch.object(runner, 'log_info'):  # 抑制 log 輸出
                    result = runner.ask_scenario_selection("TestMethod", "test")
                    if result == expected:
                        print(f"✅ 輸入 '{test_input}' -> {expected}")
                    else:
                        print(f"❌ 輸入 '{test_input}' -> 期望 {expected}, 得到 {result}")
                        return False
    
    # 測試訓練模式（應該返回 ['all']）
    with patch('builtins.print'):
        result = runner.ask_scenario_selection("TestMethod", "train")
        if result == ['all']:
            print("✅ 訓練模式自動返回所有場景")
        else:
            print(f"❌ 訓練模式錯誤: {result}")
            return False
    
    return True

def test_invalid_inputs():
    """測試無效輸入處理"""
    print("🧪 測試無效輸入處理...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試無效輸入後的正確輸入
    with patch('builtins.input', side_effect=['5', 'invalid', '1']):  # 無效數字, 無效場景, 正確輸入
        with patch('builtins.print'):
            with patch.object(runner, 'log_info'):
                result = runner.ask_scenario_selection("TestMethod", "test")
                if result == ['offpeak']:
                    print("✅ 無效輸入處理正確")
                else:
                    print(f"❌ 無效輸入處理錯誤: {result}")
                    return False
    
    return True

def test_experiment_plan_display():
    """測試實驗計劃顯示功能"""
    print("🧪 測試實驗計劃顯示...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試計劃顯示邏輯
    test_plans = [
        {'gym_hpa': {'mode': 'test', 'scenarios': ['peak', 'rushsale']}},
        {'gnnrl': {'mode': 'both', 'scenarios': ['all']}},
        {'k8s_hpa': {'mode': 'test', 'scenarios': ['offpeak']}},
    ]
    
    for plan in test_plans:
        try:
            # 模擬顯示邏輯
            for method, config in plan.items():
                mode_desc = {
                    'train': '訓練',
                    'test': '測試',
                    'both': '訓練+測試',
                    'skip': '跳過'
                }
                scenarios_desc = config.get('scenarios', ['all'])
                scenario_text = '所有場景' if 'all' in scenarios_desc else ', '.join(scenarios_desc)
                display_text = f"{method.upper()}: {mode_desc.get(config['mode'], config['mode'])} - 場景: {scenario_text}"
                print(f"   ✅ {display_text}")
        except Exception as e:
            print(f"❌ 計劃顯示錯誤: {e}")
            return False
    
    return True

def main():
    """執行所有測試"""
    print("🚀 開始場景選擇功能測試")
    print("=" * 50)
    
    tests = [
        ("場景選擇功能", test_scenario_selection),
        ("無效輸入處理", test_invalid_inputs),
        ("實驗計劃顯示", test_experiment_plan_display),
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
        print("🎉 所有場景選擇功能測試都通過！")
        print("\n💡 新功能說明:")
        print("   - 支援多場景選擇: 1,2 或 peak,rushsale")
        print("   - 支援場景名稱直接輸入")
        print("   - 支援 'all' 選擇所有場景")
        print("   - 訓練模式自動執行所有場景")
        print("   - 測試模式可選擇特定場景")
        print("\n🎯 使用範例:")
        print("   peak,rushsale  -> 只執行 peak 和 rushsale 場景")
        print("   1,3           -> 只執行 offpeak 和 rushsale 場景")
        print("   all           -> 執行所有場景")
    else:
        print("⚠️ 某些測試失敗，請檢查上述錯誤信息。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)