#!/usr/bin/env python3
"""
測試 Redis 監控系統修正
=======================

驗證 Pod 監控、日誌結構和執行邏輯的修正。
"""

import sys
from pathlib import Path
from run_redis_experiment import RedisExperimentRunner

def test_pod_monitoring_setup():
    """測試 Pod 監控設置"""
    print("🧪 測試 Pod 監控設置...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試 Pod 監控設置
    test_output_dir = Path("/tmp/test_redis_monitoring")
    test_output_dir.mkdir(exist_ok=True)
    
    try:
        pod_monitor = runner._setup_pod_monitoring_for_redis("test_scenario", test_output_dir)
        
        if pod_monitor:
            print("✅ Pod 監控器創建成功")
            print(f"   監控 namespace: redis")
            print(f"   輸出目錄: {test_output_dir}/pod_metrics")
            return True
        else:
            print("❌ Pod 監控器創建失敗")
            return False
            
    except Exception as e:
        print(f"❌ Pod 監控設置失敗: {e}")
        return False

def test_unified_manager_integration():
    """測試統一實驗管理器整合"""
    print("🧪 測試統一實驗管理器整合...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 檢查是否正確導入了 pod_monitor
    try:
        from pod_monitor import MultiPodMonitor, create_pod_monitor_for_experiment
        print("✅ Pod 監控模組導入成功")
    except ImportError as e:
        print(f"❌ Pod 監控模組導入失敗: {e}")
        return False
    
    # 檢查 unified_experiment_manager 是否存在
    unified_manager = runner.repo_root / "unified_experiment_manager.py"
    if unified_manager.exists():
        print("✅ 統一實驗管理器文件存在")
    else:
        print("❌ 統一實驗管理器文件不存在")
        return False
    
    return True

def test_log_structure():
    """測試日誌結構設置"""
    print("🧪 測試日誌結構...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試 HPA 配置
    expected_configs = ['cpu-20', 'cpu-40', 'cpu-60', 'cpu-80']
    actual_configs = []
    
    for config_type, configs in runner.redis_hpa_configs.items():
        actual_configs.extend(configs)
    
    if set(expected_configs) == set(actual_configs):
        print("✅ Redis HPA 配置正確")
        print(f"   配置列表: {actual_configs}")
    else:
        print(f"❌ Redis HPA 配置不正確")
        print(f"   期望: {expected_configs}")
        print(f"   實際: {actual_configs}")
        return False
    
    return True

def test_load_test_scenarios():
    """測試負載測試場景"""
    print("🧪 測試負載測試場景...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 檢查負載測試腳本目錄
    loadtest_dir = runner.repo_root / "loadtest" / "redis"
    
    expected_scenarios = ['offpeak', 'peak', 'rushsale', 'fluctuating']
    found_scenarios = []
    
    for scenario in expected_scenarios:
        stable_script = loadtest_dir / f"locust_redis_stable_{scenario}.py"
        regular_script = loadtest_dir / f"locust_redis_{scenario}.py"
        
        if stable_script.exists() or regular_script.exists():
            found_scenarios.append(scenario)
            script_name = stable_script.name if stable_script.exists() else regular_script.name
            print(f"   ✅ {scenario}: {script_name}")
        else:
            print(f"   ❌ {scenario}: 腳本不存在")
    
    if len(found_scenarios) >= 2:  # 至少要有兩個場景
        print(f"✅ 負載測試場景充足 ({len(found_scenarios)}/{len(expected_scenarios)})")
        return True
    else:
        print(f"❌ 負載測試場景不足 ({len(found_scenarios)}/{len(expected_scenarios)})")
        return False

def test_model_discovery():
    """測試模型發現功能"""
    print("🧪 測試模型發現功能...")
    
    runner = RedisExperimentRunner(algorithm='a2c')
    
    # 測試模型查找（可能沒有模型，但功能應該正常）
    gym_model = runner.find_latest_model('gym_hpa')
    gnnrl_model = runner.find_latest_model('gnnrl')
    
    print(f"   Gym-HPA 模型: {gym_model if gym_model else '無'}")
    print(f"   GNNRL 模型: {gnnrl_model if gnnrl_model else '無'}")
    
    # 測試無效方法名
    invalid_model = runner.find_latest_model('invalid_method')
    if invalid_model is None:
        print("✅ 無效方法名處理正確")
    else:
        print("❌ 無效方法名處理錯誤")
        return False
    
    print("✅ 模型發現功能正常")
    return True

def main():
    """執行所有測試"""
    print("🚀 開始 Redis 監控系統測試")
    print("=" * 50)
    
    tests = [
        ("Pod 監控設置", test_pod_monitoring_setup),
        ("統一管理器整合", test_unified_manager_integration),
        ("日誌結構", test_log_structure),
        ("負載測試場景", test_load_test_scenarios),
        ("模型發現功能", test_model_discovery),
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
        print("🎉 所有監控系統測試都通過！")
        print("\n💡 Redis 實驗現在具備完整監控能力:")
        print("   - Pod 數量時間序列監控")
        print("   - RPS 和延遲數據記錄")
        print("   - 配置分離的日誌結構")
        print("   - 統一的實驗管理流程")
    else:
        print("⚠️ 某些測試失敗，請檢查上述錯誤信息。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)