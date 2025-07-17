#!/usr/bin/env python3
"""
測試 Redis 實驗修正
==================

快速測試腳本，驗證所有修正是否正常工作。
"""

import sys
from pathlib import Path

def test_redis_environment_fix():
    """測試 Redis 環境觀察空間修正"""
    print("🧪 測試 Redis 環境觀察空間修正...")
    
    try:
        # 測試 GNNRL Redis 環境
        sys.path.append(str(Path(__file__).parent / "gnnrl"))
        from gnnrl.core.envs.redis import Redis
        
        # 創建環境並檢查觀察空間
        env = Redis(k8s=False, use_graph=True)
        
        # 檢查 edge_df 維度
        edge_space = env.observation_space['edge_df']
        expected_shape = (4, 7)  # 應該是 (num_nodes * num_nodes, 7)
        
        if edge_space.shape == expected_shape:
            print("✅ Redis 環境觀察空間修正成功")
            return True
        else:
            print(f"❌ Redis 環境觀察空間仍有問題: {edge_space.shape} != {expected_shape}")
            return False
            
    except Exception as e:
        print(f"❌ Redis 環境測試失敗: {e}")
        return False

def test_dependencies():
    """測試依賴套件安裝"""
    print("🧪 測試依賴套件...")
    
    missing_deps = []
    
    try:
        import locust
        print("✅ locust 可用")
    except ImportError:
        missing_deps.append("locust")
    
    try:
        import redis
        print("✅ redis 可用")
    except ImportError:
        missing_deps.append("redis")
    
    try:
        import sb3_contrib
        print("✅ sb3_contrib 可用")
    except ImportError:
        missing_deps.append("sb3_contrib")
    
    if missing_deps:
        print(f"❌ 缺少依賴套件: {', '.join(missing_deps)}")
        return False
    else:
        print("✅ 所有依賴套件都可用")
        return True

def test_redis_runner():
    """測試 Redis 實驗執行器"""
    print("🧪 測試 Redis 實驗執行器...")
    
    try:
        from run_redis_experiment import RedisExperimentRunner
        
        # 測試創建 runner
        runner = RedisExperimentRunner(
            algorithm='a2c',
            stable_loadtest=True,
            max_rps=300
        )
        
        # 檢查配置
        if runner.config['alg'] == 'a2c':
            print("✅ A2C 算法配置正確")
        else:
            print(f"❌ A2C 算法配置錯誤: {runner.config['alg']}")
            return False
        
        if runner.config['use_case'] == 'redis':
            print("✅ Redis 環境配置正確")
        else:
            print(f"❌ Redis 環境配置錯誤: {runner.config['use_case']}")
            return False
        
        print("✅ Redis 實驗執行器創建成功")
        return True
        
    except Exception as e:
        print(f"❌ Redis 實驗執行器測試失敗: {e}")
        return False

def main():
    """執行所有測試"""
    print("🚀 開始 Redis 修正測試")
    print("=" * 50)
    
    tests = [
        ("Redis 環境觀察空間", test_redis_environment_fix),
        ("依賴套件", test_dependencies),
        ("Redis 實驗執行器", test_redis_runner),
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
        print("🎉 所有測試都通過！Redis 實驗修正完成。")
        print("\n💡 現在可以執行:")
        print("   python run_autoscaling_experiment.py redis --algorithm a2c --steps 5000")
    else:
        print("⚠️ 某些測試失敗，請檢查上述錯誤信息。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)