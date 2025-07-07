#!/usr/bin/env python3
"""
Redis 實驗快速測試
================
測試 Redis 實驗的基本命令是否正確
"""

import subprocess
import sys

def test_command(name: str, cmd: list):
    """測試命令是否能正常執行 (只檢查參數解析，不實際運行)"""
    print(f"🧪 測試 {name}...")
    
    # 添加 --validate-only 參數只做驗證
    test_cmd = cmd + ["--validate-only"]
    
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ {name}: 命令參數正確")
            return True
        else:
            print(f"❌ {name}: 參數錯誤")
            print(f"   錯誤: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {name}: 測試超時")
        return False
    except Exception as e:
        print(f"❌ {name}: 測試失敗 - {e}")
        return False

def main():
    print("🚀 Redis 實驗命令快速測試")
    print("=" * 50)
    
    # 測試命令列表
    tests = [
        ("Gym-HPA Redis", [
            sys.executable, "unified_experiment_manager.py",
            "--experiment", "gym_hpa",
            "--k8s", "--use-case", "redis",
            "--goal", "latency", "--alg", "ppo",
            "--seed", "42", "--steps", "100"
        ]),
        
        ("GNNRL Redis", [
            sys.executable, "unified_experiment_manager.py", 
            "--experiment", "gnnrl",
            "--k8s", "--use-case", "redis",
            "--goal", "latency", "--model", "gat",
            "--alg", "ppo", "--seed", "42", "--steps", "100"
        ]),
        
        ("K8s-HPA Redis", [
            sys.executable, "unified_experiment_manager.py",
            "--experiment", "k8s_hpa",
            "--hpa-type", "cpu",
            "--seed", "42"
        ])
    ]
    
    results = []
    
    for name, cmd in tests:
        success = test_command(name, cmd)
        results.append((name, success))
        print()
    
    # 總結
    print("📊 測試結果總結:")
    passed = [name for name, success in results if success]
    failed = [name for name, success in results if not success]
    
    print(f"✅ 通過: {len(passed)}/{len(results)}")
    for name in passed:
        print(f"   - {name}")
    
    if failed:
        print(f"❌ 失敗: {len(failed)}")
        for name in failed:
            print(f"   - {name}")
    
    success_rate = len(passed) / len(results) * 100
    print(f"\n🎯 成功率: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n🎉 所有命令測試通過！Redis 實驗應該可以正常運行")
    else:
        print("\n⚠️ 部分命令測試失敗，需要進一步檢查")

if __name__ == "__main__":
    main()