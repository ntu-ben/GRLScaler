#!/usr/bin/env python3
"""
驗證 Redis 修復
==============
檢查修復後的 Redis 實驗是否能正常啟動
"""

import subprocess
import sys

def test_validation():
    """測試環境驗證"""
    print("🔍 測試 Redis 環境驗證...")
    
    cmd = [
        sys.executable, "unified_experiment_manager.py",
        "--experiment", "gym_hpa",
        "--use-case", "redis", 
        "--validate-only"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Redis 環境驗證通過")
            return True
        else:
            print("❌ Redis 環境驗證失敗")
            print(f"錯誤: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 環境驗證超時")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def test_gym_hpa_start():
    """測試 Gym-HPA 是否能正常啟動"""
    print("🧪 測試 Gym-HPA Redis 啟動...")
    
    cmd = [
        sys.executable, "unified_experiment_manager.py",
        "--experiment", "gym_hpa",
        "--use-case", "redis",
        "--k8s", "--steps", "1",  # 最小步數
        "--seed", "42"
    ]
    
    try:
        # 只檢查前 30 秒，看是否有明顯錯誤
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        import time
        time.sleep(30)  # 等待 30 秒
        
        if proc.poll() is None:
            # 程序還在運行，說明啟動成功
            proc.terminate()
            print("✅ Gym-HPA Redis 啟動成功 (30秒內無錯誤)")
            return True
        else:
            # 程序已結束，檢查原因
            stdout, stderr = proc.communicate()
            if "環境不完整" in stderr or "OnlineBoutique" in stderr:
                print("❌ Gym-HPA Redis 仍有環境問題")
                print(f"錯誤: {stderr}")
                return False
            else:
                print("✅ Gym-HPA Redis 正常啟動和結束")
                return True
                
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

def main():
    print("🚀 驗證 Redis 修復狀況")
    print("=" * 50)
    
    tests = [
        ("環境驗證", test_validation),
        ("Gym-HPA 啟動", test_gym_hpa_start)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n📋 執行測試: {name}")
        success = test_func()
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
        print("\n🎉 Redis 修復成功！現在可以運行完整實驗")
        print("📋 建議執行:")
        print("   python run_autoscaling_experiment.py redis --steps 1000")
    else:
        print("\n⚠️ 仍有問題需要修復")

if __name__ == "__main__":
    main()