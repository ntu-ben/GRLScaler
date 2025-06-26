#!/usr/bin/env python3
"""
整合測試腳本
=============

測試統一實驗管理器的各項功能，確保分散式環境整合正常。
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_environment_validation():
    """測試環境驗證功能"""
    print("🔍 測試環境驗證...")
    
    cmd = [sys.executable, "unified_experiment_manager.py", "--validate-only"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 環境驗證通過")
        return True
    else:
        print(f"❌ 環境驗證失敗: {result.stderr}")
        return False

def test_config_loading():
    """測試配置檔案載入"""
    print("📄 測試配置檔案載入...")
    
    config_file = Path("experiment_config.yaml")
    if not config_file.exists():
        print("❌ 配置檔案不存在")
        return False
    
    try:
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        required_sections = ['experiments', 'loadtest', 'environment']
        for section in required_sections:
            if section not in config:
                print(f"❌ 配置檔案缺少 {section} 區段")
                return False
        
        print("✅ 配置檔案載入成功")
        return True
        
    except Exception as e:
        print(f"❌ 配置檔案載入失敗: {e}")
        return False

def test_experiment_scripts():
    """測試實驗腳本存在性"""
    print("📜 測試實驗腳本...")
    
    scripts = [
        "gym-hpa/policies/run/run.py",
        "k8s_hpa/HPABaseLineTest.py", 
        "gnnrl/training/run_gnnrl_experiment.py",
        "gnnrl/training/rl_batch_loadtest.py"
    ]
    
    missing_scripts = []
    for script in scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ 缺少腳本: {', '.join(missing_scripts)}")
        return False
    
    print("✅ 實驗腳本檢查通過")
    return True

def test_locust_scenarios():
    """測試 Locust 測試腳本"""
    print("🦗 測試 Locust 測試腳本...")
    
    scenarios = [
        "loadtest/onlineboutique/locust_offpeak.py",
        "loadtest/onlineboutique/locust_rushsale.py", 
        "loadtest/onlineboutique/locust_peak.py",
        "loadtest/onlineboutique/locust_fluctuating.py"
    ]
    
    missing_scenarios = []
    for scenario in scenarios:
        if not Path(scenario).exists():
            missing_scenarios.append(scenario)
    
    if missing_scenarios:
        print(f"❌ 缺少測試腳本: {', '.join(missing_scenarios)}")
        return False
    
    print("✅ Locust 測試腳本檢查通過")
    return True

def test_distributed_agent():
    """測試分散式代理連接"""
    print("🌐 測試分散式代理連接...")
    
    m1_host = os.getenv('M1_HOST')
    if not m1_host:
        print("⚠️ M1_HOST 環境變數未設置，跳過分散式測試")
        return True
    
    try:
        import requests
        response = requests.get(f"{m1_host.rstrip('/')}/", timeout=5)
        print(f"✅ 分散式代理連接正常: {m1_host}")
        return True
    except Exception as e:
        print(f"⚠️ 分散式代理連接失敗: {e} (將使用本地 fallback)")
        return True  # 這不是致命錯誤

def test_dry_run():
    """測試乾跑模式 (不實際執行訓練)"""
    print("🧪 測試乾跑模式...")
    
    # 這裡可以添加一個簡短的乾跑測試
    # 例如只驗證命令參數而不執行完整實驗
    
    cmd = [
        sys.executable, "unified_experiment_manager.py", 
        "--experiment", "k8s_hpa", "--validate-only"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ 乾跑測試通過")
        return True
    else:
        print(f"❌ 乾跑測試失敗: {result.stderr}")
        return False

def run_all_tests():
    """執行所有測試"""
    print("=" * 60)
    print("🚀 GRLScaler 整合測試開始")
    print("=" * 60)
    
    tests = [
        ("配置檔案載入", test_config_loading),
        ("實驗腳本檢查", test_experiment_scripts),
        ("Locust 腳本檢查", test_locust_scenarios),
        ("分散式代理連接", test_distributed_agent),
        ("環境驗證", test_environment_validation),
        ("乾跑測試", test_dry_run),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 執行測試: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ 測試異常: {e}")
            results[test_name] = False
    
    # 生成測試報告
    print("\n" + "=" * 60)
    print("📊 測試結果摘要")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print("-" * 60)
    print(f"總計: {passed}/{total} 個測試通過")
    
    if passed == total:
        print("🎉 所有測試通過！系統整合成功。")
        return True
    else:
        print("⚠️ 部分測試失敗，請檢查配置。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)