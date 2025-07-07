#!/usr/bin/env python3
"""
Redis 實驗環境驗證腳本
====================
驗證 Redis 實驗所需的所有組件
"""

import subprocess
from pathlib import Path

def check_item(name: str, check_func, fix_suggestion: str = "") -> bool:
    """檢查單項"""
    print(f"🔍 檢查 {name}...")
    try:
        result = check_func()
        if result:
            print(f"✅ {name}: 正常")
            return True
        else:
            print(f"❌ {name}: 失敗")
            if fix_suggestion:
                print(f"💡 修復建議: {fix_suggestion}")
            return False
    except Exception as e:
        print(f"❌ {name}: 錯誤 - {e}")
        if fix_suggestion:
            print(f"💡 修復建議: {fix_suggestion}")
        return False

def check_redis_pods():
    """檢查 Redis Pods"""
    result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'redis'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        return False
    
    lines = result.stdout.strip().split('\n')[1:]  # 跳過標題行
    running_pods = [line for line in lines if 'Running' in line]
    return len(running_pods) >= 2

def check_redis_connectivity():
    """檢查 Redis 連接"""
    cmd = [
        'kubectl', 'run', 'redis-ping-test', '--rm', '-i', '--restart=Never',
        '--image=redis:7.2-alpine', '-n', 'redis',
        '--', 'redis-cli', '-h', 'redis-master', 'ping'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return 'PONG' in result.stdout

def check_hpa_configs():
    """檢查 HPA 配置文件"""
    redis_hpa_root = Path(__file__).parent / "macK8S" / "HPA" / "redis"
    required_configs = [
        'cpu-20', 'cpu-40', 'cpu-60', 'cpu-80',
        'mem-40', 'mem-80',
        'cpu-40-mem-40', 'cpu-80-mem-80'
    ]
    
    missing_configs = []
    for config in required_configs:
        config_dir = redis_hpa_root / config
        if not config_dir.exists() or not list(config_dir.glob("*.yaml")):
            missing_configs.append(config)
    
    return len(missing_configs) == 0

def check_loadtest_scripts():
    """檢查負載測試腳本"""
    loadtest_dir = Path(__file__).parent / "loadtest" / "redis"
    required_scripts = ['locust_redis_peak.py', 'locust_redis_offpeak.py']
    
    for script in required_scripts:
        if not (loadtest_dir / script).exists():
            return False
    return True

def check_experiment_configs():
    """檢查實驗配置"""
    config_file = Path(__file__).parent / "experiment_config.yaml"
    if not config_file.exists():
        return False
    
    content = config_file.read_text()
    return 'gym_hpa_redis:' in content and 'gnnrl_redis:' in content

def main():
    print("🚀 Redis 實驗環境完整性檢查")
    print("=" * 50)
    
    checks = [
        ("Redis Pods", check_redis_pods, 
         "kubectl apply -f MicroServiceBenchmark/redis-cluster/redis-cluster.yaml"),
        
        ("Redis 連接", check_redis_connectivity, 
         "確保 Redis master 服務正常運行"),
        
        ("HPA 配置文件", check_hpa_configs, 
         "python macK8S/HPA/redis/generate_redis_hpa.py"),
        
        ("負載測試腳本", check_loadtest_scripts, 
         "負載測試腳本已創建"),
        
        ("實驗配置", check_experiment_configs, 
         "實驗配置已更新"),
    ]
    
    results = []
    for name, check_func, fix_suggestion in checks:
        result = check_item(name, check_func, fix_suggestion)
        results.append((name, result))
        print()
    
    # 總結
    print("=" * 50)
    print("📊 檢查結果總結:")
    
    passed = [name for name, result in results if result]
    failed = [name for name, result in results if not result]
    
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
        print("\n🎉 Redis 實驗環境已就緒！")
        print("📋 可以使用以下命令開始實驗:")
        print("   python run_redis_experiment.py --steps 3000")
    else:
        print("\n⚠️ 請修復上述問題後再開始實驗")
    
    return success_rate == 100

if __name__ == "__main__":
    main()