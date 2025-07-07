#!/usr/bin/env python3
"""
Redis HPA 簡單測試
================
直接測試 HPA 配置應用
"""

import subprocess
import time
from pathlib import Path

def log_info(message: str):
    print(f"[INFO] {message}")

def log_success(message: str):
    print(f"[SUCCESS] {message}")

def log_error(message: str):
    print(f"[ERROR] {message}")

def test_hpa_config(config_name: str) -> bool:
    """測試 HPA 配置"""
    redis_hpa_root = Path(__file__).parent / "macK8S" / "HPA" / "redis"
    config_dir = redis_hpa_root / config_name
    
    if not config_dir.exists():
        log_error(f"配置目錄不存在: {config_dir}")
        return False
    
    log_info(f"測試配置: {config_name}")
    
    try:
        # 清除現有 HPA
        subprocess.run(['kubectl', 'delete', 'hpa', '--all', '-n', 'redis'], 
                      capture_output=True)
        time.sleep(3)
        
        # 應用配置
        for hpa_file in config_dir.glob("*.yaml"):
            log_info(f"應用配置文件: {hpa_file.name}")
            result = subprocess.run(['kubectl', 'apply', '-f', str(hpa_file)], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                log_error(f"應用失敗: {result.stderr}")
                return False
        
        # 檢查 HPA
        time.sleep(5)
        result = subprocess.run(['kubectl', 'get', 'hpa', '-n', 'redis'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            log_success(f"配置 {config_name} 應用成功")
            print("HPA 狀態:")
            print(result.stdout)
            return True
        else:
            log_error(f"HPA 狀態檢查失敗")
            return False
            
    except Exception as e:
        log_error(f"測試失敗: {e}")
        return False

def main():
    # 測試配置列表
    quick_configs = [
        'cpu-20', 'cpu-40', 'cpu-80',
        'mem-40', 'mem-80',
        'cpu-40-mem-40', 'cpu-80-mem-80'
    ]
    
    print("🚀 開始 Redis HPA 配置測試")
    print(f"📊 將測試 {len(quick_configs)} 個配置")
    
    results = {}
    
    for config in quick_configs:
        print(f"\n{'='*50}")
        success = test_hpa_config(config)
        results[config] = success
        time.sleep(2)
    
    # 清理
    print(f"\n{'='*50}")
    log_info("清理 HPA 配置...")
    subprocess.run(['kubectl', 'delete', 'hpa', '--all', '-n', 'redis'], 
                  capture_output=True)
    
    # 總結
    print(f"\n📊 測試結果總結:")
    successful = [c for c, r in results.items() if r]
    failed = [c for c, r in results.items() if not r]
    
    print(f"✅ 成功: {len(successful)}/{len(quick_configs)}")
    for config in successful:
        print(f"   - {config}")
    
    if failed:
        print(f"❌ 失敗: {len(failed)}")
        for config in failed:
            print(f"   - {config}")
    
    print(f"\n🎯 成功率: {len(successful)/len(quick_configs)*100:.1f}%")

if __name__ == "__main__":
    main()