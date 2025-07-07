#!/usr/bin/env python3
"""
Redis HPA 快速測試腳本
===================
快速測試所有 Redis HPA 配置的基本功能
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

class RedisHPAQuickTest:
    def __init__(self):
        self.repo_root = Path(__file__).parent
        self.redis_hpa_root = self.repo_root / "macK8S" / "HPA" / "redis"
        
        # HPA 配置列表
        self.all_configs = [
            # CPU 配置
            'cpu-20', 'cpu-40', 'cpu-60', 'cpu-80',
            # Memory 配置
            'mem-40', 'mem-80',
            # 混合配置
            'cpu-20-mem-40', 'cpu-20-mem-80',
            'cpu-40-mem-40', 'cpu-40-mem-80',
            'cpu-60-mem-40', 'cpu-60-mem-80',
            'cpu-80-mem-40', 'cpu-80-mem-80'
        ]
        
        # 快速測試配置（選擇代表性配置）
        self.quick_configs = [
            'cpu-20', 'cpu-40', 'cpu-80',  # CPU 代表性配置
            'mem-40', 'mem-80',            # Memory 配置
            'cpu-40-mem-40', 'cpu-80-mem-80'  # 混合配置代表
        ]
    
    def log_info(self, message: str):
        print(f"\\033[0;36m[INFO]\\033[0m {message}")
    
    def log_success(self, message: str):
        print(f"\\033[0;32m[SUCCESS]\\033[0m {message}")
    
    def log_error(self, message: str):
        print(f"\\033[0;31m[ERROR]\\033[0m {message}")
    
    def log_section(self, title: str):
        print(f"\\n\\033[0;35m{'=' * 50}\\033[0m")
        print(f"\\033[0;35m{title}\\033[0m")
        print(f"\\033[0;35m{'=' * 50}\\033[0m")
    
    def check_redis_environment(self) -> bool:
        """檢查 Redis 環境是否運行"""
        self.log_info("🔍 檢查 Redis 環境...")
        
        try:
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', 'redis', '--no-headers'],
                capture_output=True, text=True, check=True
            )
            
            if not result.stdout.strip():
                self.log_error("❌ Redis namespace 中沒有 Pod")
                self.log_info("💡 請先部署 Redis:")
                self.log_info("   kubectl apply -f MicroServiceBenchmark/redis-cluster/redis-cluster.yaml")
                return False
            
            running_pods = [p for p in result.stdout.strip().split('\\n') if 'Running' in p]
            redis_core_pods = [p for p in running_pods if 'redis-master' in p or 'redis-slave' in p]
            
            if len(redis_core_pods) < 2:
                self.log_error(f"❌ Redis 核心 Pod 未就緒，當前狀態：")
                print(result.stdout)
                self.log_info(f"   檢測到的核心 Pod: {redis_core_pods}")
                return False
            
            self.log_success(f"✅ Redis 環境正常，{len(running_pods)} 個 Pod 運行中")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log_error(f"❌ 檢查 Redis 環境失敗: {e}")
            return False
    
    def test_hpa_config(self, config_name: str) -> bool:
        """測試單個 HPA 配置"""
        self.log_info(f"🧪 測試 HPA 配置: {config_name}")
        
        config_dir = self.redis_hpa_root / config_name
        if not config_dir.exists():
            self.log_error(f"❌ 配置目錄不存在: {config_dir}")
            return False
        
        try:
            # 1. 清除現有 HPA
            subprocess.run(
                ['kubectl', 'delete', 'hpa', '--all', '-n', 'redis'],
                capture_output=True
            )
            time.sleep(5)
            
            # 2. 應用新 HPA 配置
            for hpa_file in config_dir.glob("*.yaml"):
                result = subprocess.run(
                    ['kubectl', 'apply', '-f', str(hpa_file)],
                    capture_output=True, text=True
                )
                if result.returncode != 0:
                    self.log_error(f"❌ 應用 HPA 配置失敗: {result.stderr}")
                    return False
            
            # 3. 等待 HPA 初始化
            self.log_info("⏳ 等待 HPA 初始化...")
            time.sleep(30)
            
            # 4. 檢查 HPA 狀態
            result = subprocess.run(
                ['kubectl', 'get', 'hpa', '-n', 'redis'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0 and 'redis' in result.stdout:
                self.log_success(f"✅ HPA 配置 {config_name} 應用成功")
                print("HPA 狀態:")
                print(result.stdout)
                return True
            else:
                self.log_error(f"❌ HPA 配置 {config_name} 狀態異常")
                return False
                
        except Exception as e:
            self.log_error(f"❌ 測試 HPA 配置 {config_name} 失敗: {e}")
            return False
    
    def run_basic_load_test(self, config_name: str) -> bool:
        """運行基本負載測試"""
        self.log_info(f"📊 執行基本負載測試: {config_name}")
        
        try:
            # 使用簡單的 Redis 壓力測試
            cmd = [
                'kubectl', 'run', f'redis-benchmark-{config_name}', 
                '--rm', '-i', '--restart=Never',
                '--image=redis:7.2-alpine', '-n', 'redis',
                '--', 'redis-benchmark', 
                '-h', 'redis-master', 
                '-c', '10',  # 10 個並發客戶端
                '-n', '1000',  # 1000 個請求
                '-d', '100',  # 100 字節數據
                '-t', 'set,get'  # 只測試 SET 和 GET
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.log_success(f"✅ 負載測試 {config_name} 完成")
                # 解析基本性能數據
                lines = result.stdout.split('\\n')
                for line in lines:
                    if 'requests per second' in line.lower():
                        self.log_info(f"📈 {line.strip()}")
                return True
            else:
                self.log_error(f"❌ 負載測試 {config_name} 失敗")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            self.log_error(f"❌ 負載測試 {config_name} 超時")
            return False
        except Exception as e:
            self.log_error(f"❌ 負載測試 {config_name} 異常: {e}")
            return False
    
    def cleanup(self):
        """清理測試環境"""
        self.log_info("🧹 清理測試環境...")
        subprocess.run(['kubectl', 'delete', 'hpa', '--all', '-n', 'redis'], 
                      capture_output=True)
        self.log_success("✅ 清理完成")
    
    def run_quick_test(self) -> bool:
        """運行快速測試"""
        self.log_section("🚀 Redis HPA 快速測試開始")
        
        # 檢查環境
        if not self.check_redis_environment():
            return False
        
        # 測試結果
        results = {}
        
        # 測試選定的配置
        for config in self.quick_configs:
            self.log_section(f"測試配置: {config}")
            
            # 測試 HPA 配置
            hpa_success = self.test_hpa_config(config)
            
            # 執行負載測試
            load_success = False
            if hpa_success:
                load_success = self.run_basic_load_test(config)
            
            results[config] = {
                'hpa_success': hpa_success,
                'load_success': load_success
            }
            
            # 配置間間隔
            time.sleep(10)
        
        # 清理
        self.cleanup()
        
        # 總結結果
        self.log_section("📊 測試結果總結")
        
        successful_configs = []
        failed_configs = []
        
        for config, result in results.items():
            if result['hpa_success'] and result['load_success']:
                successful_configs.append(config)
                self.log_success(f"✅ {config}: 完全成功")
            else:
                failed_configs.append(config)
                self.log_error(f"❌ {config}: 失敗 (HPA: {result['hpa_success']}, Load: {result['load_success']})")
        
        print(f"\\n📈 成功率: {len(successful_configs)}/{len(self.quick_configs)} ({len(successful_configs)/len(self.quick_configs)*100:.1f}%)")
        
        if successful_configs:
            print(f"✅ 成功配置: {', '.join(successful_configs)}")
        
        if failed_configs:
            print(f"❌ 失敗配置: {', '.join(failed_configs)}")
        
        return len(failed_configs) == 0
    
    def run_full_test(self) -> bool:
        """運行完整測試（所有配置）"""
        self.log_section("🚀 Redis HPA 完整測試開始")
        
        if not self.check_redis_environment():
            return False
        
        results = {}
        
        for config in self.all_configs:
            self.log_section(f"測試配置: {config}")
            
            hpa_success = self.test_hpa_config(config)
            results[config] = {'hpa_success': hpa_success}
            
            time.sleep(5)  # 短間隔
        
        self.cleanup()
        
        # 總結
        self.log_section("📊 完整測試結果")
        
        successful = [c for c, r in results.items() if r['hpa_success']]
        failed = [c for c, r in results.items() if not r['hpa_success']]
        
        print(f"📈 HPA 配置成功率: {len(successful)}/{len(self.all_configs)} ({len(successful)/len(self.all_configs)*100:.1f}%)")
        
        if successful:
            print(f"✅ 成功配置 ({len(successful)}):")
            for config in successful:
                print(f"   - {config}")
        
        if failed:
            print(f"❌ 失敗配置 ({len(failed)}):")
            for config in failed:
                print(f"   - {config}")
        
        return len(failed) == 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Redis HPA 快速測試')
    parser.add_argument('--full', action='store_true', 
                       help='運行完整測試（所有14個配置）')
    parser.add_argument('--quick', action='store_true', default=True,
                       help='運行快速測試（7個代表性配置）')
    
    args = parser.parse_args()
    
    tester = RedisHPAQuickTest()
    
    if args.full:
        success = tester.run_full_test()
    else:
        success = tester.run_quick_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()