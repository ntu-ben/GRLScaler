#!/usr/bin/env python3
"""
Redis 實驗短版本測試
==================
用少量步數快速測試 Redis 實驗流程
"""

from run_redis_experiment import RedisExperimentRunner
import sys

def main():
    print("🚀 Redis 實驗短版本測試 (300 步數)")
    print("=" * 50)
    
    # 創建 Redis 實驗執行器
    runner = RedisExperimentRunner(use_standardized_scenarios=False)
    
    # 運行短版本實驗 (300 步數，約 15-20 分鐘)
    success = runner.run_complete_redis_experiment(
        steps=300,  # 減少步數以快速測試
        goal='latency',
        model='gat'
    )
    
    if success:
        print("\n🎉 Redis 實驗短版本測試成功！")
        print("📋 可以安全地運行完整版本:")
        print("   python run_autoscaling_experiment.py redis --steps 5000")
    else:
        print("\n❌ Redis 實驗短版本測試失敗")
        print("💡 請檢查環境配置或聯絡技術支援")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()