#!/usr/bin/env python3
"""
GRLScaler 自動擴展實驗統一入口
============================

統一管理不同環境的自動擴展實驗：
- OnlineBoutique (微服務電商平台)
- Redis (內存數據庫)

支援三種自動擴展方法：
- GNNRL (圖神經網路強化學習)
- Gym-HPA (基礎強化學習) 
- K8s-HPA (原生 Kubernetes HPA)
"""

import sys
import argparse
from pathlib import Path

def show_welcome():
    """顯示歡迎信息"""
    print("🚀 GRLScaler 自動擴展實驗平台")
    print("=" * 50)
    print("📊 支援環境:")
    print("   • OnlineBoutique - 微服務電商平台 (10個服務)")
    print("   • Redis - 內存數據庫 (Master-Slave)")
    print()
    print("🧠 支援方法:")
    print("   • GNNRL - 圖神經網路強化學習")
    print("   • Gym-HPA - 基礎強化學習")
    print("   • K8s-HPA - Kubernetes 原生 HPA")
    print()

def run_onlineboutique_experiment(args):
    """執行 OnlineBoutique 實驗"""
    from run_onlineboutique_experiment import ExperimentRunner
    
    print("🛍️ 啟動 OnlineBoutique 微服務自動擴展實驗")
    print("📋 測試環境: 10個微服務 (frontend, cartservice, productcatalog...)")
    print()
    
    runner = ExperimentRunner(use_standardized_scenarios=args.standardized)
    
    if args.method:
        # 單一方法測試
        method_map = {
            'gnnrl': 'gnnrl',
            'gym-hpa': 'gym-hpa', 
            'gymhpa': 'gym-hpa',
            'k8s-hpa': 'k8s-hpa',
            'k8shpa': 'k8s-hpa',
            'hpa': 'k8s-hpa'
        }
        
        stage = method_map.get(args.method.lower())
        if not stage:
            print(f"❌ 未知方法: {args.method}")
            print("支援的方法: gnnrl, gym-hpa, k8s-hpa")
            return False
            
        success = runner.run_single_stage(stage, args.steps, args.goal, args.model)
    else:
        # 完整實驗
        skip_stages = set(args.skip) if args.skip else set()
        success = runner.run_complete_experiment(args.steps, args.goal, args.model, skip_stages)
    
    return success

def run_redis_experiment(args):
    """執行 Redis 實驗"""
    from run_redis_experiment import RedisExperimentRunner
    
    print("🗄️ 啟動 Redis 內存數據庫自動擴展實驗") 
    print("📋 測試環境: Redis Master-Slave 架構")
    print()
    
    runner = RedisExperimentRunner(use_standardized_scenarios=args.standardized)
    success = runner.run_complete_redis_experiment(args.steps, args.goal, args.model)
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description='GRLScaler 自動擴展實驗統一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # OnlineBoutique 完整實驗
  python run_autoscaling_experiment.py onlineboutique --steps 5000

  # OnlineBoutique 只測試 GNNRL
  python run_autoscaling_experiment.py onlineboutique --method gnnrl --steps 3000

  # Redis 完整實驗
  python run_autoscaling_experiment.py redis --steps 5000

  # 使用標準化場景確保公平比較
  python run_autoscaling_experiment.py onlineboutique --standardized --steps 3000
        """
    )
    
    # 環境選擇
    parser.add_argument('environment', 
                       choices=['onlineboutique', 'online-boutique', 'ob', 'redis'],
                       help='實驗環境選擇')
    
    # 實驗參數
    parser.add_argument('--steps', type=int, default=5000, 
                       help='訓練步數 (預設: 5000)')
    parser.add_argument('--goal', default='latency', 
                       choices=['latency', 'cost'],
                       help='優化目標 (預設: latency)')
    parser.add_argument('--model', default='gat',
                       choices=['gat', 'gcn', 'sage'], 
                       help='GNNRL 模型類型 (預設: gat)')
    
    # 場景選項
    parser.add_argument('--standardized', action='store_true',
                       help='使用標準化8個場景確保公平比較')
    
    # OnlineBoutique 專用選項
    parser.add_argument('--method', 
                       choices=['gnnrl', 'gym-hpa', 'gymhpa', 'k8s-hpa', 'k8shpa', 'hpa'],
                       help='只執行特定方法 (僅 OnlineBoutique)')
    parser.add_argument('--skip', nargs='+',
                       choices=['plan', 'gnnrl', 'gym-hpa', 'k8s-hpa', 'analysis'],
                       help='跳過指定階段 (僅 OnlineBoutique)')
    
    # 其他選項
    parser.add_argument('--list-configs', action='store_true',
                       help='列出可用配置')
    parser.add_argument('--verify', action='store_true', 
                       help='驗證實驗環境')
    
    args = parser.parse_args()
    
    # 顯示歡迎信息
    if not (args.list_configs or args.verify):
        show_welcome()
    
    # 環境名稱統一化
    environment = args.environment.lower()
    if environment in ['onlineboutique', 'online-boutique', 'ob']:
        environment = 'onlineboutique'
    
    # 特殊功能
    if args.list_configs:
        print("📋 可用配置:")
        if environment == 'onlineboutique':
            print("   HPA 配置: cpu-20, cpu-40, cpu-60, cpu-80")
            print("   場景: offpeak, peak, rushsale, fluctuating")
        elif environment == 'redis':
            print("   HPA 配置: cpu-20/40/60/80, mem-40/80, cpu-X-mem-Y")
            print("   場景: redis_peak, redis_offpeak")
        return
    
    if args.verify:
        print(f"🔍 驗證 {environment.title()} 實驗環境...")
        if environment == 'redis':
            from redis_environment_check import main as verify_redis
            success = verify_redis()
        else:
            # OnlineBoutique 驗證邏輯
            from run_onlineboutique_experiment import ExperimentRunner
            runner = ExperimentRunner()
            success = runner.check_prerequisites()
        
        sys.exit(0 if success else 1)
    
    # 執行實驗
    try:
        if environment == 'onlineboutique':
            success = run_onlineboutique_experiment(args)
        elif environment == 'redis':
            success = run_redis_experiment(args)
        else:
            print(f"❌ 不支援的環境: {environment}")
            success = False
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n👋 實驗被用戶中斷")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 實驗執行失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()