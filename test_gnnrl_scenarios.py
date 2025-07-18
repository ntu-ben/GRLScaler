#!/usr/bin/env python3
"""
GNNRL OnlineBoutique 場景測試腳本
=============================

這個腳本提供便捷的方式來測試GNNRL OnlineBoutique的特定場景。

使用方式：
    # 測試單個場景
    python test_gnnrl_scenarios.py peak
    
    # 測試多個場景
    python test_gnnrl_scenarios.py peak rushsale
    
    # 使用特定模型和算法
    python test_gnnrl_scenarios.py peak --model tgn --alg a2c
    
    # 使用特定模型路徑
    python test_gnnrl_scenarios.py peak --model-path logs/models/your_model.zip
"""

import sys
import argparse
import subprocess
from pathlib import Path
import glob

def find_latest_gnnrl_model(use_case='online_boutique'):
    """找到最新的GNNRL模型"""
    models_dir = Path("logs/models")
    if not models_dir.exists():
        return None
    
    # 搜尋GNNRL模型
    if use_case == 'online_boutique':
        patterns = ["gnnrl_*latency_k8s_True_steps_*.zip", "gnnrl_*_k8s_True_steps_*.zip"]
    else:
        patterns = ["gnnrl_*redis*_k8s_True_steps_*.zip"]
    
    models = []
    for pattern in patterns:
        models.extend(list(models_dir.glob(pattern)))
    
    if not models:
        return None
    
    # 返回最新的模型
    latest_model = max(models, key=lambda x: x.stat().st_mtime)
    return str(latest_model)

def main():
    parser = argparse.ArgumentParser(
        description='GNNRL OnlineBoutique 場景測試腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 測試peak場景
  python test_gnnrl_scenarios.py peak
  
  # 測試peak和rushsale場景
  python test_gnnrl_scenarios.py peak rushsale
  
  # 使用TGN模型和A2C算法
  python test_gnnrl_scenarios.py peak --model tgn --alg a2c
  
  # 使用特定模型路徑
  python test_gnnrl_scenarios.py peak --model-path logs/models/your_model.zip
        """
    )
    
    parser.add_argument('scenarios', nargs='+',
                       choices=['offpeak', 'peak', 'rushsale', 'fluctuating'],
                       help='要測試的場景')
    parser.add_argument('--model', choices=['gat', 'gcn', 'tgn'], default='gat',
                       help='GNN模型類型 (default: gat)')
    parser.add_argument('--alg', choices=['ppo', 'a2c'], default='ppo',
                       help='強化學習算法 (default: ppo)')
    parser.add_argument('--model-path', type=str,
                       help='已訓練模型的路徑 (若不指定則自動找最新模型)')
    parser.add_argument('--use-case', choices=['online_boutique', 'redis'], 
                       default='online_boutique',
                       help='應用場景 (default: online_boutique)')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子 (default: 42)')
    parser.add_argument('--k8s', action='store_true',
                       help='使用真實K8s集群 (default: False)')
    
    args = parser.parse_args()
    
    # 檢查或找到模型路徑
    model_path = args.model_path
    if not model_path:
        model_path = find_latest_gnnrl_model(args.use_case)
        if not model_path:
            print("❌ 找不到GNNRL模型。請先訓練模型或指定 --model-path")
            return False
        print(f"🔍 自動找到最新模型: {Path(model_path).name}")
    else:
        if not Path(model_path).exists():
            print(f"❌ 模型檔案不存在: {model_path}")
            return False
    
    # 顯示測試資訊
    print(f"🧪 GNNRL {args.use_case.replace('_', ' ').title()} 場景測試")
    print(f"📊 測試場景: {', '.join(args.scenarios)}")
    print(f"🧠 模型: {args.model.upper()}")
    print(f"🎯 算法: {args.alg.upper()}")
    print(f"📁 模型路徑: {model_path}")
    print(f"🎲 隨機種子: {args.seed}")
    print(f"🔧 環境: {'K8s集群' if args.k8s else '模擬模式'}")
    print()
    
    # 構建統一實驗管理器命令
    cmd = [
        sys.executable, "unified_experiment_manager.py",
        "--experiment", "gnnrl",
        "--use-case", args.use_case,
        "--model", args.model,
        "--alg", args.alg,
        "--seed", str(args.seed),
        "--testing",
        "--load-path", model_path,
        "--test-scenarios"
    ] + args.scenarios
    
    if args.k8s:
        cmd.append("--k8s")
    
    print(f"🚀 執行命令: {' '.join(cmd)}")
    print()
    
    try:
        # 執行測試
        result = subprocess.run(cmd, check=True)
        print("✅ 測試完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 測試失敗，退出碼: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\\n👋 測試被用戶中斷")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)