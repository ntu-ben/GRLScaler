#!/usr/bin/env python3
"""
簡化的 HPA 基準測試
僅執行負載測試，不重置 Kubernetes 服務
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# 設定路徑
REPO_ROOT = Path(__file__).parent
sys.path.append(str(REPO_ROOT))

def main():
    # 設定實驗參數
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"hpa_baseline_{timestamp}"
    
    print(f"🎯 執行 HPA 基準測試: {run_tag}")
    
    # 直接使用 rl_batch_loadtest.py 進行負載測試
    cmd = [
        sys.executable, "gnnrl/training/rl_batch_loadtest.py",
        "--model", "gym-hpa",  # 使用 gym-hpa 模式但不訓練
        "--run-tag", run_tag,
        "--use-case", "online_boutique",
        "--goal", "latency",
        "--seed", "42",
        "--steps", "0",  # 0 步驟表示不訓練
        "--k8s"
    ]
    
    print(f"💻 執行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        print(f"✅ HPA 基準測試完成: {run_tag}")
    except subprocess.CalledProcessError as e:
        print(f"❌ HPA 基準測試失敗: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)