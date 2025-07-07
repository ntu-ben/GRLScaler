#!/usr/bin/env python3
"""
快速生成實驗圖表腳本
===================

一鍵生成所有環境的實驗結果圖表

使用方式:
    python generate_experiment_charts.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_visualization(environment):
    """運行指定環境的可視化"""
    print(f"\n🚀 生成 {environment.upper()} 環境的實驗圖表...")
    
    cmd = [
        sys.executable, 
        "experiment_visualization.py", 
        "--auto-compare", 
        "--environment", environment
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {environment.upper()} 圖表生成完成")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ {environment.upper()} 圖表生成失敗:")
        print(e.stderr)
        return False
    
    return True

def main():
    print("📊 GRLScaler 實驗結果圖表生成器")
    print("=" * 50)
    
    # 檢查可視化腳本是否存在
    viz_script = Path("experiment_visualization.py")
    if not viz_script.exists():
        print("❌ 找不到 experiment_visualization.py 腳本")
        sys.exit(1)
    
    # 創建輸出目錄
    output_dir = Path("logs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成時間戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕐 開始時間: {timestamp}")
    
    # 支援的環境列表
    environments = ['redis', 'onlineboutique']
    
    success_count = 0
    
    # 為每個環境生成圖表
    for env in environments:
        if run_visualization(env):
            success_count += 1
    
    # 總結報告
    print("\n" + "=" * 50)
    print(f"📈 圖表生成總結:")
    print(f"   ✅ 成功: {success_count}/{len(environments)} 個環境")
    print(f"   📁 輸出目錄: {output_dir}")
    
    # 列出生成的圖表文件
    chart_files = list(output_dir.glob("*.png"))
    if chart_files:
        print(f"\n📊 生成的圖表文件 ({len(chart_files)} 個):")
        
        # 按時間排序，顯示最新的文件
        recent_files = sorted(chart_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
        
        for i, chart_file in enumerate(recent_files, 1):
            file_size = chart_file.stat().st_size / 1024  # KB
            print(f"   {i:2d}. {chart_file.name} ({file_size:.1f} KB)")
        
        if len(chart_files) > 10:
            print(f"   ... 以及其他 {len(chart_files) - 10} 個文件")
    
    print(f"\n🎉 實驗圖表生成完成!")
    print(f"💡 提示: 使用圖片查看器打開 {output_dir} 目錄中的 PNG 文件")

if __name__ == "__main__":
    main()