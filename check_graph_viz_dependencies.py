#!/usr/bin/env python3
"""
圖形可視化依賴檢查腳本
====================

檢查並安裝GNNRL圖形可視化所需的所有依賴套件
"""

import subprocess
import sys
import importlib
from pathlib import Path

# 必需套件
REQUIRED_PACKAGES = [
    ('matplotlib', 'matplotlib'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('pathlib', None),  # 內建套件
    ('json', None),     # 內建套件
    ('datetime', None), # 內建套件
]

# 可選套件（增強功能）
OPTIONAL_PACKAGES = [
    ('plotly', 'plotly', '交互式儀表板'),
    ('networkx', 'networkx', '網絡圖處理'),
    ('pillow', 'PIL', '動畫生成'),
    ('seaborn', 'seaborn', '圖表美化'),
]

def check_package(package_name, import_name=None):
    """檢查套件是否已安裝"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安裝套件"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔍 GNNRL 圖形可視化依賴檢查")
    print("=" * 50)
    
    missing_required = []
    missing_optional = []
    
    # 檢查必需套件
    print("\n📋 必需套件檢查：")
    for package_name, import_name in REQUIRED_PACKAGES:
        if import_name is None:
            print(f"  ✅ {package_name} (內建)")
        elif check_package(package_name, import_name):
            print(f"  ✅ {package_name}")
        else:
            print(f"  ❌ {package_name} (未安裝)")
            missing_required.append(package_name)
    
    # 檢查可選套件
    print("\n🎨 可選套件檢查：")
    for package_name, import_name, description in OPTIONAL_PACKAGES:
        if check_package(package_name, import_name):
            print(f"  ✅ {package_name} ({description})")
        else:
            print(f"  ❌ {package_name} ({description})")
            missing_optional.append((package_name, description))
    
    # 安裝缺失的必需套件
    if missing_required:
        print(f"\n⚠️  發現 {len(missing_required)} 個缺失的必需套件")
        response = input("是否自動安裝? (y/n): ")
        
        if response.lower() == 'y':
            print("\n📦 安裝必需套件...")
            for package in missing_required:
                print(f"  安裝 {package}...")
                if install_package(package):
                    print(f"  ✅ {package} 安裝成功")
                else:
                    print(f"  ❌ {package} 安裝失敗")
        else:
            print("\n請手動安裝缺失的套件：")
            for package in missing_required:
                print(f"  pip install {package}")
    
    # 詢問是否安裝可選套件
    if missing_optional:
        print(f"\n🎯 發現 {len(missing_optional)} 個可選套件未安裝")
        print("可選套件提供增強功能，建議安裝以獲得完整體驗")
        
        for package, description in missing_optional:
            response = input(f"是否安裝 {package} ({description})? (y/n): ")
            if response.lower() == 'y':
                print(f"  安裝 {package}...")
                if install_package(package):
                    print(f"  ✅ {package} 安裝成功")
                else:
                    print(f"  ❌ {package} 安裝失敗")
    
    # 最終檢查
    print("\n🔄 最終檢查...")
    all_good = True
    
    for package_name, import_name in REQUIRED_PACKAGES:
        if import_name is not None and not check_package(package_name, import_name):
            print(f"  ❌ {package_name} 仍未安裝")
            all_good = False
    
    if all_good:
        print("✅ 所有必需套件已安裝!")
        print("\n🎉 圖形可視化功能已就緒!")
        print("\n📚 使用方法：")
        print("  1. 啟動帶圖形可視化的訓練：")
        print("     python unified_experiment_manager.py --experiment gnnrl --use-case online_boutique")
        print("  2. 生成儀表板：")
        print("     python gnnrl/training/graph_visualization_dashboard.py --log-dir <log_dir>")
        print("  3. 查看完整指南：")
        print("     cat GRAPH_VISUALIZATION_GUIDE.md")
    else:
        print("❌ 仍有必需套件未安裝，請手動安裝或重新運行此腳本")
    
    # 顯示安裝指令摘要
    print("\n📋 完整安裝指令：")
    print("# 必需套件")
    print("pip install matplotlib numpy pandas")
    print("\n# 可選套件（建議全部安裝）")
    print("pip install plotly networkx pillow seaborn")
    print("\n# 一次性安裝所有套件")
    print("pip install matplotlib numpy pandas plotly networkx pillow seaborn")

if __name__ == "__main__":
    main()