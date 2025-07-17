#!/usr/bin/env python3
"""
OnlineBoutique 实验结果查看器
============================
"""

from pathlib import Path
import json

def show_summary():
    """显示实验汇总信息"""
    summary_file = Path("visualization_results/ob_experiment_summary.json")
    
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📊 OnlineBoutique 实验汇总")
        print("=" * 40)
        print(f"🕐 生成时间: {data['生成时间']}")
        
        if 'GNNRL' in data['数据源']:
            gnnrl = data['数据源']['GNNRL']
            print(f"\n🧠 GNNRL 分析结果:")
            print(f"   📈 数据点数量: {gnnrl['数据点数量']}")
            print(f"   ⏱️ 时间范围: {gnnrl['时间范围']}")
            print(f"   🎯 平均奖励: {gnnrl['平均奖励']:.2f}")
            print(f"   📊 总奖励: {gnnrl['总奖励']}")
            print(f"   🔧 活跃服务数: {gnnrl['活跃服务数']}")
        
        print(f"\n📁 可视化文件位置: visualization_results/")
        print("🖼️ 生成的图表:")
        
        vis_dir = Path("visualization_results")
        for png_file in vis_dir.glob("ob_*.png"):
            description = {
                "ob_scaling_actions_analysis.png": "扩缩容动作分析",
                "ob_performance_comparison.png": "性能对比分析", 
                "ob_service_scaling_heatmap.png": "服务扩缩容热力图"
            }
            desc = description.get(png_file.name, png_file.name)
            print(f"   📈 {png_file.name} - {desc}")
    else:
        print("❌ 未找到汇总报告，请先运行 python generate_ob_visualization.py")

def main():
    """主函数"""
    print("🎨 OnlineBoutique 实验结果查看器")
    print("=" * 50)
    
    # 检查可视化结果目录
    vis_dir = Path("visualization_results")
    if not vis_dir.exists():
        print("❌ 未找到可视化结果目录")
        print("💡 请运行: python generate_ob_visualization.py")
        return
    
    # 显示汇总信息
    show_summary()
    
    print("\n" + "=" * 50)
    print("💡 提示:")
    print("   • 在文件管理器中打开 visualization_results/ 目录查看图片")
    print("   • 或使用图片查看器打开 .png 文件")
    print("   • 重新生成: python generate_ob_visualization.py")

if __name__ == "__main__":
    main()