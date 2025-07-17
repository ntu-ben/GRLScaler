#!/usr/bin/env python3
"""
OnlineBoutique 实验结果可视化生成器
=================================

生成GNNRL、Gym-HPA、K8s-HPA三种方法的对比分析图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import json
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OnlineBoutiqueVisualizer:
    def __init__(self, logs_dir: Path = None):
        self.logs_dir = logs_dir or Path("logs")
        self.output_dir = Path("visualization_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 服务名称映射
        self.services = [
            "recommendationservice", "productcatalogservice", "cartservice", 
            "adservice", "paymentservice", "shippingservice", "currencyservice",
            "redis-cart", "checkoutservice", "frontend", "emailservice"
        ]
        
        # 颜色配置
        self.colors = {
            'GNNRL': '#2E86AB',  # 蓝色
            'Gym-HPA': '#A23B72', # 紫色  
            'K8s-HPA': '#F18F01'  # 橙色
        }
        
    def load_gnnrl_data(self):
        """加载GNNRL数据"""
        gnnrl_files = glob.glob(str(self.logs_dir / "gnnrl/actions/action_history_*.csv"))
        if not gnnrl_files:
            print("⚠️ 未找到GNNRL数据")
            return None
            
        # 使用最新的文件
        latest_file = max(gnnrl_files, key=lambda x: Path(x).stat().st_mtime)
        print(f"📊 加载GNNRL数据: {latest_file}")
        
        try:
            df = pd.read_csv(latest_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"❌ GNNRL数据加载失败: {e}")
            return None
    
    def load_gym_hpa_data(self):
        """加载Gym-HPA数据"""
        gym_hpa_dirs = glob.glob(str(self.logs_dir / "gym-hpa/gym_hpa_*"))
        if not gym_hpa_dirs:
            print("⚠️ 未找到Gym-HPA数据")
            return None
            
        print(f"📊 找到{len(gym_hpa_dirs)}个Gym-HPA实验")
        
        # 处理最新的测试数据
        test_dirs = [d for d in gym_hpa_dirs if 'test' in d]
        if test_dirs:
            latest_dir = max(test_dirs, key=lambda x: Path(x).stat().st_mtime)
            print(f"📊 加载Gym-HPA测试数据: {latest_dir}")
            
            # 查找场景数据
            scenario_files = glob.glob(f"{latest_dir}/*/stats_history.csv")
            if scenario_files:
                gym_data = []
                for file in scenario_files:
                    scenario = Path(file).parent.name.split('_')[0]
                    df = pd.read_csv(file)
                    df['scenario'] = scenario
                    df['method'] = 'Gym-HPA'
                    gym_data.append(df)
                return pd.concat(gym_data, ignore_index=True) if gym_data else None
        
        return None
    
    def load_k8s_hpa_data(self):
        """加载K8s-HPA数据"""
        k8s_dirs = glob.glob(str(self.logs_dir / "k8s-hpa/k8s_hpa_*"))
        if not k8s_dirs:
            print("⚠️ 未找到K8s-HPA数据")
            return None
            
        print(f"📊 找到{len(k8s_dirs)}个K8s-HPA实验")
        
        # 加载最新实验的数据
        latest_dir = max(k8s_dirs, key=lambda x: Path(x).stat().st_mtime)
        print(f"📊 加载K8s-HPA数据: {latest_dir}")
        
        k8s_data = []
        stats_files = glob.glob(f"{latest_dir}/*/*_stats_history.csv")
        
        for file in stats_files:
            path_parts = Path(file).parts
            config = path_parts[-3].split('_')[-1] if 'cpu' in path_parts[-3] else 'unknown'
            scenario = Path(file).stem.replace('_stats_history', '')
            
            df = pd.read_csv(file)
            df['config'] = config
            df['scenario'] = scenario  
            df['method'] = 'K8s-HPA'
            k8s_data.append(df)
            
        return pd.concat(k8s_data, ignore_index=True) if k8s_data else None
    
    def plot_scaling_actions_comparison(self):
        """绘制扩缩容动作对比图"""
        gnnrl_data = self.load_gnnrl_data()
        
        if gnnrl_data is None:
            print("❌ 无法生成扩缩容动作对比图：缺少GNNRL数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('OnlineBoutique 扩缩容动作分析', fontsize=16, fontweight='bold')
        
        # 1. 动作类型分布
        ax1 = axes[0, 0]
        action_counts = gnnrl_data['action_type'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(action_counts)))
        wedges, texts, autotexts = ax1.pie(action_counts.values, labels=action_counts.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax1.set_title('GNNRL 动作类型分布')
        
        # 2. 奖励随时间变化
        ax2 = axes[0, 1]
        if 'step' in gnnrl_data.columns:
            ax2.plot(gnnrl_data['step'], gnnrl_data['reward'], 'o-', alpha=0.7, 
                    color=self.colors['GNNRL'], linewidth=2)
            ax2.set_title('GNNRL 奖励随步数变化')
            ax2.set_xlabel('步数')
            ax2.set_ylabel('奖励')
            ax2.grid(True, alpha=0.3)
        
        # 3. 服务副本数变化
        ax3 = axes[1, 0]
        if 'new_replicas' in gnnrl_data.columns and 'deployment_name' in gnnrl_data.columns:
            # 选择最活跃的前5个服务
            top_services = gnnrl_data['deployment_name'].value_counts().head(5).index
            for i, service in enumerate(top_services):
                service_data = gnnrl_data[gnnrl_data['deployment_name'] == service]
                if not service_data.empty:
                    ax3.plot(service_data['step'], service_data['new_replicas'], 
                            'o-', label=service[:12], alpha=0.8, linewidth=2)
            
            ax3.set_title('主要服务副本数变化')
            ax3.set_xlabel('步数')
            ax3.set_ylabel('副本数')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # 4. 系统指标趋势
        ax4 = axes[1, 1]
        if all(col in gnnrl_data.columns for col in ['avg_latency', 'cpu_usage', 'mem_usage']):
            ax4_twin = ax4.twinx()
            
            # 延迟（左轴）
            line1 = ax4.plot(gnnrl_data['step'], gnnrl_data['avg_latency'], 
                           'r-', label='延迟 (ms)', linewidth=2)
            ax4.set_ylabel('延迟 (ms)', color='r')
            ax4.tick_params(axis='y', labelcolor='r')
            
            # CPU使用率（右轴）
            line2 = ax4_twin.plot(gnnrl_data['step'], gnnrl_data['cpu_usage'], 
                                'b-', label='CPU使用率 (%)', linewidth=2)
            ax4_twin.set_ylabel('CPU使用率 (%)', color='b')
            ax4_twin.tick_params(axis='y', labelcolor='b')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
            
            ax4.set_title('系统性能指标')
            ax4.set_xlabel('步数')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "ob_scaling_actions_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 扩缩容动作分析图已保存: {output_file}")
        plt.close()
    
    def plot_performance_comparison(self):
        """绘制三种方法性能对比图"""
        # 加载所有数据
        gnnrl_data = self.load_gnnrl_data()
        gym_hpa_data = self.load_gym_hpa_data()
        k8s_hpa_data = self.load_k8s_hpa_data()
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('OnlineBoutique 三种方法性能对比', fontsize=16, fontweight='bold')
        
        # 1. 奖励对比（如果有数据）
        ax1 = axes[0, 0]
        methods_rewards = []
        
        if gnnrl_data is not None and 'reward' in gnnrl_data.columns:
            gnnrl_rewards = gnnrl_data['reward'].values
            methods_rewards.append(('GNNRL', gnnrl_rewards))
        
        if gym_hpa_data is not None and 'reward' in gym_hpa_data.columns:
            gym_rewards = gym_hpa_data['reward'].values
            methods_rewards.append(('Gym-HPA', gym_rewards))
        
        if methods_rewards:
            box_data = [rewards for _, rewards in methods_rewards]
            box_labels = [method for method, _ in methods_rewards]
            box_colors = [self.colors.get(method, '#888888') for method in box_labels]
            
            bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                
            ax1.set_title('奖励分布对比')
            ax1.set_ylabel('奖励值')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, '暂无奖励数据', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('奖励分布对比')
        
        # 2. 延迟对比
        ax2 = axes[0, 1]
        latency_data = []
        
        if gnnrl_data is not None and 'avg_latency' in gnnrl_data.columns:
            latency_data.append(('GNNRL', gnnrl_data['avg_latency'].values))
        
        if latency_data:
            for i, (method, latencies) in enumerate(latency_data):
                ax2.hist(latencies[latencies > 0], bins=20, alpha=0.7, 
                        label=method, color=self.colors.get(method, '#888888'))
            
            ax2.set_title('延迟分布对比')
            ax2.set_xlabel('延迟 (ms)')
            ax2.set_ylabel('频次')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '暂无延迟数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('延迟分布对比')
        
        # 3. 资源使用对比
        ax3 = axes[1, 0]
        if gnnrl_data is not None:
            if 'cpu_usage' in gnnrl_data.columns and 'mem_usage' in gnnrl_data.columns:
                scatter = ax3.scatter(gnnrl_data['cpu_usage'], gnnrl_data['mem_usage'], 
                                    alpha=0.6, c=gnnrl_data['step'], cmap='viridis',
                                    label='GNNRL', s=30)
                ax3.set_xlabel('CPU使用率 (%)')
                ax3.set_ylabel('内存使用率 (%)')
                ax3.set_title('资源使用分布')
                plt.colorbar(scatter, ax=ax3, label='步数')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, '暂无资源使用数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('资源使用分布')
        
        # 4. 动作统计
        ax4 = axes[1, 1]
        if gnnrl_data is not None and 'action_value' in gnnrl_data.columns:
            action_stats = gnnrl_data['action_value'].value_counts().sort_index()
            
            bars = ax4.bar(action_stats.index, action_stats.values, 
                          color=self.colors['GNNRL'], alpha=0.7)
            ax4.set_title('GNNRL 动作选择频次')
            ax4.set_xlabel('动作值')
            ax4.set_ylabel('选择次数')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '暂无动作数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('动作选择统计')
        
        plt.tight_layout()
        output_file = self.output_dir / "ob_performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 性能对比图已保存: {output_file}")
        plt.close()
    
    def plot_service_scaling_heatmap(self):
        """绘制服务扩缩容热力图"""
        gnnrl_data = self.load_gnnrl_data()
        
        if gnnrl_data is None or 'deployment_name' not in gnnrl_data.columns:
            print("❌ 无法生成服务扩缩容热力图：缺少必要数据")
            return
        
        # 创建服务动作矩阵
        services = gnnrl_data['deployment_name'].unique()
        actions = gnnrl_data['action_value'].unique()
        
        # 构建热力图数据
        heatmap_data = []
        for service in services:
            service_data = gnnrl_data[gnnrl_data['deployment_name'] == service]
            action_counts = service_data['action_value'].value_counts()
            
            row = []
            for action in sorted(actions):
                row.append(action_counts.get(action, 0))
            heatmap_data.append(row)
        
        # 绘制热力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('OnlineBoutique 服务扩缩容行为分析', fontsize=16, fontweight='bold')
        
        # 1. 服务-动作热力图
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=[s[:15] for s in services], 
                                 columns=[f'动作{a}' for a in sorted(actions)])
        
        sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=ax1, cbar_kws={'label': '执行次数'})
        ax1.set_title('各服务动作执行频次')
        ax1.set_xlabel('动作类型')
        ax1.set_ylabel('服务名称')
        
        # 2. 副本数变化时间线
        if 'step' in gnnrl_data.columns and 'new_replicas' in gnnrl_data.columns:
            # 选择最活跃的服务
            top_services = gnnrl_data['deployment_name'].value_counts().head(6).index
            
            for i, service in enumerate(top_services):
                service_data = gnnrl_data[gnnrl_data['deployment_name'] == service]
                if not service_data.empty:
                    ax2.plot(service_data['step'], service_data['new_replicas'], 
                            'o-', label=service[:12], linewidth=2, markersize=4)
            
            ax2.set_title('主要服务副本数变化时间线')
            ax2.set_xlabel('步数')
            ax2.set_ylabel('副本数')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "ob_service_scaling_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ 服务扩缩容热力图已保存: {output_file}")
        plt.close()
    
    def generate_summary_report(self):
        """生成汇总报告"""
        gnnrl_data = self.load_gnnrl_data()
        
        report = {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "数据源": {}
        }
        
        if gnnrl_data is not None:
            report["数据源"]["GNNRL"] = {
                "数据点数量": len(gnnrl_data),
                "时间范围": f"{gnnrl_data['step'].min()} - {gnnrl_data['step'].max()}步" if 'step' in gnnrl_data.columns else "未知",
                "平均奖励": gnnrl_data['reward'].mean() if 'reward' in gnnrl_data.columns else "N/A",
                "总奖励": gnnrl_data['reward'].sum() if 'reward' in gnnrl_data.columns else "N/A",
                "活跃服务数": gnnrl_data['deployment_name'].nunique() if 'deployment_name' in gnnrl_data.columns else "N/A"
            }
        
        # 保存报告
        report_file = self.output_dir / "ob_experiment_summary.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 实验汇总报告已保存: {report_file}")
        return report
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("🎨 开始生成OnlineBoutique实验可视化图表...")
        print("=" * 50)
        
        # 生成各类图表
        self.plot_scaling_actions_comparison()
        self.plot_performance_comparison()
        self.plot_service_scaling_heatmap()
        
        # 生成汇总报告
        report = self.generate_summary_report()
        
        print("\n" + "=" * 50)
        print("🎉 OnlineBoutique可视化生成完成！")
        print(f"📁 输出目录: {self.output_dir.absolute()}")
        print("\n📊 生成的文件:")
        for file in self.output_dir.glob("ob_*.png"):
            print(f"   🖼️  {file.name}")
        for file in self.output_dir.glob("ob_*.json"):
            print(f"   📄 {file.name}")


def main():
    """主函数"""
    # 检查是否在正确的目录
    if not Path("logs").exists():
        print("❌ 未找到logs目录，请确保在GRLScaler根目录下运行")
        return
    
    # 创建可视化器并生成图表
    visualizer = OnlineBoutiqueVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()