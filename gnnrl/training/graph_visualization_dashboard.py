#!/usr/bin/env python3
"""
Graph Visualization Dashboard for GNNRL Training
==============================================

動態展示訓練過程中的圖形數據變化，包括：
- 網絡拓撲演變
- 節點特徵變化動畫
- 邊特徵變化趨勢
- 訓練指標實時監控

使用方式：
    python graph_visualization_dashboard.py --log-dir logs/gnnrl/gnnrl_train_seed42_20250711_120622/graph_visualizations
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional, Any
import webbrowser

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class GraphVisualizationDashboard:
    """GNNRL訓練圖形可視化儀表板"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.step_dirs = []
        self.data_timeline = []
        self.service_names = []
        
        # 掃描所有步驟數據
        self._scan_step_data()
        
        # 設置輸出目錄
        self.output_dir = self.log_dir / "dashboard"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _scan_step_data(self):
        """掃描所有步驟數據"""
        if not self.log_dir.exists():
            print(f"❌ 日誌目錄不存在: {self.log_dir}")
            return
            
        # 找到所有step目錄
        step_pattern = "step_*"
        self.step_dirs = sorted([d for d in self.log_dir.glob(step_pattern) if d.is_dir()])
        
        print(f"📊 找到 {len(self.step_dirs)} 個步驟數據目錄")
        
        # 加載每個步驟的數據
        for step_dir in self.step_dirs:
            raw_data_file = step_dir / "raw_data_step_*.json"
            raw_data_files = list(step_dir.glob("raw_data_step_*.json"))
            
            if raw_data_files:
                try:
                    with open(raw_data_files[0], 'r') as f:
                        data = json.load(f)
                        self.data_timeline.append(data)
                        
                        # 提取服務名稱（第一次）
                        if not self.service_names and 'service_names' in data:
                            self.service_names = data['service_names']
                            
                except Exception as e:
                    print(f"⚠️ 讀取數據失敗: {raw_data_files[0]}, {e}")
        
        print(f"✅ 成功加載 {len(self.data_timeline)} 個步驟的數據")
        
    def generate_interactive_dashboard(self):
        """生成交互式HTML儀表板"""
        if not HAS_PLOTLY:
            print("❌ 需要安裝 plotly 來生成交互式儀表板")
            print("請執行: pip install plotly")
            return
            
        if not self.data_timeline:
            print("❌ 沒有找到可視化數據")
            return
            
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Reward Trend', 'Node Features Evolution', 
                          'CPU Usage Distribution', 'Memory Usage Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 準備數據
        steps = [data['step'] for data in self.data_timeline]
        rewards = [data['reward'] for data in self.data_timeline]
        
        # 1. 訓練獎勵趨勢
        fig.add_trace(
            go.Scatter(
                x=steps, y=rewards,
                mode='lines+markers',
                name='Training Reward',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # 2. 節點特徵演化（以第一個服務為例）
        if self.service_names:
            pod_counts = [data['node_features'][0][0] for data in self.data_timeline]
            cpu_usage = [data['node_features'][0][2] for data in self.data_timeline]
            
            fig.add_trace(
                go.Scatter(
                    x=steps, y=pod_counts,
                    mode='lines+markers',
                    name=f'{self.service_names[0]} Pod Count',
                    line=dict(color='green', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )
            
            # 3. CPU使用率分佈
            latest_cpu = [node[2] for node in self.data_timeline[-1]['node_features']]
            fig.add_trace(
                go.Bar(
                    x=self.service_names,
                    y=latest_cpu,
                    name='CPU Usage (%)',
                    marker_color='orange'
                ),
                row=2, col=1
            )
            
            # 4. 記憶體使用率分佈
            latest_mem = [node[3] for node in self.data_timeline[-1]['node_features']]
            fig.add_trace(
                go.Bar(
                    x=self.service_names,
                    y=latest_mem,
                    name='Memory Usage (MB)',
                    marker_color='red'
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            title="GNNRL Training Dashboard",
            showlegend=True,
            height=800,
            font=dict(size=12)
        )
        
        # 保存HTML文件
        html_file = self.output_dir / "interactive_dashboard.html"
        pyo.plot(fig, filename=str(html_file), auto_open=False)
        
        print(f"✅ 交互式儀表板已生成: {html_file}")
        return html_file
        
    def generate_network_evolution_gif(self):
        """生成網絡演化動畫GIF"""
        if not HAS_NETWORKX:
            print("❌ 需要安裝 networkx 來生成網絡演化動畫")
            print("請執行: pip install networkx")
            return
            
        if not self.data_timeline:
            print("❌ 沒有找到可視化數據")
            return
            
        # 創建動畫
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def update_graph(frame):
            ax.clear()
            
            if frame < len(self.data_timeline):
                data = self.data_timeline[frame]
                step = data['step']
                node_features = data['node_features']
                
                # 創建網絡圖
                G = nx.DiGraph()
                
                # 添加節點
                for i, service_name in enumerate(self.service_names):
                    if i < len(node_features):
                        pod_count = node_features[i][0]
                        cpu_usage = node_features[i][2]
                        
                        G.add_node(service_name, 
                                  pod_count=pod_count,
                                  cpu_usage=cpu_usage,
                                  size=max(300, pod_count * 200))
                
                # 簡單的環形佈局
                pos = nx.circular_layout(G)
                
                # 繪製節點
                node_colors = []
                node_sizes = []
                
                for node in G.nodes():
                    cpu_usage = G.nodes[node]['cpu_usage']
                    
                    if cpu_usage > 80:
                        node_colors.append('red')
                    elif cpu_usage > 60:
                        node_colors.append('orange')
                    elif cpu_usage > 40:
                        node_colors.append('yellow')
                    else:
                        node_colors.append('lightgreen')
                    
                    node_sizes.append(G.nodes[node]['size'])
                
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                     node_size=node_sizes, alpha=0.7, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
                
                ax.set_title(f'Network Evolution - Step {step}', fontsize=14, fontweight='bold')
                ax.set_axis_off()
        
        # 創建動畫
        anim = animation.FuncAnimation(
            fig, update_graph, frames=len(self.data_timeline), 
            interval=1000, repeat=True, blit=False
        )
        
        # 保存GIF
        gif_file = self.output_dir / "network_evolution.gif"
        try:
            anim.save(str(gif_file), writer='pillow', fps=1)
            print(f"✅ 網絡演化動畫已生成: {gif_file}")
        except Exception as e:
            print(f"⚠️ 保存GIF失敗: {e}")
            print("請確保安裝了 pillow: pip install pillow")
        
        plt.close()
        
    def generate_metrics_report(self):
        """生成指標報告"""
        if not self.data_timeline:
            print("❌ 沒有找到可視化數據")
            return
            
        # 創建報告
        report = {
            'training_summary': {
                'total_steps': len(self.data_timeline),
                'step_range': f"{self.data_timeline[0]['step']} - {self.data_timeline[-1]['step']}",
                'duration': self.data_timeline[-1]['timestamp'],
                'services_monitored': len(self.service_names)
            },
            'reward_statistics': {
                'initial_reward': self.data_timeline[0]['reward'],
                'final_reward': self.data_timeline[-1]['reward'],
                'max_reward': max(data['reward'] for data in self.data_timeline),
                'min_reward': min(data['reward'] for data in self.data_timeline),
                'avg_reward': np.mean([data['reward'] for data in self.data_timeline])
            },
            'service_statistics': {}
        }
        
        # 計算服務統計
        for i, service_name in enumerate(self.service_names):
            initial_features = self.data_timeline[0]['node_features'][i]
            final_features = self.data_timeline[-1]['node_features'][i]
            
            report['service_statistics'][service_name] = {
                'pod_count_change': final_features[0] - initial_features[0],
                'cpu_usage_change': final_features[2] - initial_features[2],
                'memory_usage_change': final_features[3] - initial_features[3]
            }
        
        # 保存報告
        report_file = self.output_dir / "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ 訓練報告已生成: {report_file}")
        
        # 打印摘要
        print("\n📊 訓練摘要:")
        print(f"   總步數: {report['training_summary']['total_steps']}")
        print(f"   步驟範圍: {report['training_summary']['step_range']}")
        print(f"   監控服務: {report['training_summary']['services_monitored']}")
        print(f"   初始獎勵: {report['reward_statistics']['initial_reward']:.2f}")
        print(f"   最終獎勵: {report['reward_statistics']['final_reward']:.2f}")
        print(f"   平均獎勵: {report['reward_statistics']['avg_reward']:.2f}")
        
    def generate_all_visualizations(self):
        """生成所有可視化"""
        print("🎨 開始生成所有可視化...")
        
        # 1. 交互式儀表板
        html_file = self.generate_interactive_dashboard()
        
        # 2. 網絡演化動畫
        self.generate_network_evolution_gif()
        
        # 3. 指標報告
        self.generate_metrics_report()
        
        print(f"\n✅ 所有可視化已生成完成!")
        print(f"📂 輸出目錄: {self.output_dir}")
        print(f"🌐 查看儀表板: {html_file}")
        
        # 嘗試打開瀏覽器
        if html_file and html_file.exists():
            try:
                webbrowser.open(f"file://{html_file.absolute()}")
                print("🌐 已在瀏覽器中打開儀表板")
            except Exception as e:
                print(f"⚠️ 無法自動打開瀏覽器: {e}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="GNNRL訓練圖形可視化儀表板")
    parser.add_argument('--log-dir', required=True, help='圖形可視化日誌目錄')
    parser.add_argument('--output-dir', help='輸出目錄（可選）')
    parser.add_argument('--dashboard-only', action='store_true', help='只生成儀表板')
    parser.add_argument('--gif-only', action='store_true', help='只生成GIF動畫')
    parser.add_argument('--report-only', action='store_true', help='只生成報告')
    
    args = parser.parse_args()
    
    # 創建儀表板
    dashboard = GraphVisualizationDashboard(args.log_dir)
    
    # 設置輸出目錄
    if args.output_dir:
        dashboard.output_dir = Path(args.output_dir)
        dashboard.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 根據選項生成可視化
    if args.dashboard_only:
        dashboard.generate_interactive_dashboard()
    elif args.gif_only:
        dashboard.generate_network_evolution_gif()
    elif args.report_only:
        dashboard.generate_metrics_report()
    else:
        dashboard.generate_all_visualizations()


if __name__ == "__main__":
    main()