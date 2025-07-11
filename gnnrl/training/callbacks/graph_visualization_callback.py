"""
Graph Visualization Callback for GNNRL Training
==============================================

每500步輸出一次圖形數據，包括：
- 網絡拓撲圖
- 節點特徵變化
- 邊特徵變化  
- 訓練指標趨勢
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import networkx as nx
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class GraphVisualizationCallback(BaseCallback):
    """
    每500步輸出一次圖形數據的回調
    """
    
    def __init__(self, 
                 save_freq: int = 500,
                 output_dir: str = "graph_outputs",
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 數據存儲
        self.step_data = []
        self.node_features_history = []
        self.edge_features_history = []
        self.reward_history = []
        self.action_history = []
        
        # 圖形設置
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.dpi'] = 100
        
    def _on_step(self) -> bool:
        """每步調用"""
        if self.n_calls % self.save_freq == 0:
            self._save_graph_data()
            
        return True
        
    def _save_graph_data(self):
        """保存圖形數據"""
        step = self.n_calls
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建步驟輸出目錄
        step_dir = self.output_dir / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 獲取當前環境狀態
            env_state = self._get_env_state()
            
            if env_state:
                # 1. 生成網絡拓撲圖
                self._generate_network_topology(env_state, step_dir, step)
                
                # 2. 生成節點特徵圖
                self._generate_node_features(env_state, step_dir, step)
                
                # 3. 生成邊特徵圖
                self._generate_edge_features(env_state, step_dir, step)
                
                # 4. 生成訓練指標趨勢圖
                self._generate_training_metrics(step_dir, step)
                
                # 5. 保存原始數據
                self._save_raw_data(env_state, step_dir, step)
                
                if self.verbose > 0:
                    print(f"📊 Step {step}: Graph data saved to {step_dir}")
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Step {step}: Failed to save graph data: {e}")
    
    def _get_env_state(self) -> Optional[Dict[str, Any]]:
        """獲取環境狀態"""
        try:
            # 從訓練環境中獲取狀態
            env = self.training_env.envs[0] if hasattr(self.training_env, 'envs') else self.training_env
            
            if hasattr(env, 'get_state'):
                obs = env.get_state()
                
                # 如果是圖形環境，獲取圖形數據
                if isinstance(obs, dict) and 'svc_df' in obs:
                    return {
                        'step': self.n_calls,
                        'node_features': obs['svc_df'].copy(),
                        'edge_features': obs['edge_df'].copy(),
                        'adjacency': getattr(env, '_last_adjacency', None),
                        'service_names': getattr(env, 'service_names', [f"Service_{i}" for i in range(len(obs['svc_df']))]),
                        'reward': getattr(env, '_last_reward', 0),
                        'action': getattr(env, '_last_action', None)
                    }
            
            return None
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to get env state: {e}")
            return None
    
    def _generate_network_topology(self, env_state: Dict[str, Any], output_dir: Path, step: int):
        """生成網絡拓撲圖"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # 創建網絡圖
            G = nx.DiGraph()
            
            # 添加節點
            service_names = env_state['service_names']
            node_features = env_state['node_features']
            
            for i, service_name in enumerate(service_names):
                # 節點大小基於pod數量
                pod_count = node_features[i][0] if len(node_features[i]) > 0 else 1
                cpu_usage = node_features[i][2] if len(node_features[i]) > 2 else 0
                
                G.add_node(service_name, 
                          pod_count=pod_count,
                          cpu_usage=cpu_usage,
                          size=max(300, pod_count * 200))
            
            # 添加邊（如果有鄰接矩陣）
            if env_state['adjacency'] is not None:
                adj = env_state['adjacency']
                for i in range(len(service_names)):
                    for j in range(len(service_names)):
                        if adj[i][j] > 0:
                            G.add_edge(service_names[i], service_names[j], weight=adj[i][j])
            
            # 布局
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # 繪製節點
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                cpu_usage = G.nodes[node]['cpu_usage']
                # CPU使用率影響顏色
                if cpu_usage > 80:
                    node_colors.append('red')
                elif cpu_usage > 60:
                    node_colors.append('orange')
                elif cpu_usage > 40:
                    node_colors.append('yellow')
                else:
                    node_colors.append('lightgreen')
                
                node_sizes.append(G.nodes[node]['size'])
            
            # 繪製網絡
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
            
            if G.edges():
                nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowsize=20, ax=ax)
            
            # 添加標題和圖例
            ax.set_title(f'Service Network Topology - Step {step}', fontsize=14, fontweight='bold')
            
            # 創建圖例
            legend_elements = [
                patches.Patch(color='lightgreen', label='CPU < 40%'),
                patches.Patch(color='yellow', label='CPU 40-60%'),
                patches.Patch(color='orange', label='CPU 60-80%'),
                patches.Patch(color='red', label='CPU > 80%')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            ax.set_axis_off()
            
            plt.tight_layout()
            plt.savefig(output_dir / f'network_topology_step_{step:08d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to generate network topology: {e}")
    
    def _generate_node_features(self, env_state: Dict[str, Any], output_dir: Path, step: int):
        """生成節點特徵圖"""
        try:
            node_features = env_state['node_features']
            service_names = env_state['service_names']
            
            # 特徵名稱（根據實際環境調整）
            feature_names = ['Pod Count', 'Desired Replicas', 'CPU Usage (%)', 'Memory Usage (MB)', 'RX Traffic', 'TX Traffic']
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feature_name in enumerate(feature_names):
                if i < len(feature_names) and i < node_features.shape[1]:
                    values = node_features[:, i]
                    
                    # 條形圖
                    bars = axes[i].bar(range(len(service_names)), values, alpha=0.7)
                    
                    # 設置顏色
                    for j, bar in enumerate(bars):
                        if feature_name == 'CPU Usage (%)':
                            if values[j] > 80:
                                bar.set_color('red')
                            elif values[j] > 60:
                                bar.set_color('orange')
                            elif values[j] > 40:
                                bar.set_color('yellow')
                            else:
                                bar.set_color('lightgreen')
                        else:
                            bar.set_color('steelblue')
                    
                    axes[i].set_title(feature_name, fontweight='bold')
                    axes[i].set_xticks(range(len(service_names)))
                    axes[i].set_xticklabels(service_names, rotation=45, ha='right')
                    axes[i].grid(True, alpha=0.3)
                    
                    # 添加數值標籤
                    for j, v in enumerate(values):
                        axes[i].text(j, v + max(values) * 0.01, f'{v:.1f}', 
                                   ha='center', va='bottom', fontsize=8)
            
            plt.suptitle(f'Node Features - Step {step}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / f'node_features_step_{step:08d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to generate node features: {e}")
    
    def _generate_edge_features(self, env_state: Dict[str, Any], output_dir: Path, step: int):
        """生成邊特徵圖"""
        try:
            edge_features = env_state['edge_features']
            
            if edge_features.shape[0] > 0:
                # 邊特徵名稱
                edge_feature_names = ['Source', 'Destination', 'Active', 'QPS', 'P95 Latency', 'Error Rate', 'Custom']
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes = axes.flatten()
                
                # 只顯示活躍的邊
                active_edges = edge_features[edge_features[:, 2] > 0]  # 假設第3列是active標誌
                
                if len(active_edges) > 0:
                    # QPS分佈
                    qps_values = active_edges[:, 3]
                    axes[0].hist(qps_values, bins=20, alpha=0.7, color='skyblue')
                    axes[0].set_title('QPS Distribution', fontweight='bold')
                    axes[0].set_xlabel('QPS')
                    axes[0].set_ylabel('Frequency')
                    axes[0].grid(True, alpha=0.3)
                    
                    # P95延遲分佈
                    p95_values = active_edges[:, 4]
                    axes[1].hist(p95_values, bins=20, alpha=0.7, color='lightcoral')
                    axes[1].set_title('P95 Latency Distribution', fontweight='bold')
                    axes[1].set_xlabel('P95 Latency (ms)')
                    axes[1].set_ylabel('Frequency')
                    axes[1].grid(True, alpha=0.3)
                    
                    # 錯誤率分佈
                    error_rates = active_edges[:, 5]
                    axes[2].hist(error_rates, bins=20, alpha=0.7, color='gold')
                    axes[2].set_title('Error Rate Distribution', fontweight='bold')
                    axes[2].set_xlabel('Error Rate (%)')
                    axes[2].set_ylabel('Frequency')
                    axes[2].grid(True, alpha=0.3)
                    
                    # QPS vs P95延遲散點圖
                    axes[3].scatter(qps_values, p95_values, alpha=0.6, color='purple')
                    axes[3].set_title('QPS vs P95 Latency', fontweight='bold')
                    axes[3].set_xlabel('QPS')
                    axes[3].set_ylabel('P95 Latency (ms)')
                    axes[3].grid(True, alpha=0.3)
                else:
                    for ax in axes:
                        ax.text(0.5, 0.5, 'No Active Edges', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('No Data Available')
                
                plt.suptitle(f'Edge Features - Step {step}', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_dir / f'edge_features_step_{step:08d}.png', dpi=150, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to generate edge features: {e}")
    
    def _generate_training_metrics(self, output_dir: Path, step: int):
        """生成訓練指標趨勢圖"""
        try:
            # 記錄當前步驟的獎勵
            if hasattr(self.locals, 'rewards') and self.locals['rewards'] is not None:
                current_reward = np.mean(self.locals['rewards'])
                self.reward_history.append((step, current_reward))
            
            if len(self.reward_history) > 1:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                steps, rewards = zip(*self.reward_history)
                ax.plot(steps, rewards, 'b-', linewidth=2, alpha=0.7)
                ax.scatter(steps, rewards, color='red', s=30, alpha=0.7, zorder=5)
                
                ax.set_title('Training Reward Trend', fontsize=14, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Average Reward')
                ax.grid(True, alpha=0.3)
                
                # 添加趨勢線
                if len(steps) > 3:
                    z = np.polyfit(steps, rewards, 1)
                    p = np.poly1d(z)
                    ax.plot(steps, p(steps), "r--", alpha=0.8, linewidth=1, label=f'Trend: {z[0]:.6f}x + {z[1]:.2f}')
                    ax.legend()
                
                plt.tight_layout()
                plt.savefig(output_dir / f'training_metrics_step_{step:08d}.png', dpi=150, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to generate training metrics: {e}")
    
    def _save_raw_data(self, env_state: Dict[str, Any], output_dir: Path, step: int):
        """保存原始數據"""
        try:
            # 轉換numpy數組為列表以便JSON序列化
            data = {
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'node_features': env_state['node_features'].tolist(),
                'edge_features': env_state['edge_features'].tolist(),
                'service_names': env_state['service_names'],
                'reward': float(env_state['reward']),
                'action': env_state['action'].tolist() if env_state['action'] is not None else None
            }
            
            # 保存為JSON
            with open(output_dir / f'raw_data_step_{step:08d}.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            # 保存為CSV（節點特徵）
            node_df_path = output_dir / f'node_features_step_{step:08d}.csv'
            np.savetxt(node_df_path, env_state['node_features'], delimiter=',', 
                      header=','.join(['pod_count', 'desired_replicas', 'cpu_usage', 'mem_usage', 'rx_traffic', 'tx_traffic']),
                      comments='')
            
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Failed to save raw data: {e}")