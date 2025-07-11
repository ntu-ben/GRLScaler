"""
動態圖觀察空間
==============

支援可變節點數量和邊數量的觀察空間設計
"""

import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DynamicGraphConfig:
    """動態圖配置"""
    max_nodes: int = 20  # 最大節點數
    max_edges: int = 400  # 最大邊數 (20*20)
    node_feat_dim: int = 6  # 節點特徵維度
    edge_feat_dim: int = 7  # 邊特徵維度
    global_feat_dim: int = 4  # 全局特徵維度


class DynamicGraphSpace:
    """動態圖觀察空間管理器"""
    
    def __init__(self, config: DynamicGraphConfig):
        self.config = config
        self.current_nodes = []  # 當前節點列表
        self.node_mapping = {}   # 節點名稱到ID的映射
        self.reverse_mapping = {}  # ID到節點名稱的映射
        
    def create_observation_space(self) -> spaces.Dict:
        """創建動態觀察空間"""
        return spaces.Dict({
            'svc_df': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.config.max_nodes, self.config.node_feat_dim), 
                dtype=np.float32
            ),
            'node_df': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(1, 4), 
                dtype=np.float32
            ),
            'edge_df': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.config.max_edges, self.config.edge_feat_dim), 
                dtype=np.float32
            ),
            'flat_feats': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.config.global_feat_dim,), 
                dtype=np.float32
            ),
            'node_mask': spaces.Box(
                low=0, high=1, 
                shape=(self.config.max_nodes,), 
                dtype=np.float32
            ),
            'edge_mask': spaces.Box(
                low=0, high=1, 
                shape=(self.config.max_edges,), 
                dtype=np.float32
            ),
            'invalid_action_mask': spaces.Box(
                low=0, high=1, 
                shape=(self.config.max_nodes * 15,),  # 15個可能動作
                dtype=np.float32
            ),
        })
    
    def update_node_mapping(self, active_services: List[str]) -> Dict[str, int]:
        """更新節點映射"""
        # 保持現有節點的ID穩定性
        new_mapping = {}
        new_reverse = {}
        
        # 首先保留已存在的節點
        next_id = 0
        for service in active_services:
            if service in self.node_mapping:
                # 保持原有ID
                new_mapping[service] = self.node_mapping[service]
                new_reverse[self.node_mapping[service]] = service
                next_id = max(next_id, self.node_mapping[service] + 1)
        
        # 為新節點分配ID
        for service in active_services:
            if service not in new_mapping:
                new_mapping[service] = next_id
                new_reverse[next_id] = service
                next_id += 1
        
        self.current_nodes = active_services
        self.node_mapping = new_mapping
        self.reverse_mapping = new_reverse
        
        return new_mapping
    
    def pad_node_features(self, node_features: np.ndarray, active_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """填充節點特徵到最大維度"""
        padded_features = np.zeros((self.config.max_nodes, self.config.node_feat_dim), dtype=np.float32)
        node_mask = np.zeros(self.config.max_nodes, dtype=np.float32)
        
        # 填充實際特徵
        if active_count > 0:
            padded_features[:active_count] = node_features[:active_count]
            node_mask[:active_count] = 1.0
        
        return padded_features, node_mask
    
    def pad_edge_features(self, edge_features: np.ndarray, active_edges: int) -> Tuple[np.ndarray, np.ndarray]:
        """填充邊特徵到最大維度"""
        padded_edges = np.zeros((self.config.max_edges, self.config.edge_feat_dim), dtype=np.float32)
        edge_mask = np.zeros(self.config.max_edges, dtype=np.float32)
        
        # 填充實際邊特徵
        if active_edges > 0:
            padded_edges[:active_edges] = edge_features[:active_edges]
            edge_mask[:active_edges] = 1.0
        
        return padded_edges, edge_mask

    def pad_global_features(self, global_features: np.ndarray) -> np.ndarray:
        """
        填充全局特徵到最大維度
        
        Args:
            global_features: 全局特徵數組 (global_feat_dim,)
            
        Returns:
            padded_global: 填充後的全局特徵 (global_feat_dim,)
        """
        padded_global = np.zeros(self.config.global_feat_dim, dtype=np.float32)
        
        if len(global_features) > 0:
            actual_features = min(len(global_features), self.config.global_feat_dim)
            padded_global[:actual_features] = global_features[:actual_features]
            
        return padded_global
    
    def create_action_mask(self, deployment_list) -> np.ndarray:
        """創建動作掩碼"""
        mask = np.zeros(self.config.max_nodes * 15, dtype=np.float32)
        
        for i, deployment in enumerate(deployment_list):
            if i >= self.config.max_nodes:
                break
                
            base_idx = i * 15
            
            # 所有動作默認有效
            for action in range(15):
                action_idx = base_idx + action
                
                if action == 0:  # DO_NOTHING
                    mask[action_idx] = 1.0
                elif 1 <= action <= 7:  # ADD_REPLICA
                    n = action
                    mask[action_idx] = 1.0 if deployment.num_pods + n <= deployment.max_pods else 0.0
                elif 8 <= action <= 14:  # TERMINATE_REPLICA
                    n = action - 7
                    mask[action_idx] = 1.0 if deployment.num_pods - n >= deployment.min_pods else 0.0
        
        return mask
    
    def get_node_id(self, service_name: str) -> Optional[int]:
        """獲取服務節點ID"""
        return self.node_mapping.get(service_name)
    
    def get_service_name(self, node_id: int) -> Optional[str]:
        """獲取節點對應的服務名稱"""
        return self.reverse_mapping.get(node_id)