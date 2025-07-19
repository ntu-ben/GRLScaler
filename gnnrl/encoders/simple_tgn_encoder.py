import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
import numpy as np


class SimpleTGNEncoder(nn.Module):
    """簡化的TGN實作，避免複雜的維度問題"""
    
    def __init__(self, max_nodes: int, in_dim: int, memory_dim: int = 32):
        super().__init__()
        self.max_nodes = max_nodes
        self.in_dim = in_dim
        self.memory_dim = memory_dim
        
        # 節點記憶體
        self.register_buffer('node_memory', torch.zeros(max_nodes, memory_dim))
        
        # 時間編碼
        self.time_encoder = nn.Linear(1, memory_dim // 4)
        
        # 特徵編碼
        self.node_encoder = nn.Linear(in_dim, memory_dim)
        self.edge_encoder = nn.Linear(7, memory_dim // 4)  # edge_df has 7 features
        
        # 圖卷積層
        self.conv1 = TransformerConv(memory_dim + memory_dim // 4, memory_dim, heads=4, concat=False)
        self.conv2 = TransformerConv(memory_dim, memory_dim, heads=1, concat=False)
        
        # 記憶體更新
        self.memory_update = nn.GRUCell(memory_dim, memory_dim)
        
        # 輸出投影
        self.output_proj = nn.Linear(memory_dim, memory_dim)
        
        self.timestep = 0

    def forward(self, edge_df: torch.Tensor, node_features: torch.Tensor, 
                edge_mask: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        batch_size = node_features.shape[0] if node_features.dim() == 3 else 1
        device = node_features.device
        
        # 處理批次維度
        if node_features.dim() == 3:
            node_features = node_features[0]
        if edge_df.dim() == 3:
            edge_df = edge_df[0]
        if edge_mask.dim() == 2:
            edge_mask = edge_mask[0]
        if node_mask.dim() == 2:
            node_mask = node_mask[0]
        
        self.timestep += 1
        
        # 時間編碼
        time_emb = self.time_encoder(torch.tensor([[self.timestep]], dtype=torch.float32, device=device))
        time_emb = time_emb.expand(self.max_nodes, -1)
        
        # 節點特徵編碼
        node_emb = self.node_encoder(node_features)
        
        # 結合記憶體和時間信息
        combined_features = torch.cat([node_emb, time_emb], dim=1)
        
        # 構建邊索引
        edge_index = self._build_edge_index(edge_df, edge_mask, device)
        
        if edge_index.numel() > 0:
            # 圖卷積
            x = F.relu(self.conv1(combined_features, edge_index))
            x = F.relu(self.conv2(x, edge_index))
        else:
            # 沒有邊時，僅使用線性變換
            x = F.relu(self.output_proj(combined_features))
        
        # 更新記憶體
        valid_nodes = node_mask.bool()
        if valid_nodes.any():
            memory_input = x[valid_nodes]
            old_memory = self.node_memory[valid_nodes]
            new_memory = self.memory_update(memory_input, old_memory)
            self.node_memory[valid_nodes] = new_memory
        
        # 輸出投影
        output = self.output_proj(x)
        
        return output
    
    def _build_edge_index(self, edge_df: torch.Tensor, edge_mask: torch.Tensor, device: torch.device):
        """從edge_df構建edge_index"""
        # 獲取有效邊
        valid_edges = edge_mask.bool()
        
        if not valid_edges.any():
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        # 提取有效邊的索引
        valid_edge_data = edge_df[valid_edges]
        
        if valid_edge_data.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        
        # 確保是2D
        if valid_edge_data.dim() == 1:
            if len(valid_edge_data) >= 2:
                src = valid_edge_data[0:1].long()
                dst = valid_edge_data[1:2].long()
                edge_index = torch.stack([src, dst], dim=0)
            else:
                return torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            src = valid_edge_data[:, 0].long()
            dst = valid_edge_data[:, 1].long()
            edge_index = torch.stack([src, dst], dim=0)
        
        # 確保索引在有效範圍內
        edge_index = torch.clamp(edge_index, 0, self.max_nodes - 1)
        
        return edge_index

    def reset_memory(self):
        """重置記憶體狀態"""
        self.node_memory.zero_()
        self.timestep = 0


# 為了兼容性，保持原有的別名
TGNEncoder = SimpleTGNEncoder