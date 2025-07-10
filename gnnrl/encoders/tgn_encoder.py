"""Minimal Temporal Graph Network encoder."""

from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn.models import TGNMemory
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder, IdentityMessage, LastAggregator


class TGNEncoder(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, memory_dim: int = 32, msg_dim: int = 32):
        super().__init__()
        self.time_encoder = TimeEncoder(memory_dim)
        self.memory = TGNMemory(
            num_nodes=num_nodes, 
            raw_msg_dim=msg_dim, 
            memory_dim=memory_dim, 
            time_dim=memory_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, msg_dim),
            aggregator_module=LastAggregator()
        )
        self.conv = TransformerConv(in_dim, memory_dim, heads=2)

    def forward(self, src, dst, t, x) -> torch.Tensor:
        """Encode events."""
        # Create edge_index from src and dst
        edge_index = torch.stack([src, dst], dim=0)
        
        # Apply transformer convolution with proper edge_index
        out = self.conv(x, edge_index)
        return out

    def update_memory(self, src, dst, t, msg):
        # Use update_state instead of update_memory
        self.memory.update_state(torch.cat([src, dst]), torch.cat([t, t]), torch.cat([msg, msg]))
