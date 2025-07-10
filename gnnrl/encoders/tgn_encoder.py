"""Minimal Temporal Graph Network encoder."""

from typing import Tuple

import torch
from torch import nn
from torch_geometric.nn.models import TGNMemory
from torch_geometric.nn.models.tgn import TransformerConv, TimeEncoder


class TGNEncoder(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, memory_dim: int = 32, msg_dim: int = 32):
        super().__init__()
        self.time_encoder = TimeEncoder(memory_dim)
        self.memory = TGNMemory(num_nodes, memory_dim, message_dimension=msg_dim, time_dimension=memory_dim)
        self.conv = TransformerConv(in_dim, memory_dim, heads=2)

    def forward(self, src, dst, t, x) -> torch.Tensor:
        """Encode events."""
        t = self.time_encoder(t)
        self.memory.detach_memory()
        out = self.conv(x, src, dst, t, self.memory.memory)
        return out

    def update_memory(self, src, dst, t, msg):
        self.memory.update_memory(torch.cat([src, dst]), torch.cat([t, t]), torch.cat([msg, msg]))
