"""Temporal Graph Network encoder with dynamic node mapping support."""

from typing import Tuple, Dict, Optional
import torch
from torch import nn
from torch_geometric.nn.models import TGNMemory
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.models.tgn import TimeEncoder, IdentityMessage, LastAggregator


class DynamicTGNEncoder(nn.Module):
    """TGN encoder with dynamic node mapping and memory management."""
    
    def __init__(self, max_nodes: int, in_dim: int, memory_dim: int = 32, msg_dim: int = 32):
        super().__init__()
        self.max_nodes = max_nodes
        self.memory_dim = memory_dim
        self.msg_dim = msg_dim
        self.in_dim = in_dim
        
        # Node mapping: service_name -> node_id
        self.node_mapping: Dict[str, int] = {}
        self.active_nodes = set()
        
        # TGN components
        self.time_encoder = TimeEncoder(memory_dim)
        self.memory = TGNMemory(
            num_nodes=max_nodes, 
            raw_msg_dim=msg_dim, 
            memory_dim=memory_dim, 
            time_dim=memory_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, msg_dim),
            aggregator_module=LastAggregator()
        )
        self.conv = TransformerConv(in_dim, memory_dim, heads=2)
        
        # Memory state tracking
        self.node_memory_states = {}
        self.timestep = 0

    def update_node_mapping(self, service_names: list) -> Dict[str, int]:
        """Update node mapping based on current active services."""
        old_mapping = self.node_mapping.copy()
        
        # Create new mapping
        new_mapping = {}
        for i, service_name in enumerate(service_names):
            if i < self.max_nodes:
                new_mapping[service_name] = i
        
        # Handle mapping changes
        if old_mapping != new_mapping:
            self._remap_memory(old_mapping, new_mapping)
            self.node_mapping = new_mapping
            self.active_nodes = set(new_mapping.values())
            
        return self.node_mapping

    def _remap_memory(self, old_mapping: Dict[str, int], new_mapping: Dict[str, int]):
        """Remap memory states when node mapping changes."""
        if not hasattr(self.memory, 'memory') or self.memory.memory is None:
            return
            
        # Save memory states for services that still exist
        preserved_states = {}
        for service_name, old_id in old_mapping.items():
            if service_name in new_mapping:
                new_id = new_mapping[service_name]
                if old_id < self.memory.memory.shape[0]:
                    preserved_states[new_id] = self.memory.memory[old_id].clone()
        
        # Reset memory and restore preserved states
        self.memory.reset_state()
        if preserved_states:
            for new_id, state in preserved_states.items():
                if new_id < self.memory.memory.shape[0]:
                    self.memory.memory[new_id] = state

    def forward(self, edge_data: torch.Tensor, node_features: torch.Tensor, 
                edge_mask: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic graph support."""
        self.timestep += 1
        
        # Filter valid edges and nodes
        edge_mask_bool = edge_mask.bool()
        node_mask_bool = node_mask.bool()
        
        # Get valid edges by indexing with mask
        valid_edge_indices = torch.where(edge_mask_bool)[0]
        valid_node_indices = torch.where(node_mask_bool)[0]
        
        if len(valid_edge_indices) == 0 or len(valid_node_indices) == 0:
            return torch.zeros(self.max_nodes, self.memory_dim, dtype=torch.float32)
        
        valid_edges = edge_data[valid_edge_indices]
        valid_nodes = node_features[valid_node_indices]
        
        if len(valid_edges) == 0 or len(valid_nodes) == 0:
            # Return padded output for consistency
            return torch.zeros(self.max_nodes, self.memory_dim, dtype=torch.float32)
        
        # Extract edge information
        src = valid_edges[:, 0].long()
        dst = valid_edges[:, 1].long()
        edge_features = valid_edges[:, 3:]  # Skip src, dst, active columns
        
        # Ensure src and dst are within valid range
        src = torch.clamp(src, 0, self.max_nodes - 1)
        dst = torch.clamp(dst, 0, self.max_nodes - 1)
        
        # Create messages from edge features
        msg = self._create_messages(edge_features)
        
        # Update memory with current interactions
        if len(src) > 0:
            t = torch.full((len(src),), self.timestep, dtype=torch.float32)
            self.update_memory(src, dst, t, msg)
        
        # Get updated node representations
        memory_states = self.memory.memory if hasattr(self.memory, 'memory') else None
        if memory_states is not None:
            # Combine node features with memory
            num_valid_nodes = len(valid_nodes)
            combined_features = torch.zeros(self.max_nodes, self.in_dim, dtype=torch.float32)
            combined_features[:num_valid_nodes] = valid_nodes
            
            # Create edge index for convolution
            if len(src) > 0:
                edge_index = torch.stack([src, dst], dim=0)
                out = self.conv(combined_features, edge_index)
                
                # Add memory states
                out[:num_valid_nodes] += memory_states[:num_valid_nodes]
                return out
        
        # Fallback: return node features if no memory
        output = torch.zeros(self.max_nodes, self.memory_dim, dtype=torch.float32)
        num_valid_nodes = len(valid_nodes)
        if num_valid_nodes > 0:
            # Project node features to memory dimension
            projected = torch.nn.functional.linear(valid_nodes, 
                                                 torch.randn(self.in_dim, self.memory_dim))
            output[:num_valid_nodes] = projected
        
        return output

    def _create_messages(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Create messages from edge features."""
        if edge_features.shape[1] < self.msg_dim:
            # Pad features to message dimension
            padding = torch.zeros(len(edge_features), self.msg_dim - edge_features.shape[1])
            return torch.cat([edge_features, padding], dim=1)
        else:
            # Truncate to message dimension
            return edge_features[:, :self.msg_dim]

    def update_memory(self, src: torch.Tensor, dst: torch.Tensor, 
                     t: torch.Tensor, msg: torch.Tensor):
        """Update memory with temporal interactions."""
        # Filter nodes that are within bounds
        valid_mask = (src < self.max_nodes) & (dst < self.max_nodes)
        if valid_mask.sum() > 0:
            nodes = torch.cat([src[valid_mask], dst[valid_mask]])
            timestamps = torch.cat([t[valid_mask], t[valid_mask]])
            messages = torch.cat([msg[valid_mask], msg[valid_mask]])
            
            # Update memory state
            self.memory.update_state(nodes, timestamps, messages, messages)

    def reset_memory(self):
        """Reset memory states."""
        self.memory.reset_state()
        self.node_memory_states.clear()
        self.timestep = 0

    def get_memory_state(self) -> Optional[torch.Tensor]:
        """Get current memory state."""
        return self.memory.memory if hasattr(self.memory, 'memory') else None


# Backward compatibility
TGNEncoder = DynamicTGNEncoder
