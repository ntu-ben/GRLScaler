import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, GCNConv

from gnnrl.core.common.feature_builder import build_hetero_data
from gnnrl.encoders import TGNEncoder


class HeteroGAT(nn.Module):
    def __init__(self, metadata, hidden_dim=32, out_dim=32, num_layers=2, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for rel in metadata[1]:
                out_c = out_dim if i == num_layers - 1 else hidden_dim
                conv_dict[rel] = GATConv(
                    (-1, -1),
                    out_c,
                    heads=heads if i != num_layers - 1 else 1,
                    concat=i != num_layers - 1,
                    add_self_loops=False,  # Disable self loops for heterogeneous graphs
                )
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.elu(v) for k, v in x_dict.items()}
        return x_dict


class GCNEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim=32, out_dim=32, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for rel in metadata[1]:
                out_c = out_dim if i == num_layers - 1 else hidden_dim
                conv_dict[rel] = GCNConv(-1, out_c)
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict


class HeteroGraphEncoder(nn.Module):
    """Wrapper that pools node embeddings into a single vector."""

    def __init__(self, metadata, model="gat", hidden_dim=32, out_dim=32):
        super().__init__()
        self.model = model
        if model == "gat":
            self.encoder = HeteroGAT(metadata, hidden_dim, out_dim)
        elif model == "gcn":
            self.encoder = GCNEncoder(metadata, hidden_dim, out_dim)
        elif model == "tgn":
            # Created lazily on first forward since num_nodes may be unknown
            self.encoder = None
            self.hidden_dim = hidden_dim
            self.out_dim = out_dim
        else:
            raise ValueError(f"Unknown model: {model}")
        self.metadata = metadata
        self.out_dim = out_dim

    def forward(self, obs_dict):
        # Handle batch dimension - take first sample if batched
        svc_df = obs_dict["svc_df"]
        node_df = obs_dict["node_df"] 
        edge_df = obs_dict["edge_df"]
        
        if isinstance(svc_df, torch.Tensor) and svc_df.dim() == 3:
            svc_df = svc_df[0]  # Remove batch dimension
        if isinstance(node_df, torch.Tensor) and node_df.dim() == 3:
            node_df = node_df[0]
        if isinstance(edge_df, torch.Tensor) and edge_df.dim() == 3:
            edge_df = edge_df[0]
            
        data = build_hetero_data(svc_df, node_df, edge_df)

        if self.model == "tgn":
            num_nodes = data["svc"].x.size(0)
            if self.encoder is None:
                self.encoder = TGNEncoder(num_nodes, data["svc"].x.size(1), memory_dim=self.out_dim)
            edge_index = data["svc", "calls", "svc"].edge_index
            if edge_index.numel() == 0:
                return data["svc"].x.mean(dim=0, keepdim=True)
            src, dst = edge_index
            t = torch.zeros(src.size(0), device=src.device)
            emb = self.encoder(src, dst, t, data["svc"].x)
            self.encoder.update_memory(src, dst, t, emb[src])
            h = {"svc": emb, "node": data["node"].x}
        else:
            h = self.encoder(data.x_dict, data.edge_index_dict)
        pooled = []
        for ntype in data.node_types:
            emb = h.get(ntype)
            if emb is None:
                emb = data[ntype].x
            if emb.size(-1) < self.out_dim:
                pad = emb.new_zeros(emb.size(0), self.out_dim - emb.size(-1))
                emb = torch.cat([emb, pad], dim=-1)
            elif emb.size(-1) > self.out_dim:
                emb = emb[..., :self.out_dim]

            pooled.append(emb.mean(dim=0))

        final_embedding = torch.cat(pooled).unsqueeze(0)
        return final_embedding
