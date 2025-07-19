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
            # TGN expects edge_df, svc_df, edge_mask, node_mask
            if self.encoder is None:
                # 從觀察空間獲取正確的維度
                svc_df_tensor = obs_dict["svc_df"]
                if isinstance(svc_df_tensor, torch.Tensor) and svc_df_tensor.dim() == 3:
                    svc_df_tensor = svc_df_tensor[0]
                
                max_nodes = svc_df_tensor.shape[0]
                node_feat_dim = svc_df_tensor.shape[1]
                self.encoder = TGNEncoder(max_nodes, node_feat_dim, memory_dim=self.out_dim)
            
            # 使用動態圖數據
            edge_df = obs_dict["edge_df"]
            node_features = obs_dict["svc_df"]
            edge_mask = obs_dict.get("edge_mask", torch.ones(edge_df.shape[-2] if edge_df.dim() == 3 else edge_df.shape[0]))
            node_mask = obs_dict.get("node_mask", torch.ones(node_features.shape[-2] if node_features.dim() == 3 else node_features.shape[0]))
            
            emb = self.encoder(edge_df, node_features, edge_mask, node_mask)
            h = {"svc": emb, "node": obs_dict["node_df"]}
        else:
            h = self.encoder(data.x_dict, data.edge_index_dict)
        pooled = []
        for ntype in data.node_types:
            emb = h.get(ntype)
            if emb is None:
                emb = data[ntype].x
            if emb.size(-1) < self.out_dim:
                # 確保維度匹配
                pad_shape = list(emb.shape)
                pad_shape[-1] = self.out_dim - emb.size(-1)
                pad = emb.new_zeros(pad_shape)
                emb = torch.cat([emb, pad], dim=-1)
            elif emb.size(-1) > self.out_dim:
                emb = emb[..., :self.out_dim]

            # 確保pooled結果維度一致
            pooled_emb = emb.mean(dim=0)
            if pooled_emb.dim() == 0:  # 如果是標量，添加維度
                pooled_emb = pooled_emb.unsqueeze(0)
            elif pooled_emb.dim() > 1:  # 如果維度過高，flatten
                pooled_emb = pooled_emb.flatten()
            pooled.append(pooled_emb)

        # 確保所有pooled張量都是1維
        pooled_tensors = []
        for p in pooled:
            if p.dim() == 0:
                pooled_tensors.append(p.unsqueeze(0))
            elif p.dim() > 1:
                pooled_tensors.append(p.flatten())
            else:
                pooled_tensors.append(p)
        
        final_embedding = torch.cat(pooled_tensors).unsqueeze(0)
        return final_embedding
