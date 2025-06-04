import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from stable_baselines3.common.policies import ActorCriticPolicy

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=1):
        super().__init__()
        layers = [GATConv(in_channels, hidden_channels, heads=heads)]
        for _ in range(num_layers - 2):
            layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))
        self.convs = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)

class GNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, gat_hidden_dim=32, gat_output_dim=32, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        feat_dim = observation_space.spaces['node_features'].shape[1]
        self.gat = GATEncoder(feat_dim, gat_hidden_dim, gat_output_dim)
        self.graph_pool = lambda h: h.mean(dim=0)
        out_dim = action_space.n if hasattr(action_space, 'n') else int(sum(action_space.nvec))
        self.policy_net = nn.Linear(gat_output_dim, out_dim)
        self.value_net = nn.Linear(gat_output_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.policy_net.weight)
        nn.init.xavier_uniform_(self.value_net.weight)
        nn.init.zeros_(self.policy_net.bias)
        nn.init.zeros_(self.value_net.bias)

    def forward(self, obs, deterministic=False):
        node = torch.as_tensor(obs['node_features']).float().to(self.device)
        adj = torch.as_tensor(obs['adjacency']).to(self.device)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        node_h = self.gat(node, edge_index)
        graph_h = self.graph_pool(node_h)
        logits = self.policy_net(graph_h)
        value = self.value_net(graph_h)
        return logits, value
