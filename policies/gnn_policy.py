import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple

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

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """Extract node and edge information using a GAT encoder."""

    def __init__(self, observation_space: spaces.Dict, gat_hidden_dim: int = 32, gat_output_dim: int = 32):
        super().__init__(observation_space, features_dim=gat_output_dim)
        feat_dim = observation_space.spaces["node_features"].shape[1]
        self.gat = GATEncoder(feat_dim, gat_hidden_dim, gat_output_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        node = torch.as_tensor(observations["node_features"], dtype=torch.float32, device=device)
        adj = torch.as_tensor(observations["adjacency"], dtype=torch.float32, device=device)
        if node.dim() == 2:
            node = node.unsqueeze(0)
            adj = adj.unsqueeze(0)

        batch_embeddings = []
        for n, a in zip(node, adj):
            edge_index = a.nonzero(as_tuple=False).t().contiguous()
            h = self.gat(n, edge_index)
            batch_embeddings.append(h.mean(dim=0))

        return torch.stack(batch_embeddings, dim=0)


class GNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, gat_hidden_dim=32, gat_output_dim=32, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs=dict(gat_hidden_dim=gat_hidden_dim, gat_output_dim=gat_output_dim),
            **kwargs,
        )

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.set_training_mode(False)
        obs_tensor, _ = self.obs_to_tensor(observation)
        actions = self._predict(obs_tensor, deterministic=deterministic)
        actions = actions.reshape((-1, *self.action_space.shape))
        if isinstance(self.action_space, (spaces.Discrete, spaces.MultiDiscrete)):
            actions = actions.long()
        actions = np.array(actions.cpu().tolist(), dtype=int)
        if actions.shape[0] == 1:
            actions = actions[0]
        return actions, state
