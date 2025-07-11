import datetime
from typing import Dict, Optional

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution, MultiCategoricalDistribution

# Keys accepted by BasePolicy/BaseModel that we want to forward
_BASE_KWARGS = {
    "features_extractor_class",
    "features_extractor_kwargs",
    "features_extractor",
    "normalize_images",
    "optimizer_class",
    "optimizer_kwargs",
}

from gnnrl.core.models.gnn_encoder import HeteroGraphEncoder


class GNNPPOPolicy(ActorCriticPolicy):
    """PPO policy that combines a HeteroGraphEncoder with an MLP head."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        lr_schedule,
        metadata,
        model: str = "gat",
        embed_dim: int = 32,
        **kwargs,
    ):
        # Store GNN-specific parameters
        self.metadata = metadata
        self.model = model
        self.embed_dim = embed_dim
        
        # Filter kwargs to only keep arguments accepted by BasePolicy
        base_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in _BASE_KWARGS}
        super().__init__(observation_space, action_space, lr_schedule, **base_kwargs)
        
        # Build the networks immediately 
        self._build_networks()
        
    def _build_networks(self):
        """Build the networks."""
        self.gnn_encoder = HeteroGraphEncoder(self.metadata, model=self.model, out_dim=self.embed_dim)
        flat_dim = self.observation_space.spaces["flat_feats"].shape[0]
        num_node_types = len(self.metadata[0])
        hidden_dim = self.embed_dim * num_node_types + flat_dim
        
        self.mlp_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
        )
        
        # Action net and value net - MultiDiscrete has multiple outputs
        self.action_net = nn.Linear(64, sum(self.action_space.nvec))
        self.value_net = nn.Linear(64, 1)
        
        # Store action dimensions for distribution creation
        self.action_dims = self.action_space.nvec

    def _get_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Create graph dict for encoder
        graph_obs = {
            'svc_df': obs['svc_df'], 
            'node_df': obs['node_df'],
            'edge_df': obs['edge_df']
        }
        g_emb = self.gnn_encoder(graph_obs)
        flat = torch.as_tensor(obs["flat_feats"], dtype=torch.float32, device=g_emb.device)
        if flat.dim() == 1:
            flat = flat.unsqueeze(0)
        return torch.cat([g_emb, flat], dim=-1)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """Forward pass to get action distribution and value."""
        features = self._get_features(obs)
        latent = self.mlp_extractor(features)
        
        # Get action distribution - MultiDiscrete needs multiple distributions
        logits = self.action_net(latent)
        if "invalid_action_mask" in obs:
            logits = logits + (obs["invalid_action_mask"].to(logits.device) - 1) * 1e10
        
        # Create distribution for MultiDiscrete action space
        from stable_baselines3.common.distributions import MultiCategoricalDistribution
        
        # Split logits according to action space dimensions  
        split_logits = torch.split(logits, self.action_dims.tolist(), dim=-1)
        
        # Create the distribution with the split logits
        distribution = MultiCategoricalDistribution(self.action_dims)
        distribution = distribution.proba_distribution(action_logits=torch.cat(split_logits, dim=-1))
        
        # Get values
        values = self.value_net(latent)
        
        return distribution, values

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Predict actions from observations."""
        distribution, _ = self.forward(observation, deterministic)
        return distribution.get_actions(deterministic=deterministic)
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        """Evaluate actions according to the current policy."""
        distribution, values = self.forward(obs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values.flatten(), log_prob, entropy
    
    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        """Get action distribution for given observations."""
        distribution, _ = self.forward(obs)
        return distribution
        
    def predict_values(self, obs: Dict[str, torch.Tensor]):
        """Get value estimates for given observations."""
        _, values = self.forward(obs)
        return values.flatten()


