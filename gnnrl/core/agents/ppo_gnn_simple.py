"""
Simple GNN PPO Policy for MultiDiscrete Action Space
This approach uses a flattened action space to work with standard PPO.
"""

import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

from gnnrl.core.models.gnn_encoder import HeteroGraphEncoder


class FlattenedGNNPPOPolicy(ActorCriticPolicy):
    """
    PPO policy that flattens MultiDiscrete to Discrete for easier training.
    """
    
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
        # Store original action space
        self.original_action_space = action_space
        self.metadata = metadata
        self.model = model
        self.embed_dim = embed_dim
        
        # Create flattened action space (total combinations)
        flattened_action_space = spaces.Discrete(int(action_space.nvec.prod()))
        
        # Initialize with flattened space
        super().__init__(observation_space, flattened_action_space, lr_schedule, **kwargs)
        
        # Build networks
        self._build_networks()
    
    def _build_networks(self):
        """Build GNN encoder and policy networks."""
        self.gnn_encoder = HeteroGraphEncoder(self.metadata, model=self.model, out_dim=self.embed_dim)
        
        flat_dim = self.observation_space.spaces["flat_feats"].shape[0]
        num_node_types = len(self.metadata[0])
        hidden_dim = self.embed_dim * num_node_types + flat_dim
        
        # Feature extractor
        self.features_extractor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
        )
        
        # Policy and value networks
        self.action_net = nn.Linear(64, self.action_space.n)
        self.value_net = nn.Linear(64, 1)
    
    def _get_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features using GNN encoder."""
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
    
    def _flatten_action(self, multi_action):
        """Convert MultiDiscrete action to flattened Discrete action."""
        if len(multi_action.shape) == 1:
            multi_action = multi_action.unsqueeze(0)
        
        flattened = torch.zeros(multi_action.shape[0], dtype=torch.long, device=multi_action.device)
        multiplier = 1
        
        for i in reversed(range(len(self.original_action_space.nvec))):
            flattened += multi_action[:, i] * multiplier
            multiplier *= self.original_action_space.nvec[i]
        
        return flattened
    
    def _unflatten_action(self, flattened_action):
        """Convert flattened Discrete action to MultiDiscrete action."""
        if len(flattened_action.shape) == 0:
            flattened_action = flattened_action.unsqueeze(0)
        
        multi_action = torch.zeros((flattened_action.shape[0], len(self.original_action_space.nvec)), 
                                 dtype=torch.long, device=flattened_action.device)
        
        remaining = flattened_action.clone()
        for i in reversed(range(len(self.original_action_space.nvec))):
            multi_action[:, i] = remaining % self.original_action_space.nvec[i]
            remaining = remaining // self.original_action_space.nvec[i]
        
        return multi_action.squeeze() if multi_action.shape[0] == 1 else multi_action
    
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """Forward pass."""
        features = self._get_features(obs)
        latent = self.features_extractor(features)
        
        # Get action distribution
        logits = self.action_net(latent)
        distribution = CategoricalDistribution(logits)
        
        # Get values
        values = self.value_net(latent)
        
        return distribution, values
    
    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Predict actions."""
        distribution, _ = self.forward(observation, deterministic)
        flattened_actions = distribution.get_actions(deterministic=deterministic)
        return self._unflatten_action(flattened_actions)
    
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        """Evaluate actions."""
        # Flatten actions for evaluation
        flattened_actions = self._flatten_action(actions)
        
        distribution, values = self.forward(obs)
        log_prob = distribution.log_prob(flattened_actions)
        entropy = distribution.entropy()
        
        return values.flatten(), log_prob, entropy
    
    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        """Get action distribution."""
        distribution, _ = self.forward(obs)
        return distribution
    
    def predict_values(self, obs: Dict[str, torch.Tensor]):
        """Predict values."""
        _, values = self.forward(obs)
        return values.flatten()