import datetime
from typing import Dict, Optional

import torch
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy

from gnn_rl.models.gnn_encoder import HeteroGraphEncoder


class GNNPPOPolicy(BasePolicy):
    """PPO policy that combines a HeteroGraphEncoder with an MLP head."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Dict,
        lr_schedule,
        metadata,
        model: str = "gat",
        embed_dim: int = 32,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.gnn_encoder = HeteroGraphEncoder(metadata, model=model, out_dim=embed_dim)
        flat_dim = observation_space.spaces["flat_feats"].shape[0]
        hidden_dim = embed_dim + flat_dim
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.flat_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def _get_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        g_emb = self.gnn_encoder(obs["graph"])
        flat = torch.as_tensor(obs["flat_feats"], dtype=torch.float32, device=g_emb.device)
        return torch.cat([g_emb, flat], dim=-1)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        state = self._get_features(obs)
        logits = self.actor(state)
        values = self.critic(state)
        if "invalid_action_mask" in obs:
            logits = logits + (obs["invalid_action_mask"].to(logits.device) - 1) * 1e10
        return logits, values

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        logits, _ = self.forward(observation, deterministic)
        distribution = self._get_action_dist_from_logits(logits)
        return distribution.get_actions(deterministic=deterministic)


