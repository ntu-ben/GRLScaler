from queue import Queue
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gnn_rl.common.feature_builder import RealtimeFeatureBuilder


class K8sEnv(gym.Env):
    """Minimal env that pulls observations from RealtimeFeatureBuilder."""

    def __init__(self, queue: Optional[Queue] = None):
        super().__init__()
        self.queue = queue or Queue()
        self.builder = RealtimeFeatureBuilder(self.queue)
        # Observation/action spaces will be inferred after first reset
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        obs = self.builder.get_obs()
        self._update_spaces(obs)
        return obs

    def step(self, action):
        obs = self.builder.get_obs()
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info

    def _update_spaces(self, obs):
        if self.observation_space.spaces:
            return
        g = obs['graph']
        self.observation_space = spaces.Dict({
            'graph': spaces.Dict({
                'svc_df': spaces.Box(-np.inf, np.inf, shape=g['svc_df'].shape, dtype=np.float32),
                'node_df': spaces.Box(-np.inf, np.inf, shape=g['node_df'].shape, dtype=np.float32),
                'edge_df': spaces.Box(-np.inf, np.inf, shape=g['edge_df'].shape, dtype=np.float32),
            }),
            'flat_feats': spaces.Box(-np.inf, np.inf, shape=obs['flat_feats'].shape, dtype=np.float32),
        })
