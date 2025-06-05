import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from policies.gnn_policy import GNNActorCriticPolicy

class DummyGraphEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(-np.inf, np.inf, shape=(2, 6), dtype=np.float32),
            'adjacency': gym.spaces.Box(0, 1, shape=(2, 2), dtype=np.float32),
        })
        self.action_space = gym.spaces.MultiDiscrete([2, 15])

    def reset(self, *, seed=None, options=None):
        obs = self.observation_space.sample()
        obs = {k: v.tolist() for k, v in obs.items()}
        return obs, {}

    def step(self, action):
        obs = self.observation_space.sample()
        obs = {k: v.tolist() for k, v in obs.items()}
        return obs, 0.0, True, False, {}


def test_policy_forward():
    env = DummyGraphEnv()
    model = PPO(GNNActorCriticPolicy, env, n_steps=2, batch_size=2, n_epochs=1, learning_rate=0.001)
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    assert env.action_space.contains(action)
