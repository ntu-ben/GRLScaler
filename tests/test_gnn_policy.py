import pytest

np = pytest.importorskip("numpy")
gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")
from gnn_rl.gnn_policy import GNNActorCriticPolicy
from gnn_rl.agents.ppo_gnn import GNNPPOPolicy
PPO = sb3.PPO

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


def test_gnnppo_forward():
    obs_space = gym.spaces.Dict({
        'graph': gym.spaces.Dict({
            'svc_df': gym.spaces.Box(-np.inf, np.inf, shape=(2, 3), dtype=np.float32),
            'node_df': gym.spaces.Box(-np.inf, np.inf, shape=(1, 2), dtype=np.float32),
            'edge_df': gym.spaces.Box(-np.inf, np.inf, shape=(1, 4), dtype=np.float32),
        }),
        'flat_feats': gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
    })
    action_space = gym.spaces.MultiBinary(3)
    policy = GNNPPOPolicy(obs_space, action_space, lambda _: 0.001,
                           metadata=(['svc', 'node'], [('svc', 'calls', 'svc')]))
    obs = {
        'graph': {
            'svc_df': np.random.rand(2, 3).astype(np.float32),
            'node_df': np.random.rand(1, 2).astype(np.float32),
            'edge_df': np.random.rand(1, 4).astype(np.float32),
        },
        'flat_feats': np.zeros(2, dtype=np.float32),
    }
    logits, value = policy(obs)
    assert logits.shape[0] == 1
    assert value.shape == (1, 1)
