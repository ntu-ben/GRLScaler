import json
from queue import Queue

import pytest
np = pytest.importorskip("numpy")

gym = pytest.importorskip("gymnasium")

from gnn_rl.agents.ppo_gnn import GNNPPOPolicy
from gnn_rl.common.feature_builder import build_hetero_data
from gnn_rl.envs import K8sEnv


def test_env_reset_and_policy_forward(tmp_path):
    q = Queue()
    edges = json.load(open('tests/fixtures/edges.json'))
    metrics = json.load(open('tests/fixtures/metrics.json'))
    q.put({'edges': edges, 'metrics': metrics})
    env = K8sEnv(q)
    obs = env.reset()

    assert 'graph' in obs and 'flat_feats' in obs
    g = obs['graph']

    metadata = build_hetero_data(g['svc_df'], g['node_df'], g['edge_df']).metadata()

    obs_space = gym.spaces.Dict({
        'graph': gym.spaces.Dict({
            'svc_df': gym.spaces.Box(-np.inf, np.inf, shape=g['svc_df'].shape, dtype=np.float32),
            'node_df': gym.spaces.Box(-np.inf, np.inf, shape=g['node_df'].shape, dtype=np.float32),
            'edge_df': gym.spaces.Box(-np.inf, np.inf, shape=g['edge_df'].shape, dtype=np.float32),
        }),
        'flat_feats': gym.spaces.Box(-np.inf, np.inf, shape=obs['flat_feats'].shape, dtype=np.float32),
    })
    action_space = gym.spaces.MultiBinary(3)

    policy = GNNPPOPolicy(obs_space, action_space, lambda _: 0.001, metadata=metadata)
    logits, value = policy(obs)
    assert logits.shape[0] == 1
    assert value.shape == (1, 1)
