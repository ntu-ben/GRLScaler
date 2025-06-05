import pytest

np = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")
gym = pytest.importorskip("gym")  # gym is a dependency of the env

from gym_hpa.envs.online_boutique import OnlineBoutique, DEPLOYMENTS
from gym_hpa.envs import deployment, online_boutique


def test_fetch_service_graph(monkeypatch):
    nodes = ["recommendationservice", "productcatalogservice"]
    edges = [(0, 1)]
    monkeypatch.setattr(online_boutique, "get_jaeger_service_graph", lambda app_name=None: (nodes, edges))

    env = OnlineBoutique(k8s=False, use_graph=True)
    adj = env._fetch_service_graph()
    assert adj.shape == (len(DEPLOYMENTS), len(DEPLOYMENTS))
    assert adj[0, 1] == 1
    assert np.sum(adj) == 1


def test_reset_returns_graph(monkeypatch):
    nodes = ["recommendationservice", "productcatalogservice"]
    edges = [(0, 1)]
    monkeypatch.setattr(online_boutique, "get_jaeger_service_graph", lambda app_name=None: (nodes, edges))

    env = OnlineBoutique(k8s=False, use_graph=True)
    obs = env.reset()
    assert isinstance(obs, dict)
    assert obs["node_features"].shape == (len(DEPLOYMENTS), 6)
    assert obs["adjacency"].shape == (len(DEPLOYMENTS), len(DEPLOYMENTS))
    assert obs["adjacency"][0, 1] == 1
