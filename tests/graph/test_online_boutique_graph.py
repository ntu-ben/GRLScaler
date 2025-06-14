import pytest

np = pytest.importorskip("numpy")
pandas = pytest.importorskip("pandas")
gym = pytest.importorskip("gym")  # gym is a dependency of the env

from gnn_rl.envs.online_boutique import OnlineBoutique, DEPLOYMENTS
from gnn_rl.scraper import dataloader


def test_fetch_service_graph(monkeypatch):
    df = pandas.DataFrame({
        "src": ["recommendationservice"],
        "dst": ["productcatalogservice"],
        "rps": [1.0],
        "p99_ms": [10.0],
    })
    monkeypatch.setattr(dataloader, "fetch_edges", lambda namespace, **kw: df)

    env = OnlineBoutique(k8s=False, use_graph=True)
    adj = env._fetch_service_graph()
    assert adj.shape == (len(DEPLOYMENTS), len(DEPLOYMENTS))
    assert adj[0, 1] == 1
    assert np.sum(adj) == 1


def test_reset_returns_graph(monkeypatch):
    df = pandas.DataFrame({
        "src": ["recommendationservice"],
        "dst": ["productcatalogservice"],
        "rps": [1.0],
        "p99_ms": [10.0],
    })
    monkeypatch.setattr(dataloader, "fetch_edges", lambda namespace, **kw: df)

    env = OnlineBoutique(k8s=False, use_graph=True)
    obs = env.reset()
    assert isinstance(obs, dict)
    assert obs["node_features"].shape == (len(DEPLOYMENTS), 6)
    assert obs["adjacency"].shape == (len(DEPLOYMENTS), len(DEPLOYMENTS))
    assert obs["adjacency"][0, 1] == 1
