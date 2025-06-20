import time
import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from gnn_rl.common.feature_builder import build_hetero_data
from gnn_rl.models.gnn_encoder import HeteroGraphEncoder


def test_forward_shape_and_speed():
    svc_df = pd.DataFrame(np.random.rand(3, 4))
    node_df = pd.DataFrame(np.random.rand(2, 3))
    edge_df = pd.DataFrame({"src": [0, 1], "dst": [1, 2], "rps": [0.5, 1.0]})

    data = build_hetero_data(svc_df, node_df, edge_df)
    encoder = HeteroGraphEncoder(data.metadata(), hidden_dim=8, out_dim=8)

    obs = {"svc_df": svc_df, "node_df": node_df, "edge_df": edge_df}
    start = time.perf_counter()
    out = encoder(obs)
    duration = time.perf_counter() - start

    assert out.shape == (1, 16)
    assert duration < 0.1
