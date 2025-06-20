import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _to_numpy(data):
    if isinstance(data, pd.DataFrame):
        return data.values
    return np.asarray(data)


def build_hetero_data(svc_df, node_df, edge_df):
    """Convert service/node/edge tables to ``HeteroData``.

    ``edge_df`` should contain at least ``src`` and ``dst`` columns or the first
    two columns will be treated as such.
    """
    svc = _to_numpy(svc_df).astype(np.float32)
    node = _to_numpy(node_df).astype(np.float32)
    edges = _to_numpy(edge_df)

    if edges.size > 0:
        max_idx = int(max(edges[:, 0].max(), edges[:, 1].max()))
        if svc.shape[0] <= max_idx:
            pad = np.zeros((max_idx + 1 - svc.shape[0], svc.shape[1]), dtype=svc.dtype)
            svc = np.vstack([svc, pad])

    data = HeteroData()
    data["svc"].x = torch.tensor(svc, dtype=torch.float32)
    data["node"].x = torch.tensor(node, dtype=torch.float32)

    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    data["svc", "calls", "svc"].edge_index = torch.tensor([src, dst], dtype=torch.long)

    if edges.shape[1] > 2:
        attr = edges[:, 2:].astype(np.float32)
        data["svc", "calls", "svc"].edge_attr = torch.tensor(attr, dtype=torch.float32)

    return data


class RealtimeFeatureBuilder:
    """Build observations from raw metric dictionaries placed on a queue."""

    def __init__(self, queue):
        self.queue = queue

    def get_obs(self):
        item = self.queue.get()
        edges = item.get('edges', {}).get('edges', [])
        metrics = item.get('metrics', {})
        svc_df = pd.DataFrame(metrics.get('svc', []))
        node_df = pd.DataFrame(metrics.get('node', []))
        edge_df = pd.DataFrame(edges)
        scalar = np.asarray(metrics.get('scalar', []), dtype=np.float32)
        return {
            'graph': {
                'svc_df': svc_df,
                'node_df': node_df,
                'edge_df': edge_df,
            },
            'flat_feats': scalar,
        }
