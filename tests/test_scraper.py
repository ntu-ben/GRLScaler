import pandas as pd
import types
import pytest

from gnn_rl.scraper import dataloader


class DummyResponse:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data
    def raise_for_status(self):
        pass


def fake_get(url, params=None, timeout=5):
    if 'edges' in url:
        return DummyResponse({'edges': [{'src': {'name': 'a'}, 'dst': {'name': 'b'}, 'clientStats': {'requestRate': '1', 'latencyP99': '0.5'}}]})
    if params and 'container_cpu' in params.get('query', ''):
        return DummyResponse({'data': {'result': [{'metric': {'deployment': 'a'}, 'value': [0, '0.1']}]}})
    if params and 'memory_working_set_bytes' in params.get('query', ''):
        return DummyResponse({'data': {'result': [{'metric': {'deployment': 'a'}, 'value': [0, '1048576']}]}})
    if params and 'request_total' in params.get('query', ''):
        return DummyResponse({'data': {'result': [{'metric': {'deployment': 'a'}, 'value': [0, '2']}]}})
    if params and 'request_latency_bucket' in params.get('query', ''):
        return DummyResponse({'data': {'result': [{'metric': {'deployment': 'a'}, 'value': [0, '0.02']}]}})
    if params and 'request_success_total' in params.get('query', ''):
        return DummyResponse({'data': {'result': [{'metric': {'deployment': 'a'}, 'value': [0, '2']}]}})
    return DummyResponse({})


def test_fetchers(monkeypatch, tmp_path):
    monkeypatch.setattr(dataloader.requests, 'get', fake_get)
    edges = dataloader.fetch_edges('ns')
    nodes = dataloader.fetch_node_features('ns')
    assert list(edges.columns) == ['src', 'dst', 'rps', 'p99_ms']
    assert list(nodes.columns) == ['svc', 'cpu_m', 'mem_mib', 'in_rps', 'in_p99_ms', 'succ_rate']
    dataloader.save_data(edges, nodes, tmp_path)


def test_build_pyg():
    edges_df = pd.DataFrame({'src': ['a'], 'dst': ['b'], 'rps': [1], 'p99_ms': [2]})
    nodes_df = pd.DataFrame({
        'svc': ['a', 'b'],
        'cpu_m': [0, 0],
        'mem_mib': [0, 0],
        'in_rps': [0, 0],
        'in_p99_ms': [0, 0],
        'succ_rate': [0, 0],
    })
    data = dataloader.build_pyg(edges_df, nodes_df)
    assert data.edge_index.size(1) == len(edges_df)
    assert data.x.shape[1] == 5
