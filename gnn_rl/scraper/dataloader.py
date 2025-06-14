import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import torch
from torch_geometric.data import Data

from . import settings


def fetch_edges(namespace: str, timestamp: Optional[float] = None,
                viz_url: str = settings.VIZ_URL,
                prom_url: str = settings.PROM_URL) -> pd.DataFrame:
    """Fetch service edges via Linkerd-viz REST. Fallback to Prometheus."""
    url = f"{viz_url}/api/namespaces/{namespace}/edges"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json().get("edges", [])
        records = []
        for e in data:
            src = e.get("src", {}).get("name")
            dst = e.get("dst", {}).get("name")
            stats = e.get("clientStats", {}) or {}
            rps = float(stats.get("requestRate", stats.get("rps", 0)) or 0)
            p99 = float(stats.get("latencyP99", 0))
            records.append({"src": src, "dst": dst, "rps": rps, "p99_ms": p99})
        return pd.DataFrame(records)
    except Exception:
        # fallback using a simple PromQL query
        query = (
            f'sum by (src,dst) (rate(request_total{{namespace="{namespace}"}}[1m]))'
        )
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=5)
        r.raise_for_status()
        results = r.json().get("data", {}).get("result", [])
        records = []
        for res in results:
            m = res.get("metric", {})
            records.append({
                "src": m.get("src"),
                "dst": m.get("dst"),
                "rps": float(res.get("value", [0, 0])[1]),
                "p99_ms": float("nan"),
            })
        return pd.DataFrame(records)


def fetch_node_features(namespace: str, timestamp: Optional[float] = None,
                        prom_url: str = settings.PROM_URL) -> pd.DataFrame:
    """Fetch node features from Prometheus."""
    def q(query: str):
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=5)
        r.raise_for_status()
        return r.json().get("data", {}).get("result", [])

    cpu = q(f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[1m])) by (deployment)')
    mem = q(f'sum(container_memory_working_set_bytes{{namespace="{namespace}"}}) by (deployment)')
    rps = q(f'sum(rate(request_total{{namespace="{namespace}"}}[1m])) by (deployment)')
    p99 = q(
        'histogram_quantile(0.99, sum(irate(request_latency_bucket{namespace="%s"}[1m])) by (le, deployment))'
        % namespace)
    succ = q(f'sum(rate(request_success_total{{namespace="{namespace}"}}[1m])) by (deployment)')

    def to_map(results, key):
        m = {}
        for r in results:
            dep = r.get("metric", {}).get("deployment")
            if dep is None:
                continue
            m[dep] = float(r.get("value", [0, 0])[1])
        return m

    cpu_m = to_map(cpu, "cpu")
    mem_m = {k: v / 1024 / 1024 for k, v in to_map(mem, "mem").items()}
    rps_m = to_map(rps, "rps")
    p99_m = {k: v * 1000 for k, v in to_map(p99, "p99").items()}
    succ_m = to_map(succ, "succ")

    services = set(cpu_m) | set(mem_m) | set(rps_m) | set(p99_m) | set(succ_m)
    records = []
    for svc in services:
        records.append({
            "svc": svc,
            "cpu_m": cpu_m.get(svc, 0.0),
            "mem_mib": mem_m.get(svc, 0.0),
            "in_rps": rps_m.get(svc, 0.0),
            "in_p99_ms": p99_m.get(svc, 0.0),
            "succ_rate": succ_m.get(svc, 0.0),
        })
    return pd.DataFrame(records)


def save_data(edges_df: pd.DataFrame, nodes_df: pd.DataFrame,
              out_dir: Path = settings.DATA_DIR) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(out_dir / "edges.csv", index=False)
    nodes_df.to_csv(out_dir / "node_features.csv", index=False)


def build_pyg(edges_df: Optional[pd.DataFrame] = None,
              nodes_df: Optional[pd.DataFrame] = None,
              data_dir: Path = settings.DATA_DIR) -> Data:
    if edges_df is None:
        edges_df = pd.read_csv(data_dir / "edges.csv")
    if nodes_df is None:
        nodes_df = pd.read_csv(data_dir / "node_features.csv")

    index = {svc: i for i, svc in enumerate(nodes_df["svc"])}
    edge_index = torch.tensor(
        [[index[row.src], index[row.dst]] for row in edges_df.itertuples()],
        dtype=torch.long,
    ).t().contiguous()
    edge_attr = torch.tensor(
        edges_df[["rps", "p99_ms"]].to_numpy(dtype=float), dtype=torch.float32
    )
    x = torch.tensor(
        nodes_df[["cpu_m", "mem_mib", "in_rps", "in_p99_ms", "succ_rate"]].to_numpy(dtype=float),
        dtype=torch.float32,
    )
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
