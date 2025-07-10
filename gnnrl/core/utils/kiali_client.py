import logging
import os
from typing import List, Tuple

import pandas as pd
import requests

KIALI_URL = os.getenv("KIALI_URL", "http://localhost:20001/kiali")


def fetch_service_graph(namespace: str, duration: str = "5s") -> Tuple[List[str], pd.DataFrame]:
    """Fetch service graph with edge metrics from Kiali.

    Parameters
    ----------
    namespace: str
        Namespace to query.
    duration: str
        Query duration, default 5s.

    Returns
    -------
    Tuple of node list and edge dataframe with columns
    ``src``, ``dst``, ``qps``, ``p95`` (ms), ``err_rate``.
    """
    url = f"{KIALI_URL}/api/graph?namespaces={namespace}&duration={duration}&graphType=workload&edgeLabels=all"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception as e:  # pragma: no cover - best effort
        logging.error("Failed to query Kiali: %s", e)
        return [], pd.DataFrame(columns=["src", "dst", "qps", "p95", "err_rate"])

    data = resp.json().get("elements", {})
    nodes = []
    index = {}
    for node in data.get("nodes", []):
        name = node.get("data", {}).get("workload") or node.get("data", {}).get("app")
        if name and name not in index:
            index[name] = len(nodes)
            nodes.append(name)

    edges = []
    for edge in data.get("edges", []):
        ed = edge.get("data", {})
        src = ed.get("source")
        dst = ed.get("target")
        if src in index and dst in index:
            # Extract edge metrics; fall back to 0 if missing
            metrics = ed.get("metrics", {})
            qps = metrics.get("requestRate", 0.0)
            p95 = metrics.get("responseTime", {}).get("avg", 0.0)
            err = metrics.get("errorRate", 0.0)
            edges.append([index[src], index[dst], qps, p95, err])

    edge_df = pd.DataFrame(edges, columns=["src", "dst", "qps", "p95", "err_rate"])
    return nodes, edge_df
