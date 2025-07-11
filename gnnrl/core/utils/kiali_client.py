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
    url = f"{KIALI_URL}/api/namespaces/graph?namespaces={namespace}&duration={duration}&graphType=service&injectServiceNodes=true&edges=requestRate,requestErrorRate,responseTime"
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
            # Extract edge metrics from traffic data (Kiali v2.8+ format)
            traffic = ed.get("traffic", {})
            
            # Request rate (QPS)
            qps = traffic.get("rates", {}).get("http", 0.0)
            if qps == 0.0:
                qps = traffic.get("rates", {}).get("tcp", 0.0)
            
            # Response time (P95 latency in ms)
            p95 = traffic.get("responses", {}).get("avg", 0.0)
            if p95 == 0.0:
                # Try alternative response time fields
                p95 = traffic.get("responseTime", {}).get("avg", 0.0)
            
            # Error rate (percentage)
            err = traffic.get("rates", {}).get("httpPercentErr", 0.0)
            
            # mTLS percentage (從父級edge data獲取)
            mtls = float(ed.get("isMTLS", "0"))
            
            edges.append([index[src], index[dst], qps, p95, err, mtls])

    edge_df = pd.DataFrame(edges, columns=["src", "dst", "qps", "p95", "err_rate", "mtls"])
    return nodes, edge_df
