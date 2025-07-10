"""Dump Prometheus metrics to 1Hz CSV files."""

import csv
import os
import requests
from datetime import datetime
from pathlib import Path

PROM_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

METRICS = [
    "container_cpu_usage_seconds_total",
    "container_memory_working_set_bytes",
]

INTERVAL = 1  # seconds
DURATION = 300  # total seconds to collect


def query_range(metric: str, start: float, end: float, step: int = 1):
    params = {
        "query": metric,
        "start": start,
        "end": end,
        "step": step,
    }
    resp = requests.get(f"{PROM_URL}/api/v1/query_range", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]["result"]


def main(namespace: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now().timestamp()
    start = end - DURATION

    for metric in METRICS:
        data = query_range(metric + f'{{namespace="{namespace}"}}', start, end, step=INTERVAL)
        out_file = out_dir / f"{metric}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value"])
            for series in data:
                for ts, val in series.get("values", []):
                    writer.writerow([datetime.fromtimestamp(float(ts)).isoformat(), val])
        print(f"Saved {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump Prometheus CSV")
    parser.add_argument("namespace", help="K8s namespace")
    parser.add_argument("output", help="Output directory")
    args = parser.parse_args()

    main(args.namespace, Path(args.output))
