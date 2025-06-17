# Operations Guide

This guide describes how to run the real data collector and train the autoscaler with a Kubernetes cluster using Linkerd and Prometheus.

## 1. Configure Endpoints

Set the following environment variables or replace the CLI arguments accordingly:

- `PROMETHEUS_URL` – base URL of your Prometheus server
- `LINKERD_VIZ_API_URL` – Linkerd viz API endpoint (usually `http://localhost:8084`)

Ensure your cluster allows unauthenticated access to `/metrics` or provide the proper token.

## 2. Start the Data Collector

```bash
python -m data_collector.linkerd_prom --edges-url $LINKERD_VIZ_API_URL/api/edges \
    --metrics-url $PROMETHEUS_URL/api/v1/query
```

The collector prints JSON dictionaries to stdout. You can redirect them to a file or feed them into a message queue for `RealtimeFeatureBuilder`.

## 3. Training

Run the training script with the desired GNN encoder:

```bash
python scripts/train_gnnppo.py --model gat --steps 100000
```

TensorBoard logs are written to `runs/gnnppo/`.

## 4. Benchmark

Execute the benchmark script to compare different policies. Results are saved under the `results/` directory.

```bash
python scripts/benchmark.py --steps 10000 --seeds 3 --output results
```

After running, see `results/summary.csv` and `results/summary.md` for aggregated metrics.
