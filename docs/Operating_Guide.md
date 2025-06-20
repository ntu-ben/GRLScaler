# Operations Guide

This guide describes how to run the real data collector and train the autoscaler with a Kubernetes cluster using Istio/Kiali and Prometheus.

## 1. Configure Endpoints

Set the following environment variables or replace the CLI arguments accordingly:

- `PROMETHEUS_URL` – base URL of your Prometheus server
- `KIALI_URL` – Kiali endpoint (usually `http://localhost:30326/kiali`)

Ensure your cluster allows unauthenticated access to `/metrics` or provide the proper token.

## 2. Start the Data Collector

```bash
python -m data_collector.kiali_prom --graph-url $KIALI_URL/api/namespaces/<ns>/graph \
    --metrics-url $PROMETHEUS_URL/api/v1/query
```

The collector prints JSON dictionaries to stdout. You can redirect them to a file or feed them into a message queue for `RealtimeFeatureBuilder`.

## 3. Training

Run the training script with the desired GNN encoder. Add `--k8s` to interact with a live cluster:

```bash
python scripts/train_gnnppo.py --use-case redis --model gat --steps 100000 --k8s
```

TensorBoard logs are written to `runs/gnnppo/`.

## 4. Benchmark

Execute the benchmark script to compare different policies. Results are saved under the `results/` directory.

```bash
python scripts/benchmark.py --steps 10000 --seeds 3 --output results
```

After running, see `results/summary.csv` and `results/summary.md` for aggregated metrics.
