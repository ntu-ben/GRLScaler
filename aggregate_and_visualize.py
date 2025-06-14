#!/usr/bin/env python3
"""
aggregate_and_visualize.py
==========================
Sweep every Locust result under a **logs** folder, build a long‑form summary
CSV, draw a handful of PNG charts, and output a single `report.html` page.

Feature checklist
-----------------
✓ Walk arbitrary depth `logs/<hpa>/<scenario>` structure  
✓ Handle multiple Locust column name variants (≤ v2.8 / 2.9+ / history CSV)  
✓ Fallback to *_stats_history.csv when *_stats.csv missing or has no "Aggregated" row  
✓ Compute extra field **Fail‑rate %** = Failures / Requests ×100  
✓ Save long‑form `combined_summary.csv` & pretty HTML table  
✓ Charts (matplotlib default colors)*:  
  • Avg RPS per scenario (grouped bar)  
  • 95‑percentile latency per scenario (line)  
  • Fail‑rate heat‑map (scenario × HPA)  
*You can always tweak styles; script sticks to project rules (no seaborn, no explicit colors).

Run
---
```bash
$ python aggregate_and_visualize.py \
      --log-root /Users/hopohan/Desktop/k8s/intelliscaler/logs \
      --out-dir  /Users/hopohan/Desktop/k8s/intelliscaler/report
```
If `--out-dir` omitted → create sibling folder `report` next to log root.
"""
from __future__ import annotations
import argparse, logging, textwrap
from pathlib import Path
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt

# ── constants ───────────────────────────────────────────────────────────
COL_MAP: Dict[str, str] = {
    # Locust stats.csv v2.9+
    "Request Count": "Requests",
    "Failure Count": "Failures",
    "Requests/s":    "Avg RPS",
    "95%":           "P95 ms",
    # Locust ≤2.8 (lower‑case num_*)
    "num_requests":  "Requests",
    "num_failures":  "Failures",
    # stats_history.csv cumulative
    "Total Requests": "Requests",
    "Total Failures": "Failures",
    "Total Requests/s": "Avg RPS",
    "response_time_percentile_0.95": "P95 ms",
}

CHARTS = [
    ("Effective RPS", "effective_rps.png", "Effective RPS (Excludes Failed Requests)"),
    ("P95 ms",        "p95_latency.png", "95th‑percentile latency (ms)"),
]

# ── helpers ─────────────────────────────────────────────────────────────

def extract_metrics(stats_csv: Path | None, hist_csv: Path | None) -> dict[str, float] | None:
    """Return dict of metrics, or None if both files unusable."""
    if stats_csv and stats_csv.exists():
        df = pd.read_csv(stats_csv)
        row = df[df["Name"].isin(["Aggregated", "Total"])]
        if not row.empty:
            row = row.iloc[0]
            return _row_to_metrics(row)
    if hist_csv and hist_csv.exists():
        df = pd.read_csv(hist_csv)
        if not df.empty:
            row = df.iloc[-1]  # last timestamp
            return _row_to_metrics(row)
    return None


def _row_to_metrics(row) -> dict[str, float]:
    out: Dict[str, float] = {k: None for k in set(COL_MAP.values())}
    for orig, std in COL_MAP.items():
        if orig in row and pd.notna(row[orig]):
            out[std] = row[orig]
    # compute fail‑rate later when we have both numbers
    return out

# ── main routine ────────────────────────────────────────────────────────

def main():
    argp = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(__doc__))
    argp.add_argument("--log-root", default="logs/hpa", help="root folder holding HPA result dirs")
    argp.add_argument("--out-dir",  default=None, help="output dir (default <log_root>/../report)")
    args = argp.parse_args()

    log_root = Path(args.log_root).expanduser().resolve()
    out_dir  = Path(args.out_dir) if args.out_dir else log_root.parent / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = []

    # depth‑first search: logs/<hpa>/<scenario>
    for hpa_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
        for scn_dir in sorted(p for p in hpa_dir.iterdir() if p.is_dir()):
            stats_csv = next(scn_dir.glob("*_stats.csv"), None)
            hist_csv  = next(scn_dir.glob("*_stats_history.csv"), None)
            metrics = extract_metrics(stats_csv, hist_csv)
            if metrics is None:
                logging.warning("skip %s/%s (no usable csv)", hpa_dir.name, scn_dir.name)
                continue
            if metrics["Requests"]:
                metrics["Fail‑rate %"] = (metrics["Failures"] or 0) / metrics["Requests"] * 100
                metrics["Effective RPS"] = metrics["Avg RPS"] * (1 - metrics["Fail‑rate %"] / 100)
            else:
                metrics["Fail‑rate %"] = None
                metrics["Effective RPS"] = None
     
            rows.append({
                "HPA": hpa_dir.name,
                "Scenario": scn_dir.name,
                **metrics,
            })

    if not rows:
        logging.error("No metrics found under %s", log_root)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "combined_summary.csv", index=False)

    # ── charts ──────────────────────────────────────────────────────────
    for metric, fname, title in CHARTS:
        piv = df.pivot(index="HPA", columns="Scenario", values=metric)
        plt.figure(figsize=(12,6))
        piv.plot(kind="bar", ax=plt.gca())
        plt.title(title)
        plt.xlabel("HPA tier")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / fname)
        plt.close()

    # Fail‑rate heatmap
    piv = df.pivot(index="Scenario", columns="HPA", values="Fail‑rate %")
    fig, ax = plt.subplots(figsize=(12,6))
    cax = ax.imshow(piv, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("Failure rate (%)")
    fig.colorbar(cax, ax=ax)
    plt.tight_layout()
    plt.savefig(out_dir / "fail_rate_heatmap.png")
    plt.close()

    # ── HTML report -----------------------------------------------------
    html = ["<html><head><title>HPA Test Report</title></head><body>"]
    html.append("<h1>Combined Summary</h1>")
    html.append(df.to_html(index=False))
    for _, fname, title in CHARTS:
        html.append(f"<h2>{title}</h2><img src='{fname}' width='900'>")
    html.append("<h2>Failure rate (%)</h2><img src='fail_rate_heatmap.png' width='900'>")
    html.append("</body></html>")
    (out_dir/"report.html").write_text("\n".join(html), encoding="utf-8")
    logging.info("Report written to %s", out_dir/"report.html")

if __name__ == "__main__":
    main()

