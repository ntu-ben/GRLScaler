#!/usr/bin/env python3
"""
rl_batch_loadtest.py  v4  (2025‑06‑05)
────────────────────────────────────────────────────────────────────────────
* 重大更新：**遠端優先、失敗再本機** 的壓測流程
  1. 若環境變數 `M1_HOST` 存在 (ex. `http://m1:8000`)，
     ‑  先用 REST API 觸發 m1 的 `locust_agent` 進行壓測；
     ‑  下載 `*_stats.csv` / `*.html` 回到 m4 原有資料夾結構；
  2. 若 `M1_HOST` 未設定，或 m1 無法連線 / 回應錯誤，
     自動 fallback → 在 m4 本機直接執行 Locust。

這樣保證「只要 m1 可用，就由 m1 壓測；否則 m4 自行頂上」，滿足你的需求。
其他功能（--model/gym vs grl, 報表彙整…）保持不變。
"""
from __future__ import annotations
import argparse, subprocess, time, datetime as dt, logging, sys, os, re, json
from pathlib import Path
import pandas as pd, requests
from jinja2 import Template

# ╭─ Config (Cluster & Locust) ───────────────────────────────────────────╮
NAMESPACE     = "onlineboutique"
TARGET_HOST   = "http://frontend.onlineboutique.svc.cluster.local"
HEALTH_PATH   = "/"
HTTP_TIMEOUT  = 600
SCENARIOS = {
    "offpeak":     "locust_offpeak.py",
    "rushsale":    "locust_rushsale.py",
    "peak":        "locust_peak.py",
    "fluctuating": "locust_fluctuating.py",
}
RUN_TIME   = "15m"
LOG_ROOT   = Path("logs")
LOCUST_DIR = Path(__file__).parent / "stressTest" / "onlineboutique"

# 遠端 Locust Agent
M1_HOST        = os.getenv("M1_HOST")        # ex. "http://m1:8000"
print(M1_HOST)

STATUS_POLL    = 10      # 秒
REMOTE_TIMEOUT = 5       # 單次 HTTP timeout

# ╭─ model registry ──────────────────────────────────────────────────────╮
MODEL_ROOT = {
    "gym": Path("/Users/hopohan/Desktop/k8s/gym-hpa"),
    "grl": Path("/Users/hopohan/Desktop/k8s/GRLScaler"),
}

# ╭─ Helpers ─────────────────────────────────────────────────────────────╮

def sh(cmd, **kw):
    logging.info("$ %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def wait_frontend_ready():
    url = TARGET_HOST.rstrip("/") + HEALTH_PATH
    deadline = time.time() + HTTP_TIMEOUT
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError("frontend never became ready")

# ── Local Locust (fallback) ────────────────────────────────────────────

def run_locust_local(scn: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "locust", "-f", LOCUST_DIR / SCENARIOS[scn],
        "--headless", "--run-time", RUN_TIME,
        "--host", TARGET_HOST,
        "--csv", out_dir / scn, "--csv-full-history",
        "--html", out_dir / f"{scn}.html",
    ]
    try:
        sh(cmd)
    except subprocess.CalledProcessError as e:
        logging.warning("Locust %s exit‑code %s", scn, e.returncode)

# ── Remote Locust (preferred) ──────────────────────────────────────────

def run_locust_remote(scn: str, out_dir: Path, tag: str):
    if not M1_HOST:
        raise RuntimeError("M1_HOST not set")
    payload = {
        "tag": f"{tag}_{scn}",
        "scenario": scn,
        "target_host": TARGET_HOST,
        "run_time": RUN_TIME,
    }
    try:
        r = requests.post(f"{M1_HOST}/start", json=payload, timeout=REMOTE_TIMEOUT)
        r.raise_for_status()
        job_id = r.json()["job_id"]
    except Exception as e:
        raise RuntimeError(f"start failed: {e}") from e

    # poll until finished
    while True:
        try:
            st = requests.get(f"{M1_HOST}/status/{job_id}", timeout=REMOTE_TIMEOUT).json()
            if st.get("finished"):
                break
        except Exception:
            pass
        time.sleep(STATUS_POLL)

    # download result files
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in [f"{scn}_stats.csv", f"{scn}_stats_history.csv", f"{scn}.html"]:
        try:
            d = requests.get(f"{M1_HOST}/download/{payload['tag']}/{fn}", timeout=REMOTE_TIMEOUT)
            if d.status_code == 200:
                (out_dir / fn).write_bytes(d.content)
        except Exception as e:
            logging.warning("download %s failed: %s", fn, e)

# ── Unified wrapper ────────────────────────────────────────────────────

def run_locust(scn: str, out_dir: Path, tag: str):
    if M1_HOST:
        try:
            logging.info("🌐 remote locust (%s)…", scn)
            run_locust_remote(scn, out_dir, tag)
            return
        except Exception as e:
            logging.warning("remote failed (%s) → fallback local (%s)", scn, e)
    run_locust_local(scn, out_dir)

# ╭─ Reporting ──────────────────────────────────────────────────────────╮

def summarise(iter_dir: Path, tag: str):
    rows = []
    for scn in SCENARIOS:
        csv = next((iter_dir / scn).glob("*_stats.csv"), None)
        if not csv:
            continue
        df = pd.read_csv(csv)
        tot = df[df["Name"].isin(["Aggregated", "Total"])]
        if tot.empty:
            continue
        t = tot.iloc[0]
        rows.append({
            "Iter": tag,
            "Scenario": scn,
            "Req": t.get("Request Count", 0),
            "Fail": t.get("Failure Count", 0),
            "RPS": t.get("Requests/s", 0),
            "P95": t.get("95%", 0),
        })
    if rows:
        pd.DataFrame(rows).to_csv(iter_dir / "summary.csv", index=False)


def render_dashboard(df: pd.DataFrame, out_dir: Path):
    html = Template("""<html><head><title>{{ tag }}</title></head><body>
    {{ tbl | safe }}
    <ul>{% for _, r in df.iterrows() %}
      <li><a href='{{ r.IterDir }}/summary.csv'>{{ r.Iter }}</a></li>
    {% endfor %}</ul></body></html>""").render(tag=out_dir.name, tbl=df.to_html(index=False), df=df)
    (out_dir / "aggregate.html").write_text(html, encoding="utf-8")

# ╭─ util: resolve latest zip ───────────────────────────────────────────╮

def resolve_load_path(raw: str | None) -> str | None:
    if raw is None:
        return None
    p = Path(raw).expanduser().resolve()
    if p.is_file():
        return p.as_posix()
    if not p.is_dir():
        raise FileNotFoundError(raw)
    zips = sorted(p.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("no .zip under dir")
    zips.sort(key=lambda f: int(re.search(r"_(\d+)_steps\.zip$", f.name).group(1) if re.search(r"_(\d+)_steps\.zip$", f.name) else -1))
    return zips[-1].as_posix()

# ╭─ main ───────────────────────────────────────────────────────────────╮

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gym", "grl"], required=True)
    ap.add_argument("--rl-path")
    ap.add_argument("--run-tag")
    ap.add_argument("--alg", choices=["ppo", "recurrent_ppo", "a2c"], default="ppo")
    ap.add_argument("--k8s", action="store_true")
    ap.add_argument("--use-case", default="redis")
    ap.add_argument("--goal", default="cost")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--testing", action="store_true")
    ap.add_argument("--loading", action="store_true")
    ap.add_argument("--load-path")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--total-steps", type=int, default=5000)
    ap.add_argument("--tensorboard-log")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])

    clean_argv = [a for a in sys.argv[1:] if a not in {"\\", "\n", "\\n"}]
    args = ap.parse_args(clean_argv)

    rl_cwd = Path(args.rl_path).expanduser().resolve() if args.rl_path else MODEL_ROOT[args.model]
    if not (rl_cwd / "policies/run/run.py").exists():
        raise FileNotFoundError(f"run.py not found in {rl_cwd}")

    # ── run-tag 與 I/O 目錄 ─────────────────────────────────────────────
    default_tag = f"{args.alg}_{args.model}_{args.total_steps}"
    run_tag     = args.run_tag or default_tag
    run_root    = LOG_ROOT / run_tag                           # ← 小寫一致
    
    # ★ 這行要完整寫成「parents=True, exist_ok=True」並關好括號
    run_root.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
            level   = logging.INFO,
            format  = "%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(run_root / "batch.log", encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
                ],
    )

