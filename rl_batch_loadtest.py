#!/usr/bin/env python3
"""
rl_batch_loadtest.py  v4.2  (2025-06-06)
────────────────────────────────────────────────────────────────────────────
功能摘要
• --model {gym,gym-hpa,grl,gwydion,hpa} 決定 repo 路徑；grl 自動加 --gnn_mode
• 讀取 .env → M1_HOST=http://<m1-ip>:8099 ；遠端 locust 連不上才 fallback 本機
• 清洗 argv 中意外的 "\" / "\n" token
• 建立 logs/<run-tag>/batch.log，任何例外都同步寫 console + log
"""

from __future__ import annotations
import os, sys, logging, subprocess, time, datetime as dt, traceback, argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd, requests
from jinja2 import Template

# ──────────────────────────────────────────────────────────────────────────
# 0. 讀取 .env（若裝了 python-dotenv）
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # 假設 .env 就放在腳本同目錄；如放根目錄自行修改
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ModuleNotFoundError:
    pass  # optional

# ──────────────────────────────────────────────────────────────────────────
# 1. 全域常數（與舊版相同；節錄必要項）
# --------------------------------------------------------------------------
LOG_ROOT = Path("logs")
REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT: Dict[str, Path] = {
    # default to paths relative to this script so the repo can be cloned
    # anywhere without manual edits
    "gym": REPO_ROOT / "gnn_rl_env",
    "gym-hpa": REPO_ROOT / "gym-hpa",
    "grl": REPO_ROOT,
    # gwydion submodule contains its own package under "gwydion" folder
    "gwydion": REPO_ROOT / "gwydion" / "gwydion",
    # k8s_hpa houses the baseline tests (no RL training)
    "hpa": REPO_ROOT / "k8s_hpa",
}

NAMESPACE_OB  = os.getenv("NAMESPACE_ONLINEBOUTIQUE", "onlineboutique")
NAMESPACE_REDIS = os.getenv("NAMESPACE_REDIS", "redis")
NAMESPACE     = NAMESPACE_OB
TARGET_HOST   = os.getenv("TARGET_HOST", "http://k8s.orb.local")
HEALTH_PATH   = "/"
HTTP_TIMEOUT  = 600
RUN_TIME      = os.getenv("LOCUST_RUN_TIME", "15m")
SCENARIOS = {
    "offpeak":     "locust_offpeak.py",
    "rushsale":    "locust_rushsale.py",
    "peak":        "locust_peak.py",
    "fluctuating": "locust_fluctuating.py",
}

# ──────────────────────────────────────────────────────────────────────────
# 2. 小工具
# --------------------------------------------------------------------------
def panic(msg: str, exc: Exception | None = None) -> None:
    """同步把錯誤印到 console 與 batch.log，再結束進程"""
    logging.error(msg)
    if exc:
        _tb = "".join(traceback.format_exception(exc))
        logging.error(_tb)
        print(_tb, file=sys.stderr)
    sys.exit(1)


def sh(cmd: List[str]) -> None:
    """列印並執行 shell 指令；失敗 raise"""
    logging.info("$ %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def get_linkerd_rps(namespace: str = NAMESPACE) -> float | None:
    """Query Linkerd stats and return average RPS for all deployments."""
    api_addr = os.getenv("LINKERD_VIZ_API_URL", "localhost:8085")
    cmd = [
        "linkerd", "viz", "stat", "deploy", "-n", namespace,
        "--api-addr", api_addr,
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception as exc:
        logging.warning("linkerd stat failed: %s", exc)
        return None
    rps_vals = []
    for line in out.splitlines():
        parts = line.split()
        if not parts or parts[0] == "NAME" or parts[0].startswith("--"):
            continue
        if len(parts) >= 4:
            try:
                rps_vals.append(float(parts[3]))
            except ValueError:
                continue
    if not rps_vals:
        return None
    return sum(rps_vals) / len(rps_vals)


def wait_frontend_ready() -> None:
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


def run_locust(scenario: str, tag: str, remote: bool, out_dir: Path) -> None:
    """Start a Locust scenario either locally or via the remote agent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if remote:
        host = os.environ["M1_HOST"].rstrip("/")
        logging.info("Trigger remote locust %s on %s", scenario, host)
        payload = {
            "tag": tag,
            "scenario": scenario,
            "target_host": TARGET_HOST,
            "run_time": RUN_TIME,
        }
        logging.debug("POST %s/start %s", host, payload)
        try:
            r = requests.post(f"{host}/start", json=payload, timeout=10)
            r.raise_for_status()
            job_id = r.json()["job_id"]
            logging.debug("job id %s", job_id)
            while True:
                time.sleep(5)
                st = requests.get(f"{host}/status/{job_id}", timeout=10)
                st.raise_for_status()
                data = st.json()
                logging.debug("status %s -> %s", job_id, data)
                if data.get("finished"):
                    break
            for fname in [f"{scenario}_stats.csv", f"{scenario}_stats_history.csv", f"{scenario}.html"]:
                resp = requests.get(f"{host}/download/{tag}/{fname}", timeout=10)
                if resp.status_code == 200:
                    logging.debug("downloaded %s", fname)
                    (out_dir / fname).write_bytes(resp.content)
        except requests.RequestException as exc:
            logging.error("remote locust failed: %s", exc)
            raise
    else:
        script = Path(__file__).parent / "loadtest" / "onlineboutique" / f"locust_{scenario}.py"
        logging.info("Run local locust %s", scenario)
        cmd = [
            "locust", "-f", script, "--headless", "--run-time", RUN_TIME,
            "--host", TARGET_HOST,
            "--csv", out_dir / scenario, "--csv-full-history",
            "--html", out_dir / f"{scenario}.html",
        ]
        sh(cmd)


def summarise(run_tag: str, scenario_dirs: list[Path], namespace: str) -> pd.DataFrame:
    rows = []
    for d in scenario_dirs:
        try:
            stat_csv = next(d.glob("*_stats.csv"))
        except StopIteration:
            logging.warning("No stats CSV for %s", d)
            continue
        df = pd.read_csv(stat_csv)
        total = df[df["Name"] == "Total"]
        if total.empty:
            logging.warning("No 'Total' row in %s", stat_csv)
            continue
        tot = total.iloc[0]
        rps = get_linkerd_rps(namespace)
        rows.append({
            "Run": run_tag,
            "Scenario": d.name,
            "Requests": tot.get("Request Count", 0),
            "Failures": tot.get("Failure Count", 0),
            "Avg RPS": tot.get("Requests/s", 0),
            "Linkerd RPS": rps if rps is not None else "",
            "P95 ms": tot.get("95%", 0),
        })
    return pd.DataFrame(rows)


def render_dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    html = "<html><body>" + df.to_html(index=False) + "</body></html>"
    (out_dir / "aggregate.html").write_text(html, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────
# 3. 主程式
# --------------------------------------------------------------------------
def main() -> None:
    try:
        # 3-1 argparse（先把髒 token 清掉）
        ap = argparse.ArgumentParser()
        ap.add_argument("--model", choices=["gym", "gym-hpa", "grl", "gwydion", "hpa"], required=True)
        ap.add_argument("--rl-path")          # 可手動覆蓋 repo
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
        args = ap.parse_args(
            [a for a in sys.argv[1:] if a not in {"\\", "\\n", "\n"}]
        )

        # 3-2 決定 RL repo 路徑
        rl_cwd = Path(args.rl_path) if args.rl_path else MODEL_ROOT[args.model]
        if args.model == "gwydion":
            run_file = rl_cwd / "run.py"
        elif args.model == "gym-hpa":
            run_file = rl_cwd / "policies" / "run" / "run.py"
        elif args.model == "hpa":
            run_file = rl_cwd / "HPABaseLineTest.py"
        else:
            run_file = rl_cwd / "gnn_rl/run/run.py"
        if not run_file.exists():
            panic(f"{run_file} 不存在")

        # 3-3 run-tag & log 目錄
        default_tag = f"{args.alg}_{args.model}_{args.total_steps}"
        run_tag     = args.run_tag or default_tag
        run_root    = LOG_ROOT / args.model / run_tag
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "batch.log").touch(exist_ok=True)   # 確保檔案存在

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(run_root / "batch.log", encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.debug("🚀 rl_batch_loadtest starting…")

        # 3-4 組合 RL 子行程命令
        if args.model == "gwydion":
            rl_cmd = ["python", "run.py"]
        elif args.model == "hpa":
            rl_cmd = ["python", "HPABaseLineTest.py"]
        else:
            rl_cmd = ["python", "-m", "gnn_rl.run.run"]
        if args.model != "hpa":
            rl_cmd += [
                "--alg", args.alg,
                "--use_case", args.use_case,
                "--goal", args.goal,
                "--steps", str(args.steps),
                "--total_steps", str(args.total_steps),
            ]
        if args.model != "hpa":
            if args.k8s:      rl_cmd += ["--k8s"]
            if args.training: rl_cmd += ["--training"]
            if args.testing:
                if not args.load_path:
                    panic("--testing 需搭配 --load-path")
                rl_cmd += ["--testing", "--test_path", args.load_path]
            if args.loading:
                if not args.load_path:
                    panic("--loading 需搭配 --load-path")
                rl_cmd += ["--loading", "--load_path", args.load_path]
            if args.device: rl_cmd += ["--device", args.device]
            if args.tensorboard_log:
                rl_cmd += ["--tensorboard_log", args.tensorboard_log]
            if args.model == "grl":
                rl_cmd += ["--gnn_mode"]

        logging.debug("▶ Command: %s", " ".join(rl_cmd))
        if args.model == "hpa":
            subprocess.run(rl_cmd, cwd=rl_cwd, check=True)
            logging.info("✅ 完成 → logs/hpa")
            return
        rl = subprocess.Popen(rl_cmd, cwd=rl_cwd)

        # 3-5 決定壓測模式
        from_locust_remote = bool(os.getenv("M1_HOST"))
        logging.debug("🛠  Locust mode = %s",
                      "remote via "+os.getenv("M1_HOST") if from_locust_remote else "local")

        scenario_dirs = []
        for scn in SCENARIOS:
            out_dir = run_root / scn
            remote_tag = f"{args.model}/{run_tag}"
            run_locust(scn, remote_tag if from_locust_remote else run_tag, from_locust_remote, out_dir)
            scenario_dirs.append(out_dir)

        rl.wait()
        ns = NAMESPACE_REDIS if args.use_case == "redis" else NAMESPACE_OB
        df = summarise(run_tag, scenario_dirs, ns)
        df.to_csv(run_root / "summary.csv", index=False)
        render_dashboard(df, run_root)

        logging.info("✅ 完成 → %s", run_root)

    except Exception as e:
        panic("‼️  rl_batch_loadtest 未預期錯誤", e)


if __name__ == "__main__":
    main()

