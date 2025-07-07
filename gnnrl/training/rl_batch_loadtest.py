#!/usr/bin/env python3
"""
rl_batch_loadtest.py  v5.0  (2025-06-23)
────────────────────────────────────────────────────────────────────────────
統一實驗管理器 - 支持分散式 Locust 測試環境
────────────────────────────────────────────────────────────────────────────
功能摘要：
• 支持三種實驗模式：gym_hpa, k8s_hpa (baseline), gnnrl
• 整合分散式 Locust 測試環境 (M1_HOST 遠端代理)
• 自動協調實驗訓練與負載測試的時序
• 統一日誌管理和結果匯總
• 支持多種負載測試情境：offpeak, rushsale, peak, fluctuating

實驗架構：
• gym_hpa: 基礎強化學習 + MLP 策略
• k8s_hpa: Kubernetes HPA 基準測試
• gnnrl: 圖神經網路強化學習

分散式測試：
• 遠端 Locust 代理 (M1_HOST) 用於分散負載
• 本地 fallback 機制
• 同步訓練過程與負載測試
"""

from __future__ import annotations
import os, sys, logging, subprocess, time, datetime as dt, traceback, argparse, random
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd, requests
from jinja2 import Template

# ──────────────────────────────────────────────────────────────────────────
# 0. 讀取 .env（若裝了 python-dotenv）
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
try:
    from dotenv import load_dotenv
    # `.env` 固定放在 repo 根目錄
    load_dotenv(REPO_ROOT / ".env")
except ModuleNotFoundError:
    pass  # optional

# ──────────────────────────────────────────────────────────────────────────
# 1. 全域常數（與舊版相同；節錄必要項）
# --------------------------------------------------------------------------
LOG_ROOT = Path(os.getenv("LOG_ROOT", "logs"))
MODEL_ROOT: Dict[str, Path] = {
    # default to paths relative to this script so the repo can be cloned
    # anywhere without manual edits
    "gym": REPO_ROOT / "gnn_rl/envs",  # Legacy path (if exists)
    "gym-hpa": REPO_ROOT / "gym-hpa",  # Standard RL with MLP policies
    "grl": REPO_ROOT / "gnnrl",  # Legacy GNN-RL (if exists)
    "gnnrl": REPO_ROOT / "gnnrl",  # New unified GNNRL path
    # gwydion submodule contains its own package under "gwydion" folder
    "gwydion": REPO_ROOT / "gwydion" / "gwydion",
    # k8s_hpa houses the baseline tests (no RL training)
    "hpa": REPO_ROOT / "k8s_hpa",
}

NAMESPACE_OB  = os.getenv("NAMESPACE_ONLINEBOUTIQUE", "onlineboutique")
NAMESPACE_REDIS = os.getenv("NAMESPACE_REDIS", "redis")
NAMESPACE     = NAMESPACE_OB
TARGET_HOST   = os.getenv("TARGET_HOST", "http://k8s.orb.local")
HEALTH_PATH   = os.getenv("HEALTH_PATH", "/")
HTTP_TIMEOUT  = int(os.getenv("HTTP_TIMEOUT", "600"))
RUN_TIME      = os.getenv("LOCUST_RUN_TIME", "15m")
SCENARIOS = {
    "offpeak":     "locust_offpeak.py",
    "rushsale":    "locust_rushsale.py",
    "peak":        "locust_peak.py",
    "fluctuating": "locust_fluctuating.py",
}
_MULT = {"s": 1, "m": 60, "h": 3600}
_match = __import__("re").match
_rt = _match(r"(\d+)([smh])", RUN_TIME)
RUN_TIME_SEC = int(_rt.group(1)) * _MULT[_rt.group(2)] if _rt else 900
HALF_RUN_SEC = RUN_TIME_SEC // 2
MAX_STATUS_CHECKS = int(os.getenv("MAX_STATUS_CHECKS", "720"))  # stop polling after 1h (720 * 5s)

# ──────────────────────────────────────────────────────────────────────────
# 2. 小工具
# --------------------------------------------------------------------------
def panic(msg: str, exc: Exception | None = None) -> None:
    """同步把錯誤印到 console 與 batch.log，再結束進程"""
    logging.error(msg)
    if exc:
        _tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logging.error(_tb)
        print(_tb, file=sys.stderr)
    sys.exit(1)


def sh(cmd: List[str]) -> None:
    """列印並執行 shell 指令；失敗 raise"""
    logging.info("$ %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def record_kiali_graph(stage: str) -> None:
    """Dump Kiali service graph for the namespace."""
    logging.info("kiali graph (%s)", stage)
    kiali_base = os.getenv('KIALI_URL', 'http://localhost:20001/kiali')
    # 使用正确的 Kiali v1.7x+ 多命名空间 API 格式
    url = f"{kiali_base}/api/namespaces/graph?namespaces={NAMESPACE}&duration=600s&graphType=workload"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        # 確保 kiali 目錄存在
        kiali_dir = Path("logs/kiali")
        kiali_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kiali_file = kiali_dir / f"kiali_{stage}_{timestamp}.json"
        kiali_file.write_text(resp.text, encoding="utf-8")
        logging.info("✅ Kiali graph saved: %s", kiali_file)
    except Exception as err:
        logging.warning("kiali graph failed: %s", err)


def get_kiali_rps(namespace: str = NAMESPACE) -> float | None:
    """Query Kiali metrics and return average RPS for all workloads."""
    kiali_base = os.getenv('KIALI_URL', 'http://localhost:20001/kiali')
    url = f"{kiali_base}/api/namespaces/{namespace}/metrics?metrics=request_count"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        logging.warning("kiali metrics failed: %s", exc)
        return None
    series = resp.json().get("metrics", {}).get("request_count", [])
    total = 0.0
    count = 0
    for item in series:
        for _, val in item.get("values", []):
            try:
                total += float(val)
                count += 1
            except (TypeError, ValueError):
                continue
    if count == 0:
        return None
    return total / count


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


def run_distributed_locust(scenario: str, tag: str, remote: bool, out_dir: Path, experiment_sync: dict = None) -> None:
    """運行分散式 Locust 測試，支持與實驗訓練同步。
    
    Args:
        scenario: 測試情境名稱
        tag: 運行標籤
        remote: 是否使用遠端代理
        out_dir: 輸出目錄
        experiment_sync: 實驗同步信息 {"training_proc": subprocess, "sync_points": [...]}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 檢查實驗訓練進程狀態
    training_proc = experiment_sync.get("training_proc") if experiment_sync else None
    if training_proc and training_proc.poll() is not None:
        logging.warning("Training process terminated before loadtest %s", scenario)
    
    if remote:
        host = os.environ["M1_HOST"].rstrip("/")
        logging.info("🔗 分散式測試: M1_HOST=%s", host)
        logging.info("🚀 觸發遠端 Locust %s 在 %s", scenario, host)
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
            logging.info("📋 遠端任務 ID: %s", job_id)
            
            # 記錄開始狀態
            record_kiali_graph("start")
            
            # 中途檢查點
            time.sleep(HALF_RUN_SEC)
            record_kiali_graph("mid")
            
            # 等待完成並監控訓練進程
            for check_count in range(MAX_STATUS_CHECKS):
                time.sleep(5)
                
                # 檢查遠端測試狀態
                st = requests.get(f"{host}/status/{job_id}", timeout=10)
                st.raise_for_status()
                data = st.json()
                
                if data.get("finished"):
                    logging.info("✅ 遠端測試 %s 完成", scenario)
                    break
                    
                # 每 10 次檢查一次訓練進程
                if check_count % 10 == 0 and training_proc:
                    if training_proc.poll() is not None:
                        logging.warning("⚠️  訓練進程在測試期間終止")
                        
                logging.debug("⏳ 遠端測試狀態 [%d/%d]: %s", check_count+1, MAX_STATUS_CHECKS, 
                            "running" if not data.get("finished") else "finished")
            else:
                logging.warning("⏰ 遠端測試超時，可能仍在運行")
                return
                
            record_kiali_graph("end")
            
            # 下載結果檔案
            downloaded_files = []
            for fname in [f"{scenario}_stats.csv", f"{scenario}_stats_history.csv", f"{scenario}.html"]:
                resp = requests.get(f"{host}/download/{tag}/{fname}", timeout=10)
                if resp.status_code == 200:
                    (out_dir / fname).write_bytes(resp.content)
                    downloaded_files.append(fname)
                    logging.debug("📁 已下載: %s", fname)
                else:
                    logging.warning("❌ 下載失敗: %s (status: %d)", fname, resp.status_code)
            
            logging.info("📊 遠端測試結果: 已下載 %d/%d 檔案", len(downloaded_files), 3)
            return
            
        except requests.RequestException as exc:
            logging.error("❌ 遠端測試失敗: %s", exc)
            logging.info("🔄 切換到本地測試")

    # 本地測試 fallback
    script_path = REPO_ROOT / "loadtest" / "onlineboutique" / f"locust_{scenario}.py"
    if not script_path.exists():
        logging.error("❌ 測試腳本不存在: %s", script_path)
        return
        
    logging.info("🏠 運行本地 Locust %s", scenario)
    cmd = [
        "locust", "-f", str(script_path), "--headless", "--run-time", RUN_TIME,
        "--host", TARGET_HOST,
        "--csv", str(out_dir / scenario), "--csv-full-history",
        "--html", str(out_dir / f"{scenario}.html"),
    ]
    
    logging.debug("$ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    
    record_kiali_graph("start")
    time.sleep(HALF_RUN_SEC)
    record_kiali_graph("mid")
    
    # 等待本地測試完成，同時監控訓練進程
    while proc.poll() is None:
        time.sleep(5)
        if training_proc and training_proc.poll() is not None:
            logging.info("ℹ️  訓練進程已完成，繼續等待測試")
            training_proc = None  # 避免重複記錄
    
    record_kiali_graph("end")
    
    if proc.returncode:
        logging.warning("⚠️  本地測試 %s 結束碼: %s", scenario, proc.returncode)
    else:
        logging.info("✅ 本地測試 %s 完成", scenario)


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
        rps = get_kiali_rps(namespace)
        rows.append({
            "Run": run_tag,
            "Scenario": d.name,
            "Requests": tot.get("Request Count", 0),
            "Failures": tot.get("Failure Count", 0),
            "Avg RPS": tot.get("Requests/s", 0),
            "Kiali RPS": rps if rps is not None else "",
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
        ap.add_argument("--model", choices=["gym", "gym-hpa", "grl", "gnnrl", "gwydion", "hpa"], required=True)
        ap.add_argument("--rl-path")          # 可手動覆蓋 repo
        ap.add_argument("--run-tag")
        ap.add_argument("--alg", choices=["ppo", "recurrent_ppo", "a2c"], default="ppo")
        ap.add_argument("--gnn-model", choices=["gat", "gcn"], default="gat", help="GNN model type for gnnrl experiments")
        ap.add_argument("--k8s", action="store_true")
        ap.add_argument("--use-case", default="redis")
        ap.add_argument("--goal", default="cost")
        ap.add_argument("--training", action="store_true")
        ap.add_argument("--testing", action="store_true")
        ap.add_argument("--loading", action="store_true")
        ap.add_argument("--load-path")
        ap.add_argument("--steps", type=int, default=500)
        ap.add_argument("--total-steps", type=int, default=5000)
        ap.add_argument("--seed", type=int, default=42, help="Random seed for scenario order")
        ap.add_argument("--env-step-interval", type=float, default=15.0, help="Environment step interval in seconds")
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
        elif args.model == "gnnrl":
            run_file = rl_cwd / "training" / "run_gnnrl_experiment.py"
        elif args.model == "hpa":
            run_file = rl_cwd / "HPABaseLineTest.py"
        else:
            run_file = rl_cwd / "gnn_rl/run/run.py"
        if not run_file.exists():
            panic(f"{run_file} 不存在")

        # 3-3 run-tag & log 目錄 - 使用新的路徑管理器
        default_tag = f"{args.alg}_{args.model}_{args.total_steps}"
        run_tag = args.run_tag or default_tag
        
        # 嘗試使用新的統一路徑結構
        try:
            sys.path.append(str(REPO_ROOT))
            from experiment_path_manager import get_path_manager
            
            path_manager = get_path_manager()
            
            # 如果 run_tag 是新格式，直接使用；否則創建新的實驗目錄
            if '_' in run_tag and len(run_tag.split('_')) >= 6:
                # 新格式的 run_tag，直接使用
                run_root = path_manager.base_dir / run_tag
                run_root.mkdir(exist_ok=True)
                (run_root / "loadtest_scenarios").mkdir(exist_ok=True)
            else:
                # 舊格式，創建新的實驗目錄但保持向後兼容
                run_root = LOG_ROOT / args.model / run_tag
                run_root.mkdir(parents=True, exist_ok=True)
                
        except ImportError:
            # 如果路徑管理器不可用，使用舊方式
            run_root = LOG_ROOT / args.model / run_tag
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
        elif args.model == "gnnrl":
            # Use new GNNRL experiment runner
            rl_cmd = ["python", "training/run_gnnrl_experiment.py"]
        elif args.model == "gym-hpa":
            # Use gym-hpa run script with module import
            rl_cmd = ["python", "policies/run/run.py"]
        else:
            # Legacy gnn_rl path
            rl_cmd = ["python", "-m", "gnn_rl.run.run"]
            
        if args.model == "gnnrl":
            # GNNRL uses different parameter names
            if args.k8s: rl_cmd += ["--k8s"]
            rl_cmd += ["--steps", str(args.total_steps)]  # gnnrl uses --steps for total steps
            if args.goal: rl_cmd += ["--goal", args.goal]
            if args.gnn_model: rl_cmd += ["--model", args.gnn_model]  # Pass GNN model type
            if args.alg: rl_cmd += ["--alg", args.alg]  # Pass RL algorithm
            if args.env_step_interval: rl_cmd += ["--env-step-interval", str(args.env_step_interval)]  # Pass step interval
            if args.tensorboard_log: rl_cmd += ["--tensorboard-log", args.tensorboard_log]
        elif args.model != "hpa":
            rl_cmd += [
                "--alg", args.alg,
                "--use_case", args.use_case,
                "--goal", args.goal,
                "--steps", str(args.steps),
                "--total_steps", str(args.total_steps),
            ]
            if args.k8s:      rl_cmd += ["--k8s"]
            if args.training: rl_cmd += ["--training"]
            if args.testing:
                if not args.load_path:
                    panic("--testing 需搭配 --load-path")
                # Convert path relative to gym-hpa directory
                from pathlib import Path
                load_path = Path(args.load_path)
                if not load_path.is_absolute():
                    # Make path relative to gym-hpa directory
                    relative_load_path = "../" + str(load_path)
                else:
                    relative_load_path = str(load_path)
                rl_cmd += ["--testing", "--test_path", relative_load_path]
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
        elif args.model == "gnnrl":
            # GNNRL experiment handles its own training loop
            rl = subprocess.Popen(rl_cmd, cwd=rl_cwd)
            logging.info("🔄 GNNRL experiment started, continuing with loadtest...")
        else:
            rl = subprocess.Popen(rl_cmd, cwd=rl_cwd)

        # 3-5 統一實驗與分散式測試協調
        from_locust_remote = bool(os.getenv("M1_HOST"))
        logging.info("🔧 測試模式: %s", 
                    f"分散式 (代理: {os.getenv('M1_HOST')})" if from_locust_remote else "本地")
        
        # 為不同實驗類型設定同步策略
        experiment_sync = {"training_proc": rl} if 'rl' in locals() else None
        
        # 使用 seed 設定隨機種子
        random.seed(args.seed)
        scenario_list = list(SCENARIOS.keys())
        
        logging.info("🎲 使用隨機種子 %d，可用情境: %s", args.seed, ", ".join(scenario_list))
        
        scenario_dirs = []
        scenario_count = 0
        
        # 檢查是否有 RL 訓練進程需要等待
        has_training_proc = 'rl' in locals() and rl is not None
        
        # 持續隨機執行場景直到訓練完成 (如果有訓練進程) 或至少執行一個場景
        while True:
            # 檢查訓練是否完成
            if has_training_proc and rl.poll() is not None:
                logging.info("✅ RL 訓練進程已完成")
                break
            
            # 完全隨機選擇場景
            scn = random.choice(scenario_list)
            scenario_count += 1
            
            # 創建唯一的輸出目錄 (使用計數器避免重複)
            out_dir = run_root / f"{scn}_{scenario_count:03d}"
            logging.info("📊 執行隨機測試情境 [第%d個]: %s", scenario_count, scn)
            
            remote_tag = f"{args.model}/{run_tag}"
            
            # 分散式測試，包含實驗同步
            run_distributed_locust(
                scn, 
                remote_tag if from_locust_remote else run_tag, 
                from_locust_remote, 
                out_dir,
                experiment_sync
            )
            scenario_dirs.append(out_dir)
            
            # 情境間冷卻時間
            if has_training_proc and rl.poll() is None:
                cooldown = int(os.getenv("COOLDOWN_BETWEEN_SCENARIOS", "60"))  # 預設1分鐘
                logging.info("⏸️  情境間冷卻 %d 秒...", cooldown)
                time.sleep(cooldown)
            elif not has_training_proc:
                # 如果沒有訓練進程，執行一個場景後結束
                break

        # 最終等待訓練完成 (雙重保險)
        if has_training_proc and rl.poll() is None:
            logging.info("⏳ 最終等待訓練進程完成...")
            rl.wait()
        
        logging.info("🏁 總共執行了 %d 個隨機場景測試", len(scenario_dirs))
            
        # 生成統一報告
        ns = NAMESPACE_REDIS if args.use_case == "redis" else NAMESPACE_OB
        
        # 舊版報告（保持向後兼容）
        df = summarise(run_tag, scenario_dirs, ns)
        df.to_csv(run_root / "summary.csv", index=False)
        render_dashboard(df, run_root)
        
        # 新版統一報告
        try:
            from unified_report_generator import process_experiment_results
            
            # 準備實驗配置
            experiment_config = {
                "id": run_tag,
                "type": args.model,
                "algorithm": args.alg,
                "model": getattr(args, 'gnn_model', 'default'),
                "goal": args.goal,
                "steps": args.total_steps,
                "seed": args.seed,
                "start_time": dt.datetime.now().isoformat()
            }
            
            # 生成統一報告
            process_experiment_results(run_root, scenario_dirs, experiment_config)
            logging.info("✅ 統一實驗報告已生成")
            
        except Exception as e:
            logging.warning(f"⚠️ 統一報告生成失敗: {e}")

        logging.info("🎉 實驗完成 → %s", run_root)
        logging.info("📈 結果摘要: %s", run_root / "summary.csv")
        logging.info("🌐 儀表板: %s", run_root / "aggregate.html")
        logging.info("🔄 統一報告: experiments/ 目錄下")

    except Exception as e:
        panic("‼️  rl_batch_loadtest 未預期錯誤", e)


if __name__ == "__main__":
    main()

