#!/usr/bin/env python3
"""Simple helper to run Locust scenarios manually for distributed load testing."""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

LOCUST_BIN = shutil.which("locust") or "/usr/local/bin/locust"
SCENARIO_DIR = Path(__file__).parent / "onlineboutique"
LOG_ROOT = Path("remote_logs")


def run_locust(tag: str, scenario: str, host: str, run_time: str, 
               stable_mode: bool = False, max_rps: int = None, timeout: int = 30) -> int:
    out_dir = LOG_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 選擇腳本（穩定模式或原版）
    if stable_mode:
        script_name = f"locust_stable_{scenario}.py"
        stable_script = SCENARIO_DIR / script_name
        if not stable_script.exists():
            logging.warning(f"Stable script {script_name} not found, falling back to original")
            script_name = f"locust_{scenario}.py"
    else:
        script_name = f"locust_{scenario}.py"
    
    # 準備環境變數
    import os
    env = os.environ.copy()
    env['LOCUST_RUN_TIME'] = run_time
    if max_rps:
        env['LOCUST_MAX_RPS'] = str(max_rps)
    if timeout:
        env['LOCUST_TIMEOUT'] = str(timeout)
    
    cmd = [
        LOCUST_BIN, "-f", SCENARIO_DIR / script_name,
        "--headless", "--run-time", run_time,
        "--host", host,
        "--csv", out_dir / scenario, "--csv-full-history",
        "--html", out_dir / f"{scenario}.html",
    ]
    
    # 記錄配置信息
    logging.info(f"Starting {'stable' if stable_mode else 'standard'} loadtest: {scenario}")
    if max_rps:
        logging.info(f"Max RPS limit: {max_rps}")
    logging.debug("$ %s", " ".join(map(str, cmd)))
    logging.debug("Environment: LOCUST_MAX_RPS=%s, LOCUST_TIMEOUT=%s", 
                  env.get('LOCUST_MAX_RPS'), env.get('LOCUST_TIMEOUT'))
    
    return subprocess.call(cmd, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--target-host", required=True)
    ap.add_argument("--run-time", default="24h")
    ap.add_argument("--stable-mode", action="store_true", help="使用穩定loadtest模式")
    ap.add_argument("--max-rps", type=int, help="最高RPS限制")
    ap.add_argument("--timeout", type=int, default=30, help="請求超時時間")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s")
    rc = run_locust(args.tag, args.scenario, args.target_host, args.run_time,
                    args.stable_mode, args.max_rps, args.timeout)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
