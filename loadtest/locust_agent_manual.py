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


def run_locust(tag: str, scenario: str, host: str, run_time: str) -> int:
    out_dir = LOG_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        LOCUST_BIN, "-f", SCENARIO_DIR / f"locust_{scenario}.py",
        "--headless", "--run-time", run_time,
        "--host", host,
        "--csv", out_dir / scenario, "--csv-full-history",
        "--html", out_dir / f"{scenario}.html",
    ]
    logging.debug("$ %s", " ".join(map(str, cmd)))
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--target-host", required=True)
    ap.add_argument("--run-time", default="24h")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s")
    rc = run_locust(args.tag, args.scenario, args.target_host, args.run_time)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
