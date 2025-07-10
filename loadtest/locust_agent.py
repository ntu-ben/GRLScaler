#!/usr/bin/env python3
"""
locust_agent.py – REST API for triggering Locust runs
"""

import subprocess, shutil, uuid, os, datetime as dt, logging, re
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Locust Remote Agent")

# basic console logging so that the caller can inspect what happened
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

# 讀取環境變數
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LOCUST_BIN  = shutil.which("locust") or os.getenv("LOCUST_BIN", "/usr/local/bin/locust")
# Scenario scripts can be in "onlineboutique" or "redis" folders
ONLINEBOUTIQUE_DIR = Path(__file__).parent / "onlineboutique"
REDIS_DIR = Path(__file__).parent / "redis"
LOG_ROOT    = Path(os.getenv("LOCUST_AGENT_LOG_ROOT", "remote_logs"))      # 儲存在 m1，再由 m4 抓取

class JobReq(BaseModel):
    tag: str                   # e.g. iter01_offpeak
    scenario: str              # offpeak / rushsale / …
    target_host: str           # http://frontend.onlineboutique.svc.cluster.local
    run_time: str = "15m"
    stable_mode: bool = False  # 使用穩定loadtest模式
    max_rps: int = None        # 最高RPS限制
    timeout: int = 30          # 請求超時時間
    environment: str = "onlineboutique"  # "onlineboutique" or "redis"

jobs = {}   # job_id -> {"path":…, "ret":returncode}

@app.get("/")
def root():
    return {"status": "Locust Agent is running", "endpoints": ["/start", "/status/{job_id}", "/download/{tag}/{filename}"]}

@app.get("/health")
def health():
    return {"status": "healthy"}

def _parse_timespan(spec: str) -> int:
    """Return `spec` in seconds. Supports <num>[smhd] groups."""
    total = 0
    for amount, unit in re.findall(r"(\d+)([smhd])", spec):
        val = int(amount)
        if unit == "s":
            total += val
        elif unit == "m":
            total += val * 60
        elif unit == "h":
            total += val * 3600
        elif unit == "d":
            total += val * 86400
    if total == 0 and spec.isdigit():
        total = int(spec)
    return total


def _run_locust(job_id: str, req: JobReq):
    out_dir = LOG_ROOT / req.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 選擇環境目錄
    scenario_dir = REDIS_DIR if req.environment == "redis" else ONLINEBOUTIQUE_DIR
    
    # 選擇腳本（穩定模式或原版）
    if req.stable_mode:
        if req.environment == "redis":
            script_name = f"locust_redis_stable_{req.scenario}.py"
        else:
            script_name = f"locust_stable_{req.scenario}.py"
        
        stable_script = scenario_dir / script_name
        if not stable_script.exists():
            logging.warning(f"Stable script {script_name} not found, falling back to original")
            if req.environment == "redis":
                script_name = f"locust_redis_{req.scenario}.py"
            else:
                script_name = f"locust_{req.scenario}.py"
    else:
        if req.environment == "redis":
            script_name = f"locust_redis_{req.scenario}.py"
        else:
            script_name = f"locust_{req.scenario}.py"
    
    # 準備環境變數
    env = os.environ.copy()
    env['LOCUST_RUN_TIME'] = req.run_time
    if req.max_rps:
        env['LOCUST_TARGET_RPS'] = str(req.max_rps)  # 統一使用 LOCUST_TARGET_RPS
        env['LOCUST_MAX_RPS'] = str(req.max_rps)     # 保持舊版相容性
    if req.timeout:
        env['LOCUST_REQUEST_TIMEOUT'] = str(req.timeout)  # 統一命名
    
    cmd = [
        LOCUST_BIN, "-f", scenario_dir / script_name,
        "--headless", "--run-time", req.run_time,
        "--host", req.target_host,
        "--csv", out_dir / req.scenario, "--csv-full-history",
        "--html", out_dir / f"{req.scenario}.html"
    ]
    
    # 記錄配置信息
    logging.info(f"Starting {'stable' if req.stable_mode else 'standard'} loadtest: {req.scenario}")
    if req.max_rps:
        logging.info(f"Max RPS limit: {req.max_rps}")
    logging.debug("$ %s", " ".join(map(str, cmd)))
    logging.debug("Environment: LOCUST_TARGET_RPS=%s, LOCUST_REQUEST_TIMEOUT=%s", 
                  env.get('LOCUST_TARGET_RPS'), env.get('LOCUST_REQUEST_TIMEOUT'))
    
    proc = subprocess.Popen(cmd, env=env)
    timeout = _parse_timespan(req.run_time)
    logging.debug("locust timeout set to %ss", timeout)
    try:
        rc = proc.wait(timeout=timeout if timeout else None)
    except subprocess.TimeoutExpired:
        logging.info("timeout reached, terminating locust")
        proc.terminate()
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logging.warning("kill hung locust")
            proc.kill()
            rc = proc.wait()
    logging.debug("locust finished with rc=%s", rc)
    jobs[job_id]["ret"] = rc

@app.post("/start")
def start(req: JobReq, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    logging.info("/start tag=%s scenario=%s", req.tag, req.scenario)
    jobs[job_id] = {"req": req, "ret": None}
    bg.add_task(_run_locust, job_id, req)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id) or HTTPException(404)
    logging.debug("/status %s -> %s", job_id, job)
    return {"finished": job["ret"] is not None, "rc": job["ret"]}

@app.get("/download/{tag}/{filename:path}")
def download(tag: str, filename: str):
    file = LOG_ROOT / tag / filename
    if not file.exists():
        logging.warning("/download missing %s", file)
        raise HTTPException(404)
    logging.debug("/download %s", file)
    return FileResponse(file)

