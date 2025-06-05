#!/usr/bin/env python3
"""
locust_agent.py – REST API for triggering Locust runs
"""

import subprocess, shutil, uuid, os, datetime as dt
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Locust Remote Agent")

LOCUST_BIN  = shutil.which("locust") or "/usr/local/bin/locust"
SCENARIO_DIR = Path(__file__).parent / "stressTest" / "onlineboutique"
LOG_ROOT    = Path("remote_logs")      # 儲存在 m1，再由 m4 抓取

class JobReq(BaseModel):
    tag: str                   # e.g. iter01_offpeak
    scenario: str              # offpeak / rushsale / …
    target_host: str           # http://frontend.onlineboutique.svc.cluster.local
    run_time: str = "15m"

jobs = {}   # job_id -> {"path":…, "ret":returncode}

def _run_locust(job_id: str, req: JobReq):
    out_dir = LOG_ROOT / req.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        LOCUST_BIN, "-f", SCENARIO_DIR / f"locust_{req.scenario}.py",
        "--headless", "--run-time", req.run_time,
        "--host", req.target_host,
        "--csv", out_dir / req.scenario, "--csv-full-history",
        "--html", out_dir / f"{req.scenario}.html"
    ]
    rc = subprocess.call(cmd)
    jobs[job_id]["ret"] = rc

@app.post("/start")
def start(req: JobReq, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"req": req, "ret": None}
    bg.add_task(_run_locust, job_id, req)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id) or HTTPException(404)
    return {"finished": job["ret"] is not None, "rc": job["ret"]}

@app.get("/download/{tag}/{filename:path}")
def download(tag: str, filename: str):
    file = LOG_ROOT / tag / filename
    if not file.exists():
        raise HTTPException(404)
    return fastapi.responses.FileResponse(file)

