#!/usr/bin/env python3
"""
locust_agent.py – REST API for triggering Locust runs
"""

import subprocess, shutil, uuid, os, datetime as dt, logging
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

LOCUST_BIN  = shutil.which("locust") or "/usr/local/bin/locust"
# Scenario scripts live directly under the "onlineboutique" folder. The previous
# path used an extra "stressTest" level that does not exist and caused file
# lookup failures.
SCENARIO_DIR = Path(__file__).parent / "onlineboutique"
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
    logging.debug("$ %s", " ".join(map(str, cmd)))
    rc = subprocess.call(cmd)
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

