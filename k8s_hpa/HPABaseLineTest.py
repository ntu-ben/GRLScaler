#!/usr/bin/env python3
"""
Batch load‑test runner for Online Boutique (one‑shot, no retry on Locust failures)
==========================================================================
* 逐一套用 `HPA/onlineboutique/*` 子資料夾的 HPA
* 每組 HPA 預熱 & 健康檢查後，依序跑 4 種 Locust 情境：
      off‑peak → rush‑sale → peak → fluctuating
  ‑ 每段前先 **sleep 5 分鐘**，再確認 frontend 回 200。
  ‑ **不再因為失敗率重跑**；Locust 無論 exit‑code，都只跑一次並記錄結果。
* 失敗或產生不到 CSV 時，會寫 warning，但流程持續到下一 scenario/HPA。
* 結束後產生 `logs/<hpa>/summary.csv` 與 `logs/aggregate.html`。
"""
import subprocess, datetime as dt, time, logging, sys, os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # `.env` 固定放在 repo 根目錄
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
except ModuleNotFoundError:
    pass

import pandas as pd
import requests
from jinja2 import Template

# ── configuration ─────────────────────────────────────────────────────────
NAMESPACE_OB  = os.getenv("NAMESPACE_ONLINEBOUTIQUE", "onlineboutique")
NAMESPACE_REDIS = os.getenv("NAMESPACE_REDIS", "redis")
NAMESPACE     = NAMESPACE_OB
# 根目錄 (此檔案位於 repo/k8s_hpa/)
REPO_ROOT     = Path(__file__).resolve().parents[1]
# Online Boutique 原始碼路徑，可依需要調整
MICRO_DEMO    = REPO_ROOT / "MicroServiceBenchmark" / "microservices-demo"
MANIFEST_YAML = MICRO_DEMO / "release" / "kubernetes-manifests-withoutLoadgenerator.yaml"
# HPA YAML 目錄移至 repo/macK8S/HPA/onlineboutique
HPA_ROOT      = REPO_ROOT / "macK8S" / "HPA" / "onlineboutique"
# Scenario scripts are shared with rl_batch_loadtest under loadtest/onlineboutique
LOCUST_ROOT   = REPO_ROOT / "loadtest" / "onlineboutique"
TARGET_HOST   = os.getenv("TARGET_HOST", "http://k8s.orb.local")
HEALTH_PATH   = os.getenv("HEALTH_PATH", "/")
HTTP_TIMEOUT  = int(os.getenv("HTTP_TIMEOUT", "1200"))   # wait up to 20 min for 200 OK
COOLDOWN_SEC  = int(os.getenv("COOLDOWN_BETWEEN_SCENARIOS", "300"))   # 5 min between scenarios

LOCUST_SCRIPTS = {
    "offpeak":      "locust_offpeak.py",
    "rushsale":     "locust_rushsale.py",
    "peak":         "locust_peak.py",
    "fluctuating":  "locust_fluctuating.py",
}
RUN_TIME = os.getenv("LOCUST_RUN_TIME", "15m")
_MULT = {"s": 1, "m": 60, "h": 3600}
_match = __import__("re").match
_rt = _match(r"(\d+)([smh])", RUN_TIME)
RUN_TIME_SEC = int(_rt.group(1)) * _MULT[_rt.group(2)] if _rt else 900
HALF_RUN_SEC = RUN_TIME_SEC // 2
LOG_ROOT = Path("../logs") / "hpa"
MAX_STATUS_CHECKS = 720  # stop polling after 1h (720 * 5s)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler("batch_run.log"),
                              logging.StreamHandler(sys.stdout)])

# ── helpers ──────────────────────────────────────────────────────────────

def sh(cmd, **kw):
    if not isinstance(cmd, str):
        printable = " ".join(map(str, cmd)); cmd = list(map(str, cmd))
    else:
        printable = cmd
    logging.info("$ %s", printable)
    subprocess.run(cmd, check=True, **kw)


def record_linkerd_stat(stage: str) -> None:
    """Record Linkerd stats for deployments."""
    logging.info("linkerd stat (%s)", stage)
    try:
        sh([
            "linkerd", "viz", "stat", "deploy", "-n", NAMESPACE,
            "--api-addr", "localhost:8085",
        ])
    except subprocess.CalledProcessError as err:
        logging.warning("linkerd stat failed: %s", err)


def reset_demo():
    sh(["kubectl", "delete", "-f", MANIFEST_YAML, "-n", NAMESPACE], cwd=MICRO_DEMO)
    sh(["kubectl", "apply",  "-f", MANIFEST_YAML, "-n", NAMESPACE], cwd=MICRO_DEMO)
    # re-inject Linkerd sidecar for monitoring
    cmd = "kubectl get deploy -n {} -o yaml | linkerd inject - | kubectl apply -f -".format(NAMESPACE)
    for i in range(5):
        try:
            sh(cmd, shell=True)
            break
        except subprocess.CalledProcessError as err:
            if i == 4:
                raise
            logging.warning("Linkerd inject failed (%d/5): %s", i + 1, err)
            time.sleep(3)


def apply_hpa(folder: Path):
    logging.info("Apply HPA %s", folder.name)
    for y in sorted(folder.glob("*.yaml")):
        sh(["kubectl", "apply", "-f", y, "-n", NAMESPACE])


def delete_hpa(folder: Path):
    logging.info("Delete HPA %s", folder.name)
    for y in sorted(folder.glob("*.yaml")):
        sh(["kubectl", "delete", "-f", y, "-n", NAMESPACE])


def wait_frontend_ready():
    sh(["kubectl", "rollout", "status", "deployment/frontend", "-n", NAMESPACE])
    url = TARGET_HOST.rstrip("/") + HEALTH_PATH
    deadline = time.time() + HTTP_TIMEOUT
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                logging.info("Health‑check 200 OK")
                return
            logging.warning("Status %s, retry…", r.status_code)
        except requests.RequestException as e:
            logging.warning("%s, retry…", e)
        time.sleep(5)
    raise RuntimeError("frontend did not reach HTTP 200 in time")


def run_locust_once(scenario: str, script: Path, out_dir: Path):
    """Run Locust exactly once, optionally via the remote agent."""
    out_dir.mkdir(parents=True, exist_ok=True)
    host = os.getenv("M1_HOST")
    if host:
        host = host.rstrip("/")
        logging.info("M1_HOST=%s", host)
        tag = f"hpa_{scenario}_{int(time.time())}"
        payload = {
            "tag": tag,
            "scenario": scenario,
            "target_host": TARGET_HOST,
            "run_time": RUN_TIME,
        }
        logging.info("Trigger remote locust %s on %s", scenario, host)
        logging.debug("POST %s/start %s", host, payload)
        record_linkerd_stat("start")
        try:
            resp = requests.post(f"{host}/start", json=payload, timeout=10)
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            time.sleep(HALF_RUN_SEC)
            record_linkerd_stat("mid")
            for _ in range(MAX_STATUS_CHECKS):
                time.sleep(5)
                st = requests.get(f"{host}/status/{job_id}", timeout=10)
                st.raise_for_status()
                if st.json().get("finished"):
                    break
            else:
                logging.warning("remote locust did not finish in time")
                return
            record_linkerd_stat("end")
            for fname in [f"{scenario}_stats.csv", f"{scenario}_stats_history.csv", f"{scenario}.html"]:
                r = requests.get(f"{host}/download/{tag}/{fname}", timeout=10)
                if r.status_code == 200:
                    (out_dir / fname).write_bytes(r.content)
            return
        except requests.RequestException as exc:
            logging.error("remote locust failed: %s", exc)
            logging.info("Fallback to local locust")

    csv_prefix = out_dir / scenario
    html_file = out_dir / f"{scenario}.html"
    cmd = [
        "locust", "-f", script, "--headless", "--run-time", RUN_TIME,
        "--host", TARGET_HOST,
        "--csv", csv_prefix, "--csv-full-history",
        "--html", html_file,
    ]
    proc = subprocess.Popen(cmd)
    record_linkerd_stat("start")
    time.sleep(HALF_RUN_SEC)
    record_linkerd_stat("mid")
    proc.wait()
    record_linkerd_stat("end")
    if proc.returncode:
        logging.warning(
            "Locust %s finished with exit-code %s (failures present)", scenario, proc.returncode
        )


def summarise(hpa: str, scenario_dirs: list[Path]):
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
        rows.append({
            "HPA": hpa,
            "Scenario": d.name,
            "Requests": tot.get("Request Count", 0),
            "Failures": tot.get("Failure Count", 0),
            "Avg RPS": tot.get("Requests/s", 0),
            "P95 ms": tot.get("95%", 0),
        })
    return pd.DataFrame(rows, columns=["HPA", "Scenario", "Requests", "Failures", "Avg RPS", "P95 ms"])

# ── main ────────────────────────────────────────────────────────────────

def main():
    LOG_ROOT.mkdir(exist_ok=True)
    master_rows = []

    for folder in sorted(p for p in HPA_ROOT.iterdir() if p.is_dir()):
        hpa_name = folder.name
        logging.info("=== HPA %s ===", hpa_name)
        start = dt.datetime.now()

        reset_demo()
        apply_hpa(folder)
        wait_frontend_ready()

        scenario_dirs = []
        for scn, file in LOCUST_SCRIPTS.items():
            logging.info("Cooldown %s for %d sec", scn, COOLDOWN_SEC)
            time.sleep(COOLDOWN_SEC)
            wait_frontend_ready()
            s_dir = LOG_ROOT / hpa_name / scn
            run_locust_once(scn, LOCUST_ROOT / file, s_dir)
            scenario_dirs.append(s_dir)

        delete_hpa(folder)
        end = dt.datetime.now()

        summarise(hpa_name, scenario_dirs).to_csv(LOG_ROOT / hpa_name / "summary.csv", index=False)
        master_rows.append({"HPA": hpa_name, "Start": start, "End": end, "LogDir": (LOG_ROOT/hpa_name).resolve().as_posix()})

    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(LOG_ROOT/"master_summary.csv", index=False)
    render_dashboard(df_master, LOG_ROOT)
    logging.info("✅ All tests done → %s", (LOG_ROOT/"aggregate.html").resolve())

# ── dashboard ──────────────────────────────────────────────────────────

def render_dashboard(df: pd.DataFrame, out_dir: Path):
    html = Template("""<html><head><title>{{ tag }}</title></head><body>
    {{ tbl | safe }}
    <ul>{% for _, r in df.iterrows() %}
      <li><a href='{{ r.IterDir }}/summary.csv'>{{ r.Iter }}</a></li>
    {% endfor %}</ul>
    </body></html>""").render(
        tag=out_dir.name,
        tbl=df.to_html(index=False),
        df=df                       # ← 關鍵：把 df 傳給樣板
    )
    (out_dir / "aggregate.html").write_text(html, encoding="utf-8")

if __name__ == "__main__":
    main()

