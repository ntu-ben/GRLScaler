from pathlib import Path
import os

PROM_URL = os.getenv("PROM_URL", "http://localhost:9090")
VIZ_URL = os.getenv("VIZ_URL", "http://localhost:8085")
# default data directory at repo_root/datasets/scraper
DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "scraper"
DATA_DIR.mkdir(parents=True, exist_ok=True)
