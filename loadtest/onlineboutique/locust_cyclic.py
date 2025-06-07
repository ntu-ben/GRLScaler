"""
locust_cyclic.py  – Unified 15‑minute cyclic workload for Online Boutique
────────────────────────────────────────────────────────────────────────────
* **Why**  ‑   把原本的 off‑peak／rush‑sale／peak／fluctuating 四段流量圖一次跑完，
              方便 autoscaler 反覆實驗，不必四支 script 來回切換。
* **Cycle** –  0‑2 min off‑peak → 2‑6 min rush‑sale → 6‑10 min peak →
              10‑15 min fluctuating → loop forever (15 min 取餘數)
              ┌───────────────┐  users
              │  peak 1200    │
   users 1200 ┤               │
              │               ├───╌╌╌╌ fluctuating 200‑600 (sine‑wave)
    800 ──────┤    rush‑sale  │
              │               │
    100 ──────┤ off‑peak      │
              └───────────────┘  time 0 → 15 min
* **Host**  –  取值順序：
              1. CLI `--host` (Locust 預設)
              2. 環境變數 `TARGET_HOST`
              3. fallback = "http://localhost:80"
  這樣 m4 執行時可以 `--host http://frontend.onlineboutique.svc.cluster.local`，
  或在 m1 的 systemd 服務裡寫 `Environment=TARGET_HOST=…`。
* **Endpoints** – 只示範最常用的三條；若要 1:1 對應舊 script，把函式從
              locust_* 直接貼過來即可。

2025‑06‑05  @ChatGPT
"""
from math import sin, pi
from random import choice
import os
from locust import HttpUser, task, between
from locust.runners import LoadTestShape

# ╭─ Config ────────────────────────────────────────────────────────────────╮
DEFAULT_HOST = "http://localhost:80"
TARGET_HOST  = os.getenv("TARGET_HOST", DEFAULT_HOST)

class Shopper(HttpUser):
    """最小可用 Online Boutique 行為。若要擴充，把舊 task copy 進來即可"""

    host      = TARGET_HOST   # CLI --host 會自動覆蓋這個屬性
    wait_time = between(1, 5)

    @task(3)
    def view_home(self):
        self.client.get("/")

    @task(2)
    def view_product(self):
        product_id = choice([
            "OLJCESPC7", "66VCHSJNUP", "1YMWWN1N4O", "L9ECAV7KIM",
            "2ZYFJ3GM2N", "0PUK6V6EV0", "LS4PSXUNUM", "9SIQT8TOJO",
        ])
        self.client.get(f"/product/{product_id}")

    @task(1)
    def add_to_cart(self):
        self.client.post(
            "/cart",
            json={"item": {"product_id": "OLJCESPC7", "quantity": 1}},
        )

# ╭─ 15‑minute cyclic load shape ────────────────────────────────────────────╮
class CyclicShape(LoadTestShape):
    """Four‑stage load, repeats every 900 s (15 min)."""

    # (開始秒, 使用者數)
    stages = [
        (0,    50),   # 0‑2 min   off‑peak ramp‑up 50→100
        (120, 100),
        (120, 100),   # 2‑6 min   rush‑sale 100→800
        (360, 800),
        (360, 800),   # 6‑10 min  peak 800→1200
        (600, 1200),
        (600, 200),   # 10‑15 min fluctuating 200‑600 (sine)
        (900, 600),
    ]

    def tick(self):
        run_time = self.get_run_time() % 900  # 取餘數: 每 15 min 重新來

        # 0‑10 min: 用線性插值
        if run_time <= 600:
            for i in range(0, len(self.stages) - 1, 2):
                start, users_start = self.stages[i]
                end,   users_end   = self.stages[i + 1]
                if start <= run_time < end:
                    # 線性內插
                    pct   = (run_time - start) / (end - start)
                    users = int(users_start + pct * (users_end - users_start))
                    return users, users  # spawn_rate = users
        else:
            # 10‑15 min: 正弦波 200‑600 users
            t      = run_time - 600  # 0‑300 s
            users  = int(400 + 200 * sin(2 * pi * t / 300))  # 200‑600
            return users, users

        return None  # Safety fallback – 容許 Locust 結束

