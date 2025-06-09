from locust import HttpUser, task, between, LoadTestShape

class MyUser(HttpUser):
    wait_time = between(1, 1)    # 每位 user 約 1 RPS
    @task
    def my_task(self):
        self.client.get("/cart")

class OffPeakShape(LoadTestShape):
    """Constant low traffic.

    The previous implementation stopped after ``TOTAL_RUN_TIME``. To allow
    extended runs, we simply maintain the off‑peak user count until Locust
    exits on its own.
    """

    def tick(self):
        # 50 users → 約 50 RPS (spawn_rate = users)
        return (50, 50)

