from locust import HttpUser, task, between, LoadTestShape
import os

class MyUser(HttpUser):
    wait_time = between(1, 1)
    @task
    def my_task(self):
        self.client.get("/cart")

class FluctuatingShape(LoadTestShape):
    PHASE_DURATION = (15 * 60) / 4

    def __init__(self):
        super().__init__()
        self.time_limit = int(os.environ.get('LOCUST_RUN_TIME', 900))  # 預設 15 分鐘

    def tick(self):
        """Fluctuating traffic pattern that loops every 15 minutes."""
        
        # 檢查是否超過時間限制
        if self.get_run_time() >= self.time_limit:
            return None

        t = self.get_run_time() % (15 * 60)
        phase = int(t // self.PHASE_DURATION)
        # 四個階段 user 數：[50, 300, 50, 800]
        users = [50, 300, 50, 800][phase]
        return (users, users)

