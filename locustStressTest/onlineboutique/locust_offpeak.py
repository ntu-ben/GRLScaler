from locust import HttpUser, task, between, LoadTestShape

class MyUser(HttpUser):
    wait_time = between(1, 1)    # 每位 user 約 1 RPS
    @task
    def my_task(self):
        self.client.get("/cart")

class OffPeakShape(LoadTestShape):
    TOTAL_RUN_TIME = 15 * 60      # 15 分鐘
    def tick(self):
        if self.get_run_time() < self.TOTAL_RUN_TIME:
            # 50 users → 約 50 RPS
            return (50, 50)        # (target_user_count, spawn_rate)
        return None                 # 結束測試

