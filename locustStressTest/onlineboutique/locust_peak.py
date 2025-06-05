from locust import HttpUser, task, between, LoadTestShape

class MyUser(HttpUser):
    wait_time = between(1, 1)
    @task
    def my_task(self):
        self.client.get("/cart")

class PeakShape(LoadTestShape):
    TOTAL_RUN_TIME = 15 * 60
    def tick(self):
        if self.get_run_time() < self.TOTAL_RUN_TIME:
            # 500 users → 約 500 RPS
            return (500, 500)
        return None

