from locust import HttpUser, task, between, LoadTestShape

class MyUser(HttpUser):
    wait_time = between(1, 1)
    @task
    def my_task(self):
        self.client.get("/cart")

class FluctuatingShape(LoadTestShape):
    PHASE_DURATION = (15 * 60) / 4
    def tick(self):
        t = self.get_run_time()
        if t >= 15 * 60:
            return None
        phase = int(t // self.PHASE_DURATION)
        # 四個階段 user 數：[50, 1000, 50, 3000]
        users = [50, 300, 50, 800][phase]
        return (users, users)

