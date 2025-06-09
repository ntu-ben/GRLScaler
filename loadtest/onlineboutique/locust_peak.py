from locust import HttpUser, task, between, LoadTestShape

class MyUser(HttpUser):
    wait_time = between(1, 1)
    @task
    def my_task(self):
        self.client.get("/cart")

class PeakShape(LoadTestShape):
    """Constant peak load.

    This shape originally stopped after 15 minutes. To support long-running
    tests (e.g. 24 hours from ``locust_agent_manual.py``), we keep spawning a
    fixed number of users until Locust's own ``--run-time`` ends.
    """

    def tick(self):
        # 500 users → 約 500 RPS
        return (500, 500)

