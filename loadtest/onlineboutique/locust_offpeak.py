from locust import HttpUser, task, between, LoadTestShape
import os

class MyUser(HttpUser):
    wait_time = between(1, 1)    # 每位 user 約 1 RPS
    @task
    def my_task(self):
        self.client.get("/cart")

class OffPeakShape(LoadTestShape):
    """Constant low traffic.

    Runs for 15 minutes (900 seconds) by default, respecting the LOCUST_RUN_TIME environment variable.
    """
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取運行時間，預設15分鐘
        run_time_str = os.getenv("LOCUST_RUN_TIME", "15m")
        if run_time_str.endswith('m'):
            self.run_time_seconds = int(run_time_str[:-1]) * 60
        elif run_time_str.endswith('s'):
            self.run_time_seconds = int(run_time_str[:-1])
        elif run_time_str.endswith('h'):
            self.run_time_seconds = int(run_time_str[:-1]) * 3600
        else:
            self.run_time_seconds = 900  # 預設15分鐘

    def tick(self):
        run_time = self.get_run_time()
        
        # 如果超過設定時間，停止測試
        if run_time >= self.run_time_seconds:
            return None
            
        # 50 users → 約 50 RPS (spawn_rate = users)
        return (50, 50)

