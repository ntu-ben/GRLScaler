from locust import FastHttpUser, task, constant_throughput

class MyUser(FastHttpUser):
    wait_time = constant_throughput(100000)  # 每位使用者每 0.01 秒發送一次請求，約 100 RPS
    @task
    def my_task(self):
        self.client.get("/cart")

