from locust import HttpUser, task, constant_throughput, LoadTestShape
import os
import logging

class StableUser(HttpUser):
    """穩定壓測用戶，即使失敗也維持RPS"""
    
    # 使用constant_throughput確保穩定的請求頻率
    wait_time = constant_throughput(1)  # 每個用戶每秒1個請求
    
    def on_start(self):
        """用戶啟動時的初始化"""
        self.request_count = 0
        self.failure_count = 0
    
    @task
    def stable_load_test(self):
        """穩定的負載測試任務"""
        self.request_count += 1
        
        try:
            # 執行請求，設置較長的超時時間
            with self.client.get("/cart", timeout=30, catch_response=True) as response:
                if response.status_code >= 400:
                    # 即使響應失敗，也記錄但繼續測試
                    self.failure_count += 1
                    logging.warning(f"Request failed with status {response.status_code}, but continuing test")
                    response.failure("HTTP error")
                else:
                    response.success()
        except Exception as e:
            # 捕獲所有異常但不中斷測試
            self.failure_count += 1
            logging.warning(f"Request exception: {e}, but continuing test")

class StableRushSaleShape(LoadTestShape):
    """穩定的搶購負載，突然上升到800 RPS，然後保持穩定"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        self.base_rps = int(os.getenv("LOCUST_BASE_RPS", "100"))  # 基礎RPS
        self.rush_rps = int(os.getenv("LOCUST_RUSH_RPS", "800"))  # 搶購時RPS
        self.rush_start_time = int(os.getenv("LOCUST_RUSH_START", "180"))  # 搶購開始時間(秒)
        self.rush_duration = int(os.getenv("LOCUST_RUSH_DURATION", "300"))  # 搶購持續時間(秒)
        
        print(f"🔧 穩定RushSale壓測配置:")
        print(f"   ⏱️  運行時間: {self.run_time_seconds}秒")
        print(f"   📊 基礎RPS: {self.base_rps}")
        print(f"   🚀 搶購時RPS: {self.rush_rps}")
        print(f"   🔥 搶購時間: {self.rush_start_time}秒 ~ {self.rush_start_time + self.rush_duration}秒")
    
    def _parse_time(self, time_str):
        """解析時間字符串"""
        if time_str.endswith('m'):
            return int(time_str[:-1]) * 60
        elif time_str.endswith('s'):
            return int(time_str[:-1])
        elif time_str.endswith('h'):
            return int(time_str[:-1]) * 3600
        else:
            return 900  # 預設15分鐘
    
    def tick(self):
        """返回當前時刻的用戶數和生成速率"""
        run_time = self.get_run_time()
        
        # 檢查是否超過運行時間
        if run_time >= self.run_time_seconds:
            return None
        
        # 判斷當前階段
        if run_time < self.rush_start_time:
            # 基礎負載階段
            target_users = self.base_rps
        elif run_time < self.rush_start_time + self.rush_duration:
            # 搶購階段
            target_users = self.rush_rps
        else:
            # 搶購結束，回到基礎負載
            target_users = self.base_rps
        
        # 固定用戶數，無抖動
        return (target_users, target_users)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')