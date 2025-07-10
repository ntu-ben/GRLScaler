from locust import HttpUser, task, constant_throughput, LoadTestShape
import os
import logging

class StableUser(HttpUser):
    """穩定壓測用戶，每個用戶每秒固定1個請求"""
    
    # 每個用戶每秒固定1個請求，確保RPS = 用戶數
    wait_time = constant_throughput(1)
    
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

class StablePeakShape(LoadTestShape):
    """穩定的峰值負載，固定400 RPS，無抖動"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        self.target_rps = int(os.getenv("LOCUST_TARGET_RPS", "400"))  # 固定400 RPS
        self.target_users = self.target_rps  # 用戶數 = RPS (每用戶每秒1請求)
        
        print(f"🔧 穩定Peak壓測配置:")
        print(f"   ⏱️  運行時間: {self.run_time_seconds}秒")
        print(f"   📊 目標RPS: {self.target_rps} (固定)")
        print(f"   👥 目標用戶數: {self.target_users}")
    
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
        
        # 立即達到目標用戶數，保持穩定
        # 用戶數固定 = 目標RPS，無抖動
        return (self.target_users, self.target_users)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')