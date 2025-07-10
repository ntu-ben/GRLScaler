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

class FluctuatingShape(LoadTestShape):
    """穩定的波動負載，四階段循環，每階段固定RPS"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        
        # 如果設定了 LOCUST_TARGET_RPS，使用固定值
        if os.getenv("LOCUST_TARGET_RPS"):
            self.target_rps = int(os.getenv("LOCUST_TARGET_RPS"))
            self.fixed_mode = True
            print(f"🔧 Fluctuating固定模式:")
            print(f"   ⏱️  運行時間: {self.run_time_seconds}秒")
            print(f"   📊 固定RPS: {self.target_rps}")
        else:
            self.fixed_mode = False
            # 四階段RPS配置 [低峰, 中峰, 低峰, 高峰]
            self.phase_rps = [
                int(os.getenv("LOCUST_PHASE1_RPS", "50")),   # 第1階段: 50 RPS
                int(os.getenv("LOCUST_PHASE2_RPS", "300")),  # 第2階段: 300 RPS
                int(os.getenv("LOCUST_PHASE3_RPS", "50")),   # 第3階段: 50 RPS
                int(os.getenv("LOCUST_PHASE4_RPS", "800"))   # 第4階段: 800 RPS
            ]
            self.phase_duration = self.run_time_seconds / 4  # 每個階段平均分配時間
            
            print(f"🔧 Fluctuating動態模式:")
            print(f"   ⏱️  運行時間: {self.run_time_seconds}秒")
            print(f"   📊 四階段RPS: {self.phase_rps}")
            print(f"   ⏳ 每階段時長: {self.phase_duration:.0f}秒")
    
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
        
        if self.fixed_mode:
            # 固定模式：直接使用 LOCUST_TARGET_RPS
            return (self.target_rps, self.target_rps)
        else:
            # 動態模式：計算當前階段
            cycle_time = run_time % self.run_time_seconds
            phase = int(cycle_time // self.phase_duration)
            phase = min(phase, 3)  # 確保不超過3（四個階段：0,1,2,3）
            
            target_users = self.phase_rps[phase]
            
            # 固定用戶數，無抖動
            return (target_users, target_users)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')