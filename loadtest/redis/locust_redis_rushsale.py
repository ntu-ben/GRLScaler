from locust import User, task, constant_throughput, LoadTestShape
import os
import logging
import redis
import time

class RedisLoadUser(User):
    """Redis負載測試用戶 - 搶購模式"""
    
    # 每個用戶每秒固定1個請求，確保RPS = 用戶數
    wait_time = constant_throughput(1)
    
    def on_start(self):
        """用戶啟動時的初始化"""
        self.request_count = 0
        self.failure_count = 0
        
        # 獲取Redis連接信息
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        logging.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        
        try:
            # 連接Redis
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            # 測試連接
            self.redis_client.ping()
            logging.info(f"✅ Redis連接成功: {redis_host}:{redis_port}")
        except Exception as e:
            logging.error(f"❌ Redis連接失敗: {e}")
            self.redis_client = None
    
    @task(3)
    def redis_set_operation(self):
        """Redis SET操作 - 搶購模式高頻寫入"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        start_time = time.time()
        
        try:
            # 搶購模式：高頻寫入購物車數據
            user_id = f"user_{self.request_count % 1000}"
            cart_key = f"cart:{user_id}"
            product_id = f"product_{self.request_count % 100}"
            
            # 模擬添加商品到購物車
            self.redis_client.hset(cart_key, product_id, 1)
            
            # 設置過期時間（1小時）
            self.redis_client.expire(cart_key, 3600)
            
            # 記錄成功統計
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="SET", response_time=total_time, response_length=0, exception=None
            )
            
        except Exception as e:
            self.failure_count += 1
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="SET", response_time=total_time, response_length=0, exception=e
            )
            logging.warning(f"Redis SET操作失敗: {e}")
    
    @task(2)
    def redis_get_operation(self):
        """Redis GET操作 - 搶購模式高頻讀取"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        start_time = time.time()
        
        try:
            # 搶購模式：高頻讀取購物車數據
            user_id = f"user_{self.request_count % 1000}"
            cart_key = f"cart:{user_id}"
            
            # 獲取購物車內容
            cart_data = self.redis_client.hgetall(cart_key)
            
            # 記錄成功統計
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="GET", response_time=total_time, response_length=len(str(cart_data)), exception=None
            )
            
        except Exception as e:
            self.failure_count += 1
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="GET", response_time=total_time, response_length=0, exception=e
            )
            logging.warning(f"Redis GET操作失敗: {e}")
    
    @task(1)
    def redis_list_operation(self):
        """Redis LIST操作 - 搶購模式訂單隊列"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        start_time = time.time()
        
        try:
            # 搶購模式：訂單隊列操作
            queue_key = "order_queue"
            order_id = f"order_{self.request_count}"
            
            # 推入訂單隊列
            self.redis_client.lpush(queue_key, order_id)
            
            # 限制隊列長度（保留最新1000個訂單）
            self.redis_client.ltrim(queue_key, 0, 999)
            
            # 記錄成功統計
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="LIST", response_time=total_time, response_length=0, exception=None
            )
            
        except Exception as e:
            self.failure_count += 1
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="LIST", response_time=total_time, response_length=0, exception=e
            )
            logging.warning(f"Redis LIST操作失敗: {e}")

class RushSaleShape(LoadTestShape):
    """搶購模式負載形狀 - 高RPS搶購場景"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        
        # 搶購模式RPS配置 - 參考OnlineBoutique設計
        self.base_rps = int(os.getenv("LOCUST_BASE_RPS", "500"))      # 基礎RPS
        self.rush_rps = int(os.getenv("LOCUST_RUSH_RPS", "6000"))     # 搶購峰值RPS
        self.rush_start_time = int(os.getenv("LOCUST_RUSH_START", "180"))     # 搶購開始時間(秒)
        self.rush_duration = int(os.getenv("LOCUST_RUSH_DURATION", "300"))   # 搶購持續時間(秒)
        
        print(f"🔧 Redis搶購模式配置 (參考OnlineBoutique):")
        print(f"   ⏱️  運行時間: {self.run_time_seconds}秒")
        print(f"   📊 基礎RPS: {self.base_rps}")
        print(f"   🚀 搶購峰值RPS: {self.rush_rps}")
        print(f"   🔥 搶購開始: {self.rush_start_time}秒後")
        print(f"   ⏲️  搶購持續: {self.rush_duration}秒")
    
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
        
        # OnlineBoutique風格的時間段判斷邏輯
        if run_time < self.rush_start_time:
            # 搶購開始前：基礎負載
            target_users = self.base_rps
        elif run_time < self.rush_start_time + self.rush_duration:
            # 搶購阶段：高峰負載
            target_users = self.rush_rps
        else:
            # 搶購結束後：回到基礎負載
            target_users = self.base_rps
        
        # 穩定用戶數，確保無抖動
        return (target_users, target_users)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')