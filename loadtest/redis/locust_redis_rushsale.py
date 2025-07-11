from locust import HttpUser, task, constant_throughput, LoadTestShape
import os
import logging
import redis

class RedisLoadUser(HttpUser):
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
        
        try:
            # 搶購模式：高頻寫入購物車數據
            user_id = f"user_{self.request_count % 1000}"
            cart_key = f"cart:{user_id}"
            product_id = f"product_{self.request_count % 100}"
            
            # 模擬添加商品到購物車
            self.redis_client.hset(cart_key, product_id, 1)
            
            # 設置過期時間（1小時）
            self.redis_client.expire(cart_key, 3600)
            
        except Exception as e:
            self.failure_count += 1
            logging.warning(f"Redis SET操作失敗: {e}")
    
    @task(2)
    def redis_get_operation(self):
        """Redis GET操作 - 搶購模式高頻讀取"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        
        try:
            # 搶購模式：高頻讀取購物車數據
            user_id = f"user_{self.request_count % 1000}"
            cart_key = f"cart:{user_id}"
            
            # 獲取購物車內容
            cart_data = self.redis_client.hgetall(cart_key)
            
        except Exception as e:
            self.failure_count += 1
            logging.warning(f"Redis GET操作失敗: {e}")
    
    @task(1)
    def redis_list_operation(self):
        """Redis LIST操作 - 搶購模式訂單隊列"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        
        try:
            # 搶購模式：訂單隊列操作
            queue_key = "order_queue"
            order_id = f"order_{self.request_count}"
            
            # 推入訂單隊列
            self.redis_client.lpush(queue_key, order_id)
            
            # 限制隊列長度（保留最新1000個訂單）
            self.redis_client.ltrim(queue_key, 0, 999)
            
        except Exception as e:
            self.failure_count += 1
            logging.warning(f"Redis LIST操作失敗: {e}")

class RushSaleShape(LoadTestShape):
    """搶購模式負載形狀 - 高RPS搶購場景"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        
        # 搶購模式RPS配置
        self.base_rps = int(os.getenv("LOCUST_BASE_RPS", "100"))      # 基礎RPS
        self.peak_rps = int(os.getenv("LOCUST_PEAK_RPS", "500"))     # 峰值RPS
        self.rush_start_ratio = float(os.getenv("LOCUST_RUSH_START", "0.2"))  # 搶購開始時間比例
        self.rush_end_ratio = float(os.getenv("LOCUST_RUSH_END", "0.8"))      # 搶購結束時間比例
        
        print(f"🔧 Redis搶購模式配置:")
        print(f"   ⏱️  運行時間: {self.run_time_seconds}秒")
        print(f"   📊 基礎RPS: {self.base_rps}")
        print(f"   🚀 搶購峰值RPS: {self.peak_rps}")
        print(f"   ⏳ 搶購時間段: {self.rush_start_ratio:.1%} - {self.rush_end_ratio:.1%}")
    
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
        
        # 計算當前時間比例
        time_ratio = run_time / self.run_time_seconds
        
        # 判斷當前是否處於搶購時間段
        if self.rush_start_ratio <= time_ratio <= self.rush_end_ratio:
            # 搶購時間段：使用峰值RPS
            target_users = self.peak_rps
        else:
            # 非搶購時間段：使用基礎RPS
            target_users = self.base_rps
        
        # 穩定用戶數，確保無抖動
        return (target_users, target_users)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')