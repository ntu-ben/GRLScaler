from locust import User, task, constant_throughput, LoadTestShape
import os
import logging
import redis
import time

class RedisLoadUser(User):
    """Redis負載測試用戶 - 波動模式"""
    
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
    
    @task(4)
    def redis_get_operation(self):
        """Redis GET操作 - 波動模式混合讀寫"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        start_time = time.time()
        
        try:
            # 波動模式：混合讀取操作
            key_types = ["session", "cache", "counter"]
            key_type = key_types[self.request_count % len(key_types)]
            
            if key_type == "session":
                # 會話數據讀取
                session_key = f"session:{self.request_count % 500}"
                session_data = self.redis_client.hgetall(session_key)
                
            elif key_type == "cache":
                # 緩存數據讀取
                cache_key = f"cache:{self.request_count % 200}"
                cache_data = self.redis_client.get(cache_key)
                
            elif key_type == "counter":
                # 計數器讀取
                counter_key = f"counter:{self.request_count % 50}"
                counter_value = self.redis_client.get(counter_key)
            
            # 記錄成功統計
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="GET", response_time=total_time, response_length=0, exception=None
            )
            
        except Exception as e:
            self.failure_count += 1
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="GET", response_time=total_time, response_length=0, exception=e
            )
            logging.warning(f"Redis GET操作失敗: {e}")
    
    @task(3)
    def redis_set_operation(self):
        """Redis SET操作 - 波動模式混合寫入"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        start_time = time.time()
        
        try:
            # 波動模式：混合寫入操作
            operation_type = self.request_count % 3
            
            if operation_type == 0:
                # 會話數據寫入
                session_key = f"session:{self.request_count % 500}"
                session_data = {
                    "user_id": f"user_{self.request_count % 1000}",
                    "timestamp": str(self.request_count),
                    "action": "browse"
                }
                self.redis_client.hset(session_key, mapping=session_data)
                self.redis_client.expire(session_key, 1800)  # 30分鐘過期
                
            elif operation_type == 1:
                # 緩存數據寫入
                cache_key = f"cache:{self.request_count % 200}"
                cache_value = f"cached_data_{self.request_count}"
                self.redis_client.setex(cache_key, 600, cache_value)  # 10分鐘過期
                
            elif operation_type == 2:
                # 計數器增加
                counter_key = f"counter:{self.request_count % 50}"
                self.redis_client.incr(counter_key)
                self.redis_client.expire(counter_key, 3600)  # 1小時過期
            
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
    def redis_list_operation(self):
        """Redis LIST操作 - 波動模式隊列操作"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        start_time = time.time()
        
        try:
            # 波動模式：隊列操作
            queue_key = "event_queue"
            event_data = f"event_{self.request_count}"
            
            # 隨機進行推入或彈出操作
            if self.request_count % 2 == 0:
                # 推入操作
                self.redis_client.lpush(queue_key, event_data)
                # 限制隊列長度
                self.redis_client.ltrim(queue_key, 0, 499)
            else:
                # 彈出操作
                self.redis_client.rpop(queue_key)
            
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
    
    @task(1)
    def redis_sorted_set_operation(self):
        """Redis ZSET操作 - 波動模式排行榜"""
        if self.redis_client is None:
            return
            
        self.request_count += 1
        start_time = time.time()
        
        try:
            # 波動模式：排行榜操作
            leaderboard_key = "leaderboard"
            user_id = f"user_{self.request_count % 1000}"
            score = self.request_count % 1000
            
            # 更新排行榜
            self.redis_client.zadd(leaderboard_key, {user_id: score})
            
            # 保留top 100
            self.redis_client.zremrangebyrank(leaderboard_key, 0, -101)
            
            # 記錄成功統計
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="ZSET", response_time=total_time, response_length=0, exception=None
            )
            
        except Exception as e:
            self.failure_count += 1
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="ZSET", response_time=total_time, response_length=0, exception=e
            )
            logging.warning(f"Redis ZSET操作失敗: {e}")

class FluctuatingShape(LoadTestShape):
    """波動模式負載形狀 - 四階段循環"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        
        # 四階段RPS配置 [低峰, 中峰, 低峰, 高峰] - 參考OnlineBoutique設計
        self.phase_rps = [
            int(os.getenv("LOCUST_PHASE1_RPS", "500")),   # 第1階段: 500 RPS (低峰)
            int(os.getenv("LOCUST_PHASE2_RPS", "2000")),  # 第2階段: 2000 RPS (中峰)
            int(os.getenv("LOCUST_PHASE3_RPS", "500")),   # 第3階段: 500 RPS (低峰)
            int(os.getenv("LOCUST_PHASE4_RPS", "4000"))   # 第4階段: 4000 RPS (高峰)
        ]
        
        self.phase_duration = self.run_time_seconds / 4  # 每個階段平均分配時間
        
        print(f"🔧 Redis波動模式配置 (參考OnlineBoutique):")
        print(f"   ⏱️  運行時間: {self.run_time_seconds}秒")
        print(f"   📊 四階段RPS: {self.phase_rps} [低峰-中峰-低峰-高峰]")
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
        
        # 計算當前階段
        cycle_time = run_time % self.run_time_seconds
        phase = int(cycle_time // self.phase_duration)
        phase = min(phase, 3)  # 確保不超過3（四個階段：0,1,2,3）
        
        target_users = self.phase_rps[phase]
        
        # OnlineBoutique風格: 穩定用戶數，確保無抖動
        return (target_users, target_users)

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')