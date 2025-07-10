#!/usr/bin/env python3
"""
Redis 穩定峰值負載測試
===================
實現固定RPS，無抖動的 Redis 壓測
"""

import random
import redis
import time
import os
import logging
from locust import HttpUser, task, constant_throughput, LoadTestShape
from locust.exception import StopUser

class StableRedisUser(HttpUser):
    """穩定的 Redis 負載測試用戶，每個用戶每秒固定1個請求"""
    
    # 每個用戶每秒固定1個請求，確保RPS = 用戶數
    wait_time = constant_throughput(1)
    
    def on_start(self):
        """初始化 Redis 連接"""
        try:
            # 連接到 Redis service
            self.redis_client = redis.Redis(
                host='redis-master.redis.svc.cluster.local',
                port=6379,
                db=0,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # 測試連接
            self.redis_client.ping()
            
            # 初始化計數器
            self.request_count = 0
            self.failure_count = 0
            
        except Exception as e:
            logging.error(f"Redis 連接失敗: {e}")
            raise StopUser()
    
    @task(40)
    def redis_get(self):
        """Redis GET 操作 (40% 比重)"""
        key = f"key_{random.randint(1, 1000)}"
        start_time = time.time()
        
        try:
            result = self.redis_client.get(key)
            # 使用 HTTP 格式記錄，方便 Locust 統計
            self.environment.events.request_success.fire(
                request_type="REDIS_GET",
                name=f"GET {key}",
                response_time=(time.time() - start_time) * 1000,
                response_length=len(str(result)) if result else 0,
            )
            self.request_count += 1
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_GET",
                name=f"GET {key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )
            self.failure_count += 1
            logging.warning(f"Redis GET 失敗: {e}, 但繼續測試")
    
    @task(30)
    def redis_set(self):
        """Redis SET 操作 (30% 比重)"""
        key = f"key_{random.randint(1, 1000)}"
        value = f"value_{random.randint(1, 10000)}"
        start_time = time.time()
        
        try:
            self.redis_client.set(key, value, ex=300)  # 5分鐘過期
            self.environment.events.request_success.fire(
                request_type="REDIS_SET",
                name=f"SET {key}",
                response_time=(time.time() - start_time) * 1000,
                response_length=len(value),
            )
            self.request_count += 1
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_SET",
                name=f"SET {key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )
            self.failure_count += 1
            logging.warning(f"Redis SET 失敗: {e}, 但繼續測試")
    
    @task(20)
    def redis_list_operations(self):
        """Redis List 操作 (20% 比重)"""
        list_key = f"list_{random.randint(1, 100)}"
        value = f"item_{random.randint(1, 1000)}"
        start_time = time.time()
        
        try:
            # 隨機選擇 LPUSH 或 RPOP
            if random.choice([True, False]):
                self.redis_client.lpush(list_key, value)
                operation = "LPUSH"
            else:
                result = self.redis_client.rpop(list_key)
                operation = "RPOP"
                
            self.environment.events.request_success.fire(
                request_type=f"REDIS_{operation}",
                name=f"{operation} {list_key}",
                response_time=(time.time() - start_time) * 1000,
                response_length=len(value),
            )
            self.request_count += 1
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_LIST",
                name=f"LIST {list_key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )
            self.failure_count += 1
            logging.warning(f"Redis LIST 操作失敗: {e}, 但繼續測試")
    
    @task(10)
    def redis_hash_operations(self):
        """Redis Hash 操作 (10% 比重)"""
        hash_key = f"hash_{random.randint(1, 50)}"
        field = f"field_{random.randint(1, 100)}"
        value = f"value_{random.randint(1, 1000)}"
        start_time = time.time()
        
        try:
            if random.choice([True, False]):
                self.redis_client.hset(hash_key, field, value)
                operation = "HSET"
            else:
                result = self.redis_client.hget(hash_key, field)
                operation = "HGET"
                
            self.environment.events.request_success.fire(
                request_type=f"REDIS_{operation}",
                name=f"{operation} {hash_key}",
                response_time=(time.time() - start_time) * 1000,
                response_length=len(value),
            )
            self.request_count += 1
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_HASH",
                name=f"HASH {hash_key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )
            self.failure_count += 1
            logging.warning(f"Redis HASH 操作失敗: {e}, 但繼續測試")

class StableRedisPeakShape(LoadTestShape):
    """穩定的 Redis 峰值負載，固定300 RPS，無抖動"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        self.target_rps = int(os.getenv("LOCUST_TARGET_RPS", "300"))  # 固定300 RPS
        self.target_users = self.target_rps  # 用戶數 = RPS (每用戶每秒1請求)
        
        print(f"🔧 穩定Redis Peak壓測配置:")
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