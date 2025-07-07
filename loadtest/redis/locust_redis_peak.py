#!/usr/bin/env python3
"""
Redis Peak Load Test
===================
模擬 Redis 高峰負載場景的 Locust 測試腳本
"""

import random
import redis
from locust import HttpUser, task, between
from locust.exception import StopUser
import time

class RedisUser(HttpUser):
    """Redis 負載測試用戶"""
    wait_time = between(0.1, 0.5)  # 100-500ms 間隔
    
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
            
        except Exception as e:
            print(f"Redis 連接失敗: {e}")
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
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_GET",
                name=f"GET {key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )
    
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
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_SET",
                name=f"SET {key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )
    
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
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_LIST",
                name=f"LIST {list_key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )
    
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
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_HASH",
                name=f"HASH {hash_key}",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )

# 配置用戶負載
if __name__ == "__main__":
    # Peak 負載配置
    # 用戶數: 100-200
    # 每秒請求數: 500-800
    pass