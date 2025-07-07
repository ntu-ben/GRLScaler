#!/usr/bin/env python3
"""
Redis OffPeak Load Test
======================
模擬 Redis 低峰負載場景的 Locust 測試腳本
"""

import random
import redis
from locust import HttpUser, task, between
from locust.exception import StopUser
import time

class RedisOffPeakUser(HttpUser):
    """Redis 低峰負載測試用戶"""
    wait_time = between(1, 3)  # 1-3秒間隔，模擬低峰
    
    def on_start(self):
        """初始化 Redis 連接"""
        try:
            self.redis_client = redis.Redis(
                host='redis-master.redis.svc.cluster.local',
                port=6379,
                db=0,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis 連接失敗: {e}")
            raise StopUser()
    
    @task(50)
    def redis_get(self):
        """Redis GET 操作 (50% 比重)"""
        key = f"key_{random.randint(1, 500)}"
        start_time = time.time()
        
        try:
            result = self.redis_client.get(key)
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
        key = f"key_{random.randint(1, 500)}"
        value = f"value_{random.randint(1, 5000)}"
        start_time = time.time()
        
        try:
            self.redis_client.set(key, value, ex=600)  # 10分鐘過期
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
    def redis_basic_operations(self):
        """基本操作 (20% 比重)"""
        start_time = time.time()
        
        try:
            # 簡單的 INFO 命令
            info = self.redis_client.info('memory')
            self.environment.events.request_success.fire(
                request_type="REDIS_INFO",
                name="INFO memory",
                response_time=(time.time() - start_time) * 1000,
                response_length=len(str(info)),
            )
        except Exception as e:
            self.environment.events.request_failure.fire(
                request_type="REDIS_INFO",
                name="INFO memory",
                response_time=(time.time() - start_time) * 1000,
                exception=e,
                response_length=0,
            )

# 配置用戶負載
if __name__ == "__main__":
    # OffPeak 負載配置
    # 用戶數: 10-30
    # 每秒請求數: 50-100
    pass