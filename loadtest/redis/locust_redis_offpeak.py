#!/usr/bin/env python3
"""
Redis OffPeak Load Test
======================
模擬 Redis 低峰負載場景的 Locust 測試腳本
"""

import random
import redis
from locust import User, task, between, LoadTestShape
from locust.exception import StopUser
import time
import os
import logging

class RedisOffPeakUser(User):
    """Redis 低峰負載測試用戶"""
    wait_time = between(1, 3)  # 1-3秒間隔，模擬低峰
    
    def on_start(self):
        """初始化 Redis 連接"""
        try:
            # 從環境變數獲取 Redis 連接信息
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            
            import logging
            logging.info(f"Connecting to Redis at {redis_host}:{redis_port}")
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
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
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="GET", response_time=total_time, response_length=len(str(result)) if result else 0, exception=None
            )
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="GET", response_time=total_time, response_length=0, exception=e
            )
    
    @task(30)
    def redis_set(self):
        """Redis SET 操作 (30% 比重)"""
        key = f"key_{random.randint(1, 500)}"
        value = f"value_{random.randint(1, 5000)}"
        start_time = time.time()
        
        try:
            self.redis_client.set(key, value, ex=600)  # 10分鐘過期
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="SET", response_time=total_time, response_length=len(value), exception=None
            )
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="SET", response_time=total_time, response_length=0, exception=e
            )
    
    @task(20)
    def redis_basic_operations(self):
        """基本操作 (20% 比重)"""
        start_time = time.time()
        
        try:
            # 簡單的 INFO 命令
            info = self.redis_client.info('memory')
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="INFO", response_time=total_time, response_length=len(str(info)), exception=None
            )
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request.fire(
                request_type="Redis", name="INFO", response_time=total_time, response_length=0, exception=e
            )

class RedisOffPeakShape(LoadTestShape):
    """Redis 離峰負載，固定150 RPS，無抖動"""
    
    def __init__(self):
        super().__init__()
        # 從環境變數讀取配置
        self.run_time_seconds = self._parse_time(os.getenv("LOCUST_RUN_TIME", "15m"))
        self.target_rps = int(os.getenv("LOCUST_TARGET_RPS", "500"))  # 固定500 RPS (Redis離峰)
        self.target_users = self.target_rps  # 用戶數 = RPS (每用戶每秒1請求)
        
        print(f"🔧 Redis OffPeak壓測配置:")
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
        
        # 固定用戶數，確保無抖動
        return (self.target_users, self.target_users)

# 配置用戶負載
if __name__ == "__main__":
    # OffPeak 負載配置
    # 用戶數: 根據 RPS 動態調整
    # 每秒請求數: 150 RPS (固定)
    pass
