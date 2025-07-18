#!/usr/bin/env python3
"""
統一實驗管理器 (Unified Experiment Manager)
================================================

整合 gym_hpa, k8s_hpa, gnnrl 三個實驗，支援分散式 Locust 測試環境。

主要功能：
- 統一的命令行介面
- 自動實驗環境驗證
- 分散式負載測試協調
- 實驗結果聚合與比較
- 支援批次實驗執行

使用方式：
    # 執行單一實驗
    python unified_experiment_manager.py --experiment gnnrl --steps 5000
    
    # 批次執行所有實驗
    python unified_experiment_manager.py --batch-all --steps 3000
    
    # 僅執行負載測試 (不訓練)
    python unified_experiment_manager.py --loadtest-only
    
    # 比較實驗結果
    python unified_experiment_manager.py --compare logs/gym_hpa/run1 logs/gnnrl/run2
"""

import os
import sys
from pod_monitor import MultiPodMonitor, create_pod_monitor_for_experiment
import yaml
import argparse
import logging
import subprocess
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import requests

# 載入實驗配置
CONFIG_FILE = Path(__file__).parent / "experiment_config.yaml"

class UnifiedExperimentManager:
    def __init__(self, config_path: Path = CONFIG_FILE, stable_loadtest: bool = False, 
                 target_rps: int = None, loadtest_timeout: int = 30):
        """初始化統一實驗管理器"""
        self.repo_root = Path(__file__).parent
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._load_environment()
        
        # Stable loadtest 配置
        self.stable_loadtest = stable_loadtest
        self.target_rps = target_rps
        self.loadtest_timeout = loadtest_timeout
        
        self._setup_locust_scenarios()
        self._setup_hpa_configurations()
        
        # 初始化 timestamp 屬性
        from datetime import datetime
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def _setup_locust_scenarios(self):
        """設定 Locust 測試場景"""
        # 現在所有腳本都是穩定版本，直接使用原本命名
        self.scenarios = {
            "offpeak": "locust_offpeak.py",
            "rushsale": "locust_rushsale.py", 
            "peak": "locust_peak.py",
            "fluctuating": "locust_fluctuating.py"
        }
        
        # 從環境變數載入設定
        self.target_host = os.getenv("TARGET_HOST", "http://k8s.orb.local:8080")
        self.locust_run_time = os.getenv("LOCUST_RUN_TIME", "15m")
        self.m1_host = os.getenv("M1_HOST")
        self.kiali_url = os.getenv("KIALI_URL", "http://localhost:20001/kiali")
        self.namespace = os.getenv("NAMESPACE_ONLINEBOUTIQUE", "onlineboutique")
        self.redis_namespace = os.getenv("NAMESPACE_REDIS", "redis")
        
        # 計算運行時間（秒）
        self._parse_run_time()
        
    def _parse_run_time(self):
        """解析運行時間字串"""
        import re
        mult = {"s": 1, "m": 60, "h": 3600}
        match = re.match(r"(\d+)([smh])", self.locust_run_time)
        if match:
            self.run_time_sec = int(match.group(1)) * mult[match.group(2)]
            self.half_run_sec = self.run_time_sec // 2
        else:
            self.run_time_sec = 900  # 預設 15 分鐘
            self.half_run_sec = 450
    
    def _setup_hpa_configurations(self):
        """設定 HPA 配置選項"""
        self.hpa_configs = {
            'cpu': ['cpu-20', 'cpu-40', 'cpu-60', 'cpu-80'],  # 測試4種CPU配置
            'mem': ['mem-40', 'mem-80'],
            'hybrid': [
                'cpu-20-mem-40', 'cpu-20-mem-80',
                'cpu-40-mem-40', 'cpu-40-mem-80', 
                'cpu-60-mem-40', 'cpu-60-mem-80',
                'cpu-80-mem-40', 'cpu-80-mem-80'
            ]
        }
        
        # Redis HPA 配置 (簡化為只測試 CPU)
        self.redis_hpa_configs = {
            'cpu': ['cpu-20', 'cpu-40', 'cpu-60', 'cpu-80']
        }
        
        # HPA 配置根目錄
        self.hpa_root = self.repo_root / "macK8S" / "HPA" / "onlineboutique"
        self.redis_hpa_root = self.repo_root / "macK8S" / "HPA" / "redis"
        
    def _load_config(self, config_path: Path) -> dict:
        """載入實驗配置檔案"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """設定日誌系統"""
        # 確保 runtime 目錄存在
        runtime_dir = Path("logs/runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用時間戳創建唯一的日誌文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = runtime_dir / f"unified_experiment_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        return logging.getLogger('UnifiedExperimentManager')
    
    def _load_environment(self):
        """載入環境變數"""
        env_file = self.repo_root / '.env'
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                self.logger.info(f"✅ 已載入環境配置: {env_file}")
            except ImportError:
                self.logger.warning("python-dotenv 未安裝，手動解析 .env 檔案")
                self._manual_load_env(env_file)
        else:
            self.logger.warning("⚠️ 未找到 .env 檔案")
    
    def _manual_load_env(self, env_file: Path):
        """手動解析 .env 檔案"""
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    def validate_environment(self, use_case: str = "online_boutique") -> bool:
        """驗證實驗環境"""
        self.logger.info("🔍 驗證實驗環境...")
        
        # 檢查 Kubernetes 環境
        if not self._check_k8s_environment(use_case):
            return False
        
        # 檢查分散式測試環境
        if not self._check_distributed_testing():
            self.logger.warning("⚠️ 分散式測試環境未配置，將使用本地測試")
        
        # 檢查實驗腳本
        if not self._check_experiment_scripts():
            return False
        
        self.logger.info("✅ 環境驗證通過")
        return True
    
    def _check_k8s_environment(self, use_case: str = "online_boutique") -> bool:
        """檢查 Kubernetes 環境"""
        try:
            # 檢查 kubectl 命令
            subprocess.run(['kubectl', 'version', '--client'], 
                         capture_output=True, check=True)
            
            # 根據 use_case 選擇要檢查的 namespace 和期望的 Pod 數量
            if use_case == "redis":
                namespace = self.redis_namespace
                min_pods = 2  # redis-master, redis-slave (redis-exporter 是可選的)
                env_name = "Redis"
            else:
                namespace = self.namespace
                min_pods = 10  # OnlineBoutique 的 10 個微服務
                env_name = "OnlineBoutique"
            
            # 檢查指定 namespace 的 Pod
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', namespace, '--no-headers'],
                capture_output=True, text=True, check=True
            )
            
            if not result.stdout.strip():
                self.logger.error(f"❌ {env_name} namespace ({namespace}) 中沒有 Pod")
                return False
            
            running_pods = [p for p in result.stdout.strip().split('\n') if 'Running' in p]
            
            # 對於 Redis，只檢查核心服務
            if use_case == "redis":
                core_pods = [p for p in running_pods if 'redis-master' in p or 'redis-slave' in p]
                if len(core_pods) < min_pods:
                    self.logger.error(f"❌ {env_name} 核心服務不完整，僅 {len(core_pods)} 個核心 Pod 運行")
                    return False
            else:
                if len(running_pods) < min_pods:
                    self.logger.error(f"❌ {env_name} 環境不完整，僅 {len(running_pods)} 個 Pod 運行")
                    return False
            
            self.logger.info(f"✅ {env_name} 環境正常，{len(running_pods)} 個服務運行中")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.error(f"❌ Kubernetes 環境檢查失敗: {e}")
            return False
    
    def _check_distributed_testing(self) -> bool:
        """檢查分散式測試環境"""
        m1_host = os.getenv('M1_HOST')
        if not m1_host:
            return False
        
        try:
            import requests
            response = requests.get(f"{m1_host.rstrip('/')}/", timeout=5)
            self.logger.info(f"✅ 分散式測試代理連接正常: {m1_host}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ 分散式測試代理連接失敗: {e}")
            return False
    
    def _check_experiment_scripts(self) -> bool:
        """檢查實驗腳本存在性"""
        for exp_name, exp_config in self.config['experiments'].items():
            script_path = self.repo_root / exp_config['script_path']
            if not script_path.exists():
                self.logger.error(f"❌ 實驗腳本不存在: {script_path}")
                return False
        
        self.logger.info("✅ 實驗腳本檢查通過")
        return True
    
    def run_experiment(self, experiment: str, **kwargs) -> bool:
        """執行指定實驗"""
        if experiment not in self.config['experiments']:
            self.logger.error(f"❌ 未知實驗: {experiment}")
            return False
        
        exp_config = self.config['experiments'][experiment]
        self.logger.info(f"🚀 開始執行實驗: {exp_config['name']}")
        
        # 根據 use_case 設置正確的 namespace
        use_case = kwargs.get('use_case', 'online_boutique')
        if use_case == 'redis':
            self.namespace = self.redis_namespace
            self.target_host = "redis-master.redis.svc.cluster.local"
        else:
            self.namespace = os.getenv("NAMESPACE_ONLINEBOUTIQUE", "onlineboutique")
            self.target_host = os.getenv("TARGET_HOST", "http://k8s.orb.local:8080")
        
        self.logger.info(f"📍 設置環境: use_case={use_case}, namespace={self.namespace}")
        
        # 生成運行標籤 - 使用新的統一格式
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algorithm = kwargs.get('alg', 'ppo')
        model = kwargs.get('model', 'gat') if experiment == 'gnnrl' else 'baseline'
        goal = kwargs.get('goal', 'latency')
        steps = kwargs.get('steps', 5000)
        
        unified_tag = f"{timestamp}_{experiment}_{algorithm}_{model}_{goal}_{steps}"
        run_tag = kwargs.pop('run_tag', unified_tag)
        
        # 準備命令
        script_path = self.repo_root / exp_config['script_path']
        
        if experiment == 'gym_hpa':
            return self._run_gym_hpa_experiment(script_path, run_tag, **kwargs)
        elif experiment == 'k8s_hpa':
            return self._run_k8s_hpa_experiment(script_path, run_tag, **kwargs)
        elif experiment == 'gnnrl':
            return self._run_gnnrl_experiment(script_path, run_tag, **kwargs)
        else:
            self.logger.error(f"❌ 實驗執行器未實現: {experiment}")
            return False
    
    def _run_gym_hpa_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行 gym_hpa 實驗"""
        use_case = kwargs.get('use_case', 'online_boutique')
        self.logger.info(f"🎯 執行 Gym-HPA 實驗 (應用場景: {use_case})")
        
        # 直接調用 gym-hpa 腳本，不依賴 rl_batch_loadtest.py
        gym_hpa_script = self.repo_root / "gym-hpa" / "policies" / "run" / "run.py"
        
        cmd = [
            sys.executable, str(gym_hpa_script),
            "--alg", str(kwargs.get('alg', 'ppo')),
            "--use_case", str(use_case),
            "--goal", str(kwargs.get('goal', 'latency')),
            "--steps", str(kwargs.get('steps', 1000)),
            "--total_steps", str(kwargs.get('steps', 5000))
        ]
        
        # 測試模式或訓練模式
        if kwargs.get('testing', False):
            cmd.append("--testing")
            cmd.extend(["--test_path", kwargs.get('load_path')])
            self.logger.info("🧪 使用測試模式")
            training_proc = None
        else:
            cmd.append("--training")
            self.logger.info("🎯 使用訓練模式")
            
        if kwargs.get('k8s', False):
            cmd.append("--k8s")
            self.logger.info("✅ 啟用 K8s 集群模式")
        else:
            self.logger.info("🔄 使用模擬模式")
        
        # 開始訓練/測試進程
        if not kwargs.get('testing', False):
            # 訓練模式：並行執行
            training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gym-hpa")
            self.logger.info(f"🔄 Gym-HPA 訓練已開始，立即開始並行負載測試...")
        else:
            # 測試模式：也需要並行執行，讓測試過程中有流量
            training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gym-hpa")
            self.logger.info(f"🧪 Gym-HPA 測試已開始，立即開始並行負載測試...")
        
        # 根據測試/訓練模式選擇負載測試策略
        if kwargs.get('testing', False):
            # 測試模式：使用固定4場景評估性能
            self.logger.info("🧪 測試模式：執行固定4個場景")
            scenario_dirs = self.run_fixed_hpa_loadtest(
                "gym-hpa", run_tag, kwargs.get('seed', 42)
            )
        else:
            # 訓練模式：使用隨機場景提高泛化能力
            self.logger.info("🎯 訓練模式：使用隨機場景壓測")
            scenario_dirs = self.run_continuous_loadtest(
                "gym-hpa", run_tag, kwargs.get('seed', 42), training_proc
            )
        
        return len(scenario_dirs) > 0
    
    def _run_k8s_hpa_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行 k8s_hpa 基準測試"""
        self.logger.info("📊 執行 K8s HPA 基準測試")
        
        # HPA 基準測試只執行負載測試，不需要訓練進程
        self.logger.info("✅ 使用真實 K8s 集群進行 HPA 基準測試")
        
        # 獲取HPA配置類型選擇
        hpa_type = kwargs.get('hpa_type', 'all')  # all, cpu, mem, hybrid
        seed = kwargs.get('seed', 42)
        
        # 執行多配置HPA測試
        total_results = self.run_multi_hpa_experiment(
            "k8s-hpa", run_tag, seed, hpa_type
        )
        
        return len(total_results) > 0
    
    def _run_gnnrl_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行 GNNRL 實驗"""
        use_case = kwargs.get('use_case', 'online_boutique')
        self.logger.info(f"🧠 執行 GNNRL 實驗 (應用場景: {use_case})")
        
        # GNNRL 支持兩種環境
        if use_case == 'redis':
            self.logger.info("📊 GNNRL Redis 環境實驗")
        else:
            self.logger.info("📊 GNNRL OnlineBoutique 環境實驗")
        
        # 直接調用 GNNRL 腳本
        gnnrl_script = self.repo_root / "gnnrl" / "training" / "run_gnnrl_experiment.py"
        
        cmd = [
            sys.executable, str(gnnrl_script),
            "--steps", str(kwargs.get('steps', 5000)),
            "--goal", str(kwargs.get('goal', 'latency')),
            "--alg", str(kwargs.get('alg', 'ppo')),
            "--model", str(kwargs.get('model', 'gat')),
            "--env-step-interval", str(kwargs.get('env_step_interval', 15.0)),
            "--use-case", str(use_case)
        ]
        
        if kwargs.get('k8s', False):
            cmd.append("--k8s")
            self.logger.info("✅ 啟用 K8s 集群模式")
        else:
            self.logger.info("🔄 使用模擬模式")
        
        # GNNRL 測試模式處理
        if kwargs.get('testing', False):
            self.logger.info("🧪 GNNRL 測試模式：載入已訓練模型進行評估")
            load_path = kwargs.get('load_path')
            if not load_path or not Path(load_path).exists():
                self.logger.error(f"❌ 模型檔案不存在: {load_path}")
                return False
            
            cmd.extend([
                "--testing",
                "--load-path", str(load_path)
            ])
            
            self.logger.info(f"📂 載入模型檔案: {load_path}")
            # 測試模式：同步執行GNNRL測試和負載測試
            self.logger.info("🧪 GNNRL 測試模式：執行固定4個場景測試")
            
            # 先啟動負載測試，再啟動GNNRL測試進程
            import threading
            
            def run_gnnrl_testing():
                training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gnnrl")
                self.logger.info(f"🔄 GNNRL 測試進程已開始...")
                training_proc.wait()
                self.logger.info(f"✅ GNNRL 測試進程已完成")
            
            # 在后台启动GNNRL测试
            gnnrl_thread = threading.Thread(target=run_gnnrl_testing)
            gnnrl_thread.daemon = True
            gnnrl_thread.start()
            
            # 等待GNNRL进程初始化（3秒）
            time.sleep(3)
            
            # 立即开始负载测试场景
            selected_scenarios = kwargs.get('test_scenarios')
            if selected_scenarios:
                # 使用選定場景進行測試
                scenario_dirs = self.run_selected_scenarios_loadtest(
                    "gnnrl", run_tag, kwargs.get('seed', 42), selected_scenarios
                )
            else:
                # 使用所有場景進行測試（原來的行為）
                scenario_dirs = self.run_fixed_hpa_loadtest(
                    "gnnrl", run_tag, kwargs.get('seed', 42)
                )
            
            # 等待GNNRL测试完全结束
            gnnrl_thread.join()
        else:
            # 訓練模式：啟動 GNNRL 訓練進程
            self.logger.info("🎯 使用訓練模式")
            training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gnnrl")
            self.logger.info(f"🔄 GNNRL 訓練已開始，開始隨機場景壓測...")
            
            # 訓練模式：始終使用隨機場景壓測直到訓練完成
            # 忽略 stable_loadtest 參數，因為訓練需要隨機場景來學習不同負載模式
            scenario_dirs = self.run_continuous_loadtest(
                "gnnrl", run_tag, kwargs.get('seed', 42), training_proc
            )
        
        return len(scenario_dirs) > 0
    
    def run_batch_experiments(self, experiments: List[str], **kwargs) -> Dict[str, bool]:
        """批次執行多個實驗"""
        self.logger.info(f"🔄 批次執行實驗: {', '.join(experiments)}")
        
        results = {}
        for experiment in experiments:
            self.logger.info(f"{'='*60}")
            success = self.run_experiment(experiment, **kwargs)
            results[experiment] = success
            
            if success:
                self.logger.info(f"✅ {experiment} 實驗成功")
            else:
                self.logger.error(f"❌ {experiment} 實驗失敗")
            
            # 實驗間冷卻
            if experiment != experiments[-1]:
                cooldown = 120  # 2分鐘
                self.logger.info(f"⏸️ 實驗間冷卻 {cooldown} 秒...")
                time.sleep(cooldown)
        
        # 生成批次摘要
        self._generate_batch_summary(results)
        return results
    
    def _generate_batch_summary(self, results: Dict[str, bool]):
        """生成批次實驗摘要"""
        summary_file = self.repo_root / "logs" / "batch_summary.txt"
        summary_file.parent.mkdir(exist_ok=True)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("批次實驗執行摘要\n")
            f.write("="*50 + "\n")
            f.write(f"執行時間: {datetime.now()}\n\n")
            
            for experiment, success in results.items():
                status = "✅ 成功" if success else "❌ 失敗"
                f.write(f"{experiment}: {status}\n")
            
            success_count = sum(results.values())
            total_count = len(results)
            f.write(f"\n總計: {success_count}/{total_count} 個實驗成功\n")
        
        self.logger.info(f"📄 批次摘要已保存: {summary_file}")
    
    def record_kiali_graph(self, stage: str) -> None:
        """記錄 Kiali 服務圖"""
        self.logger.info(f"🔍 記錄 Kiali 圖表 ({stage})")
        url = f"{self.kiali_url}/api/namespaces/graph?namespaces={self.namespace}&duration=600s&graphType=workload"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            # 確保 kiali 目錄存在
            kiali_dir = Path("logs/kiali")
            kiali_dir.mkdir(parents=True, exist_ok=True)
            kiali_file = kiali_dir / f"kiali_{stage}_{self.timestamp}.json"
            kiali_file.write_text(resp.text, encoding="utf-8")
            self.logger.info(f"✅ Kiali 圖表已保存: {kiali_file}")
        except Exception as err:
            self.logger.warning(f"⚠️ Kiali 圖表記錄失敗: {err}")
    
    def _setup_pod_monitoring(self, scenario: str, out_dir: Path) -> Optional[MultiPodMonitor]:
        """設置Pod監控
        
        Args:
            scenario: 當前場景名稱
            out_dir: 輸出目錄
            
        Returns:
            配置好的Pod監控器，如果設置失敗則返回None
        """
        try:
            # 確定實驗類型 (從輸出路徑推斷)
            experiment_type = "unknown"
            if "gnnrl" in str(out_dir):
                experiment_type = "gnnrl"
            elif "gym-hpa" in str(out_dir) or "gym_hpa" in str(out_dir):
                experiment_type = "gym-hpa"
            elif "k8s-hpa" in str(out_dir) or "k8s_hpa" in str(out_dir):
                experiment_type = "k8s-hpa"
            
            # 設置Pod監控輸出目錄
            pod_monitoring_dir = out_dir / "pod_metrics"
            
            # 確定要監控的namespace列表
            namespaces_to_monitor = []
            
            # 根據當前使用的namespace添加監控
            if self.namespace:
                namespaces_to_monitor.append(self.namespace)
            
            # 如果是OnlineBoutique環境，也監控redis和default namespace（如果存在）
            if self.namespace == 'onlineboutique':
                # 檢查redis namespace是否存在
                try:
                    result = subprocess.run([
                        'kubectl', 'get', 'namespace', 'redis'
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        namespaces_to_monitor.append('redis')
                except Exception:
                    pass
            
            # 如果是Redis環境，也監控onlineboutique namespace（如果存在）
            elif self.namespace == 'redis':
                try:
                    result = subprocess.run([
                        'kubectl', 'get', 'namespace', 'onlineboutique'
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        namespaces_to_monitor.append('onlineboutique')
                except Exception:
                    pass
            
            if not namespaces_to_monitor:
                self.logger.warning("⚠️ 未找到可監控的namespace")
                return None
            
            # 創建Pod監控器
            pod_monitor = create_pod_monitor_for_experiment(
                experiment_type=experiment_type,
                scenario=scenario,
                namespaces=namespaces_to_monitor,
                output_dir=pod_monitoring_dir
            )
            
            self.logger.info(f"✅ Pod監控已設置 - 實驗類型: {experiment_type}, 場景: {scenario}, Namespaces: {namespaces_to_monitor}")
            return pod_monitor
            
        except Exception as e:
            self.logger.error(f"❌ Pod監控設置失敗: {e}")
            return None

    def run_distributed_locust(self, scenario: str, tag: str, out_dir: Path) -> bool:
        """運行分散式 Locust 測試"""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 在負載測試前重置 Pod 數量
        self.logger.info(f"🔄 場景 {scenario} 測試前重置 Pod 數量")
        if not self._reset_all_namespaces_pods():
            self.logger.warning("⚠️ Pod 重置失敗，但繼續進行負載測試")
        
        if self.m1_host:
            return self._run_remote_locust(scenario, tag, out_dir)
        else:
            return self._run_local_locust(scenario, out_dir)
    
    def _run_remote_locust(self, scenario: str, tag: str, out_dir: Path) -> bool:
        """運行遠端 Locust 測試"""
        host = self.m1_host.rstrip("/")
        self.logger.info(f"🔗 分散式測試: M1_HOST={host}")
        self.logger.info(f"🚀 觸發遠端 Locust {scenario}")
        
        # 自動判斷環境類型
        environment = 'onlineboutique' if self.namespace == 'onlineboutique' else 'redis'
        
        # 針對Redis環境，調整target_host
        target_host = self.target_host
        if environment == 'redis':
            # 對於Redis環境，使用NodePort 30379
            target_host = "redis://10.0.0.1:30379"
            self.logger.info(f"🔧 Redis環境detected，使用 target_host: {target_host}")
        
        payload = {
            "tag": tag,
            "scenario": scenario,
            "target_host": target_host,
            "run_time": self.locust_run_time,
            "environment": environment,
            "namespace": self.namespace,
            "stable_mode": self.stable_loadtest,
            "max_rps": self.target_rps,
            "timeout": self.loadtest_timeout,
        }
        
        try:
            # 開始遠端測試
            r = requests.post(f"{host}/start", json=payload, timeout=10)
            r.raise_for_status()
            job_id = r.json()["job_id"]
            self.logger.info(f"📋 遠端任務 ID: {job_id}")
            
            # 記錄開始狀態
            self.record_kiali_graph("start")
            
            # 啟動Pod監控
            pod_monitor = self._setup_pod_monitoring(scenario, out_dir)
            if pod_monitor:
                pod_monitor.start_all_monitoring(15)  # 15分鐘監控
            
            # 中途檢查點
            time.sleep(self.half_run_sec)
            self.record_kiali_graph("mid")
            
            # 等待完成
            max_checks = int(os.getenv("MAX_STATUS_CHECKS", "720"))
            for check_count in range(max_checks):
                time.sleep(5)
                
                st = requests.get(f"{host}/status/{job_id}", timeout=10)
                st.raise_for_status()
                data = st.json()
                
                if data.get("finished"):
                    self.logger.info(f"✅ 遠端測試 {scenario} 完成")
                    break
                    
                if check_count % 10 == 0:
                    self.logger.debug(f"⏳ 遠端測試狀態 [{check_count+1}/{max_checks}]: running")
            else:
                self.logger.warning("⏰ 遠端測試超時")
                return False
                
            self.record_kiali_graph("end")
            
            # 停止Pod監控
            if pod_monitor:
                pod_monitor.stop_all_monitoring()
            
            # 下載結果檔案
            downloaded_files = []
            for fname in [f"{scenario}_stats.csv", f"{scenario}_stats_history.csv", f"{scenario}.html"]:
                resp = requests.get(f"{host}/download/{tag}/{fname}", timeout=10)
                if resp.status_code == 200:
                    (out_dir / fname).write_bytes(resp.content)
                    downloaded_files.append(fname)
                else:
                    self.logger.warning(f"❌ 下載失敗: {fname}")
            
            self.logger.info(f"📊 遠端測試結果: 已下載 {len(downloaded_files)}/3 檔案")
            return len(downloaded_files) > 0
            
        except requests.RequestException as exc:
            self.logger.error(f"❌ 遠端測試失敗: {exc}")
            self.logger.info("🔄 切換到本地測試")
            return self._run_local_locust(scenario, out_dir)
    
    def _run_local_locust(self, scenario: str, out_dir: Path) -> bool:
        """運行本地 Locust 測試 - 支持兩種環境和 stable 模式"""
        # 檢查環境類型
        environment = 'onlineboutique' if self.namespace == 'onlineboutique' else 'redis'
        
        # 根據環境選擇腳本（現在都是穩定版本）
        if environment == 'redis':
            script_name = f"locust_redis_{scenario}.py"
        else:
            script_name = f"locust_{scenario}.py"
        
        # 優先嘗試環境專用腳本
        script_path = self.repo_root / "loadtest" / environment / script_name
        
        # 如果環境專用腳本不存在，嘗試fallback
        if not script_path.exists():
            if environment == 'redis':
                # Redis 環境，但腳本不存在
                self.logger.error(f"❌ Redis 測試腳本不存在: {script_name}")
                return False
            else:
                # OnlineBoutique 通用腳本
                script_path = self.repo_root / "loadtest" / "onlineboutique" / script_name
            
        if not script_path.exists():
            self.logger.error(f"❌ 測試腳本不存在: {script_path}")
            return False
            
        self.logger.info(f"🏠 運行本地 Locust {scenario} (環境: {environment})")
        
        # 準備環境變數
        env = os.environ.copy()
        
        # 設定目標 RPS（如果指定的話）
        if self.target_rps:
            env['LOCUST_TARGET_RPS'] = str(self.target_rps)
            self.logger.info(f"🎯 目標 RPS = {self.target_rps}")
        
        # 設定其他環境變數
        env['LOCUST_RUN_TIME'] = self.locust_run_time
        if hasattr(self, 'loadtest_timeout'):
            env['LOCUST_REQUEST_TIMEOUT'] = str(self.loadtest_timeout)
        
        cmd = [
            "locust", "-f", str(script_path), "--headless", "--run-time", self.locust_run_time,
            "--host", self.target_host,
            "--csv", str(out_dir / scenario), "--csv-full-history",
            "--html", str(out_dir / f"{scenario}.html"),
        ]
        
        proc = subprocess.Popen(cmd, env=env)
        
        self.record_kiali_graph("start")
        
        # 啟動Pod監控
        pod_monitor = self._setup_pod_monitoring(scenario, out_dir)
        if pod_monitor:
            pod_monitor.start_all_monitoring(15)  # 15分鐘監控
        
        time.sleep(self.half_run_sec)
        self.record_kiali_graph("mid")
        
        # 等待測試完成
        proc.wait()
        
        self.record_kiali_graph("end")
        
        # 停止Pod監控
        if pod_monitor:
            pod_monitor.stop_all_monitoring()
        
        if proc.returncode:
            self.logger.warning(f"⚠️ 本地測試 {scenario} 結束碼: {proc.returncode}")
            return False
        else:
            self.logger.info(f"✅ 本地測試 {scenario} 完成")
            return True

    def run_continuous_loadtest(self, experiment_type: str, run_tag: str, seed: int, training_proc: subprocess.Popen = None) -> List[Path]:
        """持續運行隨機 Locust 測試直到訓練完成"""
        random.seed(seed)
        scenario_list = list(self.scenarios.keys())
        scenario_dirs = []
        scenario_count = 0
        
        # 創建基礎輸出目錄
        base_output_dir = self.repo_root / "logs" / experiment_type / run_tag
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🎲 使用隨機種子 {seed}，可用情境: {', '.join(scenario_list)}")
        
        # 檢查是否有訓練進程需要等待
        has_training_proc = training_proc is not None
        
        # 持續隨機執行場景直到訓練完成或至少執行一個場景
        while True:
            # 檢查訓練是否完成
            if has_training_proc and training_proc.poll() is not None:
                self.logger.info("✅ 訓練進程已完成")
                break
            
            # 隨機選擇場景
            scenario = random.choice(scenario_list)
            scenario_count += 1
            
            # 創建唯一的輸出目錄
            out_dir = base_output_dir / f"{scenario}_{scenario_count:03d}"
            self.logger.info(f"📊 執行隨機測試情境 [第{scenario_count}個]: {scenario}")
            
            # 構建遠端標籤
            remote_tag = f"{experiment_type}/{run_tag}" if self.m1_host else run_tag
            
            # 執行 Locust 測試
            success = self.run_distributed_locust(scenario, remote_tag, out_dir)
            if success:
                scenario_dirs.append(out_dir)
            
            # 情境間冷卻時間
            if has_training_proc and training_proc.poll() is None:
                cooldown = int(os.getenv("COOLDOWN_BETWEEN_SCENARIOS", "60"))
                self.logger.info(f"⏸️ 情境間冷卻 {cooldown} 秒...")
                time.sleep(cooldown)
            elif not has_training_proc:
                # 如果沒有訓練進程，執行一個場景後結束
                break
        
        # 最終等待訓練完成
        if has_training_proc and training_proc.poll() is None:
            self.logger.info("⏳ 最終等待訓練進程完成...")
            training_proc.wait()
        
        self.logger.info(f"🏁 總共執行了 {len(scenario_dirs)} 個隨機場景測試")
        return scenario_dirs

    def run_fixed_hpa_loadtest(self, experiment_type: str, run_tag: str, seed: int) -> List[Path]:
        """運行固定的 4 個場景序列（用於基準測試和公平比較）"""
        # 測試模式：執行所有 4 個場景，使用 seed 來決定執行順序
        random.seed(seed)
        scenario_list = list(self.scenarios.keys())
        
        # 測試模式應該執行所有場景，只是順序根據 seed 決定
        if len(scenario_list) >= 4:
            fixed_sequence = scenario_list.copy()
            random.shuffle(fixed_sequence)  # 順序隨機化，但包含所有場景
        else:
            # 如果場景少於4個，就全部使用
            fixed_sequence = scenario_list
        
        self.logger.info(f"📋 測試模式：執行所有 {len(fixed_sequence)} 個場景 (順序 seed {seed}): {', '.join(fixed_sequence)}")
        
        # 創建基礎輸出目錄
        base_output_dir = self.repo_root / "logs" / experiment_type / run_tag
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        scenario_dirs = []
        
        # 執行固定序列的 4 個場景
        for i, scenario in enumerate(fixed_sequence, 1):
            out_dir = base_output_dir / f"{scenario}_{i:03d}"
            self.logger.info(f"📊 執行固定測試情境 [{i}/4]: {scenario}")
            
            # 構建遠端標籤
            remote_tag = f"{experiment_type}/{run_tag}" if self.m1_host else run_tag
            
            # 執行 Locust 測試
            success = self.run_distributed_locust(scenario, remote_tag, out_dir)
            if success:
                scenario_dirs.append(out_dir)
            
            # 場景間短暫冷卻
            if i < len(fixed_sequence):
                cooldown = 60  # 固定測試間的標準冷卻時間
                self.logger.info(f"⏸️ 固定場景間冷卻 {cooldown} 秒...")
                time.sleep(cooldown)
        
        self.logger.info(f"🏁 固定場景測試完成，執行了 {len(scenario_dirs)} 個場景")
        return scenario_dirs
    
    def run_selected_scenarios_loadtest(self, experiment_type: str, run_tag: str, seed: int, selected_scenarios: list) -> List[Path]:
        """執行選定場景的負載測試
        
        Args:
            experiment_type: 實驗類型
            run_tag: 運行標籤
            seed: 隨機種子（用於生成場景順序）
            selected_scenarios: 選定的場景列表，例如 ['peak', 'rushsale']
        
        Returns:
            執行成功的場景目錄列表
        """
        # 驗證選定場景
        available_scenarios = list(self.scenarios.keys())
        valid_scenarios = [s for s in selected_scenarios if s in available_scenarios]
        
        if not valid_scenarios:
            self.logger.error(f"❌ 沒有有效的場景。可用場景: {', '.join(available_scenarios)}")
            return []
        
        if len(valid_scenarios) < len(selected_scenarios):
            invalid_scenarios = [s for s in selected_scenarios if s not in available_scenarios]
            self.logger.warning(f"⚠️ 忽略無效場景: {', '.join(invalid_scenarios)}")
        
        # 使用種子決定場景順序
        random.seed(seed)
        test_sequence = valid_scenarios.copy()
        random.shuffle(test_sequence)  # 打亂順序但保持選定場景
        
        self.logger.info(f"📋 選定場景測試：執行 {len(test_sequence)} 個場景 (seed {seed}): {', '.join(test_sequence)}")
        
        # 創建基礎輸出目錄
        base_output_dir = self.repo_root / "logs" / experiment_type / run_tag
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        scenario_dirs = []
        
        # 執行選定場景
        for i, scenario in enumerate(test_sequence, 1):
            out_dir = base_output_dir / f"{scenario}_{i:03d}"
            self.logger.info(f"📊 執行選定測試情境 [{i}/{len(test_sequence)}]: {scenario}")
            
            # 構建遠端標籤
            remote_tag = f"{experiment_type}/{run_tag}" if self.m1_host else run_tag
            
            # 執行 Locust 測試
            success = self.run_distributed_locust(scenario, remote_tag, out_dir)
            if success:
                scenario_dirs.append(out_dir)
            
            # 場景間短暫冷卻
            if i < len(test_sequence):
                cooldown = 60  # 固定測試間的標準冷卻時間
                self.logger.info(f"⏸️ 場景間冷卻 {cooldown} 秒...")
                time.sleep(cooldown)
        
        self.logger.info(f"🏁 選定場景測試完成，執行了 {len(scenario_dirs)} 個場景")
        return scenario_dirs
    
    def run_multi_hpa_experiment(self, experiment_type: str, run_tag: str, seed: int, hpa_type: str = 'all') -> List[Path]:
        """執行多配置HPA測試
        
        Args:
            experiment_type: 實驗類型
            run_tag: 運行標籤 
            seed: 隨機種子（用於生成固定場景序列）
            hpa_type: HPA配置類型 ('all', 'cpu', 'mem', 'hybrid')
        
        Returns:
            所有測試結果目錄列表
        """
        
        # 獲取要測試的HPA配置
        if hpa_type == 'all':
            configs_to_test = []
            for config_type in self.hpa_configs:
                configs_to_test.extend(self.hpa_configs[config_type])
        elif hpa_type in self.hpa_configs:
            configs_to_test = self.hpa_configs[hpa_type]
        else:
            self.logger.error(f"❌ 不支援的HPA類型: {hpa_type}. 可用類型: all, cpu, mem, hybrid")
            return []
        
        self.logger.info(f"📈 測試HPA類型: {hpa_type}, 共 {len(configs_to_test)} 種配置")
        self.logger.info(f"📋 配置列表: {', '.join(configs_to_test)}")
        
        # 生成固定的場景序列（所有HPA配置都用相同序列）
        test_sequence = self._generate_hpa_test_sequence(seed)
        self.logger.info(f"🎲 使用固定測試序列 (seed {seed}): {', '.join(test_sequence)}")
        
        all_results = []
        
        for i, config_name in enumerate(configs_to_test, 1):
            self.logger.info(f"\n🔄 [{i}/{len(configs_to_test)}] 測試HPA配置: {config_name}")
            
            try:
                # 應用HPA配置
                if self._apply_hpa_config(config_name):
                    # 等待HPA生效
                    self.logger.info(f"⏳ 等待HPA配置生效 (30秒)...")
                    time.sleep(30)
                    
                    # 執行固定序列測試
                    config_results = self._run_hpa_config_test(
                        config_name, test_sequence, run_tag, experiment_type
                    )
                    all_results.extend(config_results)
                    
                    self.logger.info(f"✅ {config_name} 測試完成，產生 {len(config_results)} 個結果")
                else:
                    self.logger.error(f"❌ {config_name} HPA配置應用失敗")
                    
            except Exception as e:
                self.logger.error(f"❌ {config_name} 測試發生錯誤: {e}")
                continue
        
        self.logger.info(f"\n🏆 所有HPA測試完成! 共產生 {len(all_results)} 個結果")
        return all_results
    
    def _generate_hpa_test_sequence(self, seed: int) -> List[str]:
        """生成HPA測試的固定場景序列"""
        # 生成固定的場景序列（基於 seed）
        random.seed(seed)
        scenario_list = list(self.scenarios.keys())
        
        # 檢查是否已經有保存的序列
        sequence_file = self.repo_root / "logs" / "hpa_scenario_sequence.txt"
        
        if sequence_file.exists():
            # 讀取已保存的序列
            with open(sequence_file, 'r') as f:
                saved_sequences = {}
                for line in f:
                    if line.strip():
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            saved_seed, saved_sequence = parts
                            saved_sequences[int(saved_seed)] = saved_sequence.split(',')
            
            if seed in saved_sequences:
                return saved_sequences[seed]
                
        # 生成新序列（確保四個場景不重複）
        if len(scenario_list) >= 4:
            fixed_sequence = random.sample(scenario_list, 4)
        else:
            # 如果場景少於4個，就全部使用
            fixed_sequence = scenario_list
                
        return fixed_sequence
    
    def _reset_pod_replicas(self, target_namespace: str = None) -> bool:
        """重置指定 namespace 所有 deployment 的 replica 數量為 1
        
        Args:
            target_namespace: 目標 namespace，如果為 None 則使用當前 namespace
        """
        namespace = target_namespace or self.namespace
        self.logger.info(f"🔄 重置 {namespace} namespace 所有 Pod 數量到預設值 (1 replica)")
        
        try:
            # 獲取所有 deployment
            result = subprocess.run(
                ["kubectl", "get", "deployments", "-n", namespace, "-o", "name"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode != 0:
                self.logger.error(f"❌ 獲取 {namespace} deployment 列表失敗: {result.stderr}")
                return False
            
            deployments = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
            if not deployments:
                self.logger.warning(f"⚠️ 在 namespace {namespace} 中未找到 deployment")
                return True
            
            # 重置每個 deployment 的 replicas 為 1
            reset_count = 0
            for deployment in deployments:
                scale_result = subprocess.run(
                    ["kubectl", "scale", deployment, "--replicas=1", "-n", namespace],
                    capture_output=True, text=True, timeout=30
                )
                
                if scale_result.returncode == 0:
                    reset_count += 1
                    deployment_name = deployment.replace('deployment.apps/', '')
                    self.logger.info(f"✅ 重置 {namespace}/{deployment_name} 為 1 replica")
                else:
                    deployment_name = deployment.replace('deployment.apps/', '')
                    self.logger.warning(f"⚠️ 重置 {namespace}/{deployment_name} 失敗: {scale_result.stderr}")
            
            self.logger.info(f"🏁 完成 {namespace} Pod 重置，成功重置 {reset_count}/{len(deployments)} 個 deployment")
            
            # 等待 Pod 調整完成
            self.logger.info("⏳ 等待 Pod 重置生效 (30秒)...")
            time.sleep(30)
            
            return reset_count > 0
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ {namespace} Pod 重置操作超時")
            return False
        except Exception as e:
            self.logger.error(f"❌ {namespace} Pod 重置發生錯誤: {e}")
            return False

    def _reset_all_namespaces_pods(self) -> bool:
        """重置所有相關 namespace 的 Pod 數量"""
        self.logger.info("🔄 重置所有 namespace 的 Pod 數量")
        
        namespaces_to_reset = []
        
        # 根據當前應用場景確定要重置的 namespace
        if hasattr(self, 'use_case'):
            if self.use_case == 'redis':
                namespaces_to_reset.append(self.redis_namespace)
            elif self.use_case == 'online_boutique':
                namespaces_to_reset.append(self.namespace)
            else:
                # 未知場景，重置兩個都重置
                namespaces_to_reset.extend([self.namespace, self.redis_namespace])
        else:
            # 沒有 use_case 資訊，根據當前 namespace 判斷
            if self.namespace == 'redis':
                namespaces_to_reset.append(self.redis_namespace)
            else:
                namespaces_to_reset.append(self.namespace)
        
        reset_success = True
        for ns in namespaces_to_reset:
            if not self._reset_pod_replicas(ns):
                reset_success = False
                
        return reset_success

    def _apply_hpa_config(self, config_name: str) -> bool:
        """應用指定HPA配置"""
        config_dir = self.hpa_root / config_name
        
        if not config_dir.exists():
            self.logger.error(f"❌ HPA配置目錄不存在: {config_dir}")
            return False
        
        self.logger.info(f"🔧 應用HPA配置: {config_name}")
        
        try:
            # 先清除所有現有HPA
            result = subprocess.run(
                ["kubectl", "delete", "hpa", "--all", "-n", self.namespace],
                capture_output=True, text=True, timeout=30
            )
            
            # 重置所有 Pod 數量為 1
            if not self._reset_all_namespaces_pods():
                self.logger.warning("⚠️ Pod 重置失敗，但繼續應用 HPA 配置")
            
            # 應用新的HPA配置
            for hpa_file in config_dir.glob("*.yaml"):
                result = subprocess.run(
                    ["kubectl", "apply", "-f", str(hpa_file)],
                    capture_output=True, text=True, timeout=30
                )
                
                if result.returncode != 0:
                    self.logger.error(f"❌ 應用HPA檔案失敗: {hpa_file}")
                    self.logger.error(f"錯誤訊息: {result.stderr}")
                    return False
            
            self.logger.info(f"✅ HPA配置 {config_name} 應用成功")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ HPA配置應用超時: {config_name}")
            return False
        except Exception as e:
            self.logger.error(f"❌ HPA配置應用發生錯誤: {e}")
            return False
    
    def _run_hpa_config_test(self, config_name: str, test_sequence: List[str], 
                            run_tag: str, experiment_type: str) -> List[Path]:
        """執行單個HPA配置的測試"""
        results = []
        
        # 創建配置特定的輸出目錄
        config_output_dir = self.repo_root / "logs" / experiment_type / run_tag / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📋 執行 {config_name} 測試序列: {', '.join(test_sequence)}")
        
        for i, scenario in enumerate(test_sequence, 1):
            self.logger.info(f"\n📊 [{i}/4] 執行場景: {scenario}")
            
            # 為每個場景創建目錄
            scenario_dir = config_output_dir / f"{scenario}_{i:03d}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            # 執行單個場景測試
            remote_tag = f"{experiment_type}/{run_tag}/{config_name}" if self.m1_host else f"{run_tag}_{config_name}"
            if self.run_distributed_locust(scenario, remote_tag, scenario_dir):
                results.append(scenario_dir)
                self.logger.info(f"✅ {scenario} 測試完成")
            else:
                self.logger.error(f"❌ {scenario} 測試失敗")
            
            # 場景間關隔時間
            if i < len(test_sequence):
                self.logger.info(f"⏳ 場景間關隔時間 5 分鐘...")
                time.sleep(300)  # 5分鐘關隔
        
        return results
    
    def compare_experiments(self, result_paths: List[str]):
        """比較實驗結果"""
        self.logger.info("📊 比較實驗結果...")
        
        # 實現實驗結果比較邏輯
        # 這裡可以添加更詳細的比較分析
        pass

def main():
    parser = argparse.ArgumentParser(description="統一實驗管理器")
    
    # 實驗選擇
    parser.add_argument('--experiment', choices=['gym_hpa', 'k8s_hpa', 'gnnrl'], 
                       help='執行指定實驗')
    parser.add_argument('--batch-all', action='store_true',
                       help='批次執行所有實驗')
    parser.add_argument('--experiments', nargs='+', 
                       choices=['gym_hpa', 'k8s_hpa', 'gnnrl'],
                       help='批次執行指定實驗')
    
    # 實驗參數
    parser.add_argument('--steps', type=int, 
                       default=int(os.getenv('DEFAULT_STEPS', '5000')),
                       help='訓練步數')
    parser.add_argument('--goal', choices=['latency', 'cost'], 
                       default=os.getenv('DEFAULT_GOAL', 'latency'),
                       help='優化目標')
    parser.add_argument('--use-case', choices=['redis', 'online_boutique'], 
                       default=os.getenv('DEFAULT_USE_CASE', 'online_boutique'), 
                       help='應用場景')
    parser.add_argument('--alg', choices=['ppo', 'recurrent_ppo', 'a2c'], 
                       default='ppo',
                       help='強化學習算法')
    parser.add_argument('--model', choices=['gat', 'gcn', 'tgn'], 
                       default='gat',
                       help='GNN 模型類型 (僅適用於 gnnrl 實驗)')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子 (用於控制 Locust 情境執行順序)')
    parser.add_argument('--env-step-interval', type=float, default=15.0,
                       help='環境步驟間隔秒數 (模型接收新數據的頻率)')
    parser.add_argument('--run-tag', help='運行標籤')
    parser.add_argument('--hpa-type', choices=['all', 'cpu', 'mem', 'hybrid'], default='all',
                       help='K8s-HPA 測試配置類型 (all=所有, cpu=僅CPU, mem=僅記憶體, hybrid=混合)')
    parser.add_argument('--k8s', action='store_true',
                       help='啟用真實 K8s 集群模式 (預設: 模擬模式)')
    parser.add_argument('--simulation', action='store_true',
                       help='強制使用模擬模式 (覆蓋 --k8s)')
    
    # 測試模式參數
    parser.add_argument('--testing', action='store_true',
                       help='使用已訓練模型進行測試 (需搭配 --load-path)')
    parser.add_argument('--load-path', type=str,
                       help='已訓練模型的路徑 (用於測試模式)')
    parser.add_argument('--test-scenarios', nargs='+',
                       choices=['offpeak', 'peak', 'rushsale', 'fluctuating'],
                       help='選定要測試的場景，例如 --test-scenarios peak rushsale')
    
    # 其他功能
    parser.add_argument('--validate-only', action='store_true',
                       help='僅驗證環境')
    parser.add_argument('--loadtest-only', action='store_true',
                       help='僅執行負載測試')
    parser.add_argument('--enable-loadtest', action='store_true',
                       help='強制啟用負載測試（適用於測試模式）')
    parser.add_argument('--stable-loadtest', action='store_true',
                       help='使用穩定loadtest模式（失敗時維持RPS繼續測試）')
    parser.add_argument('--target-rps', type=int,
                       help='設定目標RPS數值（使用穩定loadtest模式時）')
    parser.add_argument('--loadtest-timeout', type=int, default=30,
                       help='Loadtest請求超時時間（秒）')
    parser.add_argument('--compare', nargs='+',
                       help='比較實驗結果路徑')
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = UnifiedExperimentManager(
        stable_loadtest=args.stable_loadtest,
        target_rps=args.target_rps,
        loadtest_timeout=args.loadtest_timeout
    )
    
    # 環境驗證
    if not manager.validate_environment(args.use_case):
        if not args.validate_only:
            sys.exit(1)
        else:
            return
    
    if args.validate_only:
        return
    
    # 執行實驗
    if args.experiment:
        success = manager.run_experiment(
            args.experiment,
            steps=args.steps,
            goal=args.goal,
            use_case=args.use_case,
            alg=args.alg,
            model=args.model,
            seed=args.seed,
            env_step_interval=args.env_step_interval,
            run_tag=args.run_tag,
            k8s=args.k8s and not args.simulation,
            testing=args.testing,
            load_path=args.load_path,
            hpa_type=args.hpa_type,
            test_scenarios=args.test_scenarios
        )
        sys.exit(0 if success else 1)
    
    elif args.batch_all:
        experiments = ['gym_hpa', 'k8s_hpa', 'gnnrl']
        results = manager.run_batch_experiments(
            experiments,
            steps=args.steps,
            goal=args.goal,
            use_case=args.use_case,
            alg=args.alg,
            model=args.model,
            seed=args.seed,
            env_step_interval=args.env_step_interval,
            k8s=args.k8s and not args.simulation,
            hpa_type=args.hpa_type
        )
        sys.exit(0 if all(results.values()) else 1)
    
    elif args.experiments:
        results = manager.run_batch_experiments(
            args.experiments,
            steps=args.steps,
            goal=args.goal,
            use_case=args.use_case,
            alg=args.alg,
            model=args.model,
            seed=args.seed,
            env_step_interval=args.env_step_interval,
            k8s=args.k8s and not args.simulation,
            hpa_type=args.hpa_type
        )
        sys.exit(0 if all(results.values()) else 1)
    
    elif args.compare:
        manager.compare_experiments(args.compare)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()