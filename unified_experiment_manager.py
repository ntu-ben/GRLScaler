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
import yaml
import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

# 載入實驗配置
CONFIG_FILE = Path(__file__).parent / "experiment_config.yaml"

class UnifiedExperimentManager:
    def __init__(self, config_path: Path = CONFIG_FILE):
        """初始化統一實驗管理器"""
        self.repo_root = Path(__file__).parent
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._load_environment()
        
    def _load_config(self, config_path: Path) -> dict:
        """載入實驗配置檔案"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """設定日誌系統"""
        log_file = os.getenv('UNIFIED_EXPERIMENT_LOG', 'unified_experiment.log')
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
    
    def validate_environment(self) -> bool:
        """驗證實驗環境"""
        self.logger.info("🔍 驗證實驗環境...")
        
        # 檢查 Kubernetes 環境
        if not self._check_k8s_environment():
            return False
        
        # 檢查分散式測試環境
        if not self._check_distributed_testing():
            self.logger.warning("⚠️ 分散式測試環境未配置，將使用本地測試")
        
        # 檢查實驗腳本
        if not self._check_experiment_scripts():
            return False
        
        self.logger.info("✅ 環境驗證通過")
        return True
    
    def _check_k8s_environment(self) -> bool:
        """檢查 Kubernetes 環境"""
        try:
            # 檢查 kubectl 命令
            subprocess.run(['kubectl', 'version', '--client'], 
                         capture_output=True, check=True)
            
            # 檢查 onlineboutique namespace
            result = subprocess.run(
                ['kubectl', 'get', 'pods', '-n', 'onlineboutique', '--no-headers'],
                capture_output=True, text=True, check=True
            )
            
            running_pods = [p for p in result.stdout.strip().split('\n') if 'Running' in p]
            if len(running_pods) < 10:
                self.logger.error(f"❌ OnlineBoutique 環境不完整，僅 {len(running_pods)} 個 Pod 運行")
                return False
            
            self.logger.info(f"✅ Kubernetes 環境正常，{len(running_pods)} 個服務運行中")
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
        
        # 使用現有的 rl_batch_loadtest.py
        batch_script = self.repo_root / "gnnrl" / "training" / "rl_batch_loadtest.py"
        
        cmd = [
            sys.executable, str(batch_script),
            "--model", "gym-hpa",
            "--run-tag", str(run_tag),
            "--use-case", str(use_case),
            "--goal", str(kwargs.get('goal', 'latency')),
            "--total-steps", str(kwargs.get('steps', 5000)),
            "--alg", str(kwargs.get('alg', 'ppo')),
            "--seed", str(kwargs.get('seed', 42)),
            "--env-step-interval", str(kwargs.get('env_step_interval', 15.0))
        ]
        
        # 測試模式或訓練模式
        if kwargs.get('testing', False):
            cmd.append("--testing")
            self.logger.info("🧪 使用測試模式")
        else:
            cmd.append("--training")
            self.logger.info("🎯 使用訓練模式")
            
        if kwargs.get('load_path'):
            cmd.extend(["--load-path", str(kwargs.get('load_path'))])
            self.logger.info(f"📂 載入模型: {kwargs.get('load_path')}")
        
        # 只有在指定 k8s 時才添加 --k8s 參數
        if kwargs.get('k8s', False):
            cmd.append("--k8s")
            self.logger.info("✅ 啟用 K8s 集群模式")
        else:
            self.logger.info("🔄 使用模擬模式")
        
        return self._execute_experiment_command(cmd, run_tag)
    
    def _run_k8s_hpa_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行 k8s_hpa 基準測試"""
        self.logger.info("📊 執行 K8s HPA 基準測試")
        
        # HPA 基準測試只能在真實 K8s 環境中運行
        if not kwargs.get('k8s', True):  # HPA 預設需要 K8s 環境
            self.logger.warning("⚠️ HPA 基準測試需要真實 K8s 環境，自動啟用 --k8s 模式")
        
        batch_script = self.repo_root / "gnnrl" / "training" / "rl_batch_loadtest.py"
        
        cmd = [
            sys.executable, str(batch_script),
            "--model", "hpa",
            "--run-tag", run_tag
        ]
        
        self.logger.info("✅ 使用真實 K8s 集群進行 HPA 基準測試")
        return self._execute_experiment_command(cmd, run_tag)
    
    def _run_gnnrl_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行 GNNRL 實驗"""
        use_case = kwargs.get('use_case', 'online_boutique')
        self.logger.info(f"🧠 執行 GNNRL 實驗 (應用場景: {use_case})")
        
        # 檢查 GNNRL 是否支持指定的 use case
        if use_case == 'redis':
            self.logger.warning("⚠️ GNNRL 目前主要針對 OnlineBoutique 優化，Redis 支持可能有限")
        
        batch_script = self.repo_root / "gnnrl" / "training" / "rl_batch_loadtest.py"
        
        cmd = [
            sys.executable, str(batch_script),
            "--model", "gnnrl",
            "--run-tag", str(run_tag),
            "--use-case", str(use_case),
            "--steps", str(kwargs.get('steps', 5000)),
            "--goal", str(kwargs.get('goal', 'latency')),
            "--alg", str(kwargs.get('alg', 'ppo')),
            "--gnn-model", str(kwargs.get('model', 'gat')),
            "--seed", str(kwargs.get('seed', 42)),
            "--env-step-interval", str(kwargs.get('env_step_interval', 15.0))
        ]
        
        # 只有在指定 k8s 時才添加 --k8s 參數
        if kwargs.get('k8s', False):
            cmd.append("--k8s")
            self.logger.info("✅ 啟用 K8s 集群模式")
        else:
            self.logger.info("🔄 使用模擬模式")
        
        # 測試模式參數
        if kwargs.get('testing', False):
            cmd.append("--testing")
            self.logger.info("🧪 使用測試模式")
            
        if kwargs.get('load_path'):
            cmd.extend(["--load-path", str(kwargs.get('load_path'))])
            self.logger.info(f"📂 載入模型: {kwargs.get('load_path')}")
        
        return self._execute_experiment_command(cmd, run_tag)
    
    def _execute_experiment_command(self, cmd: List[str], run_tag: str) -> bool:
        """執行實驗命令"""
        # Debug: 檢查命令參數
        for i, arg in enumerate(cmd):
            if arg is None:
                self.logger.error(f"❌ 命令參數 {i} 為 None: {cmd}")
                return False
        
        self.logger.info(f"💻 執行命令: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=False)
            execution_time = time.time() - start_time
            
            self.logger.info(f"✅ 實驗完成 ({execution_time:.2f}s): {run_tag}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ 實驗執行失敗: {e}")
            return False
    
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
    parser.add_argument('--model', choices=['gat', 'gcn'], 
                       default='gat',
                       help='GNN 模型類型 (僅適用於 gnnrl 實驗)')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子 (用於控制 Locust 情境執行順序)')
    parser.add_argument('--env-step-interval', type=float, default=15.0,
                       help='環境步驟間隔秒數 (模型接收新數據的頻率)')
    parser.add_argument('--run-tag', help='運行標籤')
    parser.add_argument('--k8s', action='store_true',
                       help='啟用真實 K8s 集群模式 (預設: 模擬模式)')
    parser.add_argument('--simulation', action='store_true',
                       help='強制使用模擬模式 (覆蓋 --k8s)')
    
    # 測試模式參數
    parser.add_argument('--testing', action='store_true',
                       help='使用已訓練模型進行測試 (需搭配 --load-path)')
    parser.add_argument('--load-path', type=str,
                       help='已訓練模型的路徑 (用於測試模式)')
    
    # 其他功能
    parser.add_argument('--validate-only', action='store_true',
                       help='僅驗證環境')
    parser.add_argument('--loadtest-only', action='store_true',
                       help='僅執行負載測試')
    parser.add_argument('--compare', nargs='+',
                       help='比較實驗結果路徑')
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = UnifiedExperimentManager()
    
    # 環境驗證
    if not manager.validate_environment():
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
            load_path=args.load_path
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
            k8s=args.k8s and not args.simulation
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
            k8s=args.k8s and not args.simulation
        )
        sys.exit(0 if all(results.values()) else 1)
    
    elif args.compare:
        manager.compare_experiments(args.compare)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()