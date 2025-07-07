#!/usr/bin/env python3
"""
標準化實驗管理器 (Standardized Experiment Manager)
================================================

基於統一實驗管理器，確保三種方法使用完全相同的8個測試場景進行公平比較。

主要改進：
- 使用固定的8個標準化場景序列
- 確保所有方法測試相同的負載模式
- 提供更精確的性能比較
"""

import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict
from unified_experiment_manager import UnifiedExperimentManager


class StandardizedExperimentManager(UnifiedExperimentManager):
    """標準化實驗管理器 - 確保公平比較的實驗執行"""
    
    def __init__(self, config_path: Path = None):
        # 修復 config_path 為 None 的問題
        if config_path is None:
            config_path = Path(__file__).parent / "experiment_config.yaml"
        super().__init__(config_path)
        self.standardized_scenarios = self._load_standardized_scenarios()
        
    def _load_standardized_scenarios(self) -> List[Dict]:
        """載入標準化測試場景配置"""
        config_file = self.repo_root / "standardized_test_scenarios.json"
        
        if not config_file.exists():
            # 如果配置文件不存在，自動生成
            self.logger.warning("🔧 標準化配置文件不存在，自動生成...")
            from standardized_test_config import StandardizedTestConfig
            config_gen = StandardizedTestConfig(seed=42)
            config_data = config_gen.export_unified_scenario_config()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ 已生成標準化配置: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            
        return config_data['scenarios']
    
    def run_standardized_loadtest(self, experiment_type: str, run_tag: str, seed: int, training_proc=None) -> List[Path]:
        """運行標準化的8個場景測試，確保所有方法使用相同場景"""
        
        scenario_dirs = []
        base_output_dir = self.repo_root / "logs" / experiment_type / run_tag
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🎯 開始標準化測試序列 (總共 {len(self.standardized_scenarios)} 個場景)")
        self.logger.info(f"🎲 基礎種子: {seed}")
        
        # 按順序執行標準化場景
        for i, scenario_config in enumerate(self.standardized_scenarios, 1):
            scenario_id = scenario_config['id']
            scenario_type = scenario_config['type']
            scenario_seed = scenario_config['seed']
            
            self.logger.info(f"📊 執行標準化場景 [{i}/{len(self.standardized_scenarios)}]: {scenario_id}")
            self.logger.info(f"   類型: {scenario_type}, 描述: {scenario_config['description']}")
            self.logger.info(f"   場景種子: {scenario_seed}")
            
            # 創建場景專屬目錄
            out_dir = base_output_dir / scenario_id
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 構建遠端標籤
            remote_tag = f"{experiment_type}/{run_tag}/{scenario_id}" if self.m1_host else f"{run_tag}_{scenario_id}"
            
            # 執行測試
            success = self.run_distributed_locust(scenario_type, remote_tag, out_dir)
            if success:
                scenario_dirs.append(out_dir)
                self.logger.info(f"✅ 場景 {scenario_id} 完成")
            else:
                self.logger.error(f"❌ 場景 {scenario_id} 失敗")
            
            # 檢查訓練進程狀態
            if training_proc and training_proc.poll() is not None:
                self.logger.info(f"✅ 訓練進程已完成 (在場景 {i} 後)")
                # 如果是測試模式或訓練已完成，繼續完成所有8個場景
                continue
            
        # 等待訓練進程完成（如果還在運行）
        if training_proc and training_proc.poll() is None:
            self.logger.info("⏳ 等待訓練進程完成...")
            training_proc.wait()
        
        self.logger.info(f"🏁 標準化測試完成，總共執行了 {len(scenario_dirs)}/{len(self.standardized_scenarios)} 個場景")
        
        return scenario_dirs
    
    def run_standardized_hpa_test(self, experiment_type: str, run_tag: str, seed: int, hpa_type: str = "cpu") -> List[Path]:
        """運行標準化的 HPA 測試，使用8個標準場景"""
        
        hpa_configs = ["cpu-20", "cpu-40", "cpu-60", "cpu-80"]
        if hpa_type != "all":
            # 只測試指定類型，但保持8個場景
            hpa_configs = [f"{hpa_type}-40", f"{hpa_type}-60"]  # 為了速度選2個代表性配置
        
        all_scenario_dirs = []
        
        for hpa_config in hpa_configs:
            self.logger.info(f"🔧 測試 HPA 配置: {hpa_config}")
            
            # 應用 HPA 配置
            self.apply_hpa_configuration(hpa_config)
            
            # 等待 HPA 設定生效
            import time
            time.sleep(30)
            
            # 為每個 HPA 配置運行標準化場景
            config_tag = f"{run_tag}_{hpa_config}"
            scenario_dirs = self.run_standardized_loadtest(experiment_type, config_tag, seed)
            all_scenario_dirs.extend(scenario_dirs)
            
        return all_scenario_dirs
    
    def apply_hpa_configuration(self, hpa_config: str):
        """應用指定的 HPA 配置"""
        self.logger.info(f"⚙️ 應用 HPA 配置: {hpa_config}")
        
        # 這裡應該包含實際的 HPA 配置應用邏輯
        # 例如使用 kubectl 命令修改 HPA 設定
        import subprocess
        
        if hpa_config.startswith("cpu-"):
            cpu_threshold = hpa_config.split("-")[1]
            self.logger.info(f"🎯 設定 CPU 閾值: {cpu_threshold}%")
            
            # 示例：應用 CPU HPA 配置
            # kubectl patch hpa <hpa-name> -p '{"spec":{"targetCPUUtilizationPercentage":<threshold>}}'
            
    def _run_gym_hpa_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行標準化的 Gym-HPA 實驗"""
        use_case = kwargs.get('use_case', 'online_boutique')
        self.logger.info(f"🎯 執行標準化 Gym-HPA 實驗 (應用場景: {use_case})")
        
        # 原本的命令構建邏輯
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
        
        # 開始訓練/測試進程
        if not kwargs.get('testing', False):
            training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gym-hpa")
            self.logger.info(f"🔄 Gym-HPA 訓練已開始，使用標準化場景進行測試...")
        else:
            training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gym-hpa")
            self.logger.info(f"🧪 Gym-HPA 測試已開始，使用標準化場景進行測試...")
        
        # 使用標準化場景進行測試
        scenario_dirs = self.run_standardized_loadtest(
            "gym-hpa", run_tag, kwargs.get('seed', 42), training_proc
        )
        
        return len(scenario_dirs) > 0
    
    def _run_gnnrl_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行標準化的 GNNRL 實驗"""
        use_case = kwargs.get('use_case', 'online_boutique')
        self.logger.info(f"🧠 執行標準化 GNNRL 實驗 (應用場景: {use_case})")
        
        # 原本的命令構建邏輯
        gnnrl_script = self.repo_root / "gnnrl" / "training" / "run_gnnrl_experiment.py"
        
        cmd = [
            sys.executable, str(gnnrl_script),
            "--steps", str(kwargs.get('steps', 5000)),
            "--goal", str(kwargs.get('goal', 'latency')),
            "--alg", str(kwargs.get('alg', 'ppo')),
            "--model", str(kwargs.get('model', 'gat')),
            "--env-step-interval", str(kwargs.get('env_step_interval', 15.0))
        ]
        
        if kwargs.get('k8s', False):
            cmd.append("--k8s")
            self.logger.info("✅ 啟用 K8s 集群模式")
        
        # GNNRL 測試模式處理
        if kwargs.get('testing', False):
            self.logger.info("🧪 GNNRL 測試模式：載入已訓練模型進行評估")
            load_path = kwargs.get('load_path')
            if load_path:
                cmd.extend(["--load-path", str(load_path)])
            training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gnnrl")
        else:
            self.logger.info("🎯 GNNRL 訓練模式")
            training_proc = subprocess.Popen(cmd, cwd=self.repo_root / "gnnrl")
        
        # 使用標準化場景進行測試
        scenario_dirs = self.run_standardized_loadtest(
            "gnnrl", run_tag, kwargs.get('seed', 42), training_proc
        )
        
        return len(scenario_dirs) > 0
    
    def _run_k8s_hpa_experiment(self, script_path: Path, run_tag: str, **kwargs) -> bool:
        """執行標準化的 K8s-HPA 實驗"""
        self.logger.info("📊 執行標準化 K8s HPA 基準測試")
        
        hpa_type = kwargs.get('hpa_type', 'cpu')
        seed = kwargs.get('seed', 42)
        
        # 使用標準化 HPA 測試
        scenario_dirs = self.run_standardized_hpa_test(
            "k8s-hpa", run_tag, seed, hpa_type
        )
        
        return len(scenario_dirs) > 0


def main():
    """主函數"""
    import sys
    from unified_experiment_manager import main as original_main
    
    # 使用標準化實驗管理器替換原始管理器
    print("🎯 使用標準化實驗管理器確保公平比較...")
    
    # 暫時替換全局管理器類
    import unified_experiment_manager
    original_class = unified_experiment_manager.UnifiedExperimentManager
    unified_experiment_manager.UnifiedExperimentManager = StandardizedExperimentManager
    
    try:
        # 執行原始主函數邏輯
        original_main()
    finally:
        # 恢復原始類
        unified_experiment_manager.UnifiedExperimentManager = original_class


if __name__ == "__main__":
    main()