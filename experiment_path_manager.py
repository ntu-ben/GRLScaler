#!/usr/bin/env python3
"""
實驗路徑管理器 (Experiment Path Manager)
=====================================

統一管理所有實驗結果的輸出路徑，確保路徑結構的一致性。
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

class ExperimentPathManager:
    """實驗路徑管理器"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 創建標準子目錄
        self.comparison_dir = self.base_dir / "comparison_reports"
        self.archive_dir = self.base_dir / "archive"
        
        self.comparison_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # 向後兼容：保持 logs 目錄的符號連接
        self.legacy_logs_dir = Path("logs")
        if not self.legacy_logs_dir.exists():
            self.legacy_logs_dir.mkdir(exist_ok=True)
    
    def create_experiment_path(self, experiment_type: str, algorithm: str = "ppo", 
                             model: str = "default", goal: str = "latency", 
                             steps: int = 5000, timestamp: Optional[str] = None) -> Path:
        """創建實驗路徑"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 統一實驗 ID 格式
        exp_id = f"{timestamp}_{experiment_type}_{algorithm}_{model}_{goal}_{steps}"
        
        # 新的統一路徑
        exp_path = self.base_dir / exp_id
        exp_path.mkdir(exist_ok=True)
        
        # 創建標準子目錄
        subdirs = [
            "loadtest_scenarios",
            "performance_charts", 
            "models",
            "models/checkpoints"
        ]
        
        for subdir in subdirs:
            (exp_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # 向後兼容：在 logs 目錄創建符號連接
        legacy_path = self.legacy_logs_dir / experiment_type / exp_id
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果符號連接不存在，創建它
        try:
            if not legacy_path.exists():
                legacy_path.symlink_to(exp_path.resolve())
        except OSError:
            # 如果符號連接失敗（比如在 Windows 上），復製結構
            pass
        
        return exp_path
    
    def get_legacy_path(self, experiment_type: str, run_tag: str) -> Path:
        """獲取舊版路徑（向後兼容）"""
        return self.legacy_logs_dir / experiment_type / run_tag
    
    def migrate_legacy_results(self):
        """遷移舊版實驗結果到新結構"""
        if not self.legacy_logs_dir.exists():
            return
        
        migrated_count = 0
        for exp_type_dir in self.legacy_logs_dir.iterdir():
            if not exp_type_dir.is_dir():
                continue
                
            exp_type = exp_type_dir.name
            for run_dir in exp_type_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                # 嘗試解析為新格式
                run_tag = run_dir.name
                if self._is_new_format(run_tag):
                    continue  # 已經是新格式
                
                # 遷移到新格式
                new_path = self._migrate_single_experiment(exp_type, run_dir)
                if new_path:
                    migrated_count += 1
        
        print(f"Migrated {migrated_count} legacy experiments to new structure")
    
    def _is_new_format(self, run_tag: str) -> bool:
        """檢查是否為新格式的 run tag"""
        parts = run_tag.split('_')
        return len(parts) >= 6 and parts[0].isdigit() and len(parts[0]) == 8
    
    def _migrate_single_experiment(self, exp_type: str, old_path: Path) -> Optional[Path]:
        """遷移單個實驗"""
        try:
            # 生成新的實驗 ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_exp_id = f"{timestamp}_{exp_type}_migrated_unknown_unknown_0"
            
            new_path = self.base_dir / new_exp_id
            new_path.mkdir(exist_ok=True)
            
            # 複製檔案結構
            import shutil
            for item in old_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, new_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, new_path / item.name)
            
            # 創建遷移信息
            migration_info = {
                "migrated_from": str(old_path),
                "migration_time": datetime.now().isoformat(),
                "original_experiment_type": exp_type
            }
            
            import json
            with open(new_path / "migration_info.json", 'w') as f:
                json.dump(migration_info, f, indent=2)
            
            return new_path
            
        except Exception as e:
            print(f"Failed to migrate {old_path}: {e}")
            return None
    
    def list_all_experiments(self) -> Dict[str, Path]:
        """列出所有實驗"""
        experiments = {}
        
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.') and \
               exp_dir.name not in ['comparison_reports', 'archive']:
                experiments[exp_dir.name] = exp_dir
        
        return experiments
    
    def archive_experiment(self, exp_id: str) -> bool:
        """歸檔實驗"""
        exp_path = self.base_dir / exp_id
        if not exp_path.exists():
            return False
        
        # 創建歸檔目錄結構 (YYYY/MM)
        now = datetime.now()
        archive_path = self.archive_dir / str(now.year) / f"{now.month:02d}"
        archive_path.mkdir(parents=True, exist_ok=True)
        
        # 移動實驗目錄
        import shutil
        target_path = archive_path / exp_id
        shutil.move(str(exp_path), str(target_path))
        
        return True

# 全域實例
_path_manager = None

def get_path_manager() -> ExperimentPathManager:
    """獲取全域路徑管理器實例"""
    global _path_manager
    if _path_manager is None:
        _path_manager = ExperimentPathManager()
    return _path_manager

def create_experiment_directory(experiment_type: str, algorithm: str = "ppo", 
                              model: str = "default", goal: str = "latency", 
                              steps: int = 5000) -> Path:
    """便利函數：創建實驗目錄"""
    return get_path_manager().create_experiment_path(
        experiment_type, algorithm, model, goal, steps
    )