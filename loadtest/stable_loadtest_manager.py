#!/usr/bin/env python3
"""
穩定Loadtest管理器
=================

管理穩定的壓力測試配置，確保：
1. 限定最高RPS，避免系統過載
2. 失敗時維持測試繼續進行
3. 提供一致的測試基準
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

class StableLoadTestManager:
    """穩定負載測試管理器"""
    
    def __init__(self, max_rps: Optional[int] = None, timeout: int = 30):
        self.max_rps = max_rps
        self.timeout = timeout
        self.loadtest_dir = Path(__file__).parent
        
        # 預設最高RPS限制
        self.default_max_rps = {
            'offpeak': 50,
            'peak': 200,  # 降低peak的預設值，避免過載
            'rushsale': 400,  # 降低rushsale的預設值
            'fluctuating': 150
        }
    
    def get_stable_script_path(self, scenario: str, environment: str = 'onlineboutique') -> Path:
        """獲取穩定loadtest腳本路徑"""
        stable_script = self.loadtest_dir / environment / f"locust_stable_{scenario}.py"
        
        if stable_script.exists():
            return stable_script
        else:
            # 如果穩定版本不存在，使用原版本
            original_script = self.loadtest_dir / environment / f"locust_{scenario}.py"
            if original_script.exists():
                print(f"⚠️  穩定版本不存在，使用原版本: {original_script}")
                return original_script
            else:
                raise FileNotFoundError(f"找不到loadtest腳本: {scenario}")
    
    def prepare_environment_variables(self, scenario: str, run_time: str = "15m") -> Dict[str, str]:
        """準備環境變數"""
        env_vars = os.environ.copy()
        
        # 設置運行時間
        env_vars['LOCUST_RUN_TIME'] = run_time
        
        # 設置最高RPS限制
        if self.max_rps:
            env_vars['LOCUST_MAX_RPS'] = str(self.max_rps)
        else:
            # 使用預設限制
            default_rps = self.default_max_rps.get(scenario, 100)
            env_vars['LOCUST_MAX_RPS'] = str(default_rps)
        
        # 設置超時時間
        env_vars['LOCUST_TIMEOUT'] = str(self.timeout)
        
        return env_vars
    
    def run_stable_loadtest(self, 
                           scenario: str, 
                           target_host: str,
                           environment: str = 'onlineboutique',
                           run_time: str = "15m",
                           output_dir: Optional[Path] = None) -> bool:
        """執行穩定的負載測試"""
        
        try:
            # 獲取腳本路徑
            script_path = self.get_stable_script_path(scenario, environment)
            
            # 準備環境變數
            env_vars = self.prepare_environment_variables(scenario, run_time)
            
            # 準備輸出目錄
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                csv_prefix = str(output_dir / f"{scenario}")
            else:
                csv_prefix = f"./logs/{environment}_{scenario}"
            
            # 構建Locust命令
            max_rps = env_vars.get('LOCUST_MAX_RPS', '100')
            
            cmd = [
                'locust',
                '-f', str(script_path),
                '--host', target_host,
                '--headless',
                '--run-time', run_time,
                '--csv', csv_prefix,
                '--print-stats',
                '--only-summary'
            ]
            
            print(f"🚀 啟動穩定loadtest:")
            print(f"   📝 腳本: {script_path.name}")
            print(f"   📊 場景: {scenario}")
            print(f"   🎯 目標: {target_host}")
            print(f"   📈 最高RPS: {max_rps}")
            print(f"   ⏱️  運行時間: {run_time}")
            print(f"   📁 輸出: {csv_prefix}")
            print()
            
            # 執行命令
            process = subprocess.Popen(
                cmd,
                env=env_vars,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=self.loadtest_dir
            )
            
            # 即時顯示輸出
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # 等待程序完成
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✅ Loadtest {scenario} 完成成功")
                return True
            else:
                print(f"❌ Loadtest {scenario} 完成但有錯誤 (返回碼: {return_code})")
                return False
                
        except Exception as e:
            print(f"❌ Loadtest執行失敗: {e}")
            return False
    
    def run_scenario_suite(self, 
                          scenarios: list,
                          target_host: str,
                          environment: str = 'onlineboutique',
                          run_time: str = "15m",
                          output_base_dir: Optional[Path] = None) -> Dict[str, bool]:
        """執行一套場景測試"""
        
        results = {}
        
        for scenario in scenarios:
            print(f"\n{'='*60}")
            print(f"🔄 執行場景: {scenario}")
            print(f"{'='*60}")
            
            # 為每個場景創建獨立的輸出目錄
            if output_base_dir:
                scenario_output_dir = output_base_dir / scenario
            else:
                scenario_output_dir = None
            
            # 執行測試
            success = self.run_stable_loadtest(
                scenario=scenario,
                target_host=target_host,
                environment=environment,
                run_time=run_time,
                output_dir=scenario_output_dir
            )
            
            results[scenario] = success
            
            # 場景間短暫休息
            if scenario != scenarios[-1]:  # 不是最後一個場景
                print(f"⏳ 場景間休息30秒...")
                time.sleep(30)
        
        return results

def main():
    """命令行界面"""
    import argparse
    
    parser = argparse.ArgumentParser(description='穩定Loadtest管理器')
    parser.add_argument('scenario', help='測試場景')
    parser.add_argument('--host', default='http://k8s.orb.local', help='目標主機')
    parser.add_argument('--max-rps', type=int, help='最高RPS限制')
    parser.add_argument('--timeout', type=int, default=30, help='請求超時時間')
    parser.add_argument('--run-time', default='15m', help='運行時間')
    parser.add_argument('--environment', default='onlineboutique', help='環境類型')
    parser.add_argument('--output-dir', type=Path, help='輸出目錄')
    
    args = parser.parse_args()
    
    manager = StableLoadTestManager(max_rps=args.max_rps, timeout=args.timeout)
    
    success = manager.run_stable_loadtest(
        scenario=args.scenario,
        target_host=args.host,
        environment=args.environment,
        run_time=args.run_time,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n✅ 測試完成成功")
    else:
        print("\n❌ 測試完成但有問題")
        exit(1)

if __name__ == '__main__':
    main()