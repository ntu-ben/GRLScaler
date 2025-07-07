#!/usr/bin/env python3
"""
標準化測試場景配置
==========================================

為確保三種自動縮放方法的公平比較，定義統一的8個測試場景。
所有方法將使用相同的 seed 和場景序列進行測試。
"""

import random
from typing import List, Dict, Tuple

class StandardizedTestConfig:
    """標準化測試配置管理器"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # 定義場景模板
        self.scenario_templates = {
            'offpeak': {
                'pattern': 'offpeak',
                'description': '低負載穩定場景',
                'expected_load': 'low',
                'duration': 15  # 分鐘
            },
            'peak': {
                'pattern': 'peak', 
                'description': '高負載穩定場景',
                'expected_load': 'high',
                'duration': 15
            },
            'rushsale': {
                'pattern': 'rushsale',
                'description': '突發銷售場景', 
                'expected_load': 'burst',
                'duration': 15
            },
            'fluctuating': {
                'pattern': 'fluctuating',
                'description': '波動負載場景',
                'expected_load': 'variable', 
                'duration': 15
            }
        }
        
    def generate_standard_scenarios(self) -> List[Dict]:
        """生成標準的8個測試場景"""
        # 重設隨機種子確保一致性
        random.seed(self.seed)
        
        scenarios = []
        scenario_id = 1
        
        # 生成平衡的8個場景：
        # 2個 offpeak, 2個 peak, 2個 rushsale, 2個 fluctuating
        scenario_types = ['offpeak'] * 2 + ['peak'] * 2 + ['rushsale'] * 2 + ['fluctuating'] * 2
        random.shuffle(scenario_types)
        
        for scenario_type in scenario_types:
            template = self.scenario_templates[scenario_type]
            scenario = {
                'id': f"{scenario_type}_{scenario_id:03d}",
                'type': scenario_type,
                'pattern': template['pattern'],
                'description': template['description'],
                'expected_load': template['expected_load'],
                'duration': template['duration'],
                'seed': self.seed + scenario_id,  # 每個場景有獨特但可重現的seed
                'sequence_order': scenario_id
            }
            scenarios.append(scenario)
            scenario_id += 1
            
        return scenarios
    
    def get_scenario_sequence_file(self) -> str:
        """生成場景序列文件內容"""
        scenarios = self.generate_standard_scenarios()
        
        content = f"""# 標準化測試場景序列 (Seed: {self.seed})
# =========================================
# 為確保三種方法公平比較，所有實驗都使用此場景序列
# 生成時間: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

場景總數: {len(scenarios)}
基礎種子: {self.seed}

測試場景序列:
"""
        
        for i, scenario in enumerate(scenarios, 1):
            content += f"""
{i}. {scenario['id']} 
   類型: {scenario['type']}
   描述: {scenario['description']} 
   預期負載: {scenario['expected_load']}
   持續時間: {scenario['duration']} 分鐘
   場景種子: {scenario['seed']}
"""
        
        content += f"""
使用方式:
1. Gym-HPA: 按序列執行上述8個場景
2. GNNRL: 按序列執行上述8個場景  
3. K8s-HPA: 按序列執行上述8個場景

確保所有方法測試相同的負載模式和條件。
"""
        return content
    
    def export_unified_scenario_config(self) -> Dict:
        """導出統一場景配置供實驗腳本使用"""
        scenarios = self.generate_standard_scenarios()
        
        config = {
            'experiment_config': {
                'seed': self.seed,
                'total_scenarios': len(scenarios),
                'scenario_duration_minutes': 15,
                'description': '標準化三方法比較實驗配置'
            },
            'scenarios': scenarios,
            'scenario_types_distribution': {
                'offpeak': 2,
                'peak': 2, 
                'rushsale': 2,
                'fluctuating': 2
            }
        }
        return config

def main():
    """生成標準化測試配置"""
    print("🔧 生成標準化測試場景配置...")
    
    config = StandardizedTestConfig(seed=42)
    
    # 生成場景序列文件
    sequence_content = config.get_scenario_sequence_file()
    with open('/Users/hopohan/Desktop/k8s/GRLScaler/standardized_scenario_sequence.txt', 'w', encoding='utf-8') as f:
        f.write(sequence_content)
    
    # 生成配置文件
    import json
    unified_config = config.export_unified_scenario_config()
    with open('/Users/hopohan/Desktop/k8s/GRLScaler/standardized_test_scenarios.json', 'w', encoding='utf-8') as f:
        json.dump(unified_config, f, ensure_ascii=False, indent=2)
    
    print("✅ 標準化配置文件已生成:")
    print("  - standardized_scenario_sequence.txt")  
    print("  - standardized_test_scenarios.json")
    
    # 顯示場景摘要
    scenarios = config.generate_standard_scenarios()
    print(f"\n📋 標準測試場景摘要 (Seed: {config.seed}):")
    print("=" * 50)
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['id']} ({scenario['type']}) - {scenario['description']}")
    
    print(f"\n🎯 每種方法將測試相同的 {len(scenarios)} 個場景，確保公平比較")

if __name__ == "__main__":
    main()