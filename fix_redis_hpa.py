#!/usr/bin/env python3

import os
import glob
import re

def fix_hpa_config(file_path):
    """修正HPA配置文件，將Resource改為ContainerResource並添加container: redis"""
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # CPU資源修正
    cpu_pattern = r'  - type: Resource\n    resource:\n      name: cpu\n      target:\n        type: Utilization\n        averageUtilization: (\d+)'
    cpu_replacement = r'  - type: ContainerResource\n    containerResource:\n      name: cpu\n      container: redis\n      target:\n        type: Utilization\n        averageUtilization: \1'
    content = re.sub(cpu_pattern, cpu_replacement, content)
    
    # Memory資源修正
    mem_pattern = r'  - type: Resource\n    resource:\n      name: memory\n      target:\n        type: Utilization\n        averageUtilization: (\d+)'
    mem_replacement = r'  - type: ContainerResource\n    containerResource:\n      name: memory\n      container: redis\n      target:\n        type: Utilization\n        averageUtilization: \1'
    content = re.sub(mem_pattern, mem_replacement, content)
    
    # 寫回文件
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ 修正完成: {file_path}")

def main():
    # 尋找所有需要修正的Redis HPA配置文件
    redis_hpa_dir = "/Users/hopohan/Desktop/k8s/GRLScaler/macK8S/HPA/redis"
    
    for root, dirs, files in os.walk(redis_hpa_dir):
        for file in files:
            if file.endswith('.yaml') and 'hpa-redis' in file:
                file_path = os.path.join(root, file)
                
                # 檢查是否包含需要修正的Resource類型
                with open(file_path, 'r') as f:
                    content = f.read()
                    if 'type: Resource' in content:
                        fix_hpa_config(file_path)

if __name__ == "__main__":
    main()