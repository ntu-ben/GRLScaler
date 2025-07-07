#!/usr/bin/env python3
"""
Redis HPA 配置文件生成器
======================
快速生成所有 Redis HPA 配置組合
"""

from pathlib import Path

# 配置參數
CPU_TARGETS = [20, 40, 60, 80]
MEM_TARGETS = [40, 80]
MIN_REPLICAS = 1
MAX_REPLICAS_MASTER = 5
MAX_REPLICAS_SLAVE = 8

def generate_hpa_yaml(config_name: str, cpu_target: int = None, mem_target: int = None):
    """生成 HPA YAML 配置"""
    
    yaml_content = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hpa-redis-master-{config_name}
  namespace: redis
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: redis-master
  minReplicas: {MIN_REPLICAS}
  maxReplicas: {MAX_REPLICAS_MASTER}
  metrics:"""

    if cpu_target:
        yaml_content += f"""
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {cpu_target}"""
    
    if mem_target:
        yaml_content += f"""
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {mem_target}"""

    yaml_content += f"""
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hpa-redis-slave-{config_name}
  namespace: redis
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: redis-slave
  minReplicas: {MIN_REPLICAS}
  maxReplicas: {MAX_REPLICAS_SLAVE}
  metrics:"""

    if cpu_target:
        yaml_content += f"""
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {cpu_target}"""
    
    if mem_target:
        yaml_content += f"""
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: {mem_target}"""

    return yaml_content

def main():
    base_dir = Path(__file__).parent
    
    print("🔧 生成 Redis HPA 配置文件...")
    
    # 1. 純 CPU 配置
    for cpu in CPU_TARGETS:
        config_name = f"cpu-{cpu}"
        config_dir = base_dir / config_name
        config_dir.mkdir(exist_ok=True)
        
        yaml_content = generate_hpa_yaml(config_name, cpu_target=cpu)
        
        with open(config_dir / f"hpa-redis-{config_name}.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ 生成 {config_name}")
    
    # 2. 純 Memory 配置
    for mem in MEM_TARGETS:
        config_name = f"mem-{mem}"
        config_dir = base_dir / config_name
        config_dir.mkdir(exist_ok=True)
        
        yaml_content = generate_hpa_yaml(config_name, mem_target=mem)
        
        with open(config_dir / f"hpa-redis-{config_name}.yaml", 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ 生成 {config_name}")
    
    # 3. CPU + Memory 混合配置
    for cpu in CPU_TARGETS:
        for mem in MEM_TARGETS:
            config_name = f"cpu-{cpu}-mem-{mem}"
            config_dir = base_dir / config_name
            config_dir.mkdir(exist_ok=True)
            
            yaml_content = generate_hpa_yaml(config_name, cpu_target=cpu, mem_target=mem)
            
            with open(config_dir / f"hpa-redis-{config_name}.yaml", 'w') as f:
                f.write(yaml_content)
            
            print(f"✅ 生成 {config_name}")
    
    print(f"\n🎉 總共生成了 {len(CPU_TARGETS) + len(MEM_TARGETS) + len(CPU_TARGETS) * len(MEM_TARGETS)} 個 Redis HPA 配置")
    print("\n📋 配置清單:")
    print("CPU 配置:", [f"cpu-{cpu}" for cpu in CPU_TARGETS])
    print("Memory 配置:", [f"mem-{mem}" for mem in MEM_TARGETS])
    print("混合配置:", [f"cpu-{cpu}-mem-{mem}" for cpu in CPU_TARGETS for mem in MEM_TARGETS])

if __name__ == "__main__":
    main()