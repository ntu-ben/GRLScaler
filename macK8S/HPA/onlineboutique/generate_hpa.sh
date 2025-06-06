#!/usr/bin/env bash
set -euo pipefail

# ----- 可調整變數 -----
NAMESPACE="onlineboutique"
EXCLUDE_DEPLOY="loadgenerator"

MIN_REPLICAS=1
MAX_REPLICAS=10

CPU_TARGETS=(20 40 60 80)
MEM_TARGETS=(40 80)

OUTDIR="$(cd "$(dirname "$0")" && pwd)"

# 取得所有 Deployment 名稱（空格分隔），過濾掉 EXCLUDE_DEPLOY
DEPLOYMENTS=$(kubectl get deploy -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' \
  | tr ' ' '\n' \
  | grep -v "^${EXCLUDE_DEPLOY}$" \
  | tr '\n' ' ')

# helper：寫出一個 HPA YAML
write_hpa() {
  local name="$1"
  local target="$2"
  local metrics_block="$3"

  cat > "${OUTDIR}/${name}.yaml" <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${name}
  namespace: ${NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ${target}
  minReplicas: ${MIN_REPLICAS}
  maxReplicas: ${MAX_REPLICAS}
  metrics:
${metrics_block}
EOF
}

count=0

# 產生 CPU-only HPA
for deploy in ${DEPLOYMENTS}; do
  for cpu in "${CPU_TARGETS[@]}"; do
    metrics="  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: ${cpu}"
    write_hpa "hpa-${deploy}-cpu-${cpu}" "${deploy}" "${metrics}"
    ((count++))
  done
done

# 產生 MEM-only HPA
for deploy in ${DEPLOYMENTS}; do
  for mem in "${MEM_TARGETS[@]}"; do
    metrics="  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: ${mem}"
    write_hpa "hpa-${deploy}-mem-${mem}" "${deploy}" "${metrics}"
    ((count++))
  done
done

# 產生 CPU+MEM HPA
for deploy in ${DEPLOYMENTS}; do
  for cpu in "${CPU_TARGETS[@]}"; do
    for mem in "${MEM_TARGETS[@]}"; do
      metrics="  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: ${cpu}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: ${mem}"
      write_hpa "hpa-${deploy}-cpu-${cpu}-mem-${mem}" "${deploy}" "${metrics}"
      ((count++))
    done
  done
done

echo "✅ 已為 ${NAMESPACE} 中的 $(echo ${DEPLOYMENTS} | wc -w) 個 Deployment（排除 ${EXCLUDE_DEPLOY}）產生 ${count} 支 HPA 定義，檔案在 ${OUTDIR}/"

