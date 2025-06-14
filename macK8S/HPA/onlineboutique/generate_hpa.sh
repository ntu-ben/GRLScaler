#!/usr/bin/env bash
# -----------------------------------------------------------------------------
#  generate_hpa.sh – 依據目前命名空間中的 Deployment 自動產生 HPA（autoscaling/v2）
#  - CPU‑only、Memory‑only、CPU+Memory 三種版本各一份
#  - 預設鎖定第一個應用容器（忽略 linkerd-proxy 等 sidecar）
#  - 若 Deployment 有多個主要容器，請自行修改 CONTAINER_SELECTOR 函式邏輯
# -----------------------------------------------------------------------------
set -euo pipefail

# ===== 使用者可調參數 =====
NAMESPACE="onlineboutique"           # 目標命名空間
EXCLUDE_DEPLOY="loadgenerator"       # 不產生 HPA 的 Deployment

MIN_REPLICAS=1
MAX_REPLICAS=7

CPU_TARGETS=(20 40 60 80)             # CPU AverageUtilization 百分比
MEM_TARGETS=(40 80)                   # Memory AverageUtilization 百分比

# 輸出目錄：與腳本同層
OUTDIR="$(cd "$(dirname "$0")" && pwd)"

# ===== Helper – 取得 Deployment 第一個容器名稱 =====
CONTAINER_SELECTOR() {
  local deploy="$1"
  kubectl get deploy "$deploy" -n "$NAMESPACE" \
    -o jsonpath='{.spec.template.spec.containers[0].name}'
}

# ===== Helper – 將 metrics 區塊與共用欄位寫成 YAML 檔 =====
write_hpa() {
  local name="$1"       # HPA 名稱
  local target="$2"     # Deployment 名稱
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

# ===== 取得所有 Deployment =====
DEPLOYMENTS=$(kubectl get deploy -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' \
  | tr ' ' '\n' | grep -v "^${EXCLUDE_DEPLOY}$" | tr '\n' ' ')

count=0

# -----------------------------------------------------------------------------
# 產生 CPU‑only HPA
# -----------------------------------------------------------------------------
for deploy in ${DEPLOYMENTS}; do
  container_name=$(CONTAINER_SELECTOR "$deploy")
  for cpu in "${CPU_TARGETS[@]}"; do
    metrics=$(cat <<EOF
  - type: ContainerResource
    containerResource:
      name: cpu
      container: ${container_name}
      target:
        type: Utilization
        averageUtilization: ${cpu}
EOF
)
    write_hpa "hpa-${deploy}-cpu-${cpu}" "$deploy" "$metrics"
    ((count++))
  done
done

# -----------------------------------------------------------------------------
# 產生 Memory‑only HPA
# -----------------------------------------------------------------------------
for deploy in ${DEPLOYMENTS}; do
  container_name=$(CONTAINER_SELECTOR "$deploy")
  for mem in "${MEM_TARGETS[@]}"; do
    metrics=$(cat <<EOF
  - type: ContainerResource
    containerResource:
      name: memory
      container: ${container_name}
      target:
        type: Utilization
        averageUtilization: ${mem}
EOF
)
    write_hpa "hpa-${deploy}-mem-${mem}" "$deploy" "$metrics"
    ((count++))
  done
done

# -----------------------------------------------------------------------------
# 產生 CPU + Memory HPA
# -----------------------------------------------------------------------------
for deploy in ${DEPLOYMENTS}; do
  container_name=$(CONTAINER_SELECTOR "$deploy")
  for cpu in "${CPU_TARGETS[@]}"; do
    for mem in "${MEM_TARGETS[@]}"; do

      metrics=$(cat <<EOF
  - type: ContainerResource
    containerResource:
      name: cpu
      container: ${container_name}
      target:
        type: Utilization
        averageUtilization: ${cpu}
  - type: ContainerResource
    containerResource:
      name: memory
      container: ${container_name}
      target:
        type: Utilization
        averageUtilization: ${mem}
EOF
)
      write_hpa "hpa-${deploy}-cpu-${cpu}-mem-${mem}" "$deploy" "$metrics"
      ((count++))
    done
  done
done

echo "✅ 已為 ${NAMESPACE} 中的 $(echo ${DEPLOYMENTS} | wc -w) 個 Deployment（排除 ${EXCLUDE_DEPLOY}）產生 ${count} 份 HPA YAML，輸出位置：${OUTDIR}"

