#!/bin/bash
set -e

# Helm repositories
helm repo add linkerd https://helm.linkerd.io/stable
helm repo update

# Namespaces
CONTROL_NS="linkerd"
VIZ_NS="linkerd-viz"

# Values for linkerd-viz installation
cat <<'VALUES' > /tmp/linkerd-viz-values.yaml
prometheus:
  enabled: false

# Expose dashboard via NodePort
web:
  service:
    type: NodePort

# Create ServiceMonitor for proxies only in specific namespaces
serviceMonitor:
  enabled: true
  proxy:
    namespaceSelector:
      matchNames:
        - onlineboutique
        - redis
VALUES

# Install linkerd-viz
helm upgrade --install linkerd-viz linkerd/linkerd-viz \
  --namespace "$VIZ_NS" --create-namespace \
  --set namespace="$CONTROL_NS" \
  -f /tmp/linkerd-viz-values.yaml

rm /tmp/linkerd-viz-values.yaml

cat <<EOM
linkerd-viz installed in namespace $VIZ_NS.
Dashboard is exposed as NodePort.
To integrate with kube-prometheus-stack, apply ServiceMonitor manifests such as macK8S/linkerd/metrics.yaml.
EOM
