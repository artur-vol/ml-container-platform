#!/usr/bin/env bash
# Validate canary health then promote or rollback.
# Run from the repository root.
set -euo pipefail

export KUBECONFIG=/home/arturvol/k3s.yaml
NAMESPACE=ml-platform
STABLE_RELEASE=ml-platform
CANARY_RELEASE=ml-platform-canary

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== canary health check ==="
if python3 "$SCRIPT_DIR/canary_check.py"; then
    echo "=== promoting canary to stable ==="

    CANARY_TAG=$(kubectl get deployment "$CANARY_RELEASE" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' | cut -d: -f2)
    echo "canary image tag: $CANARY_TAG"

    helm upgrade "$STABLE_RELEASE" "$REPO_ROOT/helm/ml-platform" \
        -f "$REPO_ROOT/helm/ml-platform/values.yaml" \
        -f "$REPO_ROOT/helm/ml-platform/values-homelab.yaml" \
        --set image.tag="$CANARY_TAG" \
        --namespace "$NAMESPACE" \
        --wait --timeout 120s

    helm uninstall "$CANARY_RELEASE" -n "$NAMESPACE"
    echo "=== canary promoted, stable updated to $CANARY_TAG ==="
else
    echo "=== canary unhealthy — rolling back ==="
    helm uninstall "$CANARY_RELEASE" -n "$NAMESPACE"
    echo "=== canary removed, stable unchanged ==="
    exit 1
fi
