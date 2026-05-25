#!/usr/bin/env bash
# Local dev driver for the ml-container-platform.
# Usage: ./ops.sh <subcommand>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
MLFLOW_BIN="$VENV/bin/mlflow"

MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"
MLFLOW_BACKEND="sqlite:///${REPO_ROOT}/mlruns/mlflow.db"
MLFLOW_ARTIFACTS="${REPO_ROOT}/mlruns/artifacts"
MLFLOW_PIDFILE="${REPO_ROOT}/mlruns/mlflow.pid"
MLFLOW_LOG="${REPO_ROOT}/mlruns/mlflow.log"

TRAIN_IMAGE="${TRAIN_IMAGE:-ml-platform/train:latest}"
SERVE_IMAGE="${SERVE_IMAGE:-ml-platform/serve:latest}"
SERVE_CONTAINER="${SERVE_CONTAINER:-ml-platform-serve}"
SERVE_PORT="${SERVE_PORT:-8000}"

KUBECONFIG="${KUBECONFIG:-$HOME/k3s.yaml}"
K8S_NS="${K8S_NS:-ml-platform}"
MONITORING_NS="${MONITORING_NS:-monitoring}"
MONITORING_CHART="$REPO_ROOT/helm/monitoring"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

log()  { printf '\033[1;34m[ops]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[ops]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[ops]\033[0m %s\n' "$*" >&2; exit 1; }

ensure_venv() {
  if [[ ! -x "$PY" ]]; then
    log "creating venv at $VENV"
    python3.11 -m venv "$VENV" 2>/dev/null || python3 -m venv "$VENV"
  fi
  "$PIP" install --quiet --upgrade pip
}

ensure_train_deps() {
  ensure_venv
  if ! "$PY" -c "import torch, torchvision, mlflow, click" 2>/dev/null; then
    log "installing train deps (this can take a few minutes)"
    "$PIP" install -r "$REPO_ROOT/requirements-train.txt"
  fi
}

ensure_serve_deps() {
  ensure_venv
  if ! "$PY" -c "import fastapi, uvicorn, mlflow, PIL, prometheus_client" 2>/dev/null; then
    log "installing serve deps"
    "$PIP" install -r "$REPO_ROOT/requirements-serve.txt"
  fi
}

mlflow_running() {
  curl -fsS "$MLFLOW_URI/health" >/dev/null 2>&1
}

start_mlflow() {
  ensure_train_deps
  mkdir -p "$MLFLOW_ARTIFACTS"
  if mlflow_running; then
    log "mlflow already up at $MLFLOW_URI"
    return 0
  fi
  log "starting mlflow server on $MLFLOW_URI"
  # Use proxied artifacts (mlflow-artifacts:/) so registered models carry
  # container-portable URIs instead of host-absolute filesystem paths.
  nohup "$MLFLOW_BIN" server \
      --backend-store-uri "$MLFLOW_BACKEND" \
      --default-artifact-root "mlflow-artifacts:/" \
      --artifacts-destination "$MLFLOW_ARTIFACTS" \
      --serve-artifacts \
      --host "$MLFLOW_HOST" \
      --port "$MLFLOW_PORT" \
      >"$MLFLOW_LOG" 2>&1 &
  echo $! >"$MLFLOW_PIDFILE"
  for _ in $(seq 1 60); do
    if mlflow_running; then
      log "mlflow ready (pid=$(cat "$MLFLOW_PIDFILE"))"
      return 0
    fi
    sleep 1
  done
  warn "last 20 lines of $MLFLOW_LOG:"
  tail -n 20 "$MLFLOW_LOG" >&2 || true
  die "mlflow did not become ready on $MLFLOW_URI"
}

stop_mlflow() {
  if [[ -f "$MLFLOW_PIDFILE" ]]; then
    local pid
    pid="$(cat "$MLFLOW_PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      log "stopping mlflow (pid=$pid)"
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$MLFLOW_PIDFILE"
  fi
  # macOS `pkill -f` is unreliable here; instead, kill whatever holds port $MLFLOW_PORT.
  local pids
  pids="$(lsof -nP -iTCP:"$MLFLOW_PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    log "killing port-$MLFLOW_PORT listeners: $(echo "$pids" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -nP -iTCP:"$MLFLOW_PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

cmd_train() {
  start_mlflow
  log "running quick training (--epochs 1 --subset 1000)"
  MLFLOW_TRACKING_URI="$MLFLOW_URI" "$PY" "$REPO_ROOT/src/train.py" \
      --epochs 1 --subset 1000 --tracking-uri "$MLFLOW_URI" "$@"
}

cmd_train_full() {
  start_mlflow
  log "running full training (--epochs 10, full dataset)"
  MLFLOW_TRACKING_URI="$MLFLOW_URI" "$PY" "$REPO_ROOT/src/train.py" \
      --epochs 10 --tracking-uri "$MLFLOW_URI" "$@"
}

cmd_promote() {
  ensure_train_deps
  start_mlflow
  MLFLOW_TRACKING_URI="$MLFLOW_URI" "$PY" "$REPO_ROOT/scripts/promote.py"
}

cmd_build() {
  log "building $TRAIN_IMAGE"
  docker build -f "$REPO_ROOT/docker/train.Dockerfile" -t "$TRAIN_IMAGE" "$REPO_ROOT"
  log "building $SERVE_IMAGE"
  docker build -f "$REPO_ROOT/docker/serve.Dockerfile" -t "$SERVE_IMAGE" "$REPO_ROOT"
  log "image sizes:"
  docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' \
    | grep -E "($(echo "$TRAIN_IMAGE" | sed 's|/|\\/|g')|$(echo "$SERVE_IMAGE" | sed 's|/|\\/|g'))" || true
}

cmd_run() {
  start_mlflow
  if docker ps -a --format '{{.Names}}' | grep -qx "$SERVE_CONTAINER"; then
    log "removing existing container $SERVE_CONTAINER"
    docker rm -f "$SERVE_CONTAINER" >/dev/null
  fi
  log "starting $SERVE_CONTAINER on :$SERVE_PORT"
  # Mirror the host mlruns path inside the container so any registered model
  # whose artifact URI is an absolute host path (pre-proxy runs) still resolves.
  docker run -d --name "$SERVE_CONTAINER" \
      -p "$SERVE_PORT:8000" \
      --add-host=host.docker.internal:host-gateway \
      -e MLFLOW_TRACKING_URI="http://host.docker.internal:${MLFLOW_PORT}" \
      -e MODEL_NAME=cifar10-mobilenet \
      -e MODEL_STAGE=Production \
      -v "$REPO_ROOT/mlruns:$REPO_ROOT/mlruns" \
      "$SERVE_IMAGE"
  log "waiting for /health…"
  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${SERVE_PORT}/health" >/dev/null 2>&1; then
      log "serve ready at http://127.0.0.1:${SERVE_PORT}"
      return 0
    fi
    sleep 1
  done
  warn "serve container logs:"
  docker logs "$SERVE_CONTAINER" >&2 || true
  die "serve did not become ready"
}

cmd_smoke_test() {
  ensure_serve_deps
  local img="$REPO_ROOT/tests/test_image.jpg"
  if [[ ! -f "$img" ]]; then
    log "generating $img"
    "$PY" "$REPO_ROOT/tests/gen_test_image.py" --out "$img"
  fi
  log "GET /health"
  curl -fsS "http://127.0.0.1:${SERVE_PORT}/health" && echo
  log "POST /predict"
  local payload
  payload="$("$PY" -c "import base64,json,sys; print(json.dumps({'image_b64': base64.b64encode(open(sys.argv[1],'rb').read()).decode()}))" "$img")"
  curl -fsS -X POST "http://127.0.0.1:${SERVE_PORT}/predict" \
      -H "Content-Type: application/json" \
      -d "$payload" && echo
}

cmd_monitoring_deploy() {
  log "updating helm dependencies for monitoring chart"
  helm dependency update "$MONITORING_CHART"
  log "deploying monitoring stack to namespace $MONITORING_NS"
  KUBECONFIG="$KUBECONFIG" helm upgrade --install monitoring "$MONITORING_CHART" \
    -f "$MONITORING_CHART/values.yaml" \
    -f "$MONITORING_CHART/values-homelab.yaml" \
    --namespace "$MONITORING_NS" --create-namespace \
    --timeout 5m \
    "$@"
  log "monitoring stack deployed — use './ops.sh monitoring-status' to check pods"
}

cmd_monitoring_status() {
  KUBECONFIG="$KUBECONFIG" kubectl get pods -n "$MONITORING_NS"
}

cmd_monitoring_open() {
  log "port-forwarding Grafana to http://127.0.0.1:${GRAFANA_PORT}  (admin/admin)"
  KUBECONFIG="$KUBECONFIG" kubectl port-forward \
    svc/monitoring-grafana "$GRAFANA_PORT:80" \
    -n "$MONITORING_NS"
}

cmd_canary_loadtest() {
  local loop=false
  [[ "${1:-}" == "--loop" ]] && loop=true

  local node_ip nodeport
  node_ip=$(KUBECONFIG="$KUBECONFIG" kubectl get nodes \
    -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
  nodeport=$(KUBECONFIG="$KUBECONFIG" kubectl -n "$K8S_NS" \
    get svc ml-platform-canary-gateway \
    -o jsonpath='{.spec.ports[0].nodePort}')
  local url="http://${node_ip}:${nodeport}/predict"
  log "canary gateway: $url"

  local img="$REPO_ROOT/tests/test_image.jpg"
  if [[ ! -f "$img" ]]; then
    log "generating test image with python3"
    python3 -c "
import struct, zlib
def png_chunk(t, d): c = zlib.crc32(t+d)&0xffffffff; return struct.pack('>I',len(d))+t+d+struct.pack('>I',c)
raw = (b'\x00'+b'\xff\x00\x00'*32)*32
data = b'\x89PNG\r\n\x1a\n'
data += png_chunk(b'IHDR', struct.pack('>IIBBBBB',32,32,8,2,0,0,0))
data += png_chunk(b'IDAT', zlib.compress(raw))
data += png_chunk(b'IEND', b'')
open('$img','wb').write(data)
"
  fi
  local img_b64 payload
  img_b64=$(base64 < "$img" | tr -d '\n')
  payload="{\"image_b64\": \"$img_b64\"}"

  local batch=200
  $loop && batch=50 && log "continuous mode (50 req every 10s) — Ctrl+C to stop" \
         || log "sending $batch /predict requests — watch Grafana Canary Split panel…"

  while true; do
    local ok=0 fail=0
    for i in $(seq 1 "$batch"); do
      if curl -fsS -o /dev/null \
          -X POST "$url" \
          -H "Content-Type: application/json" \
          -d "$payload"; then
        ok=$((ok + 1))
      else
        fail=$((fail + 1))
      fi
      printf '.'
    done
    echo " ${ok} ok / ${fail} fail"
    $loop || break
    sleep 10
  done
  log "done — check Grafana Canary Split panel for 80/20 split"
}

cmd_clean() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$SERVE_CONTAINER"; then
    log "removing $SERVE_CONTAINER"
    docker rm -f "$SERVE_CONTAINER" >/dev/null
  fi
  stop_mlflow
  log "clean done"
}

usage() {
  cat <<USAGE
Usage: ./ops.sh <command>

  train         Start MLflow + run a quick training (--epochs 1 --subset 1000).
  train-full    Start MLflow + run full training (--epochs 10).
  promote       Promote latest registered model version to Production.
  build         Build train and serve Docker images.
  run           Start the serve container on :${SERVE_PORT} (requires 'build').
  smoke-test    curl /health then /predict with tests/test_image.jpg.
  clean              Stop the serve container and the MLflow server.
  monitoring-deploy  Install / upgrade Prometheus + Grafana stack via Helm.
  monitoring-status  Show pod status in the monitoring namespace.
  monitoring-open    kubectl port-forward Grafana to :${GRAFANA_PORT} (admin/admin).
  canary-loadtest    Send 200 /predict requests through the canary gateway; shows 80/20 split in Grafana.
USAGE
}

main() {
  local cmd="${1:-}"
  [[ -n "$cmd" ]] || { usage; exit 1; }
  shift || true
  case "$cmd" in
    train)          cmd_train "$@" ;;
    train-full)     cmd_train_full "$@" ;;
    promote)        cmd_promote "$@" ;;
    build)          cmd_build "$@" ;;
    run)            cmd_run "$@" ;;
    smoke-test)          cmd_smoke_test "$@" ;;
    clean)               cmd_clean "$@" ;;
    monitoring-deploy)   cmd_monitoring_deploy "$@" ;;
    monitoring-status)   cmd_monitoring_status "$@" ;;
    monitoring-open)     cmd_monitoring_open "$@" ;;
    canary-loadtest)     cmd_canary_loadtest "$@" ;;
    -h|--help|help) usage ;;
    *) usage; die "unknown command: $cmd" ;;
  esac
}

main "$@"
