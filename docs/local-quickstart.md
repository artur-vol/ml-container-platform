# Local Quickstart

## Prerequisites

- Python 3.11
- Docker (with BuildKit enabled)
- macOS or Linux

## First-time setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt -r requirements-serve.txt
```

## Full local run

```bash
# 1. Train (quick: 1 epoch, 1000 samples — ~30 s on M1)
./ops.sh train

# 2. Promote the registered model to Production
./ops.sh promote

# 3. Build Docker images
./ops.sh build

# 4. Start the serve container on :8000
./ops.sh run

# 5. Smoke-test /health and /predict
./ops.sh smoke-test

# 6. Tear everything down
./ops.sh clean
```

For a full training run (10 epochs, full CIFAR-10 dataset), replace step 1 with:
```bash
./ops.sh train-full
```

## Cross-arch builds (M1 → homelab / GKE)

Both the homelab (i5, Ubuntu) and GKE use `linux/amd64`. Build with:

```bash
docker buildx build --platform linux/amd64 -f docker/train.Dockerfile -t ml-platform/train:latest .
docker buildx build --platform linux/amd64 -f docker/serve.Dockerfile -t ml-platform/serve:latest .
```

## Ports and endpoints

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5000 |
| Inference API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| Prometheus metrics | http://localhost:8000/metrics |

## Troubleshooting

**Port 5000 already in use on macOS** — AirPlay receiver binds to 5000. Disable it in System Settings → General → AirDrop & Handoff, or run MLflow on a different port: `MLFLOW_PORT=5001 ./ops.sh train`.

**Serve container can't reach MLflow** — ensure MLflow is running (`curl http://localhost:5000/health`) before `./ops.sh run`. The container reaches the host via `host.docker.internal`.

**Model not found on startup** — run `./ops.sh promote` before `./ops.sh run` to transition a model version to Production.
