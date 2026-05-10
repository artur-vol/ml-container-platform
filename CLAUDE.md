# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This repository is the codebase for a bachelor's thesis (БКР, НУ «Львівська політехніка», captioned April 2026): **"Проєктування та розгортання контейнерної інфраструктури для систем машинного навчання"**. The deliverable is a containerised ML platform that demonstrates the full lifecycle (train → register → deploy → monitor → A/B). The ML model itself (CIFAR-10 classifier on MobileNetV2/ResNet-18 in PyTorch) is intentionally a *demonstrator* — the thesis is graded on the **infrastructure**, not on model accuracy. Keep that priority in mind when making trade-offs.

The authoritative scope, definition-of-done, and target stack live in `docs/thesis/`:
- `thesis-scope-v1.0.md` — in/out of scope, DoD criteria 1–8, weekly plan, hardware constraints. Treat its scope table as binding; flag (don't silently expand) work that drifts out of scope.
- `thesis-draft-v2.0.md` — chapter outline of the written thesis.
- `thesis-blueprint.md` — literature-review prose for Chapter 1.

These docs are written in Ukrainian. Match the language of any new thesis-doc edits to the surrounding text; code, identifiers, and commit messages should stay in English.

## Repository state

Core ML pipeline is implemented; Helm chart, CI workflow, and `tests/smoke_test.py` are still pending. Current layout:

```
docker/        # train.Dockerfile, serve.Dockerfile (multi-stage, non-root, python:3.11-slim)
src/           # train.py, serve.py
scripts/       # promote.py (MLflow model promotion)
tests/         # gen_test_image.py (saves a CIFAR-10 sample to tests/test_image.jpg)
ops.sh         # local dev driver — see "Development commands" below
mlruns/        # MLflow SQLite backend + artifact store (gitignored)
data/          # CIFAR-10 download cache (gitignored)
helm/          # NOT YET CREATED
.github/       # NOT YET CREATED
```

A Python 3.11 venv exists at `.venv/`; activate with `source .venv/bin/activate`. `.gitignore` excludes `.venv`, `data/`, `mlruns/`, `*.pt`/`*.pth`, and `tests/test_image.jpg`, but force-includes `docs/thesis/`.

## Development commands

All local dev flows go through `ops.sh`. Pass extra CLI flags after the subcommand — they are forwarded to the underlying Python script.

```bash
./ops.sh train          # start MLflow + quick run (--epochs 1 --subset 1000)
./ops.sh train-full     # start MLflow + full run (--epochs 10, full CIFAR-10)
./ops.sh promote        # transition latest registered model version → Production
./ops.sh build          # docker build train and serve images (local arch)
./ops.sh run            # start serve container on :8000 (requires build + promote)
./ops.sh smoke-test     # curl /health then POST /predict with tests/test_image.jpg
./ops.sh clean          # stop serve container + MLflow server
```

For cross-arch images targeting the homelab or GKE (both `linux/amd64`) from the M1 Mac:
```bash
docker buildx build --platform linux/amd64 -f docker/train.Dockerfile -t ml-platform/train:latest .
docker buildx build --platform linux/amd64 -f docker/serve.Dockerfile -t ml-platform/serve:latest .
```

To run training directly (outside Docker):
```bash
source .venv/bin/activate
python src/train.py --epochs 1 --subset 1000 --tracking-uri http://localhost:5000
```

To run the serve process directly:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

## Code architecture

The pipeline follows a strict train → register → serve flow with MLflow as the registry hub:

1. **`src/train.py`** — CLI (Click) that fine-tunes MobileNetV2 (ImageNet weights, classifier head replaced) on CIFAR-10 resized to 96×96. Logs params/metrics to MLflow and registers the model under the name `cifar10-mobilenet`. Key env var: `CIFAR_DATA_DIR` (default `./data`).

2. **`scripts/promote.py`** — reads all versions of `cifar10-mobilenet` from the MLflow registry and transitions the highest version number to `Production` (archiving others). Must be run before starting the serve container. Key env var: `MLFLOW_TRACKING_URI`.

3. **`src/serve.py`** — FastAPI app that loads `models:/cifar10-mobilenet/Production` from MLflow at startup (falls back to `latest` if no Production version exists). Exposes `/predict` (base64 JPEG → `{class_name, confidence, latency_ms}`), `/health`, and `/metrics` (Prometheus). Prometheus counters/histograms are defined at module level (`PREDICT_COUNTER`, `PREDICT_LATENCY`). Key env vars: `MODEL_NAME`, `MODEL_STAGE`, `MLFLOW_TRACKING_URI`.

4. **`ops.sh`** — MLflow server is started with `--serve-artifacts` and `--default-artifact-root mlflow-artifacts:/` so artifact URIs are container-portable (not absolute host paths). The serve container mounts `$REPO_ROOT/mlruns` and uses `host.docker.internal` to reach the host MLflow server.

MLflow uses a SQLite backend (`mlruns/mlflow.db`) and stores artifacts under `mlruns/artifacts/`. This is intentional for the homelab demo — not production-grade.

## Target stack and constraints

Stack is fixed by scope §4 — don't substitute equivalents without asking:

- **Container/orchestration**: Docker (multi-stage, non-root, slim bases) → Helm 3 → k3s (homelab) **and** GKE Autopilot (cloud), same chart for both (DoD #8).
- **ML**: PyTorch + CIFAR-10 (torchvision) + MLflow tracking/registry + FastAPI/Uvicorn serving.
- **CI/CD**: GitHub Actions, push images to `ghcr.io`, smoke-test post-deploy.
- **Monitoring**: Prometheus + Grafana, with alerts on latency degradation / error-rate spikes.
- **GPU**: `nvidia-device-plugin` on the homelab only.
- **A/B**: canary via two Deployments + Service/Ingress traffic split (not blue/green, not shadow).

Hardware budget shapes design decisions:

- **Asus homelab** (k3s target): i5-6200U, **8 GB RAM**, GeForce 940M with **2 GB VRAM**, Ubuntu 24.04. The kernel is small. Set tight `resources.requests/limits`, prefer MobileNetV2, batch size 1 for GPU inference, and disable optional components when the cluster gets close to OOM. k3s was chosen over Minikube specifically for this reason.
- **GKE Autopilot**: budget ≤ 50 €. Reserve cloud deploys for the final demo (3–5 days). Develop locally.
- **MacBook M1**: dev only — write Dockerfiles/charts here, but cross-arch builds are needed for x86 homelab and GKE. Use `docker buildx` with `--platform linux/amd64` (or set up a buildx builder) when producing images intended for either cluster.

Out-of-scope items (scope §2.2) — refuse or flag if asked: distributed/multi-node training, full Kubeflow, automated retraining, production-grade RBAC/network-policies/secrets, multi-GPU, Terraform/IaC, training large models from scratch.

## Definition of Done (verification anchor)

The eight DoD criteria in scope §5 are how the thesis is graded. When closing out work, map changes back to these and flag any that regress:

1. Train job runs in container, metrics in MLflow.
2. `/predict` answers in-cluster.
3. `helm install` brings everything up Running.
4. Push-to-`main` triggers CI/CD end-to-end.
5. Grafana dashboard shows latency / throughput / error rate under load.
6. Canary splits traffic between two model versions, visible in metrics.
7. GPU inference works on homelab; CPU-vs-GPU latency comparison exists.
8. Same Helm chart deploys cleanly to both k3s and GKE.

## Working with the thesis docs

The `docs/thesis/` markdown is academic prose, not project documentation — citations follow `[Author, Year]` / `[filename.pdf, p. N]` style and must be preserved verbatim. When the user asks for thesis edits:

- Don't paraphrase across cited claims; if a claim has a citation, the rewrite must keep that citation attached to the same claim.
- Sections sometimes explicitly note "інформація відсутня у джерелах" — preserve that hedging; do not fill it in from general knowledge.
- The blueprint and draft are versioned in their filenames (`v1.0`, `v2.0`); bump the filename when producing a substantively new revision rather than overwriting.
