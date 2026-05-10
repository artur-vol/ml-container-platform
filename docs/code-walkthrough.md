# Code Walkthrough — Day-1 Local ML Stack

This document explains every file generated to deliver the day-1 goal: a fully working local ML stack (train → register → serve → smoke-test) for the bachelor's thesis project.

It is intentionally exhaustive — read top-to-bottom and you should be able to defend any design decision in a code review or thesis Q&A.

---

## 1. Repository layout

```
ml-container-platform/
├── CLAUDE.md                    # guidance for future Claude Code sessions
├── ops.sh                       # local dev driver (train/build/run/smoke-test/clean…)
├── requirements-train.txt       # train-image deps
├── requirements-serve.txt       # serve-image deps
├── docker/
│   ├── train.Dockerfile         # multi-stage train image
│   └── serve.Dockerfile         # multi-stage serve image
├── src/
│   ├── train.py                 # MobileNetV2 fine-tune + MLflow registration
│   └── serve.py                 # FastAPI inference (/predict /health /metrics)
├── scripts/
│   └── promote.py               # promote latest registered version → Production
├── tests/
│   └── gen_test_image.py        # produces tests/test_image.jpg from CIFAR-10
└── docs/
    ├── thesis/                  # academic scope/draft/blueprint (Ukrainian)
    └── code-walkthrough.md      # this file
```

Side artefacts created at runtime (all ignored in `.gitignore`): `mlruns/` (sqlite + artifact store + log + pidfile), `data/` (CIFAR-10 download).

---

## 2. The big picture

```
┌────────────────────┐  POST {"image_b64": …}    ┌───────────────────────┐
│  curl / smoke-test │ ────────────────────────▶│  serve container      │
└────────────────────┘                          │  (FastAPI, port 8000) │
                                                └─────────┬─────────────┘
                                                          │ models:/cifar10-mobilenet/Production
                                                          ▼
┌────────────────────┐                          ┌───────────────────────┐
│  src/train.py      │ ── log params/metrics ──▶│  MLflow tracking      │
│  (host venv)       │ ── register model    ──▶│  server (host :5000)  │
└────────────────────┘                          │  sqlite + mlflow-     │
                                                │  artifacts:/ proxy    │
                                                └───────────────────────┘
```

- The **MLflow tracking server** runs on the host at `127.0.0.1:5000`, backed by sqlite (`mlruns/mlflow.db`). It also serves artifacts via the `mlflow-artifacts:/` proxy, with the actual files at `mlruns/artifacts/`.
- **Training** runs in the host venv. It logs metrics + registers the model.
- **Serving** runs in a Docker container. It pulls the model artifact from the registry over the proxy and exposes a REST API.
- The container reaches the host MLflow via `host.docker.internal:5000` (Docker Desktop on macOS injects this DNS entry; we add `--add-host=host.docker.internal:host-gateway` so the same flag also works on Linux).

This split — host-resident MLflow + containerised serving — mirrors the production layout (cluster-resident MLflow + sidecar/serve pods) without needing Kubernetes today.

---

## 3. `requirements-*.txt`

Two requirements files because the train image needs torchvision-only ML deps, while the serve image additionally needs FastAPI/Pillow/Prometheus, and we want each Docker layer cache to be useful independently.

### `requirements-train.txt`

```
torch==2.4.1
torchvision==0.19.1
mlflow==2.17.2
click==8.1.7
```

Pins matter:

- `mlflow==2.17.2` is the last 2.x line that still has registry stages (`Production`, `Staging`, `Archived`). MLflow 3.x deprecates stages in favour of aliases. The thesis sticks with stages because the scope doc § 5 DoD #6 explicitly references "two model versions / canary", and stage transitions are the most pedagogically obvious form.
- `torch==2.4.1` / `torchvision==0.19.1`: matched pair. Newer torch/torchvision combinations have ABI mismatches around `MobileNet_V2_Weights.IMAGENET1K_V2`. 2.4.x has wheels for both `osx-arm64` (M1 dev) and `linux-amd64` (homelab/GKE).
- `click==8.1.7` for the train CLI; pinned to stay compatible with the older mlflow click constraints.

### `requirements-serve.txt`

```
fastapi==0.115.4
uvicorn[standard]==0.32.0
mlflow==2.17.2
pillow==11.0.0
prometheus-client==0.21.0
torch==2.4.1
torchvision==0.19.1
```

`uvicorn[standard]` pulls in `httptools`, `uvloop`, `watchfiles`, `websockets` — production-grade asyncio loop and faster HTTP parser. `torch + torchvision` are required because the registered artefact is logged via `mlflow.pytorch.log_model`, which loads via `torch.load` (it is not a self-contained pyfunc wheel).

Both files are deliberately not unified: the train image weighs more (training only needs CPU/GPU torch + dataset libs); the serve image needs the web stack but not, say, `pyarrow`. Splitting saves disk and rebuild time.

---

## 4. `src/train.py`

### CLI surface

```
--epochs INT                # default 1
--subset INT                # default None — use first N training samples
--batch-size INT            # default 64
--lr FLOAT                  # default 1e-3
--num-workers INT           # default 2
--tracking-uri TEXT         # default $MLFLOW_TRACKING_URI or http://localhost:5000
--experiment TEXT           # default cifar10-mobilenet
--register / --no-register  # default --register
```

`--subset` is the lever that makes a 1000-sample epoch finish in ~10 s on an M1, while keeping `train-full` (10 epochs, full dataset) in the same script.

### Model construction (`build_model`)

```python
weights = MobileNet_V2_Weights.IMAGENET1K_V2
model = mobilenet_v2(weights=weights)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
```

Standard transfer-learning recipe: keep the ImageNet feature extractor, replace the 1000-class head with a 10-class one. We do **not** freeze the backbone — fine-tuning the whole network is faster to convergence on CIFAR-10 than feature extraction, and we have plenty of compute headroom for a single 1-epoch demo.

### Data pipeline (`build_loaders`)

CIFAR-10 is 32×32, but MobileNetV2's ImageNet weights expect at least 96×96 inputs (the depthwise convolutions otherwise collapse the spatial dimensions). We resize to 96×96 with bilinear interpolation and normalise with ImageNet means/stds (the same statistics the pretrained weights were trained with).

When `--subset N` is provided, we slice both train and test sets:

- train: first `min(N, len)` samples
- test: first `max(N // 5, 200)` samples — enough to make `val_accuracy` meaningful even on a 1000-sample subset.

`num_workers=2` is conservative — on the homelab (8 GB RAM) more workers blow the memory budget; on M1 it's plenty.

### Evaluation (`evaluate`)

Computes:

- `loss` — average `CrossEntropyLoss` per sample.
- `accuracy` — fraction correct.
- `latency_ms_per_batch` — wall-clock forward pass timed with `time.perf_counter()`.

We deliberately log per-batch latency rather than per-sample because batch latency is what Prometheus will see in production for typical inference workloads. The `inference_latency_ms` metric logged later (single-image, 20-iter average after 3-iter warmup) is the one DoD #7 cares about (CPU-vs-GPU comparison).

### Training loop

Vanilla PyTorch:

```python
optimizer.zero_grad(); loss.backward(); optimizer.step()
```

Each epoch logs four scalars to MLflow with `step=epoch`:

- `train_loss`, `train_accuracy`
- `val_loss`, `val_accuracy`, `val_latency_ms_per_batch`

`mlflow.log_params` records the training configuration (epochs, subset, batch size, lr, model name, device, input size). These are visible as columns in the MLflow UI experiment view, making run comparison straightforward.

### Single-image latency benchmark

```python
for _ in range(3): model(sample)            # warmup
t0 = time.perf_counter()
for _ in range(20): model(sample)
single_latency_ms = (time.perf_counter() - t0) * 1000.0 / 20
mlflow.log_metric("inference_latency_ms", single_latency_ms)
```

Three warm-up calls populate the cuDNN auto-tuner / page-in CPU caches; the next 20 are timed and averaged. This is the headline number for the thesis's "CPU vs GPU latency" comparison (DoD #7). When the homelab's 940M is wired up, the same script reports the GPU number — same code path, different `device`.

### Model logging & registration

```python
mlflow.pytorch.log_model(
    pytorch_model=model.cpu(),
    artifact_path="model",
    registered_model_name=MODEL_NAME,            # "cifar10-mobilenet"
)
```

Two things to note:

- `model.cpu()` before logging ensures the artifact is portable. If we logged a CUDA-device model, loading on a CPU-only homelab would fail until torch added the right cuda libraries. CPU artifacts load anywhere.
- `registered_model_name=` is the magic flag that creates a new version under the **Model Registry** (vs just logging an artifact). On first run it creates `cifar10-mobilenet` v1; on subsequent runs it creates v2, v3, … without us needing to manage versions manually.

A `mlflow.set_tag("classes", "airplane,automobile,…")` is written so the registry remembers the class names alongside the artefact — the serving code currently re-hardcodes them, but the tag is there for any consumer that wants to introspect.

---

## 5. `src/serve.py`

### Module-level configuration

```python
MODEL_NAME    = os.environ.get("MODEL_NAME", "cifar10-mobilenet")
MODEL_STAGE   = os.environ.get("MODEL_STAGE", "Production")
TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
```

Everything is environment-driven — the same image runs against k3s-internal MLflow, GKE-internal MLflow, or the host MLflow in dev. No code change between environments.

### Prometheus metrics

```python
PREDICT_COUNTER = Counter(
    "ml_predict_requests_total", "Total /predict requests", ["status"]
)
PREDICT_LATENCY = Histogram(
    "ml_predict_latency_seconds", "End-to-end /predict latency in seconds"
)
```

- The counter has a `status` label (`ok` / `bad_request` / `error`). Grafana dashboards in chapter 4.4 of the thesis can split error rate by status without adding code.
- The histogram uses Prometheus's default bucket layout (`.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10` seconds), which is appropriate for per-request latencies in the millisecond-to-second range.

`/metrics` returns `generate_latest()` with `Content-Type: text/plain; version=0.0.4`. That's the format Prometheus's scrape job expects — once the Helm chart adds a `ServiceMonitor`/`PodMonitor`, scraping is zero-config.

### Image preprocessing

```python
PREPROCESS = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

This must match `train.py`'s eval transform exactly, otherwise inference is statistically corrupt. Identical Resize → ToTensor → Normalise pipeline. The 96×96 size is the same constraint discussed in §4 — anything smaller and MobileNetV2 collapses.

### Model loading (`_load_model`)

```python
mlflow.set_tracking_uri(TRACKING_URI)
uri          = f"models:/{MODEL_NAME}/{MODEL_STAGE}"      # e.g. models:/cifar10-mobilenet/Production
fallback_uri = f"models:/{MODEL_NAME}/latest"
try:
    model = mlflow.pytorch.load_model(uri)
except Exception as exc:
    print(f"[serve] {uri} not available ({exc}); falling back to {fallback_uri}")
    model = mlflow.pytorch.load_model(fallback_uri)
```

Why the fallback?

- On a brand-new install you have not yet run `./ops.sh promote`, so `Production` doesn't exist yet. Falling back to `latest` lets the serve container come up against an unpromoted model.
- Once you do promote, the `try` branch succeeds and the fallback is never used.

In production this fallback would be removed (`Production` should exist before deploy), but for thesis demonstration it removes a silent foot-gun.

### Lifespan handler

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = _load_model()
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model.to(app.state.device)
    yield
```

Loading happens once at startup (before `yield`), not on every request. The model is stashed on `app.state` — a FastAPI/Starlette idiom that Pydantic-validated request handlers can reach via `request.app.state` (or, as we do, via the module-level `app`).

If the model fails to load (e.g. MLflow is down), the `lifespan` raises and Starlette logs `Application startup failed. Exiting.` — exactly what Kubernetes' readiness probe should see, so the pod gets restarted instead of accepting traffic.

### Endpoints

#### `GET /health`

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

Liveness/readiness probe target. Doesn't touch the model — by design — because Kubernetes treats failure here as "pod is dead, restart"; we don't want a transient registry blip to nuke the pod.

#### `GET /metrics`

```python
return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Single-line endpoint. `prometheus-client` does all the work.

#### `POST /predict`

```python
class PredictRequest(BaseModel):
    image_b64: str

class PredictResponse(BaseModel):
    class_name: str
    confidence: float
    latency_ms: float
```

The Pydantic models give us request validation (rejects non-string `image_b64` with a 422 automatically) and OpenAPI schema generation (visit `/docs` to see Swagger).

The handler:

1. `t0 = time.perf_counter()` — start latency clock.
2. `base64.b64decode(req.image_b64, validate=True)` — strict mode rejects garbage. On failure: 400 with `bad_request` counter increment.
3. `Image.open(io.BytesIO(raw)).convert("RGB")` — Pillow auto-detects JPEG/PNG/etc. `.convert("RGB")` collapses palettes/grayscale to 3 channels (CIFAR is RGB).
4. `PREPROCESS(image).unsqueeze(0).to(device)` — same transform as eval, batched (`unsqueeze(0)`).
5. `torch.no_grad()` + softmax + argmax. On failure: 500 with `error` counter.
6. Observe latency, increment `ok` counter, return `PredictResponse`.

`latency_ms` is rounded to 0.01 ms because finer precision is meaningless given Python's `perf_counter()` resolution.

---

## 6. `docker/train.Dockerfile`

### Multi-stage rationale

Stage 1 (`builder`):

```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt
```

- `gcc/g++` are needed for any package that ships a sdist instead of a wheel. We delete the apt cache in the same RUN to keep the layer slim.
- `python -m venv /opt/venv` plus `ENV PATH=/opt/venv/bin:$PATH` means every subsequent `pip install` writes to the venv, not the system site-packages. This makes the next `COPY --from=builder /opt/venv /opt/venv` self-contained — no need to copy `/usr/local/lib/python3.11/site-packages`.

Stage 2 (runtime):

```dockerfile
FROM python:3.11-slim
RUN groupadd -r appuser -g 1000 \
 && useradd -r -u 1000 -g appuser -m -d /home/appuser appuser \
 && mkdir -p /app /app/data /app/mlruns \
 && chown -R appuser:appuser /app /home/appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/train.py /app/src/train.py

USER appuser
WORKDIR /app
ENV CIFAR_DATA_DIR=/app/data \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

ENTRYPOINT ["python", "/app/src/train.py"]
```

- Non-root `appuser` (uid 1000) is a thesis scope §2.1 requirement.
- We pre-create `/app/data` and `/app/mlruns` and `chown` them so the unprivileged user can write to those volume mounts without the container needing root.
- `gcc/g++` are NOT in the runtime stage — that's the whole point of multi-stage. Final image carries only the venv and the python script.
- `ENTRYPOINT ["python", "/app/src/train.py"]` lets you do `docker run train:latest --epochs 5 --subset 2000` — the `docker run` args become click args.

### `# syntax=docker/dockerfile:1.7`

Pinned syntax frontend so the build is reproducible across BuildKit versions. This unlocks `--mount=type=cache` etc. for future optimisation; not used today but free to add.

---

## 7. `docker/serve.Dockerfile`

Same skeleton as `train.Dockerfile`. Differences:

- `requirements-serve.txt` instead of `requirements-train.txt`.
- `EXPOSE 8000`.
- `ENTRYPOINT ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]` — uvicorn launches FastAPI directly. `0.0.0.0` is essential — `127.0.0.1` would only listen inside the container.
- `MODEL_NAME` and `MODEL_STAGE` env defaults are baked in; both are overridable at `docker run -e …`.

For a Kubernetes deployment, the same image accepts a non-default `MLFLOW_TRACKING_URI=http://mlflow.mlflow.svc.cluster.local:5000` (in-cluster service DNS) and `MODEL_STAGE=Staging` for canary. No image rebuild needed.

---

## 8. `ops.sh`

A single-file bash driver with subcommands. `set -euo pipefail` ensures any unhandled failure aborts the script — important for CI parity.

### Configuration (top of file)

```bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
```

Every value is overridable via env: `MLFLOW_PORT=5050 ./ops.sh train` works without editing the script.

### Helpers

- `log` / `warn` / `die` — coloured stderr/stdout output.
- `ensure_venv` — create a 3.11 venv on demand (`python3.11 -m venv` first, fallback to `python3 -m venv`).
- `ensure_train_deps` / `ensure_serve_deps` — idempotent; only `pip install -r …` if the marker imports are missing. Subsequent invocations are essentially free.
- `mlflow_running` — `curl /health` boolean.

### `start_mlflow`

```bash
nohup "$MLFLOW_BIN" server \
    --backend-store-uri "$MLFLOW_BACKEND" \
    --default-artifact-root "mlflow-artifacts:/" \
    --artifacts-destination "$MLFLOW_ARTIFACTS" \
    --serve-artifacts \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    >"$MLFLOW_LOG" 2>&1 &
echo $! >"$MLFLOW_PIDFILE"
```

Critical detail: `--default-artifact-root mlflow-artifacts:/` plus `--serve-artifacts` enables the **proxied artifact** mode introduced in MLflow 1.24+. With this:

- The registry stores artefact URIs as `mlflow-artifacts:/<run_id>/artifacts/model`.
- Clients (training / serving) ask the tracking server for these via HTTP.
- The tracking server reads the actual bytes from `--artifacts-destination` and streams them back.

Why this matters: without the proxy, MLflow would store **absolute filesystem paths** of the host (`/Users/arturvolinec/.../mlruns/artifacts/...`) into the registry. Inside the serve container, that path doesn't exist, and `mlflow.pytorch.load_model("models:/cifar10-mobilenet/Production")` blows up with `OSError: No such file or directory`. The proxy makes registry URIs container-portable — same registry, same URIs, runs on M1 / k3s / GKE without code changes.

The startup loop polls `/health` for up to 60 s before declaring failure. On failure, it tails the last 20 log lines so the user sees the real error.

### `stop_mlflow`

```bash
pids="$(lsof -nP -iTCP:"$MLFLOW_PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -nP -iTCP:"$MLFLOW_PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        kill -9 $pids 2>/dev/null || true
    fi
fi
```

This is more aggressive than the obvious `pkill -f "mlflow server"` because:

- macOS `pkill -f` doesn't reliably match argv on the Python framework wrapper — gunicorn workers come up as `Python -m gunicorn ... mlflow.server:app` and `pkill -f` returned no matches in testing.
- `mlflow server` spawns 4 gunicorn workers as children. Killing only the parent (via pidfile) leaves the workers as orphans, still bound to port 5000. The next `start_mlflow` then sees `mlflow_running` return true and never starts a fresh server with the new flags — silently sticking with stale config.

The lsof-based reaper kills *anything* listening on 5000, which is exactly what we want when "stop the MLflow server" means "make port 5000 free". `kill` (SIGTERM) first, fall back to `kill -9` (SIGKILL) after 1 s.

### `cmd_train` / `cmd_train_full`

Both pass `--tracking-uri "$MLFLOW_URI"` explicitly *and* set `MLFLOW_TRACKING_URI` env. Click reads the flag; MLflow itself reads the env. Setting both removes any ambiguity if some library decides to consult only one.

Quick: `--epochs 1 --subset 1000`. Full: `--epochs 10`, no subset. Anything additional the user passes (`./ops.sh train --batch-size 128`) is forwarded via `"$@"`.

### `cmd_build`

```bash
docker build -f "$REPO_ROOT/docker/train.Dockerfile" -t "$TRAIN_IMAGE" "$REPO_ROOT"
docker build -f "$REPO_ROOT/docker/serve.Dockerfile" -t "$SERVE_IMAGE" "$REPO_ROOT"
```

The build context is the repo root (so the Dockerfiles can `COPY src/` and the requirements files), but the Dockerfile path is explicit because they live in `docker/`.

### `cmd_run`

```bash
docker run -d --name "$SERVE_CONTAINER" \
    -p "$SERVE_PORT:8000" \
    --add-host=host.docker.internal:host-gateway \
    -e MLFLOW_TRACKING_URI="http://host.docker.internal:${MLFLOW_PORT}" \
    -e MODEL_NAME=cifar10-mobilenet \
    -e MODEL_STAGE=Production \
    -v "$REPO_ROOT/mlruns:$REPO_ROOT/mlruns" \
    "$SERVE_IMAGE"
```

Three subtle decisions:

1. `--add-host=host.docker.internal:host-gateway` — Docker Desktop on macOS already provides this DNS entry, but the same flag also makes it work on Linux (where `host.docker.internal` is not provided by default). One image, both platforms.
2. `-v "$REPO_ROOT/mlruns:$REPO_ROOT/mlruns"` — the *mirror mount*. We mount the host mlruns at the **same absolute path** inside the container. This is a belt-and-braces guard for any registered model whose artefact URI ended up as a host-absolute filesystem path (e.g. v1, registered before the proxy config was active). New runs go through the proxy and don't depend on this mount, but it costs nothing to keep. When moving to k3s/GKE (no host filesystem), the proxy carries you.
3. `MODEL_STAGE=Production` — explicit. If the user forgot to `./ops.sh promote`, the serve fallback in `_load_model` quietly tries `latest`. We never want to lie about which stage we served.

Then we poll `/health` for up to 60 s, and dump container logs on failure — same pattern as `start_mlflow`.

### `cmd_smoke_test`

```bash
img="$REPO_ROOT/tests/test_image.jpg"
[[ -f "$img" ]] || "$PY" "$REPO_ROOT/tests/gen_test_image.py" --out "$img"

curl -fsS "http://127.0.0.1:${SERVE_PORT}/health" && echo

payload="$("$PY" -c "import base64,json,sys; print(json.dumps({'image_b64': base64.b64encode(open(sys.argv[1],'rb').read()).decode()}))" "$img")"
curl -fsS -X POST "http://127.0.0.1:${SERVE_PORT}/predict" \
    -H "Content-Type: application/json" \
    -d "$payload" && echo
```

- We auto-generate `tests/test_image.jpg` if missing (delegated to `tests/gen_test_image.py` — it grabs a real CIFAR test sample so smoke-test inputs match the training distribution).
- The base64 + JSON encoding is done in a one-liner Python invocation (rather than `base64` and `jq` shell pipes) so the same logic works identically on macOS and Linux without flag-version differences.
- `curl -f` makes the script exit non-zero on HTTP ≥ 400, so `set -e` catches a failed predict.

### `cmd_clean`

```bash
docker rm -f "$SERVE_CONTAINER"   # if exists
stop_mlflow
```

After this, `lsof -nP -iTCP:5000 -sTCP:LISTEN | grep -v ControlCe` should be empty. `ControlCenter` is a macOS process that opportunistically binds to port 5000 (AirPlay receiver) — unrelated to MLflow.

---

## 9. `scripts/promote.py`

```python
client = MlflowClient(tracking_uri=TRACKING_URI)
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(versions, key=lambda v: int(v.version))
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest.version,
    stage="Production",
    archive_existing_versions=True,
)
```

- `search_model_versions` returns every version. We pick max numeric version (not the API's "latest" alias, which is also fine but order of returns is implementation-defined).
- `archive_existing_versions=True` makes the previous Production version transition to `Archived`. So at any time exactly one version is in Production. This is the canary's "promote" half — combined with a future "demote" command that does the reverse, you get two-version traffic split.
- `MlflowClient.transition_model_version_stage` is deprecated in 3.x. We pin 2.17.2 to keep using it, with full awareness; the magister-thesis stretch goal could migrate to aliases.

---

## 10. `tests/gen_test_image.py`

```python
ds = datasets.CIFAR10(args.data_dir, train=False, download=True)
img, label = ds[args.index]
img.save(out, format="JPEG", quality=95)
```

- Uses the existing `data/` folder so we don't re-download CIFAR-10 if training already pulled it.
- `args.index` defaults to 0 (a frog after the official torchvision shuffle, though the prediction is what we report not what we expect — accuracy after 1 epoch / 1000 samples is ~69 %, so misclassification of any single image is well within distribution).
- `quality=95` — high enough that JPEG artefacts don't dominate the 32×32 → 96×96 upsample.

---

## 11. `.gitignore`

```
.venv/                       # local venv
__pycache__/  *.pyc
*.egg-info/

data/                        # CIFAR-10 download
mlruns/                      # MLflow state, artifacts, log, pid
mlartifacts/
*.pt  *.pth                  # raw torch checkpoints if anyone saves one

tests/test_image.jpg         # auto-generated by smoke-test

.DS_Store  .idea/  .vscode/

!docs/thesis/                # never ignore the thesis docs
```

The trailing `!docs/thesis/` is a force-include — it guards against any future blanket-ignore rule (`docs/`, etc.) accidentally hiding the thesis prose.

---

## 12. `CLAUDE.md`

Top-level guidance for future Claude Code sessions. Key points:

- This repo is graded on infrastructure, not model accuracy — bias decisions accordingly.
- `docs/thesis/` is the binding scope source; flag drift instead of silently expanding.
- The stack from the scope doc is fixed; don't substitute equivalents (no Minikube swap-in for k3s, no W&B swap-in for MLflow, etc.).
- Hardware/budget constraints (8 GB homelab, 50 € GKE budget, M1 dev → cross-arch builds for x86 nodes).
- The 8 DoD criteria from the scope doc are the verification anchor.
- Thesis docs are Ukrainian academic prose — preserve `[Author, Year]` citations exactly; don't paraphrase across cited claims.

---

## 13. End-to-end run, line by line

```bash
./ops.sh train
```

1. `ensure_venv` → creates `.venv` if missing.
2. `ensure_train_deps` → `pip install -r requirements-train.txt` (one-time, ~1 min).
3. `start_mlflow` → spawns `mlflow server` with sqlite + proxied artifacts on `127.0.0.1:5000`. PID written to `mlruns/mlflow.pid`. Polls `/health` until ready.
4. `python src/train.py --epochs 1 --subset 1000 --tracking-uri http://127.0.0.1:5000`.
   - Downloads CIFAR-10 (~170 MB) on first run to `data/`.
   - Downloads MobileNetV2 weights (~14 MB) to `~/.cache/torch/hub/checkpoints/`.
   - Trains 1 epoch, evaluates each epoch, logs to MLflow.
   - 20-iter latency benchmark.
   - `mlflow.pytorch.log_model(..., registered_model_name="cifar10-mobilenet")` → registry version v1.

```bash
./ops.sh promote
```

5. `python scripts/promote.py` → registry transition v1 → Production, archives previous Production (none on first run).

```bash
./ops.sh build
```

6. Two `docker build` invocations. First run: ~3-5 min (downloads slim base, installs torch). Subsequent runs: cached, seconds.

```bash
./ops.sh run
```

7. `start_mlflow` (idempotent — already up).
8. `docker run` the serve image with the env wiring described in §8. `host.docker.internal:5000` reaches host MLflow.
9. FastAPI lifespan loads the model from the registry over HTTP via the proxy. Container becomes ready when `/health` responds 200.

```bash
./ops.sh smoke-test
```

10. Generates a CIFAR test image if missing.
11. `GET /health` → `{"status":"ok"}`.
12. `POST /predict {"image_b64": …}` → `{"class_name": "...", "confidence": 0.69, "latency_ms": 107.87}`.

```bash
./ops.sh clean
```

13. `docker rm -f` the serve container.
14. `stop_mlflow` lsof-reaps every PID listening on 5000.

---

## 14. What this delivers vs the DoD (scope §5)

| # | DoD criterion | Today's delivery | Gap to close |
|---|---|---|---|
| 1 | Train runs in container, metrics in MLflow | `train.py` logs to MLflow on every run; image is buildable via `./ops.sh build`. | Run training *inside* a container against MLflow — Helm `Job` template later. |
| 2 | `/predict` answers in K8s | Works locally via Docker. | Helm chart with `Deployment` + `Service`. |
| 3 | `helm install` brings everything up | — | Chart authoring, week 2. |
| 4 | Push-to-main triggers CI/CD | — | GitHub Actions workflow, week 4. |
| 5 | Grafana dashboard | `/metrics` already emits Prometheus counters/histograms. | Prometheus + Grafana stack, week 5. |
| 6 | Canary | Registry + `promote.py` exist; need a second version + traffic split. | Week 6. |
| 7 | GPU on homelab | `train.py` already detects CUDA via `torch.cuda.is_available()`. | nvidia-device-plugin on k3s, week 3. |
| 8 | Same chart on k3s & GKE | — | Week 7. |

What you have today is items 1, partial 2, and the metrics half of 5. That's exactly the right place to be at the end of week 1.

---

## 15. Common gotchas (already mitigated)

- **MLflow registry path corruption.** Pre-2.17 registries with `--default-artifact-root <abs path>` make container loading impossible. Mitigated by the proxy config in `start_mlflow`.
- **Macos `pkill -f` not matching gunicorn.** Mitigated by lsof-based port reaping in `stop_mlflow`.
- **First-run dependency download.** Mitigated by `ensure_*_deps` only running pip on cold cache.
- **CIFAR-10 redundant downloads.** Mitigated by sharing `./data` between `train.py` and `gen_test_image.py`.
- **Stale serve container blocking restart.** Mitigated by `docker rm -f` in `cmd_run`.
- **Cross-arch image builds.** *Not* mitigated yet — the M1 builds produce arm64 images that won't run on the homelab i5 or GKE nodes. When it's time to push to ghcr.io, switch to `docker buildx build --platform linux/amd64 --push`.
- **Race in `mlflow_running`.** Mitigated indirectly — port-reaping `stop_mlflow` ensures the next `start_mlflow` sees a clean port.

---

## 16. Where to read what

| Question | File |
|---|---|
| What is in scope, what isn't, hard hardware/budget limits | `docs/thesis/thesis-scope-v1.0.md` |
| Chapter-by-chapter outline of the written thesis | `docs/thesis/thesis-draft-v2.0.md` |
| Literature-review prose with citations | `docs/thesis/thesis-blueprint.md` |
| How to be useful to future-you in this repo | `CLAUDE.md` |
| How to use it locally | `ops.sh` (subcommands at the bottom) |
| What every line of code does | this file |
