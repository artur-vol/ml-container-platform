# Repository Guidelines

## Project Structure & Module Organization

- `src/train.py` trains, evaluates, logs metrics, and registers the model in MLflow.
- `src/serve.py` exposes the FastAPI inference service, `/health`, `/predict`, and Prometheus `/metrics`.
- `docker/` contains training and serving Dockerfiles.
- `helm/ml-platform/` contains the Kubernetes chart and environment values files.
- `scripts/promote.py` promotes the latest MLflow model to Production.
- `docs/` holds walkthroughs, thesis notes, and setup guidance.
- `tests/` currently contains utilities such as `gen_test_image.py`; add automated tests here.
- Runtime outputs such as `data/` and `mlruns/` are local artifacts and should not be treated as source.

## Build, Test, and Development Commands

Use `./ops.sh` as the main local workflow driver:

- `./ops.sh train` starts MLflow and runs quick training.
- `./ops.sh train-full` runs full training.
- `./ops.sh promote` promotes the latest registered model to Production.
- `./ops.sh build` builds both train and serve Docker images.
- `./ops.sh run` starts the serving container on `localhost:8000`.
- `./ops.sh smoke-test` calls `/health` and `/predict` using `tests/test_image.jpg`.
- `./ops.sh clean` stops the serving container and MLflow server.

Manual setup: `python3.11 -m venv .venv && source .venv/bin/activate`, then install the needed requirements file.

## Coding Style & Naming Conventions

Use Python 3.11, four-space indentation, type hints where useful, and small functions around clear responsibilities. Keep constants in `UPPER_SNAKE_CASE`, functions in `snake_case`, and environment-backed settings near the top of each module. Prefer FastAPI/Pydantic models for API payloads. `ops.sh` may use Bash features.

## Testing Guidelines

There is no full automated test suite yet. For current verification, run `./ops.sh train`, `./ops.sh promote`, `./ops.sh build`, `./ops.sh run`, and `./ops.sh smoke-test`. When adding tests, place them under `tests/`, name files `test_*.py`, and cover request validation, model loading, and CLI edge cases. Keep fixtures small.

## Commit & Pull Request Guidelines

This repository has no existing commit history, so use concise Conventional Commit-style messages such as `feat: add helm ingress values` or `fix: handle invalid image payloads`. Pull requests should include the purpose, key commands run, changed runtime assumptions, linked issues if any, and screenshots or curl output when API behavior changes.

## Security & Configuration Tips

Do not commit model artifacts, local datasets, `.venv/`, secrets, or MLflow runtime state. Configure deployments through Helm values files and environment variables such as `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_STAGE`, and `SERVE_PORT`.
