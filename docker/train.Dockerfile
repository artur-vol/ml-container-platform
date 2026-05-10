# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt \
 && find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
 && find /opt/venv -path '*/torch/test' -prune -exec rm -rf {} + 2>/dev/null || true \
 && find /opt/venv -path '*/caffe2/test' -prune -exec rm -rf {} + 2>/dev/null || true


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

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
