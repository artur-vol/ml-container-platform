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

COPY requirements-serve-cpu.txt /tmp/requirements-serve-cpu.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serve-cpu.txt


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

RUN groupadd -r appuser -g 1000 \
 && useradd -r -u 1000 -g appuser -m -d /home/appuser appuser \
 && mkdir -p /app /app/mlruns \
 && chown -R appuser:appuser /app /home/appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/serve.py /app/src/serve.py

USER appuser
WORKDIR /app

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MODEL_NAME=cifar10-mobilenet \
    MODEL_STAGE=Production

EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
