"""FastAPI inference service for the registered cifar10-mobilenet model."""
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.exceptions
import mlflow.pyfunc
import mlflow.pytorch
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from torchvision import transforms

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("serve")

Image.MAX_IMAGE_PIXELS = 25_000_000  # decompression-bomb guard (~25 megapixels)
MAX_IMAGE_BYTES = 5 * 1024 * 1024    # 5 MB decoded limit

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "cifar10-mobilenet")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

PREDICT_COUNTER = Counter(
    "ml_predict_requests_total", "Total /predict requests", ["status"]
)
PREDICT_LATENCY = Histogram(
    "ml_predict_latency_seconds", "End-to-end /predict latency in seconds"
)

PREPROCESS = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PredictRequest(BaseModel):
    image_b64: str


class PredictResponse(BaseModel):
    class_name: str
    confidence: float
    latency_ms: float


def _load_model() -> torch.nn.Module:
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    fallback_uri = f"models:/{MODEL_NAME}/latest"
    try:
        model = mlflow.pytorch.load_model(uri)
    except mlflow.exceptions.MlflowException as exc:
        # Registry may have no Production version yet (pre-promote); fall back to latest.
        logger.warning("%s not available (%s); falling back to %s", uri, exc, fallback_uri)
        model = mlflow.pytorch.load_model(fallback_uri)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = _load_model()
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model.to(app.state.device)
    logger.info("loaded %s/%s from %s on device %s", MODEL_NAME, MODEL_STAGE, TRACKING_URI, app.state.device)
    yield


app = FastAPI(title="cifar10-mobilenet inference", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    try:
        raw = base64.b64decode(req.image_b64, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            PREDICT_COUNTER.labels(status="bad_request").inc()
            raise HTTPException(status_code=413, detail="image payload too large")
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except HTTPException:
        raise
    except (binascii.Error, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        PREDICT_COUNTER.labels(status="bad_request").inc()
        logger.warning("bad request: %s", exc)
        raise HTTPException(status_code=400, detail="invalid image payload")

    tensor = PREPROCESS(image).unsqueeze(0).to(app.state.device)
    try:
        with torch.no_grad():
            logits = app.state.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, idx = torch.max(probs, dim=0)
    except Exception as exc:
        PREDICT_COUNTER.labels(status="error").inc()
        logger.error("inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="inference failed")

    latency_s = time.perf_counter() - t0
    PREDICT_LATENCY.observe(latency_s)
    PREDICT_COUNTER.labels(status="ok").inc()

    return PredictResponse(
        class_name=CIFAR10_CLASSES[int(idx.item())],
        confidence=float(confidence.item()),
        latency_ms=round(latency_s * 1000.0, 2),
    )
