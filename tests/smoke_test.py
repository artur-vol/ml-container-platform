"""Smoke tests for the serve container — run against a live /health+/predict endpoint."""
from __future__ import annotations

import base64
import io
import os

import pytest
import requests
from PIL import Image

BASE_URL = os.environ.get("SERVE_URL", "http://127.0.0.1:8000").rstrip("/")

CIFAR10_CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
}


def _jpeg_b64(width: int = 32, height: int = 32) -> str:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------

def test_predict_returns_valid_class():
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"image_b64": _jpeg_b64()},
        timeout=10,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["class_name"] in CIFAR10_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] > 0


def test_predict_larger_image():
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"image_b64": _jpeg_b64(96, 96)},
        timeout=10,
    )
    assert r.status_code == 200
    assert r.json()["class_name"] in CIFAR10_CLASSES


# ---------------------------------------------------------------------------
# /predict — error cases
# ---------------------------------------------------------------------------

def test_predict_rejects_invalid_base64():
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"image_b64": "not-valid-base64!!!"},
        timeout=5,
    )
    assert r.status_code == 400


def test_predict_rejects_missing_field():
    r = requests.post(f"{BASE_URL}/predict", json={}, timeout=5)
    assert r.status_code == 422


def test_predict_rejects_oversized_image():
    # 6 MB of zeros encoded as base64 — exceeds the 5 MB guard in serve.py
    big = base64.b64encode(b"\x00" * (6 * 1024 * 1024)).decode()
    r = requests.post(f"{BASE_URL}/predict", json={"image_b64": big}, timeout=10)
    assert r.status_code in (400, 413)  # 413 when uvicorn rejects before app code


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint():
    r = requests.get(f"{BASE_URL}/metrics", timeout=5)
    assert r.status_code == 200
    assert "predict_requests_total" in r.text
