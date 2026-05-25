"""Locust load test for the cifar10-mobilenet inference service.

Usage:
    locust -f tests/locustfile.py --host http://192.168.0.100:30800 \
        --headless -u 10 -r 2 --run-time 60s

The test image is read once at startup and reused for all requests
to avoid disk I/O overhead skewing latency measurements.
"""
from __future__ import annotations

import base64
from pathlib import Path

from locust import HttpUser, constant, task

_IMAGE_PATH = Path(__file__).parent / "test_image.jpg"
_IMAGE_B64 = base64.b64encode(_IMAGE_PATH.read_bytes()).decode()


class InferenceUser(HttpUser):
    wait_time = constant(0)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={"image_b64": _IMAGE_B64},
            name="/predict",
        )
