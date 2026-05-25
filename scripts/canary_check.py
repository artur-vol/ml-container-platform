#!/usr/bin/env python3
"""Check canary track health against Prometheus metrics.

Exits 0 if both thresholds are met, 1 otherwise.
"""
from __future__ import annotations

import math
import os
import sys

import requests

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://192.168.0.100:9090")
MAX_ERROR_RATE = float(os.environ.get("MAX_ERROR_RATE", "0.05"))
MAX_P95_LATENCY_MS = float(os.environ.get("MAX_P95_LATENCY_MS", "500"))

_ERROR_RATE_QUERY = (
    '(sum(rate(ml_predict_requests_total{status!="ok",track="canary"}[5m])) OR vector(0))'
    ' / sum(rate(ml_predict_requests_total{track="canary"}[5m]))'
)
_P95_LATENCY_QUERY = (
    'histogram_quantile(0.95, sum by(le)'
    ' (rate(ml_predict_latency_seconds_bucket{track="canary"}[5m])))'
)


def _query(expr: str) -> float | None:
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": expr},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()["data"]["result"]
    if not result:
        return None
    value = float(result[0]["value"][1])
    return None if math.isnan(value) else value


def main() -> None:
    error_rate = _query(_ERROR_RATE_QUERY)
    if error_rate is None:
        print("no canary traffic in the last 5 minutes — cannot validate", file=sys.stderr)
        sys.exit(1)

    p95_s = _query(_P95_LATENCY_QUERY)
    if p95_s is None:
        print("no canary latency data in the last 5 minutes — cannot validate", file=sys.stderr)
        sys.exit(1)
    p95_ms = p95_s * 1000

    print(f"canary error_rate={error_rate:.4f}  (threshold={MAX_ERROR_RATE})")
    print(f"canary p95_latency={p95_ms:.1f}ms   (threshold={MAX_P95_LATENCY_MS}ms)")

    failed = False
    if error_rate > MAX_ERROR_RATE:
        print(
            f"FAIL: error rate {error_rate:.4f} exceeds {MAX_ERROR_RATE}",
            file=sys.stderr,
        )
        failed = True
    if p95_ms > MAX_P95_LATENCY_MS:
        print(
            f"FAIL: p95 latency {p95_ms:.1f}ms exceeds {MAX_P95_LATENCY_MS}ms",
            file=sys.stderr,
        )
        failed = True

    if failed:
        sys.exit(1)

    print("CANARY OK")


if __name__ == "__main__":
    main()
