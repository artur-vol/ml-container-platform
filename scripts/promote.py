"""Promote the latest version of the registered model to Production."""
from __future__ import annotations

import os
import sys

from mlflow.tracking import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "cifar10-mobilenet")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"no versions registered for '{MODEL_NAME}'", file=sys.stderr)
        sys.exit(1)
    latest = max(versions, key=lambda v: int(v.version))
    # transition_model_version_stage is deprecated in MLflow 3.x (replaced by aliases),
    # but the thesis pins mlflow==2.17.2 to keep stage-based promotion for DoD #6.
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"promoted {MODEL_NAME} v{latest.version} → Production")


if __name__ == "__main__":
    main()
