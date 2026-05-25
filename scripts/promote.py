"""Promote the latest registered model version to Production with a quality gate.

Called by CI after smoke tests pass (see .github/workflows/ci.yml).
Can also be run manually: MLFLOW_TRACKING_URI=http://... python scripts/promote.py

Quality gate logic:
  - If a Production version already exists, compare val_accuracy of the candidate
    against it. Candidate must not be worse by more than QUALITY_GATE_DELTA (5%).
  - If val_accuracy is missing from either run, the gate is skipped (legacy runs).
  - If no Production version exists yet, promote unconditionally (first-time setup).

Exit codes: 0 = promoted, 1 = gate failed or no versions found.
"""
from __future__ import annotations

import os
import sys

from mlflow.tracking import MlflowClient

MODEL_NAME = os.environ.get("MODEL_NAME", "cifar10-mobilenet")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
# Maximum allowed accuracy drop from current Production to candidate.
QUALITY_GATE_DELTA = 0.05


def _get_val_accuracy(client: MlflowClient, version) -> float | None:
    # MLflow stores the last logged value for each metric key.
    metrics = client.get_run(version.run_id).data.metrics
    return metrics.get("val_accuracy")


def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"no versions registered for '{MODEL_NAME}'", file=sys.stderr)
        sys.exit(1)

    # Candidate = highest version number regardless of its current stage.
    latest = max(versions, key=lambda v: int(v.version))

    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if prod_versions:
        prod = max(prod_versions, key=lambda v: int(v.version))
        prod_acc = _get_val_accuracy(client, prod)
        cand_acc = _get_val_accuracy(client, latest)

        if prod_acc is not None and cand_acc is not None:
            threshold = prod_acc - QUALITY_GATE_DELTA
            if cand_acc < threshold:
                print(
                    f"quality gate FAILED: candidate v{latest.version} "
                    f"val_accuracy={cand_acc:.4f} is more than {QUALITY_GATE_DELTA:.0%} "
                    f"below production v{prod.version} val_accuracy={prod_acc:.4f} "
                    f"(threshold={threshold:.4f})",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(
                f"quality gate passed: candidate={cand_acc:.4f} "
                f"production={prod_acc:.4f} threshold={threshold:.4f}"
            )
        else:
            # One or both runs were trained without logging val_accuracy (e.g. older runs).
            print("quality gate skipped: val_accuracy metric not found in one or both runs")
    else:
        print("no current Production version — promoting unconditionally")

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
