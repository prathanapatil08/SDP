from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any


def log_model_run(
    *,
    run_name: str,
    model_name: str,
    task_type: str,
    target_column: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Log model artifacts only when an explicit MLflow URI is configured."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return {
            "status": "disabled",
            "reason": "Set MLFLOW_TRACKING_URI to enable MLflow run logging.",
        }

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            "Autonomous-MLOps-Models",
        )
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tags(
                {
                    "model_name": model_name,
                    "task_type": task_type,
                    "target_column": target_column,
                    "deployment_authorized": "false",
                }
            )
            for name, value in parameters.items():
                if value is not None and isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    mlflow.log_param(name, value)
            for name, value in metrics.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    mlflow.log_metric(name, float(value))
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.log_artifact(str(metadata_path), artifact_path="models")
            return {
                "status": "logged",
                "run_id": active_run.info.run_id,
                "experiment_name": experiment_name,
            }
    except Exception as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
