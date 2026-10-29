from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_operations_backup(
    *,
    root: Path,
    registry_path: Path,
    mlflow_database_path: Path,
) -> dict[str, Any]:
    backup_dir = root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    result: dict[str, Any] = {"created_at": timestamp, "files": []}

    if registry_path.is_file():
        with registry_path.open("r", encoding="utf-8") as file:
            registry = json.load(file)
        if not isinstance(registry, dict) or not isinstance(
            registry.get("models"),
            list,
        ):
            raise ValueError("The model registry is invalid; backup was not created.")
        destination = backup_dir / f"registry_{timestamp}_{suffix}.json"
        temporary = destination.with_suffix(".tmp")
        shutil.copy2(registry_path, temporary)
        temporary.replace(destination)
        result["files"].append(
            {"kind": "model_registry", "filename": destination.name}
        )

    if mlflow_database_path.is_file():
        safe_stem = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            mlflow_database_path.stem,
        )
        destination = backup_dir / f"{safe_stem}_{timestamp}_{suffix}.sqlite"
        temporary = destination.with_suffix(".tmp")
        source_connection = sqlite3.connect(
            f"file:{mlflow_database_path.resolve().as_posix()}?mode=ro",
            uri=True,
            timeout=30,
        )
        destination_connection = sqlite3.connect(temporary, timeout=30)
        try:
            source_connection.backup(destination_connection)
        finally:
            destination_connection.close()
            source_connection.close()
        temporary.replace(destination)
        result["files"].append(
            {"kind": "mlflow_database", "filename": destination.name}
        )

    if not result["files"]:
        result["status"] = "nothing_to_backup"
        result["message"] = (
            "Neither the model registry nor configured MLflow database was found."
        )
        return result

    result["status"] = "completed"
    return result


def upload_retention_policy() -> dict[str, str]:
    return {
        "temporary_source_uploads": (
            "Deleted after upload validation and cleaned-copy creation."
        ),
        "cleaned_training_datasets": (
            "Kept for review and training until explicitly deleted; CSV files "
            "older than one hour are also removed lazily when another upload "
            "or training request runs. No background cleanup scheduler is configured."
        ),
        "prediction_csv": "Processed in memory and not persisted by the prediction API.",
        "prediction_logs": (
            "Thirty days of model IDs, outcomes, error classes, row counts, and "
            "latency; uploaded feature values are never stored."
        ),
    }
