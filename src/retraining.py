from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.experiment_tracking import log_model_run
from src.prediction import create_model_metadata, describe_prediction_model
from src.training_engine import (
    CLASSIFICATION_MODELS,
    MODELS_DIR,
    REGRESSION_MODELS,
    train_and_evaluate,
    validate_dataset_frame,
)

ROOT = Path(__file__).resolve().parent.parent
EVENT_DB_PATH = ROOT / "experiments" / "automatic_retraining.sqlite"
DEFAULT_DRIFT_THRESHOLD_PERCENT = 25.0


def build_drift_event_id(
    reference_bytes: bytes,
    incoming_bytes: bytes,
    target_column: str | None,
    model_name: str | None,
    threshold_percent: float,
) -> str:
    digest = hashlib.sha256()
    for value in (
        reference_bytes,
        incoming_bytes,
        (target_column or "").encode("utf-8"),
        (model_name or "").encode("utf-8"),
        format(threshold_percent, ".8g").encode("ascii"),
    ):
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)
    return digest.hexdigest()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS retraining_events (
            event_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            trigger_reason TEXT NOT NULL,
            dataset TEXT NOT NULL,
            model_type TEXT NOT NULL,
            training_status TEXT NOT NULL,
            error TEXT,
            details_json TEXT NOT NULL
        )
        """
    )
    return connection


def _decode_event(row: sqlite3.Row, duplicate: bool = False) -> dict[str, Any]:
    result = json.loads(row["details_json"])
    result.update(
        {
            "event_id": row["event_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "trigger_reason": row["trigger_reason"],
            "dataset": row["dataset"],
            "model_type": row["model_type"],
            "training_status": row["training_status"],
            "error": row["error"],
            "duplicate_event": duplicate,
        }
    )
    return result


def _read_event(connection: sqlite3.Connection, event_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM retraining_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()


def _update_event(
    connection: sqlite3.Connection,
    event_id: str,
    training_status: str,
    error: str | None,
    details: dict[str, Any],
    model_type: str | None = None,
) -> dict[str, Any]:
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    connection.execute(
        """
        UPDATE retraining_events
        SET updated_at = ?, training_status = ?, model_type = ?, error = ?,
            details_json = ?
        WHERE event_id = ?
        """,
        (
            updated_at,
            training_status,
            model_type or details.get("model_type", "automatic"),
            error,
            json.dumps(details, allow_nan=False),
            event_id,
        ),
    )
    connection.commit()
    row = _read_event(connection, event_id)
    if row is None:
        raise RuntimeError(f"Retraining event {event_id} disappeared during update.")
    return _decode_event(row)


def run_drift_retraining(
    reference: pd.DataFrame,
    *,
    event_id: str,
    reference_name: str,
    incoming_name: str,
    target_column: str | None,
    requested_model_name: str | None,
    drift_percentage: float,
    threshold_percent: float,
    db_path: Path = EVENT_DB_PATH,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    triggered = drift_percentage > threshold_percent
    model_name = requested_model_name or "automatic"
    reason = (
        f"Feature drift {drift_percentage:.2f}% exceeded the configured "
        f"{threshold_percent:.2f}% threshold."
        if triggered
        else (
            f"Feature drift {drift_percentage:.2f}% did not exceed the "
            f"{threshold_percent:.2f}% threshold."
        )
    )
    details: dict[str, Any] = {
        "reference_dataset": reference_name,
        "incoming_dataset": incoming_name,
        "target_column": target_column,
        "drift_percentage": drift_percentage,
        "threshold_percent": threshold_percent,
        "candidate_model_path": None,
        "metrics": None,
        "deployment_authorized": False,
    }

    connection = _connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        existing = _read_event(connection, event_id)
        if existing is not None:
            connection.rollback()
            return _decode_event(existing, duplicate=True)

        initial_error = None
        if not triggered:
            initial_status = "skipped_no_drift"
        elif not target_column or target_column not in reference.columns:
            initial_status = "skipped_missing_labels"
            initial_error = (
                "Supervised retraining requires a target column in the "
                "reference dataset; incoming data is not used as a label source."
            )
            reason = reason + " " + initial_error
        elif reference[target_column].dropna().empty:
            initial_status = "skipped_missing_labels"
            initial_error = (
                f"Reference target column '{target_column}' contains no labeled rows."
            )
            reason = reason + " " + initial_error
        else:
            initial_status = "running"

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        connection.execute(
            """
            INSERT INTO retraining_events (
                event_id, created_at, updated_at, trigger_reason, dataset,
                model_type, training_status, error, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                timestamp,
                timestamp,
                reason,
                f"reference={reference_name}; incoming={incoming_name}",
                model_name,
                initial_status,
                initial_error,
                json.dumps(details, allow_nan=False),
            ),
        )
        connection.commit()

        if initial_status != "running":
            row = _read_event(connection, event_id)
            if row is None:
                raise RuntimeError(f"Could not read retraining event {event_id}.")
            return _decode_event(row)

        try:
            validation = validate_dataset_frame(
                reference,
                target_column,
            )
            task_type = validation["task_type"]
            supported_models = (
                CLASSIFICATION_MODELS
                if task_type == "classification"
                else REGRESSION_MODELS
            )
            if requested_model_name is None:
                model_name = (
                    "random_forest_classifier"
                    if task_type == "classification"
                    else "random_forest_regressor"
                )
            elif requested_model_name not in supported_models:
                raise ValueError(
                    f"Model '{requested_model_name}' is not supported for "
                    f"{task_type}. Choose one of: {sorted(supported_models)}"
                )

            training_result = train_and_evaluate(
                validation["data"],
                target_column,
                model_name=model_name,
                task_type=task_type,
            )
            safe_model_name = re.sub(
                r"[^A-Za-z0-9._-]+",
                "_",
                model_name,
            ).strip("._") or "candidate"
            candidate_path = (
                models_dir
                / f"candidate_auto_{safe_model_name}_{event_id[:16]}.joblib"
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = candidate_path.with_name(
                candidate_path.name + ".tmp"
            )
            try:
                joblib.dump(training_result["model"], temporary_path)
                temporary_path.replace(candidate_path)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

            details.update(
                {
                    "model_type": model_name,
                    "task_type": task_type,
                    "target_column": target_column,
                    "candidate_model_path": str(candidate_path),
                    "metrics": training_result["metrics"],
                    "train_rows": training_result["train_rows"],
                    "test_rows": training_result["test_rows"],
                    "deployment_authorized": False,
                }
            )
            description = describe_prediction_model(
                training_result["model"],
                candidate_path,
                name=model_name,
                version=candidate_path.stem,
                source="saved artifact",
                model_id="",
                feature_frame=validation["data"].drop(
                    columns=[target_column]
                ),
                task_type=task_type,
                target_labels=(
                    sorted(validation["data"][target_column].unique().tolist(), key=str)
                    if task_type == "classification"
                    else None
                ),
            )
            metadata = create_model_metadata(
                description,
                metrics=training_result["metrics"],
                metrics_source=(
                    f"Held-out test partition ({training_result['test_rows']} rows)."
                ),
                model_path=candidate_path,
            )
            metadata["dataset_provenance"] = {
                "reference_dataset": reference_name,
                "incoming_dataset": incoming_name,
                "labeled_reference_rows": int(len(validation["data"])),
            }
            metadata_path = candidate_path.with_name(
                candidate_path.name + ".metadata.json"
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            details["tracking"] = log_model_run(
                run_name=f"automatic_{model_name}_{event_id[:12]}",
                model_name=model_name,
                task_type=task_type,
                target_column=target_column,
                parameters={
                    "trigger_reason": reason,
                    "drift_percentage": drift_percentage,
                    "threshold_percent": threshold_percent,
                    "train_rows": training_result["train_rows"],
                    "test_rows": training_result["test_rows"],
                },
                metrics=training_result["metrics"],
                model_path=candidate_path,
                metadata_path=metadata_path,
            )
            return _update_event(
                connection,
                event_id,
                "completed",
                None,
                details,
                model_type=model_name,
            )
        except Exception as error:
            details["training_status"] = "failed"
            details["model_type"] = model_name
            return _update_event(
                connection,
                event_id,
                "failed",
                f"{type(error).__name__}: {error}",
                details,
                model_type=model_name,
            )
    finally:
        connection.close()


def list_retraining_events(db_path: Path = EVENT_DB_PATH) -> list[dict[str, Any]]:
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM retraining_events ORDER BY created_at DESC"
        ).fetchall()
        return [_decode_event(row) for row in rows]
    finally:
        connection.close()
