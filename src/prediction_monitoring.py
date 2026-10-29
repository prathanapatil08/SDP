from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 20000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_events (
            event_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            error_code TEXT,
            row_count INTEGER,
            latency_ms REAL NOT NULL
        )
        """
    )
    return connection


def record_prediction_event(
    database_path: Path,
    *,
    model_id: str,
    operation: str,
    status: str,
    error_code: str | None,
    row_count: int | None,
    latency_ms: float,
) -> None:
    created_at = datetime.now(timezone.utc)
    cutoff = (created_at - timedelta(days=30)).isoformat()
    connection = _connect(database_path)
    try:
        connection.execute(
            """
            INSERT INTO prediction_events (
                event_id, created_at, model_id, operation, status, error_code,
                row_count, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                created_at.isoformat(timespec="seconds"),
                model_id[:100],
                operation[:50],
                status[:30],
                error_code[:100] if error_code else None,
                row_count,
                max(float(latency_ms), 0.0),
            ),
        )
        connection.execute(
            "DELETE FROM prediction_events WHERE created_at < ?",
            (cutoff,),
        )
        connection.commit()
    finally:
        connection.close()


def prediction_monitoring_summary(
    database_path: Path,
    *,
    lookback_days: int = 30,
) -> dict[str, Any]:
    if not 1 <= lookback_days <= 90:
        raise ValueError("lookback_days must be between 1 and 90.")
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).isoformat(timespec="seconds")
    connection = _connect(database_path)
    try:
        totals = connection.execute(
            """
            SELECT COUNT(*) AS requests,
                   SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                   AVG(latency_ms) AS average_latency_ms
            FROM prediction_events
            WHERE created_at >= ?
            """,
            (cutoff,),
        ).fetchone()
        per_model = connection.execute(
            """
            SELECT model_id, operation, status, error_code, COUNT(*) AS count,
                   AVG(latency_ms) AS average_latency_ms
            FROM prediction_events
            WHERE created_at >= ?
            GROUP BY model_id, operation, status, error_code
            ORDER BY count DESC
            LIMIT 200
            """,
            (cutoff,),
        ).fetchall()
        return {
            "lookback_days": lookback_days,
            "requests": int(totals["requests"] or 0),
            "succeeded": int(totals["succeeded"] or 0),
            "failed": int(totals["failed"] or 0),
            "average_latency_ms": (
                float(totals["average_latency_ms"])
                if totals["average_latency_ms"] is not None
                else None
            ),
            "groups": [
                {
                    "model_id": row["model_id"],
                    "operation": row["operation"],
                    "status": row["status"],
                    "error_code": row["error_code"],
                    "count": int(row["count"]),
                    "average_latency_ms": float(row["average_latency_ms"]),
                }
                for row in per_model
            ],
        }
    finally:
        connection.close()
