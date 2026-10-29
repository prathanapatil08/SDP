import sqlite3
from fastapi.testclient import TestClient

import api.main as api_module
from api.main import app
from src.prediction_monitoring import (
    prediction_monitoring_summary,
    record_prediction_event,
)

client = TestClient(app, headers={"X-API-Key": "test-only-api-key"})


def test_monitoring_records_metadata_only_and_aggregates_failures(tmp_path):
    database_path = tmp_path / "monitoring.sqlite"
    record_prediction_event(
        database_path,
        model_id="model-1",
        operation="csv_prediction",
        status="succeeded",
        error_code=None,
        row_count=10,
        latency_ms=20.5,
    )
    record_prediction_event(
        database_path,
        model_id="model-1",
        operation="csv_prediction",
        status="failed",
        error_code="ValueError",
        row_count=2,
        latency_ms=5.0,
    )

    connection = sqlite3.connect(database_path)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prediction_events)")
        }
    finally:
        connection.close()

    assert not {"request_body", "feature_values", "uploaded_csv"} & columns
    result = prediction_monitoring_summary(database_path)
    assert result["requests"] == 2
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["average_latency_ms"] == 12.75
    assert result["groups"][0]["error_code"] in {"ValueError", None}


def test_monitoring_endpoint_limits_time_window_and_returns_no_input_data(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "monitoring.sqlite"
    record_prediction_event(
        database_path,
        model_id="model-privacy-safe-id",
        operation="csv_prediction",
        status="failed",
        error_code="ValueError",
        row_count=1,
        latency_ms=10,
    )
    monkeypatch.setattr(api_module, "PREDICTION_MONITORING_DB_PATH", database_path)

    response = client.get("/monitoring/predictions?lookback_days=30")

    assert response.status_code == 200
    report = response.json()
    assert report["failed"] == 1
    assert "private medical value" not in response.text
    invalid_window = client.get("/monitoring/predictions?lookback_days=365")
    assert invalid_window.status_code == 400


def test_old_monitoring_events_are_pruned_on_new_prediction(tmp_path):
    database_path = tmp_path / "monitoring.sqlite"
    record_prediction_event(
        database_path,
        model_id="old",
        operation="csv_prediction",
        status="succeeded",
        error_code=None,
        row_count=1,
        latency_ms=1,
    )
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE prediction_events SET created_at = ?",
            ("2000-01-01T00:00:00+00:00",),
        )
        connection.commit()
    finally:
        connection.close()

    record_prediction_event(
        database_path,
        model_id="new",
        operation="csv_prediction",
        status="succeeded",
        error_code=None,
        row_count=1,
        latency_ms=1,
    )

    result = prediction_monitoring_summary(database_path)
    assert result["requests"] == 1
    assert result["groups"][0]["model_id"] == "new"


def test_monitoring_storage_error_is_reported_without_raising(monkeypatch):
    def fail_write(*_args, **_kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(api_module, "record_prediction_event", fail_write)

    recorded = api_module._record_prediction(
        model_id="model-1",
        operation="csv_prediction",
        status="succeeded",
        error_code=None,
        row_count=1,
        started_at=api_module.time.perf_counter(),
    )

    assert recorded is False
