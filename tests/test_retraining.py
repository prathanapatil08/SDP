from pathlib import Path

import pandas as pd

import src.retraining as retraining
from src.retraining import build_drift_event_id, run_drift_retraining


def labeled_reference():
    return pd.DataFrame(
        {
            "measurement": [float(value) for value in range(60)],
            "category": ["A" if value % 2 else "B" for value in range(60)],
            "label": ["yes" if value % 2 else "no" for value in range(60)],
        }
    )


def process_event(tmp_path, reference, drift_percentage, event_suffix):
    return run_drift_retraining(
        reference,
        event_id=build_drift_event_id(
            b"reference-" + event_suffix.encode(),
            b"incoming-" + event_suffix.encode(),
            "label",
            "logistic_regression",
            25.0,
        ),
        reference_name="reference.csv",
        incoming_name="incoming.csv",
        target_column="label",
        requested_model_name="logistic_regression",
        drift_percentage=drift_percentage,
        threshold_percent=25.0,
        db_path=tmp_path / "events.sqlite",
        models_dir=tmp_path / "models",
    )


def test_drift_above_threshold_trains_separate_candidate(tmp_path):
    event = process_event(tmp_path, labeled_reference(), 50.0, "trigger")

    assert event["training_status"] == "completed"
    assert event["trigger_reason"].startswith("Feature drift 50.00% exceeded")
    assert event["dataset"] == "reference=reference.csv; incoming=incoming.csv"
    assert event["model_type"] == "logistic_regression"
    assert event["candidate_model_path"].endswith(".joblib")
    assert pd.notna(event["metrics"]["accuracy"])
    assert event["deployment_authorized"] is False
    metadata_path = event["candidate_model_path"] + ".metadata.json"
    assert Path(metadata_path).is_file()


def test_no_drift_does_not_train(tmp_path):
    event = process_event(tmp_path, labeled_reference(), 25.0, "no-drift")

    assert event["training_status"] == "skipped_no_drift"
    assert "did not exceed" in event["trigger_reason"]
    assert event["candidate_model_path"] is None
    assert not list((tmp_path / "models").glob("candidate_*.joblib"))


def test_missing_labels_are_recorded_without_training(tmp_path):
    reference = labeled_reference()
    reference["label"] = None

    event = process_event(tmp_path, reference, 50.0, "missing-labels")

    assert event["training_status"] == "skipped_missing_labels"
    assert event["error"] is not None
    assert "no labeled rows" in event["error"]
    assert event["candidate_model_path"] is None


def test_training_failure_is_recorded(tmp_path, monkeypatch):
    def fail_training(*args, **kwargs):
        raise RuntimeError("deliberate training failure")

    monkeypatch.setattr(retraining, "train_and_evaluate", fail_training)

    event = process_event(tmp_path, labeled_reference(), 50.0, "failure")

    assert event["training_status"] == "failed"
    assert event["error"] == "RuntimeError: deliberate training failure"
    assert event["candidate_model_path"] is None
    assert event["deployment_authorized"] is False


def test_duplicate_drift_event_is_not_retrained(tmp_path):
    reference = labeled_reference()
    first = process_event(tmp_path, reference, 50.0, "duplicate")
    second = process_event(tmp_path, reference, 50.0, "duplicate")

    assert first["training_status"] == "completed"
    assert second["event_id"] == first["event_id"]
    assert second["duplicate_event"] is True
    assert len(list((tmp_path / "models").glob("candidate_*.joblib"))) == 1
