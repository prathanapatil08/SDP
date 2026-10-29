
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
import api.main as api_module
from api.main import app
import src.training_engine as training_engine

client = TestClient(app, headers={"X-API-Key": "test-only-api-key"})


class IncompatibleBaseline:
    def predict(self, features):
        raise ValueError("columns are missing: {'legacy_feature'}")


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_workflow_status_endpoint():
    response = client.get("/workflow-status")
    assert response.status_code == 200
    assert response.json()["workflow"] == "COMPLETED"


def test_drift_report_endpoint():
    response = client.get("/drift-report")
    assert response.status_code == 200
    data = response.json()
    assert "features_checked" in data
    assert "drifted_features" in data
    assert "results" in data


def test_model_comparison_endpoint():
    response = client.get("/model-comparison")
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "candidate" in data


def test_promotion_decision_endpoint():
    response = client.get("/promotion-decision")
    assert response.status_code == 200
    data = response.json()
    assert "action" in data
    assert "deployment_authorized" in data
    assert data["deployment_authorized"] is False


def test_invalid_dataset_rejected():
    response = client.post(
        "/run-drift-check",
        json={"dataset": "invalid"}
    )
    assert response.status_code == 400


def test_dataset_upload_endpoint(tmp_path, monkeypatch):
    cleaned_dir = tmp_path / "cleaned"
    uploads_dir = tmp_path / "uploads"
    monkeypatch.setattr(api_module, "CLEANED_DATA_DIR", cleaned_dir)
    monkeypatch.setattr(training_engine, "UPLOADS_DIR", uploads_dir)
    response = client.post(
        "/dataset/upload",
        files={
            "file": (
                "sample.csv",
                b"age,Churn\n22,No\n34,Yes\n45,No\n52,Yes\n52,Yes\n",
                "text/csv",
            )
        },
        data={"target_column": "Churn", "task_type": "classification"},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "valid"
    assert result["row_count"] == 4
    assert result["task_type"] == "classification"
    assert result["cleaning_report"]["rows_before"] == 5
    assert result["cleaning_report"]["duplicates_removed"] == 1

    cleaned_path = cleaned_dir / result["cleaned_dataset_filename"]
    try:
        assert not list(uploads_dir.iterdir())
        assert cleaned_path.is_file()
        assert result["cleaning_report"]["rows_before"] == 5
        assert len(pd.read_csv(cleaned_path)) == 4
        download_response = client.get(
            f"/dataset/cleaned/{cleaned_path.name}"
        )
        assert download_response.status_code == 200
        assert len(pd.read_csv(pd.io.common.StringIO(download_response.text))) == 4
    finally:
        cleaned_path.unlink(missing_ok=True)


def test_cleaned_dataset_download_rejects_path_traversal():
    response = client.get("/dataset/cleaned/..%5Csecret.csv")

    assert response.status_code == 400


def test_cleaned_dataset_can_be_deleted_without_touching_neighbor_files(
    tmp_path,
    monkeypatch,
):
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    cleaned_path = cleaned_dir / "sample.csv"
    other_path = cleaned_dir / "keep.txt"
    cleaned_path.write_text("feature\n1\n", encoding="utf-8")
    other_path.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(api_module, "CLEANED_DATA_DIR", cleaned_dir)

    deleted = client.delete("/dataset/cleaned/sample.csv")
    invalid = client.delete("/dataset/cleaned/..%5Ckeep.csv")

    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert not cleaned_path.exists()
    assert other_path.read_text(encoding="utf-8") == "keep"
    assert invalid.status_code == 400


def test_generic_drift_endpoint_excludes_target_column():
    reference = (
        b"sepal_length,sepal_width,species\n"
        b"4.8,3.0,setosa\n"
        b"5.0,3.2,setosa\n"
        b"5.1,3.1,versicolor\n"
        b"5.4,3.4,versicolor\n"
    )
    incoming = (
        b"sepal_length,sepal_width,species\n"
        b"4.8,3.0,virginica\n"
        b"5.0,3.2,virginica\n"
        b"5.1,3.1,virginica\n"
        b"5.4,3.4,virginica\n"
    )

    response = client.post(
        "/drift-check",
        files={
            "reference_file": ("reference.csv", reference, "text/csv"),
            "incoming_file": ("incoming.csv", incoming, "text/csv"),
        },
        data={"target_column": "species", "auto_retrain": "false"},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["overall_status"] == "no_significant_drift"
    assert result["features_checked"] == 2
    assert result["features_flagged"] == 0
    assert result["target_column_excluded"] == "species"
    assert {item["feature"] for item in result["results"]} == {
        "sepal_length",
        "sepal_width",
    }


def test_generic_drift_endpoint_rejects_mismatched_columns():
    response = client.post(
        "/drift-check",
        files={
            "reference_file": ("reference.csv", b"feature_a\n1\n2\n", "text/csv"),
            "incoming_file": ("incoming.csv", b"feature_b\n1\n2\n", "text/csv"),
        },
    )

    assert response.status_code == 400
    assert "Feature column mismatch" in response.json()["detail"]


def test_generic_drift_endpoint_accepts_existing_incoming_dataset(
    tmp_path,
    monkeypatch,
):
    incoming_dir = tmp_path / "incoming"
    incoming_dir.mkdir()
    pd_data = "measurement,kind\n1.0,small\n2.0,large\n"
    (incoming_dir / "existing.csv").write_text(pd_data, encoding="utf-8")
    monkeypatch.setattr(api_module, "DATA_DIR", tmp_path)

    response = client.post(
        "/drift-check",
        files={
            "reference_file": (
                "reference.csv",
                pd_data.encode("utf-8"),
                "text/csv",
            )
        },
        data={"incoming_dataset": "existing.csv", "auto_retrain": "false"},
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["incoming_file"] == "existing.csv"
    assert result["overall_status"] == "no_significant_drift"
    assert result["features_checked"] == 2


def test_generic_drift_endpoint_triggers_candidate_retraining_once(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "RETRAINING_DB_PATH",
        tmp_path / "retraining_events.sqlite",
    )
    monkeypatch.setattr(api_module, "MODELS_DIR", tmp_path / "models")

    reference = pd.DataFrame(
        {
            "measurement": [float(value) for value in range(60)],
            "label": ["yes" if value % 2 else "no" for value in range(60)],
        }
    )
    incoming = reference.copy()
    incoming["measurement"] += 100
    incoming["label"] = "incoming-only"
    upload = {
        "reference_file": (
            "reference.csv",
            reference.to_csv(index=False).encode("utf-8"),
            "text/csv",
        ),
        "incoming_file": (
            "incoming.csv",
            incoming.to_csv(index=False).encode("utf-8"),
            "text/csv",
        ),
    }
    form = {
        "target_column": "label",
        "drift_threshold_percent": "25",
        "retrain_model_name": "logistic_regression",
    }

    first = client.post(
        "/drift-check",
        files=upload,
        data=form,
    )
    assert first.status_code == 200, first.text
    first_result = first.json()
    assert first_result["drift_percentage"] > 25
    assert first_result["retraining"]["training_status"] == "completed"
    assert first_result["retraining"]["deployment_authorized"] is False
    assert Path(first_result["retraining"]["candidate_model_path"]).is_file()
    candidate = joblib.load(first_result["retraining"]["candidate_model_path"])
    assert set(candidate.classes_) == {"no", "yes"}

    second = client.post(
        "/drift-check",
        files={
            "reference_file": (
                "reference.csv",
                reference.to_csv(index=False).encode("utf-8"),
                "text/csv",
            ),
            "incoming_file": (
                "incoming.csv",
                incoming.to_csv(index=False).encode("utf-8"),
                "text/csv",
            ),
        },
        data=form,
    )
    assert second.status_code == 200, second.text
    second_event = second.json()["retraining"]
    assert second_event["event_id"] == first_result["retraining"]["event_id"]
    assert second_event["duplicate_event"] is True
    assert len(list((tmp_path / "models").glob("candidate_*.joblib"))) == 1
    stored_events = client.get("/retraining-events").json()["events"]
    assert len(stored_events) == 1
    assert stored_events[0]["training_status"] == "completed"


def test_training_rejects_incompatible_baseline_before_saving_candidate(
    tmp_path,
    monkeypatch,
):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    experiments_dir = tmp_path / "experiments"
    baseline_path = models_dir / "incompatible.joblib"
    joblib.dump(IncompatibleBaseline(), baseline_path)
    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "EXPERIMENTS", experiments_dir)
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    monkeypatch.setattr(api_module, "CLEANED_DATA_DIR", cleaned_dir)
    dataset_path = cleaned_dir / "training.csv"
    pd.DataFrame(
        [
            {"measurement": float(index), "label": "yes" if index % 2 else "no"}
            for index in range(20)
        ]
    ).to_csv(dataset_path, index=False)

    response = client.post(
        "/train/generic",
        json={
            "dataset_file": "training.csv",
            "target_column": "label",
            "model_name": "logistic_regression",
            "task_type": "classification",
            "baseline_model_path": str(baseline_path),
        },
    )

    assert response.status_code == 400
    assert "baseline model is incompatible" in response.json()["detail"].lower()
    assert "Select 'None'" in response.json()["detail"]
    assert not list(models_dir.glob("candidate_*.joblib"))


def test_training_without_baseline_succeeds(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    experiments_dir = tmp_path / "experiments"
    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "EXPERIMENTS", experiments_dir)
    monkeypatch.setattr(api_module, "CLEANED_DATA_DIR", tmp_path / "cleaned")

    response = client.post(
        "/train/generic",
        json={
            "dataset": [
                {"measurement": float(index), "label": "yes" if index % 2 else "no"}
                for index in range(20)
            ],
            "target_column": "label",
            "model_name": "logistic_regression",
            "task_type": "classification",
            "random_state": 42,
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["cleaning_report"]["status"] == "ready"
    assert Path(result["cleaned_dataset_path"]).is_file()
    assert Path(result["model_path"]).is_file()
    model_metadata = json.loads(
        Path(result["model_path"] + ".metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert model_metadata["required_features"][0]["name"] == "measurement"
    assert model_metadata["dataset_provenance"]["rows_after_cleaning"] == 20
    assert model_metadata["evaluation_metrics"]["accuracy"] == result["metrics"]["accuracy"]


def test_generic_training_fits_only_on_training_partition(tmp_path, monkeypatch):
    fitted_rows = {}
    predicted_rows = {}

    class RecordingPipeline:
        def fit(self, features, labels):
            fitted_rows["features"] = set(features["measurement"].tolist())
            fitted_rows["labels"] = len(labels)
            self.prediction = labels.mode().iloc[0]
            return self

        def predict(self, features):
            predicted_rows["features"] = set(features["measurement"].tolist())
            return [self.prediction] * len(features)

    def save_test_artifact(_model, path):
        Path(path).write_bytes(b"test artifact")

    monkeypatch.setattr(
        api_module,
        "build_training_pipeline",
        lambda *args, **kwargs: RecordingPipeline(),
    )
    monkeypatch.setattr(api_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(api_module, "EXPERIMENTS", tmp_path / "experiments")
    monkeypatch.setattr(api_module, "CLEANED_DATA_DIR", tmp_path / "cleaned")
    monkeypatch.setattr(api_module.joblib, "dump", save_test_artifact)
    samples = [
        {
            "measurement": float(index),
            "label": "yes" if index % 2 else "no",
        }
        for index in range(40)
    ]

    response = client.post(
        "/train/generic",
        json={
            "dataset": samples,
            "target_column": "label",
            "model_name": "logistic_regression",
            "task_type": "classification",
            "test_size": 0.25,
            "random_state": 17,
        },
    )

    assert response.status_code == 200, response.text
    result = response.json()
    all_features = {row["measurement"] for row in samples}
    assert len(fitted_rows["features"]) == result["train_rows"]
    assert fitted_rows["features"].isdisjoint(predicted_rows["features"])
    assert fitted_rows["features"] | predicted_rows["features"] == all_features
    assert result["test_rows"] > 0
