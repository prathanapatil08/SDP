import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_module
from api.main import app
from src.prediction import (
    create_model_metadata,
    describe_prediction_model,
    list_prediction_models,
    run_csv_prediction,
    validate_prediction_frame,
)
from src.training_engine import build_training_pipeline

client = TestClient(app, headers={"X-API-Key": "test-only-api-key"})


def classification_model():
    frame = pd.DataFrame(
        {
            "measurement": [float(value) for value in range(40)],
            "category": ["A" if value % 2 else "B" for value in range(40)],
        }
    )
    labels = pd.Series(
        ["positive" if value % 2 else "negative" for value in range(40)]
    )
    model = build_training_pipeline(
        frame,
        "classification",
        "logistic_regression",
        random_state=17,
    )
    model.fit(frame, labels)
    return model


def test_prediction_uses_embedded_preprocessing_and_returns_probabilities(tmp_path):
    model_path = tmp_path / "model.joblib"
    model = classification_model()
    joblib.dump(model, model_path)
    listed = list_prediction_models(
        tmp_path,
        [],
        lambda path: joblib.load(path),
    )

    assert len(listed["models"]) == 1
    metadata = listed["models"][0]
    assert metadata["task"] == "classification"
    assert metadata["required_features"] == [
        {"name": "measurement", "data_type": "number", "role": "number"},
        {"name": "category", "data_type": "string", "role": "categorical"},
    ]
    assert metadata["probabilities_supported"] is True
    assert any("median imputation" in step for step in metadata["preprocessing"])
    assert "path" not in metadata

    incoming = pd.DataFrame(
        {
            "category": ["A", "unseen"],
            "measurement": ["2", "39"],
        }
    )
    result = run_csv_prediction(model, incoming, metadata)

    assert result["prediction_count"] == 2
    assert result["probabilities_supported"] is True
    assert all(row["confidence"] is not None for row in result["results"])
    assert all(sum(row["probabilities"].values()) == pytest.approx(1.0) for row in result["results"])


def test_regression_prediction_has_no_fabricated_confidence():
    frame = pd.DataFrame(
        {
            "measurement": [float(value) for value in range(20)],
            "category": ["A" if value % 2 else "B" for value in range(20)],
        }
    )
    targets = pd.Series([float(value * 3) for value in range(20)])
    model = build_training_pipeline(
        frame,
        "regression",
        "linear_regression",
        random_state=17,
    )
    model.fit(frame, targets)
    metadata = describe_prediction_model(
        model,
        Path("regression.joblib"),
        name="regressor",
        version="1",
        source="test",
        model_id="regression-model",
        task_type="regression",
    )

    result = run_csv_prediction(
        model,
        pd.DataFrame({"measurement": [2.0], "category": ["unseen"]}),
        metadata,
    )

    assert result["task"] == "regression"
    assert result["probabilities_supported"] is False
    assert result["results"][0]["confidence"] is None
    assert "probabilities" not in result["results"][0]


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (
            pd.DataFrame({"measurement": [1.0], "extra": ["x"]}),
            "Missing: ['category']; Extra: ['extra']",
        ),
        (
            pd.DataFrame({"measurement": ["bad"], "category": ["A"]}),
            "must be numeric",
        ),
        (
            pd.DataFrame({"measurement": [float("inf")], "category": ["A"]}),
            "non-finite",
        ),
    ],
)
def test_prediction_input_schema_and_types_are_validated(tmp_path, frame, error):
    model = classification_model()
    metadata = describe_prediction_model(
        model,
        tmp_path / "model.joblib",
        name="model",
        version="test",
        source="test",
        model_id="test-model",
    )

    with pytest.raises(ValueError) as error_info:
        validate_prediction_frame(frame, metadata["required_features"])
    assert error in str(error_info.value)


def test_prediction_endpoint_lists_schema_and_returns_csv_results(
    tmp_path,
    monkeypatch,
):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "iris_prediction.joblib"
    joblib.dump(classification_model(), model_path)
    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "list_models", lambda: [])

    listing_response = client.get("/prediction/models")

    assert listing_response.status_code == 200
    listing = listing_response.json()
    assert len(listing["models"]) == 1
    model_info = listing["models"][0]
    assert model_info["name"] == "iris_prediction"
    assert "path" not in model_info

    csv_bytes = b"measurement,category\n4,A\n35,unseen\n"
    prediction_response = client.post(
        "/predict/csv",
        files={"file": ("incoming.csv", csv_bytes, "text/csv")},
        data={"model_id": model_info["model_id"]},
    )

    assert prediction_response.status_code == 200, prediction_response.text
    result = prediction_response.json()
    assert result["prediction_count"] == 2
    assert len(result["results"]) == 2
    assert result["results"][0]["prediction"] in {"positive", "negative"}
    assert "path" not in result


def test_prediction_endpoint_reports_missing_extra_and_invalid_columns(
    tmp_path,
    monkeypatch,
):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(classification_model(), models_dir / "model.joblib")
    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "list_models", lambda: [])
    model_id = client.get("/prediction/models").json()["models"][0]["model_id"]

    for data, expected in (
        (b"measurement,extra\n1,A\n", "Missing: ['category']; Extra: ['extra']"),
        (b"measurement,category\nbad,A\n", "must be numeric"),
    ):
        response = client.post(
            "/predict/csv",
            files={"file": ("input.csv", data, "text/csv")},
            data={"model_id": model_id},
        )
        assert response.status_code == 400
        assert expected in response.json()["detail"]


def test_prediction_endpoint_rejects_non_csv_and_oversized_upload(
    tmp_path,
    monkeypatch,
):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "list_models", lambda: [])

    unsupported = client.post(
        "/predict/csv",
        files={"file": ("image.png", b"not a csv", "image/png")},
        data={"model_id": "unselected"},
    )
    assert unsupported.status_code == 400

    monkeypatch.setattr(api_module, "MAX_PREDICTION_UPLOAD_BYTES", 8)
    oversized = client.post(
        "/predict/csv",
        files={"file": ("input.csv", b"feature\n123\n", "text/csv")},
        data={"model_id": "unselected"},
    )
    assert oversized.status_code == 413


def test_unloadable_saved_models_are_reported_without_internal_paths(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "broken.joblib").write_bytes(b"not a model")

    result = list_prediction_models(
        models_dir,
        [],
        lambda _path: (_ for _ in ()).throw(RuntimeError("load failure")),
    )

    assert result["models"] == []
    assert result["unavailable_models"][0]["name"] == "broken"
    assert "runtime compatibility" in result["unavailable_models"][0]["reason"]
    assert str(tmp_path) not in str(result)


def test_model_integrity_hash_blocks_modified_artifact(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "model.joblib"
    model = classification_model()
    joblib.dump(model, model_path)
    description = describe_prediction_model(
        model,
        model_path,
        name="model",
        version="1",
        source="saved artifact",
        model_id="model-1",
    )
    metadata = create_model_metadata(
        description,
        metrics={"accuracy": 0.9},
        metrics_source="test split",
        model_path=model_path,
    )
    model_path.with_name(model_path.name + ".metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    result = list_prediction_models(
        models_dir,
        [],
        lambda path: joblib.load(path),
    )

    assert result["models"] == []
    assert "integrity check failed" in result["unavailable_models"][0]["reason"]


def test_runtime_version_mismatch_is_visible_in_model_description(tmp_path):
    model_path = tmp_path / "model.joblib"
    model = classification_model()
    joblib.dump(model, model_path)
    description = describe_prediction_model(
        model,
        model_path,
        name="model",
        version="1",
        source="saved artifact",
        model_id="model-1",
    )
    metadata = create_model_metadata(
        description,
        metrics={},
        metrics_source="test",
        model_path=model_path,
    )
    metadata["training_environment"]["scikit_learn"] = "0.0"
    model_path.with_name(model_path.name + ".metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    compatible = describe_prediction_model(
        model,
        model_path,
        name="model",
        version="1",
        source="saved artifact",
        model_id="model-1",
    )

    assert any("scikit-learn 0.0" in warning for warning in compatible["compatibility_warnings"])


def test_prediction_api_returns_actionable_error_for_unknown_model():
    response = client.post(
        "/predict/csv",
        files={"file": ("input.csv", b"feature\n1\n", "text/csv")},
        data={"model_id": "not-a-model"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_model_loader_retries_transient_oserror_once(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    calls = []

    def load(_path, _modified_time_ns):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("temporary file access failure")
        return classification_model()

    monkeypatch.setattr(api_module, "load_registered_model", load)
    monkeypatch.setattr(api_module.time, "sleep", lambda _seconds: None)

    loaded = api_module.load_prediction_model(model_path)

    assert callable(loaded.predict)
    assert len(calls) == 2


def test_nontransient_load_failure_does_not_retry(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"model")
    calls = []

    def load(_path, _modified_time_ns):
        calls.append(True)
        raise RuntimeError("incompatible artifact")

    monkeypatch.setattr(api_module, "load_registered_model", load)

    with pytest.raises(RuntimeError, match="incompatible artifact"):
        api_module.load_prediction_model(model_path)

    assert len(calls) == 1


def test_inference_failure_is_recorded_without_model_fallback(
    tmp_path,
    monkeypatch,
):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_path = models_dir / "model.joblib"
    joblib.dump(classification_model(), model_path)
    monkeypatch.setattr(api_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(api_module, "list_models", lambda: [])
    monkeypatch.setattr(
        api_module,
        "PREDICTION_MONITORING_DB_PATH",
        tmp_path / "monitoring.sqlite",
    )
    attempts = []

    def fail_prediction(*_args, **_kwargs):
        attempts.append("selected-model")
        raise RuntimeError("inference failure")

    monkeypatch.setattr(api_module, "run_csv_prediction", fail_prediction)
    model_id = client.get("/prediction/models").json()["models"][0]["model_id"]

    response = client.post(
        "/predict/csv",
        files={
            "file": (
                "input.csv",
                b"measurement,category\n2,A\n",
                "text/csv",
            )
        },
        data={"model_id": model_id},
    )

    assert response.status_code == 500
    assert "No alternative model was selected" in response.json()["detail"]
    assert attempts == ["selected-model"]
    assert api_module.prediction_monitoring_summary(
        tmp_path / "monitoring.sqlite"
    )["failed"] == 1
