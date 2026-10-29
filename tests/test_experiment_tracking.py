import sys
from types import SimpleNamespace

from src.experiment_tracking import log_model_run


def test_tracking_is_explicitly_disabled_without_tracking_uri(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    result = log_model_run(
        run_name="test",
        model_name="test_model",
        task_type="classification",
        target_column="label",
        parameters={},
        metrics={},
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "model.metadata.json",
    )

    assert result["status"] == "disabled"
    assert "MLFLOW_TRACKING_URI" in result["reason"]


def test_tracking_logs_model_metrics_and_parameters_when_configured(
    tmp_path,
    monkeypatch,
):
    calls = []

    class Run:
        info = SimpleNamespace(run_id="run-123")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(("uri", uri)),
        set_experiment=lambda name: calls.append(("experiment", name)),
        start_run=lambda run_name: (
            calls.append(("run", run_name)) or Run()
        ),
        set_tags=lambda tags: calls.append(("tags", tags)),
        log_param=lambda name, value: calls.append(("param", name, value)),
        log_metric=lambda name, value: calls.append(("metric", name, value)),
        log_artifact=lambda path, artifact_path: calls.append(
            ("artifact", path, artifact_path)
        ),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.test")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.metadata.json"
    model_path.write_bytes(b"model")
    metadata_path.write_text("{}", encoding="utf-8")

    result = log_model_run(
        run_name="model-run",
        model_name="model",
        task_type="classification",
        target_column="label",
        parameters={"random_state": 42},
        metrics={"accuracy": 0.9, "confusion_matrix": [[2, 0]]},
        model_path=model_path,
        metadata_path=metadata_path,
    )

    assert result == {
        "status": "logged",
        "run_id": "run-123",
        "experiment_name": "test-experiment",
    }
    assert ("param", "random_state", 42) in calls
    assert ("metric", "accuracy", 0.9) in calls
    assert not any(
        call[0] == "metric" and call[1] == "confusion_matrix"
        for call in calls
    )
