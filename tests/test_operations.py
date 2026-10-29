import json
import sqlite3
from pathlib import Path

from src.operations import create_operations_backup, upload_retention_policy
from src.data_cleaning import cleanup_expired_cleaned_datasets


def test_operations_backup_copies_registry_and_consistent_sqlite_database(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    registry_path = root / "models" / "registry.json"
    registry_path.parent.mkdir()
    registry_path.write_text(
        json.dumps({"models": [{"name": "sample"}]}),
        encoding="utf-8",
    )
    database_path = root / "mlflow.db"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO runs VALUES ('run-1')")
    connection.commit()
    connection.close()

    result = create_operations_backup(
        root=root,
        registry_path=registry_path,
        mlflow_database_path=database_path,
    )

    assert result["status"] == "completed"
    backups = {item["kind"]: root / "backups" / item["filename"] for item in result["files"]}
    assert json.loads(backups["model_registry"].read_text(encoding="utf-8")) == {
        "models": [{"name": "sample"}]
    }
    backup_db = sqlite3.connect(backups["mlflow_database"])
    try:
        assert backup_db.execute("SELECT run_id FROM runs").fetchone() == ("run-1",)
    finally:
        backup_db.close()


def test_registry_backup_is_rejected_if_registry_is_invalid(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{", encoding="utf-8")

    try:
        create_operations_backup(
            root=tmp_path,
            registry_path=registry_path,
            mlflow_database_path=tmp_path / "missing.db",
        )
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Invalid registry should not be backed up.")


def test_expired_cleaned_datasets_are_removed_without_touching_other_files(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    expired_csv = cleaned / "expired.csv"
    active_csv = cleaned / "active.csv"
    other_file = cleaned / "notes.txt"
    expired_csv.write_text("x\n1\n", encoding="utf-8")
    active_csv.write_text("x\n2\n", encoding="utf-8")
    other_file.write_text("keep", encoding="utf-8")
    expired_time = 1_000_000_000
    import os

    os.utime(expired_csv, (expired_time, expired_time))

    removed = cleanup_expired_cleaned_datasets(
        directory=cleaned,
        max_age_seconds=0,
    )

    assert removed == 2
    assert not expired_csv.exists()
    assert not active_csv.exists()
    assert other_file.is_file()


def test_retention_policy_documents_no_prediction_data_persistence():
    policy = upload_retention_policy()

    assert "Deleted after" in policy["temporary_source_uploads"]
    assert "not persisted" in policy["prediction_csv"]
    assert "feature values are never stored" in policy["prediction_logs"]
