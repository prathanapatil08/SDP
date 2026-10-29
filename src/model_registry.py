
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY_FILE = MODELS_DIR / "registry.json"


def _load_registry():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not REGISTRY_FILE.exists():
        return {"models": []}

    with REGISTRY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def register_model(name, version, model_path, framework="scikit-learn"):
    path = Path(model_path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    registry = _load_registry()
    metadata_path = path.with_name(path.name + ".metadata.json")
    metadata = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError("Model metadata must be a JSON object.")

    entry = {
        "name": name,
        "version": version,
        "path": str(path.resolve()),
        "framework": framework,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": _artifact_sha256(path),
        "schema": metadata.get("required_features"),
        "task_type": metadata.get("task_type"),
        "target_labels": metadata.get("target_labels"),
        "evaluation_metrics": metadata.get("evaluation_metrics"),
        "training_environment": metadata.get("training_environment"),
    }

    # Replace an existing entry with the same name and version.
    registry["models"] = [
        item for item in registry["models"]
        if not (
            item["name"] == name
            and item["version"] == version
        )
    ]

    registry["models"].append(entry)

    with REGISTRY_FILE.open("w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2)

    return entry


def list_models():
    return _load_registry()["models"]


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    register_model(
        name="customer-churn",
        version="1.0",
        model_path="models/baseline_model.joblib",
    )

    register_model(
        name="customer-churn",
        version="2.0",
        model_path="models/candidate_model.joblib",
    )

    print(json.dumps(list_models(), indent=2))
