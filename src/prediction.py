from __future__ import annotations

import hashlib
import platform
import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class PredictionModel:
    model_id: str
    name: str
    version: str
    source: str
    path: Path
    model: Any
    metadata: dict[str, Any]


def _column_names(model: Any) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    preprocessor = None
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps.get("preprocessor")
    names = getattr(preprocessor, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]

    columns = []
    for _, _, selected in getattr(preprocessor, "transformers_", []):
        if isinstance(selected, str):
            columns.append(selected)
        else:
            columns.extend(str(name) for name in selected)
    return list(dict.fromkeys(columns))


def _categorical_value_types(preprocessor: Any) -> dict[str, str]:
    value_types: dict[str, str] = {}
    for transformer_name, transformer, selected in getattr(
        preprocessor,
        "transformers_",
        [],
    ):
        if transformer_name not in {"categorical", "categorical_features"}:
            continue
        columns = [selected] if isinstance(selected, str) else list(selected)
        encoder = None
        if isinstance(transformer, Pipeline):
            encoder = next(
                (
                    step
                    for _, step in transformer.steps
                    if hasattr(step, "categories_")
                ),
                None,
            )
        categories = getattr(encoder, "categories_", [])
        for index, column in enumerate(columns):
            observed = categories[index] if index < len(categories) else []
            if len(observed) == 0:
                value_types[str(column)] = "string"
                continue
            first = observed[0]
            if isinstance(first, (str, np.str_)):
                value_types[str(column)] = "string"
            elif isinstance(first, (bool, np.bool_)):
                value_types[str(column)] = "boolean"
            elif isinstance(first, numbers.Number):
                value_types[str(column)] = "number"
            else:
                value_types[str(column)] = "unknown"
    return value_types


def _feature_schema(model: Any, feature_names: list[str]) -> list[dict[str, str]]:
    preprocessor = (
        model.named_steps.get("preprocessor")
        if isinstance(model, Pipeline)
        else None
    )
    roles: dict[str, str] = {}
    for transformer_name, _, selected in getattr(
        preprocessor,
        "transformers_",
        [],
    ):
        if isinstance(selected, str):
            columns = [selected]
        else:
            try:
                columns = list(selected)
            except TypeError:
                columns = []
        if transformer_name in {"numeric", "numerical", "numeric_features"}:
            role = "number"
        elif transformer_name in {"categorical", "categorical_features"}:
            role = "categorical"
        else:
            role = "unknown"
        roles.update({str(column): role for column in columns})

    value_types = _categorical_value_types(preprocessor)
    if preprocessor is None and feature_names:
        roles = {column: "number" for column in feature_names}

    return [
        {
            "name": name,
            "data_type": (
                value_types[name]
                if roles.get(name) == "categorical"
                and value_types.get(name) not in {None, "unknown"}
                else roles.get(name, "unknown")
            ),
            "role": roles.get(name, "unknown"),
        }
        for name in feature_names
    ]


def _preprocessing_description(model: Any) -> list[str]:
    preprocessor = (
        model.named_steps.get("preprocessor")
        if isinstance(model, Pipeline)
        else None
    )
    if preprocessor is None:
        return ["No embedded preprocessing; raw numeric features."]

    descriptions = []
    for name, transformer, selected in getattr(preprocessor, "transformers_", []):
        if isinstance(selected, str):
            columns = [selected]
        else:
            try:
                columns = list(selected)
            except TypeError:
                columns = []
        if name == "remainder" and not columns:
            continue
        steps = transformer.steps if isinstance(transformer, Pipeline) else []
        operations = []
        for step_name, step in steps:
            if step_name == "imputer":
                operations.append(
                    f"{getattr(step, 'strategy', 'configured')} imputation"
                )
            elif step_name == "scaler":
                operations.append(type(step).__name__)
            elif step_name == "encoder":
                operations.append(
                    "one-hot encoding (unknown categories: "
                    f"{getattr(step, 'handle_unknown', 'error')})"
                )
        if operations:
            descriptions.append(
                f"{name} features ({', '.join(map(str, columns))}): "
                + " -> ".join(operations)
            )
        else:
            descriptions.append(
                f"{name} features ({', '.join(map(str, columns))}): "
                f"{type(transformer).__name__}"
            )
    return descriptions or ["Embedded preprocessing is not declared."]


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _read_sidecar(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path.with_name(model_path.name + ".metadata.json")
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")
    return metadata


def artifact_sha256(model_path: Path) -> str:
    digest = hashlib.sha256()
    with model_path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compatibility_warnings(
    model: Any,
    model_path: Path,
    metadata: dict[str, Any],
) -> list[str]:
    expected_hash = metadata.get("artifact_sha256")
    if expected_hash and artifact_sha256(model_path) != expected_hash:
        raise ValueError(
            "Model artifact integrity check failed; the saved file does not "
            "match its registered SHA-256."
        )

    messages = list(
        getattr(model, "_runtime_compatibility_warnings", [])
    )
    training_environment = metadata.get("training_environment")
    if not training_environment:
        messages.append(
            "Legacy model: training library versions and integrity hash were "
            "not recorded."
        )
    else:
        trained_sklearn = training_environment.get("scikit_learn")
        if trained_sklearn and trained_sklearn != sklearn.__version__:
            messages.append(
                f"Trained with scikit-learn {trained_sklearn}; current runtime "
                f"is {sklearn.__version__}."
            )
        trained_python = training_environment.get("python")
        if trained_python and trained_python.split(".")[:2] != platform.python_version().split(".")[:2]:
            messages.append(
                f"Trained with Python {trained_python}; current runtime is "
                f"{platform.python_version()}."
            )
    return list(dict.fromkeys(messages))


def describe_prediction_model(
    model: Any,
    model_path: Path,
    *,
    name: str,
    version: str,
    source: str,
    model_id: str,
    feature_frame: pd.DataFrame | None = None,
    task_type: str | None = None,
    target_labels: list[Any] | None = None,
) -> dict[str, Any]:
    if not callable(getattr(model, "predict", None)):
        raise ValueError("The model does not provide predict().")

    metadata = _read_sidecar(model_path)
    compatibility_warnings = _compatibility_warnings(
        model,
        model_path,
        metadata,
    )
    features = _column_names(model)
    if not features and feature_frame is not None:
        features = [str(column) for column in feature_frame.columns]
    if not features:
        raise ValueError("The model artifact does not declare feature names.")

    schema = _feature_schema(model, features)
    if feature_frame is not None:
        for feature in schema:
            if feature["data_type"] == "unknown":
                column = feature["name"]
                if pd.api.types.is_numeric_dtype(feature_frame[column]):
                    feature["data_type"] = "number"
                    feature["role"] = "number"
                elif pd.api.types.is_bool_dtype(feature_frame[column]):
                    feature["data_type"] = "boolean"
                    feature["role"] = "categorical"
                else:
                    feature["data_type"] = "string"
                    feature["role"] = "categorical"
    unsupported = [item["name"] for item in schema if item["data_type"] == "unknown"]
    if unsupported:
        raise ValueError(
            "The required input types could not be verified for: "
            + ", ".join(unsupported)
        )

    task = task_type or metadata.get(
        "task_type",
        "classification" if is_classifier(model) else "regression",
    )
    if task not in {"classification", "regression"}:
        raise ValueError("The model declares an unsupported prediction task.")

    classes = getattr(model, "classes_", None)
    output_labels = (
        [_json_value(label) for label in classes]
        if classes is not None
        else (
            [_json_value(label) for label in target_labels]
            if target_labels is not None
            else metadata.get("target_labels", [])
        )
    )
    preprocessing = metadata.get("preprocessing")
    if preprocessing is None:
        preprocessing = _preprocessing_description(model)

    metrics = metadata.get("evaluation_metrics")
    metrics_source = metadata.get(
        "metrics_source",
        "Evaluation metrics were not stored with this model artifact.",
    )
    return {
        "model_id": model_id,
        "name": metadata.get("model_name", name),
        "version": metadata.get("model_version", version),
        "source": source,
        "task": task,
        "required_features": schema,
        "preprocessing": preprocessing,
        "target_column": metadata.get("target_column"),
        "expected_target_labels": output_labels,
        "evaluation_metrics": metrics,
        "metrics_source": metrics_source,
        "probabilities_supported": bool(
            task == "classification"
            and callable(getattr(model, "predict_proba", None))
        ),
        "compatibility_warnings": compatibility_warnings,
    }


def _model_identifier(
    root: Path,
    path: Path,
    name: str,
    version: str,
) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    digest = hashlib.sha256(
        f"{relative}\0{name}\0{version}".encode("utf-8")
    ).hexdigest()[:20]
    return f"model_{digest}"


def _model_entries(
    models_dir: Path,
    registry_entries: list[dict[str, Any]],
) -> tuple[list[tuple[Path, str, str, str]], list[dict[str, str]]]:
    root = models_dir.resolve()
    entries: list[tuple[Path, str, str, str]] = []
    issues: list[dict[str, str]] = []
    registered_paths: set[Path] = set()
    blocked_paths: set[Path] = set()

    for entry in registry_entries:
        model_path = Path(entry.get("path", ""))
        if not model_path.is_absolute():
            model_path = root / model_path
        try:
            resolved = model_path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            issues.append(
                {
                    "name": str(entry.get("name", "Registered model")),
                    "reason": "The registered artifact is missing or outside the model directory.",
                }
            )
            continue
        if not resolved.is_file():
            issues.append(
                {
                    "name": str(entry.get("name", "Registered model")),
                    "reason": "The registered artifact is not a file.",
                }
            )
            continue
        registered_paths.add(resolved)
        expected_hash = entry.get("artifact_sha256")
        if expected_hash and artifact_sha256(resolved) != expected_hash:
            blocked_paths.add(resolved)
            issues.append(
                {
                    "name": str(entry.get("name", resolved.stem)),
                    "reason": (
                        "Model artifact integrity check failed; the saved file "
                        "differs from the version recorded in the registry."
                    ),
                }
            )
            continue
        entries.append(
            (
                resolved,
                str(entry.get("name", resolved.stem)),
                str(entry.get("version", "registered")),
                "registry",
            )
        )

    model_paths = [
        path
        for suffix in ("*.joblib", "*.pkl", "*.pickle")
        for path in models_dir.rglob(suffix)
    ]
    for model_path in sorted(model_paths):
        try:
            resolved = model_path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved not in registered_paths and resolved not in blocked_paths:
            entries.append((resolved, resolved.stem, "local", "saved artifact"))

    return entries, issues


def list_prediction_models(
    models_dir: Path,
    registry_entries: list[dict[str, Any]],
    model_loader: Callable[[Path], Any],
) -> dict[str, list[dict[str, Any]]]:
    models: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    entries, registry_issues = _model_entries(models_dir, registry_entries)
    issues.extend(registry_issues)

    for path, name, version, source in entries:
        model_id = _model_identifier(models_dir, path, name, version)
        try:
            model = model_loader(path)
            description = describe_prediction_model(
                model,
                path,
                name=name,
                version=version,
                source=source,
                model_id=model_id,
            )
            models.append(description)
        except Exception as error:
            issues.append(
                {
                    "name": name,
                    "reason": (
                        f"{type(error).__name__}: {error}"
                        if isinstance(error, ValueError)
                        else (
                            f"Artifact could not be loaded "
                            f"({type(error).__name__}); check its integrity "
                            "and runtime compatibility."
                        )
                    ),
                }
            )
    return {"models": models, "unavailable_models": issues}


def resolve_prediction_model(
    model_id: str,
    models_dir: Path,
    registry_entries: list[dict[str, Any]],
    model_loader: Callable[[Path], Any],
) -> tuple[PredictionModel, dict[str, Any]]:
    entries, _ = _model_entries(models_dir, registry_entries)
    for path, name, version, source in entries:
        candidate_id = _model_identifier(models_dir, path, name, version)
        if candidate_id != model_id:
            continue
        model = model_loader(path)
        description = describe_prediction_model(
            model,
            path,
            name=name,
            version=version,
            source=source,
            model_id=candidate_id,
        )
        return (
            PredictionModel(
                model_id=candidate_id,
                name=description["name"],
                version=description["version"],
                source=source,
                path=path,
                model=model,
                metadata=description,
            ),
            description,
        )
    raise LookupError("Compatible prediction model was not found.")


def validate_prediction_frame(
    frame: pd.DataFrame,
    required_features: list[dict[str, str]],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("The prediction CSV must contain at least one row.")
    if not frame.columns.is_unique:
        raise ValueError("The prediction CSV contains duplicate column names.")

    required = [feature["name"] for feature in required_features]
    missing = [column for column in required if column not in frame.columns]
    extra = [str(column) for column in frame.columns if column not in required]
    if missing or extra:
        raise ValueError(
            f"Prediction feature mismatch. Missing: {missing}; Extra: {extra}"
        )

    validated = frame.loc[:, required].copy()
    for feature in required_features:
        column = feature["name"]
        expected = feature["data_type"]
        values = validated[column]
        if expected == "number":
            converted = pd.to_numeric(values, errors="coerce")
            invalid = values.notna() & converted.isna()
            if invalid.any():
                rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()[:10]
                raise ValueError(
                    f"Feature '{column}' must be numeric. Invalid CSV row(s): {rows}."
                )
            finite = converted.notna() & ~np.isfinite(converted)
            if finite.any():
                rows = (np.flatnonzero(finite.to_numpy()) + 2).tolist()[:10]
                raise ValueError(
                    f"Feature '{column}' contains non-finite numeric values "
                    f"on CSV row(s): {rows}."
                )
            validated[column] = converted
        elif expected == "string":
            validated[column] = values.astype("string")
        elif expected == "boolean":
            normalized = values.map(
                lambda value: value.strip().lower()
                if isinstance(value, str)
                else value
            )
            mapping = {"true": True, "false": False, "1": True, "0": False}
            converted = normalized.map(
                lambda value: mapping.get(value, value)
                if isinstance(value, str)
                else value
            )
            invalid = converted.notna() & ~converted.map(
                lambda value: isinstance(value, (bool, np.bool_))
            )
            if invalid.any():
                rows = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()[:10]
                raise ValueError(
                    f"Feature '{column}' must be boolean. Invalid CSV row(s): {rows}."
                )
            validated[column] = converted
        else:
            raise ValueError(
                f"Feature '{column}' has an unsupported declared type '{expected}'."
            )
    return validated


def run_csv_prediction(
    model: Any,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    features = validate_prediction_frame(
        frame,
        metadata["required_features"],
    )
    predictions = model.predict(features)
    probabilities = None
    classes = metadata["expected_target_labels"]
    supports_probabilities = metadata["probabilities_supported"]
    if supports_probabilities:
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
            raise ValueError("The model returned an invalid probability result.")
        if classes and probabilities.shape[1] != len(classes):
            raise ValueError("The model returned probabilities with an unknown class schema.")

    results = []
    for index, prediction in enumerate(predictions):
        row: dict[str, Any] = {
            "row": index + 1,
            "prediction": _json_value(prediction),
            "confidence": (
                float(np.max(probabilities[index]))
                if probabilities is not None
                else None
            ),
        }
        if probabilities is not None:
            row["probabilities"] = {
                str(label): float(probabilities[index, class_index])
                for class_index, label in enumerate(classes)
            }
        results.append(row)

    return {
        "model_id": metadata["model_id"],
        "model_name": metadata["name"],
        "model_version": metadata["version"],
        "task": metadata["task"],
        "prediction_count": len(results),
        "classes": classes,
        "probabilities_supported": probabilities is not None,
        "compatibility_warnings": metadata.get("compatibility_warnings", []),
        "results": results,
    }


def create_model_metadata(
    description: dict[str, Any],
    *,
    metrics: dict[str, Any],
    metrics_source: str,
    model_path: Path | None = None,
) -> dict[str, Any]:
    metadata = {
        "model_name": description["name"],
        "model_version": description["version"],
        "task_type": description["task"],
        "target_column": description.get("target_column"),
        "required_features": description["required_features"],
        "preprocessing": description["preprocessing"],
        "target_labels": description["expected_target_labels"],
        "evaluation_metrics": metrics,
        "metrics_source": metrics_source,
        "probabilities_supported": description["probabilities_supported"],
        "training_environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    if model_path is not None:
        metadata["artifact_sha256"] = artifact_sha256(model_path)
    return metadata
