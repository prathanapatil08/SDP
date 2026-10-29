from __future__ import annotations

import json
import re
import uuid
from inspect import isawaitable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT / "uploads"
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest_classifier": RandomForestClassifier,
    "decision_tree_classifier": DecisionTreeClassifier,
    "gradient_boosting_classifier": GradientBoostingClassifier,
}

REGRESSION_MODELS = {
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
    "decision_tree_regressor": DecisionTreeRegressor,
    "gradient_boosting_regressor": GradientBoostingRegressor,
}


def ensure_runtime_directories() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_upload_name(filename: str | None) -> str:
    if not filename:
        return "uploaded_dataset.csv"

    candidate = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    if not cleaned:
        cleaned = "uploaded_dataset.csv"
    if "." not in cleaned:
        cleaned = f"{cleaned}.csv"
    return cleaned


async def save_uploaded_csv(uploaded_file, max_size_bytes: int = 10 * 1024 * 1024) -> str:
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    if hasattr(uploaded_file, "getvalue"):
        payload = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        payload = uploaded_file.read()
        if isawaitable(payload):
            payload = await payload
        if hasattr(uploaded_file, "seek"):
            seek_result = uploaded_file.seek(0)
            if isawaitable(seek_result):
                await seek_result
    else:
        payload = b""

    if len(payload) == 0:
        raise ValueError("The uploaded file is empty.")
    if len(payload) > max_size_bytes:
        raise ValueError(
            f"Uploaded files must be smaller than {max_size_bytes / (1024 * 1024):.0f} MB."
        )

    ensure_runtime_directories()
    destination = UPLOADS_DIR / f"{uuid.uuid4().hex}_{sanitize_upload_name(getattr(uploaded_file, 'name', None))}"
    destination.write_bytes(payload)
    return str(destination)


def save_uploaded_model(uploaded_file, max_size_bytes: int = 20 * 1024 * 1024) -> str:
    if uploaded_file is None:
        raise ValueError("No model file was uploaded.")

    if hasattr(uploaded_file, "getvalue"):
        payload = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        payload = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    else:
        payload = b""

    if len(payload) == 0:
        raise ValueError("The uploaded model file is empty.")
    if len(payload) > max_size_bytes:
        raise ValueError(
            f"Model uploads must be smaller than {max_size_bytes / (1024 * 1024):.0f} MB."
        )

    suffix = Path(getattr(uploaded_file, "name", "model.joblib")).suffix.lower()
    if suffix not in {".joblib", ".pkl", ".pickle"}:
        raise ValueError("Only .joblib, .pkl, or .pickle model files are supported.")

    ensure_runtime_directories()
    destination = UPLOADS_DIR / f"{uuid.uuid4().hex}_model_{sanitize_upload_name(getattr(uploaded_file, 'name', None))}"
    destination.write_bytes(payload)
    return str(destination)


def infer_task_type(target: pd.Series) -> str:
    if target is None or target.empty:
        raise ValueError("The target column is empty.")

    cleaned = target.dropna()
    if cleaned.empty:
        raise ValueError("The target column does not contain valid values.")

    if pd.api.types.is_bool_dtype(cleaned) or pd.api.types.is_object_dtype(cleaned):
        unique_values = cleaned.nunique(dropna=True)
        return "classification" if unique_values <= 20 and unique_values > 1 else "regression"

    if pd.api.types.is_numeric_dtype(cleaned):
        unique_values = cleaned.nunique(dropna=True)
        if unique_values <= 20 and unique_values > 1:
            return "classification"
        return "regression"

    return "classification" if cleaned.nunique(dropna=True) <= 20 else "regression"


def validate_dataset_frame(
    df: pd.DataFrame,
    target_column: str,
    task_type: str | None = None,
) -> dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("The dataset is empty or missing.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the dataset.")

    cleaned = df.dropna(subset=[target_column]).copy()
    if cleaned.empty:
        raise ValueError("The target column has no non-missing values.")

    if cleaned.shape[1] < 2:
        raise ValueError("At least one feature column is required for training.")

    inferred = infer_task_type(cleaned[target_column]) if task_type is None else task_type
    if task_type is not None:
        inferred = task_type

    if inferred == "classification":
        unique_targets = cleaned[target_column].dropna().nunique()
        if unique_targets < 2:
            raise ValueError("Classification requires at least two target classes.")
        if unique_targets > 50:
            raise ValueError(
                "Classification targets should have a manageable number of classes; please choose a different task type."
            )
        cleaned[target_column] = cleaned[target_column].astype(str)
    else:
        numeric_target = pd.to_numeric(cleaned[target_column], errors="coerce")
        if numeric_target.isna().all():
            raise ValueError("Regression requires a numeric target column.")
        cleaned[target_column] = numeric_target

    feature_frame = cleaned.drop(columns=[target_column], errors="ignore")
    if feature_frame.empty:
        raise ValueError("No feature columns are available after removing the target.")

    missing_summary = {
        str(column): int(feature_frame[column].isna().sum())
        for column in feature_frame.columns
        if feature_frame[column].isna().sum() > 0
    }

    return {
        "task_type": inferred,
        "row_count": int(len(cleaned)),
        "column_count": int(len(cleaned.columns)),
        "dtypes": cleaned.dtypes.astype(str).to_dict(),
        "n_missing": int(cleaned.isna().sum().sum()),
        "missing_summary": missing_summary,
        "duplicate_rows": int(cleaned.duplicated().sum()),
        "target_unique_count": int(cleaned[target_column].nunique(dropna=True)),
        "data": cleaned,
    }


def _numeric_and_categorical_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = frame.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numeric_columns = [
        column
        for column in frame.columns
        if column not in categorical_columns and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return numeric_columns, categorical_columns


def build_preprocessor(frame: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_columns, categorical_columns = _numeric_and_categorical_columns(frame)
    steps: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        steps.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        steps.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    if not steps:
        raise ValueError("The dataset does not contain usable feature columns for preprocessing.")

    return ColumnTransformer(transformers=steps, remainder="drop")


def make_estimator(task_type: str, model_name: str, random_state: int = 42, hyperparameters: dict[str, Any] | None = None) -> Any:
    hyperparameters = dict(hyperparameters or {})
    estimator_map = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    try:
        model_key = model_name.lower().strip()
        estimator_class = estimator_map[model_key]
    except KeyError as exc:
        supported = sorted(estimator_map)
        raise ValueError(f"Unsupported {task_type} model '{model_name}'. Supported: {supported}") from exc

    params: dict[str, Any] = {}
    if model_key in {
        "logistic_regression",
        "random_forest_classifier",
        "decision_tree_classifier",
        "gradient_boosting_classifier",
        "random_forest_regressor",
        "decision_tree_regressor",
        "gradient_boosting_regressor",
    }:
        params["random_state"] = random_state
    params.update(hyperparameters)

    if model_key == "logistic_regression":
        params.setdefault("max_iter", 1000)
        return estimator_class(**params)
    if model_key == "random_forest_classifier":
        params.setdefault("n_estimators", 200)
        return estimator_class(**params)
    if model_key == "decision_tree_classifier":
        return estimator_class(**params)
    if model_key == "gradient_boosting_classifier":
        return estimator_class(**params)
    if model_key == "linear_regression":
        return estimator_class(**params)
    if model_key == "random_forest_regressor":
        params.setdefault("n_estimators", 200)
        return estimator_class(**params)
    if model_key == "decision_tree_regressor":
        return estimator_class(**params)
    if model_key == "gradient_boosting_regressor":
        return estimator_class(**params)

    raise ValueError(f"Model configuration for '{model_name}' is not supported.")


def build_training_pipeline(
    X: pd.DataFrame,
    task_type: str,
    model_name: str,
    random_state: int = 42,
    hyperparameters: dict[str, Any] | None = None,
) -> Pipeline:
    scale_numeric = model_name.lower() in {
        "logistic_regression",
        "linear_regression",
    }
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
            ("model", make_estimator(task_type, model_name, random_state=random_state, hyperparameters=hyperparameters)),
        ]
    )


def compute_classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
    }


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    errors = y_pred - np.asarray(y_true)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "residual_mean": float(errors.mean()),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    task_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = validate_dataset_frame(df, target_column, task_type=task_type)
    task = validation["task_type"]
    data = validation["data"]

    X = data.drop(columns=[target_column])
    y = data[target_column]

    stratify = y if task == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = build_training_pipeline(
        X_train,
        task,
        model_name,
        random_state=random_state,
        hyperparameters=hyperparameters,
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    if task == "classification":
        metrics = compute_classification_metrics(y_test, predictions)
    else:
        metrics = compute_regression_metrics(y_test, predictions)

    result = {
        "task_type": task,
        "target_column": target_column,
        "model_name": model_name,
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "model": pipeline,
    }
    return result


def save_model_artifact(model: Pipeline, name: str, version: str | None = None) -> str:
    ensure_runtime_directories()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "trained_model"
    suffix = version.strip() if version else "latest"
    safe_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", suffix).strip("._") or "latest"
    model_path = MODELS_DIR / f"{safe_name}_{safe_suffix}_{uuid.uuid4().hex[:8]}.joblib"
    joblib.dump(model, model_path)
    return str(model_path)


def compare_models(
    baseline_model: Any,
    candidate_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str,
) -> dict[str, Any]:
    baseline_predictions = baseline_model.predict(X_test)
    candidate_predictions = candidate_model.predict(X_test)

    if task_type == "classification":
        baseline_metrics = compute_classification_metrics(y_test, baseline_predictions)
        candidate_metrics = compute_classification_metrics(y_test, candidate_predictions)
        winner_metric = "f1"
        baseline_value = baseline_metrics[winner_metric]
        candidate_value = candidate_metrics[winner_metric]
        improvement = candidate_value - baseline_value
        winner = "candidate" if candidate_value > baseline_value else "baseline"
    else:
        baseline_metrics = compute_regression_metrics(y_test, baseline_predictions)
        candidate_metrics = compute_regression_metrics(y_test, candidate_predictions)
        winner_metric = "rmse"
        baseline_value = baseline_metrics[winner_metric]
        candidate_value = candidate_metrics[winner_metric]
        improvement = baseline_value - candidate_value
        winner = "candidate" if candidate_value < baseline_value else "baseline"

    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "winner_metric": winner_metric,
        "baseline_value": float(baseline_value),
        "candidate_value": float(candidate_value),
        "absolute_improvement": float(abs(candidate_value - baseline_value)),
        "relative_improvement": float(
            ((candidate_value - baseline_value) / baseline_value) * 100
        ) if baseline_value else 0.0,
        "improvement_direction": float(improvement),
        "winner": winner,
        "task_type": task_type,
    }


def decision_gate(comparison: dict[str, Any], task_type: str, required_gain: float = 0.01) -> dict[str, Any]:
    if task_type == "classification":
        metric_name = "f1"
        candidate_value = comparison["candidate"][metric_name]
        baseline_value = comparison["baseline"][metric_name]
        improvement = candidate_value - baseline_value
        decision = "PROMOTION_RECOMMENDED" if improvement >= required_gain else "RETAIN_BASELINE"
        return {
            "action": decision,
            "required_gain": required_gain,
            "candidate_metric": candidate_value,
            "baseline_metric": baseline_value,
            "improvement": improvement,
            "deployment_authorized": decision == "PROMOTION_RECOMMENDED",
        }

    metric_name = "rmse"
    candidate_value = comparison["candidate"][metric_name]
    baseline_value = comparison["baseline"][metric_name]
    improvement = baseline_value - candidate_value
    decision = "PROMOTION_RECOMMENDED" if improvement >= required_gain else "RETAIN_BASELINE"
    return {
        "action": decision,
        "required_gain": required_gain,
        "candidate_metric": candidate_value,
        "baseline_metric": baseline_value,
        "improvement": improvement,
        "deployment_authorized": decision == "PROMOTION_RECOMMENDED",
    }


def evaluated_model_payload(model_result: dict[str, Any], *, output_path: str | Path | None = None) -> dict[str, Any]:
    payload = {
        "task_type": model_result["task_type"],
        "target_column": model_result["target_column"],
        "model_name": model_result["model_name"],
        "test_size": model_result["test_size"],
        "random_state": model_result["random_state"],
        "train_rows": model_result["train_rows"],
        "test_rows": model_result["test_rows"],
        "metrics": model_result["metrics"],
    }

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload
