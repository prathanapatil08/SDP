
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from io import BytesIO, StringIO
import csv
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import warnings

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from src.drift_detector import detect_drift
from src.data_cleaning import (
    CLEANED_DATA_DIR,
    CleaningConfig,
    cleanup_expired_cleaned_datasets,
    clean_dataset,
    save_cleaned_dataset,
)
from src.api_security import LocalAdminKeyMiddleware
from src.experiment_tracking import log_model_run
from src.model_registry import list_models, register_model
from src.operations import create_operations_backup, upload_retention_policy
from src.prediction_monitoring import (
    prediction_monitoring_summary,
    record_prediction_event,
)
from src.prediction import (
    create_model_metadata,
    describe_prediction_model,
    list_prediction_models,
    resolve_prediction_model,
    run_csv_prediction,
)
from src.retraining import (
    DEFAULT_DRIFT_THRESHOLD_PERCENT,
    build_drift_event_id,
    list_retraining_events,
    run_drift_retraining,
)
from src.training_engine import (
    build_training_pipeline,
    compare_models,
    compute_classification_metrics,
    compute_regression_metrics,
    decision_gate,
    save_uploaded_csv,
    save_uploaded_model,
    validate_dataset_frame,
)


# --------------------------------------------------
# PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MAX_DRIFT_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PREDICTION_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PREDICTION_ROWS = 100_000
RETRAINING_DB_PATH = EXPERIMENTS / "automatic_retraining.sqlite"
PREDICTION_MONITORING_DB_PATH = EXPERIMENTS / "prediction_monitoring.sqlite"
MAX_REQUEST_BODY_BYTES = 22 * 1024 * 1024
logger = logging.getLogger("autonomous_mlops.api")

app = FastAPI(
    title="Autonomous MLOps API",
    description=(
        "API for data drift detection, model registration, "
        "model evaluation, and predictions."
    ),
    version="2.0.0",
)
app.add_middleware(LocalAdminKeyMiddleware)


@app.middleware("http")
async def apply_request_safeguards(request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    {"detail": "Request body exceeds the 22 MB service limit."},
                    status_code=413,
                )
        except ValueError:
            return JSONResponse(
                {"detail": "Request content length is invalid."},
                status_code=400,
            )

    started = time.perf_counter()
    request_id_value = request.headers.get("x-request-id") or os.urandom(12).hex()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id_value
        return response
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "api_request request_id=%s method=%s path=%s status=%s latency_ms=%.2f",
            request_id_value,
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class DriftCheckRequest(BaseModel):
    dataset: str


class ModelRegistrationRequest(BaseModel):
    name: str
    version: str
    model_path: str
    framework: str = "scikit-learn"


class PredictionRequest(BaseModel):
    records: list[dict]


class DatasetValidationRequest(BaseModel):
    dataset: list[dict] | None = None
    target_column: str
    task_type: str | None = None


class GenericTrainingRequest(BaseModel):
    dataset: list[dict] | None = None
    dataset_file: str | None = None
    target_column: str
    model_name: str = "random_forest_classifier"
    task_type: str | None = None
    test_size: float = 0.2
    random_state: int = 42
    hyperparameters: dict | None = None
    baseline_model_path: str | None = None
    remove_exact_duplicates: bool = True
    convert_numeric_columns: bool = True
    outlier_iqr_multiplier: float = 1.5


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def read_json(filename):
    file_path = EXPERIMENTS / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found.",
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {filename}: {error}",
        )


def count_drifted_features(df):
    if "drift_detected" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="Drift report is missing the drift_detected column.",
        )

    return int(
        df["drift_detected"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )


def read_drift_report():
    report_path = EXPERIMENTS / "drift_report.csv"

    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Drift report not found.",
        )

    try:
        return pd.read_csv(report_path)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read drift report: {error}",
        )


def resolve_project_model_path(model_path):
    """
    Resolve a model path and ensure it stays inside models/.
    """
    supplied_path = Path(model_path)

    if supplied_path.is_absolute():
        resolved_path = supplied_path.resolve()
    else:
        resolved_path = (ROOT / supplied_path).resolve()

    try:
        resolved_path.relative_to(MODELS_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Model files must be located inside the project's models folder.",
        )

    if not resolved_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Model file not found.",
        )

    return resolved_path


# --------------------------------------------------
# CACHED MODEL LOADING
# --------------------------------------------------

@lru_cache(maxsize=8)
def load_registered_model(model_path, modified_time_ns):
    """
    Load a trusted registered model once and reuse it.
    A file modification changes the cache key.
    """
    return joblib.load(model_path)


def load_prediction_model(model_path: Path):
    stat = model_path.stat()
    observed_warnings = []
    for attempt in range(2):
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model = load_registered_model(
                    str(model_path),
                    stat.st_mtime_ns,
                )
            observed_warnings.extend(
                str(warning.message)
                for warning in captured
                if "version" in str(warning.message).lower()
            )
            try:
                model._runtime_compatibility_warnings = observed_warnings
            except AttributeError:
                pass
            return model
        except OSError:
            if attempt == 1:
                raise
            time.sleep(0.1)

    raise RuntimeError("Model could not be loaded after a transient retry.")


def _record_prediction(
    *,
    model_id: str,
    operation: str,
    status: str,
    error_code: str | None,
    row_count: int | None,
    started_at: float,
) -> bool:
    try:
        record_prediction_event(
            PREDICTION_MONITORING_DB_PATH,
            model_id=model_id,
            operation=operation,
            status=status,
            error_code=error_code,
            row_count=row_count,
            latency_ms=(time.perf_counter() - started_at) * 1000,
        )
        return True
    except (OSError, sqlite3.Error) as error:
        logger.error(
            "prediction_monitoring_write_failed error_type=%s",
            type(error).__name__,
        )
        return False


def _database_path_from_tracking_uri() -> Path:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri.startswith("sqlite:///"):
        return ROOT / "mlflow.db"
    configured = tracking_uri.removeprefix("sqlite:///")
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Autonomous MLOps API is running",
        "status": "healthy",
    }


# --------------------------------------------------
# WORKFLOW AND REPORT ENDPOINTS
# --------------------------------------------------

@app.get("/workflow-status")
def workflow_status():
    return read_json("workflow_status.json")


@app.get("/drift-report")
def drift_report():
    df = read_drift_report()

    return {
        "features_checked": int(len(df)),
        "drifted_features": count_drifted_features(df),
        "results": df.fillna("").to_dict(orient="records"),
    }


@app.get("/model-comparison")
def model_comparison():
    return read_json("model_comparison.json")


@app.get("/promotion-decision")
def promotion_decision():
    return read_json("promotion_decision.json")


@app.get("/drift-metadata")
def drift_metadata():
    return read_json("drift_metadata.json")


# --------------------------------------------------
# RUN DRIFT DETECTION
# --------------------------------------------------

@app.post("/run-drift-check")
def run_drift_check(request: DriftCheckRequest):
    allowed_datasets = {
        "normal": (
            DATA_DIR / "incoming" / "incoming_customers.csv"
        ),
        "drifted": (
            DATA_DIR / "incoming" / "incoming_customers_drifted.csv"
        ),
    }

    if request.dataset not in allowed_datasets:
        raise HTTPException(
            status_code=400,
            detail="Choose either 'normal' or 'drifted'.",
        )

    incoming_path = allowed_datasets[request.dataset]
    detector_path = ROOT / "src" / "drift_detector.py"

    if not incoming_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {incoming_path.name}",
        )

    if not detector_path.is_file():
        raise HTTPException(
            status_code=500,
            detail="Drift detector script was not found.",
        )

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(detector_path),
                "--incoming",
                str(incoming_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Drift detection failed. "
                    f"Error: {result.stderr[-2000:]}"
                ),
            )

        report_df = read_drift_report()
        drifted_count = count_drifted_features(report_df)

        analysis_time = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        metadata = {
            "dataset": request.dataset,
            "dataset_file": incoming_path.name,
            "analysis_time": analysis_time,
            "features_checked": int(len(report_df)),
            "drifted_features": drifted_count,
        }

        EXPERIMENTS.mkdir(parents=True, exist_ok=True)

        metadata_path = EXPERIMENTS / "drift_metadata.json"

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        history_path = EXPERIMENTS / "drift_history.csv"

        history_record = {
            "analysis_time": analysis_time,
            "dataset": request.dataset,
            "dataset_file": incoming_path.name,
            "features_checked": int(len(report_df)),
            "drifted_features": drifted_count,
        }

        file_exists = (
            history_path.exists()
            and history_path.stat().st_size > 0
        )

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history_record.keys()),
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(history_record)

        return {
            "status": "completed",
            **metadata,
            "results": report_df.fillna("").to_dict(
                orient="records"
            ),
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Drift detection took longer than 120 seconds.",
        )
    except HTTPException:
        raise
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not run drift detection: {error}",
        )


@app.post("/drift-check")
async def run_uploaded_dataset_drift_check(
    reference_file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
    incoming_file: UploadFile | None = File(default=None),
    incoming_dataset: str | None = Form(default=None),
    auto_retrain: bool = Form(default=True),
    drift_threshold_percent: float = Form(
        default=DEFAULT_DRIFT_THRESHOLD_PERCENT,
    ),
    retrain_model_name: str | None = Form(default=None),
):
    if (incoming_file is None) == (incoming_dataset is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one incoming CSV upload or existing incoming dataset.",
        )
    if not 0 <= drift_threshold_percent <= 100:
        raise HTTPException(
            status_code=400,
            detail="Drift threshold must be between 0 and 100 percent.",
        )

    def read_csv_upload(
        file: UploadFile,
        label: str,
    ) -> tuple[pd.DataFrame, bytes]:
        if not file.filename or Path(file.filename).suffix.lower() != ".csv":
            raise HTTPException(
                status_code=400,
                detail=f"{label} must be a CSV file.",
            )

        payload = file.file.read(MAX_DRIFT_UPLOAD_BYTES + 1)
        if len(payload) > MAX_DRIFT_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{label} must be no larger than 10 MB.",
            )
        if not payload:
            raise HTTPException(status_code=400, detail=f"{label} is empty.")

        try:
            return pd.read_csv(BytesIO(payload)), payload
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise HTTPException(
                status_code=400,
                detail=f"{label} is not a readable CSV: {error}",
            ) from error

    reference, reference_bytes = read_csv_upload(
        reference_file,
        "Reference dataset",
    )

    if incoming_file is not None:
        incoming, incoming_bytes = read_csv_upload(
            incoming_file,
            "Incoming dataset",
        )
        incoming_name = incoming_file.filename
    else:
        if not incoming_dataset or Path(incoming_dataset).name != incoming_dataset:
            raise HTTPException(
                status_code=400,
                detail="Choose a valid incoming dataset filename.",
            )
        incoming_path = DATA_DIR / "incoming" / incoming_dataset
        if not incoming_path.is_file() or incoming_path.suffix.lower() != ".csv":
            raise HTTPException(
                status_code=404,
                detail=f"Incoming dataset not found: {incoming_dataset}",
            )
        try:
            incoming_bytes = incoming_path.read_bytes()
            incoming = pd.read_csv(BytesIO(incoming_bytes))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise HTTPException(
                status_code=400,
                detail=f"Incoming dataset is not a readable CSV: {error}",
            ) from error
        incoming_name = incoming_path.name

    try:
        report = detect_drift(
            reference,
            incoming,
            target_column=target_column or None,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    feature_count = len(report)
    flagged_count = int(report["drift_detected"].sum())
    compared_count = int(report["comparison_status"].eq("compared").sum())
    if flagged_count:
        overall_status = "drift_detected"
    elif compared_count:
        overall_status = "no_significant_drift"
    else:
        overall_status = "inconclusive"

    safe_report = report.astype(object).where(pd.notna(report), None)
    response = {
        "overall_status": overall_status,
        "reference_file": reference_file.filename,
        "incoming_file": incoming_name,
        "target_column_excluded": target_column or None,
        "reference_rows": int(len(reference)),
        "incoming_rows": int(len(incoming)),
        "features_checked": feature_count,
        "features_flagged": flagged_count,
        "features_tested": compared_count,
        "drift_percentage": (
            100.0 * flagged_count / feature_count if feature_count else 0.0
        ),
        "results": safe_report.to_dict(orient="records"),
    }
    if auto_retrain:
        event_id = build_drift_event_id(
            reference_bytes,
            incoming_bytes,
            target_column,
            retrain_model_name,
            drift_threshold_percent,
        )
        response["retraining"] = run_drift_retraining(
            reference,
            event_id=event_id,
            reference_name=reference_file.filename or "reference.csv",
            incoming_name=incoming_name or "incoming.csv",
            target_column=target_column,
            requested_model_name=retrain_model_name,
            drift_percentage=response["drift_percentage"],
            threshold_percent=drift_threshold_percent,
            db_path=RETRAINING_DB_PATH,
            models_dir=MODELS_DIR,
        )
    else:
        response["retraining"] = {
            "training_status": "disabled",
            "trigger_reason": "Automatic retraining was disabled for this drift check.",
            "model_type": retrain_model_name or "automatic",
            "error": None,
            "deployment_authorized": False,
        }
    return response


@app.get("/retraining-events")
def retraining_events():
    return {"events": list_retraining_events(RETRAINING_DB_PATH)}


@app.get("/prediction/models")
def prediction_models():
    return list_prediction_models(
        MODELS_DIR,
        list_models(),
        load_prediction_model,
    )


@app.post("/predict/csv")
async def predict_uploaded_csv(
    file: UploadFile = File(...),
    model_id: str = Form(...),
):
    started_at = time.perf_counter()
    row_count = None
    try:
        if not file.filename or Path(file.filename).suffix.lower() != ".csv":
            raise HTTPException(
                status_code=400,
                detail="Only CSV prediction uploads are supported.",
            )

        payload = file.file.read(MAX_PREDICTION_UPLOAD_BYTES + 1)
        if len(payload) > MAX_PREDICTION_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Prediction CSV files must be no larger than 10 MB.",
            )
        if not payload:
            raise HTTPException(
                status_code=400,
                detail="The prediction CSV is empty.",
            )

        headers = next(csv.reader(StringIO(payload.decode("utf-8-sig"))))
        if not headers or any(not header.strip() for header in headers):
            raise HTTPException(
                status_code=400,
                detail="The prediction CSV must have non-empty column names.",
            )
        if len(headers) != len(set(headers)):
            raise HTTPException(
                status_code=400,
                detail="The prediction CSV contains duplicate column names.",
            )
        dataframe = pd.read_csv(BytesIO(payload))
        row_count = len(dataframe)
        if row_count > MAX_PREDICTION_ROWS:
            raise HTTPException(
                status_code=413,
                detail=f"Prediction CSVs are limited to {MAX_PREDICTION_ROWS} rows.",
            )
        model, metadata = resolve_prediction_model(
            model_id,
            MODELS_DIR,
            list_models(),
            load_prediction_model,
        )
        result = run_csv_prediction(model.model, dataframe, metadata)
        result["monitoring_recorded"] = _record_prediction(
            model_id=model_id,
            operation="csv_prediction",
            status="succeeded",
            error_code=None,
            row_count=row_count,
            started_at=started_at,
        )
        return result
    except LookupError as error:
        _record_prediction(
            model_id=model_id,
            operation="csv_prediction",
            status="failed",
            error_code="model_not_found",
            row_count=row_count,
            started_at=started_at,
        )
        raise HTTPException(status_code=404, detail=str(error)) from error
    except HTTPException as error:
        _record_prediction(
            model_id=model_id,
            operation="csv_prediction",
            status="failed",
            error_code=f"http_{error.status_code}",
            row_count=row_count,
            started_at=started_at,
        )
        raise
    except (
        StopIteration,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        _record_prediction(
            model_id=model_id,
            operation="csv_prediction",
            status="failed",
            error_code=type(error).__name__,
            row_count=row_count,
            started_at=started_at,
        )
        raise HTTPException(
            status_code=400,
            detail="The prediction upload is not a readable UTF-8 CSV.",
        ) from error
    except ValueError as error:
        _record_prediction(
            model_id=model_id,
            operation="csv_prediction",
            status="failed",
            error_code="validation_or_integrity_error",
            row_count=row_count,
            started_at=started_at,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Prediction CSV validation failed: {error}",
        ) from error
    except Exception as error:
        _record_prediction(
            model_id=model_id,
            operation="csv_prediction",
            status="failed",
            error_code=type(error).__name__,
            row_count=row_count,
            started_at=started_at,
        )
        logger.error(
            "csv_prediction_failed model_id=%s error_type=%s",
            model_id[:100],
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed because the model artifact or inference "
                "pipeline could not be used. No alternative model was selected."
            ),
        ) from error


@app.get("/monitoring/predictions")
def prediction_monitoring(lookback_days: int = 30):
    try:
        return prediction_monitoring_summary(
            PREDICTION_MONITORING_DB_PATH,
            lookback_days=lookback_days,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction monitoring data is temporarily unavailable.",
        ) from error


@app.get("/operations/retention")
def operations_retention():
    return upload_retention_policy()


@app.post("/operations/backup")
def operations_backup():
    try:
        return create_operations_backup(
            root=ROOT,
            registry_path=MODELS_DIR / "registry.json",
            mlflow_database_path=_database_path_from_tracking_uri(),
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        logger.error(
            "operations_backup_failed error_type=%s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="The requested local backup could not be completed.",
        ) from error


# --------------------------------------------------
# DRIFT HISTORY
# --------------------------------------------------

@app.get("/drift-history")
def drift_history():
    history_path = EXPERIMENTS / "drift_history.csv"

    if not history_path.is_file():
        return {
            "total_analyses": 0,
            "history": [],
        }

    try:
        history_df = pd.read_csv(history_path).fillna("")
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read drift history: {error}",
        )

    return {
        "total_analyses": int(len(history_df)),
        "history": history_df.to_dict(orient="records"),
    }


@app.post("/dataset/validate")
def validate_uploaded_dataset(request: DatasetValidationRequest):
    if request.dataset is None:
        raise HTTPException(
            status_code=400,
            detail="Provide a dataset payload with one or more rows.",
        )

    df = pd.DataFrame(request.dataset)
    cleaned, cleaning_report = clean_dataset(
        df,
        target_column=request.target_column,
        task_type=request.task_type,
    )
    if cleaning_report["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail={
                "message": cleaning_report["errors"][0],
                "cleaning_report": cleaning_report,
            },
        )
    validation = validate_dataset_frame(
        cleaned,
        request.target_column,
        task_type=request.task_type,
    )

    return {
        "status": "valid",
        "task_type": validation["task_type"],
        "row_count": validation["row_count"],
        "column_count": validation["column_count"],
        "duplicate_rows": validation["duplicate_rows"],
        "missing_summary": validation["missing_summary"],
        "cleaning_report": cleaning_report,
    }


@app.post("/dataset/upload")
async def upload_dataset_file(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str | None = Form(default=None),
    remove_exact_duplicates: bool = Form(default=True),
    convert_numeric_columns: bool = Form(default=True),
    outlier_iqr_multiplier: float = Form(default=1.5),
):
    if file.filename is None or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV uploads are supported for datasets.",
        )

    try:
        cleanup_expired_cleaned_datasets(directory=CLEANED_DATA_DIR)
        dataset_path = await save_uploaded_csv(file)
        df = pd.read_csv(dataset_path)
        cleaned, cleaning_report = clean_dataset(
            df,
            target_column=target_column,
            task_type=task_type,
            config=CleaningConfig(
                remove_exact_duplicates=remove_exact_duplicates,
                convert_numeric_columns=convert_numeric_columns,
                outlier_iqr_multiplier=outlier_iqr_multiplier,
            ),
        )
        if cleaning_report["status"] != "ready":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": cleaning_report["errors"][0],
                    "cleaning_report": cleaning_report,
                },
            )
        validation = validate_dataset_frame(
            cleaned,
            target_column,
            task_type=task_type,
        )
        cleaned_path = save_cleaned_dataset(
            cleaned,
            file.filename,
            directory=CLEANED_DATA_DIR,
        )
        return {
            "status": "valid",
            "cleaned_dataset_filename": cleaned_path.name,
            "task_type": validation["task_type"],
            "row_count": validation["row_count"],
            "column_count": validation["column_count"],
            "duplicate_rows": validation["duplicate_rows"],
            "missing_summary": validation["missing_summary"],
            "columns": list(df.columns),
            "dtypes": validation["dtypes"],
            "cleaning_report": cleaning_report,
        }
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.error(
            "dataset_upload_failed error_type=%s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Could not process the dataset upload.",
        ) from error
    finally:
        if "dataset_path" in locals():
            Path(dataset_path).unlink(missing_ok=True)


@app.get("/dataset/cleaned/{filename}")
def download_cleaned_dataset(filename: str):
    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Choose a valid cleaned CSV filename.")
    cleaned_path = CLEANED_DATA_DIR / filename
    if not cleaned_path.is_file():
        raise HTTPException(status_code=404, detail="Cleaned dataset was not found.")
    return FileResponse(
        cleaned_path,
        media_type="text/csv",
        filename=filename,
    )


@app.delete("/dataset/cleaned/{filename}")
def delete_cleaned_dataset(filename: str):
    if Path(filename).name != filename or not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Choose a valid cleaned CSV filename.")
    cleaned_path = CLEANED_DATA_DIR / filename
    if not cleaned_path.is_file():
        raise HTTPException(status_code=404, detail="Cleaned dataset was not found.")
    cleaned_path.unlink()
    return {"status": "deleted", "filename": filename}


@app.post("/train/generic")
def train_generic_model(request: GenericTrainingRequest):
    if request.dataset is not None:
        dataframe = pd.DataFrame(request.dataset)
    elif request.dataset_file:
        supplied_path = Path(request.dataset_file)
        if supplied_path.is_absolute():
            dataset_path = supplied_path.resolve()
        else:
            if supplied_path.name != str(supplied_path):
                raise HTTPException(
                    status_code=400,
                    detail="Dataset files must be selected by their cleaned filename.",
                )
            dataset_path = (CLEANED_DATA_DIR / supplied_path.name).resolve()
        try:
            dataset_path.relative_to(CLEANED_DATA_DIR.resolve())
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail="Training datasets must be inside the cleaned uploads directory.",
            ) from error
        if not dataset_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Training dataset was not found or has expired.",
            )
        dataframe = pd.read_csv(dataset_path)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a dataset payload or dataset_file path for training.",
        )

    try:
        cleanup_expired_cleaned_datasets(directory=CLEANED_DATA_DIR)
        dataframe, cleaning_report = clean_dataset(
            dataframe,
            target_column=request.target_column,
            task_type=request.task_type,
            config=CleaningConfig(
                remove_exact_duplicates=request.remove_exact_duplicates,
                convert_numeric_columns=request.convert_numeric_columns,
                outlier_iqr_multiplier=request.outlier_iqr_multiplier,
            ),
        )
        if cleaning_report["status"] != "ready":
            raise HTTPException(
                status_code=400,
                detail={
                    "message": cleaning_report["errors"][0],
                    "cleaning_report": cleaning_report,
                },
            )

        cleaned_dataset_path = save_cleaned_dataset(
            dataframe,
            Path(request.dataset_file).name if request.dataset_file else "training_data.csv",
            directory=CLEANED_DATA_DIR,
        )
        validation = validate_dataset_frame(
            dataframe,
            request.target_column,
            task_type=request.task_type,
        )
        train_df = validation["data"]
        split_X = train_df.drop(columns=[request.target_column])
        split_y = train_df[request.target_column]

        stratify = (
            split_y
            if request.task_type == "classification" or validation["task_type"] == "classification"
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            split_X,
            split_y,
            test_size=request.test_size,
            random_state=request.random_state,
            stratify=stratify if stratify is not None and split_y.nunique() > 1 else None,
        )
        pipeline = build_training_pipeline(
            X_train,
            validation["task_type"],
            request.model_name,
            random_state=request.random_state,
            hyperparameters=request.hyperparameters,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = (
            compute_classification_metrics(y_test, predictions)
            if validation["task_type"] == "classification"
            else compute_regression_metrics(y_test, predictions)
        )

        baseline_model = None
        if request.baseline_model_path:
            baseline_path = Path(request.baseline_model_path)
            if not baseline_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Baseline model not found: {baseline_path}",
                )
            try:
                baseline_model = joblib.load(baseline_path)
                baseline_model.predict(X_test)
            except ValueError as error:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "The selected baseline model is incompatible with this "
                        "dataset's input features or target type. Select 'None' "
                        "or choose a baseline trained on the same features. "
                        f"Details: {error}"
                    ),
                ) from error
            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not load or validate the selected baseline model: {error}",
                ) from error

        model_out = Path(MODELS_DIR) / f"candidate_{request.model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"

        payload = {
            "status": "completed",
            "task_type": validation["task_type"],
            "target_column": request.target_column,
            "model_name": request.model_name,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "metrics": metrics,
            "model_path": str(model_out),
            "cleaned_dataset_path": str(cleaned_dataset_path),
            "cleaning_report": cleaning_report,
        }

        if baseline_model is not None:
                try:
                    comparison = compare_models(
                        baseline_model,
                        pipeline,
                        X_test,
                        y_test,
                        validation["task_type"],
                    )
                    gate = decision_gate(
                        comparison,
                        validation["task_type"],
                        required_gain=0.0,
                    )
                except ValueError as error:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "The selected baseline model cannot be compared with "
                            "this dataset and task type. Select 'None' or choose a "
                            f"compatible baseline. Details: {error}"
                        ),
                    ) from error
                payload["comparison"] = comparison
                payload["promotion_decision"] = gate

        model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_out)
        prediction_description = describe_prediction_model(
            pipeline,
            model_out,
            name=request.model_name,
            version=model_out.stem,
            source="saved artifact",
            model_id="",
            feature_frame=X_train,
            task_type=validation["task_type"],
            target_labels=(
                sorted(split_y.unique().tolist(), key=str)
                if validation["task_type"] == "classification"
                else None
            ),
        )
        prediction_description["target_column"] = request.target_column
        model_metadata = create_model_metadata(
            prediction_description,
            metrics=metrics,
            metrics_source=(
                f"Held-out test partition ({len(X_test)} rows; "
                f"random_state={request.random_state})."
            ),
            model_path=model_out,
        )
        model_metadata["dataset_provenance"] = {
            "rows_before_cleaning": cleaning_report["rows_before"],
            "rows_after_cleaning": cleaning_report["rows_after"],
            "duplicates_removed": cleaning_report["duplicates_removed"],
            "cleaning_configuration": cleaning_report["configuration"],
        }
        metadata_path = model_out.with_name(model_out.name + ".metadata.json")
        metadata_path.write_text(
            json.dumps(model_metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        payload["tracking"] = log_model_run(
            run_name=f"{request.model_name}_{model_out.stem}",
            model_name=request.model_name,
            task_type=validation["task_type"],
            target_column=request.target_column,
            parameters={
                "test_size": request.test_size,
                "random_state": request.random_state,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "cleaning_remove_duplicates": request.remove_exact_duplicates,
                "cleaning_convert_numeric": request.convert_numeric_columns,
                "cleaning_outlier_iqr_multiplier": request.outlier_iqr_multiplier,
            },
            metrics=metrics,
            model_path=model_out,
            metadata_path=metadata_path,
        )
        EXPERIMENTS.mkdir(parents=True, exist_ok=True)
        experiment_file = EXPERIMENTS / "generic_training.json"
        experiment_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Training failed: {error}") from error


@app.post("/models/upload")
async def upload_model_file(
    file: UploadFile = File(...),
    name: str = Form(...),
    version: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A model file is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".joblib", ".pkl", ".pickle"}:
        raise HTTPException(
            status_code=400,
            detail="This workflow accepts trusted scikit-learn models in .joblib, .pkl, or .pickle format only.",
        )

    try:
        uploaded_path = save_uploaded_model(file)
        model = joblib.load(uploaded_path)
        if not callable(getattr(model, "predict", None)):
            raise HTTPException(
                status_code=400,
                detail="The uploaded object is not a scikit-learn compatible model with predict().",
            )

        final_path = Path(MODELS_DIR) / f"{name}_{version}.joblib"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, final_path)
        entry = register_model(name=name, version=version, model_path=str(final_path), framework="scikit-learn")

        return {
            "status": "registered",
            "message": "Model accepted under a trusted upload workflow.",
            "warning": "Joblib and pickle files can execute arbitrary code during deserialization. Only trust sources you control.",
            "model": entry,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not validate uploaded model: {error}") from error


# --------------------------------------------------
# MODEL REGISTRY
# --------------------------------------------------

@app.get("/models")
def get_registered_models():
    return {"models": list_models()}


@app.post("/models/register")
def register_new_model(request: ModelRegistrationRequest):
    name = request.name.strip()
    version = request.version.strip()
    framework = request.framework.strip().lower()

    if not name or not version:
        raise HTTPException(
            status_code=400,
            detail="Model name and version cannot be empty.",
        )

    if framework != "scikit-learn":
        raise HTTPException(
            status_code=400,
            detail=(
                "This version supports scikit-learn models only. "
                "Additional frameworks need dedicated adapters."
            ),
        )

    model_path = resolve_project_model_path(request.model_path)

    try:
        # Validate that the trusted file loads and supports prediction.
        stat = model_path.stat()
        model = load_registered_model(
            str(model_path),
            stat.st_mtime_ns,
        )

        if not callable(getattr(model, "predict", None)):
            raise HTTPException(
                status_code=400,
                detail="The registered model does not provide predict().",
            )

        entry = register_model(
            name=name,
            version=version,
            model_path=str(model_path),
            framework=framework,
        )

        return {
            "message": "Model registered successfully.",
            "model": entry,
        }

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model file was not found.",
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not validate model file: {error}",
        )


# --------------------------------------------------
# PREDICTIONS USING A REGISTERED MODEL
# --------------------------------------------------

@app.post("/models/{name}/{version}/predict")
def predict_with_registered_model(
    name: str,
    version: str,
    request: PredictionRequest,
):
    started_at = time.perf_counter()
    model_id = f"registry:{name}:{version}"
    if not request.records:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one record in 'records'.",
        )

    registered_models = list_models()

    entry = next(
        (
            item for item in registered_models
            if item.get("name") == name
            and item.get("version") == version
        ),
        None,
    )

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="Registered model version not found.",
        )

    if entry.get("framework", "").lower() != "scikit-learn":
        raise HTTPException(
            status_code=400,
            detail="Prediction currently supports scikit-learn models only.",
        )

    model_path = resolve_project_model_path(entry["path"])

    try:
        model = load_prediction_model(model_path)
        input_data = pd.DataFrame(request.records)
        description = describe_prediction_model(
            model,
            model_path,
            name=name,
            version=version,
            source="registry",
            model_id=model_id,
        )
        prediction = run_csv_prediction(model, input_data, description)

        response = {
            "model_name": name,
            "model_version": version,
            "prediction_count": prediction["prediction_count"],
            "predictions": [
                row["prediction"] for row in prediction["results"]
            ],
            "compatibility_warnings": prediction["compatibility_warnings"],
            "monitoring_recorded": _record_prediction(
                model_id=model_id,
                operation="registered_json_prediction",
                status="succeeded",
                error_code=None,
                row_count=len(request.records),
                started_at=started_at,
            ),
        }

        if prediction["probabilities_supported"]:
            response["probabilities"] = [
                [
                    row["probabilities"][str(label)]
                    for label in prediction["classes"]
                ]
                for row in prediction["results"]
            ]
            response["classes"] = prediction["classes"]

        return response

    except HTTPException:
        raise
    except Exception as error:
        _record_prediction(
            model_id=model_id,
            operation="registered_json_prediction",
            status="failed",
            error_code=type(error).__name__,
            row_count=len(request.records),
            started_at=started_at,
        )
        logger.error(
            "registered_prediction_failed model_id=%s error_type=%s",
            model_id,
            type(error).__name__,
        )
        raise HTTPException(
            status_code=400 if isinstance(error, ValueError) else 500,
            detail=(
                f"Prediction input is incompatible: {error}"
                if isinstance(error, ValueError)
                else (
                    "Prediction failed for the selected registered model. "
                    "No alternative model was selected."
                )
            ),
        ) from error
