
import hashlib
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.training_engine import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    infer_task_type,
    validate_dataset_frame,
)
from dashboard.prediction_dashboard import render_prediction_dashboard
from dashboard.ui import apply_dashboard_styles

ROOT = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000"
MLFLOW_URL = "http://127.0.0.1:5000"
API_TIMEOUT = 30
API_SESSION = requests.Session()
if os.getenv("API_AUTH_TOKEN"):
    API_SESSION.headers["X-API-Key"] = os.environ["API_AUTH_TOKEN"]

st.set_page_config(
    page_title="Autonomous MLOps Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)
apply_dashboard_styles()

st.title("🤖 Autonomous MLOps Dashboard")
st.caption(
    "AI Model Drift Detection and Self-Healing Model Management"
)

dashboard_mode = st.sidebar.radio(
    "Application",
    ["MLOps Dashboard", "Prediction Dashboard"],
    horizontal=False,
)
if dashboard_mode == "Prediction Dashboard":
    render_prediction_dashboard()
    st.stop()

if "generic_training_result" not in st.session_state:
    st.session_state["generic_training_result"] = None
if "generic_training_context" not in st.session_state:
    st.session_state["generic_training_context"] = None
if "generic_drift_result" not in st.session_state:
    st.session_state["generic_drift_result"] = None
if "generic_drift_context" not in st.session_state:
    st.session_state["generic_drift_context"] = None
if "cleaning_upload_result" not in st.session_state:
    st.session_state["cleaning_upload_result"] = None
if "cleaning_upload_context" not in st.session_state:
    st.session_state["cleaning_upload_context"] = None
if "cleaning_approved" not in st.session_state:
    st.session_state["cleaning_approved"] = False


# -------------------- API HELPERS --------------------

def get_api_data(endpoint):
    try:
        response = API_SESSION.get(
            f"{API_URL}/{endpoint}",
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(
            f"Could not retrieve '{endpoint}': {error}. "
            "Check that FastAPI is running."
        )
    except ValueError:
        st.error(f"The API returned invalid JSON for '{endpoint}'.")

    return None


def format_percentage(value):
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def get_mlflow_runs():
    """Retrieve runs from the project's MLflow experiment."""
    try:
        response = requests.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            params={"max_results": 100},
            timeout=10,
        )
        response.raise_for_status()

        experiments = response.json().get("experiments", [])

        experiment = next(
            (
                item for item in experiments
                if item.get("name")
                == "Autonomous-MLOps-Customer-Churn"
            ),
            None,
        )

        if experiment is None:
            return None, "MLflow experiment not found."

        runs_response = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={
                "experiment_ids": [experiment["experiment_id"]],
                "max_results": 100,
                "order_by": ["start_time DESC"],
            },
            timeout=10,
        )
        runs_response.raise_for_status()

        return runs_response.json().get("runs", []), None

    except requests.RequestException as error:
        return None, str(error)


# -------------------- CUSTOM DATASET WORKFLOW --------------------

with st.container():
    st.subheader("Upload and validate a dataset")
    uploaded_csv = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help="CSV files up to 10 MB are supported. Uploaded files are stored in a dedicated runtime directory.",
    )

    uploaded_dataset = None
    target_column = None
    task_choice = "classification"
    validation_status = None

    if uploaded_csv is not None:
        try:
            uploaded_dataset = pd.read_csv(uploaded_csv)
            columns = list(uploaded_dataset.columns)
            target_column = st.selectbox("Target column", columns)
            suggested_task = infer_task_type(uploaded_dataset[target_column])
            task_choice = st.selectbox(
                "Task type",
                ["classification", "regression"],
                index=0 if suggested_task == "classification" else 1,
            )

            validation_status = validate_dataset_frame(
                uploaded_dataset,
                target_column,
                task_type=task_choice,
            )

            st.subheader("Dataset validation")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", validation_status["row_count"])
            col2.metric("Columns", validation_status["column_count"])
            col3.metric("Duplicate rows", validation_status["duplicate_rows"])

            st.write(f"Suggested task type: {suggested_task}")
            st.write(f"Selected task type: {task_choice}")

            preview_df = uploaded_dataset.head(10)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            with st.expander("Column details and missing values"):
                dtype_summary = uploaded_dataset.dtypes.astype(str).to_frame(
                    name="dtype"
                )
                st.dataframe(dtype_summary, use_container_width=True)
                if validation_status["missing_summary"]:
                    st.write(
                        pd.DataFrame.from_dict(
                            validation_status["missing_summary"],
                            orient="index",
                            columns=["missing_values"],
                        )
                    )
                else:
                    st.info("No missing values were detected.")

            if validation_status["task_type"] == "classification":
                st.success("Dataset validation passed for classification.")
            else:
                st.success("Dataset validation passed for regression.")

        except Exception as error:
            st.error(f"Dataset validation failed: {error}")

    uploaded_model = st.file_uploader(
        "Upload an existing scikit-learn model",
        type=["joblib", "pkl", "pickle"],
        help="Trusted uploads only. Joblib and pickle files can execute code during deserialization.",
    )

    if uploaded_model is not None:
        st.info(
            "Trusted upload workflow: only load model files from a source you control. "
            "The platform keeps uploaded artifacts in an isolated directory and does not replace the baseline silently."
        )

    st.subheader("Train a generic model")
    st.caption(
        "Training uploads are cleaned into a separate file. Review the report "
        "before approving training. Drift inputs remain unchanged."
    )
    remove_exact_duplicates = st.checkbox(
        "Remove exact duplicate rows",
        value=True,
        help="Only fully identical rows are removed; the original upload is preserved.",
    )
    convert_numeric_columns = st.checkbox(
        "Convert columns to numeric when every non-missing value parses safely",
        value=True,
    )
    outlier_iqr_multiplier = st.number_input(
        "Outlier reporting sensitivity (IQR multiplier)",
        min_value=0.1,
        max_value=10.0,
        value=1.5,
        step=0.1,
    )
    model_options = [
        "logistic_regression",
        "random_forest_classifier",
        "decision_tree_classifier",
        "gradient_boosting_classifier",
        "linear_regression",
        "random_forest_regressor",
        "decision_tree_regressor",
        "gradient_boosting_regressor",
    ]
    selected_model = st.selectbox("Model type", model_options)
    train_test_size = st.slider("Test split proportion", 0.10, 0.40, 0.20, 0.05)
    random_seed = st.number_input("Random seed", min_value=1, max_value=1000, value=42)

    registry_models = ((get_api_data("models") or {}).get("models", []))
    registry_options = ["None"] + [
        f"{item.get('name')}:{item.get('version')}" for item in registry_models
    ]
    selected_registry_name = st.selectbox(
        "Optional baseline model from registry",
        registry_options,
        help=(
            "The baseline must be trained on the same input features and task "
            "type as the uploaded dataset. Select None to train without comparing "
            "against a registered baseline."
        ),
    )

    dataset_bytes = uploaded_csv.getvalue() if uploaded_csv is not None else b""
    run_context = hashlib.sha256(
        b"\0".join(
            [
                dataset_bytes,
                str(target_column).encode("utf-8"),
                task_choice.encode("utf-8"),
                str(remove_exact_duplicates).encode("utf-8"),
                str(convert_numeric_columns).encode("utf-8"),
                str(outlier_iqr_multiplier).encode("utf-8"),
                selected_model.encode("utf-8"),
                str(train_test_size).encode("utf-8"),
                str(random_seed).encode("utf-8"),
                selected_registry_name.encode("utf-8"),
            ]
        )
    ).hexdigest()

    if st.session_state["generic_training_context"] != run_context:
        st.session_state["generic_training_result"] = None
        st.session_state["generic_training_context"] = run_context

    upload_context = hashlib.sha256(
        b"\0".join(
            [
                dataset_bytes,
                str(target_column).encode("utf-8"),
                task_choice.encode("utf-8"),
                str(remove_exact_duplicates).encode("utf-8"),
                str(convert_numeric_columns).encode("utf-8"),
                str(outlier_iqr_multiplier).encode("utf-8"),
            ]
        )
    ).hexdigest()
    if st.session_state["cleaning_upload_context"] != upload_context:
        st.session_state["cleaning_upload_result"] = None
        st.session_state["cleaning_approved"] = False
        st.session_state["cleaning_upload_context"] = upload_context

    if st.button(
        "Analyze and prepare cleaned dataset",
        disabled=uploaded_csv is None or target_column is None,
    ):
        try:
            with st.spinner("Validating and preparing a separate cleaned copy..."):
                upload_response = API_SESSION.post(
                    f"{API_URL}/dataset/upload",
                    files={
                        "file": (
                            uploaded_csv.name,
                            uploaded_csv.getvalue(),
                            "text/csv",
                        )
                    },
                    data={
                        "target_column": target_column,
                        "task_type": task_choice,
                        "remove_exact_duplicates": str(
                            remove_exact_duplicates
                        ).lower(),
                        "convert_numeric_columns": str(
                            convert_numeric_columns
                        ).lower(),
                        "outlier_iqr_multiplier": str(outlier_iqr_multiplier),
                    },
                    timeout=120,
                )
                upload_response.raise_for_status()
                st.session_state["cleaning_upload_result"] = upload_response.json()
                st.session_state["cleaning_approved"] = False
        except requests.HTTPError as error:
            detail = error.response.text if error.response is not None else str(error)
            if error.response is not None:
                try:
                    detail = error.response.json().get("detail", detail)
                except ValueError:
                    pass
            st.error(f"Dataset cleaning failed: {detail}")
        except requests.RequestException as error:
            st.error(f"Could not reach the dataset cleaning API: {error}")
        except ValueError as error:
            st.error(f"The cleaning API returned an invalid response: {error}")

    upload_payload = st.session_state["cleaning_upload_result"]
    if upload_payload is not None:
        cleaning_report = upload_payload["cleaning_report"]
        st.subheader("Cleaning report")
        cleaning_metrics = st.columns(4)
        cleaning_metrics[0].metric(
            "Rows before / after",
            f"{cleaning_report['rows_before']} / {cleaning_report['rows_after']}",
        )
        cleaning_metrics[1].metric(
            "Exact duplicates removed",
            cleaning_report["duplicates_removed"],
        )
        cleaning_metrics[2].metric(
            "Labeled rows usable for training",
            cleaning_report["usable_labeled_rows"],
        )
        cleaning_metrics[3].metric(
            "Type conversions",
            len(cleaning_report["changed_data_types"]),
        )
        st.dataframe(
            pd.DataFrame(cleaning_report["columns"]),
            use_container_width=True,
            hide_index=True,
        )
        if cleaning_report["warnings"]:
            for warning in cleaning_report["warnings"]:
                st.warning(warning)
        if cleaning_report["errors"]:
            for cleaning_error in cleaning_report["errors"]:
                st.error(cleaning_error)
        cleaned_filename = upload_payload["cleaned_dataset_filename"]
        try:
            cleaned_download = API_SESSION.get(
                f"{API_URL}/dataset/cleaned/{cleaned_filename}",
                timeout=API_TIMEOUT,
            )
            cleaned_download.raise_for_status()
            download_col, delete_col = st.columns(2)
            download_col.download_button(
                "Download cleaned CSV",
                data=cleaned_download.content,
                file_name=cleaned_filename,
                mime="text/csv",
            )
            if delete_col.button("Delete saved cleaned CSV"):
                try:
                    delete_response = API_SESSION.delete(
                        f"{API_URL}/dataset/cleaned/{cleaned_filename}",
                        timeout=API_TIMEOUT,
                    )
                    delete_response.raise_for_status()
                    st.session_state["cleaning_upload_result"] = None
                    st.session_state["cleaning_approved"] = False
                    st.success("The cleaned dataset was deleted.")
                    st.rerun()
                except requests.HTTPError as error:
                    detail = error.response.text if error.response is not None else str(error)
                    st.error(f"Could not delete the cleaned dataset: {detail}")
                except requests.RequestException as error:
                    st.error(f"Could not delete the cleaned dataset: {error}")
        except requests.RequestException as error:
            st.error(f"Could not retrieve the cleaned CSV: {error}")
        st.checkbox(
            "I reviewed the cleaning report and approve training with this cleaned dataset",
            value=st.session_state["cleaning_approved"],
            key="cleaning_approved",
        )

    train_btn = st.button(
        "Train model from reviewed cleaned data",
        type="primary",
        disabled=(
            uploaded_csv is None
            or target_column is None
            or upload_payload is None
            or not st.session_state["cleaning_approved"]
        ),
    )

    if train_btn and uploaded_csv is not None and target_column and upload_payload:
        st.session_state["generic_training_result"] = None
        try:
            with st.spinner("Training the selected model on the reviewed cleaned data..."):
                baseline_path = None
                if selected_registry_name != "None":
                    name, version = selected_registry_name.split(":", 1)
                    registry_entry = next(
                        (
                            item for item in registry_models
                            if item.get("name") == name and item.get("version") == version
                        ),
                        None,
                    )
                    if registry_entry:
                        baseline_path = registry_entry.get("path")

                train_payload = {
                    "dataset_file": upload_payload.get("cleaned_dataset_filename"),
                    "target_column": target_column,
                    "model_name": selected_model,
                    "task_type": task_choice,
                    "test_size": float(train_test_size),
                    "random_state": int(random_seed),
                    "baseline_model_path": baseline_path,
                    "remove_exact_duplicates": remove_exact_duplicates,
                    "convert_numeric_columns": convert_numeric_columns,
                    "outlier_iqr_multiplier": float(outlier_iqr_multiplier),
                }

                train_response = API_SESSION.post(
                    f"{API_URL}/train/generic",
                    json=train_payload,
                    timeout=180,
                )
                train_response.raise_for_status()
                st.session_state["generic_training_result"] = train_response.json()

            st.success("Training completed successfully.")

        except requests.RequestException as error:
            detail = str(error)
            if error.response is not None:
                try:
                    detail = error.response.json().get("detail", error.response.text)
                except ValueError:
                    detail = error.response.text
            st.error(f"Training failed: {detail}")
        except ValueError as error:
            st.error(f"Training request could not be parsed: {error}")
    elif train_btn:
        st.warning(
            "Upload a CSV, prepare and review its cleaning report, then approve "
            "the cleaned dataset before training."
        )

    current_result = st.session_state["generic_training_result"]
    if current_result is not None:
        st.subheader("Latest uploaded-data run")
        st.caption(
            "These results are from the most recent successful training run in this browser session."
        )
        st.json(current_result)

        tracking = current_result.get("tracking")
        if tracking:
            if tracking["status"] == "logged":
                st.success(
                    f"Model run logged to MLflow experiment "
                    f"'{tracking['experiment_name']}' "
                    f"(run {tracking['run_id']})."
                )
            elif tracking["status"] == "failed":
                st.warning(f"Model trained, but MLflow logging failed: {tracking['error']}")
            else:
                st.info(tracking.get("reason", "MLflow logging is not configured."))

        if current_result.get("metrics"):
            metric_df = pd.DataFrame(
                [
                    {"metric": key, "value": value}
                    for key, value in current_result["metrics"].items()
                    if not isinstance(value, (list, dict))
                ]
            )
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

        comparison_result = current_result.get("comparison")
        if comparison_result:
            st.subheader("Baseline and candidate comparison")
            comparison_df = pd.DataFrame(
                [
                    {"Metric": metric, "Baseline": baseline_value, "Candidate": candidate_value}
                    for metric, baseline_value in comparison_result.get("baseline", {}).items()
                    if not isinstance(baseline_value, (list, dict))
                    and (candidate_value := comparison_result.get("candidate", {}).get(metric)) is not None
                ]
            )
            if not comparison_df.empty:
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            decision = current_result.get("promotion_decision", {})
            if decision:
                st.subheader("Candidate decision")
                if decision.get("action") == "PROMOTION_RECOMMENDED":
                    st.success("Candidate model passed the configured comparison gate.")
                else:
                    st.warning("Keep the existing baseline; the candidate did not pass the comparison gate.")
                st.write(f"Decision: {decision.get('action', 'Unavailable')}")
                st.write("Deployment authorized: No")
                st.caption("A recommendation does not deploy or replace a model.")
    else:
        st.info("No uploaded-data model has been executed in this browser session yet.")


# -------------------- UPLOADED-DATA DRIFT ANALYSIS --------------------

st.header("Data drift: reference vs incoming data")
st.caption(
    "Compare input-feature distributions only. The selected target is excluded; "
    "this analysis does not measure model accuracy or performance."
)

use_training_dataset = st.checkbox(
    "Use the uploaded training dataset as the reference",
    value=uploaded_csv is not None,
    disabled=uploaded_csv is None,
)

reference_file = uploaded_csv if use_training_dataset else st.file_uploader(
    "Upload a reference CSV",
    type=["csv"],
    key="drift_reference_csv",
)

reference_frame = None
drift_target = None
if reference_file is not None:
    try:
        reference_frame = (
            uploaded_dataset.copy()
            if use_training_dataset and uploaded_dataset is not None
            else pd.read_csv(reference_file)
        )
        reference_target_options = ["(no target)"] + list(reference_frame.columns)
        default_target = (
            target_column
            if use_training_dataset and target_column in reference_target_options
            else "(no target)"
        )
        target_index = reference_target_options.index(default_target)
        selected_drift_target = st.selectbox(
            "Target column to exclude from drift checks",
            reference_target_options,
            index=target_index,
            key="drift_target_column",
        )
        drift_target = (
            None if selected_drift_target == "(no target)" else selected_drift_target
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read the reference CSV: {error}")
        reference_frame = None

incoming_mode = st.radio(
    "Incoming dataset",
    ["Upload incoming CSV", "Select existing incoming CSV"],
    horizontal=True,
)
incoming_file = None
incoming_dataset_name = None
if incoming_mode == "Upload incoming CSV":
    incoming_file = st.file_uploader(
        "Upload a separate incoming CSV",
        type=["csv"],
        key="drift_incoming_csv",
    )
else:
    existing_incoming_files = sorted(
        path.name for path in (ROOT / "data" / "incoming").glob("*.csv")
    )
    if existing_incoming_files:
        incoming_dataset_name = st.selectbox(
            "Existing incoming dataset",
            existing_incoming_files,
            key="drift_existing_incoming",
        )
    else:
        st.info("No existing incoming CSV datasets were found.")

drift_context_parts = [
    reference_file.getvalue() if reference_file is not None else b"",
    incoming_file.getvalue() if incoming_file is not None else b"",
    str(incoming_dataset_name).encode("utf-8"),
    str(drift_target).encode("utf-8"),
]
auto_retrain = st.checkbox(
    "Automatically train a candidate when drift exceeds the threshold",
    value=True,
    help=(
        "Retraining uses only labeled rows from the reference CSV. Incoming "
        "data is never treated as training labels. The candidate is saved "
        "separately and is not promoted."
    ),
)
drift_threshold_percent = st.slider(
    "Drift threshold (% of features flagged)",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=5.0,
)
if drift_target and reference_frame is not None:
    if reference_frame[drift_target].dropna().empty:
        drift_model_options = ["Automatic"]
        st.warning(
            "The selected target has no labeled rows. Drift can still be "
            "checked, but automatic supervised retraining will be skipped."
        )
    else:
        target_task = infer_task_type(reference_frame[drift_target])
        supported_drift_models = (
            CLASSIFICATION_MODELS
            if target_task == "classification"
            else REGRESSION_MODELS
        )
        drift_model_options = ["Automatic"] + sorted(supported_drift_models)
else:
    drift_model_options = ["Automatic"]
selected_drift_model = st.selectbox(
    "Candidate model for automatic retraining",
    drift_model_options,
    help="Automatic selects a random-forest model appropriate to the target type.",
)
retrain_model_name = (
    None if selected_drift_model == "Automatic" else selected_drift_model
)
drift_context_parts.extend(
    [
        str(auto_retrain).encode("utf-8"),
        str(drift_threshold_percent).encode("utf-8"),
        str(retrain_model_name).encode("utf-8"),
    ]
)
drift_context = hashlib.sha256(b"\0".join(drift_context_parts)).hexdigest()
if st.session_state["generic_drift_context"] != drift_context:
    st.session_state["generic_drift_result"] = None
    st.session_state["generic_drift_context"] = drift_context

has_incoming = incoming_file is not None or incoming_dataset_name is not None
if st.button(
    "Compare feature drift",
    type="primary",
    disabled=reference_file is None or reference_frame is None or not has_incoming,
):
    try:
        with st.spinner("Comparing reference and incoming feature distributions..."):
            files = {
                "reference_file": (
                    reference_file.name,
                    reference_file.getvalue(),
                    "text/csv",
                )
            }
            if incoming_file is not None:
                files["incoming_file"] = (
                    incoming_file.name,
                    incoming_file.getvalue(),
                    "text/csv",
                )
            data = {
                "target_column": drift_target or "",
                "auto_retrain": str(auto_retrain).lower(),
                "drift_threshold_percent": str(drift_threshold_percent),
            }
            if retrain_model_name is not None:
                data["retrain_model_name"] = retrain_model_name
            if incoming_dataset_name is not None:
                data["incoming_dataset"] = incoming_dataset_name

            response = API_SESSION.post(
                f"{API_URL}/drift-check",
                files=files,
                data=data,
                timeout=300,
            )
            response.raise_for_status()
            st.session_state["generic_drift_result"] = response.json()
    except requests.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        st.error(f"Drift analysis failed: {detail}")
    except requests.RequestException as error:
        st.error(f"Could not reach the drift detection API: {error}")
    except ValueError as error:
        st.error(f"The drift API returned an invalid response: {error}")

drift_result = st.session_state["generic_drift_result"]
if drift_result is not None:
    status = drift_result["overall_status"]
    flagged = drift_result["features_flagged"]
    checked = drift_result["features_checked"]
    percentage = drift_result["drift_percentage"]

    drift_metrics = st.columns(4)
    drift_metrics[0].metric("Overall drift status", status.replace("_", " ").title())
    drift_metrics[1].metric("Features flagged", f"{flagged} / {checked}")
    drift_metrics[2].metric("Features flagged (%)", f"{percentage:.1f}%")
    drift_metrics[3].metric(
        "Reference / incoming rows",
        f"{drift_result['reference_rows']} / {drift_result['incoming_rows']}",
    )

    if status == "drift_detected":
        st.warning(
            f"Potential feature drift detected in {flagged} of {checked} features."
        )
    elif status == "inconclusive":
        st.info("Drift analysis was inconclusive because there was too little usable data.")
    else:
        st.success("No statistically significant feature drift was detected.")

    if drift_result.get("target_column_excluded"):
        st.caption(
            f"Target column excluded from comparison: "
            f"{drift_result['target_column_excluded']}"
        )

    retraining = drift_result.get("retraining", {})
    st.subheader("Automatic retraining")
    retraining_status = retraining.get("training_status", "unavailable")
    st.write(f"Training status: **{retraining_status.replace('_', ' ').title()}**")
    if retraining.get("trigger_reason"):
        st.caption(retraining["trigger_reason"])
    if retraining.get("duplicate_event"):
        st.info("This drift event was already processed; no duplicate training run was started.")
    if retraining.get("error"):
        st.error(f"Candidate training error: {retraining['error']}")
    if retraining_status == "completed":
        st.success(
            "Candidate model trained from labeled reference data. "
            "It has not been promoted or deployed."
        )
        st.write(f"Candidate model type: {retraining.get('model_type')}")
        st.write(f"Candidate artifact: `{retraining.get('candidate_model_path')}`")
        if retraining.get("metrics"):
            st.json(retraining["metrics"])
    elif retraining_status == "skipped_missing_labels":
        st.warning(
            "Drift exceeded the threshold, but retraining was skipped because "
            "the reference dataset has no valid labels for the selected target."
        )
    elif retraining_status == "skipped_no_drift":
        st.info("Drift did not exceed the threshold; no candidate was trained.")
    elif retraining_status == "disabled":
        st.info("Automatic retraining is disabled for this drift check.")

    drift_frame = pd.DataFrame(drift_result["results"])
    if not drift_frame.empty:
        st.subheader("Feature-level drift report")
        st.dataframe(
            drift_frame[
                [
                    "feature",
                    "feature_type",
                    "test",
                    "statistic",
                    "p_value",
                    "reference_missing",
                    "incoming_missing",
                    "missingness_p_value",
                    "drift_detected",
                    "explanation",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        changed_features = drift_frame.loc[
            drift_frame["drift_detected"], "feature"
        ].tolist()
        if changed_features:
            st.write("Features with detected changes: " + ", ".join(changed_features))
        else:
            st.write(
                "No feature passed the configured significance threshold. "
                "Rows with too few non-missing values are marked in the report."
            )


# -------------------- SIDEBAR CONTROLS --------------------

st.sidebar.header("Controls")

dataset_choice = st.sidebar.selectbox(
    "Dataset for drift analysis",
    ["Drifted incoming data", "Normal incoming data"],
)

if st.sidebar.button("Run drift detection", type="primary"):
    dataset_key = (
        "drifted"
        if dataset_choice == "Drifted incoming data"
        else "normal"
    )

    try:
        with st.spinner("Analyzing incoming data..."):
            response = API_SESSION.post(
                f"{API_URL}/run-drift-check",
                json={"dataset": dataset_key},
                timeout=150,
            )
            response.raise_for_status()

        st.sidebar.success("Drift analysis completed.")
        st.rerun()

    except requests.HTTPError as error:
        detail = (
            error.response.text
            if error.response is not None
            else str(error)
        )
        st.sidebar.error(f"Drift analysis failed: {detail}")

    except requests.RequestException as error:
        st.sidebar.error(f"API request failed: {error}")

if st.sidebar.button("Refresh dashboard"):
    st.rerun()

st.sidebar.caption(
    "Drift analysis runs against the dataset selected above. "
    "Model retraining and deployment are separate operations."
)

show_legacy_reports = st.sidebar.checkbox(
    "Show historical customer-churn demo reports",
    value=False,
    help="These reports are from the original demo workflow and are not results for your uploaded dataset.",
)

# -------------------- LOAD DATA --------------------

models_data = get_api_data("models")

if show_legacy_reports:
    workflow = get_api_data("workflow-status")
    drift = get_api_data("drift-report")
    comparison = get_api_data("model-comparison")
    promotion = get_api_data("promotion-decision")
    history_data = get_api_data("drift-history")
    mlflow_runs, mlflow_error = get_mlflow_runs()


# -------------------- WORKFLOW STATUS --------------------

if show_legacy_reports:
    st.header("Historical demo workflow")
    st.warning(
        "The following reports belong to the original customer-churn demo. "
        "They are separate from the latest uploaded-data run above."
    )

    st.header("Project Status")

    if workflow is not None:
        status = workflow.get("workflow", "Unknown")
        st.metric("Latest demo pipeline status", status)

        with st.expander("Workflow details"):
            st.json(workflow)
    else:
        st.warning("Workflow status is currently unavailable.")


# -------------------- DRIFT MONITORING --------------------

    st.header("1. Data Drift Monitoring (demo dataset)")

    if drift is not None:
        drifted_count = drift.get("drifted_features", 0)
        checked_count = drift.get("features_checked", 0)

        col1, col2 = st.columns(2)
        col1.metric("Features checked", checked_count)
        col2.metric("Features with drift", drifted_count)

        if drifted_count > 0:
            st.warning(
                f"Potential drift detected in {drifted_count} "
                f"of {checked_count} monitored features."
            )
        else:
            st.success("No drift detected in the monitored features.")

        drift_df = pd.DataFrame(drift.get("results", []))

        if not drift_df.empty:
            required_columns = [
                "feature",
                "test",
                "statistic",
                "p_value",
                "drift_detected",
            ]

            available_columns = [
                column for column in required_columns
                if column in drift_df.columns
            ]

            display_drift_df = drift_df[available_columns].copy()

            st.subheader("Feature-level drift results")
            st.dataframe(
                display_drift_df,
                use_container_width=True,
                hide_index=True,
            )

            if (
                "p_value" in drift_df.columns
                and "feature" in drift_df.columns
            ):
                chart_df = drift_df[["feature", "p_value"]].copy()
                chart_df["p_value"] = pd.to_numeric(
                    chart_df["p_value"],
                    errors="coerce",
                )
                chart_df = chart_df.dropna().set_index("feature")

                if not chart_df.empty:
                    st.subheader("Statistical test p-values")
                    st.bar_chart(chart_df)
        else:
            st.info("No feature-level drift results are available.")
    else:
        st.warning("Drift report is unavailable.")


# -------------------- MODEL COMPARISON --------------------

    st.header("2. Model Performance Comparison (demo workflow)")

    if comparison is not None:
        baseline = comparison.get("baseline", {})
        candidate = comparison.get("candidate", {})

        metric_rows = [
            {
                "Metric": "Accuracy",
                "Baseline": baseline.get("accuracy"),
                "Candidate": candidate.get("accuracy"),
            },
            {
                "Metric": "Churn F1-score",
                "Baseline": baseline.get("churn_f1_score"),
                "Candidate": candidate.get("churn_f1_score"),
            },
        ]

        metrics_df = pd.DataFrame(metric_rows)

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        chart_df = metrics_df.set_index("Metric")
        st.bar_chart(chart_df)

        col1, col2 = st.columns(2)

        col1.metric(
            "Baseline accuracy",
            format_percentage(baseline.get("accuracy")),
        )
        col2.metric(
            "Candidate accuracy",
            format_percentage(candidate.get("accuracy")),
        )

        col3, col4 = st.columns(2)

        col3.metric(
            "Baseline churn F1-score",
            format_percentage(baseline.get("churn_f1_score")),
        )
        col4.metric(
            "Candidate churn F1-score",
            format_percentage(candidate.get("churn_f1_score")),
        )
    else:
        st.info("Model comparison results are unavailable.")


# -------------------- PROMOTION DECISION --------------------

    st.header("3. Promotion Decision (demo workflow)")

    if promotion is not None:
        action = promotion.get("action", "Unknown")

        if action == "PROMOTION_RECOMMENDED":
            st.success(action)
        elif action == "RETAIN_BASELINE":
            st.warning(action)
        else:
            st.info(action)

        st.write(
            promotion.get(
                "reason",
                "No decision explanation available.",
            )
        )

        st.write(
            "**Deployment authorized:**",
            promotion.get("deployment_authorized", False),
        )

        st.info(
            "A promotion recommendation is not a deployment. "
            "The baseline model is not automatically replaced."
        )

        with st.expander("Promotion decision details"):
            st.json(promotion)
    else:
        st.info("Promotion decision is unavailable.")


# -------------------- REGISTERED MODELS --------------------

if show_legacy_reports:
    st.header("4. Registered Models")

    if models_data is not None:
        registered_models = models_data.get("models", [])

        st.metric("Registered model versions", len(registered_models))

        if registered_models:
            models_df = pd.DataFrame(registered_models)

            columns_to_show = [
                column
                for column in [
                    "name",
                    "version",
                    "framework",
                    "registered_at",
                ]
                if column in models_df.columns
            ]

            st.dataframe(
                models_df[columns_to_show],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No models have been registered yet.")
    else:
        st.info(
            "Model registry is unavailable. "
            "Check whether the /models endpoint exists in FastAPI."
        )


# -------------------- DRIFT HISTORY --------------------

    st.header("5. Drift Analysis History (demo workflow)")

    if history_data is not None:
        history_records = history_data.get("history", [])

        st.metric(
            "Total analyses performed",
            history_data.get("total_analyses", len(history_records)),
        )

        if history_records:
            history_df = pd.DataFrame(history_records)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )

            if {
                "analysis_time",
                "drifted_features",
            }.issubset(history_df.columns):
                chart_df = history_df[
                    ["analysis_time", "drifted_features"]
                ].copy()

                chart_df["drifted_features"] = pd.to_numeric(
                    chart_df["drifted_features"],
                    errors="coerce",
                )

                chart_df = chart_df.dropna().set_index("analysis_time")

                if not chart_df.empty:
                    st.subheader("Drift signals across analyses")
                    st.line_chart(chart_df)
        else:
            st.info(
                "No drift history yet. Run drift detection from the sidebar."
            )
    else:
        st.info("Drift history is unavailable.")


# -------------------- MLFLOW EXPERIMENT TRACKING --------------------

if show_legacy_reports:
    st.header("6. MLflow Experiment Tracking (demo workflow)")

    if mlflow_runs is not None:
        st.metric("Recorded MLflow runs", len(mlflow_runs))

        if mlflow_runs:
            run_rows = []

            for run in mlflow_runs:
                info = run.get("info", {})
                data = run.get("data", {})

                raw_metrics = data.get("metrics", [])
                if isinstance(raw_metrics, dict):
                    metrics = raw_metrics
                elif isinstance(raw_metrics, list):
                    metrics = {
                        item["key"]: item["value"]
                        for item in raw_metrics
                        if isinstance(item, dict)
                        and "key" in item
                        and "value" in item
                    }
                else:
                    metrics = {}

                raw_tags = data.get("tags", [])
                if isinstance(raw_tags, dict):
                    tags = raw_tags
                elif isinstance(raw_tags, list):
                    tags = {
                        tag["key"]: tag["value"]
                        for tag in raw_tags
                        if isinstance(tag, dict)
                        and "key" in tag
                        and "value" in tag
                    }
                else:
                    tags = {}

                accuracy = metrics.get("accuracy")
                if accuracy is None:
                    accuracy = metrics.get("candidate_accuracy")
                if accuracy is None:
                    accuracy = metrics.get("baseline_accuracy")

                f1_metric = metrics.get("churn_f1_score")
                if f1_metric is None:
                    f1_metric = metrics.get("candidate_churn_f1_score")
                if f1_metric is None:
                    f1_metric = metrics.get("baseline_churn_f1_score")

                run_rows.append({
                    "Run name": tags.get("mlflow.runName", "Unnamed"),
                    "Workflow status": tags.get("workflow_status", "N/A"),
                    "Promotion": tags.get("promotion_action", "N/A"),
                    "Accuracy": (
                        format_percentage(accuracy)
                        if accuracy is not None else "N/A"
                    ),
                    "Churn F1-score": (
                        format_percentage(f1_metric)
                        if f1_metric is not None else "N/A"
                    ),
                    "Run ID": info.get("run_id", ""),
                    "Status": info.get("status", ""),
                })

            runs_df = pd.DataFrame(run_rows)

            st.dataframe(
                runs_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Metrics may come from different evaluation datasets. "
                "Use the model comparison section for shared holdout results."
            )

            st.markdown(f"[Open MLflow UI]({MLFLOW_URL})")

        else:
            st.info("No MLflow runs are available yet.")

    else:
        st.warning(
            f"Cannot retrieve MLflow experiment runs: {mlflow_error}. "
            "Check that the MLflow server is running on port 5000."
        )


# -------------------- OPERATIONAL MONITORING --------------------

st.header("Model and Operational Monitoring")
monitoring_report = get_api_data("monitoring/predictions")
if monitoring_report is not None:
    monitoring_metrics = st.columns(4)
    monitoring_metrics[0].metric(
        "Prediction requests",
        monitoring_report["requests"],
    )
    monitoring_metrics[1].metric(
        "Succeeded",
        monitoring_report["succeeded"],
    )
    monitoring_metrics[2].metric(
        "Failed",
        monitoring_report["failed"],
    )
    average_latency = monitoring_report["average_latency_ms"]
    monitoring_metrics[3].metric(
        "Average latency",
        f"{average_latency:.1f} ms" if average_latency is not None else "N/A",
    )
    if monitoring_report["groups"]:
        st.dataframe(
            pd.DataFrame(monitoring_report["groups"]),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Only model IDs, outcomes, error classes, row counts, and latency are "
        "stored. Prediction feature values and uploaded CSV contents are not logged."
    )

with st.expander("Upload retention and local backups"):
    retention_report = get_api_data("operations/retention")
    if retention_report:
        st.json(retention_report)
    if st.button("Back up model registry and MLflow database"):
        try:
            backup_response = API_SESSION.post(
                f"{API_URL}/operations/backup",
                timeout=60,
            )
            backup_response.raise_for_status()
            backup_result = backup_response.json()
            if backup_result["status"] == "completed":
                st.success("Local backup completed.")
                st.dataframe(
                    pd.DataFrame(backup_result["files"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(backup_result["message"])
        except requests.HTTPError as error:
            detail = error.response.text if error.response is not None else str(error)
            st.error(f"Backup failed: {detail}")
        except requests.RequestException as error:
            st.error(f"Could not complete the local backup: {error}")
