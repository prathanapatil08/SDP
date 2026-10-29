from __future__ import annotations

from io import BytesIO
import os

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
PREDICTION_TIMEOUT = 120
API_SESSION = requests.Session()
if os.getenv("API_AUTH_TOKEN"):
    API_SESSION.headers["X-API-Key"] = os.environ["API_AUTH_TOKEN"]


def _format_schema(model: dict) -> pd.DataFrame:
    schema = pd.DataFrame(model.get("required_features", []))
    if schema.empty:
        return schema
    return schema.rename(
        columns={
            "name": "Feature",
            "data_type": "Required type",
            "role": "Preprocessing role",
        }
    )


def render_prediction_dashboard() -> None:
    st.header("Prediction Dashboard")
    st.caption(
        "Prediction is a separate workflow. Uploaded prediction data never "
        "starts model training or candidate promotion."
    )

    csv_tab, image_tab = st.tabs(
        ["CSV Prediction", "Medical Image Prediction (experimental)"]
    )

    with csv_tab:
        try:
            response = API_SESSION.get(
                f"{API_URL}/prediction/models",
                timeout=15,
            )
            response.raise_for_status()
            catalog = response.json()
        except requests.HTTPError as error:
            detail = error.response.text if error.response is not None else str(error)
            st.error(f"Could not load compatible models: {detail}")
            catalog = None
        except requests.RequestException as error:
            st.error(f"Could not reach the prediction API: {error}")
            catalog = None
        except ValueError as error:
            st.error(f"The model catalog returned invalid JSON: {error}")
            catalog = None

        if catalog is not None:
            models = catalog.get("models", [])
            if not models:
                st.warning(
                    "No compatible saved models are currently available. "
                    "Train a model through the MLOps dashboard first."
                )
            else:
                options = {
                    f"{model['name']} · {model['version']} · {model['task']} "
                    f"({model['source']})": model
                    for model in models
                }
                selected_label = st.selectbox(
                    "Select a compatible trained model",
                    list(options),
                )
                selected_model = options[selected_label]
                st.subheader("Model card")
                st.write(
                    f"**Model:** {selected_model['name']} "
                    f"(version {selected_model['version']})"
                )
                st.write(f"**Task:** {selected_model['task']}")
                st.write(
                    f"**Target labels:** "
                    f"{selected_model['expected_target_labels'] or 'Not declared'}"
                )
                st.write(
                    "**Preprocessing:** "
                    + "; ".join(selected_model["preprocessing"])
                )
                st.write(
                    "**Evaluation metrics:** "
                    f"{selected_model['evaluation_metrics'] or 'Not available'}"
                )
                st.caption(
                    "Metrics source: " + selected_model["metrics_source"]
                )
                st.write("**Required input schema**")
                st.dataframe(
                    _format_schema(selected_model),
                    use_container_width=True,
                    hide_index=True,
                )
                if selected_model["probabilities_supported"]:
                    st.caption(
                        "This classifier exposes predict_proba; returned "
                        "scores are model probabilities, not calibrated clinical confidence."
                    )
                else:
                    st.caption(
                        "This model does not expose a meaningful probability output."
                    )

                prediction_file = st.file_uploader(
                    "Upload a CSV with the required input features",
                    type=["csv"],
                    key="prediction_csv_upload",
                    help="Maximum 10 MB and 100,000 rows. Extra, missing, or incompatible columns are rejected.",
                )
                if st.button(
                    "Predict every CSV row",
                    type="primary",
                    disabled=prediction_file is None,
                ):
                    try:
                        with st.spinner("Validating rows and running prediction..."):
                            prediction_response = API_SESSION.post(
                                f"{API_URL}/predict/csv",
                                files={
                                    "file": (
                                        prediction_file.name,
                                        prediction_file.getvalue(),
                                        "text/csv",
                                    )
                                },
                                data={"model_id": selected_model["model_id"]},
                                timeout=PREDICTION_TIMEOUT,
                            )
                            prediction_response.raise_for_status()
                            prediction_result = prediction_response.json()

                        result_rows = prediction_result["results"]
                        result_frame = pd.read_csv(
                            BytesIO(prediction_file.getvalue())
                        )
                        result_frame["prediction"] = [
                            row["prediction"] for row in result_rows
                        ]
                        if prediction_result["probabilities_supported"]:
                            result_frame["confidence"] = [
                                row["confidence"] for row in result_rows
                            ]
                            for label in prediction_result["classes"]:
                                result_frame[f"probability_{label}"] = [
                                    row["probabilities"][str(label)]
                                    for row in result_rows
                                ]
                        st.success(
                            f"Predictions completed for "
                            f"{prediction_result['prediction_count']} rows."
                        )
                        if prediction_result.get("compatibility_warnings"):
                            for warning in prediction_result["compatibility_warnings"]:
                                st.warning(warning)
                        if not prediction_result.get("monitoring_recorded", True):
                            st.warning(
                                "Prediction succeeded, but monitoring could not "
                                "record this event."
                            )
                        st.dataframe(
                            result_frame,
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download predictions CSV",
                            data=result_frame.to_csv(index=False).encode("utf-8"),
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    except requests.HTTPError as error:
                        detail = (
                            error.response.text
                            if error.response is not None
                            else str(error)
                        )
                        if error.response is not None:
                            try:
                                detail = error.response.json().get("detail", detail)
                            except ValueError:
                                pass
                        st.error(f"CSV prediction failed: {detail}")
                        if error.response is not None and error.response.status_code >= 500:
                            st.info(
                                "No alternative model was selected automatically. "
                                "Review the model compatibility warnings or choose "
                                "another verified model from the list."
                            )
                    except requests.RequestException as error:
                        st.error(f"Could not reach the prediction API: {error}")
                    except (ValueError, KeyError) as error:
                        st.error(f"Prediction response could not be processed: {error}")

        if catalog is not None and catalog.get("unavailable_models"):
            with st.expander("Saved models unavailable for prediction"):
                st.caption(
                    "An artifact is listed here when it is missing required "
                    "schema metadata, cannot be loaded, or uses an incompatible runtime."
                )
                st.dataframe(
                    pd.DataFrame(catalog["unavailable_models"]),
                    use_container_width=True,
                    hide_index=True,
                )

    with image_tab:
        st.subheader("Medical image prediction")
        st.warning(
            "Image inference is not enabled. The installed model artifacts are "
            "tabular models; none is declared and verified for a medical-image task."
        )
        st.file_uploader(
            "Medical image upload (disabled until a compatible model is verified)",
            type=["png", "jpg", "jpeg"],
            disabled=True,
            key="medical_image_disabled_upload",
        )
        st.info(
            "A reviewed candidate was Microsoft BiomedCLIP "
            "(BiomedCLIP-PubMedBERT_256-vit_base_patch16_224). Its model card "
            "describes broad biomedical image/text zero-shot research, not a "
            "specific disease classifier, and limits intended use to research "
            "and reproducibility; deployed use is explicitly out of scope. "
            "Therefore this project does not download, run, or present it as a "
            "medical predictor."
        )
        st.markdown(
            "[BiomedCLIP model card](https://huggingface.co/microsoft/"
            "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) · "
            "[Microsoft research paper](https://aka.ms/biomedclip-paper)"
        )
        st.caption(
            "No diagnosis or image classification result is generated. A "
            "medical image model can be enabled only after a task-specific "
            "model, license, preprocessing, image limits, labels, and intended "
            "use have been verified."
        )
