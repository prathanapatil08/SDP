\# Autonomous MLOps Platform for AI Model Drift Detection and Self-Healing Model Management



\## 1. Project Overview



This project is a software-based MLOps platform that monitors machine learning models for data drift, evaluates candidate models, and recommends whether a new model should be promoted or the existing baseline model should be retained.



The platform combines machine learning, statistical drift detection, automated workflow orchestration, a REST API, and an interactive dashboard.



\*\*Important:\*\* Model promotion is currently a recommendation only. The system does not automatically deploy or replace a production model.



\## 2. Objectives



\- Train and save a baseline machine learning model.

\- Detect changes in incoming data using statistical tests.

\- Generate drift reports and recommend further evaluation.

\- Retrain a candidate model using incoming data.

\- Compare baseline and candidate models using common evaluation data.

\- Recommend promotion only when the configured performance checks pass.

\- Display workflow status, drift results, and model metrics through a dashboard.

\- Maintain a history of drift analyses.



\## 3. Technologies Used



\- Python 3.12

\- Pandas and NumPy

\- Scikit-learn

\- SciPy

\- Joblib

\- FastAPI

\- Streamlit

\- Matplotlib

\- MLflow

\- Pytest



\## 4. Dataset



The project uses the IBM Telco Customer Churn dataset.



The dataset contains customer information and a Churn target indicating whether a customer left the service.



The cleaned dataset contains 7,032 records and 20 columns. The customer identifier and records with invalid TotalCharges values were removed during preprocessing.



\## 5. System Workflow



1\. Load and preprocess the reference dataset.

2\. Train and save the baseline model.

3\. Load incoming customer data.

4\. Compare incoming features with reference features.

5\. Detect potential drift using statistical tests.

6\. Generate a drift report.

7\. Decide whether candidate-model evaluation is recommended.

8\. Train a candidate model.

9\. Compare baseline and candidate models on the same reference holdout dataset.

10\. Check accuracy and churn F1-score.

11\. Recommend promotion or retain the baseline model.

12\. Display results through the API and dashboard.



\## 6. Model Evaluation



The current demonstration produced the following results on the shared reference holdout dataset:



| Metric | Baseline Model | Candidate Model |

|---|---:|---:|

| Accuracy | 76.26% | 79.46% |

| Churn F1-score | 0.5835 | 0.6436 |



In this run, the candidate passed both configured performance checks, so the system recommended promotion.



These are demonstration results, not proof of real-world future performance. The incoming data is sampled from the original dataset, so the current experiment has potential data leakage and overlap limitations.



\## 7. Safety and Model Promotion



The system uses a decision rule to recommend model promotion based on accuracy and churn F1-score.



The current implementation does not authorize deployment or replace the baseline model automatically. A production-ready system would require additional validation, independent test data, approval controls, and deployment safeguards.



\## 8. Project Structure



\- `data/reference/` — reference and cleaned datasets

\- `data/incoming/` — incoming and simulated drifted datasets

\- `models/` — saved baseline and candidate models

\- `src/` — training, drift detection, comparison, and orchestration scripts

\- `api/` — FastAPI backend

\- `dashboard/` — Streamlit dashboard

\- `tests/` — automated API tests

\- `experiments/` — reports, metrics, decisions, and workflow status

\- `database/` — reserved for future database integration



\## 9. How to Run



Activate the virtual environment from the project root:



```cmd

.venv\\Scripts\\activate

```



Run the complete workflow:



```cmd

python src\\run\_pipeline.py

```



Start the FastAPI backend in one CMD window:



```cmd

python -m uvicorn api.main:app --reload

```



Open the API documentation:



`http://127.0.0.1:8000/docs`



Start the Streamlit dashboard in another CMD window:



```cmd

python -m streamlit run dashboard\\app.py

```



Open the dashboard:



`http://localhost:8501`



To check drift for an uploaded dataset, upload the reference CSV in the dashboard
and select it as the reference. Choose the target column to exclude, then upload
a separate incoming CSV or select one from `data/incoming/`. Click
**Compare feature drift** to view the overall status and per-feature statistics.
Numeric features use the two-sample Kolmogorov-Smirnov test; categorical
features use a chi-square test, with a permutation p-value when expected counts
are small. Missing-value rates are tested separately. The target and model
accuracy are not used to determine data drift.

Automatic candidate retraining is enabled by default in this workflow. The
dashboard lets you set the percentage-of-features threshold and candidate
model. When drift is greater than the threshold, the selected model is trained
on valid labeled rows from the reference CSV only; incoming rows are never
assumed to have labels. Candidate artifacts are saved separately under
`models/`, and no model is promoted or deployed. Retraining events, including
skipped runs, duplicate events, and failures, are persisted in
`experiments/automatic_retraining.sqlite` and can be listed with
`GET /retraining-events`. Identical reference/incoming contents, target,
model, and threshold are processed at most once.

### Automatic cleaning before generic training

The uploaded-data workflow first creates a cleaned copy and shows a report for
review before enabling training. Exact duplicate removal and safe numeric
conversion are configurable. The report includes rows and missing values before
and after processing, dtype conversions, invalid values, potential outliers,
constant columns, suspected identifiers, possible target leakage, and
validation errors. Outliers, constant features, identifiers, and target values
are not silently modified or dropped. The raw upload remains unchanged, and
the cleaned CSV can be downloaded from the dashboard.

Missing feature values remain in the downloadable cleaned CSV to avoid fitting
imputation statistics on the held-out test data. During model training, the
existing scikit-learn preprocessing pipeline fits median numeric and
most-frequent categorical imputers on training rows only, then applies them to
the test rows. Drift comparison does not call the cleaner; reference and
incoming data remain unmodified. Generic training also cleans direct API
payloads, and its model and preprocessing pipeline are fitted on the same
training partition used for evaluation.

### Prediction dashboard

The sidebar `Application` control switches between the original **MLOps
Dashboard** and the separate **Prediction Dashboard**. The latter lists
compatible local and registered model artifacts, shows their declared input
schema and embedded preprocessing, and accepts a feature-only CSV. Every row
is checked against the selected model schema before prediction. Results can be
downloaded as CSV; classification probabilities are shown only when the model
implements `predict_proba`. Probability scores are not described as calibrated
confidence. Prediction uploads never start training or candidate promotion.

The API endpoints are `GET /prediction/models` and `POST /predict/csv`. CSV
uploads are limited to 10 MB and 100,000 rows. New generic training and
automatic-retraining candidates save target, feature-schema, preprocessing,
labels, and held-out metric metadata beside the model artifact. Legacy models
without evaluation records are labeled as having no stored metrics.

MLflow tracking is optional. Set `MLFLOW_TRACKING_URI` (and optionally
`MLFLOW_EXPERIMENT_NAME`) to log model artifacts, parameters, and scalar
metrics. Without the URI, local training works and reports tracking as
disabled.

### API access, model safeguards, and operations

The API is intended for local deployment. When `API_AUTH_TOKEN` is unset, it
accepts protected API requests only from loopback clients. To require a shared
admin key, set a unique secret in both the API and Streamlit process
environments before starting them:

```cmd
set API_AUTH_TOKEN=<your-strong-local-secret>
```

The dashboard sends that value in the `X-API-Key` header. The health check and
OpenAPI documentation routes are exempt; other API routes require the key.
Do not expose this development service directly to an untrusted network.

Prediction model loading retries one transient `OSError` once. The platform
does not switch models, retrain, or promote a replacement automatically when
loading or inference fails; errors are returned explicitly. New model metadata
records a SHA-256 artifact hash and Python/scikit-learn/Numpy/Pandas versions.
Hash mismatches block use, and runtime-version differences are shown as
compatibility warnings. This hash detects accidental changes, not a malicious
actor able to replace both the model and its metadata. Legacy artifacts may
not have hashes or training-version metadata.

`GET /monitoring/predictions` reports request outcomes and latency for the last
30 days (configurable up to 90). Stored events contain model IDs, operation,
success/failure, error class, row count, and latency—not CSV contents or
feature values. `POST /operations/backup` makes local copies of the model
registry and configured SQLite MLflow database under `backups/`; it does not
upload backups elsewhere. `GET /operations/retention` describes current
retention behavior.

Raw dataset upload files are removed after validation and cleaned-copy
creation. Cleaned CSVs can be explicitly deleted from the dashboard or through
`DELETE /dataset/cleaned/{filename}`. Files older than one hour are also
removed lazily when another upload or training request runs; there is no
background cleanup scheduler. Prediction CSVs are processed in memory.
Monitoring events older than 30 days are removed when a new event is recorded.

### Medical image prediction limitation

The image tab currently has uploads disabled and does not produce predictions.
The reviewed Microsoft BiomedCLIP artifact is MIT-licensed and supports
zero-shot biomedical image/text matching, but its model card limits use to
research/reproducibility and excludes deployed use; it is not a fixed
disease-specific classifier with verified disease labels and clinical
metrics. The project therefore does not download or use it for diagnosis or
decision support. A medically suitable task-specific model with verified
license, labels, preprocessing, and intended use is required before this
workflow can be enabled. See the [architecture and verification record](docs/ARCHITECTURE.md).



Run automated API tests:



```cmd

python -m pytest -v

```



\## 10. Testing



The automated suite covers API health and workflow routes, CSV upload and
validation, cleaning reports and download, generic drift API behavior, and
model training. Drift tests cover:



\- Unchanged numeric and categorical distributions

\- Shifted numeric and categorical distributions

\- Missing values, constant features, and small samples

\- Target exclusion and mismatched input columns

\- Drift-triggered candidate retraining and duplicate-event prevention

\- No-drift behavior, missing labels, and recorded training failures

- Cleaning duplicates, mixed types, missing values, target preservation,
  outlier/leakage warnings, and training-only imputation
- CSV prediction schema, model integrity/version checks, and inference failures
- API key/loopback access control, request-size limits, monitoring privacy,
  local backups, retention cleanup, and explicit cleaned-file deletion

Run the full suite from the project root with `python -m pytest -v`. Latest
verified result: **69 passed**, with two existing dependency/dataframe warnings.



\## 11. Future Enhancements



\- Evaluate models using an independent, time-separated dataset.

\- Add multiple-testing correction for drift detection.

\- Add scheduled retention cleanup and managed, off-machine backup rotation.

\- Add persistent storage for drift history.

\- Implement human approval and controlled deployment.

\- Add production-grade identity management, authorization roles, and monitoring.
