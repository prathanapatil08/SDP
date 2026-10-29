# Current Architecture and Extension Plan

## Current architecture (Phase 1)

The project is a Python application with one FastAPI service, one Streamlit
dashboard, and shared modules under `src/`. The dashboard is a single script
(`dashboard/app.py`) with upload/training, generic drift, and historical
customer-churn sections. It does not currently use a separate page-navigation
framework.

### Request and model flow

- `POST /dataset/upload` in `api/main.py` stores the uploaded CSV under
  `uploads/`, validates its target and task through
  `src/training_engine.py`, and returns dataset details.
- `POST /train/generic` consumes that upload or a dataframe payload, trains a
  candidate, and stores it in `models/`. The training engine builds a
  scikit-learn `Pipeline` with a `ColumnTransformer`, median numeric
  imputation, most-frequent categorical imputation, one-hot encoding, and
  optional numeric scaling. Fitted preprocessing is serialized with the model.
- `POST /drift-check` compares a reference CSV with a separate incoming CSV
  using `src/drift_detector.py`. Numeric columns use the two-sample KS test;
  categorical distributions use chi-square/permutation testing; missingness is
  compared separately. The selected target is excluded independently from
  both files. This workflow does not run a cleaning step.
- If automatic retraining is enabled, `src/retraining.py` deduplicates events
  in `experiments/automatic_retraining.sqlite` and trains a separate candidate
  from labeled reference rows. It does not promote or deploy that candidate.
- Registered models are stored in `models/registry.json` via
  `src/model_registry.py`. New entries record the model artifact hash, schema,
  task, labels, evaluation metrics, and training environment alongside the
  name, version, path, framework, and registration time.
- `POST /models/{name}/{version}/predict` accepts JSON records for registered
  scikit-learn models. It returns predictions and, when exposed by the model,
  probabilities. New generic and automatic candidates also get schema,
  preprocessing, target labels, and evaluation-metric sidecars.
- `GET /prediction/models` lists local/registered artifacts whose feature
  schema can be derived safely. `POST /predict/csv` validates input column
  names/types and invokes the model's embedded sklearn pipeline. Prediction
  responses do not include local paths. Artifact hashes are checked when
  available, and Python/scikit-learn version differences are surfaced as
  compatibility warnings. Old model artifacts without stored evaluation
  metrics are labeled as such rather than assigned invented metrics.
- The Streamlit dashboard now has a top-level `Prediction Dashboard` route
  separate from the existing MLOps dashboard, with CSV and medical-image tabs.
  Image inference remains disabled; see Phase 4 below.

### Evaluation, decisions, and experiment tracking

`src/training_engine.py` contains generic validation, model construction,
train/test splitting, metrics, model comparison, and a promotion decision
helper. `src/decision_engine.py` is a separate legacy drift-decision script.
The historical workflow writes JSON/CSV reports under `experiments/`.
`src/run_pipeline.py` and the customer-churn training workflow integrate with
MLflow. `src/mlflow_tracking.py` is a standalone test script that creates a
hard-coded test run when executed; reusable API and dashboard modules should
not import it for tracking.

### Data cleaning and current limitations

`src/data_cleaning.py` now provides conservative type normalization,
configurable exact-duplicate removal, missing/invalid-value reporting,
outlier reporting, constant/identifier/target-leakage warnings, and separate
cleaned CSV persistence. It protects the target from normalization and does
not delete outliers, constants, or suspected identifiers. Missing feature
values remain visible in the cleaned CSV; `src/training_engine.py` fits
numeric/categorical imputers as part of the serialized scikit-learn pipeline,
using training rows only. The existing dashboard adds a prepare/report/review
step before training. Drift comparison still uses its original reference and
incoming inputs without cleaning.

The generic training API previously split the data for metrics and then
retrained through a second split on the full dataframe. This was corrected:
the API now fits one preprocessing/model pipeline on its training partition
and evaluates it on its held-out partition. Tests verify both the partition
boundary and training-only imputer statistics.

Legacy artifacts may not include metric sidecars; their required features are
derived from the fitted sklearn pipeline. An artifact that cannot expose a
verifiable schema is listed as unavailable. New training artifacts persist
schema and metrics metadata. CSV prediction rejects missing/extra features,
numeric coercion failures, non-finite numeric values, oversized files, and
excess row counts. Probabilities are shown only when `predict_proba` exists;
they are not presented as calibrated confidence.

MLflow logging is optional and disabled unless `MLFLOW_TRACKING_URI` is set.
When configured, training parameters, scalar metrics, model, and metadata
artifacts are logged. A tracking failure is reported separately from the
successful local training result.

### Medical-image model verification

The Microsoft BiomedCLIP model card was checked at
`https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
The Hugging Face model metadata declares MIT licensing and the
`zero-shot-image-classification` pipeline. Its README example loads the
OpenCLIP model and uses the returned `preprocess` transform on images, then
compares image/text embeddings for user-supplied label prompts. The model card
states that use is solely for research/reproducibility and that deployed use
cases are out of scope. It is not a fixed, task-specific disease classifier
with a verified set of disease labels or clinical metrics.

Therefore, no image weights or inference dependencies were downloaded, no
image classifier was wired into the API, and the Prediction Dashboard exposes
an explicitly disabled image-upload control with the reason. No diabetes
prediction is made from images. A task-specific image model with a suitable
license, task labels, preprocessing specification, and intended use is still
needed before the image workflow can be safely enabled.

## Incremental implementation plan

1. **Phase 1 — Architecture inspection:** complete; module boundaries, flows,
   persistence, and compatibility constraints are recorded in this document.
2. **Phase 2 — Cleaning and validation:** complete. A reusable, configurable
   cleaning/reporting module handles training uploads, preserves original
   files, saves cleaned outputs separately, and keeps drift inputs unchanged.
   The generic training split issue is fixed and its leakage boundary is tested.
3. **Phase 3 — CSV prediction:** complete. Schema-aware model metadata,
   listing, CSV validation/inference, and a separate dashboard route reuse the
   serialized preprocessing pipeline. Prediction uploads do not retrain.
4. **Phase 4 — Medical-image prediction:** reviewed and intentionally blocked
   pending a compatible task-specific model. The checked BiomedCLIP model's
   research-only intended use and open-ended prompts do not justify presenting
   it as a deployed disease predictor.
5. **Phase 5 — Tracking and drift integration:** schema/metrics provenance is
   associated with new candidates; optional MLflow logging is controlled by
   `MLFLOW_TRACKING_URI`. Existing drift detection remains unchanged and
   reference/incoming datasets stay separate. No candidate is promoted.
6. **Phase 6 — Verification and documentation:** complete for the implemented
   cleaning and CSV prediction workflows. The full suite passes and live API
   and dashboard checks succeeded. Medical-image validation/inference tests
   remain deferred because the upload control and inference route are disabled
   until a compatible model is approved.

## Local safeguards and operations

The API uses `src/api_security.py`: with `API_AUTH_TOKEN` set, non-exempt
routes require a matching `X-API-Key`; with no token, only loopback clients are
allowed. Health and OpenAPI documentation routes are exempt. This is a
single-admin-key, local-development boundary, not production identity and
role-based access control. The request middleware rejects declared bodies over
22 MB, adds a request ID response header, and logs method, route, status, and
latency without logging request bodies.

Model loading retries one `OSError` once. Inference failures are reported and
recorded; the API does not silently select a different model, retrain, or
promote a candidate. New model sidecars include a SHA-256 hash and Python,
scikit-learn, NumPy, and Pandas versions. Hash mismatches block use. This
detects accidental artifact changes, but does not authenticate metadata if an
attacker can modify both the model and its sidecar. Legacy artifacts without
hashes or version metadata receive compatibility warnings.

`src/prediction_monitoring.py` stores a 30-day rolling SQLite record of model
ID, operation, status, error class, row count, and latency. It does not store
uploaded feature values or CSV content. The dashboard reads
`GET /monitoring/predictions`. The authenticated
`POST /operations/backup` endpoint uses SQLite's backup API for a consistent
copy of the configured SQLite MLflow database and copies the model registry
under `backups/`; backups remain local.

Raw dataset upload files are removed after the cleaned copy is created.
Cleaned CSVs are explicitly deletable through the dashboard or
`DELETE /dataset/cleaned/{filename}`. Files older than one hour are removed
lazily when a later upload or training request invokes cleanup; no background
scheduler is installed. Prediction files are processed in memory, and
monitoring events older than 30 days are pruned on the next event write.

The updated focused safeguarding and API tests passed (38 tests before the
cleaned-dataset deletion route was added); the final full-suite result is
reported with the task completion status.
