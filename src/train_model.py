
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "reference" / "cleaned_customer_churn.csv"
MODEL_FILE = ROOT / "models" / "baseline_model.joblib"

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Autonomous-MLOps-Customer-Churn")

# 1. Load cleaned data
df = pd.read_csv(DATA_FILE)
print("Dataset loaded:", df.shape)

# 2. Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 3. Identify column types
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:", categorical_columns)
print("\nNumerical columns:", numerical_columns)

# 4. Create preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        ),
        ("numerical", "passthrough", numerical_columns),
    ]
)

# 5. Create model pipeline
n_estimators = 100
random_state = 42
class_weight = "balanced"

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                class_weight=class_weight,
            ),
        ),
    ]
)

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=random_state,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Train, evaluate, and track the model
with mlflow.start_run(run_name="baseline_random_forest") as run:
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # 8. Predict
    y_pred = model.predict(X_test)

    # 9. Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    churn_f1 = f1_score(y_test, y_pred, pos_label="Yes")
    matrix = confusion_matrix(y_test, y_pred, labels=["No", "Yes"])

    print("\n--- MODEL RESULTS ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy percentage: {accuracy * 100:.2f}%")
    print(f"Churn F1-score: {churn_f1:.4f}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))

    print("\n--- CONFUSION MATRIX (No, Yes) ---")
    print(matrix)

    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("class_weight", class_weight)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("training_rows", len(X_train))
    mlflow.log_param("testing_rows", len(X_test))

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("churn_f1_score", churn_f1)

    mlflow.set_tag("model_role", "baseline")
    mlflow.set_tag("target", "Churn")

    # Save confusion matrix as an experiment artifact
    matrix_file = ROOT / "experiments" / "baseline_confusion_matrix.txt"
    matrix_file.parent.mkdir(parents=True, exist_ok=True)
    matrix_file.write_text(
        "Confusion matrix (labels: No, Yes)\n" + str(matrix),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(matrix_file))

    # Save the trained model file as an MLflow artifact
    mlflow.log_artifact(
        str(MODEL_FILE),
        artifact_path="model_files",
    )

    # Save model locally for the existing API
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("\nBaseline model saved successfully!")
    print(f"Location: {MODEL_FILE}")
    print(f"MLflow run ID: {run.info.run_id}")

print("\nMLflow tracking completed successfully.")
