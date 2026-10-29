
from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "incoming" / "incoming_customers.csv"
CANDIDATE_PATH = ROOT / "models" / "candidate_model.joblib"
METRICS_PATH = ROOT / "experiments" / "candidate_metrics.json"

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Autonomous-MLOps-Customer-Churn")


def main():
    # 1. Load incoming data
    df = pd.read_csv(DATA_PATH)
    print("Incoming dataset loaded:", df.shape)

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

    # 4. Prepare preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )

    # 5. Create candidate model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    # 6. Split incoming data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # 7. Train and track candidate in MLflow
    with mlflow.start_run(run_name="candidate_random_forest"):
        model.fit(X_train, y_train)

        # 8. Evaluate candidate
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, pos_label="Yes")

        metrics = {
            "accuracy": round(float(accuracy), 4),
            "churn_f1_score": round(float(f1), 4),
            "training_records": len(X_train),
            "testing_records": len(X_test),
            "candidate_model_path": str(
                CANDIDATE_PATH.relative_to(ROOT)
            )
        }

        # Log model parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42,
            "class_weight": "balanced",
            "test_size": 0.20,
            "training_records": len(X_train),
            "testing_records": len(X_test)
        })

        # Log evaluation metrics
        mlflow.log_metrics({
            "accuracy": float(accuracy),
            "churn_f1_score": float(f1)
        })

        mlflow.set_tags({
            "model_role": "candidate",
            "target": "Churn",
            "framework": "scikit-learn"
        })

        # 9. Save candidate separately
        joblib.dump(model, CANDIDATE_PATH)

        # Log the saved model file as an artifact
        mlflow.log_artifact(
            str(CANDIDATE_PATH),
            artifact_path="model_files"
        )

        # Save metrics locally and log them too
        with METRICS_PATH.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        mlflow.log_artifact(str(METRICS_PATH))

        print("\n--- CANDIDATE MODEL RESULTS ---")
        print(json.dumps(metrics, indent=4))
        print("\nCandidate saved to:", CANDIDATE_PATH)
        print("Metrics saved to:", METRICS_PATH)
        print("MLflow run ID:", mlflow.active_run().info.run_id)

    print("\nMLflow candidate tracking completed.")
    print("Baseline model was not changed.")


if __name__ == "__main__":
    main()
