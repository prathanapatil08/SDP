
import mlflow
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent.parent

# Connect to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Select the project experiment
mlflow.set_experiment("Autonomous-MLOps-Customer-Churn")

# Create a test experiment run
with mlflow.start_run(run_name="mlflow_connection_test"):
    mlflow.log_param("project", "Autonomous MLOps")
    mlflow.log_param("model_type", "RandomForestClassifier")

    mlflow.log_metric("test_accuracy", 0.80)
    mlflow.log_metric("test_f1_score", 0.65)

    mlflow.set_tag("purpose", "tracking_connection_test")

print("MLflow tracking test completed successfully.")
print("Open http://127.0.0.1:5000 to view the experiment.")
