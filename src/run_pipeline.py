
import subprocess
import sys
import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

EXPERIMENT_NAME = "Autonomous-MLOps-Customer-Churn"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

STEPS = [
    (
        "Detect data drift",
        [
            "src/drift_detector.py",
            "--incoming",
            "data/incoming/incoming_customers_drifted.csv"
        ]
    ),
    (
        "Make retraining decision",
        ["src/decision_engine.py"]
    ),
    (
        "Train candidate model",
        ["src/retrain_model.py"]
    ),
    (
        "Compare models",
        ["src/compare_models.py"]
    ),
    (
        "Evaluate promotion decision",
        ["src/promotion_decision.py"]
    )
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=" * 55)
    print("AUTONOMOUS MLOPS WORKFLOW")
    print("=" * 55)

    with mlflow.start_run(run_name="complete_mlops_workflow"):
        mlflow.set_tag("workflow_role", "end_to_end")
        mlflow.set_tag("deployment_enabled", "false")

        try:
            for name, command in STEPS:
                print(f"\n\n>>> {name}")
                print("-" * 55)

                result = subprocess.run(
                    [PYTHON] + command,
                    cwd=ROOT
                )

                if result.returncode != 0:
                    mlflow.set_tag("workflow_status", "FAILED")
                    mlflow.set_tag("failed_step", name)
                    mlflow.log_param("failed_command", " ".join(command))

                    print(f"\nWorkflow stopped: {name} failed.")
                    sys.exit(result.returncode)

            # Read the final reports
            experiments = ROOT / "experiments"

            decision = read_json(
                experiments / "promotion_decision.json"
            )
            drift = read_json(
                experiments / "decision_report.json"
            )
            comparison = read_json(
                experiments / "model_comparison.json"
            )

            status = {
                "workflow": "COMPLETED",
                "promotion_action": decision["action"],
                "deployment_authorized": False,
                "note": "No model was deployed or replaced."
            }

            status_path = experiments / "workflow_status.json"

            with open(status_path, "w", encoding="utf-8") as file:
                json.dump(status, file, indent=4)

            # Log workflow outcome
            mlflow.set_tag("workflow_status", "COMPLETED")
            mlflow.set_tag(
                "promotion_action",
                str(decision["action"])
            )
            mlflow.set_tag("deployment_authorized", "false")

            # Log drift summary when available
            if isinstance(drift, dict):
                for key in ("features_checked", "drifted_features"):
                    if isinstance(drift.get(key), (int, float)):
                        mlflow.log_metric(
                            f"drift_{key}",
                            float(drift[key])
                        )

            # Log baseline and candidate comparison metrics
            if isinstance(comparison, dict):
                for model_name in ("baseline", "candidate"):
                    model_metrics = comparison.get(model_name, {})

                    if isinstance(model_metrics, dict):
                        for metric_name in ("accuracy", "churn_f1_score"):
                            value = model_metrics.get(metric_name)

                            if isinstance(value, (int, float)):
                                mlflow.log_metric(
                                    f"{model_name}_{metric_name}",
                                    float(value)
                                )

            # Save workflow reports as MLflow artifacts
            for filename in (
                "workflow_status.json",
                "decision_report.json",
                "promotion_decision.json",
                "model_comparison.json",
                "drift_report.csv"
            ):
                report_path = experiments / filename
                if report_path.is_file():
                    mlflow.log_artifact(str(report_path))

            print("\n\n" + "=" * 55)
            print("WORKFLOW FINISHED")
            print("=" * 55)
            print(json.dumps(status, indent=4))
            print("\nStatus saved to:", status_path)
            print("MLflow run ID:", mlflow.active_run().info.run_id)

        except Exception:
            mlflow.set_tag("workflow_status", "FAILED")
            raise


if __name__ == "__main__":
    main()
