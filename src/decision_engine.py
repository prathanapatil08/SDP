
import pandas as pd
import json
from datetime import datetime

REPORT_PATH = "experiments/drift_report.csv"
DECISION_PATH = "experiments/decision_report.json"

# At least this many drifted features are required
# before recommending a retraining evaluation.
MIN_DRIFTED_FEATURES = 1


def make_decision(report):
    drifted = report[report["drift_detected"] == True]
    drift_count = len(drifted)

    if drift_count >= MIN_DRIFTED_FEATURES:
        action = "RETRAINING_EVALUATION_RECOMMENDED"
        reason = (
            f"{drift_count} feature(s) showed potential drift. "
            "Evaluate a candidate model before deployment."
        )
    else:
        action = "CONTINUE_MONITORING"
        reason = "No statistically significant feature drift detected."

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "features_checked": int(len(report)),
        "drifted_features": int(drift_count),
        "drifted_feature_names": drifted["feature"].tolist(),
        "action": action,
        "reason": reason,
        "deployment_authorized": False
    }


def main():
    report = pd.read_csv(REPORT_PATH)

    decision = make_decision(report)

    print("\n--- AUTOMATIC DECISION ---")
    print(json.dumps(decision, indent=4))

    with open(DECISION_PATH, "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=4)

    print("\nDecision saved to:", DECISION_PATH)


if __name__ == "__main__":
    main()
