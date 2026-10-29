
import json

COMPARISON_PATH = "experiments/model_comparison.json"
DECISION_PATH = "experiments/promotion_decision.json"


def main():
    with open(COMPARISON_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    baseline = results["baseline"]
    candidate = results["candidate"]

    accuracy_pass = (
        candidate["accuracy"] >= baseline["accuracy"]
    )

    f1_pass = (
        candidate["churn_f1_score"] >= baseline["churn_f1_score"]
    )

    approved = accuracy_pass and f1_pass

    if approved:
        action = "PROMOTION_RECOMMENDED"
        reason = (
            "Candidate passed both accuracy and churn F1 checks."
        )
    else:
        action = "RETAIN_BASELINE"
        reason = (
            "Candidate did not pass all required quality checks."
        )

    decision = {
        "baseline_accuracy": baseline["accuracy"],
        "candidate_accuracy": candidate["accuracy"],
        "baseline_churn_f1": baseline["churn_f1_score"],
        "candidate_churn_f1": candidate["churn_f1_score"],
        "accuracy_check_passed": accuracy_pass,
        "churn_f1_check_passed": f1_pass,
        "action": action,
        "reason": reason,
        "deployment_authorized": False
    }

    with open(DECISION_PATH, "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=4)

    print("\n--- PROMOTION DECISION ---")
    print(json.dumps(decision, indent=4))
    print("\nDecision saved to:", DECISION_PATH)
    print("\nNo model has been deployed or replaced.")


if __name__ == "__main__":
    main()
