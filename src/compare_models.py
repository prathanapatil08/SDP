
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "data/reference/cleaned_customer_churn.csv"
BASELINE_PATH = "models/baseline_model.joblib"
CANDIDATE_PATH = "models/candidate_model.joblib"
OUTPUT_PATH = "experiments/model_comparison.json"

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Recreate the baseline test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load both saved models
baseline = joblib.load(BASELINE_PATH)
candidate = joblib.load(CANDIDATE_PATH)

# Evaluate both on the same test records
baseline_predictions = baseline.predict(X_test)
candidate_predictions = candidate.predict(X_test)

results = {
    "baseline": {
        "accuracy": round(
            float(accuracy_score(y_test, baseline_predictions)), 4
        ),
        "churn_f1_score": round(
            float(f1_score(y_test, baseline_predictions, pos_label="Yes")), 4
        )
    },
    "candidate": {
        "accuracy": round(
            float(accuracy_score(y_test, candidate_predictions)), 4
        ),
        "churn_f1_score": round(
            float(f1_score(y_test, candidate_predictions, pos_label="Yes")), 4
        )
    }
}

print("\n--- MODEL COMPARISON ---")
print(json.dumps(results, indent=4))

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nComparison saved to:", OUTPUT_PATH)
