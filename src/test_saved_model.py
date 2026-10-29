
import pandas as pd
import joblib
import os

# 1. Check whether the model file exists
model_path = "models/baseline_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError("Baseline model file not found!")

print("Model file exists.")

# 2. Load the saved model
model = joblib.load(model_path)

print("Model loaded successfully.")

# 3. Load sample customers from the cleaned dataset
df = pd.read_csv(
    "data/reference/cleaned_customer_churn.csv"
)

# Use five records as sample input
X_sample = df.drop(columns=["Churn"]).head(5)
actual = df["Churn"].head(5)

# 4. Generate predictions
predictions = model.predict(X_sample)

# 5. Display actual and predicted results
results = pd.DataFrame({
    "Actual_Churn": actual.values,
    "Predicted_Churn": predictions
})

print("\n--- SAMPLE PREDICTIONS ---")
print(results)

print("\nVerification completed successfully!")
