
import pandas as pd

# Load the current incoming dataset
df = pd.read_csv(
    "data/incoming/incoming_customers.csv"
)

print("Original MonthlyCharges statistics:")
print(df["MonthlyCharges"].describe())

# Deliberately increase MonthlyCharges by 40%
# This simulates a change in the customer population.
df["MonthlyCharges"] = df["MonthlyCharges"] * 1.40

# Save to a separate file so the original incoming
# dataset remains unchanged.
df.to_csv(
    "data/incoming/incoming_customers_drifted.csv",
    index=False
)

print("\nDrifted MonthlyCharges statistics:")
print(df["MonthlyCharges"].describe())

print("\nDrifted dataset saved successfully!")
print("File: data/incoming/incoming_customers_drifted.csv")
