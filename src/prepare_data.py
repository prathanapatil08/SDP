
import pandas as pd

# 1. Load the original dataset
df = pd.read_csv("data/reference/customer_churn.csv")

print("Original dataset shape:", df.shape)

# 2. Remove customer ID because it is not a useful
# predictive feature for this initial model
df = df.drop(columns=["customerID"])

# 3. Convert TotalCharges to numeric
# Invalid or blank values will become NaN
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 4. Check missing values after conversion
print("\nMissing values after conversion:")
print(df.isnull().sum())

# 5. Remove rows with missing values
df = df.dropna()

# 6. Save the cleaned dataset
df.to_csv("data/reference/cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset shape:", df.shape)
print("Cleaned dataset saved successfully!")

print("\nTarget distribution:")
print(df["Churn"].value_counts())
