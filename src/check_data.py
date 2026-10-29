
import pandas as pd

# Load the reference dataset
file_path = "data/reference/customer_churn.csv"
df = pd.read_csv(file_path)

# Display dataset information
print("\n--- DATASET SHAPE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- CHURN DISTRIBUTION ---")
if "Churn" in df.columns:
    print(df["Churn"].value_counts())
else:
    print("WARNING: Churn column not found.")

print("\n--- DATA TYPES ---")
print(df.dtypes)
