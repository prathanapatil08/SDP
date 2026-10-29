
import pandas as pd

# Load the cleaned customer dataset
df = pd.read_csv(
    "data/reference/cleaned_customer_churn.csv"
)

# Select 1,000 records to simulate incoming customer data
incoming = df.sample(
    n=1000,
    random_state=100
).copy()

# Save incoming data separately
incoming.to_csv(
    "data/incoming/incoming_customers.csv",
    index=False
)

print("Incoming dataset created successfully!")
print("Number of records:", len(incoming))
print("Number of columns:", len(incoming.columns))

print("\nIncoming data saved to:")
print("data/incoming/incoming_customers.csv")

print("\nFirst five records:")
print(incoming.head())

print("\nChurn distribution:")
print(incoming["Churn"].value_counts())

