
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load cleaned data
df = pd.read_csv(
    "data/reference/cleaned_customer_churn.csv"
)

print("Dataset loaded:", df.shape)

# 2. Separate input features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 3. Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical columns:", categorical_columns)
print("\nNumerical columns:", numerical_columns)

# 4. Convert categorical values into numeric features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# 5. Create the model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# 6. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Train the model
print("\nTraining model...")
model.fit(X_train, y_train)

# 8. Predict on unseen test records
y_pred = model.predict(X_test)

# 9. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL RESULTS ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

# 10. Save the complete pipeline
joblib.dump(model, "models/baseline_model.joblib")

print("\nBaseline model saved successfully!")
print("Location: models/baseline_model.joblib")
