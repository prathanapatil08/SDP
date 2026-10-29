import pandas as pd
from sklearn.model_selection import train_test_split

from src.training_engine import (
    compare_models,
    decision_gate,
    infer_task_type,
    train_and_evaluate,
    validate_dataset_frame,
)


def make_classification_dataset():
    return pd.DataFrame(
        {
            "age": [22, 25, 27, 28, 30, 35, 36, 38, 42, 45, 50, 52, 54, 56, 60],
            "income": [20_000, 24_000, 30_000, 33_000, 42_000, 45_000, 50_000, 52_000, 55_000, 58_000, 62_000, 65_000, 70_000, 74_000, 80_000],
            "segment": ["A", "A", "B", "B", "A", "B", "C", "C", "A", "B", "C", "A", "B", "C", "C"],
            "churn": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        }
    )


def make_regression_dataset():
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature_b": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "target": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
        }
    )


def test_infer_task_type_for_binary_target():
    series = pd.Series(["No", "Yes", "No", "Yes"])
    assert infer_task_type(series) == "classification"


def test_validate_dataset_for_classification():
    dataset = make_classification_dataset()
    result = validate_dataset_frame(dataset, "churn", task_type="classification")
    assert result["task_type"] == "classification"
    assert result["duplicate_rows"] == 0
    assert result["data"].shape[0] == len(dataset)


def test_train_and_evaluate_classification_model():
    dataset = make_classification_dataset()
    result = train_and_evaluate(dataset, "churn", "logistic_regression", task_type="classification")
    assert result["task_type"] == "classification"
    assert set(result["metrics"]).issuperset({"accuracy", "precision", "recall", "f1"})


def test_train_and_evaluate_regression_model():
    dataset = make_regression_dataset()
    result = train_and_evaluate(dataset, "target", "linear_regression", task_type="regression")
    assert result["task_type"] == "regression"
    assert "rmse" in result["metrics"]


def test_imputation_is_fitted_on_training_rows_only():
    dataset = pd.DataFrame(
        {
            "measurement": [float(value) for value in range(24)],
            "label": ["yes" if value % 2 else "no" for value in range(24)],
        }
    )
    train_indices, test_indices = train_test_split(
        dataset.index,
        test_size=0.25,
        random_state=17,
        stratify=dataset["label"],
    )
    dataset.loc[test_indices, "measurement"] = [
        1000.0 + value for value in range(len(test_indices))
    ]
    dataset.loc[train_indices[0], "measurement"] = float("nan")

    expected_training_median = dataset.loc[
        train_indices, "measurement"
    ].median()
    full_dataset_median = dataset["measurement"].median()
    result = train_and_evaluate(
        dataset,
        "label",
        "logistic_regression",
        task_type="classification",
        test_size=0.25,
        random_state=17,
    )
    fitted_imputer = (
        result["model"]
        .named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
    )

    assert expected_training_median != full_dataset_median
    assert fitted_imputer.statistics_[0] == expected_training_median


def test_compare_models_and_gate():
    dataset = make_classification_dataset()
    train_result = train_and_evaluate(dataset, "churn", "logistic_regression", task_type="classification")
    X_test = dataset.drop(columns=["churn"]).iloc[:5].copy()
    y_test = dataset["churn"].iloc[:5].copy()
    comparison = compare_models(
        train_result["model"],
        train_result["model"],
        X_test,
        y_test,
        "classification",
    )
    gate = decision_gate(comparison, "classification", required_gain=0.0)
    assert gate["action"] in {"PROMOTION_RECOMMENDED", "RETAIN_BASELINE"}
