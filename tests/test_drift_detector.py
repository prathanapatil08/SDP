import pandas as pd
import pytest

from src.drift_detector import detect_drift


def test_unchanged_features_are_not_flagged_and_target_is_excluded():
    reference = pd.DataFrame(
        {
            "sepal_length": [4.8, 5.0, 5.1, 5.4, 5.7, 5.9, 6.1, 6.3],
            "species": ["setosa"] * 4 + ["versicolor"] * 4,
        }
    )
    incoming = reference.copy()
    incoming["species"] = ["virginica"] * len(incoming)

    report = detect_drift(reference, incoming, target_column="species")

    assert report["feature"].tolist() == ["sepal_length"]
    assert not report["drift_detected"].any()
    assert report.loc[0, "p_value"] == 1.0


def test_shifted_numeric_feature_is_flagged():
    reference = pd.DataFrame({"measurement": list(range(100))})
    incoming = pd.DataFrame({"measurement": list(range(100, 200))})

    report = detect_drift(reference, incoming)

    assert report.loc[0, "test"] == "KS two-sample test"
    assert report.loc[0, "p_value"] < 0.05
    assert bool(report.loc[0, "drift_detected"])
    assert "distribution changed" in report.loc[0, "explanation"]


def test_categorical_distribution_change_is_flagged():
    reference = pd.DataFrame({"color": ["red"] * 100 + ["blue"] * 100})
    incoming = pd.DataFrame({"color": ["red"] * 190 + ["green"] * 10})

    report = detect_drift(reference, incoming)

    assert "Chi-square" in report.loc[0, "test"]
    assert report.loc[0, "p_value"] < 0.05
    assert bool(report.loc[0, "drift_detected"])


def test_missing_values_are_compared_and_missingness_shift_is_flagged():
    reference = pd.DataFrame({"measurement": [float(value) for value in range(100)]})
    incoming = pd.DataFrame(
        {"measurement": [float(value) if value < 50 else None for value in range(100)]}
    )

    report = detect_drift(reference, incoming)

    assert report.loc[0, "reference_missing"] == 0
    assert report.loc[0, "incoming_missing"] == 50
    assert report.loc[0, "missingness_p_value"] < 0.05
    assert bool(report.loc[0, "drift_detected"])
    assert "Missing-value rate changed" in report.loc[0, "explanation"]


def test_mismatched_feature_columns_are_rejected():
    reference = pd.DataFrame({"height": [1.0, 2.0], "color": ["a", "b"]})
    incoming = pd.DataFrame({"height": [1.0, 2.0], "size": ["a", "b"]})

    with pytest.raises(ValueError, match="Feature column mismatch"):
        detect_drift(reference, incoming)


def test_constant_columns_and_small_samples_are_handled():
    reference = pd.DataFrame(
        {"constant": ["same", "same"], "measurement": [1.0, None]}
    )
    incoming = pd.DataFrame(
        {"constant": ["same", "same"], "measurement": [1.0, None]}
    )

    report = detect_drift(reference, incoming)

    assert report.loc[0, "test"] == "Categorical (constant)"
    assert report.loc[0, "p_value"] == 1.0
    assert report.loc[1, "comparison_status"] == "insufficient_data"
    assert not report["drift_detected"].any()
