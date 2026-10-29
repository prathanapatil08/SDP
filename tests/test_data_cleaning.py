import numpy as np
import pandas as pd

from src.data_cleaning import CleaningConfig, clean_dataset, save_cleaned_dataset


def test_cleaning_removes_duplicates_and_preserves_target_and_original(tmp_path):
    source = pd.DataFrame(
        {
            "measurement": ["1.0", "2.0", "2.0"],
            "label": ["yes", "no", "no"],
        }
    )
    original = source.copy(deep=True)

    cleaned, report = clean_dataset(source, target_column="label")

    assert len(cleaned) == 2
    assert cleaned["measurement"].dtype.kind == "f"
    assert cleaned["label"].tolist() == ["yes", "no"]
    assert report["duplicates_found"] == 1
    assert report["duplicates_removed"] == 1
    assert report["target_preserved"] is True
    pd.testing.assert_frame_equal(source, original)

    output_dir = tmp_path / "cleaned"
    output_path = save_cleaned_dataset(cleaned, "source.csv", directory=output_dir)

    assert output_path.is_file()
    assert output_path.parent == output_dir
    assert len(pd.read_csv(output_path)) == 2


def test_cleaning_reports_missing_categories_and_defers_train_only_imputation():
    source = pd.DataFrame(
        {
            "numeric": [1.0, np.nan, 3.0, 4.0],
            "category": ["a", None, "b", "a"],
            "label": ["yes", "no", "yes", "no"],
        }
    )

    cleaned, report = clean_dataset(source, target_column="label")

    assert cleaned["numeric"].isna().sum() == 1
    assert cleaned["category"].isna().sum() == 1
    assert report["missing_values_before"]["numeric"] == 1
    assert report["missing_values_after"]["category"] == 1
    assert report["columns"][0]["imputation_strategy"] == (
        "median, fitted on training rows"
    )
    assert report["columns"][1]["imputation_strategy"] == (
        "most_frequent, fitted on training rows"
    )
    assert report["columns"][2]["imputation_strategy"] == (
        "none (target is preserved)"
    )
    assert report["target_preserved"] is True


def test_cleaning_reports_outliers_constants_identifiers_and_leakage():
    source = pd.DataFrame(
        {
            "customer_id": range(100),
            "constant": ["same"] * 100,
            "label_copy": ["yes" if index % 2 else "no" for index in range(100)],
            "measurement": [float(index) for index in range(99)] + [10000.0],
            "label": ["yes" if index % 2 else "no" for index in range(100)],
        }
    )

    cleaned, report = clean_dataset(source, target_column="label")

    assert len(cleaned) == len(source)
    assert report["potential_outliers"]["measurement"] == 1
    assert "constant" in report["constant_columns"]
    assert "customer_id" in report["potential_identifier_columns"]
    assert "label_copy" in report["potential_target_leakage_features"]
    assert cleaned["measurement"].max() == 10000.0
    assert report["warnings"]


def test_mixed_numeric_column_is_preserved_and_reported():
    source = pd.DataFrame(
        {
            "mixed": ["1", "two", "3"],
            "label": ["a", "b", "a"],
        }
    )

    cleaned, report = clean_dataset(source, target_column="label")

    assert not pd.api.types.is_numeric_dtype(cleaned["mixed"])
    assert report["invalid_values"]["mixed"] == 1
    assert any("mixed numeric and non-numeric" in warning for warning in report["warnings"])


def test_config_can_retain_duplicates_and_uses_configured_outlier_rule():
    source = pd.DataFrame(
        {
            "measurement": [1.0, 1.0, 1.0, 100.0],
            "label": ["a", "a", "a", "b"],
        }
    )

    cleaned, report = clean_dataset(
        source,
        target_column="label",
        config=CleaningConfig(
            remove_exact_duplicates=False,
            outlier_iqr_multiplier=3.0,
        ),
    )

    assert len(cleaned) == len(source)
    assert report["duplicates_found"] == 2
    assert report["duplicates_removed"] == 0
    assert report["configuration"]["outlier_iqr_multiplier"] == 3.0


def test_invalid_or_empty_dataset_is_rejected():
    for frame in (pd.DataFrame(), pd.DataFrame({"value": []})):
        try:
            clean_dataset(frame)
        except ValueError as error:
            assert "empty" in str(error).lower()
        else:
            raise AssertionError("Expected empty datasets to be rejected.")


def test_invalid_target_is_reported_without_cleaning_target_values():
    source = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "label": [None, None, None],
        }
    )

    cleaned, report = clean_dataset(source, target_column="label")

    assert cleaned["label"].isna().all()
    assert report["status"] == "invalid"
    assert "no non-missing values" in report["errors"][0]


def test_classification_requires_enough_examples_per_target_class():
    source = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "label": ["common", "common", "common", "rare"],
        }
    )

    _, report = clean_dataset(
        source,
        target_column="label",
        task_type="classification",
    )

    assert report["status"] == "invalid"
    assert "at least two labeled rows" in report["errors"][0]
