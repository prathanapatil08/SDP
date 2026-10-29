from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.training_engine import UPLOADS_DIR, validate_dataset_frame

CLEANED_DATA_DIR = UPLOADS_DIR / "cleaned"
IDENTIFIER_NAME = re.compile(r"(^id$|(^|_)id($|_)|identifier|uuid)", re.IGNORECASE)


@dataclass(frozen=True)
class CleaningConfig:
    remove_exact_duplicates: bool = True
    convert_numeric_columns: bool = True
    outlier_iqr_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.outlier_iqr_multiplier <= 0:
            raise ValueError("outlier_iqr_multiplier must be greater than zero.")


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(column): int(count) for column, count in frame.isna().sum().items()}


def _potentially_leaky_features(
    frame: pd.DataFrame,
    target_column: str | None,
) -> list[str]:
    if not target_column or target_column not in frame.columns:
        return []

    target = frame[target_column]
    flagged = []
    for column in frame.columns:
        if column == target_column:
            continue
        common = frame[column].notna() & target.notna()
        if common.any() and frame.loc[common, column].astype(str).equals(
            target.loc[common].astype(str)
        ):
            flagged.append(str(column))
    return flagged


def clean_dataset(
    frame: pd.DataFrame,
    *,
    target_column: str | None = None,
    task_type: str | None = None,
    config: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Conservatively normalize a dataset and report issues without hiding them."""
    settings = config or CleaningConfig()
    if frame is None or frame.empty:
        raise ValueError("The dataset is empty or missing.")
    if len(frame.columns) == 0:
        raise ValueError("The dataset must contain at least one column.")
    if not frame.columns.is_unique:
        raise ValueError("The dataset contains duplicate column names.")
    if target_column and target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is not in the dataset.")

    original = frame.copy(deep=True)
    result = frame.copy(deep=True)
    before_dtypes = {str(column): str(dtype) for column, dtype in result.dtypes.items()}
    before_missing = _missing_counts(result)
    before_rows = len(result)
    warnings: list[str] = []
    dtype_changes: list[dict[str, str]] = []
    invalid_values: dict[str, int] = {}

    for column in result.columns:
        if column == target_column:
            continue

        values = result[column]
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            trimmed = values.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            trimmed = trimmed.replace(r"^\s*$", np.nan, regex=True)
            result[column] = trimmed

            if settings.convert_numeric_columns:
                non_missing = result[column].dropna()
                converted = pd.to_numeric(result[column], errors="coerce")
                if not non_missing.empty and converted[non_missing.index].notna().all():
                    result[column] = converted
                elif not non_missing.empty and converted.notna().any():
                    invalid_values[str(column)] = int(
                        (result[column].notna() & converted.isna()).sum()
                    )
                    warnings.append(
                        f"Column '{column}' contains mixed numeric and non-numeric "
                        "values; it was preserved as categorical."
                    )

        if pd.api.types.is_numeric_dtype(result[column]):
            numeric = pd.to_numeric(result[column], errors="coerce")
            infinite = numeric.notna() & ~np.isfinite(numeric)
            if infinite.any():
                invalid_values[str(column)] = (
                    invalid_values.get(str(column), 0) + int(infinite.sum())
                )
                numeric.loc[infinite] = np.nan
                result[column] = numeric
                warnings.append(
                    f"Column '{column}' contains non-finite numeric values; "
                    "they were marked missing."
                )

        after_dtype = str(result[column].dtype)
        before_dtype = before_dtypes[str(column)]
        if before_dtype != after_dtype:
            dtype_changes.append(
                {
                    "column": str(column),
                    "before": before_dtype,
                    "after": after_dtype,
                }
            )

    duplicate_count = int(result.duplicated().sum())
    if settings.remove_exact_duplicates and duplicate_count:
        result = result.drop_duplicates(keep="first").reset_index(drop=True)

    constant_columns = [
        str(column)
        for column in result.columns
        if column != target_column and result[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        warnings.append(
            "Constant or all-missing feature columns were retained for review: "
            + ", ".join(constant_columns)
        )

    identifier_columns = []
    for column in result.columns:
        if column == target_column:
            continue
        unique_ratio = result[column].nunique(dropna=True) / max(len(result), 1)
        if IDENTIFIER_NAME.search(str(column)) or unique_ratio >= 0.98:
            identifier_columns.append(str(column))
    if identifier_columns:
        warnings.append(
            "Possible identifier/high-cardinality columns were retained for "
            "review: " + ", ".join(identifier_columns)
        )

    leakage_features = _potentially_leaky_features(result, target_column)
    if leakage_features:
        warnings.append(
            "Feature values exactly match the target on labeled rows and may "
            "leak labels: " + ", ".join(leakage_features)
        )

    outlier_counts: dict[str, int] = {}
    for column in result.select_dtypes(include=["number"]).columns:
        if column == target_column:
            continue
        values = result[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            lower = q1 - settings.outlier_iqr_multiplier * iqr
            upper = q3 + settings.outlier_iqr_multiplier * iqr
            count = int(((result[column] < lower) | (result[column] > upper)).sum())
        outlier_counts[str(column)] = count
    if any(outlier_counts.values()):
        warnings.append(
            "Potential numeric outliers were reported but not removed."
        )

    # Imputation is deliberately deferred to the training pipeline, which fits
    # its imputers on the training partition only.
    imputation = {
        str(column): (
            "median, fitted on training rows"
            if pd.api.types.is_numeric_dtype(result[column])
            else "most_frequent, fitted on training rows"
        )
        for column in result.columns
        if column != target_column and result[column].isna().any()
    }

    validation_error = None
    target_rows = None
    target_classes = None
    target_class_counts = None
    if target_column:
        try:
            validation = validate_dataset_frame(
                result,
                target_column,
                task_type=task_type,
            )
            target_rows = int(validation["row_count"])
            target_classes = int(validation["target_unique_count"])
            if validation["task_type"] == "classification":
                counts = validation["data"][target_column].value_counts()
                target_class_counts = {
                    str(label): int(count) for label, count in counts.items()
                }
                if target_rows < 4:
                    raise ValueError(
                        "Classification requires at least four labeled rows "
                        "for a train/test split."
                    )
                rare_classes = [
                    label for label, count in target_class_counts.items() if count < 2
                ]
                if rare_classes:
                    raise ValueError(
                        "Each classification target class needs at least two "
                        "labeled rows. Insufficient classes: "
                        + ", ".join(rare_classes)
                    )
            elif target_rows < 2:
                raise ValueError(
                    "Regression requires at least two labeled rows for a train/test split."
                )
        except ValueError as error:
            validation_error = str(error)

    column_report = []
    for column in result.columns:
        values = result[column]
        column_report.append(
            {
                "column": str(column),
                "dtype_before": before_dtypes[str(column)],
                "dtype_after": str(values.dtype),
                "missing_before": before_missing[str(column)],
                "missing_after": int(values.isna().sum()),
                "invalid_values": invalid_values.get(str(column), 0),
                "unique_values": int(values.nunique(dropna=True)),
                "is_target": column == target_column,
                "is_constant_feature": (
                    column != target_column and values.nunique(dropna=True) <= 1
                ),
                "potential_outliers": outlier_counts.get(str(column), 0),
                "imputation_strategy": (
                    "none (target is preserved)"
                    if column == target_column
                    else imputation.get(str(column), "not required")
                ),
            }
        )

    report: dict[str, Any] = {
        "status": "invalid" if validation_error else "ready",
        "configuration": asdict(settings),
        "target_column": target_column,
        "task_type": task_type,
        "rows_before": int(before_rows),
        "rows_after": int(len(result)),
        "usable_labeled_rows": target_rows,
        "target_class_count": target_classes,
        "target_class_counts": target_class_counts,
        "columns_before": int(len(original.columns)),
        "columns_after": int(len(result.columns)),
        "duplicates_found": duplicate_count,
        "duplicates_removed": (
            duplicate_count if settings.remove_exact_duplicates else 0
        ),
        "missing_values_before": before_missing,
        "missing_values_after": _missing_counts(result),
        "changed_data_types": dtype_changes,
        "invalid_values": invalid_values,
        "potential_outliers": outlier_counts,
        "constant_columns": constant_columns,
        "potential_identifier_columns": identifier_columns,
        "potential_target_leakage_features": leakage_features,
        "warnings": warnings,
        "errors": [validation_error] if validation_error else [],
        "columns": column_report,
        "target_preserved": bool(target_column is None or target_column in result),
        "imputation_note": (
            "Missing feature values remain in the cleaned CSV; training "
            "imputers are fitted only on training rows and then applied to "
            "the held-out rows."
        ),
    }
    return result, report


def save_cleaned_dataset(
    frame: pd.DataFrame,
    source_name: str | None = None,
    directory: Path = CLEANED_DATA_DIR,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    source_stem = Path(source_name or "dataset").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_stem).strip("._")
    destination = directory / f"{safe_stem or 'dataset'}_{uuid.uuid4().hex}.csv"
    frame.to_csv(destination, index=False)
    return destination


def cleanup_expired_cleaned_datasets(
    *,
    directory: Path = CLEANED_DATA_DIR,
    max_age_seconds: int = 3600,
) -> int:
    if max_age_seconds < 0:
        raise ValueError("max_age_seconds cannot be negative.")
    if not directory.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for path in directory.glob("*.csv"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed
