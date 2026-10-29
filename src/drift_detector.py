import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, ks_2samp, random_table

REFERENCE_PATH = "data/reference/cleaned_customer_churn.csv"
SIGNIFICANCE_LEVEL = 0.05
PERMUTATION_SAMPLES = 999


def _missingness_test(
    reference: pd.Series,
    incoming: pd.Series,
) -> tuple[float | None, float]:
    table = [
        [int(reference.isna().sum()), int(reference.notna().sum())],
        [int(incoming.isna().sum()), int(incoming.notna().sum())],
    ]
    statistic, p_value = fisher_exact(table)
    finite_statistic = float(statistic)
    return (
        finite_statistic if math.isfinite(finite_statistic) else None,
        float(p_value),
    )


def _categorical_test(
    reference: pd.Series,
    incoming: pd.Series,
) -> tuple[str, float, float]:
    missing_value = object()
    pooled = pd.concat(
        [reference.reset_index(drop=True), incoming.reset_index(drop=True)],
        ignore_index=True,
    ).map(lambda value: missing_value if pd.isna(value) else str(value))
    codes, categories = pd.factorize(pooled, sort=False)
    reference_codes = codes[: len(reference)]
    incoming_codes = codes[len(reference) :]
    category_count = len(categories)

    reference_counts = np.bincount(reference_codes, minlength=category_count)
    incoming_counts = np.bincount(incoming_codes, minlength=category_count)
    observed = np.vstack([reference_counts, incoming_counts])

    if category_count == 1:
        return "Categorical (constant)", 0.0, 1.0

    chi_result = chi2_contingency(
        observed,
        correction=False,
    )
    statistic = chi_result.statistic
    asymptotic_p_value = chi_result.pvalue
    expected = chi_result.expected_freq

    if float(expected.min()) < 5:
        sampled_tables = random_table.rvs(
            observed.sum(axis=1),
            observed.sum(axis=0),
            size=PERMUTATION_SAMPLES,
            random_state=42,
        )
        expected_table = np.outer(
            observed.sum(axis=1),
            observed.sum(axis=0),
        ) / observed.sum()
        sampled_statistics = (
            (sampled_tables - expected_table) ** 2 / expected_table
        ).sum(axis=(1, 2))
        p_value = float(
            (np.count_nonzero(sampled_statistics >= statistic) + 1)
            / (PERMUTATION_SAMPLES + 1)
        )
        return (
            f"Chi-square permutation test ({PERMUTATION_SAMPLES} samples)",
            float(statistic),
            p_value,
        )

    return "Chi-square test", float(statistic), float(asymptotic_p_value)


def detect_drift(
    reference: pd.DataFrame,
    incoming: pd.DataFrame,
    target_column: str | None = None,
    significance_level: float = SIGNIFICANCE_LEVEL,
) -> pd.DataFrame:
    if reference.empty or incoming.empty:
        raise ValueError("Reference and incoming datasets must both contain rows.")

    reference_features = reference.drop(
        columns=(
            [target_column]
            if target_column and target_column in reference.columns
            else []
        ),
        errors="ignore",
    )
    incoming_features = incoming.drop(
        columns=(
            [target_column]
            if target_column and target_column in incoming.columns
            else []
        ),
        errors="ignore",
    )

    reference_columns = list(reference_features.columns)
    missing = sorted(set(reference_columns) - set(incoming_features.columns))
    extra = sorted(set(incoming_features.columns) - set(reference_columns))
    if missing or extra:
        raise ValueError(
            f"Feature column mismatch. Missing: {missing}; Extra: {extra}"
        )
    if not reference_columns:
        raise ValueError("At least one input feature is required for drift detection.")

    incoming_features = incoming_features.loc[:, reference_columns]
    results = []

    for column in reference_columns:
        ref = reference_features[column]
        new = incoming_features[column]
        missingness_statistic, missingness_p_value = _missingness_test(ref, new)

        if pd.api.types.is_numeric_dtype(ref):
            numeric_ref = pd.to_numeric(ref, errors="coerce")
            numeric_new = pd.to_numeric(new, errors="coerce")
            incoming_values = new.notna() & numeric_new.isna()
            if incoming_values.any():
                raise ValueError(
                    f"Feature '{column}' is numeric in the reference dataset "
                    "but contains non-numeric values in the incoming dataset."
                )

            valid_ref = numeric_ref.dropna()
            valid_new = numeric_new.dropna()
            if len(valid_ref) < 2 or len(valid_new) < 2:
                test_name = "KS two-sample test (insufficient values)"
                statistic = None
                p_value = None
                comparison_status = "insufficient_data"
            else:
                test = ks_2samp(valid_ref, valid_new, method="auto")
                test_name = "KS two-sample test"
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
                comparison_status = "compared"
        else:
            test_name, statistic, p_value = _categorical_test(ref, new)
            comparison_status = "compared"

        distribution_drift = p_value is not None and p_value < significance_level
        missingness_drift = missingness_p_value < significance_level
        drift_detected = distribution_drift or missingness_drift

        explanations = []
        if distribution_drift:
            explanations.append(
                f"Feature distribution changed (p={p_value:.4g})."
            )
        if missingness_drift:
            explanations.append(
                "Missing-value rate changed "
                f"(p={missingness_p_value:.4g})."
            )
        if not explanations and comparison_status == "insufficient_data":
            explanations.append(
                "Too few non-missing values for a distribution test."
            )
        if not explanations:
            explanations.append("No statistically significant change detected.")

        results.append(
            {
                "feature": column,
                "feature_type": (
                    "numeric"
                    if pd.api.types.is_numeric_dtype(ref)
                    else "categorical"
                ),
                "test": test_name,
                "statistic": statistic,
                "p_value": p_value,
                "reference_missing": int(ref.isna().sum()),
                "incoming_missing": int(new.isna().sum()),
                "missingness_statistic": missingness_statistic,
                "missingness_p_value": missingness_p_value,
                "comparison_status": comparison_status,
                "drift_detected": bool(drift_detected),
                "explanation": " ".join(explanations),
            }
        )

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Compare reference and incoming data for drift."
    )
    parser.add_argument(
        "--reference",
        default=REFERENCE_PATH,
        help="Path to the reference CSV file",
    )
    parser.add_argument(
        "--incoming",
        default="data/incoming/incoming_customers.csv",
        help="Path to the incoming CSV file",
    )
    parser.add_argument(
        "--target-column",
        default="Churn",
        help="Optional target column to exclude from feature comparisons",
    )
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    incoming = pd.read_csv(args.incoming)
    target_column = args.target_column or None
    results = detect_drift(reference, incoming, target_column=target_column)

    print("Reference records:", len(reference))
    print("Incoming records:", len(incoming))
    print("Reference file:", args.reference)
    print("Incoming file:", args.incoming)
    print("\n--- DRIFT DETECTION RESULTS ---")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    drift_count = int(results["drift_detected"].sum())
    print("\nFeatures checked:", len(results))
    print("Features with detected drift:", drift_count)
    if drift_count:
        print("Status: Potential data drift detected.")
    elif results["comparison_status"].eq("compared").any():
        print("Status: No statistically significant drift detected.")
    else:
        print("Status: Inconclusive; no features could be compared.")

    report_path = Path("experiments/drift_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_path, index=False)
    print("\nReport saved to", report_path)


if __name__ == "__main__":
    main()
