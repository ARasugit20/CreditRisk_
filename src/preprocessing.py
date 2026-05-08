"""Preprocess the real Kaggle LendingClub dataset for model training."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from utils import (
    DATA_DIR,
    DATE_COLUMN,
    EAD_COLUMN,
    LEAKAGE_COLUMNS,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    configure_logging,
    ensure_directories,
    load_csv,
    resolve_path,
)

LOW_CARDINALITY_THRESHOLD = 20
MISSINGNESS_THRESHOLD = 0.50
VALID_TARGETS = {"Charged Off": 1, "Fully Paid": 0}
IDENTIFIER_COLUMNS = [
    "id",
    "member_id",
    "url",
    "desc",
    "title",
    "zip_code",
    "policy_code",
]
ENGINEERED_FEATURE_COLUMNS = [
    "installment_to_income",
    "log_annual_inc",
    "log_revol_bal",
    "avg_fico",
    "revol_util_bins",
    "emp_length_numeric",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for real LendingClub preprocessing."""

    parser = argparse.ArgumentParser(description="Preprocess Kaggle LendingClub data.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/lending_club_sample.csv",
        help=(
            "Raw LendingClub CSV (default matches sample_data.py output). "
            "Use data/accepted_2007_to_2018Q4.csv for the full Kaggle file."
        ),
    )
    parser.add_argument(
        "--low-cardinality-threshold",
        type=int,
        default=LOW_CARDINALITY_THRESHOLD,
        help="Categorical columns with <= this many train values are one-hot encoded.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "temporal"],
        default="random",
        help="Use stratified random split or temporal date-based split.",
    )
    parser.add_argument(
        "--date-column",
        type=str,
        default=DATE_COLUMN,
        help="Date column used for temporal splitting.",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="2014-12-31",
        help="Inclusive train end date for temporal split (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--validation-end-date",
        type=str,
        default="2015-12-31",
        help="Inclusive validation end date for temporal split (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional filename prefix (e.g., temporal_) for generated splits/reports.",
    )
    return parser.parse_args()


def filter_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep Fully Paid and Charged Off loans and binarize the target."""

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing required target column `{TARGET_COLUMN}`.")

    filtered = frame.loc[frame[TARGET_COLUMN].isin(VALID_TARGETS)].copy()
    dropped = len(frame) - len(filtered)
    logging.info("Dropped %d rows with non-final loan_status values.", dropped)
    filtered[TARGET_COLUMN] = filtered[TARGET_COLUMN].map(VALID_TARGETS).astype(int)
    return filtered


def parse_emp_length(series: pd.Series) -> pd.Series:
    """Convert LendingClub employment length strings into numeric years."""

    cleaned = series.astype(str).str.lower().str.strip()
    cleaned = cleaned.str.replace("< 1 year", "0", regex=False)
    cleaned = cleaned.str.replace("10+ years", "10", regex=False)
    cleaned = cleaned.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def numeric_percent(series: pd.Series) -> pd.Series:
    """Convert percentages like '53.2%' into numeric proportions."""

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.where(numeric <= 1, numeric / 100)
    stripped = series.astype(str).str.replace("%", "", regex=False)
    return pd.to_numeric(stripped, errors="coerce") / 100


def add_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add credit-risk feature engineering from raw LendingClub fields when available."""

    engineered = frame.copy()
    created: list[str] = []

    if {"installment", "annual_inc"}.issubset(engineered.columns):
        monthly_income = pd.to_numeric(engineered["annual_inc"], errors="coerce") / 12
        installment = pd.to_numeric(engineered["installment"], errors="coerce")
        engineered["installment_to_income"] = installment / monthly_income.replace(0, np.nan)
        created.append("installment_to_income")

    if "annual_inc" in engineered.columns:
        annual_income = pd.to_numeric(engineered["annual_inc"], errors="coerce").clip(lower=0)
        engineered["log_annual_inc"] = np.log1p(annual_income)
        created.append("log_annual_inc")

    if "revol_bal" in engineered.columns:
        revolving_balance = pd.to_numeric(engineered["revol_bal"], errors="coerce").clip(lower=0)
        engineered["log_revol_bal"] = np.log1p(revolving_balance)
        created.append("log_revol_bal")

    if {"fico_range_low", "fico_range_high"}.issubset(engineered.columns):
        low = pd.to_numeric(engineered["fico_range_low"], errors="coerce")
        high = pd.to_numeric(engineered["fico_range_high"], errors="coerce")
        engineered["avg_fico"] = (low + high) / 2
        created.append("avg_fico")

    if "revol_util" in engineered.columns:
        utilization = numeric_percent(engineered["revol_util"])
        engineered["revol_util"] = utilization
        engineered["revol_util_bins"] = pd.cut(
            utilization,
            bins=[-np.inf, 0.30, 0.60, 0.80, np.inf],
            labels=["low", "medium", "high", "very_high"],
        ).astype("object")
        created.append("revol_util_bins")

    if "emp_length" in engineered.columns:
        engineered["emp_length_numeric"] = parse_emp_length(engineered["emp_length"])
        created.append("emp_length_numeric")

    logging.info("Created engineered features: %s", ", ".join(created) if created else "none")
    return engineered, created


def drop_sparse_and_identifier_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with excessive missingness and obvious identifiers/free text."""

    missing_fraction = frame.isna().mean()
    sparse_columns = missing_fraction[missing_fraction > MISSINGNESS_THRESHOLD].index.tolist()
    identifier_columns = [column for column in IDENTIFIER_COLUMNS if column in frame.columns]
    drop_columns = sorted(set(sparse_columns + identifier_columns))
    logging.info("Dropped %d sparse/identifier columns.", len(drop_columns))
    return frame.drop(columns=drop_columns, errors="ignore"), drop_columns


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 70/15/15 train, validation, and test splits."""

    train_frame, holdout_frame = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    validation_frame, test_frame = train_test_split(
        holdout_frame,
        test_size=0.50,
        stratify=holdout_frame[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    logging.info(
        "Split rows: %d train, %d validation, %d test.",
        len(train_frame),
        len(validation_frame),
        len(test_frame),
    )
    return train_frame.copy(), validation_frame.copy(), test_frame.copy()


def parse_issue_dates(series: pd.Series) -> pd.Series:
    """Parse LendingClub issue dates with support for mixed formats."""

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(series, format="%b-%Y", errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def split_temporally(
    frame: pd.DataFrame,
    date_column: str,
    train_end_date: str,
    validation_end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits using chronological boundaries."""

    if date_column not in frame.columns:
        raise ValueError(f"Temporal split requires date column `{date_column}`.")

    dated = frame.copy()
    dated["_split_date"] = parse_issue_dates(dated[date_column])
    dated = dated.dropna(subset=["_split_date"])

    train_end = pd.Timestamp(train_end_date)
    validation_end = pd.Timestamp(validation_end_date)
    if validation_end <= train_end:
        raise ValueError("validation_end_date must be after train_end_date.")

    train_frame = dated.loc[dated["_split_date"] <= train_end].drop(columns="_split_date")
    validation_frame = dated.loc[
        (dated["_split_date"] > train_end) & (dated["_split_date"] <= validation_end)
    ].drop(columns="_split_date")
    test_frame = dated.loc[dated["_split_date"] > validation_end].drop(columns="_split_date")

    if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
        raise ValueError(
            "Temporal split produced an empty partition. Adjust train/validation end dates."
        )

    logging.info(
        "Temporal split rows: %d train, %d validation, %d test.",
        len(train_frame),
        len(validation_frame),
        len(test_frame),
    )
    return train_frame.copy(), validation_frame.copy(), test_frame.copy()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model feature columns after excluding target and leakage columns."""

    excluded = {TARGET_COLUMN, DATE_COLUMN, *LEAKAGE_COLUMNS}
    return [column for column in frame.columns if column not in excluded]


def sanitize_feature_names(feature_names: list[str]) -> list[str]:
    """Make transformed feature names safe for XGBoost and CSV downstream usage."""

    counts: dict[str, int] = {}
    sanitized_names: list[str] = []
    for name in feature_names:
        safe_name = re.sub(r"[\[\]<>]", "_", str(name))
        safe_name = re.sub(r"\s+", "_", safe_name).strip("_")
        occurrence = counts.get(safe_name, 0)
        counts[safe_name] = occurrence + 1
        if occurrence:
            safe_name = f"{safe_name}_{occurrence}"
        sanitized_names.append(safe_name)
    return sanitized_names


def classify_feature_types(
    frame: pd.DataFrame,
    low_cardinality_threshold: int,
) -> tuple[list[str], list[str], list[str]]:
    """Classify features as numeric, low-cardinality categorical, or high-cardinality categorical."""

    features = feature_columns(frame)
    numeric_columns = [
        column for column in features if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [column for column in features if column not in numeric_columns]

    low_cardinality = [
        column
        for column in categorical_columns
        if frame[column].nunique(dropna=True) <= low_cardinality_threshold
    ]
    high_cardinality = [
        column for column in categorical_columns if column not in low_cardinality
    ]
    return numeric_columns, low_cardinality, high_cardinality


def make_preprocessor(
    numeric_columns: list[str],
    low_cardinality_columns: list[str],
    high_cardinality_columns: list[str],
) -> ColumnTransformer:
    """Build a sklearn transformer for imputation, encoding, and numeric scaling."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    low_cardinality_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    high_cardinality_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat_low", low_cardinality_pipeline, low_cardinality_columns),
            ("cat_high", high_cardinality_pipeline, high_cardinality_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_split(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Transform one split and append target plus portfolio EAD metadata if available."""

    transformed = preprocessor.transform(frame)
    processed = pd.DataFrame(transformed, columns=feature_names, index=frame.index)
    processed[TARGET_COLUMN] = frame[TARGET_COLUMN].to_numpy()
    if EAD_COLUMN in frame.columns:
        processed[EAD_COLUMN] = pd.to_numeric(frame[EAD_COLUMN], errors="coerce").fillna(0).to_numpy()
    return processed.reset_index(drop=True)


def save_processed_splits(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_prefix: str = "",
) -> None:
    """Save processed train, validation, and test CSV files."""

    train_frame.to_csv(DATA_DIR / f"{output_prefix}processed_train.csv", index=False)
    validation_frame.to_csv(DATA_DIR / f"{output_prefix}processed_validation.csv", index=False)
    test_frame.to_csv(DATA_DIR / f"{output_prefix}processed_test.csv", index=False)
    logging.info("Saved processed CSVs to %s.", DATA_DIR)


def save_report(report: dict[str, Any], output_prefix: str = "") -> None:
    """Save preprocessing decisions to a JSON report."""

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{output_prefix}preprocessing_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Saved preprocessing report to %s.", path)


def preprocess_lending_club(
    input_path: str | Path = "data/lending_club_sample.csv",
    low_cardinality_threshold: int = LOW_CARDINALITY_THRESHOLD,
    split_strategy: str = "random",
    date_column: str = DATE_COLUMN,
    train_end_date: str = "2014-12-31",
    validation_end_date: str = "2015-12-31",
    output_prefix: str = "",
) -> dict[str, Any]:
    """Run real LendingClub preprocessing and return a summary report."""

    ensure_directories()
    resolved_path = resolve_path(input_path, DATA_DIR)
    if resolved_path is None or not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}. Run `python src/sample_data.py` to create "
            "data/lending_club_sample.csv from data/accepted_2007_to_2018Q4.csv, or pass "
            "--input with the path to your accepted-loans CSV."
        )

    raw_frame = load_csv(resolved_path)
    target_filtered = filter_target(raw_frame)
    engineered_frame, engineered_features = add_engineered_features(target_filtered)
    cleaned_frame, dropped_columns = drop_sparse_and_identifier_columns(engineered_frame)
    if split_strategy == "temporal":
        train_frame, validation_frame, test_frame = split_temporally(
            cleaned_frame,
            date_column=date_column,
            train_end_date=train_end_date,
            validation_end_date=validation_end_date,
        )
    else:
        train_frame, validation_frame, test_frame = split_data(cleaned_frame)

    numeric_columns, low_cardinality_columns, high_cardinality_columns = classify_feature_types(
        train_frame,
        low_cardinality_threshold,
    )
    logging.info(
        "Feature groups: %d numeric, %d low-cardinality categorical, %d high-cardinality categorical.",
        len(numeric_columns),
        len(low_cardinality_columns),
        len(high_cardinality_columns),
    )

    preprocessor = make_preprocessor(
        numeric_columns,
        low_cardinality_columns,
        high_cardinality_columns,
    )
    preprocessor.fit(train_frame)
    transformed_feature_names = sanitize_feature_names(preprocessor.get_feature_names_out().tolist())

    processed_train = transform_split(preprocessor, train_frame, transformed_feature_names)
    processed_validation = transform_split(preprocessor, validation_frame, transformed_feature_names)
    processed_test = transform_split(preprocessor, test_frame, transformed_feature_names)
    save_processed_splits(
        processed_train,
        processed_validation,
        processed_test,
        output_prefix=output_prefix,
    )

    report = {
        "input_path": str(resolved_path),
        "raw_shape": list(raw_frame.shape),
        "after_target_filter_shape": list(target_filtered.shape),
        "processed_train_shape": list(processed_train.shape),
        "processed_validation_shape": list(processed_validation.shape),
        "processed_test_shape": list(processed_test.shape),
        "target_mapping": VALID_TARGETS,
        "split_strategy": split_strategy,
        "date_column": date_column if split_strategy == "temporal" else None,
        "train_end_date": train_end_date if split_strategy == "temporal" else None,
        "validation_end_date": validation_end_date if split_strategy == "temporal" else None,
        "output_prefix": output_prefix,
        "dropped_sparse_or_identifier_columns": dropped_columns,
        "engineered_features_created": engineered_features,
        "leakage_columns_excluded_from_features": [
            column for column in LEAKAGE_COLUMNS if column in cleaned_frame.columns
        ],
        "numeric_columns": numeric_columns,
        "one_hot_categorical_columns": low_cardinality_columns,
        "ordinal_encoded_categorical_columns": high_cardinality_columns,
        "final_feature_count": len(transformed_feature_names),
        "ead_metadata_column_preserved": EAD_COLUMN in cleaned_frame.columns,
    }
    save_report(report, output_prefix=output_prefix)
    return report


def main() -> None:
    """Run preprocessing from the command line."""

    configure_logging()
    args = parse_args()
    report = preprocess_lending_club(
        input_path=args.input,
        low_cardinality_threshold=args.low_cardinality_threshold,
        split_strategy=args.split_strategy,
        date_column=args.date_column,
        train_end_date=args.train_end_date,
        validation_end_date=args.validation_end_date,
        output_prefix=args.output_prefix,
    )
    logging.info(
        "Preprocessing complete: %s train rows, %s validation rows, %s test rows, %s model features.",
        report["processed_train_shape"][0],
        report["processed_validation_shape"][0],
        report["processed_test_shape"][0],
        report["final_feature_count"],
    )


if __name__ == "__main__":
    main()
