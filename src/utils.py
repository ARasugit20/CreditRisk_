"""Shared utilities for the LendingClub credit-risk pipeline."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

RANDOM_STATE = 42
F1_THRESHOLD = 0.40
LGD = 0.60
TEMPORAL_CUTOFF = "2016-01-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET_COLUMN = "loan_status"
DATE_COLUMN = "issue_d"
EAD_COLUMN = "funded_amnt"

LEAKAGE_COLUMNS = [
    "int_rate",
    "grade",
    "sub_grade",
    "funded_amnt",
    "funded_amnt_inv",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "out_prncp",
    "last_pymnt_amnt",
    "last_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "collection_recovery_fee",
    "hardship_flag",
    "debt_settlement_flag",
    "pymnt_plan",
]

ENGINEERED_FEATURES = [
    "installment_to_income",
    "log_annual_inc",
    "log_revol_bal",
    "avg_fico",
    "revol_util_bins",
    "emp_length_numeric",
]

DEFAULT_MODEL_ARTIFACTS = {
    "logistic_regression": [
        "logistic_regression.pkl",
        "logreg.pkl",
        "lr_model.pkl",
        "logistic_model.pkl",
    ],
    "random_forest": [
        "random_forest.pkl",
        "rf.pkl",
        "rf_model.pkl",
        "random_forest_model.pkl",
    ],
    "xgboost": [
        "xgboost.pkl",
        "xgb.pkl",
        "xgb_model.pkl",
        "xgboost_model.pkl",
        "xgboost.json",
        "xgb_model.json",
    ],
    "mlp": [
        "mlp.pkl",
        "mlp_model.pkl",
        "neural_network.pkl",
    ],
}


@dataclass(frozen=True)
class DataBundle:
    """Container for train/test matrices and raw test exposure data."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    test_frame: pd.DataFrame


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise logging format for command-line scripts."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_directories() -> None:
    """Create standard output/model directories if they do not exist."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_path(path: str | Path | None, default_dir: Path | None = None) -> Path | None:
    """Resolve a user-supplied path relative to the project root or a default directory."""

    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if default_dir is not None and (default_dir / candidate).exists():
        return (default_dir / candidate).resolve()
    return (PROJECT_ROOT / candidate).resolve()


def find_existing_file(candidates: Sequence[str | Path], base_dir: Path) -> Path:
    """Return the first existing file from candidate names under a base directory."""

    for candidate in candidates:
        path = resolve_path(candidate, base_dir)
        if path and path.exists():
            return path
    names = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No file found in {base_dir} for candidates: {names}")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with logging."""

    csv_path = resolve_path(path)
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    logging.info("Loading data from %s", csv_path)
    return pd.read_csv(csv_path, low_memory=False)


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert LendingClub loan_status to binary: Charged Off=1, Fully Paid=0.

    Numeric columns are returned as int (expected 0/1 from processed splits).
    String columns must be only those two statuses, matching preprocessing.filter_target.
    """

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    label_to_int = {"charged off": 1, "fully paid": 0}
    mapped = normalized.map(label_to_int)
    if mapped.isna().any():
        unknown = sorted(normalized[mapped.isna()].unique())
        raise ValueError(
            f"Unrecognized target labels in {TARGET_COLUMN}: {unknown[:10]}. "
            "Expected only 'Charged Off' and 'Fully Paid' (run preprocessing.filter_target on raw data)."
        )
    return mapped.astype(int)


def split_temporally(
    frame: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    cutoff: str = TEMPORAL_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split loans into train/test sets using an issue-date cutoff."""

    if date_column not in frame.columns:
        raise ValueError(
            f"Temporal split requires `{date_column}`. Provide train/test CSVs if absent."
        )

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"`{date_column}` contains unparsable dates.")

    cutoff_date = pd.Timestamp(cutoff)
    train = frame.loc[dates < cutoff_date].copy()
    test = frame.loc[dates >= cutoff_date].copy()
    logging.info("Temporal split: %d train rows, %d test rows", len(train), len(test))
    return train, test


def load_train_test_frames(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    data_path: str | Path | None = None,
    cutoff: str = TEMPORAL_CUTOFF,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test frames from explicit CSVs or split a single LendingClub CSV."""

    explicit_train = resolve_path(train_path, DATA_DIR)
    explicit_test = resolve_path(test_path, DATA_DIR)
    if explicit_train and explicit_test:
        return load_csv(explicit_train), load_csv(explicit_test)

    default_train_names = ["processed_train.csv", "train.csv", "lending_club_train.csv"]
    default_test_names = ["processed_test.csv", "test.csv", "lending_club_test.csv"]
    try:
        default_train = find_existing_file(default_train_names, DATA_DIR)
        default_test = find_existing_file(default_test_names, DATA_DIR)
        return load_csv(default_train), load_csv(default_test)
    except FileNotFoundError:
        pass

    raw_path = resolve_path(data_path, DATA_DIR)
    if raw_path is None:
        raw_path = find_existing_file(
            ["lending_club_raw.csv", "accepted_2007_to_2018Q4.csv", "loan.csv"],
            DATA_DIR,
        )
    raw_frame = load_csv(raw_path)
    return split_temporally(raw_frame, cutoff=cutoff)


def get_model_feature_names(model: BaseEstimator) -> list[str] | None:
    """Extract feature names from fitted sklearn, imblearn, or XGBoost estimators."""

    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]

    if hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            names = get_model_feature_names(step)
            if names:
                return names

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        if booster.feature_names:
            return [str(name) for name in booster.feature_names]

    return None


def prepare_xy(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_names: Sequence[str] | None = None,
    drop_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a feature matrix and binary target, optionally aligned to model features."""

    if target_column not in frame.columns:
        raise ValueError(f"Target column `{target_column}` is missing.")

    y = normalize_target(frame[target_column])
    drop_set = {target_column, *LEAKAGE_COLUMNS}
    if drop_columns:
        drop_set.update(drop_columns)
    if DATE_COLUMN in frame.columns:
        drop_set.add(DATE_COLUMN)

    x = frame.drop(columns=[column for column in drop_set if column in frame.columns]).copy()
    if feature_names is not None:
        for feature in feature_names:
            if feature not in x.columns:
                x[feature] = 0
        x = x.loc[:, list(feature_names)]
    else:
        x = pd.get_dummies(x, drop_first=False)

    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    return x, y


def load_data_bundle(
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    data_path: str | Path | None = None,
    model: BaseEstimator | None = None,
    cutoff: str = TEMPORAL_CUTOFF,
    drop_columns: Iterable[str] | None = None,
) -> DataBundle:
    """Load LendingClub train/test data and align columns to a fitted model if supplied."""

    train_frame, test_frame = load_train_test_frames(train_path, test_path, data_path, cutoff)
    feature_names = get_model_feature_names(model) if model is not None else None
    x_train, y_train = prepare_xy(train_frame, feature_names=feature_names, drop_columns=drop_columns)
    x_test, y_test = prepare_xy(test_frame, feature_names=feature_names, drop_columns=drop_columns)

    if feature_names is None:
        x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0)

    return DataBundle(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, test_frame=test_frame)


def load_model(model_name: str, artifact_path: str | Path | None = None) -> BaseEstimator:
    """Load a fitted model artifact by explicit path or known model-name candidates."""

    if artifact_path is not None:
        path = resolve_path(artifact_path, MODELS_DIR)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    else:
        candidates = DEFAULT_MODEL_ARTIFACTS.get(model_name)
        if not candidates:
            raise ValueError(f"Unknown model name `{model_name}`.")
        path = find_existing_file(candidates, MODELS_DIR)

    logging.info("Loading %s model from %s", model_name, path)
    if path.suffix in {".joblib", ".pkl"}:
        try:
            return joblib.load(path)
        except Exception:
            with path.open("rb") as file:
                return pickle.load(file)

    if path.suffix in {".json", ".ubj"}:
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(path)
        return model

    raise ValueError(f"Unsupported model artifact extension: {path.suffix}")


def predict_proba_positive(model: BaseEstimator, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from a fitted probabilistic classifier."""

    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not expose predict_proba().")
    probabilities = model.predict_proba(x)
    return np.asarray(probabilities)[:, 1]


def compute_probability_metrics(y_true: pd.Series, y_probability: np.ndarray) -> dict[str, float]:
    """Compute probability-focused evaluation metrics."""

    return {
        "auc": roc_auc_score(y_true, y_probability),
        "brier": brier_score_loss(y_true, y_probability),
    }


def calibrate_prefit_model(
    model: BaseEstimator,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Fit a post-hoc calibrator around an already-fitted classifier."""

    calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrator.fit(x_calibration, y_calibration)
    return calibrator


def save_model(model: BaseEstimator, path: str | Path) -> Path:
    """Persist a fitted model or calibrator with joblib."""

    raw_path = Path(path)
    destination = raw_path if raw_path.is_absolute() else MODELS_DIR / raw_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    logging.info("Saved model artifact to %s", destination)
    return destination


def save_table(frame: pd.DataFrame, filename: str) -> Path:
    """Save a table to the outputs directory as CSV."""

    ensure_directories()
    path = OUTPUTS_DIR / filename
    frame.to_csv(path, index=False)
    logging.info("Saved table to %s", path)
    return path


def log_table(title: str, frame: pd.DataFrame) -> None:
    """Log a readable table without relying on print statements."""

    logging.info("%s\n%s", title, frame.to_string(index=False))
