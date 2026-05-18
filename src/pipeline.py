"""Reusable sklearn preprocessing pipeline for LendingClub data."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import DATE_COLUMN, LEAKAGE_COLUMNS, MODELS_DIR, TARGET_COLUMN


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Collect model feature columns while excluding target and leakage fields."""

    excluded = {TARGET_COLUMN, DATE_COLUMN, *LEAKAGE_COLUMNS}
    return [column for column in frame.columns if column not in excluded]


def infer_column_types(
    frame: pd.DataFrame,
    candidate_features: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature sets from a training frame."""

    features = candidate_features or feature_columns(frame)
    numeric_columns = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_columns = [column for column in features if column not in numeric_columns]
    return numeric_columns, categorical_columns


def make_preprocessing_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build an end-to-end sklearn Pipeline wrapping a ColumnTransformer."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocessor", transformer)])


def fit_preprocessing_pipeline(
    train_frame: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> tuple[Pipeline, list[str]]:
    """Fit preprocessing on training data only and return transformed names."""

    numeric_columns, categorical_columns = infer_column_types(train_frame, feature_names)
    pipeline = make_preprocessing_pipeline(numeric_columns, categorical_columns)
    pipeline.fit(train_frame)
    transformed_names = (
        pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    return pipeline, transformed_names


def transform_with_pipeline(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Transform one split and return a dataframe with named columns."""

    transformed = pipeline.transform(frame)
    return pd.DataFrame(transformed, columns=feature_names, index=frame.index)


def save_pipeline_artifact(pipeline: Pipeline, path: str | Path = "preprocessing_pipeline.pkl") -> Path:
    """Persist the fitted preprocessing pipeline as one serialized artifact."""

    destination = Path(path)
    if not destination.is_absolute():
        destination = MODELS_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    logging.info("Saved preprocessing pipeline artifact to %s", destination)
    return destination
