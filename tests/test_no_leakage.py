"""Ensure target and post-origination fields never enter model feature columns."""

from __future__ import annotations

import pandas as pd

from pipeline import feature_columns as pipeline_feature_columns
from preprocessing import (
    add_engineered_features,
    feature_columns,
    filter_target,
    preprocess_lending_club,
    transform_split,
)
from utils import DATE_COLUMN, EAD_COLUMN, TARGET_COLUMN


FORBIDDEN_FEATURES = {DATE_COLUMN, TARGET_COLUMN}


def _model_feature_columns(processed_frame: pd.DataFrame) -> list[str]:
    """Columns used as X after preprocessing (exclude label and EAD metadata)."""

    metadata = {TARGET_COLUMN, EAD_COLUMN}
    return [column for column in processed_frame.columns if column not in metadata]


def test_feature_columns_exclude_target_and_issue_date(minimal_lending_club_frame):
    """Raw feature selection must drop loan_status and issue_d before any fit."""

    filtered = filter_target(minimal_lending_club_frame)
    engineered, _ = add_engineered_features(filtered)
    candidates = feature_columns(engineered)

    assert FORBIDDEN_FEATURES.isdisjoint(candidates)
    assert pipeline_feature_columns(engineered) == candidates


def test_preprocessed_splits_exclude_leakage_columns(minimal_lending_club_frame, tmp_path, monkeypatch):
    """End-to-end preprocessing must not place issue_d or loan_status in model features."""

    synthetic_path = tmp_path / "synthetic_lending_club.csv"
    minimal_lending_club_frame.to_csv(synthetic_path, index=False)

    monkeypatch.setattr("utils.DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr("utils.OUTPUTS_DIR", tmp_path / "outputs", raising=False)
    monkeypatch.setattr("utils.MODELS_DIR", tmp_path / "models", raising=False)
    monkeypatch.setattr("preprocessing.DATA_DIR", tmp_path, raising=False)
    monkeypatch.setattr("preprocessing.OUTPUTS_DIR", tmp_path / "outputs", raising=False)
    monkeypatch.setattr("pipeline.MODELS_DIR", tmp_path / "models", raising=False)

    report = preprocess_lending_club(
        input_path=str(synthetic_path),
        split_strategy="random",
        output_prefix="test_",
    )

    processed_train = pd.read_csv(tmp_path / "test_processed_train.csv")
    model_features = _model_feature_columns(processed_train)

    assert FORBIDDEN_FEATURES.isdisjoint(model_features)
    assert report["final_feature_count"] == len(model_features)


def test_transform_split_does_not_leak_raw_target(minimal_lending_club_frame, tmp_path, monkeypatch):
    """Transformed matrices must not carry issue_d; loan_status stays as label only."""

    from pipeline import fit_preprocessing_pipeline
    from preprocessing import sanitize_feature_names

    filtered = filter_target(minimal_lending_club_frame)
    engineered, _ = add_engineered_features(filtered)
    candidates = feature_columns(engineered)
    preprocessor, names = fit_preprocessing_pipeline(engineered, feature_names=candidates)
    names = sanitize_feature_names(names)

    processed = transform_split(preprocessor, engineered.iloc[:3], names)
    model_features = _model_feature_columns(processed)

    assert DATE_COLUMN not in model_features
    assert DATE_COLUMN not in processed.columns
    assert TARGET_COLUMN in processed.columns
    assert TARGET_COLUMN not in model_features
