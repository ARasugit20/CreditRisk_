"""Ablation study for engineered LendingClub credit-risk features."""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from utils import (
    ENGINEERED_FEATURES,
    RANDOM_STATE,
    configure_logging,
    ensure_directories,
    load_train_test_frames,
    log_table,
    prepare_xy,
    predict_proba_positive,
    save_model,
    save_table,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation study."""

    parser = argparse.ArgumentParser(description="Run engineered-feature ablation with XGBoost.")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV to temporally split.")
    parser.add_argument("--train-path", type=str, default=None, help="Preprocessed train CSV.")
    parser.add_argument("--test-path", type=str, default=None, help="Preprocessed test CSV.")
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=200_000,
        help="Stratified train sample size. Use 0 to train on all temporal training rows.",
    )
    return parser.parse_args()


def engineered_columns(frame: pd.DataFrame) -> list[str]:
    """Return exact or one-hot-expanded engineered feature columns present in a frame."""

    columns: list[str] = []
    for feature in ENGINEERED_FEATURES:
        columns.extend(
            column for column in frame.columns if column == feature or column.startswith(f"{feature}_")
        )
    return sorted(set(columns))


def stratified_training_sample(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Take a reproducible stratified sample from the temporal training set only."""

    if sample_size <= 0 or sample_size >= len(x_train):
        return x_train, y_train

    combined = x_train.copy()
    combined["_target"] = y_train.to_numpy()
    sampled = (
        combined.groupby("_target", group_keys=False)
        .sample(frac=sample_size / len(combined), random_state=RANDOM_STATE)
        .drop(columns="_target")
    )
    sampled_y = y_train.loc[sampled.index]
    logging.info("Using %d-row stratified train sample for ablation.", len(sampled))
    return sampled, sampled_y


def make_xgboost_pipeline() -> Pipeline:
    """Create an XGBoost pipeline with SMOTE confined to training folds/data."""

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("xgboost", model),
        ]
    )


def evaluate_pipeline(
    name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    """Fit an XGBoost pipeline and evaluate AUC and Brier score on temporal test data."""

    logging.info("Training %s ablation model.", name)
    pipeline = make_xgboost_pipeline()
    pipeline.fit(x_train, y_train)
    probabilities = predict_proba_positive(pipeline, x_test)
    save_model(pipeline, f"{name.lower().replace(' ', '_')}_ablation_xgboost.pkl")
    return {
        "Model": name,
        "Test AUC": roc_auc_score(y_test, probabilities),
        "Brier Score": brier_score_loss(y_test, probabilities),
    }


def run_ablation_study(
    data_path: str | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    train_sample_size: int = 200_000,
) -> pd.DataFrame:
    """Train XGBoost with and without engineered features and compare test metrics."""

    ensure_directories()
    train_frame, test_frame = load_train_test_frames(
        train_path=train_path,
        test_path=test_path,
        data_path=data_path,
    )

    full_x_train, y_train = prepare_xy(train_frame)
    full_x_test, y_test = prepare_xy(test_frame)
    full_x_train, full_x_test = full_x_train.align(full_x_test, join="outer", axis=1, fill_value=0)

    drop_columns = engineered_columns(full_x_train)
    reduced_x_train = full_x_train.drop(columns=drop_columns, errors="ignore")
    reduced_x_test = full_x_test.drop(columns=drop_columns, errors="ignore")
    logging.info("Dropping engineered features for ablation: %s", ", ".join(drop_columns))

    sampled_full_x_train, sampled_y_train = stratified_training_sample(
        full_x_train,
        y_train,
        train_sample_size,
    )
    sampled_reduced_x_train = reduced_x_train.loc[sampled_full_x_train.index]

    rows = [
        evaluate_pipeline(
            "Full Features",
            sampled_full_x_train,
            sampled_y_train,
            full_x_test,
            y_test,
        ),
        evaluate_pipeline(
            "Without Engineered Features",
            sampled_reduced_x_train,
            sampled_y_train,
            reduced_x_test,
            y_test,
        ),
    ]
    results = pd.DataFrame(rows)

    full_metrics = results.loc[results["Model"] == "Full Features"].iloc[0]
    reduced_metrics = results.loc[results["Model"] == "Without Engineered Features"].iloc[0]
    auc_gain = float(full_metrics["Test AUC"] - reduced_metrics["Test AUC"])
    brier_gain = float(reduced_metrics["Brier Score"] - full_metrics["Brier Score"])
    summary = pd.DataFrame(
        [
            {
                "AUC Improvement": auc_gain,
                "Brier Improvement": brier_gain,
                "Summary": (
                    f"Feature engineering improved AUC by {auc_gain:.4f}, "
                    f"Brier by {brier_gain:.4f}"
                ),
            }
        ]
    )
    save_table(results, "ablation_results.csv")
    save_table(summary, "ablation_summary.csv")
    log_table("Ablation Summary", summary)
    return results


def main() -> None:
    """Run ablation study from the command line."""

    configure_logging()
    args = parse_args()
    results = run_ablation_study(
        data_path=args.data_path,
        train_path=args.train_path,
        test_path=args.test_path,
        train_sample_size=args.train_sample_size,
    )
    log_table("Ablation Results", results)


if __name__ == "__main__":
    main()
