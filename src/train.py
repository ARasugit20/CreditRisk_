"""Train baseline and stronger supervised models for LendingClub default prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    configure_logging,
    ensure_directories,
    prepare_xy,
    predict_proba_positive,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""

    parser = argparse.ArgumentParser(description="Train credit-risk classifiers.")
    parser.add_argument("--train-path", type=str, default="data/processed_train.csv")
    parser.add_argument("--validation-path", type=str, default="data/processed_validation.csv")
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="",
        help="Optional suffix appended to saved model names, e.g. _temporal.",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="",
        help="Optional prefix for result files, e.g. temporal_.",
    )
    parser.add_argument(
        "--logistic-max-iter",
        type=int,
        default=2000,
        help="Max iterations for logistic regression solver.",
    )
    parser.add_argument(
        "--include-xgboost",
        action="store_true",
        help="Also train XGBoost if the package is installed.",
    )
    return parser.parse_args()


def load_processed_split(path: str | Path) -> pd.DataFrame:
    """Load a processed CSV split."""

    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")
    logging.info("Loading processed split from %s", resolved)
    return pd.read_csv(resolved)


def build_models(include_xgboost: bool, logistic_max_iter: int) -> dict[str, object]:
    """Create the supervised models used for comparison."""

    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=logistic_max_iter,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if include_xgboost:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
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

    return models


def train_models(
    train_path: str | Path = "data/processed_train.csv",
    validation_path: str | Path = "data/processed_validation.csv",
    include_xgboost: bool = False,
    model_suffix: str = "",
    results_prefix: str = "",
    logistic_max_iter: int = 2000,
) -> pd.DataFrame:
    """Train models, score validation AUC, and persist artifacts."""

    ensure_directories()
    train_frame = load_processed_split(train_path)
    validation_frame = load_processed_split(validation_path)

    if TARGET_COLUMN not in train_frame.columns or TARGET_COLUMN not in validation_frame.columns:
        raise ValueError(f"Processed splits must include `{TARGET_COLUMN}`.")

    x_train, y_train = prepare_xy(train_frame)
    x_validation, y_validation = prepare_xy(validation_frame, feature_names=x_train.columns)

    rows: list[dict[str, float | str]] = []
    for model_name, model in build_models(include_xgboost, logistic_max_iter).items():
        logging.info("Training %s.", model_name)
        model.fit(x_train, y_train)
        validation_probability = predict_proba_positive(model, x_validation)
        validation_auc = roc_auc_score(y_validation, validation_probability)

        artifact_path = MODELS_DIR / f"{model_name}{model_suffix}.pkl"
        joblib.dump(model, artifact_path)
        rows.append(
            {
                "Model": model_name,
                "Validation ROC AUC": validation_auc,
                "Artifact": str(artifact_path),
            }
        )
        logging.info("Saved %s to %s with validation AUC %.4f.", model_name, artifact_path, validation_auc)

    results = pd.DataFrame(rows).sort_values("Validation ROC AUC", ascending=False)
    results_path = RESULTS_DIR / f"{results_prefix}validation_model_comparison.csv"
    results.to_csv(results_path, index=False)
    (RESULTS_DIR / f"{results_prefix}training_metadata.json").write_text(
        json.dumps(
            {
                "train_rows": len(train_frame),
                "validation_rows": len(validation_frame),
                "feature_count": len(x_train.columns),
                "target_rate_train": float(y_train.mean()),
                "target_rate_validation": float(y_validation.mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.info("Saved validation comparison to %s.", results_path)
    return results


def main() -> None:
    """Run model training from the command line."""

    configure_logging()
    args = parse_args()
    results = train_models(
        train_path=args.train_path,
        validation_path=args.validation_path,
        include_xgboost=args.include_xgboost,
        model_suffix=args.model_suffix,
        results_prefix=args.results_prefix,
        logistic_max_iter=args.logistic_max_iter,
    )
    logging.info("Validation Model Comparison\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()
