"""Hyperparameter tuning with AP-based scoring and learning curves."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
)
from xgboost import XGBClassifier

from plot_style import apply_plot_style
from utils import DATA_DIR, OUTPUTS_DIR, RANDOM_STATE, RESULTS_DIR, TARGET_COLUMN, configure_logging, ensure_directories, load_csv, predict_proba_positive, prepare_xy


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load processed train split for cross-validated tuning."""

    frame = load_csv(DATA_DIR / "processed_train.csv")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Training data must include `{TARGET_COLUMN}`.")
    return prepare_xy(frame)


def run_searches(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, dict]:
    """Tune logistic regression and XGBoost with AP scorer."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ap_scorer = make_scorer(average_precision_score, needs_proba=True)

    searches: dict[str, GridSearchCV] = {
        "logistic_regression": GridSearchCV(
            estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            param_grid={
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
                "solver": ["liblinear"],
                "penalty": ["l2"],
            },
            scoring=ap_scorer,
            n_jobs=-1,
            cv=cv,
            refit=True,
            verbose=1,
        ),
        "xgboost": GridSearchCV(
            estimator=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_grid={
                "max_depth": [3, 4, 5],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [200, 350],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "scale_pos_weight": [1, 3],
            },
            scoring=ap_scorer,
            n_jobs=-1,
            cv=cv,
            refit=True,
            verbose=1,
        ),
    }

    best: dict[str, dict] = {}
    for model_name, search in searches.items():
        logging.info("Running GridSearchCV for %s", model_name)
        search.fit(x_train, y_train)
        best[model_name] = {
            "best_params": search.best_params_,
            "best_cv_average_precision": float(search.best_score_),
        }
        validation_ap = average_precision_score(y_train, predict_proba_positive(search.best_estimator_, x_train))
        best[model_name]["train_average_precision_at_best"] = float(validation_ap)
    return best


def plot_learning_curves(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Plot train/validation AP curves versus train set size."""

    apply_plot_style()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ap_scorer = make_scorer(average_precision_score, needs_proba=True)
    train_sizes = np.linspace(0.2, 1.0, 6)

    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=4,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, (name, estimator) in zip(axes, models.items()):
        sizes, train_scores, validation_scores = learning_curve(
            estimator=estimator,
            X=x_train,
            y=y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring=ap_scorer,
            n_jobs=-1,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        axis.plot(sizes, train_scores.mean(axis=1), marker="o", label="Train AP")
        axis.plot(sizes, validation_scores.mean(axis=1), marker="s", label="Validation AP")
        axis.fill_between(
            sizes,
            validation_scores.mean(axis=1) - validation_scores.std(axis=1),
            validation_scores.mean(axis=1) + validation_scores.std(axis=1),
            alpha=0.15,
        )
        axis.set_title(name)
        axis.set_xlabel("Training examples")
        axis.set_ylabel("Average Precision")
        axis.legend()

    figure.suptitle("Learning Curves (5-fold Stratified CV)")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run AP-driven tuning and persist best parameters."""

    configure_logging()
    ensure_directories()
    x_train, y_train = load_training_data()
    best = run_searches(x_train, y_train)
    (RESULTS_DIR / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    logging.info("Saved best parameters to %s", RESULTS_DIR / "best_params.json")
    plot_learning_curves(x_train, y_train)
    logging.info("Saved learning curve figure to %s", OUTPUTS_DIR / "learning_curves.png")


if __name__ == "__main__":
    main()
