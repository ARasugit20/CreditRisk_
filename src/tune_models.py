"""Hyperparameter tuning and validation-only model selection for credit-risk models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from utils import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    configure_logging,
    ensure_directories,
    load_csv,
    predict_proba_positive,
    prepare_xy,
    resolve_path,
)


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load processed train/validation matrices for tuning and selection."""

    train_path = resolve_path("data/processed_train.csv")
    validation_path = resolve_path("data/processed_validation.csv")
    if train_path is None or validation_path is None:
        raise FileNotFoundError("Missing processed train/validation CSVs.")

    train_frame = load_csv(train_path)
    validation_frame = load_csv(validation_path)
    if TARGET_COLUMN not in train_frame.columns or TARGET_COLUMN not in validation_frame.columns:
        raise ValueError(f"Both splits must contain `{TARGET_COLUMN}`.")

    x_train, y_train = prepare_xy(train_frame)
    x_validation, y_validation = prepare_xy(validation_frame, feature_names=x_train.columns)
    return x_train, y_train, x_validation, y_validation


def run_grid_search(
    name: str,
    estimator,
    param_grid: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict, float]:
    """Run 5-fold stratified CV search and return best params/score."""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    if name == "xgboost":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=20,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=2,
            refit=True,
        )
        logging.info("Starting RandomizedSearchCV (n_iter=20) for %s.", name)
    else:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=2,
            refit=True,
        )
        logging.info("Starting GridSearchCV for %s.", name)
    search.fit(x_train, y_train)
    logging.info("%s best ROC AUC (CV): %.6f", name, search.best_score_)
    return search.best_params_, float(search.best_score_)


def make_model(model_name: str, best_params: dict):
    """Instantiate estimator from tuned params."""

    if model_name == "logistic_regression":
        lr_params = dict(best_params)
        lr_params.pop("solver", None)
        return LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=2000,
            solver="liblinear",
            **lr_params,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params,
        )
    raise ValueError(f"Unknown model: {model_name}")


def smote_comparison(
    best_params: dict[str, dict],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """Compare validation ROC AUC before/after SMOTE (train only)."""

    rows: list[dict[str, float | str]] = []
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        base_model = make_model(model_name, best_params[model_name])
        base_model.fit(x_train, y_train)
        base_auc = roc_auc_score(y_validation, predict_proba_positive(base_model, x_validation))

        smote_model = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", clone(base_model)),
            ]
        )
        smote_model.fit(x_train, y_train)
        smote_auc = roc_auc_score(y_validation, predict_proba_positive(smote_model, x_validation))
        rows.append(
            {
                "Model": model_name,
                "Validation AUC (No SMOTE)": float(base_auc),
                "Validation AUC (SMOTE train only)": float(smote_auc),
                "Delta (SMOTE - No SMOTE)": float(smote_auc - base_auc),
            }
        )

    result = pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)
    result.to_csv(RESULTS_DIR / "smote_comparison.csv", index=False)
    return result


def threshold_comparison(
    xgb_model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, float]:
    """Find F1-optimal validation threshold for XGBoost and compare to fixed cutoffs."""

    probabilities = predict_proba_positive(xgb_model, x_validation)
    candidate_thresholds = [value / 100 for value in range(5, 95)]
    scored = []
    for threshold in candidate_thresholds:
        prediction = (probabilities >= threshold).astype(int)
        scored.append(
            (
                threshold,
                f1_score(y_validation, prediction, zero_division=0),
                precision_score(y_validation, prediction, zero_division=0),
                recall_score(y_validation, prediction, zero_division=0),
            )
        )
    best_threshold, *_ = max(scored, key=lambda row: row[1])
    selected = [0.30, 0.40, best_threshold]

    rows = []
    for threshold in sorted(set(selected)):
        prediction = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "Threshold": threshold,
                "Validation Precision": precision_score(y_validation, prediction, zero_division=0),
                "Validation Recall": recall_score(y_validation, prediction, zero_division=0),
                "Validation F1": f1_score(y_validation, prediction, zero_division=0),
                "Validation ROC AUC": roc_auc_score(y_validation, probabilities),
            }
        )
    table = pd.DataFrame(rows).sort_values("Threshold").reset_index(drop=True)
    table.to_csv(RESULTS_DIR / "threshold_comparison.csv", index=False)
    return table, float(best_threshold)


def save_best_params(best_params: dict[str, dict], best_scores: dict[str, float]) -> None:
    """Persist best hyperparameters and CV scores."""

    payload = {
        model: {"best_params": params, "best_cv_roc_auc": best_scores[model]}
        for model, params in best_params.items()
    }
    (RESULTS_DIR / "best_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fit_and_save_final_models(best_params: dict[str, dict], x_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Fit final tuned estimators on train split and overwrite production artifacts."""

    artifact_map = {
        "logistic_regression": "logistic_regression.pkl",
        "random_forest": "random_forest.pkl",
        "xgboost": "xgboost.pkl",
    }
    for model_name, artifact in artifact_map.items():
        model = make_model(model_name, best_params[model_name])
        model.fit(x_train, y_train)
        destination = MODELS_DIR / artifact
        joblib.dump(model, destination)
        logging.info("Saved tuned %s model to %s", model_name, destination)


def main() -> None:
    """Run end-to-end tuning, SMOTE comparison, thresholding, and final retraining."""

    configure_logging()
    ensure_directories()
    x_train, y_train, x_validation, y_validation = load_splits()

    grids = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["liblinear"],
                "class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [200, 500],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [5, 10, 25],
                "class_weight": [None, "balanced"],
            },
        ),
        "xgboost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "max_depth": [3, 4, 5],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [200, 500],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "scale_pos_weight": [1, 3, 5],
            },
        ),
    }

    best_params: dict[str, dict] = {}
    best_scores: dict[str, float] = {}
    for model_name, (estimator, param_grid) in grids.items():
        params, score = run_grid_search(model_name, estimator, param_grid, x_train, y_train)
        best_params[model_name] = params
        best_scores[model_name] = score

    save_best_params(best_params, best_scores)
    logging.info("Saved best parameters to %s", RESULTS_DIR / "best_params.json")

    smote_table = smote_comparison(best_params, x_train, y_train, x_validation, y_validation)
    logging.info("Saved SMOTE comparison to %s", RESULTS_DIR / "smote_comparison.csv")
    logging.info("\n%s", smote_table.to_string(index=False))

    tuned_xgb = make_model("xgboost", best_params["xgboost"])
    tuned_xgb.fit(x_train, y_train)
    threshold_table, best_threshold = threshold_comparison(tuned_xgb, x_validation, y_validation)
    logging.info("Saved threshold comparison to %s", RESULTS_DIR / "threshold_comparison.csv")
    logging.info("Best validation F1 threshold for XGBoost: %.2f", best_threshold)
    logging.info("\n%s", threshold_table.to_string(index=False))

    fit_and_save_final_models(best_params, x_train, y_train)


if __name__ == "__main__":
    main()
