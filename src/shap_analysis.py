"""SHAP analysis for the fitted XGBoost probability-of-default model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator

from utils import (
    OUTPUTS_DIR,
    RANDOM_STATE,
    configure_logging,
    ensure_directories,
    load_data_bundle,
    load_model,
    log_table,
    save_table,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SHAP analysis."""

    parser = argparse.ArgumentParser(description="Generate SHAP plots for XGBoost.")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV to temporally split.")
    parser.add_argument("--train-path", type=str, default=None, help="Preprocessed train CSV.")
    parser.add_argument("--test-path", type=str, default=None, help="Preprocessed test CSV.")
    parser.add_argument("--sample-size", type=int, default=10_000, help="Test rows to sample.")
    return parser.parse_args()


def unwrap_tree_estimator(model: BaseEstimator) -> BaseEstimator:
    """Return the underlying XGBoost estimator from a plain or pipelined model."""

    if hasattr(model, "get_booster"):
        return model

    if hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            if hasattr(step, "get_booster"):
                return step

    raise TypeError("SHAP TreeExplainer requires an XGBoost estimator with get_booster().")


def normalize_shap_values(values: np.ndarray) -> np.ndarray:
    """Normalize SHAP output to a two-dimensional matrix for binary classification."""

    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, 1]
    return array


def normalize_explanation(explanation: shap.Explanation, x_sample: pd.DataFrame) -> shap.Explanation:
    """Return a two-dimensional SHAP Explanation suitable for summary plotting."""

    if np.asarray(explanation.values).ndim != 3:
        return explanation

    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 2:
        base_values = base_values[:, 1]
    return shap.Explanation(
        values=normalize_shap_values(explanation.values),
        base_values=base_values,
        data=x_sample.to_numpy(),
        feature_names=list(x_sample.columns),
    )


def save_current_figure(path: Path) -> None:
    """Save the active Matplotlib figure at publication quality and close it."""

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info("Saved SHAP plot to %s", path)


def compute_shap_importance(
    shap_matrix: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute mean absolute SHAP importance by feature."""

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    return (
        pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )


def run_shap_analysis(
    data_path: str | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
    sample_size: int = 10_000,
) -> pd.DataFrame:
    """Generate SHAP summary, importance, and top-feature dependence plots."""

    ensure_directories()
    model = load_model("xgboost")
    data = load_data_bundle(
        train_path=train_path,
        test_path=test_path,
        data_path=data_path,
        model=model,
    )

    x_sample = data.x_test.sample(
        n=min(sample_size, len(data.x_test)),
        random_state=RANDOM_STATE,
    )
    tree_model = unwrap_tree_estimator(model)
    logging.info("Computing SHAP values for %d sampled loans.", len(x_sample))
    explainer = shap.TreeExplainer(tree_model)
    explanation = explainer(x_sample)
    explanation = normalize_explanation(explanation, x_sample)
    shap_matrix = normalize_shap_values(explanation.values)

    shap.plots.beeswarm(explanation, max_display=20, show=False)
    save_current_figure(OUTPUTS_DIR / "shap_summary.png")

    importance = compute_shap_importance(shap_matrix, list(x_sample.columns))
    save_table(importance, "shap_feature_importance.csv")

    plt.figure(figsize=(10, 7))
    top_20 = importance.head(20).iloc[::-1]
    plt.barh(top_20["Feature"], top_20["Mean |SHAP|"], color="#1f77b4")
    plt.xlabel("Mean |SHAP|")
    plt.title("XGBoost SHAP Feature Importance")
    save_current_figure(OUTPUTS_DIR / "shap_importance.png")

    for feature in importance.head(3)["Feature"]:
        shap.dependence_plot(
            feature,
            shap_matrix,
            x_sample,
            show=False,
            interaction_index="auto",
        )
        safe_feature = "".join(character if character.isalnum() else "_" for character in feature)
        save_current_figure(OUTPUTS_DIR / f"shap_dependence_{safe_feature}.png")

    return importance.head(10)


def main() -> None:
    """Run SHAP analysis from the command line."""

    configure_logging()
    args = parse_args()
    top_features = run_shap_analysis(
        data_path=args.data_path,
        train_path=args.train_path,
        test_path=args.test_path,
        sample_size=args.sample_size,
    )
    log_table("Top 10 SHAP Features", top_features)


if __name__ == "__main__":
    main()
