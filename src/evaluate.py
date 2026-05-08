"""Evaluate trained LendingClub default models on validation or test data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import (
    F1_THRESHOLD,
    MODELS_DIR,
    RESULTS_DIR,
    TARGET_COLUMN,
    configure_logging,
    ensure_directories,
    prepare_xy,
    predict_proba_positive,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate trained credit-risk models.")
    parser.add_argument("--test-path", type=str, default="data/processed_test.csv")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "xgboost"],
        help="Model artifact basenames in models/, without .pkl.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=F1_THRESHOLD,
        help="Probability threshold for class metrics.",
    )
    parser.add_argument(
        "--model-suffix",
        type=str,
        default="",
        help="Optional suffix for model artifact names, e.g. _temporal.",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="",
        help="Optional prefix for result files, e.g. temporal_.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=300,
        help="Bootstrap resamples for 95% confidence intervals (0 to disable).",
    )
    return parser.parse_args()


def load_processed_split(path: str | Path) -> pd.DataFrame:
    """Load a processed evaluation split."""

    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")
    logging.info("Loading evaluation split from %s", resolved)
    return pd.read_csv(resolved)


def load_named_model(model_name: str, model_suffix: str = "") -> object:
    """Load a model artifact from the models directory."""

    path = MODELS_DIR / f"{model_name}{model_suffix}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    logging.info("Loading model %s from %s", model_name, path)
    return joblib.load(path)


def save_curve_plots(
    model_name: str,
    y_true: pd.Series,
    probabilities,
    results_prefix: str = "",
) -> None:
    """Save ROC and precision-recall curves for one model."""

    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title(f"{model_name} ROC Curve")
    plt.savefig(
        RESULTS_DIR / f"{results_prefix}{model_name}_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.savefig(
        RESULTS_DIR / f"{results_prefix}{model_name}_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_confusion_matrix_plot(
    model_name: str,
    y_true: pd.Series,
    predictions,
    results_prefix: str = "",
) -> None:
    """Save a confusion matrix plot for one model."""

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Fully Paid", "Charged Off"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title(f"{model_name} Confusion Matrix")
    plt.savefig(
        RESULTS_DIR / f"{results_prefix}{model_name}_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error using uniform probability bins."""

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probabilities, bins, right=True) - 1
    ece = 0.0
    total = len(probabilities)
    for bin_id in range(n_bins):
        mask = indices == bin_id
        if not np.any(mask):
            continue
        bin_confidence = probabilities[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += np.abs(bin_accuracy - bin_confidence) * (mask.sum() / total)
    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric_fn,
    n_samples: int = 300,
    random_state: int = 42,
) -> tuple[float, float]:
    """Estimate a 95% bootstrap confidence interval for a metric."""

    if n_samples <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(random_state)
    values = []
    n = len(y_true)
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        sampled_y = y_true[idx]
        if len(np.unique(sampled_y)) < 2:
            continue
        values.append(metric_fn(sampled_y, probabilities[idx]))
    if not values:
        return (float("nan"), float("nan"))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def evaluate_models(
    test_path: str | Path = "data/processed_test.csv",
    model_names: list[str] | None = None,
    threshold: float = F1_THRESHOLD,
    model_suffix: str = "",
    results_prefix: str = "",
    bootstrap_samples: int = 300,
) -> pd.DataFrame:
    """Evaluate selected model artifacts and save metrics/plots."""

    ensure_directories()
    test_frame = load_processed_split(test_path)
    x_test, y_test = prepare_xy(test_frame)
    names = model_names or ["logistic_regression", "random_forest", "xgboost"]

    rows: list[dict[str, float | str]] = []
    for model_name in names:
        try:
            model = load_named_model(model_name, model_suffix=model_suffix)
        except FileNotFoundError as error:
            logging.warning("%s", error)
            continue

        feature_names = getattr(model, "feature_names_in_", x_test.columns)
        aligned_x_test, _ = prepare_xy(test_frame, feature_names=feature_names)
        probabilities = predict_proba_positive(model, aligned_x_test)
        predictions = (probabilities >= threshold).astype(int)

        report = classification_report(
            y_test,
            predictions,
            target_names=["Fully Paid", "Charged Off"],
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions)
        y_array = y_test.to_numpy(dtype=int)
        prob_array = np.asarray(probabilities, dtype=float)
        auc_ci = bootstrap_ci(y_array, prob_array, roc_auc_score, n_samples=bootstrap_samples)
        pr_auc_ci = bootstrap_ci(
            y_array,
            prob_array,
            average_precision_score,
            n_samples=bootstrap_samples,
        )
        brier_ci = bootstrap_ci(y_array, prob_array, brier_score_loss, n_samples=bootstrap_samples)
        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "ROC AUC": roc_auc_score(y_test, probabilities),
                "ROC AUC CI Low": auc_ci[0],
                "ROC AUC CI High": auc_ci[1],
                "PR AUC": average_precision_score(y_test, probabilities),
                "PR AUC CI Low": pr_auc_ci[0],
                "PR AUC CI High": pr_auc_ci[1],
                "Brier Score": brier_score_loss(y_test, probabilities),
                "Brier CI Low": brier_ci[0],
                "Brier CI High": brier_ci[1],
                "ECE": expected_calibration_error(y_array, prob_array),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "Threshold": threshold,
                "TN": int(matrix[0, 0]),
                "FP": int(matrix[0, 1]),
                "FN": int(matrix[1, 0]),
                "TP": int(matrix[1, 1]),
            }
        )

        pd.DataFrame(report).transpose().to_csv(
            RESULTS_DIR / f"{results_prefix}{model_name}_classification_report.csv"
        )
        save_confusion_matrix_plot(model_name, y_test, predictions, results_prefix=results_prefix)
        save_curve_plots(model_name, y_test, probabilities, results_prefix=results_prefix)

    results = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False)
    results.to_csv(RESULTS_DIR / f"{results_prefix}test_metrics.csv", index=False)
    (RESULTS_DIR / f"{results_prefix}metrics.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "model_suffix": model_suffix,
                "results_prefix": results_prefix,
                "bootstrap_samples": bootstrap_samples,
                "models": results.to_dict(orient="records"),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    logging.info("Saved test metrics to %s.", RESULTS_DIR / f"{results_prefix}test_metrics.csv")
    logging.info("Saved JSON metrics to %s.", RESULTS_DIR / f"{results_prefix}metrics.json")
    return results


def main() -> None:
    """Run model evaluation from the command line."""

    configure_logging()
    args = parse_args()
    results = evaluate_models(
        test_path=args.test_path,
        model_names=args.models,
        threshold=args.threshold,
        model_suffix=args.model_suffix,
        results_prefix=args.results_prefix,
        bootstrap_samples=args.bootstrap_samples,
    )
    logging.info("Test Metrics\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()
