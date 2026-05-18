"""Segment-wise discrimination and calibration analysis across models."""

from __future__ import annotations

import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from pipeline import transform_with_pipeline
from plot_style import apply_plot_style
from preprocessing import (
    add_engineered_features,
    drop_sparse_and_identifier_columns,
    filter_target,
    split_data,
    split_temporally,
)
from utils import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    TARGET_COLUMN,
    configure_logging,
    ensure_directories,
    load_csv,
    predict_proba_positive,
)


MODEL_ARTIFACTS = {
    "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "xgboost": MODELS_DIR / "xgboost.pkl",
    "mlp": MODELS_DIR / "mlp.pkl",
}


def load_preprocessing_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the preprocessing split context for segment analysis."""

    report_path = OUTPUTS_DIR / "preprocessing_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    input_path = report.get("input_path", str(DATA_DIR / "lending_club_sample.csv"))
    split_strategy = report.get("split_strategy", "random")
    date_column = report.get("date_column", "issue_d")
    train_end_date = report.get("train_end_date", "2014-12-31")
    validation_end_date = report.get("validation_end_date", "2015-12-31")

    raw = load_csv(input_path)
    filtered = filter_target(raw)
    engineered, _ = add_engineered_features(filtered)
    cleaned, _ = drop_sparse_and_identifier_columns(engineered)
    if split_strategy == "temporal":
        _, _, test_frame = split_temporally(cleaned, date_column, train_end_date, validation_end_date)
    else:
        _, _, test_frame = split_data(cleaned)
    return cleaned, test_frame.reset_index(drop=True)


def score_models_with_pipeline(test_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Run each available model on transformed test features."""

    pipeline_artifact = MODELS_DIR / "preprocessing_pipeline.pkl"
    if not pipeline_artifact.exists():
        raise FileNotFoundError(
            f"Expected preprocessing pipeline artifact at {pipeline_artifact}. Run preprocessing first."
        )
    preprocessing_pipeline = joblib.load(pipeline_artifact)
    feature_names = (
        preprocessing_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    )
    x_test = transform_with_pipeline(preprocessing_pipeline, test_frame, feature_names)

    predictions: dict[str, np.ndarray] = {}
    for name, artifact in MODEL_ARTIFACTS.items():
        if not artifact.exists():
            logging.warning("Skipping missing artifact for %s: %s", name, artifact)
            continue
        model = joblib.load(artifact)
        x_aligned = x_test.reindex(columns=getattr(model, "feature_names_in_", x_test.columns), fill_value=0)
        predictions[name] = predict_proba_positive(model, x_aligned)
    if not predictions:
        raise FileNotFoundError("No model artifacts found for segment analysis.")
    return predictions


def segment_metrics(frame: pd.DataFrame, probabilities: dict[str, np.ndarray], segment_column: str) -> pd.DataFrame:
    """Compute per-segment AUC and Brier metrics for each model."""

    rows: list[dict[str, float | str | int]] = []
    for segment_value, segment_df in frame.groupby(segment_column, dropna=False):
        positions = segment_df.index.to_numpy()
        y = segment_df[TARGET_COLUMN].to_numpy()
        for model_name, preds in probabilities.items():
            segment_preds = preds[positions]
            auc = roc_auc_score(y, segment_preds) if len(np.unique(y)) > 1 else np.nan
            rows.append(
                {
                    "segment_type": segment_column,
                    "segment_value": segment_value,
                    "model": model_name,
                    "n_samples": len(segment_df),
                    "roc_auc": auc,
                    "brier_score": brier_score_loss(y, segment_preds),
                }
            )
    return pd.DataFrame(rows)


def save_calibration_panels(frame: pd.DataFrame, probabilities: dict[str, np.ndarray]) -> None:
    """Save one calibration panel per grade with model overlays."""

    apply_plot_style()
    grades = ["A", "B", "C", "D", "E", "F", "G"]
    figure, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, grade in enumerate(grades):
        axis = axes[idx]
        subset = frame.loc[frame["grade"] == grade]
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        if subset.empty:
            axis.set_title(f"Grade {grade} (no rows)")
            continue
        y = subset[TARGET_COLUMN].to_numpy()
        for model_name, preds in probabilities.items():
            pred_subset = preds[subset.index.to_numpy()]
            frac_pos, mean_pred = calibration_curve(y, pred_subset, n_bins=8, strategy="quantile")
            axis.plot(mean_pred, frac_pos, marker="o", label=model_name)
        axis.set_title(f"Grade {grade}")
        axis.set_xlabel("Mean predicted PD")
        axis.set_ylabel("Observed default rate")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].axis("off")
        axes[-1].legend(handles, labels, loc="center")
    figure.suptitle("Calibration Curves by Grade")
    figure.tight_layout()
    figure.savefig(OUTPUTS_DIR / "segment_calibration_by_grade.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run segment-level analysis and export tabular/figure outputs."""

    configure_logging()
    ensure_directories()
    _, test_frame = load_preprocessing_context()
    probabilities = score_models_with_pipeline(test_frame)

    if "grade" in test_frame.columns:
        test_frame["grade"] = pd.Categorical(
            test_frame["grade"],
            categories=["A", "B", "C", "D", "E", "F", "G"],
            ordered=True,
        )

    metrics = []
    if "grade" in test_frame.columns:
        metrics.append(segment_metrics(test_frame, probabilities, "grade"))
    if "purpose" in test_frame.columns:
        metrics.append(segment_metrics(test_frame, probabilities, "purpose"))
    if not metrics:
        raise ValueError("Expected segment columns `grade` and/or `purpose` in test data.")

    combined = pd.concat(metrics, ignore_index=True)
    combined.to_csv(OUTPUTS_DIR / "segment_performance.csv", index=False)
    styled = (
        combined.sort_values(["segment_type", "segment_value", "model"])
        .style.background_gradient(subset=["roc_auc"], cmap="RdYlGn")
        .background_gradient(subset=["brier_score"], cmap="RdYlGn_r")
    )
    (OUTPUTS_DIR / "segment_performance_styled.html").write_text(styled.to_html(), encoding="utf-8")

    if "grade" in test_frame.columns:
        save_calibration_panels(test_frame, probabilities)
    logging.info("Saved segment analysis outputs in %s", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
