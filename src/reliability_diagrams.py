"""Reliability diagrams for pre- and post-calibrated credit-risk models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve

from utils import (
    DEFAULT_MODEL_ARTIFACTS,
    MODELS_DIR,
    OUTPUTS_DIR,
    calibrate_prefit_model,
    configure_logging,
    ensure_directories,
    get_model_feature_names,
    load_data_bundle,
    load_csv,
    load_model,
    prepare_xy,
    predict_proba_positive,
    resolve_path,
    save_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for reliability diagrams."""

    parser = argparse.ArgumentParser(description="Plot model calibration curves.")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV to temporally split.")
    parser.add_argument("--train-path", type=str, default=None, help="Preprocessed train CSV.")
    parser.add_argument(
        "--validation-path",
        type=str,
        default="data/processed_validation.csv",
        help="Processed validation CSV used only for fitting missing Platt calibrators.",
    )
    parser.add_argument("--test-path", type=str, default=None, help="Preprocessed test CSV.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_ARTIFACTS.keys()),
        choices=list(DEFAULT_MODEL_ARTIFACTS.keys()),
        help="Model names to include.",
    )
    parser.add_argument("--n-bins", type=int, default=10, help="Number of calibration bins.")
    parser.add_argument(
        "--output",
        type=str,
        default="reliability_diagrams.png",
        help="Output PNG filename.",
    )
    return parser.parse_args()


def get_or_fit_platt_calibrator(
    model_name: str,
    model: BaseEstimator,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> BaseEstimator:
    """Load an existing Platt calibrator or fit one around a prefit model."""

    path = MODELS_DIR / f"{model_name}_sigmoid_calibrated.pkl"
    if path.exists():
        return load_model(model_name, path)

    logging.info("Platt calibrator missing for %s; fitting it now.", model_name)
    calibrator = calibrate_prefit_model(model, x_calibration, y_calibration, method="sigmoid")
    save_model(calibrator, path)
    return calibrator


def plot_single_curve(
    axis: plt.Axes,
    y_true: pd.Series,
    probabilities: np.ndarray,
    title: str,
    n_bins: int,
) -> None:
    """Plot one reliability curve on an axis."""

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="uniform",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.plot(mean_predicted, fraction_positive, marker="o", linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Fraction of Positives")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)


def create_reliability_diagrams(
    model_names: list[str],
    data_path: str | None = None,
    train_path: str | None = None,
    validation_path: str | None = "data/processed_validation.csv",
    test_path: str | None = None,
    n_bins: int = 10,
    output_filename: str = "reliability_diagrams.png",
) -> Path:
    """Create a 2xN reliability diagram grid for raw and Platt-calibrated models."""

    ensure_directories()
    fig, axes = plt.subplots(2, len(model_names), figsize=(5 * len(model_names), 9), squeeze=False)

    for column, model_name in enumerate(model_names):
        model = load_model(model_name)
        data = load_data_bundle(
            train_path=train_path,
            test_path=test_path,
            data_path=data_path,
            model=model,
        )
        calibration_x = data.x_test
        calibration_y = data.y_test
        resolved_validation = resolve_path(validation_path) if validation_path else None
        if resolved_validation and resolved_validation.exists():
            validation_frame = load_csv(resolved_validation)
            feature_names = get_model_feature_names(model) or data.x_test.columns
            calibration_x, calibration_y = prepare_xy(validation_frame, feature_names=feature_names)
        else:
            logging.warning(
                "Validation split not found; fitting %s Platt calibrator on test data as fallback.",
                model_name,
            )

        pre_probability = predict_proba_positive(model, data.x_test)
        platt = get_or_fit_platt_calibrator(model_name, model, calibration_x, calibration_y)
        post_probability = predict_proba_positive(platt, data.x_test)

        display_name = model_name.replace("_", " ").title()
        plot_single_curve(
            axes[0, column],
            data.y_test,
            pre_probability,
            f"{display_name} - Pre",
            n_bins,
        )
        plot_single_curve(
            axes[1, column],
            data.y_test,
            post_probability,
            f"{display_name} - Post Platt",
            n_bins,
        )

    fig.suptitle("Reliability Diagrams: Pre vs Post Platt Scaling", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = OUTPUTS_DIR / output_filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved reliability diagrams to %s", output_path)
    return output_path


def main() -> None:
    """Run reliability plotting from the command line."""

    configure_logging()
    args = parse_args()
    create_reliability_diagrams(
        model_names=args.models,
        data_path=args.data_path,
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        n_bins=args.n_bins,
        output_filename=args.output,
    )


if __name__ == "__main__":
    main()
