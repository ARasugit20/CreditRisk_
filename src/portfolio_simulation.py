"""Expected-loss threshold simulation for a LendingClub loan portfolio."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from utils import (
    EAD_COLUMN,
    LGD,
    MODELS_DIR,
    OUTPUTS_DIR,
    calibrate_prefit_model,
    configure_logging,
    ensure_directories,
    get_model_feature_names,
    load_data_bundle,
    load_csv,
    load_model,
    log_table,
    prepare_xy,
    predict_proba_positive,
    resolve_path,
    save_model,
    save_table,
)

DEFAULT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for portfolio simulation."""

    parser = argparse.ArgumentParser(description="Simulate expected loss by PD threshold.")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV to temporally split.")
    parser.add_argument("--train-path", type=str, default=None, help="Preprocessed train CSV.")
    parser.add_argument(
        "--validation-path",
        type=str,
        default="data/processed_validation.csv",
        help="Processed validation CSV used only for fitting a missing Platt calibrator.",
    )
    parser.add_argument("--test-path", type=str, default=None, help="Preprocessed test CSV.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Approval thresholds for PD.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="portfolio_simulation.png",
        help="Output PNG filename.",
    )
    return parser.parse_args()


def get_xgboost_platt_model(
    model: BaseEstimator,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> BaseEstimator:
    """Load or fit the XGBoost Platt calibrator."""

    calibrator_path = MODELS_DIR / "xgboost_sigmoid_calibrated.pkl"
    if calibrator_path.exists():
        return load_model("xgboost", calibrator_path)

    logging.info("XGBoost Platt calibrator missing; fitting it now.")
    calibrator = calibrate_prefit_model(model, x_calibration, y_calibration, method="sigmoid")
    save_model(calibrator, calibrator_path)
    return calibrator


def simulate_thresholds(
    y_true: pd.Series,
    pd_scores: np.ndarray,
    ead: pd.Series,
    thresholds: list[float],
    lgd: float = LGD,
) -> pd.DataFrame:
    """Compute approval rate, expected loss, realized loss, and EL error by threshold."""

    rows: list[dict[str, float]] = []
    y_array = y_true.to_numpy(dtype=int)
    ead_array = ead.to_numpy(dtype=float)
    loan_expected_loss = pd_scores * lgd * ead_array

    for threshold in thresholds:
        approved = pd_scores < threshold
        predicted_el = float(loan_expected_loss[approved].sum())
        realized_loss = float((y_array[approved] * ead_array[approved]).sum())
        el_accuracy_error = (
            abs(predicted_el - realized_loss) / realized_loss if realized_loss > 0 else np.nan
        )

        rows.append(
            {
                "Threshold": threshold,
                "Approval Rate": float(approved.mean()),
                "Approved Loans": int(approved.sum()),
                "Total Portfolio EL": predicted_el,
                "Realized Loss": realized_loss,
                "EL Accuracy Error": el_accuracy_error,
            }
        )

    return pd.DataFrame(rows)


def plot_simulation(results: pd.DataFrame, output_filename: str) -> Path:
    """Plot approval rate, expected loss, and realized loss across thresholds."""

    output_path = OUTPUTS_DIR / output_filename
    fig, left_axis = plt.subplots(figsize=(11, 7))
    right_axis = left_axis.twinx()

    left_axis.plot(
        results["Threshold"],
        results["Approval Rate"] * 100,
        marker="o",
        color="#1f77b4",
        label="Approval Rate",
    )
    right_axis.plot(
        results["Threshold"],
        results["Total Portfolio EL"] / 1_000_000,
        marker="s",
        color="#ff7f0e",
        label="Total EL",
    )
    right_axis.plot(
        results["Threshold"],
        results["Realized Loss"] / 1_000_000,
        marker="^",
        color="#d62728",
        label="Realized Loss",
    )

    left_axis.set_xlabel("PD Approval Threshold")
    left_axis.set_ylabel("Approval Rate (%)")
    right_axis.set_ylabel("Loss ($ Millions)")
    left_axis.set_title("Portfolio Expected Loss Simulation")
    left_axis.grid(alpha=0.25)

    lines_left, labels_left = left_axis.get_legend_handles_labels()
    lines_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved portfolio simulation plot to %s", output_path)
    return output_path


def run_portfolio_simulation(
    data_path: str | None = None,
    train_path: str | None = None,
    validation_path: str | None = "data/processed_validation.csv",
    test_path: str | None = None,
    thresholds: list[float] | None = None,
    output_filename: str = "portfolio_simulation.png",
) -> pd.DataFrame:
    """Run the XGBoost post-Platt expected-loss threshold simulation."""

    ensure_directories()
    model = load_model("xgboost")
    data = load_data_bundle(
        train_path=train_path,
        test_path=test_path,
        data_path=data_path,
        model=model,
    )

    if EAD_COLUMN not in data.test_frame.columns:
        raise ValueError(f"Portfolio simulation requires `{EAD_COLUMN}` in the test data.")

    calibration_x = data.x_test
    calibration_y = data.y_test
    resolved_validation = resolve_path(validation_path) if validation_path else None
    if resolved_validation and resolved_validation.exists():
        validation_frame = load_csv(resolved_validation)
        feature_names = get_model_feature_names(model) or data.x_test.columns
        calibration_x, calibration_y = prepare_xy(validation_frame, feature_names=feature_names)
    else:
        logging.warning("Validation split not found; fitting XGBoost Platt calibrator on test fallback.")

    platt_model = get_xgboost_platt_model(model, calibration_x, calibration_y)
    pd_scores = predict_proba_positive(platt_model, data.x_test)
    results = simulate_thresholds(
        data.y_test,
        pd_scores,
        data.test_frame[EAD_COLUMN],
        thresholds or DEFAULT_THRESHOLDS,
    )
    save_table(results, "portfolio_simulation_summary.csv")
    plot_simulation(results, output_filename)
    return results


def main() -> None:
    """Run portfolio simulation from the command line."""

    configure_logging()
    args = parse_args()
    results = run_portfolio_simulation(
        data_path=args.data_path,
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        thresholds=args.thresholds,
        output_filename=args.output,
    )
    log_table("Portfolio Threshold Simulation", results)


if __name__ == "__main__":
    main()
