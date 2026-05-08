"""Post-hoc probability calibration for LendingClub default models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss

from utils import (
    DEFAULT_MODEL_ARTIFACTS,
    MODELS_DIR,
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for calibration."""

    parser = argparse.ArgumentParser(description="Calibrate fitted credit-risk models.")
    parser.add_argument("--data-path", type=str, default=None, help="Single CSV to temporally split.")
    parser.add_argument("--train-path", type=str, default=None, help="Preprocessed train CSV.")
    parser.add_argument(
        "--validation-path",
        type=str,
        default="data/processed_validation.csv",
        help="Processed validation CSV used only for calibration fitting.",
    )
    parser.add_argument("--test-path", type=str, default=None, help="Preprocessed test CSV.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_ARTIFACTS.keys()),
        choices=list(DEFAULT_MODEL_ARTIFACTS.keys()),
        help="Model names to calibrate.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override a model artifact path, e.g. xgboost=models/my_xgb.pkl.",
    )
    return parser.parse_args()


def parse_artifact_overrides(values: list[str]) -> dict[str, Path]:
    """Parse MODEL=PATH artifact overrides from CLI arguments."""

    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid artifact override `{value}`. Use MODEL=PATH.")
        model_name, path = value.split("=", 1)
        overrides[model_name.strip()] = Path(path.strip())
    return overrides


def calibrate_models(
    model_names: list[str],
    data_path: str | None = None,
    train_path: str | None = None,
    validation_path: str | None = "data/processed_validation.csv",
    test_path: str | None = None,
    artifact_overrides: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Fit calibrators on validation data and compare Brier scores on test data."""

    ensure_directories()
    rows: list[dict[str, float | str]] = []
    overrides = artifact_overrides or {}

    for model_name in model_names:
        model = load_model(model_name, overrides.get(model_name))
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
            logging.info("Fitting %s calibrators on validation data.", model_name)
        else:
            logging.warning(
                "Validation split not found; fitting %s calibrators on test data as fallback.",
                model_name,
            )

        pre_probability = predict_proba_positive(model, data.x_test)
        pre_brier = brier_score_loss(data.y_test, pre_probability)

        for method_name, sklearn_method in {
            "Platt sigmoid": "sigmoid",
            "Isotonic": "isotonic",
        }.items():
            calibrator = calibrate_prefit_model(
                model,
                calibration_x,
                calibration_y,
                method=sklearn_method,
            )
            post_probability = predict_proba_positive(calibrator, data.x_test)
            post_brier = brier_score_loss(data.y_test, post_probability)

            artifact_name = f"{model_name}_{sklearn_method}_calibrated.pkl"
            save_model(calibrator, MODELS_DIR / artifact_name)
            rows.append(
                {
                    "Model": model_name,
                    "Calibration": method_name,
                    "Pre-Calibration Brier": pre_brier,
                    "Post-Calibration Brier": post_brier,
                    "Brier Improvement": pre_brier - post_brier,
                }
            )

    results = pd.DataFrame(rows).sort_values(["Model", "Calibration"])
    save_table(results, "calibration_brier_comparison.csv")
    return results


def main() -> None:
    """Run calibration from the command line."""

    configure_logging()
    args = parse_args()
    overrides = parse_artifact_overrides(args.artifact)
    results = calibrate_models(
        model_names=args.models,
        data_path=args.data_path,
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        artifact_overrides=overrides,
    )
    log_table("Calibration Brier Comparison", results)


if __name__ == "__main__":
    main()
