"""Compare random-split and temporal-split experiment metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import RESULTS_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment comparison."""

    parser = argparse.ArgumentParser(description="Compare random and temporal experiment metrics.")
    parser.add_argument("--random-metrics", type=str, default="results/test_metrics.csv")
    parser.add_argument("--temporal-metrics", type=str, default="results/temporal_test_metrics.csv")
    parser.add_argument("--output", type=str, default="results/split_comparison.csv")
    return parser.parse_args()


def main() -> None:
    """Load metrics from both experiments and save a side-by-side table."""

    args = parse_args()
    random_df = pd.read_csv(args.random_metrics)
    temporal_df = pd.read_csv(args.temporal_metrics)

    random_df["Split"] = "random"
    temporal_df["Split"] = "temporal"
    combined = pd.concat([random_df, temporal_df], ignore_index=True)
    combined = combined.sort_values(["Model", "Split"]).reset_index(drop=True)

    pivot = combined.pivot_table(
        index="Model",
        columns="Split",
        values=["ROC AUC", "PR AUC", "Brier Score", "Precision", "Recall", "F1", "ECE"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    pivot = pivot.reset_index()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(output_path, index=False)
    print(f"Saved split comparison table to {output_path}")


if __name__ == "__main__":
    main()
