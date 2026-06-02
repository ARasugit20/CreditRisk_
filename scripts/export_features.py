"""
export_features.py
Exports mart_features from DuckDB to parquet for ML pipeline.
Run after: dbt run (from credit_risk_dbt/)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "credit_risk.duckdb"

# Full ML mart export (single-table).
OUT_PATH = PROJECT_ROOT / "data" / "mart_features.parquet"

# Convenience exports so the existing sklearn pipeline can train/validate
# using the same CLI arguments as processed_* CSV splits.
TRAIN_OUT_PATH = PROJECT_ROOT / "data" / "mart_features_train.parquet"
VAL_OUT_PATH = PROJECT_ROOT / "data" / "mart_features_validation.parquet"
TEST_OUT_PATH = PROJECT_ROOT / "data" / "mart_features_test.parquet"

# Align with the temporal robustness experiment used elsewhere in this repo.
TRAIN_END_DATE = pd.Timestamp("2015-12-31")
VALIDATION_END_DATE = pd.Timestamp("2016-12-31")


def export_features() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DB_PATH}. Run scripts/load_to_duckdb.py and dbt run first."
        )

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM main.mart_features").df()
    con.close()

    # `issue_d` is the temporal boundary / as-of date used by the sklearn pipeline.
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    train_df = df.loc[df["issue_d"] <= TRAIN_END_DATE]
    validation_df = df.loc[
        (df["issue_d"] > TRAIN_END_DATE) & (df["issue_d"] <= VALIDATION_END_DATE)
    ]
    test_df = df.loc[df["issue_d"] > VALIDATION_END_DATE]

    train_df.to_parquet(TRAIN_OUT_PATH, index=False)
    validation_df.to_parquet(VAL_OUT_PATH, index=False)
    test_df.to_parquet(TEST_OUT_PATH, index=False)

    print(f"Exported {len(df):,} rows to {OUT_PATH}")
    print(f"  train: {len(train_df):,} -> {TRAIN_OUT_PATH.name}")
    print(f"  val:   {len(validation_df):,} -> {VAL_OUT_PATH.name}")
    print(f"  test:  {len(test_df):,} -> {TEST_OUT_PATH.name}")


if __name__ == "__main__":
    export_features()
