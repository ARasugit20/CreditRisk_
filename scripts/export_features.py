"""
export_features.py
Exports mart_features from DuckDB to parquet for ML pipeline.
Run after: dbt run (from credit_risk_dbt/)
"""

from __future__ import annotations

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "credit_risk.duckdb"
OUT_PATH = PROJECT_ROOT / "data" / "mart_features.parquet"


def export_features() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DB_PATH}. Run scripts/load_to_duckdb.py and dbt run first."
        )

    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM main.mart_features").df()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Exported {len(df):,} rows to {OUT_PATH}")
    print(df.dtypes)
    con.close()


if __name__ == "__main__":
    export_features()
