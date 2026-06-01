"""
load_to_duckdb.py
Loads lending_club_sample.csv into DuckDB as raw.loans table.
Run before: dbt run (from credit_risk_dbt/)
"""

from __future__ import annotations

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "lending_club_sample.csv"
DB_PATH = PROJECT_ROOT / "data" / "credit_risk.duckdb"


def load_raw_loans() -> None:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Missing {RAW_CSV}. Add lending_club_sample.csv under data/ first."
        )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw.loans AS
        SELECT * FROM read_csv_auto('{RAW_CSV.as_posix()}', header=True)
        """
    )

    row_count = con.execute("SELECT COUNT(*) FROM raw.loans").fetchone()[0]
    print(f"Loaded {row_count:,} rows into raw.loans")
    print(con.execute("DESCRIBE raw.loans").df().to_string())
    con.close()


if __name__ == "__main__":
    load_raw_loans()
