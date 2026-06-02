#!/usr/bin/env bash
# Build DuckDB raw layer, run dbt, export mart_features.parquet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

echo "==> Load raw.loans into DuckDB"
"$PYTHON" scripts/load_to_duckdb.py

echo "==> dbt run + test"
(
  cd credit_risk_dbt
  dbt run --profiles-dir .
  dbt test --profiles-dir .
)

echo "==> Export mart_features.parquet"
"$PYTHON" scripts/export_features.py

echo "==> Done. ML input: data/mart_features.parquet"
