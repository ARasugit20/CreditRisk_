# credit_risk_dbt

Local [dbt](https://www.getdbt.com/) project on DuckDB for LendingClub PD features.

```bash
# From repo root (after load_to_duckdb.py)
cd credit_risk_dbt
dbt run --profiles-dir .
dbt test --profiles-dir .
```

Database file: `../data/credit_risk.duckdb` (gitignored).

See [docs/WAREHOUSE.md](../docs/WAREHOUSE.md) for the full runbook.
