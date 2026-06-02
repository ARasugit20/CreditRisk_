# dbt + DuckDB Warehouse Runbook

Features for modeling are built in SQL with explicit origination dates, then exported for sklearn.

## Quick run

```bash
python3 scripts/load_to_duckdb.py
cd credit_risk_dbt && dbt run --profiles-dir . && dbt test --profiles-dir . && cd ..
python3 scripts/export_features.py
```

Or from repo root:

```bash
bash scripts/run_warehouse.sh
```

## Models

| Layer | Model | Purpose |
|-------|--------|---------|
| staging | `stg_loans` | Clean types, parse `issue_d`, binary `is_default` |
| core | `dim_borrower` | Borrower segments (income tier, grade) |
| core | `fct_loan_performance` | Loan-level facts + derived FICO/util tiers |
| marts | `mart_features` | ML export grain (origination-only fields) |
| marts | `mart_default_metrics` | Default rate by grade × vintage |

## BigQuery migration

Update `credit_risk_dbt/profiles.yml` to a `bigquery` target; SQL models stay the same.
