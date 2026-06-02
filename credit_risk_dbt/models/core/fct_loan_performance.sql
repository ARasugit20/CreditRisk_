-- fct_loan_performance: core fact table
-- Grain: one row per loan

{{ config(materialized='table') }}

SELECT
    loan_id,
    grade,
    loan_amnt,
    funded_amnt,
    int_rate,
    installment,
    dti,
    fico_range_low,
    fico_range_high,
    delinq_2yrs,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    purpose,
    verification_status,
    issue_d,
    loan_status,
    (fico_range_low + fico_range_high) / 2.0 AS fico_mid,
    CASE
        WHEN revol_util IS NULL THEN 'unknown'
        WHEN revol_util < 30 THEN 'low'
        WHEN revol_util < 60 THEN 'mid'
        WHEN revol_util < 90 THEN 'high'
        ELSE 'very_high'
    END AS util_tier
FROM {{ ref('stg_loans') }}
