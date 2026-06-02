-- mart_features: ML-ready feature table
-- THIS IS WHAT THE MODEL READS — not raw CSVs
-- All features are origination-available (no post-origination leakage)

{{ config(materialized='table') }}

SELECT
    f.loan_id,
    f.issue_d,
    b.grade,
    b.sub_grade,
    b.income_tier,
    b.home_ownership,
    f.loan_amnt,
    f.int_rate,
    f.dti,
    f.fico_mid,
    f.util_tier,
    f.delinq_2yrs,
    f.open_acc,
    f.pub_rec,
    f.revol_bal,
    f.total_acc,
    f.purpose,
    f.verification_status,
    f.loan_status
FROM {{ ref('fct_loan_performance') }} AS f
LEFT JOIN {{ ref('dim_borrower') }} AS b USING (loan_id)
