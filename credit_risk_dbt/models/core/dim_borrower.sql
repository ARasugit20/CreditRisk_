-- dim_borrower: one row per borrower segment
-- Used for segment diagnostics

{{ config(materialized='table') }}

SELECT
    loan_id,
    grade,
    sub_grade,
    home_ownership,
    emp_length,
    addr_state,
    annual_inc,
    CASE
        WHEN annual_inc < 40000 THEN 'low'
        WHEN annual_inc < 80000 THEN 'mid'
        WHEN annual_inc < 150000 THEN 'high'
        ELSE 'very_high'
    END AS income_tier,
    issue_date
FROM {{ ref('stg_loans') }}
