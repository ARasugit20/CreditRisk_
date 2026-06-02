-- mart_default_metrics: business KPIs for dashboard/README
-- default rate by grade and vintage

{{ config(materialized='table') }}

SELECT
    grade,
    DATE_TRUNC('year', issue_d) AS vintage_year,
    COUNT(*) AS loan_count,
    SUM(loan_status) AS defaults,
    ROUND(AVG(loan_status) * 100, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_int_rate,
    ROUND(AVG(fico_mid), 0) AS avg_fico,
    ROUND(AVG(dti), 2) AS avg_dti
FROM {{ ref('fct_loan_performance') }}
GROUP BY 1, 2
ORDER BY 1, 2
