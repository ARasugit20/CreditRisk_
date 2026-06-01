-- stg_loans: clean and type-cast raw loans
-- Grain: one row per loan application
-- As-of date: issue_d (loan origination date — NO post-origination fields)

{{ config(materialized='view') }}

SELECT
    CAST(id AS VARCHAR) AS loan_id,
    CAST(loan_amnt AS DOUBLE) AS loan_amnt,
    CAST(funded_amnt AS DOUBLE) AS funded_amnt,
    CAST(int_rate AS DOUBLE) AS int_rate,
    CAST(installment AS DOUBLE) AS installment,
    grade,
    sub_grade,
    emp_length,
    home_ownership,
    CAST(annual_inc AS DOUBLE) AS annual_inc,
    verification_status,
    COALESCE(
        try_strptime(CAST(issue_d AS VARCHAR), '%b-%Y')::DATE,
        try_cast(issue_d AS DATE)
    ) AS issue_date,
    purpose,
    addr_state,
    CAST(dti AS DOUBLE) AS dti,
    CAST(delinq_2yrs AS INTEGER) AS delinq_2yrs,
    CAST(fico_range_low AS INTEGER) AS fico_range_low,
    CAST(fico_range_high AS INTEGER) AS fico_range_high,
    CAST(open_acc AS INTEGER) AS open_acc,
    CAST(pub_rec AS INTEGER) AS pub_rec,
    CAST(revol_bal AS DOUBLE) AS revol_bal,
    CAST(
        NULLIF(regexp_replace(CAST(revol_util AS VARCHAR), '%', ''), '') AS DOUBLE
    ) AS revol_util,
    CAST(total_acc AS INTEGER) AS total_acc,
    CASE
        WHEN loan_status IN (
            'Charged Off',
            'Default',
            'Does not meet the credit policy. Status:Charged Off'
        ) THEN 1
        ELSE 0
    END AS is_default
FROM {{ source('raw', 'loans') }}
WHERE loan_status IS NOT NULL
  AND issue_d IS NOT NULL
  AND loan_status IN ('Fully Paid', 'Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off')
