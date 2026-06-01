# Model Card — LendingClub Probability of Default (PD) Model

## Problem

Binary classification: predict whether a borrower will default on a LendingClub loan.
Used to support loan approval decisions and risk tier assignment (portfolio demonstration only).

## Data

- **Source:** LendingClub public dataset (Kaggle)
- **Grain:** one row per loan application
- **Raw sample in repo:** `data/lending_club_sample.csv` (~100,000 applications)
- **Training period (temporal split):** loans with `issue_d` on or before **2015-12-31** (36,491 rows after preprocessing)
- **Validation period:** **2016-01-01** through **2016-12-31** (12,872 rows)
- **Holdout period:** loans with `issue_d` after **2016-12-31** (9,978 rows)
- **Size:** **59,341** rows after target filtering (Charged Off / Fully Paid only) from the 100k sample

## Model

- **Algorithm:** XGBoost (primary), with Random Forest and Logistic Regression baselines
- **Calibration:** Post-hoc **Platt scaling** (`CalibratedClassifierCV`, method=`sigmoid`) on validation data
- **Validation strategy:** Temporal train / validation / holdout split by `issue_d` (no future leakage in features)
- **Hyperparameter tuning:** `GridSearchCV` with **average precision** scoring (`src/tune.py`)

## Metrics (Temporal Holdout — XGBoost)

| Metric | Value |
|--------|-------|
| ROC AUC | 0.707 |
| PR AUC | 0.400 |
| ECE (calibration error) | 0.046 |
| Brier Score | 0.157 |

Source: `results/temporal_metrics.json`, `results/temporal_test_metrics.csv`.

**Stratified random split (reference):** ROC AUC 0.738, PR AUC 0.417, ECE 0.014 (`results/test_metrics.csv`).

## Leakage Controls

- All features are constructed using only information available at loan origination
- Post-origination fields (payments, recoveries, `grade`, `int_rate`, etc.) are excluded via `LEAKAGE_COLUMNS` in `src/utils.py`
- `issue_d` is used only as the temporal split boundary, not as a model feature
- Automated checks: `tests/test_no_leakage.py` (CI)

## Explainability

- SHAP values computed for test-set predictions (`src/shap_analysis.py`)
- **Top features (mean \|SHAP\|):** loan term (36 months), accounts opened in past 24 months (`acc_open_past_24mths`), debt-to-income (`dti`)
- Segment analysis by loan grade, term, and borrower state: `src/segment_analysis.py`

## Limitations

- Trained on pre-2020 sample data; does not capture pandemic-era credit behavior
- Demographic fairness has not been formally audited; use with caution in regulated contexts
- Model is calibrated on in-distribution validation data; performance may degrade on out-of-distribution applicants
- Shipped sample is ~100k rows, not the full Kaggle corpus

## Intended Use

Portfolio demonstration of production-grade credit risk modeling practices.
**Not for use in real lending decisions.**
