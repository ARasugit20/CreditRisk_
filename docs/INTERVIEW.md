# Interview Notes — Credit Risk PD Modeling

Short reference for explaining this repository in a credit-risk or ML interview.

## PD vs LGD (and EAD)

| Concept | Meaning | In this repo |
|---------|---------|--------------|
| **PD** (Probability of Default) | Likelihood a borrower defaults within the horizon | **Primary target.** Binary `loan_status`: Charged Off = 1, Fully Paid = 0. Models output calibrated PD. |
| **LGD** (Loss Given Default) | Fraction of exposure lost if default occurs | Not modeled end-to-end. We assume a constant **LGD = 0.60** in `src/utils.py` for portfolio simulation (`expected_loss = PD × LGD × EAD`). |
| **EAD** (Exposure at Default) | Outstanding balance at default | Proxy: `funded_amnt`, preserved in processed splits for portfolio math only—not a model feature. |

**Talking point:** Production credit stacks often have separate PD, LGD, and EAD models. This project demonstrates PD rigor (calibration, segments, temporal validation) and uses simplified LGD/EAD for expected-loss illustration.

## Why PR-AUC for imbalanced data

- Default rates in consumer lending are typically **5–20%**; accuracy is misleading (a naive “always paid” classifier can look strong).
- **ROC AUC** can stay high when the model ranks negatives well even if positives are poorly separated.
- **PR-AUC (Average Precision)** focuses on the positive (default) class and is more informative when the positive class is rare.
- We tune with `GridSearchCV(scoring="average_precision")` in `src/tune.py` and report PR-AUC with bootstrap CIs in `src/evaluate.py`.

## Platt scaling vs isotonic calibration

| Method | Mechanism | When to use |
|--------|-----------|-------------|
| **Platt (sigmoid)** | Logistic regression on raw scores | Smaller calibration sets; less risk of overfitting |
| **Isotonic** | Monotonic piecewise-constant mapping | Larger calibration sets; can fix complex miscalibration |

This repo uses **Platt scaling** via `CalibratedClassifierCV(..., method="sigmoid")` in `src/calibration.py`. Reliability diagrams in `outputs/reliability_diagrams.png` (pinned copy: `docs/figures/reliability_diagrams.png`) show predicted vs observed default rates by decile.

**Talking point:** Calibration matters for pricing and capital—ranking (AUC) is not enough if predicted PDs are systematically too low or too high.

## Temporal split vs random split

| Split | How | Risk addressed |
|-------|-----|----------------|
| **Random (stratified)** | 70/15/15 on shuffled loans | Maximizes i.i.d. holdout for model comparison |
| **Temporal** | Train ≤ 2014, val 2015, test > 2015 by `issue_d` | **Data leakage through time**—models must generalize to future vintages |

Random splits can **inflate performance** if future macro conditions or policy changes differ from training. The temporal experiment (`--split-strategy temporal`) in `src/preprocessing.py` stress-tests robustness; compare `results/test_metrics.csv` vs `results/temporal_test_metrics.csv`.

**Train-only fit:** Preprocessing (`ColumnTransformer`, imputers, encoders) is fit on **training rows only** (`fit_preprocessing_pipeline` in `src/pipeline.py`).

## Regulatory fairness (brief)

- Credit models in the U.S. may fall under **ECOA / Fair Lending** scrutiny: disparate impact on protected classes even when those attributes are excluded from features.
- This pipeline drops direct identifiers and post-origination leakage fields, but **proxy variables** (zip, employment, purpose) can still correlate with protected attributes.
- **Segment analysis** (`src/segment_analysis.py`) is a starting point for performance-by-grade and purpose—not a substitute for formal fair-lending review (adverse-action reasons, disparate impact testing, model documentation for SR 11-7).

## Key artifacts to cite in an interview

- Metrics: `results/test_metrics.csv`
- Calibration: `outputs/calibration_brier_comparison.csv`, `docs/figures/reliability_diagrams.png`
- Explainability: `outputs/shap_feature_importance.csv`, `docs/figures/shap_summary.png`
- Leakage guard: `tests/test_no_leakage.py`

## Data labeling

| File | Type |
|------|------|
| `data/lending_club_sample.csv` | **Real** LendingClub loans (100k-row reservoir sample); safe to ship as a demo subset |
| `data/accepted_2007_to_2018Q4.csv` | **Real** full Kaggle dump — **gitignored**; download locally |
| `tests/conftest.py` `minimal_lending_club_frame` | **Synthetic** rows for fast CI only |

Never commit Kaggle credentials or the full raw CSV.
