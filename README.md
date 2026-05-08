# Credit Risk PD Pipeline

Phase 4 modules for calibrated probability-of-default modeling on LendingClub loans.

## Expected Files

Place the Kaggle LendingClub accepted-loans CSV under `data/` and saved model artifacts under `models/`.

Default data names searched:

- `data/processed_train.csv` and `data/processed_test.csv`
- raw sample input: `data/accepted_2007_to_2018Q4.csv`
- generated sample: `data/lending_club_sample.csv` (default input to `preprocessing.py`, matching `sample_data.py` output)

**Data directory layout:** Keep CSVs directly under `credit_risk_pd/data/`. Do not nest another `credit_risk_pd/` (or duplicate project tree) inside `data/`; scripts resolve paths from the project root only.

Default model artifact names searched:

- Logistic regression: `logistic_regression.pkl`, `logreg.pkl`, `lr_model.pkl`
- Random forest: `random_forest.pkl`, `rf.pkl`, `rf_model.pkl`
- XGBoost: `xgboost.pkl`, `xgb.pkl`, `xgb_model.pkl`, `xgboost.json`
- MLP: `mlp.pkl`, `mlp_model.pkl`, `neural_network.pkl`

The real-data preprocessing script filters `loan_status` to Fully Paid / Charged Off,
drops leakage features from model inputs, imputes missing values, encodes categorical
features, scales numeric features, and creates stratified 70/15/15 splits.

**Canonical train/validation/test protocol:** The default preprocessing split is **random stratified 70/15/15** (same distribution across partitions). A **temporal** split (train/validation/test by `issue_d` cutoffs) is an optional robustness experiment; use `--split-strategy temporal` and matching `--output-prefix` / train paths so metrics stay comparable to `compare_experiments.py`.

## Run Order

From the project root:

```bash
pip install -r requirements.txt
python src/sample_data.py --mode random --rows 100000
python src/preprocessing.py
python src/train.py --include-xgboost
MPLBACKEND=Agg python src/evaluate.py
MPLBACKEND=Agg python src/calibration.py
MPLBACKEND=Agg python src/reliability_diagrams.py
MPLBACKEND=Agg python src/portfolio_simulation.py
MPLBACKEND=Agg python src/shap_analysis.py
MPLBACKEND=Agg python src/ablation.py --train-sample-size 0
python src/compare_experiments.py
```

Temporal robustness experiment (recommended):

```bash
python src/preprocessing.py --split-strategy temporal --train-end-date 2015-12-31 --validation-end-date 2016-12-31 --output-prefix temporal_
python src/train.py --include-xgboost --train-path data/temporal_processed_train.csv --validation-path data/temporal_processed_validation.csv --model-suffix _temporal --results-prefix temporal_
MPLBACKEND=Agg python src/evaluate.py --test-path data/temporal_processed_test.csv --model-suffix _temporal --results-prefix temporal_ --bootstrap-samples 200
python src/compare_experiments.py
```

If your files use different names:

```bash
python src/calibration.py --train-path data/processed_train.csv --test-path data/processed_test.csv
python src/calibration.py --artifact xgboost=models/my_xgb.pkl --artifact random_forest=models/my_rf.pkl
```

## Outputs

Generated figures and tables are saved under `outputs/`, including:

- `reliability_diagrams.png`
- `portfolio_simulation.png`
- `shap_summary.png`
- `shap_importance.png`
- `shap_dependence_*.png`
- `results/metrics.json`
- `results/test_metrics.csv`
- ROC, precision-recall, and confusion-matrix plots in `results/`
- CSV summaries for calibration, portfolio simulation, SHAP importance, and ablation
