#!/usr/bin/env bash
# Reproducible end-to-end PD pipeline (run from project root).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MPLBACKEND=Agg
PYTHON="${PYTHON:-python3}"

echo "==> Installing dependencies"
"$PYTHON" -m pip install -q -r requirements.txt

echo "==> NLTK assets (nlp_analysis)"
"$PYTHON" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true

echo "==> Sample + preprocess + train"
"$PYTHON" src/sample_data.py --mode random --rows 100000
"$PYTHON" src/preprocessing.py
"$PYTHON" src/train.py --include-xgboost

echo "==> Tuning + evaluation"
"$PYTHON" src/tune.py
"$PYTHON" src/evaluate.py
"$PYTHON" src/calibration.py
"$PYTHON" src/reliability_diagrams.py
"$PYTHON" src/portfolio_simulation.py
"$PYTHON" src/shap_analysis.py

echo "==> Segment + NLP + ablation"
"$PYTHON" src/segment_analysis.py
"$PYTHON" src/nlp_analysis.py
"$PYTHON" src/ablation.py --train-sample-size 0
"$PYTHON" src/compare_experiments.py

echo "==> Done. Review outputs/ and results/"
