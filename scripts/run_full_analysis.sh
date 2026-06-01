#!/usr/bin/env bash
# Reproducible end-to-end PD pipeline (run from project root).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MPLBACKEND=Agg
PYTHON="${PYTHON:-python3}"
DATA_DIR="${ROOT}/data"

if [[ ! -d "$DATA_DIR" ]]; then
  cat <<'EOF' >&2
ERROR: data/ directory is missing.

Create it and add one of:
  • data/lending_club_sample.csv  — bundled 100k-row sample (real LendingClub, not in git as raw Kaggle dump)
  • data/accepted_2007_to_2018Q4.csv — full Kaggle accepted-loans file (gitignored; download locally)

Quick start without the full Kaggle file:
  mkdir -p data
  python src/sample_data.py --mode random --rows 100000   # requires accepted CSV above
  bash scripts/run_full_analysis.sh

Or generate synthetic test data only (unit tests):
  pytest tests/ -v
EOF
  exit 1
fi

has_input=false
for candidate in \
  "$DATA_DIR/lending_club_sample.csv" \
  "$DATA_DIR/accepted_2007_to_2018Q4.csv" \
  "$DATA_DIR/processed_train.csv"
do
  if [[ -f "$candidate" ]]; then
    has_input=true
    break
  fi
done

if [[ "$has_input" == false ]]; then
  cat <<'EOF' >&2
ERROR: data/ exists but no usable input was found.

Place one of these under data/:
  • lending_club_sample.csv (run: python src/sample_data.py --mode random --rows 100000)
  • accepted_2007_to_2018Q4.csv (Kaggle LendingClub accepted loans — keep local, never commit)
  • processed_train.csv (already preprocessed splits)

The full Kaggle CSV is gitignored. Do not commit raw LendingClub dumps to the repository.
EOF
  exit 1
fi

echo "==> Installing dependencies"
"$PYTHON" -m pip install -q -r requirements.txt

echo "==> NLTK assets (nlp_analysis)"
"$PYTHON" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true

echo "==> Sample + preprocess + train"
if [[ ! -f "$DATA_DIR/lending_club_sample.csv" && ! -f "$DATA_DIR/processed_train.csv" ]]; then
  "$PYTHON" src/sample_data.py --mode random --rows 100000
fi
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
