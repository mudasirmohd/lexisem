#!/usr/bin/env bash
# Code Ocean Reproducible Run entry point.
#
# Runs the CLASSICAL experiment (baseline vs +lexico-semantic, 5-fold),
# which is fast and CPU-friendly, and writes results into the results dir.
# The LSTM path (GPU, ~10+ min) is NOT run here by default — reproduce it
# separately with scripts/run_lstm.py (see REPRODUCE.md).
#
# Offline-safe: if pre-staged data exists (see scripts/prepare_offline_data.py)
# it is used and no network is needed. Otherwise it falls back to downloading,
# so the same script also works on a normal machine with internet.
set -euo pipefail

# Code Ocean mounts /data (read-only) and /results; fall back to repo-local
# dirs so the script also runs as ./run.sh outside a capsule.
DATA_DIR="${CO_DATA:-/data}";        [ -d "$DATA_DIR" ] || DATA_DIR="$(pwd)/data"
RESULTS_DIR="${CO_RESULTS:-/results}"; [ -d "$RESULTS_DIR" ] || RESULTS_DIR="$(pwd)/results"
mkdir -p "$RESULTS_DIR"

PRETRAINED="glove-wiki-gigaword-100"
VECTORS="$DATA_DIR/embeddings/${PRETRAINED}.bin"
NLTK_LOCAL="$DATA_DIR/nltk_data"

echo "DATA_DIR=$DATA_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"

# Use pre-staged NLTK data if present (prevents any offline network attempt).
if [ -d "$NLTK_LOCAL" ]; then
  export NLTK_DATA="$NLTK_LOCAL"
  echo "NLTK_DATA=$NLTK_DATA (pre-staged)"
else
  echo "NLTK data not pre-staged; will download if network is available."
  python scripts/download_nltk.py || true
fi

# Prefer the pre-staged local vectors; else let the script download the model.
if [ -f "$VECTORS" ]; then
  echo "Using local embeddings: $VECTORS"
  python scripts/run_experiment.py --vectors "$VECTORS" \
      --out "$RESULTS_DIR/results.json"
else
  echo "Local embeddings not found; downloading '$PRETRAINED' at runtime."
  python scripts/run_experiment.py --pretrained "$PRETRAINED" \
      --out "$RESULTS_DIR/results.json"
fi

echo "Done. Results in $RESULTS_DIR/results.json"
