#!/usr/bin/env bash
# One-command driver for a Kaggle "Save & Run All (Commit)" run (also fine on Colab / any Linux box).
#
#   bash scripts/kaggle_run.sh "<groups>" "<seeds>" [budget_hours]
#   bash scripts/kaggle_run.sh "primary" "0 1 2" 9.5          # core stages A-E, 3 seeds, stop starting new runs after 9.5 h
#   bash scripts/kaggle_run.sh "factorial topology" "0 1 2" 9.5
#
# What it does (in order):
#   1. resume   : copies results of earlier notebook runs that you attached as *inputs* (/kaggle/input/*/runs) into the output folder;
#                 finished runs are recognised and skipped (the runner checks that their saved config matches);
#   2. data     : downloads + prepares the datasets (a few minutes; nothing is kept between sessions);
#   3. train    : run_experiments.py, seed-major order (every experiment gets seed 0 first), with a wall-clock budget so that the
#                 session ends cleanly before Kaggle's 12 h limit and everything is saved;
#   4. tables   : aggregate_results.py -> /kaggle/working/tables ;   5. bundle: /kaggle/working/results_bundle.zip (no checkpoints).
#
# Environment overrides: OUT (runs dir, default /kaggle/working/runs), TAB, SKIP_DATA=1 (data already prepared in this session), NATIVE=1 (also evaluate without FOV rescaling; slow for the
# U-Net on 8-MP HRF images, so off by default), EXTRA="..." (extra run_experiments.py arguments, e.g. --datasets drive stare chase_db1 hrf).
set -u
cd "$(dirname "$0")/.."
GRP="${1:-primary}"
SEEDS="${2:-0 1 2}"
BUDGET="${3:-9.5}"
OUT="${OUT:-/kaggle/working/runs}"
TAB="${TAB:-/kaggle/working/tables}"
BUNDLE_DIR="$(dirname "$OUT")"
NW="${NW:-4}"
NATIVE_FLAG=""
if [ "${NATIVE:-0}" = "1" ]; then NATIVE_FLAG="--native-eval"; fi
mkdir -p "$OUT"

echo "=== 1/5 resume"
for d in /kaggle/input/*/runs; do
  if [ -d "$d" ]; then
    echo "copying earlier results from $d"
    cp -rn "$d"/. "$OUT"/
  fi
done
echo "runs already present: $(find "$OUT" -name per_image.csv 2>/dev/null | wc -l)"

echo "=== 2/5 data"
if [ "${SKIP_DATA:-0}" = "1" ]; then
  echo "SKIP_DATA=1: assuming data/processed is already prepared"
elif ! python scripts/download_data.py || ! python scripts/prepare_data.py; then
  echo "DATA STEP FAILED - see the messages above"; exit 1
fi

echo "=== 3/5 training (groups: $GRP | seeds: $SEEDS | budget: ${BUDGET} h)"
# shellcheck disable=SC2086   # groups / seeds / extra args are intentionally word-split
python -u scripts/run_experiments.py --groups $GRP --seeds $SEEDS --out "$OUT" --time-budget-hours "$BUDGET" \
  $NATIVE_FLAG --set train.num_workers="$NW" eval.metric_workers="$NW" ${EXTRA:-}
STATUS=$?
echo "run_experiments exit status: $STATUS (0 = all done or budget reached, 1 = some run failed; re-run to retry)"

echo "=== 4/5 tables"
python scripts/aggregate_results.py --runs "$OUT" --out "$TAB" || echo "(aggregation failed - fine if no run has finished yet)"

echo "=== 5/5 bundle"
(cd "$BUNDLE_DIR" && zip -qr results_bundle.zip "$(basename "$OUT")" "$(basename "$TAB")" -x "*.pt") || echo "(zip failed)"
echo "finished runs: $(find "$OUT" -name per_image.csv 2>/dev/null | wc -l)  | bundle: $BUNDLE_DIR/results_bundle.zip"
echo "To continue later: attach this notebook's output as an input of the next notebook and run the same command again."
exit 0
