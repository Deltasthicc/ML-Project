#!/usr/bin/env bash
# Full pipeline: every experiment -> tables -> report assets -> PDF.  Resumable (finished runs are skipped).
# Usage:  bash scripts/run_all.sh [extra overrides, e.g. train.num_workers=12 train.iters=5000]
# Add --datasets ... fives to the run commands below if FIVES has been prepared.
set -euo pipefail
cd "$(dirname "$0")/.."
OVERRIDES=("train.num_workers=${NUM_WORKERS:-12}" "$@")
mkdir -p results

python scripts/run_experiments.py --groups primary --native-eval --set "${OVERRIDES[@]}" 2>&1 | tee -a results/log_primary.txt
python scripts/run_experiments.py --groups factorial topology width --set "${OVERRIDES[@]}" 2>&1 | tee -a results/log_extra.txt

python scripts/aggregate_results.py
python scripts/auto_findings.py
python scripts/make_data_overview.py
python scripts/make_data_overview.py --strip --out results/figures/data_overview_strip.png
python scripts/make_report_assets.py
(cd report && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex) || echo "LaTeX build failed/skipped - compile report/ on Overleaf"
echo "done. Read results/findings.md, then write the \\todo{} blocks in report/main.tex"
