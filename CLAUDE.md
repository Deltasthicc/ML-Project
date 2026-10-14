# CLAUDE.md — TG-LiteNet (CSD361 final project)

Retinal vessel segmentation study: compact, topology-aware, source-only-trained model (DRIVE → STARE / CHASE_DB1 / HRF).
Read `README.md` (overview, commands) and `RUNBOOK.md` (how the full run is done) first.

## Rules for this repo
* **Do not add Claude/AI as co-author or add attribution trailers to commits or PR descriptions** (explicit instruction from the team).
* **Never fabricate or hand-edit results.** Every number in `report/` comes from `results/runs` via `scripts/aggregate_results.py` →
  `scripts/make_report_assets.py`. The Discussion/abstract-ending `\todo{}` blocks in `report/main.tex` must be written from real results.
* Protocol invariants (do not break): training uses DRIVE-train only; DRIVE-val is the only data used for checkpoint selection / tuning;
  STARE, CHASE_DB1, HRF, FIVES are test-only; the decision threshold is fixed at 0.5; all stages share optimiser/schedule/data pipeline.
* Datasets and checkpoints are not committed (`.gitignore`). Don't commit `data/`, `.venv/`, `*.pt`.
* Layout test builds with fake numbers must never be written into `report/generated` or `results/runs`.

## Commands
```bash
pip install -r requirements.txt && pip install -e .     # install torch separately (CUDA build)
pytest -q                                                # 24 tests
python scripts/run_experiments.py --groups primary --quick     # smoke test
python scripts/run_experiments.py --groups primary factorial topology width --native-eval
python scripts/aggregate_results.py && python scripts/make_report_assets.py && (cd report && pdflatex main && pdflatex main)
```

## Conventions
* All hyper-parameters in `configs/base.yaml`; experiments are entries in `tglite/experiments.py` (add a new one there, not a new YAML).
* Metrics are computed inside the FOV against NATIVE-resolution ground truth (model runs at the FOV-normalised scale, prob map resampled), threshold 0.5. Topology: 8-connectivity, β1 = β0 − Euler number.
* Python 3.12 + PyTorch; code must stay device-agnostic (CPU runs are used for tests).
