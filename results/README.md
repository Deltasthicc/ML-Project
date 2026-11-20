# results/

Exported automatically by `python scripts/run_all.py` (or `python scripts/make_report.py`) from the run folders in `runs/`.
Nothing here is edited by hand.

| Path | Content |
|---|---|
| `summary.md` | headline tables (3 evaluation modes) + pre-registered comparisons |
| `protocol/` | `decisions.json` (budget, compact backbone, freeze time + file hashes), `drive_split.json`, `PROTOCOL.md` |
| `data/` | dataset manifest (source archives, SHA-256) and statistics |
| `tables/main_{raw,clean,tuned}.{md,csv}` | one row per model: mean±sd over seeds. raw = threshold 0.5 (headline); clean = + fragment removal; tuned = threshold + fragment size from DRIVE-val |
| `tables/per_run_*.csv`, `per_dataset.csv`, `per_image_all.csv.gz` | the same numbers at finer granularity |
| `tables/comparisons.{md,csv}` | paired seed comparisons with 95% t intervals, Holm-adjusted p, image-level bootstrap |
| `tables/factorial.md`, `efficiency.*`, `uncertainty.*`, `selection.csv`, `training.csv` | 2x2x2 design, cost, RQ4, per-run DRIVE-val choices, training records |
| `figures/` | stages, cleanup/threshold, accuracy vs CPU time, risk-coverage, training curves, qualitative |
| `runs/<exp>/seed<k>/` | config, training log, DRIVE-val selections and per-image metrics of every run (checkpoints stay in `runs/`) |

Models are named `<experiment>[<variant>]`: `det` = deterministic, `mc` = MC dropout (20 passes), `ens` = 5-seed ensemble.
"Unseen" = mean over STARE, CHASE_DB1 and HRF (equal weight); FIVES is reported separately; gap = DRIVE Dice - unseen Dice.
