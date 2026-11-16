# TG-LiteNet v2 - topology-aware, compact, source-only retinal vessel segmentation

CSD361 final project (Abhiraj Agarwal, Shashwat Rajan, Tanish Shitanshu, Shiv Nadar Institution of Eminence).

A controlled study, not a new method: a model trained on **DRIVE only** is tested, without any adaptation, on
**STARE, CHASE_DB1, HRF** (and FIVES). A staged ablation asks what a topology loss (RQ1), robustness augmentation (RQ2), a
compact backbone (RQ3) and uncertainty estimates (RQ4) each contribute to overlap, connectivity, the generalisation gap,
cost and reliability. The rules of the study are fixed in [protocol/PROTOCOL.md](protocol/PROTOCOL.md) before any
target dataset is evaluated. Version 1 of the study is archived in the git tag `v1-final`.

| Stage | Model | Adds |
|---|---|---|
| A | U-Net (w64, 31 M params) | Dice + BCE baseline |
| B | A + topology | soft-clDice + skeleton head |
| C | B + DG | style/structure-augmented second view + consistency loss |
| D | C on the compact backbone | backbone chosen on DRIVE-val by accuracy per CPU-ms (<= 2 M params) |
| E | D + dropout = **TG-LiteNet** | MC-dropout uncertainty |

Plus: the full 2x2x2 factorial, topology-loss variants (cbDice, Skeleton Recall), topology-weight warm-up, ablation
controls, a width sweep, and 5-seed deep ensembles as an uncertainty baseline.

## Repository layout

```
configs/      base.yaml (all hyper-parameters) and experiments.yaml (every run, grouped)
protocol/     PROTOCOL.md (pre-registered rules), drive_split.json, decisions.json (budget, backbone, freeze record)
src/tglite/   data/ (prepare, preprocess, augment, dataset), models, losses, train, inference, selection,
              evaluate, metrics, efficiency, report, suite (orchestration + progress tracking), paths, config
scripts/      run_all.py (whole study), monitor.py (live tracker), prepare_data.py, train.py (single run), make_report.py
tests/        unit tests + resume test
results/      exported, committed final results (tables, figures, per-run records) - generated, never hand-edited
data/         downloaded archives + processed data (not committed, see data/README.md)
runs/         checkpoints, logs, probability maps, suite state (not committed)
```

## Setup (Windows, NVIDIA GPU)

```powershell
conda create -p .venv python=3.12 pip -c conda-forge      # or any Python 3.10+ venv
.venv\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\python.exe -m pip install -r requirements.txt
.venv\python.exe -m pip install -e .
.venv\python.exe -m pytest                                # unit tests
```

## Data

Download the archives listed in [data/README.md](data/README.md) into `data/` (unextracted), then

```powershell
.venv\python.exe scripts\prepare_data.py --overview       # checksums, counts, statistics, results/data/overview.png
```

## Running the study

```powershell
.venv\python.exe scripts\run_all.py --smoke               # 10-minute end-to-end check in runs_smoke/
.venv\python.exe scripts\run_all.py                       # the whole study, in a new console window with progress bars
.venv\python.exe scripts\monitor.py                       # live tracker (any terminal); or open runs\_suite\STATUS.md
```

Everything is resumable: after an interruption or a power cut, run `scripts\run_all.py` again - finished jobs are
skipped, training continues from its last checkpoint (at most 500 iterations are lost), evaluation continues at the next
dataset. The suite keeps Windows awake while it runs and refuses to start twice on the same folder.

Order: data -> pilot (training budget, DRIVE only) -> backbone selection (DRIVE only) -> **freeze** -> primary ->
factorial -> topology -> warm-up -> controls -> width. Tables and figures in `results/` are refreshed after every
group, so the first comparison of stages A-E is available as soon as the primary group finishes.
TensorBoard: `.venv\python.exe -m tensorboard.main --logdir runs`.

## Results

See [results/README.md](results/README.md) and [results/summary.md](results/summary.md) (generated).

## Branches

This branch (`v2`) holds the v2 study only. The v2.1 follow-up (~70 K model, topology novelties, `results/v21/`)
is on branch `v2.1-tglite70k`. Shared modules (`models.py`, `losses.py`, `train.py`, `augment.py`, `monitor.py`) contain
the v2.1 options as well; they are off by default and do not change any v2 run.

## Licence and data

Code: MIT ([LICENSE](LICENSE)). The datasets keep their own licences and are not redistributed; please cite DRIVE
(Staal et al. 2004), STARE (Hoover et al. 2000), CHASE_DB1 (Fraz et al. 2012), HRF (Budai et al. 2013) and FIVES
(Jin et al. 2022).
