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
  runs/         v2: config, training log, DRIVE-val choices, per-image metrics of every run; _suite/ = full suite log
  v21/          v2.1: summary, tables, figures, assets/ (example images + probability maps), runs/ (per-run records,
                v2_reeval/), logs/ (v2 and v2.1 suite logs)
reports/      write-ups for the group (GROUP_UPDATE_v2.1.md)
data/         downloaded archives + processed data (not committed, see data/README.md)
runs/         checkpoints, probability maps, TensorBoard, suite state (not committed: ~14 GB)
requirements-lock.txt   exact package versions of the environment that produced the results
```

**What is not in git and why.** Only the trained weights (`*.pt`, ~14 GB with the 31 M U-Nets), the saved
probability maps (~2.3 GB) and the datasets. Everything needed to rebuild every table, figure and number in the
reports from the repository alone is committed: per-image metrics of every model on every test image, every run's
config / training curve / DRIVE-val choices / cost, the decision records, and the suite logs. Re-running the models
themselves needs the data and either retraining (`run_all.py`, `run_v21.py`) or the weights (available on request).
`results/v21/assets/` contains one example image per dataset with its label, only for illustration.

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

## v2.1 follow-up: ~70 K model and four topology novelties

Branches: `v2` holds the v2 study only (code, protocol, `results/`); `v2.1-tglite70k` (this branch) adds v2.1 on top
(`results/v21/`, `reports/GROUP_UPDATE_v2.1.md`). The v2 files are identical on both branches.

v2.1 (`protocol/PROTOCOL_v2.1.md`, `configs/v21.yaml`) keeps the frozen v2 study untouched and adds:

* a **~70 K-parameter headline model** (capped-channel U-Net or two chained tiny U-Nets, chosen on DRIVE-val), with
  0.21 M / 0.49 M reference models and a pre-registered size rule;
* **novelty 1** - a label-free *connectivity reliability score* (fragment density of the prediction) that flags bad
  images, compared with entropy, MC dropout and deep ensembles;
* **novelty 2** - *centreline-calibre-guided test-time scale selection* (no labels, no weight change; targets thin-vessel
  datasets such as HRF);
* **novelty 3** - *topology-aware distillation* from the v2 U-Net ensemble (the student matches the teacher's soft
  centrelines);
* **novelty 4** - *skeleton-level consistency* between the clean and the perturbed view;
* stronger, still simple augmentation (RandConv, Fourier amplitude jitter), network-resolution metrics for a fairer
  comparison with published work, and an INT8 deployment check.

```
.venv\python.exe scripts\run_v21.py --smoke     # quick end-to-end check (runs_smoke/v21)
.venv\python.exe scripts\run_v21.py             # full v2.1 study in a new window (resumable) + monitor window
.venv\python.exe scripts\monitor.py --v21       # live progress
```

Results: `results/v21/` (`summary.md`, `tables/main.md`, `tables/sota.md`, `tables/novelty1_reliability.md`,
`tables/novelty2_scale.md`, `tables/ablations.md`, `tables/int8.md`, `figures/`, `runs/`, `logs/`, `assets/`).
Website / report assets (needs the weights in `runs/`): `.venv\python.exe scripts\export_assets.py`.
Group write-up: [reports/GROUP_UPDATE_v2.1.md](reports/GROUP_UPDATE_v2.1.md).

## Licence and data

Code: MIT ([LICENSE](LICENSE)). The datasets keep their own licences and are not redistributed; please cite DRIVE
(Staal et al. 2004), STARE (Hoover et al. 2000), CHASE_DB1 (Fraz et al. 2012), HRF (Budai et al. 2013) and FIVES
(Jin et al. 2022).
