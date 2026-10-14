# TG-LiteNet — topology-preserving, domain-generalizable, lightweight retinal vessel segmentation

CSD361 *Introduction to Machine Learning* — final project (Shiv Nadar Institution of Eminence)
Abhiraj Agarwal (2510110002) · Shashwat Rajan (2510111019) · Tanish Shitanshu (2510110008)

A compact depthwise-separable encoder–decoder trained on **one** source dataset (DRIVE) and evaluated on **three
never-seen** datasets (STARE, CHASE_DB1, HRF; optionally FIVES) for **accuracy** (Dice/IoU/AUC), **topology** (clDice, cbDice,
Betti-0/1 error), **generalisation gap**, **deployment cost** (params, FLOPs, latency, memory) and **uncertainty**
(MC-dropout variance vs. error, risk–coverage). It is a *trade-off study*, not a new SOTA claim — see `report/main.tex`.

```
A  U-Net baseline (Dice+BCE)  →  B  + topology loss (soft-clDice + skeleton head)  →  C  + DG augmentation + consistency
                              →  D  + compact backbone (≈0.2 M params)           →  E  TG-LiteNet = D + MC-dropout uncertainty
```
plus a full 2×2×2 factorial design (backbone × topology × DG), topology-loss variants (cbDice, Skeleton Recall) and a width sweep.

## 1. Repository layout

| Path | What |
|---|---|
| `tglite/` | the library: `models/` (U-Net, LiteNet), `data/` (preprocessing, augmentation, datasets), `losses.py`, `metrics.py`, `uncertainty.py`, `efficiency.py`, `train.py`, `evaluate.py`, `analysis.py`, `experiments.py` (registry of every run) |
| `configs/base.yaml` | all defaults/hyper-parameters (one file; experiments are overrides in `tglite/experiments.py`) |
| `scripts/` | `download_data.py`, `prepare_data.py`, `make_data_overview.py`, `train.py`, `evaluate.py`, `run_experiments.py`, `tune_lambdas.py`, `aggregate_results.py`, `auto_findings.py`, `make_report_assets.py`, `run_all.sh/.ps1` |
| `splits/drive_split.json` | fixed DRIVE train(16)/val(4)/test(20) split |
| `tests/` | unit tests (losses, Betti numbers, metrics, augmentation alignment, MC dropout, config) — `pytest -q` |
| `results/` | `runs/<exp>/seed<k>/` (config, log, `per_image.csv`, `efficiency.json`), `tables/*.csv`, `figures/`, `findings.md` |
| `report/` | IEEE paper source (`main.tex`), auto-generated `generated/*.tex` and `figures/` |
| `RUNBOOK.md` | **step-by-step checklist for the full training run on the powerful machine** |
| `data/` | *not in git*: `raw/` downloads, `processed/`, `MANIFEST.json` (checksums) |

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
# 1) PyTorch first, matching your CUDA driver (https://pytorch.org/get-started/locally/), e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu126
# 2) everything else + the package itself
pip install -r requirements.txt && pip install -e .
pytest -q                                                   # 24 tests, ~25 s on CPU
```
Developed with Python 3.12 and PyTorch 2.14 (CPU build for development; the code is device-agnostic and uses CUDA +
fp16 autocast automatically when available).

## 3. Data (not redistributed — please cite the original papers)

```bash
python scripts/download_data.py            # DRIVE, STARE, CHASE_DB1, HRF  (~140 MB)   [--fives for FIVES, 1.8 GB, optional]
python scripts/prepare_data.py             # -> data/processed/<dataset>/{images,masks,fov}/ + stats.json + splits/drive_split.json
python scripts/make_data_overview.py       # contact sheet: check labels/FOV visually  (results/figures/data_overview.png)
```

| Dataset | Images | Role | Source used by `download_data.py` | Notes |
|---|---|---|---|---|
| DRIVE [Staal 2004] | 40 (20 train / 20 test) | **source**: train 16 / val 4 / test 20 | PaddleSeg mirror of the official data (the official site needs a free login) | FOV masks are *estimated*. To use the official files, unzip them to `data/raw/drive/{training,test}/{images,1st_manual,mask}` — `prepare_data.py` prefers them automatically |
| STARE [Hoover 2000] | 20 | unseen target | official (Clemson Univ.), labels = Adam Hoover | FOV estimated |
| CHASE_DB1 [Fraz 2012] | 28 | unseen target | Hugging Face mirror `MedOtter/CHASE_DB1` (Kingston's repository was unreachable when we built this) | 1st-observer labels; FOV estimated |
| HRF [Budai 2013] | 45 | unseen target | official (FAU Erlangen), CC BY 4.0 | official FOV masks |
| FIVES [Jin 2022] | 200 (test split, stored at 1024×1024) | optional unseen target | official figshare, CC BY 4.0 (1.8 GB `.rar`) | extract with `tar`/7-zip, see RUNBOOK §2 |

Mirrors are third-party copies. `prepare_data.py` asserts image counts, sizes and a plausible vessel fraction (4–20 %);
`data/MANIFEST.json` records SHA-256 checksums so the machines can verify they hold identical files.

**Protocol (important):** DRIVE-train is the only data ever used for fitting. DRIVE-val (4 images) is used for checkpoint
selection (best Dice) and, optionally, for `tune_lambdas.py`. STARE/CHASE_DB1/HRF/FIVES are used **only for final testing**.
Every image is rescaled so its FOV is 527 px wide (DRIVE's median) — a label-free, target-free normalisation; the
`--native-eval` option additionally reports results *without* it (`per_image_native.csv`). Metrics are always computed against the **native-resolution** labels (the probability map is resampled to them), because down-scaling HRF's 1–2 px labels fragments them and inflates Betti-0. Native HRF metrics cost ~10 s/image: set `eval.metric_workers=8` (process pool) on a multi-core machine.

## 4. Running experiments

```bash
# smoke test (1 minute, CPU ok; numbers are meaningless)
python scripts/run_experiments.py --groups primary --quick

# the real thing (resumable: finished runs are skipped). See RUNBOOK.md for timing and ordering.
python scripts/tune_lambdas.py                                   # optional: choose lambda_2/lambda_3 on DRIVE-val
python scripts/run_experiments.py --groups primary --native-eval             # A-E x 5 seeds
python scripts/run_experiments.py --groups factorial topology width          # remaining ablations (4+3+4 configs)
python scripts/run_experiments.py --groups primary --datasets drive stare chase_db1 hrf fives   # if FIVES is prepared

# single run / custom config
python scripts/train.py --exp E_tglitenet --seed 0 --evaluate --set train.iters=8000 train.num_workers=12
```
Experiments (see `tglite/experiments.py`): `A_unet`, `B_unet_topo`, `C_unet_topo_dg`, `D_lite_topo_dg`, `E_tglitenet` (primary);
`X_unet_dg`, `X_lite_plain`, `X_lite_topo`, `X_lite_dg` (factorial); `V_lite_cbdice`, `V_lite_srl`, `V_lite_cldice_srl` (topology variants);
`W_lite_w{8,24,32,48}` (width; w=16 is stage D). Defaults: 5 seeds for primary/factorial, 3 for the rest.

## 5. From results to the paper

```bash
python scripts/aggregate_results.py     # results/tables/*.csv  (mean/std/95 % CI over seeds, gaps, stage deltas, uncertainty, efficiency)
python scripts/auto_findings.py         # results/findings.md   (facts with CIs, to guide the Discussion)
python scripts/make_data_overview.py && python scripts/make_data_overview.py --strip --out results/figures/data_overview_strip.png
python scripts/make_report_assets.py    # report/generated/*.tex (tables, number macros) + report/figures/*.pdf
cd report && pdflatex main && pdflatex main        # must be <= 8 pages; set \draftfalse once every \todo{} is resolved
```
Every table/number in the paper is produced by these scripts from `results/runs`, so the report matches the submitted code.
The Discussion text (red `\todo{}` blocks in `report/main.tex`) is deliberately **not** pre-written: it must follow the numbers.

## 6. Output files per run

`results/runs/<exp>/seed<k>/`: `config.yaml` (exact config), `train_log.csv` (loss terms, val Dice/clDice), `best.pt`
(checkpoint; git-ignored), `per_image.csv` (all metrics per image, per mode `det`/`mc`), `efficiency.json` (params, FLOPs, latency,
hardware), `preds/` (probability/uncertainty PNGs for seed 0, used by the qualitative figure).

## 7. Reproducibility notes

* Seeds control initialisation, data order and augmentation (`train.py: set_seed`, per-worker RNG). GPU kernels are not bit-wise
  deterministic, so re-running gives numbers that agree statistically (that is why we report mean±std/CI over seeds).
* All hyper-parameters live in `configs/base.yaml` (+ optional `configs/tuned.yaml`) and are copied into every run directory.
* Latency/FLOPs: measured at 512×512, batch 1; FLOPs = 2×MACs from `torch.utils.flop_counter`. Quote the hardware (stored in `efficiency.json`).

## 8. Limitations (also in the paper)

Small training set (16 images); FOV masks for four datasets are estimated; first-observer labels only; DRIVE and CHASE_DB1 come from
mirrors; soft-clDice, Skeleton Recall and cbDice are our own re-implementations from the papers (not the authors' code); source-only results are not comparable to
multi-source or test-time-adaptation papers.

## 9. References / licences

Datasets: see the table above (cite Staal 2004, Hoover 2000, Fraz 2012, Budai 2013, Jin 2022). The code in this repository was
written for this course project; losses follow the descriptions in clDice (Shit et al., CVPR 2021) and Skeleton Recall
(Kirchhoff et al., ECCV 2024) and are **not** copied from the authors' repositories.
