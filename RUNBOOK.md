# RUNBOOK — full training on the powerful machine

Everything here was developed and smoke-tested on a CPU-only laptop (24 unit tests, a tiny end-to-end run, and a
600-iteration sanity run that reached ≈0.79 DRIVE-validation Dice — see §9). **No full-length experiment has been run yet,
so there are no final results in the repository and none in the paper.** This runbook gets you from `git clone` to a finished PDF.

## 0. Checklist

- [ ] Python 3.10+ and an NVIDIA GPU with a CUDA build of PyTorch (`python -c "import torch; print(torch.cuda.is_available())"` → `True`)
- [ ] ≥ 8 CPU cores for the data-loader workers (augmentation runs on the CPU) — set `train.num_workers` accordingly
- [ ] ≈ 6 GB free disk for data + runs (FIVES adds ≈ 2 GB; checkpoints are small, the U-Net `best.pt` is ≈ 125 MB each)
- [ ] LaTeX (MiKTeX/TeX Live with `IEEEtran`, `adjustbox`, `algorithms`, `tikz`) *or* use Overleaf for the final PDF

## 1. Get the code and environment

```bash
git clone <your-repo-url> tglite && cd tglite
python -m venv .venv && source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the wheel that matches your driver
pip install -r requirements.txt && pip install -e .
pytest -q                                                       # all 28 tests must pass
```

## 2. Data

Either (a) re-download — `python scripts/download_data.py [--fives]` — or (b) copy the `data/raw` folder from the first machine and run
`python scripts/download_data.py --only hrf stare chase_db1 drive` (it finds the files, skips the download, and records/validates checksums).
Then:

```bash
python scripts/prepare_data.py                  # prints image counts, sizes, vessel fraction; every dataset must say "OK"
python scripts/make_data_overview.py            # open results/figures/data_overview.png and check labels/FOV outlines
```
Expected: DRIVE 40 @565×584 (12.9 % vessel), STARE 20 @700×605 (10.5 %), CHASE_DB1 28 @999×960 (10.3 %), HRF 45 @3504×2336 (9.1 %).

**FIVES (optional, tested).** `python scripts/download_data.py --fives` (1.8 GB; uses 8 parallel ranged connections because figshare throttles single connections) and then extract
only the test split with bsdtar (`tar.exe` ships with Windows 10+/macOS/Linux libarchive; 7-zip also works):
```bash
mkdir -p data/raw/fives/extracted
tar -xf data/raw/fives/FIVES.rar -C data/raw/fives/extracted --include "*/test/*"     # 400 files = 200 images + 200 labels
python scripts/prepare_data.py --only fives       # stores a half-resolution copy (1024x1024) to save disk; 200 images
```
Then add `--datasets drive stare chase_db1 hrf fives` to the run commands (the aggregation/report scripts pick FIVES up automatically as an extra column).
Expected: FIVES 200 @1024×1024, vessel fraction 10.0 %, FOV width 986 px.

## 3. Calibrate expectations (≈10 minutes)

```bash
python scripts/run_experiments.py --exp E_tglitenet --seeds 0 --quick        # pipeline smoke test (results/smoke)
python scripts/train.py --exp A_unet --seed 0 --evaluate --set train.num_workers=12 --out results/timing/A   # one full U-Net run
python scripts/train.py --exp E_tglitenet --seed 0 --evaluate --set train.num_workers=12 --out results/timing/E
```
Look at `results/timing/*/train_log.csv` (`time_s` column) and the console. Then: total ≈ 20 × t(U-Net run) [A,B,C,X_unet_dg × 5 seeds]
+ 25 × t(lite run) [D,E,X_lite_* × 5 seeds] + 21 × t(lite run) [topology + width, 3 seeds] + evaluation (≈ 1–3 min per run, MC passes included).
If it is too slow: lower `train.iters` (e.g. 3000) *for every stage* with `--set train.iters=3000` (never for only some stages), or use 3 seeds (`--seeds 0 1 2`).
GPU memory: U-Net at batch 8 × 256² with fp16 needs roughly 6–8 GB; on a smaller card use `--set train.batch_size=4` (again for all stages).
If the GPU sits idle, the CPU data pipeline is the bottleneck → increase `train.num_workers`.

Check the learning curve of the U-Net: DRIVE-val Dice should reach ≈ 0.80–0.83 within the 5000 iterations. If it is still rising at the end,
increase `train.iters` for **all** stages and re-run (that is a pre-registered protocol decision, not a tuning of the proposed method).

Evaluation note: metrics use native-resolution labels (HRF is 8 MP), costing ≈ 10 s per HRF image per mode on one core. Use
`--set eval.metric_workers=8` to compute metrics in a process pool; `eval.max_eval_pixels=2500000` caps the cost but inflates HRF Betti-0 (don't use it for final numbers).
Re-running a command skips finished runs **only if their saved config matches**; if you change a setting, the runner refuses to mix protocols and tells you to use `--force` or a new `--out`.

## 4. (Optional) choose the loss weights on DRIVE-val only

```bash
python scripts/tune_lambdas.py --set train.num_workers=12     # 9 short lite runs; writes configs/tuned.yaml (commit it!)
```
Skip it to use the defaults (λ₂ = λ₃ = 0.5). Either way state in the paper which you used.

## 5. The experiments (resumable; rerun the same command after an interruption)

```bash
export OMP_NUM_THREADS=4
python scripts/run_experiments.py --groups primary --native-eval --set train.num_workers=12 2>&1 | tee results/log_primary.txt
python scripts/run_experiments.py --groups factorial topology width --set train.num_workers=12 2>&1 | tee results/log_extra.txt
```
or everything at once: `bash scripts/run_all.sh` (Windows: `scripts\run_all.ps1`).
Priority if time is short: `primary` first (it alone answers RQ1–RQ4), then `factorial`, then `topology`, `width`.

Efficiency numbers (latency, memory) are taken on the machine that runs these commands. For the paper, quote that machine
(`results/runs/*/seed0/efficiency.json → hardware`), and ideally re-measure on a CPU-only laptop too (`python scripts/evaluate.py <run_dir> --efficiency`).

## 6. Results → tables → paper

```bash
python scripts/aggregate_results.py
python scripts/auto_findings.py                       # read results/findings.md
python scripts/make_data_overview.py --strip --out results/figures/data_overview_strip.png
python scripts/make_report_assets.py
cd report && pdflatex main && pdflatex main
```
Then write the red `\todo{}` blocks in `report/main.tex` (Sections VI–VIII and the end of the abstract) from the numbers, explaining *why* —
use `results/findings.md` and `results/tables/*.csv`. Check: ≤ 8 pages; every `\todo` gone; `\draftfalse`; the "Limitations" paragraph still true
(e.g. if you added FIVES or used the official DRIVE FOV masks, update the text); references all cited. The paper must not claim anything the tables don't show.
If a claim in the abstract/intro (written before results) is contradicted by the data, change the claim — that is a legitimate finding.

## 7. What to bring back / submit

* Commit: `results/tables/*.csv`, `results/findings.md`, `results/figures/*`, `results/runs/*/seed*/{config.yaml,train_log.csv,per_image*.csv,efficiency.json}`,
  `configs/tuned.yaml` (if used), `report/` (incl. `generated/` and `figures/`). Checkpoints (`best.pt`) and `preds/` are git-ignored; copy them separately if you want them.
* Course submission = zip of the code + README + results/log files + the PDF (dataset **not** included — the README says where to download it).

## 8. Git (no AI co-author trailers on the commits)

```bash
git init && git add . && git status                 # check: no data/, .venv/, *.pt in the list
git commit -m "TG-LiteNet: code, configs, report scaffold"
git branch -M main && git remote add origin <url> && git push -u origin main
```
`.gitignore` already excludes `data/raw`, `data/processed`, `data/cache`, `.venv`, checkpoints and LaTeX build files.

## 9. Sanity check done on the development laptop (CPU, tiny budget — not a result)

`D_lite_topo_dg`, 600 iterations, batch 4, 128-px patches, one seed: DRIVE-validation Dice 0.73 @ it 100 → 0.79 @ it 500, loss falling, so
data pipeline, labels, losses and evaluation are wired correctly. Full runs use 5000 iterations, batch 8, 256-px patches.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: ... run scripts/download_data.py and scripts/prepare_data.py first` | data step skipped, or a dataset is missing from `data.eval_datasets` |
| CUDA out of memory | `--set train.batch_size=4` (all stages) or `data.patch_size=192` |
| very slow epochs, GPU idle | more `train.num_workers`; make sure `cv2` is `opencv-python-headless`; check `OMP_NUM_THREADS` |
| `weights_only` / pickle error on load | use PyTorch ≥ 2.1; `evaluate.py` loads with `weights_only=False` on purpose (the checkpoint stores the config dict) |
| LaTeX: missing package | `tlmgr install IEEEtran adjustbox collectbox algorithms` or compile on Overleaf |
| results differ slightly between machines | expected (non-deterministic GPU kernels); the paper reports mean±std over seeds |

## 11. Kaggle / Colab (free GPUs) — the tested route

Measured on a Kaggle Tesla T4 with 4 CPU cores (default config, 5000 iterations, batch 8): U-Net (stage A) 18.5 min per run, TG-LiteNet with
augmentation (stage E) 13.8 min per run (CPU-bound augmentation), DRIVE-validation Dice ≈ 0.82–0.83 for both. Budget for 3 seeds of stages A–E ≈ 6 h.

Use **Save & Run All (Commit)** (keeps running with the tab closed / offline; the interactive session does not). Two cells:

```python
!git clone https://github.com/Deltasthicc/ML-Project.git
%cd /kaggle/working/ML-Project
!pip install -q -r requirements.txt && pip install -q -e .
```
```python
!bash scripts/kaggle_run.sh "primary" "0 1 2" 9.5
```
`kaggle_run.sh` resumes from earlier notebook outputs attached as inputs, downloads/prepares the data, trains in seed-major order with a wall-clock budget
(stops starting runs that would not finish in time, so the 12 h limit is never hit), aggregates the tables and writes `/kaggle/working/results_bundle.zip`.
Continue later (more seeds / groups) by attaching the previous notebook's *output* as an input of the new notebook and running the same command again.
In the Save dialog choose "Always save output" if the option exists. `NATIVE=1` adds the no-rescale evaluation (slow for the U-Net on HRF). On Colab write to Drive
(`OUT=/content/drive/MyDrive/tglite_runs`) and use `NW=2`.
Measure latency/memory in one session only (`python scripts/evaluate.py <run_dir> --efficiency`), because the free tiers hand out different GPU models.
