# TG-LiteNet v2 - pre-registered evaluation protocol

Written **before** any v2 model was trained or any target dataset evaluated. At the end of the DRIVE-only phases the
suite records the SHA-256 of this file, `configs/base.yaml` and `configs/experiments.yaml` in `protocol/decisions.json`
(`frozen: true`). From then on, target datasets can be evaluated (`src/tglite/evaluate.py` refuses earlier). Any analysis
added after the freeze is labelled **exploratory** in the report.

v1 of this project (tag `v1-final`) used the same targets, so the targets are *held-out*, not untouched: we know the
v1 results. This v2 protocol fixes the rules in advance so that no choice depends on the targets.

## 1. Data and roles

| Role | Data | Used for |
|---|---|---|
| train | DRIVE official training images, 15 of 20 | gradient updates |
| validation | DRIVE official training images, 5 of 20 (`protocol/drive_split.json`, `random.Random(2026)`) | checkpoint selection, training budget, backbone choice, threshold / fragment size / temperature |
| in-domain test | DRIVE official test images (20); labels from the PaddleSeg mirror, verified (training labels identical to official, test images pixel-identical) | reporting only |
| unseen targets | STARE (20, Hoover labels), CHASE_DB1 (28, 1stHO), HRF (45) | reporting only; "unseen" = equal-weight mean of the three |
| extra target | FIVES test split (200, native 2048 px) | reporting only, never averaged into "unseen" |

FOV masks: official for DRIVE and HRF, estimated for STARE, CHASE_DB1 and FIVES (thresholding; checked visually in
`results/data/overview.png`). Second observers (STARE vk, CHASE 2ndHO) are not used for any decision.

## 2. Pipeline (identical for every model)

Scale normalisation to an FOV width of 540 px (median official DRIVE FOV width) - green channel - CLAHE (clip 2, 64-px
tiles) - FOV standardisation. Predictions are resampled to native resolution; **all metrics are computed inside the FOV
against native-resolution labels**.

## 3. Training (identical for every model)

AdamW, lr 1e-3, weight decay 1e-4, 300 warm-up iterations, cosine decay to 1 %, gradient clipping 1.0, fp16 autocast,
256-px patches, **16 patch views per step for every stage** (16 clean patches; or 8 clean + their 8 DG-perturbed views),
validation on DRIVE-val every 500 iterations, best checkpoint by DRIVE-val Dice at threshold 0.5.
Loss weights fixed in advance: lambda_BCE 1, lambda_topo 0.5, lambda_skel 0.5, lambda_cons 0.5 (not tuned).

**Training budget: fixed at 10 000 iterations by the team (2026-11-11), before any v2 run was trained.** This replaces the
pilot rule below, which is kept for reference but was not run (no pilot runs exist).

*(Superseded)* **Training budget (pilot rule).** `pilot_unet` (U-Net w64) and `pilot_lite` (depthwise w16), both with the stage-C recipe,
train 20 000 iterations on one seed with a constant learning rate after warm-up (a decaying schedule would push the
plateau towards the end of the pilot regardless of convergence). For each curve, smooth DRIVE-val Dice with a centred 3-point moving average and take
the first iteration within 0.002 of its maximum. Budget = the larger of the two, rounded up to 1 000, clipped to
[4 000, 20 000]. Every later run uses this budget.

## 4. Compact backbone (selection rule)

Candidates (stage-D recipe, 3 seeds): depthwise `lite` w16, `hybrid` w16 (plain convs at the two highest resolutions,
depthwise below), plain U-Net w8, plain U-Net w16. Among candidates with <= 2 M parameters, keep those whose mean
DRIVE-val Dice is within 0.005 of the best candidate, and choose the lowest PyTorch CPU latency (4 threads, 512x512, this
machine). The choice defines `compact` for stages D, E and all compact cells. The non-chosen candidates are evaluated on
the targets after the freeze as controls.

## 5. Evaluation modes (per model, chosen on DRIVE-val only)

* **raw** (headline): threshold 0.5, no post-processing.
* **clean**: threshold 0.5 + removal of 8-connected components smaller than k, with k from {0,10,20,30,50,75,100,150,200,300}
  (px at the 540-px scale, converted to native as k / s^2) minimising mean DRIVE-val Betti-0 error (ties: smaller k).
* **tuned**: threshold from {0.30, 0.35, ..., 0.80} maximising mean DRIVE-val Dice (ties: closest to 0.5), then k chosen as
  above at that threshold.
* **temperature** T (grid 0.25-4, DRIVE-val NLL) is used only for the calibrated ECE, never for segmentation.
Choices are made separately for each run (and for each MC / ensemble variant) and stored in `val_selection_*.json`.

## 6. Metrics and comparisons

Per image: Dice, IoU, precision, sensitivity, specificity, AUC, clDice, Betti-0 and Betti-1 error (8-connectivity,
beta1 = beta0 - Euler number), ECE (15 bins), ECE after temperature scaling. Gap = DRIVE-test Dice - unseen Dice.

**Primary family** (mode raw; Holm correction over all comparisons x metrics in the family):
A->B (RQ1 topology), B->C (RQ2 DG), C->D (RQ3 compact), D->E (dropout), E det->E MC, A->E MC;
metrics: unseen Dice, unseen clDice, unseen Betti-0 error, Dice gap.
Each comparison: seed-paired mean difference with a 95 % t interval (5 seeds), Holm-adjusted p, and an image-level
paired bootstrap (2 000 resamples, stratified by dataset, equal dataset weight) on the unseen datasets.
The same family is also reported for the clean and tuned modes, and a **secondary family** (DG without topology, topology
given DG, warm-up, ensemble vs MC) is reported with its own Holm correction. A difference is called "clear" only when its
95 % interval excludes 0.

## 7. Uncertainty (RQ4)

Uncertainty = binary entropy of the mean predicted probability. Compared: E deterministic (single-pass entropy),
E with MC dropout (T = 20, p = 0.1), and a 5-seed deep ensemble of stage D (and of stage A). Per dataset: pixel-level
Spearman rho (uncertainty vs. error, 200 000 FOV pixels per image), image-level Spearman rho (mean uncertainty vs.
1 - Dice), image risk-coverage (AURC vs. random and oracle) and pixel error at 90 % coverage, ECE before / after
temperature scaling. RQ4 is supported if MC dropout or the ensemble ranks image errors better than random on the unseen
datasets.

## 8. Cost

On this machine only (RTX A4000, i9-12900K): parameters, FLOPs at 512x512, PyTorch CPU latency (4 threads and 1 thread),
ONNX Runtime CPU latency (4 threads), GPU latency fp32 / fp16, peak GPU memory; batch 1, skeleton head removed.
MC dropout and ensembles multiply the cost by 20 and 5.

## 9. Seeds

Stages A-E: 5 seeds (0-4). Backbone candidates, factorial cells, topology variants, warm-up, controls, width: 3 seeds.
No run is repeated or discarded because of its result; a crashed run is resumed from its checkpoint.
