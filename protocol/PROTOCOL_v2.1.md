# TG-LiteNet v2.1 - protocol addendum (pre-registered)

Written on 2026-11-13, after the v2 study (protocol/PROTOCOL.md, frozen 2026-11-11) had finished and before any v2.1
model was trained or any v2.1 analysis was run. The v2 protocol, configs and results are not changed; v2.1 is a separate,
clearly labelled follow-up whose new results go to `results/v21/`. The suite records the SHA-256 of this file and of
`configs/v21.yaml` in `protocol/decisions_v21.json` before it computes anything.

**Honesty note.** The v2 results on the targets are known (they motivated v2.1). v2.1 therefore cannot claim that the
targets are untouched; it fixes every new rule in advance so that no choice depends on target data. All selection
below uses DRIVE-val only. Approval: the team asked for this round to be built and run end to end on 2026-11-13.

## 1. Goal

A ~70 K-parameter headline model (the size class of DSAE-Net / LightVesselNet) that is as close as possible to the
published accuracy, plus four topology-based novelties, each tested on its own:

1. **Connectivity reliability score** - flag unreliable predictions from the prediction's own fragmentation (label-free,
   single pass), compared with entropy, MC dropout and deep ensembles.
2. **Centreline-calibre-guided test-time scale selection** - run each image at several scales and keep the one whose
   predicted vessels, measured along their own centrelines, are as wide as on the training data (no labels, no weight change).
3. **Topology-aware distillation** - the small student matches the soft centrelines of a large teacher, not only its pixels.
4. **Skeleton-level consistency** - the DG-perturbed view must reproduce the centrelines (not the pixels) of the clean view.

## 2. Unchanged from v2

Data roles, preprocessing (FOV width 540 px, green, CLAHE, standardisation), optimiser and schedule (AdamW 1e-3,
cosine, 300 warm-up, clip 1, AMP), 16 views per step, 256-px patches, 10 000 iterations, validation every 500 iterations
with best checkpoint by DRIVE-val Dice at 0.5, metrics at native resolution inside the FOV, Betti definitions, seeds.

## 3. Training recipe (configs/v21.yaml `defaults`)

* Dice + BCE + soft-clDice (lambda_topo) + skeleton head (0.5).
* DG view as in v2 plus RandConv (p 0.5) and low-frequency Fourier-amplitude jitter (p 0.5), both on the RGB image before
  CLAHE, no target data.
* Consistency (0.5) in **skeleton** form: 1 - soft-clDice(P_dg, stopgrad(P_clean)).
* Distillation from the v2 stage-C U-Net (`C_unet_topo_dg`, 5-seed ensemble, flip-averaged) on the 15 DRIVE-train images
  (maps precomputed at the normalised scale and warped with the patch): pixel term BCE(student, teacher) x 0.5 and
  skeleton term soft-clDice(student, teacher) x lambda_kd_skel. The teacher never sees DRIVE-val or any target.
* W-Net only: auxiliary Dice + BCE on the first stage (0.5).

## 4. Decisions (DRIVE-val only, in this order)

1. **Architecture**: `L_cap` (U-Net, channels 8-12-16-24-32, 66 K parameters) vs `L_wnet` (two chained U-Nets 4-8-12-16-24,
   64 K). 3 seeds each, default weights (lambda_topo 0.5, lambda_kd_skel 0.5). Keep candidates whose mean DRIVE-val Dice is
   within 0.005 of the best; choose the lower PyTorch CPU latency (4 threads, 512x512).
2. **Loss weights** on the chosen architecture: lambda_topo in {0.25, 0.5} x lambda_kd_skel in {0.5, 1.0}, seeds 0-1
   (the 0.5/0.5 cell reuses the architecture runs). Criterion: mean of (val Dice + val clDice)/2 at the best checkpoint
   (connectivity is a goal, so Dice alone is not the criterion); ties -> 0.5/0.5.
3. **Size rule** (headline): among `H_tglite70k` (chosen architecture + weights), `R_lite_w16` (0.21 M depthwise) and
   `R_unet_w8` (0.49 M), all with the same recipe, the headline is the smallest model whose mean DRIVE-val Dice over
   seeds 0-2 is within 0.005 of the best. The ~70 K model is expected to qualify; if it does not, that is reported.
4. **Per-model evaluation choices** as in v2 (threshold, fragment size, temperature on DRIVE-val).

After these decisions `decisions_v21.json` gets `frozen: true`; only then are the new models evaluated on the targets.

## 5. Runs

* Headline `H_tglite70k`: 5 seeds; 5-seed deep ensemble `ENS_H_tglite70k` (evaluation only).
* References `R_lite_w16`, `R_unet_w8`: 3 seeds.
* Ablations of the headline (3 seeds): `Z_no_kdskel` (novelty 3 off), `Z_no_kd` (no distillation), `Z_pixcons`
  (novelty 4 off: pixel consistency), `Z_noaug` (no RandConv / Fourier), `Z_v2recipe` (all v2.1 training changes off).
* Re-evaluation, without retraining, of the v2 models A, C, D, E (det and MC) and the ensembles ENS_D, ENS_A.

## 6. Evaluation modes

* **tuned** (v2.1 headline mode): threshold and fragment size chosen on DRIVE-val per model. (`raw` = 0.5 is also reported.)
* **sel** (novelty 2, as amended - see Amendment 1): the image is processed at scales {1, 1.5, 2, 2.5} x the normalised
  scale; at each scale the mean vessel calibre of the tuned-threshold prediction is measured on the prediction's own
  skeleton (2 x distance transform, at the processing resolution); keep the scale whose calibre is closest to the calibre
  the same model predicts on the 15 DRIVE-train images at scale 1; ties -> smaller scale; then tuned threshold + fragment
  size on the prediction resampled to the normalised scale. Also reported: DRIVE-val images shrunk by 0.67 / 0.5 / 0.4
  (synthetic thinner vessels), Dice gain of the rule and of the best fixed scale over scale 1. **Validation check:** if on DRIVE-val the rule lowers mean Dice by more than
  0.01 versus scale 1 for a model, the rule counts as failed for that model and is not used for claims (still reported).
* **network resolution**: Dice / clDice with prediction and label both at the 540-px-FOV scale, next to the native
  metrics, so that the comparison with papers that score downsampled images is fairer. Native resolution stays the headline.

## 7. Novelty 1: reliability scores (label-free, at the normalised scale, binary prediction at 0.5, no clean-up)

* **rel_frag** (primary): number of 8-connected components per 10^5 FOV pixels.
* rel_endpoints: skeleton end points per 100 skeleton pixels; rel_smallfrac: share of predicted vessel pixels in
  components < 100 px (secondary).
* Compared with: mean binary entropy of the prediction (single pass), of the MC-dropout mean (E, T = 20) and of the
  ensemble mean (ENS_D, ENS_A, ENS_H), and the MC / ensemble variance.
* Measures, per dataset and as the mean over the three unseen datasets: Spearman rho between score and image error
  (1 - Dice, tuned mode); AURC gain = share of the random-to-oracle AURC gap closed when the least reliable images are
  removed first. Also the pooled unseen ranking.
* **Success criterion**: for the headline and for v2 stage E, rho(rel_frag) - rho(entropy) on the unseen mean is positive
  with a seed-paired 95 % interval excluding 0, and rel_frag beats MC dropout for E. Otherwise reported as negative.

## 8. Novelties 2-4: success criteria

* Novelty 2: unseen Dice (sel - tuned) seed-paired interval excludes 0 in favour of sel, for a model whose DRIVE-val
  check passed. HRF is expected to drive any gain (thin labels at the normalised scale).
* Novelty 3: headline - `Z_no_kdskel` on unseen clDice (and Dice) - seed-paired interval (3 seeds) excludes 0.
* Novelty 4: headline - `Z_pixcons` on unseen Dice / clDice - seed-paired interval excludes 0.
* With 3 seeds these tests have low power; a non-significant result is reported as "not shown", not as "no effect".
  No Holm correction is applied across the four novelties (each is a separate question); this is stated in the report.

## 9. Cost

As in v2 (parameters, FLOPs, CPU 4t/1t, ONNX CPU, GPU fp32/fp16, memory) plus an INT8 check of headline seed 0: ONNX
Runtime static QDQ quantisation calibrated on 8 DRIVE-train images; latency, file size and per-dataset Dice vs fp32.

## 10. Comparison with published work

Published numbers (DSAE-Net, LightVesselNet, TFFM, DGSSA, WaveRNet) are quoted from the papers as context only;
training data, target access, scoring resolution and thresholds differ, so they are not ranked against ours.

## Amendment 1 (2026-11-13, before any real v2.1 run)

The smoke test (60-iteration models, 2 images per dataset, `runs_smoke/`) showed that the originally registered
novelty-2 criterion (lowest fragment density after resampling to the normalised scale) is structurally biased: fragment
density rises with the processing scale for every image, so the rule would almost never leave scale 1. The smoke output
also showed fixed-scale Dice of those 60-iteration models on 2 images per target dataset; this is disclosed here.
The replacement criterion was chosen **on DRIVE only**: DRIVE-val images were shrunk by 0.67 / 0.5 / 0.4 (vessels
thinner, as in a camera or annotation shift) and candidate label-free criteria (fragment density, end-point density,
small-fragment share, giant-component share, fragments per skeleton length, prediction confidence, calibre matching)
were compared for the v2 models D (seed 0) and C (seed 0). Calibre matching at the processing resolution was the only
criterion that never moved off scale 1 on unshrunk DRIVE-val and recovered most of the oracle gain on shrunk images
(D: +0.009 / +0.028 / +0.073 Dice; C: +0.001 / +0.009 / +0.040). It is a centreline (skeleton) based, label-free
criterion; the fragment-density score remains the novelty-1 reliability score unchanged. The simulation script is
`scripts/check_scale_rule.py`.
