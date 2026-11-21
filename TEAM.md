# Team and responsibilities

CSD361 *Introduction to Machine Learning*, Group 22, Department of CSE, Shiv Nadar Institution of Eminence.

The project is split into three equal parts, one for each part of the name **TG-LiteNet**: **T**opology, **G**eneralisation and **Lite**.
Each person owns the ML work, experiments, code, paper sections and website sections for their part, across all three versions (v1, v2 and v2.1).

| | **Person 1: Topology (T)** | **Person 2: Generalisation (G)** | **Person 3: Lite** |
|---|---|---|---|
| **Owner** | **Abhiraj Agarwal** (2510110002) | **Tanish Shitanshu** (2510110008) | **Shashwat Rajan** (2510111019) |
| Core question | Do vessels stay connected, and can we tell when a prediction is unreliable? | Does the model still work on cameras it never saw? | How small and fast can the model get without losing accuracy? |
| Step in the A–E ladder | B (topology loss), E (dropout and uncertainty) | C (augmentation + consistency) | D (compact backbone) |
| Novelties | **N1** connectivity reliability score | **N2** calibre-guided scale selection (the supported one), **N4** skeleton consistency | **N3** topology-aware distillation, plus INT8 quantisation |
| Main ML themes | topology losses, topological metrics, uncertainty, calibration, statistics | preprocessing, augmentation, domain shift, test-time adaptation | architectures, model selection, knowledge distillation, compression, optimisation |
| Infrastructure | metrics and statistics engine | data pipeline and pre-registered protocol | training loop, resumable suite, cost benchmarks |
| Website sections | threshold viewer, N1 card, Scope | scale rule, controls, versions, N2 and N4 cards | hero snapshot, model explorer, N3 card, reproduce, site-data builder |

## The ladder the parts plug into

* **A** plain U-Net, 31M parameters (the baseline)
* **B** add a topology loss so vessels stay connected (Person 1)
* **C** add augmentation that imitates other cameras (Person 2)
* **D** switch to a small, fast backbone (Person 3)
* **E** add dropout to measure uncertainty (Person 1)

---

## Person 1: Topology and reliability, Abhiraj Agarwal

*Do the vessels stay connected, and can the model tell when it is wrong?*

**Owns**
* Topology losses: soft-clDice, the skeleton prediction head, and two alternatives (cbDice, Skeleton Recall).
* Topology metrics: clDice and Betti errors (counting broken pieces and loops).
* Evaluation modes: raw (threshold 0.5), clean (small fragments removed) and tuned (threshold picked on DRIVE-val).
* Uncertainty and calibration: MC dropout, ensembles, entropy and calibration error.
* Novelty **N1**: the label-free connectivity reliability score.
* Statistics: paired confidence intervals over seeds, Holm correction and bootstrap.

**Main experiments:** stage B (U-Net + topology loss), stage E (dropout + MC dropout), the cbDice and Skeleton Recall variants, and the deep ensembles.

**Code:** the topology terms of `src/tglite/losses.py`, `metrics.py`, `inference.py`, `selection.py` (threshold, fragment-size and temperature calibration on DRIVE-val), and the N1 and statistics parts of `eval21.py` and `report.py`.

**Paper:** topology related work, problem setup, topology loss and uncertainty methods, metrics and statistics, and their results.
**Website:** threshold viewer, N1 card, Scope section.

**Key results**
* The topology loss cuts broken pieces from 72.8 to 33.9, but simply removing small fragments brings every model to about 4.5, so raw piece counts can mislead.
* With tuned thresholds, the topology loss improves unseen Dice (0.762 to 0.775).
* MC dropout adds almost nothing (+0.001 raw Dice) and is 20x slower.
* N1 did not meet its main target, but its secondary scores (end-point density, small-fragment share) are a promising lead.

---

## Person 2: Data and generalisation, Tanish Shitanshu

*Does the model still work on cameras it never saw?*

**Owns**
* Data pipeline: preparing the datasets, finding the field of view, resizing every image to the same scale, green channel, CLAHE and normalisation.
* Augmentation: changes in colour, lighting, blur, noise and shape that imitate other cameras.
* Consistency loss: the model must give the same answer on the clean and augmented image.
* Novelty **N2**: run the image at a few scales and keep the one whose vessel width looks most like the training data. No labels needed.
* Novelty **N4**: consistency on vessel centrelines instead of pixels.
* Protocol: the pre-registered rules that lock the test sets until all decisions are made.

**Main experiments:** stage C (adds augmentation and consistency; also the teacher for distillation), the 2x2x2 factorial study, and the ablations without consistency, without the new augmentations and with pixel consistency.

**Code:** the `src/tglite/data/` folder, the consistency losses in `losses.py`, scale selection in `eval21.py`, `scripts/prepare_data.py`, `scripts/check_scale_rule.py`, and the `protocol/` folder.

**Paper:** introduction, domain-generalisation related work, preprocessing and N2 method, datasets and protocol, N2/N4 and factorial results.
**Website:** scale rule, controls, versions, N2 and N4 cards.

**Key results**
* Augmentation without the topology loss gives the best raw unseen Dice in v2 (0.774).
* With tuned thresholds, stage C improves unseen Dice from 0.775 to 0.789.
* N2 helps all four small models (about +0.01 Dice), mostly on HRF (+0.034 for the 66K model). It lowers Dice on FIVES and on the large U-Net, and both are reported.
* N4 is slightly worse than plain pixel consistency (-0.009 Dice).

---

## Person 3: Model size, speed and training, Shashwat Rajan

*How small and fast can the model get, and what does it cost in accuracy?*

**Owns**
* Backbones: U-Nets of different widths, depthwise and hybrid networks, W-Net, and the final 66K U-Net.
* Model selection rules: how the backbone, architecture, loss weights and final model were chosen.
* Novelty **N3**: distillation, where the small model learns from an ensemble of five large U-Nets.
* Cost measurement: parameters, FLOPs, CPU/GPU speed and memory.
* INT8 quantisation to shrink the model file.
* Training system: optimiser, learning-rate schedule and resumable training runs.

**Main experiments:** stage D (compact backbone), backbone candidates, the width sweep, the v2.1 architecture candidates, the loss-weight grid, the 66K model and its ensemble, and the distillation ablations.

**Code:** `src/tglite/models.py`, `train.py`, the distillation losses in `losses.py`, `efficiency.py`, `int8.py`, `suite.py`, `suite21.py`, the `configs/` folder and the run scripts.

**Paper:** lightweight-network related work, network and distillation methods, training details, cost and 66K results, N3, conclusion.
**Website:** hero snapshot, explorer chart, N3 card, reproduce section, site-data scripts.

**Key results**
* The 486K U-Net is about 31x faster on CPU than the 31M U-Net, for about 0.01 Dice.
* Depthwise convolutions save parameters but not time on CPU.
* The 66K U-Net was chosen over W-Net because it is faster at the same accuracy.
* INT8 shrinks the file from 266 KB to 109 KB with no Dice loss, but it was slower on the test CPU, so only the size gain is claimed.
* N3 had no clear effect with 3 seeds.

---

## How the parts connect

* Person 2 to everyone: the cleaned data, the augmented images and the locked test sets.
* Person 2 to Person 3: the stage-C models, used as the teacher for distillation.
* Person 1 to Person 2: the tuned threshold and fragment size, used after scale selection.
* Persons 1 and 2 to Person 3: their loss terms, which run inside the training loop.
* Person 3 to everyone: the trained models and cost numbers used in every table and chart.

## What everyone should know

* **Rules:** train only on DRIVE-train, make every choice using DRIVE-val, and only look at the test sets after all decisions are frozen.
* **Headline numbers:** 66K parameters (469x smaller than the U-Net), 0.768 unseen Dice (0.779 with scale selection), 109 KB as INT8, about 25 ms per image on a CPU, and 114 models trained in v2 and v2.1.
* **Limitations:** only 15 training images from one dataset; ablations use 3 seeds, so "no clear effect" does not mean "no effect"; scale selection is best for thin vessels and lowers Dice on FIVES; INT8 was slower on the test CPU; published numbers from other papers are for context, not a ranking.
