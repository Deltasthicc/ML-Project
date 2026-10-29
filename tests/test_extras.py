"""Tests for the audit additions: new experiments, post-processing control, confusion matrix, paired image statistics."""
from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd

from tglite.analysis import confusion_summary, paired_delta, paired_image_deltas, stage_deltas, welch_delta
from tglite.config import ROOT
from tglite.experiments import GROUPS, make_config
from tglite.models import build_model, count_params

sys.path.insert(0, str(ROOT / "scripts"))
import eval_postproc  # noqa: E402


def test_new_experiments_are_registered_and_distinct():
    b8, b16 = make_config("S_unet_b8"), make_config("S_unet_b16")
    assert (b8.model.backbone, int(b8.model.base_channels)) == ("unet", 8) and int(b16.model.base_channels) == 16
    assert b8.loss.topo == "cldice" and b8.dg.enabled                                # same recipe as stage D
    assert count_params(build_model(b8)) < count_params(build_model(b16)) < 2.1e6    # both fit the <= 2 M budget
    noskel, nocons, d = make_config("Z_lite_noskel"), make_config("Z_lite_nocons"), make_config("D_lite_topo_dg")
    assert not noskel.model.skeleton_head and float(noskel.loss.lambda_skel_head) == 0.0 and d.model.skeleton_head
    assert float(nocons.loss.lambda_cons) == 0.0 and float(d.loss.lambda_cons) > 0 and nocons.dg.enabled
    assert set(GROUPS["compact_unet"]) == {"S_unet_b8", "S_unet_b16"} and set(GROUPS["controls"]) == {"Z_lite_noskel", "Z_lite_nocons"}


def test_clean_prob_removes_only_small_components():
    prob = np.zeros((200, 200), np.float32)
    prob[20:60, 20:60] = 0.9            # one big component (1600 px)
    prob[100, 100] = 0.9                # isolated speck (1 px)
    prob[150:153, 150:153] = 0.9        # small blob (9 px)
    fov = np.ones_like(prob)
    thr = 0.5
    # the cut-off is given at the 527-px FOV scale and rescaled to the image: this 200x200 test image has scale^2 ~ 0.18,
    # so k = 100 means "remove components below about 18 px" here (the 9-px blob and the speck go, the 1600-px one stays)
    kept = eval_postproc.clean_prob(prob, fov, thr, k=100) >= thr
    assert kept[30, 30] and not kept[100, 100] and not kept[151, 151]
    assert kept[20:60, 20:60].all()                                                    # the big component is untouched
    # k = 0 / tiny k leave the binarisation untouched
    assert ((eval_postproc.clean_prob(prob, fov, thr, k=1) >= thr) == (prob >= thr)).all()


def test_confusion_matrix_is_derived_exactly(tmp_path):
    # one 10x10 "image": 20 vessel pixels, 80 background, FOV = everything
    proc = tmp_path / "proc" / "stare"
    (proc / "masks").mkdir(parents=True), (proc / "fov").mkdir(parents=True)
    gt = np.zeros((10, 10), np.uint8)
    gt[0:2, :] = 255
    cv2.imwrite(str(proc / "masks" / "01.png"), gt)
    cv2.imwrite(str(proc / "fov" / "01.png"), np.full((10, 10), 255, np.uint8))
    sens, spec = 0.75, 0.9                                  # TP = 15, FN = 5, TN = 72, FP = 8
    run = tmp_path / "runs" / "E" / "seed0"
    run.mkdir(parents=True)
    pd.DataFrame([{"exp": "E", "seed": 0, "mode": "det", "dataset": "stare", "id": "01", "sens": sens, "spec": spec,
                   "acc": (15 + 72) / 100}]).to_csv(run / "per_image.csv", index=False)
    c = confusion_summary(tmp_path / "runs", tmp_path / "proc")
    r = c[(c.exp == "E") & (c.dataset == "stare")].iloc[0]
    assert np.isclose([r.tp, r.fp, r.fn, r.tn], [0.15, 0.08, 0.05, 0.72]).all()
    assert np.isclose(r.precision, 15 / 23) and np.isclose(r.f1, 30 / 43) and np.isclose(r.accuracy, 0.87)


def test_paired_image_deltas_direction_and_counts():
    rng = np.random.default_rng(0)
    rows = []
    for exp, shift in (("A", 0.0), ("B", -0.01)):
        for seed in (0, 1):
            for i in range(30):
                rows.append({"exp": exp, "seed": seed, "mode": "det", "dataset": "stare", "id": f"{i:02d}", "dice": 0.8 + 0.001 * i + shift + 0.002 * rng.normal(),
                             "cldice": 0.8, "betti0_err": 10.0})
    t = paired_image_deltas(pd.DataFrame(rows), [(("A", "det"), ("B", "det"))])
    r = t[(t.scope == "unseen") & (t.metric == "dice")].iloc[0]
    assert r.n_images == 30 and r.delta < 0 and r.hi95 < 0 and r.frac_lower > 0.9


def test_paired_delta_uses_shared_seeds_and_falls_back_to_welch():
    seeds = [0, 1, 2, 3, 4]
    base = pd.Series([0.70, 0.74, 0.78, 0.72, 0.76], index=seeds)
    better = base + 0.01 + pd.Series([0.001, -0.001, 0.0, 0.001, -0.001], index=seeds)    # constant gain on every seed
    d, ci, p, test, n = paired_delta(better, base)
    assert test == "paired-t" and n == 5 and np.isclose(d, 0.01, atol=1e-3) and p < 0.001
    assert ci < welch_delta(better.to_numpy(), base.to_numpy())[1] / 5                       # pairing removes the seed-to-seed spread
    # seeds that do not overlap: only the unpaired test is possible
    other = pd.Series([0.71, 0.75], index=[10, 11])
    assert paired_delta(other, base)[3] == "welch"
    # a stage_deltas row for a pair of runs carries the test and the number of pairs
    rows = [{"exp": e, "mode": "det", "seed": sd, "dataset": "unseen", "dice": v} for e, ser in (("A", base), ("B", better)) for sd, v in ser.items()]
    t = stage_deltas(pd.DataFrame(rows), [(("A", "det"), ("B", "det"))], datasets=("unseen",), metrics=("dice",))
    assert t.iloc[0].test == "paired-t" and t.iloc[0].n_pairs == 5


def test_unseen_image_bootstrap_is_a_macro_average_over_datasets():
    rows = []
    for ds, n, gain in (("stare", 20, 0.10), ("chase_db1", 20, 0.10), ("hrf", 40, -0.10)):    # HRF would dominate a pooled mean
        for exp, shift in (("A", 0.0), ("B", gain)):
            for i in range(n):
                rows.append({"exp": exp, "seed": 0, "mode": "det", "dataset": ds, "id": f"{i:02d}", "dice": 0.5 + shift, "cldice": 0.5, "betti0_err": 1.0})
    t = paired_image_deltas(pd.DataFrame(rows), [(("A", "det"), ("B", "det"))])
    r = t[(t.scope == "unseen") & (t.metric == "dice")].iloc[0]
    assert np.isclose(r.delta, (0.10 + 0.10 - 0.10) / 3)                                      # equal weight per dataset, as in the tables
    assert np.isclose(r.frac_lower, 0.5) and r.n_images == 80
