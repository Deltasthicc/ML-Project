"""Fast CPU tests for the building blocks (no datasets needed)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tglite import metrics as M
from tglite.config import AttrDict, make_config
from tglite.losses import Objective, soft_cldice_loss, soft_skeleton, topo_scale
from tglite.models import build_model, count_params
from tglite.selection import decide_backbone, decide_budget, modes_from_selection

DEC = {"budget_iters": 1000, "compact_model": {"backbone": "hybrid", "width": 16}}


def _line_mask(h=64, w=64, gap=False):
    m = np.zeros((h, w), np.uint8)
    m[30:33, 5:60] = 1
    if gap:
        m[30:33, 30:33] = 0
    return m


def test_overlap_metrics_perfect_and_empty():
    g = _line_mask()
    f = np.ones_like(g)
    r = M.overlap_metrics(g, g, f)
    assert r["dice"] == pytest.approx(1.0) and r["sens"] == pytest.approx(1.0)
    r0 = M.overlap_metrics(np.zeros_like(g), g, f)
    assert r0["dice"] < 1e-6


def test_betti_and_cldice_detect_a_break():
    g = _line_mask()
    f = np.ones_like(g)
    gtt = M.GTTopology(g, f)
    assert M.betti(g) == (1, 0)
    broken = M.topology_metrics(_line_mask(gap=True), gtt)
    assert broken["betti0_err"] == 1
    assert M.topology_metrics(g, gtt)["cldice"] == pytest.approx(1.0)
    ring = np.zeros((40, 40), np.uint8)
    ring[10:30, 10:30] = 1
    ring[13:27, 13:27] = 0
    assert M.betti(ring) == (1, 1)


def test_remove_small_and_fragment_scaling():
    m = _line_mask()
    m[5, 5] = 1                                     # single-pixel fragment
    assert M.betti(m)[0] == 2
    assert M.betti(M.remove_small(m, 5))[0] == 1
    prob = m.astype(np.float32)
    # 5 px at the normalised scale with scale factor 0.5 -> 20 native pixels removed threshold
    r = M.evaluate_image(prob, _line_mask(), np.ones_like(m), {"raw": (0.5, 0), "clean": (0.5, 5)}, scale=0.5)
    assert r["raw/betti0_err"] == 1 and r["clean/betti0_err"] == 0


def test_ece_calibrated_is_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200000)
    y = (rng.uniform(0, 1, p.size) < p).astype(np.float32)
    assert M.ece(p, y) < 0.01
    assert M.ece(np.clip(p * 0.5, 0, 1), y) > 0.1


@pytest.mark.parametrize("backbone,width,maxp", [("unet", 64, 32e6), ("unet", 16, 2e6), ("lite", 16, 0.3e6), ("hybrid", 16, 0.5e6)])
def test_models_shapes_and_size(backbone, width, maxp):
    cfg = AttrDict.wrap({"model": {"backbone": backbone, "width": width, "dropout": 0.1, "skeleton_head": True}})
    m = build_model(cfg)
    x = torch.randn(2, 1, 64, 80)
    out, sk = m(x, with_skeleton=True)
    assert out.shape == (2, 1, 64, 80) and sk.shape == out.shape
    assert count_params(m) < maxp


def test_soft_skeleton_and_cldice_loss_gradients():
    y = torch.from_numpy(_line_mask()).float()[None, None]
    sk = soft_skeleton(y, 5)
    assert sk.sum() > 0 and sk.max() <= 1.0 + 1e-5
    logits = torch.zeros_like(y, requires_grad=True)
    loss = soft_cldice_loss(torch.sigmoid(logits), y, torch.ones_like(y), (sk > 0.5).float())
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0


def test_objective_terms_and_warmup():
    cfg = make_config("C_unet_topo_dg", 0, ["loss.topo_warmup_iters=0.5"], decisions=DEC)
    assert cfg.loss.topo_warmup_iters == 500 and cfg.train.iters == 1000
    obj = Objective(cfg)
    y = torch.from_numpy(_line_mask()).float()[None, None]
    out = (torch.zeros_like(y), torch.zeros_like(y))
    t0 = obj.supervised(out, y, torch.ones_like(y), y, it=0)
    t1 = obj.supervised(out, y, torch.ones_like(y), y, it=999)
    assert set(t0) == {"dice", "bce", "topo", "skel_head"}
    assert float(t0["topo"]) < float(t1["topo"])
    assert topo_scale(0, 0) == 1.0


def test_config_resolution_compact_and_width_override():
    d = make_config("D_compact", 3, decisions=DEC)
    assert d.model.backbone == "hybrid" and d.model.width == 16 and d.seed == 3 and d.role == "main"
    s = make_config("S_compact_w32", 0, decisions=DEC)
    assert s.model.backbone == "hybrid" and s.model.width == 32
    with pytest.raises(RuntimeError):
        make_config("D_compact", 0, decisions={})
    with pytest.raises(RuntimeError):
        make_config("A_unet", 0, decisions={})                    # iters=auto needs the budget
    assert make_config("pilot_unet", 0, decisions={}).train.iters == 20000


def test_budget_rule(tmp_path):
    its = np.arange(500, 20001, 500)
    curve = 0.82 - 0.2 * np.exp(-its / 2500)
    d = tmp_path / "pilot_x" / "seed0"
    d.mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({"iter": its, "val_dice": curve}).to_csv(d / "train_log.csv", index=False)
    res = decide_budget([d / "train_log.csv"])
    assert 4000 <= res["budget_iters"] <= 20000 and res["budget_iters"] % 1000 == 0
    plateau = res["pilots"]["pilot_x"]["plateau_iter"]
    assert curve[list(its).index(plateau)] >= curve.max() - 0.003


def test_backbone_rule_prefers_fast_within_tolerance():
    c = {"slow_best": {"val_dice": [0.80, 0.80], "cpu_ms": 400, "params": 1e6, "model": {"backbone": "lite"}},
         "fast_close": {"val_dice": [0.797, 0.797], "cpu_ms": 100, "params": 1.5e6, "model": {"backbone": "unet"}},
         "fast_bad": {"val_dice": [0.70, 0.70], "cpu_ms": 50, "params": 1e5, "model": {"backbone": "hybrid"}},
         "too_big": {"val_dice": [0.9, 0.9], "cpu_ms": 10, "params": 5e6, "model": {"backbone": "unet"}}}
    assert decide_backbone(c)["compact_choice"] == "fast_close"


def test_modes_from_selection():
    m = modes_from_selection({"threshold": 0.6, "fragment_min_size_at_0.5": 50, "fragment_min_size_at_threshold": 30})
    assert m == {"raw": (0.5, 0), "clean": (0.5, 50), "tuned": (0.6, 30)}


def test_target_guard(monkeypatch, tmp_path):
    from tglite import evaluate as E
    f = tmp_path / "decisions.json"
    f.write_text(json.dumps({"frozen": False}))
    monkeypatch.setenv("TGLITE_DECISIONS", str(f))
    with pytest.raises(PermissionError):
        E.evaluate([], None, tmp_path, "det", ["stare"], "cpu")


def test_augment_style_and_elastic_shapes():
    from tglite.data import augment as A
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    fov = np.ones((128, 128), np.uint8)
    dg = AttrDict.wrap(make_config("C_unet_topo_dg", 0, decisions=DEC).dg)
    for k in list(dg):
        if k.startswith("p_"):
            dg[k] = 1.0
    g = A.style(rgb, fov, rng, dg)
    assert g.shape == (128, 128) and g.dtype == np.uint8
    mx, my = A.elastic_maps(64, 64, rng, 12, 8)
    gx = np.gradient(mx, axis=1)
    assert gx.min() > 0                      # no folding


def test_evaluate_image_with_uncertainty_subsample():
    rng = np.random.default_rng(1)
    g = (rng.uniform(size=(300, 300)) < 0.1).astype(np.uint8)
    p = np.clip(g * 0.7 + rng.uniform(0, 0.4, g.shape), 0, 1).astype(np.float32)
    r = M.evaluate_image(p, g, np.ones_like(g), {"raw": (0.5, 0)}, 1.0, unc=rng.uniform(size=g.shape), n_px=5000, temperature=1.5)
    assert {"ece", "ece_ts", "unc_rho_px", "px_err@90"} <= set(r)
