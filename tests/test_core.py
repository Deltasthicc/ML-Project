"""Unit tests for the pieces whose silent failure would invalidate results (run: ``pytest -q``)."""
import numpy as np
import pytest
import torch

from tglite.config import deep_update, load_config, parse_override
from tglite.data.preprocess import estimate_fov, fov_width, green_clahe, pad_to_multiple, scale_to_fov
from tglite.efficiency import count_flops
from tglite.experiments import GROUPS, REGISTRY, make_config
from tglite.losses import (TGLoss, bce_loss, dice_loss, skeleton_recall_loss, soft_cldice_loss, soft_skeleton)
from tglite.metrics import all_metrics, betti_numbers, cldice_score, segmentation_metrics
from tglite.models import LiteNet, UNet, build_model, count_params
from tglite.uncertainty import image_risk_coverage, mc_predict, pixel_risk_coverage


# ------------------------------------------------------------------------------------------------ models
def test_param_budget():
    assert 30.5e6 < count_params(UNet()) < 31.5e6               # classic U-Net
    assert count_params(LiteNet(width=16)) < 0.5e6
    assert count_params(LiteNet(width=48)) < 2.0e6              # the <= 2 M budget of the proposal


@pytest.mark.parametrize("cls", [UNet, LiteNet])
def test_shapes_and_heads(cls):
    m = cls(skeleton_head=True) if cls is LiteNet else cls(base_ch=8, skeleton_head=True)
    x = torch.randn(2, 1, 64, 80)
    assert m(x).shape == (2, 1, 64, 80)
    mask, skel = m(x, with_skeleton=True)
    assert mask.shape == skel.shape == (2, 1, 64, 80)
    plain = cls() if cls is LiteNet else cls(base_ch=8)
    assert not isinstance(plain(x, with_skeleton=True), tuple)     # no head -> single tensor


def test_padding_roundtrip():
    x = torch.randn(1, 1, 37, 53)
    xp, (h, w) = pad_to_multiple(x, 16)
    assert xp.shape[-2:] == (48, 64) and (h, w) == (37, 53)
    assert torch.equal(xp[..., :h, :w], x)


# ------------------------------------------------------------------------------------------------ losses
def _line(h=64, w=64, thick=5):
    t = torch.zeros(1, 1, h, w)
    t[..., h // 2 - thick // 2: h // 2 + thick // 2 + 1, 4:-4] = 1
    return t


def test_soft_skeleton_is_thin_and_inside():
    t = _line()
    sk = soft_skeleton(t, 10)
    assert sk.sum() < 0.5 * t.sum()
    assert (sk * (1 - t)).sum() < 1e-3                            # skeleton lies inside the object


def test_losses_zero_for_perfect_prediction_and_positive_otherwise():
    t, fov = _line(), torch.ones(1, 1, 64, 64)
    assert dice_loss(t, t, fov) < 0.02
    assert soft_cldice_loss(t, t, fov, None, 10) < 0.1
    from skimage.morphology import skeletonize
    sk = torch.from_numpy(skeletonize(t[0, 0].numpy() > 0).astype(np.float32))[None, None]
    assert skeleton_recall_loss(t, fov, sk) < 0.05
    bad = torch.zeros_like(t)
    assert dice_loss(bad, t, fov) > 0.9
    assert soft_cldice_loss(bad, t, fov, sk, 10) > 0.9
    assert skeleton_recall_loss(bad, fov, sk) > 0.9
    assert bce_loss(torch.full_like(t, -10.0), bad, fov) < 1e-3


def test_cldice_penalises_broken_vessel_more_than_dice():
    t, fov = _line(), torch.ones(1, 1, 64, 64)
    broken = t.clone()
    broken[..., :, 30:34] = 0                                      # a 4-px gap in a 56-px long vessel
    d = dice_loss(broken, t, fov).item()
    c = soft_cldice_loss(broken, t, fov, None, 10).item()
    assert c > d


def test_tgloss_terms_follow_config():
    cfg = make_config("C_unet_topo_dg")
    crit = TGLoss(cfg)
    t, fov = _line(), torch.ones(1, 1, 64, 64)
    out = (torch.randn(1, 1, 64, 64, requires_grad=True), torch.randn(1, 1, 64, 64, requires_grad=True))
    terms = crit.supervised(out, t, fov, t)
    assert {"dice", "bce", "cldice", "skel_head"} <= set(terms)
    sum(terms.values()).backward()
    assert out[0].grad is not None and out[1].grad is not None
    plain = TGLoss(make_config("A_unet")).supervised(torch.randn(1, 1, 64, 64), t, fov, t)
    assert set(plain) == {"dice", "bce"}


# ----------------------------------------------------------------------------------------------- metrics
def test_betti_numbers():
    ring = np.zeros((40, 40), np.uint8)
    ring[5:35, 5:35] = 1
    ring[12:28, 12:28] = 0
    assert betti_numbers(ring) == (1, 1)
    two = np.zeros((40, 40), np.uint8)
    two[2:8, 2:8] = 1
    two[20:30, 20:30] = 1
    assert betti_numbers(two) == (2, 0)
    diag = np.eye(10, dtype=np.uint8)                              # 8-connectivity: a diagonal line is one component
    assert betti_numbers(diag) == (1, 0)


def test_metrics_perfect_and_imperfect():
    gt = np.zeros((50, 50), np.uint8)
    gt[10:40, 24:27] = 1
    fov = np.ones_like(gt)
    m = all_metrics(gt.astype(np.float32), gt, fov)
    assert m["dice"] > 0.999 and m["cldice"] > 0.999 and m["betti0_err"] == 0 and m["betti1_err"] == 0
    cut = gt.copy()
    cut[24:26] = 0
    m2 = all_metrics(cut.astype(np.float32), gt, fov)
    assert m2["betti0_err"] == 1 and m2["dice"] < 1 and m2["cldice"] < 1


def test_fov_restriction():
    gt = np.zeros((20, 20), np.uint8)
    gt[:, :10] = 1
    prob = np.zeros((20, 20), np.float32)                          # predicts nothing
    fov = np.zeros_like(gt)
    fov[:, 10:] = 1                                                # FOV contains no vessel pixels
    m = segmentation_metrics(prob, gt, fov)
    assert m["acc"] == 1.0                                         # the wrong half is outside the FOV


# ------------------------------------------------------------------------------------------ preprocessing
def _synthetic_fundus(h=300, w=340):
    yy, xx = np.mgrid[:h, :w]
    inside = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.45 * w) ** 2
    img = np.zeros((h, w, 3), np.uint8)
    img[inside] = (180, 90, 40)
    img[h // 2 - 2: h // 2 + 2, 60:280][inside[h // 2 - 2: h // 2 + 2, 60:280]] = (150, 40, 20)   # dark "vessel"
    return img, inside.astype(np.uint8)


def test_fov_estimation_and_scale_normalisation():
    img, true_fov = _synthetic_fundus()
    est = estimate_fov(img)
    assert (est & true_fov).sum() / true_fov.sum() > 0.9 and est.sum() <= true_fov.sum()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[148:152, 60:280] = 1
    for target in (170, 700):
        i2, m2, f2 = scale_to_fov(img, mask, est, target)
        assert abs(fov_width(f2) - target) / target < 0.03
        assert i2.shape[:2] == m2.shape == f2.shape
        assert m2.sum() > 0                                        # thin structures survive down-scaling


def test_green_clahe_is_standardised_and_zero_outside():
    img, fov = _synthetic_fundus()
    x = green_clahe(img, fov)
    assert x.dtype == np.float32 and np.all(x[fov == 0] == 0)
    assert abs(x[fov > 0].mean()) < 1e-3 and abs(x[fov > 0].std() - 1) < 1e-2


# ------------------------------------------------------------------------------------------ uncertainty
def test_mc_dropout_variance():
    x = torch.randn(1, 1, 32, 32)
    drop = LiteNet(width=8, dropout=0.3)
    mean, var = mc_predict(drop, x, 8)
    assert mean.shape == var.shape == (1, 1, 32, 32) and var.max() > 0
    nodrop = LiteNet(width=8, dropout=0.0)
    _, v0 = mc_predict(nodrop, x, 4)
    assert v0.max() < 1e-8
    assert not drop.training                                       # model left in eval mode afterwards


def test_risk_coverage_with_oracle_uncertainty():
    rng = np.random.default_rng(0)
    gt = (rng.random((60, 60)) > 0.5).astype(np.uint8)
    prob = gt.astype(np.float32).copy()
    wrong = rng.random(gt.shape) < 0.2
    prob[wrong] = 1 - prob[wrong]
    fov = np.ones_like(gt)
    cov = 1.0 - float(wrong.mean())
    rc = pixel_risk_coverage(wrong.astype(np.float32), prob, gt, fov, coverages=(1.0, cov))
    assert rc[cov] < 1e-9 < rc[1.0]                                # rejecting exactly the wrong pixels removes all error
    img = image_risk_coverage(np.array([0.1, 0.5, 0.9, 0.3]), np.array([0.1, 0.5, 0.9, 0.3]))
    assert abs(img["aurc"] - img["aurc_oracle"]) < 1e-9


# -------------------------------------------------------------------------------------- config / registry
def test_config_overrides_and_registry():
    assert parse_override("a.b=3") == {"a": {"b": 3}}
    assert deep_update({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}
    cfg = load_config(["configs/base.yaml"], ["train.iters=7", "data.eval_datasets=[drive,stare]"])
    assert cfg.train.iters == 7 and cfg.data.eval_datasets == ["drive", "stare"]
    for names in GROUPS.values():
        for n in names:
            c = make_config(n, 1)
            assert c.name == n and c.seed == 1
            build_model(c)
    e = make_config("E_tglitenet")
    assert e.model.dropout > 0 and e.dg.enabled and e.loss.topo == "cldice" and e.model.backbone == "lite"
    a = make_config("A_unet")
    assert a.model.dropout == 0 and not a.dg.enabled and a.loss.topo == "none" and not a.model.skeleton_head
    assert len(REGISTRY) == 5 + 4 + 2 + 4


def test_flop_counter_orders_models():
    assert count_flops(LiteNet(width=8), (64, 64))["flops"] < count_flops(LiteNet(width=16), (64, 64))["flops"] \
        < count_flops(UNet(base_ch=16), (64, 64))["flops"]
