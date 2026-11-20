"""Unit tests for the v2.1 additions (CPU only)."""
import numpy as np
import torch

from tglite.config import AttrDict
from tglite.data import augment as A
from tglite.eval21 import choose_scale, reliability_scores
from tglite.losses import Objective, kd_skeleton_loss, skeleton_consistency_loss
from tglite.models import build_model, count_params


def _cfg(**model):
    return AttrDict.wrap({"model": {"width": 8, "skeleton_head": True, **model}})


def test_capped_and_wnet_sizes_and_shapes():
    cap = build_model(_cfg(backbone="unet", channels=[8, 12, 16, 24, 32]))
    wn = build_model(_cfg(backbone="wnet", channels=[4, 8, 12, 16, 24]))
    assert 60_000 < count_params(cap) < 75_000 and 60_000 < count_params(wn) < 75_000
    x = torch.randn(2, 1, 64, 64)
    wn.train()
    m, s = wn(x, with_skeleton=True)
    assert m.shape == s.shape == (2, 1, 64, 64) and wn.last_aux.shape == (2, 1, 64, 64)
    wn.eval()
    assert wn(x).shape == (2, 1, 64, 64) and wn.last_aux is None


def test_new_style_augmentations_keep_shape_and_fov():
    rng = np.random.default_rng(0)
    img = (rng.random((80, 90, 3)) * 255).astype(np.uint8)
    fov = np.zeros((80, 90), np.uint8)
    fov[10:70, 10:80] = 1
    rc = A.random_conv(img, fov, rng)
    ft = A.fourier_amplitude(img, rng)
    assert rc.shape == ft.shape == img.shape and rc.dtype == ft.dtype == np.uint8
    dg = AttrDict.wrap({"p_randconv": 1.0, "p_fourier": 1.0, "p_mix": 0, "p_color": 0, "p_illum": 0, "p_haze": 0, "p_lowres": 0,
                        "p_blur": 0, "p_noise": 0, "p_jpeg": 0})
    g = A.style(img, fov, rng, dg)
    assert g.shape == (80, 90) and (g[fov == 0] == 0).all()


def test_topology_distillation_and_skeleton_consistency():
    t = torch.zeros(1, 1, 32, 32)
    t[..., 14:18, :] = 1.0
    fov = torch.ones_like(t)
    assert kd_skeleton_loss(t, t, fov).item() < 0.05
    broken = t.clone()
    broken[..., 10:22] = 0
    assert kd_skeleton_loss(broken, t, fov).item() > kd_skeleton_loss(t, t, fov).item()
    lg = torch.logit(t.clamp(0.02, 0.98))
    assert skeleton_consistency_loss(lg, lg, fov).item() < 0.05
    cfg = AttrDict.wrap({"loss": {"lambda_bce": 1, "lambda_topo": 0.5, "lambda_skel_head": 0.5, "lambda_cons": 0.5, "topo": "cldice",
                                  "skel_iters": 5, "cons_mode": "skeleton"},
                         "distill": {"enabled": True, "lambda_kd": 0.5, "lambda_kd_skel": 0.5}})
    obj = Objective(cfg)
    terms = obj.supervised(lg, t, fov, t, teacher=t, aux=lg)
    assert {"kd", "kd_skel", "aux", "topo"} <= set(terms)
    assert torch.isfinite(obj.consistency(lg, lg, fov))


def test_reliability_scores_and_scale_rule():
    fov = np.ones((100, 100), np.uint8)
    one = np.zeros((100, 100), bool)
    one[50, 10:90] = True
    frag = one.copy()
    frag[50, ::6] = False
    a, b = reliability_scores(one, fov), reliability_scores(frag, fov)
    assert b["rel_frag"] > a["rel_frag"] and b["rel_endpoints"] > a["rel_endpoints"]
    thin = np.zeros((100, 100), np.float32)
    thin[50, 10:90] = 1                                  # calibre ~ 2 px
    thick = np.zeros((100, 100), np.float32)
    thick[47:54, 10:90] = 1                              # calibre ~ 8 px
    probs = {1.0: (thin, None, thin, fov), 2.0: (thick, None, thick, fov), 1.5: (thick, None, thick, fov)}
    s, cal = choose_scale(probs, 0.5, 7.0)
    assert s == 1.5 and cal[1.0] < cal[2.0]             # closest calibre wins, ties -> smaller scale
