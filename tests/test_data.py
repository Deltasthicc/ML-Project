"""Training-sample generation: geometric alignment of image / label / views, and train-time-only augmentation."""
import numpy as np
import torch

from tglite.data.datasets import TrainDataset
from tglite.experiments import make_config
from tglite.data.preprocess import estimate_fov


def _sample(h=300, w=340, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    fov = (((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.45 * w) ** 2).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for _ in range(6):                                               # random thick polylines = fake vessels
        y0, x0 = rng.integers(60, h - 60), rng.integers(60, w - 60)
        for _ in range(40):
            y0 = int(np.clip(y0 + rng.integers(-3, 4), 0, h - 1))
            x0 = int(np.clip(x0 + rng.integers(-3, 4), 0, w - 1))
            mask[max(0, y0 - 1): y0 + 2, max(0, x0 - 1): x0 + 2] = 1
    mask *= fov
    img = np.zeros((h, w, 3), np.uint8)
    img[fov > 0] = (190, 100, 50)
    img[mask > 0] = (120, 30, 15)                                    # vessels are darker in the green channel
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    return {"id": "s", "img": img, "mask": mask, "fov": fov}


def _cfg(dg):
    cfg = make_config("C_unet_topo_dg" if dg else "A_unet")
    cfg.data.patch_size = 96
    return cfg


def test_patch_alignment_and_shapes():
    ds = TrainDataset([_sample(seed=1), _sample(seed=2)], _cfg(False), dg=False, length=40)
    corr = []
    for i in range(20):
        it = ds[i]
        assert it["x"].shape == it["y"].shape == it["fov"].shape == it["skel"].shape == (1, 96, 96)
        assert it["x"].dtype == torch.float32
        assert "x_dg" not in it
        x, y, f = it["x"][0].numpy(), it["y"][0].numpy(), it["fov"][0].numpy()
        if y.sum() > 20:
            corr.append(np.corrcoef(x[f > 0], y[f > 0])[0, 1])
        assert (it["skel"] * (1 - torch.nn.functional.max_pool2d(it["y"][None], 3, 1, 1)[0])).sum() == 0   # skeleton inside mask
    assert len(corr) > 5 and np.mean(corr) < -0.3                    # dark vessels <-> positive label, after any rotation/flip


def test_dg_views_are_aligned_but_different():
    ds = TrainDataset([_sample(seed=3)], _cfg(True), dg=True, length=40)
    diffs, corr = [], []
    for i in range(20):
        it = ds[i]
        assert it["x_dg"].shape == it["x"].shape
        f = it["fov"][0].numpy() > 0
        a, b = it["x"][0].numpy()[f], it["x_dg"][0].numpy()[f]
        diffs.append(np.abs(a - b).mean())
        if a.std() > 0 and b.std() > 0:
            corr.append(np.corrcoef(a, b)[0, 1])
    assert np.mean(diffs) > 0.02                                    # the style view really differs
    assert np.mean(corr) > 0.5                                      # ...but is pixel-aligned with the clean view


def test_baseline_has_no_dg_augmentation():
    cfg = _cfg(False)
    assert not cfg.dg.enabled
    ds = TrainDataset([_sample(seed=4)], cfg, dg=cfg.dg.enabled, length=4)
    assert "x_dg" not in ds[0]


def test_sampling_is_reproducible_per_seed():
    torch.manual_seed(5)
    a = TrainDataset([_sample()], _cfg(True), dg=True, length=4)[0]["x"]
    torch.manual_seed(5)
    b = TrainDataset([_sample()], _cfg(True), dg=True, length=4)[0]["x"]
    assert torch.equal(a, b)


def test_elastic_warp_never_folds():
    from tglite.data.augment import elastic_maps
    cfg = make_config("C_unet_topo_dg")
    rng = __import__("numpy").random.default_rng(0)
    for _ in range(20):
        mx, my = elastic_maps(256, 256, rng, cfg.dg.elastic_alpha, cfg.dg.elastic_sigma)
        dxdx, dxdy = np.gradient(mx, axis=1), np.gradient(mx, axis=0)
        dydx, dydy = np.gradient(my, axis=1), np.gradient(my, axis=0)
        assert (dxdx * dydy - dxdy * dydx).min() > 0          # positive Jacobian everywhere -> no folding


def test_eval_ground_truth_is_native_by_default():
    from tglite.data.datasets import _eval_gt
    m = np.zeros((100, 120), np.uint8)
    m[10:90, 50] = 1                                           # a 1-px line, like HRF's thin vessels
    f = np.ones_like(m)
    m2, _ = _eval_gt(m, f, None)
    assert m2.shape == m.shape and m2.sum() == m.sum()
    m3, _ = _eval_gt(m, f, 3000)                               # capped -> down-scaled, so the 1-px line is damaged
    assert m3.shape != m.shape
    assert make_config("A_unet").eval.max_eval_pixels is None


def test_fast_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    from tglite.metrics import fast_auc
    r = np.random.default_rng(1)
    y = r.random(200000) < 0.1
    p = np.clip(r.normal(0.3 + 0.3 * y, 0.2), 0, 1)
    assert abs(fast_auc(p, y) - roc_auc_score(y, p)) < 1e-3
