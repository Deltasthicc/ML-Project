"""Monte-Carlo dropout uncertainty (Gal & Ghahramani, 2016) and its evaluation.

* ``mc_predict``    : T stochastic forward passes with dropout ON and BatchNorm in eval mode -> mean prob, variance.
* ``entropy``       : predictive entropy of a probability map (also usable for a *deterministic* model).
* ``spearman_unc_error`` : rank correlation between uncertainty and pixel error (FOV pixels, random subsample).
* ``risk_coverage`` : selective-prediction curves, at pixel level and at image level.

RQ4 asks whether uncertainty ranks high-error cases and supports selective rejection, especially on unseen domains.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr


def enable_mc_dropout(model: nn.Module) -> None:
    """eval() everywhere (BatchNorm uses running stats) but keep every Dropout layer stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_predict(model: nn.Module, x: torch.Tensor, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean probability, variance of probability) over ``samples`` dropout masks."""
    enable_mc_dropout(model)
    s1 = torch.zeros_like(x[:, :1])
    s2 = torch.zeros_like(s1)
    for _ in range(samples):
        p = torch.sigmoid(model(x).float())
        s1 += p
        s2 += p * p
    mean = s1 / samples
    var = (s2 / samples - mean * mean).clamp(min=0)
    model.eval()
    return mean, var


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def spearman_unc_error(unc: np.ndarray, prob: np.ndarray, gt: np.ndarray, fov: np.ndarray,
                       thr: float = 0.5, n_pixels: int = 200000, seed: int = 0) -> float:
    """Spearman rho between per-pixel uncertainty and the 0/1 misclassification indicator inside the FOV."""
    m = fov > 0
    u = unc[m]
    e = ((prob[m] >= thr) != (gt[m] > 0)).astype(np.float32)
    if len(u) > n_pixels:
        idx = np.random.default_rng(seed).choice(len(u), n_pixels, replace=False)
        u, e = u[idx], e[idx]
    if e.std() == 0 or u.std() == 0:
        return float("nan")
    return float(spearmanr(u, e).statistic)


def pixel_risk_coverage(unc: np.ndarray, prob: np.ndarray, gt: np.ndarray, fov: np.ndarray, thr: float = 0.5,
                        coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.5)) -> dict[float, float]:
    """Pixel-wise error rate after rejecting the most uncertain pixels so that only ``coverage`` of FOV pixels remain."""
    m = fov > 0
    u = unc[m]
    err = ((prob[m] >= thr) != (gt[m] > 0)).astype(np.float64)
    order = np.argsort(u, kind="stable")          # most certain first
    err_sorted = err[order]
    cum = np.cumsum(err_sorted)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(err_sorted))))
        out[float(c)] = float(cum[k - 1] / k)
    return out


def image_risk_coverage(img_unc: np.ndarray, img_err: np.ndarray, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> dict:
    """Image-level triage: reject the most uncertain images; risk = mean (1 - Dice) of the retained ones.

    Returns the curve plus the random-rejection and oracle references and AURC (area under the risk-coverage curve)."""
    order = np.argsort(img_unc, kind="stable")
    err_sorted = np.asarray(img_err)[order]
    oracle = np.sort(np.asarray(img_err))
    n = len(err_sorted)
    curve, oracle_curve = {}, {}
    for c in coverages:
        k = max(1, int(round(c * n)))
        curve[float(c)] = float(err_sorted[:k].mean())
        oracle_curve[float(c)] = float(oracle[:k].mean())
    ks = np.arange(1, n + 1)
    aurc = float((np.cumsum(err_sorted) / ks).mean())
    aurc_oracle = float((np.cumsum(oracle) / ks).mean())
    return {"curve": curve, "oracle": oracle_curve, "random": float(np.mean(img_err)), "aurc": aurc, "aurc_oracle": aurc_oracle}
