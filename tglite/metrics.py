"""Evaluation metrics (numpy, per image, FOV-restricted).

Accuracy  : Dice, IoU, sensitivity, specificity, accuracy, AUC.
Topology  : clDice, Betti-0 / Betti-1 errors (connected components / loops), computed with 8-connectivity.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score


def fast_auc(score: np.ndarray, label: np.ndarray, bins: int = 4096) -> float:
    """Histogram-based ROC-AUC (error < 1/bins in score resolution, ~1e-4 in practice); O(N) instead of O(N log N)."""
    pos = np.histogram(score[label], bins=bins, range=(0.0, 1.0))[0].astype(np.float64)
    neg = np.histogram(score[~label], bins=bins, range=(0.0, 1.0))[0].astype(np.float64)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    below = np.cumsum(neg) - neg                       # negatives with a strictly lower score bin
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def segmentation_metrics(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray, thr: float = 0.5) -> dict:
    """prob in [0,1], gt/fov in {0,1}; all pixels outside the FOV are ignored."""
    m = fov.astype(bool)
    p = prob[m] >= thr
    g = gt[m] > 0
    tp, fp = float((p & g).sum()), float((p & ~g).sum())
    fn, tn = float((~p & g).sum()), float((~p & ~g).sum())
    eps = 1e-8
    out = {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "sens": (tp + eps) / (tp + fn + eps),
        "spec": (tn + eps) / (tn + fp + eps),
        "acc": (tp + tn) / max(tp + tn + fp + fn, 1.0),
    }
    if 0 < g.sum() < g.size:
        out["auc"] = fast_auc(prob[m], g) if g.size > 2_000_000 else float(roc_auc_score(g, prob[m]))   # exact sklearn AUC for normal sizes
    else:
        out["auc"] = float("nan")
    return out


def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """clDice (Shit et al., 2021): harmonic mean of topology precision and sensitivity on binary masks."""
    p, g = pred > 0, gt > 0
    sp, sg = skeletonize(p), skeletonize(g)
    eps = 1e-8
    tprec = ((sp & g).sum() + eps) / (sp.sum() + eps)
    tsens = ((sg & p).sum() + eps) / (sg.sum() + eps)
    return float(2 * tprec * tsens / (tprec + tsens))


def cbdice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """cbDice (cl-MIN-Dice, Shi et al. MICCAI 2024, 2-D) on binary masks: clDice with radius-aware weights.

    Tprec = sum_{S_P} D_L/r_P / (|S_P minus S_L| + sum_{S_P and S_L} r_L/r_P), Tsens symmetric; D = Euclidean distance to the
    background, r = D on the skeleton, D clipped at R_max (largest skeleton radius).  Boundary-aware: a predicted centreline that is displaced from
    the true one (but still inside the vessel) scores lower than under clDice, which stays ~1 (paper, Fig. 3a).  Our re-implementation from the paper's equations.
    """
    p, g = pred > 0, gt > 0
    if not p.any() or not g.any():
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    dp, dg = ndi.distance_transform_edt(p), ndi.distance_transform_edt(g)
    rmax = max(dp[sp].max(), dg[sg].max(), 1.0)
    dpc, dgc = np.minimum(dp, rmax), np.minimum(dg, rmax)
    rp, rl = np.maximum(dp, 1.0), np.maximum(dg, 1.0)           # only used where the matching skeleton is set (D >= 1 there)
    eps = 1e-8
    prec_den = (sp & ~sg).sum() + (rl[sp & sg] / rp[sp & sg]).sum()
    sens_den = (sg & ~sp).sum() + (rp[sg & sp] / rl[sg & sp]).sum()
    tprec = ((dgc[sp] / rp[sp]).sum() + eps) / (prec_den + eps)
    tsens = ((dpc[sg] / rl[sg]).sum() + eps) / (sens_den + eps)
    return float(2 * tprec * tsens / (tprec + tsens))


def betti_numbers(mask: np.ndarray) -> tuple[int, int]:
    """(beta0, beta1) of a binary 2-D mask: components (8-conn.) and independent loops via the Euler characteristic."""
    m = mask > 0
    b0 = int(label(m, connectivity=2).max())
    chi = int(euler_number(m, connectivity=2))
    return b0, b0 - chi


def topology_metrics(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> dict:
    """clDice and Betti errors of binary masks restricted to the FOV."""
    f = fov > 0
    p, g = (pred > 0) & f, (gt > 0) & f
    p0, p1 = betti_numbers(p)
    g0, g1 = betti_numbers(g)
    return {
        "cldice": cldice_score(p, g),
        "cbdice": cbdice_score(p, g),
        "betti0_err": abs(p0 - g0),
        "betti1_err": abs(p1 - g1),
        "betti0_pred": p0, "betti0_gt": g0, "betti1_pred": p1, "betti1_gt": g1,
    }


def all_metrics(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray, thr: float = 0.5) -> dict:
    out = segmentation_metrics(prob, gt, fov, thr)
    out.update(topology_metrics(prob >= thr, gt, fov))
    return out
