"""Per-image metrics (numpy), always inside the FOV and against NATIVE-resolution ground truth.

Overlap   : Dice, IoU, precision, sensitivity, specificity, accuracy, AUC
Topology  : clDice, Betti-0 / Betti-1 error (8-connectivity; beta1 = beta0 - Euler number)
Calibration: ECE (equal-width bins) on a random FOV-pixel subsample
Post-processing: ``remove_small`` deletes connected components below a size (the pre-registered fragment rule)
"""
from __future__ import annotations

import numpy as np
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score

EPS = 1e-8


def fast_auc(score: np.ndarray, lab: np.ndarray, bins: int = 4096) -> float:
    pos = np.histogram(score[lab], bins=bins, range=(0.0, 1.0))[0].astype(np.float64)
    neg = np.histogram(score[~lab], bins=bins, range=(0.0, 1.0))[0].astype(np.float64)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def overlap_metrics(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> dict:
    m = fov > 0
    p, g = pred[m] > 0, gt[m] > 0
    tp, fp, fn = float((p & g).sum()), float((p & ~g).sum()), float((~p & g).sum())
    tn = float(m.sum()) - tp - fp - fn
    return {"dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS), "iou": (tp + EPS) / (tp + fp + fn + EPS),
            "prec": (tp + EPS) / (tp + fp + EPS), "sens": (tp + EPS) / (tp + fn + EPS), "spec": (tn + EPS) / (tn + fp + EPS),
            "acc": (tp + tn) / max(m.sum(), 1), "fp_frac": fp / max(m.sum(), 1), "fn_frac": fn / max(m.sum(), 1)}


def auc(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> float:
    m = fov > 0
    g = gt[m] > 0
    if g.all() or not g.any():
        return float("nan")
    return fast_auc(prob[m], g) if g.size > 2_000_000 else float(roc_auc_score(g, prob[m]))


def betti(mask: np.ndarray) -> tuple[int, int]:
    m = mask > 0
    b0 = int(label(m, connectivity=2).max())
    return b0, b0 - int(euler_number(m, connectivity=2))


class GTTopology:
    """Ground-truth quantities reused across thresholds / post-processing modes of one image."""

    def __init__(self, gt: np.ndarray, fov: np.ndarray):
        self.fov = fov > 0
        self.g = (gt > 0) & self.fov
        self.skel = skeletonize(self.g)
        self.b0, self.b1 = betti(self.g)


def topology_metrics(pred: np.ndarray, gtt: GTTopology) -> dict:
    p = (pred > 0) & gtt.fov
    sp = skeletonize(p)
    tprec = ((sp & gtt.g).sum() + EPS) / (sp.sum() + EPS)
    tsens = ((gtt.skel & p).sum() + EPS) / (gtt.skel.sum() + EPS)
    b0, b1 = betti(p)
    return {"cldice": float(2 * tprec * tsens / (tprec + tsens)), "betti0_err": abs(b0 - gtt.b0), "betti1_err": abs(b1 - gtt.b1),
            "betti0_pred": b0, "betti0_gt": gtt.b0}


def remove_small(pred: np.ndarray, min_size: int) -> np.ndarray:
    """Delete 8-connected components with fewer than ``min_size`` pixels."""
    p = pred > 0
    if min_size <= 0 or not p.any():
        return p
    lab = label(p, connectivity=2)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[lab]


def ece(prob: np.ndarray, gt: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the vessel probability (binary reliability, equal-width bins)."""
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    n = len(prob)
    tot = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            tot += sel.sum() / n * abs(prob[sel].mean() - gt[sel].mean())
    return float(tot)


def binarize(prob: np.ndarray, thr: float, min_size_native: int, fov: np.ndarray) -> np.ndarray:
    pred = (prob >= thr) & (fov > 0)
    return remove_small(pred, min_size_native) if min_size_native > 0 else pred


def evaluate_image(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray, modes: dict, scale: float,
                   unc: np.ndarray | None = None, n_px: int = 200000, ece_bins: int = 15, temperature: float | None = None,
                   seed: int = 0) -> dict:
    """All metrics for one image. ``modes``: {name: (threshold, min_size_at_normalised_scale)}. ``scale`` is the
    normalised/native scale factor; fragment sizes are converted to native pixels as size / scale**2."""
    out = {"auc": auc(prob, gt, fov)}
    gtt = GTTopology(gt, fov)
    for name, (thr, k) in modes.items():
        kn = int(round(k / (scale ** 2))) if k else 0
        pred = binarize(prob, thr, kn, fov)
        r = overlap_metrics(pred, gt, fov)
        r.update(topology_metrics(pred, gtt))
        out.update({f"{name}/{key}": v for key, v in r.items()})
    m = fov > 0
    pv, gv = prob[m], (gt[m] > 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pv), min(n_px, len(pv)), replace=False)
    out["ece"] = ece(pv[idx], gv[idx], ece_bins)
    if temperature:
        lg = np.log(np.clip(pv[idx], 1e-6, 1 - 1e-6) / np.clip(1 - pv[idx], 1e-6, 1))
        out["ece_ts"] = ece(1 / (1 + np.exp(-lg / temperature)), gv[idx], ece_bins)
    if unc is not None:
        from scipy.stats import spearmanr

        uv = unc[m][idx]
        err = ((pv[idx] >= 0.5) != (gv[idx] > 0)).astype(np.float32)
        out["unc_mean"] = float(unc[m].mean())
        out["unc_rho_px"] = float(spearmanr(uv, err).statistic) if err.std() > 0 and uv.std() > 0 else float("nan")
        # pixel risk-coverage: error rate of the kept pixels when the most uncertain are removed
        order = np.argsort(-uv)
        for cov in (1.0, 0.95, 0.9, 0.8, 0.7):
            keep = order[int(round((1 - cov) * len(order))):]
            out[f"px_err@{int(cov * 100)}"] = float(err[keep].mean())
    return out
