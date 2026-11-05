"""Where do the models fail?  Error analysis by vessel width, by distance from the true vessels and by image zone.

Uses the probability maps saved for seed 0 of stages A-E (``results/runs/<exp>/seed0/preds``), thresholded at the fixed 0.5, against the
ground truth at the FOV-normalised scale (the scale the model runs at, so vessel widths are comparable between datasets).  No retraining.
Because only seed 0 keeps its prediction maps, these are single-run numbers; they describe *kinds* of error and are not used for any
headline claim.

    python scripts/failure_analysis.py            # -> results/tables/failure_analysis.csv, docs/img/fig_failure.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from tglite.config import ROOT
from tglite.data import load_sample
from tglite.experiments import make_config

EXPS = [("A_unet", "A  U-Net"), ("B_unet_topo", "B  + topology"), ("C_unet_topo_dg", "C  + DG aug."), ("D_lite_topo_dg", "D  compact"),
        ("E_tglitenet", "E  TG-LiteNet")]
DATASETS = ["drive", "stare", "chase_db1", "hrf"]
WIDTH_BINS = [("thin  (<= 2 px)", 0.0, 2.0), ("medium (2-4 px)", 2.0, 4.0), ("thick (> 4 px)", 4.0, 1e9)]
ZONES = [("centre", 0.0, 0.5), ("middle", 0.5, 0.8), ("edge", 0.8, 9.0)]
NEAR_PX = 2.0          # a false positive within 2 px of a true vessel is "boundary fuzz"; further away it is a spurious vessel


def vessel_width_map(gt: np.ndarray) -> np.ndarray:
    """Local vessel width (px) at every vessel pixel: 2 x distance-to-background at the nearest skeleton pixel."""
    dist = ndi.distance_transform_edt(gt)
    skel = skeletonize(gt)
    if not skel.any():
        return np.zeros(gt.shape, np.float32)
    _, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    return (2.0 * dist[iy, ix] * gt).astype(np.float32)


def zone_map(fov: np.ndarray) -> np.ndarray:
    """Distance from the FOV centre in units of the FOV radius."""
    ys, xs = np.nonzero(fov)
    cy, cx, r = ys.mean(), xs.mean(), np.sqrt(fov.sum() / np.pi)
    yy, xx = np.indices(fov.shape)
    return np.hypot(yy - cy, xx - cx) / r


def analyse_image(prob: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> list[dict]:
    pred = (prob >= 0.5) & fov
    gt = gt & fov
    width, zones = vessel_width_map(gt), zone_map(fov)
    d_gt = ndi.distance_transform_edt(~gt)
    rows = []
    for name, lo, hi in WIDTH_BINS:                                              # recall by vessel width
        m = gt & (width > lo) & (width <= hi)
        if m.sum() >= 50:
            rows.append({"kind": "recall_by_width", "group": name, "value": float((pred & m).sum() / m.sum())})
    for name, lo, hi in ZONES:                                                   # recall and false-positive rate by zone
        z = fov & (zones >= lo) & (zones < hi)
        g = gt & z
        if g.sum() >= 50:
            rows.append({"kind": "recall_by_zone", "group": name, "value": float((pred & g).sum() / g.sum())})
        if z.sum() >= 500:
            rows.append({"kind": "fp_rate_by_zone", "group": name, "value": float((pred & ~gt & z).sum() / z.sum())})
    fp = pred & ~gt
    if fp.sum() > 0:                                                             # what are the false positives?
        rows.append({"kind": "fp_share_far", "group": "spurious (> 2 px from any vessel)", "value": float((fp & (d_gt > NEAR_PX)).sum() / fp.sum())})
    lab, n = ndi.label(pred, structure=np.ones((3, 3)))                         # stray pieces: predicted components touching no true vessel
    if n:
        touch = np.unique(lab[gt & pred])
        stray = np.setdiff1d(np.arange(1, n + 1), touch)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        rows.append({"kind": "stray_pieces_per_image", "group": "pieces touching no true vessel", "value": float(len(stray))})
        rows.append({"kind": "stray_pixels_share", "group": "of all predicted vessel pixels", "value": float(sizes[stray].sum() / max(1, pred.sum()))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = make_config("A_unet")
    runs = ROOT / a.runs
    rec = []
    for exp, _ in EXPS:
        pdir = runs / exp / f"seed{a.seed}" / "preds"
        if not pdir.exists():
            print(f"skip {exp}: no {pdir}")
            continue
        for ds in DATASETS:
            ids = sorted(p.name[:-9] for p in (pdir / ds).glob("*_prob.png"))
            cache: dict = {}
            for sid in ids:
                prob = cv2.imread(str(pdir / ds / f"{sid}_prob.png"), cv2.IMREAD_GRAYSCALE)
                if prob is None:
                    continue
                s = load_sample(ds, sid, cfg, cfg.eval.scale_mode)
                for r in analyse_image(prob.astype(np.float32) / 255.0, s["mask"] > 0, s["fov"] > 0):
                    rec.append({"exp": exp, "dataset": ds, "image": sid, **r})
            print(f"{exp:16s} {ds:10s} {len(ids)} images")
    df = pd.DataFrame(rec)
    per_ds = df.groupby(["exp", "dataset", "kind", "group"], as_index=False).value.mean()
    unseen = per_ds[per_ds.dataset != "drive"].groupby(["exp", "kind", "group"], as_index=False).value.mean().assign(dataset="unseen")
    out = pd.concat([per_ds, unseen], ignore_index=True)
    path = ROOT / "results" / "tables" / "failure_analysis.csv"
    out.to_csv(path, index=False)
    print("wrote", path)
    figure(out)


def figure(t: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_report_assets import COL, _style, _title

    _style()
    col = {"A_unet": COL["A"], "B_unet_topo": COL["B"], "C_unet_topo_dg": COL["C"], "D_lite_topo_dg": COL["D"], "E_tglitenet": COL["E (det.)"]}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
    panels = [("recall_by_width", [g for g, *_ in WIDTH_BINS], "(a) Vessels found, by width", "share of vessel pixels found"),
              ("recall_by_zone", [g for g, *_ in ZONES], "(b) Vessels found, by image zone", None),
              ("fp_rate_by_zone", [g for g, *_ in ZONES], "(c) False alarms, by image zone", "false positives (% of pixels)")]
    for ax, (kind, groups, title, ylab) in zip(axes, panels):
        w = 0.8 / len(EXPS)
        for i, (exp, label) in enumerate(EXPS):
            v = []
            for g in groups:
                r = t[(t.exp == exp) & (t.dataset == "unseen") & (t.kind == kind) & (t.group == g)].value
                v.append(float(r.iloc[0]) if len(r) else np.nan)
            v = np.array(v) * (100 if kind == "fp_rate_by_zone" else 1)
            ax.bar(np.arange(len(groups)) + (i - 2) * w, v, w * 0.92, color=col[exp], linewidth=0, label=label)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace("  (", "\n(").replace(" (", "\n(") for g in groups], fontsize=7)
        if ylab:
            ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        _title(ax, title)
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    out = ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_failure.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("wrote fig_failure.png")


if __name__ == "__main__":
    main()
