"""Figures for the README: how one image becomes a network input (and what the DG augmentation does to it).

    python scripts/make_readme_figures.py          # -> docs/img/preprocessing.png (needs the prepared data/processed/)

The report figures (``report/figures``) are produced by ``scripts/make_report_assets.py``; this script only adds the
pipeline illustration that the README uses.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

from tglite.config import ROOT, load_config
from tglite.data import load_sample
from tglite.data.augment import local_intensity_dropout, photometric
from tglite.data.datasets import split_ids
from tglite.data.preprocess import clahe_standardise

DATASETS = [("drive", "DRIVE (source)"), ("stare", "STARE"), ("chase_db1", "CHASE_DB1"), ("hrf", "HRF")]
COLUMNS = ["raw image", "green channel\n(FOV width 527 px)", "CLAHE + standardise\n(network input)", "label",
           "exact skeleton\n(topology target)", "DG view: style +\nlocal dropout"]


def main() -> None:
    cfg = load_config(["configs/base.yaml"])
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(len(DATASETS), len(COLUMNS), figsize=(9.0, 1.55 * len(DATASETS) + 0.5), gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for r, (ds, title) in enumerate(DATASETS):
        _, ids = split_ids(ds, cfg)
        raw = load_sample(ds, ids[len(ids) // 2], cfg, "none")                 # native resolution, before scale normalisation
        s = load_sample(ds, ids[len(ids) // 2], cfg, "fov")                    # what the network sees: FOV rescaled to 527 px
        fov, mask = s["fov"] > 0, s["mask"] > 0
        green = s["img"][..., 1]
        x = clahe_standardise(green, fov, float(cfg.data.clahe_clip), int(cfg.data.clahe_tile_px))
        skel = skeletonize(mask)
        skel_vis = cv2.dilate(skel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        aug = clahe_standardise(photometric(green, fov.astype(np.uint8), rng, cfg.dg), fov, float(cfg.data.clahe_clip), int(cfg.data.clahe_tile_px))
        aug = local_intensity_dropout(aug, fov, rng)
        panels = [(raw["img"], None), (np.where(fov, green, 0), "gray"), (x, "gray"), (mask, "gray"),
                  (np.where(skel_vis, 1.0, np.where(mask, 0.25, 0.0)), "gray"), (aug, "gray")]
        for c, (im, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, cmap=cmap, **({"vmin": -2.5, "vmax": 2.5} if c in (2, 5) else {}))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(COLUMNS[c], fontsize=7, pad=3)
        axes[r, 0].set_ylabel(title, fontsize=7)
    out = ROOT / "docs" / "img" / "preprocessing.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
