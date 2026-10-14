"""Contact sheet of the processed datasets (image + vessel labels in green + FOV outline in red).

    python scripts/make_data_overview.py            # -> results/figures/data_overview.png

Use it to confirm labels are aligned with the images and that estimated FOV masks (DRIVE mirror, STARE, CHASE_DB1,
FIVES) hug the retina.  Also used as the dataset figure in the report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def overlay(ds: str, sid: str, max_side: int = 640) -> np.ndarray:
    img = cv2.imread(str(PROC / ds / "images" / f"{sid}.png"))[..., ::-1].copy()
    gt = cv2.imread(str(PROC / ds / "masks" / f"{sid}.png"), 0) > 127
    fov = (cv2.imread(str(PROC / ds / "fov" / f"{sid}.png"), 0) > 127).astype(np.uint8)
    s = max_side / max(img.shape[:2])
    if s < 1:
        size = (int(img.shape[1] * s), int(img.shape[0] * s))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        fov = cv2.resize(fov, size, interpolation=cv2.INTER_NEAREST)
    img[gt] = (0.4 * img[gt] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(fov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "results" / "figures" / "data_overview.png"))
    ap.add_argument("--strip", action="store_true", help="one image per dataset in a single row (compact figure for the paper)")
    args = ap.parse_args()
    sets = [d for d in ("drive", "stare", "chase_db1", "hrf", "fives") if (PROC / d / "images").exists()]
    if args.strip:
        fig, axes = plt.subplots(1, len(sets), figsize=(2.0 * len(sets), 1.9), squeeze=False)
        for c, ds in enumerate(sets):
            ids = sorted(p.stem for p in (PROC / ds / "images").glob("*.png"))
            axes[0, c].imshow(overlay(ds, ids[len(ids) // 3], 360))
            axes[0, c].set_title(ds, fontsize=8)
            axes[0, c].axis("off")
        plt.tight_layout(pad=0.3)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=170)
        print("wrote", args.out)
        return
    fig, axes = plt.subplots(len(sets), args.per_dataset, figsize=(3.4 * args.per_dataset, 3.2 * len(sets)), squeeze=False)
    for r, ds in enumerate(sets):
        ids = sorted(p.stem for p in (PROC / ds / "images").glob("*.png"))
        pick = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, args.per_dataset)]
        for c, sid in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(overlay(ds, sid))
            ax.set_title(f"{ds}: {sid}", fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
