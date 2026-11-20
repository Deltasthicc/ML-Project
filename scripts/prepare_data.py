"""Archives in data/ -> data/processed/ (+ protocol/drive_split.json). See data/README.md for the expected files.

    python scripts/prepare_data.py                      # all five datasets
    python scripts/prepare_data.py --only drive stare
    python scripts/prepare_data.py --overview           # also write results/data/overview.png (labels + FOV outlines)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def overview(out: Path) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from tglite.data import load_native, split_ids
    from tglite.data.prepare import DATASETS

    fig, axes = plt.subplots(2, len(DATASETS), figsize=(3.2 * len(DATASETS), 6.4))
    for j, ds in enumerate(DATASETS):
        ids = split_ids("drive" if ds == "drive" else ds)
        for i, sid in enumerate(ids[:2]):
            s = load_native(ds, sid)
            img = s["img"].copy()
            img[s["mask"] > 0] = (0.4 * img[s["mask"] > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            cnts, _ = cv2.findContours(s["fov"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, (255, 0, 0), max(2, img.shape[1] // 300))
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{ds} {sid}", fontsize=8)
            axes[i, j].axis("off")
    fig.suptitle("Labels (green) and field-of-view outlines (red)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main() -> None:
    from tglite.config import AttrDict, read_yaml
    from tglite.data.prepare import DATASETS, prepare

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(DATASETS))
    ap.add_argument("--overview", action="store_true")
    a = ap.parse_args()
    cfg = AttrDict.wrap(read_yaml(ROOT / "configs" / "base.yaml"))
    prepare(cfg, a.only)
    if a.overview:
        overview(ROOT / "results" / "data" / "overview.png")


if __name__ == "__main__":
    main()
