"""Export light-weight qualitative assets (for the website / report) from the trained checkpoints in runs/.

For every dataset one test image is picked deterministically (the image with the median tuned Dice of the v2.1 ~66K model,
seed 0). For that image the script writes, at the normalised scale (FOV 540 px wide):
  image.jpg, gt.png, fov.png, prob_<model>.png (8-bit probability maps), err_<model>.png (TP black / FP red / FN blue at the
  model's DRIVE-val threshold), and for the ~66K model also the map at the scale chosen by the calibre rule (novelty 2).
plus results/v21/assets/assets.json (ids, thresholds, Dice of each map at the normalised scale) and a summary figure.
Nothing here is used for any decision; it only visualises the frozen models.

    .venv\\python.exe scripts\\export_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tglite import paths  # noqa: E402
from tglite.config import AttrDict, read_yaml  # noqa: E402
from tglite.data.dataset import load_normalised  # noqa: E402
from tglite.eval21 import choose_scale, predict_scales  # noqa: E402
from tglite.evaluate import load_model  # noqa: E402
from tglite.metrics import binarize, overlap_metrics  # noqa: E402

R = paths.runs_root()
MODELS = {  # label -> run dir (seed 0)
    "v21_H_tglite70k": R / "v21" / "H_tglite70k" / "seed0",
    "v21_R_lite_w16": R / "v21" / "R_lite_w16" / "seed0",
    "v21_Z_v2recipe": R / "v21" / "Z_v2recipe" / "seed0",
    "v2_E_tglitenet": R / "E_tglitenet" / "seed0",
    "v2_C_unet_topo_dg": R / "C_unet_topo_dg" / "seed0",
}
HEAD = "v21_H_tglite70k"
DATASETS = ("drive", "stare", "chase_db1", "hrf", "fives")


def err_map(pred, gt, fov):
    img = np.full(gt.shape + (3,), 255, np.uint8)
    img[(pred > 0) & (gt > 0)] = (0, 0, 0)
    img[(pred > 0) & (gt == 0)] = (0, 0, 220)      # BGR red
    img[(pred == 0) & (gt > 0)] = (220, 90, 0)      # BGR blue
    img[fov == 0] = (235, 235, 235)
    return img


def main():
    out = paths.ROOT / "results" / "v21" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cfgs, sels, refs = {}, {}, {}, {}
    for k, rd in MODELS.items():
        cfgs[k] = AttrDict.wrap(read_yaml(rd / "config.yaml"))
        models[k] = load_model(rd, cfgs[k], dev)
        sels[k] = json.loads((rd / "val_selection_det.json").read_text(encoding="utf-8"))
        refs[k] = json.loads((rd / "val_scale_rule_det.json").read_text(encoding="utf-8"))["ref_calibre_drive_train"]
    pi = pd.read_csv(MODELS[HEAD] / "eval21" / "per_image_det.csv")
    meta = {"note": "normalised scale (FOV 540 px wide); thresholds = each model's DRIVE-val choice; Dice here is at the "
                    "normalised scale (the reported results use native resolution)", "datasets": {}}
    for d in DATASETS:
        x = pi[pi["dataset"] == d].sort_values("tuned/dice").reset_index(drop=True)
        sid = str(x.loc[len(x) // 2, "id"])
        s = load_normalised(d, sid, cfgs[HEAD].data.fov_target_width)
        dd = out / d
        dd.mkdir(exist_ok=True)
        cv2.imwrite(str(dd / "image.jpg"), s["img"][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(dd / "gt.png"), s["mask"] * 255)
        cv2.imwrite(str(dd / "fov.png"), s["fov"] * 255)
        rec = {"id": sid, "native_hw": list(map(int, s["native_hw"])), "maps": {}}
        for k, m in models.items():
            thr = float(sels[k]["threshold"])
            scales = (1.0, 1.5, 2.0, 2.5) if k == HEAD else (1.0,)
            probs = predict_scales([m], s, cfgs[k], dev, 0, scales=scales)
            variants = {k: probs[1.0][0]}
            if k == HEAD:
                c, _ = choose_scale(probs, thr, refs[k])
                variants[f"{k}_scalesel"] = probs[c][0]
                rec["selected_scale"] = c
            for name, p in variants.items():
                b = binarize(p, thr, 0, s["fov"])
                cv2.imwrite(str(dd / f"prob_{name}.png"), np.clip(p * 255 * (s["fov"] > 0), 0, 255).astype(np.uint8))
                cv2.imwrite(str(dd / f"err_{name}.png"), err_map(b, s["mask"], s["fov"]))
                rec["maps"][name] = {"threshold": thr, "dice_normalised": round(overlap_metrics(b, s["mask"], s["fov"])["dice"], 4)}
        meta["datasets"][d] = rec
        print(d, sid, {k: v["dice_normalised"] for k, v in rec["maps"].items()})
    (out / "assets.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["image", "gt"] + [f"err_{k}" for k in (HEAD, f"{HEAD}_scalesel", "v21_R_lite_w16", "v2_E_tglitenet", "v2_C_unet_topo_dg")]
    titles = ["image", "ground truth", "~66K (v2.1)", "~66K + scale sel.", "R_lite_w16 (v2.1)", "E (v2, 486K)", "C (v2, 31M)"]
    fig, axes = plt.subplots(len(DATASETS), len(cols), figsize=(2.2 * len(cols), 2.2 * len(DATASETS)))
    for i, d in enumerate(DATASETS):
        for j, c in enumerate(cols):
            f = out / d / (f"{c}.png" if c != "image" else "image.jpg")
            im = cv2.imread(str(f), cv2.IMREAD_COLOR)[..., ::-1]
            axes[i, j].imshow(im)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=8)
            if j == 0:
                axes[i, j].set_ylabel(d, fontsize=9)
            if c.startswith("err_"):
                axes[i, j].set_xlabel(f"Dice {meta['datasets'][d]['maps'][c[4:]]['dice_normalised']:.3f}", fontsize=7)
    fig.suptitle("Median-Dice test image per dataset (black: correct vessel, red: false positive, blue: missed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(paths.ROOT / "results" / "v21" / "figures" / "qualitative_v21.png", dpi=130)
    print("assets ->", out)


if __name__ == "__main__":
    main()
