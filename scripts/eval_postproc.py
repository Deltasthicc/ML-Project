"""Evaluation-only controls on trained checkpoints (nothing is retrained).

Two questions an examiner can ask about the topology results:

1. *Can plain post-processing do the same job as the topology loss?*  Betti-0 error is dominated by many tiny floating
   fragments (ground truth: 3-12 components per image, predictions: 25-125).  Mode ``pp`` removes every connected component
   smaller than ``k`` px^2 (measured at the 527-px FOV scale, rescaled to each image's native FOV width).
2. *Is the fixed 0.5 threshold unfair to some models?*  Mode ``thr`` uses one threshold per trained model.

Both ``k`` and the threshold are selected on the four DRIVE-*validation* images only and then frozen for every target
dataset, so the source-only protocol is unchanged.  ``k`` = the value with the lowest validation Betti-0 error among those
whose validation Dice is within 0.002 of the unprocessed one; the threshold = the value with the best validation Dice.

    python scripts/eval_postproc.py --runs /content/drive/MyDrive/tglite_runs --seeds 3 4          # A-E, seeds 3 and 4
    python scripts/eval_postproc.py --runs results/smoke --exp Z_lite_nocons --seeds 0 --quick    # smoke test

Writes ``per_image_pp.csv`` into every run directory and ``postproc_summary.csv`` next to the runs (or --out).
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from tqdm import tqdm

from tglite.data import EvalDataset, check_data, load_split
from tglite.evaluate import _predict_safe, _resample, load_checkpoint
from tglite.experiments import GROUPS, make_config
from tglite.metrics import all_metrics

K_GRID = (0, 5, 10, 20, 50, 100, 200, 400)             # px^2 at the 527-px FOV scale
T_GRID = tuple(round(t, 2) for t in np.arange(0.30, 0.701, 0.05))
REF_FOV_WIDTH = 527.0
DICE_TOL = 0.002
COLS = ["dice", "cldice", "betti0_err", "betti1_err", "sens", "spec", "acc", "iou"]


def clean_prob(prob: np.ndarray, fov: np.ndarray, thr: float, k: int) -> np.ndarray:
    """A probability map whose ``>= thr`` binarisation is the prediction with components smaller than ``k`` px^2 removed."""
    pred = (prob >= thr) & (fov > 0)
    scale = math.sqrt(4.0 * float(fov.sum()) / math.pi) / REF_FOV_WIDTH          # native FOV width / 527
    min_size = int(round(k * scale ** 2))
    if min_size > 1:
        lab, _ = ndi.label(pred, structure=np.ones((3, 3), dtype=int))          # 8-connectivity, as in the Betti-0 metric
        keep = np.bincount(lab.ravel()) >= min_size
        keep[0] = False
        pred = keep[lab]
    return np.where(pred, np.maximum(prob, thr), np.minimum(prob, thr - 1e-6)).astype(np.float32)


def score(prob, gt, fov, thr, k) -> dict:
    m = all_metrics(clean_prob(prob, fov, thr, k) if k > 0 else prob, gt, fov, thr)
    return {c: m[c] for c in COLS}


def _row(meta, prob, gt, fov, thr, k) -> dict:
    return {**meta, **score(prob, gt, fov, thr, k), "thr": thr, "k": k}


def predict_maps(model, ds, device, desc):
    for i in tqdm(range(len(ds)), desc=desc, leave=False):
        item = ds[i]
        gt, fov = item["y_eval"][0].numpy(), item["fov_eval"][0].numpy()
        prob_m, _ = _predict_safe(model, item["x"][None].to(device), 0)
        yield item["id"], _resample(prob_m, gt.shape), gt, fov


def select_on_val(model, cfg, device):
    """(k*, threshold*) chosen on the DRIVE validation images only."""
    maps = list(predict_maps(model, EvalDataset(load_split("drive_val", cfg, cfg.eval.scale_mode), cfg), device, "val"))
    base_thr = float(cfg.eval.threshold)
    by_k = {k: pd.DataFrame([score(p, g, f, base_thr, k) for _, p, g, f in maps]).mean() for k in K_GRID}
    ok = [k for k in K_GRID if by_k[k]["dice"] >= by_k[0]["dice"] - DICE_TOL]
    k_star = min(ok, key=lambda k: (by_k[k]["betti0_err"], k))
    by_t = {t: pd.DataFrame([score(p, g, f, t, 0) for _, p, g, f in maps]).mean()["dice"] for t in T_GRID}
    t_star = max(T_GRID, key=lambda t: (by_t[t], -abs(t - base_thr)))
    return k_star, t_star


def evaluate_run(exp, seed, runs, device, quick, workers):
    run_dir = runs / exp / f"seed{seed}"
    cfg = make_config(exp, seed)
    if quick:
        cfg.eval["max_images"] = 2
        cfg.data["eval_datasets"] = ["drive", "stare"]
    check_data(cfg)
    model, _ = load_checkpoint(cfg, run_dir / "best.pt", device)
    k_star, t_star = select_on_val(model, cfg, device)
    print(f"  {exp} seed {seed}: k* = {k_star} px^2, thr* = {t_star}")
    rows, pending = [], []
    pool = ProcessPoolExecutor(workers) if workers > 0 else None
    for ds_name in cfg.data.eval_datasets:
        samples = load_split(ds_name, cfg, cfg.eval.scale_mode)
        if cfg.eval.get("max_images"):
            samples = samples[: int(cfg.eval.max_images)]
        for iid, prob, gt, fov in predict_maps(model, EvalDataset(samples, cfg), device, f"{exp}/s{seed}/{ds_name}"):
            for mode, thr, k in (("pp", float(cfg.eval.threshold), k_star), ("thr", t_star, 0)):
                meta = {"exp": exp, "seed": seed, "mode": mode, "dataset": ds_name, "id": iid}
                if pool is None:
                    rows.append(_row(meta, prob, gt, fov, thr, k))
                else:
                    pending.append(pool.submit(_row, meta, prob, gt, fov, thr, k))
                    while len(pending) > 2 * workers:                   # bound memory (native HRF maps are large)
                        rows.append(pending.pop(0).result())
    rows.extend(f.result() for f in pending)
    if pool is not None:
        pool.shutdown()
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_image_pp.csv", index=False)
    return df


def summarise(runs: Path, exps, seeds) -> pd.DataFrame:
    """Unseen-domain macro averages (STARE, CHASE_DB1, HRF; mean over seeds) for det / pp / thr."""
    frames = []
    for e in exps:
        for s in seeds:
            d, p = runs / e / f"seed{s}" / "per_image.csv", runs / e / f"seed{s}" / "per_image_pp.csv"
            if d.exists() and p.exists():
                det = pd.read_csv(d)
                det = det[det["mode"] == "det"][["exp", "seed", "mode", "dataset", "id"] + COLS].assign(thr=0.5, k=0)
                frames += [det, pd.read_csv(p)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["mode"] = df["mode"].replace({"det": "fixed 0.5"})
    out = []
    for (e, mode), g in df.groupby(["exp", "mode"]):
        per_seed = []
        for s, gs in g.groupby("seed"):
            unseen = gs[gs.dataset.isin(["stare", "chase_db1", "hrf"])].groupby("dataset")[COLS].mean().mean()
            drive = gs[gs.dataset == "drive"][COLS].mean()
            per_seed.append({**{f"unseen_{c}": unseen[c] for c in COLS}, **{f"drive_{c}": drive[c] for c in COLS}, "k": gs.k.iloc[0], "thr": gs.thr.iloc[0]})
        t = pd.DataFrame(per_seed)
        out.append({"exp": e, "mode": mode, "n_seeds": len(t), **t.mean().to_dict(), **{f"{c}_std": v for c, v in t.std(ddof=0).items()}})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/runs", help="directory holding <exp>/seed<k>/best.pt")
    ap.add_argument("--exp", nargs="*", default=GROUPS["primary"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0, help="metric worker processes (the topology metrics on 8-MP HRF images are slow)")
    ap.add_argument("--quick", action="store_true", help="2 images from DRIVE and STARE only (smoke test)")
    ap.add_argument("--out", default=None, help="folder for postproc_summary.csv (default: next to the runs)")
    a = ap.parse_args()
    runs = Path(a.runs)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for e in a.exp:
        for s in a.seeds:
            if (runs / e / f"seed{s}" / "best.pt").exists():
                evaluate_run(e, s, runs, device, a.quick, a.workers)
            else:
                print(f"  skip {e} seed {s}: no checkpoint in {runs}")
    summ = summarise(runs, a.exp, a.seeds)
    if summ.empty:
        print("nothing to summarise")
        return
    out_dir = Path(a.out) if a.out else runs.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_dir / "postproc_summary.csv", index=False)
    show = summ[["exp", "mode", "n_seeds", "k", "thr", "drive_dice", "unseen_dice", "unseen_cldice", "unseen_betti0_err", "unseen_betti1_err"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
