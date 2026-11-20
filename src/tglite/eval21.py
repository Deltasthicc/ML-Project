"""v2.1 evaluation (protocol/PROTOCOL_v2.1.md). Everything of v2 plus:

* novelty 1 - label-free connectivity reliability scores of each prediction (fragment density = primary, skeleton
  end-point density and small-fragment fraction = secondary), next to the usual mean entropy (single pass / MC / ensemble)
* novelty 2 - centreline-calibre-guided test-time scale selection (protocol amendment 1): the image is also run at
  1.5x, 2x and 2.5x the normalised scale; at each scale the mean vessel calibre of the prediction is measured along its
  own centrelines (2 x distance transform on the skeleton, at the processing resolution), and the scale whose calibre is
  closest to the calibre the same model predicts on DRIVE-train is kept (mode ``sel``). Checked on DRIVE-val (no shift)
  and on DRIVE-val images shrunk by 0.67 / 0.5 / 0.4 (synthetic thin vessels); used for claims only if it does not lower
  DRIVE-val Dice by more than 0.01
* network-resolution metrics (``net/<mode>/dice|cldice``): prediction and label both at the 540-px-FOV scale, as many
  papers report, next to the native-resolution metrics (headline)

Output: <run>/eval21/per_image_<variant>.csv (resumable per dataset) and <run>/val_scale_rule_<variant>.json.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import skeletonize

from .data import load_native, load_normalised, load_split, split_ids
from .data.preprocess import network_input
from .inference import entropy, predict, to_native
from .metrics import GTTopology, binarize, evaluate_image, overlap_metrics, topology_metrics
from .selection import calibrate_on_val, modes_from_selection, save_json

SCALES = (1.0, 1.5, 2.0, 2.5)
SMALL_FRAGMENT_PX = 100          # at the normalised scale (secondary score)


# ------------------------------------------------------------------------------------------- reliability scores
def reliability_scores(pred: np.ndarray, fov: np.ndarray) -> dict:
    """Label-free connectivity statistics of a binary prediction at the normalised scale (higher = less reliable)."""
    p = (pred > 0) & (fov > 0)
    area = max(int((fov > 0).sum()), 1)
    lab = label(p, connectivity=2)
    n = int(lab.max())
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
    small = float(sizes[sizes < SMALL_FRAGMENT_PX].sum()) / max(float(p.sum()), 1.0)
    sk = skeletonize(p)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk.astype(np.uint8)
    ends = int((sk & (nb == 1)).sum())
    return {"rel_frag": n / area * 1e5, "rel_endpoints": ends / max(int(sk.sum()), 1) * 100.0, "rel_smallfrac": small}


# ------------------------------------------------------------------------------------------------ multi-scale
def _input_at(sample: dict, cfg, s: float):
    img, fov = sample["img"], sample["fov"]
    if s != 1.0:
        h, w = fov.shape
        size = (int(round(w * s)), int(round(h * s)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        fov = cv2.resize(fov, size, interpolation=cv2.INTER_NEAREST)
    import torch
    return torch.from_numpy(network_input(img, fov, cfg.data.clahe_clip, cfg.data.clahe_tile_px))[None, None]


def predict_scales(models, sample: dict, cfg, device, mc_samples: int = 0, scales=SCALES) -> dict:
    """{scale: (mean prob, variance or None, mean prob at the processing resolution, FOV at that resolution)};
    the first two are resampled to the normalised resolution of ``sample``."""
    h, w = sample["fov"].shape
    out = {}
    for s in scales:
        mean, var = predict(models, _input_at(sample, cfg, s).to(device), mc_samples=mc_samples, amp=bool(cfg.train.amp))
        fproc = cv2.resize(sample["fov"], (mean.shape[1], mean.shape[0]), interpolation=cv2.INTER_NEAREST) if s != 1.0 else sample["fov"]
        proc = mean
        if s != 1.0:
            mean = cv2.resize(mean, (w, h), interpolation=cv2.INTER_AREA)
            var = None if var is None else cv2.resize(var, (w, h), interpolation=cv2.INTER_AREA)
        out[s] = (mean, var, proc, fproc)
    return out


def calibre(prob: np.ndarray, fov: np.ndarray, thr: float) -> float:
    """Mean predicted vessel calibre in px: 2 x distance-to-background sampled on the prediction's own skeleton."""
    b = binarize(prob, thr, 0, fov)
    sk = skeletonize(b)
    return float(2 * ndi.distance_transform_edt(b)[sk].mean()) if sk.any() else 0.0


def choose_scale(probs: dict, thr: float, ref_calibre: float) -> tuple[float, dict]:
    """Pre-registered rule (amendment 1): calibre at the processing resolution closest to the model's DRIVE-train
    calibre; ties -> smaller scale."""
    cal = {s: calibre(v[2], v[3], thr) for s, v in probs.items()}
    best = min(abs(c - ref_calibre) for c in cal.values())
    return min(s for s, c in cal.items() if abs(c - ref_calibre) <= best + 1e-9), cal


def reference_calibre(models, cfg, device, thr: float, mc_samples: int = 0) -> float:
    """Mean predicted calibre of the model on the 15 DRIVE-train images (scale 1) - the label-free target of the rule."""
    return float(np.mean([calibre(predict_scales(models, s, cfg, device, mc_samples, scales=(1.0,))[1.0][2], s["fov"], thr)
                          for s in load_split("drive_train", cfg)]))


def _shrink(s: dict, f: float) -> dict:
    h, w = s["fov"].shape
    sz = (max(16, int(w * f)), max(16, int(h * f)))
    return {"img": cv2.resize(s["img"], sz, interpolation=cv2.INTER_AREA), "fov": cv2.resize(s["fov"], sz, interpolation=cv2.INTER_NEAREST),
            "mask": (cv2.resize(s["mask"].astype(np.float32), sz, interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)}


def scale_rule_on_val(models, cfg, device, sel: dict, mc_samples: int = 0) -> dict:
    """DRIVE-only checks of the scale rule (before any target is touched): (i) DRIVE-val as is - mean Dice at scale 1 vs
    with selection; (ii) DRIVE-val shrunk by 0.67 / 0.5 / 0.4 (vessels thinner, as in a camera/annotation shift) -
    Dice gain of the rule over scale 1, and of the best fixed scale (oracle)."""
    thr, k = float(sel["threshold"]), int(sel["fragment_min_size_at_threshold"])
    ref = reference_calibre(models, cfg, device, thr, mc_samples)
    val = load_split("drive_val", cfg)
    d1, ds, chosen = [], [], []
    for s in val:
        probs = predict_scales(models, s, cfg, device, mc_samples)
        c, _ = choose_scale(probs, thr, ref)
        chosen.append(c)
        d1.append(overlap_metrics(binarize(probs[1.0][0], thr, k, s["fov"]), s["mask"], s["fov"])["dice"])
        ds.append(overlap_metrics(binarize(probs[c][0], thr, k, s["fov"]), s["mask"], s["fov"])["dice"])
    shrink = {}
    for f in (0.67, 0.5, 0.4):
        g, o, ch = [], [], []
        for s in val:
            sh = _shrink(s, f)
            probs = predict_scales(models, sh, cfg, device, mc_samples)
            c, _ = choose_scale(probs, thr, ref)
            dice = {sc: overlap_metrics(binarize(v[0], thr, 0, sh["fov"]), sh["mask"], sh["fov"])["dice"] for sc, v in probs.items()}
            g.append(dice[c] - dice[1.0])
            o.append(max(dice.values()) - dice[1.0])
            ch.append(c)
        shrink[str(f)] = {"gain_rule": float(np.mean(g)), "gain_oracle": float(np.mean(o)), "chosen": ch}
    res = {"ref_calibre_drive_train": ref, "val_dice_scale1": float(np.mean(d1)), "val_dice_selected": float(np.mean(ds)),
           "chosen_scales": chosen, "shrink_check": shrink,
           "rule": "calibre on own skeleton closest to the DRIVE-train calibre; used for claims only if val Dice drops <= 0.01"}
    res["passed"] = res["val_dice_selected"] >= res["val_dice_scale1"] - 0.01
    return res


# ------------------------------------------------------------------------------------------------ evaluation
def _init_worker():
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def _native_task(dataset, sid, p1, unc, psel, pfix, modes, scale, n_px, bins, temperature):
    nat = load_native(dataset, sid)
    gt, fov = nat["mask"], nat["fov"]
    row = evaluate_image(p1.astype(np.float32), gt, fov, modes, scale, None if unc is None else unc.astype(np.float32),
                         n_px, bins, temperature)
    thr, k = modes["tuned"]
    rs = evaluate_image(psel.astype(np.float32), gt, fov, {"sel": (thr, k)}, scale, None, 1000, bins, None)
    row.update({key: v for key, v in rs.items() if key.startswith("sel/")})
    for s, p in pfix.items():
        row[f"fix{s:g}/dice"] = overlap_metrics(binarize(p.astype(np.float32), thr, 0, fov), gt, fov)["dice"]
    return {"dataset": dataset, "id": sid, "scale": scale, **row}


def evaluate21(models, cfg, out_dir: Path, variant: str, datasets, device, mc_samples: int = 0, progress=None, log=print,
               allow_targets: bool = False) -> pd.DataFrame:
    from .data.dataset import TARGETS

    out_dir = Path(out_dir)
    if any(d in TARGETS for d in datasets) and not allow_targets:
        raise PermissionError("v2.1: target datasets are locked until the v2.1 protocol is frozen")
    sel_f = out_dir / f"val_selection_{variant}.json"
    if sel_f.exists():
        sel = json.loads(sel_f.read_text(encoding="utf-8"))
    else:
        sel = calibrate_on_val(models, cfg, device, mc_samples)
        save_json(sel, sel_f)
    rule_f = out_dir / f"val_scale_rule_{variant}.json"
    if rule_f.exists():
        rule = json.loads(rule_f.read_text(encoding="utf-8"))
    else:
        rule = scale_rule_on_val(models, cfg, device, sel, mc_samples)
        save_json(rule, rule_f)
    modes = modes_from_selection(sel)
    thr, k = modes["tuned"]
    ev = out_dir / "eval21"
    ev.mkdir(parents=True, exist_ok=True)
    csv_f = ev / f"per_image_{variant}.csv"
    old = pd.read_csv(csv_f) if csv_f.exists() else pd.DataFrame(columns=["dataset"])
    todo = [d for d in datasets if d not in set(old["dataset"])]
    if not todo:
        return old
    mi = cfg.eval.get("max_images")

    def ids_of(d):
        ids = split_ids(d)
        return ids[: int(mi)] if mi else ids

    n_img, done, rows = sum(len(ids_of(d)) for d in todo), 0, []
    workers = int(cfg.eval.metric_workers)
    df = old
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for d in todo:
            futs = []
            for sid in ids_of(d):
                s = load_normalised(d, sid, cfg.data.fov_target_width)
                probs = predict_scales(models, s, cfg, device, mc_samples)
                p1, var1 = probs[1.0][:2]
                c, cal = choose_scale(probs, thr, rule["ref_calibre_drive_train"])
                net = {}
                gtt = GTTopology(s["mask"], s["fov"])
                for mname in ("raw", "tuned"):
                    t_, k_ = modes[mname]
                    pb = binarize(p1, t_, k_, s["fov"])
                    net[f"net/{mname}/dice"] = overlap_metrics(pb, s["mask"], s["fov"])["dice"]
                    net[f"net/{mname}/cldice"] = topology_metrics(pb, gtt)["cldice"]
                extra = {**net, **reliability_scores(binarize(p1, 0.5, 0, s["fov"]), s["fov"]), "sel_scale": c,
                         **{f"calib@{sc:g}": v for sc, v in cal.items()}, "ent_mean": float(entropy(p1)[s["fov"] > 0].mean()),
                         "var_mean": float(var1[s["fov"] > 0].mean()) if var1 is not None else np.nan}
                nh = s["native_hw"]
                pfix = {sc: to_native(v[0], nh).astype(np.float16) for sc, v in probs.items() if sc != 1.0}
                p1n = to_native(p1, nh).astype(np.float16)
                psel = p1n if c == 1.0 else pfix[c]
                fut = pool.submit(_native_task, d, sid, p1n, to_native(entropy(p1), nh).astype(np.float16), psel, pfix, modes,
                                  s["scale"], int(cfg.eval.uncertainty_pixels), int(cfg.eval.ece_bins), sel["temperature"])
                futs.append((fut, extra))
                while len(futs) > 2 * workers:
                    f, e = futs.pop(0)
                    rows.append({**f.result(), **e})
                    done += 1
                    if progress:
                        progress(done, n_img, {"dataset": d})
            for f, e in futs:
                rows.append({**f.result(), **e})
                done += 1
                if progress:
                    progress(done, n_img, {"dataset": d})
            df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(csv_f, index=False)
    log(f"  evaluated v2.1 {variant}: {', '.join(todo)}  (scale rule on DRIVE-val: {'passed' if rule['passed'] else 'FAILED'})")
    return df
