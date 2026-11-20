"""Evaluation of a trained model (or an ensemble) on the in-domain test set and the unseen targets.

Guard: target datasets (STARE, CHASE_DB1, HRF, FIVES) can only be evaluated once protocol/decisions.json says
``frozen: true`` - i.e. after every DRIVE-val-only decision has been made and recorded.

Outputs in the run directory:
    val_selection.json                 threshold / fragment size / temperature chosen on DRIVE-val
    eval/per_image_<variant>.csv       one row per (dataset, image); variant = det | mc | ens
    preds/<variant>/<dataset>/<id>_prob.png, _unc.png    (only for seeds in eval.save_prob_maps)
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .config import is_frozen, load_decisions
from .data import load_native, load_normalised, split_ids, to_input
from .data.dataset import TARGETS
from .inference import entropy, predict, to_native
from .metrics import evaluate_image
from .models import build_model, has_dropout
from .selection import calibrate_on_val, modes_from_selection, save_json


def load_model(run_dir: Path, cfg, device):
    model = build_model(cfg).to(device)
    st = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    model.eval()
    return model


def _init_worker():
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def _metric_task(dataset, sid, prob16, unc16, modes, scale, n_px, bins, temperature):
    nat = load_native(dataset, sid)
    prob = prob16.astype(np.float32)
    unc = None if unc16 is None else unc16.astype(np.float32)
    row = evaluate_image(prob, nat["mask"], nat["fov"], modes, scale, unc, n_px, bins, temperature)
    return {"dataset": dataset, "id": sid, "scale": scale, **row}


def evaluate(models, cfg, out_dir: Path, variant: str, datasets, device, mc_samples: int = 0, save_maps: bool = False,
             progress=None, log=print) -> pd.DataFrame:
    out_dir = Path(out_dir)
    if any(d in TARGETS for d in datasets) and not is_frozen():
        raise PermissionError("target datasets can only be evaluated after the protocol is frozen (protocol/decisions.json)")
    sel_f = out_dir / f"val_selection_{variant}.json"
    if sel_f.exists():
        sel = json.loads(sel_f.read_text(encoding="utf-8"))
    else:
        sel = calibrate_on_val(models, cfg, device, mc_samples)
        sel["decisions_snapshot"] = {k: v for k, v in load_decisions().items() if k in ("budget_iters", "compact_model", "frozen")}
        save_json(sel, sel_f)
    modes = modes_from_selection(sel)
    ev = out_dir / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    csv_f = ev / f"per_image_{variant}.csv"
    old = pd.read_csv(csv_f) if csv_f.exists() else pd.DataFrame(columns=["dataset"])
    todo = [d for d in datasets if d not in set(old["dataset"])]
    if not todo:
        return old
    mi = cfg.eval.get("max_images")                     # smoke tests only

    def ids_of(d):
        ids = split_ids(d)
        return ids[: int(mi)] if mi else ids

    n_img = sum(len(ids_of(d)) for d in todo)
    done_img = 0
    rows = []
    workers = int(cfg.eval.metric_workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        for d in todo:
            futs = []
            for sid in ids_of(d):
                s = load_normalised(d, sid, cfg.data.fov_target_width)
                mean, var = predict(models, to_input(s, cfg).to(device), mc_samples=mc_samples, amp=bool(cfg.train.amp))
                unc = entropy(mean)
                pn, un = to_native(mean, s["native_hw"]), to_native(unc, s["native_hw"])
                if save_maps:
                    pd_ = out_dir / "preds" / variant / d
                    pd_.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(pd_ / f"{sid}_prob.png"), (np.clip(mean, 0, 1) * 255).astype(np.uint8))
                    cv2.imwrite(str(pd_ / f"{sid}_unc.png"), (np.clip(unc, 0, 1) * 255).astype(np.uint8))
                futs.append(pool.submit(_metric_task, d, sid, pn.astype(np.float16), un.astype(np.float16), modes,
                                        s["scale"], int(cfg.eval.uncertainty_pixels), int(cfg.eval.ece_bins), sel["temperature"]))
                while len(futs) > 2 * workers:          # back-pressure: bound the number of native maps held in memory
                    rows.append(futs.pop(0).result())
                    done_img += 1
                    if progress:
                        progress(done_img, n_img, {"dataset": d})
            for f in futs:
                rows.append(f.result())
                done_img += 1
                if progress:
                    progress(done_img, n_img, {"dataset": d})
            # write after every dataset so an interrupted evaluation resumes at dataset granularity
            df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
            df.to_csv(csv_f, index=False)
    log(f"  evaluated {variant}: {', '.join(todo)}")
    return df


def evaluate_run(run_dir: Path, cfg, datasets, device, progress=None, log=print) -> list[str]:
    """det for every model; mc in addition for models with dropout. Returns the variants evaluated."""
    model = load_model(run_dir, cfg, device)
    save = int(cfg.seed) in list(cfg.eval.save_prob_maps or [])
    evaluate([model], cfg, run_dir, "det", datasets, device, 0, save, progress, log)
    variants = ["det"]
    if has_dropout(model):
        evaluate([model], cfg, run_dir, "mc", datasets, device, int(cfg.eval.mc_samples), save, progress, log)
        variants.append("mc")
    return variants
