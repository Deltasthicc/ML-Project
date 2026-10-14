"""Evaluation of a trained run: accuracy + topology + uncertainty on every dataset, plus efficiency.

Outputs (inside the run directory)::

    per_image[_<tag>].csv   one row per (mode, dataset, image) with all metrics
    efficiency.json         params / FLOPs / latency (written once per run)
    preds/<dataset>/        probability + uncertainty PNGs for the seed given by eval.save_preds_seed

``mode`` is ``det`` (single deterministic pass, dropout off) or ``mc`` (mean of T dropout passes; only for models
trained with dropout).  Uncertainty columns are always produced: ``var`` = MC variance (mc mode only) and
``ent`` = predictive entropy of the (mean) probability map, which is also available for deterministic models and
therefore serves as the no-dropout reference for RQ4.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import EvalDataset, load_split
from .efficiency import efficiency_report
from .metrics import all_metrics
from .models import build_model
from .predict import has_dropout, predict_prob
from .uncertainty import entropy, pixel_risk_coverage, spearman_unc_error

PIXEL_COVERAGES = (0.9, 0.8, 0.5)


def load_checkpoint(cfg, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def _unc_columns(kind: str, unc: np.ndarray, prob: np.ndarray, gt: np.ndarray, fov: np.ndarray, thr: float, n_pixels: int) -> dict:
    out = {
        f"unc_mean_{kind}": float(unc[fov > 0].mean()),
        f"rho_{kind}": spearman_unc_error(unc, prob, gt, fov, thr, n_pixels),
    }
    rc = pixel_risk_coverage(unc, prob, gt, fov, thr, (1.0,) + PIXEL_COVERAGES)
    out[f"perr_{kind}_c1.0"] = rc[1.0]
    for c in PIXEL_COVERAGES:
        out[f"perr_{kind}_c{c}"] = rc[float(c)]
    return out


def compute_row(meta: dict, prob: np.ndarray, var, gt: np.ndarray, fov: np.ndarray, thr: float, n_pixels: int) -> dict:
    """All per-image metrics (top-level function so it can run in a process pool)."""
    row = dict(meta)
    row.update(all_metrics(prob, gt, fov, thr))
    row.update(_unc_columns("ent", entropy(prob), prob, gt, fov, thr, n_pixels))
    if var is not None:
        row.update(_unc_columns("var", var, prob, gt, fov, thr, n_pixels))
    return row


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def _resample(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resample a probability/variance map to the evaluation resolution (bilinear up, area down)."""
    if arr.shape == tuple(hw):
        return arr
    interp = cv2.INTER_LINEAR if hw[0] * hw[1] > arr.size else cv2.INTER_AREA
    return cv2.resize(arr, (hw[1], hw[0]), interpolation=interp)


def _predict_safe(model, x: torch.Tensor, mc_samples: int):
    """predict_prob, falling back to CPU if a very large (native-resolution) image does not fit on the GPU."""
    try:
        return predict_prob(model, x, mc_samples=mc_samples)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        dev = next(model.parameters()).device
        cpu_model = model.cpu()
        out = predict_prob(cpu_model, x.cpu(), mc_samples=mc_samples)
        model.to(dev)
        return out


def evaluate_run(cfg, run_dir, device: str | None = None, ckpt: str | Path | None = None,
                 datasets=None, scale_mode: str | None = None, tag: str = "", log=print) -> pd.DataFrame:
    run_dir = Path(run_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scale_mode = scale_mode or cfg.eval.scale_mode
    datasets = list(datasets or cfg.data.eval_datasets)
    model, _ = load_checkpoint(cfg, ckpt or run_dir / "best.pt", device)
    modes = ["det"] + (["mc"] if has_dropout(model) and int(cfg.eval.mc_samples) > 0 else [])
    save_preds = cfg.eval.save_preds_seed is not None and int(cfg.eval.save_preds_seed) == int(cfg.seed) and not tag
    thr = float(cfg.eval.threshold)

    rows, pending = [], []
    nw = int(cfg.eval.get("metric_workers") or 0)
    pool = ProcessPoolExecutor(nw) if nw > 0 else None
    for ds_name in datasets:
        samples = load_split(ds_name, cfg, scale_mode)
        if cfg.eval.get('max_images'):
            samples = samples[: int(cfg.eval.max_images)]
        ds = EvalDataset(samples, cfg)
        for i in tqdm(range(len(ds)), desc=f"{cfg.name}/s{cfg.seed}/{ds_name}", leave=False):
            item = ds[i]
            x = item["x"][None].to(device)
            gt, fov = item["y_eval"][0].numpy(), item["fov_eval"][0].numpy()      # evaluation resolution (native, capped)
            for mode in modes:
                prob_m, var_m = _predict_safe(model, x, int(cfg.eval.mc_samples) if mode == "mc" else 0)
                prob, var = _resample(prob_m, gt.shape), (_resample(var_m, gt.shape) if var_m is not None else None)
                meta = {"exp": cfg.name, "seed": int(cfg.seed), "mode": mode, "dataset": ds_name, "id": item["id"],
                        "height": gt.shape[0], "width": gt.shape[1], "scale_mode": scale_mode}
                args = (meta, prob, var, gt, fov, thr, int(cfg.eval.uncertainty_pixels))
                if pool is None:
                    rows.append(compute_row(*args))
                else:
                    pending.append(pool.submit(compute_row, *args))
                    while len(pending) > 2 * int(cfg.eval.metric_workers):       # bound memory: native HRF maps are ~30 MB each
                        rows.append(pending.pop(0).result())
                if save_preds and mode == modes[-1]:            # saved at the model's working scale (used by the qualitative figure)
                    base = run_dir / "preds" / ds_name / item["id"]
                    _save_png(Path(str(base) + "_prob.png"), prob_m)
                    _save_png(Path(str(base) + "_unc.png"), (var_m / max(var_m.max(), 1e-8)) if var_m is not None else entropy(prob_m) / np.log(2))
    rows.extend(f.result() for f in pending)
    if pool is not None:
        pool.shutdown()
    df = pd.DataFrame(rows)
    out = run_dir / (f"per_image_{tag}.csv" if tag else "per_image.csv")
    df.to_csv(out, index=False)
    log(f"  wrote {out}")
    return df


def efficiency_run(cfg, run_dir, log=print) -> dict:
    """Measure and store params / FLOPs / latency for the architecture of this run (weights are irrelevant)."""
    run_dir = Path(run_dir)
    model = build_model(cfg)
    mc = int(cfg.eval.mc_samples) if float(cfg.model.dropout) > 0 else 1
    rep = efficiency_report(model, cfg, mc_samples=mc)
    with open(run_dir / "efficiency.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    log(f"  efficiency: {rep['params_m']:.3f} M params, {rep['gflops']:.2f} GFLOPs @ {rep['input_hw']}, "
        f"cpu {rep['cpu']['median_ms']:.1f} ms")
    return rep
