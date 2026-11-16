"""Everything that is chosen from data uses DRIVE-val ONLY. Rules are fixed in protocol/PROTOCOL.md.

* ``calibrate_on_val``   per model: tuned threshold, fragment-removal size (at 0.5 and at the tuned threshold), temperature
* ``decide_budget``      training budget from the pilot learning curves
* ``decide_backbone``    TG-LiteNet backbone among the <= 2 M-parameter candidates (accuracy per CPU millisecond)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_split, to_input
from .inference import predict
from .metrics import GTTopology, binarize, overlap_metrics, topology_metrics


def _nll(p, g, T):
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)) / T
    q = 1 / (1 + np.exp(-lg))
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return float(-(g * np.log(q) + (1 - g) * np.log(1 - q)).mean())


def calibrate_on_val(models, cfg, device, mc_samples: int = 0) -> dict:
    """Pick (i) the threshold maximising mean DRIVE-val Dice (ties -> closest to 0.5), (ii) the minimum fragment size
    minimising mean DRIVE-val Betti-0 error (ties -> smallest) at 0.5 and at the tuned threshold, (iii) the temperature
    minimising DRIVE-val NLL. Sizes are in pixels at the normalised scale."""
    val = load_split("drive_val", cfg)
    probs = [predict(models, to_input(s, cfg).to(device), mc_samples=mc_samples)[0] for s in val]
    grid = [float(t) for t in cfg.eval.threshold_grid]
    dice = {t: float(np.mean([overlap_metrics(binarize(p, t, 0, s["fov"]), s["mask"], s["fov"])["dice"] for p, s in zip(probs, val)]))
            for t in grid}
    best = max(dice.values())
    thr = min((t for t in grid if dice[t] >= best - 1e-12), key=lambda t: (abs(t - 0.5), t))
    gtt = [GTTopology(s["mask"], s["fov"]) for s in val]

    def frag_rule(t):
        res = {}
        for k in cfg.eval.fragment_grid:
            res[int(k)] = float(np.mean([topology_metrics(binarize(p, t, int(k), s["fov"]), g)["betti0_err"]
                                         for p, s, g in zip(probs, val, gtt)]))
        lo = min(res.values())
        return min(k for k, v in res.items() if v <= lo + 1e-12), res

    k05, curve05 = frag_rule(0.5)
    kt, curvet = frag_rule(thr) if thr != 0.5 else (k05, curve05)
    pv = np.concatenate([p[s["fov"] > 0] for p, s in zip(probs, val)])
    gv = np.concatenate([s["mask"][s["fov"] > 0] for s in val]).astype(np.float32)
    idx = np.random.default_rng(0).choice(len(pv), min(400000, len(pv)), replace=False)
    Ts = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
    nll = [_nll(pv[idx], gv[idx], T) for T in Ts]
    T = float(Ts[int(np.argmin(nll))])
    return {"threshold": thr, "val_dice_by_threshold": dice, "fragment_min_size_at_0.5": k05,
            "fragment_min_size_at_threshold": kt, "val_betti0_by_size_at_0.5": curve05, "val_betti0_by_size_at_threshold": curvet,
            "temperature": T, "val_nll_T1": _nll(pv[idx], gv[idx], 1.0), "val_nll_T": float(min(nll)), "n_val_images": len(val)}


def modes_from_selection(sel: dict) -> dict:
    """Evaluation modes: name -> (threshold, min fragment size at the normalised scale)."""
    return {"raw": (0.5, 0), "clean": (0.5, int(sel["fragment_min_size_at_0.5"])),
            "tuned": (float(sel["threshold"]), int(sel["fragment_min_size_at_threshold"]))}


# ------------------------------------------------------------------------------------------------- budget
def decide_budget(log_files: list[Path], tol: float = 0.002, window: int = 3, step: int = 1000,
                  lo: int = 4000, hi: int = 20000) -> dict:
    """For each pilot curve: smooth DRIVE-val Dice with a centred moving average (window evaluations) and take the first
    iteration whose smoothed value is within ``tol`` of the curve's smoothed maximum. Budget = the largest such iteration
    over the pilots, rounded up to ``step``, clipped to [lo, hi]."""
    per = {}
    for f in log_files:
        df = pd.read_csv(f).dropna(subset=["val_dice"])
        sm = df["val_dice"].rolling(window, center=True, min_periods=1).mean().to_numpy()
        its = df["iter"].to_numpy()
        reach = int(its[np.argmax(sm >= sm.max() - tol)])
        per[str(Path(f).parent.parent.name)] = {"plateau_iter": reach, "max_smoothed_val_dice": float(sm.max()),
                                                "final_val_dice": float(df["val_dice"].iloc[-1])}
    raw = max(v["plateau_iter"] for v in per.values())
    budget = int(min(hi, max(lo, int(np.ceil(raw / step) * step))))
    return {"budget_iters": budget, "pilots": per,
            "rule": f"first iteration where the {window}-point moving average of DRIVE-val Dice is within {tol} of its maximum; "
                    f"max over pilots, rounded up to {step}, clipped to [{lo}, {hi}]"}


# ----------------------------------------------------------------------------------------------- backbone
def decide_backbone(cands: dict, tol: float = 0.005, max_params: float = 2e6) -> dict:
    """cands: name -> {"val_dice": [per seed], "cpu_ms": float, "params": int, "model": {...}}.
    Among candidates with <= max_params parameters, keep those whose mean DRIVE-val Dice is within ``tol`` of the best
    candidate, and choose the one with the lowest CPU latency (ties -> fewer parameters)."""
    ok = {k: v for k, v in cands.items() if v["params"] <= max_params}
    means = {k: float(np.mean(v["val_dice"])) for k, v in ok.items()}
    top = max(means.values())
    near = [k for k in ok if means[k] >= top - tol]
    choice = min(near, key=lambda k: (ok[k]["cpu_ms"], ok[k]["params"]))
    table = {k: {"mean_val_dice": means.get(k), "val_dice": v["val_dice"], "cpu_ms": v["cpu_ms"], "params": v["params"],
                 "eligible": k in ok, "within_tol": k in near} for k, v in cands.items()}
    return {"compact_model": cands[choice]["model"], "compact_choice": choice, "candidates": table,
            "rule": f"params <= {max_params:.0f}; mean DRIVE-val Dice within {tol} of the best candidate; lowest CPU latency"}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
