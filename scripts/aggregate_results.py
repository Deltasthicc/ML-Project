"""Aggregate all runs into CSV tables (results/tables/*.csv) and print a console summary.

    python scripts/aggregate_results.py                       # results/runs -> results/tables
    python scripts/aggregate_results.py --runs results/smoke --out results/smoke_tables
"""
from __future__ import annotations

import argparse

import pandas as pd

from tglite.analysis import build_all, get
from tglite.config import resolve_path
from tglite.experiments import STAGE_LABELS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    tabs = build_all(resolve_path(args.runs), resolve_path(args.out))
    if not tabs:
        raise SystemExit(f"no per_image.csv found under {args.runs}")
    s, eff = tabs["summary"], tabs.get("efficiency", pd.DataFrame())
    print(f"wrote {len(tabs)} tables to {args.out}\n")
    print(f"{'stage':26s} {'src Dice':>14s} {'unseen Dice':>14s} {'unseen clDice':>14s} {'unseen b0err':>13s} {'gap Dice':>10s}  seeds")
    modes = [("A_unet", "det"), ("B_unet_topo", "det"), ("C_unet_topo_dg", "det"), ("D_lite_topo_dg", "det"),
             ("E_tglitenet", "det"), ("E_tglitenet", "mc")]
    for exp, mode in modes:
        cells, n = [], 0
        for ds, met in (("drive", "dice"), ("unseen", "dice"), ("unseen", "cldice"), ("unseen", "betti0_err"), ("gap_unseen", "dice")):
            m, sd, n = get(s, exp, ds, met, mode)
            cells.append(f"{m:.3f}+-{sd:.3f}")
        if n:
            label = STAGE_LABELS.get(exp, exp) + ("" if mode == "det" else " (MC)")
            print(f"{label:26s} " + " ".join(f"{c:>14s}" for c in cells) + f"  {n}")
    if len(eff):
        print("\nefficiency:\n", eff[["exp", "params_m", "gflops", "cpu_ms", "cpu1_ms"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
