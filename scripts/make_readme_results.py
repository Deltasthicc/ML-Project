"""Print the results tables of the README as Markdown, straight from ``results/tables`` (so the README cannot drift from the data).

    python scripts/make_readme_results.py            # paste the output into README.md section 6
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from tglite.config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from make_report_assets import bench_latency, val  # noqa: E402

T = ROOT / "results" / "tables"
ROWS = [("A_unet", "det", "A  U-Net (Dice+BCE)"), ("B_unet_topo", "det", "B  + topology loss"), ("C_unet_topo_dg", "det", "C  + DG augmentation"),
        ("D_lite_topo_dg", "det", "D  + compact backbone"), ("E_tglitenet", "det", "E  TG-LiteNet (deterministic)"), ("E_tglitenet", "mc", "E  TG-LiteNet (MC dropout, T=20)")]
OTHER = [("X_unet_dg", "U-Net + DG, no topology loss"), ("S_unet_b8", "plain compact U-Net, base 8"), ("S_unet_b16", "plain compact U-Net, base 16"),
         ("Z_lite_noskel", "D without skeleton head"), ("Z_lite_nocons", "D without consistency loss")]


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def main() -> None:
    summ, eff = pd.read_csv(T / "summary.csv"), pd.read_csv(T / "efficiency.csv")
    lat = bench_latency(T)
    n = lambda e, m="det": int(val(summ, e, "unseen", "dice", m, "n"))  # noqa: E731

    def row(e, m, label):
        ef = eff[eff.exp == e].iloc[0]
        f = lambda ds, met, d=3: f"{val(summ, e, ds, met, m):.{d}f}"  # noqa: E731
        ms = f"{lat[e] * (20 if m == 'mc' else 1):.0f}" if e in lat else "-"
        return [label, f("drive", "dice"), f("unseen", "dice"), f("unseen", "cldice"), f("unseen", "betti0_err", 1), f"{ef.params_m:.2f}", f"{ef.gflops:.1f}", ms, str(n(e, m))]

    head = ["", "DRIVE Dice", "unseen Dice", "unseen clDice", "unseen Betti-0 err", "params (M)", "GFLOPs", "CPU ms*", "seeds"]
    print("**Staged ablation** (mean over seeds; unseen = macro average of STARE, CHASE_DB1, HRF; fixed threshold 0.5)\n")
    print(md([row(e, m, l) for e, m, l in ROWS], head))
    print("\n**Controls and compact baselines**\n")
    print(md([row(e, "det", l) for e, l in OTHER], head))
    post = pd.read_csv(T / "postproc_summary.csv")
    names = {"A_unet": "A  U-Net", "B_unet_topo": "B  + topology loss", "C_unet_topo_dg": "C  + DG augmentation", "D_lite_topo_dg": "D  + compact backbone",
             "E_tglitenet": "E  TG-LiteNet", "X_unet_dg": "U-Net + DG, no topology", "S_unet_b8": "plain compact U-Net, base 8",
             "S_unet_b16": "plain compact U-Net, base 16", "X_lite_dg": "lite + DG, no topology"}
    rows = []
    for e, label in names.items():
        g = lambda mode, col, d=3: float(post[(post.exp == e) & (post["mode"] == mode)][col].iloc[0])  # noqa: E731
        rows.append([label, str(int(post[(post.exp == e) & (post["mode"] == "pp")].n_seeds.iloc[0])), f"{g('fixed 0.5', 'unseen_dice'):.3f}", f"{g('pp', 'unseen_dice'):.3f}",
                     f"{g('thr', 'unseen_dice'):.3f}", f"{g('fixed 0.5', 'unseen_cldice'):.3f}", f"{g('pp', 'unseen_cldice'):.3f}",
                     f"{g('fixed 0.5', 'unseen_betti0_err'):.1f}", f"{g('pp', 'unseen_betti0_err'):.1f}"])
    print("\n**Evaluation-only controls** (cleanup = small components removed; tuned = threshold chosen per model on DRIVE-validation; unseen averages)\n")
    print(md(rows, ["", "seeds", "Dice fixed 0.5", "Dice cleanup", "Dice tuned thr.", "clDice fixed", "clDice cleanup", "Betti-0 fixed", "Betti-0 cleanup"]))
    print("\n\\* one 512x512 image, 4 CPU threads, controlled laptop benchmark (`results/tables/latency_laptop.csv`); the MC row is T = 20 forward passes.")


if __name__ == "__main__":
    main()
