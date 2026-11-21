"""Figures used by README.md (all values read from results/, nothing typed by hand).

    python scripts/make_readme_figs.py   # -> docs/img/fig_headline.png, fig_size.png, fig_method.png, fig_scale_mech.png, fig_qualitative.png, fig_confusion.png, fig_uncertainty.png
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
R21 = ROOT / "results" / "v21" / "tables"
INK, RED, SAND, MUTE = "#101512", "#e4432c", "#d9d2c3", "#6b6f69"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white"})


def headline() -> None:
    """Dice per unseen dataset: 31M U-Net, 486K compact, 66K, and 66K with calibre-guided scale selection."""
    pr = pd.read_csv(R21 / "per_run.csv")
    ds = ["stare", "chase_db1", "hrf", "fives"]
    names = ["STARE", "CHASE_DB1", "HRF", "FIVES"]
    spec = [("C_unet_topo_dg[det]", "tuned", "31M U-Net (stage C)", SAND), ("E_tglitenet[det]", "tuned", "486K compact (v2)", "#9aa39a"),
            ("H_tglite70k[det]", "tuned", "66K (v2.1)", INK), ("H_tglite70k[det]", "sel", "66K + scale selection", RED)]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    w = 0.2
    for i, (key, mode, lab, col) in enumerate(spec):
        sub = pr[pr["model"] == key]
        m = [sub[f"{d}/{mode}/dice"].mean() for d in ds]
        s = [sub[f"{d}/{mode}/dice"].std(ddof=1) for d in ds]
        x = np.arange(4) + (i - 1.5) * w
        ax.bar(x, m, w * 0.92, yerr=s, color=col, label=lab, error_kw={"lw": 0.9, "capsize": 2, "ecolor": MUTE}, linewidth=0)
        if "scale" in lab:
            for xi, mi in zip(x, m):
                ax.text(xi, 0.667, f"{mi:.3f}", ha="center", va="bottom", fontsize=7.2, color="white", rotation=90)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.66, 0.87)
    ax.set_ylabel("Dice (tuned threshold, native resolution)")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
    ax.set_title("Trained on 15 DRIVE images, tested on four unseen datasets (mean, whiskers: sd over seeds)", fontsize=9.5, color=MUTE, pad=24)
    fig.tight_layout()
    fig.savefig(IMG / "fig_headline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def size_bars() -> None:
    """Parameter count (log) against unseen Dice, with CPU time written on each bar."""
    m = pd.read_csv(R21 / "main.csv").set_index("model")

    def kp(s: str) -> float:
        s = str(s)
        return float(s[:-1]) * (1e3 if s.endswith("K") else 1e6 if s.endswith("M") else 1)

    spec = [("H_tglite70k[det]", "66K  TG-LiteNet v2.1", RED), ("R_lite_w16[det]", "212K  depthwise (formal size-rule pick)", INK),
            ("E_tglitenet[det]", "486K  compact U-Net (v2)", INK), ("C_unet_topo_dg[det]", "31M  U-Net (stage C)", "#9aa39a")]
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    for i, (key, lab, col) in enumerate(spec[::-1]):
        r = m.loc[key]
        p = kp(r["params"])
        dice = float(str(r["unseen Dice tuned"]).split("±")[0])
        cpu = float(r["CPU ms"])
        ax.barh(i, p, color=col, height=0.62, linewidth=0)
        ax.text(p * 1.25, i, f"Dice {dice:.3f}   {cpu:.0f} ms/image", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(spec)))
    ax.set_yticklabels([s[1] for s in spec[::-1]], fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(2e4, 2e9)
    ax.set_xlabel("parameters (log scale)")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(IMG / "fig_size.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def uncertainty_maps() -> None:
    """Interpretation figure: image, error map and per-pixel uncertainty (binary entropy of the 66K probability map).

    Numbers under each row: share of the image's errors that fall inside its 10% most uncertain pixels (from the saved
    example maps), and the dataset-level pixel error before / after rejecting those pixels (per-image records, 66K model)."""
    import json

    from PIL import Image

    meta = json.loads((ROOT / "docs" / "assets" / "assets.json").read_text())["datasets"]
    pi = pd.read_csv(R21 / "per_image_all.csv.gz")
    h = pi[pi["model"] == "H_tglite70k[det]"]
    rows = [("stare", "STARE"), ("chase_db1", "CHASE_DB1"), ("hrf", "HRF"), ("fives", "FIVES")]
    ratios = []
    for d, _ in rows:
        hh_, ww_ = np.asarray(Image.open(ROOT / "docs" / "assets" / d / "gt.png")).shape[:2]
        ratios.append(hh_ / ww_)
    colw = 2.3
    fig, axes = plt.subplots(len(rows), 3, figsize=(3 * colw + 0.45, colw * sum(ratios) + 0.42 * len(rows) + 0.3),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.0, "height_ratios": [r_ + 0.18 for r_ in ratios]})
    for r, (d, nm) in enumerate(rows):
        a = ROOT / "docs" / "assets" / d
        p = np.asarray(Image.open(a / "prob_v21_H_tglite70k.png")).astype(float) / 255
        g = np.asarray(Image.open(a / "gt.png")) > 127
        f = np.asarray(Image.open(a / "fov.png")) > 127
        thr = meta[d]["maps"]["v21_H_tglite70k"]["threshold"]
        err = ((p > thr) != g) & f
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        ent = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
        top = (ent >= np.quantile(ent[f], 0.90)) & f
        share = 100 * (err & top).sum() / max(err.sum(), 1)
        s = h[h["dataset"] == d]
        e100, e90 = 100 * s["px_err@100"].mean(), 100 * s["px_err@90"].mean()
        shown = np.where(f, ent, np.nan)
        for c, (im, lab) in enumerate(((np.asarray(Image.open(a / "image.jpg").convert("RGB")), "Image"),
                                       (np.asarray(Image.open(a / "err_v21_H_tglite70k.png").convert("RGB")), "Errors\n(red: false alarm, blue: missed)"),
                                       (None, "Uncertainty\n(bright = unsure)"))):
            ax = axes[r, c]
            if im is not None:
                ax.imshow(im)
            else:
                ax.imshow(np.full(f.shape, 0.92), cmap="gray", vmin=0, vmax=1)
                ax.imshow(shown, cmap="magma", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(lab, fontsize=8, color=INK)
        axes[r, 0].set_ylabel(nm, fontsize=10, fontweight="bold", color=INK)
        axes[r, 1].set_xlabel(f"10% most uncertain pixels hold {share:.0f}% of the errors;  dataset error {e100:.1f}% → {e90:.1f}% after rejecting them",
                              fontsize=7.6, color=MUTE, labelpad=3)
    fig.savefig(IMG / "fig_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def method_diagram() -> None:
    """Render paper/overview.tex (TikZ) to a PNG when pdflatex and pdftoppm are available."""
    if not (shutil.which("pdflatex") and shutil.which("pdftoppm")):
        print("  skipped fig_method.png (pdflatex/pdftoppm not found; the committed copy is kept)")
        return
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        ov = (ROOT / "paper" / "overview.tex").as_posix()
        (t / "m.tex").write_text("\\documentclass[border=6pt]{standalone}\\usepackage{tikz}\\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}"
                                 "\\usepackage{times}\\begin{document}\\input{" + ov + "}\\end{document}\n", encoding="utf-8")
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "m.tex"], cwd=t, capture_output=True)
        subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile", "m.pdf", "m"], cwd=t, capture_output=True)
        if (t / "m.png").exists():
            shutil.copyfile(t / "m.png", IMG / "fig_method.png")


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    headline()
    size_bars()
    uncertainty_maps()
    method_diagram()
    for f in ("fig_scale_mech", "fig_qualitative", "fig_confusion"):
        src = ROOT / "paper" / "figures" / f"{f}.png"
        if src.exists():
            shutil.copyfile(src, IMG / f"{f}.png")
    print("wrote README figures in", IMG)


if __name__ == "__main__":
    main()
