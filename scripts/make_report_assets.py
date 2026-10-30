"""Turn results/ into the LaTeX tables, number macros and figures used by report/main.tex.

    python scripts/aggregate_results.py            # first
    python scripts/make_report_assets.py           # -> report/generated/*.tex, report/figures/*.pdf

Every number in the paper's tables/abstract comes from here, so the report always matches the submitted code's
outputs.  Missing results are rendered as "--" (and the corresponding figure is skipped) instead of invented.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tglite.analysis import CORE_TARGETS, get
from tglite.config import ROOT, load_config
from tglite.data import load_sample
from tglite.experiments import FACTORIAL_CELLS, STAGE_LABELS

warnings.filterwarnings("ignore")
GEN = ROOT / "report" / "generated"
FIG = ROOT / "report" / "figures"
NICE = {"drive": "DRIVE", "stare": "STARE", "chase_db1": "CHASE\\_DB1", "hrf": "HRF", "fives": "FIVES", "unseen": "Unseen avg."}
NICE_PLAIN = {"drive": "DRIVE", "stare": "STARE", "chase_db1": "CHASE_DB1", "hrf": "HRF", "fives": "FIVES", "unseen": "unseen avg."}
STAGES = [("A_unet", "det", "A"), ("B_unet_topo", "det", "B"), ("C_unet_topo_dg", "det", "C"), ("D_lite_topo_dg", "det", "D"),
          ("E_tglitenet", "det", "E (det.)"), ("E_tglitenet", "mc", "E (MC)")]
COL = {"A": "#7f7f7f", "B": "#1f77b4", "C": "#2ca02c", "D": "#ff7f0e", "E (det.)": "#d62728", "E (MC)": "#9467bd"}


# ---------------------------------------------------------------------------------------------- helpers
def fmt(m: float, s: float | None = None, d: int = 3, bold: bool = False) -> str:
    if m is None or np.isnan(m):
        return "--"
    txt = f"{m:.{d}f}"
    if bold:
        txt = f"\\textbf{{{txt}}}"
    if s is not None and not np.isnan(s) and s > 0:
        txt += f"{{\\scriptsize$\\pm${s:.{d}f}}}"
    return txt


def val(summ, exp, ds, met, mode="det", what="mean"):
    m, s, n = get(summ, exp, ds, met, mode)
    return {"mean": m, "std": s, "n": n}[what]


def best_idx(values: list[float], lower_better: bool) -> int | None:
    arr = np.array([np.nan if v is None else v for v in values], float)
    if np.all(np.isnan(arr)):
        return None
    return int(np.nanargmin(arr) if lower_better else np.nanargmax(arr))


def write(name: str, text: str) -> None:
    """Write a generated .tex file; tabulars are scaled down to the column/text width if they would overflow."""
    if "\\begin{tabular}" in text:
        width = "\\textwidth" if "\\begin{table*}" in text else "\\columnwidth"
        text = text.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=" + width + "}\\begin{tabular}", 1)
        text = text.replace("\\end{tabular}", "\\end{tabular}\\end{adjustbox}", 1)
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / name).write_text(text, encoding="utf-8")
    print("  wrote", GEN / name)


def savefig(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print("  wrote", FIG / f"{name}.pdf")


# ---------------------------------------------------------------------------------------------- tables
def table_main(summ, eff) -> str:
    """Table III: accuracy / topology / generalisation / cost of the staged ablation."""
    cols = [("drive", "dice", "Dice", False), ("drive", "cldice", "clDice", False), ("unseen", "dice", "Dice", False),
            ("unseen", "cldice", "clDice", False), ("unseen", "cbdice", "cbDice", False),
            ("unseen", "betti0_err", "$\\beta_0$ err", True), ("gap_unseen", "dice", "$\\Delta$Dice", True)]
    best = []
    for ds, met, _, low in cols:
        best.append(best_idx([val(summ, e, ds, met, m) for e, m, _ in STAGES], low))
    lines = ["\\begin{table*}[t]", "\\centering", "\\caption{Staged ablation (mean$\\pm$std over seeds). Source = DRIVE test; unseen = macro average over STARE, CHASE\\_DB1 and HRF; "
             "$\\Delta$Dice = Dice(DRIVE)$-$Dice(unseen). Cost: one $512{\\times}512$ image, batch 1, CPU = 4 threads, Kaggle machine; E (MC) needs $T$ passes. Best per column in bold.}",
             "\\label{tab:main}", "\\small", "\\begin{tabular}{l cc ccccc rrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{Source (DRIVE)} & \\multicolumn{5}{c}{Unseen domains} & & & \\multicolumn{2}{c}{Latency (ms)} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-8}\\cmidrule(lr){11-12}",
             "Stage & Dice & clDice & Dice & clDice & cbDice & $\\beta_0$ err & $\\Delta$Dice & Params (M) & GFLOPs & CPU & GPU \\\\", "\\midrule"]
    for r, (exp, mode, lab) in enumerate(STAGES):
        cells = []
        for k, (ds, met, _, _) in enumerate(cols):
            m, s = val(summ, exp, ds, met, mode), val(summ, exp, ds, met, mode, "std")
            cells.append(fmt(m, s, 2 if met == "betti0_err" else 3, bold=(best[k] == r)))
        e = eff[eff.exp == exp] if len(eff) else pd.DataFrame()
        p = f"{e.params_m.iloc[0]:.2f}" if len(e) else "--"
        g = f"{e.gflops.iloc[0]:.1f}" if len(e) else "--"
        cpu = f"{e.cpu_ms.iloc[0]:.0f}" if len(e) else "--"
        gpu = f"{e.gpu_ms.iloc[0]:.1f}" if len(e) and "gpu_ms" in e and not pd.isna(e.gpu_ms.iloc[0]) else "--"
        if mode == "mc" and len(e):
            t = int(e.mc_samples.iloc[0])
            g, cpu = g + f" $\\times$ {t}", cpu + f" $\\times$ {t}"
            gpu = gpu + f" $\\times$ {t}" if gpu != "--" else gpu
        lines.append(f"{lab} & " + " & ".join(cells) + f" & {p} & {g} & {cpu} & {gpu} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def table_datasets(summ, datasets) -> str:
    """Table IV: per-dataset Dice / clDice / beta0 error."""
    lines = ["\\begin{table*}[t]", "\\centering", "\\caption{Per-dataset results (mean$\\pm$std over seeds). Betti-0 error is the absolute difference in the number of "
             "connected components (lower is better); counts are per image at native label resolution, so they grow with image size.}", "\\label{tab:datasets}", "\\small",
             "\\begin{tabular}{l " + " ".join(["ccc"] * len(datasets)) + "}", "\\toprule",
             "Stage & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{NICE[d]}}}" for d in datasets) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(datasets))),
             " & " + " & ".join(["Dice & clDice & $\\beta_0$"] * len(datasets)) + " \\\\", "\\midrule"]
    best = {}
    for d in datasets:
        for met, low in (("dice", False), ("cldice", False), ("betti0_err", True)):
            best[(d, met)] = best_idx([val(summ, e, d, met, m) for e, m, _ in STAGES], low)
    for r, (exp, mode, lab) in enumerate(STAGES):
        cells = []
        for d in datasets:
            for met in ("dice", "cldice", "betti0_err"):
                m, s = val(summ, exp, d, met, mode), val(summ, exp, d, met, mode, "std")
                cells.append(fmt(m, s, 1 if met == "betti0_err" else 3, bold=(best[(d, met)] == r)) if met != "betti0_err"
                             else fmt(m, None, 1, bold=(best[(d, met)] == r)))
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def table_efficiency(eff) -> str:
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Deployment cost for one $512{\\times}512$ input (batch 1, inference graph). "
             "Latency is the median over repeated runs on the machine described in Sec.~\\ref{sec:setup}; "
             "MC-dropout multiplies the latency by $T$.}", "\\label{tab:eff}", "\\small",
             "\\begin{tabular}{l rrrrr}", "\\toprule",
             "Model & Params (M) & GFLOPs & CPU-1 (ms) & CPU-4 (ms) & GPU (ms) \\\\", "\\midrule"]
    for exp, mode, lab in STAGES:
        if mode == "mc" or not len(eff):
            continue
        e = eff[eff.exp == exp]
        if e.empty:
            lines.append(f"{lab} & -- & -- & -- & -- & -- \\\\")
            continue
        e = e.iloc[0]
        gpu = f"{e.gpu_ms:.1f}" if "gpu_ms" in e and not pd.isna(e.get("gpu_ms")) else "--"
        lines.append(f"{lab} & {e.params_m:.2f} & {e.gflops:.1f} & {e.cpu1_ms:.0f} & {e.cpu_ms:.0f} & {gpu} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_confusion(conf, targets) -> str:
    """Pixel confusion matrix (share of FOV pixels) with precision / recall / F1 / accuracy: TG-LiteNet per dataset, and stages A-E on the unseen average."""
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Pixel-level confusion matrix (vessel = positive, threshold 0.5): TP, FP, FN, TN in \\% of the FOV pixels, means over images and seeds. "
             "Top: TG-LiteNet (E) per dataset. Bottom: U-Net (A) and TG-LiteNet on the unseen average (all stages: \\texttt{confusion.csv}). F1 equals the per-image Dice.}", "\\label{tab:confusion}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l cccc cccc}", "\\toprule",
             " & TP & FP & FN & TN & Precision & Recall & F1 & Accuracy \\\\", "\\midrule"]

    def row(label, exp, ds, mode="det"):
        r = conf[(conf.exp == exp) & (conf["mode"] == mode) & (conf.dataset == ds)]
        if r.empty:
            return None
        r = r.iloc[0]
        cells = [f"{100 * r[k]:.2f}" for k in ("tp", "fp", "fn", "tn")] + [f"{r[k]:.3f}" for k in ("precision", "recall", "f1", "accuracy")]
        return f"{label} & " + " & ".join(cells) + " \\\\"

    top = [row(NICE[d], "E_tglitenet", d) for d in ["drive"] + list(targets)]
    bottom = [row(f"{lab}, unseen avg.", e, "unseen") for e, mode, lab in STAGES if mode == "det" and lab in ("A", "E (det.)")]
    lines += [t for t in top if t] + ["\\midrule"] + [b for b in bottom if b] + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_factorial(summ) -> str:
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Full factorial design (backbone $\\times$ topology $\\times$ DG). Unseen = macro average over the three targets. "
             "Stages A--D (U-Net plain, U-Net+topo, U-Net+topo+DG, Lite+topo+DG) have 5 seeds, the other four cells 3.}",
             "\\label{tab:factorial}", "\\small",
             "\\begin{tabular}{lcc cc cc}", "\\toprule", "Backbone & Topo & DG & Src Dice & Unseen Dice & Unseen clDice & $\\Delta$Dice \\\\", "\\midrule"]
    tick = lambda b: "\\checkmark" if b else "--"  # noqa: E731
    for (bb, topo, dg), exp in sorted(FACTORIAL_CELLS.items(), key=lambda kv: (kv[0][0] != "unet", kv[0][1] != "none", kv[0][2])):
        cells = [fmt(val(summ, exp, ds, met), val(summ, exp, ds, met, what="std"), 3)
                 for ds, met in (("drive", "dice"), ("unseen", "dice"), ("unseen", "cldice"), ("gap_unseen", "dice"))]
        lines.append(f"{'U-Net' if bb == 'unet' else 'Lite'} & {tick(topo != 'none')} & {tick(dg)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def bench_latency(tdir: Path) -> dict:
    """exp -> CPU latency (ms, 4 threads, 512x512) from the controlled benchmark (scripts/benchmark_latency.py).

    The per-run efficiency.json files are measured right after training and vary up to 2x for identical models, so the
    cross-architecture comparisons use this benchmark (interleaved rounds on an idle machine), mapped via each experiment's architecture."""
    from tglite.experiments import REGISTRY, make_config
    files = sorted(tdir.glob("latency_*.csv"))
    if not files:
        return {}
    table = pd.read_csv(files[0]).set_index("arch").cpu_ms.to_dict()
    out = {}
    for exp in REGISTRY:
        m = make_config(exp).model
        arch = f"U-Net base {int(m.base_channels)}" if m.backbone == "unet" else f"LiteNet w{int(m.width)}"
        if arch in table:
            out[exp] = float(table[arch])
    return out


def table_variants(summ, eff, lat) -> str:
    """Stage-D variants, ingredient controls, plain compact U-Nets and the width sweep, with same-session CPU latency."""
    groups = [
        [("D_lite_topo_dg", "soft-clDice + head (D)"), ("V_lite_cbdice", "cbDice"), ("V_lite_srl", "Skeleton Recall"), ("V_lite_cldice_srl", "clDice + Skel.\\ Recall")],
        [("Z_lite_noskel", "D without head"), ("Z_lite_nocons", "D without consistency"), ("S_unet_b8", "plain U-Net, base 8"), ("S_unet_b16", "plain U-Net, base 16")],
        [("W_lite_w8", "width $w{=}8$"), ("D_lite_topo_dg", "width $w{=}16$ (D)"), ("W_lite_w24", "width $w{=}24$"), ("W_lite_w32", "width $w{=}32$"), ("W_lite_w48", "width $w{=}48$")],
    ]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Variants of stage D (unseen averages): topology loss (top), removed ingredients and plain compact U-Nets with the same recipe (middle), lite width (bottom). "
             "Stage D has 5 seeds, other rows 3. CPU = 4 threads, controlled laptop benchmark.}",
             "\\label{tab:variants}", "\\small", "\\begin{tabular}{l ccc rr}", "\\toprule",
             " & Dice & clDice & $\\beta_0$ err & Params (M) & CPU (ms) \\\\", "\\midrule"]

    def row(exp, label):
        cells = [fmt(val(summ, exp, ds, met), val(summ, exp, ds, met, what="std"), 3 if met != "betti0_err" else 1)
                 for ds, met in (("unseen", "dice"), ("unseen", "cldice"), ("unseen", "betti0_err"))]
        cells[2] = fmt(val(summ, exp, "unseen", "betti0_err"), None, 1)
        e = eff[eff.exp == exp] if len(eff) else pd.DataFrame()
        pg = f"{e.params_m.iloc[0]:.2f}" if len(e) else "--"
        ms = f"{lat[exp]:.0f}" if exp in lat else "--"
        return f"{label} & " + " & ".join(cells) + f" & {pg} & {ms} \\\\"

    for k, g in enumerate(groups):
        lines += [row(e, l) for e, l in g] + (["\\midrule"] if k < len(groups) - 1 else [])
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_dataset_info(cfg) -> str:
    """Datasets table from data/processed/<name>/stats.json (written by prepare_data.py): measured, not assumed."""
    import json as _json
    meta = {  # name+citation, camera/FOV (from the dataset papers), role
        "drive": (r"DRIVE~\cite{staal2004}", "source"),
        "stare": (r"STARE~\cite{hoover2000}", "target"),
        "chase_db1": (r"CHASE\_DB1~\cite{fraz2012}", "target"),
        "hrf": (r"HRF~\cite{budai2013}", "target"),
        "fives": (r"FIVES~\cite{jin2022fives}", "target (opt.)"),
    }
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Datasets (all statistics measured on the files used in our experiments). FOV = field-of-view width, "
             r"used to normalise the scale of every image; calibre = mean vessel width (2$\times$ distance transform on the label skeleton) at native "
             r"resolution and after normalisation to a 527\,px FOV; G = green-channel mean$\pm$std inside the FOV (0--255). "
             r"Source = train (16) / validation (4) / test (20 official test images); targets are test-only.}",
             r"\label{tab:data}", r"\small", r"\begin{tabular}{l r c r r c c l}", r"\toprule",
             r"Dataset & \# & Size (px) & Vessel \% & FOV (px) & Calibre (px) & G & Role \\", r"\midrule"]
    summ_f = ROOT / "results" / "tables" / "summary.csv"
    evaluated = set(pd.read_csv(summ_f).dataset) if summ_f.exists() else set()
    for ds, (name, role) in meta.items():
        f = ROOT / "data" / "processed" / ds / "stats.json"
        if not f.exists() or (ds == "fives" and "fives" not in evaluated):   # list only datasets that were actually evaluated
            continue
        st = _json.loads(f.read_text())
        hw = next(iter(st["per_image"].values()))["hw"]
        cal = (f"{st['vessel_width_native_px']:.1f}$\\to${st['vessel_width_at_527px']:.1f}" if "vessel_width_native_px" in st else "--")
        g = (f"{st['green_mean']:.0f}$\\pm${st['green_std']:.0f}" if "green_mean" in st else "--")
        lines.append(f"{name} & {st['n_images']} & {hw[1]}$\\times${hw[0]} & {100 * st['vessel_fraction_mean']:.1f} & "
                     f"{st['fov_width_median']:.0f} & {cal} & {g} & {role} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def macros(summ, eff, unc) -> str:
    """\\newcommand numbers for the abstract / text; XX.X when a result is missing."""
    out = ["% auto-generated by scripts/make_report_assets.py - do not edit"]
    def mac(name, v, d=3):
        out.append(f"\\newcommand{{\\{name}}}{{{'XX' if v is None or np.isnan(v) else f'{v:.{d}f}'}}}")
    for tag, exp, mode in (("A", "A_unet", "det"), ("D", "D_lite_topo_dg", "det"), ("E", "E_tglitenet", "mc")):
        mac(f"srcDice{tag}", val(summ, exp, "drive", "dice", mode))
        mac(f"unDice{tag}", val(summ, exp, "unseen", "dice", mode))
        mac(f"unCl{tag}", val(summ, exp, "unseen", "cldice", mode))
        mac(f"unBetti{tag}", val(summ, exp, "unseen", "betti0_err", mode), 1)
        mac(f"gapDice{tag}", val(summ, exp, "gap_unseen", "dice", mode))
    for tag, exp in (("A", "A_unet"), ("E", "E_tglitenet")):
        e = eff[eff.exp == exp] if len(eff) else pd.DataFrame()
        mac(f"params{tag}", e.params_m.iloc[0] if len(e) else None, 2)
        mac(f"gflops{tag}", e.gflops.iloc[0] if len(e) else None, 1)
        mac(f"cpums{tag}", e.cpu_ms.iloc[0] if len(e) else None, 0)
    if len(eff) and {"A_unet", "E_tglitenet"} <= set(eff.exp):
        a, e = eff[eff.exp == "A_unet"].iloc[0], eff[eff.exp == "E_tglitenet"].iloc[0]
        mac("paramRatio", a.params_m / e.params_m, 0)
        mac("flopRatio", a.gflops / e.gflops, 0)
    else:
        mac("paramRatio", None), mac("flopRatio", None)
    import json as _json
    hw = _json.loads(eff.hardware.iloc[0]) if len(eff) and "hardware" in eff else {}
    def esc(t):
        return str(t).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    out.append(r"\newcommand{\hwGPU}{" + esc(hw.get("gpu", "GPU (see efficiency.json)")) + "}")
    out.append(r"\newcommand{\hwCPU}{" + (esc(hw.get("processor", "CPU")) if hw else "CPU (see efficiency.json)") + "}")
    out.append(r"\newcommand{\hwTorch}{" + esc(hw.get("torch", "PyTorch")) + "}")
    nseeds = max([val(summ, "E_tglitenet", "drive", "dice", "mc", "n")] + [0])
    out.append(f"\\newcommand{{\\nSeeds}}{{{int(nseeds) if nseeds else 'N'}}}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------- figures
SHORT = {"S_unet_b16": "U16", "W_lite_w48": "w48"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })


def _letter(ax, s: str) -> None:
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def fig_results(summ, targets):
    """Per-dataset Dice / clDice / Betti-0 error of stages A--E (mean over seeds, bars = std over seeds)."""
    if summ.empty:
        return
    stages = [s for s in STAGES if s[1] == "det"]
    ds_list = ["drive"] + targets
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.6))
    w = 0.8 / len(stages)
    for ax, (met, lab, log) in zip(axes, [("dice", "Dice", False), ("cldice", "clDice", False), ("betti0_err", "Betti-0 error (log)", True)]):
        for i, (exp, mode, name) in enumerate(stages):
            m = np.array([val(summ, exp, d, met, mode) for d in ds_list], float)
            s = np.array([val(summ, exp, d, met, mode, "std") for d in ds_list], float)
            ax.bar(np.arange(len(ds_list)) + (i - (len(stages) - 1) / 2) * w, m, w * 0.92, yerr=s, color=COL[name], linewidth=0,
                   label=name.replace(" (det.)", ""), error_kw={"elinewidth": 0.5, "capsize": 0.8, "capthick": 0.5})
        ax.axvline(0.5, color="k", lw=0.5, ls=":")
        ax.set_xticks(np.arange(len(ds_list)))
        ax.set_xticklabels([NICE_PLAIN[d].replace("CHASE_DB1", "CHASE") for d in ds_list], fontsize=6.5)
        ax.set_ylabel(lab, fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        if log:
            ax.set_yscale("log")
            ax.set_yticks([20, 50, 100, 200])
            ax.set_yticklabels(["20", "50", "100", "200"])
            ax.minorticks_off()
            ax.set_ylim(15, 260)
        else:
            lo = np.nanmin([val(summ, e, d, met, m_) for e, m_, _ in stages for d in ds_list])
            ax.set_ylim(max(0, lo - 0.06), min(1.0, np.nanmax([val(summ, e, d, met, m_) for e, m_, _ in stages for d in ds_list]) + 0.03))
    axes[0].text(0.0, 1.0, "source", transform=axes[0].get_xaxis_transform(), fontsize=6, color="0.35", va="bottom", ha="center")
    axes[0].text(2.0, 1.0, "unseen targets", transform=axes[0].get_xaxis_transform(), fontsize=6, color="0.35", va="bottom", ha="center")
    for ax, s in zip(axes, "abc"):
        _letter(ax, f"({s})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=False, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(pad=0.4, w_pad=1.0)
    savefig(fig, "fig_results")


CLEAN_SETS = [("A_unet", "A"), ("B_unet_topo", "B"), ("C_unet_topo_dg", "C"), ("D_lite_topo_dg", "D"), ("E_tglitenet", "E"),
              ("X_unet_dg", "U+DG"), ("S_unet_b8", "U8"), ("S_unet_b16", "U16"), ("X_lite_dg", "L+DG")]


def fig_cleanup(post, nat_summ, per_run):
    """Evaluation-only checks: raw vs cleaned Betti-0 error, and Dice at the fixed threshold, the DRIVE-val-tuned threshold and without FOV rescaling.

    Stages A--E use all five seeds, the other four configurations three; the no-rescale evaluation exists for A, D and E on seeds 3--4 only, so it is
    drawn next to the fixed-threshold Dice of the same two seeds (hollow marker).
    """
    if post is None or post.empty:
        return
    sets = [c for c in CLEAN_SETS if ((post.exp == c[0]) & (post["mode"] == "pp")).any()]
    names = [c[1] for c in sets]

    def g(exp, mode, col):
        r = post[(post.exp == exp) & (post["mode"] == mode)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    x = np.arange(len(sets))
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6))
    raw = [g(e, "fixed 0.5", "unseen_betti0_err") for e, _ in sets]
    pp = [g(e, "pp", "unseen_betti0_err") for e, _ in sets]
    ax = axes[0]
    ax.bar(x - 0.2, raw, 0.38, color="#b0b0b0", label="raw", linewidth=0)
    ax.bar(x + 0.2, pp, 0.38, color="#1f77b4", label="cleanup", linewidth=0)
    for xi, v in zip(x, pp):
        ax.text(xi + 0.2, v * 1.1, f"{v:.1f}", ha="center", fontsize=4.2, rotation=90, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(2.5, 400)
    ax.set_yticks([3, 10, 30, 100, 300])
    ax.set_yticklabels(["3", "10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_ylabel("unseen Betti-0 error", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper right", handlelength=1.0)
    ax = axes[1]
    fx = [g(e, "fixed 0.5", "unseen_dice") for e, _ in sets]
    th = [g(e, "thr", "unseen_dice") for e, _ in sets]
    for xi, a, b in zip(x, fx, th):
        ax.plot([xi, xi], [a, b], color="0.6", lw=0.8, zorder=1)
    ax.scatter(x, fx, s=16, color="k", label="fixed 0.5", zorder=3)
    ax.scatter(x, th, s=18, marker="^", color="#d62728", label="tuned thr.", zorder=3)
    if len(nat_summ) and per_run is not None:
        nx, nv, rv = [], [], []
        for k, (e, _) in enumerate(sets):
            n = val(nat_summ, e, "unseen", "dice")
            two = per_run[(per_run.exp == e) & (per_run["mode"] == "det") & (per_run.dataset == "unseen") & per_run.seed.isin([3, 4])].dice
            if np.isfinite(n) and len(two) == 2:
                nx.append(k), nv.append(n), rv.append(two.mean())
        ax.scatter(nx, rv, s=16, facecolors="none", edgecolors="k", marker="D", label="fixed, seeds 3-4", zorder=3)
        ax.scatter(nx, nv, s=18, marker="s", color="#2ca02c", label="no rescale", zorder=3)
        for xi, a, b in zip(nx, rv, nv):
            ax.plot([xi, xi], [a, b], color="0.6", lw=0.8, zorder=1)
    ax.set_ylabel("unseen Dice", fontsize=7)
    ax.legend(fontsize=5, frameon=False, loc="lower right", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
    ax.set_ylim(0.60, 0.82)
    for ax, s_ in zip(axes, "ab"):
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=5.5, rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        ax.axvline(4.5, color="k", lw=0.5, ls=":")
        _letter(ax, f"({s_})")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    savefig(fig, "fig_cleanup")


def fig_tradeoff(summ, eff, lat):
    """Unseen Dice against unseen clDice (overlap vs connectivity) and against CPU latency (cost); marker area grows with parameters."""
    if summ.empty or not len(eff):
        return
    pts = []
    for exp in sorted(set(summ.exp)):
        e = eff[eff.exp == exp]
        if e.empty:
            continue
        pts.append((exp, float(lat.get(exp, e.cpu_ms.iloc[0])), val(summ, exp, "unseen", "dice"), val(summ, exp, "unseen", "cldice"), float(e.params_m.iloc[0])
                    if "params_m" in e else 1.0))
    if not pts:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85))
    off = {0: {"A": (4, 3), "B": (4, -3), "C": (4, 1), "D": (-8, 3), "E": (4, -8), "U16": (3, -8), "w48": (-10, 3)},
           1: {"A": (5, 3), "B": (5, -3), "C": (5, -8), "D": (-9, -9), "E": (4, 3), "U16": (-4, 5), "w48": (-6, -9)}}
    for k, (ax, (xi, xl, logx)) in enumerate(zip(axes, [(3, "unseen clDice", False), (1, "CPU latency, ms (log)", True)])):
        for p in sorted(pts, key=lambda q: -q[4]):
            stage = next((s for e, m, s in STAGES if e == p[0] and m == "det"), None)
            marker = "s" if p[0].startswith(("A_", "B_", "C_", "S_", "X_unet")) else "o"
            size = 10 + 14 * np.log10(1 + 30 * p[4])
            ax.scatter(p[xi], p[2], s=size, marker=marker, color=COL.get(stage, "#d9d9d9"), edgecolor="k", linewidth=0.4, zorder=3 if stage else 2)
            lab = stage.replace(" (det.)", "") if stage and not stage.startswith("E (MC)") else SHORT.get(p[0])
            if lab in off[k]:
                ax.annotate(lab, (p[xi], p[2]), fontsize=6, xytext=off[k][lab], textcoords="offset points")
        if logx:
            ax.set_xscale("log")
            ax.set_xticks([100, 300, 1000, 3000])
            ax.set_xticklabels(["100", "300", "1k", "3k"])
            ax.minorticks_off()
            ax.set_xlim(55, 7000)
        ax.set_xlabel(xl, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("unseen Dice", fontsize=7)
    axes[1].tick_params(labelleft=False)
    axes[0].sharey(axes[1])
    for ax, s in zip(axes, "ab"):
        _letter(ax, f"({s})")
    fig.tight_layout(pad=0.4, w_pad=0.6)
    savefig(fig, "fig_tradeoff")


def fig_training(runs_dir: Path):
    """Training curves (DRIVE validation Dice, mean and min--max over seeds) of stages A--D, used in the README only."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    any_ = False
    for exp, mode, name in STAGES:
        if mode == "mc":
            continue
        curves = []
        for p in sorted((runs_dir / exp).glob("seed*/train_log.csv")):
            d = pd.read_csv(p).dropna(subset=["val_dice"])
            if len(d):
                curves.append(d.set_index("iter")["val_dice"])
        if curves:
            c = pd.concat(curves, axis=1)
            ax.plot(c.index, c.mean(axis=1), color=COL[name], label=name)
            ax.fill_between(c.index, c.min(axis=1), c.max(axis=1), color=COL[name], alpha=0.15)
            any_ = True
    if not any_:
        plt.close(fig)
        return
    ax.set_xlabel("iteration", fontsize=7)
    ax.set_ylabel("DRIVE validation Dice", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig_training")


def _read_png(path: Path):
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if a is None else a.astype(np.float32) / 255.0


def _err_overlay(img, pred, gt, fov):
    out = (img.astype(np.float32) / 255.0 * 0.45)
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    for m, c in ((tp, (1, 1, 1)), (fp, (1, 0.15, 0.15)), (fn, (0.2, 0.55, 1))):
        out[m & fov] = c
    return out


def _crop_box(a, b, fov, frac=0.3):
    """Window (fraction of the shorter side) where predictions a and b disagree most (integral image over the FOV)."""
    H, W = a.shape
    h = w = int(frac * min(H, W))
    d = ((a != b) & fov).astype(np.float32)
    ii = cv2.integral(d)
    best, pos = -1.0, (0, 0)
    step = max(4, h // 8)
    for y in range(0, H - h + 1, step):
        for x in range(0, W - w + 1, step):
            s = ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]
            if s > best:
                best, pos = s, (y, x)
    return pos[0], pos[1], h, w


def fig_qualitative(runs_dir: Path, cfg, targets):
    """One test image per dataset (seed 0): full image with the zoom window, ground truth, error maps of A, D, E and the MC uncertainty of E in the window."""
    exps = [("A_unet", "A"), ("D_lite_topo_dg", "D"), ("E_tglitenet", "E")]
    seed_dir = {e: runs_dir / e / "seed0" for e, _ in exps}
    if not all((d / "preds").exists() for d in seed_dir.values()):
        print("  (qualitative figure skipped: preds/ missing for A, D or E seed 0)")
        return
    ds_list = ["drive"] + targets
    fig, axes = plt.subplots(len(ds_list), 6, figsize=(7.1, 1.18 * len(ds_list) + 0.25), gridspec_kw={"wspace": 0.03, "hspace": 0.04})
    for r, ds in enumerate(ds_list):
        ids = sorted(p.name[:-9] for p in (seed_dir["E_tglitenet"] / "preds" / ds).glob("*_prob.png"))
        if not ids:
            continue
        sid = ids[len(ids) // 2]
        s = load_sample(ds, sid, cfg, cfg.eval.scale_mode)
        gt, fov = s["mask"] > 0, s["fov"] > 0
        probs = {t: _read_png(seed_dir[e] / "preds" / ds / f"{sid}_prob.png") for e, t in exps}
        if any(p is None for p in probs.values()):
            continue
        y, x, h, w = _crop_box(probs["A"] >= 0.5, probs["E"] >= 0.5, fov)
        sl = (slice(y, y + h), slice(x, x + w))
        u = _read_png(seed_dir["E_tglitenet"] / "preds" / ds / f"{sid}_unc.png")
        u = np.zeros(gt.shape, np.float32) if u is None else u
        full = s["img"]
        panels = [("image (zoom window)", full, None), ("ground truth", np.stack([gt] * 3, -1).astype(float)[sl], None)]
        for e, t in exps:
            panels.append((t, _err_overlay(full, probs[t] >= 0.5, gt, fov)[sl], None))
        uc = u[sl]
        hi = max(float(np.percentile(u[fov], 99.5)), 1e-6)
        panels.append(("E uncertainty", uc, hi))
        for c, (title, im, hi_) in enumerate(panels):
            ax = axes[r, c]
            if hi_ is None:
                ax.imshow(im)
            else:
                ax.imshow(im, cmap="magma", vmin=0, vmax=hi_)
            if c == 0:
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#ffd400", lw=1.0))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title({"A": "A: U-Net", "D": "D: lite", "E": "E: TG-LiteNet"}.get(title, title), fontsize=7, pad=2)
        axes[r, 0].set_ylabel(NICE_PLAIN[ds].replace("CHASE_DB1", "CHASE"), fontsize=7)
    savefig(fig, "fig_qualitative")


def fig_risk_coverage(unc, targets):
    """Risk--coverage of the MC-dropout variance: pixel error and mean (1 - Dice) of the retained images, with the random baseline."""
    if unc is None or unc.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.5))
    cols = {"drive": "k", "stare": "#1f77b4", "chase_db1": "#2ca02c", "hrf": "#d62728"}
    pcov, icov = [1.0, 0.9, 0.8, 0.5], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for ds in ["drive"] + targets:
        r = unc[(unc.exp == "E_tglitenet") & (unc["mode"] == "mc") & (unc.dataset == ds)]
        if r.empty:
            continue
        r = r.iloc[0]
        axes[0].plot(pcov, [r.get(f"perr_var_c{c}_mean", np.nan) for c in pcov], color=cols[ds], marker="o", ms=2, lw=1, label=NICE_PLAIN[ds].replace("CHASE_DB1", "CHASE"))
        ys = [r.get(f"img_risk_var_c{c}_mean", np.nan) for c in icov]
        axes[1].plot(icov, ys, color=cols[ds], marker="o", ms=2, lw=1)
        axes[1].axhline(ys[0], color=cols[ds], lw=0.5, ls=":")
    axes[0].set_xlabel("pixel coverage", fontsize=7)
    axes[0].set_ylabel("pixel error rate", fontsize=7)
    axes[1].set_xlabel("image coverage", fontsize=7)
    axes[1].set_ylabel("mean 1 − Dice, retained", fontsize=7)
    for ax, s in zip(axes, "ab"):
        ax.invert_xaxis()
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.3, lw=0.4)
        _letter(ax, f"({s})")
    axes[0].legend(fontsize=5.5, frameon=False, handlelength=1.2, loc="upper right")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    savefig(fig, "fig_risk_coverage")


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--report-dir", default="report", help="where generated/ and figures/ are written")
    ap.add_argument("--placeholders", action="store_true", help="write empty ('--') tables so the paper compiles before results exist")
    args = ap.parse_args()
    global GEN, FIG
    GEN, FIG = ROOT / args.report_dir / "generated", ROOT / args.report_dir / "figures"
    tdir, runs = ROOT / args.tables, ROOT / args.runs
    if args.placeholders:
        tdir = ROOT / "results" / "_empty"
        tdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["exp", "mode", "dataset", "metric", "mean", "std", "n_seeds", "ci95"]).to_csv(tdir / "summary.csv", index=False)
    if not (tdir / "summary.csv").exists():
        raise SystemExit("run scripts/aggregate_results.py first")
    summ = pd.read_csv(tdir / "summary.csv")
    eff = pd.read_csv(tdir / "efficiency.csv") if (tdir / "efficiency.csv").exists() else pd.DataFrame()
    unc = pd.read_csv(tdir / "uncertainty.csv") if (tdir / "uncertainty.csv").exists() else pd.DataFrame()
    cfg = load_config(["configs/base.yaml"])
    targets = [d for d in CORE_TARGETS if d in set(summ.dataset)]
    extra = [d for d in ("fives",) if d in set(summ.dataset)]
    print("tables:")
    write("tab_main.tex", table_main(summ, eff))
    write("tab_datasets.tex", table_datasets(summ, ["drive"] + targets + extra))
    write("tab_efficiency.tex", table_efficiency(eff))
    conf = pd.read_csv(tdir / "confusion.csv") if (tdir / "confusion.csv").exists() else pd.DataFrame()
    if len(conf):
        write("tab_confusion.tex", table_confusion(conf, targets))
    write("tab_factorial.tex", table_factorial(summ))
    lat = bench_latency(tdir)
    write("tab_variants.tex", table_variants(summ, eff, lat))
    post = pd.read_csv(tdir / "postproc_summary.csv") if (tdir / "postproc_summary.csv").exists() else pd.DataFrame()
    nat_summ = pd.read_csv(tdir / "native_summary.csv") if (tdir / "native_summary.csv").exists() else pd.DataFrame()
    write("tab_datasets_info.tex", table_dataset_info(cfg))
    write("numbers.tex", macros(summ, eff, unc))
    ov = ROOT / "results" / "figures" / "data_overview_strip.png"
    if ov.exists():
        FIG.mkdir(parents=True, exist_ok=True)
        (FIG / "data_overview.png").write_bytes(ov.read_bytes())
    print("figures:")
    _style()
    fig_results(summ, targets)
    fig_cleanup(post, nat_summ, pd.read_csv(tdir / "per_run.csv") if (tdir / "per_run.csv").exists() else None)
    fig_tradeoff(summ, eff, lat)
    fig_training(runs)
    fig_qualitative(runs, cfg, targets)
    fig_risk_coverage(unc, targets)


if __name__ == "__main__":
    main()
