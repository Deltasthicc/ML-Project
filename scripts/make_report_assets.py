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
    lines = ["\\begin{table*}[t]", "\\centering", "\\caption{Staged ablation (mean$\\pm$std over seeds). Source = DRIVE test; unseen = macro average over STARE, "
             "CHASE\\_DB1 and HRF (never used for training or model selection); $\\Delta$Dice = Dice(DRIVE)$-$Dice(unseen). "
             "Cost: one $512{\\times}512$ image, batch 1, CPU = 4 threads, Kaggle machine; latency is machine-dependent; E (MC) needs $T$ forward passes. Best per column in bold.}",
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


def table_uncertainty(unc, targets) -> str:
    """Table VI: does uncertainty track error?  (image-level and pixel-level, source vs unseen)."""
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Uncertainty quality of TG-LiteNet (E). $\\rho_{px}$: mean per-image Spearman correlation between "
             "pixel uncertainty and misclassification; $\\rho_{img}$: Spearman between mean image uncertainty and $1-$Dice; "
             "AURC: area under the image-level risk--coverage curve (lower is better; random = mean error, oracle = lower bound). "
             "\\emph{Var} = MC-dropout variance, \\emph{Ent} = predictive entropy.}",
             "\\label{tab:unc}", "\\small", "\\begin{tabular}{l cc cc ccc}", "\\toprule",
             " & \\multicolumn{2}{c}{$\\rho_{px}$} & \\multicolumn{2}{c}{$\\rho_{img}$} & \\multicolumn{3}{c}{AURC} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}",
             "Domain & Var & Ent & Var & Ent & Var & Rand. & Oracle \\\\", "\\midrule"]
    if unc is None or unc.empty:
        return "\n".join(lines + ["\\multicolumn{8}{c}{(no results yet)} \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"])
    for ds in ["drive"] + targets:
        r = unc[(unc.exp == "E_tglitenet") & (unc["mode"] == "mc") & (unc.dataset == ds)]
        if r.empty:
            continue
        r = r.iloc[0]
        g = lambda k: fmt(r.get(f"{k}_mean", np.nan), None, 3)  # noqa: E731
        lines.append(f"{NICE[ds]} & {g('rho_var')} & {g('rho_ent')} & {g('img_spearman_var')} & {g('img_spearman_ent')} & "
                     f"{g('aurc_var')} & {g('risk_random')} & {g('aurc_oracle')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_confusion(conf, targets) -> str:
    """Pixel confusion matrix (share of FOV pixels) with precision / recall / F1 / accuracy: TG-LiteNet per dataset, and stages A-E on the unseen average."""
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Pixel-level confusion matrix and classification metrics (vessel = positive class, threshold 0.5). TP, FP, FN, TN are shares of the "
             "FOV pixels in \\%; all values are means over images and seeds. Top: TG-LiteNet (E) on every dataset. Bottom: the U-Net (A) and TG-LiteNet on the "
             "unseen average (other stages: \\texttt{results/tables/confusion.csv}). F1 equals the per-image Dice.}", "\\label{tab:confusion}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
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
             "U-Net plain, U-Net+topo, U-Net+topo+DG and Lite+topo+DG are stages A--D (3 seeds); the other cells have 2 seeds, so their $\\pm$ is only indicative.}",
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
        [("D_lite_topo_dg", "soft-clDice + skel.\\ head (D)"), ("V_lite_cbdice", "cbDice"), ("V_lite_srl", "Skeleton Recall"), ("V_lite_cldice_srl", "clDice + Skel.\\ Recall")],
        [("Z_lite_noskel", "D without skeleton head"), ("Z_lite_nocons", "D without consistency loss"), ("S_unet_b8", "plain U-Net, base 8 (topo+DG)"), ("S_unet_b16", "plain U-Net, base 16 (topo+DG)")],
        [("W_lite_w8", "width $w{=}8$"), ("D_lite_topo_dg", "width $w{=}16$ (D)"), ("W_lite_w24", "width $w{=}24$"), ("W_lite_w32", "width $w{=}32$"), ("W_lite_w48", "width $w{=}48$")],
    ]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Variants of stage D: topology loss (top), removed ingredients and plain compact U-Nets with the same recipe (middle), lite width (bottom). "
             "Stage D has 5 seeds, all other rows 3. CPU latency (4 threads) is from a controlled benchmark on one laptop CPU (Sec.~\\ref{sec:setup}).}",
             "\\label{tab:variants}", "\\small", "\\begin{tabular}{l cccc rr r}", "\\toprule",
             " & Src Dice & Unseen Dice & Unseen clDice & Unseen $\\beta_0$ & Params (M) & GFLOPs & CPU (ms) \\\\", "\\midrule"]

    def row(exp, label):
        cells = [fmt(val(summ, exp, ds, met), val(summ, exp, ds, met, what="std"), 3 if met != "betti0_err" else 1)
                 for ds, met in (("drive", "dice"), ("unseen", "dice"), ("unseen", "cldice"), ("unseen", "betti0_err"))]
        cells[3] = fmt(val(summ, exp, "unseen", "betti0_err"), None, 1)
        e = eff[eff.exp == exp] if len(eff) else pd.DataFrame()
        pg = f"{e.params_m.iloc[0]:.2f} & {e.gflops.iloc[0]:.1f}" if len(e) else "-- & --"
        ms = f"{lat[exp]:.0f}" if exp in lat else "--"
        return f"{label} & " + " & ".join(cells) + f" & {pg} & {ms} \\\\"

    for k, g in enumerate(groups):
        lines += [row(e, l) for e, l in g] + (["\\midrule"] if k < len(groups) - 1 else [])
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def table_checks(post, nat_summ) -> str:
    """Evaluation-only checks on seeds 3 and 4: small-component removal, validation-tuned threshold and no-rescale evaluation."""
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Evaluation-only checks on the trained models (seeds 3 and 4, unseen average). Cleanup removes connected components below a size chosen on DRIVE-validation; "
             "tuned = one threshold per model chosen on DRIVE-validation; native = no FOV rescaling (run for A, D and E only).}",
             "\\label{tab:checks}", "\\small", "\\begin{tabular}{l cc ccc c}", "\\toprule",
             " & \\multicolumn{2}{c}{Betti-0 error} & \\multicolumn{4}{c}{Dice} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}",
             "Stage & fixed 0.5 & cleanup & fixed 0.5 & cleanup & tuned thr. & native \\\\", "\\midrule"]
    for exp, mode, lab in STAGES:
        if mode != "det":
            continue
        g = lambda m, c: (lambda r: float(r[c].iloc[0]) if len(r) else float("nan"))(post[(post.exp == exp) & (post["mode"] == m)])  # noqa: E731
        nat = val(nat_summ, exp, "unseen", "dice") if len(nat_summ) else float("nan")
        lines.append(f"{lab} & {g('fixed 0.5', 'unseen_betti0_err'):.1f} & {g('pp', 'unseen_betti0_err'):.1f} & {g('fixed 0.5', 'unseen_dice'):.3f} & "
                     f"{g('pp', 'unseen_dice'):.3f} & {g('thr', 'unseen_dice'):.3f} & {fmt(nat, None, 3)} \\\\")
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


def table_hparams(cfg) -> str:
    c = cfg
    rows = [
        ("Optimiser", r"AdamW, lr $%g$, wd $%g$" % (c.train.lr, c.train.weight_decay)),
        ("Schedule", r"%d warm-up it., cosine to $1\%%$" % c.train.warmup_iters),
        ("Iterations / batch / patch", r"%d / %d / %d$^2$ px" % (c.train.iters, c.train.batch_size, c.data.patch_size)),
        ("Loss weights", r"$\lambda_1{=}%g,\ \lambda_2{=}%g,\ \lambda_3{=}%g$; skel. head $%g$" % (c.loss.lambda_bce, c.loss.lambda_topo, c.loss.lambda_cons, c.loss.lambda_skel_head)),
        ("Soft-skeleton iterations", "%d" % c.loss.skel_iters),
        ("CLAHE", "clip %g, %d px tiles" % (c.data.clahe_clip, c.data.clahe_tile_px)),
        ("Scale normalisation", "FOV width $=%d$ px" % c.data.fov_target_width),
        ("Lite width $w$ / U-Net base", "%d / %d" % (c.model.width, c.model.base_channels)),
        ("MC dropout (stage E)", r"$p{=}0.1$, $T{=}%d$ passes" % c.eval.mc_samples),
        ("Threshold", "%g (fixed; never tuned on targets)" % c.eval.threshold),
    ]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Training configuration shared by all stages (generated from \texttt{configs/base.yaml}).}",
             r"\label{tab:hparams}", r"\small", r"\begin{tabular}{ll}", r"\toprule", r"Item & Value \\", r"\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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
def fig_ablation(summ, targets):
    if summ.empty:
        return
    mets = [("dice", "Dice"), ("cldice", "clDice")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False)
    ds_list = ["drive"] + targets
    w = 0.8 / len(STAGES)
    for ax, (met, lab) in zip(axes, mets):
        for i, (exp, mode, name) in enumerate(STAGES):
            m = [val(summ, exp, d, met, mode) for d in ds_list]
            s = [val(summ, exp, d, met, mode, "std") for d in ds_list]
            ax.bar(np.arange(len(ds_list)) + (i - (len(STAGES) - 1) / 2) * w, m, w, yerr=s, color=COL[name], label=name, capsize=1.5, linewidth=0)
        ax.set_xticks(np.arange(len(ds_list)))
        ax.set_xticklabels([NICE_PLAIN[d] for d in ds_list], fontsize=7)
        ax.set_ylabel(lab, fontsize=8)
        lo = np.nanmin([val(summ, e, d, met, m) for e, m, _ in STAGES for d in ds_list])
        ax.set_ylim(max(0, lo - 0.08), 1.0 if met == "dice" else 1.0)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=6, ncol=3, loc="lower left")
    fig.tight_layout()
    savefig(fig, "fig_ablation")


SHORT = {"X_unet_dg": "U+DG", "X_lite_plain": "L", "X_lite_topo": "L+T", "X_lite_dg": "L+DG", "V_lite_cbdice": "cbD", "V_lite_srl": "SRL",
         "V_lite_cldice_srl": "cl+SRL", "W_lite_w8": "w8", "W_lite_w24": "w24", "W_lite_w32": "w32", "W_lite_w48": "w48",
         "S_unet_b8": "U8", "S_unet_b16": "U16", "Z_lite_noskel": "-head", "Z_lite_nocons": "-cons"}
LABELLED = {"X_unet_dg", "X_lite_plain", "X_lite_dg", "W_lite_w8", "W_lite_w48", "S_unet_b8", "S_unet_b16", "Z_lite_noskel", "Z_lite_nocons"}


def fig_tradeoff(summ, eff, lat):
    """Two readable panels: unseen Dice vs unseen clDice (overlap vs connectivity) and vs CPU latency from one Colab session (the cost)."""
    if summ.empty or not len(eff):
        return
    pts = []
    for exp in sorted(set(summ.exp)):
        e = eff[eff.exp == exp]
        if e.empty:
            continue
        pts.append((exp, float(lat.get(exp, e.cpu_ms.iloc[0])), val(summ, exp, "unseen", "dice"), val(summ, exp, "unseen", "cldice")))
    if not pts:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95))
    for ax, (xi, xl, logx) in zip(axes, [(3, "Unseen clDice", False), (1, "CPU latency, ms (log, laptop)", True)]):
        for p in pts:
            stage = next((s for e, m, s in STAGES if e == p[0] and m == "det"), None)
            marker = "s" if ("unet" in p[0] or p[0][0] in "ABC") and not p[0].startswith(("D_", "E_")) else "o"
            ax.scatter(p[xi], p[2], s=16, marker=marker, color=COL.get(stage, "#bbbbbb"), edgecolor="k", linewidth=0.4, zorder=3)
            lab = stage if stage and not stage.startswith("E (MC)") else SHORT.get(p[0]) if p[0] in LABELLED else None
            if lab:
                ax.annotate(lab.replace(" (det.)", ""), (p[xi], p[2]), fontsize=5.5, xytext=(2, 1.5), textcoords="offset points")
        if logx:
            ax.set_xscale("log")
            ax.set_xticks([100, 300, 1000, 3000])
            ax.set_xticklabels(["100", "300", "1000", "3000"])
            ax.minorticks_off()
            ax.set_xlim(90, 6500)
        ax.set_xlabel(xl, fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Unseen Dice", fontsize=6)
    fig.tight_layout(pad=0.4)
    savefig(fig, "fig_tradeoff")


def fig_training(runs_dir: Path):
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
    for m, c in ((tp, (1, 1, 1)), (fp, (1, 0.15, 0.15)), (fn, (0.2, 0.5, 1))):
        out[m & fov] = c
    return out


def fig_qualitative(runs_dir: Path, cfg, targets):
    exps = [("A_unet", "A"), ("D_lite_topo_dg", "D"), ("E_tglitenet", "E")]
    seed_dir = {e: runs_dir / e / "seed0" for e, _ in exps}
    if not all((d / "preds").exists() for d in seed_dir.values()):
        print("  (qualitative figure skipped: preds/ missing for A, D or E seed 0)")
        return
    ds_list = ["drive"] + targets
    fig, axes = plt.subplots(len(ds_list), 6, figsize=(7.2, 1.25 * len(ds_list) + 0.3))
    for r, ds in enumerate(ds_list):
        ids = sorted(p.name[:-9] for p in (seed_dir["E_tglitenet"] / "preds" / ds).glob("*_prob.png"))
        if not ids:
            continue
        sid = ids[len(ids) // 2]
        s = load_sample(ds, sid, cfg, cfg.eval.scale_mode)
        gt, fov = s["mask"] > 0, s["fov"] > 0
        panels = [("image", s["img"]), ("ground truth", np.stack([gt] * 3, -1).astype(float))]
        for e, tag in exps:
            p = _read_png(seed_dir[e] / "preds" / ds / f"{sid}_prob.png")
            panels.append((f"{tag}", _err_overlay(s["img"], p >= 0.5, gt, fov) if p is not None else np.zeros_like(s["img"])))
        u = _read_png(seed_dir["E_tglitenet"] / "preds" / ds / f"{sid}_unc.png")
        panels.append(("E uncertainty", u if u is not None else np.zeros(gt.shape)))
        for c, (title, im) in enumerate(panels[:6]):
            ax = axes[r, c]
            ax.imshow(im, cmap="magma" if title == "E uncertainty" else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=7)
        axes[r, 0].text(-0.05, 0.5, NICE_PLAIN[ds], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    fig.text(0.5, 0.0, "white = TP, red = FP, blue = FN (threshold 0.5, seed 0)", ha="center", fontsize=6)
    fig.tight_layout()
    savefig(fig, "fig_qualitative")


def fig_risk_coverage(unc, targets):
    if unc is None or unc.empty:
        return
    covs = [1.0, 0.9, 0.8, 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ds, col in zip(["drive"] + targets, ["k", "#1f77b4", "#2ca02c", "#d62728"]):
        r = unc[(unc.exp == "E_tglitenet") & (unc["mode"] == "mc") & (unc.dataset == ds)]
        if r.empty:
            continue
        r = r.iloc[0]
        for kind, ls in (("var", "-"), ("ent", "--")):
            ys = [r.get(f"perr_{kind}_c{c}_mean", np.nan) for c in covs]
            axes[0].plot(covs, ys, ls, color=col, marker="o", ms=2.5, label=f"{NICE_PLAIN[ds]} ({kind})")
        ys = [r.get(f"img_risk_var_c{c}_mean", np.nan) for c in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]]
        axes[1].plot([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], ys, color=col, marker="o", ms=2.5, label=NICE_PLAIN[ds])
    axes[0].set_xlabel("pixel coverage", fontsize=7)
    axes[0].set_ylabel("pixel error rate", fontsize=7)
    axes[1].set_xlabel("image coverage", fontsize=7)
    axes[1].set_ylabel("mean (1 $-$ Dice) of retained images", fontsize=7)
    for ax in axes:
        ax.invert_xaxis()
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=5, ncol=2)
    axes[1].legend(fontsize=5)
    fig.tight_layout()
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
    write("tab_uncertainty.tex", table_uncertainty(unc, targets + extra))
    conf = pd.read_csv(tdir / "confusion.csv") if (tdir / "confusion.csv").exists() else pd.DataFrame()
    if len(conf):
        write("tab_confusion.tex", table_confusion(conf, targets))
    write("tab_factorial.tex", table_factorial(summ))
    lat = bench_latency(tdir)
    write("tab_variants.tex", table_variants(summ, eff, lat))
    post = pd.read_csv(tdir / "postproc_summary.csv") if (tdir / "postproc_summary.csv").exists() else pd.DataFrame()
    nat_summ = pd.read_csv(tdir / "native_summary.csv") if (tdir / "native_summary.csv").exists() else pd.DataFrame()
    if len(post):
        write("tab_checks.tex", table_checks(post, nat_summ))
    write("tab_datasets_info.tex", table_dataset_info(cfg))
    write("tab_hparams.tex", table_hparams(cfg))
    write("numbers.tex", macros(summ, eff, unc))
    ov = ROOT / "results" / "figures" / "data_overview_strip.png"
    if ov.exists():
        FIG.mkdir(parents=True, exist_ok=True)
        (FIG / "data_overview.png").write_bytes(ov.read_bytes())
    print("figures:")
    fig_ablation(summ, targets + extra)
    fig_tradeoff(summ, eff, lat)
    fig_training(runs)
    fig_qualitative(runs, cfg, targets)
    fig_risk_coverage(unc, targets + extra)


if __name__ == "__main__":
    main()
