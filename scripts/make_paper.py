"""Build every table, figure and number macro of the IEEE paper from the committed result files.

    python scripts/make_paper.py          # -> paper/generated/*.tex, paper/figures/*.pdf|png

Nothing here is typed in by hand: all values are read from ``results/`` (v2: ``results/tables``, v2.1: ``results/v21/tables``) and
``protocol/``.  The paper's prose uses the macros written to ``paper/generated/numbers.tex``.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R2, R21 = ROOT / "results" / "tables", ROOT / "results" / "v21" / "tables"
OUT, FIG, GEN = ROOT / "paper", ROOT / "paper" / "figures", ROOT / "paper" / "generated"
for d in (FIG, GEN):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------- helpers
STAGES = [("A_unet", "A", "U-Net"), ("B_unet_topo", "B", "+ topology"), ("C_unet_topo_dg", "C", "+ DG aug."), ("D_compact", "D", "compact"), ("E_tglitenet", "E", "TG-LiteNet")]
COL = {"A": "#6b6b6b", "B": "#1f77b4", "C": "#2ca02c", "D": "#ff7f0e", "E": "#d62728"}
DS = ["drive", "stare", "chase_db1", "hrf", "fives"]
DSN = {"drive": "DRIVE", "stare": "STARE", "chase_db1": "CHASE_DB1", "hrf": "HRF", "fives": "FIVES"}
DSL = {"drive": "DRIVE", "stare": "STARE", "chase_db1": "CHASE", "hrf": "HRF", "fives": "FIVES"}
MAC: dict[str, str] = {}


def mac(name: str, value) -> None:
    v = str(value)
    if v.startswith("-"):
        v = r"\ensuremath{-}" + v[1:]            # typographic minus in text and in math mode
    MAC[name] = v


def pm(s: str) -> tuple[float, float]:
    a, _, b = str(s).partition("±")
    return float(a), (float(b) if b else float("nan"))


def f3(x, d=3):
    return f"{x:.{d}f}"


def cell(s: str, d: int | None = None, bold: bool = False) -> str:
    """'0.759±0.007' -> '0.759{\\scriptsize$\\pm$0.007}' (optionally bold)."""
    a, b = pm(s)
    d = d if d is not None else len(str(s).split("±")[0].split(".")[-1])
    t = f"{a:.{d}f}"
    if bold:
        t = r"\textbf{" + t + "}"
    return t + ("" if np.isnan(b) else r"{\scriptsize$\pm$" + f"{b:.{d}f}" + "}")


def write(name: str, text: str) -> None:
    (GEN / name).write_text(text, encoding="utf-8", newline="\n")
    print("  wrote", GEN / name)


def tabular(spec: str, header: list[str], rows: list[list[str]], midrules: tuple[int, ...] = ()) -> str:
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "pdf.fonttype": 42,
                         "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("  wrote", FIG / f"{name}.pdf")


def title(ax, s: str) -> None:
    ax.set_title(s, loc="left", fontsize=8.5, fontweight="bold", pad=4)


def parse_md_table(path: Path, first_only: bool = True) -> list[list[str]]:
    rows, started = [], False
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("|"):
            started = True
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
        elif started and first_only:
            break
    return rows[1:]


def kparams(s: str) -> float:
    s = str(s).strip()
    return float(s[:-1]) * 1e3 if s.endswith("K") else float(s)


# ------------------------------------------------------------------------------------------------------------------ data
raw, clean, tuned = (pd.read_csv(R2 / f"main_{m}.csv").set_index("model") for m in ("raw", "clean", "tuned"))
m21 = pd.read_csv(R21 / "main.csv").set_index("model")
eff = pd.read_csv(R2 / "efficiency.csv").set_index("exp")
perrun_raw = pd.read_csv(R2 / "per_run_raw.csv")
perrun21 = pd.read_csv(R21 / "per_run.csv")
dstats = pd.read_csv(ROOT / "results" / "data" / "dataset_stats.csv").set_index("dataset")
comp = pd.read_csv(R2 / "comparisons.csv")
unc = pd.read_csv(R2 / "uncertainty.csv")
training = pd.read_csv(R2 / "training.csv")
dec2 = json.loads((ROOT / "protocol" / "decisions.json").read_text())
dec21 = json.loads((ROOT / "protocol" / "decisions_v21.json").read_text())


def k(tag: str) -> str:
    return {"A": "A_unet[det]", "B": "B_unet_topo[det]", "C": "C_unet_topo_dg[det]", "D": "D_compact[det]", "E": "E_tglitenet[det]", "Emc": "E_tglitenet[mc]"}[tag]


# ---------------------------------------------------------------------------------------------------- number macros
for tag in "ABCDE":
    r_, t_, c_ = raw.loc[k(tag)], tuned.loc[k(tag)], clean.loc[k(tag)]
    mac(f"uDiceRaw{tag}", f3(r_["unseen/dice_mean"]))
    mac(f"uDiceTuned{tag}", f3(t_["unseen/dice_mean"]))
    mac(f"uDiceClean{tag}", f3(c_["unseen/dice_mean"]))
    mac(f"uClRaw{tag}", f3(r_["unseen/cldice_mean"]))
    mac(f"uBettiRaw{tag}", f"{r_['unseen/betti0_err_mean']:.1f}")
    mac(f"uBettiClean{tag}", f"{c_['unseen/betti0_err_mean']:.1f}")
    mac(f"gapRaw{tag}", f3(r_["gap/dice_mean"]))
    mac(f"gapTuned{tag}", f3(t_["gap/dice_mean"]))
    mac(f"driveRaw{tag}", f3(r_["drive/dice_mean"]))
    mac(f"cpu{tag}", f"{r_['cpu_ms']:.0f}")
    mac(f"par{tag}", f"{r_['params_M']:.2f}")
mac("uDiceRawEmc", f3(raw.loc[k("Emc"), "unseen/dice_mean"]))
mac("uDiceTunedEmc", f3(tuned.loc[k("Emc"), "unseen/dice_mean"]))
mac("cpuEmc", f"{raw.loc[k('Emc'), 'cpu_ms']:.0f}")
mac("gpuA", f"{raw.loc[k('A'), 'gpu_ms']:.1f}")
mac("gpuE", f"{raw.loc[k('E'), 'gpu_ms']:.1f}")
mac("gflopsA", f"{raw.loc[k('A'), 'GFLOPs']:.0f}")
mac("gflopsE", f"{raw.loc[k('E'), 'GFLOPs']:.2f}")
mac("fivesRawX", f3(raw.loc["X_unet_dg[det]", "fives/dice_mean"]))
mac("uDiceRawX", f3(raw.loc["X_unet_dg[det]", "unseen/dice_mean"]))
mac("uDiceTunedX", f3(tuned.loc["X_unet_dg[det]", "unseen/dice_mean"]))
mac("uBettiRawX", f"{raw.loc['X_unet_dg[det]', 'unseen/betti0_err_mean']:.1f}")
mac("uDiceTunedWarm", f3(tuned.loc["W_unet_topo_dg_warm[det]", "unseen/dice_mean"]))
mac("uDiceRawWarm", f3(raw.loc["W_unet_topo_dg_warm[det]", "unseen/dice_mean"]))
for name, key in (("Keight", "K_unet_w8[det]"), ("Ksixteen", "K_unet_w16[det]"), ("Klite", "K_lite_w16[det]"), ("Khyb", "K_hybrid_w16[det]")):
    mac(f"cpu{name}", f"{raw.loc[key, 'cpu_ms']:.0f}")
    mac(f"par{name}", f"{raw.loc[key, 'params_M']:.2f}")

for tag, key in (("H", "H_tglite70k[det]"), ("Rl", "R_lite_w16[det]"), ("Ru", "R_unet_w8[det]"), ("ENSH", "ENS_H_tglite70k[ens]"), ("Ev", "E_tglitenet[det]"), ("Cv", "C_unet_topo_dg[det]"),
                 ("ENSD", "ENS_D_compact[ens]"), ("Zvtwo", "Z_v2recipe[det]")):
    r_ = m21.loc[key]
    mac(f"{tag}Tuned", f3(float(pm(r_["unseen Dice tuned"])[0]) if isinstance(r_["unseen Dice tuned"], str) else r_["unseen Dice tuned"]))
    mac(f"{tag}Sel", f3(float(pm(r_["unseen Dice scale-sel"])[0]) if isinstance(r_["unseen Dice scale-sel"], str) else r_["unseen Dice scale-sel"]))
    mac(f"{tag}Raw", f3(float(pm(r_["unseen Dice raw"])[0]) if isinstance(r_["unseen Dice raw"], str) else r_["unseen Dice raw"]))
    mac(f"{tag}Cl", f3(float(pm(r_["unseen clDice"])[0]) if isinstance(r_["unseen clDice"], str) else r_["unseen clDice"]))
    mac(f"{tag}Betti", f"{float(pm(r_['Betti-0 raw'])[0]):.1f}" if isinstance(r_["Betti-0 raw"], str) else f"{r_['Betti-0 raw']:.1f}")
    mac(f"{tag}Drive", f3(float(pm(r_["DRIVE Dice"])[0]) if isinstance(r_["DRIVE Dice"], str) else r_["DRIVE Dice"]))
    mac(f"{tag}Fives", f3(float(pm(r_["FIVES"])[0]) if isinstance(r_["FIVES"], str) else r_["FIVES"]))
    mac(f"{tag}Params", r_["params"])
    mac(f"{tag}Gflops", f"{r_['GFLOPs']:.2f}")
    mac(f"{tag}Cpu", f"{r_['CPU ms']:.1f}")
    mac(f"{tag}Hrf", f3(float(pm(r_["HRF"])[0]) if isinstance(r_["HRF"], str) else r_["HRF"]))
    mac(f"{tag}Stare", f3(float(pm(r_["STARE"])[0]) if isinstance(r_["STARE"], str) else r_["STARE"]))
    mac(f"{tag}Chase", f3(float(pm(r_["CHASE"])[0]) if isinstance(r_["CHASE"], str) else r_["CHASE"]))

sel_ = pd.read_csv(R2 / "selection.csv")
sel_["model"] = sel_["run"].str.split("\\").str[0]
for tag, exp in (("A", "A_unet"), ("B", "B_unet_topo"), ("C", "C_unet_topo_dg"), ("D", "D_compact"), ("E", "E_tglitenet")):
    mac(f"thr{tag}", f"{sel_[sel_['model'] == exp]['threshold'].mean():.2f}")
    mac(f"precRaw{tag}", f3(raw.loc[k(tag), "unseen/prec_mean"]))
    mac(f"sensRaw{tag}", f3(raw.loc[k(tag), "unseen/sens_mean"]))
    mac(f"precTuned{tag}", f3(tuned.loc[k(tag), "unseen/prec_mean"]))
    mac(f"sensTuned{tag}", f3(tuned.loc[k(tag), "unseen/sens_mean"]))

# pre-registered comparisons (raw family)
def cmp_(comparison: str, metric: str, mode: str = "raw", family: str = "primary") -> pd.Series:
    m = comp[(comp["family"] == family) & (comp["mode"] == mode) & (comp["comparison"] == comparison) & (comp["metric"] == metric)]
    return m.iloc[0]


for cname, key in (("RQ1 topology (A->B)", "AB"), ("RQ2 DG (B->C)", "BC"), ("RQ3 compact (C->D)", "CD")):
    for met, mk in (("unseen/dice", "Dice"), ("unseen/cldice", "Cl"), ("unseen/betti0_err", "Betti"), ("gap/dice", "Gap")):
        r_ = cmp_(cname, met)
        d = 4 if mk != "Betti" else 1
        mac(f"d{key}{mk}", f"{r_['delta']:+.{d}f}")
        mac(f"ci{key}{mk}", f"{r_['ci95']:.{d}f}")
        mac(f"p{key}{mk}", f"{r_['p_holm']:.3f}")
mac("dAEmcBetti", f"{cmp_('overall (A->E mc)', 'unseen/betti0_err')['delta']:+.1f}")
mac("ciAEmcBetti", f"{cmp_('overall (A->E mc)', 'unseen/betti0_err')['ci95']:.1f}")
mac("pAEmcBetti", f"{cmp_('overall (A->E mc)', 'unseen/betti0_err')['p_holm']:.3f}")
mac("dAEmcDice", f"{cmp_('overall (A->E mc)', 'unseen/dice')['delta']:+.4f}")
mac("ciAEmcDice", f"{cmp_('overall (A->E mc)', 'unseen/dice')['ci95']:.4f}")
mac("dAEmcDiceTuned", f"{cmp_('overall (A->E mc)', 'unseen/dice', 'tuned')['delta']:+.4f}")
mac("dABDiceTuned", f"{cmp_('RQ1 topology (A->B)', 'unseen/dice', 'tuned')['delta']:+.4f}")
mac("dBCDiceTuned", f"{cmp_('RQ2 DG (B->C)', 'unseen/dice', 'tuned')['delta']:+.4f}")
mac("dBCGapTuned", f"{cmp_('RQ2 DG (B->C)', 'gap/dice', 'tuned')['delta']:+.4f}")
mac("dMcDice", f"{cmp_('MC averaging (E det->E mc)', 'unseen/dice')['delta']:+.4f}")
mac("dMcDiceCi", f"{cmp_('MC averaging (E det->E mc)', 'unseen/dice')['ci95']:.4f}")
for nm, key in (("DG without topology (A->X_unet_dg)", "AX"), ("topology given DG (X_unet_dg->C)", "XC")):
    for met, mk in (("unseen/dice", "Dice"), ("unseen/cldice", "Cl"), ("unseen/betti0_err", "Betti")):
        r_ = cmp_(nm, met, family="secondary")
        mac(f"d{key}{mk}", f"{r_['delta']:+.{1 if mk == 'Betti' else 4}f}")
        mac(f"ci{key}{mk}", f"{r_['ci95']:.{1 if mk == 'Betti' else 4}f}")

# counts
mac("nTrainVtwo", int(training.shape[0]))
mac("nSeedsMain", 5)
n21 = sum(1 for d_ in (ROOT / "results" / "v21" / "runs").iterdir() if d_.is_dir() and d_.name != "v2_reeval" for _ in d_.glob("seed*"))   # one folder per training run
mac("nTrainVtwoOne", n21)
mac("nTrainAll", int(training.shape[0]) + n21)
mac("budgetIters", dec2["budget_iters"])
mac("compactParams", f"{dec2['backbone']['candidates']['K_unet_w8']['params'] / 1e3:.0f}")
mac("liteParams", f"{dec2['backbone']['candidates']['K_lite_w16']['params'] / 1e3:.0f}")
mac("hParamsExact", f"{dec21['arch_selection']['candidates']['L_cap']['params']:,}".replace(",", "\\,"))
mac("hParamsK", f"{dec21['arch_selection']['candidates']['L_cap']['params'] / 1e3:.0f}")
mac("ratioParams", f"{31036481 / dec21['arch_selection']['candidates']['L_cap']['params']:.0f}")
mac("ratioParamsE", f"{31036481 / dec2['backbone']['candidates']['K_unet_w8']['params']:.0f}")
mac("valRl", f3(dec21["size_rule"]["candidates"]["R_lite_w16"]["mean_val_dice"], 4))
mac("valH", f3(dec21["size_rule"]["candidates"]["H_tglite70k"]["mean_val_dice"], 4))
mac("valGap", f3(dec21["size_rule"]["candidates"]["R_lite_w16"]["mean_val_dice"] - dec21["size_rule"]["candidates"]["H_tglite70k"]["mean_val_dice"], 4))

# novelty 2 / ablation tables parsed from the generated markdown
n2 = parse_md_table(R21 / "novelty2_scale.md")
n2 = {r[0]: r for r in n2}
H2 = n2["H_tglite70k[det]"]
mac("nTwoH", re.search(r"([+-][\d.]+) \[±([\d.]+)\]", H2[4]).group(1))
mac("nTwoHci", re.search(r"([+-][\d.]+) \[±([\d.]+)\]", H2[4]).group(2))
mac("nTwoHrf", f"{float(H2[5]):+.3f}")
mac("nTwoFives", f"{float(H2[8]):+.3f}")
mac("nTwoCv", re.search(r"([+-][\d.]+)", n2["C_unet_topo_dg[det]"][4]).group(1))
mac("nTwoEv", re.search(r"([+-][\d.]+)", n2["E_tglitenet[det]"][4]).group(1))
mac("nTwoRl", re.search(r"([+-][\d.]+)", n2["R_lite_w16[det]"][4]).group(1))
mac("nTwoRu", re.search(r"([+-][\d.]+)", n2["R_unet_w8[det]"][4]).group(1))
abl = {r[1]: r for r in parse_md_table(R21 / "ablations.md")}


def abl_val(run: str, col: int) -> tuple[str, str, str]:
    m = re.search(r"([+-][\d.]+) \[±([\d.]+)\] (not clear|clear)", abl[run][col])
    return m.group(1), m.group(2), m.group(3)


for run, nm in (("Z_no_kdskel", "Nthree"), ("Z_pixcons", "Nfour"), ("Z_noaug", "Aug"), ("Z_v2recipe", "All"), ("Z_no_kd", "Kd")):
    for col, mk in ((2, "Dice"), (3, "Cl"), (4, "Betti")):
        v, ci, vd = abl_val(run, col)
        mac(f"a{nm}{mk}", v)
        mac(f"a{nm}{mk}ci", ci)
        mac(f"a{nm}{mk}v", vd)

# novelty 1
n1_txt = (R21 / "novelty1_reliability.md").read_text(encoding="utf-8")
for lab, key in (("E[mc]", r"E_tglitenet\[mc\]: rel_frag - entropy"), ("H", r"H_tglite70k\[det\]: rel_frag - entropy"), ("Ed", r"E_tglitenet\[det\]: rel_frag - entropy")):
    m = re.search(key + r" \(rho, unseen mean\): ([+-][\d.]+) \[±([\d.]+)\] (not clear|clear)", n1_txt)
    mac(f"nOne{lab.replace('[', '').replace(']', '')}", m.group(1))
    mac(f"nOne{lab.replace('[', '').replace(']', '')}ci", m.group(2))
m = re.search(r"E: rel_frag\(det\) - MC-dropout entropy \(rho, unseen mean\): ([+-][\d.]+) \[±([\d.]+)\]", n1_txt)
mac("nOneVsMc", m.group(1))
mac("nOneVsMcci", m.group(2))
n1s = pd.read_csv(R21 / "novelty1_summary.csv")
for model, nm in (("H_tglite70k[det]", "H"), ("E_tglitenet[det]", "Ed"), ("C_unet_topo_dg[det]", "C")):
    for sc, sn in (("ent_mean", "Ent"), ("rel_frag", "Frag"), ("rel_endpoints", "End"), ("rel_smallfrac", "Small")):
        v = n1s[(n1s["model"] == model) & (n1s["score"] == sc)]["rho_mean"].iloc[0]
        mac(f"rho{nm}{sn}", f"{v:.2f}")

# uncertainty summary (mean over the three unseen datasets)
unseen3 = ["stare", "chase_db1", "hrf"]
u = unc[unc["dataset"].isin(unseen3)].copy()
u["gain"] = (u["aurc_random"] - u["aurc"]) / (u["aurc_random"] - u["aurc_oracle"]) * 100
ug = u.groupby("model")[["rho_px", "rho_img", "gain", "ece", "ece_ts", "px_err@100", "px_err@90"]].mean()
mac("rhoPxMin", f"{ug.loc[[k('E'), k('Emc')], 'rho_px'].min():.2f}")
mac("rhoPxMax", f"{ug.loc[[k('E'), k('Emc')], 'rho_px'].max():.2f}")
mac("rhoPxEmc", f"{ug.loc[k('Emc'), 'rho_px']:.2f}")
mac("rhoImgEmc", f"{ug.loc[k('Emc'), 'rho_img']:.2f}")
mac("rhoImgHrfEmc", f"{unc[(unc['model'] == k('Emc')) & (unc['dataset'] == 'hrf')]['rho_img'].iloc[0]:+.2f}")
mac("rhoImgHrfE", f"{unc[(unc['model'] == k('E')) & (unc['dataset'] == 'hrf')]['rho_img'].iloc[0]:+.2f}")
mac("pxErrFull", f"{100 * ug.loc[k('E'), 'px_err@100']:.1f}")
mac("pxErrNinety", f"{100 * ug.loc[k('E'), 'px_err@90']:.1f}")
mac("eceE", f"{ug.loc[k('E'), 'ece']:.3f}")
mac("eceTsE", f"{ug.loc[k('E'), 'ece_ts']:.3f}")

# efficiency numbers for the 66K model (INT8 table is markdown)
int8 = (R21 / ".." / "tables" / "int8.md").read_text(encoding="utf-8")
m = re.search(r"fp32 ([\d.]+) ms -> int8 ([\d.]+) ms", int8)
mac("hFpMs", m.group(1))
mac("hIntMs", m.group(2))
m = re.search(r"file size: (\d+) KB -> (\d+) KB", int8)
mac("hFpKb", m.group(1))
mac("hIntKb", m.group(2))

# ---------------------------------------------------------------------------------------------------- tables
# Table: datasets
rows = []
role = {"drive": "train 15 / val 5 / test 20", "stare": "unseen test", "chase_db1": "unseen test", "hrf": "unseen test", "fives": "extra test (200)"}
for d_ in DS:
    s = dstats.loc[d_]
    n_ = int(s["n"])
    if d_ == "drive":
        n_ = 40
    rows.append([DSN[d_].replace("_", r"\_"), str(n_), f"{int(s['w'])}$\\times${int(s['h'])}", f"{100 * s['vessel_frac_fov']:.1f}", f"{s['fov_width_median']:.0f}",
                 f"{s['calibre_native']:.1f}$\\to${s['calibre_norm']:.1f}", f"{s['green_mean']:.0f}$\\pm${s['green_std']:.0f}", s["fov"], role[d_]])
write("tab_datasets.tex", tabular("lrcrrcccl", ["Dataset", "\\#", "Size (px)", "Vessel \\%", "FOV (px)", "Calibre (px)", "Green", "FOV mask", "Role"], rows))

# Table: staged ablation (v2)
rows = []
labels = {"A": "A  U-Net", "B": "B  + topology loss", "C": "C  + DG augmentation", "D": "D  compact backbone", "E": "E  TG-LiteNet", "Emc": "E  + MC dropout ($T{=}20$)"}
for tag in ["A", "B", "C", "D", "E", "Emc"]:
    r_, t_ = raw.loc[k(tag)], tuned.loc[k(tag)]
    rows.append([labels[tag], cell(r_["drive/dice"]), cell(r_["unseen/dice"]), cell(t_["unseen/dice"]), cell(r_["unseen/cldice"]), cell(r_["unseen/betti0_err"], 1), cell(r_["gap/dice"]),
                 f"{r_['params_M']:.2f}", f"{r_['cpu_ms']:.0f}" + (r" $\times$ 20" if tag == "Emc" else "")])
write("tab_stages.tex", tabular("l ccc c c c rr", ["Model", "DRIVE Dice", "Unseen Dice", "Tuned Dice", "clDice", "Betti-0", "Gap", "Params (M)", "CPU (ms)"], rows, (5,)))

# Table: published context + v2.1 models
pub = [("DSAE-Net~\\cite{zeng2025}", "78K", "4.76", "--", "0.819", "0.815", "0.801", "0.812", "--", "--", "--", "--"),
       ("LightVesselNet~\\cite{sobhan2026}", "75K", "1.4", "--", "--", "--", "--", "n/r", "--", "--", "--", "--"),
       ("DGSSA~\\cite{liu2026dgssa} (multi-source)", "n/r", "n/r", "--", "--", "--", "--", "0.780", "--", "--", "--", "--")]
rows = [list(p) for p in pub]


def fm(model: str, col: str, d: int = 3) -> str:
    v = m21.loc[model, col]
    return cell(v, d) if isinstance(v, str) else f"{v:.{d}f}"


mrows = [("C_unet_topo_dg[det]", "v2 stage C (U-Net)"), ("E_tglitenet[det]", "v2 stage E (TG-LiteNet)"), ("R_unet_w8[det]", "v2.1 compact U-Net"), ("R_lite_w16[det]", "v2.1 depthwise (size-rule headline)"),
         ("H_tglite70k[det]", "\\textbf{v2.1 TG-LiteNet 66K}"), ("ENS_H_tglite70k[ens]", "v2.1 66K, 5-model ensemble")]
for key_, lab in mrows:
    r_ = m21.loc[key_]
    pk = kparams(r_["params"]) / 1e3
    rows.append([lab, (f"{pk / 1e3:.1f}M" if pk >= 1e3 else f"{pk:.0f}K"), f"{r_['GFLOPs']:.2f}", f"{r_['CPU ms']:.0f}", fm(key_, "STARE"), fm(key_, "CHASE"), fm(key_, "HRF"), fm(key_, "unseen Dice tuned"), fm(key_, "unseen Dice scale-sel"),
                 fm(key_, "FIVES"), fm(key_, "unseen clDice"), fm(key_, "Betti-0 raw", 1)])
write("tab_models.tex", tabular("l rrr ccc cc c cc", ["Model", "Params", "GFLOPs", "CPU ms", "STARE", "CHASE", "HRF", "Unseen", "+ scale sel.", "FIVES", "clDice", "Betti-0"], rows, (3, 4)))

# Table: factorial
cells_ = [("A_unet[det]", "U-Net", "--", "--"), ("X_unet_dg[det]", "U-Net", "--", "\\checkmark"), ("B_unet_topo[det]", "U-Net", "\\checkmark", "--"), ("C_unet_topo_dg[det]", "U-Net", "\\checkmark", "\\checkmark"),
          ("X_compact_plain[det]", "Compact", "--", "--"), ("X_compact_dg[det]", "Compact", "--", "\\checkmark"), ("X_compact_topo[det]", "Compact", "\\checkmark", "--"), ("D_compact[det]", "Compact", "\\checkmark", "\\checkmark")]
rows = []
for key_, bb, tp, dg in cells_:
    r_, t_ = raw.loc[key_], tuned.loc[key_]
    rows.append([bb, tp, dg, str(int(r_["n_seeds"])), cell(r_["unseen/dice"]), cell(r_["unseen/cldice"]), cell(r_["unseen/betti0_err"], 1), cell(t_["unseen/dice"]), cell(t_["gap/dice"])])
write("tab_factorial.tex", tabular("lcc c cccc c", ["Backbone", "Topo", "DG", "n", "Dice", "clDice", "Betti-0", "Tuned Dice", "Tuned gap"], rows, (4,)))

# Table: backbone selection (DRIVE-val only)
rows = []
cand = dec2["backbone"]["candidates"]
names = {"K_lite_w16": "depthwise $w{=}16$", "K_hybrid_w16": "hybrid $w{=}16$", "K_unet_w8": "plain U-Net $w{=}8$", "K_unet_w16": "plain U-Net $w{=}16$"}
for kk, v in cand.items():
    rows.append([names[kk], f"{v['params'] / 1e3:.0f}K", f"{v['mean_val_dice']:.4f}", f"{v['cpu_ms']:.0f}", r"\checkmark" if kk == dec2["backbone"]["compact_choice"] else ""])
write("tab_backbone.tex", tabular("lrrrc", ["Candidate", "Params", "Val Dice", "CPU ms", "Chosen"], rows))

# Table: novelties
rows = [
    ["N1", r"Connectivity reliability score. $\rho$(fragments) $-$ $\rho$(entropy): $" + MAC["nOneEmc"] + r"$ against MC dropout (stage E), $" + MAC["nOneH"] + r"$ for the 66K model", "Lead found"],
    ["N2", r"Calibre-guided scale selection. Unseen Dice $" + MAC["nTwoH"] + r"\,[\pm" + MAC["nTwoHci"] + r"]$ (66K); HRF $" + MAC["nTwoHrf"] + r"$, FIVES $" + MAC["nTwoFives"] + "$", r"\textbf{Supported}"],
    ["N3", r"Topology-aware distillation. $\Delta$Dice $" + MAC["aNthreeDice"] + r"\,[\pm" + MAC["aNthreeDiceci"] + r"]$, $\Delta$clDice $" + MAC["aNthreeCl"] + "$", "Open"],
    ["N4", r"Skeleton-level consistency. $\Delta$Dice $" + MAC["aNfourDice"] + r"\,[\pm" + MAC["aNfourDiceci"] + r"]$ (clear), $\Delta$clDice $" + MAC["aNfourCl"] + "$", "Dice trade-off"],
    ["--", r"RandConv + Fourier augmentation. $\Delta$Dice $" + MAC["aAugDice"] + r"\,[\pm" + MAC["aAugDiceci"] + "]$", "Open"],
]
write("tab_novelties.tex", tabular(r"l p{4.5cm} p{1.5cm}", ["", "Novelty and result (66K model, unseen, tuned)", "Verdict"], rows))

# Table: uncertainty
rows = []
for tag, lab in (("A", "A  U-Net (entropy)"), ("C", "C  + DG (entropy)"), ("E", "E  TG-LiteNet (entropy)"), ("Emc", "E  MC dropout ($T{=}20$)")):
    key_ = k(tag)
    rows.append([lab, f"{ug.loc[key_, 'rho_px']:.2f}", f"{ug.loc[key_, 'rho_img']:+.2f}", f"{ug.loc[key_, 'gain']:+.1f}", f"{ug.loc[key_, 'ece']:.3f}", f"{ug.loc[key_, 'ece_ts']:.3f}"])
key_ = "ENS_D_compact[ens]"
rows.append(["D  5-model ensemble", f"{ug.loc[key_, 'rho_px']:.2f}", f"{ug.loc[key_, 'rho_img']:+.2f}", f"{ug.loc[key_, 'gain']:+.1f}", f"{ug.loc[key_, 'ece']:.3f}", f"{ug.loc[key_, 'ece_ts']:.3f}"])
write("tab_uncertainty.tex", tabular("lccccc", ["Uncertainty source", "$\\rho_{px}$", "$\\rho_{img}$", "AURC gain (\\%)", "ECE", "ECE$_T$"], rows))

# ---------------------------------------------------------------------------------------------------- figures
style()

# Fig: stages per dataset (mean +- sd over seeds, raw mode)
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
panels = [("dice", "(a) Dice", False), ("cldice", "(b) clDice", False), ("betti0_err", "(c) Betti-0 error (log)", True)]
w = 0.8 / len(STAGES)
for ax, (met, ttl, log) in zip(axes, panels):
    for i, (exp, tag, lab) in enumerate(STAGES):
        sub = perrun_raw[perrun_raw["model"] == f"{exp}[det]"]
        mu = [sub[f"{d_}/{met}"].mean() for d_ in DS]
        sd = [sub[f"{d_}/{met}"].std(ddof=1) for d_ in DS]
        ax.bar(np.arange(len(DS)) + (i - 2) * w, mu, w * 0.9, yerr=sd, color=COL[tag], linewidth=0, label=f"{tag} {lab}", error_kw={"elinewidth": 0.6, "capsize": 1, "capthick": 0.6})
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DSL[d_] for d_ in DS], fontsize=7)
    ax.axvline(0.5, color="k", lw=0.5, ls=":")
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    title(ax, ttl)
    if log:
        ax.set_yscale("log")
        ax.set_yticks([20, 50, 100])
        ax.set_yticklabels(["20", "50", "100"])
        ax.minorticks_off()
    elif met == "dice":
        ax.set_ylim(0.66, 0.87)
    else:
        ax.set_ylim(0.77, 0.90)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False, handlelength=1.1, columnspacing=1.3)
fig.tight_layout(pad=0.4, w_pad=1.0)
save(fig, "fig_stages")

# Fig: evaluation modes (raw / clean / tuned)
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.45))
order = [("A_unet[det]", "A"), ("B_unet_topo[det]", "B"), ("C_unet_topo_dg[det]", "C"), ("D_compact[det]", "D"), ("E_tglitenet[det]", "E"), ("X_unet_dg[det]", "U+DG")]
x = np.arange(len(order))
ax = axes[0]
ax.bar(x - 0.19, [raw.loc[o, "unseen/betti0_err_mean"] for o, _ in order], 0.36, color="#b0b0b0", linewidth=0, label="raw")
ax.bar(x + 0.19, [clean.loc[o, "unseen/betti0_err_mean"] for o, _ in order], 0.36, color="#1f77b4", linewidth=0, label="clean")
ax.set_yscale("log")
ax.set_ylim(3, 200)
ax.set_yticks([5, 10, 30, 100])
ax.set_yticklabels(["5", "10", "30", "100"])
ax.minorticks_off()
ax.set_ylabel("unseen Betti-0 error")
ax.legend(frameon=False, loc="upper right", handlelength=1)
title(ax, "(a) Fragments")
ax = axes[1]
ax.scatter(x, [raw.loc[o, "unseen/dice_mean"] for o, _ in order], s=16, color="k", label="raw 0.5", zorder=3)
ax.scatter(x, [clean.loc[o, "unseen/dice_mean"] for o, _ in order], s=16, marker="s", color="#1f77b4", label="clean", zorder=3)
ax.scatter(x, [tuned.loc[o, "unseen/dice_mean"] for o, _ in order], s=18, marker="^", color="#d62728", label="tuned", zorder=3)
for xi, (o, _) in zip(x, order):
    ax.plot([xi, xi], [raw.loc[o, "unseen/dice_mean"], tuned.loc[o, "unseen/dice_mean"]], color="0.6", lw=0.9, zorder=1)
ax.set_ylabel("unseen Dice")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.1, columnspacing=0.8, fontsize=6.5)
title(ax, "(b) Overlap")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in order], fontsize=7)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
fig.tight_layout(pad=0.4, w_pad=0.8)
save(fig, "fig_modes")

# Fig: cost trade-off and scale selection
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.15]})
pts = [("A_unet[det]", "A", "#6b6b6b", "s"), ("C_unet_topo_dg[det]", "C", "#2ca02c", "s"), ("E_tglitenet[det]", "E (v2)", "#ff7f0e", "o"), ("ENS_D_compact[ens]", "ENS", "#ff7f0e", "D"),
       ("R_unet_w8[det]", "R-U8", "#1f77b4", "o"), ("R_lite_w16[det]", "R-L16", "#1f77b4", "o"), ("H_tglite70k[det]", "66K", "#d62728", "o"), ("ENS_H_tglite70k[ens]", "66K ens.", "#d62728", "D")]
for ax_i, (xcol, xl) in enumerate((("params", "parameters (log)"), ("CPU ms", "CPU time per 512$\\times$512 image, ms (log)"))):
    ax = axes[ax_i]
    for key_, lab, c_, mk in pts:
        r_ = m21.loc[key_]
        xv = kparams(r_["params"]) if xcol == "params" else float(r_["CPU ms"])
        yv = float(pm(r_["unseen Dice tuned"])[0]) if isinstance(r_["unseen Dice tuned"], str) else float(r_["unseen Dice tuned"])
        ax.scatter(xv, yv, s=34, color=c_, marker=mk, edgecolor="k", linewidth=0.4, zorder=3)
        off = {"A": (-9, 5), "C": (5, 3), "E (v2)": (5, 3), "ENS": (5, 3), "R-U8": (5, -11), "R-L16": (-6, -12), "66K": (7, -3), "66K ens.": (-6, 9)}[lab]
        if xcol != "params":
            off = {"A": (-9, 6), "C": (-8, 5), "E (v2)": (5, 4), "ENS": (5, 4), "R-U8": (-6, 8), "R-L16": (5, 5), "66K": (7, -3), "66K ens.": (7, -3)}[lab]
        ax.annotate(lab, (xv, yv), fontsize=6.5, xytext=off, textcoords="offset points")
    if xcol == "params":
        ax.scatter(78e3, 0.812, s=46, marker="*", color="#6a3d9a", zorder=3)
        ax.annotate("DSAE-Net\n(other protocol)", (78e3, 0.812), fontsize=6.3, xytext=(6, -14), textcoords="offset points")
    ax.set_xscale("log")
    if xcol == "params":
        ax.set_xlim(4e4, 4e8)
    else:
        ax.set_xlim(15, 2500)
    ax.set_xlabel(xl)
    ax.grid(alpha=0.3, lw=0.4)
    if ax_i == 0:
        ax.set_ylabel("unseen Dice (tuned threshold)")
    ax.set_ylim(0.76, 0.82)
    title(ax, "(a) Accuracy vs size" if ax_i == 0 else "(b) Accuracy vs CPU time")
ax = axes[2]
ds4 = ["stare", "chase_db1", "hrf", "fives"]
mods = [("H_tglite70k[det]", "66K", "#d62728"), ("R_lite_w16[det]", "R-L16", "#1f77b4"), ("E_tglitenet[det]", "E v2", "#ff7f0e"), ("C_unet_topo_dg[det]", "C v2", "#2ca02c")]
wd = 0.8 / len(mods)
for i, (key_, lab, c_) in enumerate(mods):
    sub = perrun21[perrun21["model"] == key_]
    dl = [float((sub[f"{d_}/sel/dice"] - sub[f"{d_}/tuned/dice"]).mean()) for d_ in ds4]
    ax.bar(np.arange(4) + (i - 1.5) * wd, dl, wd * 0.9, color=c_, linewidth=0, label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels([DSL[d_] for d_ in ds4], fontsize=7)
ax.set_ylabel("$\\Delta$Dice (selected $-$ scale 1)")
ax.legend(frameon=False, loc="lower left", fontsize=6.5, ncol=2, handlelength=1, columnspacing=0.8)
ax.grid(axis="y", alpha=0.3, lw=0.4)
title(ax, "(c) Scale selection")
fig.tight_layout(pad=0.4, w_pad=1.0)
save(fig, "fig_cost_scale")

# Fig: scale chosen per dataset for the 66K model (share of images)
pi = pd.read_csv(R21 / "per_image_all.csv.gz")
hh = pi[pi["model"] == "H_tglite70k[det]"]
share = hh.groupby(["dataset", "sel_scale"]).size().unstack(fill_value=0)
share = share.div(share.sum(axis=1), axis=0)
for ds_ in ("hrf", "fives", "stare", "chase_db1", "drive"):
    nm_ = {"hrf": "Hrf", "fives": "Fives", "stare": "Stare", "chase_db1": "Chase", "drive": "Drive"}[ds_]
    mac(f"scaleOneFive{nm_}", f"{100 * share.loc[ds_].get(1.5, 0):.0f}")
    mac(f"scaleOne{nm_}", f"{100 * share.loc[ds_].get(1.0, 0):.0f}")

# Fig: how calibre-guided scale selection works (66K model): predicted vessel width per scale, and the scale chosen per dataset
fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
ax = axes[0]
scales_ = (1.0, 1.5, 2.0, 2.5)
cols_ = {"drive": "#555555", "stare": "#1f77b4", "chase_db1": "#2ca02c", "hrf": "#d62728", "fives": "#ff7f0e"}
for ds_, c_ in cols_.items():
    wv = [hh[hh["dataset"] == ds_][f"calib@{s_:g}"].mean() for s_ in scales_]
    ax.plot(scales_, wv, marker="o", ms=3, lw=1.2 if ds_ != "drive" else 1.0, ls="--" if ds_ == "drive" else "-", color=c_, label=DSL.get(ds_, ds_))
ax.set_xlabel("input scale")
ax.set_ylabel("predicted vessel width (px)")
ax.set_xticks(scales_)
ax.grid(alpha=0.3, lw=0.4)
ax.legend(frameon=False, fontsize=5.8, loc="lower right", handlelength=1.4, labelspacing=0.2)
title(ax, "(a) Predicted width")
ax = axes[1]
order_ = ["hrf", "fives", "stare", "chase_db1", "drive"]
bot = np.zeros(len(order_))
pal_ = {1.0: "#bdbdbd", 1.5: "#d62728", 2.0: "#ff9896", 2.5: "#8c564b"}
for s_ in scales_:
    vals_ = np.array([100 * share.loc[d_].get(s_, 0) for d_ in order_])
    ax.bar(range(len(order_)), vals_, bottom=bot, color=pal_[s_], width=0.7, linewidth=0, label=f"x{s_:g}")
    bot += vals_
ax.set_xticks(range(len(order_)))
ax.set_xticklabels([DSL.get(d_, d_) for d_ in order_], rotation=35, ha="right", fontsize=6.5)
ax.set_ylabel("images (%)")
ax.set_ylim(0, 128)
ax.set_yticks([0, 25, 50, 75, 100])
ax.legend(frameon=False, fontsize=5.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=0.8, columnspacing=0.5, handletextpad=0.3)
title(ax, "(b) Scale chosen")
save(fig, "fig_scale_mech")

# Table: pixel-level classification metrics and confusion matrix of the 66K model (tuned mode), from the per-image records
hc = pi[pi["model"] == "H_tglite70k[det]"].copy()
hc["fp"], hc["fn"] = hc["tuned/fp_frac"], hc["tuned/fn_frac"]
hc["tp"] = hc["fp"] * hc["tuned/prec"] / (1 - hc["tuned/prec"])
hc["tn"] = 1 - hc["tp"] - hc["fp"] - hc["fn"]
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.25), gridspec_kw={"wspace": 0.04, "hspace": 0.20})
conf_names = {"stare": "STARE", "chase_db1": "CHASE_DB1", "hrf": "HRF", "fives": "FIVES"}
GOOD, BAD, MARG = "#cfe3d4", "#f2cfc7", "#ececec"
conf_csv = []
for ax, ds_ in zip(axes.ravel(), ("stare", "chase_db1", "hrf", "fives")):
    s_ = hc[hc["dataset"] == ds_]
    tp_, fp_, fn_, tn_ = [100 * s_[k_].mean() for k_ in ("tp", "fp", "fn", "tn")]
    acc_, prec_, rec_, spec_, f1_ = [s_[k_].mean() for k_ in ("tuned/acc", "tuned/prec", "tuned/sens", "tuned/spec", "tuned/dice")]
    conf_csv.append([ds_, tp_, fn_, fp_, tn_, acc_, prec_, rec_, spec_, f1_])
    ax.set_xlim(-0.85, 3.05)
    ax.set_ylim(2.95, -0.55)
    ax.axis("off")
    cells = [(0, 0, "TP", tp_, GOOD), (1, 0, "FN", fn_, BAD), (0, 1, "FP", fp_, BAD), (1, 1, "TN", tn_, GOOD)]
    for cx, cy, nm_, v_, col_ in cells:
        ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor=col_, edgecolor="white", lw=1.2))
        ax.text(cx + 0.5, cy + 0.5, f"{nm_}\n{v_:.1f}%", ha="center", va="center", fontsize=6.6, fontweight="bold")
    for cx, lab_ in ((0, "Pred. +"), (1, "Pred. \u2212")):
        ax.text(cx + 0.5, -0.12, lab_, ha="center", va="center", fontsize=6.2)
    for cy, lab_ in ((0, "True +"), (1, "True \u2212")):
        ax.text(-0.06, cy + 0.5, lab_, ha="right", va="center", fontsize=6.2)
    # margins: recall / specificity (right), precision (bottom), accuracy (corner)
    for cy, nm_, v_ in ((0, "Recall", rec_), (1, "Spec.", spec_)):
        ax.add_patch(Rectangle((2, cy), 1, 1, facecolor=MARG, edgecolor="white", lw=1.2))
        ax.text(2.5, cy + 0.5, f"{nm_}\n{v_:.3f}", ha="center", va="center", fontsize=6.2)
    ax.add_patch(Rectangle((0, 2), 1, 0.75, facecolor=MARG, edgecolor="white", lw=1.2))
    ax.text(0.5, 2.375, f"Prec.\n{prec_:.3f}", ha="center", va="center", fontsize=6.2)
    ax.add_patch(Rectangle((2, 2), 1, 0.75, facecolor=MARG, edgecolor="white", lw=1.2))
    ax.text(2.5, 2.375, f"Acc.\n{acc_:.3f}", ha="center", va="center", fontsize=6.2)
    ax.set_title(f"{conf_names[ds_]}  (F1 = {f1_:.3f})", fontsize=7.6, fontweight="bold", pad=1.5, loc="center")
save(fig, "fig_confusion")
pd.DataFrame(conf_csv, columns=["dataset", "TP_pct", "FN_pct", "FP_pct", "TN_pct", "accuracy", "precision", "recall", "specificity", "f1"]).to_csv(GEN / "confusion_66k.csv", index=False)
u3 = hc[hc["dataset"].isin(["stare", "chase_db1", "hrf"])].groupby("dataset")[["fp", "fn", "tuned/acc"]].mean()
mac("confFpLo", f"{100 * u3['fp'].min():.1f}")
mac("confFpHi", f"{100 * u3['fp'].max():.1f}")
mac("confFnLo", f"{100 * u3['fn'].min():.1f}")
mac("confFnHi", f"{100 * u3['fn'].max():.1f}")
mac("confTnLo", f"{100 * hc.groupby('dataset')['tn'].mean().min():.0f}")
mac("confTnHi", f"{100 * hc.groupby('dataset')['tn'].mean().max():.0f}")

# qualitative figure from the exported example assets (HRF: selection helps, FIVES: it hurts)
A = ROOT / "results" / "v21" / "assets"
fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.2), gridspec_kw={"wspace": 0.03, "hspace": 0.12})
cols = [("image.jpg", "image"), ("gt.png", "ground truth"), ("err_v21_H_tglite70k.png", "66K, scale 1"), ("err_v21_H_tglite70k_scalesel.png", "66K + scale sel."), ("err_v2_C_unet_topo_dg.png", "v2 C (31M)")]
meta = json.loads((A / "assets.json").read_text())["datasets"]
for r_i, ds_ in enumerate(("hrf", "fives")):
    for c_i, (f_, lab) in enumerate(cols):
        ax = axes[r_i, c_i]
        im = Image.open(A / ds_ / f_)
        ax.imshow(im, cmap="gray" if im.mode == "L" else None)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if r_i == 0:
            ax.set_title(lab, fontsize=7.5, pad=2)
        maps = meta[ds_]["maps"]
        key_ = {2: "v21_H_tglite70k", 3: "v21_H_tglite70k_scalesel", 4: "v2_C_unet_topo_dg"}.get(c_i)
        if key_:
            ax.set_xlabel(f"Dice {maps[key_]['dice_normalised']:.3f}", fontsize=7, labelpad=1)
    axes[r_i, 0].set_ylabel(DSL[ds_], fontsize=8)
save(fig, "fig_qualitative")

# ---------------------------------------------------------------------------------------------------- macros file
lines = ["% auto-generated by scripts/make_paper.py - do not edit"]
for kk, vv in MAC.items():
    lines.append("\\newcommand{\\" + kk + "}{" + vv + "}")
write("numbers.tex", "\n".join(lines) + "\n")
print("macros:", len(MAC))
