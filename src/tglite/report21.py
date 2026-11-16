"""v2.1 report: everything under results/v21/ is generated here from runs/ (never edited by hand).

tables/main.md          all models: native-resolution metrics (raw / tuned / scale-selected), network-resolution Dice, cost
tables/sota.md          published numbers (context only - different protocols) next to ours
tables/novelty1_reliability.md   label-free connectivity scores vs entropy / MC dropout / ensembles (image error ranking)
tables/novelty2_scale.md         topology-guided scale selection vs scale 1
tables/ablations.md     novelties 3, 4 and the new augmentation: headline vs one-ingredient-removed (seed-paired)
tables/decisions.md     DRIVE-val decisions of v2.1 (architecture, loss weights, size rule)
tables/int8.md          INT8 deployment check
figures/*.png           Dice vs parameters, risk-coverage, scale selection
summary.md              short, plain-language summary of the numbers above
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

UNSEEN = ["stare", "chase_db1", "hrf"]
ALL_DS = ["drive", "stare", "chase_db1", "hrf", "fives"]

# Published numbers quoted in the preliminary report / v1 paper (context only: protocols differ, see PROTOCOL_v2.1.md)
SOTA = [
    {"model": "DSAE-Net [19]", "protocol": "cross-dataset tests", "params": "78K", "gflops": "4.76", "stare": 81.92, "chase_db1": 81.46,
     "hrf": 80.12, "unseen": 81.17, "cldice": "-", "betti0": "-", "uncertainty": "no"},
    {"model": "LightVesselNet [12]", "protocol": "train one, test four", "params": "75K", "gflops": "1.4", "stare": "-", "chase_db1": "-",
     "hrf": "-", "unseen": "-", "cldice": "-", "betti0": "-", "uncertainty": "no"},
    {"model": "TFFM [21]", "protocol": "source-only (Fundus-AVSeg)", "params": "n/r", "gflops": "n/r", "stare": "-", "chase_db1": "-",
     "hrf": "-", "unseen": "79.9-82.2", "cldice": "68.7-73.6", "betti0": "24.2-50.5", "uncertainty": "no"},
    {"model": "DGSSA [15]", "protocol": "multi-source DG", "params": "n/r", "gflops": "n/r", "stare": "-", "chase_db1": "-", "hrf": "-",
     "unseen": 77.98, "cldice": "-", "betti0": "-", "uncertainty": "no"},
    {"model": "WaveRNet [16]", "protocol": "multi-source DG (no HRF)", "params": ">90M", "gflops": "n/r", "stare": "-", "chase_db1": "-",
     "hrf": "-", "unseen": 69.49, "cldice": "-", "betti0": "-", "uncertainty": "no"},
    {"model": "TG-LiteNet v1 (E, MC)", "protocol": "source-only DRIVE", "params": "0.21M", "gflops": "~5.5", "stare": "-", "chase_db1": "-",
     "hrf": "-", "unseen": 75.6, "cldice": 83.3, "betti0": 42.7, "uncertainty": "MC dropout"},
]


# ------------------------------------------------------------------------------------------------ loading
def _collect(v2_runs: Path, root21: Path) -> pd.DataFrame:
    frames = []
    pats = [(v2_runs, "*/seed*/eval21/per_image_*.csv", "v2"), (v2_runs, "ENS_*/eval21/per_image_*.csv", "v2"),
            (root21, "*/seed*/eval21/per_image_*.csv", "v2.1"), (root21, "ENS_*/eval21/per_image_*.csv", "v2.1")]
    for base, pat, ver in pats:
        for f in base.glob(pat):
            if ver == "v2" and "v21" in f.parts:
                continue
            df = pd.read_csv(f)
            variant = f.stem.replace("per_image_", "")
            if f.parent.parent.name.startswith("seed"):
                exp, seed = f.parent.parent.parent.name, int(f.parent.parent.name[4:])
            else:
                exp, seed = f.parent.parent.name, 0
            df["exp"], df["seed"], df["variant"], df["version"] = exp, seed, variant, ver
            df["model"] = f"{exp}[{variant}]"
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _per_run(df: pd.DataFrame, cols) -> pd.DataFrame:
    """mean per (model, seed, dataset) -> wide with dataset prefixes, + unseen means + gap."""
    g = df.groupby(["model", "seed", "dataset"])[cols].mean().reset_index()
    out = []
    for (m, s), sub in g.groupby(["model", "seed"]):
        r = {"model": m, "seed": s}
        for _, row in sub.iterrows():
            for c in cols:
                r[f"{row['dataset']}/{c}"] = row[c]
        for c in cols:
            vals = [r.get(f"{d}/{c}") for d in UNSEEN]
            r[f"unseen/{c}"] = float(np.mean(vals)) if all(v is not None and not pd.isna(v) for v in vals) else np.nan
        out.append(r)
    pr = pd.DataFrame(out)
    for mode in ("raw", "tuned", "sel"):
        if f"drive/{mode}/dice" in pr and f"unseen/{mode}/dice" in pr:
            pr[f"gap/{mode}/dice"] = pr[f"drive/{mode}/dice"] - pr[f"unseen/{mode}/dice"]
    return pr


def _ms(v) -> str:
    v = [x for x in v if not pd.isna(x)]
    if not v:
        return "-"
    return f"{np.mean(v):.3f}" if len(v) == 1 else f"{np.mean(v):.3f}±{np.std(v, ddof=1):.3f}"


def _tci(d) -> str:
    from scipy import stats
    d = np.asarray([x for x in d if not pd.isna(x)], float)
    if len(d) < 2:
        return f"{d.mean():+.4f} (n={len(d)})" if len(d) else "-"
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    clear = "clear" if abs(d.mean()) > h else "not clear"
    return f"{d.mean():+.4f} [±{h:.4f}] {clear} (n={len(d)})"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _cost(v2_runs: Path, root21: Path, exp: str, dec21: dict) -> dict:
    for base in (root21, v2_runs):
        f = base / exp / "efficiency.json"
        if f.exists():
            return json.loads(f.read_text())
    if exp.startswith(("Z_", "H_", "ENS_H")):
        f = root21 / "H_tglite70k" / "efficiency.json"
        if f.exists():
            return json.loads(f.read_text())
        a = dec21.get("arch")
        f = root21 / a / "efficiency.json" if a else None
        if f and f.exists():
            return json.loads(f.read_text())
    if exp.startswith("ENS_"):
        base = {"ENS_D_compact": "D_compact", "ENS_A_unet": "A_unet"}.get(exp)
        if base and (v2_runs / base / "efficiency.json").exists():
            r = json.loads((v2_runs / base / "efficiency.json").read_text())
            return {**r, "params": r["params"] * 5, "gflops": r["gflops"] * 5, "cpu_4t": {"median_ms": r["cpu_4t"]["median_ms"] * 5}}
    return {}


# ------------------------------------------------------------------------------------------------ analyses
def _aurc(score: np.ndarray, err: np.ndarray) -> tuple[float, float, float]:
    """(AURC, random, oracle): images are removed most-unreliable-first; risk = mean error of the kept images."""
    n = len(err)
    def curve(order):
        kept = err[order]            # ascending reliability problem: keep the lowest scores first
        return float(np.mean([kept[:k].mean() for k in range(1, n + 1)]))
    return curve(np.argsort(score, kind="stable")), float(err.mean()), curve(np.argsort(err, kind="stable"))


def novelty1(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from scipy.stats import spearmanr
    scores = ["rel_frag", "rel_endpoints", "rel_smallfrac", "ent_mean", "var_mean"]
    rows, per_seed = [], []
    for (m, s), sub in df.groupby(["model", "seed"]):
        for d in ALL_DS + ["unseen_pooled"]:
            x = sub[sub["dataset"].isin(UNSEEN)] if d == "unseen_pooled" else sub[sub["dataset"] == d]
            if len(x) < 4:
                continue
            err = 1 - x["tuned/dice"].to_numpy(float)
            for sc in scores:
                if sc not in x or x[sc].isna().all():
                    continue
                v = x[sc].to_numpy(float)
                rho = spearmanr(v, err).statistic if np.std(v) > 0 else np.nan
                a, r, o = _aurc(v, err)
                gain = 100 * (r - a) / (r - o) if r > o else np.nan
                per_seed.append({"model": m, "seed": s, "dataset": d, "score": sc, "rho": rho, "aurc_gain_pct": gain})
    ps = pd.DataFrame(per_seed)
    if ps.empty:
        return ps, ps, {}
    tab = ps.groupby(["model", "score", "dataset"])[["rho", "aurc_gain_pct"]].mean().reset_index()
    um = ps[ps["dataset"].isin(UNSEEN)].groupby(["model", "seed", "score"])[["rho", "aurc_gain_pct"]].mean().reset_index()
    summ = um.groupby(["model", "score"])[["rho", "aurc_gain_pct"]].agg(["mean", "std", "count"]).reset_index()
    summ.columns = ["model", "score", "rho_mean", "rho_sd", "n", "gain_mean", "gain_sd", "n2"]
    # pre-registered verdicts: fragment density vs the model's own entropy (det) and vs MC dropout (E)
    verdict = {}
    for m in um["model"].unique():
        a = um[(um["model"] == m) & (um["score"] == "rel_frag")].set_index("seed")["rho"]
        b = um[(um["model"] == m) & (um["score"] == "ent_mean")].set_index("seed")["rho"]
        common = a.index.intersection(b.index)
        if len(common):
            verdict[f"{m}: rel_frag - entropy (rho, unseen mean)"] = _tci((a[common] - b[common]).to_numpy())
    e_det = um[(um["model"] == "E_tglitenet[det]") & (um["score"] == "rel_frag")].set_index("seed")["rho"]
    e_mc = um[(um["model"] == "E_tglitenet[mc]") & (um["score"] == "ent_mean")].set_index("seed")["rho"]
    common = e_det.index.intersection(e_mc.index)
    if len(common):
        verdict["E: rel_frag(det) - MC-dropout entropy (rho, unseen mean)"] = _tci((e_det[common] - e_mc[common]).to_numpy())
    return tab, summ, verdict


def novelty2(pr: pd.DataFrame, models) -> pd.DataFrame:
    rows = []
    for m in models:
        x = pr[pr["model"] == m]
        if x.empty or "unseen/sel/dice" not in x:
            continue
        r = {"model": m, "n": len(x), "unseen tuned (scale 1)": _ms(x["unseen/tuned/dice"]), "unseen scale-selected": _ms(x["unseen/sel/dice"]),
             "delta unseen": _tci(x["unseen/sel/dice"] - x["unseen/tuned/dice"])}
        for d in ("hrf", "stare", "chase_db1", "fives", "drive"):
            if f"{d}/sel/dice" in x:
                r[f"delta {d}"] = f"{(x[f'{d}/sel/dice'] - x[f'{d}/tuned/dice']).mean():+.4f}"
        rows.append(r)
    return pd.DataFrame(rows)


def ablations(pr: pd.DataFrame) -> pd.DataFrame:
    h = pr[pr["model"] == "H_tglite70k[det]"].set_index("seed")
    rows = []
    what = {"Z_no_kdskel": "novelty 3 (skeleton distillation)", "Z_no_kd": "all distillation", "Z_pixcons": "novelty 4 (skeleton consistency)",
            "Z_noaug": "new augmentation (RandConv + Fourier)", "Z_v2recipe": "all v2.1 training changes"}
    for z, desc in what.items():
        zz = pr[pr["model"] == f"{z}[det]"].set_index("seed")
        common = h.index.intersection(zz.index)
        if not len(common):
            continue
        r = {"ablation (removed)": desc, "run": z}
        for c, lab in (("unseen/tuned/dice", "unseen Dice"), ("unseen/tuned/cldice", "unseen clDice"),
                       ("unseen/raw/betti0_err", "unseen Betti-0 (raw)"), ("gap/tuned/dice", "gap")):
            if c in h and c in zz:
                r[f"headline - ablation: {lab}"] = _tci((h.loc[common, c] - zz.loc[common, c]).to_numpy())
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures
def _figures(pr, df, cost_of, out: Path, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11})
    # Dice vs parameters
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for m in sorted(pr["model"].unique()):
        exp = m.split("[")[0]
        c = cost_of(exp)
        if not c or "unseen/tuned/dice" not in pr:
            continue
        y = pr.loc[pr["model"] == m, "unseen/tuned/dice"].mean() * 100
        if pd.isna(y):
            continue
        new = (exp.startswith(("H_", "R_", "ENS_H")))
        ax.scatter(c["params"], y, s=60 if new else 35, marker="o" if new else "s", color="tab:red" if exp.startswith(("H_", "ENS_H")) else
                   ("tab:orange" if new else "tab:gray"), zorder=3)
        if not exp.startswith("Z_"):
            ax.annotate(m.replace("[det]", ""), (c["params"], y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.scatter([78e3], [81.17], marker="*", s=150, color="tab:blue", zorder=3)
    ax.annotate("DSAE-Net (published,\ndifferent protocol)", (78e3, 81.17), fontsize=8, xytext=(5, -18), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("unseen Dice (%), threshold from DRIVE-val")
    ax.set_title("Accuracy vs size (higher and further left is better)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "dice_vs_params.png", dpi=150)
    plt.close(fig)
    # risk-coverage (unseen pooled) for the headline, seed 0
    sub = df[(df["model"] == "H_tglite70k[det]") & (df["seed"] == 0) & (df["dataset"].isin(UNSEEN))]
    if len(sub) > 4:
        err = 1 - sub["tuned/dice"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        n = len(err)
        for sc, lab in (("rel_frag", "fragment density (novelty 1)"), ("ent_mean", "mean entropy")):
            o = np.argsort(sub[sc].to_numpy(float), kind="stable")
            cov = np.arange(1, n + 1) / n
            ax.plot(cov[::-1] * 100, [err[o][:k].mean() for k in range(1, n + 1)][::-1], label=lab)
        ax.axhline(err.mean(), ls=":", color="k", label="random")
        ax.set_xlabel("images kept (%) - least reliable removed first")
        ax.set_ylabel("mean 1 - Dice of kept images")
        ax.set_title("H_tglite70k (~66K, seed 0), unseen datasets pooled")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "risk_coverage_headline.png", dpi=150)
        plt.close(fig)
    # scale selection
    if "sel_scale" in df:
        t = df[df["model"] == "H_tglite70k[det]"].groupby("dataset")["sel_scale"].value_counts(normalize=True).unstack(fill_value=0)
        if len(t):
            ax = t.plot(kind="bar", stacked=True, figsize=(6.5, 4))
            ax.set_ylabel("share of images")
            ax.set_title("Scale chosen by the calibre rule (H_tglite70k)")
            plt.tight_layout()
            plt.savefig(out / "scale_selection.png", dpi=150)
            plt.close("all")


# ------------------------------------------------------------------------------------------------ main
def build_report21(v2_runs: Path, root21: Path, out: Path, log=print) -> None:
    from .suite21 import decisions21_file
    out = Path(out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    dec21 = json.loads(decisions21_file().read_text(encoding="utf-8")) if decisions21_file().exists() else {}
    (out / "protocol").mkdir(exist_ok=True)
    for f in (paths.PROTOCOL / "PROTOCOL_v2.1.md", decisions21_file(), paths.CONFIGS / "v21.yaml"):
        if f.exists():
            shutil.copy2(f, out / "protocol" / f.name)
    df = _collect(Path(v2_runs), Path(root21))
    if df.empty:
        (out / "summary.md").write_text("# TG-LiteNet v2.1\n\nNo v2.1 evaluations yet.\n", encoding="utf-8")
        return
    df.to_csv(out / "tables" / "per_image_all.csv.gz", index=False)
    metric_cols = [c for c in df.columns if c.split("/")[0] in ("raw", "clean", "tuned", "sel", "net") and not c.endswith(("_pred", "_gt"))]
    metric_cols += [c for c in df.columns if c.startswith("fix")]
    pr = _per_run(df, metric_cols)
    pr.to_csv(out / "tables" / "per_run.csv", index=False)
    cost_of = lambda e: _cost(Path(v2_runs), Path(root21), e, dec21)  # noqa: E731

    # main table
    order = lambda m: (0 if m.startswith(("H_", "ENS_H")) else 1 if m.startswith("R_") else 2 if m.startswith("Z_") else 3, m)  # noqa: E731
    rows = []
    for m in sorted(pr["model"].unique(), key=order):
        x = pr[pr["model"] == m]
        c = cost_of(m.split("[")[0])
        rows.append({"model": m, "n": len(x), "DRIVE Dice": _ms(x.get("drive/tuned/dice", [])),
                     "unseen Dice raw": _ms(x.get("unseen/raw/dice", [])), "unseen Dice tuned": _ms(x.get("unseen/tuned/dice", [])),
                     "unseen Dice scale-sel": _ms(x.get("unseen/sel/dice", [])),
                     "STARE": _ms(x.get("stare/tuned/dice", [])), "CHASE": _ms(x.get("chase_db1/tuned/dice", [])),
                     "HRF": _ms(x.get("hrf/tuned/dice", [])), "FIVES": _ms(x.get("fives/tuned/dice", [])),
                     "unseen clDice": _ms(x.get("unseen/tuned/cldice", [])), "Betti-0 raw": _ms(x.get("unseen/raw/betti0_err", [])),
                     "Betti-0 clean": _ms(x.get("unseen/clean/betti0_err", [])), "gap": _ms(x.get("gap/tuned/dice", [])),
                     "unseen Dice @network res": _ms(x.get("unseen/net/tuned/dice", [])),
                     "params": f"{c['params'] / 1e3:.0f}K" if c else "-", "GFLOPs": f"{c['gflops']:.2f}" if c else "-",
                     "CPU ms": f"{c['cpu_4t']['median_ms']:.1f}" if c else "-"})
    main = pd.DataFrame(rows)
    main.to_csv(out / "tables" / "main.csv", index=False)
    (out / "tables" / "main.md").write_text(
        "### v2.1 main results (mean±sd over seeds). 'tuned' = threshold + fragment size chosen on DRIVE-val (v2.1 headline mode);\n"
        "### 'raw' = threshold 0.5; 'scale-sel' = tuned + topology-guided scale (novelty 2). Native resolution unless stated.\n\n"
        + _md(main), encoding="utf-8")

    # SOTA
    s = [dict(r) for r in SOTA]
    for r in s:
        r.setdefault("unseen_scale_sel", "-")
    for m, lab in (("H_tglite70k[det]", "TG-LiteNet v2.1 ~66K (H_tglite70k)"), ("ENS_H_tglite70k[ens]", "v2.1 ~66K, 5-seed ensemble"),
                   ("R_lite_w16[det]", "v2.1 R_lite_w16 (size-rule headline)"), ("R_unet_w8[det]", "v2.1 R_unet_w8"),
                   ("E_tglitenet[det]", "TG-LiteNet v2 (E)"), ("C_unet_topo_dg[det]", "v2 stage C U-Net (31M)")):
        x = pr[pr["model"] == m]
        if x.empty:
            continue
        c = cost_of(m.split("[")[0])
        f = lambda k: f"{100 * x[k].mean():.1f}" if k in x else "-"  # noqa: E731
        s.append({"model": lab, "protocol": "source-only DRIVE (native res.)", "params": f"{c['params'] / 1e3:.0f}K" if c else "-",
                  "gflops": f"{c['gflops']:.2f}" if c else "-", "stare": f("stare/tuned/dice"), "chase_db1": f("chase_db1/tuned/dice"),
                  "hrf": f("hrf/tuned/dice"), "unseen": f"{f('unseen/tuned/dice')} (net-res {f('unseen/net/tuned/dice')})",
                  "unseen_scale_sel": f("unseen/sel/dice"),
                  "cldice": f("unseen/tuned/cldice"), "betti0": f"{x['unseen/raw/betti0_err'].mean():.1f}" if "unseen/raw/betti0_err" in x else "-",
                  "uncertainty": "connectivity score + entropy"})
    (out / "tables" / "sota.md").write_text(
        "### Published results (context only - training data, scoring resolution and thresholds differ) vs ours\n"
        "### Dice / clDice in %, 'unseen' = mean of STARE, CHASE_DB1, HRF. Ours: threshold chosen on DRIVE-val, native-resolution "
        "scoring (network-resolution in brackets).\n\n" + _md(pd.DataFrame(s)), encoding="utf-8")

    # novelty 1
    tab, summ, verdict = novelty1(df)
    if not summ.empty:
        summ.to_csv(out / "tables" / "novelty1_summary.csv", index=False)
        tab.to_csv(out / "tables" / "novelty1_per_dataset.csv", index=False)
        sm = summ.copy()
        for c in ("rho_mean", "rho_sd", "gain_mean", "gain_sd"):
            sm[c] = sm[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}" if "rho" in c else f"{v:.1f}")
        txt = ("### Novelty 1 - label-free reliability of whole images. Spearman rho between the score and the image error "
               "(1 - Dice, tuned mode), averaged over the unseen datasets, then over seeds; AURC gain = % of the random-to-oracle "
               "gap closed when the least reliable images are rejected first.\n### Scores: rel_frag (fragment density, primary), "
               "rel_endpoints, rel_smallfrac (connectivity, label-free, single pass); ent_mean (mean entropy; MC / ensemble mean for "
               "[mc] / [ens]); var_mean (MC / ensemble variance).\n\n" + _md(sm[["model", "score", "rho_mean", "rho_sd", "n",
                                                                                   "gain_mean", "gain_sd"]]))
        txt += "\n\n### Pre-registered comparisons (seed-paired t interval)\n\n" + "\n".join(f"* {k}: {v}" for k, v in verdict.items())
        (out / "tables" / "novelty1_reliability.md").write_text(txt, encoding="utf-8")

    # novelty 2
    models2 = [m for m in sorted(pr["model"].unique(), key=order)]
    n2 = novelty2(pr, models2)
    rules = []
    for f in list(Path(root21).glob("*/seed*/val_scale_rule_*.json")) + list(Path(v2_runs).glob("*/seed*/val_scale_rule_*.json")):
        try:
            r = json.loads(f.read_text())
            sk = "; ".join(f"shrink {k}: rule {v['gain_rule']:+.3f} / oracle {v['gain_oracle']:+.3f}" for k, v in r.get("shrink_check", {}).items())
            rules.append(f"{f.parent.parent.name} {f.parent.name} {f.stem.split('_')[-1]}: val Dice {r['val_dice_scale1']:.4f} -> "
                         f"{r['val_dice_selected']:.4f}, {'passed' if r['passed'] else 'FAILED'}  ({sk})")
        except Exception:  # noqa: BLE001
            pass
    (out / "tables" / "novelty2_scale.md").write_text(
        "### Novelty 2 - centreline-calibre-guided test-time scale selection (scales 1, 1.5, 2, 2.5; the scale whose predicted "
        "vessel calibre, measured on the prediction's own skeleton, is closest to the model's DRIVE-train calibre wins). "
        "Unseen Dice in the tuned mode at scale 1 vs with selection; delta = selected - scale 1 (seed-paired).\n\n" + _md(n2)
        + "\n\n### DRIVE-val check of the rule (rule used for claims only if it passed)\n\n" + "\n".join(f"* {r}" for r in sorted(rules)),
        encoding="utf-8")

    # ablations
    ab = ablations(pr)
    (out / "tables" / "ablations.md").write_text(
        "### Novelties 3, 4 and the new augmentation - H_tglite70k (~66K, full v2.1 recipe) minus ablation (seed-paired over the shared seeds; "
        "tuned mode; Betti-0 lower is better, so a negative Betti-0 difference favours H_tglite70k)\n\n" + _md(ab), encoding="utf-8")

    # decisions + int8
    (out / "tables" / "decisions.md").write_text("### v2.1 decisions (DRIVE-val only)\n\n```json\n" + json.dumps(dec21, indent=2) + "\n```\n",
                                                encoding="utf-8")
    i8 = Path(root21) / "H_tglite70k" / "int8.json"
    if i8.exists():
        r = json.loads(i8.read_text())
        if not r.get("failed"):
            t = ["| dataset | Dice fp32 | Dice int8 | n |", "|---|---|---|---|"]
            t += [f"| {d} | {v['fp32']:.4f} | {v['int8']:.4f} | {v['n']} |" for d, v in r["dice"].items()]
            (out / "tables" / "int8.md").write_text(
                f"### INT8 (ONNX Runtime, static QDQ, calibrated on 8 DRIVE-train images) - H_tglite70k (~66K) seed 0\n\n"
                f"* CPU 512x512, 4 threads: fp32 {r['cpu_fp32_onnx_ms']:.1f} ms -> int8 {r['cpu_int8_onnx_ms']:.1f} ms\n"
                f"* file size: {r['size_fp32_kb']:.0f} KB -> {r['size_int8_kb']:.0f} KB\n\n" + "\n".join(t), encoding="utf-8")

    try:
        _figures(pr, df, cost_of, out / "figures", log)
    except Exception as exc:  # noqa: BLE001
        log(f"  v2.1 figures failed: {exc}")

    # summary
    L = ["# TG-LiteNet v2.1 - summary", "", "Generated by `src/tglite/report21.py` from `runs/`. Protocol: `protocol/PROTOCOL_v2.1.md`.", ""]
    if dec21:
        L += [f"* architecture chosen on DRIVE-val: **{dec21.get('arch', 'pending')}**; loss weights: {dec21.get('weights', 'pending')}; "
              f"headline by the size rule: **{dec21.get('headline', 'pending')}**; frozen: {dec21.get('frozen', False)}", ""]
    L += ["## Main results", "", _md(main), ""]
    for name in ("sota.md", "novelty1_reliability.md", "novelty2_scale.md", "ablations.md", "int8.md"):
        f = out / "tables" / name
        if f.exists():
            L += [f"## {name[:-3]}", "", f.read_text(encoding="utf-8"), ""]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    log(f"  v2.1 report: {df['model'].nunique()} models, {len(df)} image rows -> {out}")
