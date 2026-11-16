"""runs/ -> results/  (tables, statistics, figures, per-run records). Works on partial results: everything that exists is used.

results/
  README.md            what is where (generated)          summary.md      headline numbers + automatic findings
  protocol/            decisions.json, drive_split.json, PROTOCOL.md (copies, as frozen)
  data/                manifest.json, dataset_stats.csv
  tables/              main_{raw,clean,tuned}.{csv,md}, per_dataset.csv, comparisons.{csv,md}, factorial.md,
                       efficiency.{csv,md}, uncertainty.{csv,md}, selection.csv, training.csv
  figures/             stages.png, cleanup_threshold.png, tradeoff.png, risk_coverage.png, training_curves.png, qualitative.png
  runs/<exp>/seed<k>/  config.yaml, train_log.csv, train_done.json, val_selection_*.json, per_image_*.csv
  runs/<exp>/efficiency.json
Every number comes from the run files; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import paths

UNSEEN = ("stare", "chase_db1", "hrf")
MODES = ("raw", "clean", "tuned")
HEAD = ["dice", "cldice", "betti0_err", "betti1_err", "sens", "prec"]
STAGES = ["A_unet", "B_unet_topo", "C_unet_topo_dg", "D_compact", "E_tglitenet"]
LABELS = {"A_unet": "A U-Net", "B_unet_topo": "B +topology", "C_unet_topo_dg": "C +DG aug.", "D_compact": "D compact",
          "E_tglitenet": "E TG-LiteNet", "E_tglitenet[mc]": "E TG-LiteNet (MC)", "X_unet_dg": "U-Net +DG only"}
# pre-registered comparisons (protocol/PROTOCOL.md, section 6): (name, first, second)
PRIMARY_COMPARISONS = [
    ("RQ1 topology (A->B)", "A_unet[det]", "B_unet_topo[det]"),
    ("RQ2 DG (B->C)", "B_unet_topo[det]", "C_unet_topo_dg[det]"),
    ("RQ3 compact (C->D)", "C_unet_topo_dg[det]", "D_compact[det]"),
    ("dropout (D->E det)", "D_compact[det]", "E_tglitenet[det]"),
    ("MC averaging (E det->E mc)", "E_tglitenet[det]", "E_tglitenet[mc]"),
    ("overall (A->E mc)", "A_unet[det]", "E_tglitenet[mc]"),
]
SECONDARY_COMPARISONS = [
    ("DG without topology (A->X_unet_dg)", "A_unet[det]", "X_unet_dg[det]"),
    ("topology given DG (X_unet_dg->C)", "X_unet_dg[det]", "C_unet_topo_dg[det]"),
    ("warm-up U-Net (C->W)", "C_unet_topo_dg[det]", "W_unet_topo_dg_warm[det]"),
    ("warm-up compact (D->W)", "D_compact[det]", "W_compact_warm[det]"),
    ("ensemble vs MC (E mc->ENS D)", "E_tglitenet[mc]", "ENS_D_compact[ens]"),
]
PRIMARY_METRICS = [("unseen", "dice"), ("unseen", "cldice"), ("unseen", "betti0_err"), ("gap", "dice")]


# ------------------------------------------------------------------------------------------------- loading
def load_all(runs: Path) -> pd.DataFrame:
    """Long per-image table: model (= exp[variant]), exp, variant, seed, dataset, id, metric columns."""
    frames = []
    for f in sorted(runs.glob("*/seed*/eval/per_image_*.csv")) + sorted(runs.glob("ENS_*/eval/per_image_*.csv")):
        variant = f.stem.replace("per_image_", "")
        if f.parent.parent.name.startswith("seed"):
            exp, seed = f.parent.parent.parent.name, int(f.parent.parent.name[4:])
        else:
            exp, seed = f.parent.parent.name, -1
        df = pd.read_csv(f)
        if df.empty:
            continue
        df.insert(0, "seed", seed)
        df.insert(0, "variant", variant)
        df.insert(0, "exp", exp)
        df.insert(0, "model", f"{exp}[{variant}]")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_run(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One row per (model, seed): per-dataset means, unseen mean (equal dataset weight) and gap."""
    cols = {m: f"{mode}/{m}" for m in HEAD}
    extra = ["auc", "ece", "ece_ts", "unc_mean", "unc_rho_px"]
    use = [c for c in list(cols.values()) + extra if c in df.columns]
    g = df.groupby(["model", "exp", "variant", "seed", "dataset"])[use].mean().reset_index()
    g = g.rename(columns={v: k for k, v in cols.items()})
    rows = []
    for (model, exp, variant, seed), sub in g.groupby(["model", "exp", "variant", "seed"]):
        r = {"model": model, "exp": exp, "variant": variant, "seed": seed}
        for _, d in sub.iterrows():
            for m in list(cols) + extra:
                if m in d:
                    r[f"{d['dataset']}/{m}"] = d[m]
        un = sub[sub.dataset.isin(UNSEEN)]
        if len(un) == len(UNSEEN):
            for m in list(cols) + extra:
                if m in un:
                    r[f"unseen/{m}"] = un[m].mean()
            if "drive/dice" in r:
                r["gap/dice"] = r["drive/dice"] - r["unseen/dice"]
                r["gap/cldice"] = r["drive/cldice"] - r["unseen/cldice"]
        rows.append(r)
    return pd.DataFrame(rows)


def _ms(x) -> str:
    x = pd.Series(x).dropna()
    if x.empty:
        return "-"
    big = x.abs().mean() >= 10
    f = "{:.1f}" if big else "{:.3f}"
    return f.format(x.mean()) + ("" if len(x) < 2 else "±" + f.format(x.std(ddof=1)))


def summary_table(pr: pd.DataFrame, eff: dict, cols) -> pd.DataFrame:
    rows = []
    for model, sub in pr.groupby("model", sort=False):
        exp = sub.exp.iloc[0]
        r = {"model": model, "n_seeds": int((sub.seed >= 0).sum()) or 1}
        for c in cols:
            r[c] = _ms(sub[c]) if c in sub else "-"
            r[c + "_mean"] = sub[c].mean() if c in sub else np.nan
        e = eff.get(exp.replace("ENS_", "")) or eff.get(exp)
        if e:
            r["params_M"] = round(e["params_m"], 3)
            r["GFLOPs"] = round(e["gflops"], 2)
            r["cpu_ms"] = round(e["cpu_4t"]["median_ms"], 1)
            r["gpu_ms"] = round(e.get("gpu_fp32", {}).get("median_ms", np.nan), 2)
            mult = e["mc_samples"] if "[mc]" in model else 5 if "[ens]" in model else 1   # MC passes / ensemble members
            r["cpu_ms"] = round(r["cpu_ms"] * mult, 1)
            r["gpu_ms"] = round(r["gpu_ms"] * mult, 2)
            if "[ens]" in model:
                r["params_M"] = round(r["params_M"] * 5, 3)
        rows.append(r)
    return pd.DataFrame(rows)


def _order(models) -> list[str]:
    def key(m):
        exp = m.split("[")[0]
        pri = STAGES.index(exp) if exp in STAGES else 50 + ord(exp[0])
        return (pri, m)
    return sorted(models, key=key)


def md_table(df: pd.DataFrame, cols, title: str = "") -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in df.to_dict("records"))
    return (f"### {title}\n\n" if title else "") + head + body


# ------------------------------------------------------------------------------------------------- statistics
def paired(pr: pd.DataFrame, a: str, b: str, col: str):
    A = pr[pr.model == a].set_index("seed")[col] if col in pr else pd.Series(dtype=float)
    B = pr[pr.model == b].set_index("seed")[col] if col in pr else pd.Series(dtype=float)
    if b.startswith("ENS_") or a.startswith("ENS_"):           # ensemble: one value -> compare with the other model's seed mean
        if A.empty or B.empty:
            return None
        return {"n": 1, "delta": float(B.mean() - A.mean()), "ci95": np.nan, "p": np.nan}
    common = A.index.intersection(B.index)
    if len(common) < 2:
        return None
    d = (B.loc[common] - A.loc[common]).to_numpy(dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    half = stats.t.ppf(0.975, len(d) - 1) * se
    p = float(stats.ttest_1samp(d, 0).pvalue) if se > 0 else 0.0
    return {"n": len(d), "delta": float(d.mean()), "ci95": float(half), "p": p}


def image_bootstrap(df: pd.DataFrame, a: str, b: str, metric: str, mode: str = "raw", n_boot: int = 2000, seed: int = 0):
    """Paired image-level bootstrap on the unseen datasets: images averaged over shared seeds, resampled within each dataset,
    datasets weighted equally. Returns (delta, lo, hi)."""
    col = f"{mode}/{metric}"
    if col not in df:
        return None
    A, B = df[df.model == a], df[df.model == b]
    shared = sorted(set(A.seed) & set(B.seed)) if not (a.startswith("ENS_") or b.startswith("ENS_")) else None
    if shared is not None:
        A, B = A[A.seed.isin(shared)], B[B.seed.isin(shared)]
    ga = A[A.dataset.isin(UNSEEN)].groupby(["dataset", "id"])[col].mean()
    gb = B[B.dataset.isin(UNSEEN)].groupby(["dataset", "id"])[col].mean()
    d = (gb - ga).dropna()
    if d.empty or d.index.get_level_values(0).nunique() < len(UNSEEN):
        return None
    rng = np.random.default_rng(seed)
    per = {k: d.xs(k).to_numpy() for k in UNSEEN}
    est = np.mean([v.mean() for v in per.values()])
    boots = np.mean([np.array([rng.choice(v, len(v)).mean() for _ in range(n_boot)]) for v in per.values()], axis=0)
    return float(est), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def comparisons(pr_by_mode: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, comps in (("primary", PRIMARY_COMPARISONS), ("secondary", SECONDARY_COMPARISONS)):
        for mode in MODES:
            pr = pr_by_mode[mode]
            fam = []
            for name, a, b in comps:
                for scope, metric in PRIMARY_METRICS:
                    r = paired(pr, a, b, f"{scope}/{metric}")
                    if r is None:
                        continue
                    bs = image_bootstrap(df, a, b, metric, mode) if scope == "unseen" else None
                    fam.append({"family": family, "mode": mode, "comparison": name, "first": a, "second": b,
                                "metric": f"{scope}/{metric}", **r,
                                "img_boot_lo": bs[1] if bs else np.nan, "img_boot_hi": bs[2] if bs else np.nan})
            if fam:
                ps = [x["p"] for x in fam]
                ok = [i for i, v in enumerate(ps) if not np.isnan(v)]
                if ok:
                    adj = holm([ps[i] for i in ok])
                    for i, v in zip(ok, adj):
                        fam[i]["p_holm"] = float(v)
                rows += fam
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- uncertainty
def uncertainty_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per (model, dataset): pixel Spearman, image-level Spearman(uncertainty, 1-Dice), image AURC vs random/oracle, ECE."""
    rows, curves = [], {}
    if "unc_mean" not in df:
        return pd.DataFrame(), curves
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        per_seed = []
        for seed, s in sub.groupby("seed"):
            u, risk = s["unc_mean"].to_numpy(), 1 - s["raw/dice"].to_numpy()
            n = len(u)
            order = np.argsort(-u)
            covs = np.linspace(1.0, 0.5, 6)
            kept = [risk[order[int(round((1 - c) * n)):]].mean() for c in covs]
            oracle = [np.sort(risk)[: max(1, int(round(c * n)))].mean() for c in covs]
            per_seed.append({"rho_img": stats.spearmanr(u, risk).statistic if n > 2 else np.nan,
                             "aurc": float(np.mean(kept)), "aurc_random": float(risk.mean()), "aurc_oracle": float(np.mean(oracle)),
                             "rho_px": s["unc_rho_px"].mean(), "ece": s["ece"].mean(), "ece_ts": s.get("ece_ts", pd.Series([np.nan])).mean(),
                             "px_err@100": s["px_err@100"].mean(), "px_err@90": s["px_err@90"].mean(), "curve": kept})
        ps = pd.DataFrame(per_seed)
        r = {"model": model, "dataset": dataset, "n_seeds": len(ps)}
        for c in ("rho_px", "rho_img", "aurc", "aurc_random", "aurc_oracle", "ece", "ece_ts", "px_err@100", "px_err@90"):
            r[c] = float(ps[c].mean())
        rows.append(r)
        curves[(model, dataset)] = np.mean(np.stack(ps["curve"].to_list()), axis=0)
    return pd.DataFrame(rows), curves


# ------------------------------------------------------------------------------------------------- figures
def figures(out: Path, pr_by_mode: dict, df: pd.DataFrame, summ: pd.DataFrame, curves: dict, runs: Path, log) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    pr = pr_by_mode["raw"]
    stages = [m for m in [f"{s}[det]" for s in STAGES] if m in set(pr.model)]
    dsets = [d for d in ("drive", *UNSEEN, "fives") if f"{d}/dice" in pr]
    colors = ["#6b6b6b", "#2c7fb8", "#41ab5d", "#fd8d3c", "#d7301f"]
    if stages:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (m, title, logy) in zip(axes, [("dice", "Dice (higher is better)", False), ("cldice", "clDice (higher is better)", False),
                                              ("betti0_err", "Betti-0 error (lower is better)", True)]):
            w = 0.8 / len(stages)
            for i, s in enumerate(stages):
                sub = pr[pr.model == s]
                mu = [sub[f"{d}/{m}"].mean() for d in dsets]
                sd = [sub[f"{d}/{m}"].std(ddof=1) if len(sub) > 1 else 0 for d in dsets]
                ax.bar(np.arange(len(dsets)) + i * w, mu, w, yerr=sd, color=colors[i % 5], label=LABELS.get(s[:-5], s), capsize=2)
            ax.set_xticks(np.arange(len(dsets)) + 0.4 - w / 2, dsets)
            ax.set_title(title)
            if logy:
                ax.set_yscale("log")
            else:
                lo = np.nanmin([pr[pr.model.isin(stages)][f"{d}/{m}"].min() for d in dsets])
                ax.set_ylim(max(0, lo - 0.05), None)
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "stages.png", dpi=160)
        plt.close(fig)

    # threshold / cleanup check
    models = [m for m in _order(set(pr.model)) if m.endswith("[det]") or m.endswith("[mc]")][:12]
    if models and "unseen/dice" in pr:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        x = np.arange(len(models))
        for k, (mode, col, mk) in enumerate([("raw", "#333333", "o"), ("clean", "#2c7fb8", "s"), ("tuned", "#d7301f", "^")]):
            p = pr_by_mode[mode]
            axes[0].scatter(x + (k - 1) * 0.15, [p[p.model == m]["unseen/betti0_err"].mean() for m in models], c=col, marker=mk, label=mode)
            axes[1].scatter(x + (k - 1) * 0.15, [p[p.model == m]["unseen/dice"].mean() for m in models], c=col, marker=mk, label=mode)
        axes[0].set_yscale("log")
        axes[0].set_title("Unseen Betti-0 error by evaluation mode")
        axes[1].set_title("Unseen Dice by evaluation mode")
        for ax in axes:
            ax.set_xticks(x, [m.replace("[det]", "") for m in models], rotation=45, ha="right", fontsize=8)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "cleanup_threshold.png", dpi=160)
        plt.close(fig)

    # accuracy vs CPU time
    s2 = summ.dropna(subset=["cpu_ms"]) if "cpu_ms" in summ else pd.DataFrame()
    s2 = s2[~s2.model.str.contains(r"\[mc\]|\[ens\]")] if not s2.empty else s2
    if not s2.empty and "unseen/dice_mean" in s2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, col, t in ((axes[0], "unseen/dice_mean", "Unseen Dice"), (axes[1], "unseen/cldice_mean", "Unseen clDice")):
            ax.scatter(s2.cpu_ms, s2[col], s=20 + 60 * np.log1p(s2.params_M * 10), c="#2c7fb8", alpha=0.8)
            for _, r in s2.iterrows():
                ax.annotate(r.model.replace("[det]", ""), (r.cpu_ms, r[col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_xlabel("CPU time per 512x512 image (ms, 4 threads, log)")
            ax.set_title(t + " vs CPU time (marker size ~ parameters)")
        fig.tight_layout()
        fig.savefig(fig_dir / "tradeoff.png", dpi=160)
        plt.close(fig)

    # image-level risk-coverage
    unc_models = [m for m in ("E_tglitenet[det]", "E_tglitenet[mc]", "ENS_D_compact[ens]", "D_compact[det]") if any(k[0] == m for k in curves)]
    if unc_models:
        ds = [d for d in ("drive", *UNSEEN, "fives") if any(k == (unc_models[0], d) for k in curves)]
        fig, axes = plt.subplots(1, len(ds), figsize=(3.4 * len(ds), 3.4), sharey=False)
        axes = np.atleast_1d(axes)
        covs = np.linspace(100, 50, 6)
        for ax, d in zip(axes, ds):
            for m, c in zip(unc_models, ["#888888", "#d7301f", "#2c7fb8", "#41ab5d"]):
                if (m, d) in curves:
                    ax.plot(covs, curves[(m, d)], marker="o", c=c, label=m, ms=3)
            ax.invert_xaxis()
            ax.set_title(d)
            ax.set_xlabel("images kept (%)")
        axes[0].set_ylabel("mean 1-Dice of kept images")
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "risk_coverage.png", dpi=160)
        plt.close(fig)

    # training curves
    logs = {e: sorted((runs / e).glob("seed*/train_log.csv")) for e in ("pilot_unet", "pilot_lite", *STAGES)}
    logs = {k: v for k, v in logs.items() if v}
    if logs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for e, fs in logs.items():
            d = pd.read_csv(fs[0]).dropna(subset=["val_dice"])
            ax.plot(d["iter"], d["val_dice"], label=f"{e} ({fs[0].parent.name})")
        ax.set_xlabel("iteration")
        ax.set_ylabel("DRIVE-val Dice")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=160)
        plt.close(fig)
    try:
        qualitative(fig_dir, runs)
    except Exception as exc:  # noqa: BLE001
        log(f"  qualitative figure skipped: {exc}")


def qualitative(fig_dir: Path, runs: Path) -> None:
    import cv2
    import matplotlib.pyplot as plt

    from .data import load_normalised
    show = [("A_unet", "det"), ("D_compact", "det"), ("E_tglitenet", "mc")]
    avail = [(e, v) for e, v in show if (runs / e / "seed0" / "preds" / v).exists()]
    if not avail:
        return
    base = runs / avail[0][0] / "seed0" / "preds" / avail[0][1]
    ds = [d.name for d in sorted(base.iterdir()) if d.is_dir()]
    fig, axes = plt.subplots(len(ds), 2 + len(avail), figsize=(2.6 * (2 + len(avail)), 2.6 * len(ds)))
    axes = np.atleast_2d(axes)
    for i, d in enumerate(ds):
        sid = sorted(p.name[:-9] for p in (base / d).glob("*_prob.png"))[0]
        s = load_normalised(d, sid, 540)
        h, w = s["mask"].shape
        cy, cx, r = h // 2, w // 2, min(h, w) // 5
        crop = (slice(cy - r, cy + r), slice(cx - r, cx + r))
        axes[i, 0].imshow(s["img"][crop])
        axes[i, 0].set_ylabel(d)
        axes[i, 1].imshow(s["mask"][crop], cmap="gray")
        for j, (e, v) in enumerate(avail):
            p = cv2.imread(str(runs / e / "seed0" / "preds" / v / d / f"{sid}_prob.png"), 0)
            if p is None:
                continue
            pred = p[crop] >= 128
            gt = s["mask"][crop] > 0
            err = np.ones(gt.shape + (3,))
            err[pred & ~gt] = (0.85, 0.15, 0.1)
            err[~pred & gt] = (0.1, 0.3, 0.9)
            err[pred & gt] = (0.2, 0.2, 0.2)
            axes[i, 2 + j].imshow(err)
            if i == 0:
                axes[i, 2 + j].set_title(f"{e} [{v}]", fontsize=8)
    axes[0, 0].set_title("image (centre crop)", fontsize=8)
    axes[0, 1].set_title("ground truth", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Errors at threshold 0.5 (black: correct vessel, red: false positive, blue: missed), seed 0", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "qualitative.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- export
def copy_run_records(runs: Path, out: Path) -> None:
    dst_root = out / "runs"
    for rd in sorted(runs.glob("*/seed*")) + sorted(runs.glob("ENS_*")):
        if not rd.is_dir() or rd.name.startswith("_"):
            continue
        rel = rd.relative_to(runs)
        dst = dst_root / rel
        for pat in ("config.yaml", "train_log.csv", "train_done.json", "run_meta.json", "val_selection_*.json", "members.json"):
            for f in rd.glob(pat):
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst / f.name)
        for f in rd.glob("eval/per_image_*.csv"):
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.name)
    for f in runs.glob("*/efficiency.json"):
        (dst_root / f.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst_root / f.parent.name / "efficiency.json")


def build_report(runs: Path, out: Path, log=print) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    # protocol + data provenance
    (out / "protocol").mkdir(exist_ok=True)
    for f in (paths.decisions_file(), paths.PROTOCOL / "drive_split.json", paths.PROTOCOL / "PROTOCOL.md"):
        if Path(f).exists():
            shutil.copy2(f, out / "protocol" / Path(f).name)
    mf = paths.processed_dir() / "MANIFEST.json"
    if mf.exists():
        (out / "data").mkdir(exist_ok=True)
        shutil.copy2(mf, out / "data" / "manifest.json")
        man = json.loads(mf.read_text())
        pd.DataFrame([{"dataset": k, **v["stats"], "fov": v.get("fov"), "labels": v.get("labels")} for k, v in man.items()]
                     ).to_csv(out / "data" / "dataset_stats.csv", index=False)
    copy_run_records(runs, out)

    # training / selection records
    tr_rows, sel_rows = [], []
    for f in sorted(runs.glob("*/seed*/train_done.json")):
        tr_rows.append({"exp": f.parent.parent.name, "seed": f.parent.name, **json.loads(f.read_text())})
    for f in sorted(runs.glob("*/seed*/val_selection_*.json")) + sorted(runs.glob("ENS_*/val_selection_*.json")):
        s = json.loads(f.read_text())
        sel_rows.append({"run": str(f.parent.relative_to(runs)), "variant": f.stem.split("_")[-1], "threshold": s["threshold"],
                         "fragment@0.5": s["fragment_min_size_at_0.5"], "fragment@thr": s["fragment_min_size_at_threshold"],
                         "temperature": s["temperature"]})
    if tr_rows:
        pd.DataFrame(tr_rows).to_csv(out / "tables" / "training.csv", index=False)
    if sel_rows:
        pd.DataFrame(sel_rows).to_csv(out / "tables" / "selection.csv", index=False)

    eff = {f.parent.name: json.loads(f.read_text()) for f in runs.glob("*/efficiency.json")}
    if eff:
        er = [{"exp": k, "params_M": round(v["params_m"], 4), "GFLOPs@512": round(v["gflops"], 2),
               "cpu_4t_ms": round(v["cpu_4t"]["median_ms"], 1), "cpu_1t_ms": round(v["cpu_1t"]["median_ms"], 1),
               "onnx_cpu_4t_ms": round(v.get("onnx_cpu_4t", {}).get("median_ms", np.nan), 1),
               "gpu_fp32_ms": round(v.get("gpu_fp32", {}).get("median_ms", np.nan), 2),
               "gpu_fp16_ms": round(v.get("gpu_fp16", {}).get("median_ms", np.nan), 2),
               "gpu_peak_MB": round(v.get("gpu_peak_mem_mb", np.nan), 1), "mc_samples": v["mc_samples"]} for k, v in sorted(eff.items())]
        e_df = pd.DataFrame(er)
        e_df.to_csv(out / "tables" / "efficiency.csv", index=False)
        (out / "tables" / "efficiency.md").write_text(md_table(e_df, list(e_df.columns), "Cost on this machine (batch 1, 512x512)"), encoding="utf-8")

    df = load_all(runs)
    if df.empty:
        log("  report: no evaluated runs yet (tables for training/selection/cost only)")
        write_readme(out, None)
        return
    df.to_csv(out / "tables" / "per_image_all.csv.gz", index=False, compression="gzip")
    pr_by_mode = {m: per_run(df, m) for m in MODES}
    cols = ["drive/dice", "unseen/dice", "unseen/cldice", "unseen/betti0_err", "unseen/betti1_err", "gap/dice",
            "stare/dice", "chase_db1/dice", "hrf/dice", "fives/dice", "unseen/prec", "unseen/sens"]
    summ = {}
    for mode in MODES:
        pr = pr_by_mode[mode]
        pr.to_csv(out / "tables" / f"per_run_{mode}.csv", index=False)
        s = summary_table(pr, eff, cols)
        s = s.set_index("model").loc[_order(s.model)].reset_index()
        summ[mode] = s
        s.to_csv(out / "tables" / f"main_{mode}.csv", index=False)
        show = ["model", "n_seeds"] + cols + [c for c in ("params_M", "cpu_ms") if c in s]
        desc = {"raw": "threshold 0.5, no post-processing (headline)", "clean": "threshold 0.5 + fragment removal (size chosen on DRIVE-val)",
                "tuned": "threshold and fragment size chosen on DRIVE-val"}[mode]
        (out / "tables" / f"main_{mode}.md").write_text(md_table(s, show, f"Main results - {desc}; mean±sd over seeds"), encoding="utf-8")
    pdset = []
    for mode in MODES:
        g = df.groupby(["model", "dataset"])[[f"{mode}/{m}" for m in HEAD if f"{mode}/{m}" in df]].mean().reset_index()
        g.insert(1, "mode", mode)
        pdset.append(g.rename(columns=lambda c: c.split("/")[-1]))
    pd.concat(pdset).to_csv(out / "tables" / "per_dataset.csv", index=False)

    comp = comparisons(pr_by_mode, df)
    if not comp.empty:
        comp.to_csv(out / "tables" / "comparisons.csv", index=False)
        c2 = comp.copy()
        c2["delta [95% CI]"] = [f"{d:+.4f} [±{c:.4f}]" if not np.isnan(c) else f"{d:+.4f}" for d, c in zip(c2.delta, c2.ci95)]
        c2["image bootstrap"] = [f"[{lo:+.4f}, {hi:+.4f}]" if not np.isnan(lo) else "" for lo, hi in zip(c2.img_boot_lo, c2.img_boot_hi)]
        c2["p (Holm)"] = [f"{p:.3f}" if not np.isnan(p) else "" for p in c2.get("p_holm", pd.Series(np.nan, index=c2.index))]
        txt = ""
        for (fam, mode), sub in c2.groupby(["family", "mode"], sort=False):
            txt += md_table(sub, ["comparison", "metric", "n", "delta [95% CI]", "image bootstrap", "p (Holm)"],
                            f"{fam} comparisons - mode {mode} (second minus first; seed-paired t interval; Holm within family)") + "\n"
        (out / "tables" / "comparisons.md").write_text(txt, encoding="utf-8")

    # factorial view
    fact = {("unet", "-", "-"): "A_unet", ("unet", "topo", "-"): "B_unet_topo", ("unet", "-", "DG"): "X_unet_dg", ("unet", "topo", "DG"): "C_unet_topo_dg",
            ("compact", "-", "-"): "X_compact_plain", ("compact", "topo", "-"): "X_compact_topo", ("compact", "-", "DG"): "X_compact_dg",
            ("compact", "topo", "DG"): "D_compact"}
    s = summ["raw"].set_index("model")
    fr = [{"backbone": k[0], "topology": k[1], "DG": k[2], **{c: s.loc[f"{v}[det]", c] for c in ("n_seeds", "drive/dice", "unseen/dice", "unseen/cldice", "unseen/betti0_err", "gap/dice")}}
          for k, v in fact.items() if f"{v}[det]" in s.index]
    if fr:
        (out / "tables" / "factorial.md").write_text(md_table(pd.DataFrame(fr), list(fr[0].keys()), "2x2x2 factorial (threshold 0.5)"), encoding="utf-8")

    unc, curves = uncertainty_table(df)
    if not unc.empty:
        unc.to_csv(out / "tables" / "uncertainty.csv", index=False)
        u2 = unc.copy()
        for c in u2.columns[3:]:
            u2[c] = u2[c].map(lambda v: f"{v:.3f}")
        (out / "tables" / "uncertainty.md").write_text(md_table(u2, list(u2.columns), "Uncertainty quality (entropy of the mean prediction)"), encoding="utf-8")
    figures(out, pr_by_mode, df, summ["raw"], curves, runs, log)
    write_summary(out, summ, comp)
    write_readme(out, summ["raw"])
    log(f"  report: {df.model.nunique()} models, {len(df)} image rows -> {out}")


def write_summary(out: Path, summ: dict, comp: pd.DataFrame) -> None:
    dec = json.loads(paths.decisions_file().read_text()) if paths.decisions_file().exists() else {}
    L = ["# Results summary (auto-generated from runs/ - do not edit by hand)", ""]
    if dec:
        L += [f"* training budget: **{dec.get('budget_iters')}** iterations (pilot rule, DRIVE-val only)",
              f"* compact backbone: **{dec.get('backbone', {}).get('compact_choice')}** {dec.get('compact_model')} (DRIVE-val only)",
              f"* protocol frozen: {dec.get('frozen')} at {dec.get('frozen_at')}", ""]
    for mode in MODES:
        s = summ[mode]
        cols = [c for c in ("model", "n_seeds", "drive/dice", "unseen/dice", "unseen/cldice", "unseen/betti0_err", "gap/dice", "fives/dice", "params_M", "cpu_ms") if c in s]
        L.append(md_table(s, cols, f"mode = {mode}"))
    if not comp.empty:
        L += ["### Pre-registered primary comparisons (mode raw)", ""]
        for _, r in comp[(comp.family == "primary") & (comp["mode"] == "raw")].iterrows():
            clear = "clear" if (not np.isnan(r.ci95) and abs(r.delta) > r.ci95) else "not clear"
            L.append(f"* {r.comparison}, {r.metric}: {r.delta:+.4f} (±{r.ci95:.4f}, n={r.n}, Holm p={r.get('p_holm', np.nan):.3f}) - {clear}")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_readme(out: Path, summ) -> None:
    txt = """# results/

Exported automatically by `python scripts/run_all.py` (or `python scripts/make_report.py`) from the run folders in `runs/`.
Nothing here is edited by hand.

| Path | Content |
|---|---|
| `summary.md` | headline tables (3 evaluation modes) + pre-registered comparisons |
| `protocol/` | `decisions.json` (budget, compact backbone, freeze time + file hashes), `drive_split.json`, `PROTOCOL.md` |
| `data/` | dataset manifest (source archives, SHA-256) and statistics |
| `tables/main_{raw,clean,tuned}.{md,csv}` | one row per model: mean±sd over seeds. raw = threshold 0.5 (headline); clean = + fragment removal; tuned = threshold + fragment size from DRIVE-val |
| `tables/per_run_*.csv`, `per_dataset.csv`, `per_image_all.csv.gz` | the same numbers at finer granularity |
| `tables/comparisons.{md,csv}` | paired seed comparisons with 95% t intervals, Holm-adjusted p, image-level bootstrap |
| `tables/factorial.md`, `efficiency.*`, `uncertainty.*`, `selection.csv`, `training.csv` | 2x2x2 design, cost, RQ4, per-run DRIVE-val choices, training records |
| `figures/` | stages, cleanup/threshold, accuracy vs CPU time, risk-coverage, training curves, qualitative |
| `runs/<exp>/seed<k>/` | config, training log, DRIVE-val selections and per-image metrics of every run (checkpoints stay in `runs/`) |

Models are named `<experiment>[<variant>]`: `det` = deterministic, `mc` = MC dropout (20 passes), `ens` = 5-seed ensemble.
"Unseen" = mean over STARE, CHASE_DB1 and HRF (equal weight); FIVES is reported separately; gap = DRIVE Dice - unseen Dice.
"""
    (out / "README.md").write_text(txt, encoding="utf-8")
