"""Aggregation of run outputs into the tables behind the paper (pure pandas; no plotting here).

Terminology
-----------
run        one (experiment, seed) pair.             method  an experiment evaluated in one mode
(``E_tglitenet`` has two methods: ``det`` = dropout off, ``mc`` = T dropout passes; all others only ``det``).
``unseen`` = macro average over STARE, CHASE_DB1 and HRF (each dataset weighs the same, whatever its size).
Confidence intervals are 95 % Student-t intervals over *seeds* (training randomness), not over images.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .experiments import FACTORIAL_CELLS, GROUPS
from .uncertainty import image_risk_coverage

CORE_TARGETS = ["stare", "chase_db1", "hrf"]
SEG = ["dice", "iou", "sens", "spec", "auc"]
TOPO = ["cldice", "cbdice", "betti0_err", "betti1_err"]
METRICS = SEG + TOPO
GAP_METRICS = ["dice", "iou", "auc", "cldice", "cbdice"]          # Betti counts are not comparable across image sizes
LOWER_IS_BETTER = {"betti0_err", "betti1_err", "gap_dice", "gap_cldice", "gap_iou", "gap_auc"}


# -------------------------------------------------------------------------------------------- loading
def load_runs(runs_dir: str | Path, tag: str = "") -> pd.DataFrame:
    """Concatenate every ``per_image[_tag].csv`` under ``runs_dir``."""
    runs_dir = Path(runs_dir)
    name = f"per_image_{tag}.csv" if tag else "per_image.csv"
    frames = [pd.read_csv(p) for p in sorted(runs_dir.glob(f"*/seed*/{name}"))]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_efficiency(runs_dir: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_dir).glob("*/seed*/efficiency.json")):
        e = json.loads(p.read_text())
        exp = p.parent.parent.name
        row = {"exp": exp, "params_m": e["params_m"], "gflops": e["gflops"], "gmacs": e["gmacs"],
               "cpu_ms": e["cpu"]["median_ms"], "cpu1_ms": e["cpu_1thread"]["median_ms"],
               "mc_samples": e.get("mc_samples", 1), "input_hw": "x".join(map(str, e["input_hw"]))}
        if "gpu" in e:
            row.update(gpu_ms=e["gpu"]["median_ms"], gpu_mem_mb=e["gpu"].get("peak_mem_mb", np.nan))
        row["hardware"] = json.dumps(e.get("hardware", {}))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("exp", keep="first")


# ------------------------------------------------------------------------------------------ run tables
def per_run_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over images for each (exp, mode, seed, dataset) + the macro-average 'unseen' row."""
    cols = [c for c in METRICS if c in df.columns]
    base = df.groupby(["exp", "mode", "seed", "dataset"], as_index=False)[cols].mean()
    tgt = base[base.dataset.isin(CORE_TARGETS)]
    have = tgt.groupby(["exp", "mode", "seed"]).dataset.nunique()
    keep = have[have == len(CORE_TARGETS)].index
    dropped = sorted(set(base.set_index(["exp", "mode", "seed"]).index) - set(keep))
    if dropped:
        import warnings
        warnings.warn(f"{len(dropped)} run(s) lack one of {CORE_TARGETS} and are excluded from the 'unseen' average: {dropped[:5]}...")
    if len(keep):
        un = tgt.set_index(["exp", "mode", "seed"]).loc[keep].reset_index().groupby(["exp", "mode", "seed"], as_index=False)[cols].mean()
        un["dataset"] = "unseen"
        base = pd.concat([base, un], ignore_index=True)
    return base


def add_gaps(run_tab: pd.DataFrame) -> pd.DataFrame:
    """Add rows ``dataset = gap_<target>`` holding  m(DRIVE) - m(target)  per run, for GAP_METRICS."""
    src = run_tab[run_tab.dataset == "drive"].set_index(["exp", "mode", "seed"])
    out = []
    for ds in [d for d in run_tab.dataset.unique() if d != "drive" and not d.startswith("gap_")]:
        tgt = run_tab[run_tab.dataset == ds].set_index(["exp", "mode", "seed"])
        idx = src.index.intersection(tgt.index)
        g = (src.loc[idx, GAP_METRICS] - tgt.loc[idx, GAP_METRICS]).reset_index()
        g["dataset"] = f"gap_{ds}"
        out.append(g)
    return pd.concat([run_tab] + out, ignore_index=True) if out else run_tab


def summarise(run_tab: pd.DataFrame) -> pd.DataFrame:
    """mean / std / 95 % CI over seeds -> long table (exp, mode, dataset, metric, ...)."""
    long = run_tab.melt(id_vars=["exp", "mode", "seed", "dataset"], var_name="metric", value_name="value").dropna(subset=["value"])
    g = long.groupby(["exp", "mode", "dataset", "metric"])["value"]
    s = g.agg(["mean", "std", "count"]).reset_index().rename(columns={"count": "n_seeds"})
    s["std"] = s["std"].fillna(0.0)
    tcrit = np.where(s.n_seeds > 1, stats.t.ppf(0.975, np.maximum(s.n_seeds - 1, 1)), np.nan)
    s["ci95"] = tcrit * s["std"] / np.sqrt(s.n_seeds)
    return s


def method_name(exp: str, mode: str) -> str:
    return exp if mode == "det" else f"{exp}[{mode}]"


def welch_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """mean(a) - mean(b), with 95 % Welch CI half-width and two-sided p-value (nan when n < 2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    if len(a) < 2 or len(b) < 2:
        return d, float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return d, 0.0, 0.0 if d != 0 else 1.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return d, float(stats.t.ppf(0.975, df) * se), float(2 * stats.t.sf(abs(d / se), df))


def paired_delta(a: pd.Series, b: pd.Series) -> tuple[float, float, float, str, int]:
    """mean(a) - mean(b) with a 95 % interval half-width and two-sided p-value, plus the test used and the number of pairs.

    ``a`` and ``b`` are per-seed values indexed by seed.  Methods trained with the same seed identifier are compared as
    pairs (Student-t on the per-seed differences, which keeps the seed-to-seed correlation); the unpaired Welch test is
    only used when fewer than two seeds are shared.
    """
    shared = a.index.intersection(b.index)
    if len(shared) >= 2:
        diff = (a.loc[shared] - b.loc[shared]).to_numpy(float)
        d, sd = diff.mean(), diff.std(ddof=1)
        if sd == 0:
            return float(d), 0.0, (1.0 if d == 0 else 0.0), "paired-t", len(shared)
        se = sd / np.sqrt(len(diff))
        return float(d), float(stats.t.ppf(0.975, len(diff) - 1) * se), float(2 * stats.t.sf(abs(d / se), len(diff) - 1)), "paired-t", len(shared)
    d, ci, p = welch_delta(a.to_numpy(float), b.to_numpy(float))
    return d, ci, p, "welch", 0


def stage_deltas(run_tab: pd.DataFrame, pairs: list[tuple[tuple[str, str], tuple[str, str]]],
                 datasets=("drive", "unseen", "gap_unseen"), metrics=("dice", "cldice", "cbdice", "betti0_err", "auc")) -> pd.DataFrame:
    """Differences (second minus first) between methods, each given as (exp, mode); paired over seeds where possible."""
    rows = []
    for (e1, m1), (e2, m2) in pairs:
        for ds in datasets:
            for met in metrics:
                sel = lambda e, m: run_tab[(run_tab.exp == e) & (run_tab["mode"] == m) & (run_tab.dataset == ds)].dropna(subset=[met]).set_index("seed")[met]  # noqa: E731
                a, b = sel(e2, m2), sel(e1, m1)
                if len(a) == 0 or len(b) == 0:
                    continue
                d, ci, p, test, n_pairs = paired_delta(a, b)
                rows.append({"from": method_name(e1, m1), "to": method_name(e2, m2), "dataset": ds, "metric": met,
                             "delta": d, "ci95": ci, "p": p, "test": test, "n_pairs": n_pairs, "n_from": len(b), "n_to": len(a)})
    return pd.DataFrame(rows)


def paired_image_deltas(df: pd.DataFrame, pairs, metrics=("dice", "cldice", "betti0_err"), n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Paired comparison on the test *images*: for every image the metric is averaged over the seeds of each method and the
    second method is compared with the first on that same image.

    ``unseen`` uses a stratified bootstrap that matches the macro average of the tables: images are resampled within each of
    STARE, CHASE_DB1 and HRF, the three dataset means of the difference are averaged, and the 2.5/97.5 percentiles of that
    statistic give the interval (``delta`` is the same macro average).  ``frac_higher``/``frac_lower`` and the Wilcoxon p-value
    are pooled over the 93 images and are descriptive.  This captures test-set sampling variability only; training randomness
    is covered by the seed-level intervals of ``stage_deltas``.  Many comparisons are made without multiplicity correction.
    """
    d = df.assign(key=df.dataset + "/" + df.id.astype(str))
    rng = np.random.default_rng(seed)
    rows = []
    for (e1, m1), (e2, m2) in pairs:
        for scope, dsets in (("unseen", CORE_TARGETS), ("drive", ["drive"])):
            for met in metrics:
                a = d[(d.exp == e1) & (d["mode"] == m1) & d.dataset.isin(dsets)].groupby("key")[met].mean()
                b = d[(d.exp == e2) & (d["mode"] == m2) & d.dataset.isin(dsets)].groupby("key")[met].mean()
                diff = (b - a).dropna()
                if len(diff) < 5:
                    continue
                strata = [diff[diff.index.str.startswith(ds + "/")].to_numpy() for ds in dsets]
                strata = [x for x in strata if len(x)]
                macro = float(np.mean([x.mean() for x in strata]))
                boots = np.mean([x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1) for x in strata], axis=0)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                pooled = diff.to_numpy()
                try:
                    p = float(stats.wilcoxon(pooled).pvalue)
                except ValueError:
                    p = float("nan")
                rows.append({"from": method_name(e1, m1), "to": method_name(e2, m2), "scope": scope, "metric": met, "delta": macro,
                             "lo95": float(lo), "hi95": float(hi), "frac_higher": float((pooled > 0).mean()), "frac_lower": float((pooled < 0).mean()),
                             "p_wilcoxon": p, "n_images": len(pooled)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ uncertainty
def uncertainty_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (exp, mode, seed, dataset): Spearman(pixel), image-level Spearman, AURC and pixel error vs coverage."""
    rows = []
    for (exp, mode, seed, ds), g in df.groupby(["exp", "mode", "seed", "dataset"]):
        row = {"exp": exp, "mode": mode, "seed": seed, "dataset": ds, "n_images": len(g)}
        err = (1.0 - g["dice"]).to_numpy()
        for kind in ("var", "ent"):
            if f"rho_{kind}" not in g or g[f"unc_mean_{kind}"].isna().all():
                continue
            row[f"rho_{kind}"] = g[f"rho_{kind}"].mean()
            u = g[f"unc_mean_{kind}"].to_numpy()
            row[f"img_spearman_{kind}"] = float(stats.spearmanr(u, err).statistic) if len(g) > 3 and np.std(u) > 0 else np.nan
            rc = image_risk_coverage(u, err)
            row[f"aurc_{kind}"] = rc["aurc"]
            row["aurc_oracle"] = rc["aurc_oracle"]
            row["risk_random"] = rc["random"]
            for c, v in rc["curve"].items():
                row[f"img_risk_{kind}_c{c}"] = v
            for col in g.columns:
                if col.startswith(f"perr_{kind}_c"):
                    row[col] = g[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_uncertainty(unc: pd.DataFrame) -> pd.DataFrame:
    val = [c for c in unc.columns if c not in ("exp", "mode", "seed", "dataset", "n_images")]
    g = unc.groupby(["exp", "mode", "dataset"])[val]
    m, s = g.mean(), g.std().fillna(0.0)
    out = m.add_suffix("_mean").join(s.add_suffix("_std")).reset_index()
    return out


# ------------------------------------------------------------------------------------------ confusion matrix
def _class_sizes(df: pd.DataFrame, processed_dir: Path) -> dict:
    """{(dataset, id): (vessel pixels, background pixels)} inside the FOV, at the native resolution the metrics use."""
    import cv2

    sizes = {}
    for ds, iid in df[["dataset", "id"]].drop_duplicates().itertuples(index=False):
        root = processed_dir / ("drive" if ds.startswith("drive") else ds)
        gt = cv2.imread(str(root / "masks" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE) > 127
        fov = cv2.imread(str(root / "fov" / f"{iid}.png"), cv2.IMREAD_GRAYSCALE) > 127
        sizes[(ds, iid)] = (float((gt & fov).sum()), float(fov.sum() - (gt & fov).sum()))
    return sizes


def confusion_summary(runs_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    """Pixel-level confusion matrix, precision, recall, F1 and accuracy per (experiment, mode, dataset).

    The stored per-image sensitivity and specificity plus the exact ground-truth class sizes inside the FOV give the
    confusion counts without re-running any model: TP = sens * P, FN = P - TP, TN = spec * N, FP = N - TN.  Cells are
    reported as a share of the FOV pixels; every statistic is a mean over images and then over seeds, like all other
    metrics (F1 equals the per-image Dice).  ``unseen`` is the macro average over STARE, CHASE_DB1 and HRF.
    """
    frames = []
    for f in sorted(Path(runs_dir).glob("*/seed*/per_image.csv")):
        frames.append(pd.read_csv(f, dtype={"id": str}, usecols=["exp", "seed", "mode", "dataset", "id", "sens", "spec", "acc"]))
    if not frames or not Path(processed_dir).exists():
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    sizes = _class_sizes(df, Path(processed_dir))
    P = np.array([sizes[(d, i)][0] for d, i in zip(df.dataset, df.id)])
    N = np.array([sizes[(d, i)][1] for d, i in zip(df.dataset, df.id)])
    F = P + N
    df["tp"], df["fn"] = df.sens * P / F, (1 - df.sens) * P / F
    df["tn"], df["fp"] = df.spec * N / F, (1 - df.spec) * N / F
    assert np.abs(df.tp + df.tn - df.acc).max() < 1e-6, "derived accuracy disagrees with the stored accuracy"
    df["precision"] = df.tp / (df.tp + df.fp).replace(0, np.nan)
    df["recall"] = df.sens
    df["f1"] = 2 * df.tp / (2 * df.tp + df.fp + df.fn)
    df["accuracy"] = df.tp + df.tn
    cols = ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"]
    per_seed = df.groupby(["exp", "mode", "seed", "dataset"])[cols].mean().reset_index()
    out = per_seed.groupby(["exp", "mode", "dataset"])[cols].mean().reset_index()
    tgt = out[out.dataset.isin(CORE_TARGETS)].groupby(["exp", "mode"])[cols].mean().reset_index().assign(dataset="unseen")
    return pd.concat([out, tgt], ignore_index=True)


# ------------------------------------------------------------------------------------------ the bundle
def build_all(runs_dir: str | Path, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Compute and write every CSV table; return them (empty dict if there are no runs)."""
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_runs(runs_dir)
    if df.empty:
        return {}
    tabs: dict[str, pd.DataFrame] = {}
    run_tab = add_gaps(per_run_table(df))
    tabs["per_run"] = run_tab
    tabs["summary"] = summarise(run_tab)
    tabs["efficiency"] = load_efficiency(runs_dir)
    unc = uncertainty_table(df)
    tabs["uncertainty_per_run"] = unc
    tabs["uncertainty"] = summarise_uncertainty(unc)

    present = set(zip(run_tab.exp, run_tab["mode"]))
    chain = [("A_unet", "det"), ("B_unet_topo", "det"), ("C_unet_topo_dg", "det"), ("D_lite_topo_dg", "det"),
             ("E_tglitenet", "det"), ("E_tglitenet", "mc")]
    pairs = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    pairs += [(("D_lite_topo_dg", "det"), ("E_tglitenet", "mc")), (("A_unet", "det"), ("E_tglitenet", "mc"))]
    pairs = [(a, b) for a, b in pairs if a in present and b in present]
    tabs["stage_deltas"] = stage_deltas(run_tab, pairs)

    cells = []
    for (bb, topo, dg), exp in FACTORIAL_CELLS.items():
        if (exp, "det") in present:
            cells.append({"backbone": bb, "topology": topo != "none", "dg": dg, "exp": exp})
    if cells:
        tabs["factorial_cells"] = pd.DataFrame(cells)

    # paired comparisons used in the ablation section: factor effects within the factorial, and variants / widths vs stage D
    cell = FACTORIAL_CELLS.get
    fpairs = []
    for (bb, topo, dg), exp in FACTORIAL_CELLS.items():
        if topo != "none" and cell((bb, "none", dg)):                       # effect of the topology loss
            fpairs.append(((cell((bb, "none", dg)), "det"), (exp, "det")))
        if dg and cell((bb, topo, False)):                                   # effect of DG augmentation
            fpairs.append(((cell((bb, topo, False)), "det"), (exp, "det")))
        if bb == "lite" and cell(("unet", topo, dg)):                        # effect of the compact backbone
            fpairs.append(((cell(("unet", topo, dg)), "det"), (exp, "det")))
    fpairs.append((("A_unet", "det"), ("X_lite_dg", "det")))
    fpairs += [(("D_lite_topo_dg", "det"), (e, "det")) for e in
               ("V_lite_cbdice", "V_lite_srl", "V_lite_cldice_srl", "W_lite_w8", "W_lite_w24", "W_lite_w32", "W_lite_w48",
                "S_unet_b8", "S_unet_b16", "Z_lite_noskel", "Z_lite_nocons")]
    fpairs = [(a, b) for a, b in fpairs if a in present and b in present]
    if fpairs:
        tabs["factorial_deltas"] = stage_deltas(run_tab, fpairs)
    all_pairs = list(dict.fromkeys(pairs + fpairs))                          # the stage chain and the factorial list overlap
    tabs["paired_image_deltas"] = paired_image_deltas(df, all_pairs)

    conf = confusion_summary(runs_dir, Path(__file__).resolve().parents[1] / "data" / "processed")
    if len(conf):
        tabs["confusion"] = conf

    nat = load_runs(runs_dir, "native")
    if not nat.empty:
        keys = set(zip(nat.exp, nat["mode"], nat.seed))                            # native evaluation exists for a few runs only
        src = df[df.dataset == "drive"]
        src = src[[k in keys for k in zip(src.exp, src["mode"], src.seed)]]
        nt = add_gaps(per_run_table(pd.concat([nat, src.assign(scale_mode="fov")], ignore_index=True)))
        tabs["native_summary"] = summarise(nt)
    for name, t in tabs.items():
        if len(t):
            t.to_csv(out_dir / f"{name}.csv", index=False)
    return tabs


def get(summary: pd.DataFrame, exp: str, dataset: str, metric: str, mode: str = "det") -> tuple[float, float, int]:
    """(mean, std, n) from a summary table; (nan, nan, 0) if missing."""
    r = summary[(summary.exp == exp) & (summary["mode"] == mode) & (summary.dataset == dataset) & (summary.metric == metric)]
    if r.empty:
        return float("nan"), float("nan"), 0
    r = r.iloc[0]
    return float(r["mean"]), float(r["std"]), int(r.n_seeds)
