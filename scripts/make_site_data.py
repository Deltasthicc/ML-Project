"""Export everything the project website (``docs/``) shows, straight from ``results/`` (never typed in by hand).

    python scripts/make_site_data.py     # -> docs/data/site.js, docs/assets/ (copy of results/v21/assets), docs/img/ (paper figures), docs/paper.pdf

The live viewer thresholds the exported probability maps in the browser; the thresholds are each model's DRIVE-validation choice
(``assets.json``), so the on-page Dice equals the value written to ``assets.json`` (checked in ``docs/assets/assets.json``).
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R2, R21 = ROOT / "results" / "tables", ROOT / "results" / "v21" / "tables"
DOCS = ROOT / "docs"


def pm(s) -> tuple[float, float | None]:
    if not isinstance(s, str):
        return float(s), None
    a, _, b = s.partition("±")
    return float(a), (float(b) if b else None)


def kparams(s) -> float:
    s = str(s).strip()
    return float(s[:-1]) * 1e3 if s.endswith("K") else float(s)


def md_rows(path: Path) -> list[list[str]]:
    rows, started = [], False
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("|"):
            started = True
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
        elif started:
            break
    return rows[1:]


LABEL = {
    "A_unet[det]": ("A  U-Net", "v2", "stage"), "B_unet_topo[det]": ("B  + topology loss", "v2", "stage"), "C_unet_topo_dg[det]": ("C  + DG augmentation", "v2", "stage"),
    "D_compact[det]": ("D  compact U-Net (486K)", "v2", "stage"), "E_tglitenet[det]": ("E  TG-LiteNet v2 (486K)", "v2", "stage"), "E_tglitenet[mc]": ("E  + MC dropout", "v2", "stage"),
    "ENS_A_unet[ens]": ("5 x U-Net ensemble", "v2", "ensemble"), "ENS_D_compact[ens]": ("5 x compact ensemble", "v2", "ensemble"),
    "H_tglite70k[det]": ("TG-LiteNet v2.1 (66K)", "v2.1", "headline"), "ENS_H_tglite70k[ens]": ("5 x 66K ensemble", "v2.1", "ensemble"),
    "R_lite_w16[det]": ("Depthwise 212K (size-rule headline)", "v2.1", "reference"), "R_unet_w8[det]": ("U-Net w8 486K (v2.1 recipe)", "v2.1", "reference"),
    "Z_no_kd[det]": ("66K without distillation", "v2.1", "ablation"), "Z_no_kdskel[det]": ("66K without N3 (centreline distillation)", "v2.1", "ablation"),
    "Z_noaug[det]": ("66K without RandConv + Fourier", "v2.1", "ablation"), "Z_pixcons[det]": ("66K with pixel consistency (no N4)", "v2.1", "ablation"),
    "Z_v2recipe[det]": ("66K with the v2 recipe", "v2.1", "ablation"),
}


def build() -> dict:
    m21 = pd.read_csv(R21 / "main.csv").set_index("model")
    out: dict = {}

    models = []
    for key, r in m21.iterrows():
        lab, ver, kind = LABEL[key]
        g = lambda c: pm(r[c])  # noqa: E731
        models.append({"key": key, "label": lab, "version": ver, "kind": kind, "n": int(r["n"]), "params": kparams(r["params"]), "gflops": float(r["GFLOPs"]), "cpu_ms": float(r["CPU ms"]),
                       "drive": g("DRIVE Dice")[0], "stare": g("STARE")[0], "chase": g("CHASE")[0], "hrf": g("HRF")[0], "fives": g("FIVES")[0],
                       "raw": g("unseen Dice raw")[0], "tuned": g("unseen Dice tuned")[0], "tuned_sd": g("unseen Dice tuned")[1], "sel": g("unseen Dice scale-sel")[0], "sel_sd": g("unseen Dice scale-sel")[1],
                       "cldice": g("unseen clDice")[0], "betti_raw": g("Betti-0 raw")[0], "betti_clean": g("Betti-0 clean")[0], "gap": g("gap")[0], "net_res": g("unseen Dice @network res")[0]})
    out["models"] = models

    # v2 stages and variants: per dataset, per mode, mean and sd over seeds (per-run tables)
    stage_models = {"A": "A_unet[det]", "B": "B_unet_topo[det]", "C": "C_unet_topo_dg[det]", "D": "D_compact[det]", "E": "E_tglitenet[det]"}
    ds = ["drive", "stare", "chase_db1", "hrf", "fives"]
    stages = {}
    for mode in ("raw", "clean", "tuned"):
        pr = pd.read_csv(R2 / f"per_run_{mode}.csv")
        stages[mode] = {}
        for tag, key in stage_models.items():
            sub = pr[pr["model"] == key]
            e = {}
            for met in ("dice", "cldice", "betti0_err"):
                for d in ds + ["unseen"]:
                    col = f"{d}/{met}"
                    e[f"{d}_{met}"] = [round(float(sub[col].mean()), 5), round(float(sub[col].std(ddof=1)), 5)]
            stages[mode][tag] = e
    out["stages"] = stages
    out["stage_labels"] = {k: LABEL[v][0] for k, v in stage_models.items()}

    raw, tuned = (pd.read_csv(R2 / f"main_{m}.csv").set_index("model") for m in ("raw", "tuned"))
    fac = [("A_unet[det]", "U-Net", False, False), ("X_unet_dg[det]", "U-Net", False, True), ("B_unet_topo[det]", "U-Net", True, False), ("C_unet_topo_dg[det]", "U-Net", True, True),
           ("X_compact_plain[det]", "Compact", False, False), ("X_compact_dg[det]", "Compact", False, True), ("X_compact_topo[det]", "Compact", True, False), ("D_compact[det]", "Compact", True, True)]
    out["factorial"] = [{"key": k, "backbone": b, "topology": t, "dg": g, "n": int(raw.loc[k, "n_seeds"]), "dice_raw": float(raw.loc[k, "unseen/dice_mean"]), "cldice_raw": float(raw.loc[k, "unseen/cldice_mean"]),
                         "betti_raw": float(raw.loc[k, "unseen/betti0_err_mean"]), "dice_tuned": float(tuned.loc[k, "unseen/dice_mean"]), "gap_tuned": float(tuned.loc[k, "gap/dice_mean"])} for k, b, t, g in fac]

    # novelty 2: per dataset change in Dice from scale selection (seed-paired mean), and chosen scales for the 66K model
    pr21 = pd.read_csv(R21 / "per_run.csv")
    d4 = ["stare", "chase_db1", "hrf", "fives"]
    scale = {}
    for key in ("H_tglite70k[det]", "R_lite_w16[det]", "R_unet_w8[det]", "E_tglitenet[det]", "D_compact[det]", "C_unet_topo_dg[det]", "A_unet[det]"):
        sub = pr21[pr21["model"] == key]
        scale[key] = {d: float((sub[f"{d}/sel/dice"] - sub[f"{d}/tuned/dice"]).mean()) for d in d4}
    out["scale_delta"] = scale
    n2 = {r[0]: r for r in md_rows(R21 / "novelty2_scale.md")}
    out["scale_unseen"] = {k: {"text": v[4]} for k, v in n2.items()}
    pi = pd.read_csv(R21 / "per_image_all.csv.gz")
    hh = pi[pi["model"] == "H_tglite70k[det]"]
    share = hh.groupby(["dataset", "sel_scale"]).size().unstack(fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)
    out["scale_share"] = {d: {str(c): float(share.loc[d].get(c, 0)) for c in (1.0, 1.5, 2.0, 2.5)} for d in share.index}

    # ablations (novelties 3, 4, augmentation) and novelty 1
    abl = {r[1]: r for r in md_rows(R21 / "ablations.md")}

    def cellv(run, col):
        m = re.search(r"([+-][\d.]+) \[±([\d.]+)\] (not clear|clear)", abl[run][col])
        return {"delta": float(m.group(1)), "ci": float(m.group(2)), "clear": m.group(3) == "clear"}

    out["ablations"] = {run: {"dice": cellv(run, 2), "cldice": cellv(run, 3), "betti": cellv(run, 4)} for run in abl}
    n1 = (R21 / "novelty1_reliability.md").read_text(encoding="utf-8")

    def n1v(pat):
        m = re.search(pat + r" \(rho, unseen mean\): ([+-][\d.]+) \[±([\d.]+)\] (not clear|clear)", n1)
        return {"delta": float(m.group(1)), "ci": float(m.group(2)), "clear": m.group(3) == "clear"}

    out["novelty1"] = {"E_mc": n1v(r"E_tglitenet\[mc\]: rel_frag - entropy"), "E_det": n1v(r"E_tglitenet\[det\]: rel_frag - entropy"), "H": n1v(r"H_tglite70k\[det\]: rel_frag - entropy")}
    m = re.search(r"E: rel_frag\(det\) - MC-dropout entropy \(rho, unseen mean\): ([+-][\d.]+) \[±([\d.]+)\]", n1)
    out["novelty1"]["E_vs_mc"] = {"delta": float(m.group(1)), "ci": float(m.group(2)), "clear": True}

    # uncertainty (v2, unseen mean)
    unc = pd.read_csv(R2 / "uncertainty.csv")
    u = unc[unc["dataset"].isin(["stare", "chase_db1", "hrf"])].copy()
    u["gain"] = (u["aurc_random"] - u["aurc"]) / (u["aurc_random"] - u["aurc_oracle"]) * 100
    ug = u.groupby("model")[["rho_px", "rho_img", "gain", "ece", "ece_ts", "px_err@100", "px_err@90"]].mean()
    out["uncertainty"] = {k: {c: float(ug.loc[k, c]) for c in ug.columns} for k in ("A_unet[det]", "C_unet_topo_dg[det]", "E_tglitenet[det]", "E_tglitenet[mc]", "ENS_D_compact[ens]")}

    # datasets, cost, decisions, int8
    st = pd.read_csv(ROOT / "results" / "data" / "dataset_stats.csv")
    out["datasets"] = [{"id": r.dataset, "n": int(r.n), "w": int(r.w), "h": int(r.h), "vessel_pct": round(100 * r.vessel_frac_fov, 1), "fov": round(r.fov_width_median), "calibre_native": round(r.calibre_native, 1),
                        "calibre_norm": round(r.calibre_norm, 1), "green": f"{r.green_mean:.0f} ± {r.green_std:.0f}", "fov_mask": r.fov} for r in st.itertuples()]
    dec21 = json.loads((ROOT / "protocol" / "decisions_v21.json").read_text())
    dec2 = json.loads((ROOT / "protocol" / "decisions.json").read_text())
    out["decisions"] = {"arch": dec21["arch_selection"]["candidates"], "size_rule": dec21["size_rule"]["candidates"], "backbone": dec2["backbone"]["candidates"], "backbone_choice": dec2["backbone"]["compact_choice"],
                        "frozen_v2": dec2["frozen_at"], "frozen_v21": dec21["frozen_at"]}
    int8 = (R21 / "int8.md").read_text(encoding="utf-8")
    a = re.search(r"fp32 ([\d.]+) ms -> int8 ([\d.]+) ms", int8)
    b = re.search(r"file size: (\d+) KB -> (\d+) KB", int8)
    out["int8"] = {"fp32_ms": float(a.group(1)), "int8_ms": float(a.group(2)), "fp32_kb": int(b.group(1)), "int8_kb": int(b.group(2))}
    eff = pd.read_csv(R2 / "efficiency.csv").set_index("exp")
    out["efficiency"] = {k: {"params_M": float(r["params_M"]), "gflops": float(r["GFLOPs@512"]), "cpu_4t": float(r["cpu_4t_ms"]), "gpu_fp32": float(r["gpu_fp32_ms"])} for k, r in eff.iterrows()}
    out["n_trainings"] = {"v2": int(pd.read_csv(R2 / "training.csv").shape[0]),
                          "v21": sum(1 for d in (ROOT / "results" / "v21" / "runs").iterdir() if d.is_dir() and d.name != "v2_reeval" for _ in d.glob("seed*"))}

    assets = json.loads((DOCS / "assets" / "assets.json").read_text())
    out["viewer"] = assets["datasets"]
    return out


def main() -> None:
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "img").mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "results" / "v21" / "assets", DOCS / "assets", dirs_exist_ok=True)
    for f in ("fig_stages", "fig_modes", "fig_cost_scale"):
        src = ROOT / "paper" / "figures" / f"{f}.png"
        if src.exists():
            shutil.copyfile(src, DOCS / "img" / f"{f}.png")
    shutil.copyfile(ROOT / "results" / "v21" / "figures" / "qualitative_v21.png", DOCS / "img" / "qualitative_v21.png")
    pdf = ROOT / "paper" / "main.pdf"
    if pdf.exists():
        shutil.copyfile(pdf, DOCS / "paper.pdf")
    data = build()
    (DOCS / "data" / "site.js").write_text("window.SITE = " + json.dumps(data, indent=1) + ";\n", encoding="utf-8")
    print("wrote docs/data/site.js:", len(data["models"]), "models")


if __name__ == "__main__":
    main()
