"""Export everything the project website (``docs/``) shows, straight from ``results/tables`` and ``results/runs`` so the page cannot drift from the data.

    python scripts/make_website_data.py     # -> docs/data/results.js, docs/img/*.png (copies of the report figures), docs/assets/viewer/*

The interactive viewer needs, for one image per dataset (seed 0): the photograph, the ground truth, the FOV mask and the probability maps of
stages A, C and E (plus the MC-dropout uncertainty of E), all at the FOV-normalised scale the models run at.
"""
from __future__ import annotations

import json
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

from tglite.config import ROOT
from tglite.data import load_sample
from tglite.experiments import make_config

sys.path.insert(0, str(ROOT / "scripts"))
from make_report_assets import bench_latency, val  # noqa: E402

T = ROOT / "results" / "tables"
DOCS = ROOT / "docs"
DATASETS = ["drive", "stare", "chase_db1", "hrf"]
STAGES = [("A_unet", "det", "A", "U-Net"), ("B_unet_topo", "det", "B", "+ topology loss"), ("C_unet_topo_dg", "det", "C", "+ DG augmentation"),
          ("D_lite_topo_dg", "det", "D", "+ compact backbone"), ("E_tglitenet", "det", "E", "TG-LiteNet"), ("E_tglitenet", "mc", "E-MC", "TG-LiteNet + MC dropout")]
CONTROLS = [("A_unet", "A  U-Net"), ("B_unet_topo", "B  + topology"), ("C_unet_topo_dg", "C  + DG aug."), ("D_lite_topo_dg", "D  compact"),
            ("E_tglitenet", "E  TG-LiteNet"), ("X_unet_dg", "U-Net + DG only"), ("S_unet_b8", "plain U-Net, base 8"), ("S_unet_b16", "plain U-Net, base 16"),
            ("X_lite_dg", "compact + DG only")]
VIEW_MODELS = [("A_unet", "A"), ("C_unet_topo_dg", "C"), ("E_tglitenet", "E")]
FIGURES = ["fig_results", "fig_cleanup", "fig_tradeoff", "fig_risk_coverage", "fig_qualitative", "data_overview"]


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def build() -> dict:
    summ, eff = pd.read_csv(T / "summary.csv"), pd.read_csv(T / "efficiency.csv")
    lat = bench_latency(T)
    post, choices = pd.read_csv(T / "postproc_summary.csv"), pd.read_csv(T / "postproc_choices.csv")
    unc = pd.read_csv(T / "uncertainty.csv")
    out: dict = {"datasets": DATASETS}

    stages = []
    for exp, mode, tag, label in STAGES:
        e = eff[eff.exp == exp].iloc[0]
        row = {"tag": tag, "label": label, "exp": exp, "params_m": r(e.params_m, 4), "gflops": r(e.gflops, 1), "cpu_ms": r(lat.get(exp), 0),
               "mc_passes": int(e.mc_samples) if mode == "mc" else 1, "seeds": int(val(summ, exp, "drive", "dice", mode, "n"))}
        for ds in DATASETS + ["unseen"]:
            for met in ("dice", "cldice", "betti0_err"):
                row[f"{ds}_{met}"] = r(val(summ, exp, ds, met, mode))
                row[f"{ds}_{met}_std"] = r(val(summ, exp, ds, met, mode, "std"))
        row["gap"] = r(val(summ, exp, "gap_unseen", "dice", mode))
        stages.append(row)
    out["stages"] = stages

    controls = []
    for exp, label in CONTROLS:
        d = {"exp": exp, "label": label, "seeds": int(post[(post.exp == exp) & (post["mode"] == "pp")].n_seeds.iloc[0])}
        for mode, key in (("fixed 0.5", "fixed"), ("pp", "clean"), ("thr", "tuned")):
            q = post[(post.exp == exp) & (post["mode"] == mode)].iloc[0]
            d[key] = {m: r(q[f"unseen_{m}"]) for m in ("dice", "cldice", "betti0_err")}
        t = choices[(choices.exp == exp) & (choices["mode"] == "thr")].thr
        d["thr_range"] = [r(t.min(), 2), r(t.max(), 2)]
        controls.append(d)
    out["controls"] = controls

    names = {"A_unet": "A", "B_unet_topo": "B", "C_unet_topo_dg": "C", "D_lite_topo_dg": "D", "E_tglitenet": "E", "S_unet_b8": "plain U-Net, base 8",
             "S_unet_b16": "plain U-Net, base 16", "X_unet_dg": "U-Net + DG only", "X_lite_dg": "compact + DG only"}
    pts = []
    for exp in sorted(set(summ.exp)):
        e = eff[eff.exp == exp]
        if e.empty or exp not in lat:
            continue
        pts.append({"exp": exp, "name": names.get(exp, exp), "main": exp in names, "cpu_ms": r(lat[exp], 0), "params_m": r(e.params_m.iloc[0], 3),
                    "dice": r(val(summ, exp, "unseen", "dice")), "cldice": r(val(summ, exp, "unseen", "cldice")),
                    "betti0": r(val(summ, exp, "unseen", "betti0_err"), 1)})
    out["tradeoff"] = pts

    info = {}
    for ds in DATASETS:
        st = json.loads((ROOT / "data" / "processed" / ds / "stats.json").read_text())
        hw = next(iter(st["per_image"].values()))["hw"]
        info[ds] = {"n": int(st["n_images"]), "w": int(hw[1]), "h": int(hw[0]), "vessel_pct": r(100 * st["vessel_fraction_mean"], 1), "fov_px": int(round(st["fov_width_median"])),
                    "calibre_native": r(st["vessel_width_native_px"], 1), "calibre_norm": r(st["vessel_width_at_527px"], 1), "green": f"{st['green_mean']:.0f} ± {st['green_std']:.0f}"}
    out["dataset_info"] = info

    cells = pd.read_csv(T / "factorial_cells.csv")
    out["factorial"] = [{"backbone": c.backbone, "topology": bool(c.topology != "none" and c.topology is not False and str(c.topology) != "False"),
                         "dg": bool(c.dg), "exp": c.exp, "dice": r(val(summ, c.exp, "unseen", "dice")), "cldice": r(val(summ, c.exp, "unseen", "cldice")),
                         "betti0": r(val(summ, c.exp, "unseen", "betti0_err"), 1), "gap": r(val(summ, c.exp, "gap_unseen", "dice"))}
                        for c in cells.itertuples()]

    risk = {}
    pcov, icov = [1.0, 0.9, 0.8, 0.5], [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for ds in DATASETS:
        q = unc[(unc.exp == "E_tglitenet") & (unc["mode"] == "mc") & (unc.dataset == ds)].iloc[0]
        risk[ds] = {"rho_px": r(q.rho_var_mean, 3), "rho_img": r(q.img_spearman_var_mean, 3), "pixel": [r(q[f"perr_var_c{c}_mean"]) for c in pcov],
                    "image": [r(q[f"img_risk_var_c{c}_mean"]) for c in icov]}
    out["risk"] = {"pixel_cov": pcov, "image_cov": icov, "datasets": risk}

    fa = T / "failure_analysis.csv"
    if fa.exists():
        f = pd.read_csv(fa)
        f = f[f.dataset == "unseen"]
        out["failure"] = [{"exp": e, "kind": k, "group": g, "value": r(v)} for e, k, g, v in zip(f.exp, f.kind, f.group, f.value)]
    return out


def viewer(cfg) -> dict:
    dest = DOCS / "assets" / "viewer"
    dest.mkdir(parents=True, exist_ok=True)
    runs = ROOT / "results" / "runs"
    manifest = {}
    for ds in DATASETS:
        pdir = runs / "E_tglitenet" / "seed0" / "preds" / ds
        ids = sorted(p.name[:-9] for p in pdir.glob("*_prob.png"))
        sid = ids[len(ids) // 2]
        s = load_sample(ds, sid, cfg, cfg.eval.scale_mode)
        h, w = s["mask"].shape
        cv2.imwrite(str(dest / f"{ds}_img.jpg"), cv2.cvtColor(s["img"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(dest / f"{ds}_gt.png"), ((s["mask"] > 0) * 255).astype(np.uint8))
        cv2.imwrite(str(dest / f"{ds}_fov.png"), ((s["fov"] > 0) * 255).astype(np.uint8))
        for exp, tag in VIEW_MODELS:
            shutil.copyfile(runs / exp / "seed0" / "preds" / ds / f"{sid}_prob.png", dest / f"{ds}_prob_{tag}.png")
        shutil.copyfile(pdir / f"{sid}_unc.png", dest / f"{ds}_unc_E.png")
        manifest[ds] = {"id": sid, "w": int(w), "h": int(h)}
        print("viewer", ds, sid, w, h)
    return manifest


def main() -> None:
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "img").mkdir(parents=True, exist_ok=True)
    data = build()
    data["viewer"] = viewer(make_config("A_unet"))
    (DOCS / "data" / "results.js").write_text("window.RESULTS = " + json.dumps(data, indent=1) + ";\n", encoding="utf-8")
    for f in FIGURES:
        shutil.copyfile(ROOT / "report" / "figures" / f"{f}.png", DOCS / "img" / f"{f}.png")
    shutil.copyfile(ROOT / "report" / "main.pdf", DOCS / "paper.pdf")
    print("wrote docs/data/results.js and", len(FIGURES), "figures")


if __name__ == "__main__":
    main()
