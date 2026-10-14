"""Print (and save to results/findings.md) the factual numbers behind each research question, with confidence intervals.

    python scripts/aggregate_results.py && python scripts/auto_findings.py

This is a *writing aid*: it states what the data show (differences, 95 % Welch intervals over seeds, whether the interval
excludes zero) but does not interpret them -- the Discussion in the paper must explain the *why*.
"""
from __future__ import annotations

import argparse

import pandas as pd

from tglite.config import resolve_path


def fmt_delta(row, lower_better: bool) -> str:
    d, ci = row["delta"], row["ci95"]
    sig = "n/a" if pd.isna(ci) else ("CI excludes 0" if abs(d) > ci else "CI includes 0")
    better = "" if pd.isna(ci) or abs(d) <= ci else (" (improves)" if (d < 0) == lower_better else " (worsens)")
    ci_txt = "n/a" if pd.isna(ci) else f"±{ci:.4f}"
    return f"{d:+.4f} [{ci_txt}] {sig}{better}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()
    t = resolve_path(args.tables)
    sd = pd.read_csv(t / "stage_deltas.csv")
    summ = pd.read_csv(t / "summary.csv")
    eff = pd.read_csv(t / "efficiency.csv") if (t / "efficiency.csv").exists() else pd.DataFrame()
    unc = pd.read_csv(t / "uncertainty.csv") if (t / "uncertainty.csv").exists() else pd.DataFrame()
    low = {"betti0_err": True, "dice": False, "cldice": False, "auc": False}
    out = ["# Auto-generated findings (facts only)\n", "Deltas are (second − first) in mean over seeds; brackets are 95 % Welch intervals.\n"]

    def block(title, a, b):
        out.append(f"\n## {title}\n")
        sub = sd[(sd["from"] == a) & (sd["to"] == b)]
        if sub.empty:
            out.append("(no data)\n")
            return
        for ds in ("drive", "unseen", "gap_unseen"):
            for met in ("dice", "cldice", "betti0_err", "auc"):
                r = sub[(sub.dataset == ds) & (sub.metric == met)]
                if len(r):
                    lb = low.get(met, False) or ds.startswith("gap")
                    out.append(f"- {ds:11s} {met:11s} {a} → {b}: {fmt_delta(r.iloc[0], lb)}")

    block("RQ1 topology: A → B (adds soft-clDice + skeleton head)", "A_unet", "B_unet_topo")
    block("RQ2 generalisation: B → C (adds DG augmentation + consistency)", "B_unet_topo", "C_unet_topo_dg")
    block("RQ3 trade-off: C → D (compact backbone)", "C_unet_topo_dg", "D_lite_topo_dg")
    block("E (dropout trained in, deterministic eval) vs D", "D_lite_topo_dg", "E_tglitenet")
    block("MC averaging effect: E det → E MC", "E_tglitenet", "E_tglitenet[mc]")
    block("Overall: A → E (MC)", "A_unet", "E_tglitenet[mc]")

    if len(eff):
        out.append("\n## Cost\n")
        a, d, e = (eff[eff.exp == x] for x in ("A_unet", "D_lite_topo_dg", "E_tglitenet"))
        if len(a) and len(e):
            a, e = a.iloc[0], e.iloc[0]
            out.append(f"- parameters: U-Net {a.params_m:.2f} M vs TG-LiteNet {e.params_m:.3f} M  (×{a.params_m / e.params_m:.0f} fewer)")
            out.append(f"- FLOPs @512²: {a.gflops:.1f} vs {e.gflops:.2f} GFLOPs (×{a.gflops / e.gflops:.0f})")
            out.append(f"- CPU latency (4 threads): {a.cpu_ms:.0f} ms vs {e.cpu_ms:.0f} ms; MC dropout multiplies the latter by T={int(e.mc_samples)}")

    if len(unc):
        out.append("\n## RQ4 uncertainty (stage E, MC mode)\n")
        for ds in ("drive", "stare", "chase_db1", "hrf"):
            r = unc[(unc.exp == "E_tglitenet") & (unc["mode"] == "mc") & (unc.dataset == ds)]
            if r.empty:
                continue
            r = r.iloc[0]
            out.append(f"- {ds:10s} rho_px(var)={r.get('rho_var_mean', float('nan')):.3f}  rho_px(ent)={r.get('rho_ent_mean', float('nan')):.3f}  "
                       f"rho_img(var)={r.get('img_spearman_var_mean', float('nan')):.3f}  AURC(var)={r.get('aurc_var_mean', float('nan')):.4f} "
                       f"(random {r.get('risk_random_mean', float('nan')):.4f}, oracle {r.get('aurc_oracle_mean', float('nan')):.4f})  "
                       f"pixel err@90% cov: {r.get('perr_var_c0.9_mean', float('nan')):.4f} vs {r.get('perr_var_c1.0_mean', float('nan')):.4f} at 100%")

    text = "\n".join(out) + "\n"
    resolve_path("results/findings.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
