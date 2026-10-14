"""Choose the loss weights lambda_2 (topology) and lambda_3 (consistency) on the DRIVE *validation* images only.

    python scripts/tune_lambdas.py                       # 3 x 3 grid on the stage-D recipe, 1 seed each
    python scripts/tune_lambdas.py --l2 0.25 0.5 1 --l3 0.1 0.5 1 --seeds 0 1

Criterion: mean of validation Dice and validation clDice at the best checkpoint (4 DRIVE-val images).  No STARE / CHASE_DB1 /
HRF image is touched.  The winner is written to ``configs/tuned.yaml``; ``make_config`` merges that file automatically
(commit it so the final runs are reproducible).  If you skip this script the defaults in configs/base.yaml are used.

Caveat: 4 validation images make this selection noisy; differences smaller than ~0.005 should be treated as ties
(we then keep the smaller value of each weight, i.e. the closest to the default).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch
import yaml

from tglite.config import ROOT, resolve_path
from tglite.experiments import make_config
from tglite.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="D_lite_topo_dg")
    ap.add_argument("--l2", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--l3", nargs="*", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    args = ap.parse_args()

    out = resolve_path("results/tuning")
    rows = []
    for l2, l3 in itertools.product(args.l2, args.l3):
        for seed in args.seeds:
            cfg = make_config(args.exp, seed, overrides=[f"loss.lambda_topo={l2}", f"loss.lambda_cons={l3}", *args.set])
            run_dir = out / f"l2_{l2}_l3_{l3}" / f"seed{seed}"
            ck = run_dir / "best.pt"
            if not ck.exists():
                train(cfg, run_dir, args.device)
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            rows.append({"lambda_topo": l2, "lambda_cons": l3, "seed": seed, "val_dice": meta["val_dice"], "val_cldice": meta["val_cldice"]})
            print(rows[-1])
    df = pd.DataFrame(rows)
    df["score"] = (df.val_dice + df.val_cldice) / 2
    g = df.groupby(["lambda_topo", "lambda_cons"], as_index=False).score.mean().sort_values(["score", "lambda_topo", "lambda_cons"], ascending=[False, True, True])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tuning_runs.csv", index=False)
    g.to_csv(out / "tuning_summary.csv", index=False)
    top = g.score.iloc[0]
    best = g[g.score >= top - 0.005].sort_values(["lambda_topo", "lambda_cons"]).iloc[0]       # tie -> smaller weights
    tuned = {"loss": {"lambda_topo": float(best.lambda_topo), "lambda_cons": float(best.lambda_cons)}}
    (ROOT / "configs" / "tuned.yaml").write_text("# written by scripts/tune_lambdas.py (selected on DRIVE validation only)\n" + yaml.safe_dump(tuned))
    print(g.to_string(index=False))
    print("selected:", tuned, "-> configs/tuned.yaml")


if __name__ == "__main__":
    main()
