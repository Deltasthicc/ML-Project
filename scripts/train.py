"""Train one model.   python scripts/train.py --exp E_tglitenet --seed 0 [--set train.iters=2000]

Use ``--config my.yaml`` to layer your own YAML on top of base.yaml for a custom experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tglite.config import resolve_path
from tglite.evaluate import efficiency_run, evaluate_run
from tglite.experiments import make_config
from tglite.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True, help="experiment name from tglite/experiments.py (or any name with --config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--evaluate", action="store_true", help="also evaluate + measure efficiency after training")
    args = ap.parse_args()
    cfg = make_config(args.exp, args.seed, args.config, args.set)
    run_dir = Path(args.out) if args.out else resolve_path(cfg.paths.runs) / args.exp / f"seed{args.seed}"
    train(cfg, run_dir, args.device)
    if args.evaluate:
        evaluate_run(cfg, run_dir, args.device)
        efficiency_run(cfg, run_dir)


if __name__ == "__main__":
    main()
