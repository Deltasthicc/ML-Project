"""Evaluate an existing run directory.

    python scripts/evaluate.py results/runs/E_tglitenet/seed0
    python scripts/evaluate.py results/runs/E_tglitenet/seed0 --scale none --tag native --datasets stare chase_db1 hrf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tglite.config import AttrDict
from tglite.evaluate import efficiency_run, evaluate_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--scale", choices=["fov", "none"], default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--efficiency", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = AttrDict(yaml.safe_load(open(run_dir / "config.yaml", encoding="utf-8")))
    evaluate_run(cfg, run_dir, args.device, datasets=args.datasets, scale_mode=args.scale, tag=args.tag)
    if args.efficiency:
        efficiency_run(cfg, run_dir)


if __name__ == "__main__":
    main()
