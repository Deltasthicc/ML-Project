"""Train (or resume) ONE run outside the suite - for debugging and quick experiments.

    python scripts/train.py --exp A_unet --seed 0
    python scripts/train.py --exp D_compact --seed 0 --set train.iters=2000 --out runs_scratch/D_test
Runs written to a custom --out are never picked up by the report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from tqdm import tqdm

    from tglite import paths
    from tglite.config import make_config
    from tglite.train import train_run

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = make_config(a.exp, a.seed, a.set)
    out = Path(a.out) if a.out else paths.run_dir(a.exp, a.seed)
    bar = tqdm(total=int(cfg.train.iters), desc=f"{a.exp} s{a.seed}")

    def prog(it, total, info):
        bar.n = it
        bar.set_postfix({k: round(v, 4) for k, v in info.items() if isinstance(v, float)})

    print(train_run(cfg, out, progress=prog, log=tqdm.write))


if __name__ == "__main__":
    main()
