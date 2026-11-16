"""Rebuild results/ from runs/ (the suite does this automatically after every group).

    python scripts/make_report.py
    python scripts/make_report.py --smoke      # runs_smoke/ -> runs_smoke/results/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        os.environ["TGLITE_RUNS"] = str(ROOT / "runs_smoke")
        os.environ["TGLITE_DECISIONS"] = str(ROOT / "runs_smoke" / "decisions.json")
        os.environ["TGLITE_RESULTS"] = str(ROOT / "runs_smoke" / "results")
    from tglite import paths
    from tglite.report import build_report
    from tglite.suite import results_root

    build_report(paths.runs_root(), results_root())


if __name__ == "__main__":
    main()
