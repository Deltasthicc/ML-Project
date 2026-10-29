"""List, for every trained run, the code commit and software/hardware environment recorded in its ``env.json``.

    python scripts/make_run_manifest.py                  # results/runs -> results/tables/run_manifest.csv

The 70 reported runs were produced in two environments (Kaggle T4 / CUDA 12.8 and Colab L4 / CUDA 13.0) at three code commits,
so they are statistically but not bitwise reproducible; this table is the record of which run came from where.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build(runs: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs.glob("*/seed*/env.json")):
        env = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"exp": p.parent.parent.name, "seed": int(p.parent.name.removeprefix("seed")),
                     "git_commit": str(env.get("git_commit", ""))[:7], "python": env.get("python"), "torch": env.get("torch"),
                     "cuda": env.get("cuda"), "gpu": env.get("gpu"), "platform": str(env.get("platform", "")).split("-")[0]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/tables/run_manifest.csv")
    a = ap.parse_args()
    df = build(Path(a.runs))
    if df.empty:
        raise SystemExit(f"no env.json found under {a.runs}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} runs, {df.exp.nunique()} configurations -> {a.out}")
    print(df.groupby(["gpu", "torch", "cuda"]).size().to_string())
    print(df.groupby("git_commit").size().to_string())


if __name__ == "__main__":
    main()
