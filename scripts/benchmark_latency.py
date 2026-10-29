"""Controlled latency benchmark of every architecture family used in the paper.

The per-run ``efficiency.json`` files are measured right after training, while data-loader workers and other jobs may still
compete for the CPU, so identical architectures differ by up to 2x between runs.  This script measures on an otherwise idle
machine instead: all models are built once, then measured in several *interleaved* rounds (so that slow drifts of the machine
affect every model equally) and the median of the per-round medians is reported.

    python scripts/benchmark_latency.py --name laptop            # -> results/tables/latency_laptop.csv  (CPU, 4 threads)
    python scripts/benchmark_latency.py --name colab_l4 --threads 4   # on CUDA machines a GPU column is added

Close other programs while it runs.  Input is one 512x512 image, batch 1, eager PyTorch (framework-level latency, not an
optimised edge runtime).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
import torch

from tglite.config import ROOT
from tglite.efficiency import count_flops, hardware_description, measure_latency
from tglite.models import count_params
from tglite.models.litenet import LiteNet
from tglite.models.unet import UNet

ARCHS = [("U-Net base 8", lambda: UNet(base_ch=8)), ("U-Net base 16", lambda: UNet(base_ch=16)), ("U-Net base 64", lambda: UNet(base_ch=64)),
         ("LiteNet w8", lambda: LiteNet(width=8)), ("LiteNet w16", lambda: LiteNet(width=16)), ("LiteNet w24", lambda: LiteNet(width=24)),
         ("LiteNet w32", lambda: LiteNet(width=32)), ("LiteNet w48", lambda: LiteNet(width=48))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="machine", help="label of this machine; output is latency_<name>.csv")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()

    models = {n: f().eval() for n, f in ARCHS}
    cpu: dict[str, list[float]] = {n: [] for n in models}
    gpu: dict[str, float] = {}
    for r in range(a.rounds):
        for n, m in models.items():
            cpu[n].append(measure_latency(m, "cpu", warmup=a.warmup, repeats=a.repeats, threads=a.threads)["median_ms"])
        print(f"round {r + 1}/{a.rounds}: " + ", ".join(f"{n} {cpu[n][-1]:.0f}" for n in models), flush=True)
    if torch.cuda.is_available():
        for n, m in models.items():
            gpu[n] = measure_latency(m, "cuda", warmup=10, repeats=30)["median_ms"]
    rows = []
    for n, m in models.items():
        fl = count_flops(m)["gflops"]
        rows.append({"arch": n, "params_m": count_params(m) / 1e6, "gflops": fl, "cpu_ms": statistics.median(cpu[n]),
                     "cpu_min_ms": min(cpu[n]), "cpu_max_ms": max(cpu[n]), "gpu_ms": gpu.get(n, float("nan")), "threads": a.threads})
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"latency_{a.name}.csv", index=False)
    (out / f"latency_{a.name}.json").write_text(json.dumps(hardware_description(), indent=2))
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
