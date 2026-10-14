"""Deployment-cost accounting: parameters, FLOPs/MACs, latency (CPU & GPU) and peak memory.

All numbers are for the *inference* graph (mask head only, BatchNorm in eval mode, batch size 1) at a fixed
input size, so that models are comparable.  ``flops`` follows PyTorch's ``FlopCounterMode`` convention
(2 x multiply-accumulates); ``macs`` is half of it.  Latency depends on the machine: the hardware description is
returned with the numbers and must be quoted with them.
"""
from __future__ import annotations

import platform
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from .models import count_params


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def count_flops(model: torch.nn.Module, hw=(512, 512)) -> dict:
    model = model.eval().cpu()
    x = torch.zeros(1, 1, *hw)
    with FlopCounterMode(display=False) as fc:
        model(x)
    flops = int(fc.get_total_flops())
    return {"flops": flops, "gflops": flops / 1e9, "gmacs": flops / 2e9}


@torch.no_grad()
def measure_latency(model: torch.nn.Module, device: str, hw=(512, 512), warmup: int = 10, repeats: int = 50,
                    threads: int | None = None) -> dict:
    """Median / IQR single-image latency in ms (and peak GPU memory in MB on CUDA)."""
    dev = torch.device(device)
    prev_threads = torch.get_num_threads()
    if dev.type == "cpu" and threads:
        torch.set_num_threads(threads)
    model = model.eval().to(dev)
    x = torch.randn(1, 1, *hw, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(repeats):
        _sync(dev)
        t0 = time.perf_counter()
        model(x)
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1000.0)
    q = statistics.quantiles(times, n=4) if len(times) >= 4 else [min(times), statistics.median(times), max(times)]
    out = {"median_ms": statistics.median(times), "q1_ms": q[0], "q3_ms": q[2], "fps": 1000.0 / statistics.median(times)}
    if dev.type == "cuda":
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    else:
        out["threads"] = torch.get_num_threads()
    torch.set_num_threads(prev_threads)
    model.cpu()
    return out


def hardware_description() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
            "cpu_count": __import__("os").cpu_count()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def efficiency_report(model: torch.nn.Module, cfg, mc_samples: int = 1) -> dict:
    """Full efficiency record for one model (call on a model *without* training state)."""
    e = cfg.efficiency
    hw = tuple(e.input_hw)
    model = model.eval()
    rep = {
        "params": count_params(model),
        "params_m": count_params(model) / 1e6,
        "input_hw": list(hw),
        **count_flops(model, hw),
        "cpu": measure_latency(model, "cpu", hw, e.warmup, max(5, e.repeats // 5), e.cpu_threads),
        "cpu_1thread": measure_latency(model, "cpu", hw, 3, max(5, e.repeats // 5), 1),
        "hardware": hardware_description(),
        "mc_samples": mc_samples,
    }
    if torch.cuda.is_available():
        rep["gpu"] = measure_latency(model, "cuda", hw, e.warmup, e.repeats)
    for key in ("cpu", "cpu_1thread", "gpu"):
        if key in rep and mc_samples > 1:
            rep[key]["median_ms_total_mc"] = rep[key]["median_ms"] * mc_samples
    return rep
