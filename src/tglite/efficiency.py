"""Deployment cost of an architecture, measured on THIS machine only (one fixed input size, batch 1).

params, FLOPs (torch FlopCounterMode: 2 x multiply-accumulates), GPU latency fp32 + fp16 autocast and peak memory,
PyTorch CPU latency with 4 threads and 1 thread, ONNX Runtime CPU latency (4 threads). Skeleton head excluded (training only).
"""
from __future__ import annotations

import copy
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .models import build_model, count_params


def _deploy_model(cfg):
    c = copy.deepcopy(cfg)
    c["model"]["skeleton_head"] = False
    m = build_model(c)
    m.eval()
    return m


def _time(fn, warmup: int, repeats: int, sync=None) -> dict:
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}


def flops(model, hw) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.zeros(1, 1, *hw)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return float(fc.get_total_flops())


@torch.no_grad()
def measure(cfg, log=print) -> dict:
    e = cfg.efficiency
    hw = tuple(int(v) for v in e.input_hw)
    model = _deploy_model(cfg)
    torch.backends.cudnn.benchmark = True          # same setting as training/inference; autotuning happens during warm-up
    rep = {"params": count_params(model), "params_m": count_params(model) / 1e6, "input_hw": list(hw), "gflops": flops(model, hw) / 1e9,
           "hardware": {"cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "torch": torch.__version__}}
    x = torch.randn(1, 1, *hw)
    for threads in (int(e.cpu_threads), 1):
        old = torch.get_num_threads()
        torch.set_num_threads(threads)
        reps = max(5, int(e.repeats) // (5 if threads == 1 else 2))
        rep[f"cpu_{threads}t"] = _time(lambda: model(x), 3, reps)
        torch.set_num_threads(old)
    try:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "m.onnx"
            torch.onnx.export(model, x, str(f), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(e.cpu_threads)
            sess = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
            xn = x.numpy()
            rep[f"onnx_cpu_{int(e.cpu_threads)}t"] = _time(lambda: sess.run(None, {"x": xn}), 3, max(5, int(e.repeats) // 2))
    except Exception as exc:  # noqa: BLE001
        log(f"  onnx latency skipped: {exc}")
    if torch.cuda.is_available():
        m = model.cuda()
        xg = x.cuda()
        sync = torch.cuda.synchronize
        rep["gpu_fp32"] = _time(lambda: m(xg), int(e.warmup), int(e.repeats), sync)

        def amp():
            with torch.autocast("cuda", dtype=torch.float16):
                m(xg)
        rep["gpu_fp16"] = _time(amp, int(e.warmup), int(e.repeats), sync)
        torch.cuda.reset_peak_memory_stats()
        m(xg)
        sync()
        rep["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        model.cpu()
    T = int(cfg.eval.mc_samples)
    rep["mc_samples"] = T if float(cfg.model.get("dropout", 0) or 0) > 0 else 1
    return rep
