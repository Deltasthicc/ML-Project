"""INT8 deployment check of one trained model (ONNX Runtime, static QDQ quantisation, CPU).

Calibration uses DRIVE-train images only. Reports CPU latency (fp32 ONNX vs int8 ONNX, 512x512, 4 threads), file size,
and the per-dataset Dice of the int8 model at the run's DRIVE-val threshold vs the same model in fp32 (PyTorch).
"""
from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_native, load_normalised, load_split, split_ids, to_input
from .inference import to_native
from .metrics import binarize, overlap_metrics
from .models import build_model


def _deploy(run_dir: Path, cfg):
    c = copy.deepcopy(cfg)
    c["model"]["skeleton_head"] = False
    m = build_model(c)
    st = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)["model"]
    m.load_state_dict({k: v for k, v in st.items() if "skel_head" not in k}, strict=True)
    return m.eval()


def _pad(x: np.ndarray, mult: int = 16):
    h, w = x.shape[-2:]
    H, W = -(-h // mult) * mult, -(-w // mult) * mult
    return np.pad(x, ((0, 0), (0, 0), (0, H - h), (0, W - w))), (h, w)


def _lat(fn, n=30):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def int8_report(run_dir: Path, cfg, datasets, log=print) -> dict:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    model = _deploy(run_dir, cfg)
    sel = json.loads((Path(run_dir) / "val_selection_det.json").read_text(encoding="utf-8"))
    thr = float(sel["threshold"])
    with tempfile.TemporaryDirectory() as td:
        f32, f8 = Path(td) / "m32.onnx", Path(td) / "m8.onnx"
        x0 = torch.zeros(1, 1, 512, 512)
        torch.onnx.export(model, x0, str(f32), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False,
                          dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}})
        calib = [_pad(to_input(s, cfg).numpy())[0] for s in load_split("drive_train", cfg)[:8]]

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter(calib)

            def get_next(self):
                v = next(self.it, None)
                return None if v is None else {"x": v.astype(np.float32)}

        quantize_static(str(f32), str(f8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.efficiency.cpu_threads)
        s32 = ort.InferenceSession(str(f32), so, providers=["CPUExecutionProvider"])
        s8 = ort.InferenceSession(str(f8), so, providers=["CPUExecutionProvider"])
        xb = np.random.default_rng(0).standard_normal((1, 1, 512, 512)).astype(np.float32)
        rep = {"cpu_fp32_onnx_ms": _lat(lambda: s32.run(None, {"x": xb})), "cpu_int8_onnx_ms": _lat(lambda: s8.run(None, {"x": xb})),
               "size_fp32_kb": f32.stat().st_size / 1024, "size_int8_kb": f8.stat().st_size / 1024, "threshold": thr,
               "calibration": "8 DRIVE-train images", "dice": {}}
        mi = cfg.eval.get("max_images")
        for d in datasets:
            ids = split_ids(d)[: int(mi)] if mi else split_ids(d)
            d32, d8 = [], []
            for sid in ids:
                s = load_normalised(d, sid, cfg.data.fov_target_width)
                nat = load_native(d, sid)
                xp, (h, w) = _pad(to_input(s, cfg).numpy())
                for sess, acc in ((s32, d32), (s8, d8)):
                    lg = sess.run(None, {"x": xp.astype(np.float32)})[0][0, 0, :h, :w]
                    p = to_native(1 / (1 + np.exp(-lg)), s["native_hw"])
                    acc.append(overlap_metrics(binarize(p, thr, 0, nat["fov"]), nat["mask"], nat["fov"])["dice"])
            rep["dice"][d] = {"fp32": float(np.mean(d32)), "int8": float(np.mean(d8)), "n": len(ids)}
    log(f"  int8: CPU {rep['cpu_fp32_onnx_ms']:.1f} -> {rep['cpu_int8_onnx_ms']:.1f} ms, "
        f"{rep['size_fp32_kb']:.0f} -> {rep['size_int8_kb']:.0f} KB, Dice {rep['dice']}")
    return rep
