"""Full-image inference: deterministic, MC dropout, or an ensemble of models; result resampled to native resolution."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from .data.preprocess import pad_to_multiple


def _enable_mc(model: nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def predict(models, x: torch.Tensor, mc_samples: int = 0, amp: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """x: (1,1,H,W) on the models' device. One model + mc_samples>0 -> MC dropout; several models -> ensemble.
    Returns (mean probability, predictive variance or None) at the input resolution."""
    models = models if isinstance(models, (list, tuple)) else [models]
    xp, (h, w) = pad_to_multiple(x, 16)
    use_amp = amp and xp.is_cuda
    probs = []
    for model in models:
        passes = mc_samples if (mc_samples and len(models) == 1) else 1
        if passes > 1:
            _enable_mc(model)
        else:
            model.eval()
        for _ in range(passes):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(xp)
            probs.append(torch.sigmoid(out.float())[0, 0, :h, :w])
        model.eval()
    st = torch.stack(probs)
    mean = st.mean(0).cpu().numpy()
    var = st.var(0, unbiased=False).cpu().numpy() if len(probs) > 1 else None
    return mean, var


def to_native(arr: np.ndarray, native_hw) -> np.ndarray:
    h, w = int(native_hw[0]), int(native_hw[1])
    if arr.shape == (h, w):
        return arr
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)
