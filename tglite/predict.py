"""Full-image inference helpers (deterministic or MC dropout), with padding to a multiple of 16."""
from __future__ import annotations

import numpy as np
import torch

from .data.preprocess import pad_to_multiple
from .uncertainty import mc_predict


@torch.no_grad()
def predict_prob(model, x: torch.Tensor, mc_samples: int = 0):
    """x: (1, 1, H, W) standardised input on the model's device.

    Returns ``(prob, var)`` as numpy (H, W) arrays; ``var`` is None for deterministic inference.
    mc_samples > 0 -> mean/variance over that many dropout masks (the model must contain dropout).
    """
    xp, (h, w) = pad_to_multiple(x, 16)
    if mc_samples and mc_samples > 0:
        mean, var = mc_predict(model, xp, mc_samples)
        return mean[0, 0, :h, :w].cpu().numpy(), var[0, 0, :h, :w].cpu().numpy()
    model.eval()
    p = torch.sigmoid(model(xp).float())
    return p[0, 0, :h, :w].cpu().numpy(), None


def has_dropout(model) -> bool:
    return any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)) for m in model.modules())
