"""Segmentation backbones and the model factory."""
from __future__ import annotations

from .litenet import LiteNet
from .unet import UNet


def build_model(cfg) -> "UNet | LiteNet":
    """Build the network described by ``cfg.model``."""
    m = cfg.model
    kwargs = dict(in_ch=1, dropout=float(m.get("dropout", 0.0)), skeleton_head=bool(m.get("skeleton_head", False)))
    if m.backbone == "unet":
        return UNet(base_ch=int(m.base_channels), **kwargs)
    if m.backbone == "lite":
        return LiteNet(width=int(m.width), **kwargs)
    raise ValueError(f"unknown backbone {m.backbone!r}")


def count_params(model, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


__all__ = ["UNet", "LiteNet", "build_model", "count_params"]
