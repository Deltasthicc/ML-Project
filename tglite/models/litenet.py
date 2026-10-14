"""TG-LiteNet backbone: a depthwise-separable encoder-decoder.

Design goals (preliminary report, Sec. VIII-A): fixed compact model, <= ~2 M parameters, cheap to run on a CPU.
 * every 3x3 convolution is a depthwise 3x3 followed by a pointwise 1x1 (MobileNet-style), with BN + ReLU6;
 * each block keeps a 1x1 residual shortcut so that the very small channel counts still train stably;
 * encoder: 4 down-sampling stages, widths  w, 2w, 4w, 8w, 8w  (the deepest stage is not doubled: it is the
   cheapest place to save parameters and vessels do not need a huge receptive-field channel count);
 * decoder: bilinear up-sampling + skip concatenation + one separable block per level;
 * heads: mask head (always), auxiliary skeleton head (training only), Dropout2d for MC-dropout uncertainty.

``width`` is the single capacity knob; scripts/width_sweep reports params/FLOPs/latency/accuracy for several values.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Sequential):
    """Depthwise 3x3 + pointwise 1x1 + BN + ReLU6."""

    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=True),
        )


class SepBlock(nn.Module):
    """Two separable convs with a 1x1 residual shortcut."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(SepConv(cin, cout), SepConv(cout, cout))
        self.short = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.short(x)


class LiteNet(nn.Module):
    n_down = 4

    def __init__(self, in_ch: int = 1, width: int = 24, dropout: float = 0.0, skeleton_head: bool = False):
        super().__init__()
        w = width
        c = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c[0], 3, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU6(inplace=True))
        self.enc = nn.ModuleList([SepBlock(c[0], c[0])] + [SepBlock(c[i], c[i + 1]) for i in range(4)])
        self.dec = nn.ModuleList([SepBlock(c[i + 1] + c[i], c[i]) for i in reversed(range(4))])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.mask_head = nn.Conv2d(c[0], 1, 1)
        self.skel_head = nn.Conv2d(c[0], 1, 1) if skeleton_head else None

    @property
    def has_skeleton_head(self) -> bool:
        return self.skel_head is not None

    def forward(self, x: torch.Tensor, with_skeleton: bool = False):
        x = self.stem(x)
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < self.n_down:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        x = self.drop(x)
        for dec in self.dec:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
            x = self.drop(x)
        mask = self.mask_head(x)
        if with_skeleton and self.skel_head is not None:
            return mask, self.skel_head(x)
        return mask
