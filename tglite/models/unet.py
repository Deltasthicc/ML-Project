"""Classic U-Net (Ronneberger et al., 2015) used as the heavy baseline.

Standard configuration (base_ch=64): four 2x2 max-pool stages, double 3x3 conv + BN + ReLU blocks,
transposed-conv up-sampling -> ~31 M parameters.  ``dropout`` optionally inserts spatial dropout after the
bottleneck and every decoder block (only used for MC-dropout experiments); ``skeleton_head`` adds the
auxiliary skeleton output used by the topology loss during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    n_down = 4

    def __init__(self, in_ch: int = 1, base_ch: int = 64, dropout: float = 0.0, skeleton_head: bool = False):
        super().__init__()
        c = [base_ch * m for m in (1, 2, 4, 8, 16)]
        self.enc = nn.ModuleList([DoubleConv(in_ch, c[0])] + [DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([DoubleConv(2 * c[i], c[i]) for i in reversed(range(4))])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.mask_head = nn.Conv2d(c[0], 1, 1)
        self.skel_head = nn.Conv2d(c[0], 1, 1) if skeleton_head else None

    @property
    def has_skeleton_head(self) -> bool:
        return self.skel_head is not None

    def forward(self, x: torch.Tensor, with_skeleton: bool = False):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < self.n_down:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        x = self.drop(x)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([skips.pop(), x], dim=1))
            x = self.drop(x)
        mask = self.mask_head(x)
        if with_skeleton and self.skel_head is not None:
            return mask, self.skel_head(x)
        return mask
