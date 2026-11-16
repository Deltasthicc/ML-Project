"""Segmentation backbones.

* ``unet``   plain U-Net (Ronneberger et al. 2015): double 3x3 conv blocks, widths (w, 2w, 4w, 8w, 16w), transposed-conv
             up-sampling. w=64 is the classic ~31 M-parameter baseline; w=8 / 16 are the "just use fewer channels" compact rivals.
* ``lite``   depthwise-separable encoder-decoder (MobileNet-style blocks with a residual 1x1), widths (w, 2w, 4w, 8w, 8w).
* ``hybrid`` same as ``lite`` but the two highest-resolution levels (encoder and decoder) use plain 3x3 convolutions:
             depthwise convs are memory-bound and slow exactly where the feature maps are largest (v1 finding), while
             the deep, low-resolution levels hold most of the parameters.

All share: optional Dropout2d at the bottleneck and after every decoder block (MC dropout), a 1x1 mask head and an
optional 1x1 skeleton head (training only).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SepConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.Conv2d(cin, cout, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class SepBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(SepConv(cin, cout), SepConv(cout, cout))
        self.short = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.short(x)


class EncoderDecoder(nn.Module):
    """Generic 4-level U-shaped network; ``block(level)`` picks the conv block per resolution level."""

    def __init__(self, chans, block_for_level, stem: bool, up_mode: str, in_ch: int = 1, dropout: float = 0.0,
                 skeleton_head: bool = False):
        super().__init__()
        c = chans
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c[0], 3, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True)) if stem else None
        first_in = c[0] if stem else in_ch
        self.enc = nn.ModuleList([block_for_level(0)(first_in, c[0])] + [block_for_level(i + 1)(c[i], c[i + 1]) for i in range(4)])
        self.up_mode = up_mode
        if up_mode == "transpose":
            self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
            self.dec = nn.ModuleList([block_for_level(i)(2 * c[i], c[i]) for i in reversed(range(4))])
        else:
            self.up = None
            self.dec = nn.ModuleList([block_for_level(i)(c[i + 1] + c[i], c[i]) for i in reversed(range(4))])
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.mask_head = nn.Conv2d(c[0], 1, 1)
        self.skel_head = nn.Conv2d(c[0], 1, 1) if skeleton_head else None

    @property
    def has_skeleton_head(self) -> bool:
        return self.skel_head is not None

    def forward(self, x: torch.Tensor, with_skeleton: bool = False):
        if self.stem is not None:
            x = self.stem(x)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < 4:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        x = self.drop(x)
        for j, dec in enumerate(self.dec):
            skip = skips.pop()
            if self.up is not None:
                x = self.up[j](x)
            else:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.drop(dec(torch.cat([skip, x], dim=1)))
        mask = self.mask_head(x)
        if with_skeleton and self.skel_head is not None:
            return mask, self.skel_head(x)
        return mask


class WNet(nn.Module):
    """Two tiny U-Nets in sequence (Galdran et al. 2022): the second sees the image and the first one's probability map
    and refines it. The first stage's logits are kept in ``last_aux`` for an auxiliary loss during training."""

    def __init__(self, chans, dropout: float = 0.0, skeleton_head: bool = False):
        super().__init__()
        self.s1 = EncoderDecoder(chans, lambda lvl: DoubleConv, stem=False, up_mode="transpose", in_ch=1, dropout=dropout)
        self.s2 = EncoderDecoder(chans, lambda lvl: DoubleConv, stem=False, up_mode="transpose", in_ch=2, dropout=dropout,
                                 skeleton_head=skeleton_head)
        self.last_aux = None

    @property
    def has_skeleton_head(self) -> bool:
        return self.s2.has_skeleton_head

    def forward(self, x: torch.Tensor, with_skeleton: bool = False):
        a = self.s1(x)
        self.last_aux = a if self.training else None
        return self.s2(torch.cat([x, torch.sigmoid(a)], dim=1), with_skeleton=with_skeleton)


def build_model(cfg) -> nn.Module:
    m = cfg.model
    w = int(m.width)
    kw = dict(dropout=float(m.get("dropout", 0.0)), skeleton_head=bool(m.get("skeleton_head", False)))
    chans = m.get("channels")
    if m.backbone == "wnet":
        return WNet(list(chans), **kw)
    if m.backbone == "unet" and chans:                      # explicit (capped) channel list, e.g. the ~70 K v2.1 model
        return EncoderDecoder(list(chans), lambda lvl: DoubleConv, stem=False, up_mode="transpose", **kw)
    if m.backbone == "unet":
        return EncoderDecoder([w, 2 * w, 4 * w, 8 * w, 16 * w], lambda lvl: DoubleConv, stem=False, up_mode="transpose", **kw)
    if m.backbone == "lite":
        return EncoderDecoder([w, 2 * w, 4 * w, 8 * w, 8 * w], lambda lvl: SepBlock, stem=True, up_mode="bilinear", **kw)
    if m.backbone == "hybrid":
        return EncoderDecoder([w, 2 * w, 4 * w, 8 * w, 8 * w], lambda lvl: DoubleConv if lvl <= 1 else SepBlock,
                              stem=False, up_mode="bilinear", **kw)
    raise ValueError(f"unknown backbone {m.backbone!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def has_dropout(model: nn.Module) -> bool:
    return any(isinstance(mm, (nn.Dropout, nn.Dropout2d)) for mm in model.modules())
