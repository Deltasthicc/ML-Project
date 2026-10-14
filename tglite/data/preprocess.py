"""Deterministic image preprocessing: FOV handling, scale normalisation, green channel + CLAHE.

The same function is used at training and test time, so a (fixed) preprocessing step can never be the
source of a train/test mismatch.  Pipeline for one fundus image:

    RGB  ->  green channel  ->  fill outside-FOV with the FOV mean  ->  CLAHE (fixed tile size in pixels)
         ->  per-image standardisation inside the FOV  ->  zero outside the FOV
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------------------- FOV helpers
def estimate_fov(rgb: np.ndarray, thresh: int | None = None, erode_frac: float = 0.01) -> np.ndarray:
    """Binary FOV mask (uint8 {0,1}) for images that ship without one (STARE, CHASE_DB1, DRIVE mirror, FIVES).

    The retina is brighter than the black surround, so we threshold the (blurred) max-channel, keep the
    biggest connected component, fill holes, and erode slightly to drop the noisy rim.
    """
    gray = rgb.max(axis=2) if rgb.ndim == 3 else rgb
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
    if thresh is None:
        # robust adaptive threshold: halfway between the background level and the retina level
        bg = float(np.percentile(gray, 2))
        fg = float(np.percentile(gray, 60))
        thresh = int(np.clip(bg + 0.25 * (fg - bg), 10, 60))
    m = gray > thresh
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return np.ones(gray.shape, np.uint8)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    it = max(1, int(round(erode_frac * min(m.shape))))
    m = ndi.binary_erosion(m, iterations=it)
    return m.astype(np.uint8)


def fov_width(fov: np.ndarray) -> float:
    """Horizontal extent of the FOV in pixels (robust to a tiny stray pixel via 1 % trimming)."""
    cols = np.nonzero(fov.any(axis=0))[0]
    if len(cols) == 0:
        return float(fov.shape[1])
    return float(cols[-1] - cols[0] + 1)


# --------------------------------------------------------------------------------- scale normalisation
def scale_to_fov(img: np.ndarray, mask: np.ndarray, fov: np.ndarray, target_width: float):
    """Rescale (img, mask, fov) so the FOV is ``target_width`` px wide.

    Cameras differ hugely in resolution (HRF is ~6x larger than DRIVE); without this step a model trained on
    DRIVE sees vessels of a completely different pixel width.  This uses *no labels and no target statistics
    beyond the image's own FOV*, so it is a legitimate part of a source-only deployment pipeline.
    """
    s = target_width / fov_width(fov)
    if abs(s - 1.0) < 0.02:
        return img, mask, fov
    h, w = img.shape[:2]
    nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    img2 = cv2.resize(img, (nw, nh), interpolation=interp)
    # masks: interpolate as float then threshold, so thin vessels are not erased by nearest-neighbour sampling
    mask2 = (cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
    fov2 = (cv2.resize(fov.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
    return img2, mask2, fov2


# --------------------------------------------------------------------------------------- main transform
def green_clahe(rgb: np.ndarray, fov: np.ndarray, clip: float = 2.0, tile_px: int = 64) -> np.ndarray:
    """RGB uint8 -> standardised float32 (H, W) network input (zero outside the FOV)."""
    return clahe_standardise(rgb[..., 1], fov, clip, tile_px)


def clahe_standardise(green: np.ndarray, fov: np.ndarray, clip: float = 2.0, tile_px: int = 64) -> np.ndarray:
    """uint8 green channel -> CLAHE -> FOV-standardised float32 (zero outside the FOV)."""
    g = green.copy()
    inside = fov.astype(bool)
    if inside.sum() == 0:
        return np.zeros(g.shape, np.float32)
    g[~inside] = int(g[inside].mean())
    h, w = g.shape
    grid = (max(2, math.ceil(w / tile_px)), max(2, math.ceil(h / tile_px)))
    g = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(g)
    x = g.astype(np.float32) / 255.0
    mu, sd = x[inside].mean(), x[inside].std() + 1e-6
    x = (x - mu) / sd
    x[~inside] = 0.0
    return x.astype(np.float32)


def pad_to_multiple(x, multiple: int = 16):
    """Pad a (..., H, W) torch tensor on the bottom/right by reflection so H, W are multiples of ``multiple``."""
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    mode = "reflect" if (ph < h and pw < w) else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)
