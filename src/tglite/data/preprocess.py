"""Deterministic preprocessing, identical for training and testing.

    RGB -> (FOV-width scale normalisation) -> green channel -> fill outside-FOV with the FOV mean -> CLAHE
        -> standardise inside the FOV -> zero outside the FOV

Scale normalisation uses only the image's own FOV (no labels, no target statistics), so it is a legitimate part of a
source-only deployment pipeline.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def estimate_fov(rgb: np.ndarray, erode_frac: float = 0.01) -> np.ndarray:
    """Binary FOV (uint8 {0,1}) for datasets without one: threshold the blurred max-channel halfway-ish between background
    and retina, keep the largest component, fill holes, erode slightly to drop the noisy rim."""
    gray = rgb.max(axis=2) if rgb.ndim == 3 else rgb
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
    bg, fg = float(np.percentile(gray, 2)), float(np.percentile(gray, 60))
    m = gray > int(np.clip(bg + 0.25 * (fg - bg), 10, 60))
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return np.ones(gray.shape, np.uint8)
    m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_erosion(m, iterations=max(1, int(round(erode_frac * min(m.shape)))))
    return m.astype(np.uint8)


def fov_width(fov: np.ndarray) -> float:
    cols = np.nonzero(np.asarray(fov).any(axis=0))[0]
    return float(cols[-1] - cols[0] + 1) if len(cols) else float(fov.shape[1])


def vessel_calibre(mask: np.ndarray) -> float:
    """Mean vessel width in px: 2 x distance transform sampled on the label skeleton."""
    m = mask > 0
    sk = skeletonize(m)
    if not sk.any():
        return 0.0
    return float(2 * ndi.distance_transform_edt(m)[sk].mean())


def scale_factor(fov: np.ndarray, target_width: float) -> float:
    return float(target_width) / fov_width(fov)


def resize_to(img: np.ndarray, mask: np.ndarray | None, fov: np.ndarray, s: float):
    """Rescale image (+ optional label) and FOV by factor ``s``. Labels are resampled as float and thresholded so thin
    vessels survive down-scaling better than with nearest neighbour."""
    if abs(s - 1.0) < 0.02:
        return img, mask, fov
    h, w = img.shape[:2]
    nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    lin = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    mask2 = None if mask is None else (cv2.resize(mask.astype(np.float32), (nw, nh), interpolation=lin) >= 0.5).astype(np.uint8)
    fov2 = (cv2.resize(fov.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
    return img2, mask2, fov2


def clahe_standardise(gray: np.ndarray, fov: np.ndarray, clip: float = 2.0, tile_px: int = 64) -> np.ndarray:
    """uint8 single channel -> CLAHE -> FOV-standardised float32 (zero outside the FOV)."""
    g = gray.copy()
    inside = fov.astype(bool)
    if not inside.any():
        return np.zeros(g.shape, np.float32)
    g[~inside] = int(g[inside].mean())
    h, w = g.shape
    grid = (max(2, math.ceil(w / tile_px)), max(2, math.ceil(h / tile_px)))
    g = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(g)
    x = g.astype(np.float32) / 255.0
    x = (x - x[inside].mean()) / (x[inside].std() + 1e-6)
    x[~inside] = 0.0
    return x.astype(np.float32)


def network_input(rgb: np.ndarray, fov: np.ndarray, clip: float = 2.0, tile_px: int = 64) -> np.ndarray:
    return clahe_standardise(rgb[..., 1], fov, clip, tile_px)


def pad_to_multiple(x, multiple: int = 16):
    """Pad a (..., H, W) tensor bottom/right so H and W are multiples of ``multiple``; returns (padded, (H, W))."""
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    mode = "reflect" if (ph < h and pw < w) else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=mode), (h, w)
