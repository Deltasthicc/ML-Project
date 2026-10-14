"""Train-time augmentation.

Two families, deliberately kept separate so the ablation can switch the second one on/off cleanly:

1. *Standard geometric augmentation* (all stages, including the baseline): random crop around a FOV pixel,
   flips, rotation, isotropic scaling.
2. *Domain-generalisation (DG) augmentation* (stage C onwards; Sec. IV-C of the paper):
   - **style**:   per-channel colour gain, gamma, brightness/contrast, smooth illumination field, blur, noise,
                  JPEG compression  (operates on the green channel *before* the deterministic preprocessing, so the
                  preprocessed view really changes);
   - **structure**: elastic deformation (applied identically to both views and the label) and *local intensity
                  dropout* (random elliptical regions whose contrast is attenuated, mimicking lesions, shadows,
                  dust).  Both are much cheaper than synthesising vessel trees (DGSSA's space-colonisation).

Everything takes a ``numpy.random.Generator`` so runs are reproducible given the dataloader worker seed.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------------------------- style
def _smooth_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Random smooth illumination field in roughly [-1, 1]: a tilted plane plus a couple of Gaussian blobs.

    Computed on a coarse (<= 48 px) grid and bilinearly up-sampled: the field is low-frequency by construction.
    """
    gh, gw = max(8, h // 12), max(8, w // 12)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    yy, xx = yy / gh - 0.5, xx / gw - 0.5
    ang = float(rng.uniform(0, 2 * np.pi))
    field = np.float32(np.cos(ang)) * xx + np.float32(np.sin(ang)) * yy
    field = field / (np.abs(field).max() + 1e-6)
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = (float(v) for v in rng.uniform(-0.5, 0.5, 2))
        sg = float(rng.uniform(0.15, 0.5))
        field = field + np.float32(rng.uniform(-1, 1)) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg * sg)).astype(np.float32)
    field = field / (np.abs(field).max() + 1e-6)
    return cv2.resize(field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def photometric(green: np.ndarray, fov: np.ndarray, rng: np.random.Generator, dg) -> np.ndarray:
    """Style augmentation of the green channel (uint8 HxW) -> uint8 HxW (zero outside the FOV).

    Only the green channel is processed because it is the only channel the network ever sees; a per-channel colour
    gain therefore reduces to a gain on green.  Point-wise operations (gain, gamma, contrast, brightness) are folded
    into one 256-entry look-up table.
    """
    h, w = green.shape
    out = green
    if rng.random() < dg.p_color:
        x = np.arange(256, dtype=np.float32) / 255.0
        x = np.clip(x * np.float32(np.exp(rng.normal(0.0, 0.15))), 0, 1) ** np.float32(np.exp(rng.normal(0.0, 0.25)))
        c, b = np.float32(rng.uniform(0.7, 1.35)), np.float32(rng.uniform(-0.12, 0.12))
        lut = (np.clip((x - 0.5) * c + 0.5 + b, 0, 1) * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    if rng.random() < dg.p_illum:
        amp = np.float32(rng.uniform(0.15, 0.5))
        out = np.clip(out.astype(np.float32) * (1.0 + amp * _smooth_field(h, w, rng)), 0, 255).astype(np.uint8)
    if rng.random() < dg.p_blur:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.8)))
    if rng.random() < dg.p_noise:
        n = rng.standard_normal(out.shape, dtype=np.float32) * np.float32(255.0 * rng.uniform(0.005, 0.03))
        out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
    if rng.random() < dg.p_jpeg:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    out = out.copy()
    out[fov == 0] = 0
    return out


# ------------------------------------------------------------------------------------------- structure
def elastic_maps(h: int, w: int, rng: np.random.Generator, alpha: float, sigma: float, max_grad: float = 0.5):
    """Smooth random displacement field -> (map_x, map_y) for ``cv2.remap``.

    ``alpha`` is the peak displacement in pixels; the field is additionally scaled down so that no displacement
    gradient exceeds ``max_grad`` (< 1), which guarantees a positive Jacobian, i.e. the warp never folds the image
    (a folded warp would corrupt vessel topology in the label)."""
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    scale = alpha / (max(np.abs(dx).max(), np.abs(dy).max()) + 1e-12)
    dx, dy = dx * scale, dy * scale
    gmax = max(np.abs(np.gradient(d, axis=a)).max() for d in (dx, dy) for a in (0, 1))
    if gmax > max_grad:
        dx, dy = dx * (max_grad / gmax), dy * (max_grad / gmax)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return xx + dx, yy + dy


def remap(arr: np.ndarray, maps, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.remap(arr, maps[0], maps[1], interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def local_intensity_dropout(x: np.ndarray, fov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Attenuate contrast inside 1-4 random soft ellipses of a standardised (H, W) patch."""
    h, w = x.shape
    out = x.copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(1, 5))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        rx, ry = rng.uniform(10, 45), rng.uniform(10, 45)
        th = rng.uniform(0, np.pi)
        c, s = np.cos(th), np.sin(th)
        u = ((xx - cx) * c + (yy - cy) * s) / rx
        v = (-(xx - cx) * s + (yy - cy) * c) / ry
        soft = np.clip(1.5 - np.sqrt(u * u + v * v), 0, 1)       # 1 in the core, soft edge
        keep = float(rng.uniform(0.0, 0.6))                      # remaining contrast
        out = out * (1 - soft * (1 - keep))
    out[fov == 0] = 0
    return out


# ------------------------------------------------------------------------------------------- geometry
def random_affine(patch: int, fov_yx: tuple[np.ndarray, np.ndarray], rng: np.random.Generator,
                  scale_range=(0.8, 1.25), rotate: bool = True, flip: bool = True) -> np.ndarray:
    """Forward 2x3 affine matrix mapping source pixels -> a ``patch`` x ``patch`` crop centred on a random FOV pixel.

    ``fov_yx`` = ``np.nonzero(fov)`` (precomputed once per image)."""
    ys, xs = fov_yx
    k = int(rng.integers(len(ys)))
    cx, cy = float(xs[k]), float(ys[k])
    s = float(np.exp(rng.uniform(np.log(scale_range[0]), np.log(scale_range[1]))))
    th = float(rng.uniform(-np.pi, np.pi)) if rotate else 0.0
    fx = -1.0 if (flip and rng.random() < 0.5) else 1.0
    fy = -1.0 if (flip and rng.random() < 0.5) else 1.0
    c, sn = np.cos(th), np.sin(th)
    lin = np.array([[c, -sn], [sn, c]], np.float64) @ np.diag([s * fx, s * fy])
    off = np.array([patch / 2.0, patch / 2.0]) - lin @ np.array([cx, cy])
    return np.concatenate([lin, off[:, None]], axis=1).astype(np.float32)
