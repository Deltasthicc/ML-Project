"""Training augmentation.

1. Geometric (every stage, baseline included): crop around a random FOV pixel, flips, rotation, isotropic scaling.
2. DG augmentation (stage C onwards) of a SECOND view of the same patch:
   * style, applied to the RGB image BEFORE the deterministic preprocessing (so the network input really changes):
       - random channel mixing before the green channel is taken (camera spectral response)
       - gain / gamma / contrast / brightness   (largely normalised away by CLAHE - kept for completeness)
       - smooth multiplicative illumination field
       - additive low-frequency haze veil  (lowers local vessel contrast; NOT undone by CLAHE)
       - resolution loss: down- then up-sampling (optics / sensor resolution; NOT undone by CLAHE)
       - blur, Gaussian noise, JPEG compression
   * structure: elastic warp (both views + label, never folds) and local contrast dropout (perturbed view only).

v1 lesson: point-wise intensity changes are cancelled by CLAHE + standardisation, so most of the style budget now goes
to perturbations that survive it (haze, resolution, noise/JPEG, illumination, channel mixing).
"""
from __future__ import annotations

import cv2
import numpy as np


def _smooth_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Random low-frequency field in [-1, 1]: a tilted plane plus up to two Gaussian blobs."""
    gh, gw = max(8, h // 12), max(8, w // 12)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    yy, xx = yy / gh - 0.5, xx / gw - 0.5
    ang = float(rng.uniform(0, 2 * np.pi))
    f = np.float32(np.cos(ang)) * xx + np.float32(np.sin(ang)) * yy
    f = f / (np.abs(f).max() + 1e-6)
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = (float(v) for v in rng.uniform(-0.5, 0.5, 2))
        sg = float(rng.uniform(0.15, 0.5))
        f = f + np.float32(rng.uniform(-1, 1)) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sg * sg)).astype(np.float32)
    f = f / (np.abs(f).max() + 1e-6)
    return cv2.resize(f.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def random_conv(rgb: np.ndarray, fov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """RandConv (Xu et al., ICLR 2021) on the RGB image: a random k x k 3->3 convolution (k in {1,3,5,7}), re-standardised
    per channel inside the FOV and mixed with the original. Changes local texture / colour response, keeps geometry."""
    k = int(rng.choice([1, 3, 5, 7]))
    W = rng.normal(0.0, 1.0 / np.sqrt(3 * k * k), (3, 3, k, k)).astype(np.float32)
    src = rgb.astype(np.float32) / 255.0
    m = fov > 0
    out = np.empty_like(src)
    for co in range(3):
        acc = np.zeros(src.shape[:2], np.float32)
        for ci in range(3):
            acc += cv2.filter2D(src[..., ci], -1, W[co, ci], borderType=cv2.BORDER_REFLECT)
        v = acc[m] if m.any() else acc.ravel()
        o_ref = src[..., co][m] if m.any() else src[..., co].ravel()
        acc = (acc - v.mean()) / (v.std() + 1e-6) * (o_ref.std() + 1e-6) + o_ref.mean()
        out[..., co] = acc
    a = np.float32(rng.uniform(0.3, 1.0))
    mixed = a * out + (1 - a) * src
    return (np.clip(mixed, 0, 1) * 255).astype(np.uint8)


def fourier_amplitude(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency amplitude-spectrum jitter (FDA / amplitude-mix family without any target image): the amplitude of
    the lowest frequencies of every channel is multiplied by a smooth random log-normal field; the phase (structure,
    i.e. the vessels) is kept."""
    h, w = rgb.shape[:2]
    beta = float(rng.uniform(0.01, 0.08))
    sig = float(rng.uniform(0.1, 0.5))
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h // 2) / h) ** 2 + ((xx - w // 2) / w) ** 2)
    band = r < beta
    out = np.empty(rgb.shape, np.float32)
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(rgb[..., c].astype(np.float32)))
        amp, ph = np.abs(F), np.angle(F)
        field = np.exp(rng.normal(0.0, sig, (h, w))).astype(np.float32)
        field = cv2.GaussianBlur(field, (0, 0), 2.0)
        amp = np.where(band, amp * field, amp)
        amp[h // 2, w // 2] = np.abs(F[h // 2, w // 2])          # keep the mean brightness (DC) unchanged
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * ph))))
    return np.clip(out, 0, 255).astype(np.uint8)


def style(rgb: np.ndarray, fov: np.ndarray, rng: np.random.Generator, dg) -> np.ndarray:
    """RGB uint8 -> perturbed single-channel uint8 'green' image (zero outside the FOV)."""
    h, w = rgb.shape[:2]
    p_rc, p_ft = float(dg.get("p_randconv", 0.0) or 0.0), float(dg.get("p_fourier", 0.0) or 0.0)
    if p_ft > 0 and rng.random() < p_ft:                   # v2.1 only (0 in v2 configs: no extra random draws there)
        rgb = fourier_amplitude(rgb, rng)
    if p_rc > 0 and rng.random() < p_rc:
        rgb = random_conv(rgb, fov, rng)
    if rng.random() < dg.p_mix:
        wr, wb = rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.3)
        wg = 1.0 - 0.5 * (wr + wb)
        g = np.clip(wr * rgb[..., 0] + wg * rgb[..., 1] + wb * rgb[..., 2], 0, 255).astype(np.uint8)
    else:
        g = rgb[..., 1].copy()
    if rng.random() < dg.p_color:
        x = np.arange(256, dtype=np.float32) / 255.0
        x = np.clip(x * np.float32(np.exp(rng.normal(0.0, 0.15))), 0, 1) ** np.float32(np.exp(rng.normal(0.0, 0.25)))
        c, b = np.float32(rng.uniform(0.7, 1.35)), np.float32(rng.uniform(-0.12, 0.12))
        g = cv2.LUT(g, (np.clip((x - 0.5) * c + 0.5 + b, 0, 1) * 255).astype(np.uint8))
    out = g.astype(np.float32)
    if rng.random() < dg.p_illum:
        out *= 1.0 + np.float32(rng.uniform(0.15, 0.5)) * _smooth_field(h, w, rng)
    if rng.random() < dg.p_haze:                       # veil: out = (1 - a) * out + a * L(x), L low-frequency and bright
        a = (np.float32(rng.uniform(0.15, 0.5)) * (0.5 + 0.5 * _smooth_field(h, w, rng))).clip(0, 0.6)
        level = np.float32(rng.uniform(0.6, 1.0)) * out[fov > 0].max() if (fov > 0).any() else 255.0
        out = (1 - a) * out + a * level
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < dg.p_lowres:
        s = float(rng.uniform(0.35, 0.8))
        small = cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC)
    if rng.random() < dg.p_blur:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.5, 1.6)))
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


def elastic_maps(h: int, w: int, rng: np.random.Generator, alpha: float, sigma: float, max_grad: float = 0.5):
    """Smooth displacement field (peak ``alpha`` px) for ``cv2.remap``; gradients capped below 1 so the warp never folds."""
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


def local_dropout(x: np.ndarray, fov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Attenuate contrast inside 1-4 soft random ellipses of a standardised patch."""
    h, w = x.shape
    out = x.copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(int(rng.integers(1, 5))):
        cx, cy, rx, ry, th = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(10, 45), rng.uniform(10, 45), rng.uniform(0, np.pi)
        c, s = np.cos(th), np.sin(th)
        u = ((xx - cx) * c + (yy - cy) * s) / rx
        v = (-(xx - cx) * s + (yy - cy) * c) / ry
        soft = np.clip(1.5 - np.sqrt(u * u + v * v), 0, 1)
        out = out * (1 - soft * (1 - float(rng.uniform(0.0, 0.6))))
    out[fov == 0] = 0
    return out


def random_affine(patch: int, fov_yx, rng: np.random.Generator, scale_range=(0.75, 1.33)) -> np.ndarray:
    """2x3 forward affine: source -> ``patch`` x ``patch`` crop centred on a random FOV pixel (random scale, rotation, flips)."""
    ys, xs = fov_yx
    k = int(rng.integers(len(ys)))
    cx, cy = float(xs[k]), float(ys[k])
    s = float(np.exp(rng.uniform(np.log(scale_range[0]), np.log(scale_range[1]))))
    th = float(rng.uniform(-np.pi, np.pi))
    fx = -1.0 if rng.random() < 0.5 else 1.0
    fy = -1.0 if rng.random() < 0.5 else 1.0
    c, sn = np.cos(th), np.sin(th)
    lin = np.array([[c, -sn], [sn, c]], np.float64) @ np.diag([s * fx, s * fy])
    off = np.array([patch / 2.0, patch / 2.0]) - lin @ np.array([cx, cy])
    return np.concatenate([lin, off[:, None]], axis=1).astype(np.float32)
