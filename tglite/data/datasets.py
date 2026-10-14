"""Dataset access: processed-folder loading, scale-normalised caching, and the train / eval ``Dataset`` classes.

Processed layout (written by ``scripts/prepare_data.py``)::

    data/processed/<dataset>/images/<id>.png   RGB fundus image
                            /masks/<id>.png    vessel ground truth, 0/255 (1st human observer)
                            /fov/<id>.png      field-of-view mask, 0/255

Dataset names:  drive (official test split), drive_train, drive_val, stare, chase_db1, hrf, fives.
DRIVE train/val/test ids come from ``splits/drive_split.json`` (16/4 split of the official 20 training images + the
official 20 test images). STARE / CHASE_DB1 / HRF / FIVES are *never* used for training, tuning or model selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from ..config import resolve_path
from . import augment as A
from .preprocess import clahe_standardise, green_clahe, scale_to_fov

DATASETS = ("drive", "stare", "chase_db1", "hrf", "fives")


def split_ids(name: str, cfg) -> tuple[str, list[str]]:
    """Map an evaluation/training dataset name to (folder, list of ids)."""
    if name in ("drive", "drive_train", "drive_val"):
        with open(resolve_path(cfg.paths.split_file), "r", encoding="utf-8") as fh:
            split = json.load(fh)
        key = {"drive": "test", "drive_train": "train", "drive_val": "val"}[name]
        return "drive", list(split[key])
    folder = resolve_path(cfg.paths.processed) / name / "images"
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found - run scripts/download_data.py and scripts/prepare_data.py first")
    return name, sorted(p.stem for p in folder.glob("*.png"))


def _read(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    arr = cv2.imread(str(path), flags)
    if arr is None:
        raise FileNotFoundError(path)
    return arr


def _eval_gt(mask: np.ndarray, fov: np.ndarray, max_pixels: float):
    """Ground truth used for *metrics*: native resolution (``max_pixels`` null/0), or capped at ``max_pixels`` for a cheaper approximation.

    Down-scaling thin labels (HRF's contain 1-2 px lines) fragments them and inflates Betti-0 even at half scale, so metrics are
    computed on native ground truth and the model's probability map is resampled to it.
    """
    h, w = mask.shape
    if not max_pixels or h * w <= max_pixels:
        return mask, fov
    s = (max_pixels / (h * w)) ** 0.5
    size = (max(8, int(round(w * s))), max(8, int(round(h * s))))
    m = (cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
    f = (cv2.resize(fov.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
    return m, f


def load_sample(name: str, sid: str, cfg, scale_mode: str | None = None) -> dict:
    """Load one sample: uint8 RGB ``img``, {0,1} ``mask`` and ``fov`` (scale-normalised unless scale_mode == 'none'),
    plus ``mask_eval`` / ``fov_eval``: the ground truth at evaluation resolution (see ``_eval_gt``)."""
    scale_mode = scale_mode or cfg.eval.scale_mode
    folder = "drive" if name.startswith("drive") else name
    width = int(cfg.data.fov_target_width)
    max_px = cfg.eval.get("max_eval_pixels") or 0
    cache = None
    if scale_mode == "fov":
        cache = resolve_path(cfg.paths.cache) / f"fov{width}_e{int(max_px) or 'native'}" / folder / f"{sid}.npz"
        if cache.exists():
            z = np.load(cache)
            return {"id": sid, "img": z["img"], "mask": z["mask"], "fov": z["fov"], "mask_eval": z["mask_eval"], "fov_eval": z["fov_eval"]}
    root = resolve_path(cfg.paths.processed) / folder
    img = _read(root / "images" / f"{sid}.png")[..., ::-1].copy()
    mask = (_read(root / "masks" / f"{sid}.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    fov = (_read(root / "fov" / f"{sid}.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    mask_e, fov_e = _eval_gt(mask, fov, max_px)
    if scale_mode == "fov":
        img, mask, fov = scale_to_fov(img, mask, fov, width)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, img=img, mask=mask, fov=fov, mask_eval=mask_e, fov_eval=fov_e)
    return {"id": sid, "img": img, "mask": mask, "fov": fov, "mask_eval": mask_e, "fov_eval": fov_e}


def load_split(name: str, cfg, scale_mode: str | None = None) -> list[dict]:
    _, ids = split_ids(name, cfg)
    return [load_sample(name, i, cfg, scale_mode) for i in ids]


# ------------------------------------------------------------------------------------------- evaluation
class EvalDataset(Dataset):
    """Full images -> standardised network input, label and FOV (one image per item; batch size 1)."""

    def __init__(self, samples: list[dict], cfg):
        self.samples, self.cfg = samples, cfg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = green_clahe(s["img"], s["fov"], self.cfg.data.clahe_clip, self.cfg.data.clahe_tile_px)
        return {
            "id": s["id"],
            "x": torch.from_numpy(x)[None],
            "y": torch.from_numpy(s["mask"].astype(np.float32))[None],
            "fov": torch.from_numpy(s["fov"].astype(np.float32))[None],
            "y_eval": torch.from_numpy(s["mask_eval"].astype(np.float32))[None],
            "fov_eval": torch.from_numpy(s["fov_eval"].astype(np.float32))[None],
        }


# ---------------------------------------------------------------------------------------------- training
class TrainDataset(Dataset):
    """Infinite random-patch sampler over the DRIVE training images.

    Each item is a dict with tensors ``x`` (clean view), optionally ``x_dg`` (style/structure-perturbed view of the
    *same* patch, pixel-aligned with ``x``), ``y`` (vessels), ``fov`` and ``skel`` (exact skeleton of ``y``).
    """

    def __init__(self, samples: list[dict], cfg, dg: bool, length: int):
        self.cfg, self.dg, self.length = cfg, dg, length
        self.patch = int(cfg.data.patch_size)
        self.samples = samples
        c = cfg.data
        self.pre = [green_clahe(s["img"], s["fov"], c.clahe_clip, c.clahe_tile_px) for s in samples]
        self.fov_yx = [np.nonzero(s["fov"]) for s in samples]
        self._rng = None
        self._rng_seed = None

    def __len__(self):
        return self.length

    def _get_rng(self) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        # Re-seed every epoch-worker pair deterministically from torch's per-worker base seed.
        seed = (torch.initial_seed() + (info.id if info else 0)) % (2 ** 32)
        if self._rng is None or self._rng_seed != seed:
            self._rng, self._rng_seed = np.random.default_rng(seed), seed
        return self._rng

    def __getitem__(self, _):
        rng = self._get_rng()
        c, dgc = self.cfg.data, self.cfg.dg
        k = int(rng.integers(len(self.samples)))
        s, x1 = self.samples[k], self.pre[k]
        P = self.patch
        M = A.random_affine(P, self.fov_yx[k], rng)

        def warp(a, interp=cv2.INTER_LINEAR):
            return cv2.warpAffine(a, M, (P, P), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        x1p = warp(x1)
        yp = (warp(s["mask"].astype(np.float32)) >= 0.5).astype(np.uint8)
        fp = (warp(s["fov"].astype(np.float32)) >= 0.5).astype(np.uint8)
        x2p = None
        if self.dg:
            g2 = A.photometric(s["img"][..., 1], s["fov"], rng, dgc)
            x2 = clahe_standardise(g2, s["fov"], c.clahe_clip, c.clahe_tile_px)
            x2p = warp(x2)
            if rng.random() < dgc.p_elastic:                      # structural: same deformation for both views + label
                maps = A.elastic_maps(P, P, rng, dgc.elastic_alpha, dgc.elastic_sigma)
                x1p, x2p = A.remap(x1p, maps), A.remap(x2p, maps)
                yp = (A.remap(yp.astype(np.float32), maps) >= 0.5).astype(np.uint8)
                fp = (A.remap(fp.astype(np.float32), maps) >= 0.5).astype(np.uint8)
            if rng.random() < dgc.p_dropout:                      # structural: view 2 only
                x2p = A.local_intensity_dropout(x2p, fp, rng)
        skel = skeletonize(yp > 0).astype(np.float32)
        out = {
            "x": torch.from_numpy(np.ascontiguousarray(x1p, dtype=np.float32))[None],
            "y": torch.from_numpy(yp.astype(np.float32))[None],
            "fov": torch.from_numpy(fp.astype(np.float32))[None],
            "skel": torch.from_numpy(skel)[None],
        }
        if x2p is not None:
            out["x_dg"] = torch.from_numpy(np.ascontiguousarray(x2p, dtype=np.float32))[None]
        return out
