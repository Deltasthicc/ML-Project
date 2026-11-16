"""Dataset access.

Names: ``drive_train`` / ``drive_val`` (15 / 5 of the official DRIVE training images, protocol/drive_split.json),
``drive`` (the 20 official DRIVE test images = in-domain test), ``stare``, ``chase_db1``, ``hrf``, ``fives``.
STARE / CHASE_DB1 / HRF / FIVES are never used for training, tuning or selection (enforced in ``evaluate.py``).
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from .. import paths
from . import augment as A
from .preprocess import clahe_standardise, network_input, resize_to, scale_factor

SOURCE_SPLITS = ("drive_train", "drive_val")
TARGETS = ("stare", "chase_db1", "hrf", "fives")


def folder_of(name: str) -> str:
    return "drive" if name.startswith("drive") else name


def split_ids(name: str) -> list[str]:
    if name.startswith("drive"):
        f = paths.PROTOCOL / "drive_split.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run scripts/prepare_data.py")
        split = json.loads(f.read_text(encoding="utf-8"))
        return list(split[{"drive": "test", "drive_train": "train", "drive_val": "val"}[name]])
    folder = paths.processed_dir() / name / "images"
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing - run scripts/prepare_data.py")
    return sorted(p.stem for p in folder.glob("*.png"))


def _read(path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    arr = cv2.imread(str(path), flags)
    if arr is None:
        raise FileNotFoundError(path)
    return arr


def load_native(name: str, sid: str) -> dict:
    root = paths.processed_dir() / folder_of(name)
    return {"img": _read(root / "images" / f"{sid}.png")[..., ::-1].copy(),
            "mask": (_read(root / "masks" / f"{sid}.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8),
            "fov": (_read(root / "fov" / f"{sid}.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)}


def load_normalised(name: str, sid: str, target_width: float) -> dict:
    """Image/label/FOV rescaled so the FOV is ``target_width`` px wide (cached), plus the scale factor and native shape."""
    cache = paths.cache_dir() / f"fov{int(target_width)}" / folder_of(name) / f"{sid}.npz"
    if cache.exists():
        z = np.load(cache)
        return {"id": sid, "img": z["img"], "mask": z["mask"], "fov": z["fov"], "scale": float(z["scale"]), "native_hw": tuple(z["native_hw"])}
    nat = load_native(name, sid)
    s = scale_factor(nat["fov"], target_width)
    img, mask, fov = resize_to(nat["img"], nat["mask"], nat["fov"], s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez_compressed(tmp, img=img, mask=mask, fov=fov, scale=s, native_hw=np.array(nat["mask"].shape))
    tmp.replace(cache)
    return {"id": sid, "img": img, "mask": mask, "fov": fov, "scale": s, "native_hw": nat["mask"].shape}


def load_split(name: str, cfg) -> list[dict]:
    return [load_normalised(name, i, cfg.data.fov_target_width) for i in split_ids(name)]


def to_input(sample: dict, cfg) -> torch.Tensor:
    return torch.from_numpy(network_input(sample["img"], sample["fov"], cfg.data.clahe_clip, cfg.data.clahe_tile_px))[None, None]


class TrainPatches(Dataset):
    """Random patches from DRIVE-train. Items: ``x`` (clean), ``y``, ``fov``, ``skel`` (+ ``x_dg`` perturbed view, + ``dist``)."""

    def __init__(self, samples: list[dict], cfg, length: int):
        self.cfg, self.length, self.samples = cfg, length, samples
        self.dg = bool(cfg.dg.enabled)
        self.cbdice = cfg.loss.topo == "cbdice"
        self.P = int(cfg.data.patch_size)
        self.pre = [network_input(s["img"], s["fov"], cfg.data.clahe_clip, cfg.data.clahe_tile_px) for s in samples]
        self.fov_yx = [np.nonzero(s["fov"]) for s in samples]
        self._rng, self._seed = None, None
        kd = cfg.get("distill") or {}
        self.teacher = None
        if kd.get("enabled"):                       # v2.1: teacher probability maps (DRIVE-train, normalised scale)
            from pathlib import Path
            td = Path(kd["teacher_dir"])
            self.teacher = [np.load(td / f"{s['id']}.npy").astype(np.float32) for s in samples]

    def __len__(self):
        return self.length

    def _rng_get(self) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        seed = (torch.initial_seed() + (info.id if info else 0)) % (2 ** 32)
        if self._rng is None or seed != self._seed:
            self._rng, self._seed = np.random.default_rng(seed), seed
        return self._rng

    def __getitem__(self, _):
        rng, c, d, P = self._rng_get(), self.cfg.data, self.cfg.dg, self.P
        k = int(rng.integers(len(self.samples)))
        s = self.samples[k]
        M = A.random_affine(P, self.fov_yx[k], rng)

        def warp(a):
            return cv2.warpAffine(a, M, (P, P), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        x1 = warp(self.pre[k])
        y = (warp(s["mask"].astype(np.float32)) >= 0.5).astype(np.uint8)
        f = (warp(s["fov"].astype(np.float32)) >= 0.5).astype(np.uint8)
        tp = warp(self.teacher[k]) if self.teacher is not None else None
        x2 = None
        if self.dg:
            x2 = warp(clahe_standardise(A.style(s["img"], s["fov"], rng, d), s["fov"], c.clahe_clip, c.clahe_tile_px))
            if rng.random() < d.p_elastic:
                maps = A.elastic_maps(P, P, rng, d.elastic_alpha, d.elastic_sigma)
                x1, x2 = A.remap(x1, maps), A.remap(x2, maps)
                y = (A.remap(y.astype(np.float32), maps) >= 0.5).astype(np.uint8)
                f = (A.remap(f.astype(np.float32), maps) >= 0.5).astype(np.uint8)
                if tp is not None:
                    tp = A.remap(tp, maps)
            if rng.random() < d.p_dropout:
                x2 = A.local_dropout(x2, f, rng)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]  # noqa: E731
        out = {"x": t(x1), "y": t(y), "fov": t(f), "skel": t(skeletonize(y > 0))}
        if self.cbdice:
            out["dist"] = t(ndi.distance_transform_edt(y > 0))
        if x2 is not None:
            out["x_dg"] = t(x2)
        if tp is not None:
            out["teacher"] = t(np.clip(tp, 0, 1))
        return out
