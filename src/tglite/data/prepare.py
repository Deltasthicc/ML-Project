"""Archives in data/ -> standardised ``data/processed/<dataset>/{images,masks,fov[,masks2]}/<id>.png``.

* images : RGB PNG (lossless copy of the original pixels, native resolution)
* masks  : vessel label of the FIRST human observer, 0/255
* masks2 : second observer where it exists (STARE vk, CHASE 2ndHO) - only used for a human-performance reference
* fov    : field-of-view mask 0/255. Official masks for DRIVE and HRF; estimated (``preprocess.estimate_fov``) for STARE,
           CHASE_DB1 and FIVES, which ship without one.

Zip/tar archives are read in memory; only the FIVES .rar is extracted (test split only) with the system ``tar``
(bsdtar/libarchive, built into Windows 10+). SHA-256 of every source archive is recorded in ``processed/MANIFEST.json``
and checked against the reference values below where we have one.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import re
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .. import paths
from .preprocess import estimate_fov, fov_width, vessel_calibre

# SHA-256 of archives verified in the first iteration of the project (same files from the same URLs)
KNOWN_SHA256 = {
    "all.zip": "a914d02cda161b7f33f25d0397c276d50e9a6cbc705e9b364a54f0adafed57e4",                 # HRF
    "labels-ah.tar": "ebf2f1e17ca955f24579d9edd990e2dae79a5c82def69f0985d8e24f826ddd2f",           # STARE, Hoover
    "labels-vk.tar": "47474a701536b0cfdb369fdce012be36141e9f44d80387f0179446b5cb0f5576",           # STARE, Kouznetsova
    "drive_paddleseg_mirror.zip": "1097cde905f07b3d0ef187f456f4e5df0fe502fa0715ed09a5cfb892461e0ab3",
    "stare-images.tar": "5f7b509b6067cad4f1be84933145d783de9ec087b3eaaf4db0103dd0144dd433",
    "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation.rar": "be72f9af286b107bcebcc08a9dae7fc55c3fb0959409b689e14c72f9fdc4ad8e",
}
CHASE_HF_REPO = "MedOtter/CHASE_DB1"
CHASE_HF_REVISION = "c8051462ef60e5c5ac081415493a7782826f19b5"
FIVES_RAR = "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation.rar"

EXPECTED = {"drive": 40, "stare": 20, "chase_db1": 28, "hrf": 45, "fives": 200}
DATASETS = tuple(EXPECTED)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source(name: str, log) -> dict:
    p = paths.data_root() / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (see data/README.md)")
    digest = sha256(p)
    ref = KNOWN_SHA256.get(name)
    if ref and ref != digest:
        raise RuntimeError(f"{name}: SHA-256 {digest} does not match the reference {ref}")
    log(f"  {name}: {p.stat().st_size / 1e6:.1f} MB, sha256 {digest[:12]}... {'(matches reference)' if ref else '(no reference)'}")
    return {"file": name, "bytes": p.stat().st_size, "sha256": digest, "verified": bool(ref)}


def _pil_rgb(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def _pil_bin(data: bytes, thr: int = 127) -> np.ndarray:
    return (np.array(Image.open(io.BytesIO(data)).convert("L")) > thr).astype(np.uint8)


def _label01(data: bytes) -> np.ndarray:
    """Label PNG stored either as {0,255} or as {0,1} (PaddleSeg): anything > 0 is vessel."""
    a = np.array(Image.open(io.BytesIO(data)).convert("L"))
    assert len(np.unique(a)) <= 2, np.unique(a)
    return (a > 0).astype(np.uint8)


def _gz_ppm(data: bytes, rgb: bool) -> np.ndarray:
    im = Image.open(io.BytesIO(gzip.decompress(data)))
    return np.array(im.convert("RGB")) if rgb else (np.array(im.convert("L")) > 127).astype(np.uint8)


def _num(name: str) -> str:
    return re.match(r"(\d+)", Path(name).name).group(1)


class Writer:
    def __init__(self, ds: str):
        self.root = paths.processed_dir() / ds
        if self.root.exists():
            shutil.rmtree(self.root)
        for sub in ("images", "masks", "fov"):
            (self.root / sub).mkdir(parents=True)
        self.ids: list[str] = []

    def save(self, sid: str, img: np.ndarray, mask: np.ndarray, fov: np.ndarray | None, mask2: np.ndarray | None = None):
        assert img.ndim == 3 and img.shape[:2] == mask.shape, (sid, img.shape, mask.shape)
        fov_src = "official" if fov is not None else "estimated"
        if fov is None:
            fov = estimate_fov(img)
        cv2.imwrite(str(self.root / "images" / f"{sid}.png"), img[..., ::-1])
        cv2.imwrite(str(self.root / "masks" / f"{sid}.png"), (mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(self.root / "fov" / f"{sid}.png"), (fov > 0).astype(np.uint8) * 255)
        if mask2 is not None:
            (self.root / "masks2").mkdir(exist_ok=True)
            cv2.imwrite(str(self.root / "masks2" / f"{sid}.png"), (mask2 > 0).astype(np.uint8) * 255)
        self.ids.append(sid)
        self.fov_src = fov_src


# --------------------------------------------------------------------------------------------------- DRIVE
def prep_drive(cfg, log) -> dict:
    src = [_source("datasets.zip", log), _source("drive_paddleseg_mirror.zip", log)]
    outer = zipfile.ZipFile(paths.data_root() / "datasets.zip")
    inner = {Path(n).name: zipfile.ZipFile(io.BytesIO(outer.read(n))) for n in outer.namelist() if n.endswith(".zip")}
    tr, te = inner["training.zip"], inner["test.zip"]
    mirror = zipfile.ZipFile(paths.data_root() / "drive_paddleseg_mirror.zip")
    mnames = mirror.namelist()

    def mirror_file(prefix: str, sid: str, suffix: str) -> str:
        hits = [n for n in mnames if n.startswith(prefix) and not n.endswith("/") and _num(n) == sid and n.endswith(suffix)]
        assert len(hits) == 1, (prefix, sid, hits)
        return hits[0]

    w = Writer("drive")
    train_ids, test_ids, checks = [], [], {"train_label_identical": 0, "test_image_max_abs_diff": 0.0}
    for n in sorted(x for x in tr.namelist() if x.startswith("training/images/") and x.endswith(".tif")):
        sid = _num(n)
        img = _pil_rgb(tr.read(n))
        gt = _pil_bin(tr.read(f"training/1st_manual/{sid}_manual1.gif"))
        fov = _pil_bin(tr.read(f"training/mask/{sid}_training_mask.gif"))
        # integrity check of the mirror: its training labels must equal the official ones
        mgt = _label01(mirror.read(mirror_file("annotations/training/", sid, ".png")))
        checks["train_label_identical"] += int(np.array_equal(mgt, gt))
        w.save(sid, img, gt, fov)
        train_ids.append(sid)
    for n in sorted(x for x in te.namelist() if x.startswith("test/images/") and x.endswith(".tif")):
        sid = _num(n)
        img = _pil_rgb(te.read(n))
        fov = _pil_bin(te.read(f"test/mask/{sid}_test_mask.gif"))
        mimg = _pil_rgb(mirror.read(mirror_file("images/validation/", sid, ".png")))
        diff = float(np.abs(mimg.astype(np.int16) - img.astype(np.int16)).max()) if mimg.shape == img.shape else 999.0
        checks["test_image_max_abs_diff"] = max(checks["test_image_max_abs_diff"], diff)
        gt = _label01(mirror.read(mirror_file("annotations/validation/", sid, ".png")))
        w.save(sid, img, gt, fov)
        test_ids.append(sid)
    assert len(train_ids) == 20 and len(test_ids) == 20, (len(train_ids), len(test_ids))
    if checks["train_label_identical"] != 20 or checks["test_image_max_abs_diff"] > 2:
        raise RuntimeError(f"DRIVE mirror does not match the official files: {checks}")
    log(f"  mirror check: 20/20 training labels identical to official, test images max |diff| = {checks['test_image_max_abs_diff']:.0f}")

    rnd = random.Random(int(cfg.data.split_seed))
    val = sorted(rnd.sample(sorted(train_ids), int(cfg.data.n_val)))
    split = {"train": sorted(set(train_ids) - set(val)), "val": val, "test": sorted(test_ids),
             "rule": f"val = random.Random({cfg.data.split_seed}).sample(sorted(official training ids), {cfg.data.n_val}); "
                     "test = the 20 official DRIVE test images (labels from the PaddleSeg mirror)"}
    paths.PROTOCOL.mkdir(exist_ok=True)
    (paths.PROTOCOL / "drive_split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    return {"sources": src, "fov": "official", "labels": "official 1st_manual (train); PaddleSeg mirror 1st_manual (test)", "checks": checks}


# --------------------------------------------------------------------------------------------------- STARE
def prep_stare(cfg, log) -> dict:
    root = paths.data_root()
    src = [_source("labels-ah.tar", log), _source("labels-vk.tar", log)]

    def read_tar(name: str) -> dict[str, bytes]:
        with tarfile.open(root / name) as tf:
            return {Path(m.name).name[:6]: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}

    ah, vk = read_tar("labels-ah.tar"), read_tar("labels-vk.tar")
    if (root / "stare-images.tar").exists():
        src.append(_source("stare-images.tar", log))
        imgs = {k: _gz_ppm(v, True) for k, v in read_tar("stare-images.tar").items() if k in ah}
    else:
        src.append(_source("all-images.zip", log))
        z = zipfile.ZipFile(root / "all-images.zip")
        imgs = {}
        for n in z.namelist():
            sid = Path(n).name[:6]
            if sid in ah and not n.endswith("/"):
                data = z.read(n)
                imgs[sid] = _gz_ppm(data, True) if n.endswith(".gz") else _pil_rgb(data)
    w = Writer("stare")
    for sid in sorted(ah):
        w.save(sid, imgs[sid], _gz_ppm(ah[sid], False), None, _gz_ppm(vk[sid], False))
    return {"sources": src, "fov": "estimated", "labels": "Hoover (ah) = 1st observer; Kouznetsova (vk) = 2nd"}


# ----------------------------------------------------------------------------------------------- CHASE_DB1
def prep_chase(cfg, log) -> dict:
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset(CHASE_HF_REPO, revision=CHASE_HF_REVISION, cache_dir=str(paths.data_root() / "hf_cache"))
    full = concatenate_datasets([ds[k] for k in sorted(ds.keys())])
    w = Writer("chase_db1")
    for r in sorted(full, key=lambda r: r["image_id"]):
        img = np.array(r["image"].convert("RGB"))
        assert img.shape[:2] == (960, 999), (r["image_id"], img.shape)
        w.save(r["image_id"], img, np.array(r["mask"]) > 0, None, np.array(r["mask_2ndHO"]) > 0)
    log(f"  Hugging Face {CHASE_HF_REPO}@{CHASE_HF_REVISION[:10]}: {len(full)} images")
    return {"sources": [{"hf_repo": CHASE_HF_REPO, "revision": CHASE_HF_REVISION,
                         "note": "byte-identical copy of the official Kingston archive per the dataset card; Kingston server offline"}],
            "fov": "estimated", "labels": "1stHO = 1st observer; 2ndHO = 2nd"}


# ---------------------------------------------------------------------------------------------------- HRF
def prep_hrf(cfg, log) -> dict:
    src = [_source("all.zip", log)]
    z = zipfile.ZipFile(paths.data_root() / "all.zip")
    names = z.namelist()
    w = Writer("hrf")
    for n in sorted(x for x in names if x.startswith("images/") and not x.endswith("/")):
        sid = Path(n).stem
        gt = next(x for x in names if x.startswith("manual1/") and Path(x).stem == sid)
        mk = next(x for x in names if x.startswith("mask/") and Path(x).stem == f"{sid}_mask")
        w.save(sid, _pil_rgb(z.read(n)), _pil_bin(z.read(gt)), _pil_bin(z.read(mk)))
    return {"sources": src, "fov": "official", "labels": "manual1"}


# -------------------------------------------------------------------------------------------------- FIVES
def prep_fives(cfg, log) -> dict:
    rar = paths.data_root() / FIVES_RAR
    src = [_source(FIVES_RAR, log)]
    out = paths.data_root() / "extracted" / "fives"
    if not any(out.rglob("*.png")) if out.exists() else True:
        out.mkdir(parents=True, exist_ok=True)
        tar = shutil.which("tar") if not Path(r"C:\Windows\System32\tar.exe").exists() else r"C:\Windows\System32\tar.exe"
        log("  extracting the FIVES test split (bsdtar) ...")
        subprocess.run([tar, "-xf", str(rar), "-C", str(out), "--include", "*/test/*"], check=True)

    def find(name: str) -> Path:
        return next(d for d in out.rglob("*") if d.is_dir() and d.name.lower() == name and d.parent.name.lower() == "test")

    orig, gtd = find("original"), find("ground truth")
    w = Writer("fives")
    for p in sorted(orig.glob("*.png")):
        w.save(p.stem, np.array(Image.open(p).convert("RGB")), _pil_bin((gtd / p.name).read_bytes()), None)
    return {"sources": src, "fov": "estimated", "labels": "FIVES test split ground truth (native 2048 x 2048)"}


STEPS = {"drive": prep_drive, "stare": prep_stare, "chase_db1": prep_chase, "hrf": prep_hrf, "fives": prep_fives}


# ------------------------------------------------------------------------------------------------- statistics
def dataset_stats(ds: str, target_width: float) -> dict:
    root = paths.processed_dir() / ds
    rows = []
    for p in sorted((root / "images").glob("*.png")):
        img = cv2.imread(str(p))
        m = cv2.imread(str(root / "masks" / p.name), 0) > 127
        f = cv2.imread(str(root / "fov" / p.name), 0) > 127
        fw = fov_width(f)
        cal = vessel_calibre(m)
        rows.append({"id": p.stem, "h": img.shape[0], "w": img.shape[1], "vessel_frac_fov": float(m[f].mean()),
                     "fov_width": fw, "calibre_native": cal, "calibre_norm": cal * target_width / fw,
                     "green_mean": float(img[..., 1][f].mean()), "green_std": float(img[..., 1][f].std())})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "id"}
    agg["n"] = len(rows)
    agg["fov_width_median"] = float(np.median([r["fov_width"] for r in rows]))
    return {"summary": agg, "per_image": rows}


def prepare(cfg, names=None, log=print) -> dict:
    manifest_path = paths.processed_dir() / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for ds in names or DATASETS:
        log(f"== {ds}")
        info = STEPS[ds](cfg, log)
        st = dataset_stats(ds, float(cfg.data.fov_target_width))
        n = st["summary"]["n"]
        if n != EXPECTED[ds]:
            raise RuntimeError(f"{ds}: expected {EXPECTED[ds]} images, got {n}")
        vf = st["summary"]["vessel_frac_fov"]
        if not 0.04 < vf < 0.25:
            raise RuntimeError(f"{ds}: implausible vessel fraction {vf:.3f} (label polarity?)")
        s = st["summary"]
        log(f"  {ds}: {n} images {int(s['w'])}x{int(s['h'])}, vessel {100 * vf:.1f}% of FOV, FOV width {s['fov_width_median']:.0f}px, "
            f"calibre {s['calibre_native']:.1f}->{s['calibre_norm']:.1f}px, green {s['green_mean']:.0f}+-{s['green_std']:.0f}")
        manifest[ds] = {**info, "prepared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stats": s}
        (paths.processed_dir() / ds / "stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # scale-normalised caches depend on these files
    if paths.cache_dir().exists():
        shutil.rmtree(paths.cache_dir())
    return manifest


def is_prepared(names=DATASETS) -> bool:
    mf = paths.processed_dir() / "MANIFEST.json"
    if not mf.exists():
        return False
    man = json.loads(mf.read_text(encoding="utf-8"))
    return all(n in man for n in names) and (paths.PROTOCOL / "drive_split.json").exists()
