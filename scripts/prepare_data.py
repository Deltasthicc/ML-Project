"""Convert the raw downloads into the uniform layout used by the code and sanity-check them.

    python scripts/prepare_data.py              # all datasets found in data/raw
    python scripts/prepare_data.py --only drive stare

Output per dataset ``data/processed/<name>/{images,masks,fov}/<id>.png`` (+ ``stats.json``):
    images : RGB uint8 fundus image
    masks  : vessel ground truth (255 = vessel); first human observer where several exist
    fov    : field-of-view mask (255 inside). Official masks are used where the dataset ships them (HRF, official DRIVE);
             otherwise they are estimated (``tglite.data.preprocess.estimate_fov``) and should be eyeballed with
             ``scripts/make_data_overview.py``.

It also writes ``splits/drive_split.json`` (16 train / 4 validation images drawn once, seed 2026, from the 20 official
DRIVE training images; the 20 official test images form the in-domain test set).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from tglite.data.preprocess import estimate_fov

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

EXPECTED = {  # dataset -> (n_images, (height, width))
    "drive": (40, (584, 565)),
    "stare": (20, (605, 700)),
    "chase_db1": (28, (960, 999)),
    "hrf": (45, (2336, 3504)),
}


# ----------------------------------------------------------------------------------------------- utils
def pil_rgb(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def pil_bin(data: bytes) -> np.ndarray:
    """Binary label -> uint8 {0,255}. Works for 0/1, 0/255 and 1-bit images."""
    a = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return ((a > 0) * 255).astype(np.uint8)


def save(ds: str, sid: str, img: np.ndarray, mask: np.ndarray, fov: np.ndarray | None) -> None:
    if fov is None:
        fov = estimate_fov(img) * 255
    for sub, arr in (("images", img[..., ::-1]), ("masks", mask), ("fov", fov)):
        d = OUT / ds / sub
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{sid}.png"), np.ascontiguousarray(arr))


def finish(ds: str) -> None:
    """Validate counts / shapes, compute simple EDA statistics and write stats.json."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    imgs = sorted((OUT / ds / "images").glob("*.png"))
    stats = {"n_images": len(imgs), "per_image": {}}
    vf, ff, widths, vw, gm, gs = [], [], [], [], [], []
    for p in imgs:
        im = cv2.imread(str(p))
        m = cv2.imread(str(OUT / ds / "masks" / p.name), 0)
        f = cv2.imread(str(OUT / ds / "fov" / p.name), 0)
        assert im.shape[:2] == m.shape == f.shape, f"{ds}/{p.stem}: shape mismatch {im.shape} {m.shape} {f.shape}"
        inside, vessel = f > 127, m > 127
        v = float(vessel[inside].mean())
        cols = np.nonzero(inside.any(axis=0))[0]
        fw = int(cols[-1] - cols[0] + 1)
        sk = skeletonize(vessel)
        width = float(2 * ndi.distance_transform_edt(vessel)[sk].mean())   # mean vessel calibre (px, native resolution)
        green = im[..., 1][inside].astype(np.float32)
        vf.append(v); ff.append(float(inside.mean())); widths.append(fw); vw.append(width)
        gm.append(float(green.mean())); gs.append(float(green.std()))
        stats["per_image"][p.stem] = {"hw": list(im.shape[:2]), "vessel_frac": round(v, 4), "fov_width": fw, "vessel_width_px": round(width, 2)}
    med_fw = float(np.median(widths))
    stats.update(vessel_fraction_mean=float(np.mean(vf)), fov_fraction_mean=float(np.mean(ff)), fov_width_median=med_fw,
                 vessel_width_native_px=float(np.mean(vw)), vessel_width_at_527px=float(np.mean(vw)) * 527.0 / med_fw,
                 green_mean=float(np.mean(gm)), green_std=float(np.mean(gs)))
    (OUT / ds / "stats.json").write_text(json.dumps(stats, indent=1))
    exp = EXPECTED.get(ds)
    note = ""
    if exp:
        ok_n = len(imgs) == exp[0]
        shape = stats["per_image"][imgs[0].stem]["hw"]
        ok_s = tuple(shape) == exp[1]
        note = "OK" if ok_n and ok_s else f"!! expected {exp}, got n={len(imgs)} hw={shape}"
    print(f"  {ds}: {len(imgs)} images, vessel fraction {np.mean(vf):.3f}, FOV width (median) {med_fw:.0f}px, "
          f"vessel width {np.mean(vw):.1f}px native / {stats['vessel_width_at_527px']:.1f}px normalised, green {np.mean(gm):.0f}+-{np.mean(gs):.0f}  {note}")
    assert 0.04 < np.mean(vf) < 0.20, f"{ds}: implausible vessel fraction {np.mean(vf):.3f} - label polarity wrong?"


# ----------------------------------------------------------------------------------------------- DRIVE
def prep_drive() -> None:
    ids_train, ids_test = [], []
    official = RAW / "drive"
    mirror = RAW / "drive_mirror" / "drive.zip"
    if (official / "training" / "images").exists():
        print("  using OFFICIAL DRIVE files in data/raw/drive")
        for part, suffix in (("training", "training"), ("test", "test")):
            for p in sorted((official / part / "images").glob("*.tif")):
                n = p.stem.split("_")[0]
                img = np.array(Image.open(p).convert("RGB"))
                gt = np.array(Image.open(official / part / "1st_manual" / f"{n}_manual1.gif").convert("L"))
                msk = np.array(Image.open(official / part / "mask" / f"{n}_{suffix}_mask.gif").convert("L"))
                save("drive", n, img, ((gt > 0) * 255).astype(np.uint8), ((msk > 0) * 255).astype(np.uint8))
                (ids_train if part == "training" else ids_test).append(n)
    elif mirror.exists():
        print("  using PaddleSeg mirror (training/ + validation/ == official training/test); FOV masks estimated")
        z = zipfile.ZipFile(mirror)
        for part, bucket in (("training", ids_train), ("validation", ids_test)):
            for name in sorted(n for n in z.namelist() if n.startswith(f"images/{part}/") and n.endswith(".png")):
                sid = Path(name).stem
                gt_name = f"annotations/{part}/{sid}_manual1.png"
                save("drive", sid, pil_rgb(z.read(name)), pil_bin(z.read(gt_name)), None)
                bucket.append(sid)
    else:
        print("  DRIVE: nothing found, skipping")
        return
    assert len(ids_train) == 20 and len(ids_test) == 20, (len(ids_train), len(ids_test))
    rnd = random.Random(2026)
    val = sorted(rnd.sample(ids_train, 4))
    split = {"train": sorted(set(ids_train) - set(val)), "val": val, "test": sorted(ids_test),
             "note": "train/val = official DRIVE training images (16/4, seed 2026); test = official DRIVE test images"}
    (ROOT / "splits").mkdir(exist_ok=True)
    (ROOT / "splits" / "drive_split.json").write_text(json.dumps(split, indent=1))
    finish("drive")


# ----------------------------------------------------------------------------------------------- STARE
def _ppm_gz(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(gzip.decompress(data))).convert("RGB"))


def prep_stare() -> None:
    d = RAW / "stare"
    if not (d / "stare-images.tar").exists():
        print("  STARE: not downloaded, skipping")
        return
    def read_tar(path: Path) -> dict[str, bytes]:
        with tarfile.open(path) as tf:
            return {Path(m.name).name[:6]: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}

    imgs, lab = read_tar(d / "stare-images.tar"), read_tar(d / "labels-ah.tar")
    for sid in sorted(imgs):
        img = _ppm_gz(imgs[sid])
        gt = (np.array(Image.open(io.BytesIO(gzip.decompress(lab[sid]))).convert("L")) > 127).astype(np.uint8) * 255
        save("stare", sid, img, gt, None)
    finish("stare")


# --------------------------------------------------------------------------------------------- CHASE_DB1
def prep_chase() -> None:
    d = RAW / "chase_hf"
    files = [d / "train.parquet", d / "test.parquet"]
    if not all(f.exists() for f in files):
        print("  CHASE_DB1: not downloaded, skipping")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for _, r in df.iterrows():
        save("chase_db1", r["image_id"], pil_rgb(r["image"]["bytes"]), pil_bin(r["mask"]["bytes"]), None)
    finish("chase_db1")


# ------------------------------------------------------------------------------------------------- HRF
def prep_hrf() -> None:
    zp = RAW / "hrf" / "all.zip"
    if not zp.exists():
        print("  HRF: not downloaded, skipping")
        return
    z = zipfile.ZipFile(zp)
    names = z.namelist()
    for name in sorted(n for n in names if n.startswith("images/") and not n.endswith("/")):
        sid = Path(name).stem                       # e.g. 01_dr
        gt = next(n for n in names if n.startswith("manual1/") and Path(n).stem == sid)
        mk = next(n for n in names if n.startswith("mask/") and Path(n).stem == f"{sid}_mask")
        save("hrf", sid, pil_rgb(z.read(name)), pil_bin(z.read(gt)), pil_bin(z.read(mk)))
    finish("hrf")


# ----------------------------------------------------------------------------------------------- FIVES
def prep_fives() -> None:
    """FIVES test split (200 images: 50 normal / AMD / DR / glaucoma), stored at half resolution to save disk."""
    root = RAW / "fives" / "extracted"
    if not root.exists():
        print("  FIVES: not extracted (see README), skipping")
        return
    def find(parent: str, name: str):
        return next((d for d in root.rglob("*") if d.is_dir() and d.name.lower() == name and d.parent.name.lower() == parent), None)

    orig, gtd = find("test", "original"), find("test", "ground truth")
    if orig is None or gtd is None:
        print("  FIVES: could not find test/Original and test/Ground truth, skipping")
        return
    for p in sorted(orig.glob("*.png")):
        img = np.array(Image.open(p).convert("RGB"))
        gt = pil_bin(open(gtd / p.name, "rb").read())
        half = (img.shape[1] // 2, img.shape[0] // 2)
        img = cv2.resize(img, half, interpolation=cv2.INTER_AREA)
        gt = ((cv2.resize(gt.astype(np.float32) / 255, half, interpolation=cv2.INTER_AREA) >= 0.5) * 255).astype(np.uint8)
        save("fives", p.stem, img, gt, None)
    finish("fives")


STEPS = {"drive": prep_drive, "stare": prep_stare, "chase_db1": prep_chase, "hrf": prep_hrf, "fives": prep_fives}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(STEPS))
    args = ap.parse_args()
    cache = ROOT / "data" / "cache"          # scale-normalised copies depend on the files written here -> drop them
    if cache.exists():
        import shutil
        shutil.rmtree(cache)
    for name in args.only or list(STEPS):
        print(f"== {name}")
        STEPS[name]()


if __name__ == "__main__":
    main()
