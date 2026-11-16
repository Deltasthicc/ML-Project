"""Amendment 1 of protocol/PROTOCOL_v2.1.md: DRIVE-val-only simulation used to choose the novelty-2 scale criterion
(DRIVE-val images shrunk by f; which label-free criterion recovers the best scale). Run: .venv/python.exe scripts/check_scale_rule.py"""
import sys, json, numpy as np, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from tglite.config import make_config
from tglite.data import load_split
from tglite.evaluate import load_model
from tglite.eval21 import _input_at, SCALES
from tglite.inference import predict
from tglite.metrics import binarize, overlap_metrics
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from tglite import paths

def crit_native(p, fov, thr):
    b = binarize(p, thr, 0, fov)
    sk = skeletonize(b); dist = ndi.distance_transform_edt(b)
    calib = 2 * dist[sk].mean() if sk.any() else 0
    lab = label(b, connectivity=2); sizes = np.bincount(lab.ravel())[1:]
    n = len(sizes); skl = max(sk.sum(), 1)
    return {"calib": calib, "frag_per_len": n / skl * 1000, "giant": -(sizes.max() / max(b.sum(), 1)) if n else 0,
            "mean_prob_conf": -np.mean(np.abs(p[fov > 0] - 0.5))}

dev = "cuda"
for exp in ("D_compact", "C_unet_topo_dg"):
    cfg = make_config(exp, 0); m = load_model(paths.run_dir(exp, 0), cfg, dev)
    sel = json.loads((paths.run_dir(exp, 0) / "val_selection_det.json").read_text()); thr = sel["threshold"]
    def run(s):
        out = {}
        h, w = s["fov"].shape
        for sc in SCALES:
            x = _input_at(s, cfg, sc).to(dev)
            p, _ = predict([m], x)
            fv = cv2.resize(s["fov"], (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST)
            out[sc] = (p, fv, cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA) if sc != 1 else p)
        return out
    ref = np.mean([crit_native(run(s)[1.0][0], s["fov"], thr)["calib"] for s in load_split("drive_train", cfg)[:5]])
    for f in (1.0, 0.67, 0.5, 0.4):
        best, picks, gain = [], {}, {}
        for s in load_split("drive_val", cfg):
            h, w = s["fov"].shape; sz = (int(w * f), int(h * f))
            sh = {"img": cv2.resize(s["img"], sz, interpolation=cv2.INTER_AREA), "fov": cv2.resize(s["fov"], sz, interpolation=cv2.INTER_NEAREST),
                  "mask": (cv2.resize(s["mask"].astype(np.float32), sz, interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)}
            o = run(sh)
            dice = {sc: overlap_metrics(binarize(v[2], thr, 0, sh["fov"]), sh["mask"], sh["fov"])["dice"] for sc, v in o.items()}
            cr = {sc: crit_native(v[0], v[1], thr) for sc, v in o.items()}
            best.append(max(dice, key=dice.get))
            for c in ("calib", "frag_per_len", "giant", "mean_prob_conf"):
                sc = min(cr, key=lambda q: abs(cr[q]["calib"] - ref)) if c == "calib" else min(cr, key=lambda q: (cr[q][c], q))
                picks.setdefault(c, []).append(sc); gain.setdefault(c, []).append(dice[sc] - dice[1.0])
            picks.setdefault("calib_vals", []).append([round(cr[q]["calib"], 2) for q in SCALES])
        print(exp, f, "ref", round(ref, 2), "best", best, {c: (picks[c], round(float(np.mean(gain[c])), 4)) for c in gain}, "calibs", picks["calib_vals"][:2])
