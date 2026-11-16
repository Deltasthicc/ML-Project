"""One training run (config + seed), fully resumable.

Run directory (runs/<exp>/seed<k>/):
    config.yaml      resolved config        env.json       software / hardware record
    last.pt          resumable state (model, optimiser, scaler, iteration, best score, log position) - written atomically
    best.pt          weights with the best DRIVE-val Dice (checkpoint selection uses DRIVE-val only)
    train_log.csv    loss terms every log_every iterations, DRIVE-val metrics every val_every
    tb/              TensorBoard
    train_done.json  written when training has finished
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config_hash, dump_config
from .data import TrainPatches, load_split, to_input
from .inference import predict
from .losses import Objective
from .metrics import GTTopology, overlap_metrics, topology_metrics
from .models import build_model, count_params

LOG_FIELDS = ["iter", "lr", "loss", "dice", "bce", "topo", "skel_head", "cons", "aux", "kd", "kd_skel", "val_dice", "val_cldice", "val_betti0", "it_per_s", "elapsed_s"]


def env_record() -> dict:
    import skimage
    import sklearn

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                                cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    return {"git_commit": commit, "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "scikit-image": skimage.__version__, "scikit-learn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cpu": platform.processor()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(_):
    cv2.setNumThreads(0)


def lr_at(it: int, total: int, warmup: int, base: float, schedule: str = "cosine") -> float:
    if it < warmup:
        return base * (it + 1) / warmup
    if schedule == "constant":
        return base
    t = (it - warmup) / max(1, total - warmup)
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model, val_samples, cfg, device) -> dict:
    """DRIVE-val metrics at threshold 0.5 (normalised scale; DRIVE's scale factor is ~1)."""
    d, c, b = [], [], []
    for s in val_samples:
        prob, _ = predict(model, to_input(s, cfg).to(device), amp=bool(cfg.train.amp))
        pred = (prob >= 0.5) & (s["fov"] > 0)
        d.append(overlap_metrics(pred, s["mask"], s["fov"])["dice"])
        t = topology_metrics(pred, GTTopology(s["mask"], s["fov"]))
        c.append(t["cldice"])
        b.append(t["betti0_err"])
    return {"val_dice": float(np.mean(d)), "val_cldice": float(np.mean(c)), "val_betti0": float(np.mean(b))}


def train_run(cfg, run_dir: Path, device: str | None = None, progress=None, log=print) -> dict:
    """Train (or resume) one run; returns the contents of train_done.json. ``progress(it, total, info)`` is called often."""
    run_dir = Path(run_dir)
    done_f = run_dir / "train_done.json"
    if done_f.exists():
        return json.loads(done_f.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    h = config_hash(cfg)
    cfg_f = run_dir / "config.yaml"
    meta_f = run_dir / "run_meta.json"
    if meta_f.exists() and json.loads(meta_f.read_text())["config_hash"] != h:
        raise RuntimeError(f"{run_dir} was started with a different config (hash mismatch) - refusing to mix protocols. "
                           "Delete the run directory to restart it.")
    dump_config(cfg, cfg_f)
    meta_f.write_text(json.dumps({"config_hash": h, "env": env_record()}, indent=2), encoding="utf-8")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = str(device).startswith("cuda")
    torch.backends.cudnn.benchmark = is_cuda
    tr = cfg.train
    total = int(tr.iters)
    dg = bool(cfg.dg.enabled)
    batch = int(tr.views_per_step) // (2 if dg else 1)

    set_seed(int(cfg.seed))
    model = build_model(cfg).to(device)
    obj = Objective(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr.lr), weight_decay=float(tr.weight_decay))
    use_amp = bool(tr.amp) and is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best, best_it, elapsed_before = 0, -1.0, -1, 0.0
    last_f, best_f = run_dir / "last.pt", run_dir / "best.pt"
    if last_f.exists():
        st = torch.load(last_f, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        start, best, best_it, elapsed_before = st["iter"], st["best"], st["best_iter"], st.get("elapsed", 0.0)
        torch.set_rng_state(st["torch_rng"].cpu())
        log(f"  resumed {cfg.name} seed {cfg.seed} at iteration {start}/{total}")
    log_f = run_dir / "train_log.csv"
    if start == 0 or not log_f.exists():
        with open(log_f, "w", newline="") as fh:
            csv.DictWriter(fh, LOG_FIELDS).writeheader()
    else:                                    # drop log rows written after the last checkpoint
        with open(log_f, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if int(r["iter"]) <= start]
        with open(log_f, "w", newline="") as fh:
            w = csv.DictWriter(fh, LOG_FIELDS)
            w.writeheader()
            w.writerows(rows)

    train_samples = load_split("drive_train", cfg)
    val_samples = load_split("drive_val", cfg)
    if start >= total:
        remaining = 0
    else:
        remaining = total - start
    ds = TrainPatches(train_samples, cfg, length=max(1, remaining) * batch)
    gen = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + start)
    nw = int(tr.num_workers) if is_cuda else min(int(tr.num_workers), 2)
    loader = DataLoader(ds, batch_size=batch, num_workers=nw, drop_last=True, worker_init_fn=_worker_init, generator=gen,
                        pin_memory=is_cuda, persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(run_dir / "tb"), purge_step=start)
    except Exception:  # noqa: BLE001
        tb = None

    log(f"  {cfg.name} seed {cfg.seed}: {cfg.model.backbone} w={cfg.model.width} params={count_params(model) / 1e6:.3f}M, "
        f"{total} iters x {int(tr.views_per_step)} views, device={device}")
    t0, t_last, it = time.time(), time.time(), start
    acc: dict = {}
    model.train()
    for batch_d in loader:
        if it >= total:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it, total, int(tr.warmup_iters), float(tr.lr), str(tr.get("lr_schedule", "cosine")))
        x, y, fov, skel = (batch_d[k].to(device, non_blocking=True) for k in ("x", "y", "fov", "skel"))
        dist = batch_d["dist"].to(device, non_blocking=True) if "dist" in batch_d else None
        teacher = batch_d["teacher"].to(device, non_blocking=True) if "teacher" in batch_d else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if dg:
                n = x.shape[0]
                out = model(torch.cat([x, batch_d["x_dg"].to(device, non_blocking=True)]), with_skeleton=model.has_skeleton_head)
                two = lambda t: None if t is None else torch.cat([t, t])  # noqa: E731
                terms = obj.supervised(out, two(y), two(fov), two(skel), two(dist), it, teacher=two(teacher),
                                       aux=getattr(model, "last_aux", None))
                logits = out[0] if isinstance(out, tuple) else out
                if obj.lc > 0:
                    terms["cons"] = obj.consistency(logits[n:], logits[:n], fov)
            else:
                out = model(x, with_skeleton=model.has_skeleton_head)
                terms = obj.supervised(out, y, fov, skel, dist, it, teacher=teacher, aux=getattr(model, "last_aux", None))
            loss = sum(terms.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if tr.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.grad_clip))
        scaler.step(opt)
        scaler.update()
        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        acc["loss"] = acc.get("loss", 0.0) + float(loss.detach())
        acc["_n"] = acc.get("_n", 0) + 1
        it += 1
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(f"{cfg.name} seed {cfg.seed}: non-finite loss at iteration {it}")

        if it % int(tr.log_every) == 0 or it % int(tr.val_every) == 0 or it == total:
            now = time.time()
            n = acc.pop("_n")
            row = {"iter": it, "lr": opt.param_groups[0]["lr"], **{k: v / n for k, v in acc.items()},
                   "it_per_s": n / max(now - t_last, 1e-9), "elapsed_s": elapsed_before + now - t0}
            t_last, acc = now, {}
            if it % int(tr.val_every) == 0 or it == total:
                model.eval()
                row.update(validate(model, val_samples, cfg, device))
                model.train()
                if row["val_dice"] > best:
                    best, best_it = row["val_dice"], it
                    _atomic_save({"model": model.state_dict(), "iter": it, "val": row}, best_f)
                _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "iter": it,
                              "best": best, "best_iter": best_it, "elapsed": row["elapsed_s"], "torch_rng": torch.get_rng_state()}, last_f)
            with open(log_f, "a", newline="") as fh:
                csv.DictWriter(fh, LOG_FIELDS, extrasaction="ignore").writerow(row)
            if tb is not None:
                for k, v in row.items():
                    if k != "iter" and isinstance(v, (int, float)):
                        tb.add_scalar(k, v, it)
            if progress:
                progress(it, total, {"loss": row["loss"], "val_dice": row.get("val_dice"), "best": best, "it_s": row["it_per_s"]})
    if tb is not None:
        tb.close()
    done = {"iters": total, "best_val_dice": best, "best_iter": best_it, "params": count_params(model),
            "train_time_s": elapsed_before + time.time() - t0}
    done_f.write_text(json.dumps(done, indent=2), encoding="utf-8")
    return done
