"""Training loop (one run = one config + one seed)."""
from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import dump_config
from .data import EvalDataset, TrainDataset, load_split
from .losses import TGLoss
from .metrics import cldice_score, segmentation_metrics
from .models import build_model, count_params
from .predict import predict_prob


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(_):
    cv2.setNumThreads(0)   # avoid CPU oversubscription: parallelism comes from the DataLoader workers


def lr_at(it: int, total: int, warmup: int, base: float) -> float:
    if it < warmup:
        return base * (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def validate(model, val_samples, cfg, device) -> dict:
    ds = EvalDataset(val_samples, cfg)
    dices, aucs, cls = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        prob, _ = predict_prob(model, item["x"][None].to(device), mc_samples=0)
        gt, fov = item["y"][0].numpy(), item["fov"][0].numpy()
        m = segmentation_metrics(prob, gt, fov, cfg.eval.threshold)
        dices.append(m["dice"])
        aucs.append(m["auc"])
        cls.append(cldice_score((prob >= cfg.eval.threshold) & (fov > 0), (gt > 0) & (fov > 0)))
    return {"val_dice": float(np.mean(dices)), "val_auc": float(np.nanmean(aucs)), "val_cldice": float(np.mean(cls))}


def train(cfg, run_dir: Path, device: str | None = None, log=print) -> Path:
    """Train and return the path of the best (by source-validation Dice) checkpoint."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg.seed))
    tr = cfg.train
    assert int(tr.val_every) % int(tr.log_every) == 0, "train.val_every must be a multiple of train.log_every"

    model = build_model(cfg).to(device)
    log(f"[{cfg.name} seed={cfg.seed}] backbone={cfg.model.backbone} params={count_params(model)/1e6:.3f}M device={device}")
    criterion = TGLoss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr.lr), weight_decay=float(tr.weight_decay))
    is_cuda = str(device).startswith("cuda")
    use_amp = bool(tr.amp) and is_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_samples = load_split("drive_train", cfg, scale_mode="fov")
    val_samples = load_split("drive_val", cfg, scale_mode="fov")
    dg = bool(cfg.dg.enabled)
    ds = TrainDataset(train_samples, cfg, dg=dg, length=int(tr.iters) * int(tr.batch_size))
    gen = torch.Generator().manual_seed(int(cfg.seed))
    nw = int(tr.num_workers)
    loader = DataLoader(ds, batch_size=int(tr.batch_size), num_workers=nw, drop_last=True, shuffle=False,
                        worker_init_fn=_worker_init, generator=gen, pin_memory=is_cuda,
                        persistent_workers=nw > 0)

    best, best_path = -1.0, run_dir / "best.pt"
    log_path = run_dir / "train_log.csv"
    fields = ["iter", "lr", "loss", "val_dice", "val_auc", "val_cldice", "time_s"] + ["t_" + k for k in ("dice", "bce", "cldice", "srl", "skel_head", "cons")]
    t0 = time.time()
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        run_terms: dict[str, float] = {}
        for it, batch in enumerate(loader):
            model.train()
            lr = lr_at(it, int(tr.iters), int(tr.warmup_iters), float(tr.lr))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y, fov, skel = (batch[k].to(device, non_blocking=True) for k in ("x", "y", "fov", "skel"))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if dg:
                    xx = torch.cat([x, batch["x_dg"].to(device, non_blocking=True)])
                    out = model(xx, with_skeleton=model.has_skeleton_head)
                    rep = lambda t: torch.cat([t, t])  # noqa: E731
                    terms = criterion.supervised(out, rep(y), rep(fov), rep(skel))
                    logits = out[0] if isinstance(out, tuple) else out
                    n = x.shape[0]
                    terms["cons"] = criterion.lc * criterion.consistency(logits[n:], logits[:n], fov)
                else:
                    out = model(x, with_skeleton=model.has_skeleton_head)
                    terms = criterion.supervised(out, y, fov, skel)
                loss = sum(terms.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tr.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.grad_clip))
            scaler.step(opt)
            scaler.update()

            for k, v in terms.items():
                run_terms[k] = run_terms.get(k, 0.0) + float(v.detach())
            run_terms["loss"] = run_terms.get("loss", 0.0) + float(loss.detach())
            step = it + 1
            if step % int(tr.log_every) == 0 or step == int(tr.iters):
                row = {"iter": step, "lr": lr, "time_s": round(time.time() - t0, 1)}
                cnt = int(tr.log_every) if step % int(tr.log_every) == 0 else step % int(tr.log_every)
                row["loss"] = run_terms.pop("loss", 0.0) / cnt
                for k in list(run_terms):
                    row["t_" + k] = run_terms.pop(k) / cnt
                if step % int(tr.val_every) == 0 or step == int(tr.iters):
                    v = validate(model, val_samples, cfg, device)
                    row.update(v)
                    if v["val_dice"] > best:
                        best = v["val_dice"]
                        torch.save({"model": model.state_dict(), "iter": step, "val_dice": best,
                                    "val_cldice": v["val_cldice"], "cfg": dict(cfg)}, best_path)
                wr.writerow({k: row.get(k, "") for k in fields})
                fh.flush()
                log(f"  it {step:5d}/{tr.iters} loss {row['loss']:.4f} lr {lr:.2e}" + (f" val_dice {row['val_dice']:.4f}" if "val_dice" in row else ""))
    if not best_path.exists():  # val never ran (tiny smoke runs)
        torch.save({"model": model.state_dict(), "iter": int(tr.iters), "val_dice": float("nan"), "cfg": dict(cfg)}, best_path)
    return best_path
