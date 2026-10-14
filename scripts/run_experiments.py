"""Run (train -> evaluate -> efficiency) for experiment groups x seeds.  Resumable: finished runs are skipped.

Examples
--------
    python scripts/run_experiments.py --groups primary                      # stages A-E, 5 seeds each
    python scripts/run_experiments.py --groups primary factorial topology width --native-eval
    python scripts/run_experiments.py --exp E_tglitenet --seeds 0 1 2       # single experiment
    python scripts/run_experiments.py --groups primary --quick              # 1-minute smoke test (tiny, not for results!)
    python scripts/run_experiments.py --groups primary --datasets drive stare chase_db1 hrf fives

Layout: results/runs/<experiment>/seed<k>/{config.yaml, train_log.csv, best.pt, per_image.csv, efficiency.json, preds/}
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import torch
import yaml

from tglite.config import _plain as plain, resolve_path
from tglite.evaluate import efficiency_run, evaluate_run
from tglite.experiments import DEFAULT_SEEDS, GROUPS, make_config
from tglite.train import train

QUICK = ["train.iters=30", "train.batch_size=2", "data.patch_size=128", "train.val_every=10", "train.log_every=10",
         "train.num_workers=0", "train.warmup_iters=5", "eval.max_images=2", "eval.mc_samples=3",
         "efficiency.repeats=5", "efficiency.warmup=1", "data.eval_datasets=[drive]"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", default=[], choices=list(GROUPS))
    ap.add_argument("--exp", nargs="*", default=[], help="explicit experiment names (see tglite/experiments.py)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: 5 seeds (primary/factorial) or 3 (others)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="output root (default results/runs)")
    ap.add_argument("--datasets", nargs="*", default=None, help="override evaluation datasets, e.g. add fives")
    ap.add_argument("--native-eval", action="store_true", help="additionally evaluate at native resolution (no FOV rescale)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="config overrides applied to every run")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (writes to results/smoke)")
    ap.add_argument("--no-train", action="store_true", help="only (re-)evaluate existing checkpoints")
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    args = ap.parse_args()

    jobs: list[tuple[str, int]] = []
    for g in args.groups:
        seeds = args.seeds if args.seeds is not None else list(range(DEFAULT_SEEDS[g]))
        jobs += [(e, s) for e in GROUPS[g] for s in seeds]
    for e in args.exp:
        seeds = args.seeds if args.seeds is not None else list(range(5))
        jobs += [(e, s) for s in seeds]
    jobs = list(dict.fromkeys(jobs))
    if not jobs:
        ap.error("nothing to run: pass --groups and/or --exp")

    overrides = list(args.set) + (QUICK if args.quick else [])
    if args.datasets:
        overrides.append(f"data.eval_datasets=[{','.join(args.datasets)}]")
    out_root = Path(args.out) if args.out else resolve_path("results/smoke" if args.quick else "results/runs")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(jobs)} runs on {device}; output -> {out_root}")

    t_all = time.time()
    mismatched, failed = [], []
    for i, (name, seed) in enumerate(jobs, 1):
        cfg = make_config(name, seed, overrides=overrides)
        run_dir = out_root / name / f"seed{seed}"
        exists = (run_dir / "per_image.csv").exists() and (run_dir / "best.pt").exists()
        print(f"\n=== [{i}/{len(jobs)}] {name} seed {seed}")
        try:
            if exists and not args.force and not args.no_train:
                diff = config_diff(run_dir / "config.yaml", cfg)
                if diff:   # never silently mix protocols: a finished run made with another config is NOT reused
                    print(f"!! existing results in {run_dir} were produced with a different config ({', '.join(diff[:6])}); "
                          f"re-run with --force (or another --out) to replace them. SKIPPED.")
                    mismatched.append((name, seed))
                    continue
                print("    (done, skipping training and evaluation)")
            else:
                t0 = time.time()
                if not args.no_train:
                    train(cfg, run_dir, device)
                evaluate_run(cfg, run_dir, device)
                print(f"    finished in {(time.time()-t0)/60:.1f} min")
            if not list((out_root / name).glob("seed*/efficiency.json")):     # once per experiment, whichever seed runs first
                efficiency_run(cfg, run_dir)
            if args.native_eval and name in ("A_unet", "D_lite_topo_dg", "E_tglitenet") and                     (not (run_dir / "per_image_native.csv").exists() or args.force):
                nat = [d for d in cfg.data.eval_datasets if d != "drive"] or ["drive"]
                evaluate_run(cfg, run_dir, device, datasets=nat, scale_mode="none", tag="native")
        except Exception:  # noqa: BLE001 - keep going so a crash does not lose the rest of an overnight run
            traceback.print_exc()
            print(f"!! {name} seed {seed} FAILED - continuing")
            failed.append((name, seed))
    print(f"\nall done in {(time.time()-t_all)/60:.1f} min")
    if mismatched:
        print(f"!! {len(mismatched)} run(s) skipped because of a config mismatch: {mismatched}")
    if failed:
        print(f"!! {len(failed)} run(s) FAILED: {failed}  (re-run the same command to retry them)")
        raise SystemExit(1)


# keys that may legitimately differ between machines / resumptions without changing the protocol
_IGNORE = {"train.num_workers", "train.log_every", "eval.metric_workers", "eval.save_preds_seed", "efficiency.repeats",
           "efficiency.warmup", "efficiency.cpu_threads", "train.amp"}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def config_diff(saved_path: Path, cfg) -> list[str]:
    """Keys (excluding machine-specific ones) whose value differs between a saved run config and ``cfg``."""
    if not saved_path.exists():
        return ["config.yaml missing"]
    old = _flat(yaml.safe_load(open(saved_path, encoding="utf-8")))
    new = _flat(yaml.safe_load(yaml.safe_dump(plain(cfg))))
    return sorted(k for k in set(old) | set(new) if k not in _IGNORE and old.get(k) != new.get(k))


if __name__ == "__main__":
    main()
