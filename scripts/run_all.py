"""Run the complete study (resumable). Re-run the same command after an interruption or power cut to continue.

    python scripts/run_all.py                       # everything, in a NEW console window with progress bars
    python scripts/run_all.py --here                # same, in this terminal
    python scripts/run_all.py --groups primary      # only some groups (pilot/backbone/freeze always run first if needed)
    python scripts/run_all.py --smoke               # tiny end-to-end check -> runs_smoke/ (never touches runs/ or results/)

Watch progress from any terminal:  python scripts/monitor.py      (or open runs/_suite/STATUS.md)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SMOKE_OVERRIDES = ["train.iters=60", "train.val_every=20", "train.log_every=10", "train.num_workers=2", "train.warmup_iters=10",
                   "eval.mc_samples=3", "eval.max_images=2", "eval.metric_workers=4", "efficiency.repeats=5", "efficiency.warmup=2"]


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _keep_awake() -> None:
    """Stop Windows from sleeping while the study runs (released automatically when the process exits)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="*", help="subset of groups from configs/experiments.yaml (default: all)")
    ap.add_argument("--seeds", nargs="*", type=int, help="override the seed lists (testing only)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (applied to EVERY run)")
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline check in runs_smoke/")
    ap.add_argument("--here", action="store_true", help="run in this terminal instead of a new window")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.smoke:
        os.environ["TGLITE_RUNS"] = str(ROOT / "runs_smoke")
        os.environ["TGLITE_DECISIONS"] = str(ROOT / "runs_smoke" / "decisions.json")
        os.environ["TGLITE_RESULTS"] = str(ROOT / "runs_smoke" / "results")
        a.set = SMOKE_OVERRIDES + a.set
        if a.seeds is None:
            a.seeds = [0, 1]
        if a.groups is None:
            a.groups = ["pilot", "backbone", "primary"]

    if os.name == "nt" and not a.here and not a.worker:
        args = [sys.executable, str(Path(__file__).resolve()), "--worker", *sys.argv[1:]]
        cmd = subprocess.list2cmdline(args)
        title = "TG-LiteNet study" + (" (smoke)" if a.smoke else "")
        subprocess.Popen(["cmd.exe", "/k", f"title {title} && {cmd}"], cwd=str(ROOT), creationflags=subprocess.CREATE_NEW_CONSOLE)
        subprocess.Popen([sys.executable, str(ROOT / "scripts" / "monitor.py"), "--window", "--smoke" if a.smoke else "--real"], cwd=str(ROOT))
        print(f"Started in a new window: {title}  (+ a monitor window).\n"
              f"Re-open the monitor any time: double-click monitor.cmd, or  .venv\\python.exe scripts\\monitor.py")
        return

    from tglite import paths
    from tglite.suite import Suite

    lock = paths.runs_root() / "_suite" / "lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid != os.getpid() and _pid_alive(pid):
            sys.exit(f"another study process (pid {pid}) is already running on {paths.runs_root()}")
    lock.write_text(str(os.getpid()))
    _keep_awake()
    try:
        Suite(groups=a.groups, seeds=a.seeds, overrides=a.set, smoke=a.smoke).run()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
