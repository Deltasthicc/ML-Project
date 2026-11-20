"""Live progress monitor for the study. Refreshes every few seconds; Ctrl+C (or close the window) to quit.

    monitor.cmd                          # double-click in Explorer, or type in any terminal: opens a pop-up monitor window
    python scripts/monitor.py            # live view in THIS terminal (auto-detects the active study: runs/ or runs_smoke/)
    python scripts/monitor.py --window   # open the live view in a new pop-up window
    python scripts/monitor.py --once     # print once and exit
    python scripts/monitor.py --smoke    # force runs_smoke/   (--real forces runs/)

Shows: running / stopped state, GPU load, overall progress bar (jobs done / left), current job progress bar with ETA,
whole-study ETA and finish time, protocol decisions, finished runs with their DRIVE-val Dice, exported results, last log lines.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(s) -> str:
    return "-" if s is None else str(timedelta(seconds=int(s)))


def bar(frac: float, width: int = 50) -> str:
    frac = min(max(frac or 0.0, 0.0), 1.0)
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + f"] {100 * frac:5.1f}%"


def alive(pid) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        u, mu, mt, t = [x.strip() for x in out.split(",")]
        return f"GPU {u}% | {int(mu) / 1024:.1f}/{int(mt) / 1024:.1f} GB | {t} C"
    except Exception:  # noqa: BLE001
        return "GPU n/a"


def locations(which: str):
    real = (ROOT / "runs", ROOT / "results", ROOT / "protocol" / "decisions.json", "REAL STUDY")
    smoke = (ROOT / "runs_smoke", ROOT / "runs_smoke" / "results", ROOT / "runs_smoke" / "decisions.json", "SMOKE TEST")
    v21 = (ROOT / "runs" / "v21", ROOT / "results" / "v21", ROOT / "protocol" / "decisions_v21.json", "v2.1 STUDY")
    v21s = (ROOT / "runs_smoke" / "v21", ROOT / "runs_smoke" / "results_v21", ROOT / "runs_smoke" / "v21" / "decisions_v21.json",
            "v2.1 SMOKE TEST")
    pick = {"real": real, "smoke": smoke, "v21": v21, "v21-smoke": v21s}
    if which in pick:
        return pick[which]
    # auto: the study whose status file was updated most recently
    def mtime(loc):
        f = loc[0] / "_suite" / "status.json"
        return f.stat().st_mtime if f.exists() else -1
    return max((real, smoke, v21, v21s), key=mtime)


def render(runs: Path, results: Path, decisions: Path, label: str) -> str:
    st_f = runs / "_suite" / "status.json"
    if not st_f.exists():
        return (f"No study has been started yet ({runs} has no status).\n\n"
                f"Start one with:   .venv\\python.exe scripts\\run_all.py        (or --smoke for the quick check)\n"
                f"This window keeps checking every few seconds.")
    try:
        st = json.loads(st_f.read_text())
    except json.JSONDecodeError:
        return "status file is being written - retrying"
    lock = runs / "_suite" / "lock"
    txt = lock.read_text().strip() if lock.exists() else ""
    running = alive(int(txt)) if txt.isdigit() else False
    upd = datetime.fromisoformat(st["updated"])
    stale = (datetime.now() - upd).total_seconds()
    if running:
        state = "RUNNING"
    elif st.get("phase") == "finished":
        state = "FINISHED"
    else:
        script = "run_v21.py" if "v2.1" in label else "run_all.py"
        state = f"STOPPED - run  .venv\\python.exe scripts\\{script}" + (" --smoke" if "SMOKE" in label else "") + "  to resume"
    done, planned = int(st.get("done_jobs", 0)), int(st.get("planned_jobs", 0) or 1)
    eta = st.get("eta_total_s") if running else None
    det = st.get("job_detail") or {}
    det_s = "  ".join(f"{k}={v}" for k, v in det.items())
    L = [f"TG-LiteNet {label}  -  {datetime.now():%a %d %b %H:%M:%S}",
         f"state: {state}        {gpu()}",
         "=" * 104,
         f"phase   : {st.get('phase')}",
         f"overall : {bar(done / planned)}   {done} done / {max(planned - done, 0)} left (of {planned} planned so far)",
         f"job     : {st.get('job')}",
         f"          {bar(st.get('job_progress', 0))}   job ETA {fmt(st.get('eta_job_s') if running else None)}",
         f"          {det_s}",
         f"study   : remaining ~ {fmt(eta)}" + (f"   ->  finish ~ {(datetime.now() + timedelta(seconds=eta)):%a %d %b %H:%M}" if eta else ""),
         f"updated : {upd:%H:%M:%S} ({stale:.0f}s ago)"]
    if decisions.exists() and "v2.1" in label:
        d = json.loads(decisions.read_text())
        L += ["", "v2.1 decisions (DRIVE-val only):",
              f"  rules recorded = {d.get('rules_frozen', False)}   architecture = {d.get('arch', '(pending)')}   "
              f"loss weights = {d.get('weights', '(pending)')}",
              f"  headline (size rule) = {d.get('headline', '(pending)')}   frozen = {d.get('frozen', False)} {d.get('frozen_at', '')}"]
    elif decisions.exists():
        d = json.loads(decisions.read_text())
        L += ["", "protocol decisions (DRIVE-val only):",
              f"  budget = {d.get('budget_iters', '(pending)')} iters   compact = {d.get('backbone', {}).get('compact_choice', '(pending)')} "
              f"{d.get('compact_model', '')}   frozen = {d.get('frozen', False)} {d.get('frozen_at', '')}"]
    rows = []
    for f in sorted(runs.glob("*/seed*/train_done.json"), key=lambda p: p.stat().st_mtime):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        ev = (f.parent / "eval" / "eval_done.json").exists()
        rows.append(f"  {f.parent.parent.name:22s} {f.parent.name:6s} val Dice {r['best_val_dice']:.4f} @ {r['best_iter']:>6}  "
                    f"{r['train_time_s'] / 60:6.1f} min  {'evaluated' if ev else ''}")
    L += ["", f"finished runs: {len(rows)} (last 10)"] + rows[-10:]
    summ = results / "summary.md"
    if summ.exists():
        L += ["", f"results exported: {summ}  (updated {datetime.fromtimestamp(summ.stat().st_mtime):%a %H:%M})"]
    L += ["", "recent log:"] + ["  " + x[:150] for x in st.get("recent", [])[-8:]]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--window", action="store_true", help="open the live monitor in a new console window")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--real", action="store_true")
    g.add_argument("--v21", action="store_true", help="the v2.1 study (runs/v21)")
    g.add_argument("--v21-smoke", action="store_true", help="the v2.1 smoke test (runs_smoke/v21)")
    ap.add_argument("--every", type=float, default=3.0)
    a = ap.parse_args()
    which = "smoke" if a.smoke else "real" if a.real else "v21" if a.v21 else "v21-smoke" if a.v21_smoke else "auto"
    if a.window:
        args = [sys.executable, str(Path(__file__).resolve())] + [x for x in sys.argv[1:] if x != "--window"]
        if os.name == "nt":
            cmd = subprocess.list2cmdline(args)
            subprocess.Popen(["cmd.exe", "/c", f"title TG-LiteNet monitor && mode con: cols=110 lines=45 && {cmd}"], cwd=str(ROOT),
                             creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["x-terminal-emulator", "-e", *args])
        print("monitor opened in a new window")
        return
    if a.once:
        print(render(*locations(which)))
        return
    try:
        while True:
            txt = render(*locations(which))
            os.system("cls" if os.name == "nt" else "clear")
            print(txt, flush=True)
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
