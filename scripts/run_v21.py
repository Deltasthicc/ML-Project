"""Run the complete TG-LiteNet v2.1 study (resumable). Re-run the same command after an interruption or power cut.

    python scripts/run_v21.py            # everything, in a NEW console window with progress bars (+ a monitor window)
    python scripts/run_v21.py --here     # same, in this terminal
    python scripts/run_v21.py --smoke    # tiny end-to-end check -> runs_smoke/v21 (never touches runs/ or results/)

Protocol: protocol/PROTOCOL_v2.1.md. Results: results/v21/. Monitor: python scripts/monitor.py --v21
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_all import SMOKE_OVERRIDES, _keep_awake, _pid_alive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (every run)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--here", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.smoke:
        os.environ["TGLITE_RUNS"] = str(ROOT / "runs_smoke")
        os.environ["TGLITE_DECISIONS"] = str(ROOT / "runs_smoke" / "decisions.json")
        os.environ["TGLITE_RESULTS"] = str(ROOT / "runs_smoke" / "results")
        os.environ["TGLITE_DECISIONS_V21"] = str(ROOT / "runs_smoke" / "v21" / "decisions_v21.json")
        os.environ["TGLITE_RESULTS_V21"] = str(ROOT / "runs_smoke" / "results_v21")
        a.set = SMOKE_OVERRIDES + a.set
    if os.name == "nt" and not a.here and not a.worker:
        args = [sys.executable, str(Path(__file__).resolve()), "--worker", *sys.argv[1:]]
        title = "TG-LiteNet v2.1 study" + (" (smoke)" if a.smoke else "")
        subprocess.Popen(["cmd.exe", "/k", f"title {title} && {subprocess.list2cmdline(args)}"], cwd=str(ROOT),
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
        subprocess.Popen([sys.executable, str(ROOT / "scripts" / "monitor.py"), "--window", "--v21-smoke" if a.smoke else "--v21"], cwd=str(ROOT))
        print(f"Started in a new window: {title} (+ a monitor window). Monitor any time: .venv\\python.exe scripts\\monitor.py --v21")
        return

    from tglite.suite21 import Suite21, v21_root

    lock = v21_root() / "_suite" / "lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid != os.getpid() and _pid_alive(pid):
            sys.exit(f"another v2.1 study process (pid {pid}) is already running")
    lock.write_text(str(os.getpid()))
    _keep_awake()
    try:
        Suite21(overrides=a.set, smoke=a.smoke).run()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
