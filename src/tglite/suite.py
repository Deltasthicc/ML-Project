"""The complete study as one resumable job list.

Phases (each job is skipped when its output already exists, so re-running the same command resumes):
  1 prepare    archives -> data/processed (once)
  2 pilot      2 long runs on DRIVE-train/val -> decide the training budget            (DRIVE only)
  3 backbone   4 compact candidates x 3 seeds + CPU latency -> decide the compact model  (DRIVE only)
  4 freeze     record all decisions + hashes of the protocol files; targets unlocked from here on
  5 groups     primary, factorial, topology, warmup, controls, width: train -> evaluate -> cost, per run;
               backbone candidates are evaluated as controls; ensembles after primary
  6 report     tables, figures and summary exported to results/ after EVERY group (early results appear after `primary`)

Progress: tqdm bars (overall + current job) and runs/_suite/{status.json, STATUS.md, suite.log}; see scripts/monitor.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import paths
from .config import config_hash, experiments_spec, group_runs, is_frozen, load_decisions, make_config, save_decisions


# ------------------------------------------------------------------------------------------- progress / status
class Tracker:
    """tqdm bars + status.json / STATUS.md / suite.log, with an ETA model learnt from finished jobs."""

    def __init__(self, root: Path, total_jobs_hint: int = 0, use_tqdm: bool = True):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_f = self.root / "suite.log"
        self.status_f = self.root / "status.json"
        self.hist_f = self.root / "history.json"
        self.history = json.loads(self.hist_f.read_text()) if self.hist_f.exists() else {"jobs": {}, "speed": {}}
        self.state = {"started": datetime.now().isoformat(timespec="seconds"), "phase": "", "job": "", "job_progress": 0.0,
                      "job_detail": {}, "done_jobs": 0, "planned_jobs": total_jobs_hint, "recent": [], "eta_job_s": None,
                      "eta_total_s": None, "pid": os.getpid()}
        self.job_t0 = time.time()
        self._last_write = 0.0
        self.counted: set[str] = set()
        self.bar_all = self.bar_job = None
        if use_tqdm:
            from tqdm import tqdm

            self.bar_all = tqdm(total=max(total_jobs_hint, 1), position=0, desc="study", unit="job", dynamic_ncols=True)
            self.bar_job = tqdm(total=100, position=1, desc="job", dynamic_ncols=True, bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f} {postfix}")

    def log(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        with open(self.log_f, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.bar_all is not None:
            self.bar_all.write(line)
        else:
            print(line, flush=True)
        self.state["recent"] = (self.state["recent"] + [line])[-15:]
        self.flush(force=True)

    def set_plan(self, n: int) -> None:
        self.state["planned_jobs"] = n
        if self.bar_all is not None:
            self.bar_all.total = n
            self.bar_all.refresh()

    def finish(self) -> None:
        """Mark the study complete: overall and job bars at 100 %, no remaining time."""
        done = self.state["done_jobs"]
        self.set_plan(done)
        self.state.update(phase="finished", job="study finished", job_kind="done", job_progress=1.0, job_detail={},
                          eta_job_s=0, eta_total_s=0)
        if self.bar_all is not None:
            self.bar_all.n = done
            self.bar_all.refresh()
        if self.bar_job is not None:
            self.bar_job.reset(total=1)
            self.bar_job.set_description("study finished")
            self.bar_job.update(1)
        self.flush(force=True)

    def phase(self, name: str) -> None:
        self.state["phase"] = name
        self.log(f"===== {name}")

    def start_job(self, name: str, total: int, kind: str, eta_s: float | None = None) -> None:
        self.state.update(job=name, job_kind=kind, job_progress=0.0, job_detail={}, eta_job_s=eta_s)
        self.job_t0 = time.time()
        if self.bar_job is not None:
            self.bar_job.reset(total=max(total, 1))
            self.bar_job.set_description(name[:48])
        self.flush(force=True)

    def update(self, n: int, total: int, info: dict | None = None) -> None:
        info = info or {}
        el = time.time() - self.job_t0
        frac = n / max(total, 1)
        self.state["job_progress"] = frac
        self.state["job_detail"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items() if v is not None}
        if frac > 0.02 and info.get("it_s"):
            self.state["eta_job_s"] = (total - n) / max(info["it_s"], 1e-6)
        elif frac > 0.02:
            self.state["eta_job_s"] = el / frac * (1 - frac)
        if self.bar_job is not None:
            self.bar_job.total = total
            self.bar_job.n = n
            self.bar_job.set_postfix({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in self.state["job_detail"].items()}, refresh=False)
            self.bar_job.refresh()
        self.flush()

    def end_job(self, name: str, kind: str, key: str | None = None, units: float | None = None) -> None:
        dur = time.time() - self.job_t0
        self.history["jobs"][name] = {"kind": kind, "seconds": dur, "finished": datetime.now().isoformat(timespec="seconds")}
        if key and units:
            self.history["speed"].setdefault(key, []).append(dur / units)
        self.hist_f.write_text(json.dumps(self.history, indent=1))
        self.skip_job(name)
        self.flush(force=True)

    def skip_job(self, name: str) -> None:
        """Count a job as done (once per name, however often it is reached)."""
        if name in self.counted:
            return
        self.counted.add(name)
        self.state["done_jobs"] += 1
        if self.bar_all is not None:
            self.bar_all.update(1)

    def speed(self, key: str, default: float) -> float:
        v = self.history["speed"].get(key)
        return float(np.median(v)) if v else default

    def flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < 3:
            return
        self._last_write = now
        self.state["updated"] = datetime.now().isoformat(timespec="seconds")
        tmp = self.status_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        os.replace(tmp, self.status_f)
        write_status_md(self.root, self.state)


def _fmt_s(s) -> str:
    return "-" if s is None else str(timedelta(seconds=int(s)))


def write_status_md(root: Path, st: dict) -> None:
    eta_t = st.get("eta_total_s")
    finish = (datetime.now() + timedelta(seconds=eta_t)).strftime("%a %d %b %H:%M") if eta_t else "-"
    lines = [f"# TG-LiteNet study - status ({st.get('updated', '')})", "",
             f"* phase: **{st.get('phase')}**",
             f"* jobs: {st.get('done_jobs')} / {st.get('planned_jobs')} (planned so far; later phases are added when reached)",
             f"* current: `{st.get('job')}` - {100 * st.get('job_progress', 0):.0f}%  ETA job {_fmt_s(st.get('eta_job_s'))}",
             f"* detail: {st.get('job_detail')}",
             f"* estimated remaining (whole study): {_fmt_s(eta_t)} -> finish ~ {finish}", "", "## recent log", "```"]
    lines += st.get("recent", []) + ["```"]
    (root / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- the suite
class Suite:
    def __init__(self, groups=None, seeds=None, overrides=(), device=None, smoke=False, use_tqdm=True, report_every_group=True):
        self.groups = groups
        self.seeds = seeds
        self.overrides = list(overrides)
        self.smoke = smoke
        self.device = device
        self.report_every_group = report_every_group
        self.runs = paths.runs_root()
        self.tr = Tracker(self.runs / "_suite", use_tqdm=use_tqdm)
        self.log = self.tr.log

    # --------------------------------------------------------------- helpers
    def cfg(self, exp: str, seed: int):
        return make_config(exp, seed, self.overrides)

    def _train_key(self, cfg) -> str:
        return f"train:{cfg.model.backbone}:{cfg.model.width}:{'dg' if cfg.dg.enabled else 'nodg'}:{cfg.loss.topo}"

    def _default_speed(self, cfg) -> float:
        """seconds per iteration before anything has been measured (rough A4000 guesses)"""
        # measured on the RTX A4000 / i9-12900K with 12 loader workers (runs_scratch/timing): U-Net w64 5.1 it/s (4.5 with
        # topology + DG), compact w16 models 11-14.5 it/s; + ~60 s start-up per run
        heavy = cfg.model.backbone == "unet" and int(cfg.model.width) >= 32
        if heavy:
            return 0.22 if cfg.dg.enabled else 0.20
        return 0.085 * max(1.0, (int(cfg.model.width) / 16) ** 1.5)

    def est_train_s(self, cfg) -> float:
        return 60 + int(cfg.train.iters) * self.tr.speed(self._train_key(cfg), self._default_speed(cfg))

    def est_eval_s(self, cfg) -> float:
        mc = float(cfg.model.get("dropout", 0) or 0) > 0
        return self.tr.speed("eval", 240.0) * (3 if mc else 1)

    def train(self, exp: str, seed: int) -> dict:
        from .train import train_run

        cfg = self.cfg(exp, seed)
        rd = paths.run_dir(exp, seed)
        name = f"train {exp} s{seed}"
        if (rd / "train_done.json").exists():
            self.tr.skip_job(name)
            return json.loads((rd / "train_done.json").read_text())
        self.tr.start_job(name, int(cfg.train.iters), "train", self.est_train_s(cfg))
        start_it = 0
        if (rd / "last.pt").exists():
            import torch
            start_it = int(torch.load(rd / "last.pt", map_location="cpu", weights_only=False)["iter"])
        done = train_run(cfg, rd, self.device, progress=self.tr.update, log=self.log)
        self.tr.end_job(name, "train", self._train_key(cfg), max(1, int(cfg.train.iters) - start_it))
        self.log(f"  done {exp} s{seed}: best DRIVE-val Dice {done['best_val_dice']:.4f} @ {done['best_iter']}, {done['train_time_s'] / 60:.1f} min")
        return done

    def datasets(self):
        base = self.cfg_base()
        return ["drive", *base.data.targets, *base.data.extra_targets]

    def cfg_base(self):
        from .config import AttrDict, deep_update, parse_overrides, read_yaml
        c = read_yaml(paths.CONFIGS / "base.yaml")
        deep_update(c, parse_overrides(self.overrides))
        return AttrDict.wrap(c)

    def evaluate(self, exp: str, seed: int) -> None:
        from .evaluate import evaluate_run

        cfg = self.cfg(exp, seed)
        rd = paths.run_dir(exp, seed)
        marker = rd / "eval" / "eval_done.json"
        name = f"eval {exp} s{seed}"
        if marker.exists():
            self.tr.skip_job(name)
            return
        self.tr.start_job(name, 1, "eval", self.est_eval_s(cfg))
        dev = self._dev()
        variants = evaluate_run(rd, cfg, self.datasets(), dev, progress=self.tr.update, log=self.log)
        marker.write_text(json.dumps({"variants": variants, "datasets": self.datasets(),
                                      "decisions_sha": _sha(paths.decisions_file())}, indent=1))
        self.tr.end_job(name, "eval", "eval", 2 if "mc" in variants else 1)

    def efficiency(self, exp: str) -> None:
        from .efficiency import measure

        cfg = self.cfg(exp, 0)
        f = self.runs / exp / "efficiency.json"
        name = f"cost {exp}"
        if f.exists():
            self.tr.skip_job(name)
            return
        self.tr.start_job(name, 1, "efficiency", 60)
        rep = measure(cfg, self.log)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rep, indent=2))
        self.tr.end_job(name, "efficiency")
        self.log(f"  cost {exp}: {rep['params_m']:.3f} M params, {rep['gflops']:.2f} GFLOPs, CPU {rep['cpu_4t']['median_ms']:.0f} ms, "
                 f"GPU {rep.get('gpu_fp32', {}).get('median_ms', float('nan')):.1f} ms")

    def ensemble(self, name: str, spec: dict) -> None:
        from .evaluate import evaluate, load_model

        rd = self.runs / name
        marker = rd / "eval" / "eval_done.json"
        if marker.exists():
            self.tr.skip_job(f"eval {name}")
            return
        seeds = self.seeds if self.seeds is not None else spec["seeds"]
        members = [paths.run_dir(spec["members"], s) for s in seeds]
        if not all((m / "best.pt").exists() for m in members):
            self.log(f"  {name}: members missing, skipped")
            self.tr.skip_job(f"eval {name}")
            return
        cfg = self.cfg(spec["members"], 0)
        dev = self._dev()
        models = [load_model(m, self.cfg(spec["members"], s), dev) for m, s in zip(members, seeds)]
        self.tr.start_job(f"eval {name}", 1, "eval", self.est_eval_s(cfg) * 2)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "members.json").write_text(json.dumps({"members": [str(m) for m in members]}, indent=1))
        evaluate(models, cfg, rd, "ens", self.datasets(), dev, 0, True, self.tr.update, self.log)
        (rd / "eval").mkdir(exist_ok=True)
        marker.write_text(json.dumps({"variants": ["ens"], "datasets": self.datasets()}, indent=1))
        self.tr.end_job(f"eval {name}", "eval")

    def _dev(self):
        import torch
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------- planning / ETA
    def update_eta(self, remaining_jobs: list[tuple[str, str, int]]) -> None:
        """remaining_jobs: [(kind, exp, seed)] still to run (best effort; unknown compact configs use stage-D guesses)."""
        tot = 0.0
        for kind, exp, seed in remaining_jobs:
            try:
                cfg = self.cfg(exp, seed)
            except Exception:  # noqa: BLE001 - compact backbone or budget not decided yet
                from .config import AttrDict
                cfg = AttrDict.wrap({"train": {"iters": load_decisions().get("budget_iters", 12000)},
                                     "model": {"backbone": "unet" if "unet" in exp or exp[0] in "ABC" else "lite", "width": 64 if exp[0] in "ABC" else 16,
                                               "dropout": 0.1 if exp.startswith("E") else 0},
                                     "dg": {"enabled": True}, "loss": {"topo": "cldice"}})
            tot += self.est_train_s(cfg) if kind == "train" else self.est_eval_s(cfg) if kind == "eval" else 60
        self.tr.state["eta_total_s"] = tot
        self.tr.flush(force=True)

    def plan(self) -> list[tuple[str, str, int]]:
        spec = experiments_spec()
        out = []
        for exp, seed, g in group_runs(self.groups, self.seeds):
            role = spec["experiments"][exp].get("role", "main")
            out.append(("train", exp, seed))
            if role != "pilot":
                out.append(("eval", exp, seed))
        return out

    # --------------------------------------------------------------- phases
    def run(self) -> None:
        try:
            self._run()
        except KeyboardInterrupt:
            self.log("interrupted - re-run the same command to resume (training resumes from the last checkpoint)")
            raise
        except Exception:
            self.log("FAILED:\n" + traceback.format_exc())
            raise

    def _todo(self, plan):
        rem = []
        for kind, exp, seed in plan:
            rd = paths.run_dir(exp, seed)
            if kind == "train" and not (rd / "train_done.json").exists():
                rem.append((kind, exp, seed))
            if kind == "eval" and not (rd / "eval" / "eval_done.json").exists():
                rem.append((kind, exp, seed))
        return rem

    def _run(self) -> None:
        from .data.prepare import is_prepared, prepare

        spec = experiments_spec()
        plan = self.plan()
        groups_run = [g for g in spec["order"] if g in (self.groups or spec["order"])]
        n_cost = len({e for g in groups_run if g != "pilot" for e in spec["groups"][g]["experiments"]})
        n_ens = len(spec.get("ensembles", {})) if "primary" in groups_run else 0
        self.tr.set_plan(1 + len(plan) + n_cost + n_ens + 1)
        self.log(f"study started: groups={self.groups or spec['order']}, runs dir={self.runs}, smoke={self.smoke}")
        self.update_eta(self._todo(plan))

        # 1 data
        self.tr.phase("prepare data")
        if not is_prepared():
            self.tr.start_job("prepare data", 1, "prepare")
            prepare(self.cfg_base(), log=self.log)
            self.tr.end_job("prepare data", "prepare")
        else:
            self.tr.skip_job("prepare data")
            self.log("  data already prepared")

        wanted = self.groups or spec["order"]
        dec = load_decisions()
        # 2 pilot -> budget
        if "budget_iters" not in dec:
            self.tr.phase("pilot (DRIVE only): training budget")
            for exp in spec["groups"]["pilot"]["experiments"]:
                self.train(exp, 0)
            from .selection import decide_budget
            res = decide_budget([paths.run_dir(e, 0) / "train_log.csv" for e in spec["groups"]["pilot"]["experiments"]])
            dec = load_decisions()
            dec["budget_iters"] = res["budget_iters"]
            dec["budget"] = res
            save_decisions(dec)
            self.log(f"  DECISION budget = {res['budget_iters']} iterations  ({res['pilots']})")
            self.update_eta(self._todo(plan))
        # 3 backbone
        if "compact_model" not in dec:
            self.tr.phase("backbone selection (DRIVE only)")
            g = spec["groups"]["backbone"]
            seeds = self.seeds if self.seeds is not None else g["seeds"]
            for s in seeds:
                for exp in g["experiments"]:
                    self.train(exp, s)
            cands = {}
            for exp in g["experiments"]:
                self.efficiency(exp)
                cfg = self.cfg(exp, 0)
                eff = json.loads((self.runs / exp / "efficiency.json").read_text())
                cands[exp] = {"val_dice": [json.loads((paths.run_dir(exp, s) / "train_done.json").read_text())["best_val_dice"] for s in seeds],
                              "cpu_ms": eff["cpu_4t"]["median_ms"], "params": eff["params"],
                              "model": {"backbone": cfg.model.backbone, "width": int(cfg.model.width)}}
            from .selection import decide_backbone
            res = decide_backbone(cands)
            dec = load_decisions()
            dec["compact_model"] = res["compact_model"]
            dec["backbone"] = res
            save_decisions(dec)
            self.log(f"  DECISION compact backbone = {res['compact_choice']} {res['compact_model']}")
        # 4 freeze
        if not dec.get("frozen"):
            self.tr.phase("freeze protocol")
            dec = load_decisions()
            dec["frozen"] = True
            dec["frozen_at"] = datetime.now().isoformat(timespec="seconds")
            dec["frozen_files_sha256"] = {f: _sha(paths.ROOT / f) for f in ("protocol/PROTOCOL.md", "configs/base.yaml", "configs/experiments.yaml")}
            save_decisions(dec)
            self.log("  protocol frozen: target datasets unlocked")
        self.update_eta(self._todo(plan))

        # 5 groups
        for g in [x for x in spec["order"] if x in wanted and x != "pilot"]:
            self.tr.phase(f"group {g}")
            gs = spec["groups"][g]
            seeds = self.seeds if self.seeds is not None else gs["seeds"]
            for s in seeds:
                for exp in gs["experiments"]:
                    self.train(exp, s)
                    self.evaluate(exp, s)
                    self.update_eta(self._todo(plan))
                if g == "primary" and self.report_every_group:     # early results: A-E table after every seed round
                    self.report()
            for exp in gs["experiments"]:
                self.efficiency(exp)
            if g == "primary":
                for name, es in spec.get("ensembles", {}).items():
                    self.ensemble(name, es)
            if self.report_every_group:
                self.report()
        self.report()
        self.tr.finish()
        self.log("study finished - results exported to " + str(results_root()))

    def report(self) -> None:
        from .report import build_report

        self.tr.start_job("report + export", 1, "report", 120)
        self.tr.counted.discard("report + export")
        if self.tr.state["done_jobs"] >= self.tr.state["planned_jobs"]:
            self.tr.set_plan(self.tr.state["planned_jobs"] + 1)
        try:
            build_report(self.runs, results_root(), log=self.log)
        except Exception:  # noqa: BLE001 - a reporting bug must never kill a multi-day run
            self.log("report failed (study continues):\n" + traceback.format_exc())
        self.tr.update(1, 1)
        self.tr.end_job("report + export", "report")


def results_root() -> Path:
    return Path(os.environ.get("TGLITE_RESULTS", paths.RESULTS))


def _sha(p: Path) -> str | None:
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


__all__ = ["Suite", "Tracker", "config_hash", "is_frozen"]
