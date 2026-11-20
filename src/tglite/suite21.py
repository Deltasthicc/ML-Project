"""TG-LiteNet v2.1 study (protocol/PROTOCOL_v2.1.md) as one resumable job list.

Phases (every job is skipped when its output exists, so re-running the same command resumes; training resumes from
the last checkpoint):
  1 rules     record the SHA-256 of the v2.1 protocol + config (rules fixed before any v2.1 result exists)
  2 teacher   probability maps of the v2 stage-C U-Net ensemble on DRIVE-train (distillation targets)
  3 reeval    v2 models (A, C, D, E det/mc, ensembles) with the v2.1 evaluation -> novelties 1 and 2 on existing models
  4 arch      L_cap vs L_wnet (3 seeds, DRIVE-val)                         -> decisions_v21.arch
  5 weights   lambda_topo x lambda_kd_skel grid (2 seeds, DRIVE-val)        -> decisions_v21.weights
  6 train     headline (5 seeds), references (3), ablations (3)
  7 freeze    size rule (DRIVE-val) + freeze -> targets unlocked for the new models
  8 eval      v2.1 evaluation of every new model + 5-seed ensemble, cost, INT8
  9 report    results/v21/ (also written after phase 3 for early results)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from . import paths
from .config import AttrDict, deep_update, make_config, parse_overrides, read_yaml
from .suite import Tracker


def v21_root() -> Path:
    return paths.runs_root() / "v21"


def decisions21_file() -> Path:
    return Path(os.environ.get("TGLITE_DECISIONS_V21", paths.PROTOCOL / "decisions_v21.json"))


def results21_root() -> Path:
    return Path(os.environ.get("TGLITE_RESULTS_V21", paths.RESULTS / "v21"))


def load_dec21() -> dict:
    f = decisions21_file()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_dec21(d: dict) -> None:
    f = decisions21_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d, indent=2), encoding="utf-8")


def spec21() -> dict:
    return read_yaml(paths.CONFIGS / "v21.yaml")


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def make_config21(exp: str, seed: int, overrides=(), dec: dict | None = None, extra: dict | None = None) -> AttrDict:
    spec = spec21()
    dec = load_dec21() if dec is None else dec
    entry = copy.deepcopy(spec["experiments"][exp]) if exp in spec["experiments"] else {}
    cfg = read_yaml(paths.CONFIGS / "base.yaml")
    deep_update(cfg, copy.deepcopy(spec["defaults"]))
    arch = entry.pop("arch", None)
    if arch == "headline":
        if "arch" not in dec:
            raise RuntimeError(f"{exp}: the headline architecture has not been decided yet")
        arch = dec["arch"]
    if arch:
        deep_update(cfg["model"], copy.deepcopy(spec["archs"][arch]))
    if entry.pop("weights", None) == "chosen":
        if "weights" not in dec:
            raise RuntimeError(f"{exp}: the loss weights have not been decided yet")
        deep_update(cfg, copy.deepcopy(dec["weights"]))
    role = entry.pop("role", "main")
    deep_update(cfg, entry)
    if extra:
        deep_update(cfg, copy.deepcopy(extra))
    deep_update(cfg, parse_overrides(overrides))
    cfg["distill"]["teacher_dir"] = str(v21_root() / "_teacher")
    cfg.update(name=exp, seed=int(seed), role=role, protocol="v2.1")
    return AttrDict.wrap(cfg)


def grid_name(lt: float, lk: float) -> str:
    return f"G_t{int(round(lt * 100)):03d}_k{int(round(lk * 100)):03d}"


class Suite21:
    def __init__(self, overrides=(), smoke=False, device=None, use_tqdm=True):
        self.overrides = list(overrides)
        self.smoke = smoke
        self.device = device
        self.root = v21_root()
        self.tr = Tracker(self.root / "_suite", use_tqdm=use_tqdm)
        self.log = self.tr.log
        self.spec = spec21()
        sd = self.spec["seeds"]
        cut = (lambda v: v[:2]) if smoke else (lambda v: v)
        self.seeds = {k: cut(list(v)) for k, v in sd.items()}
        self.reeval = {k: cut(list(v)) for k, v in self.spec["reeval"]["runs"].items()}

    # ------------------------------------------------------------------ helpers
    def _dev(self):
        import torch
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def rd(self, exp, seed) -> Path:
        return self.root / exp / f"seed{seed}"

    def cfg(self, exp, seed, extra=None):
        return make_config21(exp, seed, self.overrides, extra=extra)

    def datasets(self):
        b = read_yaml(paths.CONFIGS / "base.yaml")["data"]
        return ["drive", *b["targets"], *b["extra_targets"]]

    def _speed_default(self, cfg) -> float:
        return {"wnet": 0.07, "lite": 0.1}.get(cfg.model.backbone, 0.06)

    def est_train(self, cfg) -> float:
        return 60 + int(cfg.train.iters) * self.tr.speed(f"train21:{cfg.model.backbone}:{cfg.model.width}", self._speed_default(cfg))

    def train(self, exp, seed, extra=None, name=None):
        from .train import train_run

        name = name or exp
        cfg = self.cfg(exp, seed, extra)
        cfg["name"] = name
        rd = self.rd(name, seed)
        job = f"train {name} s{seed}"
        if (rd / "train_done.json").exists():
            self.tr.skip_job(job)
            return json.loads((rd / "train_done.json").read_text())
        self.tr.start_job(job, int(cfg.train.iters), "train", self.est_train(cfg))
        start = 0
        if (rd / "last.pt").exists():
            import torch
            start = int(torch.load(rd / "last.pt", map_location="cpu", weights_only=False)["iter"])
        done = train_run(cfg, rd, self.device, progress=self.tr.update, log=self.log)
        self.tr.end_job(job, "train", f"train21:{cfg.model.backbone}:{cfg.model.width}", max(1, int(cfg.train.iters) - start))
        self.log(f"  done {name} s{seed}: best DRIVE-val Dice {done['best_val_dice']:.4f} @ {done['best_iter']}, "
                 f"{done['train_time_s'] / 60:.1f} min, {done['params']} params")
        return done

    def val_at_best(self, name, seed) -> dict:
        import pandas as pd
        rd = self.rd(name, seed)
        done = json.loads((rd / "train_done.json").read_text())
        df = pd.read_csv(rd / "train_log.csv").dropna(subset=["val_dice"])
        row = df.loc[df["iter"] == done["best_iter"]].iloc[0]
        return {"val_dice": float(row["val_dice"]), "val_cldice": float(row["val_cldice"])}

    def efficiency(self, name, cfg) -> dict:
        from .efficiency import measure
        f = self.root / name / "efficiency.json"
        job = f"cost {name}"
        if f.exists():
            self.tr.skip_job(job)
            return json.loads(f.read_text())
        self.tr.start_job(job, 1, "efficiency", 90)
        rep = measure(cfg, self.log)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rep, indent=2))
        self.tr.end_job(job, "efficiency")
        self.log(f"  cost {name}: {rep['params']} params, {rep['gflops']:.2f} GFLOPs, CPU {rep['cpu_4t']['median_ms']:.1f} ms")
        return rep

    def eval_models(self, job, models_fn, cfg, out_dir: Path, variant: str, mc: int = 0) -> None:
        from .eval21 import evaluate21
        marker = Path(out_dir) / "eval21" / f"done_{variant}.json"
        if marker.exists():
            self.tr.skip_job(job)
            return
        self.tr.start_job(job, 1, "eval", self.tr.speed("eval21", 300.0) * (4 if mc else 1))
        models = models_fn()
        evaluate21(models, cfg, out_dir, variant, self.datasets(), self._dev(), mc, self.tr.update, self.log, allow_targets=True)
        marker.write_text(json.dumps({"variant": variant, "at": datetime.now().isoformat(timespec="seconds")}))
        self.tr.end_job(job, "eval", "eval21" if not mc else None, 1)

    def report(self):
        from .report21 import build_report21
        self.tr.start_job("report + export", 1, "report", 120)
        self.tr.counted.discard("report + export")
        if self.tr.state["done_jobs"] >= self.tr.state["planned_jobs"]:
            self.tr.set_plan(self.tr.state["planned_jobs"] + 1)
        try:
            build_report21(paths.runs_root(), self.root, results21_root(), log=self.log)
        except Exception:  # noqa: BLE001 - a reporting bug must never kill the run
            self.log("report failed (study continues):\n" + traceback.format_exc())
        self.tr.update(1, 1)
        self.tr.end_job("report + export", "report")

    # ------------------------------------------------------------------ plan size (for the progress bar)
    def _plan_size(self) -> int:
        s = self.seeds
        n = 1 + 1                                                          # rules + teacher
        n += sum(len(v) for v in self.reeval.values()) + len(self.reeval.get("E_tglitenet", [])) + len(self.spec["reeval"]["ensembles"])
        n += 2 * len(s["selection"]) + 2                                   # arch runs + cost
        n += 4 * min(2, len(s["selection"]))                               # weight grid
        n += len(s["headline"]) + 2 * len(s["reference"]) + 5 * len(s["ablation"])
        n += 1                                                             # freeze
        n += len(s["headline"]) + 2 * len(s["reference"]) + 5 * len(s["ablation"]) + 1   # evaluations + ensemble
        n += 3 + 1 + 2                                                     # cost, int8, reports
        return n

    # ------------------------------------------------------------------ run
    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            self.log("interrupted - re-run the same command to resume")
            raise
        except Exception:
            self.log("FAILED:\n" + traceback.format_exc())
            raise

    def _run(self):
        from .config import load_decisions
        from .data.prepare import is_prepared

        if not is_prepared():
            raise RuntimeError("data not prepared - run scripts/prepare_data.py")
        if not load_decisions().get("frozen"):
            raise RuntimeError("the v2 protocol is not frozen; v2.1 builds on the finished v2 study")
        self.tr.set_plan(self._plan_size())
        self.log(f"v2.1 study started: root={self.root}, smoke={self.smoke}")
        s = self.seeds

        # 1 rules -------------------------------------------------------------------------------------------
        self.tr.phase("v2.1 rules")
        dec = load_dec21()
        if not dec.get("rules_frozen"):
            dec.update(rules_frozen=True, rules_frozen_at=datetime.now().isoformat(timespec="seconds"),
                       rules_sha256={f: _sha(paths.ROOT / f) for f in ("protocol/PROTOCOL_v2.1.md", "configs/v21.yaml")})
            save_dec21(dec)
            self.log("  v2.1 rules recorded (protocol + config hashes)")
        self.tr.skip_job("rules")

        # 2 teacher maps ------------------------------------------------------------------------------------
        self.tr.phase("teacher maps (DRIVE-train)")
        self.teacher_maps()

        # 3 re-evaluate v2 models --------------------------------------------------------------------------
        self.tr.phase("re-evaluate v2 models (novelties 1, 2)")
        from .evaluate import load_model
        dev = self._dev()
        for exp, seeds in self.reeval.items():
            for sd in seeds:
                rd = paths.run_dir(exp, sd)
                if not (rd / "best.pt").exists():
                    self.log(f"  {exp} s{sd}: no checkpoint, skipped")
                    self.tr.skip_job(f"reeval {exp} s{sd}")
                    continue
                c2 = make_config(exp, sd, self.overrides)
                self.eval_models(f"reeval {exp} s{sd}", lambda c2=c2, rd=rd: [load_model(rd, c2, dev)], c2, rd, "det")
                if float(c2.model.get("dropout", 0) or 0) > 0:
                    self.eval_models(f"reeval {exp} s{sd} mc", lambda c2=c2, rd=rd: [load_model(rd, c2, dev)], c2, rd, "mc",
                                     int(c2.eval.mc_samples))
        spec2 = read_yaml(paths.CONFIGS / "experiments.yaml")
        for ens in self.spec["reeval"]["ensembles"]:
            es = spec2["ensembles"][ens]
            seeds = self.reeval.get(es["members"], es["seeds"])
            members = [paths.run_dir(es["members"], sd) for sd in seeds]
            if not all((m / "best.pt").exists() for m in members):
                self.tr.skip_job(f"reeval {ens}")
                continue
            c2 = make_config(es["members"], 0, self.overrides)
            self.eval_models(f"reeval {ens}", lambda es=es, seeds=seeds, members=members: [
                load_model(m, make_config(es["members"], sd, self.overrides), dev) for m, sd in zip(members, seeds)],
                c2, paths.runs_root() / ens, "ens")
        self.report()

        # 4 architecture ------------------------------------------------------------------------------------
        dec = load_dec21()
        if "arch" not in dec:
            self.tr.phase("v2.1 architecture (DRIVE-val)")
            for sd in s["selection"]:
                for a in ("L_cap", "L_wnet"):
                    self.train(a, sd)
            cands = {}
            for a in ("L_cap", "L_wnet"):
                rep = self.efficiency(a, self.cfg(a, 0))
                cands[a] = {"val_dice": [self.val_at_best(a, sd)["val_dice"] for sd in s["selection"]],
                            "cpu_ms": rep["cpu_4t"]["median_ms"], "params": rep["params"]}
            means = {a: float(np.mean(v["val_dice"])) for a, v in cands.items()}
            top = max(means.values())
            near = [a for a in cands if means[a] >= top - 0.005]
            choice = min(near, key=lambda a: (cands[a]["cpu_ms"], cands[a]["params"]))
            dec = load_dec21()
            dec["arch"] = choice
            dec["arch_selection"] = {"candidates": {a: {**v, "mean_val_dice": means[a]} for a, v in cands.items()},
                                     "rule": "mean DRIVE-val Dice within 0.005 of the best candidate; lowest CPU latency"}
            save_dec21(dec)
            self.log(f"  DECISION v2.1 architecture = {choice}  (val Dice {means})")
        else:
            for sd in s["selection"]:
                for a in ("L_cap", "L_wnet"):
                    self.tr.skip_job(f"train {a} s{sd}")
            for a in ("L_cap", "L_wnet"):
                self.tr.skip_job(f"cost {a}")

        # 5 loss weights ------------------------------------------------------------------------------------
        dec = load_dec21()
        grid = self.spec["weight_grid"]
        gseeds = s["selection"][:2]
        if "weights" not in dec:
            self.tr.phase("v2.1 loss weights (DRIVE-val)")
            arch = dec["arch"]
            score = {}
            for lt in grid["lambda_topo"]:
                for lk in grid["lambda_kd_skel"]:
                    name = grid_name(lt, lk)
                    vals = []
                    for sd in gseeds:
                        if lt == 0.5 and lk == 0.5:          # identical to the architecture run -> reuse it
                            self.tr.skip_job(f"train {name} s{sd}")
                            vals.append(self.val_at_best(arch, sd))
                            continue
                        self.train(arch, sd, extra={"loss": {"lambda_topo": lt}, "distill": {"lambda_kd_skel": lk}}, name=name)
                        vals.append(self.val_at_best(name, sd))
                    score[name] = {"lambda_topo": lt, "lambda_kd_skel": lk,
                                   "score": float(np.mean([(v["val_dice"] + v["val_cldice"]) / 2 for v in vals])),
                                   "val_dice": float(np.mean([v["val_dice"] for v in vals])),
                                   "val_cldice": float(np.mean([v["val_cldice"] for v in vals]))}
            best = max(v["score"] for v in score.values())
            tied = [k for k, v in score.items() if v["score"] >= best - 1e-9]
            pick = grid_name(0.5, 0.5) if grid_name(0.5, 0.5) in tied else tied[0]
            dec = load_dec21()
            dec["weights"] = {"loss": {"lambda_topo": score[pick]["lambda_topo"]}, "distill": {"lambda_kd_skel": score[pick]["lambda_kd_skel"]}}
            dec["weight_selection"] = {"grid": score, "choice": pick,
                                       "rule": "max mean (val Dice + val clDice)/2 at the best checkpoint, 2 seeds; ties -> defaults"}
            save_dec21(dec)
            self.log(f"  DECISION v2.1 loss weights = {dec['weights']}")
        else:
            for lt in grid["lambda_topo"]:
                for lk in grid["lambda_kd_skel"]:
                    for sd in gseeds:
                        self.tr.skip_job(f"train {grid_name(lt, lk)} s{sd}")

        # 6 training ----------------------------------------------------------------------------------------
        self.tr.phase("v2.1 training: headline, references, ablations")
        abl = ["Z_no_kdskel", "Z_no_kd", "Z_pixcons", "Z_noaug", "Z_v2recipe"]
        for sd in s["headline"]:
            self.train("H_tglite70k", sd)
        for sd in s["reference"]:
            for e in ("R_lite_w16", "R_unet_w8"):
                self.train(e, sd)
        for sd in s["ablation"]:
            for e in abl:
                self.train(e, sd)

        # 7 size rule + freeze ------------------------------------------------------------------------------
        dec = load_dec21()
        if not dec.get("frozen"):
            self.tr.phase("v2.1 size rule + freeze")
            common = s["reference"]
            cands = {}
            for e in ("H_tglite70k", "R_lite_w16", "R_unet_w8"):
                rep = self.efficiency(e, self.cfg(e, 0))
                cands[e] = {"mean_val_dice": float(np.mean([self.val_at_best(e, sd)["val_dice"] for sd in common])),
                            "params": rep["params"]}
            top = max(v["mean_val_dice"] for v in cands.values())
            ok = [e for e, v in cands.items() if v["mean_val_dice"] >= top - 0.005]
            head = min(ok, key=lambda e: cands[e]["params"])
            dec = load_dec21()
            dec["headline"] = head
            dec["size_rule"] = {"candidates": cands, "rule": "smallest model whose mean DRIVE-val Dice (seeds "
                                f"{common}) is within 0.005 of the best"}
            dec["frozen"] = True
            dec["frozen_at"] = datetime.now().isoformat(timespec="seconds")
            save_dec21(dec)
            self.log(f"  DECISION v2.1 headline model = {head}  ({cands});  v2.1 protocol frozen: targets unlocked")
        else:
            for e in ("H_tglite70k", "R_lite_w16", "R_unet_w8"):
                self.tr.skip_job(f"cost {e}")
        self.tr.skip_job("freeze")

        # 8 evaluation --------------------------------------------------------------------------------------
        self.tr.phase("v2.1 evaluation")
        from .evaluate import load_model
        runs = [("H_tglite70k", sd) for sd in s["headline"]] + [(e, sd) for sd in s["reference"] for e in ("R_lite_w16", "R_unet_w8")] \
            + [(e, sd) for sd in s["ablation"] for e in abl]
        for e, sd in runs:
            c = self.cfg(e, sd)
            self.eval_models(f"eval {e} s{sd}", lambda c=c, e=e, sd=sd: [load_model(self.rd(e, sd), c, dev)], c, self.rd(e, sd), "det")
        c0 = self.cfg("H_tglite70k", 0)
        self.eval_models("eval ENS_H", lambda: [load_model(self.rd("H_tglite70k", sd), self.cfg("H_tglite70k", sd), dev)
                                                 for sd in s["headline"]], c0, self.root / "ENS_H_tglite70k", "ens")
        for e in ("Z_v2recipe",):
            self.efficiency(e, self.cfg(e, 0))
        self.int8()
        self.report()
        self.tr.finish()
        self.log("v2.1 study finished - results exported to " + str(results21_root()))

    # ------------------------------------------------------------------ teacher + int8
    def teacher_maps(self):
        out = self.root / "_teacher"
        job = "teacher maps"
        if (out / "done.json").exists():
            self.tr.skip_job(job)
            return
        import torch
        from .data import load_split, to_input
        from .evaluate import load_model
        from .inference import predict
        self.tr.start_job(job, 1, "teacher", 60)
        dev = self._dev()
        t = self.spec["defaults"]["distill"]
        seeds = self.reeval.get(t["teacher"], t["teacher_seeds"])
        models = [load_model(paths.run_dir(t["teacher"], sd), make_config(t["teacher"], sd, self.overrides), dev) for sd in seeds]
        c = make_config(t["teacher"], 0, self.overrides)
        out.mkdir(parents=True, exist_ok=True)
        for smp in load_split("drive_train", c):
            x = to_input(smp, c).to(dev)
            ps = [predict(models, xx)[0] for xx in (x, torch.flip(x, [3]), torch.flip(x, [2]))]
            p = (ps[0] + ps[1][:, ::-1] + ps[2][::-1, :]) / 3.0
            p[smp["fov"] == 0] = 0
            np.save(out / f"{smp['id']}.npy", p.astype(np.float16))
        (out / "done.json").write_text(json.dumps({"teacher": t["teacher"], "seeds": seeds, "tta": "identity + h/v flips"}))
        self.tr.end_job(job, "teacher")
        self.log(f"  teacher maps: {t['teacher']} seeds {seeds} (+ flip averaging) on DRIVE-train -> {out}")

    def int8(self):
        job = "int8 headline"
        f = self.root / "H_tglite70k" / "int8.json"
        if f.exists():
            self.tr.skip_job(job)
            return
        self.tr.start_job(job, 1, "int8", 300)
        try:
            from .int8 import int8_report
            rep = int8_report(self.rd("H_tglite70k", 0), self.cfg("H_tglite70k", 0), self.datasets(), log=self.log)
            f.write_text(json.dumps(rep, indent=2))
        except Exception:  # noqa: BLE001 - optional deliverable
            self.log("int8 failed (study continues):\n" + traceback.format_exc())
            f.write_text(json.dumps({"failed": True}))
        self.tr.end_job(job, "int8")
