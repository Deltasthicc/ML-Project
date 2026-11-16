"""Configuration: base.yaml + experiment entry + protocol decisions + ``key=value`` overrides."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from . import paths


class AttrDict(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict):
            return AttrDict({k: AttrDict.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [AttrDict.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_overrides(items) -> dict:
    """``["train.iters=100", "dg.enabled=true"]`` -> nested dict (values parsed as YAML)."""
    out: dict = {}
    for it in items or ():
        key, _, val = it.partition("=")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


# ------------------------------------------------------------------------------------------- protocol decisions
def load_decisions() -> dict:
    f = paths.decisions_file()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_decisions(dec: dict) -> None:
    f = paths.decisions_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dec, indent=2), encoding="utf-8")


def is_frozen() -> bool:
    return bool(load_decisions().get("frozen"))


# ---------------------------------------------------------------------------------------------- experiments
def experiments_spec() -> dict:
    return read_yaml(paths.CONFIGS / "experiments.yaml")


def make_config(exp: str, seed: int = 0, overrides=(), decisions: dict | None = None) -> AttrDict:
    spec = experiments_spec()
    if exp not in spec["experiments"]:
        raise KeyError(f"unknown experiment {exp!r}; known: {sorted(spec['experiments'])}")
    entry = copy.deepcopy(spec["experiments"][exp])
    dec = load_decisions() if decisions is None else decisions
    cfg = read_yaml(paths.CONFIGS / "base.yaml")

    # compact backbone: base -> decided compact model -> explicit experiment keys (e.g. width sweep)
    m = entry.get("model", {})
    if m.get("backbone") == "compact":
        if "compact_model" not in dec:
            raise RuntimeError(f"{exp} uses the compact backbone, which is chosen in the 'backbone' phase "
                               "(protocol/decisions.json has no compact_model yet)")
        deep_update(cfg["model"], dec["compact_model"])
        m = {k: v for k, v in m.items() if k != "backbone"}
        entry["model"] = m
    role = entry.pop("role", "main")
    deep_update(cfg, entry)
    deep_update(cfg, parse_overrides(overrides) if not isinstance(overrides, dict) else overrides)

    if cfg["train"]["iters"] == "auto":
        if "budget_iters" not in dec:
            raise RuntimeError(f"{exp}: train.iters=auto but the training budget has not been decided (pilot phase)")
        cfg["train"]["iters"] = int(dec["budget_iters"])
    wu = cfg["loss"].get("topo_warmup_iters", 0) or 0
    if 0 < wu < 1:                                          # fraction of the budget
        cfg["loss"]["topo_warmup_iters"] = int(round(wu * cfg["train"]["iters"]))
    cfg.update(name=exp, seed=int(seed), role=role)
    return AttrDict.wrap(cfg)


def config_hash(cfg: dict) -> str:
    """Hash of everything that changes training (used to refuse resuming a run under a different protocol)."""
    keep = {k: v for k, v in dict(cfg).items() if k not in ("eval", "efficiency")}
    keep["train"] = {k: v for k, v in dict(keep["train"]).items() if k not in ("num_workers", "log_every")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dump_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(json.loads(json.dumps(cfg)), fh, sort_keys=False)


def group_runs(groups=None, seeds_override=None) -> list[tuple[str, int, str]]:
    """[(experiment, seed, group)] in execution order (seed-major inside a group so early seeds finish first)."""
    spec = experiments_spec()
    out = []
    for g in spec["order"]:
        if groups and g not in groups:
            continue
        gs = spec["groups"][g]
        seeds = seeds_override if seeds_override is not None else gs["seeds"]
        for s in seeds:
            for e in gs["experiments"]:
                out.append((e, int(s), g))
    return out
