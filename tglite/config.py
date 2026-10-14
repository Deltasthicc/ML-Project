"""YAML configuration helpers: layered files + ``key.sub=value`` command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

# Project root = parent of the ``tglite`` package directory (works for editable installs / plain checkouts).
ROOT = Path(__file__).resolve().parent.parent


class AttrDict(dict):
    """dict with attribute access (``cfg.model.backbone``); nested dicts are converted lazily."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover - mirrors normal attribute errors
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def deep_update(base: dict, new: dict) -> dict:
    """Recursively merge ``new`` into ``base`` (in place) and return ``base``."""
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def parse_override(text: str) -> dict:
    """``"a.b.c=3"`` -> ``{"a": {"b": {"c": 3}}}`` (value parsed as YAML so lists/bools/floats work)."""
    key, _, raw = text.partition("=")
    if not _:
        raise ValueError(f"override must look like key=value, got {text!r}")
    value = yaml.safe_load(raw)
    out: dict = {}
    cur = out
    parts = key.strip().split(".")
    for part in parts[:-1]:
        cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
    return out


def load_config(files: Iterable[str | Path], overrides: Iterable[str] = ()) -> AttrDict:
    """Load and merge YAML files in order, then apply ``key=value`` overrides."""
    cfg: dict = {}
    for f in files:
        path = Path(f)
        if not path.is_absolute() and not path.exists():
            path = ROOT / path
        with open(path, "r", encoding="utf-8") as fh:
            deep_update(cfg, yaml.safe_load(fh) or {})
    for ov in overrides:
        deep_update(cfg, parse_override(ov))
    return AttrDict(cfg)


def resolve_path(p: str | Path) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def dump_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_plain(cfg), fh, sort_keys=False)


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj
