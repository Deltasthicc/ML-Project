"""Filesystem layout. Data and heavy run artefacts can be moved with environment variables.

    TGLITE_DATA  (default <repo>/data)   archives as downloaded + processed/ cache/ extracted/ hf_cache/
    TGLITE_RUNS  (default <repo>/runs)   checkpoints, logs, probability maps, suite state (git-ignored)
    results/                             exported, committed final results
    protocol/                            pre-registered protocol + selection decisions (committed)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
PROTOCOL = ROOT / "protocol"
RESULTS = ROOT / "results"


def data_root() -> Path:
    return Path(os.environ.get("TGLITE_DATA", ROOT / "data"))


def runs_root() -> Path:
    return Path(os.environ.get("TGLITE_RUNS", ROOT / "runs"))


def processed_dir() -> Path:
    return data_root() / "processed"


def cache_dir() -> Path:
    return data_root() / "cache"


def decisions_file() -> Path:
    return Path(os.environ.get("TGLITE_DECISIONS", PROTOCOL / "decisions.json"))


def run_dir(exp: str, seed: int, root: Path | None = None) -> Path:
    return (root or runs_root()) / exp / f"seed{seed}"
