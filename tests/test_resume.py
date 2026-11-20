"""Training resumes from last.pt after an interruption (CPU, real DRIVE data; skipped if data are not prepared)."""
from __future__ import annotations

import json

import pytest

from tglite import paths
from tglite.config import make_config

pytestmark = pytest.mark.skipif(not (paths.PROTOCOL / "drive_split.json").exists() or not (paths.processed_dir() / "drive").exists(),
                                reason="DRIVE not prepared")

DEC = {"budget_iters": 12, "compact_model": {"backbone": "lite", "width": 4}}
OV = ["train.val_every=4", "train.log_every=2", "train.num_workers=0", "train.views_per_step=2", "data.patch_size=64", "train.warmup_iters=2"]


def test_interrupt_and_resume(tmp_path):
    from tglite.train import train_run

    cfg = make_config("D_compact", 0, OV, decisions=DEC)
    calls = []

    class Stop(Exception):
        pass

    def prog(it, total, info):
        calls.append(it)
        if it == 8:
            raise Stop

    with pytest.raises(Stop):
        train_run(cfg, tmp_path, device="cpu", progress=prog, log=lambda *_: None)
    assert (tmp_path / "last.pt").exists() and not (tmp_path / "train_done.json").exists()
    done = train_run(cfg, tmp_path, device="cpu", log=lambda *_: None)
    assert done["iters"] == 12 and (tmp_path / "best.pt").exists()
    import pandas as pd
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert log["iter"].is_unique and log["iter"].max() == 12
    # a changed protocol must not silently resume
    (tmp_path / "train_done.json").unlink()
    with pytest.raises(RuntimeError):
        train_run(make_config("D_compact", 0, OV + ["loss.lambda_topo=0.9"], decisions=DEC), tmp_path, device="cpu", log=lambda *_: None)
    json.loads((tmp_path / "run_meta.json").read_text())
