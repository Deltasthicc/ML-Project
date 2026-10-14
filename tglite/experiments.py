"""Experiment registry: every run in the paper is ``configs/base.yaml`` + one entry below (+ a seed).

Groups
------
primary    Staged ablation of the paper (Table II):  A -> B -> C -> D -> E, each adding exactly one component.
factorial  The remaining cells of the 2 (backbone) x 2 (topology) x 2 (DG) full-factorial design, so interaction
           effects (e.g. "does topology supervision still help a compact network under shift?", survey gap 4) can
           be read off.  Together with A-D they give all 8 cells.
topology   Topology-loss variants on the stage-D recipe: soft-clDice (= D), radius-aware cbDice, Skeleton Recall, clDice + Skeleton Recall.
width      Capacity sweep of the lite backbone on the stage-D recipe (accuracy / topology / gap vs params, FLOPs, latency).
"""
from __future__ import annotations

from .config import ROOT, AttrDict, deep_update, load_config

TOPO = {"topo": "cldice"}


def _e(backbone: str, topo: str = "none", dg: bool = False, dropout: float = 0.0, **model_extra) -> dict:
    model = {"backbone": backbone, "skeleton_head": topo != "none", "dropout": dropout}
    model.update(model_extra)
    return {"model": model, "loss": {"topo": topo}, "dg": {"enabled": dg}}


REGISTRY: dict[str, dict] = {
    # ---- primary staged ablation -------------------------------------------------------------------------
    "A_unet": _e("unet"),
    "B_unet_topo": _e("unet", "cldice"),
    "C_unet_topo_dg": _e("unet", "cldice", dg=True),
    "D_lite_topo_dg": _e("lite", "cldice", dg=True),
    "E_tglitenet": _e("lite", "cldice", dg=True, dropout=0.1),
    # ---- remaining factorial cells -----------------------------------------------------------------------
    "X_unet_dg": _e("unet", dg=True),
    "X_lite_plain": _e("lite"),
    "X_lite_topo": _e("lite", "cldice"),
    "X_lite_dg": _e("lite", dg=True),
    # ---- topology-loss variants (stage-D recipe) ---------------------------------------------------------
    "V_lite_srl": _e("lite", "skeleton_recall", dg=True),
    "V_lite_cldice_srl": _e("lite", "cldice_skeleton_recall", dg=True),
    "V_lite_cbdice": _e("lite", "cbdice", dg=True),
}
for _w in (8, 24, 32, 48):                                   # width 16 is stage D itself
    REGISTRY[f"W_lite_w{_w}"] = _e("lite", "cldice", dg=True, width=_w)

GROUPS: dict[str, list[str]] = {
    "primary": ["A_unet", "B_unet_topo", "C_unet_topo_dg", "D_lite_topo_dg", "E_tglitenet"],
    "factorial": ["X_unet_dg", "X_lite_plain", "X_lite_topo", "X_lite_dg"],
    "topology": ["V_lite_cbdice", "V_lite_srl", "V_lite_cldice_srl"],
    "width": [f"W_lite_w{w}" for w in (8, 24, 32, 48)],
}
# default number of seeds per group (small, cheap models can afford more seeds)
DEFAULT_SEEDS = {"primary": 5, "factorial": 5, "topology": 3, "width": 3}
# the stage-D recipe is shared by "primary" and the supplementary groups (referenced by the aggregation script)
FACTORIAL_CELLS = {
    ("unet", "none", False): "A_unet", ("unet", "cldice", False): "B_unet_topo",
    ("unet", "none", True): "X_unet_dg", ("unet", "cldice", True): "C_unet_topo_dg",
    ("lite", "none", False): "X_lite_plain", ("lite", "cldice", False): "X_lite_topo",
    ("lite", "none", True): "X_lite_dg", ("lite", "cldice", True): "D_lite_topo_dg",
}

STAGE_LABELS = {
    "A_unet": "A  U-Net baseline",
    "B_unet_topo": "B  + topology loss",
    "C_unet_topo_dg": "C  + DG augmentation",
    "D_lite_topo_dg": "D  + compact backbone",
    "E_tglitenet": "E  TG-LiteNet (full)",
}


def make_config(name: str, seed: int = 0, extra_files=(), overrides=()) -> AttrDict:
    """base.yaml + registry entry + optional yaml files + ``key=value`` overrides."""
    files = ["configs/base.yaml"]
    if (ROOT / "configs" / "tuned.yaml").exists():       # written by scripts/tune_lambdas.py (loss weights chosen on DRIVE-val only)
        files.append("configs/tuned.yaml")
    cfg = load_config(files)
    if name in REGISTRY:
        deep_update(cfg, REGISTRY[name])
    elif not extra_files:
        raise KeyError(f"unknown experiment {name!r}; known: {sorted(REGISTRY)}")
    if extra_files:                                       # user YAMLs win over the registry entry
        deep_update(cfg, load_config(extra_files))
    cfg["name"] = name
    cfg["seed"] = int(seed)
    deep_update(cfg, load_config([], overrides))
    return cfg
