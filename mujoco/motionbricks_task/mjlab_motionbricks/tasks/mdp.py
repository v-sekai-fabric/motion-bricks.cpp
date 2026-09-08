"""The ROM terms: a soft penalty outside the permissive E.3 envelope and the
clearance metric inside the conservative one, both read from the envelope
artefact (`rom_envelope.py`), so the reward, the metric and the gate agree."""
from __future__ import annotations

import math

import pyarrow.parquet as pq
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_motionbricks.robot.g1_constants import ENVELOPE_PARQUET, GATE_JOINTS

_BOUNDS: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
_SHIFT_DEG = 0.0


def set_envelope_shift(deg: float) -> None:
    """The negative control: shift the envelope so a passing policy must read low."""
    global _SHIFT_DEG
    _SHIFT_DEG = deg
    _BOUNDS.clear()


def _envelope_rad(kind: str) -> dict[str, tuple[float, float]]:
    if not ENVELOPE_PARQUET.is_file():
        raise FileNotFoundError(f"{ENVELOPE_PARQUET} is missing; run `pixi run envelope` first")
    t = pq.read_table(ENVELOPE_PARQUET).to_pylist()
    out = {}
    for r in t:
        lo, hi = r["qpos_min_deg"] + _SHIFT_DEG, r["qpos_max_deg"] + _SHIFT_DEG
        if kind == "permissive":
            # the permissive bound reaches E.3's angle_deg_max instead of angle_deg_min
            extra = (r["e3_deg"] and 25.0) or 0.0
            if r["flexion_sign"] > 0:
                hi = min(hi + extra, r["mjcf_max_deg"])
            else:
                lo = max(lo - extra, r["mjcf_min_deg"])
        out[r["joint"]] = (math.radians(lo), math.radians(hi))
    missing = [j for j in GATE_JOINTS if j not in out]
    if missing:
        raise ValueError(f"envelope artefact lacks {missing}")
    return out


def _joint_names(asset: Entity) -> list[str]:
    for attr in ("joint_names",):
        names = getattr(asset, attr, None)
        if names is not None:
            return list(names)
    return list(asset.data.joint_names)


def _gate_ids(asset_cfg: SceneEntityCfg, names: list[str]) -> list[int]:
    """The gate joints' indices: from the manager-resolved ids, or from the cfg's
    regexes when the cfg was never resolved (a roll-out calling the term directly)."""
    if not isinstance(asset_cfg.joint_ids, slice):
        return list(asset_cfg.joint_ids)
    import re
    pats = asset_cfg.joint_names or (".*",)
    return [i for i, n in enumerate(names) if any(re.fullmatch(p, n) for p in pats)]


def _bounds(env: ManagerBasedRlEnv, asset: Entity, asset_cfg: SceneEntityCfg, kind: str):
    names = _joint_names(asset)
    ids = _gate_ids(asset_cfg, names)
    key = (id(env), kind, tuple(ids))
    if key not in _BOUNDS:
        env_rad = _envelope_rad(kind)
        selected = [names[i] for i in ids]
        lo = torch.tensor([env_rad[n][0] for n in selected], device=env.device)
        hi = torch.tensor([env_rad[n][1] for n in selected], device=env.device)
        _BOUNDS[key] = (ids, lo, hi)
    return _BOUNDS[key]


def rom_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Radians outside the permissive envelope, summed over the gate joints. Shape [B]."""
    asset: Entity = env.scene[asset_cfg.name]
    ids, lo, hi = _bounds(env, asset, asset_cfg, "permissive")
    q = asset.data.joint_pos[:, ids]
    return torch.sum(torch.relu(lo - q) + torch.relu(q - hi), dim=1)


def rom_clearance(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """The fraction of gate joints inside the conservative envelope, per env. Shape [B].
    Its mean over a rollout is the ">= 90 % ROM clearance" figure; the denominator is
    the eight gate joints times the frames."""
    asset: Entity = env.scene[asset_cfg.name]
    ids, lo, hi = _bounds(env, asset, asset_cfg, "conservative")
    q = asset.data.joint_pos[:, ids]
    inside = (q >= lo) & (q <= hi)
    return inside.float().mean(dim=1)
