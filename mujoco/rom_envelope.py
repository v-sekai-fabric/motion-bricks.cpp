"""The eight-joint E.3 envelope on the G1 as one artefact on disk (RFD 2238).

SPDX-License-Identifier: Apache-2.0

``rom_calibrate`` measures the anatomical neutral and the flexion sign for the
knee, elbow and shoulder pitch from the model; ``rom_map`` holds the Appendix
E.3 rows verbatim and the wrist mapping (anatomical wrist flexion/extension is
``wrist_yaw`` on this rig). Neither writes anything down. This module writes
``data/g1_rom_envelope.parquet``: one row per gate joint and bound side, in MJCF
qpos degrees, with the E.3 row and the calibration that produced it, so the
training reward, the clearance metric and the gate all read the same numbers.

    python rom_envelope.py                 # write the parquet and print it
    python rom_envelope.py --check         # refuse when a bound is outside jnt_range,
                                           # a neutral is outside its range, or an E.3
                                           # row is not one of the pinned eight
    python rom_envelope.py --self-test     # the planted swapped sign must be refused
    python rom_envelope.py --refetch       # the pinned E.3 rows against the dataset

The wrist sign is measured too: the right wrist's yaw axis is the left's mirrored
across the sagittal plane, so the flexion direction flips with it. E.3 allows no
hyperextension at the knee, elbow or shoulder (their rows are one-sided); the
wrist has both rows and is two-sided.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rom_calibrate import JointCalibration, calibrate_all
from rom_map import E3_DEG, MJCF_RANGE_RAD

HERE = Path(__file__).resolve().parent
DEFAULT_XML = HERE / "g1.xml"
DEFAULT_OUT = HERE / "data" / "g1_rom_envelope.parquet"
DATASET = "chibifire/starforged-std-3001-appendix-e"

ONE_SIDED = {
    "knee_joint": "knee_flexion",
    "elbow_joint": "elbow_flexion",
    "shoulder_pitch_joint": "shoulder_flexion",
}
TWO_SIDED = {"wrist_yaw_joint": ("wrist_flexion", "wrist_extension")}


@dataclass(frozen=True)
class Bound:
    joint: str
    side: str
    e3_row: str
    e3_deg: float
    q_neutral_deg: float
    flexion_sign: int
    qpos_min_deg: float
    qpos_max_deg: float
    mjcf_min_deg: float
    mjcf_max_deg: float
    binding: str
    method: str


def _wrist_calibration(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, JointCalibration]:
    """Neutral at q = 0 (the hand in line with the forearm in the stand pose);
    flexion +q on the left, and on the right whichever sign the mirrored axis gives."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    axes = {}
    for side in ("left", "right"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_yaw_joint")
        axes[side] = data.xaxis[jid].copy()
    mirrored_left = axes["left"] * np.array([1.0, -1.0, 1.0])
    right_sign = 1 if float(mirrored_left @ axes["right"]) > 0 else -1
    out = {}
    for side, sign in (("left", 1), ("right", right_sign)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_yaw_joint")
        lo, hi = model.jnt_range[jid]
        out[f"{side}_wrist_yaw_joint"] = JointCalibration(
            joint=f"{side}_wrist_yaw_joint", q_neutral_deg=0.0, flexion_sign=sign,
            mjcf_lo_deg=math.degrees(lo), mjcf_hi_deg=math.degrees(hi),
            method=f"wrist_yaw axis mirrored across the sagittal plane (dot {float(mirrored_left @ axes['right']):+.3f})")
    return out


def envelope(xml_path: Path = DEFAULT_XML, conservative: bool = True) -> list[Bound]:
    pick = 0 if conservative else 1
    cal = calibrate_all(str(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    cal.update(_wrist_calibration(model, mujoco.MjData(model)))
    rows: list[Bound] = []
    for side in ("left", "right"):
        for suffix, row in ONE_SIDED.items():
            c = cal[f"{side}_{suffix}"]
            q_flex = c.qpos_for_flexion(E3_DEG[row][pick])
            lo, hi = sorted((q_flex, c.q_neutral_deg))
            qmin, qmax = max(lo, c.mjcf_lo_deg), min(hi, c.mjcf_hi_deg)
            binding = "E.3" if (qmin > c.mjcf_lo_deg + 1e-6 or qmax < c.mjcf_hi_deg - 1e-6) else "MJCF"
            rows.append(Bound(c.joint, side, row, E3_DEG[row][pick], c.q_neutral_deg, c.flexion_sign,
                              qmin, qmax, c.mjcf_lo_deg, c.mjcf_hi_deg, binding, c.method))
        for suffix, (flex_row, ext_row) in TWO_SIDED.items():
            c = cal[f"{side}_{suffix}"]
            q_flex = c.qpos_for_flexion(E3_DEG[flex_row][pick])
            q_ext = c.qpos_for_flexion(-E3_DEG[ext_row][pick])
            lo, hi = sorted((q_flex, q_ext))
            qmin, qmax = max(lo, c.mjcf_lo_deg), min(hi, c.mjcf_hi_deg)
            binding = "E.3" if (qmin > c.mjcf_lo_deg + 1e-6 or qmax < c.mjcf_hi_deg - 1e-6) else "MJCF"
            rows.append(Bound(c.joint, side, f"{ext_row}+{flex_row}", E3_DEG[flex_row][pick], c.q_neutral_deg,
                              c.flexion_sign, qmin, qmax, c.mjcf_lo_deg, c.mjcf_hi_deg, binding, c.method))
    return rows


def table(rows: list[Bound]) -> pa.Table:
    return pa.table({
        "joint": [r.joint for r in rows],
        "side": [r.side for r in rows],
        "e3_row": [r.e3_row for r in rows],
        "e3_deg": pa.array([r.e3_deg for r in rows], pa.float64()),
        "q_neutral_deg": pa.array([r.q_neutral_deg for r in rows], pa.float64()),
        "flexion_sign": pa.array([r.flexion_sign for r in rows], pa.int8()),
        "qpos_min_deg": pa.array([r.qpos_min_deg for r in rows], pa.float64()),
        "qpos_max_deg": pa.array([r.qpos_max_deg for r in rows], pa.float64()),
        "mjcf_min_deg": pa.array([r.mjcf_min_deg for r in rows], pa.float64()),
        "mjcf_max_deg": pa.array([r.mjcf_max_deg for r in rows], pa.float64()),
        "binding": [r.binding for r in rows],
        "method": [r.method for r in rows],
    })


def check(rows: list[Bound]) -> list[str]:
    """Every defect by name; an empty list is a pass."""
    problems = []
    if len(rows) != 8:
        problems.append(f"{len(rows)} gate joints, expected 8")
    for r in rows:
        lo, hi = (math.degrees(v) for v in MJCF_RANGE_RAD[r.joint])
        if r.qpos_min_deg < lo - 1e-6 or r.qpos_max_deg > hi + 1e-6:
            problems.append(f"{r.joint}: envelope [{r.qpos_min_deg:+.1f}, {r.qpos_max_deg:+.1f}] leaves jnt_range [{lo:+.1f}, {hi:+.1f}]")
        if not (lo - 1e-6 <= r.q_neutral_deg <= hi + 1e-6):
            problems.append(f"{r.joint}: neutral {r.q_neutral_deg:+.1f} outside jnt_range")
        if r.qpos_min_deg >= r.qpos_max_deg:
            problems.append(f"{r.joint}: empty envelope")
        for e3 in r.e3_row.split("+"):
            if e3 not in E3_DEG:
                problems.append(f"{r.joint}: E.3 row '{e3}' is not one of the pinned eight")
        if r.flexion_sign not in (1, -1):
            problems.append(f"{r.joint}: flexion sign {r.flexion_sign}")
        # A flexed pose must lie inside the envelope on the flexion side of neutral.
        probe = r.q_neutral_deg + r.flexion_sign * min(r.e3_deg, 10.0)
        if not (r.qpos_min_deg - 1e-6 <= probe <= r.qpos_max_deg + 1e-6):
            problems.append(f"{r.joint}: 10 deg of flexion ({probe:+.1f}) falls outside the envelope; the sign is wrong")
    return problems


def self_test(rows: list[Bound]) -> None:
    assert not check(rows), check(rows)
    planted = list(rows)
    r = planted[0]
    planted[0] = Bound(r.joint, r.side, r.e3_row, r.e3_deg, r.q_neutral_deg, -r.flexion_sign,
                       r.qpos_min_deg, r.qpos_max_deg, r.mjcf_min_deg, r.mjcf_max_deg, r.binding, r.method)
    if not check(planted):
        raise SystemExit("self-test: the planted swapped sign was accepted; the check is decoration")
    planted = list(rows)
    r = planted[1]
    planted[1] = Bound(r.joint, r.side, r.e3_row, r.e3_deg, r.q_neutral_deg, r.flexion_sign,
                       r.mjcf_min_deg - 30.0, r.qpos_max_deg, r.mjcf_min_deg, r.mjcf_max_deg, r.binding, r.method)
    if not check(planted):
        raise SystemExit("self-test: a bound outside jnt_range was accepted")
    print("self-test: 2 controls refused, the envelope passes")


def refetch() -> None:
    from datasets import load_dataset  # the publish/fetch dependency, not a training one

    ds = load_dataset(DATASET, "human", split="train")
    live = {}
    for row in ds:
        if row["section"] == "E.3" and row["category"] == "range_of_motion":
            live.setdefault(row["subcategory"], {})[row["metric"]] = float(row["value"])
    pinned = {k: {"angle_deg_min": v[0], "angle_deg_max": v[1]} for k, v in E3_DEG.items()}
    if live != pinned:
        raise SystemExit(f"E.3 drifted from the pinned rows:\n live   {live}\n pinned {pinned}")
    print(f"refetch: {len(live)} E.3 rows agree with the pinned eight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    if args.refetch:
        refetch()
        return
    rows = envelope(args.xml)
    if args.self_test:
        self_test(rows)
        return
    problems = check(rows)
    for p in problems:
        print("FAIL", p)
    if problems:
        sys.exit(1)
    if not args.check:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table(rows), args.out, compression="zstd")
        print(f"wrote {args.out} sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]}")
    print(f"{'joint':<26}{'neutral':>9}{'sign':>6}{'envelope (qpos deg)':>26}{'mjcf':>24}  binding")
    for r in rows:
        print(f"{r.joint:<26}{r.q_neutral_deg:>+8.1f}d{('+q' if r.flexion_sign > 0 else '-q'):>6}"
              f"{f'[{r.qpos_min_deg:+.1f}, {r.qpos_max_deg:+.1f}]':>26}"
              f"{f'[{r.mjcf_min_deg:+.1f}, {r.mjcf_max_deg:+.1f}]':>24}  {r.binding}")


if __name__ == "__main__":
    main()
