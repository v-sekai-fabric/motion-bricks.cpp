"""Calibrate Appendix E.3 anatomical angles onto Unitree G1 MJCF qpos.

SPDX-License-Identifier: Apache-2.0

Why this file exists
--------------------
``rom_map.py``'s first pass assigned each E.3 bound a ``[negative, positive]``
envelope in *anatomical* degrees and compared it directly against the MJCF
``jnt_range``.  That is wrong, and measurably so:

* **MJCF joint zero is not anatomical neutral.**  The G1's left elbow is
  anatomically straight at ``q = +75.8 deg``, not at zero.  The knee is
  straight at ``q = +13.8 deg``.
* **The flexion direction is not always +q.**  The elbow and the shoulder
  pitch both flex toward *negative* q; the knee flexes toward positive q.

So an E.3 bound of "130 deg of elbow flexion" is not ``q <= 130`` and is not
``q in [0, 130]``.  It is ``q >= 75.8 - 130 = -54.2``.  Comparing raw qpos
limits against anatomical degrees produced two wrong conclusions in the first
pass -- that the MJCF dominated the elbow, and that the shoulder bound was
within 3 deg of the mechanical limit.  Both were artefacts of mixing two
coordinate systems.

Everything below is measured from the model at import time rather than
asserted, so it stays correct if the rig is re-pinned.

Method
------
For a limb hinge (knee, elbow) the anatomical neutral is the ``q`` maximising
the interior angle at the joint, measured between the proximal and distal
joint anchors -- i.e. the straightest the limb gets.  Flexion is whichever
direction reduces that angle.

For shoulder pitch, neutral is the ``q`` minimising the upper arm's elevation
from hanging-straight-down, and flexion is the direction that carries the arm
*forward* (+x in the pelvis frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from rom_map import E3_DEG

DEFAULT_XML = "g1.xml"


@dataclass(frozen=True)
class JointCalibration:
    joint: str
    q_neutral_deg: float
    flexion_sign: int  # +1 if flexion is +q, -1 if flexion is -q
    mjcf_lo_deg: float
    mjcf_hi_deg: float
    method: str

    def flexion_travel_deg(self) -> float:
        """Anatomical flexion the hardware permits, from neutral."""
        if self.flexion_sign > 0:
            return self.mjcf_hi_deg - self.q_neutral_deg
        return self.q_neutral_deg - self.mjcf_lo_deg

    def opposite_travel_deg(self) -> float:
        """Hyperextension the hardware permits, from neutral."""
        if self.flexion_sign > 0:
            return self.q_neutral_deg - self.mjcf_lo_deg
        return self.mjcf_hi_deg - self.q_neutral_deg

    def qpos_for_flexion(self, flex_deg: float) -> float:
        return self.q_neutral_deg + self.flexion_sign * flex_deg


def _anchor(model, data, joint: str) -> np.ndarray:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return data.xanchor[jid].copy()


def _interior_angle(model, data, prox: str, mid: str, dist: str) -> float:
    a, b, c = (_anchor(model, data, j) for j in (prox, mid, dist))
    u, v = a - b, c - b
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def _calibrate_hinge(model, data, joint, prox, dist, samples=2001) -> JointCalibration:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qadr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]

    best_q, best_angle = None, -1.0
    for q in np.linspace(lo, hi, samples):
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        angle = _interior_angle(model, data, prox, joint, dist)
        if angle > best_angle:
            best_q, best_angle = float(q), angle

    # Flexion is the direction that reduces the interior angle. Probe both.
    def angle_at(q: float) -> float:
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        return _interior_angle(model, data, prox, joint, dist)

    # Flexion direction must be decided from the RANGE ENDPOINTS, not from a
    # small step either side of neutral.  Near a smooth maximum both sides fall
    # away at almost the same rate, so a +/-5 deg probe reads local curvature
    # and flips on noise -- it reported the knee and elbow backwards.  The
    # endpoint that reaches the smaller interior angle is the flexed one.
    sign = 1 if angle_at(hi) < angle_at(lo) else -1

    return JointCalibration(
        joint=joint,
        q_neutral_deg=math.degrees(best_q),
        flexion_sign=sign,
        mjcf_lo_deg=math.degrees(lo),
        mjcf_hi_deg=math.degrees(hi),
        method=f"max interior angle {best_angle:.1f} deg via {prox}/{dist}",
    )


def _calibrate_shoulder(model, data, joint, upper_link, shoulder_link,
                        samples=2001) -> JointCalibration:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    qadr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]
    down = np.array([0.0, 0.0, -1.0])

    def state(q: float) -> tuple[float, float]:
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = q
        mujoco.mj_forward(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, shoulder_link)
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, upper_link)
        v = data.xpos[eid] - data.xpos[sid]
        elev = math.degrees(math.acos(
            np.clip(float(v @ down / np.linalg.norm(v)), -1.0, 1.0)))
        return elev, float(v[0])

    best_q, best_elev = None, 1e9
    for q in np.linspace(lo, hi, samples):
        elev, _ = state(q)
        if elev < best_elev:
            best_q, best_elev = float(q), elev

    step = math.radians(10.0)
    _, fwd_plus = state(min(best_q + step, hi))
    _, fwd_minus = state(max(best_q - step, lo))
    sign = 1 if fwd_plus > fwd_minus else -1

    return JointCalibration(
        joint=joint,
        q_neutral_deg=math.degrees(best_q),
        flexion_sign=sign,
        mjcf_lo_deg=math.degrees(lo),
        mjcf_hi_deg=math.degrees(hi),
        method=f"min arm elevation {best_elev:.1f} deg from hanging",
    )


def calibrate_all(xml_path: str = DEFAULT_XML) -> dict[str, JointCalibration]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    out: dict[str, JointCalibration] = {}
    for side in ("left", "right"):
        out[f"{side}_knee_joint"] = _calibrate_hinge(
            model, data, f"{side}_knee_joint",
            f"{side}_hip_yaw_joint", f"{side}_ankle_roll_joint")
        out[f"{side}_elbow_joint"] = _calibrate_hinge(
            model, data, f"{side}_elbow_joint",
            f"{side}_shoulder_yaw_joint", f"{side}_wrist_yaw_joint")
        out[f"{side}_shoulder_pitch_joint"] = _calibrate_shoulder(
            model, data, f"{side}_shoulder_pitch_joint",
            f"{side}_elbow_link", f"{side}_shoulder_pitch_link")
    return out


# E.3 row -> (joint suffix, flexion row, opposing row or None)
E3_TO_JOINT = {
    "knee": ("knee_joint", "knee_flexion", None),
    "elbow": ("elbow_joint", "elbow_flexion", None),
    "shoulder": ("shoulder_pitch_joint", "shoulder_flexion", None),
}


def e3_qpos_envelope(conservative: bool = True) -> dict[str, tuple[float, float]]:
    """E.3 bounds expressed in MJCF qpos degrees, intersected with jnt_range."""
    pick = 0 if conservative else 1
    cal = calibrate_all()
    out: dict[str, tuple[float, float]] = {}
    for side in ("left", "right"):
        for _, (suffix, flex_row, opp_row) in E3_TO_JOINT.items():
            joint = f"{side}_{suffix}"
            c = cal[joint]
            flex_limit = E3_DEG[flex_row][pick]
            q_flex = c.qpos_for_flexion(flex_limit)
            q_neutral = c.q_neutral_deg  # no hyperextension allowed by E.3
            lo, hi = sorted((q_flex, q_neutral))
            out[joint] = (max(lo, c.mjcf_lo_deg), min(hi, c.mjcf_hi_deg))
    return out


def _report() -> None:
    cal = calibrate_all()
    print("Anatomical calibration of the G1 against MJCF qpos")
    print("=" * 92)
    print(f"{'joint':<30}{'q_neutral':>11}{'flex dir':>10}"
          f"{'flex travel':>13}{'hyperext':>11}   method")
    for joint, c in cal.items():
        print(f"{joint:<30}{c.q_neutral_deg:>+10.1f}d"
              f"{('+q' if c.flexion_sign > 0 else '-q'):>10}"
              f"{c.flexion_travel_deg():>12.1f}d"
              f"{c.opposite_travel_deg():>10.1f}d   {c.method}")

    print("\nE.3 conservative bounds mapped into qpos, intersected with jnt_range")
    print("=" * 92)
    env = e3_qpos_envelope(conservative=True)
    print(f"{'joint':<30}{'MJCF range':>22}{'E.3-clipped qpos':>24}   binds?")
    for joint, (lo, hi) in env.items():
        c = cal[joint]
        binds = (lo > c.mjcf_lo_deg + 1e-6) or (hi < c.mjcf_hi_deg - 1e-6)
        print(f"{joint:<30}"
              f"{f'[{c.mjcf_lo_deg:+.1f}, {c.mjcf_hi_deg:+.1f}]':>22}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>24}   "
              f"{'E.3 BINDS' if binds else 'MJCF dominates'}")

    print("\nHardware headroom vs the E.3 civilian bound")
    print("=" * 92)
    for joint, c in cal.items():
        row = {"knee": "knee_flexion", "elbo": "elbow_flexion",
               "shou": "shoulder_flexion"}[joint.split("_")[1][:4]]
        e3 = E3_DEG[row][0]
        print(f"  {joint:<30} hardware {c.flexion_travel_deg():6.1f}d  "
              f"vs E.3 {e3:5.1f}d  -> "
              f"{'E.3 is the binding constraint' if e3 < c.flexion_travel_deg() else 'HARDWARE is the binding constraint'}")


if __name__ == "__main__":
    _report()
