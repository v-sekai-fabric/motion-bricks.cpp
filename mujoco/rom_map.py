"""Explicit Appendix E.3 -> Unitree G1 joint map for MotionBricks #174.

SPDX-License-Identifier: Apache-2.0

This file pins down, before any PPO step burns 4090 hours, exactly which G1
joints the Appendix E.3 civilian range-of-motion bounds apply to, in which
direction, and what the ROM-clearance gate is actually measured over.

Provenance of every number here
-------------------------------
* E.3 bounds: ``chibifire/starforged-std-3001-appendix-e``, config ``human``,
  section ``E.3`` (16 rows = 8 subcategories x ``angle_deg_min`` /
  ``angle_deg_max``), source field ``AAOS Norkin-White civilian orthopedic ROM
  reference``.  Fetched and checked 2026-09-06.  Run ``--refetch`` to re-pull.
* MJCF limits: ``google-deepmind/mujoco_menagerie`` ``unitree_g1/g1.xml``,
  itself derived from ``unitreerobotics/unitree_ros`` ``g1_29dof_rev_1_0.xml``.
  Licence is BSD-3-Clause (Unitree Robotics); scanned for non-commercial and
  share-alike clauses, none present, so it is clear of the CLAUDE.md CC-BY-SA
  row that blocks the microduck 3D models.
* 34-joint skeleton names: ``scripts/convert_to_gguf.py`` ``JOINT_NAMES`` in
  this repository.

READ THIS FIRST: 34 and 29 are different things
-----------------------------------------------
Both numbers are real and the #174 spec uses them interchangeably.  They are
not interchangeable.

* **34** is the MotionBricks *animation skeleton*: 34 joints carrying local
  quaternions, ``local_rotation_xyzw [frames, 34, 4]``.  This is the right
  number for the skintoken / retarget interface (``skeleton_kind=g1_humanoid``)
  and for ``src/motion_rep.hpp``'s ``g1_joint_count = 34U``.
* **29** is the MuJoCo *actuated degrees of freedom* in ``g1.xml`` -- what a
  PPO policy can actually command.  ``docs/IMPLEMENTATION.md`` is explicit that
  "MuJoCo qpos is an adapter/debug output, not the primary animation format".

The 5 skeleton joints with no actuator are ``pelvis_skel`` (the floating base,
driven by a freejoint, not an actuator), ``left_toe_base``, ``right_toe_base``,
``left_hand_roll_skel`` and ``right_hand_roll_skel``.  29 + 5 = 34.

Consequences for the observation/action contract, which currently says
``68 joint (34 x [pos, vel]) + 6 root state + ...`` and ``Action: 34-dim``:

1. A 34-dim action would command the floating base and four unactuated joints.
   The actuated action space is **29**.
2. ``34 x [pos, vel]`` double-counts the root: ``pelvis_skel`` is joint 0 of the
   34, and ``6 root state`` already carries it.
3. So the joint block should be ``29 x [pos, vel] = 58``, not 68, and
   ``obs_dim = 58 + 6 + 48 * 7 = 400``, not 410.
4. ONNX export is then ``1 x 400 -> 1 x 29`` (+ M command on the input side),
   not ``1 x obs_dim -> 1 x 34``.

This is unresolved -- see ``motionbricks-open-questions-174``.  The module
below is written against the 29 actuated DoF because that is what the ROM gate
can physically constrain, and asserts loudly if the caller hands it a different
count.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Appendix E.3, verbatim. percentile is empty and sex is "any" on all 16 rows:
# these are the low/high end of the published NORMAL MAXIMUM excursion for one
# direction of motion, NOT a two-sided travel envelope. See
# `rom-e3-bounds-are-endpoint-ranges`.
# --------------------------------------------------------------------------
E3_DEG: dict[str, tuple[float, float]] = {
    # subcategory:        (angle_deg_min, angle_deg_max)
    "elbow_flexion": (130.0, 155.0),
    "knee_flexion": (130.0, 155.0),
    "shoulder_flexion": (150.0, 180.0),
    "hip_flexion_seated": (100.0, 125.0),
    "neck_flexion": (40.0, 65.0),
    "neck_extension": (45.0, 75.0),
    "wrist_flexion": (55.0, 80.0),
    "wrist_extension": (50.0, 75.0),
}

# --------------------------------------------------------------------------
# g1.xml actuated joint limits, radians as published, in declaration order.
# --------------------------------------------------------------------------
MJCF_RANGE_RAD: dict[str, tuple[float, float]] = {
    "left_hip_pitch_joint": (-2.5307, 2.8798),
    "left_hip_roll_joint": (-0.5236, 2.9671),
    "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798),
    "left_ankle_pitch_joint": (-0.87267, 0.5236),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798),
    "right_hip_roll_joint": (-2.9671, 0.5236),
    "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798),
    "right_ankle_pitch_joint": (-0.87267, 0.5236),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618),
    "waist_roll_joint": (-0.52, 0.52),
    "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}

ACTUATED_DOF = 29
SKELETON_JOINTS = 34


@dataclass(frozen=True)
class RomBound:
    """One E.3 bound, resolved onto concrete G1 joints.

    ``negative`` / ``positive`` are the E.3-derived envelope in degrees about
    the joint's own zero, already sign-corrected for this joint's MJCF axis.
    ``None`` on a side means E.3 says nothing there and only the MJCF limit
    applies -- it does NOT mean zero.
    """

    e3_rows: tuple[str, ...]
    joints: tuple[str, ...]
    negative: float | None
    positive: float | None
    sign_verified: bool = False
    note: str = ""
    applies_to_locomotion: bool = True


def _conservative(row: str) -> float:
    """angle_deg_min -- the least-mobile normal individual. Hard-reject bound."""
    return E3_DEG[row][0]


def _permissive(row: str) -> float:
    """angle_deg_max -- inflection point for the soft ROM-clearance penalty."""
    return E3_DEG[row][1]


def build_bounds(conservative: bool = True) -> list[RomBound]:
    """Resolve the 8 E.3 subcategories onto G1 joints.

    SUPERSEDED for the numeric envelopes -- use ``rom_calibrate.py``.

    The ``negative``/``positive`` values here are in ANATOMICAL degrees about a
    presumed neutral of zero, and the report below compares them directly
    against MJCF ``jnt_range``.  That comparison mixes two coordinate systems
    and is wrong: measured against the actual model, the G1's elbow is straight
    at ``q = +75.8 deg`` and flexes toward NEGATIVE q, and the shoulder pitch
    flexes toward negative q from ``q = +4.7 deg``.  Only the knee flexes the
    way the joint name suggests.

    Corrected E.3 envelopes, in MJCF qpos degrees (conservative bound):

        knee            [ +13.8, +143.8]
        elbow           [ -54.2,  +75.8]
        shoulder_pitch  [-145.3,   +4.7]

    E.3 is the binding constraint on all six, with hardware headroom of
    151.2 / 135.8 / 181.7 deg respectively.  What survives from this function
    is the *structure* -- which E.3 rows map to which joints, which are
    two-sided, which are unmappable, and the gate denominator.  Take the
    numbers from ``rom_calibrate.e3_qpos_envelope()``.

    ``conservative=True`` uses ``angle_deg_min`` for the hard-reject envelope,
    per the operator ruling relayed 2026-09-06.  ``False`` uses
    ``angle_deg_max``, the inflection for the soft penalty.
    """
    pick = _conservative if conservative else _permissive
    return [
        RomBound(
            e3_rows=("knee_flexion",),
            joints=("left_knee_joint", "right_knee_joint"),
            negative=0.0,
            positive=pick("knee_flexion"),
            note="MJCF allows 165 deg; E.3 is the binding constraint.",
        ),
        RomBound(
            e3_rows=("elbow_flexion",),
            joints=("left_elbow_joint", "right_elbow_joint"),
            negative=0.0,
            positive=pick("elbow_flexion"),
            note=(
                "Asymmetric. On the FLEXION side the MJCF caps at 120 deg, tighter "
                "than the 130 deg E.3 bound, so the E.3 flexion term can never "
                "fire -- a policy cannot violate E.3 elbow flexion on this rig. "
                "On the EXTENSION side E.3 does bind: it forbids hyperextension "
                "past 0 deg where the MJCF would allow -60 deg. So the joint "
                "belongs in the gate, but only its extension half is live."
            ),
        ),
        RomBound(
            e3_rows=("shoulder_flexion",),
            joints=("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
            negative=0.0,
            positive=pick("shoulder_flexion"),
            note=(
                "Binds by only 3 deg against the MJCF's 153 deg. Sign convention "
                "matters more here than anywhere else -- verify before trusting."
            ),
        ),
        RomBound(
            e3_rows=("wrist_extension", "wrist_flexion"),
            joints=("left_wrist_yaw_joint", "right_wrist_yaw_joint"),
            negative=-pick("wrist_extension"),
            positive=pick("wrist_flexion"),
            sign_verified=True,
            note=(
                "CORRECTED 2026-09-06: anatomical wrist flexion/extension is "
                "wrist_YAW, not wrist_pitch. Verified on g1_with_hands.xml by "
                "comparing rotation axes: the middle-finger flexion axis is "
                "parallel to wrist_yaw (|dot| = 1.000) and orthogonal to "
                "wrist_pitch (|dot| = 0.000), and that holds at every "
                "wrist_roll value from -90 to +90 deg -- roll is proximal to "
                "both, so it rotates them together and cannot change which "
                "axis is which. The joint NAMED pitch is radial/ulnar "
                "deviation. Mapping E.3 wrist rows onto wrist_pitch would "
                "clamp deviation while leaving real flexion unbounded. "
                "Signs are mirrored: flexion is -q on the left, +q on the "
                "right, so the qpos envelopes are NOT the same on both sides "
                "-- left [-55, +50], right [-50, +55]. Both sit inside the "
                "+/-92.5 deg mechanical range, so E.3 binds on both."
            ),
        ),
        RomBound(
            e3_rows=("neck_extension", "neck_flexion"),
            joints=(),
            negative=-pick("neck_extension"),
            positive=pick("neck_flexion"),
            note=(
                "UNMAPPABLE: the 29-DoF G1 has no neck or head joint at all. The "
                "closest actuated joints are waist pitch/roll, which are torso, not "
                "cervical. These 2 of 8 E.3 rows constrain nothing on this rig."
            ),
        ),
        RomBound(
            e3_rows=("hip_flexion_seated",),
            joints=("left_hip_pitch_joint", "right_hip_pitch_joint"),
            negative=0.0,
            positive=pick("hip_flexion_seated"),
            applies_to_locomotion=False,
            note=(
                "SEATED ONLY. The 100-125 deg figure is measured with trunk "
                "contribution and is not a standing/locomotion hip bound. Omitted "
                "for idle and all locomotion styles per the 2026-09-06 ruling."
            ),
        ),
    ]


def mjcf_deg(joint: str) -> tuple[float, float]:
    lo, hi = MJCF_RANGE_RAD[joint]
    return math.degrees(lo), math.degrees(hi)


def effective_envelope(joint: str, bound: RomBound) -> tuple[float, float]:
    """Intersection of the MJCF mechanical limit and the E.3 envelope.

    Per the 2026-09-06 coverage ruling: usable bound is the MJCF range AND the
    E.3-derived envelope where an entry exists.  Joints with no E.3 entry are
    constrained by the MJCF alone.
    """
    lo, hi = mjcf_deg(joint)
    if bound.negative is not None:
        lo = max(lo, bound.negative)
    if bound.positive is not None:
        hi = min(hi, bound.positive)
    return lo, hi


def gate_joints(locomotion: bool = True) -> list[str]:
    """The denominator for the >= 90% ROM-clearance first-deliverable gate.

    Only joints where E.3 actually binds tighter than the MJCF belong here.  A
    joint whose MJCF limit already dominates can never fail the E.3 test, so
    including it silently inflates the pass rate.
    """
    out: list[str] = []
    for bound in build_bounds(conservative=True):
        if locomotion and not bound.applies_to_locomotion:
            continue
        for joint in bound.joints:
            mj_lo, mj_hi = mjcf_deg(joint)
            eff_lo, eff_hi = effective_envelope(joint, bound)
            if eff_lo > mj_lo + 1e-9 or eff_hi < mj_hi - 1e-9:
                out.append(joint)
    return out


def unmapped_actuated_joints(locomotion: bool = True) -> list[str]:
    """Actuated joints E.3 has no opinion on -- MJCF limits are all they get."""
    covered = {
        j
        for b in build_bounds()
        if not (locomotion and not b.applies_to_locomotion)
        for j in b.joints
    }
    return [j for j in MJCF_RANGE_RAD if j not in covered]


def _report() -> None:
    assert len(MJCF_RANGE_RAD) == ACTUATED_DOF, (
        f"expected {ACTUATED_DOF} actuated joints, got {len(MJCF_RANGE_RAD)}"
    )
    print(f"Unitree G1: {ACTUATED_DOF} actuated DoF; "
          f"MotionBricks skeleton: {SKELETON_JOINTS} joints "
          f"(+pelvis, +2 toe_base, +2 hand_roll, none actuated)\n")

    print(f"{'joint':<30}{'MJCF (deg)':<20}{'E.3 (deg)':<20}"
          f"{'effective':<20}binds")
    print("-" * 98)
    for bound in build_bounds(conservative=True):
        if not bound.joints:
            print(f"{'(' + '+'.join(bound.e3_rows) + ')':<30}"
                  f"{'--':<20}"
                  f"{f'[{bound.negative:g}, {bound.positive:g}]':<20}"
                  f"{'NO SUCH JOINT':<20}never")
            continue
        for joint in bound.joints:
            mj = mjcf_deg(joint)
            eff = effective_envelope(joint, bound)
            e3s = (f"[{bound.negative:g}, {bound.positive:g}]"
                   if bound.negative is not None else "--")
            binds = "E.3" if joint in gate_joints() else "MJCF"
            if not bound.applies_to_locomotion:
                binds = "n/a (seated)"
            print(f"{joint:<30}"
                  f"{f'[{mj[0]:.1f}, {mj[1]:.1f}]':<20}"
                  f"{e3s:<20}"
                  f"{f'[{eff[0]:.1f}, {eff[1]:.1f}]':<20}{binds}")

    gate = gate_joints(locomotion=True)
    print(f"\nGATE DENOMINATOR (locomotion/idle styles): {len(gate)} joints")
    for j in gate:
        print(f"  {j}")
    print(f"\nActuated joints with NO E.3 opinion: "
          f"{len(unmapped_actuated_joints())} of {ACTUATED_DOF}")

    print("\nUnverified assumptions -- resolve before trusting the gate:")
    for bound in build_bounds():
        if bound.joints and not bound.sign_verified:
            print(f"  - {'/'.join(bound.e3_rows)}: sign convention unverified")
    print("  Run verify_signs() against a loaded MuJoCo model to settle these.")


def verify_signs(xml_path: str = "g1.xml") -> dict[str, str]:
    """Confirm each mapped joint's positive direction is anatomical flexion.

    Perturbs one joint at a time by +5 degrees from the neutral pose and
    measures how the distal body moves.  For the hinge pairs (knee, elbow) the
    unambiguous test is limb-chain shortening: flexion brings the distal end
    closer to the proximal anchor.  For shoulder pitch the test is the hand's
    forward displacement in the pelvis frame.

    Returns a dict of joint -> verdict.  Wrist pitch is deliberately NOT
    decided here: distinguishing flexion from extension needs a palm normal,
    which lives in the VRM hand geometry, not the G1 rig.
    """
    import mujoco  # imported lazily; only needed for verification
    import numpy as np

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def body_xpos(name: str) -> "np.ndarray":
        bid = mujoco.mj_id2name  # noqa: F841  (keep symmetry with id lookup)
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert idx >= 0, f"no body {name}"
        return data.xpos[idx].copy()

    # proximal anchor, distal end, joint
    chain_tests = [
        ("left_knee_joint", "pelvis", "left_ankle_roll_link"),
        ("right_knee_joint", "pelvis", "right_ankle_roll_link"),
        ("left_elbow_joint", "left_shoulder_pitch_link", "left_wrist_yaw_link"),
        ("right_elbow_joint", "right_shoulder_pitch_link", "right_wrist_yaw_link"),
    ]

    verdicts: dict[str, str] = {}
    eps = math.radians(5.0)

    for joint, proximal, distal in chain_tests:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        qadr = model.jnt_qposadr[jid]

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        d0 = np.linalg.norm(body_xpos(distal) - body_xpos(proximal))

        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = eps
        mujoco.mj_forward(model, data)
        d_plus = np.linalg.norm(body_xpos(distal) - body_xpos(proximal))

        shortens = d_plus < d0
        verdicts[joint] = (
            "POSITIVE = flexion (chain shortens)" if shortens
            else "POSITIVE = EXTENSION -- envelope sign must be flipped"
        )

    # Shoulder pitch: flexion raises/advances the hand in the pelvis frame.
    for joint, distal in [
        ("left_shoulder_pitch_joint", "left_wrist_yaw_link"),
        ("right_shoulder_pitch_joint", "right_wrist_yaw_link"),
    ]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        qadr = model.jnt_qposadr[jid]

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        p0 = body_xpos(distal) - body_xpos("pelvis")

        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = eps
        mujoco.mj_forward(model, data)
        p1 = body_xpos(distal) - body_xpos("pelvis")

        forward = p1[0] - p0[0]
        verdicts[joint] = (
            f"POSITIVE moves hand {'FORWARD' if forward > 0 else 'BACKWARD'} "
            f"(dx={forward:+.4f} m) -- "
            + ("consistent with flexion" if forward > 0
               else "this is EXTENSION; envelope sign must be flipped")
        )

    for joint in ("left_wrist_pitch_joint", "right_wrist_pitch_joint"):
        verdicts[joint] = (
            "UNDECIDABLE on the G1 alone -- flexion vs extension needs a palm "
            "normal, which lives in the VRM hand geometry"
        )

    return verdicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refetch", action="store_true",
                        help="re-pull E.3 from HuggingFace and diff against E3_DEG")
    args = parser.parse_args()
    if args.refetch:
        raise SystemExit("--refetch not yet wired; see rom_map_refetch.py")
    _report()
