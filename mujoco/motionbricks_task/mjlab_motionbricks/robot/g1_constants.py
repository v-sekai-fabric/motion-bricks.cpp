"""The rig: mjlab's own Unitree G1 config (menagerie MJCF, BSD-3-Clause), re-exported,
with the 29 actuated joints in MJCF order and the eight-joint ROM gate."""
from __future__ import annotations

from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg  # noqa: F401

ACTUATED_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
ACTION_DIM = len(ACTUATED_JOINTS)

GATE_JOINTS: tuple[str, ...] = (
    "left_knee_joint", "left_elbow_joint", "left_shoulder_pitch_joint", "left_wrist_yaw_joint",
    "right_knee_joint", "right_elbow_joint", "right_shoulder_pitch_joint", "right_wrist_yaw_joint",
)
GATE_REGEX = "^(" + "|".join(GATE_JOINTS) + ")$"

MUJOCO_DIR = Path(__file__).resolve().parents[3]
ENVELOPE_PARQUET = MUJOCO_DIR / "data" / "g1_rom_envelope.parquet"
