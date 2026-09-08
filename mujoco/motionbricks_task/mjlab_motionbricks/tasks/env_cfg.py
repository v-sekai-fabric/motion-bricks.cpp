"""The flat velocity task on the G1 at 50 Hz with the ROM terms (RFD 2238).

Everything not named here is mjlab's own flat G1 velocity configuration. The
observation is mjlab's actor group as it stands (joint positions and velocities,
projected gravity, base angular and linear velocity, last action, the velocity
command); its dimension is measured from the built environment and recorded,
never asserted, because the observation contract is still an open ruling.
"""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from mjlab_motionbricks.robot.g1_constants import GATE_REGEX
from mjlab_motionbricks.tasks import mdp

CONTROL_HZ = 50
ROM_PENALTY_WEIGHT = -0.5


def motionbricks_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)
    # microduck's rate: 50 Hz control over a 5 ms physics step.
    cfg.sim.mujoco.timestep = 0.005
    cfg.decimation = 4
    assert abs(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation) - CONTROL_HZ) < 1e-9
    gate = SceneEntityCfg("robot", joint_names=(GATE_REGEX,))
    cfg.rewards["rom_penalty"] = RewardTermCfg(func=mdp.rom_penalty, weight=ROM_PENALTY_WEIGHT, params={"asset_cfg": gate})
    cfg.metrics["rom_clearance"] = MetricsTermCfg(func=mdp.rom_clearance, params={"asset_cfg": gate})
    return cfg
