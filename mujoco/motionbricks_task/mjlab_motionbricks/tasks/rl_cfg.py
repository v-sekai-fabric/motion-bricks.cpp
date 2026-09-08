"""mjlab's G1 PPO configuration, logged to TensorBoard under its own experiment name."""
from __future__ import annotations

from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def motionbricks_g1_ppo_runner_cfg():
    cfg = unitree_g1_ppo_runner_cfg()
    cfg.experiment_name = "motionbricks_g1_flat"
    if hasattr(cfg, "logger"):
        cfg.logger = "tensorboard"
    if hasattr(cfg, "upload_model"):
        cfg.upload_model = False
    return cfg
