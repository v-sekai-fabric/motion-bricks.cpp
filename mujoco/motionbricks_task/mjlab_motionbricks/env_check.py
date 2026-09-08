"""Build the task, measure its observation and action dimensions, step it, and
show the ROM terms in the reward and metric dictionaries."""
from __future__ import annotations

import json
import sys
import time

from mjlab_motionbricks.gpu import select_gpu_by_name


def main(num_envs: int = 16, steps: int = 100) -> dict:
    select_gpu_by_name("4090")
    import torch
    import mjlab_motionbricks  # noqa: F401  (registers the task)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_motionbricks import TASK_FLAT
    from mjlab_motionbricks.robot.g1_constants import ACTION_DIM, ACTUATED_JOINTS

    cfg = load_env_cfg(TASK_FLAT)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs, _ = env.reset()
    dims = {k: int(v.shape[-1]) for k, v in obs.items()} if isinstance(obs, dict) else {"obs": int(obs.shape[-1])}
    action_dim = int(env.action_manager.total_action_dim)
    robot = env.scene["robot"]
    names = list(getattr(robot, "joint_names", None) or robot.data.joint_names)
    if tuple(names) != ACTUATED_JOINTS:
        sys.exit(f"FAIL: joint order differs from rom_map's: {names}")
    if action_dim != ACTION_DIM:
        sys.exit(f"FAIL: action dim {action_dim}, expected {ACTION_DIM}")
    t0 = time.perf_counter()
    reward_terms, metric_terms = set(), set()
    for _ in range(steps):
        act = torch.zeros(num_envs, action_dim, device=env.device)
        _, rew, _, _, extras = env.step(act)
        reward_terms |= set(env.reward_manager.active_terms)
        if hasattr(env, "metrics_manager"):
            metric_terms |= set(env.metrics_manager.active_terms)
    wall = time.perf_counter() - t0
    out = {
        "task": TASK_FLAT, "num_envs": num_envs, "steps": steps,
        "obs_dims": dims, "action_dim": action_dim,
        "control_hz": round(1.0 / env.step_dt, 3),
        "reward_terms": sorted(reward_terms), "metric_terms": sorted(metric_terms),
        "env_steps_per_s": round(num_envs * steps / wall, 1),
    }
    print(json.dumps(out, indent=2))
    if "rom_penalty" not in reward_terms:
        sys.exit("FAIL: rom_penalty is not an active reward term")
    if "rom_clearance" not in metric_terms:
        sys.exit("FAIL: rom_clearance is not an active metric term")
    env.close()
    return out


if __name__ == "__main__":
    main()
