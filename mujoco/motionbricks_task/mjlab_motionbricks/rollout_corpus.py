"""Roll the policy out over command sweeps and seeds and record every frame
(RFD 2238): the body's PPO corpus, microduck's `--save-csv` done as data.

    pixi run rollouts --checkpoint logs/.../model_299.pt --seeds 8 --envs 64 --steps 1000

Seeds 0..5 train, 6 test, 7 evaluation: whole seeds and whole sweeps held out, so
the evaluation split is not the training distribution reshuffled. The run table
records the checkpoint, the task, the measured obs_dim, the versions and the note
that this is generated data sampled from a learned policy (CLAUDE.md's four
conditions), never evaluation material.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from mjlab_motionbricks.gpu import select_gpu_by_name

SWEEPS = {
    # name: (lin_vel_x, lin_vel_y, ang_vel_z) ranges the command is drawn from
    "stand": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    "walk_slow": ((0.3, 0.5), (0.0, 0.0), (0.0, 0.0)),
    "walk": ((0.8, 1.0), (0.0, 0.0), (0.0, 0.0)),
    "run": ((1.5, 2.0), (0.0, 0.0), (0.0, 0.0)),
    "lateral": ((0.0, 0.0), (-0.8, 0.8), (0.0, 0.0)),
    "turn": ((0.0, 0.3), (0.0, 0.0), (-0.7, 0.7)),
    "mixed": ((-1.0, 1.5), (-0.8, 0.8), (-0.7, 0.7)),
}


def split_of(seed: int, seeds: int) -> str:
    if seed == seeds - 1:
        return "evaluation"
    if seed == seeds - 2:
        return "test"
    return "train"


def git_sha(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def roll(checkpoint: Path, out: Path, seed: int, seeds: int, sweep: str, num_envs: int, steps: int) -> dict:
    import torch
    import mjlab_motionbricks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.recorder_manager import RecorderTermCfg
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    from mjlab_motionbricks import TASK_FLAT
    from mjlab_motionbricks.record import RolloutRecorder

    split = split_of(seed, seeds)
    env_cfg = load_env_cfg(TASK_FLAT, play=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    twist = env_cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    lx, ly, az = SWEEPS[sweep]
    twist.ranges.lin_vel_x, twist.ranges.lin_vel_y, twist.ranges.ang_vel_z = lx, ly, az
    twist.resampling_time_range = (1e6, 1e6)  # one command per episode
    env_cfg.recorders = {"rollout": RecorderTermCfg(func=RolloutRecorder, params={
        "out_dir": str(out), "split": split, "seed": seed, "sweep": sweep, "step_dt": env_cfg.sim.mujoco.timestep * env_cfg.decimation})}
    agent_cfg = load_rl_cfg(TASK_FLAT)
    base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_FLAT) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), None, "cuda:0")
    runner.load(str(checkpoint), map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    obs = env.get_observations()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            obs, _, _, _ = env.step(policy(obs))
    wall = time.perf_counter() - t0
    rec = base.recorder_manager.get_term("rollout")
    obs_dim = int(base.observation_manager.group_obs_dim["actor"][0])
    env.close()
    return {"seed": seed, "split": split, "sweep": sweep, "num_envs": num_envs, "steps": steps,
            "frames": rec.frames_written + len(rec.rows), "episodes": len(rec.episode_rows), "wall_s": round(wall, 1),
            "frames_per_s": round(num_envs * steps / wall, 1), "obs_dim": obs_dim}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("rollouts"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--sweeps", default=",".join(SWEEPS))
    args = ap.parse_args()
    select_gpu_by_name("4090")
    from mjlab_motionbricks import TASK_FLAT
    from mjlab_motionbricks.robot.g1_constants import ENVELOPE_PARQUET, MUJOCO_DIR

    runs = []
    for seed in range(args.seeds):
        for sweep in args.sweeps.split(","):
            r = roll(args.checkpoint, args.out, seed, args.seeds, sweep, args.envs, args.steps)
            print(json.dumps(r))
            runs.append(r)
    run = {
        "task": TASK_FLAT, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "obs_dim": runs[0]["obs_dim"], "action_dim": 29, "control_hz": 50,
        "seeds": args.seeds, "splits": {"train": "seeds 0..n-3", "test": "seed n-2", "evaluation": "seed n-1"},
        "sweeps": {k: {"lin_vel_x": v[0], "lin_vel_y": v[1], "ang_vel_z": v[2]} for k, v in SWEEPS.items() if k in args.sweeps.split(",")},
        "versions": {d: md.version(d) for d in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "torch")},
        "repo_sha": git_sha(MUJOCO_DIR), "envelope_sha256": hashlib.sha256(ENVELOPE_PARQUET.read_bytes()).hexdigest(),
        "synthetic_class": "generated", "generating_model": "PPO policy from the task above at the checkpoint above",
        "conditioning": "the twist command per sweep, the seed, mjlab's play-mode randomisation",
        "not_for_evaluation": True, "not_sole_distribution": True,
        "note": "the smoke checkpoint is not a gait; regenerate from a real checkpoint when the rulings land (RFD 2238)",
        "rolls": runs, "frames_total": sum(r["frames"] for r in runs), "episodes_total": sum(r["episodes"] for r in runs),
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    d = args.out / "data" / "run"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in run.items()}]), d / "run.parquet", compression="zstd")
    print(json.dumps({k: v for k, v in run.items() if k != "rolls"}, indent=2))


if __name__ == "__main__":
    main()
