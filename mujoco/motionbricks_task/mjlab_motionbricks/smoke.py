"""The smoke run (RFD 2238): train a few hundred iterations, roll the policy out with
the ROM clearance logged, run the shifted-envelope negative control, export, and
write the record. A smoke policy is pipeline validation, not a gait; nothing here
is published.

    pixi run smoke --num-envs 1024 --iterations 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from mjlab_motionbricks.gpu import select_gpu_by_name


def rollout(checkpoint: Path, steps: int, shift_deg: float, num_envs: int = 64) -> dict:
    import torch
    import mjlab_motionbricks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab_motionbricks import TASK_FLAT
    from mjlab_motionbricks.tasks import mdp

    mdp.set_envelope_shift(shift_deg)
    env_cfg = load_env_cfg(TASK_FLAT, play=True)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(TASK_FLAT)
    base = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_FLAT) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), None, "cuda:0")
    runner.load(str(checkpoint), map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    obs = env.get_observations()
    clearance_sum, reward_sum, frames = 0.0, 0.0, 0
    gate = env_cfg.metrics["rom_clearance"].params["asset_cfg"]
    with torch.inference_mode():
        for _ in range(steps):
            act = policy(obs)
            obs, rew, _, _ = env.step(act)
            clearance_sum += float(mdp.rom_clearance(base, gate).mean())
            reward_sum += float(rew.mean())
            frames += 1
    mdp.set_envelope_shift(0.0)
    env.close()
    return {"steps": steps, "num_envs": num_envs, "shift_deg": shift_deg,
            "rom_clearance": round(clearance_sum / frames, 4),
            "denominator": f"{frames} frames x {num_envs} envs x 8 gate joints",
            "mean_reward_per_step": round(reward_sum / frames, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--rollout-steps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("policies/smoke"))
    ap.add_argument("--run", type=Path, default=None, help="reuse this run directory instead of training")
    args = ap.parse_args()
    select_gpu_by_name("4090")
    from mjlab_motionbricks import TASK_FLAT

    wall_train = 0.0
    if args.run is None:
        run_name = f"smoke_{int(time.time())}"
        t0 = time.perf_counter()
        cmd = [sys.executable, "-m", "mjlab_motionbricks.train_cli", TASK_FLAT,
               "--env.scene.num-envs", str(args.num_envs), "--agent.max-iterations", str(args.iterations),
               "--agent.run-name", run_name, "--agent.save-interval", str(args.iterations)]
        print("train:", " ".join(cmd))
        train = subprocess.run(cmd, text=True)
        wall_train = time.perf_counter() - t0
        if train.returncode != 0:
            sys.exit(f"FAIL: training exited {train.returncode}")
        runs = sorted(Path("logs/rsl_rl/motionbricks_g1_flat").glob(f"*_{run_name}"))
    else:
        runs = [args.run]
    if not runs:
        sys.exit("FAIL: no run directory written")
    ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        sys.exit("FAIL: no checkpoint written")
    checkpoint = ckpts[-1]
    first = rollout(checkpoint, args.rollout_steps, 0.0)
    control = rollout(checkpoint, min(args.rollout_steps, 200), 30.0)
    if control["rom_clearance"] >= 0.9:
        sys.exit(f"FAIL: the shifted envelope still reads {control['rom_clearance']}; the metric is decoration")
    from mjlab_motionbricks.export import export

    manifest = export(checkpoint, args.out)
    record = {
        "task": TASK_FLAT, "run": str(runs[-1]), "checkpoint": str(checkpoint),
        "num_envs": args.num_envs, "iterations": args.iterations, "train_wall_s": round(wall_train, 1),
        "rollout": first, "shifted_envelope_control": control,
        "obs_dim": manifest["obs_dim"], "action_dim": manifest["action_dim"],
        "onnx": manifest["onnx"], "normalizer_in_graph": manifest["normalizer_in_graph"],
    }
    (args.out / "smoke_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
