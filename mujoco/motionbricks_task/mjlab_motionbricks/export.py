"""Export a checkpoint to ONNX with the observation normaliser baked in, assert the
graph's shape, and write a manifest beside it (the microduck rule: export through
the runner, never hand-convert a checkpoint).

    pixi run export --checkpoint logs/rsl_rl/motionbricks_g1_flat/<run>/model_299.pt --out policies/smoke
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from mjlab_motionbricks.gpu import select_gpu_by_name


def git_sha(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def export(checkpoint: Path, out_dir: Path, filename: str = "policy.onnx") -> dict:
    select_gpu_by_name("4090")
    import onnx
    import mjlab_motionbricks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab_motionbricks import TASK_FLAT
    from mjlab_motionbricks.robot.g1_constants import ACTION_DIM, ENVELOPE_PARQUET, MUJOCO_DIR

    env_cfg = load_env_cfg(TASK_FLAT, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(TASK_FLAT)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0"), clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_FLAT) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), None, "cuda:0")
    runner.load(str(checkpoint), map_location="cuda:0")
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(out_dir), filename)
    onnx_path = out_dir / filename
    metadata = get_base_metadata(env.unwrapped, run_path=str(checkpoint))
    attach_metadata_to_onnx(str(onnx_path), metadata)

    model = onnx.load(str(onnx_path))
    def dims(v):
        return [d.dim_value for d in v.type.tensor_type.shape.dim]
    inputs = {i.name: dims(i) for i in model.graph.input}
    outputs = {o.name: dims(o) for o in model.graph.output}
    obs_dim = int(env.unwrapped.observation_manager.group_obs_dim["actor"][0])
    in_shape = next(iter(inputs.values()))
    out_shape = next(iter(outputs.values()))
    if in_shape != [1, obs_dim] or out_shape != [1, ACTION_DIM]:
        raise SystemExit(f"FAIL: graph is {inputs} -> {outputs}, expected [1, {obs_dim}] -> [1, {ACTION_DIM}]")
    normalizer = any("norm" in n.name.lower() or "Normalization" in n.op_type for n in model.graph.node) or \
        any("norm" in i.name.lower() for i in model.graph.initializer)
    manifest = {
        "task": TASK_FLAT, "checkpoint": str(checkpoint), "onnx": str(onnx_path),
        "obs_dim": obs_dim, "action_dim": ACTION_DIM, "inputs": inputs, "outputs": outputs,
        "normalizer_in_graph": bool(normalizer),
        "versions": {d: md.version(d) for d in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "torch", "onnx")},
        "repo_sha": git_sha(MUJOCO_DIR), "envelope_sha256": hashlib.sha256(ENVELOPE_PARQUET.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "published": False, "note": "a smoke export; no policy ships before the obs_dim, K_max and ROM rulings (RFD 2238)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    env.close()
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("policies/smoke"))
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
