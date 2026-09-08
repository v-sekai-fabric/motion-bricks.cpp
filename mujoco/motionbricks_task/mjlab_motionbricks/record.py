"""The roll-out recorder (RFD 2238): every control step of every environment, as
ETNF parquet shards.

`frames` is the wide table: one row per env per step with the actor observation,
the action, joint positions and velocities, the root pose and body-frame
velocities, projected gravity, the twist command, the reward and each reward
term, the ROM clearance and the reset flags. `episodes` is its satellite: one row
per (env, episode) with the seed, the command sweep and how it ended. The
terminal transition is written from `record_pre_reset`, where the action is
still the one that ended the episode.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from mjlab_motionbricks.robot.g1_constants import GATE_REGEX
from mjlab.managers.scene_entity_config import SceneEntityCfg

FLUSH_ROWS = 200_000


class RolloutRecorder(RecorderTerm):
    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.out = Path(p["out_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.split = p["split"]
        self.seed = int(p["seed"])
        self.sweep = p.get("sweep", "mixed")
        self.step_dt = float(p.get("step_dt", 0.02))
        n = env.num_envs
        self.episode = torch.zeros(n, dtype=torch.int64, device=env.device)
        self.step = torch.zeros(n, dtype=torch.int64, device=env.device)
        self.rows: list[dict] = []
        self.episode_rows: list[dict] = []
        self.part = 0
        self.frames_written = 0
        self._gate = SceneEntityCfg("robot", joint_names=(GATE_REGEX,))

    # ---- the row -----------------------------------------------------------
    def _rows(self, ids: torch.Tensor, terminal: bool) -> list[dict]:
        env = self._env
        from mjlab_motionbricks.tasks import mdp

        robot = env.scene["robot"]
        obs = env.obs_buf["actor"][ids].cpu().numpy()
        act = env.action_manager.action[ids].cpu().numpy()
        jp = robot.data.joint_pos[ids].cpu().numpy()
        jv = robot.data.joint_vel[ids].cpu().numpy()
        pos = robot.data.root_link_pos_w[ids].cpu().numpy()
        quat = robot.data.root_link_quat_w[ids].cpu().numpy()
        lin = robot.data.root_link_lin_vel_b[ids].cpu().numpy()
        ang = robot.data.root_link_ang_vel_b[ids].cpu().numpy()
        grav = robot.data.projected_gravity_b[ids].cpu().numpy()
        cmd = env.command_manager.get_command("twist")[ids].cpu().numpy()
        rew = env.reward_buf[ids].cpu().numpy()
        terms = {}
        rm = env.reward_manager
        for name in rm.active_terms:
            try:
                terms[name] = rm.get_term(name)[ids].cpu().numpy() if hasattr(rm, "get_term") else None
            except Exception:  # noqa: BLE001
                terms[name] = None
        clearance = mdp.rom_clearance(env, self._gate)[ids].cpu().numpy()
        terminated = env.reset_terminated[ids].cpu().numpy() if terminal else [False] * len(ids)
        timed_out = env.reset_time_outs[ids].cpu().numpy() if terminal else [False] * len(ids)
        ep = self.episode[ids].cpu().numpy()
        st = self.step[ids].cpu().numpy()
        out = []
        for i, e in enumerate(ids.cpu().numpy()):
            row = {
                "env": int(e), "episode": int(ep[i]), "step": int(st[i]), "t": float(st[i]) * self.step_dt,
                "obs": obs[i].astype("float32").tolist(), "action": act[i].astype("float32").tolist(),
                "joint_pos": jp[i].astype("float32").tolist(), "joint_vel": jv[i].astype("float32").tolist(),
                "root_pos_w": pos[i].astype("float32").tolist(), "root_quat_w": quat[i].astype("float32").tolist(),
                "root_lin_vel_b": lin[i].astype("float32").tolist(), "root_ang_vel_b": ang[i].astype("float32").tolist(),
                "projected_gravity_b": grav[i].astype("float32").tolist(), "command": cmd[i].astype("float32").tolist(),
                "reward": float(rew[i]), "rom_clearance": float(clearance[i]),
                "terminal": bool(terminal), "terminated": bool(terminated[i]), "timed_out": bool(timed_out[i]),
            }
            for name, vals in terms.items():
                row[f"reward_{name}"] = float(vals[i]) if vals is not None else 0.0
            out.append(row)
        return out

    # ---- the hooks ---------------------------------------------------------
    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        self.step[env_ids] = 0

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.rows += self._rows(env_ids, terminal=True)
        ep = self.episode[env_ids].cpu().numpy()
        st = self.step[env_ids].cpu().numpy()
        term = self._env.reset_terminated[env_ids].cpu().numpy()
        for i, e in enumerate(env_ids.cpu().numpy()):
            self.episode_rows.append({"env": int(e), "episode": int(ep[i]), "seed": self.seed, "sweep": self.sweep,
                                      "length": int(st[i]) + 1, "ended": "terminated" if term[i] else "time_out"})
        self.episode[env_ids] += 1
        self._maybe_flush()

    def record_post_step(self) -> None:
        mask = ~self._env.reset_buf
        ids = torch.nonzero(mask).flatten()
        if len(ids):
            self.rows += self._rows(ids, terminal=False)
        self.step[mask] += 1
        self._maybe_flush()

    # ---- the files ---------------------------------------------------------
    def _maybe_flush(self) -> None:
        if len(self.rows) >= FLUSH_ROWS:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            d = self.out / self.split / "data" / "frames"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(self.rows), d / f"{self.split}-seed{self.seed}-{self.sweep}-{self.part:05d}.parquet", compression="zstd")
            self.frames_written += len(self.rows)
            self.rows = []
            self.part += 1

    def close(self) -> None:
        # open episodes end with the run
        n = self._env.num_envs
        ids = torch.arange(n, device=self._env.device)
        ep = self.episode.cpu().numpy()
        st = self.step.cpu().numpy()
        for e in range(n):
            if st[e] > 0:
                self.episode_rows.append({"env": e, "episode": int(ep[e]), "seed": self.seed, "sweep": self.sweep,
                                          "length": int(st[e]), "ended": "run_end"})
        self.flush()
        if self.episode_rows:
            d = self.out / self.split / "data" / "episodes"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(self.episode_rows), d / f"{self.split}-seed{self.seed}-{self.sweep}.parquet", compression="zstd")
        del ids
