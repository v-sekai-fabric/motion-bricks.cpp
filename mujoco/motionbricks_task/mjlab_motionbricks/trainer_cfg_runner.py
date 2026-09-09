"""Applies a plan of trainer CALL steps to a fresh task config and reads the term table
back from the config objects, so the teacher's trainer family is scored by the
configuration mjlab would run rather than by the program's own claim.

    python -m mjlab_motionbricks.trainer_cfg_runner --plan plan.json --out table.json
    python -m mjlab_motionbricks.trainer_cfg_runner --serve      # one JSON line in, one out
    python -m mjlab_motionbricks.trainer_cfg_runner --self-test  # controls: unknown term, bad number
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys

from mjlab_motionbricks.tasks.env_cfg import motionbricks_g1_flat_env_cfg

REWARD_PARAM = {
    "track_linear_velocity": "std", "track_angular_velocity": "std", "upright": "std",
    "foot_clearance": "target_height", "foot_swing_height": "target_height", "self_collisions": "force_threshold",
}


def _num(s, what: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: '{s}' is not a number") from None


def _int(s, what: str) -> int:
    v = _num(s, what)
    if v != int(v):
        raise ValueError(f"{what}: '{s}' is not an integer")
    return int(v)


def _bool(s, what: str) -> bool:
    if str(s).upper() in ("TRUE", "1"):
        return True
    if str(s).upper() in ("FALSE", "0"):
        return False
    raise ValueError(f"{what}: '{s}' is not a BOOL")


def _set(obj, field: str, value):
    try:
        setattr(obj, field, value)
        return obj
    except dataclasses.FrozenInstanceError:
        return dataclasses.replace(obj, **{field: value})


def apply(cfg, command: str, args: list[str]) -> None:
    cls, method = command.split(".", 1)
    what = command
    if cls == "Reward":
        term = cfg.rewards.get(method)
        if term is None:
            raise ValueError(f"{what}: no such reward term")
        term.weight = _num(args[0], what + " weight")
        if method in REWARD_PARAM:
            key = REWARD_PARAM[method]
            if key not in term.params:
                raise ValueError(f"{what}: the term has no parameter {key}")
            term.params[key] = _num(args[1], f"{what} {key}")
    elif cls == "Observation":
        terms = cfg.observations["actor"].terms
        term = terms.get(method)
        if term is None:
            raise ValueError(f"{what}: no such observation term")
        if args:
            n = _num(args[0], what + " noise")
            if n < 0:
                raise ValueError(f"{what}: noise {n} is negative")
            # the prototype comes from a fresh config: an earlier step may have zeroed this term's noise
            proto = motionbricks_g1_flat_env_cfg().observations["actor"].terms["joint_pos"].noise
            term.noise = None if n == 0 else dataclasses.replace(proto, n_min=-n, n_max=n)
    elif cls == "Metric":
        if method not in cfg.metrics:
            raise ValueError(f"{what}: no such metric")
    elif cls == "Termination":
        term = cfg.terminations.get(method)
        if term is None:
            raise ValueError(f"{what}: no such termination")
        if method == "fell_over":
            term.params["limit_angle"] = math.radians(_num(args[0], what + " limit_deg"))
    elif cls == "Action":
        if method != "joint_pos":
            raise ValueError(f"{what}: no such action")
        act = cfg.actions["joint_pos"]
        act.scale = _num(args[0], what + " scale")
        act.use_default_offset = _bool(args[1], what + " use_default_offset")
    elif cls == "Command":
        if method != "twist":
            raise ValueError(f"{what}: no such command")
        vals = [_num(a, what) for a in args]
        r = cfg.commands["twist"].ranges
        for i, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_z")):
            lo, hi = vals[2 * i], vals[2 * i + 1]
            if lo > hi:
                raise ValueError(f"{what}: {name} range [{lo}, {hi}] is inverted")
            r = _set(r, name, (lo, hi))
        cfg.commands["twist"].ranges = r
    elif cls == "Sim":
        if method == "control":
            ts = _num(args[0], what + " timestep")
            dec = _int(args[1], what + " decimation")
            if ts <= 0 or dec <= 0:
                raise ValueError(f"{what}: timestep and decimation must be positive")
            cfg.sim.mujoco.timestep = ts
            cfg.decimation = dec
        elif method == "episode":
            cfg.episode_length_s = _num(args[0], what + " length_s")
        else:
            raise ValueError(f"{what}: no such sim setting")
    elif cls == "Scene":
        if method != "num_envs":
            raise ValueError(f"{what}: no such scene setting")
        cfg.scene.num_envs = _int(args[0], what)
    elif cls == "Event":
        ev = cfg.events.get(method)
        if ev is None:
            raise ValueError(f"{what}: no such event")
        vals = [_num(a, what) for a in args]
        if method == "foot_friction":
            ev.params["ranges"] = (vals[0], vals[1])
        elif method == "encoder_bias":
            ev.params["bias_range"] = (vals[0], vals[1])
        elif method == "base_com":
            ev.params["ranges"] = {0: (-vals[0], vals[0]), 1: (-vals[1], vals[1]), 2: (-vals[2], vals[2])}
        elif method == "push_robot":
            ev.interval_range_s = (vals[0], vals[1])
            ev.params["velocity_range"]["x"] = (-vals[2], vals[2])
            ev.params["velocity_range"]["y"] = (-vals[2], vals[2])
        else:
            raise ValueError(f"{what}: the runner does not set this event")
    elif cls == "Actuator":
        if method != "pd":
            raise ValueError(f"{what}: no actuator class for this model in the environment")
        vals = [_num(a, what) for a in args]
        art = cfg.scene.entities["robot"].articulation
        acts = tuple(dataclasses.replace(a, stiffness=vals[0], damping=vals[1], effort_limit=vals[2]) for a in art.actuators)
        cfg.scene.entities["robot"].articulation = _set(art, "actuators", acts)
    else:
        raise ValueError(f"{what}: no such surface")


def table(cfg) -> dict:
    rewards = {}
    for name, term in cfg.rewards.items():
        entry = {"weight": float(term.weight)}
        for k, v in term.params.items():
            if k in ("std", "target_height", "force_threshold") and isinstance(v, (int, float)):
                entry[k] = float(v)
        rewards[name] = entry
    obs = {}
    for name, term in cfg.observations["actor"].terms.items():
        obs[name] = {"noise": float(term.noise.n_max) if term.noise is not None else 0.0}
    terms = {}
    for name, term in cfg.terminations.items():
        entry = {}
        if "limit_angle" in term.params:
            entry["limit_deg"] = math.degrees(term.params["limit_angle"])
        terms[name] = entry
    act = cfg.actions["joint_pos"]
    r = cfg.commands["twist"].ranges
    events = {}
    for name, ev in cfg.events.items():
        p = ev.params
        if name == "foot_friction" and "ranges" in p:
            events[name] = {"min": p["ranges"][0], "max": p["ranges"][1]}
        elif name == "encoder_bias" and "bias_range" in p:
            events[name] = {"min": p["bias_range"][0], "max": p["bias_range"][1]}
        elif name == "base_com" and "ranges" in p:
            events[name] = {"x": p["ranges"][0][1], "y": p["ranges"][1][1], "z": p["ranges"][2][1]}
        elif name == "push_robot":
            events[name] = {"interval_min_s": ev.interval_range_s[0], "interval_max_s": ev.interval_range_s[1],
                            "vel": p["velocity_range"]["x"][1]}
        else:
            events[name] = {}
    acts = cfg.scene.entities["robot"].articulation.actuators
    return {
        "rewards": rewards,
        "observations": obs,
        "metrics": sorted(cfg.metrics.keys()),
        "terminations": terms,
        "actions": {"joint_pos": {"scale": act.scale if isinstance(act.scale, (int, float)) else "per-joint",
                                  "use_default_offset": bool(act.use_default_offset)}},
        "commands": {"twist": {"lin_vel_x": list(r.lin_vel_x), "lin_vel_y": list(r.lin_vel_y), "ang_vel_z": list(r.ang_vel_z)}},
        "sim": {"timestep": cfg.sim.mujoco.timestep, "decimation": cfg.decimation,
                "control_hz": 1.0 / (cfg.sim.mujoco.timestep * cfg.decimation), "episode_length_s": cfg.episode_length_s},
        "scene": {"num_envs": cfg.scene.num_envs},
        "events": events,
        "actuators": {"pd": {"stiffness": [a.stiffness for a in acts], "damping": [a.damping for a in acts],
                             "effort_limit": [a.effort_limit for a in acts]}},
    }


def perform(steps: list[dict]) -> dict:
    cfg = motionbricks_g1_flat_env_cfg()
    for step in steps:
        if step.get("kind") != "call":
            raise ValueError(f"step {step.get('index')}: kind {step.get('kind')} is not a call")
        apply(cfg, step["command"], list(step.get("args", [])))
    return table(cfg)


def handle(req: dict) -> dict:
    try:
        return {"ok": True, "table": perform(req["steps"])}
    except Exception as e:  # noqa: BLE001 - the refusal is the result
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def self_test() -> None:
    t = perform([{"kind": "call", "command": "Reward.track_linear_velocity", "args": ["2.5", "0.5"]}])
    assert t["rewards"]["track_linear_velocity"] == {"weight": 2.5, "std": 0.5}, t["rewards"]["track_linear_velocity"]
    r = handle({"steps": [{"kind": "call", "command": "Reward.jump_height", "args": ["1.0"]}]})
    assert not r["ok"] and "no such reward" in r["error"], r
    r = handle({"steps": [{"kind": "call", "command": "Reward.upright", "args": ["fast", "0.4"]}]})
    assert not r["ok"] and "not a number" in r["error"], r
    r = handle({"steps": [{"kind": "call", "command": "Actuator.bam", "args": ["6", "0.1", "20", "0", "0.1"]}]})
    assert not r["ok"], r
    t = perform([{"kind": "call", "command": "Actuator.pd", "args": ["30", "2", "60"]}])
    assert t["actuators"]["pd"]["stiffness"] == [30.0] * 6, t["actuators"]
    print("self-test: 1 identity, 3 control(s) refused")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan")
    ap.add_argument("--out")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.serve:
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            print(json.dumps(handle(json.loads(line))), flush=True)
        return
    plan = json.load(open(args.plan, encoding="utf-8"))
    res = handle(plan)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f)
    print("ok" if res["ok"] else res["error"])
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
