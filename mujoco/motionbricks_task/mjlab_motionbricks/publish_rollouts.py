"""Publish a roll-out corpus to the Hub with train, test and evaluation named in the
viewer, the run table as provenance, and the census as the card's numbers.

The token is the `hf_token` field of this desk's `agents/<cn>` row in OpenBao (a
`-no-store` login with the desk client certificate). Refuses a string column that
carries an absolute desk path, and a corpus whose census has not been written.

    pixi run rollouts-publish --hub chibifire/motionbricks-g1-rollouts [--private]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

FORBIDDEN = ("c:/users", "c:\\users", "/home/", "/private/tmp", "/mnt/c/users")
SPLITS = (("train", "train/data"), ("test", "test/data"), ("evaluation", "evaluation/data"))


def hf_token_from_bao() -> str:
    env = dict(os.environ, BAO_ADDR="https://100.124.200.34:8200", BAO_TLS_SERVER_NAME="weftspun-bao.internal",
               BAO_CACERT=os.path.expanduser("~/.magi/ca-bundle.pem"),
               BAO_CLIENT_CERT=os.path.expanduser("~/.magi/v3-leaf-int.pem"),
               BAO_CLIENT_KEY=os.path.expanduser("~/.magi/bao-client-v3.key"))
    bao = os.path.expanduser("~/bin/bao.exe")
    # the login prints a note about not storing the token above the token itself
    tok = subprocess.run([bao, "login", "-method=cert", "-no-store", "-field=token"], env=env,
                         capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[-1].strip()
    env["BAO_TOKEN"] = tok
    hf = subprocess.run([bao, "kv", "get", "-field=hf_token", "agents/magi-16739d.agents.weftspun"], env=env,
                        capture_output=True, text=True, timeout=60).stdout.strip()
    if not hf.startswith("hf_"):
        sys.exit("FAIL: no hf_token in bao agents row")
    return hf


def refuse_if_absolute(root: Path) -> None:
    hits = []
    for p in root.rglob("*.parquet"):
        t = pq.read_table(p)
        for col in t.column_names:
            if str(t.schema.field(col).type) == "string":
                for v in t.column(col).to_pylist()[:2000]:
                    if v and any(m in v.lower() for m in FORBIDDEN):
                        hits.append((p.name, col, v[:80]))
                        break
    if hits:
        sys.exit(f"REFUSED: absolute desk path in a string column: {hits[:5]}")


def readme(run: dict, census: dict, hub: str) -> str:
    configs = "".join(
        f"- config_name: {n}\n" + ("  default: true\n" if n == "frames" else "")
        + "  data_files:\n" + "".join(f"  - split: {sp}\n    path: {d}/{n}/*.parquet\n" for sp, d in SPLITS)
        for n in ("frames", "episodes"))
    configs += "- config_name: run\n  data_files:\n  - split: train\n    path: data/run/*.parquet\n"
    rows = "\n".join(
        f"| {sp} | {c['shards']} | {c['frames']:,} | {c['episodes']:,} | {', '.join(str(s) for s in c['seeds'])} | {c['rom_clearance_mean']} | {c['reward_mean']} | {c['terminal_fraction']} |"
        for sp, c in census["splits"].items())
    return f"""---
license: mit
tags:
- reinforcement-learning
- mujoco
- mjlab
- unitree-g1
- motionbricks
- generated-synthetic
configs:
{configs}---

# {hub.split("/")[-1]}

Policy roll-outs of the G1 flat velocity task ({run["task"]}) at {run["control_hz"]} Hz:
one `frames` row per environment per control step (the actor observation, the action,
joint state, root state, the twist command, the reward and its terms, the ROM
clearance, the reset flags), an `episodes` satellite (seed, sweep, length, how it
ended), and a `run` table naming the checkpoint (sha256 `{run["checkpoint_sha256"][:16]}...`),
the measured observation width ({run["obs_dim"]}) and the versions.

**This is generated synthetic data**: every row was sampled from a learned policy
({run["generating_model"]}), conditioned on {run["conditioning"]}. It is manifested
apart from constructed and real data, it is never the sole distribution for anything
deployed on real inputs, and nothing is evaluated on it. The checkpoint behind this
corpus is the pipeline's smoke run, not a gait; the same command regenerates the
corpus from a real checkpoint.

Three splits by whole seed: `train` (seeds 0 to 5), `test` (seed 6, the gate after
training), `evaluation` (seed 7, never trained or tuned on).

| split | shards | frames | episodes | seeds | ROM clearance | reward/step | terminal fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
{rows}

Nulls: {census["nulls"]}. Source: v-sekai-fabric/motion-bricks.cpp, `mujoco/motionbricks_task`.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("rollouts"))
    ap.add_argument("--census", type=Path, default=Path("bench/rollouts_census.json"))
    ap.add_argument("--hub", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--readme-only", action="store_true")
    args = ap.parse_args()
    if not args.census.is_file():
        sys.exit("FAIL: write the census first (rollouts-census --out bench/rollouts_census.json)")
    census = json.loads(args.census.read_text(encoding="utf-8"))
    run = pq.read_table(args.root / "data" / "run" / "run.parquet").to_pylist()[0]
    refuse_if_absolute(args.root)
    (args.root / "README.md").write_text(readme(run, census, args.hub), encoding="utf-8")
    (args.root / "census.json").write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token_from_bao())
    api.create_repo(args.hub, repo_type="dataset", exist_ok=True, private=args.private)
    if args.readme_only:
        api.upload_file(path_or_fileobj=str(args.root / "README.md"), path_in_repo="README.md",
                        repo_id=args.hub, repo_type="dataset")
    else:
        api.upload_large_folder(folder_path=str(args.root), repo_id=args.hub, repo_type="dataset")
    files = api.list_repo_files(args.hub, repo_type="dataset")
    print(f"published {args.hub}: {len(files)} file(s), {sum(f.endswith('.parquet') for f in files)} parquet(s)")


if __name__ == "__main__":
    main()
