"""Census of a roll-out corpus: what each split holds, enumerated.

Reads every shard (rule 5: a fixed population is counted, not sampled) and prints
rows, episodes, frames per sweep, ROM clearance and reward per split, the terminal
fraction, the null count, and the run table's provenance. Refuses a seed that
appears in two splits and a shard whose obs width differs from the run's obs_dim;
`--self-test` plants both and asserts the refusal.

    pixi run rollouts-census [--root rollouts] [--self-test]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SPLITS = ("train", "test", "evaluation")


def _shards(root: Path, split: str, table: str) -> list[Path]:
    return sorted((root / split / "data" / table).glob("*.parquet"))


def census(root: Path) -> dict:
    run = pq.read_table(root / "data" / "run" / "run.parquet").to_pylist()
    if len(run) != 1:
        raise SystemExit(f"REFUSED: the run table carries {len(run)} row(s), not one")
    run = run[0]
    obs_dim = int(run["obs_dim"])
    seeds: dict[int, set[str]] = defaultdict(set)
    out: dict = {"run": {k: run[k] for k in ("task", "checkpoint_sha256", "obs_dim", "action_dim", "control_hz",
                                                 "synthetic_class", "generating_model", "conditioning")
                         if k in run},
                 "splits": {}}
    nulls = 0
    for split in SPLITS:
        frames = _shards(root, split, "frames")
        episodes = _shards(root, split, "episodes")
        if not frames:
            raise SystemExit(f"REFUSED: split {split} has no frames shard")
        rows = 0
        terminal = 0
        clearance = 0.0
        reward = 0.0
        per_sweep: dict[str, int] = defaultdict(int)
        for f in frames:
            t = pq.read_table(f)
            nulls += sum(c.null_count for c in t.columns)
            width = len(t.column("obs")[0].as_py())
            if width != obs_dim:
                raise SystemExit(f"REFUSED: {f.name} carries obs of width {width}, the run says {obs_dim}")
            sweep = f.name.split("-")[2]
            seed = int(f.name.split("-")[1][4:])
            seeds[seed].add(split)
            rows += t.num_rows
            per_sweep[sweep] += t.num_rows
            terminal += sum(t.column("terminal").to_pylist())
            clearance += sum(t.column("rom_clearance").to_pylist())
            reward += sum(t.column("reward").to_pylist())
        ep_rows = 0
        ended: dict[str, int] = defaultdict(int)
        for f in episodes:
            t = pq.read_table(f)
            nulls += sum(c.null_count for c in t.columns)
            ep_rows += t.num_rows
            for r in t.column("ended").to_pylist():
                ended[r] += 1
        out["splits"][split] = {
            "shards": len(frames), "frames": rows, "episodes": ep_rows,
            "seeds": sorted(s for s, sp in seeds.items() if split in sp),
            "frames_per_sweep": dict(sorted(per_sweep.items())),
            "terminal_fraction": round(terminal / rows, 4) if rows else 0.0,
            "rom_clearance_mean": round(clearance / rows, 4) if rows else 0.0,
            "reward_mean": round(reward / rows, 4) if rows else 0.0,
            "ended": dict(sorted(ended.items())),
        }
    leaks = {s: sorted(sp) for s, sp in seeds.items() if len(sp) > 1}
    if leaks:
        raise SystemExit(f"REFUSED: seed(s) in more than one split: {leaks}")
    out["nulls"] = nulls
    out["seeds_per_split"] = {sp: sorted(s for s, v in seeds.items() if sp in v) for sp in SPLITS}
    return out


def self_test(root: Path) -> None:
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="census_rollouts_"))
    try:
        shutil.copytree(root, tmp / "a")
        src = _shards(tmp / "a", "test", "frames")[0]
        shutil.copy(src, tmp / "a" / "train" / "data" / "frames" / src.name.replace("test-", "train-"))
        try:
            census(tmp / "a")
        except SystemExit as e:
            assert "more than one split" in str(e), e
        else:
            raise SystemExit("FAIL: the planted seed leak passed")
        shutil.copytree(root, tmp / "b")
        run = tmp / "b" / "data" / "run" / "run.parquet"
        t = pq.read_table(run).to_pylist()
        t[0]["obs_dim"] = int(t[0]["obs_dim"]) + 1
        import pyarrow as pa
        pq.write_table(pa.Table.from_pylist(t), run, compression="zstd")
        try:
            census(tmp / "b")
        except SystemExit as e:
            assert "obs of width" in str(e), e
        else:
            raise SystemExit("FAIL: the planted obs width mismatch passed")
        print("self-test: 2 control(s) refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("rollouts"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.self_test:
        self_test(args.root)
        return
    c = census(args.root)
    text = json.dumps(c, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if c["nulls"]:
        sys.exit(f"FAIL: {c['nulls']} null(s)")


if __name__ == "__main__":
    main()
