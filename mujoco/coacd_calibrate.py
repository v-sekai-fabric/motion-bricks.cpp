"""Sweep CoACD parameters on representative VRM bones (task #177).

SPDX-License-Identifier: Apache-2.0

CoACD's defaults (threshold=0.05, mcts_iterations=150, mcts_nodes=20) took over
two minutes on the FIRST of 80 bones of the pixiv VRM -- unusable for an avatar,
let alone a corpus.  This finds a setting that is fast enough to run over a
corpus, coarse enough to stay inside the ~2-5 hulls/bone budget, and small
enough to commit to GitHub as plain git objects (Git LFS is banned there).

Each configuration runs in a subprocess under a wall-clock timeout, because
CoACD cannot be cancelled once started.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from vrm_coacd import bone_submeshes, load_glb, node_names

PIXIV = Path(
    r"C:\fabric-starforged\6-datasource\sk-vrm1-constraint-twist-sample"
    r"\Constraint_Twist_Sample\Art\VRM1\VRM1_Constraint_Twist_Sample_01.vrm"
)
PY = Path("./.venv/Scripts/python.exe")
CACHE = Path("bone_cache")


def cache_bones(vrm: Path, wanted: list[str]) -> dict[str, Path]:
    """Extract named bone submeshes once and cache them as .npz."""
    CACHE.mkdir(exist_ok=True)
    js, blob = load_glb(vrm)
    names = node_names(js)
    subs = bone_submeshes(js, blob)
    by_name = {names.get(n, str(n)): (n, V, F) for n, (V, F) in subs.items()}

    out: dict[str, Path] = {}
    for w in wanted:
        if w not in by_name:
            print(f"  !! bone {w!r} not found", file=sys.stderr)
            continue
        _n, V, F = by_name[w]
        p = CACHE / f"{w}.npz"
        if not p.exists():
            np.savez_compressed(p, V=V, F=F)
        out[w] = p
        print(f"  cached {w:<34} {len(V):>6} verts {len(F):>6} tris")
    return out


def run_cfg(bone_npz: Path, params: dict, timeout: float) -> dict:
    try:
        r = subprocess.run(
            [str(PY), "coacd_worker.py", str(bone_npz), json.dumps(params)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"TIMEOUT>{timeout:g}s", "params": params}
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return {"ok": False, "reason": (tail[-1] if tail else "nonzero exit"),
                "params": params}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "reason": "unparseable output", "params": params}


def show(label: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'config':<40}{'hulls':>6}{'time_s':>9}{'cover':>8}"
          f"{'bloat':>8}{'h_verts':>9}")
    for name, r in rows:
        if not r.get("ok"):
            print(f"{name:<40}{'--':>6}{r.get('reason','fail'):>9}")
            continue
        print(f"{name:<40}{r['n_hulls']:>6}{r['time_s']:>9.2f}"
              f"{r['coverage']:>8.4f}{(r['bloat'] or 0):>8.3f}"
              f"{r['hull_verts']:>9}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--stage", default="all")
    args = ap.parse_args()

    bones = ["J_Sec_L_TopsUpperLegSide", "J_Bip_C_Head",
             "J_Bip_C_Hips", "J_Bip_L_Foot"]
    print("caching representative bones:")
    cached = cache_bones(PIXIV, bones)
    heavy = cached.get("J_Sec_L_TopsUpperLegSide")

    # ---- Stage A: threshold, on the pathological bone -------------------
    if args.stage in ("all", "a") and heavy:
        rows = []
        for th in (0.05, 0.1, 0.15, 0.2, 0.3):
            t = time.time()
            r = run_cfg(heavy, {"threshold": th}, args.timeout)
            rows.append((f"threshold={th}", r))
            print(f"  ran threshold={th} in {time.time()-t:.1f}s", flush=True)
        show(f"Stage A: threshold sweep on J_Sec_L_TopsUpperLegSide "
             f"(12906 tris, defaults elsewhere)", rows)

    # ---- Stage B: MCTS + preprocess cost at a fixed threshold ------------
    if args.stage in ("all", "b") and heavy:
        rows = []
        cfgs = [
            ("mcts_iter=150 nodes=20 (default)", {}),
            ("mcts_iter=50  nodes=20", {"mcts_iterations": 50}),
            ("mcts_iter=20  nodes=10", {"mcts_iterations": 20, "mcts_nodes": 10}),
            ("mcts_iter=20 nodes=10 depth=2",
             {"mcts_iterations": 20, "mcts_nodes": 10, "mcts_max_depth": 2}),
            ("+ preprocess_resolution=30",
             {"mcts_iterations": 20, "mcts_nodes": 10, "preprocess_resolution": 30}),
            ("+ resolution=1000",
             {"mcts_iterations": 20, "mcts_nodes": 10, "resolution": 1000}),
        ]
        for name, extra in cfgs:
            p = {"threshold": 0.15, **extra}
            t = time.time()
            r = run_cfg(heavy, p, args.timeout)
            rows.append((name, r))
            print(f"  ran {name} in {time.time()-t:.1f}s", flush=True)
        show("Stage B: search cost at threshold=0.15", rows)

    # ---- Stage C: hull cap + vertex budget (on-disk size) ---------------
    if args.stage in ("all", "c") and heavy:
        rows = []
        base = {"threshold": 0.15, "mcts_iterations": 20, "mcts_nodes": 10}
        cfgs = [
            ("base", {}),
            ("max_convex_hull=8", {"max_convex_hull": 8}),
            ("max_convex_hull=4", {"max_convex_hull": 4}),
            ("max_ch_vertex=64", {"max_convex_hull": 4, "max_ch_vertex": 64}),
            ("max_ch_vertex=32", {"max_convex_hull": 4, "max_ch_vertex": 32}),
            ("decimate+32", {"max_convex_hull": 4, "max_ch_vertex": 32,
                             "decimate": True}),
        ]
        for name, extra in cfgs:
            t = time.time()
            r = run_cfg(heavy, {**base, **extra}, args.timeout)
            rows.append((name, r))
            print(f"  ran {name} in {time.time()-t:.1f}s", flush=True)
        show("Stage C: hull cap + vertex budget at threshold=0.15", rows)

    # ---- Stage D: candidate applied across all four bones ---------------
    if args.stage in ("all", "d"):
        cand = {"threshold": 0.15, "mcts_iterations": 20, "mcts_nodes": 10,
                "max_convex_hull": 4, "max_ch_vertex": 32}
        rows = []
        for name, p in cached.items():
            t = time.time()
            r = run_cfg(p, cand, args.timeout)
            rows.append((name, r))
            print(f"  ran {name} in {time.time()-t:.1f}s", flush=True)
        show(f"Stage D: candidate {cand} across bones", rows)


if __name__ == "__main__":
    main()
