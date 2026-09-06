"""VRM -> per-bone convex hulls via CoACD, for MuJoCo collision (task #177).

SPDX-License-Identifier: Apache-2.0

MuJoCo's rigid pipeline collides convex geoms and auto-convexifies meshes; a
skinned VRM mesh deforms every frame and cannot be a rigid collision geom.
So the collision proxy is built offline: assign each triangle to the bone that
dominates its skin weights, convex-decompose each bone's submesh with CoACD,
and attach the resulting hulls rigidly to that bone.

CoACD is pinned in the manifest at ``3-interactor/coacd-upstream``
(``1401ce2a7ae1ed89c65ab958b48d489350c233c7``, release 1.0.14, MIT).  The
``coacd`` PyPI wheel used here reports the same 1.0.14.
"""

from __future__ import annotations

import argparse
import json
import struct
import time

from pathlib import Path

import numpy as np

COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    _, _, _ = struct.unpack("<III", raw[:12])
    off, js, bin_chunk = 12, None, b""
    while off < len(raw):
        clen, ctype = struct.unpack("<II", raw[off : off + 8])
        chunk = raw[off + 8 : off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            bin_chunk = chunk
        off += 8 + clen
        off += (4 - off % 4) % 4 if off % 4 else 0
    assert js is not None, "no JSON chunk"
    return js, bin_chunk


def accessor(js: dict, blob: bytes, idx: int) -> np.ndarray:
    """Decode a glTF accessor.

    Vectorised deliberately: the obvious per-element ``frombuffer`` loop is
    O(count) in Python and did not finish parsing a 10 MB VRM in 900 s.  The
    tightly-packed case is a single reshape; the interleaved case uses one
    strided view.  Both are O(1) in Python-level work.
    """
    acc = js["accessors"][idx]
    n, ctype = acc["count"], acc["componentType"]
    ncomp = NCOMP[acc["type"]]
    bv = js["bufferViews"][acc["bufferView"]]
    start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    dt = np.dtype(COMPONENT[ctype])
    packed = dt.itemsize * ncomp
    stride = bv.get("byteStride") or packed

    if stride == packed:
        return np.frombuffer(blob, dtype=dt, count=n * ncomp,
                             offset=start).reshape(n, ncomp)

    raw = np.frombuffer(blob, dtype=np.uint8, offset=start,
                        count=(n - 1) * stride + packed)
    view = np.lib.stride_tricks.as_strided(
        raw, shape=(n, packed), strides=(stride, 1), writeable=False)
    return np.ascontiguousarray(view).view(dt).reshape(n, ncomp)


def bone_submeshes(js: dict, blob: bytes) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group triangles by weight-dominant joint. Returns {node_index: (V, F)}."""
    groups: dict[int, list] = {}
    for mesh in js["meshes"]:
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                continue
            pos = accessor(js, blob, attrs["POSITION"]).astype(np.float64)
            joints = accessor(js, blob, attrs["JOINTS_0"]).astype(np.int32)
            weights = accessor(js, blob, attrs["WEIGHTS_0"]).astype(np.float64)
            if weights.dtype != np.float64 or weights.max() > 1.5:
                weights = weights / max(weights.max(), 1.0)
            dom = joints[np.arange(len(joints)), weights.argmax(axis=1)]

            idx = accessor(js, blob, prim["indices"]).reshape(-1).astype(np.int64)
            tris = idx.reshape(-1, 3)
            # A triangle belongs to the bone dominating the majority of its
            # vertices. Vectorised majority-of-3: if any two agree take that
            # value, else fall back to the first vertex's bone. (A per-triangle
            # Counter loop is another O(ntris) Python bottleneck.)
            a, b, c = dom[tris[:, 0]], dom[tris[:, 1]], dom[tris[:, 2]]
            tri_bone = np.where(a == b, a, np.where(a == c, a,
                                np.where(b == c, b, a))).astype(np.int32)
            skin = js["skins"][prim.get("skin", 0)] if "skins" in js else None
            joint_nodes = skin["joints"] if skin else list(range(int(dom.max()) + 1))

            for b in np.unique(tri_bone):
                sel = tris[tri_bone == b]
                used, remap = np.unique(sel, return_inverse=True)
                node = joint_nodes[b] if b < len(joint_nodes) else int(b)
                groups.setdefault(node, []).append(
                    (pos[used], remap.reshape(-1, 3))
                )

    merged: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for node, parts in groups.items():
        vs, fs, off = [], [], 0
        for v, f in parts:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        merged[node] = (np.vstack(vs), np.vstack(fs))
    return merged


def node_names(js: dict) -> dict[int, str]:
    return {i: n.get("name", f"node{i}") for i, n in enumerate(js.get("nodes", []))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vrm", type=Path)
    # Defaults below are the CALIBRATED values, not CoACD's own defaults.
    # See coacd_calibrate.py: stock settings (threshold=0.05, mcts_iterations=150,
    # mcts_nodes=20) took 94 s on ONE bone of 80. MCTS search cost dominates --
    # dropping iterations 150->20 is a 4x speedup at identical hull count and
    # coverage. threshold=0.4 keeps the mean inside the 2-5 hulls/bone budget.
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="CoACD concavity threshold; higher = fewer, coarser hulls")
    ap.add_argument("--mcts-iterations", type=int, default=20)
    ap.add_argument("--mcts-nodes", type=int, default=10)
    ap.add_argument("--preprocess-resolution", type=int, default=30)
    ap.add_argument("--min-tris", type=int, default=12,
                    help="skip bones with fewer triangles than this")
    ap.add_argument("--out", type=Path, default=Path("hulls.npz"))
    args = ap.parse_args()

    import coacd
    coacd.set_log_level("error")
    cparams = dict(threshold=args.threshold,
                   mcts_iterations=args.mcts_iterations,
                   mcts_nodes=args.mcts_nodes,
                   preprocess_resolution=args.preprocess_resolution)
    print(f"CoACD params: {cparams}", flush=True)

    js, blob = load_glb(args.vrm)
    names = node_names(js)
    subs = bone_submeshes(js, blob)
    print(f"skinned bones with geometry: {len(subs)}", flush=True)

    total_hulls, total_tris_in, rows, store = 0, 0, [], {}
    for node, (V, F) in sorted(subs.items()):
        if len(F) < args.min_tris:
            continue
        total_tris_in += len(F)
        _t0 = time.time()
        parts = coacd.run_coacd(coacd.Mesh(V, F), **cparams)
        _dt = time.time() - _t0
        total_hulls += len(parts)
        bname = names.get(node, str(node))
        print(f"  [{len(rows)+1}] {bname} tris={len(F)} hulls={len(parts)} "
              f"t={_dt:.2f}s", flush=True)
        rows.append((bname, len(V), len(F), len(parts), _dt))
        for k, (hv, hf) in enumerate(parts):
            # key carries the BONE NAME so downstream MJCF assembly can map
            # VRM bone -> G1 body. "|" is the separator because bone names
            # themselves contain underscores (J_Sec_L_TopsUpperLegSide).
            bname = names.get(node, str(node))
            store[f"hull|{bname}|{k}|v"] = np.asarray(hv, dtype=np.float32)
            store[f"hull|{bname}|{k}|f"] = np.asarray(hf, dtype=np.int32)

    rows.sort(key=lambda r: -r[3])
    print(f"\n{'bone':<34}{'verts':>8}{'tris':>8}{'hulls':>7}")
    for name, nv, nf, nh in rows[:20]:
        print(f"{name:<34}{nv:>8}{nf:>8}{nh:>7}")
    if len(rows) > 20:
        print(f"... {len(rows)-20} more bones")

    print(f"\nbones decomposed : {len(rows)}")
    print(f"input triangles  : {total_tris_in}")
    print(f"TOTAL HULLS      : {total_hulls}")
    print(f"mean hulls/bone  : {total_hulls/max(len(rows),1):.1f}")
    _tt = sum(r[4] for r in rows)
    print(f"decomposition    : {_tt:.1f}s total, {_tt/max(len(rows),1):.2f}s/bone mean, "
          f"{max(r[4] for r in rows):.1f}s worst")
    np.savez_compressed(args.out, **store)
    print(f"written          : {args.out}")


if __name__ == "__main__":
    main()
