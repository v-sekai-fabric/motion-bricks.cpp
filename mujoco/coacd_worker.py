"""Run one CoACD configuration on one cached bone submesh, and report metrics.

SPDX-License-Identifier: Apache-2.0

Isolated in a subprocess on purpose: CoACD offers no cancellation, and the
default parameters take minutes on a single dense bone.  Running each config
under an external timeout is the only way to sweep without a pathological
setting stalling the whole run.

Usage:  coacd_worker.py <bone.npz> '<params-json>'
Emits a single JSON object on stdout.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


def hull_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume via the divergence theorem over triangles."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def coverage(points: np.ndarray, parts: list, tol: float = 1e-4) -> float:
    """Fraction of source vertices inside at least one hull.

    A collision proxy that does not contain the mesh is broken, so this is the
    metric that actually matters.  Uses scipy's ConvexHull plane equations --
    a library call, not a hand-written vectorised kernel.
    """
    from scipy.spatial import ConvexHull

    inside = np.zeros(len(points), dtype=bool)
    for hv, _hf in parts:
        hv = np.asarray(hv, dtype=np.float64)
        if len(hv) < 4:
            continue
        try:
            eq = ConvexHull(hv).equations
        except Exception:
            continue
        # points @ n + d <= tol  for every face plane
        inside |= np.all(points @ eq[:, :3].T + eq[:, 3] <= tol, axis=1)
    return float(inside.mean())


def main() -> None:
    bone_path, params_json = sys.argv[1], sys.argv[2]
    params = json.loads(params_json)

    data = np.load(bone_path)
    V = data["V"].astype(np.float64)
    F = data["F"].astype(np.int32)

    import coacd

    coacd.set_log_level("error")

    t0 = time.time()
    parts = coacd.run_coacd(coacd.Mesh(V, F), **params)
    elapsed = time.time() - t0

    hull_vols = [hull_volume(np.asarray(hv), np.asarray(hf)) for hv, hf in parts]
    total_hull_verts = int(sum(len(hv) for hv, _ in parts))
    total_hull_faces = int(sum(len(hf) for _, hf in parts))

    # Reference: the source submesh's OWN convex hull. Well defined even when
    # the submesh is an open shell (bone submeshes are cut pieces, so their own
    # volume is meaningless).
    from scipy.spatial import ConvexHull

    try:
        src_ch_vol = float(ConvexHull(V).volume)
    except Exception:
        src_ch_vol = float("nan")

    out = {
        "ok": True,
        "params": params,
        "n_hulls": len(parts),
        "time_s": round(elapsed, 3),
        "hull_vol_sum": round(sum(hull_vols), 9),
        "src_convex_hull_vol": round(src_ch_vol, 9),
        # <1 means the decomposition is tighter than the source's convex hull,
        # i.e. it captured concavity. Sum ignores overlap, so it is an upper
        # bound; fine for ranking configs.
        "bloat": round(sum(hull_vols) / src_ch_vol, 4) if src_ch_vol > 0 else None,
        "coverage": round(coverage(V, parts), 5),
        "hull_verts": total_hull_verts,
        "hull_faces": total_hull_faces,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
