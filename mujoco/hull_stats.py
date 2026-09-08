"""Per-bone hull statistics: triangle count, hull count, volume bloat, concavity.

SPDX-License-Identifier: Apache-2.0

Answers two questions the benchmark alone cannot:

* **What drives decomposition cost?**  Triangle count does not -- on the
  calibration bones the correlation between tris and time is *negative*
  (``J_Bip_C_Hips`` is 10x smaller than ``J_Sec_L_TopsUpperLegSide`` and took
  8x longer).  Concavity is the candidate explanation, so it is measured here:
  ``concavity = 1 - mesh_volume / convex_hull_volume``, which is 0 for an
  already-convex shell and approaches 1 for a highly concave one.
* **How much does a hull over-cover its bone?**  ``bloat = sum(hull volumes) /
  source convex hull volume``.  A garment fitted to the true surface sits
  *inside* a bloated hull, so per-bone bloat is the per-bone fidelity budget
  for the dress-on identity gate.

Source submeshes are open shells (they are cut out of a closed body mesh), so
their raw signed volume is unreliable; the convex hull volume is well defined
either way and is used as the reference throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from vrm_coacd import bone_submeshes, load_glb, node_names


def tri_mesh_volume(V: np.ndarray, F: np.ndarray) -> float:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("vrm", type=Path)
    ap.add_argument("--hulls", type=Path, default=Path("hulls_pixiv.npz"))
    ap.add_argument("--out", type=Path, default=Path("hull_stats.json"))
    a = ap.parse_args()

    js, blob = load_glb(a.vrm)
    names = node_names(js)
    subs = bone_submeshes(js, blob)

    data = np.load(a.hulls)
    per_bone: dict[str, list] = {}
    for k in data.files:
        if not k.startswith("hull|") or not k.endswith("|v"):
            continue
        bone = k.split("|")[1]
        idx = k.split("|")[2]
        per_bone.setdefault(bone, []).append(idx)

    rows = []
    for node, (V, F) in subs.items():
        bone = names.get(node, str(node))
        if bone not in per_bone:
            continue
        try:
            ch = ConvexHull(V)
            ch_vol = float(ch.volume)
        except Exception:
            continue
        mesh_vol = tri_mesh_volume(V, F)
        hv_sum, hverts = 0.0, 0
        for idx in per_bone[bone]:
            hv = np.asarray(data[f"hull|{bone}|{idx}|v"], dtype=np.float64)
            hf = np.asarray(data[f"hull|{bone}|{idx}|f"], dtype=np.int64)
            hv_sum += tri_mesh_volume(hv, hf)
            hverts += len(hv)
        rows.append({
            "bone": bone,
            "tris": int(len(F)),
            "verts": int(len(V)),
            "hulls": len(per_bone[bone]),
            "hull_verts": hverts,
            "mesh_vol": round(mesh_vol, 9),
            "ch_vol": round(ch_vol, 9),
            # 0 = already convex, ->1 = highly concave
            "concavity": round(1.0 - min(mesh_vol / ch_vol, 1.0), 4) if ch_vol > 0 else None,
            # >1 = hulls enclose more than the bone's own convex hull
            "bloat_vs_ch": round(hv_sum / ch_vol, 4) if ch_vol > 0 else None,
            "bloat_vs_mesh": round(hv_sum / mesh_vol, 4) if mesh_vol > 0 else None,
        })

    rows.sort(key=lambda r: -r["tris"])
    print(f"{'bone':<34}{'tris':>7}{'hulls':>6}{'hverts':>8}"
          f"{'concav':>8}{'bloat_ch':>10}{'bloat_mesh':>12}")
    for r in rows[:25]:
        print(f"{r['bone']:<34}{r['tris']:>7}{r['hulls']:>6}{r['hull_verts']:>8}"
              f"{(r['concavity'] or 0):>8.3f}{(r['bloat_vs_ch'] or 0):>10.3f}"
              f"{(r['bloat_vs_mesh'] or 0):>12.3f}")
    if len(rows) > 25:
        print(f"... {len(rows)-25} more bones")

    cv = [r["concavity"] for r in rows if r["concavity"] is not None]
    bl = [r["bloat_vs_mesh"] for r in rows if r["bloat_vs_mesh"] is not None]
    print(f"\nbones               : {len(rows)}")
    print(f"total hulls         : {sum(r['hulls'] for r in rows)}")
    print(f"total hull verts    : {sum(r['hull_verts'] for r in rows)}")
    print(f"concavity  min/med/max : {min(cv):.3f} / {np.median(cv):.3f} / {max(cv):.3f}")
    print(f"bloat_mesh min/med/max : {min(bl):.3f} / {np.median(bl):.3f} / {max(bl):.3f}")
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"written             : {a.out}")


if __name__ == "__main__":
    main()
