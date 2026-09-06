"""Load a VRM into flat skinning arrays for batched LBS + silhouette rasterisation.

SPDX-License-Identifier: Apache-2.0

Flattens every skinned primitive in the file into one vertex/index buffer with
GLOBAL joint indices, so a single LBS kernel covers the whole avatar rather than
one dispatch per primitive.  A VRM commonly carries several skins (this sample
has three: Body/83 joints, Face/137, Hair/137) whose ``JOINTS_0`` indices are
*skin-local*; they are remapped through ``skin.joints`` to node indices here.

Reuses the vectorised accessor decoder from ``vrm_coacd`` -- the per-element
Python loop it replaced could not parse this file in 900 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vrm_coacd import accessor as _dense_accessor
from vrm_coacd import load_glb

_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor(js: dict, blob: bytes, idx: int) -> np.ndarray:
    """Accessor decode that also handles the zero-initialised and sparse forms.

    An accessor with no ``bufferView`` is defined by glTF as all zeros, which a
    morph target uses when only a handful of vertices move; the moved ones then
    arrive via ``sparse``.  The dense decoder in ``vrm_coacd`` raises KeyError on
    these, and several of this VRM's 57 face morphs are exactly that shape.
    """
    acc = js["accessors"][idx]
    ncomp = _NCOMP[acc["type"]]
    if "bufferView" in acc:
        out = _dense_accessor(js, blob, idx).astype(np.float32)
    else:
        out = np.zeros((acc["count"], ncomp), dtype=np.float32)

    sparse = acc.get("sparse")
    if sparse:
        out = np.array(out, dtype=np.float32)
        ind = _read_raw(js, blob, sparse["indices"], sparse["count"], 1,
                        sparse["indices"]["componentType"]).reshape(-1)
        val = _read_raw(js, blob, sparse["values"], sparse["count"], ncomp,
                        acc["componentType"])
        out[ind.astype(np.int64)] = val.astype(np.float32)
    return out


def _read_raw(js: dict, blob: bytes, ref: dict, count: int, ncomp: int,
              ctype: int) -> np.ndarray:
    fmt = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}[ctype]
    bv = js["bufferViews"][ref["bufferView"]]
    start = bv.get("byteOffset", 0) + ref.get("byteOffset", 0)
    dt = np.dtype(fmt)
    return np.frombuffer(blob, dtype=dt, count=count * ncomp,
                         offset=start).reshape(count, ncomp)


@dataclass
class SkinnedAvatar:
    positions: np.ndarray      # (V, 3) rest positions
    joints: np.ndarray         # (V, 4) global node indices
    weights: np.ndarray        # (V, 4) normalised
    indices: np.ndarray        # (T, 3)
    inverse_bind: np.ndarray   # (N, 4, 4) per node, identity where unused
    rest_local: np.ndarray     # (N, 4, 4) node local TRS at rest
    parents: np.ndarray        # (N,) parent node index, -1 for roots
    node_names: list[str]
    morph_offsets: np.ndarray  # (M, V, 3) morph deltas, zero outside owning prim
    morph_names: list[str]

    @property
    def n_verts(self) -> int:
        return len(self.positions)

    @property
    def n_tris(self) -> int:
        return len(self.indices)

    @property
    def n_nodes(self) -> int:
        return len(self.parents)


def _trs(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    r = np.array(node.get("rotation", [0, 0, 0, 1]), dtype=np.float64)  # xyzw
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    x, y, z, w = r
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rot * s[None, :]
    m[:3, 3] = t
    return m


def load(path: str | Path, with_morphs: bool = True) -> SkinnedAvatar:
    js, blob = load_glb(Path(path))
    nodes = js["nodes"]
    n_nodes = len(nodes)

    parents = np.full(n_nodes, -1, dtype=np.int32)
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parents[c] = i
    rest_local = np.stack([_trs(n) for n in nodes]).astype(np.float32)
    names = [n.get("name", f"node{i}") for i, n in enumerate(nodes)]

    inverse_bind = np.tile(np.eye(4, dtype=np.float32), (n_nodes, 1, 1))
    for skin in js.get("skins", []):
        if "inverseBindMatrices" not in skin:
            continue
        ibm = accessor(js, blob, skin["inverseBindMatrices"]).astype(np.float32)
        ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)  # glTF is column-major
        for local, node_idx in enumerate(skin["joints"]):
            inverse_bind[node_idx] = ibm[local]

    P, J, W, I = [], [], [], []
    morph_chunks: list[tuple[int, np.ndarray]] = []
    morph_names: list[str] = []
    offset = 0

    for mesh in js["meshes"]:
        targets_meta = mesh.get("extras", {}).get("targetNames", [])
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
                continue
            pos = accessor(js, blob, attrs["POSITION"]).astype(np.float32)
            jl = accessor(js, blob, attrs["JOINTS_0"]).astype(np.int32)
            wt = accessor(js, blob, attrs["WEIGHTS_0"]).astype(np.float32)
            if wt.max() > 1.5:            # normalised integer weights
                wt = wt / np.iinfo(np.uint16).max
            s = wt.sum(axis=1, keepdims=True)
            wt = np.divide(wt, s, out=np.zeros_like(wt), where=s > 0)

            skin_idx = prim.get("skin", js["nodes"][0].get("skin", 0))
            # a primitive inherits its skin from the node that references the mesh
            skin_idx = _skin_for_mesh(js, mesh)
            joint_nodes = np.array(js["skins"][skin_idx]["joints"], dtype=np.int32)
            jg = joint_nodes[np.clip(jl, 0, len(joint_nodes) - 1)]

            idx = accessor(js, blob, prim["indices"]).reshape(-1).astype(np.int32)
            I.append(idx.reshape(-1, 3) + offset)
            P.append(pos)
            J.append(jg)
            W.append(wt)

            if with_morphs:
                for k, tgt in enumerate(prim.get("targets", [])):
                    if "POSITION" not in tgt:
                        continue
                    d = accessor(js, blob, tgt["POSITION"]).astype(np.float32)
                    morph_chunks.append((offset, d))
                    nm = targets_meta[k] if k < len(targets_meta) else f"morph{k}"
                    morph_names.append(nm)
            offset += len(pos)

    positions = np.vstack(P)
    total_v = len(positions)
    morphs = np.zeros((len(morph_chunks), total_v, 3), dtype=np.float32)
    for m, (off, d) in enumerate(morph_chunks):
        morphs[m, off:off + len(d)] = d

    return SkinnedAvatar(
        positions=positions,
        joints=np.vstack(J).astype(np.int32),
        weights=np.vstack(W),
        indices=np.vstack(I).astype(np.int32),
        inverse_bind=inverse_bind,
        rest_local=rest_local,
        parents=parents,
        node_names=names,
        morph_offsets=morphs,
        morph_names=morph_names,
    )


def _skin_for_mesh(js: dict, mesh: dict) -> int:
    mi = js["meshes"].index(mesh)
    for n in js["nodes"]:
        if n.get("mesh") == mi and "skin" in n:
            return n["skin"]
    return 0


def world_matrices(av: SkinnedAvatar, local: np.ndarray | None = None) -> np.ndarray:
    """Compose node-local matrices up the hierarchy. Roots first by construction."""
    lm = av.rest_local if local is None else local
    out = np.empty_like(lm)
    order = np.argsort(_depth(av.parents))
    for i in order:
        p = av.parents[i]
        out[i] = lm[i] if p < 0 else out[p] @ lm[i]
    return out


def _depth(parents: np.ndarray) -> np.ndarray:
    d = np.zeros(len(parents), dtype=np.int32)
    for i in range(len(parents)):
        p, k = parents[i], 0
        while p >= 0:
            k += 1
            p = parents[p]
        d[i] = k
    return d


def skin_matrices(av: SkinnedAvatar, local: np.ndarray | None = None) -> np.ndarray:
    """(N, 4, 4) matrices taking rest-space vertices to posed space."""
    return world_matrices(av, local) @ av.inverse_bind


def skin_cpu(av: SkinnedAvatar, skinmat: np.ndarray,
             morph_w: np.ndarray | None = None) -> np.ndarray:
    """Reference LBS on CPU -- the correctness oracle for the Slang kernel."""
    v = av.positions.astype(np.float64)
    if morph_w is not None and len(av.morph_offsets):
        v = v + np.einsum("m,mvc->vc", morph_w.astype(np.float64),
                          av.morph_offsets.astype(np.float64))
    vh = np.concatenate([v, np.ones((len(v), 1))], axis=1)
    out = np.zeros((len(v), 3), dtype=np.float64)
    for k in range(4):
        w = av.weights[:, k].astype(np.float64)
        m = skinmat[av.joints[:, k]].astype(np.float64)
        out += w[:, None] * np.einsum("vij,vj->vi", m[:, :3, :], vh)
    return out


if __name__ == "__main__":
    import sys
    av = load(sys.argv[1] if len(sys.argv) > 1 else
              r"C:\fabric-starforged\6-datasource\sk-vrm1-constraint-twist-sample"
              r"\Constraint_Twist_Sample\Art\VRM1\VRM1_Constraint_Twist_Sample_01.vrm")
    print(f"verts {av.n_verts}  tris {av.n_tris}  nodes {av.n_nodes}  "
          f"morphs {len(av.morph_offsets)}")
    sm = skin_matrices(av)
    rest = skin_cpu(av, sm)
    print(f"rest-pose skin max |v'-v| = {np.abs(rest - av.positions).max():.3e}"
          "   (should be ~0: skinning the rest pose is the identity)")
    print(f"bbox {rest.min(0).round(3)} .. {rest.max(0).round(3)}")
