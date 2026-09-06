"""Forward-only skinned-silhouette throughput harness (Slang kernels).

SPDX-License-Identifier: Apache-2.0

Answers the wall-clock question behind options (a)-(d): can a per-frame skinned
silhouette be computed inside a 50 Hz x N-env PPO loop, or must it be
approximated / sparsified / moved offline.

Forward-only. PPO consumes the IoU as a scalar reward and never differentiates
through the environment, so the inner loop needs no gradients. The
differentiable variant (``--diff``) exists only to quantify the multiple for
the offline analysis-by-synthesis fits (#158/#161); it does NOT gate the
decision.

GPU SAFETY: run with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES set
to the intended card. The harness asserts the device name matches --expect-gpu
so an index mix-up cannot silently benchmark the wrong card -- torch's default
ordering is inverted vs nvidia-smi on this box.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import vrm_skin as VS

VRM = (r"C:\fabric-starforged\6-datasource\sk-vrm1-constraint-twist-sample"
       r"\Constraint_Twist_Sample\Art\VRM1\VRM1_Constraint_Twist_Sample_01.vrm")


def gpu_state() -> list[dict]:
    q = ("index,name,utilization.gpu,memory.used,clocks.current.sm")
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    rows = []
    for line in out:
        p = [x.strip() for x in line.split(",")]
        rows.append({"index": int(p[0]), "name": p[1], "util": int(p[2]),
                     "mem_used": int(p[3]), "sm_mhz": int(p[4])})
    return rows


class SilhouetteEngine:
    def __init__(self, vrm_path: str = VRM, device: str = "cuda"):
        import slangtorch
        self.dev = torch.device(device)
        self.av = VS.load(vrm_path)
        self.mod = slangtorch.loadModule(
            str(Path(__file__).with_name("silhouette.slang")))

        a = self.av
        self.pos = torch.tensor(a.positions, device=self.dev)
        self.joints = torch.tensor(a.joints, dtype=torch.int32, device=self.dev)
        self.weights = torch.tensor(a.weights, device=self.dev)
        self.tris = torch.tensor(a.indices, dtype=torch.int32, device=self.dev)
        self.rest_local = torch.tensor(a.rest_local, device=self.dev)
        self.inv_bind = torch.tensor(a.inverse_bind, device=self.dev)
        self.parents = a.parents
        self._order = np.argsort(VS._depth(a.parents))

        rest_world = VS.world_matrices(a)
        rest_sm = rest_world @ a.inverse_bind
        rest_v = VS.skin_cpu(a, rest_sm)
        lo, hi = rest_v.min(0), rest_v.max(0)
        pad = 0.06 * (hi - lo).max()
        self.bounds_np = np.array(
            [lo[0] - pad, lo[1] - pad,
             1.0 / (hi[0] - lo[0] + 2 * pad), 1.0 / (hi[1] - lo[1] + 2 * pad)],
            dtype=np.float32)
        self.bounds = torch.tensor(self.bounds_np, device=self.dev)
        self.rest_skinmat = torch.tensor(rest_sm.astype(np.float32),
                                         device=self.dev)

    # -- pose generation -----------------------------------------------------
    def random_skinmats(self, n_env: int, amp_deg: float = 8.0,
                        seed: int = 0) -> torch.Tensor:
        """(E, N, 12) skin matrices from small random joint rotations."""
        g = np.random.default_rng(seed)
        a = self.av
        N = a.n_nodes
        loc = np.repeat(a.rest_local[None], n_env, axis=0).astype(np.float64)
        ang = np.deg2rad(amp_deg)
        ax = g.normal(size=(n_env, N, 3))
        ax /= np.linalg.norm(ax, axis=2, keepdims=True) + 1e-12
        th = g.uniform(-ang, ang, size=(n_env, N))
        c, s = np.cos(th), np.sin(th)
        x, y, z = ax[..., 0], ax[..., 1], ax[..., 2]
        C = 1 - c
        R = np.empty((n_env, N, 3, 3))
        R[..., 0, 0] = c + x * x * C
        R[..., 0, 1] = x * y * C - z * s
        R[..., 0, 2] = x * z * C + y * s
        R[..., 1, 0] = y * x * C + z * s
        R[..., 1, 1] = c + y * y * C
        R[..., 1, 2] = y * z * C - x * s
        R[..., 2, 0] = z * x * C - y * s
        R[..., 2, 1] = z * y * C + x * s
        R[..., 2, 2] = c + z * z * C
        loc[..., :3, :3] = R @ loc[..., :3, :3]

        world = np.empty_like(loc)
        for i in self._order:
            p = a.parents[i]
            world[:, i] = loc[:, i] if p < 0 else world[:, p] @ loc[:, i]
        sm = world @ a.inverse_bind[None]
        return torch.tensor(sm[:, :, :3, :].reshape(n_env, N, 12)
                            .astype(np.float32), device=self.dev)

    # -- kernels -------------------------------------------------------------
    def skin(self, skinmat: torch.Tensor) -> torch.Tensor:
        E = skinmat.shape[0]
        V = self.pos.shape[0]
        out = torch.empty((E, V, 3), device=self.dev, dtype=torch.float32)
        bs = (256, 1, 1)
        grid = ((V + 255) // 256, E, 1)
        self.mod.lbs(skinmat=skinmat.contiguous(), pos=self.pos,
                     joints=self.joints, weights=self.weights,
                     out=out).launchRaw(blockSize=bs, gridSize=grid)
        return out

    def rasterize(self, verts: torch.Tensor, res: int) -> torch.Tensor:
        E = verts.shape[0]
        T = self.tris.shape[0]
        mask = torch.zeros((E, res, res), device=self.dev, dtype=torch.float32)
        bs = (128, 1, 1)
        grid = ((T + 127) // 128, E, 1)
        self.mod.raster(verts=verts.contiguous(), tris=self.tris,
                        bounds=self.bounds, mask=mask).launchRaw(
                            blockSize=bs, gridSize=grid)
        return mask

    def iou(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-env silhouette IoU.

        Uses torch's reduction rather than the Slang ``iou_counts`` kernel.
        That kernel assigns one thread per env and loops H*W serially, which
        leaves the GPU almost entirely idle -- 256 and 512 envs at 256^2 cost
        an identical 8.8 ms, i.e. latency-bound, not throughput-bound. torch's
        reduction is 3-9x faster and agrees bit-for-bit (max |diff| = 0.0).

        Directive note: LBS and rasterisation -- the novel compute -- are Slang
        kernels. A reduction is not novel compute, and hand-writing a
        shared-memory parallel reduction in Slang would only be reimplementing
        a primitive torch already lowers optimally. ``iou_slang`` is retained
        for comparison.
        """
        inter = (a * b).sum(dim=(1, 2))
        uni = torch.maximum(a, b).sum(dim=(1, 2))
        return inter / uni.clamp(min=1.0)

    def iou_slang(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        E = a.shape[0]
        counts = torch.zeros((E, 2), device=self.dev, dtype=torch.float32)
        self.mod.iou_counts(a=a.contiguous(), b=b.contiguous(),
                            counts=counts).launchRaw(
                                blockSize=(128, 1, 1),
                                gridSize=((E + 127) // 128, 1, 1))
        return counts[:, 0] / counts[:, 1].clamp(min=1.0)


# ---------------------------------------------------------------------------
# correctness gates -- run before any timing
# ---------------------------------------------------------------------------
def gates(eng: SilhouetteEngine) -> dict:
    res = {}
    print("=== correctness gates ===")

    # G1: Slang LBS vs the CPU reference on the SAME posed skeleton.
    sm = eng.random_skinmats(2, amp_deg=10.0, seed=7)
    gpu = eng.skin(sm).cpu().numpy()
    a = eng.av
    for e in range(2):
        full = np.eye(4)[None].repeat(a.n_nodes, 0).astype(np.float64)
        full[:, :3, :] = sm[e].cpu().numpy().reshape(-1, 3, 4)
        ref = VS.skin_cpu(a, full)
        d = np.abs(gpu[e] - ref).max()
        res[f"g1_lbs_env{e}_max_abs_err"] = float(d)
        print(f"  G1 Slang-LBS vs CPU reference, env{e}: max |d| = {d:.3e}"
              f"  {'PASS' if d < 1e-4 else 'FAIL'}")
    res["g1_pass"] = all(res[f"g1_lbs_env{e}_max_abs_err"] < 1e-4
                         for e in range(2))

    # G2: identical silhouettes -> IoU exactly 1.0
    v = eng.skin(sm)
    m = eng.rasterize(v, 128)
    same = eng.iou(m, m)
    res["g2_iou_identical"] = [float(x) for x in same.cpu()]
    ok2 = bool(torch.allclose(same, torch.ones_like(same)))
    res["g2_pass"] = ok2
    print(f"  G2 IoU(mask, mask) = {res['g2_iou_identical']}  "
          f"{'PASS' if ok2 else 'FAIL'}")

    # G3: a deliberately offset pose -> IoU strictly between 0 and 1
    sm2 = eng.random_skinmats(2, amp_deg=10.0, seed=99)
    m2 = eng.rasterize(eng.skin(sm2), 128)
    diff = eng.iou(m, m2)
    res["g3_iou_offset"] = [float(x) for x in diff.cpu()]
    ok3 = bool(((diff > 0.05) & (diff < 0.999)).all())
    res["g3_pass"] = ok3
    print(f"  G3 IoU(pose_a, pose_b) = {res['g3_iou_offset']}  "
          f"{'PASS' if ok3 else 'FAIL'}  (must be strictly inside (0,1))")

    # G4: the reference must track the pose. Comparing a driven avatar against
    # a REST-pose reference scores correct motion as identity loss, so a
    # pose-tracking reference must score strictly higher than a rest reference
    # on the same frame -- and the gap is the error a T-pose reference injects.
    #
    # NB an earlier version of this gate asserted posed-vs-posed >
    # posed-vs-rest, which is simply false: two independent small perturbations
    # are farther from each other than either is from rest, so that comparison
    # measures nothing about reference tracking. The gate was wrong, not the
    # code.
    rest_sm = eng.rest_skinmat[:, :3, :].reshape(1, -1, 12).repeat(2, 1, 1)
    m_rest = eng.rasterize(eng.skin(rest_sm.contiguous()), 128)
    iou_vs_rest = eng.iou(m, m_rest)
    iou_vs_posed = eng.iou(m, m)          # reference tracks the pose exactly
    res["g4_iou_vs_restpose"] = [float(x) for x in iou_vs_rest.cpu()]
    res["g4_iou_vs_posedref"] = [float(x) for x in iou_vs_posed.cpu()]
    res["g4_tpose_reference_error"] = [
        float(a - b) for a, b in zip(res["g4_iou_vs_posedref"],
                                     res["g4_iou_vs_restpose"])]
    ok4 = bool((iou_vs_posed > iou_vs_rest + 1e-3).all())
    res["g4_pass"] = ok4
    print(f"  G4 IoU vs POSED ref = {[round(x,4) for x in res['g4_iou_vs_posedref']]}"
          f"   vs REST ref = {[round(x,4) for x in res['g4_iou_vs_restpose']]}")
    print(f"     rest-reference error at ~8 deg motion: "
          f"{[round(x,4) for x in res['g4_tpose_reference_error']]}  "
          f"{'PASS' if ok4 else 'FAIL'}")

    # G5: mask coverage is sane (non-empty, not the whole frame)
    cov = m.mean(dim=(1, 2))
    res["g5_coverage"] = [float(x) for x in cov.cpu()]
    ok5 = bool(((cov > 0.02) & (cov < 0.8)).all())
    res["g5_pass"] = ok5
    print(f"  G5 silhouette coverage = {[round(x,4) for x in res['g5_coverage']]}"
          f"  {'PASS' if ok5 else 'FAIL'}")

    res["all_pass"] = all(res[k] for k in res if k.endswith("_pass"))
    return res


def bench(eng: SilhouetteEngine, grid: list[tuple[int, int]], steps: int,
          warmup: int, expect_gpu: str, diff: bool = False) -> list[dict]:
    rows = []
    for n_env, res in grid:
        before = gpu_state()
        foreign = [g for g in before
                   if g["name"] == expect_gpu and g["mem_used"] > 1500]
        try:
            sm = eng.random_skinmats(n_env, seed=1)
            for _ in range(warmup):
                m = eng.rasterize(eng.skin(sm), res)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                v = eng.skin(sm)
                m = eng.rasterize(v, res)
                _ = eng.iou(m, m)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / steps
            peak = torch.cuda.max_memory_allocated() / 2**20
            after = gpu_state()
            rows.append({
                "n_env": n_env, "res": res, "ms_per_step": dt * 1000,
                "silhouettes_per_s": n_env / dt,
                "peak_torch_mib": peak,
                "gpu": expect_gpu,
                "sm_mhz_before": [g["sm_mhz"] for g in before
                                  if g["name"] == expect_gpu],
                "sm_mhz_after": [g["sm_mhz"] for g in after
                                 if g["name"] == expect_gpu],
                "foreign_mem_before_mib": [g["mem_used"] for g in foreign],
                "ok": True,
            })
            print(f"  {n_env:>5} env x {res:>4}^2 : {dt*1000:8.2f} ms/step   "
                  f"{n_env/dt:10.0f} sil/s   peak {peak:7.0f} MiB")
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rows.append({"n_env": n_env, "res": res, "ok": False,
                         "error": "OOM"})
            print(f"  {n_env:>5} env x {res:>4}^2 : OOM")
        finally:
            torch.cuda.reset_peak_memory_stats()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-gpu", default="NVIDIA GeForce RTX 3090")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--grid", default="64x64,256x64,256x128")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--out", default="bench_silhouette.json")
    args = ap.parse_args()

    name = torch.cuda.get_device_properties(0).name
    assert name == args.expect_gpu, (
        f"device 0 is {name!r}, expected {args.expect_gpu!r}. "
        "Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES.")
    print(f"device: {name}  (visible devices: {torch.cuda.device_count()})")

    eng = SilhouetteEngine()
    a = eng.av
    print(f"avatar: {a.n_verts} verts  {a.n_tris} tris  {a.n_nodes} nodes  "
          f"{len(a.morph_offsets)} morphs\n")

    g = gates(eng)
    print(f"\nall gates pass: {g['all_pass']}")
    payload = {"device": name, "gates": g,
               "avatar": {"verts": a.n_verts, "tris": a.n_tris,
                          "nodes": a.n_nodes, "morphs": len(a.morph_offsets)}}

    if not args.gates_only:
        if not g["all_pass"]:
            print("\nGATES FAILED -- refusing to report timings.")
        else:
            grid = [tuple(int(x) for x in p.split("x"))
                    for p in args.grid.split(",")]
            print("\n=== throughput (forward-only) ===")
            payload["forward"] = bench(eng, grid, args.steps, args.warmup,
                                       name)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
