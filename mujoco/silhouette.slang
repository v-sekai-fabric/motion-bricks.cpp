// Batched LBS skinning + binary silhouette rasterisation.
// SPDX-License-Identifier: Apache-2.0
//
// Per the operator directive, vectorised compute goes through Shader Slang
// compiled to a backend rather than ad-hoc numpy/CUDA. Built with slangtorch,
// which lowers these to CUDA and binds them to torch tensors.
//
// Forward-only by design: PPO consumes the silhouette IoU as a scalar reward
// and never differentiates through the environment, so no gradients are needed
// in the inner loop. A differentiable variant is a separate concern for the
// offline analysis-by-synthesis fits (#158/#161).

// ---------------------------------------------------------------------------
// Linear blend skinning.
//   skinmat [E, N, 12]  row-major 3x4 per node (world @ inverseBind)
//   pos     [V, 3]      rest positions, morph deltas already applied
//   joints  [V, 4]      global node indices
//   weights [V, 4]      normalised
//   out     [E, V, 3]
// One thread per (vertex, env). Fused: never materialises a per-vertex matrix,
// which at 4096 envs x 24951 verts would be ~6.5 GB of intermediates.
// ---------------------------------------------------------------------------
[AutoPyBindCUDA]
[CUDAKernel]
void lbs(
    TensorView<float> skinmat,
    TensorView<float> pos,
    TensorView<int> joints,
    TensorView<float> weights,
    TensorView<float> out)
{
    uint3 tid = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint v = tid.x;
    uint e = tid.y;
    if (v >= pos.size(0) || e >= out.size(0)) return;

    float px = pos[uint2(v, 0)];
    float py = pos[uint2(v, 1)];
    float pz = pos[uint2(v, 2)];

    float ox = 0.0f, oy = 0.0f, oz = 0.0f;

    [ForceUnroll]
    for (int k = 0; k < 4; k++)
    {
        float w = weights[uint2(v, uint(k))];
        if (w == 0.0f) continue;
        uint j = uint(joints[uint2(v, uint(k))]);

        float m00 = skinmat[uint3(e, j, 0)];
        float m01 = skinmat[uint3(e, j, 1)];
        float m02 = skinmat[uint3(e, j, 2)];
        float m03 = skinmat[uint3(e, j, 3)];
        float m10 = skinmat[uint3(e, j, 4)];
        float m11 = skinmat[uint3(e, j, 5)];
        float m12 = skinmat[uint3(e, j, 6)];
        float m13 = skinmat[uint3(e, j, 7)];
        float m20 = skinmat[uint3(e, j, 8)];
        float m21 = skinmat[uint3(e, j, 9)];
        float m22 = skinmat[uint3(e, j, 10)];
        float m23 = skinmat[uint3(e, j, 11)];

        ox += w * (m00 * px + m01 * py + m02 * pz + m03);
        oy += w * (m10 * px + m11 * py + m12 * pz + m13);
        oz += w * (m20 * px + m21 * py + m22 * pz + m23);
    }

    out[uint3(e, v, 0)] = ox;
    out[uint3(e, v, 1)] = oy;
    out[uint3(e, v, 2)] = oz;
}

// ---------------------------------------------------------------------------
// Binary silhouette rasteriser, orthographic.
//   verts  [E, V, 3]
//   tris   [T, 3]
//   bounds [4]          minx, miny, invSpanX, invSpanY
//   mask   [E, H, W]    written 1.0 where covered
// One thread per (triangle, env). No atomics: every writer stores the same
// value, so the race is benign -- a binary mask does not need InterlockedOr.
// ---------------------------------------------------------------------------
[AutoPyBindCUDA]
[CUDAKernel]
void raster(
    TensorView<float> verts,
    TensorView<int> tris,
    TensorView<float> bounds,
    TensorView<float> mask)
{
    uint3 tid = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    uint t = tid.x;
    uint e = tid.y;
    if (t >= tris.size(0) || e >= mask.size(0)) return;

    uint H = mask.size(1);
    uint W = mask.size(2);

    float minx = bounds[0];
    float miny = bounds[1];
    float isx = bounds[2];
    float isy = bounds[3];

    uint i0 = uint(tris[uint2(t, 0)]);
    uint i1 = uint(tris[uint2(t, 1)]);
    uint i2 = uint(tris[uint2(t, 2)]);

    // Project orthographically into pixel space (x right, y up -> row flip).
    float ax = (verts[uint3(e, i0, 0)] - minx) * isx * float(W);
    float ay = (verts[uint3(e, i0, 1)] - miny) * isy * float(H);
    float bx = (verts[uint3(e, i1, 0)] - minx) * isx * float(W);
    float by = (verts[uint3(e, i1, 1)] - miny) * isy * float(H);
    float cx = (verts[uint3(e, i2, 0)] - minx) * isx * float(W);
    float cy = (verts[uint3(e, i2, 1)] - miny) * isy * float(H);

    float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (abs(area) < 1e-12f) return;
    float inv = 1.0f / area;

    int lox = int(floor(min(min(ax, bx), cx)));
    int hix = int(ceil(max(max(ax, bx), cx)));
    int loy = int(floor(min(min(ay, by), cy)));
    int hiy = int(ceil(max(max(ay, by), cy)));

    lox = max(lox, 0);
    loy = max(loy, 0);
    hix = min(hix, int(W) - 1);
    hiy = min(hiy, int(H) - 1);

    for (int py = loy; py <= hiy; py++)
    {
        for (int px = lox; px <= hix; px++)
        {
            float fx = float(px) + 0.5f;
            float fy = float(py) + 0.5f;
            float w0 = ((bx - ax) * (fy - ay) - (by - ay) * (fx - ax)) * inv;
            float w1 = ((cx - bx) * (fy - by) - (cy - by) * (fx - bx)) * inv;
            float w2 = ((ax - cx) * (fy - cy) - (ay - cy) * (fx - cx)) * inv;
            if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
            {
                uint row = H - 1u - uint(py);   // flip so +y is up
                mask[uint3(e, row, uint(px))] = 1.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-env IoU between two binary masks. counts [E, 2] = (intersection, union).
// ---------------------------------------------------------------------------
[AutoPyBindCUDA]
[CUDAKernel]
void iou_counts(
    TensorView<float> a,
    TensorView<float> b,
    TensorView<float> counts)
{
    uint e = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (e >= a.size(0)) return;

    uint H = a.size(1);
    uint W = a.size(2);
    float inter = 0.0f;
    float uni = 0.0f;
    for (uint r = 0; r < H; r++)
    {
        for (uint c = 0; c < W; c++)
        {
            float x = a[uint3(e, r, c)];
            float y = b[uint3(e, r, c)];
            inter += x * y;
            uni += max(x, y);
        }
    }
    counts[uint2(e, 0)] = inter;
    counts[uint2(e, 1)] = uni;
}
