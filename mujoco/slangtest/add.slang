[AutoPyBindCUDA]
[CUDAKernel]
void add_one(TensorView<float> a, TensorView<float> out)
{
    uint i = cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
    if (i >= a.size(0)) return;
    out[i] = a[i] + 1.0f;
}
