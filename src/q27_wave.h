// q27_wave.h -- wave-level helpers shared by the projection kernels (device code only).
#pragma once
// v = v + (lane i+O within the 16-lane row), as one DPP row_shl:O add; the DPP control must be a literal constant.
template <int O>
__device__ __forceinline__ float q27_dpp_shl_add(float v) {
    return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x100 | O, 0xF, 0xF, true));
}
// Exact re-emission of NR copies of the tree `for (o = 32; o; o >>= 1) v += __shfl_down(v, o, 64)` (lane 0's
// result only ever consumes lanes < o at each step; fadd is commutative): the o=32/16 steps as ds_bpermute issued
// GB accumulators at a time before any wait, the o=8/4/2/1 steps as DPP row_shl adds (in-row, no LDS).
// Lane 0 holds the same bits as the shfl_down tree; other lanes are don't-care.
template <int NR, int GB>
__device__ __forceinline__ void q27_wave_reduce_multi(float* v, int lane) {
    static_assert(NR % GB == 0, "GB must divide NR");
#pragma unroll
    for (int o = 32; o >= 16; o >>= 1) {
        const int addr = ((lane + o) & 63) << 2;
#pragma unroll
        for (int b0 = 0; b0 < NR; b0 += GB) {
            float t[GB];
#pragma unroll
            for (int k = 0; k < GB; ++k) t[k] = __int_as_float(__builtin_amdgcn_ds_bpermute(addr, __float_as_int(v[b0 + k])));
#pragma unroll
            for (int k = 0; k < GB; ++k) v[b0 + k] += t[k];
        }
    }
#pragma unroll
    for (int k = 0; k < NR; ++k) v[k] = q27_dpp_shl_add<8>(v[k]);
#pragma unroll
    for (int k = 0; k < NR; ++k) v[k] = q27_dpp_shl_add<4>(v[k]);
#pragma unroll
    for (int k = 0; k < NR; ++k) v[k] = q27_dpp_shl_add<2>(v[k]);
#pragma unroll
    for (int k = 0; k < NR; ++k) v[k] = q27_dpp_shl_add<1>(v[k]);
}
