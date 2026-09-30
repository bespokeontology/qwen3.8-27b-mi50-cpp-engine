// exl3_gemv_gfx906.hip -- FUSED EXL3 decode + M=1 GEMV for gfx906, any K.
//
// WHY THIS SHAPE. The obvious port target was moe_su_gfx906's su_sweep, but that kernel's
// beautiful lane-contiguous 1 KB sweep is K=4 SPECIFIC: it maps 8 lanes x one uint4 = exactly the
// 128 B cell of K=4. At K=6 a cell is 192 B = 12 uint4 and the mapping does not divide. Worse,
// dq4<6> (the K=6 decoder) does MODULAR indexing over the whole 48-dword cell, so it needs random
// access to the cell rather than a streaming register window.
//
// So this fuses into reconstruct_kernel's structure instead, which is already K-generic and already
// VERIFIED to decode our store correctly (test_exl3_q27 / test_exl3_fwd): stage 8 cells in LDS,
// dq_dispatch<K,cb>, frags_to_tile -> a row-major 16x128 fp16 tile, and then -- instead of storing
// 89M halves to global -- dot that tile against x. Every piece of bit math and the whole
// tensor-core-to-row-major element mapping are REUSED, not re-derived, so the one failure mode that
// matters here (a wrong element mapping producing plausible garbage) cannot be introduced.
//
// Correctness is checked against the reconstruct+CPU reference, not against plausibility.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "exl3_dequant_gfx906.h"
#include <array>
#include <cstdio>

namespace exl3gfx {

template <int K, int cb>
__global__ __launch_bounds__(256)
void gemv_kernel(float* __restrict__ y,             // [out_features] fp32, pre-zeroed
                 const __half* __restrict__ x,      // [in_features] fp16 (already suh-scaled + rotated)
                 const uint16_t* __restrict__ g_packed,
                 int packed_blocks_n,
                 __half* __restrict__ dbg_tile)   // block (0,0) dumps its 16x128 tile here
{
    constexpr int packed_size = 256 * K / 16;       // uint16s per cell
    const int t = threadIdx.x, lane_id = t % 32, warp_id = t / 32;
    const int k = blockIdx.y;                        // which 16-row k block
    const int n = blockIdx.x * 8;                    // first cell of this 128-column block

    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (size_t)(k * packed_blocks_n + n) * packed_size;
    if (t < packed_size) ((int4*)s_packed)[t] = ((const int4*)g_packed)[t];
    __syncthreads();

    FragB frag[2];
    dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    __shared__ __half2 tile[16][8][8];               // == row-major [16][128] fp16
    frags_to_tile(frag, lane_id, warp_id, tile);

    __shared__ float xs[16];
    if (t < 16) xs[t] = __half2float(x[k * 16 + t]);
    __syncthreads();

    // y[n*16 + c] += sum_r xs[r] * W[r][c]   for the 128 columns of this block
    const __half* W = (const __half*)tile;
    if (dbg_tile && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int i = t; i < 16 * 128; i += 256) dbg_tile[i] = W[i];
    }
    if (t < 128) {
        float acc = 0.f;
        #pragma unroll
        for (int r = 0; r < 16; ++r) acc = fmaf(xs[r], __half2float(W[r * 128 + t]), acc);
        atomicAdd(&y[(size_t)n * 16 + t], acc);
    }
}

#define __(i, cb) gemv_kernel<i, cb>
constexpr auto gemv_instances = std::array{
    __(1,0), __(2,0), __(3,0), __(4,0), __(5,0), __(6,0), __(7,0), __(8,0),
    __(1,1), __(2,1), __(3,1), __(4,1), __(5,1), __(6,1), __(7,1), __(8,1),
    __(1,2), __(2,2), __(3,2), __(4,2), __(5,2), __(6,2), __(7,2), __(8,2)
};
#undef __


// ---------------------------------------------------------------------------
// LEVER 1: dot straight from the fragments. The 4 KB LDS tile in gemv_kernel exists only because
// reconstruct_kernel needed a ROW-MAJOR layout to STORE; a GEMV never needs the layout at all.
// This reproduces frags_to_tile's m0..m7 and its r0..r3 / c0,c1 arithmetic VERBATIM -- the same
// __shfl_down(...,4,32) merge, the same row and column formulas -- and then multiplies each value
// by x[row] instead of writing it to shared memory. Reusing the arithmetic verbatim is deliberate:
// re-deriving it is exactly how a wrong element mapping becomes plausible garbage (it already cost
// one debugging round on the codebook index). Diffed against gemv_kernel, which is bit-exact
// against exl3_reconstruct_gfx906.
template <int K, int cb>
__global__ __launch_bounds__(256)
void gemv_kernel_frag(float* __restrict__ y,
                      const __half* __restrict__ x,
                      const uint16_t* __restrict__ g_packed,
                      int packed_blocks_n)
{
    constexpr int packed_size = 256 * K / 16;
    const int t = threadIdx.x, lane_id = t % 32, warp_id = t / 32;
    const int k = blockIdx.y, n = blockIdx.x * 8;

    __shared__ uint32_t s_packed[8][packed_size / 2];
    __shared__ float acc_c[128];                     // 128 output columns of this block
    g_packed += (size_t)(k * packed_blocks_n + n) * packed_size;
    if (t < packed_size) ((int4*)s_packed)[t] = ((const int4*)g_packed)[t];
    if (t < 128) acc_c[t] = 0.f;
    __shared__ float xs[16];
    if (t < 16) xs[t] = __half2float(x[k * 16 + t]);
    __syncthreads();

    FragB frag[2];
    dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // --- verbatim from frags_to_tile ---
    int n0b = __shfl_down((int)half2_to_bits(frag[0][0]), 4, 32);
    int n1b = __shfl_down((int)half2_to_bits(frag[0][1]), 4, 32);
    int n2b = __shfl_down((int)half2_to_bits(frag[1][0]), 4, 32);
    int n3b = __shfl_down((int)half2_to_bits(frag[1][1]), 4, 32);
    if (!(lane_id & 4))
    {
        __half2 n0 = half2_from_bits((uint32_t)n0b);
        __half2 n1 = half2_from_bits((uint32_t)n1b);
        __half2 n2 = half2_from_bits((uint32_t)n2b);
        __half2 n3 = half2_from_bits((uint32_t)n3b);
        __half2 m0 = __halves2half2(__low2half (frag[0][0]), __low2half (n0));
        __half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        __half2 m2 = __halves2half2(__low2half (frag[0][1]), __low2half (n1));
        __half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        __half2 m4 = __halves2half2(__low2half (frag[1][0]), __low2half (n2));
        __half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        __half2 m6 = __halves2half2(__low2half (frag[1][1]), __low2half (n3));
        __half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        const int r0 = (lane_id % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = lane_id / 8,       c1 = c0 + 4;
        // tile[r][warp_id][c] is a __half2 covering columns warp_id*16 + c*2 + {0,1}
        const int b0 = warp_id * 16 + c0 * 2;
        const int b1 = warp_id * 16 + c1 * 2;
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        p0 = fmaf(xs[r0], __low2float(m0), p0);  p1 = fmaf(xs[r0], __high2float(m0), p1);
        p0 = fmaf(xs[r1], __low2float(m1), p0);  p1 = fmaf(xs[r1], __high2float(m1), p1);
        p0 = fmaf(xs[r2], __low2float(m2), p0);  p1 = fmaf(xs[r2], __high2float(m2), p1);
        p0 = fmaf(xs[r3], __low2float(m3), p0);  p1 = fmaf(xs[r3], __high2float(m3), p1);
        p2 = fmaf(xs[r0], __low2float(m4), p2);  p3 = fmaf(xs[r0], __high2float(m4), p3);
        p2 = fmaf(xs[r1], __low2float(m5), p2);  p3 = fmaf(xs[r1], __high2float(m5), p3);
        p2 = fmaf(xs[r2], __low2float(m6), p2);  p3 = fmaf(xs[r2], __high2float(m6), p3);
        p2 = fmaf(xs[r3], __low2float(m7), p2);  p3 = fmaf(xs[r3], __high2float(m7), p3);
        atomicAdd(&acc_c[b0    ], p0);
        atomicAdd(&acc_c[b0 + 1], p1);
        atomicAdd(&acc_c[b1    ], p2);
        atomicAdd(&acc_c[b1 + 1], p3);
    }
    __syncthreads();
    if (t < 128) atomicAdd(&y[(size_t)n * 16 + t], acc_c[t]);
}

#define __(i, cb) gemv_kernel_frag<i, cb>
constexpr auto gemv_frag_instances = std::array{
    __(1,0), __(2,0), __(3,0), __(4,0), __(5,0), __(6,0), __(7,0), __(8,0),
    __(1,1), __(2,1), __(3,1), __(4,1), __(5,1), __(6,1), __(7,1), __(8,1),
    __(1,2), __(2,2), __(3,2), __(4,2), __(5,2), __(6,2), __(7,2), __(8,2)
};
#undef __

}  // namespace exl3gfx

// Host entry. Mirrors exl3_reconstruct_gfx906's dispatch and constraints.
void exl3_gemv_frag_gfx906(float* y, const __half* x, const uint16_t* packed,
                           int K, bool mcg, bool mul1, int in_features, int out_features,
                           hipStream_t stream)
{
    int idx = K - 1;
    if (mcg) idx += 8;
    else if (mul1) idx += 16;
    dim3 grid(out_features / 128, in_features / 16);
    hipLaunchKernelGGL(exl3gfx::gemv_frag_instances[idx], grid, dim3(256), 0, stream,
                       y, x, packed, out_features / 16);
}

void exl3_gemv_gfx906(float* y, const __half* x, const uint16_t* packed,
                      int K, bool mcg, bool mul1, int in_features, int out_features,
                      hipStream_t stream, __half* dbg_tile)
{
    // MIRROR exl3_reconstruct_gfx906's index arithmetic EXACTLY (reconstruct.cu:127-129).
    // mcg -> +8 (cb 1), mul1 -> +16 (cb 2), neither -> cb 0. The name "mul1" does NOT mean the
    // multiply-add codebook at cb 0; assuming it did produced a tile that differed from
    // reconstruct's in 2046 of 2048 elements, i.e. plausible-looking garbage.
    int idx = K - 1;
    if (mcg) idx += 8;
    else if (mul1) idx += 16;
    dim3 grid(out_features / 128, in_features / 16);
    dim3 blk(256);
    hipLaunchKernelGGL(exl3gfx::gemv_instances[idx], grid, blk, 0, stream,
                       y, x, packed, out_features / 16, dbg_tile);
}
