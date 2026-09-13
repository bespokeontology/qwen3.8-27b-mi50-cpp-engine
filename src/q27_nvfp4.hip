// q27_nvfp4.hip — NVFP4 projection GEMV + its activation quantizer, for gfx906 (wave64, no MFMA).
//
// This file owns THE hot kernel of the decode step. It serves
//     gate/up  [17408, 5120]   x2 per layer x 64
//     down     [ 5120, 17408]  x1 per layer x 64
//     lm_head  [248320, 5120]  x1 per token
// which together are 58.8% of the decode bytes.
//
// ABI: include/q27.h. Mathematics and storage: see include/q27.h §6.
//
// ---------------------------------------------------------------------------------------------
// PROVENANCE OF THE GEOMETRY (do not "improve" it without a paired measurement)
//   reference/nvfp4_dot4_v3.hip kernel kE, measured on a real gate_proj [17408,5120] on card 3:
//       0.1496 ms, 335.1 GB/s of weight bytes — 16.8x the float-dequant path, and faster than a
//       pre-converted int8 arena while reading half the bytes.
//   Its structure, reproduced here verbatim and then generalized to arbitrary (rows, K):
//       * 32 lanes per row, 8 rows per 256-thread block  (one row per half-wave; no cross-row talk)
//       * five uint4 weight loads per lane at stride 32   -> ALL loads issued, then ONE wait
//       * q27_dq4 byte-permute decode (2x v_perm_b32 + 1x v_bfi_b32 per 4 nibbles)
//       * four v_dot4_i32_i8 per group of 16, against an activation PRE-PERMUTED even-then-odd
//       * one fmaf per group applying q27_e4m3(gs) * xs[g] * 0.5f
//       * 32-lane shuffle reduction, epilogue scalar on lane 0
//   The 0.5f folds out the factor 2 that the int8 weight representation carries
//   (E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} x2 = {0,1,2,3,4,6,8,12}, contract §6).
//
// GENERALIZATION
//   uint4 per row  NU4 = K/32   (one uint4 = 16 packed bytes = 32 weights = exactly 2 groups of 16)
//   uint4 per lane UPL = K/1024 :  K=5120 -> 5 (=kE),  K=17408 -> 17.
//   K=5120 is dispatched to a fully-unrolled compile-time kernel, so the measured path is
//   reproduced instruction-for-instruction. K=17408 would blow VGPR fully unrolled, so it goes to
//   a blocked kernel: an outer `#pragma unroll 1` loop whose BODY still issues Q27_NVFP4_UNROLL
//   independent uint4 loads before any decode. That keeps the dependency-graph property (few
//   vmcnt waits) which is what the estate measured as load-bearing, while capping live weight
//   registers at 4x uint4.
//
// WHAT WAS VERIFIED WITHOUT A GPU (2026-09-08; no GPU run was authorized for this file)
//   codegen equivalence, ROCm 5.7.1 gfx906, q27_k_nvfp4_fixed<5> vs reference kE:
//       v_dot4 40/40   v_perm_b32 120/120   v_bfi_b32 40/40   global_load 35/35
//       ds_bpermute 5/5   fma 10/10   s_waitcnt vmcnt(0) 1/1   spill 0/0   LDS 0/0
//       VGPR 90 vs 89 (the one extra holds `rows`, which kE had as a compile-time constant),
//       occupancy 2 waves/SIMD in both. The measured path is reproduced, not re-derived.
//   q27_k_nvfp4_blocked<4>: VGPR 86, occupancy 2, 0 spill, vmcnt(0) count 2 (both are loop-exit
//       fences). Its steady-state loop issues four global_load_dwordx4 at offsets -1024/-512/0/+512
//       back to back and then waits vmcnt(2) — the counted-wait shape, not a serialized one.
//   numerics: a CPU emulation of v_perm_b32 / q27_dq4 / v_dot4_i32_i8 confirms (a) all 16 E2M1
//       codes decode to exactly 2x their value, (b) over 20000 random groups the full chain
//       quantizer-permutation -> nibble split -> dot4 -> e4m3(gs)*xs*0.5 -> in_scale*ws2 matches a
//       direct fp64 sum(w*x) to 3.5e-16 of sum|terms|. UNMEASURED here: wall time on the card.
//
// SCALE COMPOSITION (contract §5 — a lost activation scale compounds through SwiGLU; this is the
// single worst documented failure in the estate, so it is spelled out)
//     stored weight        w[r,k] = E2M1(code) * e4m3(gs[r, k>>4]) * ws2
//     activation, 2 level  global = 1/in_scale ; the quantizer emits int8 q and a per-16 float xs
//                          such that  x[k] * global  ==  q[k] * xs[k>>4]
//     therefore
//       y[r] = SUM_k w[r,k] x[k]
//            = (in_scale * ws2) * SUM_g  e4m3(gs[r,g]) * xs[g] * 0.5f * SUM_{k in g} (2*E2M1) * q
//     -> alpha = in_scale * ws2, formed ONCE on the host and applied ONCE on lane 0. Never per
//        block, never per group. Both scales are MULTIPLIERS; no input_scale in the checkpoint is
//        1.0 (range 0.0013079 .. 0.602679), so dropping it is not a no-op.
//
//   Deliberate deviation, recorded: the model's own activation path stores a 4-bit e2m1 value with
//   an e4m3-ROUNDED per-16 scale, because CUTLASS NVFP4 requires a scale in an e4m3 register. This
//   engine's execution representation is int8 activations (contract §6 sizes the accumulator at
//   17408 * 12 * 127 = 26,529,792, i.e. the FULL int8 activation range), and our per-16 scale
//   multiplies an fp32 partial sum, so no e4m3 register exists to satisfy. We therefore keep the
//   dynamic scale at full fp32 precision: strictly more accurate, same composition. Define
//   Q27_QUANT_E4M3_SCALE to reproduce the model's e4m3-rounded scale exactly for an A/B.
//
// ---------------------------------------------------------------------------------------------
#include "q27.h"
#include "q27_wave.h"
#include "q27_nr.h"
#include <cstdio>
#include <cstdint>

// ---------------- tunables ----------------
#define Q27_NVFP4_BLOCK   256      // 8 rows x 32 lanes
#define Q27_NVFP4_ROWS    8
#ifndef Q27_NVFP4_UNROLL           // independent uint4 loads issued per outer iteration (blocked)
#define Q27_NVFP4_UNROLL  4
#endif

// =============================================================================================
//  device helpers
// =============================================================================================

// One group of 16 weights against one group of 16 pre-permuted int8 activations.
//   pk.x holds packed elements 0..7   (byte j = {low nibble: element 2j, high nibble: 2j+1})
//   pk.y holds packed elements 8..15
//   xe   holds the EVEN activations 0,2,4,6 | 8,10,12,14   (dwords 0,1 of the permuted group)
//   xo   holds the ODD  activations 1,3,5,7 | 9,11,13,15   (dwords 2,3)
// The even/odd split is exactly what falls out of the nibble packing, which is why the
// permutation is done ONCE upstream (q27.h ACTIVATION PERMUTATION CONTRACT) and never here.
__device__ __forceinline__ int q27_grp16(uint2 pk, uint2 xe, uint2 xo)
{
    int a = 0;
    a = q27_d4(q27_dq4( pk.x       & 0x0F0F0F0Fu), xe.x, a);
    a = q27_d4(q27_dq4((pk.x >> 4) & 0x0F0F0F0Fu), xo.x, a);
    a = q27_d4(q27_dq4( pk.y       & 0x0F0F0F0Fu), xe.y, a);
    a = q27_d4(q27_dq4((pk.y >> 4) & 0x0F0F0F0Fu), xo.y, a);
    return a;
}

// One uint4 of packed weights = 32 weights = groups (2u, 2u+1). Two dot-product groups, two fmaf.
// NON-TEMPORAL WEIGHT LOADS. The FP8 family already does this (q27_ld16nt, 4 sites) and measures
// at ~741 GB/s; NVFP4 had none and measures ~465-534 on the same cards with the same 8-byte lane
// load. Weights are streamed ONCE per token and never reused, so allocating 2.585 GB/token/card
// into a 16 KB/CU L1 evicts the one thing that IS reused: every wave re-reads the same activation
// and scale vectors. glc=1 slc=1 keeps the weight stream out of L1 and marks it evict-first in L2.
// ONLY the weights get this. Xp/Xs/S stay cached deliberately -- they are the reused working set.
// Compile-time so the A/B compares two binaries rather than branching per load inside the
// inner loop, which would measure the branch instead of the cache policy.
#ifndef Q27_NVFP4_NT
#define Q27_NVFP4_NT 0   // MEASURED 2026-09-10: NT lost (~0.18 ms vs 1.5 predicted, 3/6 paired).
                         // Default matches the 59.604 authority binary. Q27_NVFP4_NT=1 re-enables.
#endif
typedef unsigned int q27_u32x2 __attribute__((ext_vector_type(2)));
static __device__ __forceinline__ uint2 q27_ld8nt(const void* p) {
#if Q27_NVFP4_NT
    const q27_u32x2 v = __builtin_nontemporal_load((const q27_u32x2*)p);
    return make_uint2(v.x, v.y);
#else
    return *(const uint2*)p;
#endif
}
typedef unsigned int q27_u32x4nv __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ uint4 q27_ld16nt_nv(const void* p) {
#if Q27_NVFP4_NT
    const q27_u32x4nv v = __builtin_nontemporal_load((const q27_u32x4nv*)p);
    return make_uint4(v.x, v.y, v.z, v.w);
#else
    return *(const uint4*)p;
#endif
}

__device__ __forceinline__ float q27_pair32(uint4 wv, int u,
        const unsigned char* __restrict__ sr, const unsigned* __restrict__ Xp,
        const float* __restrict__ Xs, float acc)
{
    const int gA = u << 1, gB = gA + 1;
    const uint2 pA = make_uint2(wv.x, wv.y);
    const uint2 pB = make_uint2(wv.z, wv.w);
    const uint2 eA = make_uint2(Xp[gA * 4 + 0], Xp[gA * 4 + 1]);
    const uint2 oA = make_uint2(Xp[gA * 4 + 2], Xp[gA * 4 + 3]);
    const uint2 eB = make_uint2(Xp[gB * 4 + 0], Xp[gB * 4 + 1]);
    const uint2 oB = make_uint2(Xp[gB * 4 + 2], Xp[gB * 4 + 3]);
    acc = fmaf((float)q27_grp16(pA, eA, oA), q27_e4m3(sr[gA]) * Xs[gA] * 0.5f, acc);
    acc = fmaf((float)q27_grp16(pB, eB, oB), q27_e4m3(sr[gB]) * Xs[gB] * 0.5f, acc);
    return acc;
}

// =============================================================================================
//  q27_proj_nvfp4 kernels
// =============================================================================================

// Fully-unrolled form. UPL = K/1024 uint4 per lane. UPL==5 is kE exactly (K=5120: gate, up, lm_head).
// All UPL loads are issued back to back with no intervening use, so the compiler emits one
// s_waitcnt for the whole batch instead of one per load. Count barriers, not instructions.
template<int UPL>
__global__ __launch_bounds__(Q27_NVFP4_BLOCK) void q27_k_nvfp4_fixed(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * Q27_NVFP4_ROWS + (threadIdx.x >> 5);
    if (row >= rows) return;                       // whole half-wave exits together: shuffle is safe
    const int lane = threadIdx.x & 31;
    const int KP = UPL * 32 * 16;                  // K/2  bytes  per row
    const int NG = UPL * 32 * 2;                   // K/16 groups per row
    const uint4*         wr = (const uint4*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;

    uint4 wv[UPL];
    #pragma unroll
    for (int i = 0; i < UPL; ++i) wv[i] = q27_ld16nt_nv(&wr[lane + i * 32]);  // stride 32, non-temporal

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < UPL; ++i) acc = q27_pair32(wv[i], lane + i * 32, sr, Xp, Xs, acc);

    #pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o, 32);
    if (lane == 0) Y[row] = acc * alpha;

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// ---------------------------------------------------------------------------------------------
// FP8 -> NVFP4 CONVERSION, AT LOAD. The checkpoint ships the attention and GDN projections as
// e4m3 (fp8) while the MLP is NVFP4; the fp8 bucket is 29% of GPU-busy. This re-quantizes an
// fp8 tensor into the NVFP4 representation (E2M1 codes + per-16 E4M3 scales, ws2 = the fp8
// tensor's weight_scale) so the same consumer kernels can stream half the bytes. The recipe is
// the engine's own (E4M3-rounded scales, nearest E2M1); representation is free under the
// numerical law, and the quality gate is real generations. Q27_FP8X4=1 converts at load.
__device__ __forceinline__ unsigned char q27_f32_to_e4m3(float v)
{
    // non-negative encode, OCP e4m3fn, correct-by-construction against the q27_e4m3 decoder:
    //   decode: e==0 -> m*2^-9;  else (1 + m/8) * 2^(e-7)
    //   so a normal value r = (1+m/8)*2^(e-7) has e = ex2+6 (frexpf: r = fr*2^ex2, fr in [0.5,1))
    //   and m = round((fr*2 - 1)*8); subnormals (r < 2^-6) use e=0, m = round(r*2^9).
    if (!(v > 0.f)) return 0;
    if (v >= 448.f) return 0x7e;
    int ex; frexpf(v, &ex);                          // v = f*2^ex, f in [0.5,1)
    int se = ex - 4;                                 // value step = 2^se (mantissa LSB = binade-3)
    if (se < -9) se = -9;                            // subnormal floor
    const float step = ldexpf(1.0f, se);
    const float r = rintf(v / step) * step;          // nearest representable
    if (r >= 448.f) return 0x7e;
    int ex2; const float fr = frexpf(r, &ex2);
    if (ex2 >= -5) {
        const int e = ex2 + 6;
        int m = (int)rintf((fr * 2.0f - 1.0f) * 8.0f);
        if (m >= 8) { m = 0; return (unsigned char)(((e + 1) << 3) | m); }   // carry
        return (unsigned char)((e << 3) | m);
    } else {
        const int m = (int)rintf(r * 512.0f);
        if (m >= 8) return (unsigned char)(1 << 3);  // rounded up into the smallest normal
        return (unsigned char)(m & 7);
    }
}
__device__ __forceinline__ unsigned char q27_f32_to_e2m1(float v)
{
    // |v| in [0,6]: nearest of {0,0.5,1,1.5,2,3,4,6}; sign in bit 3
    const float a = fabsf(v);
    int c = 0;
    if (a > 0.25f) c = 1;                            // 0.5
    if (a > 0.75f) c = 2;                            // 1
    if (a > 1.25f) c = 3;                            // 1.5
    if (a > 1.75f) c = 4;                            // 2
    if (a > 2.5f)  c = 5;                            // 3
    if (a > 3.5f)  c = 6;                            // 4
    if (a > 5.0f)  c = 7;                            // 6
    return (unsigned char)(c | ((v < 0.f) ? 8 : 0));
}
__global__ void q27_k_fp8_to_nvfp4(const unsigned char* __restrict__ F,  // [rows][K] e4m3
                                   unsigned char* __restrict__ W,        // [rows][K/2] packed
                                   unsigned char* __restrict__ S,        // [rows][K/16] e4m3
                                   int rows, int K)
{
    const int g = blockIdx.x * 256 + threadIdx.x;
    const int NG = K >> 4;
    if (g >= rows * NG) return;
    const int r  = g / NG;
    const int gr = g % NG;
    const unsigned char* fr = F + (size_t)r * K + gr * 16;

    float vals[16]; float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { vals[j] = q27_e4m3s(fr[j]); mx = fmaxf(mx, fabsf(vals[j])); }
    const unsigned char sb = q27_f32_to_e4m3(mx * (1.0f / 6.0f));
    S[(size_t)r * NG + gr] = sb;
    const float rs = q27_e4m3(sb);
    const float inv = (rs > 0.f) ? (1.0f / rs) : 0.f;
    unsigned char* wr = W + (size_t)r * (K >> 1) + gr * 8;
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
        const unsigned char c0 = q27_f32_to_e2m1(vals[j]     * inv);
        const unsigned char c1 = q27_f32_to_e2m1(vals[j + 1] * inv);
        wr[j >> 1] = (unsigned char)(c0 | (c1 << 4));
    }
}
extern "C" void q27_fp8_to_nvfp4_conv(const q27_fp8_t* src, q27_nvfp4_t* dst, hipStream_t st) {
    dst->rows = src->rows; dst->K = src->K;
    dst->ws2 = src->wscale; dst->in_scale = src->in_scale;
    const size_t WB = (size_t)src->rows * (size_t)(src->K >> 1);
    const size_t SB = (size_t)src->rows * (size_t)(src->K >> 4);
    if (hipMalloc((void**)&dst->w, WB) != hipSuccess ||
        hipMalloc((void**)&dst->gs, SB) != hipSuccess) {
        std::fprintf(stderr, "FP8X4: conversion allocation failed\n");
        dst->w = 0; dst->gs = 0;
        return;
    }
    const int NG = src->K >> 4;
    hipLaunchKernelGGL(q27_k_fp8_to_nvfp4, dim3((size_t)((size_t)src->rows * NG + 255) / 256),
                       dim3(256), 0, st, src->w, (unsigned char*)dst->w, (unsigned char*)dst->gs,
                       src->rows, src->K);
}

// Q27_LS_Q8: int8 EXECUTION mirror of an fp8 projection (gfx906 has dot4, not an fp8 datapath).
// Per 16-group along K: gs = e4m3(max|w|/127), w8 = rint(w/gs) clamped to [-127,127]; w is the e4m3 value
// (the per-tensor wscale stays in ws2). Consumers: q27_k_wide_i8 (natural K order, dot4, gs*xs per group).
__global__ void q27_k_fp8_to_i8g(const unsigned char* __restrict__ F, signed char* __restrict__ W8,
                                 unsigned char* __restrict__ S, int rows, int K)
{
    const int g = blockIdx.x * 256 + threadIdx.x;
    const int NG = K >> 4;
    if (g >= rows * NG) return;
    const int r = g / NG, gr = g % NG;
    const unsigned char* fr = F + (size_t)r * K + gr * 16;
    float vals[16]; float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { vals[j] = q27_e4m3s(fr[j]); mx = fmaxf(mx, fabsf(vals[j])); }
    const unsigned char sb = q27_f32_to_e4m3(mx * (1.0f / 127.0f));
    S[(size_t)r * NG + gr] = sb;
    const float rs = q27_e4m3(sb);
    const float inv = (rs > 0.f) ? (1.0f / rs) : 0.f;
    signed char* wr = W8 + (size_t)r * K + gr * 16;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int q = (int)rintf(vals[j] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        wr[j] = (signed char)q;
    }
}
extern "C" void q27_fp8_to_i8g_conv(const q27_fp8_t* src, q27_nvfp4_t* dst, hipStream_t st) {
    dst->rows = src->rows; dst->K = src->K;
    dst->ws2 = src->wscale; dst->in_scale = src->in_scale;
    const size_t WB = (size_t)src->rows * (size_t)src->K;
    const size_t SB = (size_t)src->rows * (size_t)(src->K >> 4);
    if (hipMalloc((void**)&dst->w, WB) != hipSuccess || hipMalloc((void**)&dst->gs, SB) != hipSuccess) {
        std::fprintf(stderr, "Q8: int8 mirror allocation failed\n");
        dst->w = 0; dst->gs = 0;
        return;
    }
    const int NG = src->K >> 4;
    hipLaunchKernelGGL(q27_k_fp8_to_i8g, dim3((size_t)((size_t)src->rows * NG + 255) / 256), dim3(256), 0, st,
                       src->w, (signed char*)dst->w, (unsigned char*)dst->gs, src->rows, src->K);
}

// w64 with BF16 output, for the converted q/k/v/in_qkv/in_z whose consumers want BF16
// (the fp8 path wrote BF16 in the epilogue; this preserves the shape of the producer).
template<int UPL2>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64b(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, unsigned short* __restrict__ Y)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int KP = UPL2 * 64 * 8;
    const int NG = UPL2 * 64;
    const uint2*         wr = (const uint2*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;

    uint2 wv[UPL2];
    uint4 xv[UPL2];
    float sc[UPL2];
#pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;
        wv[i] = q27_ld8nt(&wr[g]);
        xv[i] = *(const uint4*)(Xp + (size_t)g * 4);
        sc[i] = q27_e4m3(sr[g]) * Xs[g] * 0.5f;
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const uint2 e = make_uint2(xv[i].x, xv[i].y);
        const uint2 o = make_uint2(xv[i].z, xv[i].w);
        acc = fmaf((float)q27_grp16(wv[i], e, o), sc[i], acc);
    }
#pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) Y[row] = q27_f2bf(acc * alpha);

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
// BATCHED FUSED gate+up (prefill): C positions share both weight streams. Per-position
// accumulation order identical to q27_k_nvfp4_w64_gu -> bit-identical tokens.
template <int UPL2, int C>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64_gu_b(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int KP = UPL2 * 64 * 8;
    const int NG = UPL2 * 64;
    const uint2* wg = (const uint2*)(WG + (size_t)row * (size_t)KP);
    const uint2* wu = (const uint2*)(WU + (size_t)row * (size_t)KP);
    const unsigned char* sg = SG + (size_t)row * (size_t)NG;
    const unsigned char* su = SU + (size_t)row * (size_t)NG;

    float accg[C], accu[C];
#pragma unroll
    for (int c = 0; c < C; ++c) { accg[c] = 0.f; accu[c] = 0.f; }
#pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;
        const uint2 vg = q27_ld8nt(&wg[g]);
        const uint2 vu = q27_ld8nt(&wu[g]);
        const float cg0 = q27_e4m3(sg[g]);
        const float cu0 = q27_e4m3(su[g]);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const uint4 xv = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)g * 4);
            const float xsc = Xs[(size_t)c * UPL2 * 64 + g] * 0.5f;
            const uint2 e = make_uint2(xv.x, xv.y);
            const uint2 o = make_uint2(xv.z, xv.w);
            accg[c] = fmaf((float)q27_grp16(vg, e, o), cg0 * xsc, accg[c]);
            accu[c] = fmaf((float)q27_grp16(vu, e, o), cu0 * xsc, accu[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float vg2 = accg[c], vu2 = accu[c];
#pragma unroll
        for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
        if (lane == 0) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
extern "C" void q27_proj_nvfp4_gu_b(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120) return;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g64((unsigned)((wg->rows + 3) / 4)), b64(256);
    // C is padded UP to the next instantiated size: the per-position slots are 8 wide, so
    // the extra positions compute into slots nobody reads. Launching NOTHING for C in
    // {3,5,6,7} left STALE slots for the tail chunk of any prompt whose length is not a
    // multiple of 8 (205 = 25*8+5). Caught by the batched-GDN gate.
    const int Cp = (C <= 1) ? 1 : (C <= 2) ? 2 : (C <= 4) ? 4 : 8;
    if (Cp == 8)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_gu_b<5,8>), g64, b64, 0, s,
                           wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs,
                           ag, au, wg->rows, yg, yu);
    else if (Cp == 4)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_gu_b<5,4>), g64, b64, 0, s,
                           wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs,
                           ag, au, wg->rows, yg, yu);
    else if (Cp == 2)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_gu_b<5,2>), g64, b64, 0, s,
                           wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs,
                           ag, au, wg->rows, yg, yu);
    else if (Cp == 1)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_gu_b<5,1>), g64, b64, 0, s,
                           wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs,
                           ag, au, wg->rows, yg, yu);
}

// BATCHED down (prefill, K=4352): the clamped full-width tail from w64t2, C accumulators.
// radd (the residual share) is the per-position hid array: [C][rows] bf16, contiguous.
template <int FULL2, int TAIL, int C>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64t2_b(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd = nullptr, float rs = 0.f)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int NG = FULL2 * 64 + TAIL;
    const int KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4;     // K/4 u32 per position
    const int NGS = FULL2 * 64 + TAIL;          // K/16 floats per position
    const uint2* wr = (const uint2*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;
    const bool tailact = (lane < TAIL);

    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int g = i * 64 + ((i < FULL2) ? lane : (tailact ? lane : 0));
        const uint2 wv = q27_ld8nt(&wr[g]);
        const float s0 = q27_e4m3(sr[g]) * 0.5f;
        if (i < FULL2 || tailact) {          // the clamped tail's contribution is masked off-lane
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)g * 4);
                const float s = s0 * Xs[(size_t)c * NGS + g];
                const uint2 e = make_uint2(xv.x, xv.y);
                const uint2 o = make_uint2(xv.z, xv.w);
                acc[c] = fmaf((float)q27_grp16(wv, e, o), s, acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float v = acc[c];
#pragma unroll
        for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
        if (lane == 0) {
            v *= alpha;
            if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
            Y[(size_t)c * rows + row] = v;
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
// BATCHED (prefill): C positions share every weight load. Xp/Xs are per-position arrays with
// stride UPL2*256 u32s (K/4) and UPL2*64 floats (K/16); Y is [C][rows]. The per-position
// accumulation order is IDENTICAL to q27_k_nvfp4_w64, so tokens are bit-identical by design.
template <int UPL2, int C>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64_b(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int KP = UPL2 * 64 * 8;
    const int NG = UPL2 * 64;
    const uint2*         wr = (const uint2*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;

    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;
        const uint2 wv = q27_ld8nt(&wr[g]);                  // ONE weight load serves C positions
        const float s0 = q27_e4m3(sr[g]) * 0.5f;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const uint4 xv = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)g * 4);
            const float s = s0 * Xs[(size_t)c * UPL2 * 64 + g];
            const uint2 e = make_uint2(xv.x, xv.y);
            const uint2 o = make_uint2(xv.z, xv.w);
            acc[c] = fmaf((float)q27_grp16(wv, e, o), s, acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float v = acc[c];
#pragma unroll
        for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
        if (lane == 0) Y[(size_t)c * rows + row] = v * alpha;
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
extern "C" void q27_proj_nvfp4_b(const q27_nvfp4_t* w, const signed char* xq_perm,
                                 const float* xs, float* y, int C, hipStream_t s) {
    if (w->K != 5120) return;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g64((unsigned)((w->rows + 3) / 4)), b64(256);
    // C is padded UP to the next instantiated size: the per-position slots are 8 wide, so
    // the extra positions compute into slots nobody reads. Launching NOTHING for C in
    // {3,5,6,7} left STALE slots for the tail chunk of any prompt whose length is not a
    // multiple of 8 (205 = 25*8+5). Caught by the batched-GDN gate.
    const int Cp = (C <= 1) ? 1 : (C <= 2) ? 2 : (C <= 4) ? 4 : 8;
    if (Cp == 8)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_b<5,8>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y);
    else if (Cp == 4)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_b<5,4>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y);
    else if (Cp == 2)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_b<5,2>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y);
    else if (Cp == 1)
        hipLaunchKernelGGL((q27_k_nvfp4_w64_b<5,1>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y);
}

extern "C" int q27_proj_nvfp4_bf16(const q27_nvfp4_t* w, const signed char* xq_perm,
                                   const float* xs, unsigned short* y, hipStream_t s) {
    if (!w || w->K != 5120) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g64((unsigned)((w->rows + 3) / 4)), b64(256);
    hipLaunchKernelGGL((q27_k_nvfp4_w64b<5>), g64, b64, 0, s,
                       w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y);
    return 1;
}

// ---------------------------------------------------------------------------------------------
// K-WIDENED LOADS. The w64 family loads dwordx2 (8 B) per lane per iteration, so one wave
// iteration is a 512-B single-tensor burst, and the family measures 465-534 GB/s. The fp8
// family loads dwordx4 (16 B) per lane -> 1024-B bursts and measures ~741 GB/s on the same
// cards. Neither family is issue-bound (~0.06 instr/B against ~1.4 Ginstr/s/CU), so the
// burst width is the binding resource: one 1024-B contiguous stream per wave-iteration beats
// two 512-B ones. This form pairs two groups per lane: one dwordx4 weight load, one ushort
// scale load, one dwordx2 activation-scale load. Q27_NVFP4_WIDE=1 selects it; =0 is the
// shipped w64 for the A/B.
template<int WIDE2, int NARROW>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64w(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int NG = (WIDE2 * 2 + NARROW) * 64;
    const int KP = NG * 8;
    const uint4*         wr4 = (const uint4*)(W + (size_t)row * (size_t)KP);   // 16 B units
    const uint2*         wr2 = (const uint2*)(W + (size_t)row * (size_t)KP);   // 8 B units
    const unsigned char* sr = S + (size_t)row * (size_t)NG;

    uint4 wv[WIDE2];  uint4 xv0[WIDE2];  uint4 xv1[WIDE2];
    unsigned short sc2[WIDE2];  uint2 xs2[WIDE2];
    #pragma unroll
    for (int i = 0; i < WIDE2; ++i) {
        const int l2 = lane + i * 64;
        const int g  = 2 * l2;
        wv[i]   = q27_ld16nt_nv(&wr4[l2]);                   // 16 B = groups 2*l2, 2*l2+1
        xv0[i]  = *(const uint4*)(Xp + (size_t)g * 4);
        xv1[i]  = *(const uint4*)(Xp + (size_t)(g + 1) * 4);
        sc2[i]  = *(const unsigned short*)(sr + g);          // both scale bytes, one load
        xs2[i]  = *(const uint2*)(Xs + g);                   // both activation scales
    }
    const int gt = (WIDE2 * 2) * 64 + lane;                  // narrow tail: one group, full width
    const uint2 wvt = q27_ld8nt(&wr2[gt]);
    const uint4 xvt = *(const uint4*)(Xp + (size_t)gt * 4);
    const unsigned char sct = sr[gt];
    const float xst = Xs[gt];

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < WIDE2; ++i) {
        const uint2 e0 = make_uint2(xv0[i].x, xv0[i].y);
        const uint2 o0 = make_uint2(xv0[i].z, xv0[i].w);
        const uint2 e1 = make_uint2(xv1[i].x, xv1[i].y);
        const uint2 o1 = make_uint2(xv1[i].z, xv1[i].w);
        const float s0 = q27_e4m3((unsigned char)(sc2[i] & 0xFF)) * xs2[i].x * 0.5f;
        const float s1 = q27_e4m3((unsigned char)(sc2[i] >> 8))   * xs2[i].y * 0.5f;
        acc = fmaf((float)q27_grp16(make_uint2(wv[i].x, wv[i].y), e0, o0), s0, acc);
        acc = fmaf((float)q27_grp16(make_uint2(wv[i].z, wv[i].w), e1, o1), s1, acc);
    }
    {
        const uint2 e = make_uint2(xvt.x, xvt.y);
        const uint2 o = make_uint2(xvt.z, xvt.w);
        acc = fmaf((float)q27_grp16(wvt, e, o), q27_e4m3(sct) * xst * 0.5f, acc);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) Y[row] = acc * alpha;

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// ---------------------------------------------------------------------------------------------
// 64-lanes-per-row form. MEASURED 1.3876x over the 32-lane form above on the real gate_proj
// (0.1077 vs 0.1495 ms, 465.4 vs 335.4 GB/s, six of seven reps within 0.3%; one rep showed a
// 0.9168 ms transient, reported not deleted). Correctness vs the 32-lane form: relL2 1.158e-07,
// cosine 1.00000000 - pure float reassociation.
//
// Why it wins: a uint2 is 8 bytes = 16 weights = exactly ONE scale group, so 64 lanes issue one
// contiguous 512 B burst per step against the 32-lane form's two 512 B bursts from two different
// rows. Two independent priors agreed before the measurement: this engine's own FP8 projection
// uses 64 lanes/row and reaches ~533 GB/s, and Pangu's rejected ledger measured 64 lanes beating
// 32 by 1.6x on a wide-K dense projection ("64 lanes = one 512-B transaction").
// UPL2 = K/1024 uint2 per lane: K=5120 -> 5, K=17408 -> 17.
template<int UPL2>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y)
{
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;                        // a whole wave exits together
    const int lane = threadIdx.x & 63;
    const int KP = UPL2 * 64 * 8;                   // K/2  bytes  per row
    const int NG = UPL2 * 64;                       // K/16 groups per row
    const uint2*         wr = (const uint2*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;

    uint2 wv[UPL2];
    // The comment that used to sit here said "all loads before any decode". That was true of the
    // WEIGHTS only. The accumulate loop below still issued six more dependent loads per iteration --
    // four separate activation words, the group scale and the activation scale -- so at UPL2=5 the
    // real graph was 30 interleaved loads, not 5. Two fixes:
    //   * Xp[g*4+0..3] are four CONSECUTIVE unsigned, i.e. one 16-B load, not four.
    //   * every activation and scale load is hoisted alongside the weight loads.
    uint4 xv[UPL2];
    float sc[UPL2];
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;                // one uint2 == one group of 16
        wv[i] = q27_ld8nt(&wr[g]);
        xv[i] = *(const uint4*)(Xp + (size_t)g * 4);
        sc[i] = q27_e4m3(sr[g]) * Xs[g] * 0.5f;
    }

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const uint2 e = make_uint2(xv[i].x, xv[i].y);
        const uint2 o = make_uint2(xv[i].z, xv[i].w);
        acc = fmaf((float)q27_grp16(wv[i], e, o), sc[i], acc);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) Y[row] = acc * alpha;
}

// =============================================================================================
// FUSED gate+up. They are the SAME shape (rows=4352, K=5120 under 4-way TP) and read the IDENTICAL
// quantized activation. Two launches per layer became one: 128 -> 64 per token, the activation and
// its scales are loaded once instead of twice, and both accumulators share the same decode of the
// activation nibbles. Same trick that took the gated-delta bf16 GEMV from 3.783 to 0.740 ms.
template<int UPL2>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64_gu(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int KP = UPL2 * 64 * 8;
    const int NG = UPL2 * 64;
    const uint2*         wg = (const uint2*)(WG + (size_t)row * (size_t)KP);
    const uint2*         wu = (const uint2*)(WU + (size_t)row * (size_t)KP);
    const unsigned char* sg = SG + (size_t)row * (size_t)NG;
    const unsigned char* su = SU + (size_t)row * (size_t)NG;

    uint2 vg[UPL2], vu[UPL2];  uint4 xv[UPL2];  float xsc[UPL2];
    unsigned char cg[UPL2], cu[UPL2];
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;
        vg[i] = wg[g];  vu[i] = wu[g];
        cg[i] = sg[g];  cu[i] = su[g];
        xv[i] = *(const uint4*)(Xp + (size_t)g * 4);
        xsc[i] = Xs[g] * 0.5f;
    }

    float accg = 0.f, accu = 0.f;
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const uint2 e = make_uint2(xv[i].x, xv[i].y);
        const uint2 o = make_uint2(xv[i].z, xv[i].w);
        accg = fmaf((float)q27_grp16(vg[i], e, o), q27_e4m3(cg[i]) * xsc[i], accg);
        accu = fmaf((float)q27_grp16(vu[i], e, o), q27_e4m3(cu[i]) * xsc[i], accu);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) { accg += __shfl_down(accg, o, 64); accu += __shfl_down(accu, o, 64); }
    if (lane == 0) { YG[row] = accg * ag; YU[row] = accu * au; }
}

// K-widened FUSED gate+up. Same mechanism as q27_k_nvfp4_w64w: two groups per lane per
// iteration, one dwordx4 per matrix -> two 1024-B single-tensor bursts per wave-iteration
// instead of two 512-B ones, plus the shared activation loads amortise over 32 weights.
// Q27_NVFP4_GUW=1 selects it; =0 is the shipped w64_gu for the A/B.
template<int WIDE2, int NARROW>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64_guw(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int NG = (WIDE2 * 2 + NARROW) * 64;
    const int KP = NG * 8;
    const uint4* wg4 = (const uint4*)(WG + (size_t)row * (size_t)KP);
    const uint4* wu4 = (const uint4*)(WU + (size_t)row * (size_t)KP);
    const uint2* wg2 = (const uint2*)(WG + (size_t)row * (size_t)KP);
    const uint2* wu2 = (const uint2*)(WU + (size_t)row * (size_t)KP);
    const unsigned char* sg = SG + (size_t)row * (size_t)NG;
    const unsigned char* su = SU + (size_t)row * (size_t)NG;

    uint4 vg[WIDE2], vu[WIDE2];  uint4 xv0[WIDE2], xv1[WIDE2];
    unsigned short cg2[WIDE2], cu2[WIDE2];  uint2 xs2[WIDE2];
    #pragma unroll
    for (int i = 0; i < WIDE2; ++i) {
        const int l2 = lane + i * 64;
        const int g  = 2 * l2;
        vg[i]  = q27_ld16nt_nv(&wg4[l2]);
        vu[i]  = q27_ld16nt_nv(&wu4[l2]);
        cg2[i] = *(const unsigned short*)(sg + g);
        cu2[i] = *(const unsigned short*)(su + g);
        xv0[i] = *(const uint4*)(Xp + (size_t)g * 4);
        xv1[i] = *(const uint4*)(Xp + (size_t)(g + 1) * 4);
        xs2[i] = *(const uint2*)(Xs + g);
    }
    const int gt = (WIDE2 * 2) * 64 + lane;
    const uint2 vgt = q27_ld8nt(&wg2[gt]);
    const uint2 vut = q27_ld8nt(&wu2[gt]);
    const unsigned char cgt = sg[gt], cut = su[gt];
    const uint4 xvt = *(const uint4*)(Xp + (size_t)gt * 4);
    const float xst = Xs[gt];

    float accg = 0.f, accu = 0.f;
    #pragma unroll
    for (int i = 0; i < WIDE2; ++i) {
        const uint2 e0 = make_uint2(xv0[i].x, xv0[i].y);
        const uint2 o0 = make_uint2(xv0[i].z, xv0[i].w);
        const uint2 e1 = make_uint2(xv1[i].x, xv1[i].y);
        const uint2 o1 = make_uint2(xv1[i].z, xv1[i].w);
        const float g0 = q27_e4m3((unsigned char)(cg2[i] & 0xFF)) * xs2[i].x * 0.5f;
        const float g1 = q27_e4m3((unsigned char)(cg2[i] >> 8))   * xs2[i].y * 0.5f;
        const float u0 = q27_e4m3((unsigned char)(cu2[i] & 0xFF)) * xs2[i].x * 0.5f;
        const float u1 = q27_e4m3((unsigned char)(cu2[i] >> 8))   * xs2[i].y * 0.5f;
        accg = fmaf((float)q27_grp16(make_uint2(vg[i].x, vg[i].y), e0, o0), g0, accg);
        accg = fmaf((float)q27_grp16(make_uint2(vg[i].z, vg[i].w), e1, o1), g1, accg);
        accu = fmaf((float)q27_grp16(make_uint2(vu[i].x, vu[i].y), e0, o0), u0, accu);
        accu = fmaf((float)q27_grp16(make_uint2(vu[i].z, vu[i].w), e1, o1), u1, accu);
    }
    {
        const uint2 e = make_uint2(xvt.x, xvt.y);
        const uint2 o = make_uint2(xvt.z, xvt.w);
        accg = fmaf((float)q27_grp16(vgt, e, o), q27_e4m3(cgt) * xst * 0.5f, accg);
        accu = fmaf((float)q27_grp16(vut, e, o), q27_e4m3(cut) * xst * 0.5f, accu);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) { accg += __shfl_down(accg, o, 64); accu += __shfl_down(accu, o, 64); }
    if (lane == 0) { YG[row] = accg * ag; YU[row] = accu * au; }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_nvfp4_gu(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                 const signed char* xq_perm, const float* xs,
                                 float* yg, float* yu, hipStream_t s)
{
    if (!q27_env_flag("Q27_NVFP4_GU", true)) return 0;
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120) return 0;
    // Q27_GEMV_ALT (plain-decode A/B of the row-batched forms at NR=1; identical arithmetic): 1 = register NR form, 2 = LDS-staged form
    static const int alt = q27_env_int("Q27_GEMV_ALT", 0);
    if (alt == 1 && q27_proj_nvfp4_gu_nr(wg, wu, xq_perm, xs, yg, yu, wg->rows, 1, s)) return 1;
    if (alt == 2 && q27_proj_nvfp4_gu_lds(wg, wu, xq_perm, xs, yg, yu, wg->rows, 1, s)) return 1;
    if (alt == 3 && q27_proj_nvfp4_gu_hg(wg, wu, xq_perm, xs, yg, yu, wg->rows, 1, s)) return 1;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g64((unsigned)((wg->rows + 3) / 4)), b64(256);
    if (q27_env_flag("Q27_NVFP4_GUW", false))
        hipLaunchKernelGGL((q27_k_nvfp4_w64_guw<2,1>), g64, b64, 0, s,
                           wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs,
                           ag, au, wg->rows, yg, yu);
    else
        hipLaunchKernelGGL((q27_k_nvfp4_w64_gu<5>), g64, b64, 0, s,
                           wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs,
                           ag, au, wg->rows, yg, yu);
    return 1;
}

// ---------------------------------------------------------------------------------------------
// PREDICATED-TAIL 64-LANE FORM. Tensor parallelism splits K on the three row-parallel matrices,
// and down_proj becomes K = 17408/4 = 4352. K/16 = 272 uint2 per row, and 272 = 4*64 + 16, so it
// is NOT a whole number of 64-lane iterations and q27_k_nvfp4_w64 above cannot express it. The
// dispatcher therefore dropped it to the generic blocked kernel while gate/up (K=5120, exactly 5
// iterations) kept the fast one. TP makes those two matrices exactly the same size --
// 4352 x 5120 = 5120 x 4352 = 22,282,240 elements -- and down measured 1.53x the per-call cost of
// gate/up. That gap was dispatcher selection, not arithmetic.
//
// This recovers the winning geometry for a ragged K. FULL2 full 64-lane iterations run with the
// SAME loads, the SAME unroll and NO bounds test in the loop body, so the hot path is what the
// authority already measured; the remainder is ONE predicated iteration outside the loop with
// TAIL lanes active. All 64 lanes did the full iterations, so the wave-wide reduction below is
// unchanged and correct.
//   K=4352 -> FULL2=4, TAIL=16.
template<int FULL2, int TAIL, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64t(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, void* __restrict__ Yv,
        const unsigned short* __restrict__ radd = nullptr, float rs = 0.f)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;                        // a whole wave exits together
    const int lane = threadIdx.x & 63;
    const int NG = FULL2 * 64 + TAIL;               // K/16 groups per row
    const int KP = NG * 8;                          // K/2  bytes  per row
    const uint2*         wr = (const uint2*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;

    // Same hoist as q27_k_nvfp4_w64: activation and scale loads join the weight loads, and the four
    // consecutive activation words become one 16-B load. Inactive tail lanes read a VALID address.
    uint2 wv[FULL2];  uint4 xv[FULL2];  float sc[FULL2];
    #pragma unroll
    for (int i = 0; i < FULL2; ++i) {
        const int g = lane + i * 64;
        wv[i] = q27_ld8nt(&wr[g]);
        xv[i] = *(const uint4*)(Xp + (size_t)g * 4);
        sc[i] = q27_e4m3(sr[g]) * Xs[g] * 0.5f;
    }
    const bool act = (lane < TAIL);
    const int  gt  = FULL2 * 64 + (act ? lane : 0);
    const uint2 wt = q27_ld8nt(&wr[gt]);
    const uint4 xt = *(const uint4*)(Xp + (size_t)gt * 4);
    const float st = q27_e4m3(sr[gt]) * Xs[gt] * 0.5f;

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < FULL2; ++i) {
        const uint2 e = make_uint2(xv[i].x, xv[i].y);
        const uint2 o = make_uint2(xv[i].z, xv[i].w);
        acc = fmaf((float)q27_grp16(wv[i], e, o), sc[i], acc);
    }
    if (act) {
        const uint2 e = make_uint2(xt.x, xt.y);
        const uint2 o = make_uint2(xt.z, xt.w);
        acc = fmaf((float)q27_grp16(wt, e, o), st, acc);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) {
        float v = acc * alpha;
        // Residual share, same rationale as the fp8 epilogue: this is a PARTIAL feeding an
        // all-reduce, and a separate launch here would be uniform work on all four cards.
        if (radd) v += rs * q27_bf2f(radd[row]);
        if (OUTBF) ((unsigned short*)Yv)[row] = q27_f2bf(v);   // feeds the BF16 collective
        else       ((float*)Yv)[row]         = v;
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// K=4352 wide: q27_k_nvfp4_w64w's mechanism with q27_k_nvfp4_w64t2's clamped tail.
// Two wide iterations cover 256 of the 272 groups; the last 16 groups run on 16 clamped lanes.
// Q27_K4352W=1 selects it; =0 keeps the w64t2 path for the A/B.
template<int WIDE2, int TAIL, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64tw(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, void* __restrict__ Yv,
        const unsigned short* __restrict__ radd = nullptr, float rs = 0.f)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int NG = WIDE2 * 2 * 64 + TAIL;
    const int KP = NG * 8;
    const uint4*         wr4 = (const uint4*)(W + (size_t)row * (size_t)KP);
    const uint2*         wr2 = (const uint2*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;
    const bool tailact = (lane < TAIL);

    uint4 wv[WIDE2];  uint4 xv0[WIDE2], xv1[WIDE2];
    unsigned short sc2[WIDE2];  uint2 xs2[WIDE2];
    #pragma unroll
    for (int i = 0; i < WIDE2; ++i) {
        const int l2 = lane + i * 64;
        const int g  = 2 * l2;
        wv[i]   = q27_ld16nt_nv(&wr4[l2]);
        xv0[i]  = *(const uint4*)(Xp + (size_t)g * 4);
        xv1[i]  = *(const uint4*)(Xp + (size_t)(g + 1) * 4);
        sc2[i]  = *(const unsigned short*)(sr + g);
        xs2[i]  = *(const uint2*)(Xs + g);
    }
    const int gt = WIDE2 * 2 * 64 + (tailact ? lane : 0);       // clamped narrow tail
    const uint2 wvt = q27_ld8nt(&wr2[gt]);
    const uint4 xvt = *(const uint4*)(Xp + (size_t)gt * 4);
    const unsigned char sct = sr[gt];
    const float xst = Xs[gt];

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < WIDE2; ++i) {
        const uint2 e0 = make_uint2(xv0[i].x, xv0[i].y);
        const uint2 o0 = make_uint2(xv0[i].z, xv0[i].w);
        const uint2 e1 = make_uint2(xv1[i].x, xv1[i].y);
        const uint2 o1 = make_uint2(xv1[i].z, xv1[i].w);
        const float s0 = q27_e4m3((unsigned char)(sc2[i] & 0xFF)) * xs2[i].x * 0.5f;
        const float s1 = q27_e4m3((unsigned char)(sc2[i] >> 8))   * xs2[i].y * 0.5f;
        acc = fmaf((float)q27_grp16(make_uint2(wv[i].x, wv[i].y), e0, o0), s0, acc);
        acc = fmaf((float)q27_grp16(make_uint2(wv[i].z, wv[i].w), e1, o1), s1, acc);
    }
    if (tailact) {
        const uint2 e = make_uint2(xvt.x, xvt.y);
        const uint2 o = make_uint2(xvt.z, xvt.w);
        acc = fmaf((float)q27_grp16(wvt, e, o), q27_e4m3(sct) * xst * 0.5f, acc);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) {
        float v = acc * alpha;
        if (radd) v += rs * q27_bf2f(radd[row]);
        if (OUTBF) ((unsigned short*)Yv)[row] = q27_f2bf(v);
        else       ((float*)Yv)[row]         = v;
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// K=4352 v2. The w64t predicated tail runs the fifth iteration on 16 of 64 lanes, and the ISA
// shows its loads serialized BEHIND the main loop's arithmetic (a quarter-width tail round-trip
// after four groups of depth-2 load pipelines). This form runs FIVE FULL-WIDTH iterations:
// iteration 4 loads CLAMPED addresses on inactive lanes and masks the contribution, so every
// iteration is a uniform 512-B weight burst and the whole load phase can batch before any
// decode. Numerically identical to w64t it replaces (active-lane addresses unchanged).
// Q27_K4352V2=0 restores w64t for the A/B.
template<int FULL2, int TAIL, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64t2(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, void* __restrict__ Yv,
        const unsigned short* __restrict__ radd = nullptr, float rs = 0.f)
{
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int NG = FULL2 * 64 + TAIL;
    const int KP = NG * 8;
    const uint2*         wr = (const uint2*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;
    const bool tailact = (lane < TAIL);

    uint2 wv[FULL2 + 1];  uint4 xv[FULL2 + 1];  unsigned char cg[FULL2 + 1];  float xs[FULL2 + 1];
    #pragma unroll
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int g = i * 64 + ((i < FULL2) ? lane : (tailact ? lane : 0));   // clamped on tail
        wv[i] = q27_ld8nt(&wr[g]);
        xv[i] = *(const uint4*)(Xp + (size_t)g * 4);
        cg[i] = sr[g];
        xs[i] = Xs[g];
    }

    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < FULL2; ++i) {
        const uint2 e = make_uint2(xv[i].x, xv[i].y);
        const uint2 o = make_uint2(xv[i].z, xv[i].w);
        acc = fmaf((float)q27_grp16(wv[i], e, o), q27_e4m3(cg[i]) * xs[i] * 0.5f, acc);
    }
    if (tailact) {
        const uint2 e = make_uint2(xv[FULL2].x, xv[FULL2].y);
        const uint2 o = make_uint2(xv[FULL2].z, xv[FULL2].w);
        acc = fmaf((float)q27_grp16(wv[FULL2], e, o), q27_e4m3(cg[FULL2]) * xs[FULL2] * 0.5f, acc);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) {
        float v = acc * alpha;
        if (radd) v += rs * q27_bf2f(radd[row]);
        if (OUTBF) ((unsigned short*)Yv)[row] = q27_f2bf(v);
        else       ((float*)Yv)[row]         = v;
    }
}

// Blocked form for large K (K=17408 -> 17 uint4/lane; fully unrolled that blows VGPR).
// Outer loop is `#pragma unroll 1` — capped, load-bearing — but the body still issues B
// independent loads before the first decode, preserving the one-wait dependency graph.
template<int B>
__global__ __launch_bounds__(Q27_NVFP4_BLOCK) void q27_k_nvfp4_blocked(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, int KP, int NG, int NU4, float* __restrict__ Y)
{
#ifdef Q27_EXPERIMENTS
    const int row = blockIdx.x * Q27_NVFP4_ROWS + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const uint4*         wr = (const uint4*)(W + (size_t)row * (size_t)KP);
    const unsigned char* sr = S + (size_t)row * (size_t)NG;

    float acc = 0.f;
    int u = lane;
    const int nfull = NU4 / (32 * B);
    #pragma unroll 1
    for (int c = 0; c < nfull; ++c) {
        uint4 wv[B];
        #pragma unroll
        for (int i = 0; i < B; ++i) wv[i] = q27_ld16nt_nv(&wr[u + i * 32]);
        #pragma unroll
        for (int i = 0; i < B; ++i) acc = q27_pair32(wv[i], u + i * 32, sr, Xp, Xs, acc);
        u += 32 * B;
    }
    // Remainder. RAGGED K IS SUPPORTED HERE and this loop is the mechanism, not an edge case.
    // This comment previously asserted "K is a multiple of 1024 (every NVFP4 tensor in this
    // checkpoint)", which was true when written and FALSE 8h32m later: four-way TP splits K on the
    // row-parallel matrices, giving down_proj K = 17408/4 = 4352 (NU4 = 136, not a multiple of 32).
    // The CODE was always correct -- the `u < NU4` guard covers each uint4 exactly once and every
    // lane reaches the reduction below -- only the reasoning around it went stale. What actually
    // happens at a ragged K is lane divergence in this tail: at NU4=136 with B=4 the unrolled loop
    // above runs once (u -> 128) and 8 of 32 lanes execute one further iteration, so tail lane
    // occupancy is 25%. That is a cost, not a defect. K=4352 no longer reaches this kernel anyway
    // (q27_k_nvfp4_w64t handles it), but any future ragged K will, and it will be correct.
    #pragma unroll 1
    for (; u < NU4; u += 32) acc = q27_pair32(wr[u], u, sr, Xp, Xs, acc);

    #pragma unroll
    for (int o = 16; o; o >>= 1) acc += __shfl_down(acc, o, 32);
    if (lane == 0) Y[row] = acc * alpha;

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// =============================================================================================
//  q27_quant_perm kernel
// =============================================================================================
#ifdef Q27_QUANT_E4M3_SCALE
// Round a non-negative float to the nearest OCP e4m3fn magnitude (subnormals down to 2^-9,
// max finite 448). Only used for the A/B against the model's own scale encoding.
__device__ __forceinline__ float q27_e4m3_round(float v)
{
    if (!(v > 0.f))   return 0.f;
    if (v >= 448.f)   return 448.f;
    int ex; frexpf(v, &ex);                     // v = f * 2^ex, f in [0.5, 1)
    int se = ex - 4;                            // step = 2^(binade-3), binade = ex-1
    if (se < -9) se = -9;                       // subnormal floor
    const float step = ldexpf(1.0f, se);
    const float r = rintf(v / step) * step;     // rintf is ties-to-even
    return (r > 448.f) ? 448.f : r;
}
#endif

// One thread owns one group of 16. The absmax is therefore purely thread-local: no cross-lane
// reduction, no LDS. (q27.h: never stage an operand in LDS without a second consumer.)
// VEC==1 takes the 32B-in / 16B-out vector path; VEC==0 is the unaligned fallback.
template<int VEC>
__global__ __launch_bounds__(256) void q27_k_quant_perm(
        const unsigned short* __restrict__ X, signed char* __restrict__ Q,
        float* __restrict__ XS, int ngrp, float gscale)
{
    const int g = blockIdx.x * 256 + threadIdx.x;
    if (g >= ngrp) return;

    float v[16];
    if (VEC) {
        const uint4* p = (const uint4*)(X + ((size_t)g << 4));   // 16 bf16 = 32 B, 16B-aligned
        const uint4 a = p[0], b = p[1];
        const unsigned s[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[2 * i + 0] = q27_bf2f((unsigned short)(s[i] & 0xFFFFu)) * gscale;
            v[2 * i + 1] = q27_bf2f((unsigned short)(s[i] >> 16))     * gscale;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = q27_bf2f(X[((size_t)g << 4) + j]) * gscale;
    }

    float mx = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) mx = fmaxf(mx, fabsf(v[j]));

    float sc = mx * (1.0f / 127.0f);
#ifdef Q27_QUANT_E4M3_SCALE
    sc = q27_e4m3_round(sc);
#endif
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;   // all-zero group -> scale 0, codes 0
    XS[g] = sc;

    // EVEN-THEN-ODD inside the group (q27.h ACTIVATION PERMUTATION CONTRACT):
    //     dst[(g<<4) + (j&1 ? 8+(j>>1) : (j>>1))] = src[g*16 + j]
    // j is a compile-time constant here, so the destination dword and byte shift fold away and
    // the 16 bytes assemble in registers.
    unsigned d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        int q = (int)rintf(v[j] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);      // |v| <= mx makes this defensive only
        const unsigned byte = (unsigned)(q & 0xFF);
        const int d  = (j & 1) ? (8 + (j >> 1)) : (j >> 1);
        const int sh = (d & 3) * 8;
        if      ((d >> 2) == 0) d0 |= byte << sh;
        else if ((d >> 2) == 1) d1 |= byte << sh;
        else if ((d >> 2) == 2) d2 |= byte << sh;
        else                    d3 |= byte << sh;
    }

    signed char* q8 = Q + ((size_t)g << 4);
    if (VEC) {
        *(uint4*)q8 = make_uint4(d0, d1, d2, d3);
    } else {
        const unsigned dd[4] = { d0, d1, d2, d3 };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            q8[i * 4 + 0] = (signed char)(dd[i]      );
            q8[i * 4 + 1] = (signed char)(dd[i] >>  8);
            q8[i * 4 + 2] = (signed char)(dd[i] >> 16);
            q8[i * 4 + 3] = (signed char)(dd[i] >> 24);
        }
    }
}

// =============================================================================================
//  host entry points  (declared extern "C" by q27.h)
// =============================================================================================

// x_bf16 [n] BF16 -> xq_perm [n] int8 (even-then-odd per 16) + xs [n/16] fp32.
// Contract: n % 16 == 0. `in_scale` is the tensor's STATIC input_scale from the checkpoint; the
// emitted pair satisfies  x[k] / in_scale == xq_perm[perm(k)] * xs[k>>4], so q27_proj_nvfp4's
// epilogue alpha = in_scale * ws2 completes the composition.
void q27_quant_perm(const unsigned short* x_bf16, signed char* xq_perm, float* xs,
                    int n, float in_scale, hipStream_t s)
{
    if (n <= 0 || (n & 15)) {
        fprintf(stderr, "q27_quant_perm: n=%d must be positive and a multiple of 16\n", n);
        return;
    }
    if (!(in_scale > 0.f)) {
        fprintf(stderr, "q27_quant_perm: in_scale=%g is not positive (checkpoint range "
                        "0.0013079..0.602679; never 1.0, never 0)\n", (double)in_scale);
        return;
    }
    const int ngrp = n >> 4;
    const dim3 grid((unsigned)((ngrp + 255) / 256)), blk(256);
    const uintptr_t mix = (uintptr_t)x_bf16 | (uintptr_t)xq_perm;
    if ((mix & 15u) == 0u)
        hipLaunchKernelGGL((q27_k_quant_perm<1>), grid, blk, 0, s,
                           x_bf16, xq_perm, xs, ngrp, 1.0f / in_scale);
    else
        hipLaunchKernelGGL((q27_k_quant_perm<0>), grid, blk, 0, s,
                           x_bf16, xq_perm, xs, ngrp, 1.0f / in_scale);
}

// y[rows] = W x, W in NVFP4. `w` is a HOST-resident handle whose .w/.gs are device pointers
// (q27_load.h populates it that way). xq_perm/xs come from q27_quant_perm on the same activation.
// Set immediately before the down_proj call that feeds an all-reduce, cleared after. Each card's
// decode thread owns its own device and its own stream, and these are read only inside that
// thread's launch, so they are per-thread state, not shared.
static thread_local const unsigned short* g_nv_radd = nullptr;
static thread_local float                 g_nv_rs   = 0.f;
extern "C" void q27_nvfp4_set_res(const unsigned short* radd, float rs) {
    g_nv_radd = radd; g_nv_rs = rs;
}


// =============================================================================================
// SMALL-M PREFILL MLP KERNELS (Q27_PF_MLP2). The batched kernels above keep one wave per ROW and
// loop over the C=8 positions inside it, so every row's wave re-loads all eight activation vectors
// from L2 (40 KB per row, ~350 MB per gate+up call) and re-decodes each weight group's nibbles once
// per position. Measured 212 us per 8-position call for a 22 MB weight stream (105 GB/s) against
// 38 us for the single-position kernel. Here a block stages one K-chunk of all eight activations
// (8 KB + 2 KB of scales) in LDS, every wave handles R rows against it, and each weight group is
// decoded ONCE into four dwords before the eight sdot4 chains. The per-position arithmetic and
// its order are unchanged (same q27_dq4/q27_d4 sequence per group, same fmaf order over groups,
// same wave reduction), so the outputs are bit-identical to q27_k_nvfp4_w64_gu_b / w64t2_b.
template <int UPL2, int C, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_gu_b2(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        __syncthreads();                                   // previous chunk fully consumed
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, g = t & 63;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const int g = i * 64 + lane;
        uint2 vg[R], vu[R]; unsigned cg[R], cu[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            vg[r] = q27_ld8nt(WG + (size_t)rr * KP + (size_t)g * 8);
            vu[r] = q27_ld8nt(WU + (size_t)rr * KP + (size_t)g * 8);
            cg[r] = SG[(size_t)rr * NG + g]; cu[r] = SU[(size_t)rr * NG + g];
        }
        uint4 xv[C]; float xs[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { xv[c] = sx[c][lane]; xs[c] = ss[c][lane]; }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned g0 = q27_dq4( vg[r].x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg[r].y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg[r].y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu[r].x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu[r].y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu[r].y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg[r]), suf = q27_e4m3(cu[r]);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                int a = q27_d4(g0, xv[c].x, 0); a = q27_d4(g1, xv[c].z, a); a = q27_d4(g2, xv[c].y, a); a = q27_d4(g3, xv[c].w, a);
                int b = q27_d4(u0, xv[c].x, 0); b = q27_d4(u1, xv[c].z, b); b = q27_d4(u2, xv[c].y, b); b = q27_d4(u3, xv[c].w, b);
                accg[r][c] = fmaf((float)a, sgf * xs[c], accg[r][c]);
                accu[r][c] = fmaf((float)b, suf * xs[c], accu[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
#ifdef Q27_EXPERIMENTS
// v3 (Q27_PF_MLP3): gu_b2 with the weight fetches of ALL k-chunks issued up front. b2 fetched one
// chunk's 8 bytes per row per k-chunk and stalled on HBM latency five times per wave with ~0.3 us
// of arithmetic between stalls (the 128-token trace: 115 us per call against a 30 us stream floor,
// ~20% of issue peak). Same q27_dq4/q27_d4 sequence per group, same fmaf order over groups, same
// wave reduction: outputs bit-identical to b2. Scales are prefetched one chunk ahead.
template <int UPL2, int C, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_gu_b3(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    uint2 vg[UPL2][R], vu[UPL2][R];
#pragma unroll
    for (int i = 0; i < UPL2; ++i)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int g = i * 64 + lane;
            vg[i][r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)g * 8);
            vu[i][r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)g * 8);
        }
    unsigned cg[R], cu[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { cg[r] = SG[(size_t)rr[r] * NG + lane]; cu[r] = SU[(size_t)rr[r] * NG + lane]; }
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, g = t & 63;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        unsigned cgn[R], cun[R];
        if (i + 1 < UPL2) {
#pragma unroll
            for (int r = 0; r < R; ++r) { cgn[r] = SG[(size_t)rr[r] * NG + (i + 1) * 64 + lane]; cun[r] = SU[(size_t)rr[r] * NG + (i + 1) * 64 + lane]; }
        }
        unsigned g0[R], g1[R], g2[R], g3[R], u0[R], u1[R], u2[R], u3[R]; float sgf[R], suf[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            g0[r] = q27_dq4( vg[i][r].x       & 0x0F0F0F0Fu); g1[r] = q27_dq4((vg[i][r].x >> 4) & 0x0F0F0F0Fu);
            g2[r] = q27_dq4( vg[i][r].y       & 0x0F0F0F0Fu); g3[r] = q27_dq4((vg[i][r].y >> 4) & 0x0F0F0F0Fu);
            u0[r] = q27_dq4( vu[i][r].x       & 0x0F0F0F0Fu); u1[r] = q27_dq4((vu[i][r].x >> 4) & 0x0F0F0F0Fu);
            u2[r] = q27_dq4( vu[i][r].y       & 0x0F0F0F0Fu); u3[r] = q27_dq4((vu[i][r].y >> 4) & 0x0F0F0F0Fu);
            sgf[r] = q27_e4m3(cg[r]); suf[r] = q27_e4m3(cu[r]);
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int a = q27_d4(g0[r], xv.x, 0); a = q27_d4(g1[r], xv.z, a); a = q27_d4(g2[r], xv.y, a); a = q27_d4(g3[r], xv.w, a);
                int b = q27_d4(u0[r], xv.x, 0); b = q27_d4(u1[r], xv.z, b); b = q27_d4(u2[r], xv.y, b); b = q27_d4(u3[r], xv.w, b);
                accg[r][c] = fmaf((float)a, sgf[r] * xsc, accg[r][c]);
                accu[r][c] = fmaf((float)b, suf[r] * xsc, accu[r][c]);
            }
        }
        if (i + 1 < UPL2) {
#pragma unroll
            for (int r = 0; r < R; ++r) { cg[r] = cgn[r]; cu[r] = cun[r]; }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b3(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > 8) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    if (R == 1) { const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b3<5, 8, 1>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (R == 2) { const dim3 g((unsigned)((wg->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b3<5, 8, 2>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}

#else   // closed experiments compiled out (see docs/AMD_REFERENCE_MINING.md); stubs keep the flag chains linkable
extern "C" int q27_proj_nvfp4_gu_b3(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int R, hipStream_t s) { (void)0; return 0; }
#endif
// v5 (Q27_PF_MLP5): b2's exact loop, minus the per-chunk register cache of the eight staged
// activations (xv[8] = 32 VGPRs, xs[8] = 8): each is read from LDS where it is used. The resource
// report puts b2<5,8,2> at 119 VGPRs = 2 waves/SIMD and every heavier variant at 1 wave; the census
// says the kernel is latency-bound, so occupancy is the lever. __launch_bounds__ asks for 3 waves.
// Same q27_dq4/q27_d4 sequence, same fmaf order, same reduction: bit-identical to b2.
template <int UPL2, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b5(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, g = t & 63;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const int g = i * 64 + lane;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            const uint2 vg = q27_ld8nt(WG + (size_t)rr * KP + (size_t)g * 8);
            const uint2 vu = q27_ld8nt(WU + (size_t)rr * KP + (size_t)g * 8);
            const unsigned cg = SG[(size_t)rr * NG + g], cu = SU[(size_t)rr * NG + g];
            const unsigned g0 = q27_dq4( vg.x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg.x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg.y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg.y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu.x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu.x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu.y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu.y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg), suf = q27_e4m3(cu);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
                int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                accg[r][c] = fmaf((float)a_, sgf * xsc, accg[r][c]);
                accu[r][c] = fmaf((float)b_, suf * xsc, accu[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b5(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    if (R == 1) { const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b5<5, Q27_PF_MC, 1, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (R == 2) { const dim3 g((unsigned)((wg->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b5<5, Q27_PF_MC, 2, 3>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}
template <int FULL2, int TAIL, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b5(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const bool act = (i < FULL2) || (lane < TAIL);
        const int g = i * 64 + (act ? lane : 0);
        const int lsel = act ? lane : 0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            const uint2 wv = q27_ld8nt(W + (size_t)rr * KP + (size_t)g * 8);
            const unsigned cs = S[(size_t)rr * NG + g];
            if (act) {
                const unsigned w0 = q27_dq4( wv.x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv.x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv.y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv.y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lsel]; const float xsc = ss[c][lsel];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[r][c] = fmaf((float)a_, sf * xsc, acc[r][c]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
    }
}
extern "C" int q27_proj_nvfp4_bt5(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, int R, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b5<4, 16, Q27_PF_MC, 1, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    if (R == 2) { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b5<4, 16, Q27_PF_MC, 2, 3>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

#ifdef Q27_EXPERIMENTS
// v6 (Q27_PF_MLP6): b5's loop with TWO k-chunks staged per iteration (128 groups per position in
// LDS, 20 KB), so each wave fetches 16 B of weights per row per stage instead of 8 and the block
// pays half the barriers. Per (row, position) the groups are consumed in the same order as b2's
// consecutive chunks, so the fmaf chain is unchanged: bit-identical. R=1 keeps ~4 waves/SIMD.
template <int UPL2, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b6(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][128];
    __shared__ float ss[C][128];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; i += 2) {
        const int ng = (i + 1 < UPL2) ? 128 : 64;             // groups staged this iteration
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            const int ga = i * 64 + lane, gb = ga + 64;
            const bool hasb = (i + 1 < UPL2);
            const uint2 vga = q27_ld8nt(WG + (size_t)rr * KP + (size_t)ga * 8);
            const uint2 vua = q27_ld8nt(WU + (size_t)rr * KP + (size_t)ga * 8);
            const uint2 vgb = q27_ld8nt(WG + (size_t)rr * KP + (size_t)(hasb ? gb : ga) * 8);
            const uint2 vub = q27_ld8nt(WU + (size_t)rr * KP + (size_t)(hasb ? gb : ga) * 8);
            const unsigned cga = SG[(size_t)rr * NG + ga], cua = SU[(size_t)rr * NG + ga];
            const unsigned cgb = SG[(size_t)rr * NG + (hasb ? gb : ga)], cub = SU[(size_t)rr * NG + (hasb ? gb : ga)];
            {   // chunk i
                const unsigned g0 = q27_dq4( vga.x       & 0x0F0F0F0Fu), g1 = q27_dq4((vga.x >> 4) & 0x0F0F0F0Fu);
                const unsigned g2 = q27_dq4( vga.y       & 0x0F0F0F0Fu), g3 = q27_dq4((vga.y >> 4) & 0x0F0F0F0Fu);
                const unsigned u0 = q27_dq4( vua.x       & 0x0F0F0F0Fu), u1 = q27_dq4((vua.x >> 4) & 0x0F0F0F0Fu);
                const unsigned u2 = q27_dq4( vua.y       & 0x0F0F0F0Fu), u3 = q27_dq4((vua.y >> 4) & 0x0F0F0F0Fu);
                const float sgf = q27_e4m3(cga), suf = q27_e4m3(cua);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
                    int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                    int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                    accg[r][c] = fmaf((float)a_, sgf * xsc, accg[r][c]);
                    accu[r][c] = fmaf((float)b_, suf * xsc, accu[r][c]);
                }
            }
            if (hasb) {   // chunk i+1, consumed after chunk i exactly as b2's next iteration would
                const unsigned g0 = q27_dq4( vgb.x       & 0x0F0F0F0Fu), g1 = q27_dq4((vgb.x >> 4) & 0x0F0F0F0Fu);
                const unsigned g2 = q27_dq4( vgb.y       & 0x0F0F0F0Fu), g3 = q27_dq4((vgb.y >> 4) & 0x0F0F0F0Fu);
                const unsigned u0 = q27_dq4( vub.x       & 0x0F0F0F0Fu), u1 = q27_dq4((vub.x >> 4) & 0x0F0F0F0Fu);
                const unsigned u2 = q27_dq4( vub.y       & 0x0F0F0F0Fu), u3 = q27_dq4((vub.y >> 4) & 0x0F0F0F0Fu);
                const float sgf = q27_e4m3(cgb), suf = q27_e4m3(cub);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][64 + lane]; const float xsc = ss[c][64 + lane];
                    int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                    int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                    accg[r][c] = fmaf((float)a_, sgf * xsc, accg[r][c]);
                    accu[r][c] = fmaf((float)b_, suf * xsc, accu[r][c]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b6(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    if (R == 1) { const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b6<5, Q27_PF_MC, 1, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}
// down v6: full chunks in pairs, then the TAIL chunk alone (lanes >= TAIL fetch group i*64 and skip).
template <int FULL2, int TAIL, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b6(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][128];
    __shared__ float ss[C][128];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; i += 2) {
        const bool tail_only = (i >= FULL2);                    // this iteration is the TAIL chunk
        const bool hasb = (!tail_only) && (i + 1 < FULL2);      // second full chunk staged
        const int ng = tail_only ? TAIL : (hasb ? 128 : 64);
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const bool act = (!tail_only) || (lane < TAIL);
        const int lsel = act ? lane : 0;
        const int ga = i * 64 + lsel, gb = ga + 64;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            const uint2 wva = q27_ld8nt(W + (size_t)rr * KP + (size_t)ga * 8);
            const uint2 wvb = q27_ld8nt(W + (size_t)rr * KP + (size_t)(hasb ? gb : ga) * 8);
            const unsigned csa = S[(size_t)rr * NG + ga], csb = S[(size_t)rr * NG + (hasb ? gb : ga)];
            if (act) {
                const unsigned w0 = q27_dq4( wva.x       & 0x0F0F0F0Fu), w1 = q27_dq4((wva.x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wva.y       & 0x0F0F0F0Fu), w3 = q27_dq4((wva.y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(csa);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lsel]; const float xsc = ss[c][lsel];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[r][c] = fmaf((float)a_, sf * xsc, acc[r][c]);
                }
            }
            if (hasb) {
                const unsigned w0 = q27_dq4( wvb.x       & 0x0F0F0F0Fu), w1 = q27_dq4((wvb.x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wvb.y       & 0x0F0F0F0Fu), w3 = q27_dq4((wvb.y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(csb);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][64 + lane]; const float xsc = ss[c][64 + lane];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[r][c] = fmaf((float)a_, sf * xsc, acc[r][c]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
    }
}
extern "C" int q27_proj_nvfp4_bt6(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, int R, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b6<4, 16, Q27_PF_MC, 1, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    if (R == 2) { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b6<4, 16, Q27_PF_MC, 2, 3>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

// v7 (Q27_PF_MLP7): the persistent-row form. A block stages ALL k-chunks of the chunk's activations
// once (50 KB of LDS -> one block per CU, one wave per SIMD) and each wave then streams a run of
// rows back to back with the NEXT row's five weight fetches in flight while the current row's
// arithmetic runs. No per-chunk barrier, no LDS re-staging, a deep in-wave pipeline instead of
// wave count: the one design on this list that trades occupancy for bytes in flight on purpose.
// Grid = ceil(rows / (4 * RPW)); the launcher picks RPW so the grid is one round over the CUs.
// Per (row, position) the group order and the fmaf chain are b2's: bit-identical.
template <int UPL2, int C>
__global__ __launch_bounds__(256, 1) void q27_k_nvfp4_gu_b7(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, int rpw, float* __restrict__ YG, float* __restrict__ YU)
{
    constexpr int NG = UPL2 * 64;
    __shared__ uint4 sx[C][NG];
    __shared__ float ss[C][NG];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int KP = NG * 8;
    for (int t = threadIdx.x; t < C * NG; t += 256) {
        const int c = t / NG, g = t % NG;
        sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)g * 4);
        ss[c][g] = Xs[(size_t)c * NG + g] * 0.5f;
    }
    __syncthreads();
    const int rbeg = (blockIdx.x * 4 + wave) * rpw;
    if (rbeg >= rows) return;
    const int rend = (rbeg + rpw < rows) ? (rbeg + rpw) : rows;
    // weights of the current row (all k-chunks) and the next row, in flight
    uint2 vg[UPL2], vu[UPL2], nvg[UPL2], nvu[UPL2]; unsigned cg[UPL2], cu[UPL2], ncg[UPL2], ncu[UPL2];
    auto fetch = [&](int row, uint2* g_, uint2* u_, unsigned* cg_, unsigned* cu_) {
#pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            const int g = i * 64 + lane;
            g_[i] = q27_ld8nt(WG + (size_t)row * KP + (size_t)g * 8);
            u_[i] = q27_ld8nt(WU + (size_t)row * KP + (size_t)g * 8);
            cg_[i] = SG[(size_t)row * NG + g]; cu_[i] = SU[(size_t)row * NG + g];
        }
    };
    fetch(rbeg, vg, vu, cg, cu);
    for (int row = rbeg; row < rend; ++row) {
        if (row + 1 < rend) fetch(row + 1, nvg, nvu, ncg, ncu);
        float accg[C], accu[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[c] = 0.f; accu[c] = 0.f; }
#pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            const unsigned g0 = q27_dq4( vg[i].x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg[i].x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg[i].y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg[i].y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu[i].x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu[i].x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu[i].y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu[i].y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg[i]), suf = q27_e4m3(cu[i]);
            const int g = i * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][g]; const float xsc = ss[c][g];
                int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                accg[c] = fmaf((float)a_, sgf * xsc, accg[c]);
                accu[c] = fmaf((float)b_, suf * xsc, accu[c]);
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[c], vu2 = accu[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
        if (row + 1 < rend) {
#pragma unroll
            for (int i = 0; i < UPL2; ++i) { vg[i] = nvg[i]; vu[i] = nvu[i]; cg[i] = ncg[i]; cu[i] = ncu[i]; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b7(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int rpw_req, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const int ncu = 60;
    int rpw = (rpw_req > 0) ? rpw_req : (wg->rows + 4 * ncu - 1) / (4 * ncu);   // one round over the CUs
    const int nblk = (wg->rows + 4 * rpw - 1) / (4 * rpw);
    hipLaunchKernelGGL((q27_k_nvfp4_gu_b7<5, Q27_PF_MC>), dim3(nblk), dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs,
                       (const unsigned*)xq_perm, xs, ag, au, wg->rows, rpw, yg, yu);
    return 1;
}
// down v7: same form over FULL2 full chunks + the TAIL chunk; lanes >= TAIL skip the tail group.
template <int FULL2, int TAIL, int C>
__global__ __launch_bounds__(256, 1) void q27_k_nvfp4_t2_b7(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, int rpw, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    constexpr int NG = FULL2 * 64 + TAIL;
    constexpr int NCH = FULL2 + 1;
    __shared__ uint4 sx[C][NG];
    __shared__ float ss[C][NG];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4;
    for (int t = threadIdx.x; t < C * NG; t += 256) {
        const int c = t / NG, g = t % NG;
        sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)g * 4);
        ss[c][g] = Xs[(size_t)c * NG + g] * 0.5f;
    }
    __syncthreads();
    const int rbeg = (blockIdx.x * 4 + wave) * rpw;
    if (rbeg >= rows) return;
    const int rend = (rbeg + rpw < rows) ? (rbeg + rpw) : rows;
    uint2 wv[NCH], nwv[NCH]; unsigned cs[NCH], ncs[NCH];
    auto fetch = [&](int row, uint2* w_, unsigned* c_) {
#pragma unroll
        for (int i = 0; i < NCH; ++i) {
            const bool act = (i < FULL2) || (lane < TAIL);
            const int g = i * 64 + (act ? lane : 0);
            w_[i] = q27_ld8nt(W + (size_t)row * KP + (size_t)g * 8);
            c_[i] = S[(size_t)row * NG + g];
        }
    };
    fetch(rbeg, wv, cs);
    for (int row = rbeg; row < rend; ++row) {
        if (row + 1 < rend) fetch(row + 1, nwv, ncs);
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
        for (int i = 0; i < NCH; ++i) {
            const bool act = (i < FULL2) || (lane < TAIL);
            if (act) {
                const unsigned w0 = q27_dq4( wv[i].x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv[i].x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv[i].y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv[i].y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs[i]);
                const int g = i * 64 + lane;
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][g]; const float xsc = ss[c][g];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[c] = fmaf((float)a_, sf * xsc, acc[c]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
        if (row + 1 < rend) {
#pragma unroll
            for (int i = 0; i < NCH; ++i) { wv[i] = nwv[i]; cs[i] = ncs[i]; }
        }
    }
}
extern "C" int q27_proj_nvfp4_bt7(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, int rpw_req, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    const int ncu = 60;
    int rpw = (rpw_req > 0) ? rpw_req : (w->rows + 4 * ncu - 1) / (4 * ncu);
    const int nblk = (w->rows + 4 * rpw - 1) / (4 * rpw);
    hipLaunchKernelGGL((q27_k_nvfp4_t2_b7<4, 16, Q27_PF_MC>), dim3(nblk), dim3(256), 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                       alpha, w->rows, rpw, y, radd, rs);
    return 1;
}

// v8 (Q27_PF_MLP8): b5 (no register cache of the staged activations) plus double-buffered staging:
// the NEXT chunk's global loads are issued before this chunk's arithmetic and written to the other
// LDS buffer after it, one barrier per chunk. b4 tried this on top of b2's register caches and
// reached 157 VGPRs with spills; here the live state across the arithmetic is two uint4 and two
// floats per thread plus the next chunk's weights. Bit-identical to b2.
template <int UPL2, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b8(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    static_assert(C * 64 == 512, "the staging assumes C = 8 (two uint4 per thread per chunk)");
    __shared__ uint4 sx[2][C][64];
    __shared__ float ss[2][C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    // staging share of this thread: entries t0 = tid and t1 = tid + 256 of the C*64 grid
    const int c0 = threadIdx.x >> 6, g0 = threadIdx.x & 63, c1 = c0 + 4, g1 = g0;
    // chunk 0 synchronously
    sx[0][c0][g0] = *(const uint4*)(Xp + (size_t)c0 * (UPL2 * 256) + (size_t)g0 * 4);
    ss[0][c0][g0] = Xs[(size_t)c0 * NG + g0] * 0.5f;
    sx[0][c1][g1] = *(const uint4*)(Xp + (size_t)c1 * (UPL2 * 256) + (size_t)g1 * 4);
    ss[0][c1][g1] = Xs[(size_t)c1 * NG + g1] * 0.5f;
    uint2 vg[R], vu[R]; unsigned cg[R], cu[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        vg[r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)lane * 8);
        vu[r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)lane * 8);
        cg[r] = SG[(size_t)rr[r] * NG + lane]; cu[r] = SU[(size_t)rr[r] * NG + lane];
    }
    __syncthreads();
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        const int cur = i & 1;
        const bool more = (i + 1 < UPL2);
        uint4 nx0 = make_uint4(0u,0u,0u,0u), nx1 = nx0; float ns0 = 0.f, ns1 = 0.f;
        uint2 nvg[R], nvu[R]; unsigned ncg[R], ncu[R];
        if (more) {
            const int gn = (i + 1) * 64;
            nx0 = *(const uint4*)(Xp + (size_t)c0 * (UPL2 * 256) + (size_t)(gn + g0) * 4); ns0 = Xs[(size_t)c0 * NG + gn + g0] * 0.5f;
            nx1 = *(const uint4*)(Xp + (size_t)c1 * (UPL2 * 256) + (size_t)(gn + g1) * 4); ns1 = Xs[(size_t)c1 * NG + gn + g1] * 0.5f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                nvg[r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)(gn + lane) * 8);
                nvu[r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)(gn + lane) * 8);
                ncg[r] = SG[(size_t)rr[r] * NG + gn + lane]; ncu[r] = SU[(size_t)rr[r] * NG + gn + lane];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned w0 = q27_dq4( vg[r].x       & 0x0F0F0F0Fu), w1 = q27_dq4((vg[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned w2 = q27_dq4( vg[r].y       & 0x0F0F0F0Fu), w3 = q27_dq4((vg[r].y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu[r].x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu[r].y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu[r].y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg[r]), suf = q27_e4m3(cu[r]);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[cur][c][lane]; const float xsc = ss[cur][c][lane];
                int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                accg[r][c] = fmaf((float)a_, sgf * xsc, accg[r][c]);
                accu[r][c] = fmaf((float)b_, suf * xsc, accu[r][c]);
            }
        }
        if (more) {
            sx[cur ^ 1][c0][g0] = nx0; ss[cur ^ 1][c0][g0] = ns0;
            sx[cur ^ 1][c1][g1] = nx1; ss[cur ^ 1][c1][g1] = ns1;
#pragma unroll
            for (int r = 0; r < R; ++r) { vg[r] = nvg[r]; vu[r] = nvu[r]; cg[r] = ncg[r]; cu[r] = ncu[r]; }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b8(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > 8) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    if (R == 1) { const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b8<5, 8, 1, 3>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}

// v9 (Q27_PF_MLP9): b5 with ONLY the weight fetch pipelined one chunk ahead (two uint2 and two
// scale bytes per row: ~6 VGPRs). The activation staging stays synchronous. b5 issues its 8-byte
// weight loads at the top of each chunk, after the barrier, so an HBM latency sits in front of every
// chunk's arithmetic; here the loads for chunk i+1 are in flight during chunk i. Bit-identical to b2.
template <int UPL2, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b9(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    uint2 vg[R], vu[R]; unsigned cg[R], cu[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        vg[r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)lane * 8);
        vu[r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)lane * 8);
        cg[r] = SG[(size_t)rr[r] * NG + lane]; cu[r] = SU[(size_t)rr[r] * NG + lane];
    }
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, g = t & 63;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        uint2 nvg[R], nvu[R]; unsigned ncg[R], ncu[R];
        const bool more = (i + 1 < UPL2);
        if (more) {
            const int gn = (i + 1) * 64 + lane;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                nvg[r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)gn * 8);
                nvu[r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)gn * 8);
                ncg[r] = SG[(size_t)rr[r] * NG + gn]; ncu[r] = SU[(size_t)rr[r] * NG + gn];
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned g0 = q27_dq4( vg[r].x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg[r].y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg[r].y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu[r].x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu[r].y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu[r].y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg[r]), suf = q27_e4m3(cu[r]);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
                int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                accg[r][c] = fmaf((float)a_, sgf * xsc, accg[r][c]);
                accu[r][c] = fmaf((float)b_, suf * xsc, accu[r][c]);
            }
        }
        if (more) {
#pragma unroll
            for (int r = 0; r < R; ++r) { vg[r] = nvg[r]; vu[r] = nvu[r]; cg[r] = ncg[r]; cu[r] = ncu[r]; }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b9(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    if (R == 1) { const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b9<5, Q27_PF_MC, 1, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}
// down v9: the same weight prefetch on t2_b5 (full chunks, then the TAIL chunk).
template <int FULL2, int TAIL, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b9(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    uint2 wv[R]; unsigned cs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { wv[r] = q27_ld8nt(W + (size_t)rr[r] * KP + (size_t)lane * 8); cs[r] = S[(size_t)rr[r] * NG + lane]; }
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        uint2 nwv[R]; unsigned ncs[R];
        const bool more = (i + 1 < FULL2 + 1);
        if (more) {
            const bool actn = (i + 1 < FULL2) || (lane < TAIL);
            const int gn = (i + 1) * 64 + (actn ? lane : 0);
#pragma unroll
            for (int r = 0; r < R; ++r) { nwv[r] = q27_ld8nt(W + (size_t)rr[r] * KP + (size_t)gn * 8); ncs[r] = S[(size_t)rr[r] * NG + gn]; }
        }
        __syncthreads();
        const bool act = (i < FULL2) || (lane < TAIL);
        const int lsel = act ? lane : 0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (act) {
                const unsigned w0 = q27_dq4( wv[r].x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv[r].x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv[r].y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv[r].y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs[r]);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lsel]; const float xsc = ss[c][lsel];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[r][c] = fmaf((float)a_, sf * xsc, acc[r][c]);
                }
            }
        }
        if (more) {
#pragma unroll
            for (int r = 0; r < R; ++r) { wv[r] = nwv[r]; cs[r] = ncs[r]; }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
    }
}
extern "C" int q27_proj_nvfp4_bt9(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, int R, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b9<4, 16, Q27_PF_MC, 1, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    if (R == 2) { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b9<4, 16, Q27_PF_MC, 2, 3>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

// v10 (Q27_PF_MLP10): b5 with the chunk's weight fetch issued BEFORE the activation staging of the
// same chunk, so the HBM latency of the weights overlaps the L2 latency of the staging instead of
// following the barrier. No new live state (the same vg/vu/cg/cu registers, loaded earlier), so the
// register shape is b5's; bit-identical to b2.
template <int UPL2, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b10(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        const int g = i * 64 + lane;
        uint2 vg[R], vu[R]; unsigned cg[R], cu[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {                     // weights first: their latency overlaps the staging below
            vg[r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)g * 8);
            vu[r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)g * 8);
            cg[r] = SG[(size_t)rr[r] * NG + g]; cu[r] = SU[(size_t)rr[r] * NG + g];
        }
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, gg = t & 63;
            sx[c][gg] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + gg) * 4);
            ss[c][gg] = Xs[(size_t)c * NG + i * 64 + gg] * 0.5f;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned g0 = q27_dq4( vg[r].x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg[r].y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg[r].y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu[r].x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu[r].y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu[r].y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg[r]), suf = q27_e4m3(cu[r]);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
                int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                accg[r][c] = fmaf((float)a_, sgf * xsc, accg[r][c]);
                accu[r][c] = fmaf((float)b_, suf * xsc, accu[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b10(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                     const signed char* xq_perm, const float* xs,
                                     float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    if (R == 1) { const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b10<5, Q27_PF_MC, 1, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}
template <int FULL2, int TAIL, int C, int R, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b10(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        const bool act = (i < FULL2) || (lane < TAIL);
        const int g = i * 64 + (act ? lane : 0);
        const int lsel = act ? lane : 0;
        uint2 wv[R]; unsigned cs[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { wv[r] = q27_ld8nt(W + (size_t)rr[r] * KP + (size_t)g * 8); cs[r] = S[(size_t)rr[r] * NG + g]; }
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, gg = t % ng;
            sx[c][gg] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + gg) * 4);
            ss[c][gg] = Xs[(size_t)c * NGS + i * 64 + gg] * 0.5f;
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (act) {
                const unsigned w0 = q27_dq4( wv[r].x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv[r].x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv[r].y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv[r].y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs[r]);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lsel]; const float xsc = ss[c][lsel];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[r][c] = fmaf((float)a_, sf * xsc, acc[r][c]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
    }
}
extern "C" int q27_proj_nvfp4_bt10(const q27_nvfp4_t* w, const signed char* xq_perm,
                                   const float* xs, float* y, int C, int R, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b10<4, 16, Q27_PF_MC, 1, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    if (R == 2) { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b10<4, 16, Q27_PF_MC, 2, 3>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

#else   // closed experiments compiled out (see docs/AMD_REFERENCE_MINING.md); stubs keep the flag chains linkable
extern "C" int q27_proj_nvfp4_gu_b6(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt6(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_gu_b7(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int rpw_req, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt7(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int rpw_req, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_gu_b8(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_gu_b9(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt9(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_gu_b10(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt10(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int R, hipStream_t s) { (void)0; return 0; }
#endif
#ifdef Q27_EXPERIMENTS
// v12 (Q27_PF_MLP12): b5's loop untouched, the EPILOGUE rebuilt. b5's disassembly ends in 96 ds_bpermute_b32 with
// 96 s_waitcnt (never more than two in flight, positions one after another): a serial LDS-latency chain per wave as
// long as the whole main loop's arithmetic. The same tree (lane i adds lane i+o for o = 32..1; lane 0's result only
// ever consumes lanes < o at each step, and fadd is commutative) is re-emitted with the o=32/16 steps as bpermutes
// issued GB accumulators at a time before any wait, and the o=8/4/2/1 steps as DPP row_shl adds (in-row, no LDS).
// Bit-identical to the __shfl_down tree: identical pairings and order per lane.
template <int UPL2, int C, int GB, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b12(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * 4 + wave;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    float acc[2 * C];
#pragma unroll
    for (int c = 0; c < 2 * C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, g = t & 63;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const int g = i * 64 + lane;
        {
            const int rr = (row0 < rows) ? row0 : (rows - 1);
            const uint2 vg = q27_ld8nt(WG + (size_t)rr * KP + (size_t)g * 8);
            const uint2 vu = q27_ld8nt(WU + (size_t)rr * KP + (size_t)g * 8);
            const unsigned cg = SG[(size_t)rr * NG + g], cu = SU[(size_t)rr * NG + g];
            const unsigned g0 = q27_dq4( vg.x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg.x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg.y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg.y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu.x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu.x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu.y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu.y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg), suf = q27_e4m3(cu);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
                int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                acc[c]     = fmaf((float)a_, sgf * xsc, acc[c]);
                acc[C + c] = fmaf((float)b_, suf * xsc, acc[C + c]);
            }
        }
    }
    q27_wave_reduce_multi<2 * C, GB>(acc, lane);
    if (lane == 0 && row0 < rows) {
#pragma unroll
        for (int c = 0; c < C; ++c) { YG[(size_t)c * rows + row0] = acc[c] * ag; YU[(size_t)c * rows + row0] = acc[C + c] * au; }
    }
}
extern "C" int q27_proj_nvfp4_gu_b12(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                     const signed char* xq_perm, const float* xs,
                                     float* yg, float* yu, int C, int gb, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
    if (gb == 4)  { hipLaunchKernelGGL((q27_k_nvfp4_gu_b12<5, Q27_PF_MC, 4, 4>),  g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (gb == 8)  { hipLaunchKernelGGL((q27_k_nvfp4_gu_b12<5, Q27_PF_MC, 8, 4>),  g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (gb == 16) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b12<5, Q27_PF_MC, 16, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}
// down v12: t2_b5's loop, the same rebuilt epilogue.
template <int FULL2, int TAIL, int C, int GB, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b12(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * 4 + wave;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const bool act = (i < FULL2) || (lane < TAIL);
        const int g = i * 64 + (act ? lane : 0);
        const int lsel = act ? lane : 0;
        {
            const int rr = (row0 < rows) ? row0 : (rows - 1);
            const uint2 wv = q27_ld8nt(W + (size_t)rr * KP + (size_t)g * 8);
            const unsigned cs = S[(size_t)rr * NG + g];
            if (act) {
                const unsigned w0 = q27_dq4( wv.x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv.x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv.y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv.y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lsel]; const float xsc = ss[c][lsel];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[c] = fmaf((float)a_, sf * xsc, acc[c]);
                }
            }
        }
    }
    q27_wave_reduce_multi<C, GB>(acc, lane);
    if (lane == 0 && row0 < rows) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[c] * alpha;
            if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row0]);
            Y[(size_t)c * rows + row0] = v;
        }
    }
}
extern "C" int q27_proj_nvfp4_bt12(const q27_nvfp4_t* w, const signed char* xq_perm,
                                   const float* xs, float* y, int C, int gb, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
    if (gb == 4) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b12<4, 16, Q27_PF_MC, 4, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    if (gb == 8) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b12<4, 16, Q27_PF_MC, 8, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

#else   // closed experiments compiled out (see docs/AMD_REFERENCE_MINING.md); stubs keep the flag chains linkable
extern "C" int q27_proj_nvfp4_gu_b12(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int gb, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt12(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int gb, hipStream_t s) { (void)0; return 0; }
#endif
// v13 (Q27_PF_MLP13): v12 with the weight/scale addressing made wave-uniform on purpose. b12's code forms every
// weight and scale address as a 64-bit VGPR pair (60 64-bit address ops, no saddr loads) although the row base is
// the same for all 64 lanes: readfirstlane on the row index puts the four row pointers in SGPRs and leaves a 32-bit
// lane offset in VGPRs, which is what a fifth wave per SIMD (<= 48 VGPRs) needs. Same arithmetic, bit-identical.
template <int UPL2, int C, int GB, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b13(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * 4 + wave;
    const int rr = __builtin_amdgcn_readfirstlane((row0 < rows) ? row0 : (rows - 1));
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    const unsigned char* __restrict__ WGr = WG + (size_t)rr * KP;
    const unsigned char* __restrict__ WUr = WU + (size_t)rr * KP;
    const unsigned char* __restrict__ SGr = SG + (size_t)rr * NG;
    const unsigned char* __restrict__ SUr = SU + (size_t)rr * NG;
    float acc[2 * C];
#pragma unroll
    for (int c = 0; c < 2 * C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, g = t & 63;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const unsigned g = (unsigned)(i * 64 + lane);
        {
            const uint2 vg = q27_ld8nt(WGr + g * 8u);
            const uint2 vu = q27_ld8nt(WUr + g * 8u);
            const unsigned cg = SGr[g], cu = SUr[g];
            const unsigned g0 = q27_dq4( vg.x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg.x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg.y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg.y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu.x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu.x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu.y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu.y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg), suf = q27_e4m3(cu);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
                int a_ = q27_d4(g0, xv.x, 0); a_ = q27_d4(g1, xv.z, a_); a_ = q27_d4(g2, xv.y, a_); a_ = q27_d4(g3, xv.w, a_);
                int b_ = q27_d4(u0, xv.x, 0); b_ = q27_d4(u1, xv.z, b_); b_ = q27_d4(u2, xv.y, b_); b_ = q27_d4(u3, xv.w, b_);
                acc[c]     = fmaf((float)a_, sgf * xsc, acc[c]);
                acc[C + c] = fmaf((float)b_, suf * xsc, acc[C + c]);
            }
        }
    }
    q27_wave_reduce_multi<2 * C, GB>(acc, lane);
    if (lane == 0 && row0 < rows) {
#pragma unroll
        for (int c = 0; c < C; ++c) { YG[(size_t)c * rows + row0] = acc[c] * ag; YU[(size_t)c * rows + row0] = acc[C + c] * au; }
    }
}
extern "C" int q27_proj_nvfp4_gu_b13(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                     const signed char* xq_perm, const float* xs,
                                     float* yg, float* yu, int C, int minw, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
    if (minw == 4) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b13<5, Q27_PF_MC, 8, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (minw == 5) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b13<5, Q27_PF_MC, 8, 5>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (minw == 6) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b13<5, Q27_PF_MC, 8, 6>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }   // LDS-limited ceiling (6 x 10 KB)
    return 0;
}
// down v13: t2_b12 with the same uniform addressing; MINW 5 or 6.
template <int FULL2, int TAIL, int C, int GB, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b13(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * 4 + wave;
    const int rr = __builtin_amdgcn_readfirstlane((row0 < rows) ? row0 : (rows - 1));
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    const unsigned char* __restrict__ Wr = W + (size_t)rr * KP;
    const unsigned char* __restrict__ Sr = S + (size_t)rr * NG;
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const bool act = (i < FULL2) || (lane < TAIL);
        const unsigned g = (unsigned)(i * 64 + (act ? lane : 0));
        const int lsel = act ? lane : 0;
        {
            const uint2 wv = q27_ld8nt(Wr + g * 8u);
            const unsigned cs = Sr[g];
            if (act) {
                const unsigned w0 = q27_dq4( wv.x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv.x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv.y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv.y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lsel]; const float xsc = ss[c][lsel];
                    int a_ = q27_d4(w0, xv.x, 0); a_ = q27_d4(w1, xv.z, a_); a_ = q27_d4(w2, xv.y, a_); a_ = q27_d4(w3, xv.w, a_);
                    acc[c] = fmaf((float)a_, sf * xsc, acc[c]);
                }
            }
        }
    }
    q27_wave_reduce_multi<C, GB>(acc, lane);
    if (lane == 0 && row0 < rows) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[c] * alpha;
            if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row0]);
            Y[(size_t)c * rows + row0] = v;
        }
    }
}
extern "C" int q27_proj_nvfp4_bt13(const q27_nvfp4_t* w, const signed char* xq_perm,
                                   const float* xs, float* y, int C, int minw, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
    if (minw == 5) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b13<4, 16, Q27_PF_MC, 8, 5>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    if (minw == 6) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b13<4, 16, Q27_PF_MC, 8, 6>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

#ifdef Q27_EXPERIMENTS
// ---- int8 mirror (Q27_PF_W8): q27_dq4 applied ONCE at load. Each 8 packed bytes (16 nibbles) become 16 int8 in the
// order the kernels consume them: [dq4(lo x), dq4(hi x), dq4(lo y), dq4(hi y)]. The v14 kernels feed these dwords to
// the same v_dot4 with the same activation pairing, so every partial sum is bit-identical to the nibble kernels;
// the price is 2x weight bytes (gate+up+down: 67 MB per layer per card, 4.3 GB per card over 64 layers).
__global__ void q27_k_nvfp4_expand_w8(const unsigned char* __restrict__ W, unsigned char* __restrict__ W8, size_t ngroups) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngroups) return;
    const uint2 v = ((const uint2*)W)[i];
    uint4 o;
    o.x = q27_dq4( v.x       & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
    o.z = q27_dq4( v.y       & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
    ((uint4*)W8)[i] = o;
}
// side table keyed by the packed-weight pointer (the struct keeps its release layout; see receipt v93)
static const unsigned char* s_w8_key[2048];
static unsigned char*       s_w8_val[2048];
static int                  s_w8_n = 0;
extern "C" const unsigned char* q27_nvfp4_w8(const q27_nvfp4_t* w) {
    if (!w || !w->w) return nullptr;
    for (int i = 0; i < s_w8_n; ++i) if (s_w8_key[i] == w->w) return s_w8_val[i];
    return nullptr;
}
extern "C" void q27_nvfp4_free_w8(const q27_nvfp4_t* w) {
    if (!w) return;
    for (int i = 0; i < s_w8_n; ++i) if (s_w8_key[i] == w->w) {
        hipFree(s_w8_val[i]); s_w8_key[i] = s_w8_key[s_w8_n - 1]; s_w8_val[i] = s_w8_val[s_w8_n - 1]; --s_w8_n; return;
    }
}
extern "C" int q27_nvfp4_make_w8(const q27_nvfp4_t* w, hipStream_t s) {
    if (!w || !w->w || w->rows <= 0 || w->K <= 0 || (w->K % 16)) return 0;
    if (q27_nvfp4_w8(w)) return 1;
    if (s_w8_n >= 2048) return 0;
    const size_t bytes = (size_t)w->rows * (size_t)w->K;
    unsigned char* p = nullptr;
    if (hipMalloc((void**)&p, bytes) != hipSuccess) return 0;
    const size_t ng = bytes / 16;
    hipLaunchKernelGGL(q27_k_nvfp4_expand_w8, dim3((unsigned)((ng + 255) / 256)), dim3(256), 0, s, w->w, p, ng);
    s_w8_key[s_w8_n] = w->w; s_w8_val[s_w8_n] = p; ++s_w8_n;
    return 1;
}
// v14 (Q27_PF_MLP14): v13 reading the int8 mirror -- no per-chunk nibble decode (64 of the 227 VALU per chunk), 16-byte
// weight loads. Same dot4 operands and order as b5/b13: bit-identical.
template <int UPL2, int C, int GB, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b14(
        const unsigned char* __restrict__ WG8, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU8, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * 4 + wave;
    const int rr = __builtin_amdgcn_readfirstlane((row0 < rows) ? row0 : (rows - 1));
    const int KP8 = UPL2 * 64 * 16, NG = UPL2 * 64;
    const unsigned char* __restrict__ WGr = WG8 + (size_t)rr * KP8;
    const unsigned char* __restrict__ WUr = WU8 + (size_t)rr * KP8;
    const unsigned char* __restrict__ SGr = SG + (size_t)rr * NG;
    const unsigned char* __restrict__ SUr = SU + (size_t)rr * NG;
    float acc[2 * C];
#pragma unroll
    for (int c = 0; c < 2 * C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, g = t & 63;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NG + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const unsigned g = (unsigned)(i * 64 + lane);
        {
            const uint4 vg = q27_ld16ntv(WGr + g * 16u);
            const uint4 vu = q27_ld16ntv(WUr + g * 16u);
            const unsigned cg = SGr[g], cu = SUr[g];
            const float sgf = q27_e4m3(cg), suf = q27_e4m3(cu);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
                int a_ = q27_d4(vg.x, xv.x, 0); a_ = q27_d4(vg.y, xv.z, a_); a_ = q27_d4(vg.z, xv.y, a_); a_ = q27_d4(vg.w, xv.w, a_);
                int b_ = q27_d4(vu.x, xv.x, 0); b_ = q27_d4(vu.y, xv.z, b_); b_ = q27_d4(vu.z, xv.y, b_); b_ = q27_d4(vu.w, xv.w, b_);
                acc[c]     = fmaf((float)a_, sgf * xsc, acc[c]);
                acc[C + c] = fmaf((float)b_, suf * xsc, acc[C + c]);
            }
        }
    }
    q27_wave_reduce_multi<2 * C, GB>(acc, lane);
    if (lane == 0 && row0 < rows) {
#pragma unroll
        for (int c = 0; c < C; ++c) { YG[(size_t)c * rows + row0] = acc[c] * ag; YU[(size_t)c * rows + row0] = acc[C + c] * au; }
    }
}
extern "C" int q27_proj_nvfp4_gu_b14(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                     const signed char* xq_perm, const float* xs,
                                     float* yg, float* yu, int C, int minw, hipStream_t s) {
    const unsigned char* g8 = q27_nvfp4_w8(wg); const unsigned char* u8 = q27_nvfp4_w8(wu);
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC || !g8 || !u8) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
    if (minw == 5) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b14<5, Q27_PF_MC, 8, 5>), g, b, 0, s, g8, wg->gs, u8, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (minw == 6) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b14<5, Q27_PF_MC, 8, 6>), g, b, 0, s, g8, wg->gs, u8, wu->gs, (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}
template <int FULL2, int TAIL, int C, int GB, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b14(
        const unsigned char* __restrict__ W8, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * 4 + wave;
    const int rr = __builtin_amdgcn_readfirstlane((row0 < rows) ? row0 : (rows - 1));
    const int NG = FULL2 * 64 + TAIL, KP8 = NG * 16;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    const unsigned char* __restrict__ Wr = W8 + (size_t)rr * KP8;
    const unsigned char* __restrict__ Sr = S + (size_t)rr * NG;
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const bool act = (i < FULL2) || (lane < TAIL);
        const unsigned g = (unsigned)(i * 64 + (act ? lane : 0));
        const int lsel = act ? lane : 0;
        {
            const uint4 wv = q27_ld16ntv(Wr + g * 16u);
            const unsigned cs = Sr[g];
            if (act) {
                const float sf = q27_e4m3(cs);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    const uint4 xv = sx[c][lsel]; const float xsc = ss[c][lsel];
                    int a_ = q27_d4(wv.x, xv.x, 0); a_ = q27_d4(wv.y, xv.z, a_); a_ = q27_d4(wv.z, xv.y, a_); a_ = q27_d4(wv.w, xv.w, a_);
                    acc[c] = fmaf((float)a_, sf * xsc, acc[c]);
                }
            }
        }
    }
    q27_wave_reduce_multi<C, GB>(acc, lane);
    if (lane == 0 && row0 < rows) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[c] * alpha;
            if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row0]);
            Y[(size_t)c * rows + row0] = v;
        }
    }
}
extern "C" int q27_proj_nvfp4_bt14(const q27_nvfp4_t* w, const signed char* xq_perm,
                                   const float* xs, float* y, int C, int minw, hipStream_t s) {
    const unsigned char* w8 = q27_nvfp4_w8(w);
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC || !w8) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
    if (minw == 5) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b14<4, 16, Q27_PF_MC, 8, 5>), g, b, 0, s, w8, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    if (minw == 6) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b14<4, 16, Q27_PF_MC, 8, 6>), g, b, 0, s, w8, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

// v11 (Q27_PF_MLP11): b5 with the per-position LDS reads pipelined. The disassembly of b5 shows one
// ds_read_b128 into the same four registers per position followed by s_waitcnt lgkmcnt(0) and ~14
// instructions: eight exposed LDS round trips per k-chunk, forty per call, against ~450 cycles of
// arithmetic per chunk. DEPTH staged activations are read ahead (DEPTH=2: +5 VGPRs, stays in the
// 4-wave tier; DEPTH=8: all positions preloaded, 3-wave tier). Same arithmetic order: bit-identical.
template <int UPL2, int C, int DEPTH, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_gu_b11(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    static_assert(C % DEPTH == 0, "DEPTH must divide C");
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row = blockIdx.x * 4 + wave;
    const int rr = (row < rows) ? row : (rows - 1);
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    float accg[C], accu[C];
#pragma unroll
    for (int c = 0; c < C; ++c) { accg[c] = 0.f; accu[c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        const int g = i * 64 + lane;
        const uint2 vg = q27_ld8nt(WG + (size_t)rr * KP + (size_t)g * 8);
        const uint2 vu = q27_ld8nt(WU + (size_t)rr * KP + (size_t)g * 8);
        const unsigned cg = SG[(size_t)rr * NG + g], cu = SU[(size_t)rr * NG + g];
        __syncthreads();
        for (int t = threadIdx.x; t < C * 64; t += 256) {
            const int c = t >> 6, gg = t & 63;
            sx[c][gg] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)(i * 64 + gg) * 4);
            ss[c][gg] = Xs[(size_t)c * NG + i * 64 + gg] * 0.5f;
        }
        __syncthreads();
        const unsigned g0 = q27_dq4( vg.x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg.x >> 4) & 0x0F0F0F0Fu);
        const unsigned g2 = q27_dq4( vg.y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg.y >> 4) & 0x0F0F0F0Fu);
        const unsigned u0 = q27_dq4( vu.x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu.x >> 4) & 0x0F0F0F0Fu);
        const unsigned u2 = q27_dq4( vu.y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu.y >> 4) & 0x0F0F0F0Fu);
        const float sgf = q27_e4m3(cg), suf = q27_e4m3(cu);
#pragma unroll
        for (int cb = 0; cb < C; cb += DEPTH) {
            uint4 xv[DEPTH]; float xsc[DEPTH];
#pragma unroll
            for (int k = 0; k < DEPTH; ++k) { xv[k] = sx[cb + k][lane]; xsc[k] = ss[cb + k][lane]; }   // DEPTH reads in flight
#pragma unroll
            for (int k = 0; k < DEPTH; ++k) {
                const int c = cb + k;
                int a_ = q27_d4(g0, xv[k].x, 0); a_ = q27_d4(g1, xv[k].z, a_); a_ = q27_d4(g2, xv[k].y, a_); a_ = q27_d4(g3, xv[k].w, a_);
                int b_ = q27_d4(u0, xv[k].x, 0); b_ = q27_d4(u1, xv[k].z, b_); b_ = q27_d4(u2, xv[k].y, b_); b_ = q27_d4(u3, xv[k].w, b_);
                accg[c] = fmaf((float)a_, sgf * xsc[k], accg[c]);
                accu[c] = fmaf((float)b_, suf * xsc[k], accu[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float vg2 = accg[c], vu2 = accu[c];
#pragma unroll
        for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
        if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
    }
}
extern "C" int q27_proj_nvfp4_gu_b11(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                     const signed char* xq_perm, const float* xs,
                                     float* yg, float* yu, int C, int depth, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
    if (depth == 2) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b11<5, Q27_PF_MC, 2, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (depth == 4) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b11<5, Q27_PF_MC, 4, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (depth == 8) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b11<5, Q27_PF_MC, 8, 3>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}
// down v11: the same LDS read pipelining on t2_b5 (R=1).
template <int FULL2, int TAIL, int C, int DEPTH, int MINW>
__global__ __launch_bounds__(256, MINW) void q27_k_nvfp4_t2_b11(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    static_assert(C % DEPTH == 0, "DEPTH must divide C");
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row = blockIdx.x * 4 + wave;
    const int rr = (row < rows) ? row : (rows - 1);
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        const bool act = (i < FULL2) || (lane < TAIL);
        const int g = i * 64 + (act ? lane : 0);
        const int lsel = act ? lane : 0;
        const uint2 wv = q27_ld8nt(W + (size_t)rr * KP + (size_t)g * 8);
        const unsigned cs = S[(size_t)rr * NG + g];
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, gg = t % ng;
            sx[c][gg] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + gg) * 4);
            ss[c][gg] = Xs[(size_t)c * NGS + i * 64 + gg] * 0.5f;
        }
        __syncthreads();
        if (act) {
            const unsigned w0 = q27_dq4( wv.x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv.x >> 4) & 0x0F0F0F0Fu);
            const unsigned w2 = q27_dq4( wv.y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv.y >> 4) & 0x0F0F0F0Fu);
            const float sf = q27_e4m3(cs);
#pragma unroll
            for (int cb = 0; cb < C; cb += DEPTH) {
                uint4 xv[DEPTH]; float xsc[DEPTH];
#pragma unroll
                for (int k = 0; k < DEPTH; ++k) { xv[k] = sx[cb + k][lsel]; xsc[k] = ss[cb + k][lsel]; }
#pragma unroll
                for (int k = 0; k < DEPTH; ++k) {
                    const int c = cb + k;
                    int a_ = q27_d4(w0, xv[k].x, 0); a_ = q27_d4(w1, xv[k].z, a_); a_ = q27_d4(w2, xv[k].y, a_); a_ = q27_d4(w3, xv[k].w, a_);
                    acc[c] = fmaf((float)a_, sf * xsc[k], acc[c]);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float v = acc[c];
#pragma unroll
        for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
        if (lane == 0 && row < rows) {
            v *= alpha;
            if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
            Y[(size_t)c * rows + row] = v;
        }
    }
}
extern "C" int q27_proj_nvfp4_bt11(const q27_nvfp4_t* w, const signed char* xq_perm,
                                   const float* xs, float* y, int C, int depth, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
    if (depth == 2) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b11<4, 16, Q27_PF_MC, 2, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    if (depth == 4) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b11<4, 16, Q27_PF_MC, 4, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    if (depth == 8) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b11<4, 16, Q27_PF_MC, 8, 3>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

#else   // closed experiments compiled out (see docs/AMD_REFERENCE_MINING.md); stubs keep the flag chains linkable
extern "C" const unsigned char* q27_nvfp4_w8(const q27_nvfp4_t* w) { (void)0; return nullptr; }
extern "C" void q27_nvfp4_free_w8(const q27_nvfp4_t* w) { (void)0;  }
extern "C" int q27_nvfp4_make_w8(const q27_nvfp4_t* w, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_gu_b14(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int minw, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt14(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int minw, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_gu_b11(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int depth, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt11(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int depth, hipStream_t s) { (void)0; return 0; }

// PRESHUFFLED int8 mirror (Q27_LS_W8): [kk][tile][tx][4 rows][16 B] with the dq4 even-then-odd
// lane order baked in. A warp reading one (kk, tile) walks 1 KB of contiguous bytes -- the
// w8-bypass wide GEMMs' per-thread weight loads are fully coalesced and decode-free.
__global__ void q27_k_nvfp4_expand_w8s(const unsigned char* __restrict__ W, unsigned char* __restrict__ W8,
                                       int KP, int ntiles, int NG) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NG * ntiles * 16) return;
    const int tx = i & 15, rem = i >> 4;
    const int kk = rem / ntiles, tile = rem % ntiles;
    unsigned char* dst = W8 + ((size_t)((size_t)kk * ntiles + tile) * 16 + tx) * 64;
    const int row0 = tile * 64 + tx * 4;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint2 v = ((const uint2*)(W + (size_t)(row0 + r) * KP))[kk];
        uint4 o;
        o.x = q27_dq4( v.x       & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        o.z = q27_dq4( v.y       & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ((uint4*)dst)[r] = o;
    }
}
static const unsigned char* s_w8s_key[2048];
static unsigned char*       s_w8s_val[2048];
static int                  s_w8s_n = 0;
extern "C" const unsigned char* q27_nvfp4_w8s(const q27_nvfp4_t* w) {
    if (!w || !w->w) return nullptr;
    for (int i = 0; i < s_w8s_n; ++i) if (s_w8s_key[i] == w->w) return s_w8s_val[i];
    return nullptr;
}
extern "C" int q27_nvfp4_make_w8s(const q27_nvfp4_t* w, hipStream_t s) {
    if (!w || !w->w || w->rows <= 0 || w->K <= 0 || (w->K % 16) || (w->rows % 64)) return 0;
    if (q27_nvfp4_w8s(w)) return 1;
    if (s_w8s_n >= 2048) return 0;
    const size_t bytes = (size_t)w->rows * (size_t)w->K;
    unsigned char* p = nullptr;
    if (hipMalloc((void**)&p, bytes) != hipSuccess) return 0;
    const int ntiles = w->rows / 64, NG = w->K >> 4, KP = w->K >> 1;
    const int ntx = NG * ntiles * 16;
    hipLaunchKernelGGL(q27_k_nvfp4_expand_w8s, dim3((ntx + 255) / 256), dim3(256), 0, s, w->w, p, KP, ntiles, NG);
    s_w8s_key[s_w8s_n] = w->w; s_w8s_val[s_w8s_n] = p; ++s_w8s_n;
    return 1;
}

#endif
#ifdef Q27_EXPERIMENTS
// v4 (Q27_PF_MLP4): b2 with the k-chunk pipeline SOFTWARE-PIPELINED. The M-knee census showed b2
// flat per position from C=4 to C=8 at ~200 GB/s of weights (12% of dot4 issue peak): each k-chunk
// stages its activation slice global->LDS synchronously (load, barrier, compute) and fetches its
// weights at the top of the chunk, so every chunk pays a global latency with nothing to overlap it.
// Here the NEXT chunk's activation slice and weights are loaded into registers before this chunk's
// arithmetic, written to the other LDS buffer after it, one barrier per chunk. Same q27_dq4/q27_d4
// sequence, same fmaf order, same wave reduction: bit-identical to b2. Extra registers: ~16.
template <int UPL2, int C, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_gu_b4(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU)
{
    static_assert(C * 64 % 256 == 0, "C*64 must be a multiple of the block");
    constexpr int PT = C * 64 / 256;                     // staged uint4 per thread per chunk
    __shared__ uint4 sx[2][C][64];
    __shared__ float ss[2][C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int KP = UPL2 * 64 * 8, NG = UPL2 * 64;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    // chunk 0: stage synchronously, weights for chunk 0
    for (int t = threadIdx.x; t < C * 64; t += 256) {
        const int c = t >> 6, g = t & 63;
        sx[0][c][g] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)g * 4);
        ss[0][c][g] = Xs[(size_t)c * NG + g] * 0.5f;
    }
    uint2 vg[R], vu[R]; unsigned cg[R], cu[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        vg[r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)lane * 8);
        vu[r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)lane * 8);
        cg[r] = SG[(size_t)rr[r] * NG + lane]; cu[r] = SU[(size_t)rr[r] * NG + lane];
    }
    __syncthreads();
    float accg[R][C], accu[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) { accg[r][c] = 0.f; accu[r][c] = 0.f; }
#pragma unroll 1
    for (int i = 0; i < UPL2; ++i) {
        const int cur = i & 1;
        // issue chunk i+1's loads now (activations for this thread's staging share, weights)
        uint4 nx[PT]; float ns[PT]; uint2 nvg[R], nvu[R]; unsigned ncg[R], ncu[R];
        const bool more = (i + 1 < UPL2);
        if (more) {
#pragma unroll
            for (int p = 0; p < PT; ++p) {
                const int t = threadIdx.x + p * 256, c = t >> 6, g = t & 63;
                nx[p] = *(const uint4*)(Xp + (size_t)c * (UPL2 * 256) + (size_t)((i + 1) * 64 + g) * 4);
                ns[p] = Xs[(size_t)c * NG + (i + 1) * 64 + g] * 0.5f;
            }
            const int gn = (i + 1) * 64 + lane;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                nvg[r] = q27_ld8nt(WG + (size_t)rr[r] * KP + (size_t)gn * 8);
                nvu[r] = q27_ld8nt(WU + (size_t)rr[r] * KP + (size_t)gn * 8);
                ncg[r] = SG[(size_t)rr[r] * NG + gn]; ncu[r] = SU[(size_t)rr[r] * NG + gn];
            }
        }
        // compute chunk i (b2's arithmetic, verbatim)
        uint4 xv[C]; float xs[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { xv[c] = sx[cur][c][lane]; xs[c] = ss[cur][c][lane]; }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned g0 = q27_dq4( vg[r].x       & 0x0F0F0F0Fu), g1 = q27_dq4((vg[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned g2 = q27_dq4( vg[r].y       & 0x0F0F0F0Fu), g3 = q27_dq4((vg[r].y >> 4) & 0x0F0F0F0Fu);
            const unsigned u0 = q27_dq4( vu[r].x       & 0x0F0F0F0Fu), u1 = q27_dq4((vu[r].x >> 4) & 0x0F0F0F0Fu);
            const unsigned u2 = q27_dq4( vu[r].y       & 0x0F0F0F0Fu), u3 = q27_dq4((vu[r].y >> 4) & 0x0F0F0F0Fu);
            const float sgf = q27_e4m3(cg[r]), suf = q27_e4m3(cu[r]);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                int a = q27_d4(g0, xv[c].x, 0); a = q27_d4(g1, xv[c].z, a); a = q27_d4(g2, xv[c].y, a); a = q27_d4(g3, xv[c].w, a);
                int b = q27_d4(u0, xv[c].x, 0); b = q27_d4(u1, xv[c].z, b); b = q27_d4(u2, xv[c].y, b); b = q27_d4(u3, xv[c].w, b);
                accg[r][c] = fmaf((float)a, sgf * xs[c], accg[r][c]);
                accu[r][c] = fmaf((float)b, suf * xs[c], accu[r][c]);
            }
        }
        if (more) {
#pragma unroll
            for (int p = 0; p < PT; ++p) {
                const int t = threadIdx.x + p * 256, c = t >> 6, g = t & 63;
                sx[cur ^ 1][c][g] = nx[p]; ss[cur ^ 1][c][g] = ns[p];
            }
#pragma unroll
            for (int r = 0; r < R; ++r) { vg[r] = nvg[r]; vu[r] = nvu[r]; cg[r] = ncg[r]; cu[r] = ncu[r]; }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float vg2 = accg[r][c], vu2 = accu[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vg2 += __shfl_down(vg2, o, 64); vu2 += __shfl_down(vu2, o, 64); }
            if (lane == 0 && row < rows) { YG[(size_t)c * rows + row] = vg2 * ag; YU[(size_t)c * rows + row] = vu2 * au; }
        }
    }
}
extern "C" int q27_proj_nvfp4_gu_b4(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    if (R == 1) { const dim3 g((unsigned)((wg->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b4<5, Q27_PF_MC, 1>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (R == 2) { const dim3 g((unsigned)((wg->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b4<5, Q27_PF_MC, 2>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}

// down v4: the same pipeline over FULL2 full chunks + the TAIL chunk (lanes >= TAIL fetch group i*64
// and skip the arithmetic, as b2 does). Bit-identical to t2_b2.
template <int FULL2, int TAIL, int C, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_t2_b4(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    static_assert(C * 64 % 256 == 0, "C*64 must be a multiple of the block");
    constexpr int PT = C * 64 / 256;
    __shared__ uint4 sx[2][C][64];
    __shared__ float ss[2][C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    for (int t = threadIdx.x; t < C * 64; t += 256) {
        const int c = t >> 6, g = t & 63;
        sx[0][c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)g * 4);
        ss[0][c][g] = Xs[(size_t)c * NGS + g] * 0.5f;
    }
    uint2 wv[R]; unsigned cs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { wv[r] = q27_ld8nt(W + (size_t)rr[r] * KP + (size_t)lane * 8); cs[r] = S[(size_t)rr[r] * NG + lane]; }
    __syncthreads();
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int cur = i & 1;
        const bool more = (i + 1 < FULL2 + 1);
        uint4 nx[PT]; float ns[PT]; uint2 nwv[R]; unsigned ncs[R];
        if (more) {
            const int ngn = (i + 1 < FULL2) ? 64 : TAIL;          // groups in chunk i+1
#pragma unroll
            for (int p = 0; p < PT; ++p) {
                const int t = threadIdx.x + p * 256, c = t / 64, g = t % 64;
                if (g < ngn) {
                    nx[p] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)((i + 1) * 64 + g) * 4);
                    ns[p] = Xs[(size_t)c * NGS + (i + 1) * 64 + g] * 0.5f;
                } else { nx[p] = make_uint4(0u, 0u, 0u, 0u); ns[p] = 0.f; }
            }
            const bool actn = (i + 1 < FULL2) || (lane < TAIL);
            const int gn = (i + 1) * 64 + (actn ? lane : 0);
#pragma unroll
            for (int r = 0; r < R; ++r) { nwv[r] = q27_ld8nt(W + (size_t)rr[r] * KP + (size_t)gn * 8); ncs[r] = S[(size_t)rr[r] * NG + gn]; }
        }
        const bool act = (i < FULL2) || (lane < TAIL);
        if (act) {
            uint4 xv[C]; float xs[C];
#pragma unroll
            for (int c = 0; c < C; ++c) { xv[c] = sx[cur][c][lane]; xs[c] = ss[cur][c][lane]; }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const unsigned w0 = q27_dq4( wv[r].x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv[r].x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv[r].y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv[r].y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs[r]);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    int a = q27_d4(w0, xv[c].x, 0); a = q27_d4(w1, xv[c].z, a); a = q27_d4(w2, xv[c].y, a); a = q27_d4(w3, xv[c].w, a);
                    acc[r][c] = fmaf((float)a, sf * xs[c], acc[r][c]);
                }
            }
        }
        if (more) {
#pragma unroll
            for (int p = 0; p < PT; ++p) {
                const int t = threadIdx.x + p * 256, c = t / 64, g = t % 64;
                sx[cur ^ 1][c][g] = nx[p]; ss[cur ^ 1][c][g] = ns[p];
            }
#pragma unroll
            for (int r = 0; r < R; ++r) { wv[r] = nwv[r]; cs[r] = ncs[r]; }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
    }
}
extern "C" int q27_proj_nvfp4_bt4(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, int R, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b4<4, 16, Q27_PF_MC, 1>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    if (R == 2) { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b4<4, 16, Q27_PF_MC, 2>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

// bench-only launchers with the position count as a parameter (C = 8 or 16, R = 2): the M-knee
// census. The production slots are 8 wide, so these are not used by the engine.
extern "C" int q27_proj_nvfp4_gu_b2_c(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                      const signed char* xq_perm, const float* xs,
                                      float* yg, float* yu, int C, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g((unsigned)((wg->rows + 7) / 8)), b(256);
    if (C == 16) { hipLaunchKernelGGL((q27_k_nvfp4_gu_b2<5, 16, 2>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                                      (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (C == 8)  { hipLaunchKernelGGL((q27_k_nvfp4_gu_b2<5, 8, 2>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                                      (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    if (C == 4)  { hipLaunchKernelGGL((q27_k_nvfp4_gu_b2<5, 4, 2>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                                      (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    return 0;
}

#else   // closed experiments compiled out (see docs/AMD_REFERENCE_MINING.md); stubs keep the flag chains linkable
extern "C" int q27_proj_nvfp4_gu_b4(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt4(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_gu_b2_c(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int C, hipStream_t s) { (void)0; return 0; }
#endif
extern "C" int q27_proj_nvfp4_gu_b2(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int C, int R, hipStream_t s) {
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    // always C=Q27_PF_MC: the slots are 8 wide, the extra positions of a tail chunk compute into slots nobody reads
    if (R == 4) { const dim3 g((unsigned)((wg->rows + 15) / 16)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_gu_b2<5, Q27_PF_MC, 4>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                           (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
    { const dim3 g((unsigned)((wg->rows + 7) / 8)), b(256);
      hipLaunchKernelGGL((q27_k_nvfp4_gu_b2<5, Q27_PF_MC, 2>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs,
                         (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu); return 1; }
}

// down (K=4352 = FULL2 x 1024 + TAIL x 16), same design; the tail chunk stages TAIL groups and only
// lanes < TAIL contribute, exactly as w64t2_b's clamped tail.
template <int FULL2, int TAIL, int C, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_t2_b2(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;            // groups in this chunk
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const bool act = (i < FULL2) || (lane < TAIL);
        const int g = i * 64 + (act ? lane : 0);
        uint2 wv[R]; unsigned cs[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            wv[r] = q27_ld8nt(W + (size_t)rr * KP + (size_t)g * 8);
            cs[r] = S[(size_t)rr * NG + g];
        }
        if (act) {
            uint4 xv[C]; float xs[C];
#pragma unroll
            for (int c = 0; c < C; ++c) { xv[c] = sx[c][act ? lane : 0]; xs[c] = ss[c][act ? lane : 0]; }
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const unsigned w0 = q27_dq4( wv[r].x       & 0x0F0F0F0Fu), w1 = q27_dq4((wv[r].x >> 4) & 0x0F0F0F0Fu);
                const unsigned w2 = q27_dq4( wv[r].y       & 0x0F0F0F0Fu), w3 = q27_dq4((wv[r].y >> 4) & 0x0F0F0F0Fu);
                const float sf = q27_e4m3(cs[r]);
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    int a = q27_d4(w0, xv[c].x, 0); a = q27_d4(w1, xv[c].z, a); a = q27_d4(w2, xv[c].y, a); a = q27_d4(w3, xv[c].w, a);
                    acc[r][c] = fmaf((float)a, sf * xs[c], acc[r][c]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
    }
}
#ifdef Q27_EXPERIMENTS
// down v3: t2_b2 with all FULL2+1 chunk fetches issued up front (tail chunk: lanes >= TAIL fetch
// group i*64, as b2 does, and skip the arithmetic). Bit-identical to t2_b2.
template <int FULL2, int TAIL, int C, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_t2_b3(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    const int NG = FULL2 * 64 + TAIL, KP = NG * 8;
    const int XPS = FULL2 * 256 + TAIL * 4, NGS = NG;
    int rr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rr[r] = (row0 + r < rows) ? (row0 + r) : (rows - 1);
    uint2 wv[FULL2 + 1][R];
#pragma unroll
    for (int i = 0; i < FULL2 + 1; ++i) {
        const bool act = (i < FULL2) || (lane < TAIL);
        const int g = i * 64 + (act ? lane : 0);
#pragma unroll
        for (int r = 0; r < R; ++r) wv[i][r] = q27_ld8nt(W + (size_t)rr[r] * KP + (size_t)g * 8);
    }
    unsigned cs[R];
#pragma unroll
    for (int r = 0; r < R; ++r) cs[r] = S[(size_t)rr[r] * NG + lane];
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int ng = (i < FULL2) ? 64 : TAIL;
        __syncthreads();
        for (int t = threadIdx.x; t < C * ng; t += 256) {
            const int c = t / ng, g = t % ng;
            sx[c][g] = *(const uint4*)(Xp + (size_t)c * XPS + (size_t)(i * 64 + g) * 4);
            ss[c][g] = Xs[(size_t)c * NGS + i * 64 + g] * 0.5f;
        }
        __syncthreads();
        const bool act = (i < FULL2) || (lane < TAIL);
        unsigned csn[R];
        if (i + 1 < FULL2 + 1) {
            const bool actn = (i + 1 < FULL2) || (lane < TAIL);
            const int gn = (i + 1) * 64 + (actn ? lane : 0);
#pragma unroll
            for (int r = 0; r < R; ++r) csn[r] = S[(size_t)rr[r] * NG + gn];
        }
        if (act) {
            unsigned w0[R], w1[R], w2[R], w3[R]; float sf[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                w0[r] = q27_dq4( wv[i][r].x       & 0x0F0F0F0Fu); w1[r] = q27_dq4((wv[i][r].x >> 4) & 0x0F0F0F0Fu);
                w2[r] = q27_dq4( wv[i][r].y       & 0x0F0F0F0Fu); w3[r] = q27_dq4((wv[i][r].y >> 4) & 0x0F0F0F0Fu);
                sf[r] = q27_e4m3(cs[r]);
            }
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = sx[c][lane]; const float xsc = ss[c][lane];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int a = q27_d4(w0[r], xv.x, 0); a = q27_d4(w1[r], xv.z, a); a = q27_d4(w2[r], xv.y, a); a = q27_d4(w3[r], xv.w, a);
                    acc[r][c] = fmaf((float)a, sf[r] * xsc, acc[r][c]);
                }
            }
        }
        if (i + 1 < FULL2 + 1) {
#pragma unroll
            for (int r = 0; r < R; ++r) cs[r] = csn[r];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)c * rows + row]);
                Y[(size_t)c * rows + row] = v;
            }
        }
    }
}
extern "C" int q27_proj_nvfp4_bt3(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, int R, hipStream_t s) {
    if (!w || w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    const unsigned short* radd = g_nv_radd; const float rs = g_nv_rs;   // same contract as bt2 (q27_nvfp4_set_res)
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b3<4, 16, 8, 1>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    if (R == 2) { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b3<4, 16, 8, 2>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, radd, rs); return 1; }
    return 0;
}

extern "C" int q27_proj_nvfp4_bt2_c(const q27_nvfp4_t* w, const signed char* xq_perm,
                                    const float* xs, float* y, int C, hipStream_t s) {
    if (w->K != 4352) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
    if (C == 16) { hipLaunchKernelGGL((q27_k_nvfp4_t2_b2<4, 16, 16, 2>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                                      alpha, w->rows, y, nullptr, 0.f); return 1; }
    if (C == 8)  { hipLaunchKernelGGL((q27_k_nvfp4_t2_b2<4, 16, 8, 2>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                                      alpha, w->rows, y, nullptr, 0.f); return 1; }
    if (C == 4)  { hipLaunchKernelGGL((q27_k_nvfp4_t2_b2<4, 16, 4, 2>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                                      alpha, w->rows, y, nullptr, 0.f); return 1; }
    return 0;
}

#else   // closed experiments compiled out (see docs/AMD_REFERENCE_MINING.md); stubs keep the flag chains linkable
extern "C" int q27_proj_nvfp4_bt3(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, int R, hipStream_t s) { (void)0; return 0; }
extern "C" int q27_proj_nvfp4_bt2_c(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int C, hipStream_t s) { (void)0; return 0; }
#endif
extern "C" int q27_proj_nvfp4_bt2(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, int R, hipStream_t s) {
    if (w->K != 4352 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->ws2;
    if (R == 4) { const dim3 g((unsigned)((w->rows + 15) / 16)), b(256);
        hipLaunchKernelGGL((q27_k_nvfp4_t2_b2<4, 16, Q27_PF_MC, 4>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                           alpha, w->rows, y, g_nv_radd, g_nv_rs); return 1; }
    { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
      hipLaunchKernelGGL((q27_k_nvfp4_t2_b2<4, 16, Q27_PF_MC, 2>), g, b, 0, s, w->w, w->gs, (const unsigned*)xq_perm, xs,
                         alpha, w->rows, y, g_nv_radd, g_nv_rs); return 1; }
}

extern "C" void q27_proj_nvfp4_bt(const q27_nvfp4_t* w, const signed char* xq_perm,
                                  const float* xs, float* y, int C, hipStream_t s) {
    if (w->K != 4352) return;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g64((unsigned)((w->rows + 3) / 4)), b64(256);
    // C is padded UP to the next instantiated size: the per-position slots are 8 wide, so
    // the extra positions compute into slots nobody reads. Launching NOTHING for C in
    // {3,5,6,7} left STALE slots for the tail chunk of any prompt whose length is not a
    // multiple of 8 (205 = 25*8+5). Caught by the batched-GDN gate.
    const int Cp = (C <= 1) ? 1 : (C <= 2) ? 2 : (C <= 4) ? 4 : 8;
    if (Cp == 8)
        hipLaunchKernelGGL((q27_k_nvfp4_w64t2_b<4,16,8>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y,
                           g_nv_radd, g_nv_rs);
    else if (Cp == 4)
        hipLaunchKernelGGL((q27_k_nvfp4_w64t2_b<4,16,4>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y,
                           g_nv_radd, g_nv_rs);
    else if (Cp == 2)
        hipLaunchKernelGGL((q27_k_nvfp4_w64t2_b<4,16,2>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y,
                           g_nv_radd, g_nv_rs);
    else if (Cp == 1)
        hipLaunchKernelGGL((q27_k_nvfp4_w64t2_b<4,16,1>), g64, b64, 0, s,
                           w->w, w->gs, (const unsigned*)xq_perm, xs, alpha, w->rows, y,
                           g_nv_radd, g_nv_rs);
}


void q27_proj_nvfp4(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                    float* y, hipStream_t s)
{
    const int rows = w->rows, K = w->K;
    if (rows <= 0 || K <= 0 || (K & 31)) {
        fprintf(stderr, "q27_proj_nvfp4: rows=%d K=%d — K must be a positive multiple of 32 "
                        "(the ABI says 512)\n", rows, K);
        return;
    }
    const int KP  = K >> 1;                       // packed bytes per row
    const int NG  = K >> 4;                       // group scales per row
    const int NU4 = K >> 5;                       // uint4 per row
    const float alpha = w->in_scale * w->ws2;     // contract §5: formed once, applied once
    const dim3 grid((unsigned)((rows + Q27_NVFP4_ROWS - 1) / Q27_NVFP4_ROWS)), blk(Q27_NVFP4_BLOCK);
    const unsigned* Xp = (const unsigned*)xq_perm;

    static const bool use32 = q27_env_flag("Q27_NVFP4_W32", false);
    if (!use32) {                       // measured 1.39x faster; W32=1 restores the old geometry
        const dim3 g64((unsigned)((rows + 3) / 4)), b64(256);
        switch (K) {
        case  5120:
            if (q27_env_flag("Q27_NVFP4_WIDE", false))
                hipLaunchKernelGGL((q27_k_nvfp4_w64w<2,1>), g64, b64, 0, s,
                                   w->w, w->gs, Xp, xs, alpha, rows, y);
            else
                hipLaunchKernelGGL((q27_k_nvfp4_w64<5>),  g64, b64, 0, s,
                                   w->w, w->gs, Xp, xs, alpha, rows, y);
            return;
        case 17408: hipLaunchKernelGGL((q27_k_nvfp4_w64<17>), g64, b64, 0, s,
                        w->w, w->gs, Xp, xs, alpha, rows, y); return;
        // TP row-parallel down_proj: K = 17408/4. 272 uint2/row = 4*64 + 16, so it needs the
        // predicated-tail form. Q27_K4352=0 restores the old fall-through for A/B.
        // converted o_proj/out_proj under Q27_FP8X4: K = 6144/4 = 1536 = 64 + 32 groups
        case  1536:
            if (q27_env_flag("Q27_COLL_BF16", false))
                hipLaunchKernelGGL((q27_k_nvfp4_w64t<1,32,1>), g64, b64, 0, s,
                                   w->w, w->gs, Xp, xs, alpha, rows, y);
            else
                hipLaunchKernelGGL((q27_k_nvfp4_w64t<1,32,0>), g64, b64, 0, s,
                                   w->w, w->gs, Xp, xs, alpha, rows, y, g_nv_radd, g_nv_rs);
            return;
        case  4352:
            {   static const int alt = q27_env_int("Q27_GEMV_ALT", 0);
                if (alt == 1 && q27_proj_nvfp4_down_nr(w, xq_perm, xs, y, rows, g_nv_radd, 0, g_nv_rs, 1, s)) return;
                if (alt == 2 && q27_proj_nvfp4_down_lds(w, xq_perm, xs, y, rows, g_nv_radd, 0, g_nv_rs, 1, s)) return;
                if (alt == 3 && q27_proj_nvfp4_down_hg(w, xq_perm, xs, y, rows, g_nv_radd, 0, g_nv_rs, 1, s)) return; }
            if (q27_env_flag("Q27_K4352W", false)) {       // wide main + clamped tail, A/B
                if (q27_env_flag("Q27_COLL_BF16", false))
                    hipLaunchKernelGGL((q27_k_nvfp4_w64tw<2,16,1>), g64, b64, 0, s,
                                       w->w, w->gs, Xp, xs, alpha, rows, y);
                else
                    hipLaunchKernelGGL((q27_k_nvfp4_w64tw<2,16,0>), g64, b64, 0, s,
                                       w->w, w->gs, Xp, xs, alpha, rows, y, g_nv_radd, g_nv_rs);
                return;
            }
            if (q27_env_flag("Q27_K4352V2", true)) {       // KEPT: 18 pairs, 14/18 paired,
                                                                  // blocks +5.21/-1.36/+1.10% (pooled ~+2.1%);
                                                                  // ISA shows the w64t tail divergence it removes
                if (q27_env_flag("Q27_COLL_BF16", false))
                    hipLaunchKernelGGL((q27_k_nvfp4_w64t2<4,16,1>), g64, b64, 0, s,
                                       w->w, w->gs, Xp, xs, alpha, rows, y);
                else
                    hipLaunchKernelGGL((q27_k_nvfp4_w64t2<4,16,0>), g64, b64, 0, s,
                                       w->w, w->gs, Xp, xs, alpha, rows, y, g_nv_radd, g_nv_rs);
                return;
            }
            if (q27_env_flag("Q27_K4352", true)) {
                if (q27_env_flag("Q27_COLL_BF16", false))
                    hipLaunchKernelGGL((q27_k_nvfp4_w64t<4,16,1>), g64, b64, 0, s,
                                       w->w, w->gs, Xp, xs, alpha, rows, y);
                else
                    hipLaunchKernelGGL((q27_k_nvfp4_w64t<4,16,0>), g64, b64, 0, s,
                                       w->w, w->gs, Xp, xs, alpha, rows, y, g_nv_radd, g_nv_rs);
                return;
            }
            break;
        default: break;                 // fall through to the 32-lane forms for other K
        }
    }
    switch (K) {
#define Q27_FIXED(KK, UPL) \
    case KK: hipLaunchKernelGGL((q27_k_nvfp4_fixed<UPL>), grid, blk, 0, s, \
                               w->w, w->gs, Xp, xs, alpha, rows, y); return;
    Q27_FIXED(1024, 1)
    Q27_FIXED(2048, 2)
    Q27_FIXED(3072, 3)
    Q27_FIXED(4096, 4)
    Q27_FIXED(5120, 5)        // gate, up, lm_head — the measured kE geometry, unrolled
#undef Q27_FIXED
    default:                  // K=17408 (down) and anything else
        hipLaunchKernelGGL((q27_k_nvfp4_blocked<Q27_NVFP4_UNROLL>), grid, blk, 0, s,
                           w->w, w->gs, Xp, xs, alpha, rows, KP, NG, NU4, y);
        return;
    }
}

// =============================================================================================
//  standalone self-test  (not part of the engine build)
//
//  build:  /opt/rocm/bin/hipcc -O3 --offload-arch=gfx906 -Iinclude -DQ27_NVFP4_SELFTEST \
//            src/q27_nvfp4.hip -o /tmp/q27_nvfp4_selftest
//  run:    /tmp/q27_nvfp4_selftest [shard] [checkrows] [iters] [dev]
//
//  It reads TWO real tensors straight out of the merged checkpoint — layers.0.mlp.gate_proj
//  (rows 17408, K 5120 -> the fully-unrolled kE path) and layers.0.mlp.down_proj (rows 5120,
//  K 17408 -> the blocked path) — and gates on:
//    1. q27_quant_perm byte-for-byte against a host reference of the permutation contract;
//    2. q27_proj_nvfp4 against the exact fp64 evaluation of the SAME int8 scheme  (should be
//       fp32-rounding small: this is the "did I lose a scale" gate);
//    3. q27_proj_nvfp4 against a full float dequant of the unquantized activation (this is the
//       activation-quantization error, not a defect);
//    4. throughput, interleaved paired reps, against kE's measured 0.1496 ms / 335.1 GB/s.
// =============================================================================================
#ifdef Q27_NVFP4_SELFTEST
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define HC(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ \
    printf("HIP %s @%d\n", hipGetErrorString(e_), __LINE__); exit(1);} }while(0)

// ---- host mirrors of the device numerics ----
static float h_bf2f(unsigned short b){ unsigned u=((unsigned)b)<<16; float f; memcpy(&f,&u,4); return f; }
static unsigned short h_f2bf(float f){ unsigned u; memcpy(&u,&f,4);
    return (unsigned short)((u + 0x7FFFu + ((u>>16)&1u)) >> 16); }
static float h_e4m3(unsigned b){                 // scales are non-negative: sign bit is always 0
    unsigned e=b>>3, m=b&7u;
    if(e==0) return (float)m * 0.001953125f;     // 2^-9
    if(e==15 && m==7) return INFINITY;
    unsigned bits=((e+120u)<<23)|(m<<20); float f; memcpy(&f,&bits,4); return f; }
static float h_e2m1(unsigned c){
    unsigned e=(c>>1)&3u, m=c&1u;
    float v = (e==0) ? (m?0.5f:0.0f) : (1.0f+0.5f*(float)m)*ldexpf(1.0f,(int)e-1);
    return (c&8u) ? -v : v; }
static int h_e2m1_x2(unsigned c){                // the int8 the weight decoder must produce
    unsigned e=(c>>1)&3u, m=c&1u;
    int mag = (e==0) ? (int)m : (int)((2u+m) << (e-1));
    return (c&8u) ? -mag : mag; }

// ---- minimal safetensors reader (8B LE header length, JSON header, payload) ----
struct St { FILE* f; std::string hdr; long long base; };
static bool st_open(St& s, const char* path){
    s.f = fopen(path, "rb");
    if(!s.f){ printf("open failed: %s\n", path); return false; }
    unsigned long long hl = 0;
    if(fread(&hl, 8, 1, s.f) != 1) return false;
    if(hl == 0 || hl > (1ull<<28)) return false;
    s.hdr.assign((size_t)hl, '\0');
    if(fread(&s.hdr[0], 1, (size_t)hl, s.f) != hl) return false;
    s.base = 8 + (long long)hl;
    return true;
}
static const char* st_after(const char* p, const char* end, const char* pat){
    const size_t n = strlen(pat);
    for(const char* q = p; q + n <= end; ++q) if(!memcmp(q, pat, n)) return q + n;
    return 0;
}
static bool st_get(const St& s, const char* name, long long* off, long long* len,
                   long long* shape, int* ndim){
    std::string key = "\""; key += name; key += "\":{";
    const size_t p = s.hdr.find(key);
    if(p == std::string::npos){ printf("tensor absent: %s\n", name); return false; }
    const size_t e = s.hdr.find('}', p);                 // tensor objects have no nested objects
    if(e == std::string::npos) return false;
    const char* b  = s.hdr.data() + p;
    const char* en = s.hdr.data() + e;
    const char* sp = st_after(b, en, "\"shape\":[");
    const char* dp = st_after(b, en, "\"data_offsets\":[");
    if(!sp || !dp) return false;
    int nd = 0;
    for(const char* q = sp; q < en && *q != ']'; ){
        if(*q >= '0' && *q <= '9'){ char* ep; long long v = strtoll(q, &ep, 10);
            if(nd < 8) shape[nd++] = v; q = ep; }
        else ++q;
    }
    *ndim = nd;
    { char* ep; long long a = strtoll(dp, &ep, 10);
      while(*ep == ',' || *ep == ' ') ++ep;
      long long z = strtoll(ep, &ep, 10);
      *off = s.base + a; *len = z - a; }
    return true;
}
static std::vector<unsigned char> st_read(St& s, long long off, long long len){
    std::vector<unsigned char> v((size_t)len);
    if(fseek(s.f, (long)off, SEEK_SET) != 0){ printf("seek failed\n"); exit(1); }
    if(fread(v.data(), 1, (size_t)len, s.f) != (size_t)len){ printf("short read\n"); exit(1); }
    return v;
}
static float st_f32(St& s, const char* name){
    long long off, len, sh[8]; int nd;
    if(!st_get(s, name, &off, &len, sh, &nd) || len != 4){ printf("bad scalar %s\n", name); exit(1); }
    std::vector<unsigned char> v = st_read(s, off, 4);
    float f; memcpy(&f, v.data(), 4); return f;
}

struct Tim { hipEvent_t a,b; Tim(){HC(hipEventCreate(&a));HC(hipEventCreate(&b));}
    void s(){HC(hipEventRecord(a));}
    float e(){HC(hipEventRecord(b));HC(hipEventSynchronize(b));
              float m; HC(hipEventElapsedTime(&m,a,b)); return m;} };

// One tensor end to end.
static void run_one(St& st, const char* prefix, int checkrows, int iters, double kE_ms, double kE_gbs)
{
    std::string nW = std::string(prefix) + ".weight";
    std::string nS = std::string(prefix) + ".weight_scale";
    long long wo, wl, so, sl, sh[8], sh2[8]; int nd, nd2;
    if(!st_get(st, nW.c_str(), &wo, &wl, sh,  &nd )) exit(1);
    if(!st_get(st, nS.c_str(), &so, &sl, sh2, &nd2)) exit(1);
    const int R  = (int)sh[0];
    const int K  = (int)sh[1] * 2;                  // packed U8 [R, K/2]
    const int KP = K >> 1, NG = K >> 4;
    if(nd != 2 || nd2 != 2 || sh2[0] != R || sh2[1] != NG){ printf("shape mismatch\n"); exit(1); }
    const float ws2      = st_f32(st, (std::string(prefix) + ".weight_scale_2").c_str());
    const float in_scale = st_f32(st, (std::string(prefix) + ".input_scale").c_str());
    if(checkrows > R) checkrows = R;

    printf("\n================ %s  [%d, %d] ================\n", prefix, R, K);
    printf("  packed %.2f MB   scales %.2f MB   ws2=%.9g   input_scale=%.9g\n",
           (double)R*KP/1e6, (double)R*NG/1e6, (double)ws2, (double)in_scale);
    if(in_scale == 1.0f) printf("  !! input_scale == 1.0 — the contract says no tensor has that\n");

    std::vector<unsigned char> Wp = st_read(st, wo, wl);
    std::vector<unsigned char> Sp = st_read(st, so, sl);

    // deterministic BF16 activation of width K
    std::vector<unsigned short> Xb(K);
    std::vector<float>          Xf(K);
    { unsigned s = 12345u;
      for(int i = 0; i < K; ++i){
          s = s*1664525u + 1013904223u;
          float v = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
          Xb[i] = h_f2bf(v); Xf[i] = h_bf2f(Xb[i]); } }

    // ---- host reference for q27_quant_perm (the permutation contract, verbatim) ----
    const float gsc = 1.0f / in_scale;
    const int   NGX = K >> 4;
    std::vector<signed char> Qref(K);
    std::vector<float>       Sref(NGX);
    for(int g = 0; g < NGX; ++g){
        float mx = 0.f;
        for(int j = 0; j < 16; ++j) mx = fmaxf(mx, fabsf(Xf[g*16+j] * gsc));
        const float sc  = mx * (1.0f/127.0f);
        const float inv = (sc > 0.f) ? 1.0f/sc : 0.f;
        Sref[g] = sc;
        for(int j = 0; j < 16; ++j){
            int q = (int)rintf(Xf[g*16+j] * gsc * inv);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            Qref[(g<<4) + ((j&1) ? 8+(j>>1) : (j>>1))] = (signed char)q;
        }
    }

    // ---- device ----
    unsigned char *dW, *dS; signed char *dQ; unsigned short *dX; float *dXs, *dY;
    HC(hipMalloc(&dW, (size_t)R*KP)); HC(hipMemcpy(dW, Wp.data(), (size_t)R*KP, hipMemcpyHostToDevice));
    HC(hipMalloc(&dS, (size_t)R*NG)); HC(hipMemcpy(dS, Sp.data(), (size_t)R*NG, hipMemcpyHostToDevice));
    HC(hipMalloc(&dX, (size_t)K*2));  HC(hipMemcpy(dX, Xb.data(), (size_t)K*2, hipMemcpyHostToDevice));
    HC(hipMalloc(&dQ, (size_t)K));
    HC(hipMalloc(&dXs, (size_t)NGX*4));
    HC(hipMalloc(&dY,  (size_t)R*4));

    q27_quant_perm(dX, dQ, dXs, K, in_scale, 0);
    HC(hipDeviceSynchronize()); HC(hipGetLastError());

    std::vector<signed char> Qdev(K); std::vector<float> Sdev(NGX);
    HC(hipMemcpy(Qdev.data(), dQ,  (size_t)K,     hipMemcpyDeviceToHost));
    HC(hipMemcpy(Sdev.data(), dXs, (size_t)NGX*4, hipMemcpyDeviceToHost));
    { int bad = 0, badsc = 0;
      for(int i = 0; i < K;   ++i) if(Qdev[i] != Qref[i]) ++bad;
      for(int g = 0; g < NGX; ++g) if(Sdev[g] != Sref[g]) ++badsc;
      printf("  [1] q27_quant_perm : %d/%d code mismatches, %d/%d scale mismatches  -> %s\n",
             bad, K, badsc, NGX, (bad==0 && badsc==0) ? "PASS" : "FAIL");
      // spot-check the contract itself on group 0
      printf("      perm spot-check g0: src j=0,1,2,3 -> dst 0,8,1,9  (%d,%d,%d,%d)\n",
             (int)Qdev[0], (int)Qdev[8], (int)Qdev[1], (int)Qdev[9]); }

    q27_nvfp4_t h = { dW, dS, ws2, in_scale, R, K };
    q27_proj_nvfp4(&h, dQ, dXs, dY, 0);
    HC(hipDeviceSynchronize()); HC(hipGetLastError());
    std::vector<float> Y(R);
    HC(hipMemcpy(Y.data(), dY, (size_t)R*4, hipMemcpyDeviceToHost));

    // ---- [2] exact evaluation of the SAME int8 scheme, in double ----
    // ---- [3] full float dequant against the UNquantized activation ----
    std::vector<double> Yq(checkrows), Ye(checkrows);
    for(int r = 0; r < checkrows; ++r){
        const unsigned char* wr = Wp.data() + (size_t)r*KP;
        const unsigned char* sr = Sp.data() + (size_t)r*NG;
        double aq = 0.0, ae = 0.0;
        for(int g = 0; g < NGX; ++g){
            const double gsW = (double)h_e4m3(sr[g]);
            long long dot = 0;
            for(int j = 0; j < 16; ++j){
                const int   e = g*16 + j;
                const unsigned c = (wr[e>>1] >> ((e&1)*4)) & 0xFu;
                const int   d = (g<<4) + ((j&1) ? 8+(j>>1) : (j>>1));
                dot += (long long)h_e2m1_x2(c) * (long long)Qref[d];
                ae  += (double)h_e2m1(c) * gsW * (double)ws2 * (double)Xf[e];
            }
            aq += (double)dot * gsW * (double)Sref[g] * 0.5;
        }
        Yq[r] = aq * (double)in_scale * (double)ws2;
        Ye[r] = ae;
    }
    auto cmp = [&](const char* tag, const std::vector<double>& ref){
        double d = 0, s = 0, mx = 0, dot = 0, na = 0, nb = 0;
        for(int r = 0; r < checkrows; ++r){
            const double a = Y[r], b = ref[r];
            d += (a-b)*(a-b); s += b*b; mx = fmax(mx, fabs(a-b));
            dot += a*b; na += a*a; nb += b*b;
        }
        printf("  %s relL2=%.6e  maxabs=%.6e  cosine=%.8f\n",
               tag, sqrt(d/s), mx, dot/sqrt(na*nb));
    };
    printf("  [2] vs exact int8-scheme (scale composition gate, %d rows):\n", checkrows);
    cmp("      ", Yq);
    printf("  [3] vs float dequant of the raw activation (quantization error, %d rows):\n", checkrows);
    cmp("      ", Ye);

    // ---- [4] throughput ----
    Tim t; double best = 1e18, sum = 0; const int reps = 7;
    for(int rep = 0; rep < reps; ++rep){
        t.s();
        for(int i = 0; i < iters; ++i) q27_proj_nvfp4(&h, dQ, dXs, dY, 0);
        const double ms = t.e() / iters;
        best = fmin(best, ms); sum += ms;
        printf("      rep %d: %.4f ms\n", rep, ms);
    }
    const double bytes = (double)R*KP + (double)R*NG;
    printf("  [4] best %.4f ms   %.1f GB/s of weight bytes   (mean %.4f ms)\n",
           best, bytes/(best*1e-3)/1e9, sum/reps);
    if(kE_ms > 0.0)
        printf("      vs kE measured %.4f ms / %.1f GB/s  ->  %.3fx  %s\n",
               kE_ms, kE_gbs, kE_ms/best, (best <= kE_ms*1.02) ? "NO REGRESSION" : "REGRESSION");
    printf("      whole-MLP projection of this class, 64 layers x 3: %.1f ms/token\n", best*64.0*3.0);

    HC(hipFree(dW)); HC(hipFree(dS)); HC(hipFree(dX));
    HC(hipFree(dQ)); HC(hipFree(dXs)); HC(hipFree(dY));
}

int main(int argc, char** argv)
{
    const char* shard = (argc > 1) ? argv[1]
        : "/data/qwen38-27b/model/model-00000-of-00003.safetensors";
    const int checkrows = (argc > 2) ? atoi(argv[2]) : 2048;
    const int iters     = (argc > 3) ? atoi(argv[3]) : 200;
    const int dev       = (argc > 4) ? atoi(argv[4]) : 0;

    HC(hipSetDevice(dev));
    hipDeviceProp_t pr; HC(hipGetDeviceProperties(&pr, dev));
    printf("device %d: %s  CUs=%d  wave=%d\n", dev, pr.gcnArchName, pr.multiProcessorCount, pr.warpSize);

    St st;
    if(!st_open(st, shard)){ printf("cannot open %s\n", shard); return 1; }
    printf("shard %s  (header %zu B)\n", shard, st.hdr.size());

    // K=5120 -> fully-unrolled kE geometry; the 0.1496 ms / 335.1 GB/s receipt is this shape.
    run_one(st, "model.language_model.layers.0.mlp.gate_proj", checkrows, iters, 0.1496, 335.1);
    // K=17408 -> blocked path (17 uint4/lane). No prior receipt; report the number.
    run_one(st, "model.language_model.layers.0.mlp.down_proj", checkrows, iters, 0.0, 0.0);

    fclose(st.f);
    return 0;
}
#endif // Q27_NVFP4_SELFTEST

// =============================================================================================
// FUSED SwiGLU + PERMUTING QUANTIZER.
// swiglu read gate/up (fp32), wrote d.act (bf16), and the very next kernel read d.act back to
// quantize it -- 64 launches per token plus a 8.7 KiB round trip through HBM for a value that was
// already in registers. This does the activation and the quantization in one pass and never
// materialises d.act. Emits the EVEN-THEN-ODD group order the nvfp4 consumer requires, per the
// activation permutation contract; the permutation stays in the producer, as it always has.
__global__ __launch_bounds__(256) void q27_k_swiglu_quant(
        const float* __restrict__ G, const float* __restrict__ U,
        signed char* __restrict__ Q, float* __restrict__ XS, int ngrp, float gscale)
{
    const int g = blockIdx.x * 256 + threadIdx.x;
    if (g >= ngrp) return;
    const size_t base = (size_t)g << 4;

    float v[16];
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float a = G[base + j];
        const float b = U[base + j];
        const float sw = a / (1.0f + __expf(-a));      // SiLU
        v[j] = sw * b * gscale;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    const float sc  = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    XS[g] = sc;

    unsigned d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int c = (int)rintf(v[j] * inv);
        c = c < -127 ? -127 : (c > 127 ? 127 : c);
        const unsigned byte = (unsigned)(c & 0xFF);
        const int dp = (j & 1) ? (8 + (j >> 1)) : (j >> 1);   // even-then-odd
        const int sh = (dp & 3) * 8;
        if      ((dp >> 2) == 0) d0 |= byte << sh;
        else if ((dp >> 2) == 1) d1 |= byte << sh;
        else if ((dp >> 2) == 2) d2 |= byte << sh;
        else                     d3 |= byte << sh;
    }
    *(uint4*)(Q + base) = make_uint4(d0, d1, d2, d3);
}

// CHUNK-BATCHED form: C positions in one launch (blockIdx.y = position), identical per-position
// arithmetic; the prefill sweep called the single-position kernel once per position per layer.
__global__ __launch_bounds__(256) void q27_k_swiglu_quant_b(
        const float* __restrict__ G, const float* __restrict__ U,
        signed char* __restrict__ Q, float* __restrict__ XS, int ngrp, float gscale,
        size_t gu_stride, size_t q_stride, size_t xs_stride)
{
    const int c = blockIdx.y;
    G += (size_t)c * gu_stride; U += (size_t)c * gu_stride; Q += (size_t)c * q_stride; XS += (size_t)c * xs_stride;
    const int g = blockIdx.x * 256 + threadIdx.x;
    if (g >= ngrp) return;
    const size_t base = (size_t)g << 4;
    float v[16];
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float a = G[base + j];
        const float b = U[base + j];
        const float sw = a / (1.0f + __expf(-a));
        v[j] = sw * b * gscale;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    const float sc  = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    XS[g] = sc;
    unsigned d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int cq = (int)rintf(v[j] * inv);
        cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
        const unsigned byte = (unsigned)(cq & 0xFF);
        const int dp = (j & 1) ? (8 + (j >> 1)) : (j >> 1);
        const int sh = (dp & 3) * 8;
        if      ((dp >> 2) == 0) d0 |= byte << sh;
        else if ((dp >> 2) == 1) d1 |= byte << sh;
        else if ((dp >> 2) == 2) d2 |= byte << sh;
        else                     d3 |= byte << sh;
    }
    *(uint4*)(Q + base) = make_uint4(d0, d1, d2, d3);
}
extern "C" int q27_swiglu_quant_b(const float* gate, const float* up, signed char* xq, float* xs,
                                  int n, float in_scale, int C, hipStream_t s)
{
    if (n <= 0 || (n & 15) || !(in_scale > 0.f) || C < 1) return 0;
    if (((uintptr_t)gate | (uintptr_t)up | (uintptr_t)xq) & 15u) return 0;
    const int ngrp = n >> 4;
    hipLaunchKernelGGL(q27_k_swiglu_quant_b, dim3((ngrp + 255) / 256, C), dim3(256), 0, s,
                       gate, up, xq, xs, ngrp, 1.0f / in_scale, (size_t)n, (size_t)n, (size_t)ngrp);
    return 1;
}

extern "C" int q27_swiglu_quant(const float* gate, const float* up, signed char* xq, float* xs,
                                int n, float in_scale, hipStream_t s)
{
    if (n <= 0 || (n & 15) || !(in_scale > 0.f)) return 0;
    if (((uintptr_t)gate | (uintptr_t)up | (uintptr_t)xq) & 15u) return 0;
    const int ngrp = n >> 4;
    hipLaunchKernelGGL(q27_k_swiglu_quant, dim3((ngrp + 255) / 256), dim3(256), 0, s,
                       gate, up, xq, xs, ngrp, 1.0f / in_scale);
    return 1;
}

// =============================================================================================
// gate+up -> SwiGLU -> permuting quantize, AS ONE PRODUCTION OPERATION.
// The production graph, not the old kernel boundaries, sets the kernel. gate and up have identical
// geometry, identical activation and identical activation scales, and their outputs are consumed
// only by SwiGLU, whose output is consumed only by the quantizer, whose output is consumed only by
// down_proj. Nothing between needs to exist in HBM.
//
// ONE BLOCK OWNS 16 CONSECUTIVE OUTPUT ROWS = EXACTLY ONE nvfp4 QUANTIZATION GROUP. That is what
// makes the fusion possible: the quantizer needs a group of 16 contiguous outputs, so a row-parallel
// projection with 4 rows per block could never own one. 256 threads = 4 waves, each wave taking 4
// rows in sequence.
//
// Second win, less obvious than the deleted buffers: the activation and its scales are loaded ONCE
// per block and reused across all 4 rows AND both matrices. The two-kernel form reloaded them per
// row per matrix.
// Deletes d.pa and d.pb entirely (2 x 4352 x 4 B per layer) plus one launch per layer.
template<int UPL2>
__global__ __launch_bounds__(256) void q27_k_nvfp4_gu_swiglu_q(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float dsc,
        signed char* __restrict__ Q, float* __restrict__ QS)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wv   = threadIdx.x >> 6;
    const int grp  = blockIdx.x;
    const int row0 = grp * 16 + wv * 4;
    const int KP   = UPL2 * 64 * 8;
    const int NG   = UPL2 * 64;

    // activation once per BLOCK, reused by 4 rows x 2 matrices
    uint4 xv[UPL2]; float xsc[UPL2];
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;
        xv[i]  = *(const uint4*)(Xp + (size_t)g * 4);
        xsc[i] = Xs[g] * 0.5f;
    }

    __shared__ float sg16[16], su16[16];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row0 + r;
        const int rr  = (row < rows) ? row : (rows - 1);      // clamp, never read OOB
        const uint2*         wg = (const uint2*)(WG + (size_t)rr * (size_t)KP);
        const uint2*         wu = (const uint2*)(WU + (size_t)rr * (size_t)KP);
        const unsigned char* sg = SG + (size_t)rr * (size_t)NG;
        const unsigned char* su = SU + (size_t)rr * (size_t)NG;

        uint2 vg[UPL2], vu[UPL2]; unsigned char cg[UPL2], cu[UPL2];
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            const int g = lane + i * 64;
            vg[i] = q27_ld8nt(&wg[g]); vu[i] = q27_ld8nt(&wu[g]);   // weights: non-temporal
            cg[i] = sg[g]; cu[i] = su[g];                           // scales: keep cached
        }
        float accg = 0.f, accu = 0.f;
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            const uint2 e = make_uint2(xv[i].x, xv[i].y);
            const uint2 o = make_uint2(xv[i].z, xv[i].w);
            accg = fmaf((float)q27_grp16(vg[i], e, o), q27_e4m3(cg[i]) * xsc[i], accg);
            accu = fmaf((float)q27_grp16(vu[i], e, o), q27_e4m3(cu[i]) * xsc[i], accu);
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1) { accg += __shfl_down(accg, o, 64); accu += __shfl_down(accu, o, 64); }
        if (lane == 0 && row < rows) { sg16[wv * 4 + r] = accg * ag; su16[wv * 4 + r] = accu * au; }
    }
    __syncthreads();

    // Epilogue across 16 LANES, not one thread. The first version ran this on threadIdx.x==0 while
    // 255 threads waited at the barrier, serialising 16 __expf calls; that alone made the whole
    // fusion lose (proj_nvfp4 3.004 -> 3.660 ms). One lane per value, max by shuffle.
    if (threadIdx.x < 64) {
        const int j = lane & 15;
        const float a = sg16[j], b = su16[j];
        const float vj = (a / (1.0f + __expf(-a))) * b * dsc;
        float mx = fabsf(vj);
        #pragma unroll
        for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 16));
        mx = __shfl(mx, 0, 16);
        const float sc  = mx * (1.0f / 127.0f);
        const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
        if (lane == 0) QS[grp] = sc;
        int c = (int)rintf(vj * inv);
        c = c < -127 ? -127 : (c > 127 ? 127 : c);
        const int dp = (j & 1) ? (8 + (j >> 1)) : (j >> 1);   // even-then-odd, in the producer
        if (lane < 16) ((signed char*)Q)[((size_t)grp << 4) + dp] = (signed char)c;
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_mlp_gu_swiglu_q(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                   const signed char* xq_perm, const float* xs,
                                   signed char* dq, float* ds, float down_in_scale, hipStream_t s)
{
    // REJECTED on measurement, kept behind a flag as evidence. Full gate+up+SwiGLU+quant fusion
// composed 20.181 ms mean against 19.160 for the two-kernel form (gate+up fused, SwiGLU+quant
// fused). First version also had a single-thread epilogue serialising 16 __expf calls; fixing that
// to 16 lanes narrowed it but did not reverse it. The cost is structural: 4 rows per wave serialise
// on each row's shuffle reduction, where 1 row per wave gives 4x the blocks. Deleting the two
// intermediate buffers did not pay for that. Q27_MLP_FUSE=1 re-enables.
    if (!q27_env_flag("Q27_MLP_FUSE", false)) return 0;
    if (wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120) return 0;
    if (wg->rows & 15) return 0;                       // whole groups of 16 only
    if (!(down_in_scale > 0.f)) return 0;
    hipLaunchKernelGGL((q27_k_nvfp4_gu_swiglu_q<5>), dim3((unsigned)(wg->rows >> 4)), dim3(256),
                       0, s, wg->w, wg->gs, wu->w, wu->gs, (const unsigned*)xq_perm, xs,
                       wg->in_scale * wg->ws2, wu->in_scale * wu->ws2, wg->rows,
                       1.0f / down_in_scale, dq, ds);
    return 1;
}

// Q27_LS_Q8: int8 PER-64-GROUP execution mirror of an NVFP4 tensor. Each E2M1 value times its e4m3
// per-16 scale is re-expressed as int8 against one fp32 scale per 64 elements (exact whenever the four
// sub-group scales are within ~21x of each other), natural K order, so the consumer accumulates 64 MACs
// in int32 before touching a float. gfx906 executes int8 dot4 only; this is the representation it wants.
__global__ void q27_k_nvfp4_to_i8g64(const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
                                     signed char* __restrict__ W8, float* __restrict__ S64, int rows, int K)
{
    const int g = blockIdx.x * 256 + threadIdx.x;
    const int NG = K >> 6, NG16 = K >> 4;
    if (g >= rows * NG) return;
    const int r = g / NG, gr = g % NG;
    static const float e2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v[64]; float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        const int e = gr * 64 + j;
        const unsigned b = W[(size_t)r * (K >> 1) + (e >> 1)];
        const unsigned nib = (e & 1) ? (b >> 4) : (b & 15u);
        const float gs = q27_e4m3(S[(size_t)r * NG16 + (e >> 4)]);
        v[j] = e2m1[nib & 7] * ((nib & 8) ? -1.f : 1.f) * gs;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    const float sc = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    S64[(size_t)r * NG + gr] = sc;
    signed char* wr = W8 + (size_t)r * K + gr * 64;
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        int q = (int)rintf(v[j] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        wr[j] = (signed char)q;
    }
}
extern "C" int q27_nvfp4_to_i8g64_conv(const q27_nvfp4_t* src, q27_i8g_t* dst, hipStream_t st) {
    if (!src || !src->w || (src->K % 64)) return 0;
    dst->rows = src->rows; dst->K = src->K; dst->alpha = src->in_scale * src->ws2;
    signed char* w8 = nullptr; float* s64 = nullptr;
    if (hipMalloc((void**)&w8, (size_t)src->rows * src->K) != hipSuccess ||
        hipMalloc((void**)&s64, (size_t)src->rows * (src->K >> 6) * 4) != hipSuccess) {
        std::fprintf(stderr, "Q8: int8/64 mirror allocation failed\n"); dst->w = nullptr; dst->s = nullptr; return 0;
    }
    const int NG = src->K >> 6;
    hipLaunchKernelGGL(q27_k_nvfp4_to_i8g64, dim3((size_t)((size_t)src->rows * NG + 255) / 256), dim3(256), 0, st,
                       src->w, src->gs, w8, s64, src->rows, src->K);
    dst->w = w8; dst->s = s64;
    return 1;
}

// CHUNK-BATCHED swiglu + int8 quantize with PER-64 scales, natural order (the int8/64 down's input).
// 16 elements per thread as q27_k_swiglu_quant_b; the 64-group max is four consecutive threads.
__global__ __launch_bounds__(256) void q27_k_swiglu_quant_b_g64(
        const float* __restrict__ G, const float* __restrict__ U,
        signed char* __restrict__ Q, float* __restrict__ XS, int ngrp, float gscale,
        size_t gu_stride, size_t q_stride, size_t xs_stride)
{
    const int c = blockIdx.y;
    G += (size_t)c * gu_stride; U += (size_t)c * gu_stride; Q += (size_t)c * q_stride; XS += (size_t)c * xs_stride;
    const int g = blockIdx.x * 256 + threadIdx.x;
    const bool ok = g < ngrp;
    const size_t base = (size_t)(ok ? g : ngrp - 1) << 4;
    float v[16];
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float a = G[base + j];
        const float b = U[base + j];
        const float sw = a / (1.0f + __expf(-a));
        v[j] = sw * b * gscale;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    mx = fmaxf(mx, __shfl_down(mx, 1, 4));
    mx = fmaxf(mx, __shfl_down(mx, 2, 4));
    mx = __shfl(mx, 0, 4);
    const float sc  = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    if (ok && (g & 3) == 0) XS[g >> 2] = sc;
    unsigned d[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int cq = (int)rintf(v[j] * inv);
        cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
        d[j >> 2] |= ((unsigned)(cq & 0xFF)) << ((j & 3) * 8);
    }
    if (ok) *(uint4*)(Q + base) = make_uint4(d[0], d[1], d[2], d[3]);
}
extern "C" int q27_swiglu_quant_b_g64(const float* gate, const float* up, signed char* xq, float* xs,
                                      int n, float in_scale, int C, hipStream_t s)
{
    if (n <= 0 || (n & 63) || !(in_scale > 0.f) || C < 1) return 0;
    if (((uintptr_t)gate | (uintptr_t)up | (uintptr_t)xq) & 15u) return 0;
    const int ngrp = n >> 4;
    hipLaunchKernelGGL(q27_k_swiglu_quant_b_g64, dim3((ngrp + 255) / 256, C), dim3(256), 0, s,
                       gate, up, xq, xs, ngrp, 1.0f / in_scale, (size_t)n, (size_t)n, (size_t)(n >> 6));
    return 1;
}

// Q27_LS_Q8 stage 2: int8 PER-64-GROUP execution mirror of an fp8 projection (fp32 group scales), natural K order.
__global__ void q27_k_fp8_to_i8g64(const unsigned char* __restrict__ F, signed char* __restrict__ W8,
                                   float* __restrict__ S64, int rows, int K)
{
    const int g = blockIdx.x * 256 + threadIdx.x;
    const int NG = K >> 6;
    if (g >= rows * NG) return;
    const int r = g / NG, gr = g % NG;
    const unsigned char* fr = F + (size_t)r * K + gr * 64;
    float v[64]; float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 64; ++j) { v[j] = q27_e4m3s(fr[j]); mx = fmaxf(mx, fabsf(v[j])); }
    const float sc = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    S64[(size_t)r * NG + gr] = sc;
    signed char* wr = W8 + (size_t)r * K + gr * 64;
#pragma unroll
    for (int j = 0; j < 64; ++j) { int q = (int)rintf(v[j] * inv); q = q < -127 ? -127 : (q > 127 ? 127 : q); wr[j] = (signed char)q; }
}
extern "C" int q27_fp8_to_i8g64_conv(const q27_fp8_t* src, q27_i8g_t* dst, hipStream_t st) {
    if (!src || !src->w || (src->K % 64)) return 0;
    dst->rows = src->rows; dst->K = src->K; dst->alpha = src->in_scale * src->wscale;
    signed char* w8 = nullptr; float* s64 = nullptr;
    if (hipMalloc((void**)&w8, (size_t)src->rows * src->K) != hipSuccess ||
        hipMalloc((void**)&s64, (size_t)src->rows * (src->K >> 6) * 4) != hipSuccess) {
        std::fprintf(stderr, "Q8: int8/64 projection mirror allocation failed\n"); dst->w = nullptr; dst->s = nullptr; return 0;
    }
    const int NG = src->K >> 6;
    hipLaunchKernelGGL(q27_k_fp8_to_i8g64, dim3((size_t)((size_t)src->rows * NG + 255) / 256), dim3(256), 0, st, src->w, w8, s64, src->rows, src->K);
    dst->w = w8; dst->s = s64;
    return 1;
}

// Q27_LS_Q8: several fp8 projections that share the same input -> ONE contiguous int8/64 mirror (rows stacked), so
// the rocBLAS path can run them as one GEMM (k/v alone: 9 TMAC/s; inside the q GEMM: 17.6). Each dst keeps its own
// rows/alpha and points into the shared buffers.
extern "C" int q27_fp8_to_i8g64_cat(const q27_fp8_t* const* srcs, int n, q27_i8g_t* dsts, hipStream_t st) {
    if (!srcs || n < 1 || !dsts) return 0;
    size_t rows = 0; const int K = srcs[0]->K;
    for (int i = 0; i < n; ++i) { if (!srcs[i] || !srcs[i]->w || srcs[i]->K != K || (K % 64)) return 0; rows += (size_t)srcs[i]->rows; }
    signed char* w8 = nullptr; float* s64 = nullptr;
    if (hipMalloc((void**)&w8, rows * (size_t)K) != hipSuccess || hipMalloc((void**)&s64, rows * (size_t)(K >> 6) * 4) != hipSuccess) {
        std::fprintf(stderr, "Q8: concatenated int8/64 mirror allocation failed\n"); return 0;
    }
    const int NG = K >> 6; size_t off = 0;
    for (int i = 0; i < n; ++i) {
        const q27_fp8_t* s = srcs[i];
        hipLaunchKernelGGL(q27_k_fp8_to_i8g64, dim3((size_t)((size_t)s->rows * NG + 255) / 256), dim3(256), 0, st,
                           s->w, w8 + off * (size_t)K, s64 + off * (size_t)NG, s->rows, K);
        dsts[i].w = w8 + off * (size_t)K; dsts[i].s = s64 + off * (size_t)NG; dsts[i].rows = s->rows; dsts[i].K = K; dsts[i].alpha = s->in_scale * s->wscale;
        off += (size_t)s->rows;
    }
    return 1;
}


// =============================================================================================
// ROW-BATCHED DECODE FORMS (Q27_SPEC verify). NR activation rows of ONE sequence share every
// weight load. The 16-nibble weight group is decoded to int8x4 words ONCE per (row, group) and
// only the four dot4 + one fmaf are repeated per activation row: the single-row form spends most
// of its VALU in the decode, and repeating it per row is what made NR=2 cost 1.4x (measured).
// Per-(row, n) arithmetic and accumulation order are those of q27_k_nvfp4_w64 / w64_gu / w64t2
// (same integer dot4 sequence, same fmaf chain), so every output is bit-identical.
// Activation rows are contiguous ([NR][K/2] permuted int8 as unsigned words, [NR][K/16] scales).
__device__ __forceinline__ void q27_grp16_pre(uint2 pk, unsigned* w) {
    w[0] = q27_dq4( pk.x       & 0x0F0F0F0Fu);
    w[1] = q27_dq4((pk.x >> 4) & 0x0F0F0F0Fu);
    w[2] = q27_dq4( pk.y       & 0x0F0F0F0Fu);
    w[3] = q27_dq4((pk.y >> 4) & 0x0F0F0F0Fu);
}
__device__ __forceinline__ int q27_grp16_dot(const unsigned* w, uint4 xv) {   // xe = (x, y), xo = (z, w)
    int a = 0;
    a = q27_d4(w[0], xv.x, a);
    a = q27_d4(w[1], xv.z, a);
    a = q27_d4(w[2], xv.y, a);
    a = q27_d4(w[3], xv.w, a);
    return a;
}

template<int UPL2, int NR, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride, int xst)
{
    const int row0 = (blockIdx.x * 4 + (threadIdx.x >> 6)) * R;
    if (row0 >= rows) return;
    const int lane = threadIdx.x & 63;
    const int KP = UPL2 * 64 * 8;
    const int NG = UPL2 * 64;
    const uint2*         wr[R];
    const unsigned char* sr[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        wr[r] = (const uint2*)(W + (size_t)rr * (size_t)KP);
        sr[r] = S + (size_t)rr * (size_t)NG;
    }
    uint2 wv[R][UPL2];  unsigned char cg[R][UPL2];  uint4 xv[NR][UPL2];  float xs[NR][UPL2];
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;
        #pragma unroll
        for (int r = 0; r < R; ++r) { wv[r][i] = q27_ld8nt(&wr[r][g]); cg[r][i] = sr[r][g]; }
        #pragma unroll
        for (int n = 0; n < NR; ++n) {
            xv[n][i] = *(const uint4*)(Xp + (size_t)n * (xst >> 2) + (size_t)g * 4);
            xs[n][i] = Xs[(size_t)n * (xst >> 4) + g];
        }
    }
    float acc[R][NR];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int n = 0; n < NR; ++n) acc[r][n] = 0.f;
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float gsc = (NR > 1) ? q27_e4m3_sel(cg[r][i]) : q27_e4m3(cg[r][i]);   // select form for the multi-row instances (the branchy form is faster at NR=1)
            unsigned wd[4]; q27_grp16_pre(wv[r][i], wd);
            #pragma unroll
            for (int n = 0; n < NR; ++n)
                acc[r][n] = fmaf((float)q27_grp16_dot(wd, xv[n][i]), gsc * xs[n][i] * 0.5f, acc[r][n]);
        }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1)
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int n = 0; n < NR; ++n) acc[r][n] += __shfl_down(acc[r][n], o, 64);
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) continue;
            #pragma unroll
            for (int n = 0; n < NR; ++n) Y[(size_t)n * ystride + row0 + r] = acc[r][n] * alpha;
        }
    }
}

template<int UPL2, int NR, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64_gu_nr(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU, int ystride, int xst)
{
    const int row0 = (blockIdx.x * 4 + (threadIdx.x >> 6)) * R;
    if (row0 >= rows) return;
    const int lane = threadIdx.x & 63;
    const int KP = UPL2 * 64 * 8;
    const int NG = UPL2 * 64;
    const uint2* wg[R]; const uint2* wu[R]; const unsigned char* sg[R]; const unsigned char* su[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        wg[r] = (const uint2*)(WG + (size_t)rr * (size_t)KP);
        wu[r] = (const uint2*)(WU + (size_t)rr * (size_t)KP);
        sg[r] = SG + (size_t)rr * (size_t)NG;
        su[r] = SU + (size_t)rr * (size_t)NG;
    }
    uint2 vg[R][UPL2], vu[R][UPL2];  unsigned char cg[R][UPL2], cu[R][UPL2];
    uint4 xv[NR][UPL2];  float xsc[NR][UPL2];
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        const int g = lane + i * 64;
        #pragma unroll
        for (int r = 0; r < R; ++r) { vg[r][i] = wg[r][g]; vu[r][i] = wu[r][g]; cg[r][i] = sg[r][g]; cu[r][i] = su[r][g]; }
        #pragma unroll
        for (int n = 0; n < NR; ++n) {
            xv[n][i]  = *(const uint4*)(Xp + (size_t)n * (xst >> 2) + (size_t)g * 4);
            xsc[n][i] = Xs[(size_t)n * (xst >> 4) + g] * 0.5f;
        }
    }
    float accg[R][NR], accu[R][NR];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int n = 0; n < NR; ++n) { accg[r][n] = 0.f; accu[r][n] = 0.f; }
    #pragma unroll
    for (int i = 0; i < UPL2; ++i) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float eg = (NR > 1) ? q27_e4m3_sel(cg[r][i]) : q27_e4m3(cg[r][i]), eu = (NR > 1) ? q27_e4m3_sel(cu[r][i]) : q27_e4m3(cu[r][i]);
            unsigned wgd[4], wud[4]; q27_grp16_pre(vg[r][i], wgd); q27_grp16_pre(vu[r][i], wud);
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                accg[r][n] = fmaf((float)q27_grp16_dot(wgd, xv[n][i]), eg * xsc[n][i], accg[r][n]);
                accu[r][n] = fmaf((float)q27_grp16_dot(wud, xv[n][i]), eu * xsc[n][i], accu[r][n]);
            }
        }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1)
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int n = 0; n < NR; ++n) { accg[r][n] += __shfl_down(accg[r][n], o, 64); accu[r][n] += __shfl_down(accu[r][n], o, 64); }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) continue;
            #pragma unroll
            for (int n = 0; n < NR; ++n) { YG[(size_t)n * ystride + row0 + r] = accg[r][n] * ag; YU[(size_t)n * ystride + row0 + r] = accu[r][n] * au; }
        }
    }
}

template<int FULL2, int TAIL, int NR, int R>
__global__ __launch_bounds__(256) void q27_k_nvfp4_w64t2_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs, int xst)
{
    const int row0 = (blockIdx.x * 4 + (threadIdx.x >> 6)) * R;
    if (row0 >= rows) return;
    const int lane = threadIdx.x & 63;
    const int NG = FULL2 * 64 + TAIL;
    const int KP = NG * 8;
    const bool tailact = (lane < TAIL);
    const uint2* wr[R]; const unsigned char* sr[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        wr[r] = (const uint2*)(W + (size_t)rr * (size_t)KP);
        sr[r] = S + (size_t)rr * (size_t)NG;
    }
    uint2 wv[R][FULL2 + 1];  unsigned char cg[R][FULL2 + 1];  uint4 xv[NR][FULL2 + 1];  float xs[NR][FULL2 + 1];
    #pragma unroll
    for (int i = 0; i < FULL2 + 1; ++i) {
        const int g = i * 64 + ((i < FULL2) ? lane : (tailact ? lane : 0));   // clamped on tail
        #pragma unroll
        for (int r = 0; r < R; ++r) { wv[r][i] = q27_ld8nt(&wr[r][g]); cg[r][i] = sr[r][g]; }
        #pragma unroll
        for (int n = 0; n < NR; ++n) {
            xv[n][i] = *(const uint4*)(Xp + (size_t)n * (xst >> 2) + (size_t)g * 4);
            xs[n][i] = Xs[(size_t)n * (xst >> 4) + g];
        }
    }
    float acc[R][NR];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int n = 0; n < NR; ++n) acc[r][n] = 0.f;
    #pragma unroll
    for (int i = 0; i < FULL2; ++i) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float ec = (NR > 1) ? q27_e4m3_sel(cg[r][i]) : q27_e4m3(cg[r][i]);
            unsigned wd[4]; q27_grp16_pre(wv[r][i], wd);
            #pragma unroll
            for (int n = 0; n < NR; ++n)
                acc[r][n] = fmaf((float)q27_grp16_dot(wd, xv[n][i]), ec * xs[n][i] * 0.5f, acc[r][n]);
        }
    }
    if (tailact) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float ec = (NR > 1) ? q27_e4m3_sel(cg[r][FULL2]) : q27_e4m3(cg[r][FULL2]);
            unsigned wd[4]; q27_grp16_pre(wv[r][FULL2], wd);
            #pragma unroll
            for (int n = 0; n < NR; ++n)
                acc[r][n] = fmaf((float)q27_grp16_dot(wd, xv[n][FULL2]), ec * xs[n][FULL2] * 0.5f, acc[r][n]);
        }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1)
        #pragma unroll
        for (int r = 0; r < R; ++r)
            #pragma unroll
            for (int n = 0; n < NR; ++n) acc[r][n] += __shfl_down(acc[r][n], o, 64);
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) continue;
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                float v = acc[r][n] * alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)n * rstride + row0 + r]);
                Y[(size_t)n * ystride + row0 + r] = v;
            }
        }
    }
}

// native 8-row forms (defined after the LDS kernels below); the _nr/_nrs entries route NR > 4 here
static int q27_nr_lds_min();   // Q27_NR_LDS_MIN threshold (defined with the LDS forms below)
static int q27_nvfp4_lds8(const q27_nvfp4_t* w, const signed char* xq, int xst, const float* xs, float* y, int ystride, int NR, hipStream_t s);
static int q27_nvfp4_gu_lds8(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, int xst, const float* xs, float* yg, float* yu, int ystride, int NR, hipStream_t s);
static int q27_nvfp4_down_lds8(const q27_nvfp4_t* w, const signed char* xq, int xst, const float* xs, float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s);
#define Q27_NR_CASES(LAUNCH) \
    switch (NR) { case 1: LAUNCH(1); break; case 2: LAUNCH(2); break; \
                  case 3: LAUNCH(3); break; case 4: LAUNCH(4); break; default: return 0; }
#define Q27_NR_R 1      // weight rows per wave in the row-batched NVFP4 forms (R=2 measured slower: fewer waves than SIMDs on the small shards)

// K=5120, fp32 out (lm_head shard).
extern "C" int q27_proj_nvfp4_nr(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                                 float* y, int ystride, int NR, hipStream_t s)
{
    if (!w || w->K != 5120 || NR < 1 || NR > 8) return 0;
    if (NR >= q27_nr_lds_min()) return q27_nvfp4_lds8(w, xq_perm, w->K, xs, y, ystride, NR, s);
    const float alpha = w->in_scale * w->ws2;
    const dim3 g64((unsigned)((w->rows + 4 * Q27_NR_R - 1) / (4 * Q27_NR_R))), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_w64_nr<5,N,Q27_NR_R>), g64, b64, 0, s, w->w, w->gs, \
                     (const unsigned*)xq_perm, xs, alpha, w->rows, y, ystride, w->K)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}

// K=5120 fused gate+up (rows=4352 under 4-way TP).
extern "C" int q27_proj_nvfp4_gu_nr(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int ystride, int NR, hipStream_t s)
{
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || NR < 1 || NR > 8) return 0;
    if (NR >= q27_nr_lds_min()) return q27_nvfp4_gu_lds8(wg, wu, xq_perm, wg->K, xs, yg, yu, ystride, NR, s);
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g64((unsigned)((wg->rows + 4 * Q27_NR_R - 1) / (4 * Q27_NR_R))), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_w64_gu_nr<5,N,Q27_NR_R>), g64, b64, 0, s, wg->w, wg->gs, wu->w, wu->gs, \
                     (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu, ystride, wg->K)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}

// K=4352 (down_proj under 4-way TP), fp32 partial out with the residual share folded in.
extern "C" int q27_proj_nvfp4_down_nr(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                                      float* y, int ystride, const unsigned short* radd, int rstride,
                                      float rs, int NR, hipStream_t s)
{
    if (!w || w->K != 4352 || NR < 1 || NR > 8) return 0;
    if (NR >= q27_nr_lds_min()) return q27_nvfp4_down_lds8(w, xq_perm, w->K, xs, y, ystride, radd, rstride, rs, NR, s);
    const float alpha = w->in_scale * w->ws2;
    const dim3 g64((unsigned)((w->rows + 4 * Q27_NR_R - 1) / (4 * Q27_NR_R))), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_w64t2_nr<4,16,N,Q27_NR_R>), g64, b64, 0, s, w->w, w->gs, \
                     (const unsigned*)xq_perm, xs, alpha, w->rows, y, ystride, radd, rstride, rs, w->K)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}


// Q27_SPEC draft layer: the register forms with an explicit activation row stride (slices of wider rows).
extern "C" int q27_proj_nvfp4_gu_nrs(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, int xst,
                                     const float* xs, float* yg, float* yu, int ystride, int NR, hipStream_t s)
{
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || NR < 1 || NR > 8) return 0;
    if (NR >= q27_nr_lds_min()) return q27_nvfp4_gu_lds8(wg, wu, xq_perm, xst, xs, yg, yu, ystride, NR, s);
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g64((unsigned)((wg->rows + 3) / 4)), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_w64_gu_nr<5,N,1>), g64, b64, 0, s, wg->w, wg->gs, wu->w, wu->gs, \
                     (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu, ystride, xst)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
extern "C" int q27_proj_nvfp4_down_nrs(const q27_nvfp4_t* w, const signed char* xq_perm, int xst, const float* xs,
                                       float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s)
{
    if (!w || w->K != 4352 || NR < 1 || NR > 8) return 0;
    if (NR >= q27_nr_lds_min()) return q27_nvfp4_down_lds8(w, xq_perm, xst, xs, y, ystride, radd, rstride, rs, NR, s);
    const float alpha = w->in_scale * w->ws2;
    const dim3 g64((unsigned)((w->rows + 3) / 4)), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_w64t2_nr<4,16,N,1>), g64, b64, 0, s, w->w, w->gs, \
                     (const unsigned*)xq_perm, xs, alpha, w->rows, y, ystride, radd, rstride, rs, xst)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}

// =============================================================================================
// LDS-STAGED ROW-BATCHED FORMS (Q27_NR_LDS). Measured: the register-resident NR forms hold all
// NR x UPL2 activation quads in VGPRs (74 / 116 / ~150 VGPRs at NR = 2 / 3 / 4 -> 3 / 2 / 1
// waves per SIMD) and lose exactly the occupancy the single-row kernel (63 VGPRs, 4 waves) uses
// to keep HBM busy. Here the block stages the NR activation rows and scales in LDS once, each
// wave walks rpw rows with the row's weights issued upfront, and the activations of ONE group at
// a time are read from LDS, double-buffered behind compiler barriers so nothing is hoisted: the
// live set is weights + two groups. Per-(row, n) arithmetic and accumulation order are those of
// the single-row kernels, so outputs stay bit-identical.
#define Q27_LDS_FENCE() asm volatile("" ::: "memory")
template<int UPL2, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_lds_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride, int rpw)
{
    constexpr int NG = UPL2 * 64, KP = NG * 8, KB = NG * 16;
    __shared__ __align__(16) unsigned char sx[NR][KB];
    __shared__ float ss[NR][NG];
    for (int i = threadIdx.x; i < NR * (KB / 16); i += 256) {
        const int n = i / (KB / 16), o = i - n * (KB / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)Xp)[(size_t)n * (KB / 16) + o];
    }
    for (int i = threadIdx.x; i < NR * NG; i += 256) { const int n = i / NG, o = i - n * NG; ss[n][o] = Xs[(size_t)n * NG + o]; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    for (int r = 0; r < rpw; ++r) {
        const int row = rowbase + r;
        if (row >= rows) break;
        uint2 wv[UPL2]; unsigned char cg[UPL2];
        {
            const uint2* wr = (const uint2*)(W + (size_t)row * KP);  const unsigned char* sr = S + (size_t)row * NG;
            #pragma unroll
            for (int i = 0; i < UPL2; ++i) { const int g = lane + i * 64; wv[i] = q27_ld8nt(&wr[g]); cg[i] = sr[g]; }
        }
        float acc[NR];
        uint4 xa[NR]; float sa[NR];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { acc[n] = 0.f; xa[n] = *(const uint4*)(sx[n] + (size_t)lane * 16); sa[n] = ss[n][lane]; }
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            uint4 xb[NR]; float sb[NR];
            if (i + 1 < UPL2) {
                const int g1 = lane + (i + 1) * 64;
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + (size_t)g1 * 16); sb[n] = ss[n][g1]; }
            }
            Q27_LDS_FENCE();
            const float gsc = q27_e4m3(cg[i]);
            unsigned wd[4]; q27_grp16_pre(wv[i], wd);
            #pragma unroll
            for (int n = 0; n < NR; ++n) acc[n] = fmaf((float)q27_grp16_dot(wd, xa[n]), gsc * sa[n] * 0.5f, acc[n]);
            Q27_LDS_FENCE();
            if (i + 1 < UPL2) {
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xa[n] = xb[n]; sa[n] = sb[n]; }
            }
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1)
            #pragma unroll
            for (int n = 0; n < NR; ++n) acc[n] += __shfl_down(acc[n], o, 64);
        if (lane == 0) {
            #pragma unroll
            for (int n = 0; n < NR; ++n) Y[(size_t)n * ystride + row] = acc[n] * alpha;
        }
    }
}

template<int UPL2, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_gu_lds_nr(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU, int ystride, int rpw)
{
    constexpr int NG = UPL2 * 64, KP = NG * 8, KB = NG * 16;
    __shared__ __align__(16) unsigned char sx[NR][KB];
    __shared__ float ss[NR][NG];
    for (int i = threadIdx.x; i < NR * (KB / 16); i += 256) {
        const int n = i / (KB / 16), o = i - n * (KB / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)Xp)[(size_t)n * (KB / 16) + o];
    }
    for (int i = threadIdx.x; i < NR * NG; i += 256) { const int n = i / NG, o = i - n * NG; ss[n][o] = Xs[(size_t)n * NG + o] * 0.5f; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    for (int r = 0; r < rpw; ++r) {
        const int row = rowbase + r;
        if (row >= rows) break;
        uint2 vg[UPL2], vu[UPL2]; unsigned char cg[UPL2], cu[UPL2];
        {
            const uint2* wg = (const uint2*)(WG + (size_t)row * KP); const uint2* wu = (const uint2*)(WU + (size_t)row * KP);
            const unsigned char* sg = SG + (size_t)row * NG; const unsigned char* su = SU + (size_t)row * NG;
            #pragma unroll
            for (int i = 0; i < UPL2; ++i) { const int g = lane + i * 64; vg[i] = wg[g]; vu[i] = wu[g]; cg[i] = sg[g]; cu[i] = su[g]; }
        }
        float accg[NR], accu[NR];
        uint4 xa[NR]; float sa[NR];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { accg[n] = 0.f; accu[n] = 0.f; xa[n] = *(const uint4*)(sx[n] + (size_t)lane * 16); sa[n] = ss[n][lane]; }
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            uint4 xb[NR]; float sb[NR];
            if (i + 1 < UPL2) {
                const int g1 = lane + (i + 1) * 64;
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + (size_t)g1 * 16); sb[n] = ss[n][g1]; }
            }
            Q27_LDS_FENCE();
            const float eg = q27_e4m3(cg[i]), eu = q27_e4m3(cu[i]);
            unsigned wgd[4], wud[4]; q27_grp16_pre(vg[i], wgd); q27_grp16_pre(vu[i], wud);
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                accg[n] = fmaf((float)q27_grp16_dot(wgd, xa[n]), eg * sa[n], accg[n]);
                accu[n] = fmaf((float)q27_grp16_dot(wud, xa[n]), eu * sa[n], accu[n]);
            }
            Q27_LDS_FENCE();
            if (i + 1 < UPL2) {
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xa[n] = xb[n]; sa[n] = sb[n]; }
            }
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1)
            #pragma unroll
            for (int n = 0; n < NR; ++n) { accg[n] += __shfl_down(accg[n], o, 64); accu[n] += __shfl_down(accu[n], o, 64); }
        if (lane == 0) {
            #pragma unroll
            for (int n = 0; n < NR; ++n) { YG[(size_t)n * ystride + row] = accg[n] * ag; YU[(size_t)n * ystride + row] = accu[n] * au; }
        }
    }
}

template<int FULL2, int TAIL, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_t2_lds_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs, int rpw)
{
    constexpr int NG = FULL2 * 64 + TAIL, KP = NG * 8, KB = NG * 16;
    __shared__ __align__(16) unsigned char sx[NR][KB];
    __shared__ float ss[NR][NG];
    for (int i = threadIdx.x; i < NR * (KB / 16); i += 256) {
        const int n = i / (KB / 16), o = i - n * (KB / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)Xp)[(size_t)n * (KB / 16) + o];
    }
    for (int i = threadIdx.x; i < NR * NG; i += 256) { const int n = i / NG, o = i - n * NG; ss[n][o] = Xs[(size_t)n * NG + o]; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const bool tailact = (lane < TAIL);
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    for (int r = 0; r < rpw; ++r) {
        const int row = rowbase + r;
        if (row >= rows) break;
        uint2 wv[FULL2 + 1]; unsigned char cg[FULL2 + 1];
        {
            const uint2* wr = (const uint2*)(W + (size_t)row * KP);  const unsigned char* sr = S + (size_t)row * NG;
            #pragma unroll
            for (int i = 0; i < FULL2 + 1; ++i) { const int g = i * 64 + ((i < FULL2) ? lane : (tailact ? lane : 0)); wv[i] = q27_ld8nt(&wr[g]); cg[i] = sr[g]; }
        }
        float acc[NR];
        uint4 xa[NR]; float sa[NR];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { acc[n] = 0.f; xa[n] = *(const uint4*)(sx[n] + (size_t)lane * 16); sa[n] = ss[n][lane]; }
        #pragma unroll
        for (int i = 0; i < FULL2 + 1; ++i) {
            uint4 xb[NR]; float sb[NR];
            if (i + 1 < FULL2 + 1) {
                const int g1 = (i + 1) * 64 + (((i + 1) < FULL2) ? lane : (tailact ? lane : 0));
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + (size_t)g1 * 16); sb[n] = ss[n][g1]; }
            }
            Q27_LDS_FENCE();
            if (i < FULL2 || tailact) {
                const float ec = q27_e4m3(cg[i]);
                unsigned wd[4]; q27_grp16_pre(wv[i], wd);
                #pragma unroll
                for (int n = 0; n < NR; ++n) acc[n] = fmaf((float)q27_grp16_dot(wd, xa[n]), ec * sa[n] * 0.5f, acc[n]);
            }
            Q27_LDS_FENCE();
            if (i + 1 < FULL2 + 1) {
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xa[n] = xb[n]; sa[n] = sb[n]; }
            }
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1)
            #pragma unroll
            for (int n = 0; n < NR; ++n) acc[n] += __shfl_down(acc[n], o, 64);
        if (lane == 0) {
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                float v = acc[n] * alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)n * rstride + row]);
                Y[(size_t)n * ystride + row] = v;
            }
        }
    }
}



// =============================================================================================
// Q27_SCHED_FENCE: the plain asm memory clobber (Q27_LDS_FENCE) did NOT stop the machine scheduler from
// hoisting every group's LDS reads and nibble decodes above the first dot (ISA: 40 ds_read_b128 and
// 120 v_perm before the first v_dot4 -> 256 VGPRs + 92 spills at NR=8). sched_barrier(0) is a hard
// scheduling fence: nothing crosses it in either direction.
#ifndef Q27_SCHED_FENCE
#define Q27_SCHED_FENCE() do { __builtin_amdgcn_sched_barrier(0); asm volatile("" ::: "memory"); } while (0)
// Q27_PIN_ROWS: sched_barrier pinned the LOADS in program order but LLVM had already sunk every group's
// arithmetic below the last group's loads at the IR level (first v_dot4 after all 40 ds_reads). An asm
// statement that "modifies" each accumulator forces group i's dots to complete here, and its memory
// clobber keeps group i+2's loads below it: the live set is two groups, not five.
#define Q27_PIN_ROWS(A) do { _Pragma("unroll") for (int n_ = 0; n_ < NR; ++n_) asm volatile("" : "+v"(A[n_]) : : "memory"); } while (0)
#endif
// NATIVE 8-ROW FORMS (Q27_NR_CH=8, K=7 verify). The register forms stop at NR=4 (VGPRs) and the
// 4-row chunking of an 8-row verify streams every weight twice (layers 71.7 ms at NR=8 vs 37.2 at
// NR=4). Here a 256-thread block stages up to 8 activation rows (40-61 KB) in LDS once, so one
// block per CU (LDS-bound) = one wave per SIMD, and the wave is SOFTWARE-PIPELINED so that one
// wave keeps the VALU busy: the next row's weights are in flight while the current row computes
// (weights are the only HBM traffic), and the next activation group is read from LDS while the
// current group's dots run. Rows per wave = ceil(rows / (4 x CUs)). The first version of these
// kernels (v1: 512-thread blocks, no prefetch, rpw 9-16) measured 138.8 ms for the NR=8 layer
// pass; v2 added an LDS double buffer on top of the weight prefetch and the gate/up form spilled
// 92 VGPRs at NR=8 (256 cap), so v3 keeps only the next-row weight prefetch (one group live).
// pass (vs 71.7 chunked): every row paid its HBM latency and every group its LDS latency in
// the open, with 2 waves per SIMD to hide it. Weights are decoded ONCE per row and applied to all
// NR rows; per-(row, n) arithmetic and accumulation order are those of the register forms.
#define Q27_NR_CASES8(LAUNCH) \
    switch (NR) { case 2: LAUNCH(2); break; case 3: LAUNCH(3); break; case 4: LAUNCH(4); break; \
                  case 5: LAUNCH(5); break; case 6: LAUNCH(6); break; \
                  case 7: LAUNCH(7); break; case 8: LAUNCH(8); break; default: return 0; }
// Q27_NR_LDS_MIN: rows from which the pipelined LDS-staged forms replace the register forms (default 5 = NR>4 only)
static int q27_nr_lds_min() { static int v = 0; if (!v) { const char* e = getenv("Q27_NR_LDS_MIN"); v = e ? atoi(e) : 5; if (v < 2) v = 2; if (v > 9) v = 9; } return v; }
static int q27_lds8_ncu() {
    static int ncu = 0;
    if (!ncu) { int dev = 0, v = 0; hipGetDevice(&dev); hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev); ncu = v > 0 ? v : 60; }
    return ncu;
}
static inline int q27_nr_rpw8(int rows) { const int w = 4 * q27_lds8_ncu(); int r = (rows + w - 1) / w; return r < 1 ? 1 : r; }

// v6: TWO weight rows per iteration for the single-matrix forms (lm_head K=5120, down K=4352). With one
// wave per SIMD the one-row-ahead prefetch only hides HBM latency when a row's arithmetic is longer than
// the latency; gate/up (two matrices, 5 groups) is, lm_head/down (one matrix) is not: profiled 127 us per
// down call (22 MB) and 115 us per fp8 K=1536 call (2 MB) -> latency-bound. A row pair doubles the
// arithmetic per prefetch and shares every activation read.
template<int UPL2, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_lds8_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride, int rpw, int xst)
{
    constexpr int NG = UPL2 * 64, KP = NG * 8, KB = NG * 16;
    __shared__ __align__(16) unsigned char sx[NR][KB];
    __shared__ float ss[NR][NG];
    for (int i = threadIdx.x; i < NR * (KB / 16); i += 256) {
        const int n = i / (KB / 16), o = i - n * (KB / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)(Xp + (size_t)n * (xst >> 2)))[o];
    }
    for (int i = threadIdx.x; i < NR * NG; i += 256) { const int n = i / NG, o = i - n * NG; ss[n][o] = Xs[(size_t)n * (xst >> 4) + o]; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    const int rend = (rowbase + rpw < rows) ? rowbase + rpw : rows;
    uint2 wv[2][UPL2]; unsigned char cg[2][UPL2];
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int rr = (rowbase + p < rows) ? rowbase + p : rows - 1;
        const uint2* wr = (const uint2*)(W + (size_t)rr * KP);  const unsigned char* sr = S + (size_t)rr * NG;
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) { const int g = lane + i * 64; wv[p][i] = q27_ld8nt(&wr[g]); cg[p][i] = sr[g]; }
    }
    for (int row = rowbase; row < rend; row += 2) {
        uint2 wn[2][UPL2]; unsigned char cn[2][UPL2];
        if (row + 2 < rend) {
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int rr = (row + 2 + p < rows) ? row + 2 + p : rows - 1;
                const uint2* wr = (const uint2*)(W + (size_t)rr * KP);  const unsigned char* sr = S + (size_t)rr * NG;
                #pragma unroll
                for (int i = 0; i < UPL2; ++i) { const int g = lane + i * 64; wn[p][i] = q27_ld8nt(&wr[g]); cn[p][i] = sr[g]; }
            }
        }
        float acc0[NR], acc1[NR];
        uint4 xa[NR]; float sa[NR];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { acc0[n] = 0.f; acc1[n] = 0.f; xa[n] = *(const uint4*)(sx[n] + (size_t)lane * 16); sa[n] = ss[n][lane]; }
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            uint4 xb[NR]; float sb[NR];
            if (i + 1 < UPL2) {
                const int g1 = lane + (i + 1) * 64;
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + (size_t)g1 * 16); sb[n] = ss[n][g1]; }
            }
            const float gs0 = q27_e4m3_sel(cg[0][i]), gs1 = q27_e4m3_sel(cg[1][i]);
            unsigned wd0[4], wd1[4]; q27_grp16_pre(wv[0][i], wd0); q27_grp16_pre(wv[1][i], wd1);
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                acc0[n] = fmaf((float)q27_grp16_dot(wd0, xa[n]), gs0 * sa[n] * 0.5f, acc0[n]);
                acc1[n] = fmaf((float)q27_grp16_dot(wd1, xa[n]), gs1 * sa[n] * 0.5f, acc1[n]);
            }
            Q27_PIN_ROWS(acc0); Q27_PIN_ROWS(acc1);
            if (i + 1 < UPL2) {
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xa[n] = xb[n]; sa[n] = sb[n]; }
            }
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1)
            #pragma unroll
            for (int n = 0; n < NR; ++n) { acc0[n] += __shfl_down(acc0[n], o, 64); acc1[n] += __shfl_down(acc1[n], o, 64); }
        if (lane == 0) {
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                Y[(size_t)n * ystride + row] = acc0[n] * alpha;
                if (row + 1 < rend) Y[(size_t)n * ystride + row + 1] = acc1[n] * alpha;
            }
        }
        if (row + 2 < rend) {
            #pragma unroll
            for (int p = 0; p < 2; ++p)
                #pragma unroll
                for (int i = 0; i < UPL2; ++i) { wv[p][i] = wn[p][i]; cg[p][i] = cn[p][i]; }
        }
    }
}

template<int UPL2, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_gu_lds8_nr(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU, int ystride, int rpw, int xst)
{
    constexpr int NG = UPL2 * 64, KP = NG * 8, KB = NG * 16;
    __shared__ __align__(16) unsigned char sx[NR][KB];
    __shared__ float ss[NR][NG];
    for (int i = threadIdx.x; i < NR * (KB / 16); i += 256) {
        const int n = i / (KB / 16), o = i - n * (KB / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)(Xp + (size_t)n * (xst >> 2)))[o];
    }
    for (int i = threadIdx.x; i < NR * NG; i += 256) { const int n = i / NG, o = i - n * NG; ss[n][o] = Xs[(size_t)n * (xst >> 4) + o] * 0.5f; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    const int rend = (rowbase + rpw < rows) ? rowbase + rpw : rows;
    uint2 vg[UPL2], vu[UPL2]; unsigned char cg[UPL2], cu[UPL2];
    {
        const uint2* wg = (const uint2*)(WG + (size_t)rowbase * KP); const uint2* wu = (const uint2*)(WU + (size_t)rowbase * KP);
        const unsigned char* sg = SG + (size_t)rowbase * NG; const unsigned char* su = SU + (size_t)rowbase * NG;
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) { const int g = lane + i * 64; vg[i] = wg[g]; vu[i] = wu[g]; cg[i] = sg[g]; cu[i] = su[g]; }
    }
    for (int row = rowbase; row < rend; ++row) {
        uint2 ng_[UPL2], nu_[UPL2]; unsigned char ncg[UPL2], ncu[UPL2];
        if (row + 1 < rend) {
            const uint2* wg = (const uint2*)(WG + (size_t)(row + 1) * KP); const uint2* wu = (const uint2*)(WU + (size_t)(row + 1) * KP);
            const unsigned char* sg = SG + (size_t)(row + 1) * NG; const unsigned char* su = SU + (size_t)(row + 1) * NG;
            #pragma unroll
            for (int i = 0; i < UPL2; ++i) { const int g = lane + i * 64; ng_[i] = wg[g]; nu_[i] = wu[g]; ncg[i] = sg[g]; ncu[i] = su[g]; }
        }
        float accg[NR], accu[NR];
        uint4 xa[NR]; float sa[NR];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { accg[n] = 0.f; accu[n] = 0.f; xa[n] = *(const uint4*)(sx[n] + (size_t)lane * 16); sa[n] = ss[n][lane]; }
        #pragma unroll
        for (int i = 0; i < UPL2; ++i) {
            uint4 xb[NR]; float sb[NR];
            if (i + 1 < UPL2) {   // next group in flight while this group computes
                const int g1 = lane + (i + 1) * 64;
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + (size_t)g1 * 16); sb[n] = ss[n][g1]; }
            }
            const float eg = q27_e4m3_sel(cg[i]), eu = q27_e4m3_sel(cu[i]);
            unsigned wgd[4], wud[4]; q27_grp16_pre(vg[i], wgd); q27_grp16_pre(vu[i], wud);
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                accg[n] = fmaf((float)q27_grp16_dot(wgd, xa[n]), eg * sa[n], accg[n]);
                accu[n] = fmaf((float)q27_grp16_dot(wud, xa[n]), eu * sa[n], accu[n]);
            }
            Q27_PIN_ROWS(accg); Q27_PIN_ROWS(accu);
            if (i + 1 < UPL2) {
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xa[n] = xb[n]; sa[n] = sb[n]; }
            }
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1)
            #pragma unroll
            for (int n = 0; n < NR; ++n) { accg[n] += __shfl_down(accg[n], o, 64); accu[n] += __shfl_down(accu[n], o, 64); }
        if (lane == 0) {
            #pragma unroll
            for (int n = 0; n < NR; ++n) { YG[(size_t)n * ystride + row] = accg[n] * ag; YU[(size_t)n * ystride + row] = accu[n] * au; }
        }
        if (row + 1 < rend) {
            #pragma unroll
            for (int i = 0; i < UPL2; ++i) { vg[i] = ng_[i]; vu[i] = nu_[i]; cg[i] = ncg[i]; cu[i] = ncu[i]; }
        }
    }
}

template<int FULL2, int TAIL, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_t2_lds8_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs, int rpw, int xst)
{
    constexpr int NG = FULL2 * 64 + TAIL, KP = NG * 8, KB = NG * 16;
    __shared__ __align__(16) unsigned char sx[NR][KB];
    __shared__ float ss[NR][NG];
    for (int i = threadIdx.x; i < NR * (KB / 16); i += 256) {
        const int n = i / (KB / 16), o = i - n * (KB / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)(Xp + (size_t)n * (xst >> 2)))[o];
    }
    for (int i = threadIdx.x; i < NR * NG; i += 256) { const int n = i / NG, o = i - n * NG; ss[n][o] = Xs[(size_t)n * (xst >> 4) + o]; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const bool tailact = (lane < TAIL);
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    const int rend = (rowbase + rpw < rows) ? rowbase + rpw : rows;
    uint2 wv[2][FULL2 + 1]; unsigned char cg[2][FULL2 + 1];
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int rr = (rowbase + p < rows) ? rowbase + p : rows - 1;
        const uint2* wr = (const uint2*)(W + (size_t)rr * KP);  const unsigned char* sr = S + (size_t)rr * NG;
        #pragma unroll
        for (int i = 0; i < FULL2 + 1; ++i) { const int g = i * 64 + ((i < FULL2) ? lane : (tailact ? lane : 0)); wv[p][i] = q27_ld8nt(&wr[g]); cg[p][i] = sr[g]; }
    }
    for (int row = rowbase; row < rend; row += 2) {
        uint2 wn[2][FULL2 + 1]; unsigned char cn[2][FULL2 + 1];
        if (row + 2 < rend) {
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int rr = (row + 2 + p < rows) ? row + 2 + p : rows - 1;
                const uint2* wr = (const uint2*)(W + (size_t)rr * KP);  const unsigned char* sr = S + (size_t)rr * NG;
                #pragma unroll
                for (int i = 0; i < FULL2 + 1; ++i) { const int g = i * 64 + ((i < FULL2) ? lane : (tailact ? lane : 0)); wn[p][i] = q27_ld8nt(&wr[g]); cn[p][i] = sr[g]; }
            }
        }
        // residual share for the epilogue, issued NOW so its latency overlaps the arithmetic: read in the
        // epilogue it was 16 dependent loads each behind s_waitcnt vmcnt(0) (ISA), ~0.7 us apiece = the
        // whole 127 us of a down call
        float ra0[NR], ra1[NR];
        if (radd) {
            #pragma unroll
            for (int n = 0; n < NR; ++n) { ra0[n] = q27_bf2f(radd[(size_t)n * rstride + row]); ra1[n] = (row + 1 < rend) ? q27_bf2f(radd[(size_t)n * rstride + row + 1]) : 0.f; }
        } else {
            #pragma unroll
            for (int n = 0; n < NR; ++n) { ra0[n] = 0.f; ra1[n] = 0.f; }
        }
        float acc0[NR], acc1[NR];
        uint4 xa[NR]; float sa[NR];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { acc0[n] = 0.f; acc1[n] = 0.f; xa[n] = *(const uint4*)(sx[n] + (size_t)lane * 16); sa[n] = ss[n][lane]; }
        #pragma unroll
        for (int i = 0; i < FULL2 + 1; ++i) {
            uint4 xb[NR]; float sb[NR];
            if (i + 1 < FULL2 + 1) {
                const int g1 = (i + 1) * 64 + (((i + 1) < FULL2) ? lane : (tailact ? lane : 0));
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + (size_t)g1 * 16); sb[n] = ss[n][g1]; }
            }
            if (i < FULL2 || tailact) {
                const float e0 = q27_e4m3_sel(cg[0][i]), e1 = q27_e4m3_sel(cg[1][i]);
                unsigned wd0[4], wd1[4]; q27_grp16_pre(wv[0][i], wd0); q27_grp16_pre(wv[1][i], wd1);
                #pragma unroll
                for (int n = 0; n < NR; ++n) {
                    acc0[n] = fmaf((float)q27_grp16_dot(wd0, xa[n]), e0 * sa[n] * 0.5f, acc0[n]);
                    acc1[n] = fmaf((float)q27_grp16_dot(wd1, xa[n]), e1 * sa[n] * 0.5f, acc1[n]);
                }
            }
            Q27_PIN_ROWS(acc0); Q27_PIN_ROWS(acc1);
            if (i + 1 < FULL2 + 1) {
                #pragma unroll
                for (int n = 0; n < NR; ++n) { xa[n] = xb[n]; sa[n] = sb[n]; }
            }
        }
        #pragma unroll
        for (int o = 32; o; o >>= 1)
            #pragma unroll
            for (int n = 0; n < NR; ++n) { acc0[n] += __shfl_down(acc0[n], o, 64); acc1[n] += __shfl_down(acc1[n], o, 64); }
        if (lane == 0) {
            #pragma unroll
            for (int n = 0; n < NR; ++n) {
                float v = acc0[n] * alpha;
                v += rs * ra0[n];
                Y[(size_t)n * ystride + row] = v;
                if (row + 1 < rend) Y[(size_t)n * ystride + row + 1] = acc1[n] * alpha + rs * ra1[n];
            }
        }
        if (row + 2 < rend) {
            #pragma unroll
            for (int p = 0; p < 2; ++p)
                #pragma unroll
                for (int i = 0; i < FULL2 + 1; ++i) { wv[p][i] = wn[p][i]; cg[p][i] = cn[p][i]; }
        }
    }
}
static int q27_nvfp4_lds8(const q27_nvfp4_t* w, const signed char* xq, int xst, const float* xs, float* y, int ystride, int NR, hipStream_t s)
{
    if (!w || w->K != 5120 || NR < 2 || NR > 8) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int rpw = q27_nr_rpw8(w->rows);
    const dim3 g((unsigned)((w->rows + 4 * rpw - 1) / (4 * rpw))), b(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_lds8_nr<5,N>), g, b, 0, s, w->w, w->gs, \
                     (const unsigned*)xq, xs, alpha, w->rows, y, ystride, rpw, xst)
    Q27_NR_CASES8(Q27_L)
#undef Q27_L
    return 1;
}
static int q27_nvfp4_gu_lds8(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, int xst, const float* xs, float* yg, float* yu, int ystride, int NR, hipStream_t s)
{
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || NR < 2 || NR > 8) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const int rpw = q27_nr_rpw8(wg->rows);
    const dim3 g((unsigned)((wg->rows + 4 * rpw - 1) / (4 * rpw))), b(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_gu_lds8_nr<5,N>), g, b, 0, s, wg->w, wg->gs, wu->w, wu->gs, \
                     (const unsigned*)xq, xs, ag, au, wg->rows, yg, yu, ystride, rpw, xst)
    Q27_NR_CASES8(Q27_L)
#undef Q27_L
    return 1;
}
static int q27_nvfp4_down_lds8(const q27_nvfp4_t* w, const signed char* xq, int xst, const float* xs, float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s)
{
    if (!w || w->K != 4352 || NR < 2 || NR > 8) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int rpw = q27_nr_rpw8(w->rows);
    const dim3 g((unsigned)((w->rows + 4 * rpw - 1) / (4 * rpw))), b(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_t2_lds8_nr<4,16,N>), g, b, 0, s, w->w, w->gs, \
                     (const unsigned*)xq, xs, alpha, w->rows, y, ystride, radd, rstride, rs, rpw, xst)
    Q27_NR_CASES8(Q27_L)
#undef Q27_L
    return 1;
}

// rows per wave: keep >= ~3 waves per SIMD (240 SIMDs) so the small shards stay parallel
static inline int q27_nr_rpw(int rows) { int r = rows / 720; return r < 1 ? 1 : (r > 4 ? 4 : r); }

extern "C" int q27_proj_nvfp4_lds(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                                  float* y, int ystride, int NR, hipStream_t s)
{
    if (!w || w->K != 5120 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int rpw = q27_nr_rpw(w->rows);
    const dim3 g64((unsigned)((w->rows + 4 * rpw - 1) / (4 * rpw))), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_lds_nr<5,N>), g64, b64, 0, s, w->w, w->gs, \
                     (const unsigned*)xq_perm, xs, alpha, w->rows, y, ystride, rpw)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
extern "C" int q27_proj_nvfp4_gu_lds(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                     const signed char* xq_perm, const float* xs,
                                     float* yg, float* yu, int ystride, int NR, hipStream_t s)
{
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || NR < 1 || NR > 4) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const int rpw = q27_nr_rpw(wg->rows);
    const dim3 g64((unsigned)((wg->rows + 4 * rpw - 1) / (4 * rpw))), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_gu_lds_nr<5,N>), g64, b64, 0, s, wg->w, wg->gs, wu->w, wu->gs, \
                     (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu, ystride, rpw)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
extern "C" int q27_proj_nvfp4_down_lds(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                                       float* y, int ystride, const unsigned short* radd, int rstride,
                                       float rs, int NR, hipStream_t s)
{
    if (!w || w->K != 4352 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int rpw = q27_nr_rpw(w->rows);
    const dim3 g64((unsigned)((w->rows + 4 * rpw - 1) / (4 * rpw))), b64(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_t2_lds_nr<4,16,N>), g64, b64, 0, s, w->w, w->gs, \
                     (const unsigned*)xq_perm, xs, alpha, w->rows, y, ystride, radd, rstride, rs, rpw)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}


// =============================================================================================
// SPLIT-ROW ROW-BATCHED FORMS (Q27_NR_HG). Two waves per weight row, each owning a CONTIGUOUS half
// of the row's groups (NG/2 = 160 for K=5120: lanes take groups base+lane, base+64+lane and, for
// lane < 32, base+128+lane), so every wave-load is still one contiguous 512-B burst. Per lane that
// is 2-3 groups instead of 5: half the weight, activation and scale registers of the 64-lane forms
// (NR=3 fits 4 waves/SIMD where the 64-lane form had 2), twice the waves per matrix. The two
// half-row partials meet through LDS. The fp32 chain (2-3 groups per lane, 64-lane tree, one add)
// differs from the 64-lane kernels' in association -- plain decode under Q27_GEMV_ALT=3 uses the
// SAME forms, keeping spec == plain an exact-token gate. (The first attempt split each group's two
// 4-B words across the waves: 8-B-strided 4-B loads halved the memory pipeline: plain 67 -> 41.)
template<int NG, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_hg_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride)
{
    constexpr int KP = NG * 8, NGH = NG / 2, CH = (NGH + 63) / 64, REM = NGH - 64 * (CH - 1);
    __shared__ float part[4][NR];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row = blockIdx.x * 2 + (wave >> 1), h = wave & 1;
    const int rr = (row < rows) ? row : (rows - 1);
    const uint2*         wr = (const uint2*)(W + (size_t)rr * KP);
    const unsigned char* sr = S + (size_t)rr * NG;
    const int base = h * NGH;
    uint2 wv[CH]; unsigned char cg[CH]; uint4 xv[NR][CH]; float xs[NR][CH];
    #pragma unroll
    for (int i = 0; i < CH; ++i) {
        const int g = base + ((i < CH - 1 || lane < REM) ? (i * 64 + lane) : 0);   // clamped tail
        wv[i] = q27_ld8nt(&wr[g]); cg[i] = sr[g];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { xv[n][i] = *(const uint4*)(Xp + ((size_t)n * NG + g) * 4); xs[n][i] = Xs[(size_t)n * NG + g]; }
    }
    float acc[NR];
    #pragma unroll
    for (int n = 0; n < NR; ++n) acc[n] = 0.f;
    #pragma unroll
    for (int i = 0; i < CH; ++i) {
        if (i == CH - 1 && lane >= REM) break;
        const float gsc = q27_e4m3(cg[i]);
        unsigned wd[4]; q27_grp16_pre(wv[i], wd);
        #pragma unroll
        for (int n = 0; n < NR; ++n) acc[n] = fmaf((float)q27_grp16_dot(wd, xv[n][i]), gsc * xs[n][i] * 0.5f, acc[n]);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1)
        #pragma unroll
        for (int n = 0; n < NR; ++n) acc[n] += __shfl_down(acc[n], o, 64);
    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < NR; ++n) part[wave][n] = acc[n];
    }
    __syncthreads();
    if (threadIdx.x < 2 * NR) {
        const int rb = threadIdx.x / NR, n = threadIdx.x - rb * NR;
        const int ro = blockIdx.x * 2 + rb;
        if (ro < rows) Y[(size_t)n * ystride + ro] = (part[rb * 2][n] + part[rb * 2 + 1][n]) * alpha;
    }
}

template<int NG, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_gu_hg_nr(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float ag, float au, int rows, float* __restrict__ YG, float* __restrict__ YU, int ystride)
{
    constexpr int KP = NG * 8, NGH = NG / 2, CH = (NGH + 63) / 64, REM = NGH - 64 * (CH - 1);
    __shared__ float pg[4][NR], pu[4][NR];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row = blockIdx.x * 2 + (wave >> 1), h = wave & 1;
    const int rr = (row < rows) ? row : (rows - 1);
    const uint2* wg = (const uint2*)(WG + (size_t)rr * KP); const uint2* wu = (const uint2*)(WU + (size_t)rr * KP);
    const unsigned char* sg = SG + (size_t)rr * NG; const unsigned char* su = SU + (size_t)rr * NG;
    const int base = h * NGH;
    uint2 vg[CH], vu[CH]; unsigned char cg[CH], cu[CH]; uint4 xv[NR][CH]; float xs[NR][CH];
    #pragma unroll
    for (int i = 0; i < CH; ++i) {
        const int g = base + ((i < CH - 1 || lane < REM) ? (i * 64 + lane) : 0);
        vg[i] = wg[g]; vu[i] = wu[g]; cg[i] = sg[g]; cu[i] = su[g];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { xv[n][i] = *(const uint4*)(Xp + ((size_t)n * NG + g) * 4); xs[n][i] = Xs[(size_t)n * NG + g] * 0.5f; }
    }
    float accg[NR], accu[NR];
    #pragma unroll
    for (int n = 0; n < NR; ++n) { accg[n] = 0.f; accu[n] = 0.f; }
    #pragma unroll
    for (int i = 0; i < CH; ++i) {
        if (i == CH - 1 && lane >= REM) break;
        const float eg = q27_e4m3(cg[i]), eu = q27_e4m3(cu[i]);
        unsigned wgd[4], wud[4]; q27_grp16_pre(vg[i], wgd); q27_grp16_pre(vu[i], wud);
        #pragma unroll
        for (int n = 0; n < NR; ++n) {
            accg[n] = fmaf((float)q27_grp16_dot(wgd, xv[n][i]), eg * xs[n][i], accg[n]);
            accu[n] = fmaf((float)q27_grp16_dot(wud, xv[n][i]), eu * xs[n][i], accu[n]);
        }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1)
        #pragma unroll
        for (int n = 0; n < NR; ++n) { accg[n] += __shfl_down(accg[n], o, 64); accu[n] += __shfl_down(accu[n], o, 64); }
    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < NR; ++n) { pg[wave][n] = accg[n]; pu[wave][n] = accu[n]; }
    }
    __syncthreads();
    if (threadIdx.x < 2 * NR) {
        const int rb = threadIdx.x / NR, n = threadIdx.x - rb * NR;
        const int ro = blockIdx.x * 2 + rb;
        if (ro < rows) {
            YG[(size_t)n * ystride + ro] = (pg[rb * 2][n] + pg[rb * 2 + 1][n]) * ag;
            YU[(size_t)n * ystride + ro] = (pu[rb * 2][n] + pu[rb * 2 + 1][n]) * au;
        }
    }
}

// K=4352 down (NG=272): the same split, 136 groups per wave (2 full + 8 lanes); residual share folded.
template<int NG, int NR>
__global__ __launch_bounds__(256) void q27_k_nvfp4_t2_hg_nr(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const unsigned* __restrict__ Xp, const float* __restrict__ Xs,
        float alpha, int rows, float* __restrict__ Y, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs)
{
    constexpr int KP = NG * 8, NGH = NG / 2, CH = (NGH + 63) / 64, REM = NGH - 64 * (CH - 1);
    __shared__ float part[4][NR];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row = blockIdx.x * 2 + (wave >> 1), h = wave & 1;
    const int rr = (row < rows) ? row : (rows - 1);
    const uint2*         wr = (const uint2*)(W + (size_t)rr * KP);
    const unsigned char* sr = S + (size_t)rr * NG;
    const int base = h * NGH;
    uint2 wv[CH]; unsigned char cg[CH]; uint4 xv[NR][CH]; float xs[NR][CH];
    #pragma unroll
    for (int i = 0; i < CH; ++i) {
        const int g = base + ((i < CH - 1 || lane < REM) ? (i * 64 + lane) : 0);
        wv[i] = q27_ld8nt(&wr[g]); cg[i] = sr[g];
        #pragma unroll
        for (int n = 0; n < NR; ++n) { xv[n][i] = *(const uint4*)(Xp + ((size_t)n * NG + g) * 4); xs[n][i] = Xs[(size_t)n * NG + g]; }
    }
    float acc[NR];
    #pragma unroll
    for (int n = 0; n < NR; ++n) acc[n] = 0.f;
    #pragma unroll
    for (int i = 0; i < CH; ++i) {
        if (i == CH - 1 && lane >= REM) break;
        const float gsc = q27_e4m3(cg[i]);
        unsigned wd[4]; q27_grp16_pre(wv[i], wd);
        #pragma unroll
        for (int n = 0; n < NR; ++n) acc[n] = fmaf((float)q27_grp16_dot(wd, xv[n][i]), gsc * xs[n][i] * 0.5f, acc[n]);
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1)
        #pragma unroll
        for (int n = 0; n < NR; ++n) acc[n] += __shfl_down(acc[n], o, 64);
    if (lane == 0) {
        #pragma unroll
        for (int n = 0; n < NR; ++n) part[wave][n] = acc[n];
    }
    __syncthreads();
    if (threadIdx.x < 2 * NR) {
        const int rb = threadIdx.x / NR, n = threadIdx.x - rb * NR;
        const int ro = blockIdx.x * 2 + rb;
        if (ro < rows) {
            float v = (part[rb * 2][n] + part[rb * 2 + 1][n]) * alpha;
            if (radd) v += rs * q27_bf2f(radd[(size_t)n * rstride + ro]);
            Y[(size_t)n * ystride + ro] = v;
        }
    }
}

extern "C" int q27_proj_nvfp4_hg(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                                 float* y, int ystride, int NR, hipStream_t s)
{
    if (!w || w->K != 5120 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g2((unsigned)((w->rows + 1) / 2)), b256(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_hg_nr<320,N>), g2, b256, 0, s, w->w, w->gs, \
                     (const unsigned*)xq_perm, xs, alpha, w->rows, y, ystride)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
extern "C" int q27_proj_nvfp4_gu_hg(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                                    const signed char* xq_perm, const float* xs,
                                    float* yg, float* yu, int ystride, int NR, hipStream_t s)
{
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || wg->K != 5120 || NR < 1 || NR > 4) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g2((unsigned)((wg->rows + 1) / 2)), b256(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_gu_hg_nr<320,N>), g2, b256, 0, s, wg->w, wg->gs, wu->w, wu->gs, \
                     (const unsigned*)xq_perm, xs, ag, au, wg->rows, yg, yu, ystride)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
extern "C" int q27_proj_nvfp4_down_hg(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                                      float* y, int ystride, const unsigned short* radd, int rstride,
                                      float rs, int NR, hipStream_t s)
{
    if (!w || w->K != 4352 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g2((unsigned)((w->rows + 1) / 2)), b256(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_nvfp4_t2_hg_nr<272,N>), g2, b256, 0, s, w->w, w->gs, \
                     (const unsigned*)xq_perm, xs, alpha, w->rows, y, ystride, radd, rstride, rs)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
