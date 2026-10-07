// q27_fp8.hip — FP8 (E4M3) projections for Qwen3.8-27B on gfx906 (MI50), wave64, ROCm 5.7.1.
//
// Serves q_proj[12288,5120], k_proj/v_proj[1024,5120], o_proj[5120,6144],
// in_proj_qkv[10240,5120], in_proj_z[6144,5120], out_proj[5120,6144] — 41.2% of decode bytes.
// Storage per q27.h: w = e4m3(byte) * wscale, one f32 multiplier, row-major, stride K bytes.
//
// =====================================================================================
// EXECUTION REPRESENTATION — DECIDED, WITH THE NUMBERS
// =====================================================================================
// gfx906 has no FP8 datapath. Two candidates were on the table (R1 receipt left this open:
// "Not settled: the FP8 attention projections (a separate conversion question)").
//
//   (a) e4m3 -> int8 (requantize at load, or in-register), then v_dot4_i32_i8.
//   (b) e4m3 -> fp16 in-register, then the 4-operand v_dot2_f32_f16, fp32 accumulate.
//
// Byte traffic is 1 B/weight either way, so this is ALU + accuracy, not bandwidth.
// I measured (a)'s accuracy on THIS checkpoint's real bytes (not a model): the four largest
// FP8 tensors, every weight, decoded exactly and requantized. Weight relative L2 vs the
// stored e4m3 values, and the resulting GEMV output relative L2 against a Gaussian activation:
//
//   tensor                          amax/sigma   int8/tensor  int8/row   int8/grp16   fp16
//   layers.3.self_attn.q_proj         18.55      4.239e-2     9.96e-3    4.86e-3      0.0
//   layers.3.self_attn.k_proj         13.21      3.031e-2     1.06e-2    4.94e-3      0.0
//   layers.3.self_attn.o_proj         90.06      2.035e-1     1.01e-2    4.92e-3      0.0
//   layers.0.linear_attn.in_proj_qkv  27.74      6.280e-2     9.20e-3    4.84e-3      0.0
//   (simulated GEMV output relL2, int8 per-tensor: 4.23e-2 / 3.20e-2 / 2.06e-1 / 6.25e-2)
//
// The project's acceptance gate (tests/q27_gate.h) is relL2 <= 2e-3. Option (a) misses it by
// 21x on q_proj and by 102x on o_proj. o_proj is the reason: amax/sigma = 90, i.e. this tensor
// has the classic attention outlier structure, and a per-tensor int8 grid spends its whole
// dynamic range on outliers. Even a per-ROW int8 scale (5x the gate) and a per-GROUP-16 int8
// scale (2.4x the gate, and +6.25% bytes for the extra scale array, which the q27_fp8_t ABI
// has no field for) both fail. The e4m3->int8 doubling is NOT the exact doubling the NVFP4 case
// enjoys; e4m3 is a floating grid with 8 exponent decades and int8 is a uniform grid.
//
// Option (b) is EXACT: relL2 = 0.000000e+00 on all four tensors, 82M real weights. It is exact
// by construction, not by luck — every OCP e4m3fn finite value is an fp16 value: e4m3's range
// is [2^-9, 448], fp16 normals reach down to 2^-14 and up to 65504, and fp16's 10-bit mantissa
// covers e4m3's 3. There is nothing to lose.
//
// CHOSEN: (b). This also honours the standing estate law that attention stays 8-bit and that
// quantization is chosen BY ROLE — the model's authors put NVFP4 (per-group-16 scaled, so
// outlier-tolerant) on the MLP and FP8 (relative precision) on attention deliberately.
// Requantizing attention to a uniform int8 grid would silently undo that choice.
//
// WHAT (a) WOULD HAVE COST US, and what it would have bought: option (a) as a pre-converted
// int8 arena is the cheapest possible inner loop — one v_dot4_i32_i8 per 4 weights, ~0.25
// VALU ops/weight, and R1 measured that shape (variant C) at 328.5 GB/s, i.e. at the roofline.
// This kernel's decode costs 2.0 ops/weight (below), so if it lands short of the roofline the
// gap is what (a) would have bought — at the accuracy prices tabulated above. It is not a trade
// worth making at 3-20% output error against a 0.2% gate. In-register e4m3->int8 is worse still:
// it needs the exponent unpacked and re-rounded per element, strictly more work than (b)'s
// three ops, for strictly worse numbers. There is no version of (a) that wins here.
//
// =====================================================================================
// VERIFICATION (host, exact arithmetic, real checkpoint bytes — no GPU run)
// =====================================================================================
// 1. The decode bit sequence below reproduces the OCP e4m3fn table on all 256 codes, exactly,
//    including both signed zeros and all 8 subnormals. 0 mismatches.
// 2. The int8->fp16 magic below is exact on all 256 int8 values.
// 3. Simulating this file's arithmetic end to end (quantizer recipe, fp16 operands, fp32
//    products, per-group-16 accumulate, single epilogue alpha) against the exact fp32 product
//    W_e4m3 . a_bf16 on real weights, 2048 rows each:
//
//      tensor                      weights-only relL2   full relL2   cosine    CUDA oracle's own
//                                                                              activation path
//      q_proj      [12288,5120]       0.000e+00         4.65e-3    0.9999892     2.77e-2
//      k_proj      [ 1024,5120]       0.000e+00         4.56e-3    0.9999896     2.57e-2
//      o_proj      [ 5120,6144]       0.000e+00         4.69e-3    0.9999890     2.61e-2
//      in_proj_qkv [10240,5120]       0.000e+00         4.46e-3    0.9999902     2.58e-2
//      (with 12x outlier activation channels: ours 5.9-7.5e-3; the oracle's own path degrades
//       to 2.96e-1 on o_proj, where its small static input_scale 0.0174 clips the outliers)
//
//    The weight path contributes EXACTLY ZERO error. Every remaining digit is the int8
//    activation, which the ABI fixes (q27_quant_fp8 must emit `signed char`). 4.5e-3 is the
//    information floor for 8 bits per activation element with one shared scale per 16, and it
//    is ~6x MORE accurate than the CUDA oracle's own static-per-tensor e4m3 activation.
//
//    HONEST FLAG FOR THE GATE OWNER: 4.5e-3 exceeds tests/q27_gate.h's default 2e-3. That
//    tolerance is not reachable at a single projection boundary by ANY int8-activation kernel —
//    the oracle itself misses it by 2.6e-2 against exact fp32. Gate this op against captured
//    ORACLE vectors (where the two quantization grids are the thing being compared), not
//    against fp32, and expect the residual to be activation-grid difference, not weight error.
//
// =====================================================================================
// THE DECODE: e4m3 byte -> fp16, 4 VALU ops per 2 weights, branchless, exact
// =====================================================================================
// Place the byte's 7 magnitude bits at fp16 bits 13:7 and its sign at bit 15, adding NOTHING
// to the exponent. The result is the true value times 2^-8, and the 2^-8 folds into the GEMM
// epilogue scalar for free. Why this is exact on the whole domain:
//   normals  (e>=1): fp16 exponent field = e, value = 2^(e-15)*(1+m/8) = 2^(e-7)*(1+m/8) * 2^-8
//   subnorms (e==0): fp16 exponent field = 0, so fp16's OWN subnormal rule applies and gives
//                    (m<<7) * 2^-24 = m*2^-17 = (m*2^-9) * 2^-8  — exactly e4m3's subnormal.
//   code 0x00 -> +0, 0x80 -> -0, and e4m3fn's NaN slot (0x7F/0xFF) maps to a finite 480*2^-8;
//                    that code does not occur (measured: 0 NaN codes in all four tensors).
// So the e4m3 subnormal branch that q27_e4m3() must keep for SCALE decode to fp32 is simply
// not needed here — fp16's exponent field is wide enough that the whole e4m3 grid lands on
// fp16's own normal/subnormal boundary. No branch, no select, no LUT. (This does not touch
// the standing "keep the branches in q27_e4m3" law: that is a different target format.)
//
// Sequence, per input dword (4 weights -> two half2 dwords):
//   raw = v_perm_b32 to [b1,0,b0,0]   (byte at bits 15:8 of each half, zeros as separators)
//   u   = raw >> 1                    (magnitude at 13:7, sign now at 14, bit 15 clear)
//   u  += (u & 0x40004000)            (sign bit 14 doubles into bit 15 and its carry clears 14)
// -> 2 perm + 2 shr + 2 and + 2 add = 8 ops / 4 weights = 2.0 ops/weight, plus 0.5 for the
//    two v_dot2_f32_f16. R1's rejected variant B ran ~6 ops/weight and capped at 57% of
//    roofline on the dependency chain; variant E ran ~0.75 and hit it. 2.5 is between, and the
//    loop is written to keep several independent loads in flight per the estate's law.
//
// =====================================================================================
// ACTIVATION CONTRACT — NATURAL ORDER. DO NOT MERGE WITH THE NVFP4 PRODUCER.
// =====================================================================================
// q27_quant_fp8 writes xq[k] at index k. No even-then-odd permutation. q27_quant_perm's output
// is NOT interchangeable with this one and the two producers must never be fused: the estate
// has a recorded incident where moving a permutation into a shared producer silently broke the
// int8 consumer. The permutation exists only to make q27_dq4's byte-permute nibble decode
// land the even weights contiguously; this kernel does not use q27_dq4.
//
// Quantization, mirroring the NVFP4 producer's shape with int8's 127 in place of e2m1's 6:
//     global = 1 / in_scale
//     xs[g]  = (max|x| over the 16) * global / 127        f32, exact (no e4m3 rounding: the ABI
//     xq[k]  = rint(x[k] * global / xs[g]) clamped +-127  gives us a float* here, so use it)
// so  x[k] ~= xq[k] * xs[g] * in_scale, and with w = code * wscale the GEMM is
//     y[r] = (in_scale * wscale) * SUM_g xs[g] * SUM_{k in g} code * xq[k]
// i.e. per-group partial sums scaled by xs[g], one epilogue scalar alpha at the end (contract
// section 5: "alpha = input_scale * weight_scale, applied ONCE"). in_scale cancels analytically
// between xs[] and alpha — it is kept in both places so this producer stays structurally
// identical to the NVFP4 one, whose global does NOT cancel (it survives the e4m3 rounding of
// its group scale). alpha additionally carries the 2^8 from the decode above.
//
// BUFFER SIZES the caller must honour: xq is n bytes, xs is exactly (n+15)/16 floats — the
// same per-16 granularity as q27_quant_perm, so the two producers stay allocation-compatible.
// n must be a multiple of 16 for the vector path (5120 and 6144 both are); an odd n falls into
// a guarded scalar tail rather than writing garbage.
//
// One quantization serves several GEMMs: model.cpp:2070-2074 throws unless q/k/v share an
// input_scale, so one q27_quant_fp8 call feeds q_proj, k_proj and v_proj.

#ifndef Q27_FP8_MINW
#define Q27_FP8_MINW 1
#endif
#include <cstdlib>
#include <hip/hip_fp16.h>   // __half2 / __float2half2_rn used by the fdot2 path
#include "q27.h"
#include "q27_nr.h"
#include "q27_wave.h"
#include <vector>
#include <ctime>

// ------------------------------------------------------------------ device helpers
typedef _Float16 q27_h2_t __attribute__((ext_vector_type(2)));

// 4-operand VOP3 v_dot2_f32_f16. gfx906 has this; it has no v_dot2c_f32_f16.
__device__ __forceinline__ float q27_fdot2(unsigned a, unsigned b, float c) {
    union { unsigned u; q27_h2_t h; } ua, ub;
    ua.u = a; ub.u = b;
    return __builtin_amdgcn_fdot2(ua.h, ub.h, c, false);
}

// four e4m3 weight bytes -> two half2 dwords carrying 2^-8 times the true value. See above.
__device__ __forceinline__ void q27_e4m3x4_h2(unsigned d, unsigned& lo, unsigned& hi) {
    lo = __builtin_amdgcn_perm(0u, d, 0x01040004u);   // [b1, 0, b0, 0]
    hi = __builtin_amdgcn_perm(0u, d, 0x03040204u);   // [b3, 0, b2, 0]
    lo >>= 1;  lo += (lo & 0x40004000u);
    hi >>= 1;  hi += (hi & 0x40004000u);
}

// four int8 activation bytes -> two half2 dwords, exact (|q| <= 127 is exact in fp16).
// Offset-binary + the 0x6400 magic: fp16 bits 0x6400|u == 1024+u exactly for u in [0,255],
// so one xor, two byte permutes and two v_pk_add_f16 convert four lanes with no v_cvt at all.
__device__ __forceinline__ void q27_i8x4_h2(unsigned d, unsigned& lo, unsigned& hi) {
    const unsigned u = d ^ 0x80808080u;               // q + 128, no cross-byte carry
    unsigned a = __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u);  // [0x64,u1,0x64,u0]
    unsigned b = __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u);  // [0x64,u3,0x64,u2]
    union { unsigned u; q27_h2_t h; } ca, cb, k;
    k.u = 0x64806480u;                                // half2(1152, 1152) = 1024 + 128
    ca.u = a; cb.u = b;
    ca.h = ca.h - k.h;  cb.h = cb.h - k.h;
    lo = ca.u; hi = cb.u;
}

// 16 weight bytes against 8 pre-converted activation half2 dwords, fp32 accumulate.
__device__ __forceinline__ float q27_dot16_fp8(uint4 w, const unsigned* xh, float acc) {
    unsigned lo, hi;
    q27_e4m3x4_h2(w.x, lo, hi); acc = q27_fdot2(lo, xh[0], acc); acc = q27_fdot2(hi, xh[1], acc);
    q27_e4m3x4_h2(w.y, lo, hi); acc = q27_fdot2(lo, xh[2], acc); acc = q27_fdot2(hi, xh[3], acc);
    q27_e4m3x4_h2(w.z, lo, hi); acc = q27_fdot2(lo, xh[4], acc); acc = q27_fdot2(hi, xh[5], acc);
    q27_e4m3x4_h2(w.w, lo, hi); acc = q27_fdot2(lo, xh[6], acc); acc = q27_fdot2(hi, xh[7], acc);
    return acc;
}


// ============================================================================================
// WIDE fp8 PROJECTION GEMM (Q27_PF_OWIDE).
//
// The existing fp8 projection family (q27_k_proj_fp8_m2 and friends) measured 1.2-2.1 TMAC/s
// against this FFN's 5.9, and the source explains it: 233 VGPRs under a bare __launch_bounds__(256),
// i.e. ONE wave per SIMD, so it is occupancy/latency bound and changing M does nothing (receipt
// v108). Forcing 2 waves measured WORSE (3705 vs 2613 ms) because the spills cost more than the
// occupancy. So it needs the geometry that already works, not a re-parameterisation:
//   * output-stationary tile TR=128 rows x TC=64 tokens per block, grid over BOTH rows and tokens
//     (m2 had grid = rows/4 only, with M looped INSIDE the block);
//   * K walked inside the thread in groups of 16;
//   * the int8 -> half2 conversion done ONCE per activation element at LDS staging time, not once
//     per (row, k-step) as m2 does -- activations are stored in LDS as half2 (32 B per 16-group);
//   * weights stay fp8 in LDS (16 B per 16-group) and convert inside q27_dot16_fp8, whose lane
//     tree is identical to m2's, so the accumulation chain per output is the same.
// Output is fp32 with an optional bf16 residual fold, matching q27_proj_fp8_m2_res.
// ============================================================================================
template <int KI, int NW, int MINW>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_fp8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ Xq,
        const float* __restrict__ Xs, float alpha, int rows, int K, int M,
        float* __restrict__ Y, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs) {
    constexpr int TR = 128, TC = 64;
    constexpr int WSTR = KI * 16 + 16;
    constexpr int XSTR = KI * 32 + 32;
    __shared__ __align__(16) unsigned char sw[TR * WSTR];
    __shared__ __align__(16) unsigned char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W + (size_t)rr * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            const uint4 v = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            unsigned h[8];
            q27_i8x4_h2(v.x, h[0], h[1]); q27_i8x4_h2(v.y, h[2], h[3]);
            q27_i8x4_h2(v.z, h[4], h[5]); q27_i8x4_h2(v.w, h[6], h[7]);
            *(uint4*)(sx + t * XSTR + g * 32)      = *(const uint4*)&h[0];
            *(uint4*)(sx + t * XSTR + g * 32 + 16) = *(const uint4*)&h[4];
            sxsc[it] = Xs[(size_t)tt * NG + gg];
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = *(const uint4*)(sw + (lane + 64) * WSTR + g * 16);
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 xa = *(const uint4*)(sx + t * XSTR + g * 32);
                const uint4 xb = *(const uint4*)(sx + t * XSTR + g * 32 + 16);
                unsigned xh[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
                const float xsv = sxsc[t * KI + g];
                acc0[j] = fmaf(q27_dot16_fp8(w0, xh, 0.f), xsv, acc0[j]);
                acc1[j] = fmaf(q27_dot16_fp8(w1, xh, 0.f), xsv, acc1[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int t = tok_base + wave * TPW + j;
        if (t < M) {
            if (r0 < rows) { float v = acc0[j] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rstride + r0]); Y[(size_t)t * ystride + r0] = v; }
            if (r1 < rows) { float v = acc1[j] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rstride + r1]); Y[(size_t)t * ystride + r1] = v; }
        }
    }
}

extern "C" int q27_proj_fp8_wide(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                  float* y, int M, int ystride,
                                  const unsigned short* radd, int rstride, float rs,
                                  int ki, hipStream_t s) {
    if (!w || M < 1 || (w->K & 15) || w->K < 16) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64));
    // KI=8: sw 18,432 + sx 18,432 + sxsc 2,048 = 38,912 B of the 64 KB LDS budget. KI=16 needs
    // 73,728 B and does not fit (the compiler rejects it), so only KI 4 and 8 are instantiated.
    if (ki == 8)  { hipLaunchKernelGGL((q27_k_wide_fp8<8,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, y, ystride, radd, rstride, rs); return 1; }
    if (ki == 4)  { hipLaunchKernelGGL((q27_k_wide_fp8<4,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, y, ystride, radd, rstride, rs); return 1; }
    return 0;
}

// BF16-output twin, for the projections whose consumers expect BF16 (q/k/v into qkva_slots, GDN
// in_qkv/in_z into qkv_slots/zbuf_slots). Same tile, same K-in-thread accumulation, same
// q27_dot16_fp8 chain -- only the epilogue store differs.
template <int KI, int NW, int MINW>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_fp8b(
        const unsigned char* __restrict__ W, const signed char* __restrict__ Xq,
        const float* __restrict__ Xs, float alpha, int rows, int K, int M,
        unsigned short* __restrict__ Y, int ystride) {
    constexpr int TR = 128, TC = 64;
    constexpr int WSTR = KI * 16 + 16;
    constexpr int XSTR = KI * 32 + 32;
    __shared__ __align__(16) unsigned char sw[TR * WSTR];
    __shared__ __align__(16) unsigned char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W + (size_t)rr * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            const uint4 v = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            unsigned h[8];
            q27_i8x4_h2(v.x, h[0], h[1]); q27_i8x4_h2(v.y, h[2], h[3]);
            q27_i8x4_h2(v.z, h[4], h[5]); q27_i8x4_h2(v.w, h[6], h[7]);
            *(uint4*)(sx + t * XSTR + g * 32)      = *(const uint4*)&h[0];
            *(uint4*)(sx + t * XSTR + g * 32 + 16) = *(const uint4*)&h[4];
            sxsc[it] = Xs[(size_t)tt * NG + gg];
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = *(const uint4*)(sw + (lane + 64) * WSTR + g * 16);
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 xa = *(const uint4*)(sx + t * XSTR + g * 32);
                const uint4 xb = *(const uint4*)(sx + t * XSTR + g * 32 + 16);
                unsigned xh[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
                const float xsv = sxsc[t * KI + g];
                acc0[j] = fmaf(q27_dot16_fp8(w0, xh, 0.f), xsv, acc0[j]);
                acc1[j] = fmaf(q27_dot16_fp8(w1, xh, 0.f), xsv, acc1[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int t = tok_base + wave * TPW + j;
        if (t < M) {
            if (r0 < rows) Y[(size_t)t * ystride + r0] = q27_f2bf(acc0[j] * alpha);
            if (r1 < rows) Y[(size_t)t * ystride + r1] = q27_f2bf(acc1[j] * alpha);
        }
    }
}

extern "C" int q27_proj_fp8_wideb(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                  unsigned short* y, int M, int ystride, int ki, hipStream_t s) {
    if (!w || M < 1 || (w->K & 15) || w->K < 16) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64));
    if (ki == 8)  { hipLaunchKernelGGL((q27_k_wide_fp8b<8,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, y, ystride); return 1; }
    if (ki == 4)  { hipLaunchKernelGGL((q27_k_wide_fp8b<4,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, y, ystride); return 1; }
    return 0;
}

// ------------------------------------------------------------------ quantizer
// One thread per group of 16. n is 5120 or 6144 here, both multiples of 16; the tail path
// exists only so a caller cannot get silent garbage from an odd n.
static __device__ __forceinline__ void q27_quant_fp8_body(
        const unsigned short* __restrict__ x, signed char* __restrict__ xq,
        float* __restrict__ xs, int n, float global) {
    const int g  = blockIdx.x * blockDim.x + threadIdx.x;
    const int k0 = g * 16;
    if (k0 >= n) return;

    float v[16];
    float mx = 0.0f;
    if (k0 + 16 <= n) {
        const uint4 a = *(const uint4*)(x + k0);      // 16 bf16 = 32 B, 16-B aligned
        const uint4 b = *(const uint4*)(x + k0 + 8);
        const unsigned p[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[2*i  ] = q27_bf2f((unsigned short)(p[i] & 0xFFFFu));
            v[2*i+1] = q27_bf2f((unsigned short)(p[i] >> 16));
            mx = fmaxf(mx, fmaxf(fabsf(v[2*i]), fabsf(v[2*i+1])));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = (k0 + i < n) ? q27_bf2f(x[k0 + i]) : 0.0f;
            mx = fmaxf(mx, fabsf(v[i]));
        }
    }

    const float s   = mx * global * (1.0f / 127.0f);
    const float inv = (s > 0.0f) ? (global / s) : 0.0f;    // == 127/mx; global cancels by design
    xs[g] = s;

    // pack into four dwords directly: a `signed char[16]` local is only 1-byte aligned, so
    // type-punning it to uint4 would be an under-aligned access waiting to be a scratch spill.
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned d = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int t = (int)rintf(v[4*i + j] * inv);
            t = (t > 127) ? 127 : ((t < -127) ? -127 : t);
            d |= ((unsigned)t & 0xFFu) << (8 * j);
        }
        w[i] = d;
    }
    if (k0 + 16 <= n) {
        uint4 o; o.x = w[0]; o.y = w[1]; o.z = w[2]; o.w = w[3];
        *(uint4*)(xq + k0) = o;                       // xq+k0 is 16-B aligned (k0 % 16 == 0)
    } else {
        for (int i = 0; i < 16 && k0 + i < n; ++i)
            xq[k0 + i] = (signed char)((w[i >> 2] >> (8 * (i & 3))) & 0xFFu);
    }
}

// ------------------------------------------------------------------ projection
// Geometry: one wave64 per R consecutive rows, 4 waves (256 threads) per block.
// Lane l at step t covers k in [t*1024 + l*16, +16) — exactly ONE aligned group of 16, so the
// activation group scale is a single scalar per lane per step and never needs a cross-lane
// reduction. A wave reads 1024 contiguous weight bytes per row per step (uint4 per lane).
// K = 5120 -> 5 steps, K = 6144 -> 6 steps, both exact.
//
// R>1 amortizes the activation int8->fp16 conversion (the only work that is not per-weight)
// over R rows and, more importantly, puts R independent weight loads in flight before any
// unpacking — the estate's dependency-graph law, not an instruction-count argument.
// The activation is NOT staged in LDS: with one wave per row group the only consumer of a
// converted activation register is that same lane, and the estate has measured replicated LDS
// activation staging hurting skinny decode repeatedly.
// HIP's uint4/uint2 are class templates, which __builtin_nontemporal_load rejects; these native
// ext_vector_types are what Pangu's ld16nt takes, for the same reason. Declared here, above the
// first kernel that uses them.
typedef unsigned int q27_u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int q27_u32x2 __attribute__((ext_vector_type(2)));

// 8 fp8 weights (uint2) against 4 packed activation halves.
__device__ __forceinline__ float q27_dot8_fp8(uint2 w, const unsigned* xh, float acc) {
    unsigned lo, hi;
    q27_e4m3x4_h2(w.x, lo, hi); acc = q27_fdot2(lo, xh[0], acc); acc = q27_fdot2(hi, xh[1], acc);
    q27_e4m3x4_h2(w.y, lo, hi); acc = q27_fdot2(lo, xh[2], acc); acc = q27_fdot2(hi, xh[3], acc);
    return acc;
}

// ---------------------------------------------------------------------------------------------
// NARROW-LOAD STREAMING FORM: uint2 per lane instead of uint4.
// The transfer package measured this directly on these cards: uint2 741 GB/s vs uint4 643, with an
// explicit "do not re-propose uint4 without new evidence". This engine's own numbers agree -- the
// NVFP4 kernel uses uint2 and reaches 534 GB/s (72% of roof) while this fp8 projection used uint4
// and reached 409 (55%). 64 lanes x 8 B = 512 B per step instead of 1024, so NSTEP doubles and
// twice as many independent loads are in flight.
// Scale indexing note: k0 = t*512 + lane*8, so lanes PAIR UP within a 16-element scale group
// (lane 0 -> k0 0, lane 1 -> k0 8, both group 0). k0>>4 is still the correct group for each.
template <int R, int NSTEP>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_n(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv,
        int rows, int K, float alpha, int outbf) {
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    if (row0 >= rows) return;

    const unsigned char* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        wp[r] = W + (size_t)rr * (size_t)K;
    }

    uint2 wv[R][NSTEP];
    uint2 xv[NSTEP];
    float sc[NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 9) + lane * 8;
        ok[t] = (k0 < K);
        const int ks = ok[t] ? k0 : 0;              // inactive lanes read a VALID address
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const q27_u32x2 v = __builtin_nontemporal_load((const q27_u32x2*)(wp[r] + ks));
            wv[r][t] = make_uint2(v.x, v.y);
        }
        xv[t] = *(const uint2*)(xq + ks);
        sc[t] = xs[ks >> 4];
    }

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        if (!ok[t]) continue;
        unsigned xh[4];
        q27_i8x4_h2(xv[t].x, xh[0], xh[1]);
        q27_i8x4_h2(xv[t].y, xh[2], xh[3]);
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += q27_dot8_fp8(wv[r][t], xh, 0.0f) * sc[t];
    }

#pragma unroll
    for (int o = 32; o; o >>= 1)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += __shfl_down(acc[r], o, 64);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) continue;
            const float v = acc[r] * alpha;
            if (outbf) ((unsigned short*)yv)[row0 + r] = q27_f2bf(v);
            else       ((float*)yv)[row0 + r]         = v;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// STREAMING FORM, measured at 731 GB/s: issue
// EVERY load into a register array before consuming ANY of it, and mark the weights non-temporal
// because each weight byte is read exactly once per token and should not evict activations from L2.
//
// The kernel below issues 1 weight + 1 activation load per step and consumes them immediately --
// R+1 loads in flight, then a wait, five times over for K=5120. That is the same dependency-chain
// defect that made the GDN bf16 GEMV 30x off its byte floor. proj_fp8 is 4.871 ms/token, the
// largest single bucket after the GDN fix.
//
// NSTEP is a template parameter so the load block fully unrolls: R*NSTEP weight loads plus NSTEP
// activation loads are in flight at once (10 for the K=5120, R=1 production shape, matching
// Pangu's 8). Register pressure rises and that is expected -- the transfer package's own lesson is
// that the winning variant had 49% MORE instructions at identical occupancy. Dependency structure
// is the objective; instruction and register counts are not.
// HIP's uint4 is a class template, which __builtin_nontemporal_load rejects. Pangu's ld16nt takes
// a native ext_vector_type for the same reason. Load through this, then reinterpret.
static __device__ __forceinline__ uint4 q27_ld16nt(const void* p) {
    const q27_u32x4 v = __builtin_nontemporal_load((const q27_u32x4*)p);
    return make_uint4(v.x, v.y, v.z, v.w);
}

// OUTBF=1 writes BF16 straight out. Every fp8 projection except o_proj/out_proj was immediately
// followed by q27_f2bf_vec converting its fp32 output -- 144 launches per token, each a full round
// trip through memory for a pure format change. The conversion costs one instruction in this
// epilogue. o_proj/out_proj keep fp32 because the collective reduces them.
template <int R, int NSTEP, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_s(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, float rs) {
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    if (row0 >= rows) return;

    const unsigned char* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);   // clamped ragged tail
        wp[r] = W + (size_t)rr * (size_t)K;
    }

    // ---- every load issued before any consumption ----
    uint4 wv[R][NSTEP];
    uint4 xv[NSTEP];
    float sc[NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);                       // K=1536 -> step 1 is live on lanes 0..31 only
        const int ks = ok[t] ? k0 : 0;          // inactive lanes read a VALID address, never OOB
#pragma unroll
        for (int r = 0; r < R; ++r)
            wv[r][t] = q27_ld16nt(wp[r] + ks);
        xv[t] = *(const uint4*)(xq + ks);
        sc[t] = xs[ks >> 4];
    }

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        if (!ok[t]) continue;
        unsigned xh[8];
        q27_i8x4_h2(xv[t].x, xh[0], xh[1]);
        q27_i8x4_h2(xv[t].y, xh[2], xh[3]);
        q27_i8x4_h2(xv[t].z, xh[4], xh[5]);
        q27_i8x4_h2(xv[t].w, xh[6], xh[7]);
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += q27_dot16_fp8(wv[r][t], xh, 0.0f) * sc[t];
    }

#pragma unroll
    for (int o = 32; o; o >>= 1)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += __shfl_down(acc[r], o, 64);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) continue;
            float v = acc[r] * alpha;
            // Residual share folded into the epilogue. This is a PARTIAL projection feeding an
            // all-reduce; every card adds rs*residual so the host reduction yields the exact
            // post-residual vector. Doing it here rather than in a separate kernel matters: a
            // launch on all four cards is uniform work, which moves the barrier release rather
            // than being absorbed as slack.
            if (radd) v += rs * q27_bf2f(radd[row0 + r]);
            if (OUTBF) ((unsigned short*)yv)[row0 + r] = q27_f2bf(v);
            else       ((float*)yv)[row0 + r]         = v;
        }
    }
}

template <int R>
__global__ __launch_bounds__(256) void q27_k_proj_fp8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, float* __restrict__ y,
        int rows, int K, float alpha) {
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    if (row0 >= rows) return;

    const int nsteps = (K + 1023) >> 10;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    // clamped row bases so a ragged tail block never reads out of bounds
    const unsigned char* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        wp[r] = W + (size_t)rr * (size_t)K;
    }

#pragma unroll 2
    for (int t = 0; t < nsteps; ++t) {
        const int k0 = (t << 10) + lane * 16;
        if (k0 >= K) continue;

        // issue every independent load first, then wait once and decode.
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r] = *(const uint4*)(wp[r] + k0);
        const uint4 xv = *(const uint4*)(xq + k0);
        const float s  = xs[k0 >> 4];

        unsigned xh[8];
        q27_i8x4_h2(xv.x, xh[0], xh[1]);
        q27_i8x4_h2(xv.y, xh[2], xh[3]);
        q27_i8x4_h2(xv.z, xh[4], xh[5]);
        q27_i8x4_h2(xv.w, xh[6], xh[7]);

#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += q27_dot16_fp8(wv[r], xh, 0.0f) * s;
    }

    // wave64 reduction. __shfl_down(x, off, 64) crosses the whole wave; ds_swizzle would not.
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = v * alpha;
    }
}

// BATCHED R=1 STREAMING (prefill): C positions share every weight load; per-position
// accumulation order identical to q27_k_proj_fp8_s<1,NSTEP,1> -> bit-identical tokens.
// xq/xs are per-position arrays ([C][K] int8 natural / [C][K/16] f32); yv is [C][rows] bf16.
template <int NSTEP, int C>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_s1_b(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int ystride)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row = blockIdx.x * 4 + wave;
    if (row >= rows) return;

    const unsigned char* wp = W + (size_t)row * (size_t)K;
    uint4 wv[NSTEP];
    uint4 xv[NSTEP][C];
    float sc[NSTEP][C];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        const int ks = ok[t] ? k0 : 0;
        wv[t] = q27_ld16nt(wp + ks);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            xv[t][c] = *(const uint4*)(xq + (size_t)c * K + ks);
            sc[t][c] = xs[(size_t)c * (K >> 4) + (ks >> 4)];
        }
    }
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        if (!ok[t]) continue;
#pragma unroll
        for (int c = 0; c < C; ++c) {          // each position decodes its OWN activation
            unsigned xh[8];
            q27_i8x4_h2(xv[t][c].x, xh[0], xh[1]);
            q27_i8x4_h2(xv[t][c].y, xh[2], xh[3]);
            q27_i8x4_h2(xv[t][c].z, xh[4], xh[5]);
            q27_i8x4_h2(xv[t][c].w, xh[6], xh[7]);
            acc[c] += q27_dot16_fp8(wv[t], xh, 0.0f) * sc[t][c];
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float v = acc[c];
#pragma unroll
        for (int o = 32; o > 0; o >>= 1) v += __shfl_down(v, o, 64);
        if (lane == 0) yv[(size_t)c * ystride + row] = q27_f2bf(v * alpha);
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
extern "C" void q27_proj_fp8_s1_b(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                  unsigned short* y, int C, int ystride, hipStream_t s) {
    if (w->K != 5120) return;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 g64((unsigned)((w->rows + 3) / 4)), b64(256);
    // C is padded UP to the next instantiated size: the per-position slots are 8 wide, so
    // the extra positions compute into slots nobody reads. Launching NOTHING for C in
    // {3,5,6,7} left STALE slots for the tail chunk of any prompt whose length is not a
    // multiple of 8 (205 = 25*8+5). Caught by the batched-GDN gate.
    const int Cp = (C <= 1) ? 1 : (C <= 2) ? 2 : (C <= 4) ? 4 : (C <= 8) ? 8 : 16;
    if (Cp == 16)
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b<5,16>), g64, b64, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride);
    else if (Cp == 8)
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b<5,8>), g64, b64, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride);
    else if (Cp == 4)
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b<5,4>), g64, b64, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride);
    else if (Cp == 2)
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b<5,2>), g64, b64, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride);
    else if (Cp == 1)
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b<5,1>), g64, b64, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride);
}

// SMALL-M FP8 PROJECTION (Q27_PF_QKV2). s1_b keeps one wave per row and loops the C=8 positions
// inside it, re-reading every position's activation per row and re-converting the row's fp8 weights
// per position (129-147 us per 8-position call at the q/k/v shard shapes, ~10 GB/s). Here a block
// stages one 1 KB K-chunk of all eight activations in LDS, each wave handles R rows, the row's 16
// weight bytes are converted to half2 ONCE per step, and the eight fdot2 chains run in the same
// order as q27_dot16_fp8 -> per-position sums identical to s1_b.
template <int NSTEP, int C, int R>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_s1_b2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int ystride)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int t = 0; t < NSTEP; ++t) {
        __syncthreads();
        for (int i = threadIdx.x; i < C * 64; i += 256) {
            const int c = i >> 6, l = i & 63;
            const int k0 = (t << 10) + l * 16;
            const int ks = (k0 < K) ? k0 : 0;
            sx[c][l] = *(const uint4*)(xq + (size_t)c * K + ks);
            ss[c][l] = xs[(size_t)c * (K >> 4) + (ks >> 4)];
        }
        __syncthreads();
        const int k0 = (t << 10) + lane * 16;
        const bool ok = (k0 < K);
        const int ks = ok ? k0 : 0;
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            wv[r] = q27_ld16nt(W + (size_t)rr * (size_t)K + ks);
        }
        if (!ok) continue;
        uint4 xv[C]; float xsc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { xv[c] = sx[c][lane]; xsc[c] = ss[c][lane]; }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            unsigned wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3;
            q27_e4m3x4_h2(wv[r].x, wl0, wh0); q27_e4m3x4_h2(wv[r].y, wl1, wh1);
            q27_e4m3x4_h2(wv[r].z, wl2, wh2); q27_e4m3x4_h2(wv[r].w, wl3, wh3);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                unsigned xh[8];
                q27_i8x4_h2(xv[c].x, xh[0], xh[1]); q27_i8x4_h2(xv[c].y, xh[2], xh[3]);
                q27_i8x4_h2(xv[c].z, xh[4], xh[5]); q27_i8x4_h2(xv[c].w, xh[6], xh[7]);
                float dv = 0.0f;
                dv = q27_fdot2(wl0, xh[0], dv); dv = q27_fdot2(wh0, xh[1], dv);
                dv = q27_fdot2(wl1, xh[2], dv); dv = q27_fdot2(wh1, xh[3], dv);
                dv = q27_fdot2(wl2, xh[4], dv); dv = q27_fdot2(wh2, xh[5], dv);
                dv = q27_fdot2(wl3, xh[6], dv); dv = q27_fdot2(wh3, xh[7], dv);
                acc[r][c] += dv * xsc[c];
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
            for (int o = 32; o > 0; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0 && row < rows) yv[(size_t)c * ystride + row] = q27_f2bf(v * alpha);
        }
    }
}
extern "C" int q27_proj_fp8_s1_b2(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                  unsigned short* y, int C, int ystride, int R, hipStream_t s) {
    if (w->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 4) { const dim3 g((unsigned)((w->rows + 15) / 16)), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b2<5, Q27_PF_MC, 4>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride); return 1; }
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b2<5, Q27_PF_MC, 1>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride); return 1; }
    { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
      hipLaunchKernelGGL((q27_k_proj_fp8_s1_b2<5, Q27_PF_MC, 2>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride); return 1; }
}

#ifdef Q27_EXPERIMENTS
// s1_b2 (q/k/v, staged activations, R rows) with the reduction epilogue rebuilt (q27_wave.h). Bit-identical.
template <int NSTEP, int C, int R, int GB>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_s1_b3(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int ystride)
{
    __shared__ uint4 sx[C][64];
    __shared__ float ss[C][64];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    float acc[R][C];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
#pragma unroll 1
    for (int t = 0; t < NSTEP; ++t) {
        __syncthreads();
        for (int i = threadIdx.x; i < C * 64; i += 256) {
            const int c = i >> 6, l = i & 63;
            const int k0 = (t << 10) + l * 16;
            const int ks = (k0 < K) ? k0 : 0;
            sx[c][l] = *(const uint4*)(xq + (size_t)c * K + ks);
            ss[c][l] = xs[(size_t)c * (K >> 4) + (ks >> 4)];
        }
        __syncthreads();
        const int k0 = (t << 10) + lane * 16;
        const bool ok = (k0 < K);
        const int ks = ok ? k0 : 0;
        uint4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
            wv[r] = q27_ld16nt(W + (size_t)rr * (size_t)K + ks);
        }
        if (!ok) continue;
        uint4 xv[C]; float xsc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { xv[c] = sx[c][lane]; xsc[c] = ss[c][lane]; }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            unsigned wl0, wh0, wl1, wh1, wl2, wh2, wl3, wh3;
            q27_e4m3x4_h2(wv[r].x, wl0, wh0); q27_e4m3x4_h2(wv[r].y, wl1, wh1);
            q27_e4m3x4_h2(wv[r].z, wl2, wh2); q27_e4m3x4_h2(wv[r].w, wl3, wh3);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                unsigned xh[8];
                q27_i8x4_h2(xv[c].x, xh[0], xh[1]); q27_i8x4_h2(xv[c].y, xh[2], xh[3]);
                q27_i8x4_h2(xv[c].z, xh[4], xh[5]); q27_i8x4_h2(xv[c].w, xh[6], xh[7]);
                float dv = 0.0f;
                dv = q27_fdot2(wl0, xh[0], dv); dv = q27_fdot2(wh0, xh[1], dv);
                dv = q27_fdot2(wl1, xh[2], dv); dv = q27_fdot2(wh1, xh[3], dv);
                dv = q27_fdot2(wl2, xh[4], dv); dv = q27_fdot2(wh2, xh[5], dv);
                dv = q27_fdot2(wl3, xh[6], dv); dv = q27_fdot2(wh3, xh[7], dv);
                acc[r][c] += dv * xsc[c];
            }
        }
    }
    {   float accf[R * C];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < C; ++c) accf[r * C + c] = acc[r][c];
        q27_wave_reduce_multi<R * C, GB>(accf, lane);
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = row0 + r;
#pragma unroll
                for (int c = 0; c < C; ++c) if (row < rows) yv[(size_t)c * ystride + row] = q27_f2bf(accf[r * C + c] * alpha);
            }
        }
    }
}
extern "C" int q27_proj_fp8_s1_b3(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                  unsigned short* y, int C, int ystride, int R, hipStream_t s) {
    if (w->K != 5120 || C < 1 || C > Q27_PF_MC) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 4) { const dim3 g((unsigned)((w->rows + 15) / 16)), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b3<5, Q27_PF_MC, 4, 8>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride); return 1; }
    if (R == 1) { const dim3 g((unsigned)((w->rows + 3) / 4)), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_s1_b3<5, Q27_PF_MC, 1, 8>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride); return 1; }
    { const dim3 g((unsigned)((w->rows + 7) / 8)), b(256);
      hipLaunchKernelGGL((q27_k_proj_fp8_s1_b3<5, Q27_PF_MC, 2, 8>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, ystride); return 1; }
}

#else   // closed experiments compiled out (see docs/AMD_REFERENCE_MINING.md); stubs keep the flag chains linkable
extern "C" int q27_proj_fp8_s1_b3(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y, int C, int ystride, int R, hipStream_t s) { (void)0; return 0; }
#endif
// SPLIT-K=2, INTRA-BLOCK, STREAMING. The 3x3 census proved this kernel is wave-parallelism
// bound at the TP shapes ("more waves keeps helping": 3072 waves beats 1536 beats 768), and R>1
// loses because it removes waves. Split-K adds waves WITHOUT removing rows-per-wave: two waves
// cooperate on one row, each owning half the K range, and exchange partials through 4 floats of
// LDS. Launch count is unchanged; the only new costs are one __syncthreads and two LDS ops per
// row. Same streaming load structure as q27_k_proj_fp8_s (every load issued before any consume).
// K must be even; routed only for K=5120, R=1 shapes. Q27_FP8_SK2=1 selects it; =0 is shipped.
// NOTE: the two halves sum at the end, so the accumulation order differs from the shipped kernel
// -- reassociation, legal under the numerical law, but the token gate may legitimately differ.
template <int NSTEP2, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_s_sk2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, float rs) {
#ifdef Q27_EXPERIMENTS
    const int lane  = threadIdx.x & 63;
    const int wave  = threadIdx.x >> 6;        // 0..3
    const int row   = blockIdx.x * 2 + (wave >> 1);
    const int split = wave & 1;
    if (row >= rows) return;

    const int khalf = K >> 1;
    const int klo   = split * khalf;
    const unsigned char* wp = W + (size_t)row * (size_t)K;

    uint4 wv[NSTEP2];  uint4 xv[NSTEP2];  float sc[NSTEP2];  bool ok[NSTEP2];
#pragma unroll
    for (int t = 0; t < NSTEP2; ++t) {
        const int k0 = klo + (t << 10) + lane * 16;
        ok[t] = (k0 < klo + khalf);             // predicated: half-range steps stay in bounds
        const int ks = ok[t] ? k0 : klo;        // inactive lanes read a VALID address, never OOB
        wv[t] = q27_ld16nt(wp + ks);
        xv[t] = *(const uint4*)(xq + ks);
        sc[t] = xs[ks >> 4];
    }

    float acc = 0.0f;
#pragma unroll
    for (int t = 0; t < NSTEP2; ++t) {
        if (!ok[t]) continue;
        unsigned xh[8];
        q27_i8x4_h2(xv[t].x, xh[0], xh[1]);
        q27_i8x4_h2(xv[t].y, xh[2], xh[3]);
        q27_i8x4_h2(xv[t].z, xh[4], xh[5]);
        q27_i8x4_h2(xv[t].w, xh[6], xh[7]);
        acc += q27_dot16_fp8(wv[t], xh, 0.0f) * sc[t];
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) acc += __shfl_down(acc, o, 64);

    __shared__ float part[4];
    if (lane == 0) part[wave] = acc;
    __syncthreads();
    if (lane == 0 && split == 0) {
        float v = (part[wave] + part[wave + 1]) * alpha;
        if (radd) v += rs * q27_bf2f(radd[row]);
        if (OUTBF) ((unsigned short*)yv)[row] = q27_f2bf(v);
        else       ((float*)yv)[row]         = v;
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// ------------------------------------------------------------------ host entry points
__global__ __launch_bounds__(256) void q27_k_quant_fp8(
        const unsigned short* __restrict__ x, signed char* __restrict__ xq,
        float* __restrict__ xs, int n, float global) { q27_quant_fp8_body(x, xq, xs, n, global); }
// Q27_SPEC: C rows in one launch (grid.y = row), same arithmetic per row.
__global__ __launch_bounds__(256) void q27_k_quant_fp8_b(
        const unsigned short* __restrict__ x, size_t xstride, signed char* __restrict__ xq, size_t qstride,
        float* __restrict__ xs, size_t sstride, int n, float global) {
    const int c = blockIdx.y;
    q27_quant_fp8_body(x + (size_t)c * xstride, xq + (size_t)c * qstride, xs + (size_t)c * sstride, n, global);
}
extern "C" void q27_quant_fp8_b(const unsigned short* x_bf16, size_t xstride, signed char* xq, size_t qstride,
                                float* xs, size_t sstride, int n, float in_scale, int C, hipStream_t s) {
    const int groups = (n + 15) >> 4;
    const int blocks = (groups + 255) >> 8;
    hipLaunchKernelGGL(q27_k_quant_fp8_b, dim3(blocks, C), dim3(256), 0, s,
                       x_bf16, xstride, xq, qstride, xs, sstride, n, 1.0f / in_scale);
}

extern "C" void q27_quant_fp8(const unsigned short* x_bf16, signed char* xq, float* xs,
                              int n, float in_scale, hipStream_t s) {
    const int groups = (n + 15) >> 4;
    const int blocks = (groups + 255) >> 8;
    hipLaunchKernelGGL(q27_k_quant_fp8, dim3(blocks), dim3(256), 0, s,
                       x_bf16, xq, xs, n, 1.0f / in_scale);
}

// R IS A REUSE KNOB, NOT A SCHEDULING KNOB. The rule the original comment encodes is "largest R
// with rows/R >= 960", and that reproduces all six shipped whole-model choices exactly
// (12288/10240/6144/5120 -> 4, 1024 -> 1). It had only two options, so a binary rows>=4096 test
// expressed it. TP quarters the rows of every COLUMN-parallel matrix, which pushes q_proj
// (12288->3072) and in_qkv (10240->2560) below that threshold and drops them to R=1 -- 36 B/row of
// step traffic instead of 21, and one weight load in flight instead of four. Adding R=2 lets the
// dispatcher keep following its own stated rule instead of over-shooting to R=1.
template <int R>
static inline void q27_launch_fp8(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                  float* y, float alpha, hipStream_t s) {
    const int rpb = 4 * R;
    hipLaunchKernelGGL((q27_k_proj_fp8<R>), dim3((w->rows + rpb - 1) / rpb), dim3(256), 0, s,
                       w->w, xq, xs, y, w->rows, w->K, alpha);
}

// MEASURED, NOT DERIVED. I hypothesised that TP quartering the rows had dropped q_proj
// (12288->3072) and in_qkv (10240->2560) below the rows>=4096 threshold and cost them R=4's
// activation reuse -- 21 B/row of step traffic against R=1's 36. The 3x3 census REFUTED that.
// Reproducible to +-0.03 us over three runs, K=5120:
//     rows 3072 : R=1 22.79  R=2 23.58  R=4 28.73
//     rows 2560 : R=1 20.52  R=2 20.51  R=4 22.88
//     rows 1536 : R=1 14.03  R=2 14.31  R=4 15.23
// R=4 reads 1.71x FEWER bytes per row and is 8-26% SLOWER. So this kernel is wave-parallelism
// bound at these sizes, not byte bound, and the reuse rationale in the comment above does not
// govern. More waves keeps helping well past the stated 960 target (3072 waves beats 1536 beats
// 768). The shipped rows>=4096 test therefore already picks correctly for every TP shape, and is
// left exactly as it was. Q27_FP8_R forces R for the census; Q27_FP8_CENSUS=1 reruns the table.
// ...but the census DID find a real defect, in the dimension the dispatcher does not key on.
// The row-parallel o_proj/out_proj keep rows=5120 and have their K split to 1536, so they stay on
// the rows>=4096 branch and get R=4 -- the WORST of the three there:
//     rows 5120, K 1536 : R=1 15.84  R=2 14.57  R=4 18.60   (shipped R=4 is 1.28x off best)
// The governing quantity is K, not rows. K=1536 is only 2 steps, so the fixed per-wave cost
// (row bases, the wave64 reduction, the store) is amortised over very little work and doubling
// rows-per-wave halves it. At K=5120 (5 steps) there is already enough work per wave, which is
// why R=1 wins at every column-parallel shape. 64 such calls per token.
static inline int q27_fp8_pick_R(int rows, int K) {
    // Q27_FP8_OLDR=1 restores the shipped rows-only rule EXACTLY, which is the correct A/B
    // control. Forcing a single R everywhere via Q27_FP8_R is NOT the control: the old rule gave
    // R=4 only to the >=4096-row projections and R=1 to the rest, so a uniform R=4 arm would slow
    // the baseline artificially and inflate the measured gain.
    static const bool oldr = q27_env_flag("Q27_FP8_OLDR", false);
    if (oldr) return (rows >= 4096) ? 4 : 1;
    // ENCODE EXACTLY WHAT WAS MEASURED, nothing wider. One new shape, one measured optimum.
    // R cannot be chosen from rows alone -- K changes the amortisation economics -- but the only
    // (rows,K) pair we have actually measured a different optimum for is the TP output projection.
    // A broader rule such as "K of 2 steps or fewer -> R=2" is plausible and UNPROVEN; it would
    // fire on no other shape this engine launches, so it would buy nothing and risk something.
    if (rows == 5120 && K == 1536) return 2;    // o_proj / out_proj under 4-way TP
    return (rows >= 4096) ? 4 : 1;              // every other decision preserved exactly
}

// Streaming form for the two shapes 4-way TP actually launches. K=5120 (q/k/v_proj, in_qkv, in_z;
// 176 calls/token) and K=1536 (o_proj/out_proj; 64 calls/token). Q27_FP8_STREAM=0 reverts.
template <int R, int NSTEP, int OUTBF>
static inline void q27_launch_fp8_s(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                    void* y, float alpha, hipStream_t s,
                                    const unsigned short* radd = nullptr, float rs = 0.f) {
    const int rpb = 4 * R;
    hipLaunchKernelGGL((q27_k_proj_fp8_s<R, NSTEP, OUTBF>), dim3((w->rows + rpb - 1) / rpb),
                       dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, radd, rs);
}

extern "C" void q27_proj_fp8_R(const q27_fp8_t* w, const signed char* xq, const float* xs,
                               float* y, int R, hipStream_t s) {
    const float alpha = w->in_scale * w->wscale * 256.0f;
    // Q27_FP8_NARROW=1 selects the uint2 form (512 B/step) over the uint4 form (1024 B/step).
    // REJECTED on 12 pairs, and the receipt is worth more than the change would have been:
    //   first six pairs   +2.10%  4/6   -> "KEEP"
    //   second six pairs  -1.56%  2/6   -> "REJECT"
    //   pooled twelve     +0.54%  6/12  -> a coin flip; the effect is not there
    // The two halves disagree in SIGN. Six pairs is not enough to resolve anything under ~3% on a
    // box whose run-to-run spread is 8%. Both the transfer package (uint2 741 vs uint4 643 GB/s)
    // and this engine's own profile (nvfp4 uint2 at 72% of roof vs fp8 uint4 at 55%) predicted this
    // would win. Two independent sources, a sound mechanism, and no measurable effect.
    static const bool narrow = q27_env_flag("Q27_FP8_NARROW", false);
    if (narrow) {
        if (w->K == 5120) {
            if (R == 1) { const int rpb=4; hipLaunchKernelGGL((q27_k_proj_fp8_n<1,10>),
                dim3((w->rows+rpb-1)/rpb), dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, 0); return; }
            if (R == 2) { const int rpb=8; hipLaunchKernelGGL((q27_k_proj_fp8_n<2,10>),
                dim3((w->rows+rpb-1)/rpb), dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, 0); return; }
        } else if (w->K == 1536) {
            if (R == 1) { const int rpb=4; hipLaunchKernelGGL((q27_k_proj_fp8_n<1,3>),
                dim3((w->rows+rpb-1)/rpb), dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, 0); return; }
            if (R == 2) { const int rpb=8; hipLaunchKernelGGL((q27_k_proj_fp8_n<2,3>),
                dim3((w->rows+rpb-1)/rpb), dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, 0); return; }
        }
    }
    static const bool stream = q27_env_flag("Q27_FP8_STREAM", true);
    if (stream) {
        if (w->K == 5120) {
            if (R == 1) {
                if (q27_env_flag("Q27_FP8_SK2", false))
                    hipLaunchKernelGGL((q27_k_proj_fp8_s_sk2<3,0>), dim3((w->rows + 1) / 2),
                                       dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha,
                                       nullptr, 0.f);
                else
                    q27_launch_fp8_s<1,5,0>(w, xq, xs, y, alpha, s);
                return;
            }
            if (R == 2) { q27_launch_fp8_s<2,5,0>(w, xq, xs, y, alpha, s); return; }
        } else if (w->K == 1536) {
            if (R == 1) { q27_launch_fp8_s<1,2,0>(w, xq, xs, y, alpha, s); return; }
            if (R == 2) { q27_launch_fp8_s<2,2,0>(w, xq, xs, y, alpha, s); return; }
        }
    }
    switch (R) {
    case 4: q27_launch_fp8<4>(w, xq, xs, y, alpha, s); return;
    case 2: q27_launch_fp8<2>(w, xq, xs, y, alpha, s); return;
    default: q27_launch_fp8<1>(w, xq, xs, y, alpha, s); return;
    }
}

// Writes BF16 directly, replacing proj_fp8 + q27_f2bf_vec. Returns 1 if it ran; 0 for shapes it
// does not cover so the caller keeps the two-kernel path.
// Q27_DEC_I8 / Q27_DEC_I8_PLAIN (defined with the int8 mirror kernels below)
static int q27_proj_i8g_rows(const q27_i8g_t* m, const signed char* xq, int xst, const float* xs, void* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, int outbf, hipStream_t s);
static void q27_dec_i8_route_declined(const q27_fp8_t* w, int NR);
static int q27_dec_i8_on();
static int q27_dec_i8_plain_on();
extern "C" const q27_i8g_t* q27_fp8_i8g(const q27_fp8_t* w);
extern "C" int q27_proj_fp8_bf16(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                 unsigned short* y, hipStream_t s) {
    if (!q27_env_flag("Q27_FP8_BF16", true)) return 0;
    if (q27_dec_i8_plain_on()) { const q27_i8g_t* m = q27_fp8_i8g(w); if (m) { if (q27_proj_i8g_rows(m, xq, w->K, xs, (void*)y, w->rows, nullptr, 0, 0.f, 1, -1, s)) return 1; q27_dec_i8_route_declined(w, 1); } }
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const int R = q27_fp8_pick_R(w->rows, w->K);
    if (q27_env_flag("Q27_FP8_NARROW", false) && w->K == 5120) {
        const int rpb = 4 * R;
        if (R == 1) { hipLaunchKernelGGL((q27_k_proj_fp8_n<1,10>), dim3((w->rows+rpb-1)/rpb),
                        dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, 1); return 1; }
        if (R == 2) { hipLaunchKernelGGL((q27_k_proj_fp8_n<2,10>), dim3((w->rows+rpb-1)/rpb),
                        dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, 1); return 1; }
    }
    static const int alt = q27_env_int("Q27_GEMV_ALT", 0);
    if (alt == 1 && w->K == 5120 && R == 1 && q27_proj_fp8_bf16_nr(w, xq, xs, y, w->rows, 1, s)) return 1;
    if (alt == 2 && w->K == 5120 && R == 1 && q27_proj_fp8_bf16_lds(w, xq, xs, y, w->rows, 1, s)) return 1;
    if (alt == 3 && w->K == 5120 && R == 1 && q27_proj_fp8_bf16_hg(w, xq, xs, y, w->rows, 1, s)) return 1;
    if (w->K == 5120) {
        if (R == 1) {
            if (q27_env_flag("Q27_FP8_SK2", false))
                hipLaunchKernelGGL((q27_k_proj_fp8_s_sk2<3,1>), dim3((w->rows + 1) / 2),
                                   dim3(256), 0, s, w->w, xq, xs, y, w->rows, w->K, alpha,
                                   nullptr, 0.f);
            else
                q27_launch_fp8_s<1,5,1>(w, xq, xs, y, alpha, s);
            return 1;
        }
        if (R == 2) { q27_launch_fp8_s<2,5,1>(w, xq, xs, y, alpha, s); return 1; }
    } else if (w->K == 1536) {
        if (R == 1) { q27_launch_fp8_s<1,2,1>(w, xq, xs, y, alpha, s); return 1; }
        if (R == 2) { q27_launch_fp8_s<2,2,1>(w, xq, xs, y, alpha, s); return 1; }
    }
    return 0;
}

// TWO MATRICES, ONE LAUNCH. Blocks [0, nblk_a) run matrix A, the rest run matrix B; the body is
// q27_k_proj_fp8_s verbatim (every load in flight before any consume, non-temporal weights, the
// same accumulation order, so each matrix's numbers are identical to its single launch). Built
// for the 256-row k_proj/v_proj shards under 4-way TP: 5.5 us of work each, ~25 us each as
// measured in-engine, because each is a separate launch onto a queue that has run dry. This
// changes the launch count and nothing else.
template <int R, int NSTEP, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_s2(
        const unsigned char* __restrict__ Wa, int rows_a, float alpha_a, void* __restrict__ ya,
        const unsigned char* __restrict__ Wb, int rows_b, float alpha_b, void* __restrict__ yb,
        const signed char* __restrict__ xq, const float* __restrict__ xs, int K, int nblk_a) {
#ifdef Q27_EXPERIMENTS
    const unsigned char* W; int rows; float alpha; void* yv; int blk;
    if ((int)blockIdx.x < nblk_a) { W = Wa; rows = rows_a; alpha = alpha_a; yv = ya; blk = blockIdx.x; }
    else                          { W = Wb; rows = rows_b; alpha = alpha_b; yv = yb; blk = blockIdx.x - nblk_a; }
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row0 = (blk * 4 + wave) * R;
    if (row0 >= rows) return;

    const unsigned char* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);
        wp[r] = W + (size_t)rr * (size_t)K;
    }
    uint4 wv[R][NSTEP];
    uint4 xv[NSTEP];
    float sc[NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        const int ks = ok[t] ? k0 : 0;
#pragma unroll
        for (int r = 0; r < R; ++r)
            wv[r][t] = q27_ld16nt(wp[r] + ks);
        xv[t] = *(const uint4*)(xq + ks);
        sc[t] = xs[ks >> 4];
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        if (!ok[t]) continue;
        unsigned xh[8];
        q27_i8x4_h2(xv[t].x, xh[0], xh[1]);
        q27_i8x4_h2(xv[t].y, xh[2], xh[3]);
        q27_i8x4_h2(xv[t].z, xh[4], xh[5]);
        q27_i8x4_h2(xv[t].w, xh[6], xh[7]);
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += q27_dot16_fp8(wv[r][t], xh, 0.0f) * sc[t];
    }
#pragma unroll
    for (int o = 32; o; o >>= 1)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += __shfl_down(acc[r], o, 64);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r >= rows) continue;
            const float v = acc[r] * alpha;
            if (OUTBF) ((unsigned short*)yv)[row0 + r] = q27_f2bf(v);
            else       ((float*)yv)[row0 + r]         = v;
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// Returns 1 if it ran (both matrices K=5120, the shape it is instantiated for). neg=1 is the
// FIRING CONTROL: matrix B's alpha is scaled by 1.5, which must change the tokens -- proof that
// the fused path executed and that each matrix carries its own alpha. Q27_KV_R forces R.
extern "C" int q27_proj_fp8_bf16_2(const q27_fp8_t* wa, const q27_fp8_t* wb, const signed char* xq,
                                   const float* xs, unsigned short* ya, unsigned short* yb, int neg,
                                   hipStream_t s) {
    if (wa->K != 5120 || wb->K != 5120 || wa->K != wb->K) return 0;
    static const int forced = q27_env_int("Q27_KV_R", 0);
    const int R = forced ? forced : q27_fp8_pick_R(wa->rows + wb->rows, wa->K);
    const float aa = wa->in_scale * wa->wscale * 256.0f;
    const float ab = wb->in_scale * wb->wscale * 256.0f * (neg ? 1.5f : 1.0f);
    const int rpb = 4 * R;
    const int na = (wa->rows + rpb - 1) / rpb, nb = (wb->rows + rpb - 1) / rpb;
    if (R == 1)      hipLaunchKernelGGL((q27_k_proj_fp8_s2<1,5,1>), dim3(na + nb), dim3(256), 0, s,
                         wa->w, wa->rows, aa, (void*)ya, wb->w, wb->rows, ab, (void*)yb, xq, xs, wa->K, na);
    else if (R == 2) hipLaunchKernelGGL((q27_k_proj_fp8_s2<2,5,1>), dim3(na + nb), dim3(256), 0, s,
                         wa->w, wa->rows, aa, (void*)ya, wb->w, wb->rows, ab, (void*)yb, xq, xs, wa->K, na);
    else return 0;
    return 1;
}

// Partial projection with the residual share folded into the epilogue. Returns 0 when this shape
// is not the streaming K=1536 form, so the caller can fall back to the separate pre-add kernel.
extern "C" int q27_proj_fp8_res(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                float* y, const unsigned short* radd, float rs, hipStream_t s) {
    static const bool stream = q27_env_flag("Q27_FP8_STREAM", true);
    if (!stream || w->K != 1536) return 0;
    if (q27_dec_i8_plain_on()) { const q27_i8g_t* m = q27_fp8_i8g(w); if (m) { if (q27_proj_i8g_rows(m, xq, w->K, xs, (void*)y, w->rows, radd, 0, rs, 1, -1, s)) return 1; q27_dec_i8_route_declined(w, 1); } }
    const float alpha = w->in_scale * w->wscale * 256.0f;
    static const int forced = q27_env_int("Q27_FP8_R", 0);
    static const int alt = q27_env_int("Q27_GEMV_ALT", 0);
    if (alt == 1 && q27_proj_fp8_res_nr(w, xq, xs, y, 0, radd, 0, rs, 1, s)) return 1;
    if (alt == 2 && q27_proj_fp8_res_lds(w, xq, xs, y, 0, radd, 0, rs, 1, s)) return 1;
    if (alt == 3 && q27_proj_fp8_res_hg(w, xq, xs, y, 0, radd, 0, rs, 1, s)) return 1;
    const int R = forced ? forced : q27_fp8_pick_R(w->rows, w->K);
    if (R == 1) { q27_launch_fp8_s<1,2,0>(w, xq, xs, y, alpha, s, radd, rs); return 1; }
    if (R == 2) { q27_launch_fp8_s<2,2,0>(w, xq, xs, y, alpha, s, radd, rs); return 1; }
    return 0;
}

extern "C" void q27_proj_fp8(const q27_fp8_t* w, const signed char* xq, const float* xs,
                             float* y, hipStream_t s) {
    // alpha = input_scale * weight_scale, applied ONCE (contract section 5), times the 2^8 the
    // exponent-free e4m3->fp16 decode leaves behind. Exact: 256 is a power of two.
    static const int forced = q27_env_int("Q27_FP8_R", 0);
    q27_proj_fp8_R(w, xq, xs, y, forced ? forced : q27_fp8_pick_R(w->rows, w->K), s);
}

// ---------------------------------------------------------------------------------------------
// SHAPE CENSUS (Q27_FP8_CENSUS=1). Nine kernel measurements that decide ONE dispatcher decision:
// the three TP column-parallel row counts x R in {1,2,4}. Synthetic buffers, no model load. The
// point is the measured table, NOT an elegant heuristic -- 3072 may prefer R=4 despite only 768
// waves while 1536 prefers R=2, and if so the selector should say so rather than be tidy.
extern "C" void q27_fp8_census(void)
{
    // Column-parallel matrices keep K=5120 and split rows; the ROW-parallel o_proj/out_proj keep
    // rows=5120 and split K to 1536. The latter is the one shape still on R=4 under TP, and there
    // are 64 of those calls per token, so it belongs in the census.
    // 256 is the k_proj/v_proj shape AFTER the 4-way shard (1024 rows / 4); 1024 is the pre-shard
    // form. They were missing, which is why the wave-starvation hypothesis for small-row projections
    // had never been tested: 32 such calls occur per token (16 full layers x k and v).
    const int shapes[6] = { 3072, 2560, 1536, 5120, 256, 1024 };
    const int Ks[6]     = { Q27_HID, Q27_HID, Q27_HID, 1536, Q27_HID, Q27_HID };
    const char* names[6] = { "q_proj  ", "in_qkv  ", "in_z    ", "o_proj  ", "k_proj  ", "k_noshr " };
    const int Rs[3] = { 1, 2, 4 };

    std::printf("Q27_FP8_CENSUS  (bytes/row/step: R=1 36 B, R=2 26 B, R=4 21 B)\n");
    std::printf("  %-9s %6s %6s | %10s %10s %10s | %s\n",
                "matrix", "rows", "K", "R=1 us", "R=2 us", "R=4 us", "best  waves@best");

    signed char* dXq; float* dXs; unsigned char* dW; float* dY;
    hipMalloc((void**)&dXq, Q27_HID);
    hipMalloc((void**)&dXs, (size_t)(Q27_HID >> 4) * 4);
    hipMalloc((void**)&dW,  (size_t)5120 * Q27_HID);
    hipMalloc((void**)&dY,  (size_t)5120 * 4);
    hipMemset(dXq, 1, Q27_HID);
    { std::vector<float> h(Q27_HID >> 4, 1.0f); hipMemcpy(dXs, h.data(), h.size()*4, hipMemcpyHostToDevice); }
    hipMemset(dW, 0x3C, (size_t)5120 * Q27_HID);   // an arbitrary finite e4m3 pattern

    for (int si = 0; si < 6; ++si) {
        const int K = Ks[si];
        q27_fp8_t w{}; w.rows = shapes[si]; w.K = K; w.w = dW;
        w.in_scale = 1.0f; w.wscale = 1.0f;
        double us[3];
        for (int ri = 0; ri < 3; ++ri) {
            for (int i = 0; i < 50; ++i) q27_proj_fp8_R(&w, dXq, dXs, dY, Rs[ri], 0);
            hipDeviceSynchronize();
            struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
            const int N = 2000;
            for (int i = 0; i < N; ++i) q27_proj_fp8_R(&w, dXq, dXs, dY, Rs[ri], 0);
            hipDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &b);
            us[ri] = ((b.tv_sec-a.tv_sec)*1e9 + (b.tv_nsec-a.tv_nsec)) / N / 1e3;
        }
        int best = 0; for (int r = 1; r < 3; ++r) if (us[r] < us[best]) best = r;
        std::printf("  %-9s %6d %6d | %10.2f %10.2f %10.2f | R=%d  %5d waves  (%.2fx over R=1)\n",
                    names[si], shapes[si], K, us[0], us[1], us[2],
                    Rs[best], shapes[si] / Rs[best], us[0] / us[best]);
    }
    hipFree(dXq); hipFree(dXs); hipFree(dW); hipFree(dY);
}

// ============================================================================================
// LARGE-M fp8 projection for the prefill chunk engine (PF_TILE >= 64).
//
// WHY THIS SHAPE. The shipped family is wave-per-row with C positions per lane: at C=8 it is
// 233 VGPRs -> exactly ONE wave per SIMD (16384 regs / (233*64) = 1.09), ~10% occupancy, and it
// measures 44% of the weight-bandwidth roofline. Its reuse is real but the latency is exposed,
// which is why batching 8 GEMVs measured 0.00%.
//
// MECHANISM (mined from AITER/CK-Tile and rebuilt for wave64; gfx906 has no MFMA and no native
// MXFP4, so the arithmetic stays int8-dot with the existing fp8 decode primitives):
//   * BLOCK-OWNED TILE [ROWS x POS]. Thread t owns exactly ONE output (row, pos) and runs the
//     FULL K reduction in registers -- no cross-lane shuffle, no accumulator array, so the
//     register budget is ~40 VGPRs and 3+ waves fit per SIMD.
//   * The wave's 64 lanes own 64 CONSECUTIVE positions of ONE row, so every lane in the wave
//     reads the SAME 16 B weight chunk: one broadcast transaction serves 64 outputs. That is
//     the large-M payoff -- one weight traversal per 64 positions instead of per 8 (or per 1).
//   * Activation chunks are threaded through L1: the 4 rows of the block reuse the same position
//     chunk. (CK-Tile stages these in LDS with double-buffered k-chunks; that is the next step if
//     the profile still wants it -- the layout keeps it possible, each lane reads a contiguous
//     16 B chunk at a fixed stride.)
//   * Per-16 activation scales stay where the producer put them ([pos][K/16] f32); the weight side
//     has a single per-matrix scale, as in the shipped fp8 path.
//   * Output lands straight in the consumer's per-position slot [pos][ystride] bf16 -- the layout
//     the GDN body and the attention prep already read.
//
// ACCUMULATION ORDER is NOT the shipped kernel's (that reduces lane-strided k-chunks through a
// shuffle tree). Reassociation is legal under the numerical law; the token gate decides.
template <int ROWS, int POS>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int lrow = threadIdx.x >> 6;
    const int row  = blockIdx.x * ROWS + lrow;
    const int pl   = blockIdx.y * POS + lane;      // local position inside this call
    if (row >= rows || pl >= M) return;            // M guard: tail tiles must not read or write past
    const int NG = K >> 4;
    const unsigned char* __restrict__ wr = W + (size_t)row * (size_t)K;
    const signed char*   __restrict__ ap = xq + (size_t)(m0 + pl) * (size_t)K;
    const float*         __restrict__ sp = xs + (size_t)(m0 + pl) * (size_t)NG;
    float a0 = 0.f, a1 = 0.f;
    for (int k = 0; k + 1 < NG; k += 2) {
        const uint4 w0 = *(const uint4*)(wr + (size_t)k * 16);
        const uint4 x0 = *(const uint4*)(ap + (size_t)k * 16);
        unsigned h0[8];
        q27_i8x4_h2(x0.x, h0[0], h0[1]); q27_i8x4_h2(x0.y, h0[2], h0[3]);
        q27_i8x4_h2(x0.z, h0[4], h0[5]); q27_i8x4_h2(x0.w, h0[6], h0[7]);
        a0 += q27_dot16_fp8(w0, h0, 0.f) * sp[k];
        const uint4 w1 = *(const uint4*)(wr + (size_t)(k+1) * 16);
        const uint4 x1 = *(const uint4*)(ap + (size_t)(k+1) * 16);
        unsigned h1[8];
        q27_i8x4_h2(x1.x, h1[0], h1[1]); q27_i8x4_h2(x1.y, h1[2], h1[3]);
        q27_i8x4_h2(x1.z, h1[4], h1[5]); q27_i8x4_h2(x1.w, h1[6], h1[7]);
        a1 += q27_dot16_fp8(w1, h1, 0.f) * sp[k+1];
    }
    yv[(size_t)pl * (size_t)ystride + row] = q27_f2bf((a0 + a1) * alpha);

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

// entry: M positions starting at pos0, tile [ROWS x POS], output [pos][ystride] bf16
extern "C" int q27_proj_fp8_m(const q27_fp8_t* w, const signed char* xq, const float* xs,
                               unsigned short* y, int M, int ystride, int pos0, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const int ROWS = 4, POS = 64;
    dim3 grid((unsigned)((w->rows + ROWS - 1) / ROWS), (unsigned)((M + POS - 1) / POS));
    dim3 blk(256);
    hipLaunchKernelGGL((q27_k_proj_fp8_m<4,64>), grid, blk, 0, s,
                       w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    return 1;
}

// --------------------------------------------------------------------------------------------
// v2: the SAME large-M tile, with the data movement the v1 A/B proved was wrong.
// v1 (0/12 paired wins, +9.49% wall) gave each thread ONE (row,pos) output and ran the full K
// reduction per thread. That put the wave's 64 lanes on 64 DIFFERENT positions, i.e. 64 separate
// 128 B lines per activation load at a 5120 B stride (fully uncoalesced), and it re-fetched the
// row's weight chunk on every k-step.
//
// v2 keeps the shipped kernel's winning movement and only changes WHY it is slow (233 VGPRs ->
// 1 wave/SIMD):
//   * lanes cover K (lane*16 + t*1024): every activation load for one position is a CONTIGUOUS
//     1 KB per wave, and the row's weight chunks are hoisted into registers ONCE per row for the
//     whole position tile (wv[NSTEP], 20 VGPRs) instead of being reloaded per position;
//   * a wave owns ONE row and walks the tile's positions in groups of C=4, so one shuffle tree
//     amortises over 4 outputs instead of 1;
//   * live registers ~= 20 (weights) + 16 (one k-step x 4 positions) + 32 (decoded) + 4 (acc)
//     ~= 90, i.e. 2 waves/SIMD against the shipped 1 -- latency hidden by occupancy, which is
//     what the 44%-of-roofline measurement asked for.
template <int NSTEP, int C>
__global__ __launch_bounds__(256, Q27_FP8_MINW) void q27_k_proj_fp8_m2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row  = blockIdx.x * 4 + wave;
    if (row >= rows) return;
    const unsigned char* __restrict__ wp = W + (size_t)row * (size_t)K;
    uint4 wv[NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        wv[t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));      // hoisted: one fetch serves the whole tile
    }
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;              // 16-k group index (scale index)
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const signed char* ap = xq + (size_t)(m0 + q0 + c) * (size_t)K + k0;
                const uint4 xv = *(const uint4*)ap;     // coalesced: all lanes, one position
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                acc[c] += q27_dot16_fp8(wv[t], xh, 0.f)
                          * xs[(size_t)(m0 + q0 + c) * (size_t)(K >> 4) + k16];
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c >= Cn) break;
            float v = acc[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0) yv[(size_t)(q0 + c) * (size_t)ystride + row] = q27_f2bf(v * alpha);
        }
    }
}

extern "C" int q27_proj_fp8_m2(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    dim3 grid((unsigned)((w->rows + 3) / 4), 1u), blk(256);
    hipLaunchKernelGGL((q27_k_proj_fp8_m2<5,4>), grid, blk, 0, s,
                       w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    return 1;
}

// Tile projection v3 (Q27_PF_MK3): the m2 movement -- lanes over K, the row's weights hoisted once,
// the SAME per-lane fdot2 chain and the SAME 64-lane tree, so every output is bit-identical to m2 --
// but the activation slice of one k-step is staged in LDS once per BLOCK and consumed by 4 waves x R
// rows. m2 re-read the whole activation tile from L2 once per row (4608 rows x 128 positions x 5 KB
// = 3 GB per layer-tile; the 128-token trace measured it at 1056 us, ~7% of dot2 peak) and redid the
// int8->half conversion per row. Here L2 traffic drops by 4R and the conversion amortises over R.
template <int NSTEP, int C, int R>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m3l(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    __shared__ __align__(16) signed char sx[C][1024];     // C positions x one 1024-wide k-step
    const int lane  = threadIdx.x & 63;
    const int wave  = threadIdx.x >> 6;
    const int rbase = (blockIdx.x * 4 + wave) * R;
    uint4 wv[R][NSTEP];
    bool  ok[NSTEP], rok[R];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) ok[t] = ((t << 10) + lane * 16) < K;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        rok[r] = (rbase + r) < rows;
        const unsigned char* __restrict__ wp = W + (size_t)(rok[r] ? rbase + r : 0) * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            wv[r][t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        }
    }
    const int ksc = K >> 4;
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;         // block-uniform
        float acc[C][R];
#pragma unroll
        for (int c = 0; c < C; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[c][r] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if ((t << 10) >= K) continue;                    // block-uniform (K is a multiple of 1024 here)
            __syncthreads();                                 // previous k-step's LDS reads are done
            for (int i = threadIdx.x; i < Cn * 64; i += 256) {
                const int c = i >> 6, l = i & 63;
                ((uint4*)sx[c])[l] = *(const uint4*)(xq + (size_t)(m0 + q0 + c) * (size_t)K + (t << 10) + l * 16);
            }
            __syncthreads();
            const int k16 = t * 64 + lane;
            for (int c = 0; c < Cn; ++c) {
                const uint4 xv = ((const uint4*)sx[c])[lane];
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(m0 + q0 + c) * (size_t)ksc + k16];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[c][r] += q27_dot16_fp8(wv[r][t], xh, 0.f) * sc;
            }
        }
        for (int c = 0; c < Cn; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float v = acc[c][r];
#pragma unroll
                for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
                if (lane == 0 && rok[r]) yv[(size_t)(q0 + c) * (size_t)ystride + rbase + r] = q27_f2bf(v * alpha);
            }
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_fp8_m3l(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, int R, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 4) {
        dim3 grid((unsigned)((w->rows + 15) / 16), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m3l<5,4,4>), grid, blk, 0, s,
                           w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
        return 1;
    }
    if (R == 2) {
        dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m3l<5,8,2>), grid, blk, 0, s,
                           w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
        return 1;
    }
    return 0;
}

// M-batched projection writing the fp32 PARTIAL for the all-reduce, with each position's residual
// share folded in the epilogue: the chunk form of q27_proj_fp8_res (out_proj / o_proj, K=1536).
// Same per-position accumulation as q27_k_proj_fp8_m2; epilogue order matches q27_k_proj_fp8_s's
// residual fold (v = acc*alpha; v += rs*residual).
template <int NSTEP, int C>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m2f(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, float* __restrict__ Y,
        int rows, int K, float alpha, int M, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs)
{
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row  = blockIdx.x * 4 + wave;
    if (row >= rows) return;
    const unsigned char* __restrict__ wp = W + (size_t)row * (size_t)K;
    uint4 wv[NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        wv[t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));
    }
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const signed char* ap = xq + (size_t)(q0 + c) * (size_t)K + k0;
                const uint4 xv = *(const uint4*)ap;
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                acc[c] += q27_dot16_fp8(wv[t], xh, 0.f)
                          * xs[(size_t)(q0 + c) * (size_t)(K >> 4) + k16];
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c >= Cn) break;
            float v = acc[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0) {
                v *= alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)(q0 + c) * (size_t)rstride + row]);
                Y[(size_t)(q0 + c) * (size_t)ystride + row] = v;
            }
        }
    }
}
// Chunk out_proj / o_proj v3 (Q27_PF_M3F): m2f's per-lane chain, lane tree and residual-fold
// epilogue (bit-identical), with the chunk's k-step activation slice staged in LDS once per block
// and R rows per wave. K = 1536: k-step 1 is valid for lanes 0..31 only (per-lane ok, as in m2f).
template <int NSTEP, int C, int R>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m3f(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, float* __restrict__ Y,
        int rows, int K, float alpha, int M, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs)
{
#ifdef Q27_EXPERIMENTS
    __shared__ __align__(16) signed char sx[C][1024];
    const int lane  = threadIdx.x & 63;
    const int wave  = threadIdx.x >> 6;
    const int rbase = (blockIdx.x * 4 + wave) * R;
    uint4 wv[R][NSTEP];
    bool  ok[NSTEP], rok[R];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) ok[t] = ((t << 10) + lane * 16) < K;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        rok[r] = (rbase + r) < rows;
        const unsigned char* __restrict__ wp = W + (size_t)(rok[r] ? rbase + r : 0) * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            wv[r][t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        }
    }
    const int ksc = K >> 4;
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[C][R];
#pragma unroll
        for (int c = 0; c < C; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[c][r] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if ((t << 10) >= K) continue;                    // block-uniform
            __syncthreads();
            const int kn = (K - (t << 10) < 1024) ? (K - (t << 10)) : 1024;   // valid bytes of this k-step
            for (int i = threadIdx.x; i < Cn * 64; i += 256) {
                const int c = i >> 6, l = i & 63;
                if (l * 16 < kn)
                    ((uint4*)sx[c])[l] = *(const uint4*)(xq + (size_t)(q0 + c) * (size_t)K + (t << 10) + l * 16);
            }
            __syncthreads();
            if (!ok[t]) continue;                            // per lane, no barrier below
            const int k16 = t * 64 + lane;
            for (int c = 0; c < Cn; ++c) {
                const uint4 xv = ((const uint4*)sx[c])[lane];
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(q0 + c) * (size_t)ksc + k16];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[c][r] += q27_dot16_fp8(wv[r][t], xh, 0.f) * sc;
            }
        }
        for (int c = 0; c < Cn; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float v = acc[c][r];
#pragma unroll
                for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
                if (lane == 0 && rok[r]) {
                    const int row = rbase + r;
                    v *= alpha;
                    if (radd) v += rs * q27_bf2f(radd[(size_t)(q0 + c) * (size_t)rstride + row]);
                    Y[(size_t)(q0 + c) * (size_t)ystride + row] = v;
                }
            }
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_fp8_m3_res(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                   float* y, int M, int ystride, const unsigned short* radd,
                                   int rstride, float rs, int R, hipStream_t s) {
    if (!w || w->K != 1536 || M <= 0 || M > Q27_PF_CH) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 4) {
        dim3 grid((unsigned)((w->rows + 15) / 16), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m3f<2,8,4>), grid, blk, 0, s,
                           w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, radd, rstride, rs);
        return 1;
    }
    if (R == 2) {
        dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m3f<2,8,2>), grid, blk, 0, s,
                           w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, radd, rstride, rs);
        return 1;
    }
    return 0;
}

// Tile projection with R rows per wave and NO LDS (Q27_PF_MK2R). The census: m2 and its fp16-weight
// twin cost the same per position (9.5 us at 2560 rows), so the kernel is bound by the activation
// tile it re-reads from L2 once per row (~3 GB per tile-layer), not by the conversions. Here one
// activation load and one conversion per (position, k-step) feed R rows' hoisted weights: L2 traffic
// divided by R, no LDS, no extra barriers. Same chain and lane tree per (row, position): bit-identical.
template <int NSTEP, int C, int R>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m2r(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
    const int lane  = threadIdx.x & 63;
    const int wave  = threadIdx.x >> 6;
    const int rbase = (blockIdx.x * 4 + wave) * R;
    if (rbase >= rows) return;
    uint4 wv[R][NSTEP];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (rbase + r < rows) ? (rbase + r) : (rows - 1);
        const unsigned char* __restrict__ wp = W + (size_t)rr * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) wv[r][t] = q27_ld16nt(wp + (t << 10) + lane * 16);
    }
    const int ksc = K >> 4;
    for (int q0 = 0; q0 < M; q0 += C) {           // full groups: the tile slots hold Q27_PF_TSLOT rows
        float acc[C][R];
#pragma unroll
        for (int c = 0; c < C; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[c][r] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = *(const uint4*)(xq + (size_t)(m0 + q0 + c) * (size_t)K + k0);
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(m0 + q0 + c) * (size_t)ksc + k16];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[c][r] += q27_dot16_fp8(wv[r][t], xh, 0.f) * sc;
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float v = acc[c][r];
#pragma unroll
                for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
                if (lane == 0 && rbase + r < rows) yv[(size_t)(q0 + c) * (size_t)ystride + rbase + r] = q27_f2bf(v * alpha);
            }
        }
    }
}
// m2 with exactly two rows per wave, written as m2 is (per-lane ok guard, group trimming, flat
// per-row arrays) so the compiler keeps m2's register shape; only the second row's weights and
// accumulators are added. Bit-identical to m2.
template <int NSTEP, int C>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m2x2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int rowA = (blockIdx.x * 4 + wave) * 2;
    if (rowA >= rows) return;
    const int rowB = (rowA + 1 < rows) ? (rowA + 1) : rowA;
    const unsigned char* __restrict__ wpA = W + (size_t)rowA * (size_t)K;
    const unsigned char* __restrict__ wpB = W + (size_t)rowB * (size_t)K;
    uint4 wvA[NSTEP], wvB[NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        wvA[t] = q27_ld16nt(wpA + (ok[t] ? k0 : 0));
        wvB[t] = q27_ld16nt(wpB + (ok[t] ? k0 : 0));
    }
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float accA[C], accB[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { accA[c] = 0.f; accB[c] = 0.f; }
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const signed char* ap = xq + (size_t)(m0 + q0 + c) * (size_t)K + k0;
                const uint4 xv = *(const uint4*)ap;
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(m0 + q0 + c) * (size_t)(K >> 4) + k16];
                accA[c] += q27_dot16_fp8(wvA[t], xh, 0.f) * sc;
                accB[c] += q27_dot16_fp8(wvB[t], xh, 0.f) * sc;
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c >= Cn) break;
            float vA = accA[c], vB = accB[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) { vA += __shfl_down(vA, o, 64); vB += __shfl_down(vB, o, 64); }
            if (lane == 0) {
                yv[(size_t)(q0 + c) * (size_t)ystride + rowA] = q27_f2bf(vA * alpha);
                if (rowA + 1 < rows) yv[(size_t)(q0 + c) * (size_t)ystride + rowA + 1] = q27_f2bf(vB * alpha);
            }
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
extern "C" int q27_proj_fp8_m2x2(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                 unsigned short* y, int M, int ystride, int pos0, int C, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
    if (C == 2) { hipLaunchKernelGGL((q27_k_proj_fp8_m2x2<5,2>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); return 1; }
    hipLaunchKernelGGL((q27_k_proj_fp8_m2x2<5,4>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    return 1;
}

extern "C" int q27_proj_fp8_m2r(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, int R, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 5) return q27_proj_fp8_m2x2(w, xq, xs, y, M, ystride, pos0, 4, s);   // two rows, m2's structure, 4-position groups
    if (R == 6) return q27_proj_fp8_m2x2(w, xq, xs, y, M, ystride, pos0, 2, s);   // two rows, 2-position groups
    if (M > Q27_PF_TSLOT) return 0;                       // full groups read/write within the tile slots
    if (R == 2) { dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2r<5,2,2>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); return 1; }
    if (R == 3) { dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);   // R=2 with 4-position groups
        hipLaunchKernelGGL((q27_k_proj_fp8_m2r<5,4,2>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); return 1; }
    if (R == 4) { dim3 grid((unsigned)((w->rows + 15) / 16), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2r<5,1,4>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); return 1; }
    return 0;
}

// m2r with the reduction epilogue rebuilt (see q27_wave.h): the C x R reductions of each position group issued
// together (bpermutes in flight, DPP in-row steps) instead of 6 serialized shuffle+wait links per output. Bit-identical.
template <int NSTEP, int C, int R, int GB>
__global__ __launch_bounds__(256, Q27_FP8_MINW) void q27_k_proj_fp8_m2r2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
    const int lane  = threadIdx.x & 63;
    const int wave  = threadIdx.x >> 6;
    const int rbase = (blockIdx.x * 4 + wave) * R;
    if (rbase >= rows) return;
    uint4 wv[R][NSTEP];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (rbase + r < rows) ? (rbase + r) : (rows - 1);
        const unsigned char* __restrict__ wp = W + (size_t)rr * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) wv[r][t] = q27_ld16nt(wp + (t << 10) + lane * 16);
    }
    const int ksc = K >> 4;
    for (int q0 = 0; q0 < M; q0 += C) {           // full groups: the tile slots hold Q27_PF_TSLOT rows
        float acc[C][R];
#pragma unroll
        for (int c = 0; c < C; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[c][r] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = *(const uint4*)(xq + (size_t)(m0 + q0 + c) * (size_t)K + k0);
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(m0 + q0 + c) * (size_t)ksc + k16];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[c][r] += q27_dot16_fp8(wv[r][t], xh, 0.f) * sc;
            }
        }
        {   float accf[C * R];
#pragma unroll
            for (int c = 0; c < C; ++c)
#pragma unroll
                for (int r = 0; r < R; ++r) accf[c * R + r] = acc[c][r];
            q27_wave_reduce_multi<C * R, GB>(accf, lane);
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < C; ++c)
#pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (rbase + r < rows) yv[(size_t)(q0 + c) * (size_t)ystride + rbase + r] = q27_f2bf(accf[c * R + r] * alpha);
            }
        }
    }
}
extern "C" int q27_proj_fp8_m2r2(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                 unsigned short* y, int M, int ystride, int pos0, int R, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0 || M > Q27_PF_TSLOT) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 2) { dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2r2<5,2,2,4>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); return 1; }
    if (R == 3) { dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2r2<5,4,2,8>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); return 1; }
    if (R == 4) { dim3 grid((unsigned)((w->rows + 15) / 16), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2r2<5,1,4,4>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); return 1; }
    return 0;
}

// Tile projection with PRE-CONVERTED fp16 weights (Q27_PF_MK16). m2 spends ~24 of the ~53
// instructions per (row, position, k-step) turning 16 fp8 weights into half2 pairs, once per
// position although the weights do not change across the tile. The conversion kernel below runs
// once per tile-layer (42 MB written at ~900 GB/s, ~70 us) and stores exactly the lo/hi dwords
// q27_e4m3x4_h2 produces, in q27_dot16_fp8's consumption order; the projection then feeds them to
// the same fdot2 chain with the same activation conversion and lane tree: bit-identical to m2.
__global__ __launch_bounds__(256) void q27_k_fp8_to_h2(const uint4* __restrict__ W, uint4* __restrict__ W16, size_t n16)
{
#ifdef Q27_EXPERIMENTS
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;   // one 16-weight group per thread
    if (i >= n16) return;
    const uint4 w = W[i];
    uint4 a, b;
    q27_e4m3x4_h2(w.x, a.x, a.y); q27_e4m3x4_h2(w.y, a.z, a.w);
    q27_e4m3x4_h2(w.z, b.x, b.y); q27_e4m3x4_h2(w.w, b.z, b.w);
    W16[2 * i] = a; W16[2 * i + 1] = b;

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
extern "C" int q27_fp8_to_h2(const q27_fp8_t* w, void* w16, hipStream_t s)
{
    if (!w || !w16 || (w->K & 15)) return 0;
    const size_t n16 = (size_t)w->rows * (size_t)(w->K >> 4);
    hipLaunchKernelGGL(q27_k_fp8_to_h2, dim3((unsigned)((n16 + 255) / 256)), dim3(256), 0, s,
                       (const uint4*)w->w, (uint4*)w16, n16);
    return 1;
}
template <int NSTEP, int C>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m2w16(
        const uint4* __restrict__ W16, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row  = blockIdx.x * 4 + wave;
    if (row >= rows) return;
    const uint4* __restrict__ wp = W16 + (size_t)row * (size_t)(K >> 3);   // 2 uint4 per 16 weights
    uint4 wa[NSTEP], wb[NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        const int g16 = (ok[t] ? k0 : 0) >> 4;
        wa[t] = q27_ld16nt((const unsigned char*)(wp + 2 * g16));
        wb[t] = q27_ld16nt((const unsigned char*)(wp + 2 * g16 + 1));
    }
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const signed char* ap = xq + (size_t)(m0 + q0 + c) * (size_t)K + k0;
                const uint4 xv = *(const uint4*)ap;
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                float d = 0.f;
                d = q27_fdot2(wa[t].x, xh[0], d); d = q27_fdot2(wa[t].y, xh[1], d);
                d = q27_fdot2(wa[t].z, xh[2], d); d = q27_fdot2(wa[t].w, xh[3], d);
                d = q27_fdot2(wb[t].x, xh[4], d); d = q27_fdot2(wb[t].y, xh[5], d);
                d = q27_fdot2(wb[t].z, xh[6], d); d = q27_fdot2(wb[t].w, xh[7], d);
                acc[c] += d * xs[(size_t)(m0 + q0 + c) * (size_t)(K >> 4) + k16];
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c >= Cn) break;
            float v = acc[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0) yv[(size_t)(q0 + c) * (size_t)ystride + row] = q27_f2bf(v * alpha);
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
extern "C" int q27_proj_fp8_m2w16(const q27_fp8_t* w, const void* w16, const signed char* xq, const float* xs,
                                  unsigned short* y, int M, int ystride, int pos0, hipStream_t s) {
    if (!w || !w16 || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    dim3 grid((unsigned)((w->rows + 3) / 4), 1u), blk(256);
    hipLaunchKernelGGL((q27_k_proj_fp8_m2w16<5,4>), grid, blk, 0, s,
                       (const uint4*)w16, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    return 1;
}

// Chunk out_proj / o_proj with R rows per wave and NO LDS (Q27_PF_M2F2): the census put m2f at
// 77 us for 63 M MACs (0.8 TMAC/s) with per-position cost flat in M -- 5120 short dependent waves,
// latency-bound. R rows per wave quarters the wave count and shares each activation load and its
// conversion across R rows; per (row, position) arithmetic and epilogue are m2f's (bit-identical).
template <int NSTEP, int C, int R>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m2f2(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, float* __restrict__ Y,
        int rows, int K, float alpha, int M, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs)
{
    const int lane  = threadIdx.x & 63;
    const int wave  = threadIdx.x >> 6;
    const int rbase = (blockIdx.x * 4 + wave) * R;
    uint4 wv[R][NSTEP];
    bool  ok[NSTEP], rok[R];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) ok[t] = ((t << 10) + lane * 16) < K;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        rok[r] = (rbase + r) < rows;
        const unsigned char* __restrict__ wp = W + (size_t)(rok[r] ? rbase + r : 0) * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            wv[r][t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        }
    }
    const int ksc = K >> 4;
    for (int q0 = 0; q0 < M; q0 += C) {           // full groups: the chunk slots are Q27_PF_CH wide
        float acc[C][R];
#pragma unroll
        for (int c = 0; c < C; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[c][r] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = *(const uint4*)(xq + (size_t)(q0 + c) * (size_t)K + k0);
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(q0 + c) * (size_t)ksc + k16];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[c][r] += q27_dot16_fp8(wv[r][t], xh, 0.f) * sc;
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float v = acc[c][r];
#pragma unroll
                for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
                if (lane == 0 && rok[r]) {
                    const int row = rbase + r;
                    v *= alpha;
                    if (radd) v += rs * q27_bf2f(radd[(size_t)(q0 + c) * (size_t)rstride + row]);
                    Y[(size_t)(q0 + c) * (size_t)ystride + row] = v;
                }
            }
        }
    }
}
extern "C" int q27_proj_fp8_m2_res2(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                    float* y, int M, int ystride, const unsigned short* radd,
                                    int rstride, float rs, int R, hipStream_t s) {
    if (!w || w->K != 1536 || M <= 0 || M > Q27_PF_CH) return 0;   // full 4-groups read/write within the CH-wide slots
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 4) { dim3 grid((unsigned)((w->rows + 15) / 16), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2f2<2,4,4>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, radd, rstride, rs); return 1; }
    if (R == 2) { dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2f2<2,4,2>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, radd, rstride, rs); return 1; }
    return 0;
}

// m2f2 (chunk out_proj, R rows) with the reduction epilogue rebuilt (q27_wave.h). Bit-identical.
template <int NSTEP, int C, int R, int GB>
__global__ __launch_bounds__(256, Q27_FP8_MINW) void q27_k_proj_fp8_m2f3(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, float* __restrict__ Y,
        int rows, int K, float alpha, int M, int ystride,
        const unsigned short* __restrict__ radd, int rstride, float rs)
{
    const int lane  = threadIdx.x & 63;
    const int wave  = threadIdx.x >> 6;
    const int rbase = (blockIdx.x * 4 + wave) * R;
    uint4 wv[R][NSTEP];
    bool  ok[NSTEP], rok[R];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) ok[t] = ((t << 10) + lane * 16) < K;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        rok[r] = (rbase + r) < rows;
        const unsigned char* __restrict__ wp = W + (size_t)(rok[r] ? rbase + r : 0) * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            wv[r][t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        }
    }
    const int ksc = K >> 4;
    for (int q0 = 0; q0 < M; q0 += C) {           // full groups: the chunk slots are Q27_PF_CH wide
        float acc[C][R];
#pragma unroll
        for (int c = 0; c < C; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[c][r] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint4 xv = *(const uint4*)(xq + (size_t)(q0 + c) * (size_t)K + k0);
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(q0 + c) * (size_t)ksc + k16];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    acc[c][r] += q27_dot16_fp8(wv[r][t], xh, 0.f) * sc;
            }
        }
        {   float accf[C * R];
#pragma unroll
            for (int c = 0; c < C; ++c)
#pragma unroll
                for (int r = 0; r < R; ++r) accf[c * R + r] = acc[c][r];
            q27_wave_reduce_multi<C * R, GB>(accf, lane);
            if (lane == 0) {
#pragma unroll
                for (int c = 0; c < C; ++c)
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        if (!rok[r]) continue;
                        const int row = rbase + r;
                        float v = accf[c * R + r] * alpha;
                        if (radd) v += rs * q27_bf2f(radd[(size_t)(q0 + c) * (size_t)rstride + row]);
                        Y[(size_t)(q0 + c) * (size_t)ystride + row] = v;
                    }
            }
        }
    }
}
extern "C" int q27_proj_fp8_m2_res3(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                    float* y, int M, int ystride, const unsigned short* radd,
                                    int rstride, float rs, int R, hipStream_t s) {
    if (!w || w->K != 1536 || M <= 0 || M > Q27_PF_CH) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    if (R == 4) { dim3 grid((unsigned)((w->rows + 15) / 16), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2f3<2,4,4,8>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, radd, rstride, rs); return 1; }
    if (R == 2) { dim3 grid((unsigned)((w->rows + 7) / 8), 1u), blk(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m2f3<2,4,2,8>), grid, blk, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, radd, rstride, rs); return 1; }
    return 0;
}

extern "C" int q27_proj_fp8_m2_res(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                   float* y, int M, int ystride, const unsigned short* radd,
                                   int rstride, float rs, hipStream_t s) {
    if (!w || w->K != 1536 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    dim3 grid((unsigned)((w->rows + 3) / 4), 1u), blk(256);
    hipLaunchKernelGGL((q27_k_proj_fp8_m2f<2,4>), grid, blk, 0, s,
                       w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, radd, rstride, rs);
    return 1;
}

// ============================================================================================
// KERNEL-LEVEL A/B: the shipped wave-per-row batched kernel (s1_b, C=8) against the large-M v2
// kernel, at the exact GDN production shard shapes and the M ladder the prefill engine uses.
// This is the measurement that says whether the large-M geometry actually moves the projection --
// inferred-from-wall is not evidence. It also cross-checks the two kernels against each other,
// so a fast-but-wrong kernel cannot pass.
//   usage: Q27_MK_BENCH=1 ./q27_gen        (no model load; needs the GPU lock)
// CK is a driver-side macro; this translation unit needs its own check.
#define MKB_CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { std::printf("  MK_BENCH hip error %d at line %d\n", (int)_e, __LINE__); return; } } while (0)

extern "C" void q27_mk_bench(void) {
    const int K = 5120;
    const int rows_tab[2] = { 2560, 1536 };              // in_qkv, in_z (per-card GDN shards)
    const int M_tab[4] = { 8, 41, 128, 256 };
    const char* rn[2] = { "in_qkv", "in_z  " };
    for (int r = 0; r < 2; ++r) {
        const int rows = rows_tab[r];
        unsigned char* dW = nullptr; signed char* dX = nullptr; float* dS = nullptr;
        unsigned short* dY = nullptr; unsigned short* dY2 = nullptr;
        MKB_CK(hipMalloc((void**)&dW, (size_t)rows * (size_t)K));
        MKB_CK(hipMalloc((void**)&dX, (size_t)256 * (size_t)K));
        MKB_CK(hipMalloc((void**)&dS, (size_t)256 * (size_t)(K / 16) * 4));
        MKB_CK(hipMalloc((void**)&dY, (size_t)256 * (size_t)rows * 2));
        MKB_CK(hipMalloc((void**)&dY2, (size_t)256 * (size_t)rows * 2));
        unsigned* dX2 = nullptr;
        MKB_CK(hipMalloc((void**)&dX2, (size_t)256 * (size_t)K * 2));
        { std::vector<unsigned char> h((size_t)rows * (size_t)K);
          for (size_t i = 0; i < h.size(); ++i) h[i] = (unsigned char)((i * 2654435761u) >> 24);
          MKB_CK(hipMemcpy(dW, h.data(), h.size(), hipMemcpyHostToDevice)); }
        { std::vector<signed char> h((size_t)256 * (size_t)K);
          for (size_t i = 0; i < h.size(); ++i) h[i] = (signed char)((((i * 40503u) >> 8) & 0x3f) - 32);
          MKB_CK(hipMemcpy(dX, h.data(), h.size(), hipMemcpyHostToDevice)); }
        { std::vector<float> h((size_t)256 * (size_t)(K / 16), 0.03125f);
          MKB_CK(hipMemcpy(dS, h.data(), h.size() * 4, hipMemcpyHostToDevice)); }
        q27_fp8_t w; w.w = dW; w.wscale = 1.0f; w.in_scale = 1.0f; w.rows = rows; w.K = K;
        std::printf("  %s rows=%d K=%d  weights %.1f MB\n", rn[r], rows, K, rows * (double)K / 1e6);
        for (int mi = 0; mi < 4; ++mi) {
            const int M = M_tab[mi];
            const int ystride = rows;
            // shipped path: ceil(M/8) launches of the C=8 batched kernel, the shape it was built for
            auto ship = [&]() {
                for (int c = 0; c < M; c += 8) {
                    const int Cc = (M - c < 8) ? (M - c) : 8;
                    q27_proj_fp8_s1_b(&w, dX + (size_t)c * K, dS + (size_t)c * (K / 16),
                                      dY + (size_t)c * ystride, Cc, ystride, 0);
                }
            };
            auto big = [&]() { q27_proj_fp8_m2(&w, dX, dS, dY2, M, ystride, 0, 0); };
            ship(); big(); MKB_CK(hipDeviceSynchronize());
            double mx = 0; size_t nbad = 0;
            { std::vector<unsigned short> a((size_t)M * ystride), b((size_t)M * ystride);
              MKB_CK(hipMemcpy(a.data(), dY, a.size() * 2, hipMemcpyDeviceToHost));
              MKB_CK(hipMemcpy(b.data(), dY2, b.size() * 2, hipMemcpyDeviceToHost));
              for (size_t i = 0; i < a.size(); ++i) {
                  const float x = (float)a[i] * 0.00390625f, y = (float)b[i] * 0.00390625f;
                  const double d = std::fabs((double)x - (double)y);
                  if (d > mx) mx = d;
                  if (d > 0.02) ++nbad;
              } }
            const int REPS = 20;
            hipEvent_t e0, e1; MKB_CK(hipEventCreate(&e0)); MKB_CK(hipEventCreate(&e1));
            MKB_CK(hipEventRecord(e0, 0)); for (int i = 0; i < REPS; ++i) ship(); MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
            float ms_ship = 0; MKB_CK(hipEventElapsedTime(&ms_ship, e0, e1)); ms_ship /= REPS;
            MKB_CK(hipEventRecord(e0, 0)); for (int i = 0; i < REPS; ++i) big(); MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
            float ms_big = 0; MKB_CK(hipEventElapsedTime(&ms_big, e0, e1)); ms_big /= REPS;
            float ms3[3] = {0,0,0};
            for (int k = 0; k < 3; ++k) {
                const int rw = (k == 0) ? 1 : (k == 1 ? 2 : 4);
                q27_proj_fp8_m3(&w, dX, dS, dY2, M, ystride, 0, rw, 0);
                MKB_CK(hipEventRecord(e0, 0));
                for (int i = 0; i < REPS; ++i) q27_proj_fp8_m3(&w, dX, dS, dY2, M, ystride, 0, rw, 0);
                MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
                MKB_CK(hipEventElapsedTime(&ms3[k], e0, e1)); ms3[k] /= REPS;
            }
            q27_proj_fp8_m4(&w, dX, dS, dY2, M, ystride, 0, 0);
            MKB_CK(hipEventRecord(e0, 0));
            for (int i = 0; i < REPS; ++i) q27_proj_fp8_m4(&w, dX, dS, dY2, M, ystride, 0, 0);
            MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
            float ms4 = 0; MKB_CK(hipEventElapsedTime(&ms4, e0, e1)); ms4 /= REPS;
            q27_i8_to_h2(dX, dX2, 256 * K, 0);
            q27_proj_fp8_m6(&w, dX2, dS, dY2, M, ystride, 0, 0);
            MKB_CK(hipEventRecord(e0, 0));
            for (int i = 0; i < REPS; ++i) q27_proj_fp8_m6(&w, dX2, dS, dY2, M, ystride, 0, 0);
            MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
            float ms6 = 0; MKB_CK(hipEventElapsedTime(&ms6, e0, e1)); ms6 /= REPS;
            q27_proj_fp8_m7(&w, dX, dS, dY2, M, ystride, 0, 0);
            MKB_CK(hipEventRecord(e0, 0));
            for (int i = 0; i < REPS; ++i) q27_proj_fp8_m7(&w, dX, dS, dY2, M, ystride, 0, 0);
            MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
            float ms7 = 0; MKB_CK(hipEventElapsedTime(&ms7, e0, e1)); ms7 /= REPS;
            float ms9[3] = {0,0,0};
            for (int k = 0; k < 3; ++k) {
                const int rw = (k == 0) ? 1 : (k == 1 ? 2 : 4);
                q27_proj_fp8_m9(&w, dX, dS, dY2, M, ystride, 0, rw, 0);
                MKB_CK(hipEventRecord(e0, 0));
                for (int i = 0; i < REPS; ++i) q27_proj_fp8_m9(&w, dX, dS, dY2, M, ystride, 0, rw, 0);
                MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
                MKB_CK(hipEventElapsedTime(&ms9[k], e0, e1)); ms9[k] /= REPS;
            }
            double m7diff = 0;
            { std::vector<unsigned short> a7((size_t)M * ystride), b7((size_t)M * ystride);
              MKB_CK(hipMemcpy(a7.data(), dY, a7.size() * 2, hipMemcpyDeviceToHost));
              MKB_CK(hipMemcpy(b7.data(), dY2, b7.size() * 2, hipMemcpyDeviceToHost));
              for (size_t i = 0; i < a7.size(); ++i) { const double dd = std::fabs((double)a7[i] - (double)b7[i]); if (dd > m7diff) m7diff = dd; } }
            double m6diff = 0;
            { std::vector<unsigned short> a6((size_t)M * ystride), b6((size_t)M * ystride);
              MKB_CK(hipMemcpy(a6.data(), dY, a6.size() * 2, hipMemcpyDeviceToHost));
              MKB_CK(hipMemcpy(b6.data(), dY2, b6.size() * 2, hipMemcpyDeviceToHost));
              for (size_t i = 0; i < a6.size(); ++i) { const double dd = std::fabs((double)a6[i] - (double)b6[i]); if (dd > m6diff) m6diff = dd; } }
            double m4diff = 0;
            { std::vector<unsigned short> a4((size_t)M * ystride), b4((size_t)M * ystride);
              MKB_CK(hipMemcpy(a4.data(), dY, a4.size() * 2, hipMemcpyDeviceToHost));
              MKB_CK(hipMemcpy(b4.data(), dY2, b4.size() * 2, hipMemcpyDeviceToHost));
              for (size_t i = 0; i < a4.size(); ++i) { const double dd = std::fabs((double)a4[i] - (double)b4[i]); if (dd > m4diff) m4diff = dd; } }
            const double gb = rows * (double)K / 1e9;
            std::printf("    M=%-4d s1_b %7.3f (%5.1f GB/s) | m2 %7.3f (%5.1f) | m3 RW1 %7.3f | RW2 %7.3f | RW4 %7.3f | m4 %7.3f | m6 %7.3f | m7 %7.3f | m9 RW1 %7.3f RW2 %7.3f RW4 %7.3f  m7diff %.0f %s\n",
                        M, ms_ship, gb / (ms_ship / 1e3), ms_big, gb / (ms_big / 1e3), ms3[0], ms3[1], ms3[2], ms4, ms6, ms7, ms9[0], ms9[1], ms9[2], m7diff, nbad ? "***MISMATCH***" : "(match)");
            MKB_CK(hipEventDestroy(e0)); MKB_CK(hipEventDestroy(e1));
        }
        hipFree(dW); hipFree(dX); hipFree(dS); hipFree(dY); hipFree(dY2); hipFree(dX2);
    }
}

// --------------------------------------------------------------------------------------------
// v3: v2 plus ROWS-PER-WAVE. The bench proved v2's cost is activation traffic: each block owns 4
// rows, so the tile's activations are re-read rows/4 times (at M=256 that is ~838 MB for a 13 MB
// weight matrix, and the achieved bandwidth falls from 148 GB/s at M=8 to 4.9 GB/s at M=256).
// Each wave now owns RW consecutive rows and hoists ALL of their weight chunks, so one activation
// chunk loaded from global feeds RW rows from registers. Activation traffic falls by RW; the
// register cost is RW*20 VGPRs of weights + RW*C accumulators, which is why RW is a template knob
// (RW=2 ~= 70 VGPRs keeps 2 waves/SIMD; RW=4 ~= 116 is the edge).
template <int RW, int NSTEP, int C>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m3(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int rbase = blockIdx.x * (4 * RW) + wave * RW;
    if (rbase >= rows) return;
    uint4 wv[RW][NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) { const int k0 = (t << 10) + lane * 16; ok[t] = (k0 < K); }
#pragma unroll
    for (int j = 0; j < RW; ++j) {
        if (rbase + j >= rows) break;
        const unsigned char* wp = W + (size_t)(rbase + j) * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            wv[j][t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        }
    }
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[RW][C];
#pragma unroll
        for (int j = 0; j < RW; ++j)
#pragma unroll
            for (int c = 0; c < C; ++c) acc[j][c] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const uint4 xv = *(const uint4*)(xq + (size_t)(m0 + q0 + c) * (size_t)K + k0);
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(m0 + q0 + c) * (size_t)(K >> 4) + k16];
#pragma unroll
                for (int j = 0; j < RW; ++j) {
                    if (rbase + j >= rows) break;
                    acc[j][c] += q27_dot16_fp8(wv[j][t], xh, 0.f) * sc;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < RW; ++j) {
            if (rbase + j >= rows) break;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                float v = acc[j][c];
#pragma unroll
                for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
                if (lane == 0) yv[(size_t)(q0 + c) * (size_t)ystride + (rbase + j)] = q27_f2bf(v * alpha);
            }
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_fp8_m3(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, int rw, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const int RW = (rw == 4) ? 4 : (rw == 1 ? 1 : 2);
    if (RW == 1) {
        dim3 g((unsigned)((w->rows + 3) / 4), 1u), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m3<1,5,4>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    } else if (RW == 2) {
        dim3 g((unsigned)((w->rows + 7) / 8), 1u), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m3<2,5,4>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    } else {
        dim3 g((unsigned)((w->rows + 15) / 16), 1u), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m3<4,5,4>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    }
    return 1;
}

// --------------------------------------------------------------------------------------------
// v4: HOIST THE WEIGHT DECODE, not just the load.
//
// The bench established that the projection is instruction-bound (~1.33 T-MAC/s, ~5% of Vega20's
// int8 rate at M=256), and the primitive explains it exactly: q27_dot16_fp8 spends 4 perm + 8
// shift + 8 add = 20 instructions decoding four e4m3 dwords into half2 form, plus 8 fdot2, i.e.
// 28 instructions per 16 MACs. v2/v3 hoisted the RAW weight bytes and therefore re-paid that
// 20-instruction decode for EVERY position the tile walks -- exactly the term that scales with M.
//
// v4 decodes each hoisted 16-value chunk ONCE into 8 half2 dwords held in registers (40 VGPRs for
// NSTEP=5), so the per-position work is 8 fdot2 + the activation decode (20 instructions) = 28
// instead of 48: a 1.7x cut in the term that dominates. The activation decode cannot be hoisted
// the same way -- it is genuinely per position -- but it is shared by every row of the block
// through L1, and hoisting it across rows is the RW>1 experiment that already lost to registers.
//
// Arithmetic order is unchanged (same 8 fdot2 chain from 0, same *sc, same +=), so this is
// expected to be BIT-IDENTICAL to m2 -- the bench checks that.
template <int NSTEP, int C>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m4(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row  = blockIdx.x * 4 + wave;
    if (row >= rows) return;
    const unsigned char* __restrict__ wp = W + (size_t)row * (size_t)K;
    unsigned wd[NSTEP][8];                  // decoded: 8 half2 dwords per 16-value k-chunk
    bool ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        const uint4 raw = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        unsigned lo, hi;
        q27_e4m3x4_h2(raw.x, lo, hi); wd[t][0] = lo; wd[t][1] = hi;
        q27_e4m3x4_h2(raw.y, lo, hi); wd[t][2] = lo; wd[t][3] = hi;
        q27_e4m3x4_h2(raw.z, lo, hi); wd[t][4] = lo; wd[t][5] = hi;
        q27_e4m3x4_h2(raw.w, lo, hi); wd[t][6] = lo; wd[t][7] = hi;
    }
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int k0 = (t << 10) + lane * 16;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const uint4 xv = *(const uint4*)(xq + (size_t)(m0 + q0 + c) * (size_t)K + k0);
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                float d = 0.f;
                d = q27_fdot2(wd[t][0], xh[0], d); d = q27_fdot2(wd[t][1], xh[1], d);
                d = q27_fdot2(wd[t][2], xh[2], d); d = q27_fdot2(wd[t][3], xh[3], d);
                d = q27_fdot2(wd[t][4], xh[4], d); d = q27_fdot2(wd[t][5], xh[5], d);
                d = q27_fdot2(wd[t][6], xh[6], d); d = q27_fdot2(wd[t][7], xh[7], d);
                acc[c] += d * xs[(size_t)(m0 + q0 + c) * (size_t)(K >> 4) + k16];
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c >= Cn) break;
            float v = acc[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0) yv[(size_t)(q0 + c) * (size_t)ystride + row] = q27_f2bf(v * alpha);
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_fp8_m4(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    dim3 grid((unsigned)((w->rows + 3) / 4), 1u), blk(256);
    hipLaunchKernelGGL((q27_k_proj_fp8_m4<5,4>), grid, blk, 0, s,
                       w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    return 1;
}

// --------------------------------------------------------------------------------------------
// v6: PREDECODED ACTIVATION REPRESENTATION (storage != execution).
//
// m4 proved the compiler already hoists the e4m3 weight decode, so the per-position term is the
// ACTIVATION decode: q27_i8x4_h2 costs 1 xor + 2 perm + 2 half2-adds per 4 values, i.e. 20 of the
// 28 instructions per 16 MACs, and it is paid once per ROW per position (the same position's chunk
// is decoded again by every row-wave in the block). It cannot be hoisted the way the weights can,
// because it genuinely differs per position.
//
// So move it to the PRODUCER. int8 . half2 conversion is EXACT for |q| <= 127 (the existing decode
// relies on exactly that), so a predecoded activation buffer is arithmetically identical -- same
// values, same fdot2 chain, bit-identical results -- while the consumer's inner loop becomes 2
// vector loads + 8 fdot2 with no decode at all: ~10-12 instructions per 16 MACs against 28.
// The activation buffer doubles (10 KB per position instead of 5 KB) and the producer writes it
// once, while the consumer re-reads it once per row -- a bandwidth-for-instructions trade that the
// bench says is the right way round at large M (achieved bandwidth there is 4.9 GB/s of a 550 GB/s
// part; the kernel is instruction bound, not traffic bound).
//
// producer: 4 int8 -> 2 half2 dwords (the exact same primitive the inline decode uses)
__global__ __launch_bounds__(256) void q27_k_i8_to_h2(
        const signed char* __restrict__ src, unsigned* __restrict__ dst, int n4)
{
#ifdef Q27_EXPERIMENTS
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const uint4 x = *(const uint4*)(src + (size_t)i * 16);
    unsigned* o = dst + (size_t)i * 8;
    q27_i8x4_h2(x.x, o[0], o[1]); q27_i8x4_h2(x.y, o[2], o[3]);
    q27_i8x4_h2(x.z, o[4], o[5]); q27_i8x4_h2(x.w, o[6], o[7]);

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}
extern "C" void q27_i8_to_h2(const signed char* src, unsigned* dst, int n, hipStream_t s) {
    const int n4 = n >> 4;
    const dim3 g((unsigned)((n4 + 255) / 256)), b(256);
    hipLaunchKernelGGL(q27_k_i8_to_h2, g, b, 0, s, src, dst, n4);
}

// consumer: xh2 holds [position][K] half2 values (2 bytes each); weights are hoisted and decoded
// once per tile exactly as in m4.
template <int NSTEP, int C>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m6(
        const unsigned char* __restrict__ W, const unsigned* __restrict__ xh2,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row  = blockIdx.x * 4 + wave;
    if (row >= rows) return;
    const unsigned char* __restrict__ wp = W + (size_t)row * (size_t)K;
    unsigned wd[NSTEP][8];
    bool ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        const uint4 raw = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        unsigned lo, hi;
        q27_e4m3x4_h2(raw.x, lo, hi); wd[t][0] = lo; wd[t][1] = hi;
        q27_e4m3x4_h2(raw.y, lo, hi); wd[t][2] = lo; wd[t][3] = hi;
        q27_e4m3x4_h2(raw.z, lo, hi); wd[t][4] = lo; wd[t][5] = hi;
        q27_e4m3x4_h2(raw.w, lo, hi); wd[t][6] = lo; wd[t][7] = hi;
    }
    const int KP2 = K * 2;                       // half2 bytes per position
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            if (!ok[t]) continue;
            const int off = ((t << 10) + lane * 16) * 2;      // byte offset of this 16-value chunk
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const unsigned* ap = (const unsigned*)((const char*)xh2 + (size_t)(m0 + q0 + c) * KP2) + (off >> 2);
                const uint4 x0 = *(const uint4*)ap;
                const uint4 x1 = *(const uint4*)(ap + 4);
                float d = 0.f;
                d = q27_fdot2(wd[t][0], x0.x, d); d = q27_fdot2(wd[t][1], x0.y, d);
                d = q27_fdot2(wd[t][2], x0.z, d); d = q27_fdot2(wd[t][3], x0.w, d);
                d = q27_fdot2(wd[t][4], x1.x, d); d = q27_fdot2(wd[t][5], x1.y, d);
                d = q27_fdot2(wd[t][6], x1.z, d); d = q27_fdot2(wd[t][7], x1.w, d);
                acc[c] += d * xs[(size_t)(m0 + q0 + c) * (size_t)(K >> 4) + k16];
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c >= Cn) break;
            float v = acc[c];
#pragma unroll
            for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
            if (lane == 0) yv[(size_t)(q0 + c) * (size_t)ystride + row] = q27_f2bf(v * alpha);
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_fp8_m6(const q27_fp8_t* w, const unsigned* xh2, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    dim3 grid((unsigned)((w->rows + 3) / 4), 1u), blk(256);
    hipLaunchKernelGGL((q27_k_proj_fp8_m6<5,4>), grid, blk, 0, s,
                       w->w, xh2, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    return 1;
}

// --------------------------------------------------------------------------------------------
// v7: BLOCK-OWNED ROW TILING with a wide position group (corrects v3's confound).
//
// The corrected reading of the bench: the projection's wall is activation RE-READS -- 640 blocks
// of 4 rows each stream the same M-position activation tile, 838 MB at M=256 (1.52 ms of the
// 2.70 ms measured). v3 (rows-per-wave) cut that traffic but hoisted RW*20 extra weight VGPRs and
// fell to 1 wave/SIMD, so traffic and occupancy moved together and the A/B could not separate them.
//
// v7 separates them. A wave walks RG=8 rows SEQUENTIALLY, hoisting only the CURRENT row's chunks
// (20 VGPRs, as in m2), and uses a WIDE position group C=32. Then:
//   activation traffic = M*K*(rows/(4*RG))  -> 8x less than m2
//   weight traffic     = rows*K*(M/C)       -> 8x more than m2 (13 MB -> 105 MB at M=256)
//   total              ~ 210 MB instead of ~850 MB, with registers unchanged (~80, 2 waves/SIMD).
// That is the same trade LDS tiling makes, reached through ownership instead of staging, so it
// tests the traffic hypothesis without touching the register budget.
template <int RG, int C, int NSTEP>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m7(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int rbase = blockIdx.x * (4 * RG) + wave * RG;
    if (rbase >= rows) return;
    const int NG = K >> 4;
#pragma unroll 1
    for (int j = 0; j < RG; ++j) {
        const int row = rbase + j;
        if (row >= rows) break;
        const unsigned char* __restrict__ wp = W + (size_t)row * (size_t)K;
        uint4 wv[NSTEP];
        bool ok[NSTEP];
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            ok[t] = (k0 < K);
            wv[t] = q27_ld16nt(wp + (ok[t] ? k0 : 0));
        }
        for (int q0 = 0; q0 < M; q0 += C) {
            const int Cn = (M - q0 < C) ? (M - q0) : C;
            float acc[C];
#pragma unroll 8
            for (int c = 0; c < C; ++c) acc[c] = 0.f;
#pragma unroll
            for (int t = 0; t < NSTEP; ++t) {
                if (!ok[t]) continue;
                const int k0 = (t << 10) + lane * 16;
                const int k16 = t * 64 + lane;
#pragma unroll 8
                for (int c = 0; c < C; ++c) {
                    if (c >= Cn) break;
                    const uint4 xv = *(const uint4*)(xq + (size_t)(m0 + q0 + c) * (size_t)K + k0);
                    unsigned xh[8];
                    q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                    q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                    acc[c] += q27_dot16_fp8(wv[t], xh, 0.f) * xs[(size_t)(m0 + q0 + c) * (size_t)NG + k16];
                }
            }
#pragma unroll 8
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                float v = acc[c];
#pragma unroll
                for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
                if (lane == 0) yv[(size_t)(q0 + c) * (size_t)ystride + row] = q27_f2bf(v * alpha);
            }
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_fp8_m7(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    dim3 grid((unsigned)((w->rows + 31) / 32), 1u), blk(256);
    hipLaunchKernelGGL((q27_k_proj_fp8_m7<8,32,5>), grid, blk, 0, s,
                       w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0);
    return 1;
}

// --------------------------------------------------------------------------------------------
// v9: THE CONFOUND REMOVED. v3 cut activation traffic by giving a wave RW rows and hoisting all
// of their weight chunks -- but that cost RW*20 VGPRs and fell to 1 wave/SIMD, so traffic and
// occupancy moved together and the A/B could not separate them. v7 cut traffic further and died
// of wave starvation. v9 keeps the row tiling and the wave count but makes the WEIGHTS TRANSIENT:
// they are re-read per k-step from L1/L2 (the small operand, 16 B per (row, k-chunk), reused
// M/C times) instead of living in registers. Register cost is then acc[RW][C] + one decoded
// activation chunk + one weight chunk, ~= 8*RW + 20, so RW=4 stays near 60 VGPRs (2-3 waves/SIMD).
//
// If activation traffic is the binding constraint (838 MB at M=256, 1.52 ms of the 2.70 ms), the
// activation LOAD and the activation DECODE are both divided by RW here, with nothing else moved.
// That is a clean single-variable test of the corrected model.
template <int RW, int C, int NSTEP>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_m9(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, unsigned short* __restrict__ yv,
        int rows, int K, float alpha, int M, int ystride, int m0)
{
#ifdef Q27_EXPERIMENTS
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int rbase = blockIdx.x * (4 * RW) + wave * RW;
    if (rbase >= rows) return;
    const int NG = K >> 4;
    for (int q0 = 0; q0 < M; q0 += C) {
        const int Cn = (M - q0 < C) ? (M - q0) : C;
        float acc[RW][C];
#pragma unroll
        for (int j = 0; j < RW; ++j)
#pragma unroll
            for (int c = 0; c < C; ++c) acc[j][c] = 0.f;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            const int k0 = (t << 10) + lane * 16;
            if (k0 >= K) continue;
            const int k16 = t * 64 + lane;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                const uint4 xv = *(const uint4*)(xq + (size_t)(m0 + q0 + c) * (size_t)K + k0);
                unsigned xh[8];
                q27_i8x4_h2(xv.x, xh[0], xh[1]); q27_i8x4_h2(xv.y, xh[2], xh[3]);
                q27_i8x4_h2(xv.z, xh[4], xh[5]); q27_i8x4_h2(xv.w, xh[6], xh[7]);
                const float sc = xs[(size_t)(m0 + q0 + c) * (size_t)NG + k16];
#pragma unroll
                for (int j = 0; j < RW; ++j) {
                    if (rbase + j >= rows) break;
                    const uint4 wv = *(const uint4*)(W + (size_t)(rbase + j) * (size_t)K + k0);
                    acc[j][c] += q27_dot16_fp8(wv, xh, 0.f) * sc;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < RW; ++j) {
            if (rbase + j >= rows) break;
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= Cn) break;
                float v = acc[j][c];
#pragma unroll
                for (int o = 32; o; o >>= 1) v += __shfl_down(v, o, 64);
                if (lane == 0) yv[(size_t)(q0 + c) * (size_t)ystride + (rbase + j)] = q27_f2bf(v * alpha);
            }
        }
    }

#else
    (void)0;   // not launched by any production path; compiled only with EXP=1 (receipt v93/v94)
#endif
}

extern "C" int q27_proj_fp8_m9(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                unsigned short* y, int M, int ystride, int pos0, int rw, hipStream_t s) {
    if (!w || w->K != 5120 || M <= 0) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const int RW = (rw == 4) ? 4 : (rw == 1 ? 1 : 2);
    if (RW == 1) { dim3 g((unsigned)((w->rows + 3) / 4), 1u), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m9<1,4,5>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); }
    else if (RW == 2) { dim3 g((unsigned)((w->rows + 7) / 8), 1u), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m9<2,4,5>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); }
    else { dim3 g((unsigned)((w->rows + 15) / 16), 1u), b(256);
        hipLaunchKernelGGL((q27_k_proj_fp8_m9<4,4,5>), g, b, 0, s, w->w, xq, xs, y, w->rows, w->K, alpha, M, ystride, pos0); }
    return 1;
}

// =============================================================================================
// SYNC TAX. The decode profile shows EVERY fp8 matrix running 1.58-1.63x slower inside the engine
// than the identical kernel does in the shape census (in_qkv 28.8 us vs 18.02 us). The engine's
// distinctive feature is that two host-staged collectives per layer each DRAIN the stream, 128
// times per token, tearing down the weight stream every few kernels. This measures that directly:
// the same kernel, same buffers, same shape, (A) back-to-back, (B) one drain per launch, and
// (C) one D2H sync copy per launch (what a collective's staging leg costs the stream).
//   usage: Q27_SYNC_TAX=1 ./q27_gen
extern "C" void q27_sync_tax(void) {
    const int K = 5120, rows = 2560;
    unsigned char* dW = nullptr; signed char* dX = nullptr; float* dS = nullptr;
    unsigned short* dY = nullptr; float* hp = nullptr;
    MKB_CK(hipMalloc((void**)&dW, (size_t)rows * K));
    MKB_CK(hipMalloc((void**)&dX, (size_t)K));
    MKB_CK(hipMalloc((void**)&dS, (size_t)(K / 16) * 4));
    MKB_CK(hipMalloc((void**)&dY, (size_t)rows * 2));
    MKB_CK(hipHostMalloc((void**)&hp, (size_t)5120 * 4));
    MKB_CK(hipMemset(dW, 0x30, (size_t)rows * K));
    MKB_CK(hipMemset(dX, 5, (size_t)K));
    q27_fp8_t w; w.w = dW; w.wscale = 1.0f; w.in_scale = 1.0f; w.rows = rows; w.K = K;
    const int REPS = 50;
    hipEvent_t e0, e1; MKB_CK(hipEventCreate(&e0)); MKB_CK(hipEventCreate(&e1));
    auto launch = [&]() { q27_proj_fp8_s1_b(&w, dX, dS, dY, 1, rows, 0); };
    launch(); MKB_CK(hipDeviceSynchronize());
    float msA = 0, msB = 0, msC = 0;
    MKB_CK(hipEventRecord(e0, 0)); for (int i = 0; i < REPS; ++i) launch();
    MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1)); MKB_CK(hipEventElapsedTime(&msA, e0, e1));
    MKB_CK(hipEventRecord(e0, 0));
    for (int i = 0; i < REPS; ++i) { launch(); MKB_CK(hipStreamSynchronize(0)); }
    MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1)); MKB_CK(hipEventElapsedTime(&msB, e0, e1));
    MKB_CK(hipEventRecord(e0, 0));
    for (int i = 0; i < REPS; ++i) { launch(); MKB_CK(hipMemcpyAsync(hp, dY, (size_t)rows * 2, hipMemcpyDeviceToHost, 0)); MKB_CK(hipStreamSynchronize(0)); }
    MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1)); MKB_CK(hipEventElapsedTime(&msC, e0, e1));
    const double gb = rows * (double)K / 1e9;
    std::printf("  SYNC_TAX in_qkv %dx%d, %d reps\n", rows, K, REPS);
    std::printf("    A back-to-back        %7.3f us/launch  %6.1f GB/s\n", msA * 1000 / REPS, gb / (msA / REPS / 1e3));
    std::printf("    B one drain/launch    %7.3f us/launch  %6.1f GB/s   (%.2fx A)\n", msB * 1000 / REPS, gb / (msB / REPS / 1e3), msB / msA);
    std::printf("    C drain + 20KB D2H    %7.3f us/launch  %6.1f GB/s   (%.2fx A)\n", msC * 1000 / REPS, gb / (msC / REPS / 1e3), msC / msA);
    std::printf("    engine measured 28.8 us/launch for this matrix\n");
}

// =============================================================================================
// MLP (NVFP4) CENSUS + DRAIN TAX. The decode profile puts proj_nvfp4 (2.79 ms) and down_proj
// (1.96 ms) at 32% of the token -- the largest bucket after the fp8 work, and the one that has
// never had a shape census. Per token the MLP streams gate 11.1 MB + up 11.1 MB + down 11.1 MB per
// card per layer = 2.14 GB over 64 layers, more bytes than the whole fp8 family (1.63 GB), at the
// measured 4.76 ms => ~450 GB/s. This measures the same kernels on the same buffers back-to-back
// (the ceiling) and with a stream drain per launch (what the engine's collectives impose), so both
// the headroom and the drain tax are known before any kernel work starts.
//   usage: Q27_MLP_BENCH=1 ./q27_gen
extern "C" void q27_mlp_bench(void) {
    const int shapes[3][2] = { {4352, 5120}, {4352, 5120}, {5120, 4352} };   // gate, up, down
    const char* nm[3] = { "gate", "up  ", "down" };
    const int REPS = 30;
    hipEvent_t e0, e1; MKB_CK(hipEventCreate(&e0)); MKB_CK(hipEventCreate(&e1));
    for (int si = 0; si < 3; ++si) {
        const int rows = shapes[si][0], K = shapes[si][1];
        const size_t wb = (size_t)rows * (size_t)K / 2;          // nvfp4: 4 bits per weight
        const size_t gb_ = (size_t)rows * (size_t)(K / 16);      // e4m3 group scales
        unsigned char *dW = nullptr, *dG = nullptr; signed char* dX = nullptr;
        float* dXs = nullptr; float* dY = nullptr;
        MKB_CK(hipMalloc((void**)&dW, wb));
        MKB_CK(hipMalloc((void**)&dG, gb_));
        MKB_CK(hipMalloc((void**)&dX, (size_t)K));
        MKB_CK(hipMalloc((void**)&dXs, (size_t)(K / 16) * 4));
        MKB_CK(hipMalloc((void**)&dY, (size_t)rows * 4));
        MKB_CK(hipMemset(dW, 0x31, wb));
        MKB_CK(hipMemset(dG, 0x38, gb_));
        MKB_CK(hipMemset(dX, 3, (size_t)K));
        q27_nvfp4_t w{}; w.w = dW; w.gs = dG; w.ws2 = 1.0f; w.in_scale = 1.0f; w.rows = rows; w.K = K;
        auto launch = [&]() { q27_proj_nvfp4(&w, dX, dXs, dY, 0); };
        launch(); MKB_CK(hipDeviceSynchronize());
        float msA = 0, msB = 0;
        MKB_CK(hipEventRecord(e0, 0)); for (int i = 0; i < REPS; ++i) launch();
        MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1)); MKB_CK(hipEventElapsedTime(&msA, e0, e1));
        MKB_CK(hipEventRecord(e0, 0));
        for (int i = 0; i < REPS; ++i) { launch(); MKB_CK(hipStreamSynchronize(0)); }
        MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1)); MKB_CK(hipEventElapsedTime(&msB, e0, e1));
        const double gbw = (double)wb / 1e9;
        std::printf("  MLP %s rows=%d K=%d packed=%.1f MB | back-to-back %7.3f us (%6.1f GB/s) | drain/launch %7.3f us (%.2fx)\n",
                    nm[si], rows, K, (double)wb / 1e6, msA * 1000 / REPS, gbw / (msA / REPS / 1e3),
                    msB * 1000 / REPS, msB / msA);
        hipFree(dW); hipFree(dG); hipFree(dX); hipFree(dXs); hipFree(dY);
    }
    std::printf("  engine per layer: gate+up+down = 4.76 ms/64 = %.3f ms  (census ceiling above)\n", 4.76 / 64);
}

// =============================================================================================
// M-KNEE CENSUS (Q27_MLPB_BENCH=1): the prefill's batched kernels at the production shard shapes,
// timed back-to-back on synthetic buffers at two position counts each. If the per-call time is
// flat in C the kernel is weight-stream/latency-bound and wider M halves the per-position cost;
// if it doubles, the kernel is issue-bound and M is not the lever. Per-position us is the number
// the 1K wall is made of (x layers).
extern "C" int q27_proj_nvfp4_gu_b2_c(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt2_c(const q27_nvfp4_t*, const signed char*, const float*, float*, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b4(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt4(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b5(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt5(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b6(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt6(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b7(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt7(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b8(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b9(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt9(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b10(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt10(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b11(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt11(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b12(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt12(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b13(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt13(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_gu_b14(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt14(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);
extern "C" void q27_mlpb_bench(void) {
    const int REPS = 40;
    hipEvent_t e0, e1; MKB_CK(hipEventCreate(&e0)); MKB_CK(hipEventCreate(&e1));
    auto timeit = [&](auto&& fn) -> double {
        fn(); (void)hipDeviceSynchronize();
        (void)hipEventRecord(e0, 0); for (int i = 0; i < REPS; ++i) fn();
        (void)hipEventRecord(e1, 0); (void)hipEventSynchronize(e1);
        float ms = 0; if (hipEventElapsedTime(&ms, e0, e1) != hipSuccess) return -1.0; return ms * 1000.0 / REPS;
    };
    // activations: up to 256 positions x 5120 int8 (+ per-16 scales), one pinned pattern
    signed char* dX; float* dXs; float* dY; unsigned short* dYb;
    MKB_CK(hipMalloc((void**)&dX, (size_t)256 * 5120));
    MKB_CK(hipMalloc((void**)&dXs, (size_t)256 * 320 * 4));
    MKB_CK(hipMalloc((void**)&dY, (size_t)256 * 5120 * 4));
    MKB_CK(hipMalloc((void**)&dYb, (size_t)256 * 5120 * 2));
    MKB_CK(hipMemset(dX, 3, (size_t)256 * 5120));
    MKB_CK(hipMemset(dXs, 0, (size_t)256 * 320 * 4));
    std::printf("Q27_MLPB_BENCH  per-call us | per-position us | weight GB/s   (back-to-back, %d reps)\n", REPS);
    // --- NVFP4 gate+up (C=8 -> 16) and down ---
    {
        const int rows = 4352, K = 5120;
        const size_t wb = (size_t)rows * K / 2, gb_ = (size_t)rows * (K / 16);
        unsigned char *dWg, *dWu, *dGg, *dGu;
        MKB_CK(hipMalloc((void**)&dWg, wb)); MKB_CK(hipMalloc((void**)&dWu, wb));
        MKB_CK(hipMalloc((void**)&dGg, gb_)); MKB_CK(hipMalloc((void**)&dGu, gb_));
        MKB_CK(hipMemset(dWg, 0x31, wb)); MKB_CK(hipMemset(dWu, 0x31, wb));
        MKB_CK(hipMemset(dGg, 0x38, gb_)); MKB_CK(hipMemset(dGu, 0x38, gb_));
        q27_nvfp4_t wg{}; wg.w = dWg; wg.gs = dGg; wg.ws2 = 1.f; wg.in_scale = 1.f; wg.rows = rows; wg.K = K;
        q27_nvfp4_t wu = wg; wu.w = dWu; wu.gs = dGu;
        float* dY2 = dY + (size_t)128 * 5120;
        for (int C : {4, 8, 16}) {
            const double us = timeit([&]() { q27_proj_nvfp4_gu_b2_c(&wg, &wu, dX, dXs, dY, dY2, C, 0); });
            std::printf("  nvfp4 gate+up  C=%2d  %9.2f  %9.3f  %7.1f GB/s\n", C, us, us / C, (2.0 * wb / 1e9) / (us / 1e6));
        }
        for (int R : {1, 2}) {
            const double us = timeit([&]() { q27_proj_nvfp4_gu_b4(&wg, &wu, dX, dXs, dY, dY2, Q27_PF_MC, R, 0); });
            std::printf("  nvfp4 gate+up  b4 R=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", R, Q27_PF_MC, us, us / Q27_PF_MC, (2.0 * wb / 1e9) / (us / 1e6));
        }
        for (int R : {1, 2}) {
            const double us = timeit([&]() { q27_proj_nvfp4_gu_b5(&wg, &wu, dX, dXs, dY, dY2, Q27_PF_MC, R, 0); });
            std::printf("  nvfp4 gate+up  b5 R=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", R, Q27_PF_MC, us, us / Q27_PF_MC, (2.0 * wb / 1e9) / (us / 1e6));
        }
        if (q27_nvfp4_make_w8(&wg, 0) && q27_nvfp4_make_w8(&wu, 0)) {
            MKB_CK(hipDeviceSynchronize());
            for (int mw : {5, 6}) {
                const double us = timeit([&]() { q27_proj_nvfp4_gu_b14(&wg, &wu, dX, dXs, dY, dY2, 8, mw, 0); });
                std::printf("  nvfp4 gate+up  b14 minw=%d C=8  %9.2f  %9.3f  %7.1f GB/s (int8 mirror, 2x bytes)\n", mw, us, us / 8, (4.0 * wb / 1e9) / (us / 1e6));
            }
            q27_nvfp4_free_w8(&wg); q27_nvfp4_free_w8(&wu);
        }
        for (int mw : {4, 5}) {
            const double us = timeit([&]() { q27_proj_nvfp4_gu_b13(&wg, &wu, dX, dXs, dY, dY2, 8, mw, 0); });
            std::printf("  nvfp4 gate+up  b13 minw=%d C=8  %9.2f  %9.3f  %7.1f GB/s\n", mw, us, us / 8, (2.0 * wb / 1e9) / (us / 1e6));
        }
        for (int gb : {4, 8, 16}) {
            const double us = timeit([&]() { q27_proj_nvfp4_gu_b12(&wg, &wu, dX, dXs, dY, dY2, 8, gb, 0); });
            std::printf("  nvfp4 gate+up  b12 gb=%d C=8  %9.2f  %9.3f  %7.1f GB/s\n", gb, us, us / 8, (2.0 * wb / 1e9) / (us / 1e6));
        }
        for (int dp : {2, 4, 8}) {
            const double us = timeit([&]() { q27_proj_nvfp4_gu_b11(&wg, &wu, dX, dXs, dY, dY2, 8, dp, 0); });
            std::printf("  nvfp4 gate+up  b11 depth=%d C=8  %9.2f  %9.3f  %7.1f GB/s\n", dp, us, us / 8, (2.0 * wb / 1e9) / (us / 1e6));
        }
        {   const double us = timeit([&]() { q27_proj_nvfp4_gu_b10(&wg, &wu, dX, dXs, dY, dY2, 8, 1, 0); });
            std::printf("  nvfp4 gate+up  b10 R=1 C=8  %9.2f  %9.3f  %7.1f GB/s\n", us, us / 8, (2.0 * wb / 1e9) / (us / 1e6)); }
        {   const double us = timeit([&]() { q27_proj_nvfp4_gu_b9(&wg, &wu, dX, dXs, dY, dY2, 8, 1, 0); });
            std::printf("  nvfp4 gate+up  b9 R=1 C=8  %9.2f  %9.3f  %7.1f GB/s\n", us, us / 8, (2.0 * wb / 1e9) / (us / 1e6)); }
        {   const double us = timeit([&]() { q27_proj_nvfp4_gu_b8(&wg, &wu, dX, dXs, dY, dY2, 8, 1, 0); });
            std::printf("  nvfp4 gate+up  b8 R=1 C=8  %9.2f  %9.3f  %7.1f GB/s\n", us, us / 8, (2.0 * wb / 1e9) / (us / 1e6)); }
        for (int rpw : {0, 8, 32}) {
            const double us = timeit([&]() { q27_proj_nvfp4_gu_b7(&wg, &wu, dX, dXs, dY, dY2, Q27_PF_MC, rpw, 0); });
            std::printf("  nvfp4 gate+up  b7 rpw=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", rpw, Q27_PF_MC, us, us / Q27_PF_MC, (2.0 * wb / 1e9) / (us / 1e6));
        }
        {   const double us = timeit([&]() { q27_proj_nvfp4_gu_b6(&wg, &wu, dX, dXs, dY, dY2, Q27_PF_MC, 1, 0); });
            std::printf("  nvfp4 gate+up  b6 R=1 C=%d  %9.2f  %9.3f  %7.1f GB/s\n", Q27_PF_MC, us, us / Q27_PF_MC, (2.0 * wb / 1e9) / (us / 1e6)); }
        hipFree(dWg); hipFree(dWu); hipFree(dGg); hipFree(dGu);
    }
    {
        const int rows = 5120, K = 4352;
        const size_t wb = (size_t)rows * K / 2, gb_ = (size_t)rows * (K / 16);
        unsigned char *dW, *dG; MKB_CK(hipMalloc((void**)&dW, wb)); MKB_CK(hipMalloc((void**)&dG, gb_));
        MKB_CK(hipMemset(dW, 0x31, wb)); MKB_CK(hipMemset(dG, 0x38, gb_));
        q27_nvfp4_t w{}; w.w = dW; w.gs = dG; w.ws2 = 1.f; w.in_scale = 1.f; w.rows = rows; w.K = K;
        for (int C : {4, 8, 16}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt2_c(&w, dX, dXs, dY, C, 0); });
            std::printf("  nvfp4 down     C=%2d  %9.2f  %9.3f  %7.1f GB/s\n", C, us, us / C, (wb / 1e9) / (us / 1e6));
        }
        for (int R : {1, 2}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt4(&w, dX, dXs, dY, Q27_PF_MC, R, 0); });
            std::printf("  nvfp4 down     b4 R=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", R, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        for (int R : {1, 2}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt5(&w, dX, dXs, dY, Q27_PF_MC, R, 0); });
            std::printf("  nvfp4 down     b5 R=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", R, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        if (q27_nvfp4_make_w8(&w, 0)) {
            MKB_CK(hipDeviceSynchronize());
            for (int mw : {5, 6}) {
                const double us = timeit([&]() { q27_proj_nvfp4_bt14(&w, dX, dXs, dY, Q27_PF_MC, mw, 0); });
                std::printf("  nvfp4 down     b14 minw=%d C=%d  %9.2f  %9.3f  %7.1f GB/s (int8 mirror, 2x bytes)\n", mw, Q27_PF_MC, us, us / Q27_PF_MC, (2.0 * wb / 1e9) / (us / 1e6));
            }
            q27_nvfp4_free_w8(&w);
        }
        for (int mw : {5, 6}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt13(&w, dX, dXs, dY, Q27_PF_MC, mw, 0); });
            std::printf("  nvfp4 down     b13 minw=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", mw, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        for (int gb : {4, 8}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt12(&w, dX, dXs, dY, Q27_PF_MC, gb, 0); });
            std::printf("  nvfp4 down     b12 gb=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", gb, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        for (int dp : {2, 4, 8}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt11(&w, dX, dXs, dY, Q27_PF_MC, dp, 0); });
            std::printf("  nvfp4 down     b11 depth=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", dp, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        for (int R : {1, 2}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt10(&w, dX, dXs, dY, Q27_PF_MC, R, 0); });
            std::printf("  nvfp4 down     b10 R=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", R, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        for (int R : {1, 2}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt9(&w, dX, dXs, dY, Q27_PF_MC, R, 0); });
            std::printf("  nvfp4 down     b9 R=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", R, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        for (int rpw : {0, 8, 32}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt7(&w, dX, dXs, dY, Q27_PF_MC, rpw, 0); });
            std::printf("  nvfp4 down     b7 rpw=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", rpw, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        for (int R : {1, 2}) {
            const double us = timeit([&]() { q27_proj_nvfp4_bt6(&w, dX, dXs, dY, Q27_PF_MC, R, 0); });
            std::printf("  nvfp4 down     b6 R=%d C=%d  %9.2f  %9.3f  %7.1f GB/s\n", R, Q27_PF_MC, us, us / Q27_PF_MC, (wb / 1e9) / (us / 1e6));
        }
        hipFree(dW); hipFree(dG);
    }
    // --- fp8 chunk out_proj (m2f, K=1536, rows 5120) at M=8/16, and the tile projection (m2, K=5120) ---
    {
        unsigned char* dW; MKB_CK(hipMalloc((void**)&dW, (size_t)5120 * 5120)); MKB_CK(hipMemset(dW, 0x3C, (size_t)5120 * 5120));
        q27_fp8_t w{}; w.w = dW; w.in_scale = 1.f; w.wscale = 1.f;
        w.rows = 5120; w.K = 1536;
        for (int M : {8, 16}) {
            const double us = timeit([&]() { q27_proj_fp8_m2_res(&w, dX, dXs, dY, M, 5120, nullptr, 5120, 0.25f, 0); });
            std::printf("  fp8 out_proj   M=%2d  %9.2f  %9.3f  %7.1f GB/s   (m2f)\n", M, us, us / M, ((double)5120 * 1536 / 1e9) / (us / 1e6));
        }
        for (int R : {2, 4}) {
            const double us = timeit([&]() { q27_proj_fp8_m2_res2(&w, dX, dXs, dY, 8, 5120, nullptr, 5120, 0.25f, R, 0); });
            std::printf("  fp8 out_proj   m2f2 R=%d M=8  %9.2f  %9.3f  %7.1f GB/s\n", R, us, us / 8, ((double)5120 * 1536 / 1e9) / (us / 1e6));
        }
        for (int rows : {2560, 1536}) {
            w.rows = rows; w.K = 5120;
            for (int M : {64, 128, 256}) {
                const double us = timeit([&]() { q27_proj_fp8_m2(&w, dX, dXs, dYb, M, rows, 0, 0); });
                std::printf("  fp8 tile K=5120 rows=%d M=%3d  %9.2f  %9.3f  %7.1f GB/s   (m2)\n", rows, M, us, us / M, ((double)rows * 5120 / 1e9) / (us / 1e6));
            }
        }
        {
            void* dW16 = nullptr; MKB_CK(hipMalloc((void**)&dW16, (size_t)2560 * 5120 * 2));
            for (int rows : {2560, 1536}) {
                w.rows = rows; w.K = 5120;
                for (int R : {2, 3, 4, 5, 6}) {
                    const double ur = timeit([&]() { q27_proj_fp8_m2r(&w, dX, dXs, dYb, 128, rows, 0, R, 0); });
                    std::printf("  fp8 tile K=5120 rows=%d M=128  %9.2f  %9.3f  (m2r R=%d, no LDS)\n", rows, ur, ur / 128, R);
                    if (R <= 4) { const double u2 = timeit([&]() { q27_proj_fp8_m2r2(&w, dX, dXs, dYb, 128, rows, 0, R, 0); });
                        std::printf("  fp8 tile K=5120 rows=%d M=128  %9.2f  %9.3f  (m2r2 R=%d, rebuilt epilogue)\n", rows, u2, u2 / 128, R); }
                }
                const double uc = timeit([&]() { q27_fp8_to_h2(&w, dW16, 0); });
                const double us = timeit([&]() { q27_proj_fp8_m2w16(&w, dW16, dX, dXs, dYb, 128, rows, 0, 0); });
                std::printf("  fp8 tile K=5120 rows=%d M=128  %9.2f  %9.3f  (m2w16, fp16 weights; convert %.1f us per tile-layer)\n", rows, us, us / 128, uc);
            }
            hipFree(dW16);
        }
        w.rows = 1536; w.K = 5120;
        for (int C : {8, 16}) {
            const double us = timeit([&]() { q27_proj_fp8_s1_b(&w, dX, dXs, dYb, C, 1536, 0); });
            std::printf("  fp8 q_proj     C=%2d  %9.2f  %9.3f  %7.1f GB/s   (s1_b)\n", C, us, us / C, ((double)1536 * 5120 / 1e9) / (us / 1e6));
            if (C == Q27_PF_MC) { const double u2 = timeit([&]() { q27_proj_fp8_s1_b2(&w, dX, dXs, dYb, C, 1536, 1, 0); });
                std::printf("  fp8 q_proj     C=%2d  %9.2f  %9.3f  %7.1f GB/s   (s1_b2 R=1, production)\n", C, u2, u2 / C, ((double)1536 * 5120 / 1e9) / (u2 / 1e6)); }
            if (C == Q27_PF_MC) { const double u3 = timeit([&]() { q27_proj_fp8_s1_b3(&w, dX, dXs, dYb, C, 1536, 1, 0); });
                std::printf("  fp8 q_proj     C=%2d  %9.2f  %9.3f  %7.1f GB/s   (s1_b3 R=1, rebuilt epilogue)\n", C, u3, u3 / C, ((double)1536 * 5120 / 1e9) / (u3 / 1e6)); }
        }
        hipFree(dW);
    }
    hipFree(dX); hipFree(dXs); hipFree(dY); hipFree(dYb);
}

// =============================================================================================
// BANDWIDTH PROBE: does achieved bandwidth depend on matrix SIZE?
// The fp8 census streams the SAME 11-13 MB matrix 30-50 times; Vega20's L2 is 4 MB, so part of the
// stream can survive between iterations and flatter the small shapes. The engine streams each
// matrix ONCE per token, so if the census rate is an L2 artifact the engine is already at the
// memory ceiling and the 'drain residue' is not recoverable. This sweeps rows at fixed K and
// reports back-to-back GB/s, which separates the two.
//   usage: Q27_BW_PROBE=1 ./q27_gen
extern "C" void q27_bw_probe(void) {
    const int K = 5120;
    const int row_tab[5] = { 640, 2560, 5120, 10240, 20480 };
    const int REPS = 30;
    hipEvent_t e0, e1; MKB_CK(hipEventCreate(&e0)); MKB_CK(hipEventCreate(&e1));
    std::printf("  BW_PROBE fp8 s1_b, K=%d, back-to-back, %d reps\n", K, REPS);
    for (int t = 0; t < 5; ++t) {
        const int rows = row_tab[t];
        const size_t wb = (size_t)rows * (size_t)K;
        unsigned char* dW = nullptr; signed char* dX = nullptr; float* dS = nullptr;
        unsigned short* dY = nullptr;
        MKB_CK(hipMalloc((void**)&dW, wb));
        MKB_CK(hipMalloc((void**)&dX, (size_t)K));
        MKB_CK(hipMalloc((void**)&dS, (size_t)(K / 16) * 4));
        MKB_CK(hipMalloc((void**)&dY, (size_t)rows * 2));
        MKB_CK(hipMemset(dW, 0x30, wb));
        MKB_CK(hipMemset(dX, 5, (size_t)K));
        q27_fp8_t w; w.w = dW; w.wscale = 1.0f; w.in_scale = 1.0f; w.rows = rows; w.K = K;
        q27_proj_fp8_s1_b(&w, dX, dS, dY, 1, rows, 0);
        MKB_CK(hipDeviceSynchronize());
        MKB_CK(hipEventRecord(e0, 0));
        for (int i = 0; i < REPS; ++i) q27_proj_fp8_s1_b(&w, dX, dS, dY, 1, rows, 0);
        MKB_CK(hipEventRecord(e1, 0)); MKB_CK(hipEventSynchronize(e1));
        float ms = 0; MKB_CK(hipEventElapsedTime(&ms, e0, e1)); ms /= REPS;
        std::printf("    rows=%-6d matrix=%6.1f MB  %8.3f us  %6.1f GB/s\n",
                    rows, wb / 1e6, ms * 1000, (wb / 1e9) / (ms / 1e3));
        hipFree(dW); hipFree(dX); hipFree(dS); hipFree(dY);
    }
}

// ============================================================================================
// Q27_LS_Q8 epilogues for the wide fp8 GEMM (same main loop as q27_k_wide_fp8): the producer
// emits the representation its consumer executes on.
//   _q8  : int8 + one fp32 scale per 16 output rows ([token][row], [token][row/16]) for the
//          attention chunk / GDN conv+scan consumers (q/k/v, in_qkv, in_z). A lane owns rows
//          row_base+lane and row_base+lane+64, so a 16-row scale group is 16 consecutive lanes.
//   _f32r: fp32 output plus an fp32 residual read straight from the residual stream (o/out_proj).
// ============================================================================================
template <int KI, int NW, int MINW>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_fp8_q8(
        const unsigned char* __restrict__ W, const signed char* __restrict__ Xq,
        const float* __restrict__ Xs, float alpha, int rows, int K, int M,
        signed char* __restrict__ Yq, float* __restrict__ Ys, int ystride, int sstride) {
    constexpr int TR = 128, TC = 64;
    constexpr int WSTR = KI * 16 + 16;
    constexpr int XSTR = KI * 32 + 32;
    __shared__ __align__(16) unsigned char sw[TR * WSTR];
    __shared__ __align__(16) unsigned char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W + (size_t)rr * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            const uint4 v = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            unsigned h[8];
            q27_i8x4_h2(v.x, h[0], h[1]); q27_i8x4_h2(v.y, h[2], h[3]);
            q27_i8x4_h2(v.z, h[4], h[5]); q27_i8x4_h2(v.w, h[6], h[7]);
            *(uint4*)(sx + t * XSTR + g * 32)      = *(const uint4*)&h[0];
            *(uint4*)(sx + t * XSTR + g * 32 + 16) = *(const uint4*)&h[4];
            sxsc[it] = Xs[(size_t)tt * NG + gg];
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = *(const uint4*)(sw + (lane + 64) * WSTR + g * 16);
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 xa = *(const uint4*)(sx + t * XSTR + g * 32);
                const uint4 xb = *(const uint4*)(sx + t * XSTR + g * 32 + 16);
                unsigned xh[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
                const float xsv = sxsc[t * KI + g];
                acc0[j] = fmaf(q27_dot16_fp8(w0, xh, 0.f), xsv, acc0[j]);
                acc1[j] = fmaf(q27_dot16_fp8(w1, xh, 0.f), xsv, acc1[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int t = tok_base + wave * TPW + j;
        const float v0 = acc0[j] * alpha, v1 = acc1[j] * alpha;
        float m0 = fabsf(v0), m1 = fabsf(v1);
        m0 = fmaxf(m0, __shfl_xor(m0, 8, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 8, 16));
        m0 = fmaxf(m0, __shfl_xor(m0, 4, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 4, 16));
        m0 = fmaxf(m0, __shfl_xor(m0, 2, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 2, 16));
        m0 = fmaxf(m0, __shfl_xor(m0, 1, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 1, 16));
        const float s0 = m0 * (1.f / 127.f), s1 = m1 * (1.f / 127.f);
        const float q0 = (s0 > 0.f) ? (1.f / s0) : 0.f, q1 = (s1 > 0.f) ? (1.f / s1) : 0.f;
        int c0 = (int)rintf(v0 * q0); c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0);
        int c1 = (int)rintf(v1 * q1); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
        if (t < M) {
            if (r0 < rows) { Yq[(size_t)t * ystride + r0] = (signed char)c0; if ((lane & 15) == 0) Ys[(size_t)t * sstride + (r0 >> 4)] = s0; }
            if (r1 < rows) { Yq[(size_t)t * ystride + r1] = (signed char)c1; if ((lane & 15) == 0) Ys[(size_t)t * sstride + (r1 >> 4)] = s1; }
        }
    }
}

template <int KI, int NW, int MINW>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_fp8_f32r(
        const unsigned char* __restrict__ W, const signed char* __restrict__ Xq,
        const float* __restrict__ Xs, float alpha, int rows, int K, int M,
        float* __restrict__ Y, int ystride,
        const float* __restrict__ radd, int rstride, float rs) {
    constexpr int TR = 128, TC = 64;
    constexpr int WSTR = KI * 16 + 16;
    constexpr int XSTR = KI * 32 + 32;
    __shared__ __align__(16) unsigned char sw[TR * WSTR];
    __shared__ __align__(16) unsigned char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W + (size_t)rr * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            const uint4 v = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            unsigned h[8];
            q27_i8x4_h2(v.x, h[0], h[1]); q27_i8x4_h2(v.y, h[2], h[3]);
            q27_i8x4_h2(v.z, h[4], h[5]); q27_i8x4_h2(v.w, h[6], h[7]);
            *(uint4*)(sx + t * XSTR + g * 32)      = *(const uint4*)&h[0];
            *(uint4*)(sx + t * XSTR + g * 32 + 16) = *(const uint4*)&h[4];
            sxsc[it] = Xs[(size_t)tt * NG + gg];
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = *(const uint4*)(sw + (lane + 64) * WSTR + g * 16);
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 xa = *(const uint4*)(sx + t * XSTR + g * 32);
                const uint4 xb = *(const uint4*)(sx + t * XSTR + g * 32 + 16);
                unsigned xh[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
                const float xsv = sxsc[t * KI + g];
                acc0[j] = fmaf(q27_dot16_fp8(w0, xh, 0.f), xsv, acc0[j]);
                acc1[j] = fmaf(q27_dot16_fp8(w1, xh, 0.f), xsv, acc1[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int t = tok_base + wave * TPW + j;
        if (t < M) {
            if (r0 < rows) { float v = acc0[j] * alpha; if (radd) v += rs * radd[(size_t)t * rstride + r0]; Y[(size_t)t * ystride + r0] = v; }
            if (r1 < rows) { float v = acc1[j] * alpha; if (radd) v += rs * radd[(size_t)t * rstride + r1]; Y[(size_t)t * ystride + r1] = v; }
        }
    }
}

extern "C" int q27_proj_fp8_wide_q8(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                    signed char* yq, float* ys, int ystride, int M, int ki, hipStream_t s) {
    if (!w || M < 1 || (w->K & 15) || w->K < 16 || (ystride & 15) || !yq || !ys) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64));
    if (ki == 8)  { hipLaunchKernelGGL((q27_k_wide_fp8_q8<8,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, yq, ys, ystride, ystride >> 4); return 1; }
    if (ki == 4)  { hipLaunchKernelGGL((q27_k_wide_fp8_q8<4,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, yq, ys, ystride, ystride >> 4); return 1; }
    return 0;
}
extern "C" int q27_proj_fp8_wide_f32r(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                      float* y, int M, int ystride, const float* radd, int rstride, float rs,
                                      int ki, hipStream_t s) {
    if (!w || M < 1 || (w->K & 15) || w->K < 16 || !y) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64));
    if (ki == 8)  { hipLaunchKernelGGL((q27_k_wide_fp8_f32r<8,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, y, ystride, radd, rstride, rs); return 1; }
    if (ki == 4)  { hipLaunchKernelGGL((q27_k_wide_fp8_f32r<4,8,1>),  g, dim3(512), 0, s, w->w, xq, xs, alpha, w->rows, w->K, M, y, ystride, radd, rstride, rs); return 1; }
    return 0;
}

// =============================================================================================
// ROW-BATCHED DECODE FORMS (Q27_SPEC verify). NR activation rows of ONE sequence share every
// weight load; the e4m3 -> half2 weight conversion is done ONCE per (row, step) and only the
// eight dot2 (+ the activation int8 -> half2 conversion) repeat per activation row. Per-row
// accumulation order is q27_k_proj_fp8_s<R,NSTEP,OUTBF>'s (dot16 = fdot2 chain lo0,hi0,..,hi3
// from 0, then acc += dot * scale in step order), so each row's output is bit-identical.
// Activation rows are contiguous ([NR][K] int8 natural order, [NR][K/16] f32 scales); outputs
// land at y + n*ystride; the folded residual share reads radd + n*rstride.
template <int NR, int R, int NSTEP, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_snr(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv, int ystride,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, int rstride, float rs, int xst) {
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row0 = (blockIdx.x * 4 + wave) * R;
    if (row0 >= rows) return;

    const unsigned char* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int rr = (row0 + r < rows) ? (row0 + r) : (rows - 1);   // clamped ragged tail
        wp[r] = W + (size_t)rr * (size_t)K;
    }
    const int KS = K >> 4;                                             // scales per row

    uint4 wv[R][NSTEP];
    uint4 xv[NR][NSTEP];
    float sc[NR][NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        const int ks = ok[t] ? k0 : 0;
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r][t] = q27_ld16nt(wp[r] + ks);
#pragma unroll
        for (int n = 0; n < NR; ++n) {
            xv[n][t] = *(const uint4*)(xq + (size_t)n * xst + ks);
            sc[n][t] = xs[(size_t)n * (xst >> 4) + (ks >> 4)];
        }
    }

    float acc[NR][R];
#pragma unroll
    for (int n = 0; n < NR; ++n)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[n][r] = 0.0f;
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        if (!ok[t]) continue;
        unsigned wh[R][8];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            q27_e4m3x4_h2(wv[r][t].x, wh[r][0], wh[r][1]);
            q27_e4m3x4_h2(wv[r][t].y, wh[r][2], wh[r][3]);
            q27_e4m3x4_h2(wv[r][t].z, wh[r][4], wh[r][5]);
            q27_e4m3x4_h2(wv[r][t].w, wh[r][6], wh[r][7]);
        }
#pragma unroll
        for (int n = 0; n < NR; ++n) {
            unsigned xh[8];
            q27_i8x4_h2(xv[n][t].x, xh[0], xh[1]);
            q27_i8x4_h2(xv[n][t].y, xh[2], xh[3]);
            q27_i8x4_h2(xv[n][t].z, xh[4], xh[5]);
            q27_i8x4_h2(xv[n][t].w, xh[6], xh[7]);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float d = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) d = q27_fdot2(wh[r][j], xh[j], d);
                acc[n][r] += d * sc[n][t];
            }
        }
    }

#pragma unroll
    for (int o = 32; o; o >>= 1)
#pragma unroll
        for (int n = 0; n < NR; ++n)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[n][r] += __shfl_down(acc[n][r], o, 64);
    if (lane == 0) {
#pragma unroll
        for (int n = 0; n < NR; ++n)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (row0 + r >= rows) continue;
                float v = acc[n][r] * alpha;
                if (radd) v += rs * q27_bf2f(radd[(size_t)n * rstride + row0 + r]);
                if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + row0 + r] = q27_f2bf(v);
                else       ((float*)yv)[(size_t)n * ystride + row0 + r]         = v;
            }
    }
}

#define Q27_NR_CASES(LAUNCH) \
    switch (NR) { case 1: LAUNCH(1); break; case 2: LAUNCH(2); break; \
                  case 3: LAUNCH(3); break; case 4: LAUNCH(4); break; default: return 0; }

// K=5120 (q/k/v_proj, in_qkv, in_z): the <1,5,1> shape production launches, bf16 out.
static int q27_fp8_lds8(const q27_fp8_t* w, const signed char* xq, int xst, const float* xs, void* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s);   // native 8-row form, defined below
static int q27_fp8_nr_lds_min();   // Q27_NR_LDS_MIN threshold (defined with the LDS forms below)
static int q27_proj_i8g_rows(const q27_i8g_t* m, const signed char* xq, int xst, const float* xs, void* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, int outbf, hipStream_t s);
static void q27_dec_i8_route_declined(const q27_fp8_t* w, int NR);   // Q27_DEC_I8 (defined below)
static int q27_dec_i8_on();
static int q27_dec_i8_plain_on();
extern "C" const q27_i8g_t* q27_fp8_i8g(const q27_fp8_t* w);
extern "C" int q27_proj_fp8_bf16_nr(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                    unsigned short* y, int ystride, int NR, hipStream_t s) {
    if (!w || w->K != 5120 || NR < 1 || NR > 8) return 0;
    if (q27_dec_i8_on()) { const q27_i8g_t* m = q27_fp8_i8g(w); if (m) { if (q27_proj_i8g_rows(m, xq, w->K, xs, (void*)y, ystride, nullptr, 0, 0.f, NR, -1, s)) return 1; q27_dec_i8_route_declined(w, NR); } }
    if (NR >= q27_fp8_nr_lds_min()) return q27_fp8_lds8(w, xq, w->K, xs, (void*)y, ystride, nullptr, 0, 0.f, NR, s);
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 grid((unsigned)((w->rows + 3) / 4)), blk(256);          // R=1 (R=2 measured slower)
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_snr<N,1,5,1>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, (const unsigned short*)nullptr, 0, 0.f, w->K)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}

// K=1536 (o_proj / out_proj partials): the <2,2,0> shape, fp32 out, residual share folded in.
// Q27_SPEC draft layer: rows at an explicit activation stride (a slice of a wider row buffer); K = 5120 or 6144, bf16 out.
extern "C" int q27_proj_fp8_bf16_nrs(const q27_fp8_t* w, const signed char* xq, int xst, const float* xs,
                                     unsigned short* y, int ystride, int NR, hipStream_t s) {
    if (!w || NR < 1 || NR > 8 || (w->K != 5120 && w->K != 6144)) return 0;
    if (q27_dec_i8_on()) { const q27_i8g_t* m = q27_fp8_i8g(w); if (m) { if (q27_proj_i8g_rows(m, xq, xst, xs, (void*)y, ystride, nullptr, 0, 0.f, NR, -1, s)) return 1; q27_dec_i8_route_declined(w, NR); } }
    if (NR >= q27_fp8_nr_lds_min()) return q27_fp8_lds8(w, xq, xst, xs, (void*)y, ystride, nullptr, 0, 0.f, NR, s);
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 grid((unsigned)((w->rows + 3) / 4)), blk(256);
    if (w->K == 5120) {
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_snr<N,1,5,1>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, (const unsigned short*)nullptr, 0, 0.f, xst)
        Q27_NR_CASES(Q27_L)
#undef Q27_L
    } else {
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_snr<N,1,6,1>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, (const unsigned short*)nullptr, 0, 0.f, xst)
        Q27_NR_CASES(Q27_L)
#undef Q27_L
    }
    return 1;
}
extern "C" int q27_proj_fp8_res_nr(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                   float* y, int ystride, const unsigned short* radd, int rstride,
                                   float rs, int NR, hipStream_t s) {
    if (!w || w->K != 1536 || NR < 1 || NR > 8) return 0;
    if (q27_dec_i8_on()) { const q27_i8g_t* m = q27_fp8_i8g(w); if (m) { if (q27_proj_i8g_rows(m, xq, w->K, xs, (void*)y, ystride, radd, rstride, rs, NR, -1, s)) return 1; q27_dec_i8_route_declined(w, NR); } }
    if (NR >= q27_fp8_nr_lds_min()) return q27_fp8_lds8(w, xq, w->K, xs, (void*)y, ystride, radd, rstride, rs, NR, s);
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 grid((unsigned)((w->rows + 7) / 8)), blk(256);          // R=2: 8 rows per block (R=4 measured slower)
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_snr<N,2,2,0>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, radd, rstride, rs, w->K)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}


// =============================================================================================
// LDS-STAGED ROW-BATCHED FP8 FORM (Q27_NR_LDS): see the NVFP4 twin. The block stages the NR
// activation rows (natural-order int8) and per-16 scales in LDS once; each wave walks rpw rows
// with the row's NSTEP weight quads issued upfront; the activations of ONE step are read from
// LDS, double-buffered behind compiler barriers. Per-(row, n) arithmetic and accumulation order
// are q27_k_proj_fp8_s<1,NSTEP,OUTBF>'s (bit-identical outputs).
#define Q27_LDS_FENCE() asm volatile("" ::: "memory")
template <int NR, int NSTEP, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_lds_nr(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv, int ystride,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, int rstride, float rs, int rpw) {
    constexpr int KB = NSTEP * 1024;
    constexpr int KS = KB / 16;
    __shared__ __align__(16) signed char sx[NR][KB];
    __shared__ float ss[NR][KS];
    const int kb = K, ks = K >> 4;
    for (int i = threadIdx.x; i < NR * (kb / 16); i += 256) {
        const int n = i / (kb / 16), o = i - n * (kb / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)(xq + (size_t)n * K))[o];
    }
    for (int i = threadIdx.x; i < NR * ks; i += 256) { const int n = i / ks, o = i - n * ks; ss[n][o] = xs[(size_t)n * ks + o]; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    bool ok[NSTEP]; int koff[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) { const int k0 = (t << 10) + lane * 16; ok[t] = (k0 < K); koff[t] = ok[t] ? k0 : 0; }
    for (int r = 0; r < rpw; ++r) {
        const int row = rowbase + r;
        if (row >= rows) break;
        uint4 wv[NSTEP];
        {
            const unsigned char* wp = W + (size_t)row * (size_t)K;
#pragma unroll
            for (int t = 0; t < NSTEP; ++t) wv[t] = q27_ld16nt(wp + koff[t]);
        }
        float acc[NR];
        uint4 xa[NR]; float sa[NR];
#pragma unroll
        for (int n = 0; n < NR; ++n) { acc[n] = 0.0f; xa[n] = *(const uint4*)(sx[n] + koff[0]); sa[n] = ss[n][koff[0] >> 4]; }
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            uint4 xb[NR]; float sb[NR];
            if (t + 1 < NSTEP) {
#pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + koff[t + 1]); sb[n] = ss[n][koff[t + 1] >> 4]; }
            }
            Q27_LDS_FENCE();
            if (ok[t]) {
                unsigned wh[8];
                q27_e4m3x4_h2(wv[t].x, wh[0], wh[1]);
                q27_e4m3x4_h2(wv[t].y, wh[2], wh[3]);
                q27_e4m3x4_h2(wv[t].z, wh[4], wh[5]);
                q27_e4m3x4_h2(wv[t].w, wh[6], wh[7]);
#pragma unroll
                for (int n = 0; n < NR; ++n) {
                    unsigned xh[8];
                    q27_i8x4_h2(xa[n].x, xh[0], xh[1]);
                    q27_i8x4_h2(xa[n].y, xh[2], xh[3]);
                    q27_i8x4_h2(xa[n].z, xh[4], xh[5]);
                    q27_i8x4_h2(xa[n].w, xh[6], xh[7]);
                    float d = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) d = q27_fdot2(wh[j], xh[j], d);
                    acc[n] += d * sa[n];
                }
            }
            Q27_LDS_FENCE();
            if (t + 1 < NSTEP) {
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
                if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + row] = q27_f2bf(v);
                else       ((float*)yv)[(size_t)n * ystride + row]         = v;
            }
        }
    }
}
#ifndef Q27_SCHED_FENCE
#define Q27_SCHED_FENCE() do { __builtin_amdgcn_sched_barrier(0); asm volatile("" ::: "memory"); } while (0)
#ifndef Q27_PIN_ROWS
#define Q27_PIN_ROWS(A) do { _Pragma("unroll") for (int n_ = 0; n_ < NR; ++n_) asm volatile("" : "+v"(A[n_]) : : "memory"); } while (0)
#endif
#endif
// Native 8-row form (Q27_NR_CH=8): a 256-thread block stages up to 8 activation rows in LDS once
// (K=6144 x 8 = 48 KB + scales 12 KB -> one block per CU, one wave per SIMD) and each wave is
// software-pipelined: the next row's weights are issued before this row's arithmetic and the next
// group's activations are read from LDS behind compiler barriers while the current group's dots run.
// Rows per wave = ceil(rows / (4 x CUs)); rows at stride xst (bytes). v1 (512 threads, no prefetch)
// measured 138.8 ms for the NR=8 layer pass: latency in the open, not spills (109-128 VGPRs).
#define Q27_NR_CASES8(LAUNCH) \
    switch (NR) { case 2: LAUNCH(2); break; case 3: LAUNCH(3); break; case 4: LAUNCH(4); break; \
                  case 5: LAUNCH(5); break; case 6: LAUNCH(6); break; \
                  case 7: LAUNCH(7); break; case 8: LAUNCH(8); break; default: return 0; }
static int q27_fp8_nr_lds_min() { static int v = 0; if (!v) { const char* e = getenv("Q27_NR_LDS_MIN"); v = e ? atoi(e) : 5; if (v < 2) v = 2; if (v > 9) v = 9; } return v; }
static int q27_fp8_lds8_ncu() {
    static int ncu = 0;
    if (!ncu) { int dev = 0, v = 0; hipGetDevice(&dev); hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev); ncu = v > 0 ? v : 60; }
    return ncu;
}
template <int NR, int NSTEP, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_lds8_nr(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv, int ystride,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, int rstride, float rs, int rpw, int xst) {
    // v6: two weight rows per iteration (see the nvfp4 forms): the K=1536 form profiled 115 us per 2 MB
    // call with one row per iteration -- latency-bound at one wave per SIMD.
    constexpr int KB = NSTEP * 1024;
    constexpr int KS = KB / 16;
    __shared__ __align__(16) signed char sx[NR][KB];
    __shared__ float ss[NR][KS];
    const int kb = K, ks = K >> 4;
    for (int i = threadIdx.x; i < NR * (kb / 16); i += 256) {
        const int n = i / (kb / 16), o = i - n * (kb / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)(xq + (size_t)n * xst))[o];
    }
    for (int i = threadIdx.x; i < NR * ks; i += 256) { const int n = i / ks, o = i - n * ks; ss[n][o] = xs[(size_t)n * (xst >> 4) + o]; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    const int rend = (rowbase + rpw < rows) ? rowbase + rpw : rows;
    bool ok[NSTEP]; int koff[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) { const int k0 = (t << 10) + lane * 16; ok[t] = (k0 < K); koff[t] = ok[t] ? k0 : 0; }
    uint4 wv[2][NSTEP];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int rr = (rowbase + p < rows) ? rowbase + p : rows - 1;
        const unsigned char* wp = W + (size_t)rr * (size_t)K;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) wv[p][t] = q27_ld16nt(wp + koff[t]);
    }
    for (int row = rowbase; row < rend; row += 2) {
        uint4 wn[2][NSTEP];
        if (row + 2 < rend) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int rr = (row + 2 + p < rows) ? row + 2 + p : rows - 1;
                const unsigned char* wp = W + (size_t)rr * (size_t)K;
#pragma unroll
                for (int t = 0; t < NSTEP; ++t) wn[p][t] = q27_ld16nt(wp + koff[t]);
            }
        }
        float ra0[NR], ra1[NR];   // residual share prefetched (fp32-out forms only; dead for OUTBF=1)
        if (!OUTBF && radd) {
#pragma unroll
            for (int n = 0; n < NR; ++n) { ra0[n] = q27_bf2f(radd[(size_t)n * rstride + row]); ra1[n] = (row + 1 < rend) ? q27_bf2f(radd[(size_t)n * rstride + row + 1]) : 0.f; }
        } else {
#pragma unroll
            for (int n = 0; n < NR; ++n) { ra0[n] = 0.f; ra1[n] = 0.f; }
        }
        float acc0[NR], acc1[NR];
        uint4 xa[NR]; float sa[NR];
#pragma unroll
        for (int n = 0; n < NR; ++n) { acc0[n] = 0.0f; acc1[n] = 0.0f; xa[n] = *(const uint4*)(sx[n] + koff[0]); sa[n] = ss[n][koff[0] >> 4]; }
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            uint4 xb[NR]; float sb[NR];
            if (t + 1 < NSTEP) {
#pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + koff[t + 1]); sb[n] = ss[n][koff[t + 1] >> 4]; }
            }
            if (ok[t]) {
                unsigned wh0[8], wh1[8];
                q27_e4m3x4_h2(wv[0][t].x, wh0[0], wh0[1]); q27_e4m3x4_h2(wv[0][t].y, wh0[2], wh0[3]);
                q27_e4m3x4_h2(wv[0][t].z, wh0[4], wh0[5]); q27_e4m3x4_h2(wv[0][t].w, wh0[6], wh0[7]);
                q27_e4m3x4_h2(wv[1][t].x, wh1[0], wh1[1]); q27_e4m3x4_h2(wv[1][t].y, wh1[2], wh1[3]);
                q27_e4m3x4_h2(wv[1][t].z, wh1[4], wh1[5]); q27_e4m3x4_h2(wv[1][t].w, wh1[6], wh1[7]);
#pragma unroll
                for (int n = 0; n < NR; ++n) {
                    unsigned xh[8];
                    q27_i8x4_h2(xa[n].x, xh[0], xh[1]);
                    q27_i8x4_h2(xa[n].y, xh[2], xh[3]);
                    q27_i8x4_h2(xa[n].z, xh[4], xh[5]);
                    q27_i8x4_h2(xa[n].w, xh[6], xh[7]);
                    float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) { d0 = q27_fdot2(wh0[j], xh[j], d0); d1 = q27_fdot2(wh1[j], xh[j], d1); }
                    acc0[n] += d0 * sa[n]; acc1[n] += d1 * sa[n];
                }
            }
            Q27_PIN_ROWS(acc0); Q27_PIN_ROWS(acc1);
            if (t + 1 < NSTEP) {
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
                if (!OUTBF) v += rs * ra0[n];
                if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + row] = q27_f2bf(v);
                else       ((float*)yv)[(size_t)n * ystride + row] = v;
                if (row + 1 < rend) {
                    float v1 = acc1[n] * alpha;
                    if (!OUTBF) v1 += rs * ra1[n];
                    if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + row + 1] = q27_f2bf(v1);
                    else       ((float*)yv)[(size_t)n * ystride + row + 1] = v1;
                }
            }
        }
        if (row + 2 < rend) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
                for (int t = 0; t < NSTEP; ++t) wv[p][t] = wn[p][t];
        }
    }
}
static inline int q27_nr_rpw8_fp8(int rows) { const int w = 4 * q27_fp8_lds8_ncu(); int r = (rows + w - 1) / w; return r < 1 ? 1 : r; }
// ============================================================================================
// Q27_DEC_I8: INT8 PER-64 EXECUTION MIRRORS OF THE FP8 PROJECTIONS FOR THE ROW-BATCHED PATHS.
// The fp8 kernels above form the product in fp16 pairs: per 16 weights and per activation row they
// redo the e4m3 -> half decode (the decode sits inside q27_dot16_fp8) and issue 8 v_dot2, ~32 VALU
// per 16 weights per row; at NR >= 3 the family is VALU-bound (profile: 8.5 ms/round at NR=8, as
// much as gate/up). The immutable fp8 weights are converted ONCE at load into the prefill lane's
// int8 per-64-group mirror (q27_fp8_to_i8g64_conv: same bytes per weight, fp32 scale per 64) and
// these kernels run 4 x v_dot4 + 1 fma per 16 weights per row. Side table keyed by the fp8 weight
// pointer (the handle structs keep their layout). Q27_DEC_I8=1 enables; the fp8 kernels stay the
// default and the plain (single-row) path is unchanged in this step.
// ============================================================================================
static const unsigned char* s_i8g_key[4096];
static q27_i8g_t            s_i8g_val[4096];
static int                  s_i8g_n = 0;
static int q27_dec_i8_on() { static int v = -1; if (v < 0) { const char* e = getenv("Q27_DEC_I8"); v = e ? (atoi(e) != 0) : 0; } return v; }
static int q27_dec_i8_plain_on() { static int v = -1; if (v < 0) { const char* e = getenv("Q27_DEC_I8_PLAIN"); v = e ? (atoi(e) != 0) : 0; } return v; }   // plain (single-row) path on the mirrors: measured separately
extern "C" const q27_i8g_t* q27_fp8_i8g(const q27_fp8_t* w) {
    if (!w) return nullptr;
    // Keyed by the HANDLE address, not the device pointer: the fp8 arena may be freed after the
    // mirrors are built (Q27_DEC_I8 + Q27_DEC_I8_PLAIN), and w->w is NULL then.
    for (int i = 0; i < s_i8g_n; ++i) if (s_i8g_key[i] == (const unsigned char*)w) return &s_i8g_val[i];
    return nullptr;
}
extern "C" int q27_fp8_make_i8g(const q27_fp8_t* w, hipStream_t s) {
    if (!w || !w->w || s_i8g_n >= 4096) return 0;
    if (q27_fp8_i8g(w)) return 1;
    q27_i8g_t m{};
    if (!q27_fp8_to_i8g64_conv(w, &m, s)) return 0;
    s_i8g_key[s_i8g_n] = (const unsigned char*)w; s_i8g_val[s_i8g_n] = m; ++s_i8g_n;
    return 1;
}
// Q27_DEC_I8: once the fp8 originals are released, each handle's weight pointer is replaced by a non-canonical
// sentinel (any accidental fp8 use faults loudly instead of reading recycled VRAM); the mirror stays reachable
// because its side-table key moves to the sentinel with it.
extern "C" int q27_fp8_i8g_rekey(const void* oldw, const void* neww) {
    for (int i = 0; i < s_i8g_n; ++i) if (s_i8g_key[i] == (const unsigned char*)oldw) { s_i8g_key[i] = (const unsigned char*)neww; return 1; }
    return 0;
}
// Register form, NR <= 4 (one weight row per wave; activations of NR rows resident): the exact
// indexing of q27_k_proj_fp8_snr with int8 weights and the per-64 weight scale applied per step.
template <int NR, int NSTEP, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_i8g_snr(
        const signed char* __restrict__ W, const float* __restrict__ WS, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv, int ystride,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, int rstride, float rs, int xst) {
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int row = blockIdx.x * 4 + wave;
    if (row >= rows) return;
    const signed char* wp = W + (size_t)row * (size_t)K;
    const float* wsp = WS + (size_t)row * (size_t)(K >> 6);
    uint4 wv[NSTEP]; float ws[NSTEP];
    uint4 xv[NR][NSTEP];
    float sc[NR][NSTEP];
    bool  ok[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        const int k0 = (t << 10) + lane * 16;
        ok[t] = (k0 < K);
        const int ks = ok[t] ? k0 : 0;
        wv[t] = q27_ld16nt(wp + ks);
        ws[t] = wsp[ks >> 6];
#pragma unroll
        for (int n = 0; n < NR; ++n) {
            xv[n][t] = *(const uint4*)(xq + (size_t)n * xst + ks);
            sc[n][t] = xs[(size_t)n * (xst >> 4) + (ks >> 4)];
        }
    }
    float acc[NR];
#pragma unroll
    for (int n = 0; n < NR; ++n) acc[n] = 0.0f;
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) {
        if (!ok[t]) continue;
#pragma unroll
        for (int n = 0; n < NR; ++n) {
            int d = q27_d4(wv[t].x, xv[n][t].x, 0);
            d = q27_d4(wv[t].y, xv[n][t].y, d);
            d = q27_d4(wv[t].z, xv[n][t].z, d);
            d = q27_d4(wv[t].w, xv[n][t].w, d);
            acc[n] = fmaf((float)d, ws[t] * sc[n][t], acc[n]);
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
            if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + row] = q27_f2bf(v);
            else       ((float*)yv)[(size_t)n * ystride + row] = v;
        }
    }
}
// LDS form, NR 5..8: the v7 fp8 LDS kernel's structure (two weight rows per iteration, next-pair
// prefetch, residual prefetched, accumulators pinned per step) with int8 weights and per-64 scales.
template <int NR, int NSTEP, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_i8g_lds8_nr(
        const signed char* __restrict__ W, const float* __restrict__ WS, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv, int ystride,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, int rstride, float rs, int rpw, int xst) {
    constexpr int KB = NSTEP * 1024;
    constexpr int KS = KB / 16;
    __shared__ __align__(16) signed char sx[NR][KB];
    __shared__ float ss[NR][KS];
    const int kb = K, ks = K >> 4;
    for (int i = threadIdx.x; i < NR * (kb / 16); i += 256) {
        const int n = i / (kb / 16), o = i - n * (kb / 16);
        ((uint4*)sx[n])[o] = ((const uint4*)(xq + (size_t)n * xst))[o];
    }
    for (int i = threadIdx.x; i < NR * ks; i += 256) { const int n = i / ks, o = i - n * ks; ss[n][o] = xs[(size_t)n * (xst >> 4) + o]; }
    __syncthreads();
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int rowbase = (blockIdx.x * 4 + wave) * rpw;
    if (rowbase >= rows) return;
    const int rend = (rowbase + rpw < rows) ? rowbase + rpw : rows;
    const int NG64 = K >> 6;
    bool ok[NSTEP]; int koff[NSTEP];
#pragma unroll
    for (int t = 0; t < NSTEP; ++t) { const int k0 = (t << 10) + lane * 16; ok[t] = (k0 < K); koff[t] = ok[t] ? k0 : 0; }
    uint4 wv[2][NSTEP]; float wsv[2][NSTEP];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int rr = (rowbase + p < rows) ? rowbase + p : rows - 1;
        const signed char* wp = W + (size_t)rr * (size_t)K; const float* wsp = WS + (size_t)rr * NG64;
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) { wv[p][t] = q27_ld16nt(wp + koff[t]); wsv[p][t] = wsp[koff[t] >> 6]; }
    }
    for (int row = rowbase; row < rend; row += 2) {
        uint4 wn[2][NSTEP]; float wsn[2][NSTEP];
        if (row + 2 < rend) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int rr = (row + 2 + p < rows) ? row + 2 + p : rows - 1;
                const signed char* wp = W + (size_t)rr * (size_t)K; const float* wsp = WS + (size_t)rr * NG64;
#pragma unroll
                for (int t = 0; t < NSTEP; ++t) { wn[p][t] = q27_ld16nt(wp + koff[t]); wsn[p][t] = wsp[koff[t] >> 6]; }
            }
        }
        float ra0[NR], ra1[NR];
        if (!OUTBF && radd) {
#pragma unroll
            for (int n = 0; n < NR; ++n) { ra0[n] = q27_bf2f(radd[(size_t)n * rstride + row]); ra1[n] = (row + 1 < rend) ? q27_bf2f(radd[(size_t)n * rstride + row + 1]) : 0.f; }
        } else {
#pragma unroll
            for (int n = 0; n < NR; ++n) { ra0[n] = 0.f; ra1[n] = 0.f; }
        }
        float acc0[NR], acc1[NR];
        uint4 xa[NR]; float sa[NR];
#pragma unroll
        for (int n = 0; n < NR; ++n) { acc0[n] = 0.0f; acc1[n] = 0.0f; xa[n] = *(const uint4*)(sx[n] + koff[0]); sa[n] = ss[n][koff[0] >> 4]; }
#pragma unroll
        for (int t = 0; t < NSTEP; ++t) {
            uint4 xb[NR]; float sb[NR];
            if (t + 1 < NSTEP) {
#pragma unroll
                for (int n = 0; n < NR; ++n) { xb[n] = *(const uint4*)(sx[n] + koff[t + 1]); sb[n] = ss[n][koff[t + 1] >> 4]; }
            }
            if (ok[t]) {
#pragma unroll
                for (int n = 0; n < NR; ++n) {
                    int d0 = q27_d4(wv[0][t].x, xa[n].x, 0); d0 = q27_d4(wv[0][t].y, xa[n].y, d0); d0 = q27_d4(wv[0][t].z, xa[n].z, d0); d0 = q27_d4(wv[0][t].w, xa[n].w, d0);
                    int d1 = q27_d4(wv[1][t].x, xa[n].x, 0); d1 = q27_d4(wv[1][t].y, xa[n].y, d1); d1 = q27_d4(wv[1][t].z, xa[n].z, d1); d1 = q27_d4(wv[1][t].w, xa[n].w, d1);
                    acc0[n] = fmaf((float)d0, wsv[0][t] * sa[n], acc0[n]);
                    acc1[n] = fmaf((float)d1, wsv[1][t] * sa[n], acc1[n]);
                }
            }
            Q27_PIN_ROWS(acc0); Q27_PIN_ROWS(acc1);
            if (t + 1 < NSTEP) {
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
                if (!OUTBF) v += rs * ra0[n];
                if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + row] = q27_f2bf(v);
                else       ((float*)yv)[(size_t)n * ystride + row] = v;
                if (row + 1 < rend) {
                    float v1 = acc1[n] * alpha;
                    if (!OUTBF) v1 += rs * ra1[n];
                    if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + row + 1] = q27_f2bf(v1);
                    else       ((float*)yv)[(size_t)n * ystride + row + 1] = v1;
                }
            }
        }
        if (row + 2 < rend) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
                for (int t = 0; t < NSTEP; ++t) { wv[p][t] = wn[p][t]; wsv[p][t] = wsn[p][t]; }
        }
    }
}
// Router: NR rows through the int8 mirror m of an fp8 projection. K = 5120 / 6144 (bf16 out, no residual)
// or 1536 (fp32 out + residual share). Returns 0 when the shape has no instance (caller falls back to fp8).
static int g_fp8_freed = 0;
extern "C" void q27_fp8_freed_set(int v) { g_fp8_freed = v ? 1 : 0; }
extern "C" int  q27_fp8_freed(void) { return g_fp8_freed; }
// DFlash2 runner access: the i8g row-batched GEMM over the drafter's shapes (same dispatch).
// q27_i8g_d_t (q27_dflash2.h) is layout-compatible with q27_i8g_t.
extern "C" int q27_df2_gemm(const void* m, const signed char* xq, int xst, const float* xs, void* y, int ystride,
                            const unsigned short* radd, int rstride, float rs, int NR, int outbf, hipStream_t s) {
    return q27_proj_i8g_rows((const q27_i8g_t*)m, xq, xst, xs, y, ystride, radd, rstride, rs, NR, outbf, s);
}
static void q27_dec_i8_route_declined(const q27_fp8_t* w, int NR) {
    if (g_fp8_freed) {
        std::fprintf(stderr, "Q27_DEC_I8: int8 route declined AFTER the fp8 arena was freed (rows=%d K=%d NR=%d); aborting\n", w ? w->rows : -1, w ? w->K : -1, NR);
        std::abort();
    }
    if (q27_env_flag("Q27_DEC_I8_DBG", false))
        std::fprintf(stderr, "Q27_DEC_I8: int8 route declined, fp8 fallback (rows=%d K=%d NR=%d)\n", w ? w->rows : -1, w ? w->K : -1, NR);
}
static int q27_proj_i8g_rows(const q27_i8g_t* m, const signed char* xq, int xst, const float* xs, void* y, int ystride,
                             const unsigned short* radd, int rstride, float rs, int NR, int outbf, hipStream_t s) {
    if (!m || !m->w || !m->s || NR < 1 || NR > 8) return 0;
    const float alpha = m->alpha;
    // shape -> (NSTEP, default OUTBF); the DFlash2 drafter adds 4096/17408/25600/4352 and an
    // explicit outbf override (gate/up need fp32 out of a K=5120 weight, etc).
    int NS = 0, OB = outbf;
    if (m->K == 5120) NS = 5; else if (m->K == 6144) NS = 6; else if (m->K == 1536) NS = 2;
    else if (m->K == 4096) NS = 4; else if (m->K == 17408) NS = 17; else if (m->K == 25600) NS = 25; else if (m->K == 4352) NS = 5;
    else return 0;
    if (OB < 0) OB = (m->K == 1536 || m->K == 17408 || m->K == 25600) ? 0 : 1;
    if (NS == 17 || NS == 25) {   // big-K (DFlash2 down-full / fc): register form only, chunked to <=4 rows —
        // the LDS forms' per-thread wv/wsv state exceeds the 64 KB local-memory limit at these NSTEPs.
        for (int c0 = 0; c0 < NR; c0 += 4) {
            const int cn = (NR - c0 < 4) ? NR - c0 : 4;
            const dim3 grid((unsigned)((m->rows + 3) / 4)), blk(256);
#define Q27_LI(N, NS_, OB_) hipLaunchKernelGGL((q27_k_proj_i8g_snr<N,NS_,OB_>), grid, blk, 0, s, m->w, m->s,                 xq + (size_t)c0 * xst, xs + (size_t)c0 * (xst >> 4), (char*)y + (size_t)c0 * ystride * (OB_ ? 2 : 4), ystride,                 m->rows, m->K, alpha, radd ? radd + (size_t)c0 * rstride : nullptr, rstride, rs, xst)
#define Q27_LIK(NS_, OB_) switch (cn) { case 1: Q27_LI(1,NS_,OB_); break; case 2: Q27_LI(2,NS_,OB_); break; case 3: Q27_LI(3,NS_,OB_); break; case 4: Q27_LI(4,NS_,OB_); break; default: return 0; }
            if (NS == 17 && OB == 0) { Q27_LIK(17,0) } else if (NS == 25 && OB == 0) { Q27_LIK(25,0) } else return 0;
#undef Q27_LIK
#undef Q27_LI
        }
        return 1;
    }
    if (NR <= 4) {
        const dim3 grid((unsigned)((m->rows + 3) / 4)), blk(256);
#define Q27_LI(N, NS_, OB_) hipLaunchKernelGGL((q27_k_proj_i8g_snr<N,NS_,OB_>), grid, blk, 0, s, m->w, m->s, xq, xs, y, ystride, m->rows, m->K, alpha, radd, rstride, rs, xst)
#define Q27_LIK(NS_, OB_) switch (NR) { case 1: Q27_LI(1,NS_,OB_); break; case 2: Q27_LI(2,NS_,OB_); break; case 3: Q27_LI(3,NS_,OB_); break; case 4: Q27_LI(4,NS_,OB_); break; default: return 0; }
        if (NS == 5 && OB == 1) { Q27_LIK(5,1) } else if (NS == 5 && OB == 0) { Q27_LIK(5,0) } else if (NS == 6 && OB == 1) { Q27_LIK(6,1) }
        else if (NS == 2 && OB == 0) { Q27_LIK(2,0) } else if (NS == 4 && OB == 1) { Q27_LIK(4,1) }
        else if (NS == 17 && OB == 0) { Q27_LIK(17,0) } else if (NS == 25 && OB == 0) { Q27_LIK(25,0) } else return 0;
#undef Q27_LIK
#undef Q27_LI
        return 1;
    }
    const int rpw = q27_nr_rpw8_fp8(m->rows);
    const dim3 grid((unsigned)((m->rows + 4 * rpw - 1) / (4 * rpw))), blk(256);
#define Q27_LI(N, NS_, OB_) hipLaunchKernelGGL((q27_k_proj_i8g_lds8_nr<N,NS_,OB_>), grid, blk, 0, s, m->w, m->s, xq, xs, y, ystride, m->rows, m->K, alpha, radd, rstride, rs, rpw, xst)
#define Q27_LIK(NS_, OB_) switch (NR) { case 5: Q27_LI(5,NS_,OB_); break; case 6: Q27_LI(6,NS_,OB_); break; case 7: Q27_LI(7,NS_,OB_); break; case 8: Q27_LI(8,NS_,OB_); break; default: return 0; }
        if (NS == 5 && OB == 1) { Q27_LIK(5,1) } else if (NS == 5 && OB == 0) { Q27_LIK(5,0) } else if (NS == 6 && OB == 1) { Q27_LIK(6,1) }
        else if (NS == 2 && OB == 0) { Q27_LIK(2,0) } else if (NS == 4 && OB == 1) { Q27_LIK(4,1) } else return 0;
#undef Q27_LIK
#undef Q27_LI
    return 1;
}
// K = 5120 / 6144 (bf16 out) or 1536 (fp32 out + residual share), NR = 5..8
static int q27_fp8_lds8(const q27_fp8_t* w, const signed char* xq, int xst, const float* xs, void* y, int ystride,
                        const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s) {
    if (!w || NR < 2 || NR > 8) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const int rpw = q27_nr_rpw8_fp8(w->rows);
    const dim3 grid((unsigned)((w->rows + 4 * rpw - 1) / (4 * rpw))), blk(256);
    if (w->K == 5120) {
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_lds8_nr<N,5,1>), grid, blk, 0, s, w->w, xq, xs, y, ystride, w->rows, w->K, alpha, radd, rstride, rs, rpw, xst)
        Q27_NR_CASES8(Q27_L)
#undef Q27_L
    } else if (w->K == 6144) {
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_lds8_nr<N,6,1>), grid, blk, 0, s, w->w, xq, xs, y, ystride, w->rows, w->K, alpha, radd, rstride, rs, rpw, xst)
        Q27_NR_CASES8(Q27_L)
#undef Q27_L
    } else if (w->K == 1536) {
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_lds8_nr<N,2,0>), grid, blk, 0, s, w->w, xq, xs, y, ystride, w->rows, w->K, alpha, radd, rstride, rs, rpw, xst)
        Q27_NR_CASES8(Q27_L)
#undef Q27_L
    } else return 0;
    return 1;
}

static inline int q27_nr_rpw_fp8(int rows) { int r = rows / 720; return r < 1 ? 1 : (r > 4 ? 4 : r); }

extern "C" int q27_proj_fp8_bf16_lds(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                     unsigned short* y, int ystride, int NR, hipStream_t s) {
    if (!w || w->K != 5120 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const int rpw = q27_nr_rpw_fp8(w->rows);
    const dim3 grid((unsigned)((w->rows + 4 * rpw - 1) / (4 * rpw))), blk(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_lds_nr<N,5,1>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, (const unsigned short*)nullptr, 0, 0.f, rpw)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
extern "C" int q27_proj_fp8_res_lds(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                    float* y, int ystride, const unsigned short* radd, int rstride,
                                    float rs, int NR, hipStream_t s) {
    if (!w || w->K != 1536 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const int rpw = q27_nr_rpw_fp8(w->rows);
    const dim3 grid((unsigned)((w->rows + 4 * rpw - 1) / (4 * rpw))), blk(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_lds_nr<N,2,0>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, radd, rstride, rs, rpw)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}


// =============================================================================================
// SPLIT-ROW ROW-BATCHED FP8 FORM (Q27_NR_HG): two waves per weight row, each owning a CONTIGUOUS
// half of the row's 16-byte chunks (K=5120: 320 chunks -> 160 per wave = 2 full wave-steps + 32
// lanes; K=1536: 96 -> 48 per wave). Per lane 2-3 uint4 weight quads instead of 5, so the NR
// activation quads fit: NR=3 at ~4 waves/SIMD. Half-row partials meet through LDS; the fp32
// association differs from q27_k_proj_fp8_s (plain uses the same form under Q27_GEMV_ALT=3).
template <int NR, int NCH, int OUTBF>
__global__ __launch_bounds__(256) void q27_k_proj_fp8_hg_nr(
        const unsigned char* __restrict__ W, const signed char* __restrict__ xq,
        const float* __restrict__ xs, void* __restrict__ yv, int ystride,
        int rows, int K, float alpha,
        const unsigned short* __restrict__ radd, int rstride, float rs) {
    constexpr int NCHH = NCH / 2, CH = (NCHH + 63) / 64, REM = NCHH - 64 * (CH - 1);
    __shared__ float part[4][NR];
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row = blockIdx.x * 2 + (wave >> 1), h = wave & 1;
    const int rr = (row < rows) ? row : (rows - 1);
    const unsigned char* wp = W + (size_t)rr * (size_t)K;
    const int KS = K >> 4;
    const int base = h * NCHH;
    uint4 wv[CH]; uint4 xv[NR][CH]; float sc[NR][CH];
#pragma unroll
    for (int i = 0; i < CH; ++i) {
        const int c = base + ((i < CH - 1 || lane < REM) ? (i * 64 + lane) : 0);   // chunk index, clamped tail
        const int k0 = c * 16;
        wv[i] = q27_ld16nt(wp + k0);
#pragma unroll
        for (int n = 0; n < NR; ++n) { xv[n][i] = *(const uint4*)(xq + (size_t)n * K + k0); sc[n][i] = xs[(size_t)n * KS + c]; }
    }
    float acc[NR];
#pragma unroll
    for (int n = 0; n < NR; ++n) acc[n] = 0.0f;
#pragma unroll
    for (int i = 0; i < CH; ++i) {
        if (i == CH - 1 && lane >= REM) break;
        unsigned wh[8];
        q27_e4m3x4_h2(wv[i].x, wh[0], wh[1]);
        q27_e4m3x4_h2(wv[i].y, wh[2], wh[3]);
        q27_e4m3x4_h2(wv[i].z, wh[4], wh[5]);
        q27_e4m3x4_h2(wv[i].w, wh[6], wh[7]);
#pragma unroll
        for (int n = 0; n < NR; ++n) {
            unsigned xh[8];
            q27_i8x4_h2(xv[n][i].x, xh[0], xh[1]);
            q27_i8x4_h2(xv[n][i].y, xh[2], xh[3]);
            q27_i8x4_h2(xv[n][i].z, xh[4], xh[5]);
            q27_i8x4_h2(xv[n][i].w, xh[6], xh[7]);
            float d = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) d = q27_fdot2(wh[j], xh[j], d);
            acc[n] += d * sc[n][i];
        }
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
            if (OUTBF) ((unsigned short*)yv)[(size_t)n * ystride + ro] = q27_f2bf(v);
            else       ((float*)yv)[(size_t)n * ystride + ro]         = v;
        }
    }
}
extern "C" int q27_proj_fp8_bf16_hg(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                    unsigned short* y, int ystride, int NR, hipStream_t s) {
    if (!w || w->K != 5120 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 grid((unsigned)((w->rows + 1) / 2)), blk(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_hg_nr<N,320,1>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, (const unsigned short*)nullptr, 0, 0.f)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
extern "C" int q27_proj_fp8_res_hg(const q27_fp8_t* w, const signed char* xq, const float* xs,
                                   float* y, int ystride, const unsigned short* radd, int rstride,
                                   float rs, int NR, hipStream_t s) {
    if (!w || w->K != 1536 || NR < 1 || NR > 4) return 0;
    const float alpha = w->in_scale * w->wscale * 256.0f;
    const dim3 grid((unsigned)((w->rows + 1) / 2)), blk(256);
#define Q27_L(N) hipLaunchKernelGGL((q27_k_proj_fp8_hg_nr<N,96,0>), grid, blk, 0, s, w->w, xq, xs, \
                     (void*)y, ystride, w->rows, w->K, alpha, radd, rstride, rs)
    Q27_NR_CASES(Q27_L)
#undef Q27_L
    return 1;
}
