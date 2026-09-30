// q27_sr.hip -- single-residency ("SR") layout helpers for gfx906 (MI50), wave64, ROCm 5.7.1.
//
// Under SR each card keeps ONE copy of its 16 owned layers: MLP as block-scaled FP8 (e4m3 bytes + a
// [rows/128][K/128] BF16 weight_scale_inv plane), the seven fp8 projections as int8 per-64 mirrors. The
// prefill sweep runs rocBLAS int8 GEMMs, which want per-ROW int8 (one fp32 scale per row). Today that
// form is manufactured IN PLACE from a per-64 mirror (q27_i8g64_rowify), which destroys the resident
// copy -- fine when a second copy existed, fatal under SR. Everything here writes OUT OF PLACE into a
// caller-owned transient, once per layer per prefill chunk, so each producer must be bandwidth-bound:
// the [17408][5120] gate/up tensor is 89 MB in + 89 MB out, ~0.25 ms at the MI50's ~800 GB/s.
//
// GEOMETRY. One wave (64 lanes) per row, four rows per 256-thread block. A lane owns the 16-byte chunk
// at k = t*1024 + lane*16 for t = 0..NCH-1, so every load/store instruction moves 1 KB contiguous per
// wave (fully coalesced) and NCH = ceil(K/1024) <= 17 chunks cover K <= 17408 in registers. The fp8
// producers need the row's max before they can quantize, and holding NCH uint4 in VGPRs (68 at NCH=17)
// is cheaper than reading the row twice; NCH is a template parameter so the K=5120 shapes do not pay the
// register cost of the K=17408 one. Waves never synchronise, so a ragged row tail simply returns.
//
// ARITHMETIC IS THE ENGINE'S, NOT A REDERIVATION (see include/q27_sr.h and tools/q27_sr_test.cpp):
//   fp8 weight   v = q27_e4m3s(w) * bf16(bs[(row>>7)*bs_kblk + (k>>7)])   (q27_k_proj_fp8_snr<...,BS=1>)
//   fp8 row      sc = max|v| * (1/127); inv = sc>0 ? 1/sc : 0; q = clamp(rintf(v*inv), -127, 127)
//   per-64 row   sc = max of the per-64 scales; q = clamp(rintf((float)w * s64 * inv))  (q27_k_i8g64_to_grp)
//   swiglu       q27_k_swiglu_quant_b verbatim, natural byte order instead of even-then-odd
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include "q27.h"
#include "q27_sr.h"

namespace {

constexpr int SR_MAXK   = 17 * 1024;   // 17408 = Q27_INTER: the widest row any SR producer sees
constexpr int SR_ROWS_PB = 4;          // rows per 256-thread block (one wave each)

// Wave-wide max. fmaxf is exact and commutative, so the reduction order cannot change the result.
__device__ __forceinline__ float sr_wavemax(float v) {
    v = fmaxf(v, __shfl_xor(v, 32, 64)); v = fmaxf(v, __shfl_xor(v, 16, 64)); v = fmaxf(v, __shfl_xor(v, 8, 64));
    v = fmaxf(v, __shfl_xor(v, 4, 64));  v = fmaxf(v, __shfl_xor(v, 2, 64));  v = fmaxf(v, __shfl_xor(v, 1, 64));
    return v;
}
__device__ __forceinline__ int sr_q8(float v) {   // the engine's rounding: rintf then symmetric clamp
    int q = (int)rintf(v);
    return q < -127 ? -127 : (q > 127 ? 127 : q);
}
__device__ __forceinline__ unsigned sr_pack4(int c0, int c1, int c2, int c3) {   // c0 -> lowest address
    // Two byte-permutes and an OR instead of the and/shift/or chain: the low byte of a negative int IS
    // its two's-complement int8. Selector semantics as the engine's q27_e4m3x4_h2: 0-3 pick bytes of the
    // second operand, 4-7 of the first, 0x0C is a zero byte.
    const unsigned lo = __builtin_amdgcn_perm((unsigned)c1, (unsigned)c0, 0x0C0C0400u);   // [0, 0, c1, c0]
    const unsigned hi = __builtin_amdgcn_perm((unsigned)c3, (unsigned)c2, 0x04000C0Cu);   // [c3, c2, 0, 0]
    return lo | hi;
}

// ---- block-scaled fp8: four bytes of one dword (byte 0 = lowest k) times ONE block scale --------
// 16 | 128, so a lane's 16-byte chunk never straddles a 128-wide K block: one bsf per chunk is exact.
//
// THE ROW MAX COMES FROM THE BYTE ENCODINGS, NOT FROM DECODED VALUES. |e4m3(b)| is a non-decreasing
// function of the 7-bit magnitude field (b & 0x7F): sign-magnitude, exponent above mantissa, and the
// subnormal top (7 * 2^-9) sits below the smallest normal (2^-6). So max|v| over a chunk that shares
// one block scale is e4m3s(max magnitude byte) * |bsf| -- one decode per 16 bytes instead of 16, and
// bit-identical to taking the max of the 16 rounded products (RN(|a*b|) = |RN(a*b)|). This is what
// turned the producers from ALU-bound (282 GB/s) into bandwidth-bound; the full q27_e4m3s decode is
// ~12 VALU per byte and the first version ran it twice per byte.
__device__ __forceinline__ unsigned sr_maxbyte4(unsigned d) {   // max of the four 7-bit magnitude fields (v_bfe + v_max3)
    const unsigned b0 = d & 0x7Fu, b1 = (d >> 8) & 0x7Fu, b2 = (d >> 16) & 0x7Fu, b3 = (d >> 24) & 0x7Fu;
    return max(max(b0, b1), max(b2, b3));
}
__device__ __forceinline__ unsigned sr_maxbyte16(uint4 w) {
    return max(max(sr_maxbyte4(w.x), sr_maxbyte4(w.y)), max(sr_maxbyte4(w.z), sr_maxbyte4(w.w)));
}
__device__ __forceinline__ float sr_fp8x16_absmax(uint4 w, float bsf) {
    return q27_e4m3s(sr_maxbyte16(w)) * fabsf(bsf);
}
// THE QUANTISE PASS DECODES THE ENGINE'S WAY: q27_e4m3x4_h2's two perms + shift + add turn four e4m3
// bytes into two half2 dwords carrying EXACTLY 2^-8 times the value (every e4m3 value times 2^-8 is
// an fp16 value, subnormals included; only the never-present NaN codes 0x7F/0xFF differ: 480 here,
// INF in q27_e4m3s). The 2^-8 is repaid by feeding bsf * 256 -- exact, a power of two -- so
//     (float)h * (bsf * 256) == RN(e4m3(b) * bsf)
// bit for bit, i.e. the reference arithmetic at ~7 VALU per byte instead of ~20.
typedef _Float16 sr_h2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void sr_e4m3x4_h2(unsigned d, unsigned& lo, unsigned& hi) {   // == q27_e4m3x4_h2 (src/q27_fp8.hip)
    lo = __builtin_amdgcn_perm(0u, d, 0x01040004u);   // [b1, 0, b0, 0]
    hi = __builtin_amdgcn_perm(0u, d, 0x03040204u);   // [b3, 0, b2, 0]
    lo >>= 1;  lo += (lo & 0x40004000u);
    hi >>= 1;  hi += (hi & 0x40004000u);
}
__device__ __forceinline__ unsigned sr_fp8x4_quant(unsigned d, float bsf256, float inv) {
    unsigned lo, hi; sr_e4m3x4_h2(d, lo, hi);
    union { unsigned u; sr_h2_t h; } a, b; a.u = lo; b.u = hi;
    const float v0 = (float)a.h.x * bsf256, v1 = (float)a.h.y * bsf256;
    const float v2 = (float)b.h.x * bsf256, v3 = (float)b.h.y * bsf256;
    return sr_pack4(sr_q8(v0 * inv), sr_q8(v1 * inv), sr_q8(v2 * inv), sr_q8(v3 * inv));
}
__device__ __forceinline__ uint4 sr_fp8x16_quant(uint4 w, float bsf, float inv) {
    const float bsf256 = bsf * 256.0f;
    return make_uint4(sr_fp8x4_quant(w.x, bsf256, inv), sr_fp8x4_quant(w.y, bsf256, inv),
                      sr_fp8x4_quant(w.z, bsf256, inv), sr_fp8x4_quant(w.w, bsf256, inv));
}

// Up to four K-slices of one logical [rows][nsl*Ks] tensor, each with its own compacted bs plane.
// nsl = 1 is the plain single-tensor case (Ks = K). Passed by value: it is the kernel's argument block.
struct sr_fp8_slices {
    const unsigned char*  w[4];
    const unsigned short* bs[4];
    int bs_kblk[4];
    int nsl, Ks, Ktot, rows;
};
template <typename T> __device__ __forceinline__ T sr_sel4(const T (&a)[4], int c) {
    // A per-lane index into a kernel-argument array would be a per-lane load from the kernarg
    // segment; three v_cndmask are what we want instead.
    return c == 0 ? a[0] : (c == 1 ? a[1] : (c == 2 ? a[2] : a[3]));
}

template <int NCH>
__global__ __launch_bounds__(256) void q27_k_sr_fp8bs_row(sr_fp8_slices p, signed char* __restrict__ out,
                                                          size_t ldw_out, float* __restrict__ rowscale)
{
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int row = blockIdx.x * SR_ROWS_PB + wave;
    if (row >= p.rows) return;
    uint4 wv[NCH];
    float bsf[NCH];
    // 1. Issue every load before touching any data. A lane past the row end loads chunk 0 instead
    //    (clamped, unconditional -- the value is masked out below): a divergent `if` around each load
    //    would put the loads in separate exec regions and, with the max taken inside, made the compiler
    //    wait for each chunk before issuing the next (the first version: 17 serialized latencies per
    //    wave, 104 GB/s on the down_proj shape).
#pragma unroll
    for (int t = 0; t < NCH; ++t) {
        const int k0 = (t << 10) + (lane << 4);
        const int kc = (k0 < p.Ktot) ? k0 : 0;
        // The down_proj slice width (4352) is not a multiple of 1024, so a chunk's slice is a
        // per-(t, lane) property; 16 | Ks and 128 | Ks guarantee a chunk never straddles a slice
        // boundary or a block-scale boundary.
        const int c   = (p.nsl > 1) ? ((kc >= p.Ks) + (kc >= 2 * p.Ks) + (kc >= 3 * p.Ks)) : 0;
        const int off = kc - c * p.Ks;
        wv[t]  = *(const uint4*)(sr_sel4(p.w, c) + (size_t)row * (size_t)p.Ks + off);
        bsf[t] = q27_bf2f(sr_sel4(p.bs, c)[(size_t)(row >> 7) * (size_t)sr_sel4(p.bs_kblk, c) + (off >> 7)]);
    }
    // 2. Row max from the byte encodings (one decode per chunk).
    float mx = 0.f;
#pragma unroll
    for (int t = 0; t < NCH; ++t) {
        const int k0 = (t << 10) + (lane << 4);
        const float m = sr_fp8x16_absmax(wv[t], bsf[t]);
        mx = fmaxf(mx, (k0 < p.Ktot) ? m : 0.f);
    }
    mx = sr_wavemax(mx);
    const float sc  = mx * (1.0f / 127.0f);           // the engine's form (q27_k_fp8_to_i8g64), not mx/127
    const float inv = (sc > 0.f) ? 1.0f / sc : 0.f;   // an all-zero (or INF-max) row quantises to zeros
    if (lane == 0) rowscale[row] = sc;
    signed char* orow = out + (size_t)row * ldw_out;
#pragma unroll
    for (int t = 0; t < NCH; ++t) {
        const int k0 = (t << 10) + (lane << 4);
        if (k0 < p.Ktot) *(uint4*)(orow + k0) = sr_fp8x16_quant(wv[t], bsf[t], inv);
    }
}

// ---- block-scaled fp8 -> per-64 int8 MIRROR (the q27_i8g_t form the layer-local MLP arena holds) ----
// q27_k_fp8_to_i8g64 does one THREAD per 64-group with 64 scalar byte loads; that is fine at load time
// and wrong for a per-chunk transient. Here FOUR CONSECUTIVE LANES own one 64-group (16 bytes each, so a
// wave's load instruction covers 1 KB contiguous), the group max is a 2-step shuffle inside the quad,
// and each lane stores its own 16 int8. 64 | 128: one block scale per group. Out-of-range quads still
// execute the shuffles on a clamped chunk (the same trick as q27_k_swiglu_quant_b_g64) so no lane
// ever reads an inactive partner.
__global__ __launch_bounds__(256) void q27_k_sr_fp8bs_g64(sr_fp8_slices p, signed char* __restrict__ w8, float* __restrict__ s64)
{
    const int nchunk = p.Ktot >> 4;                                   // 16-byte chunks per row (multiple of 4)
    const size_t total = (size_t)p.rows * (size_t)nchunk;
    const size_t gt = (size_t)blockIdx.x * 256 + threadIdx.x;
    const bool ok = gt < total;
    const size_t ct = ok ? gt : (total - 1);
    const int row = (int)(ct / (size_t)nchunk);
    const int k0  = (int)(ct - (size_t)row * (size_t)nchunk) << 4;
    const int c   = (p.nsl > 1) ? ((k0 >= p.Ks) + (k0 >= 2 * p.Ks) + (k0 >= 3 * p.Ks)) : 0;
    const int off = k0 - c * p.Ks;
    const unsigned char*  wp = sr_sel4(p.w, c);
    const unsigned short* bp = sr_sel4(p.bs, c);
    const int kb = sr_sel4(p.bs_kblk, c);
    const uint4 wv  = *(const uint4*)(wp + (size_t)row * (size_t)p.Ks + off);
    const float bsf = q27_bf2f(bp[(size_t)(row >> 7) * (size_t)kb + (off >> 7)]);
    float mx = sr_fp8x16_absmax(wv, bsf);
    mx = fmaxf(mx, __shfl_xor(mx, 1, 4));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 4));
    const float sc  = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? 1.0f / sc : 0.f;
    if (!ok) return;
    if ((k0 & 63) == 0) s64[(size_t)row * (size_t)(p.Ktot >> 6) + (k0 >> 6)] = sc;
    *(uint4*)(w8 + (size_t)row * (size_t)p.Ktot + k0) = sr_fp8x16_quant(wv, bsf, inv);
}

// ---- per-64 int8 mirror -> per-row int8, out of place ---------------------------------------------
// Bit-identical to q27_k_i8g64_to_grp at GS = K: the row scale is the max of the per-64 scales (so it
// does not depend on the weight bytes at all), and the requantisation is (float)w * s64 * inv evaluated
// left to right, rintf, clamp. The max is taken from the scale plane alone, so a single streaming pass
// over the bytes suffices; NCH chunks are still hoisted so NCH loads are in flight per lane.
__device__ __forceinline__ unsigned sr_i8x4_requant(unsigned d, float s64, float inv) {
    const float v0 = (float)(int)(signed char)(unsigned char)(d & 0xFFu)         * s64 * inv;
    const float v1 = (float)(int)(signed char)(unsigned char)((d >> 8) & 0xFFu)  * s64 * inv;
    const float v2 = (float)(int)(signed char)(unsigned char)((d >> 16) & 0xFFu) * s64 * inv;
    const float v3 = (float)(int)(signed char)(unsigned char)(d >> 24)           * s64 * inv;
    return sr_pack4(sr_q8(v0), sr_q8(v1), sr_q8(v2), sr_q8(v3));
}
template <int NCH>
__global__ __launch_bounds__(256) void q27_k_sr_i8g64_row(const signed char* __restrict__ w, const float* __restrict__ sg,
                                                          int K, int ldw, int lds, int rows,
                                                          signed char* __restrict__ out, size_t ldw_out,
                                                          float* __restrict__ rowscale)
{
    const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
    const int row = blockIdx.x * SR_ROWS_PB + wave;
    if (row >= rows) return;
    const int NS = K >> 6;
    const signed char* wr = w + (size_t)row * (size_t)ldw;
    const float* sr = sg + (size_t)row * (size_t)lds;
    float mx = 0.f;
    for (int i = lane; i < NS; i += 64) mx = fmaxf(mx, sr[i]);
    mx = sr_wavemax(mx);
    const float s   = mx;
    const float inv = (s > 0.f) ? 1.f / s : 0.f;
    if (lane == 0) rowscale[row] = s;
    uint4 wv[NCH];
    float s64[NCH];
#pragma unroll
    for (int t = 0; t < NCH; ++t) {
        const int k0 = (t << 10) + (lane << 4);
        if (k0 < K) { wv[t] = *(const uint4*)(wr + k0); s64[t] = sr[k0 >> 6]; }   // 16 | 64: one scale per chunk
        else        { wv[t] = make_uint4(0u, 0u, 0u, 0u); s64[t] = 0.f; }
    }
    signed char* orow = out + (size_t)row * ldw_out;
#pragma unroll
    for (int t = 0; t < NCH; ++t) {
        const int k0 = (t << 10) + (lane << 4);
        if (k0 < K)
            *(uint4*)(orow + k0) = make_uint4(sr_i8x4_requant(wv[t].x, s64[t], inv), sr_i8x4_requant(wv[t].y, s64[t], inv),
                                              sr_i8x4_requant(wv[t].z, s64[t], inv), sr_i8x4_requant(wv[t].w, s64[t], inv));
    }
}

// ---- chunk-batched swiglu + per-16 int8 quantise, NATURAL code order ------------------------------
// q27_k_swiglu_quant_b (src/q27_nvfp4.hip) verbatim except for the byte placement: that kernel stores
// code j at dp = (j&1) ? 8+(j>>1) : (j>>1) because its consumer is the NVFP4 nibble reader; the SR
// decode consumers read int8 in natural K order, so code j goes to position j.
__global__ __launch_bounds__(256) void q27_k_sr_swiglu_quant_nat_b(
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
    unsigned d[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int cq = (int)rintf(v[j] * inv);
        cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
        d[j >> 2] |= ((unsigned)(cq & 0xFF)) << ((j & 3) * 8);
    }
    *(uint4*)(Q + base) = make_uint4(d[0], d[1], d[2], d[3]);
}

// ---- 4-way fp32 sum -------------------------------------------------------------------------------
// No __restrict__ on dst/a: dst may alias a (each element is read before it is written by the same
// thread, so aliasing is safe, but the compiler must not be told the two are disjoint).
__global__ __launch_bounds__(256) void q27_k_sr_sum4_v4(float* dst, const float* a, const float* b,
                                                        const float* c, const float* d, size_t n4, size_t tail)
{
    const size_t stride = (size_t)gridDim.x * 256;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += stride) {
        const float4 x = ((const float4*)a)[i], y = ((const float4*)b)[i];
        const float4 z = ((const float4*)c)[i], w = ((const float4*)d)[i];
        float4 r;
        r.x = ((x.x + y.x) + z.x) + w.x; r.y = ((x.y + y.y) + z.y) + w.y;
        r.z = ((x.z + y.z) + z.z) + w.z; r.w = ((x.w + y.w) + z.w) + w.w;
        ((float4*)dst)[i] = r;
    }
    if (blockIdx.x == 0 && threadIdx.x < tail) {   // tail < 4 elements after the float4 span
        const size_t i = n4 * 4 + threadIdx.x;
        dst[i] = ((a[i] + b[i]) + c[i]) + d[i];
    }
}
__global__ __launch_bounds__(256) void q27_k_sr_sum4_s(float* dst, const float* a, const float* b,
                                                       const float* c, const float* d, size_t n)
{
    const size_t stride = (size_t)gridDim.x * 256;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += stride)
        dst[i] = ((a[i] + b[i]) + c[i]) + d[i];
}

inline bool sr_aligned16(const void* p) { return (((uintptr_t)p) & 15u) == 0; }
inline bool sr_overlap(const void* a, size_t na, const void* b, size_t nb) {
    const uintptr_t a0 = (uintptr_t)a, b0 = (uintptr_t)b;
    return a0 < b0 + nb && b0 < a0 + na;
}
inline unsigned sr_grid_rows(int rows) { return (unsigned)((rows + SR_ROWS_PB - 1) / SR_ROWS_PB); }

// One instantiation per chunk count; K <= 17408 means NCH <= 17.
#define Q27_SR_NCH_CASES(NCH, LAUNCH) \
    switch (NCH) { \
        case 1:  LAUNCH(1);  break; case 2:  LAUNCH(2);  break; case 3:  LAUNCH(3);  break; case 4:  LAUNCH(4);  break; \
        case 5:  LAUNCH(5);  break; case 6:  LAUNCH(6);  break; case 7:  LAUNCH(7);  break; case 8:  LAUNCH(8);  break; \
        case 9:  LAUNCH(9);  break; case 10: LAUNCH(10); break; case 11: LAUNCH(11); break; case 12: LAUNCH(12); break; \
        case 13: LAUNCH(13); break; case 14: LAUNCH(14); break; case 15: LAUNCH(15); break; case 16: LAUNCH(16); break; \
        case 17: LAUNCH(17); break; default: return 0; }

int sr_launch_fp8(const sr_fp8_slices& p, signed char* w_out, size_t ldw_out, float* rowscale_out, hipStream_t s)
{
    const int nch = (p.Ktot + 1023) >> 10;
    const dim3 grid(sr_grid_rows(p.rows)), block(256);
#define Q27_SR_L(N) hipLaunchKernelGGL((q27_k_sr_fp8bs_row<N>), grid, block, 0, s, p, w_out, ldw_out, rowscale_out)
    Q27_SR_NCH_CASES(nch, Q27_SR_L)
#undef Q27_SR_L
    return 1;
}

}  // namespace

// ==================================================================================================
extern "C" int q27_sr_fp8bs_to_i8row(const q27_fp8_t* src, signed char* w_out, size_t ldw_out, float* rowscale_out, hipStream_t s)
{
    if (!src || !src->w || !src->bs || !w_out || !rowscale_out) return 0;
    const int K = src->K, rows = src->rows;
    if (rows < 1 || K < 128 || (K % 128) || K > SR_MAXK) return 0;
    if (src->bs_kblk != K / 128) return 0;                       // the plane must be this tensor's own, not a wider parent's
    if (ldw_out < (size_t)K || (ldw_out & 15u)) return 0;        // 16-byte stores at row stride
    if (!sr_aligned16(src->w) || !sr_aligned16(w_out)) return 0;
    if (sr_overlap(src->w, (size_t)rows * K, w_out, (size_t)rows * ldw_out)) return 0;   // out of place is the contract
    sr_fp8_slices p;
    for (int c = 0; c < 4; ++c) { p.w[c] = src->w; p.bs[c] = src->bs; p.bs_kblk[c] = src->bs_kblk; }
    p.nsl = 1; p.Ks = K; p.Ktot = K; p.rows = rows;
    return sr_launch_fp8(p, w_out, ldw_out, rowscale_out, s);
}

extern "C" int q27_sr_fp8bs4_to_i8row(const q27_fp8_t* const src4[4], signed char* w_out, float* rowscale_out, hipStream_t s)
{
    if (!src4 || !w_out || !rowscale_out) return 0;
    for (int c = 0; c < 4; ++c) if (!src4[c] || !src4[c]->w || !src4[c]->bs) return 0;
    const int Ks = src4[0]->K, rows = src4[0]->rows;
    if (rows < 1 || Ks < 128 || (Ks % 128) || 4 * Ks > SR_MAXK) return 0;
    const size_t ldw_out = (size_t)4 * Ks;
    sr_fp8_slices p;
    for (int c = 0; c < 4; ++c) {
        const q27_fp8_t* t = src4[c];
        if (t->K != Ks || t->rows != rows || t->bs_kblk != Ks / 128) return 0;   // one logical tensor: all slices agree
        if (!sr_aligned16(t->w)) return 0;
        if (sr_overlap(t->w, (size_t)rows * Ks, w_out, (size_t)rows * ldw_out)) return 0;
        p.w[c] = t->w; p.bs[c] = t->bs; p.bs_kblk[c] = t->bs_kblk;
    }
    if (!sr_aligned16(w_out)) return 0;
    p.nsl = 4; p.Ks = Ks; p.Ktot = 4 * Ks; p.rows = rows;
    return sr_launch_fp8(p, w_out, ldw_out, rowscale_out, s);
}

extern "C" int q27_sr_i8g64_to_i8row(const q27_i8g_t* src, signed char* w_out, size_t ldw_out, float* rowscale_out, hipStream_t s)
{
    if (!src || !src->w || !src->s || !w_out || !rowscale_out) return 0;
    const int K = src->K, rows = src->rows;
    if (rows < 1 || K < 64 || (K % 64) || K > SR_MAXK) return 0;
    const int ldw = src->ldw ? src->ldw : K;            // K-view strides: 0 = dense at this tensor's width
    const int lds = src->lds ? src->lds : (K >> 6);
    if (ldw < K || (ldw & 15) || lds < (K >> 6)) return 0;   // a view narrower than its own K is impossible
    if (ldw_out < (size_t)K || (ldw_out & 15u)) return 0;
    if (!sr_aligned16(src->w) || !sr_aligned16(w_out)) return 0;
    if (sr_overlap(src->w, (size_t)rows * ldw, w_out, (size_t)rows * ldw_out)) return 0;   // in place is q27_i8g64_rowify's job
    const int nch = (K + 1023) >> 10;
    const dim3 grid(sr_grid_rows(rows)), block(256);
#define Q27_SR_L(N) hipLaunchKernelGGL((q27_k_sr_i8g64_row<N>), grid, block, 0, s, src->w, src->s, K, ldw, lds, rows, w_out, ldw_out, rowscale_out)
    Q27_SR_NCH_CASES(nch, Q27_SR_L)
#undef Q27_SR_L
    return 1;
}

extern "C" int q27_sr_swiglu_quant_nat_b(const float* gate, const float* up, signed char* xq, float* xs,
                                         int n, float in_scale, int C, hipStream_t s)
{
    if (!gate || !up || !xq || !xs) return 0;
    if (n <= 0 || (n & 15) || !(in_scale > 0.f) || C < 1) return 0;       // q27_swiglu_quant_b's contract
    if (((uintptr_t)gate | (uintptr_t)up | (uintptr_t)xq) & 15u) return 0;
    const int ngrp = n >> 4;
    hipLaunchKernelGGL(q27_k_sr_swiglu_quant_nat_b, dim3((ngrp + 255) / 256, C), dim3(256), 0, s,
                       gate, up, xq, xs, ngrp, 1.0f / in_scale, (size_t)n, (size_t)n, (size_t)ngrp);
    return 1;
}

extern "C" int q27_sr_sum4_f32(float* dst, const float* a, const float* b, const float* c, const float* d, size_t n, hipStream_t s)
{
    if (!dst || !a || !b || !c || !d) return 0;
    if (n == 0) return 1;
    const bool v4 = sr_aligned16(dst) && sr_aligned16(a) && sr_aligned16(b) && sr_aligned16(c) && sr_aligned16(d);
    if (v4) {
        const size_t n4 = n >> 2, tail = n & 3;
        const size_t want = (n4 + 255) / 256;
        const unsigned grid = (unsigned)(want < 1 ? 1 : (want > 4096 ? 4096 : want));
        hipLaunchKernelGGL(q27_k_sr_sum4_v4, dim3(grid), dim3(256), 0, s, dst, a, b, c, d, n4, tail);
    } else {
        const size_t want = (n + 255) / 256;
        const unsigned grid = (unsigned)(want > 4096 ? 4096 : want);
        hipLaunchKernelGGL(q27_k_sr_sum4_s, dim3(grid), dim3(256), 0, s, dst, a, b, c, d, n);
    }
    return 1;
}

namespace {
// Shared by (f) and (g): stamp the descriptor exactly as q27_nvfp4_to_i8g64_into does (the arena consumer
// reads every field, so ldw/lds MUST be written to 0, never left as the caller's stack contents), then
// launch the quad-per-group kernel. alpha carries wscale even though a block-scaled tensor's wscale is
// 1.0: that keeps this the same formula as q27_fp8_to_i8g64_conv and nothing else to remember.
int sr_launch_g64(const sr_fp8_slices& p, float alpha, q27_i8g_t* dst, signed char* w8, float* s64, hipStream_t s)
{
    dst->rows = p.rows; dst->K = p.Ktot; dst->alpha = alpha;
    dst->ldw = 0; dst->lds = 0;
    const size_t total = (size_t)p.rows * (size_t)(p.Ktot >> 4);
    hipLaunchKernelGGL(q27_k_sr_fp8bs_g64, dim3((unsigned)((total + 255) / 256)), dim3(256), 0, s, p, w8, s64);
    dst->w = w8; dst->s = s64;
    return 1;
}
}  // namespace

extern "C" int q27_sr_fp8bs_to_i8g64_into(const q27_fp8_t* src, q27_i8g_t* dst, signed char* w8, float* s64, hipStream_t s)
{
    if (!src || !src->w || !src->bs || !dst || !w8 || !s64) return 0;
    const int K = src->K, rows = src->rows;
    if (rows < 1 || K < 128 || (K % 128)) return 0;
    if (src->bs_kblk != K / 128) return 0;
    if (!sr_aligned16(src->w) || !sr_aligned16(w8)) return 0;
    if (sr_overlap(src->w, (size_t)rows * K, w8, (size_t)rows * K)) return 0;
    sr_fp8_slices p;
    for (int c = 0; c < 4; ++c) { p.w[c] = src->w; p.bs[c] = src->bs; p.bs_kblk[c] = src->bs_kblk; }
    p.nsl = 1; p.Ks = K; p.Ktot = K; p.rows = rows;
    return sr_launch_g64(p, src->in_scale * src->wscale, dst, w8, s64, s);
}

extern "C" int q27_sr_fp8bs4_to_i8g64_into(const q27_fp8_t* const src4[4], q27_i8g_t* dst, signed char* w8, float* s64, hipStream_t s)
{
    if (!src4 || !dst || !w8 || !s64) return 0;
    for (int c = 0; c < 4; ++c) if (!src4[c] || !src4[c]->w || !src4[c]->bs) return 0;
    const int Ks = src4[0]->K, rows = src4[0]->rows;
    if (rows < 1 || Ks < 128 || (Ks % 128)) return 0;
    const size_t Ktot = (size_t)4 * Ks;
    sr_fp8_slices p;
    for (int c = 0; c < 4; ++c) {
        const q27_fp8_t* t = src4[c];
        if (t->K != Ks || t->rows != rows || t->bs_kblk != Ks / 128) return 0;
        if (!sr_aligned16(t->w)) return 0;
        if (sr_overlap(t->w, (size_t)rows * Ks, w8, (size_t)rows * Ktot)) return 0;
        p.w[c] = t->w; p.bs[c] = t->bs; p.bs_kblk[c] = t->bs_kblk;
    }
    if (!sr_aligned16(w8)) return 0;
    p.nsl = 4; p.Ks = Ks; p.Ktot = (int)Ktot; p.rows = rows;
    return sr_launch_g64(p, src4[0]->in_scale * src4[0]->wscale, dst, w8, s64, s);
}
