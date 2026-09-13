// q27_elem.hip — elementwise, norm and reduction kernels for the Qwen3.8-27B gfx906 engine.
//
// Entry points (declared in q27.h):
//   q27_rmsnorm       contract §2  — rsqrt(Σx²/N + 1e-6), fp32 accumulate, BF16 RNE out,
//                                    weight applied as (1+W) except linear_attn.norm (RAW W).
//   q27_add_inplace   contract §1  — the two residual adds, BF16 h += fp32 mixer output.
//   q27_swiglu        contract §1  — SiLU on the GATE operand, times up.
//   q27_embed_gather  contract §1  — one BF16 row of [248320, 5120].
//   q27_argmax        contract §7  — non-finite filtered, ties broken by LOWEST index.
//
// Design notes, in the estate's terms:
//  * These run 64+ times per token and move kilobytes, so they are LAUNCH bound, not bandwidth
//    bound. Everything here is one or two launches on the caller's stream, zero allocations,
//    zero host synchronisation, zero hipMemset. argmax's inter-block scratch is a file-scope
//    __device__ arena, not a per-token allocation.
//  * Dependency graph over instruction count: every kernel issues all of its independent global
//    loads before the first use, so the wave takes one wait instead of a chain of them. RMSNorm
//    holds BOTH x and the norm weight in registers across the block reduction — the weight load
//    does not depend on the reduction, so it must not sit behind it.
//  * Nothing is staged in LDS except the per-wave reduction partials, which have many consumers.
//    Activations have exactly one consumer per element per block and are kept in registers.
//  * wave64 throughout: __shfl_xor(v, off, 64) with off up to 32 crosses the whole wave
//    (ds_bpermute), unlike ds_swizzle.
#include "q27.h"
#include <cstdlib>

#define Q27_WAVE    64
#define Q27_EL_BLK  256          // elementwise block width
#define Q27_AM_BLK  256          // argmax block width
#define Q27_AM_GRID 256          // argmax stage-1 blocks == stage-2 block width

typedef unsigned short q27_u16;

// ============================== reductions (wave64) ==============================

// Butterfly: the total lands in EVERY lane, so no broadcast and no second sync.
__device__ __forceinline__ float q27_wave_sum(float v) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, Q27_WAVE);
    return v;
}

// One __syncthreads for the whole block reduction. The final fan-in reads the per-wave partials
// from LDS with every lane at the same address (LDS broadcast), so all threads leave with the sum.
// Call at most once per kernel, or add a __syncthreads before reusing `sm`.
template<int BLOCK>
__device__ __forceinline__ float q27_block_sum(float v, float* sm) {
    v = q27_wave_sum(v);
    if (BLOCK > Q27_WAVE) {
        const int nw = BLOCK / Q27_WAVE;
        if ((threadIdx.x & (Q27_WAVE - 1)) == 0) sm[threadIdx.x >> 6] = v;
        __syncthreads();
        float t = 0.0f;
#pragma unroll
        for (int k = 0; k < nw; ++k) t += sm[k];
        v = t;
    }
    return v;
}

// ============================== RMSNorm (contract §2) ==============================
//
// plus == 1.0f  -> weight is (1 + W): input_layernorm, post_attention_layernorm, model.norm,
//                  q_norm, k_norm.
// plus == 0.0f  -> RAW W: linear_attn.norm ONLY (mean +0.869, not zero-centred; applying (1+W)
//                  there would roughly double the mixer output of 48 of 64 layers).
//
// One block, so no grid synchronisation and no second launch. The widths in play (128, 256,
// 5120, 17408) are 10–35 KB of traffic; a single workgroup of 16 waves covers the latency and a
// second launch would cost more than the extra CUs would save.

// Vectorised path: BLOCK*VPT*4 >= n, x and w cached in registers across the reduction.
template<int BLOCK, int VPT>
__global__ __launch_bounds__(BLOCK) void q27_k_rmsnorm_v4(
        const ushort4* __restrict__ x, const ushort4* __restrict__ w,
        ushort4* __restrict__ y, int nv, float rn, float plus)
{
    __shared__ float sm[BLOCK / Q27_WAVE];
    const int t = threadIdx.x;

    ushort4 xr[VPT], wr[VPT];
    // Issue every load, then wait once. x and w are independent of each other and of the sum.
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int i = t + k * BLOCK;
        if (i < nv) { xr[k] = x[i]; wr[k] = w[i]; }
    }

    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        if (t + k * BLOCK < nv) {
            const float a = q27_bf2f(xr[k].x), b = q27_bf2f(xr[k].y);
            const float c = q27_bf2f(xr[k].z), d = q27_bf2f(xr[k].w);
            ss = fmaf(a, a, fmaf(b, b, fmaf(c, c, fmaf(d, d, ss))));   // fp32 accumulate
        }
    }

    const float inv = rsqrtf(q27_block_sum<BLOCK>(ss, sm) * rn + Q27_RMS_EPS);

#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int i = t + k * BLOCK;
        if (i < nv) {
            ushort4 o;
            o.x = q27_f2bf(q27_bf2f(xr[k].x) * inv * (plus + q27_bf2f(wr[k].x)));
            o.y = q27_f2bf(q27_bf2f(xr[k].y) * inv * (plus + q27_bf2f(wr[k].y)));
            o.z = q27_f2bf(q27_bf2f(xr[k].z) * inv * (plus + q27_bf2f(wr[k].z)));
            o.w = q27_f2bf(q27_bf2f(xr[k].w) * inv * (plus + q27_bf2f(wr[k].w)));
            y[i] = o;
        }
    }
}

// Generic path: any n, any alignment. x is re-read in pass 2 (it is resident in L1 by then).
template<int BLOCK>
__global__ __launch_bounds__(BLOCK) void q27_k_rmsnorm_gen(
        const q27_u16* __restrict__ x, const q27_u16* __restrict__ w,
        q27_u16* __restrict__ y, int n, float rn, float plus)
{
    __shared__ float sm[BLOCK / Q27_WAVE];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += BLOCK) { const float a = q27_bf2f(x[i]); ss = fmaf(a, a, ss); }
    const float inv = rsqrtf(q27_block_sum<BLOCK>(ss, sm) * rn + Q27_RMS_EPS);
    for (int i = threadIdx.x; i < n; i += BLOCK)
        y[i] = q27_f2bf(q27_bf2f(x[i]) * inv * (plus + q27_bf2f(w[i])));
}

// ============================== residual add ==============================
// h (BF16, in place) += m (fp32 mixer / MLP output).

__global__ __launch_bounds__(Q27_EL_BLK) void q27_k_add_v4(
        ushort4* __restrict__ h, const float4* __restrict__ m, int nv)
{
    const int i = blockIdx.x * Q27_EL_BLK + threadIdx.x;
    if (i >= nv) return;
    const ushort4 a = h[i];          // two independent loads issued,
    const float4  b = m[i];          // one wait
    ushort4 o;
    o.x = q27_f2bf(q27_bf2f(a.x) + b.x);
    o.y = q27_f2bf(q27_bf2f(a.y) + b.y);
    o.z = q27_f2bf(q27_bf2f(a.z) + b.z);
    o.w = q27_f2bf(q27_bf2f(a.w) + b.w);
    h[i] = o;
}

__global__ __launch_bounds__(Q27_EL_BLK) void q27_k_add_gen(
        q27_u16* __restrict__ h, const float* __restrict__ m, int n)
{
    const int i = blockIdx.x * Q27_EL_BLK + threadIdx.x;
    if (i >= n) return;
    h[i] = q27_f2bf(q27_bf2f(h[i]) + m[i]);
}

// ============================== SwiGLU ==============================
// out = SiLU(gate) * up.  SiLU is on the GATE operand (contract §1: `a = SiLU(a) * b`).
// Not sigmoid, not on `up`. Contrast full attention's output gate, which IS plain sigmoid (§3),
// and gated delta's, which is swish on z (§4).
// v_rcp_f32 (~1 ulp), NOT `1.0f/x`. Plain division lowers to the IEEE sequence
// v_div_scale / v_rcp / v_div_fmas / v_div_fixup — 5 instructions per element instead of 1, and
// measured in the ISA it bloated the gate arithmetic enough that the scheduler SANK the `up`
// load below it, exposing its full latency. The result is rounded to BF16 (8-bit mantissa), so
// the exact division buys nothing. inf-safe as before: rcp(inf) = 0, g*0 = -0.
__device__ __forceinline__ float q27_silu(float g) {
    return g * __builtin_amdgcn_rcpf(1.0f + __expf(-g));
}

__global__ __launch_bounds__(Q27_EL_BLK) void q27_k_swiglu_v4(
        const float4* __restrict__ g, const float4* __restrict__ u,
        ushort4* __restrict__ o, int nv)
{
    const int i = blockIdx.x * Q27_EL_BLK + threadIdx.x;
    if (i >= nv) return;
    const float4 a = g[i];           // both loads issued before either is used
    const float4 b = u[i];
    // Transcendental first, on the gate only: the four v_exp_f32 are independent of `b`, so the
    // wait on the `up` load overlaps them.
    const float sx = q27_silu(a.x), sy = q27_silu(a.y);
    const float sz = q27_silu(a.z), sw = q27_silu(a.w);
    ushort4 r;
    r.x = q27_f2bf(sx * b.x);
    r.y = q27_f2bf(sy * b.y);
    r.z = q27_f2bf(sz * b.z);
    r.w = q27_f2bf(sw * b.w);
    o[i] = r;
}

__global__ __launch_bounds__(Q27_EL_BLK) void q27_k_swiglu_gen(
        const float* __restrict__ g, const float* __restrict__ u,
        q27_u16* __restrict__ o, int n)
{
    const int i = blockIdx.x * Q27_EL_BLK + threadIdx.x;
    if (i >= n) return;
    const float a = g[i], b = u[i];
    o[i] = q27_f2bf(q27_silu(a) * b);
}

// ============================== embedding gather ==============================
// One BF16 row of [Q27_VOCAB, Q27_HID]. Row stride is 10240 bytes, so a row base inherits the
// table's alignment; the vector path only needs the table itself to be 8B aligned.

__global__ __launch_bounds__(Q27_EL_BLK) void q27_k_embed_v4(
        const ushort4* __restrict__ tbl, ushort4* __restrict__ out, unsigned long long base)
{
    const int i = blockIdx.x * Q27_EL_BLK + threadIdx.x;
    if (i < (Q27_HID >> 2)) out[i] = tbl[base + (unsigned long long)i];
}

__global__ __launch_bounds__(Q27_EL_BLK) void q27_k_embed_gen(
        const q27_u16* __restrict__ tbl, q27_u16* __restrict__ out, unsigned long long base)
{
    const int i = blockIdx.x * Q27_EL_BLK + threadIdx.x;
    if (i < Q27_HID) out[i] = tbl[base + (unsigned long long)i];
}

// ============================== argmax (contract §7) ==============================
// 248320 fp32 logits. Non-finite entries are filtered out; ties go to the LOWEST index.
// Two stages on the caller's stream — no host sync between them. Inter-stage scratch is a
// file-scope device arena (process-global, no per-token allocation). Consequence: q27_argmax is
// not re-entrant across concurrent streams; the engine is single-user by construction.
__device__ float        g_q27_am_val[Q27_AM_GRID];
__device__ unsigned int g_q27_am_idx[Q27_AM_GRID];

#define Q27_AM_NONE 0xFFFFFFFFu

__device__ __forceinline__ bool q27_finite(float v) {
    return (__float_as_uint(v) & 0x7F800000u) != 0x7F800000u;   // catches +-inf and NaN
}

// Total order on (value desc, index asc) -> the merge is associative and commutative, so the
// butterfly below is order independent and every lane ends with the same winner.
__device__ __forceinline__ void q27_am_merge(float& bv, unsigned int& bi, float v, unsigned int i) {
    const bool take = (v > bv) || (v == bv && i < bi);
    bv = take ? v : bv;
    bi = take ? i : bi;
}
__device__ __forceinline__ void q27_am_cand(float& bv, unsigned int& bi, float v, unsigned int i) {
    if (q27_finite(v)) q27_am_merge(bv, bi, v, i);
}

template<int BLOCK>
__device__ __forceinline__ void q27_block_amax(float& bv, unsigned int& bi,
                                               float* sv, unsigned int* si) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        const float        ov = __shfl_xor(bv, off, Q27_WAVE);
        const unsigned int oi = __shfl_xor(bi, off, Q27_WAVE);
        q27_am_merge(bv, bi, ov, oi);
    }
    if (BLOCK > Q27_WAVE) {
        const int nw = BLOCK / Q27_WAVE;
        if ((threadIdx.x & (Q27_WAVE - 1)) == 0) {
            sv[threadIdx.x >> 6] = bv;
            si[threadIdx.x >> 6] = bi;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < nw; ++k) q27_am_merge(bv, bi, sv[k], si[k]);   // self-merge is a no-op
    }
}

template<int BLOCK, bool VEC>
__global__ __launch_bounds__(BLOCK) void q27_k_amax1(const float* __restrict__ p, int n) {
    __shared__ float        sv[BLOCK / Q27_WAVE];
    __shared__ unsigned int si[BLOCK / Q27_WAVE];

    float        bv = -INFINITY;
    unsigned int bi = Q27_AM_NONE;
    const int gid    = blockIdx.x * BLOCK + threadIdx.x;
    const int stride = gridDim.x * BLOCK;

    if (VEC) {
        // One dwordx4 per thread per step: four candidates from a single load, so the loop body
        // has one wait rather than four dependent ones.
        const float4* q = (const float4*)p;
        const int nv = n >> 2;
        for (int i = gid; i < nv; i += stride) {
            const float4 f = q[i];
            const unsigned int b = (unsigned int)i << 2;
            q27_am_cand(bv, bi, f.x, b);
            q27_am_cand(bv, bi, f.y, b + 1u);
            q27_am_cand(bv, bi, f.z, b + 2u);
            q27_am_cand(bv, bi, f.w, b + 3u);
        }
        for (int i = (nv << 2) + gid; i < n; i += stride) q27_am_cand(bv, bi, p[i], (unsigned int)i);
    } else {
        for (int i = gid; i < n; i += stride) q27_am_cand(bv, bi, p[i], (unsigned int)i);
    }

    q27_block_amax<BLOCK>(bv, bi, sv, si);
    if (threadIdx.x == 0) { g_q27_am_val[blockIdx.x] = bv; g_q27_am_idx[blockIdx.x] = bi; }
}

// Stage 2 that also emits the winning VALUE.  The tensor-parallel head needs it: each card owns
// 62080 of the 248320 logits, and the four cards settle the token with a 4-way max over
// (value, GLOBAL index) -- 4x8 bytes instead of an all-gather of 993 KB.
__global__ __launch_bounds__(Q27_AM_GRID) void q27_k_amax2_val(unsigned int* __restrict__ out,
                                                               float* __restrict__ oval, int nblk) {
    __shared__ float        sv[Q27_AM_GRID / Q27_WAVE];
    __shared__ unsigned int si[Q27_AM_GRID / Q27_WAVE];
    float        bv = -INFINITY;
    unsigned int bi = Q27_AM_NONE;
    const int t = threadIdx.x;
    if (t < nblk) { bv = g_q27_am_val[t]; bi = g_q27_am_idx[t]; }
    q27_block_amax<Q27_AM_GRID>(bv, bi, sv, si);
    if (t == 0) {
        const bool none = (bi == Q27_AM_NONE);
        *out  = none ? 0u : bi;
        *oval = none ? -INFINITY : bv;      // a card with no finite logit can never win the 4-way
    }
}

__global__ __launch_bounds__(Q27_AM_GRID) void q27_k_amax2(unsigned int* __restrict__ out, int nblk) {
    __shared__ float        sv[Q27_AM_GRID / Q27_WAVE];
    __shared__ unsigned int si[Q27_AM_GRID / Q27_WAVE];
    float        bv = -INFINITY;
    unsigned int bi = Q27_AM_NONE;
    const int t = threadIdx.x;
    if (t < nblk) { bv = g_q27_am_val[t]; bi = g_q27_am_idx[t]; }
    q27_block_amax<Q27_AM_GRID>(bv, bi, sv, si);
    if (t == 0) *out = (bi == Q27_AM_NONE) ? 0u : bi;   // all-non-finite -> token 0, never garbage
}

// ============================== host entry points ==============================

static __forceinline__ bool q27_aligned(const void* a, const void* b, const void* c, unsigned m) {
    return ((((uintptr_t)a) | ((uintptr_t)b) | ((uintptr_t)c)) & (uintptr_t)m) == 0;
}

extern "C" {

void q27_rmsnorm(const unsigned short* x, const unsigned short* wgt, unsigned short* y,
                 int n, int plus_one, hipStream_t s)
{
    if (n <= 0) return;
    const float rn   = 1.0f / (float)n;
    const float plus = plus_one ? 1.0f : 0.0f;      // (1+W) everywhere except linear_attn.norm

    if (((n & 3) == 0) && q27_aligned(x, wgt, y, 7u)) {
        const int      nv = n >> 2;
        const ushort4* xv = (const ushort4*)x;
        const ushort4* wv = (const ushort4*)wgt;
        ushort4*       yv = (ushort4*)y;
        // Buckets sized so VPT is tight for the widths the model actually uses:
        //   n=128/256 -> nv 32/64 ; n=5120 -> nv 1280 (256x5) ; n=17408 -> nv 4352 (1024x5)
        if (nv <= 64) {
            q27_k_rmsnorm_v4<64, 1><<<1, 64, 0, s>>>(xv, wv, yv, nv, rn, plus);   return;
        } else if (nv <= 256) {
            q27_k_rmsnorm_v4<64, 4><<<1, 64, 0, s>>>(xv, wv, yv, nv, rn, plus);   return;
        } else if (nv <= 1280) {
            q27_k_rmsnorm_v4<256, 5><<<1, 256, 0, s>>>(xv, wv, yv, nv, rn, plus); return;
        } else if (nv <= 5120) {
            q27_k_rmsnorm_v4<1024, 5><<<1, 1024, 0, s>>>(xv, wv, yv, nv, rn, plus); return;
        }
    }
    q27_k_rmsnorm_gen<1024><<<1, 1024, 0, s>>>(x, wgt, y, n, rn, plus);
}

void q27_add_inplace(unsigned short* h, const float* m, int n, hipStream_t s)
{
    if (n <= 0) return;
    if (((n & 3) == 0) && (((uintptr_t)h & 7u) == 0) && (((uintptr_t)m & 15u) == 0)) {
        const int nv = n >> 2;
        q27_k_add_v4<<<(nv + Q27_EL_BLK - 1) / Q27_EL_BLK, Q27_EL_BLK, 0, s>>>(
            (ushort4*)h, (const float4*)m, nv);
        return;
    }
    q27_k_add_gen<<<(n + Q27_EL_BLK - 1) / Q27_EL_BLK, Q27_EL_BLK, 0, s>>>(h, m, n);
}

void q27_swiglu(const float* gate, const float* up, unsigned short* out, int n, hipStream_t s)
{
    if (n <= 0) return;
    if (((n & 3) == 0) && (((uintptr_t)gate | (uintptr_t)up) & 15u) == 0 &&
        (((uintptr_t)out & 7u) == 0)) {
        const int nv = n >> 2;
        q27_k_swiglu_v4<<<(nv + Q27_EL_BLK - 1) / Q27_EL_BLK, Q27_EL_BLK, 0, s>>>(
            (const float4*)gate, (const float4*)up, (ushort4*)out, nv);
        return;
    }
    q27_k_swiglu_gen<<<(n + Q27_EL_BLK - 1) / Q27_EL_BLK, Q27_EL_BLK, 0, s>>>(gate, up, out, n);
}

void q27_embed_gather(const unsigned short* tbl, unsigned tok, unsigned short* out, hipStream_t s)
{
    // Clamp rather than fault: a bad id must not take the process down mid-stream.
    const unsigned long long id = (tok < (unsigned)Q27_VOCAB) ? (unsigned long long)tok
                                                              : (unsigned long long)(Q27_VOCAB - 1);
    if ((((uintptr_t)tbl | (uintptr_t)out) & 7u) == 0) {
        const int nv = Q27_HID >> 2;
        q27_k_embed_v4<<<(nv + Q27_EL_BLK - 1) / Q27_EL_BLK, Q27_EL_BLK, 0, s>>>(
            (const ushort4*)tbl, (ushort4*)out, id * (unsigned long long)nv);
        return;
    }
    q27_k_embed_gen<<<(Q27_HID + Q27_EL_BLK - 1) / Q27_EL_BLK, Q27_EL_BLK, 0, s>>>(
        tbl, out, id * (unsigned long long)Q27_HID);
}

void q27_argmax(const float* logits, unsigned* out, int n, hipStream_t s)
{
    if (n <= 0) return;
    const bool vec = ((n >> 2) > 0) && ((((uintptr_t)logits) & 15u) == 0);
    // Cover n with as few blocks as possible, capped at the scratch arena.
    const int units = vec ? (n >> 2) : n;
    int grid = (units + Q27_AM_BLK - 1) / Q27_AM_BLK;
    if (grid > Q27_AM_GRID) grid = Q27_AM_GRID;
    if (grid < 1) grid = 1;

    if (vec) q27_k_amax1<Q27_AM_BLK, true ><<<grid, Q27_AM_BLK, 0, s>>>(logits, n);
    else     q27_k_amax1<Q27_AM_BLK, false><<<grid, Q27_AM_BLK, 0, s>>>(logits, n);
    q27_k_amax2<<<1, Q27_AM_GRID, 0, s>>>(out, grid);
}

// Same reduction, but the winning value comes back too (see q27_k_amax2_val).
void q27_argmax_val(const float* logits, unsigned* out_idx, float* out_val, int n, hipStream_t s)
{
    if (n <= 0) return;
    const bool vec = ((n >> 2) > 0) && ((((uintptr_t)logits) & 15u) == 0);
    const int units = vec ? (n >> 2) : n;
    int grid = (units + Q27_AM_BLK - 1) / Q27_AM_BLK;
    if (grid > Q27_AM_GRID) grid = Q27_AM_GRID;
    if (grid < 1) grid = 1;

    if (vec) q27_k_amax1<Q27_AM_BLK, true ><<<grid, Q27_AM_BLK, 0, s>>>(logits, n);
    else     q27_k_amax1<Q27_AM_BLK, false><<<grid, Q27_AM_BLK, 0, s>>>(logits, n);
    q27_k_amax2_val<<<1, Q27_AM_GRID, 0, s>>>(out_idx, out_val, grid);
}

}   // extern "C"

// ================= FUSED RMSNORM + QUANTIZE =================================================
// Measured on the first TP authority (a8e3289): rmsnorm 6.7% and quantize 10.5% of TP compute,
// and every rmsnorm in the layer is IMMEDIATELY followed by a quantize of its own output. Both
// are single-block, tiny-vector, latency-bound kernels at M=1 on a quarter-width shard, so the
// pair costs two launches and a full 10 KiB round trip through HBM for arithmetic that is already
// sitting in registers.
//
// This fuses them. The normalised bf16 goes to LDS (10 KiB for Q27_HID) instead of being re-read
// from global, and the same block quantises it. y is still written because the GDN layers' two
// bf16 GEMVs read the normalised vector directly.
//
// MODE 0 = fp8 contract   : scale = mx*in_scale/127, codes in natural order.
// MODE 1 = nvfp4 contract : values pre-scaled by 1/in_scale, scale = mx/127, and the codes are
//                           written EVEN-THEN-ODD within each group of 16 per the ACTIVATION
//                           PERMUTATION CONTRACT in q27.h. The permutation is produced HERE, in
//                           the quantizer, exactly as before -- never in the consumer.
// ADD is folded in too. Every q27_add_inplace in the layer is IMMEDIATELY followed by an rmsnorm
// of the buffer it just wrote -- residual 1 feeds post_norm, residual 2 feeds the next layer's
// input_norm. 128 launches per token that exist only as a kernel boundary, each a full read-modify-
// write of the residual through HBM. Folded here it is two extra instructions on a value already
// being loaded. `a` null keeps the plain norm.
template<int BLOCK, int VPT, int MODE>
__global__ __launch_bounds__(BLOCK) void q27_k_rmsnorm_quant(
        const ushort4* __restrict__ x, const ushort4* __restrict__ w, ushort4* __restrict__ y,
        signed char* __restrict__ q, float* __restrict__ qs,
        int nv, float rn, float plus, float sc_in, const void* __restrict__ a,
        ushort4* __restrict__ xw, int abf)
{
    __shared__ float   sm[BLOCK / Q27_WAVE];
    // Declared as ushort4, NOT unsigned short[]: a __shared__ array of unsigned short is only
    // 2-byte aligned, so storing a ushort4 through a cast of it is an under-aligned 8-byte store.
    __shared__ ushort4 ldsv[BLOCK * VPT];
    const unsigned short* const lds = (const unsigned short*)ldsv;
    const int t = threadIdx.x;

    ushort4 xr[VPT], wr[VPT];
    float4  ar[VPT];
    ushort4 br[VPT];
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int i = t + k * BLOCK;
        if (i < nv) {
            xr[k] = x[i]; wr[k] = w[i];
            if (a) { if (abf) br[k] = ((const ushort4*)a)[i]; else ar[k] = ((const float4*)a)[i]; }
        }
    }
    if (a) {                                   // residual add, folded
#pragma unroll
        for (int k = 0; k < VPT; ++k) {
            const int i = t + k * BLOCK;
            if (i < nv) {
                const float m0 = abf ? q27_bf2f(br[k].x) : ar[k].x;
                const float m1 = abf ? q27_bf2f(br[k].y) : ar[k].y;
                const float m2 = abf ? q27_bf2f(br[k].z) : ar[k].z;
                const float m3 = abf ? q27_bf2f(br[k].w) : ar[k].w;
                xr[k].x = q27_f2bf(q27_bf2f(xr[k].x) + m0);
                xr[k].y = q27_f2bf(q27_bf2f(xr[k].y) + m1);
                xr[k].z = q27_f2bf(q27_bf2f(xr[k].z) + m2);
                xr[k].w = q27_f2bf(q27_bf2f(xr[k].w) + m3);
                xw[i] = xr[k];                 // the residual stream must persist
            }
        }
    }

    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        if (t + k * BLOCK < nv) {
            const float a = q27_bf2f(xr[k].x), b = q27_bf2f(xr[k].y);
            const float c = q27_bf2f(xr[k].z), d = q27_bf2f(xr[k].w);
            ss = fmaf(a, a, fmaf(b, b, fmaf(c, c, fmaf(d, d, ss))));
        }
    }
    const float inv = rsqrtf(q27_block_sum<BLOCK>(ss, sm) * rn + Q27_RMS_EPS);

#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        const int i = t + k * BLOCK;
        if (i < nv) {
            ushort4 o;
            o.x = q27_f2bf(q27_bf2f(xr[k].x) * inv * (plus + q27_bf2f(wr[k].x)));
            o.y = q27_f2bf(q27_bf2f(xr[k].y) * inv * (plus + q27_bf2f(wr[k].y)));
            o.z = q27_f2bf(q27_bf2f(xr[k].z) * inv * (plus + q27_bf2f(wr[k].z)));
            o.w = q27_f2bf(q27_bf2f(xr[k].w) * inv * (plus + q27_bf2f(wr[k].w)));
            y[i] = o;
            ldsv[i] = o;
        }
    }
    __syncthreads();

    const int ngrp = nv >> 2;                       // nv ushort4 = nv*4 scalars = nv/4 groups of 16
    for (int gi = t; gi < ngrp; gi += BLOCK) {
        const unsigned short* src = lds + ((size_t)gi << 4);
        float v[16];
        float mx = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = q27_bf2f(src[j]) * ((MODE == 1) ? sc_in : 1.0f);
            mx = fmaxf(mx, fabsf(v[j]));
        }
        float sc, qinv;
        if (MODE == 1) {                            // nvfp4: sc_in is 1/in_scale, applied above
            sc   = mx * (1.0f / 127.0f);
            qinv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        } else {                                    // fp8: sc_in is in_scale; it cancels by design
            sc   = mx * sc_in * (1.0f / 127.0f);
            qinv = (sc > 0.0f) ? (sc_in / sc) : 0.0f;
        }
        qs[gi] = sc;

        unsigned d0 = 0, d1 = 0, d2 = 0, d3 = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int c = (int)rintf(v[j] * qinv);
            c = c < -127 ? -127 : (c > 127 ? 127 : c);
            const unsigned byte = (unsigned)(c & 0xFF);
            const int dpos = (MODE == 1) ? ((j & 1) ? (8 + (j >> 1)) : (j >> 1)) : j;
            const int sh   = (dpos & 3) * 8;
            if      ((dpos >> 2) == 0) d0 |= byte << sh;
            else if ((dpos >> 2) == 1) d1 |= byte << sh;
            else if ((dpos >> 2) == 2) d2 |= byte << sh;
            else                       d3 |= byte << sh;
        }
        *(uint4*)(q + ((size_t)gi << 4)) = make_uint4(d0, d1, d2, d3);
    }
}

// Returns 1 if the fused path ran, 0 if the caller must fall back to the two separate kernels.
// FAIL-SAFE BY CONSTRUCTION: every shape or alignment this kernel does not cover returns 0.
template<int MODE>
static int q27_rmsnorm_quant_impl(const unsigned short* x, const unsigned short* wgt,
                                  unsigned short* y, signed char* xq, float* xs,
                                  int n, int plus_one, float in_scale, const void* add, int abf,
                                  hipStream_t s)
{
    if (n <= 0 || (n & 15) || (n >> 2) > 1280) return 0;
    if (!q27_aligned(x, wgt, y, 7u)) return 0;
    if (((uintptr_t)xq & 15u) || !(in_scale > 0.f)) return 0;
    if (add && ((uintptr_t)add & 15u)) return 0;      // float4 needs 16-B alignment
    // BOTH modes take the RECIPROCAL. q27_quant_fp8's launcher passes 1.0f/in_scale into its
    // kernel's `global` parameter, so reading the kernel body alone and assuming it receives
    // in_scale directly makes the fp8 scale wrong by in_scale squared. That produced fluent token
    // salad, caught by the oracle gate. The two modes differ only in whether the values are
    // pre-multiplied by it, not in which quantity it is.
    const float sc_in = 1.0f / in_scale;
    // ONE BLOCK is forced: the RMS statistic is a reduction over the whole vector. But the block
    // was 256 threads = 4 waves, i.e. 4 waves on a 60-CU card, and rmsnorm measured 2.338 ms/token
    // for ~2 MB of traffic (~1 GB/s). The width is free parallelism inside that one block:
    // Q27_HID/4 = 1280 ushort4, and 640 threads x VPT=2 covers exactly 1280 with no idle lanes,
    // giving 10 waves instead of 4. 640 is a clean multiple of the 64-wide wave.
    // Width selectable at RUNTIME so it can be A/B'd end to end like everything else.
    // Q27_RN_WAVES=4 -> 256 threads VPT=5 (the original), 10 -> 640 threads VPT=2 (exact cover of
    // Q27_HID/4 = 1280, no idle lanes). The 10-wave form was kept on a +1.0% three-pair result,
    // which is inside the machine's own spread; this makes revalidating it a one-flag test.
    // DEFAULT 4, NOT 10. The 10-wave form was kept on THREE pairs at +1.0%, inside this machine's
    // own spread (the control alone ranged 18.664-20.383 ms over six runs, 8.6%). Revalidated at
    // six interleaved pairs it does NOT reproduce:
    //   CONTROL   (4 waves)  51.918 tok/s  19.261 ms/token  range [18.664, 20.383]
    //   CANDIDATE (10 waves) 51.286 tok/s  19.498 ms/token  range [18.833, 19.829]
    //   DELTA -1.22%, candidate faster in 3/6  -> REJECT
    // Kernel time really is lower at 10 waves (1.962 vs 2.338 ms). The engine disagrees, and the
    // engine decides.
    static const int rnw = q27_env_int("Q27_RN_WAVES", 4);
    if (rnw == 4)
        q27_k_rmsnorm_quant<256, 5, MODE><<<1, 256, 0, s>>>(
            (const ushort4*)x, (const ushort4*)wgt, (ushort4*)y, xq, xs,
            n >> 2, 1.0f / (float)n, plus_one ? 1.0f : 0.0f, sc_in, add, (ushort4*)x, abf);
    else
        q27_k_rmsnorm_quant<640, 2, MODE><<<1, 640, 0, s>>>(
            (const ushort4*)x, (const ushort4*)wgt, (ushort4*)y, xq, xs,
            n >> 2, 1.0f / (float)n, plus_one ? 1.0f : 0.0f, sc_in, add, (ushort4*)x, abf);
    return 1;
}

int q27_rmsnorm_quant_fp8(const unsigned short* x, const unsigned short* wgt, unsigned short* y,
                          signed char* xq, float* xs, int n, int plus_one, float in_scale,
                          const void* add, int abf, hipStream_t s)
{ return q27_rmsnorm_quant_impl<0>(x, wgt, y, xq, xs, n, plus_one, in_scale, add, abf, s); }

int q27_rmsnorm_quant_perm(const unsigned short* x, const unsigned short* wgt, unsigned short* y,
                           signed char* xq, float* xs, int n, int plus_one, float in_scale,
                           const void* add, int abf, hipStream_t s)
{ return q27_rmsnorm_quant_impl<1>(x, wgt, y, xq, xs, n, plus_one, in_scale, add, abf, s); }

// BF16 residual add, for the final pending residual when the collective transports BF16.
__global__ __launch_bounds__(256) void q27_k_add_bf16(unsigned short* __restrict__ h,
                                                      const unsigned short* __restrict__ m, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) h[i] = q27_f2bf(q27_bf2f(h[i]) + q27_bf2f(m[i]));
}
extern "C" void q27_add_inplace_bf16(unsigned short* h, const unsigned short* m, int n,
                                     hipStream_t s) {
    if (n <= 0) return;
    hipLaunchKernelGGL(q27_k_add_bf16, dim3((n + 255) / 256), dim3(256), 0, s, h, m, n);
}

// =============================================================================================
// MULTI-BLOCK RMSNORM + QUANTIZE. Two stages, both with a real grid.
// The single-block form is 1.96 ms/token for 2.6 MB (about 1 GB/s) and width is exhausted: 4 -> 10
// waves helped, 10 -> 16 hurt. One block is forced only because the RMS statistic spans the whole
// vector -- so split the statistic instead.
//
//   stage A (grid 5):  fold the residual, write the residual stream back, and reduce a PARTIAL
//                      sum of squares per block into a 5-float scratch.
//   stage B (grid 20): sum the 5 partials (cheap, every block does it), rsqrt, then normalize and
//                      quantize 16 groups per block with 16 lanes per group.
// Two launches instead of one, but grids of 5 and 20 instead of 1. At 14.9 us/call there is a large
// budget for a second launch (~1.25 us).
#define Q27_RN_A_BLOCKS 5
#define Q27_RN_B_BLOCKS 20

__global__ __launch_bounds__(256) void q27_k_rn_partial(
        ushort4* __restrict__ x, const void* __restrict__ a, int abf, int nv,
        float* __restrict__ part)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    float ss = 0.f;
    if (i < nv) {
        ushort4 v = x[i];
        if (a) {
            float m0, m1, m2, m3;
            if (abf) { const ushort4 b = ((const ushort4*)a)[i];
                       m0=q27_bf2f(b.x); m1=q27_bf2f(b.y); m2=q27_bf2f(b.z); m3=q27_bf2f(b.w); }
            else     { const float4 b = ((const float4*)a)[i]; m0=b.x; m1=b.y; m2=b.z; m3=b.w; }
            v.x = q27_f2bf(q27_bf2f(v.x) + m0);  v.y = q27_f2bf(q27_bf2f(v.y) + m1);
            v.z = q27_f2bf(q27_bf2f(v.z) + m2);  v.w = q27_f2bf(q27_bf2f(v.w) + m3);
            x[i] = v;                                   // residual stream persists
        }
        const float f0=q27_bf2f(v.x), f1=q27_bf2f(v.y), f2=q27_bf2f(v.z), f3=q27_bf2f(v.w);
        ss = fmaf(f0,f0, fmaf(f1,f1, fmaf(f2,f2, f3*f3)));
    }
    __shared__ float sm[4];
    ss = q27_block_sum<256>(ss, sm);
    if (threadIdx.x == 0) part[blockIdx.x] = ss;
}

template<int MODE>
__global__ __launch_bounds__(256) void q27_k_rn_apply(
        const unsigned short* __restrict__ x, const unsigned short* __restrict__ w,
        unsigned short* __restrict__ y, signed char* __restrict__ q, float* __restrict__ qs,
        int ngrp, float rn, float plus, float sc_in, const float* __restrict__ part)
{
    float tot = 0.f;
    #pragma unroll
    for (int i = 0; i < Q27_RN_A_BLOCKS; ++i) tot += part[i];   // 5 loads, every block, cheap
    const float inv = rsqrtf(tot * rn + Q27_RMS_EPS);

    const int lane16 = threadIdx.x & 15;                        // 16 lanes per group of 16
    const int g      = blockIdx.x * 16 + (threadIdx.x >> 4);
    if (g >= ngrp) return;
    const int k = (g << 4) + lane16;

    const float nv = q27_bf2f(x[k]) * inv * (plus + q27_bf2f(w[k]));
    y[k] = q27_f2bf(nv);
    const float v = (MODE == 1) ? nv * sc_in : nv;

    float mx = fabsf(v);
    #pragma unroll
    for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 16));
    mx = __shfl(mx, 0, 16);
    float sc, qi;
    if (MODE == 1) { sc = mx * (1.f/127.f);            qi = (sc > 0.f) ? 1.0f/sc : 0.f; }
    else           { sc = mx * sc_in * (1.f/127.f);    qi = (sc > 0.f) ? sc_in/sc : 0.f; }
    if (lane16 == 0) qs[g] = sc;

    int c = (int)rintf(v * qi);
    c = c < -127 ? -127 : (c > 127 ? 127 : c);
    const int dp = (MODE == 1) ? ((lane16 & 1) ? (8 + (lane16 >> 1)) : (lane16 >> 1)) : lane16;
    q[(g << 4) + dp] = (signed char)c;
}


// ---------------------------------------------------------------------------------------------
// HOST-SUMSQ RMSNORM. The post-barrier tail was a whole-vector sum-of-squares, which forces a
// cross-block reduction and therefore ONE block -- one CU of sixty -- 128x per token, sitting on
// the serial critical path where nothing can hide it. Widening it lost twice (-4.0%, -2.73%)
// because more blocks do not remove the reduction.
//
// So the reduction is moved off the GPU entirely. Every card holds the same residual r, so if
// each card adds r/4 to its own partial before the copy out, the host reduction of the four
// partials yields sum(p_i) + r -- the exact post-residual vector -- and can accumulate sum(x^2)
// in the SAME vectorised traversal, while those floats are already in registers. The host then
// publishes one scalar. What is left after the barrier is a pure elementwise map with no
// reduction, no intermediate residual write-back and no second pass, so it can use the machine.
//
// This kernel is deliberately the whole post-barrier operation: normalize, weight, write the new
// residual stream, and quantize into the consumer's representation. Leaving a standalone
// quantizer behind it would just recreate a second tail.
// Q27_RN_BLK: threads per block of the host-sumsq norm kernels (16/32/64). The pinned-memory read is
// PCIe-latency bound (20 KB in ~8 us); spreading the same float4 reads over more blocks / CUs raises
// the number of outstanding requests. Same per-element math at any block size.
static int q27_rn_blk() { const char* v = getenv("Q27_RN_BLK"); int b = v ? atoi(v) : 64; if (b != 16 && b != 32 && b != 64) b = 64; return b; }
// The batched (NR-row) form reads NR x 20 KB: measured on the 2-row verify, 16-thread blocks (4x the
// blocks in flight) are +4% end-to-end where the single-row kernel prefers 64 (-4% at 16).
static int q27_rn_blk_b() { const char* v = getenv("Q27_RN_BLK_B"); int b = v ? atoi(v) : 16; if (b != 16 && b != 32 && b != 64) b = 64; return b; }
template<int MODE>
__global__ __launch_bounds__(64) void q27_k_rn_hostss(
        const float4* __restrict__ x, const unsigned short* __restrict__ w,
        unsigned short* __restrict__ hid, unsigned short* __restrict__ y,
        signed char* __restrict__ q, float* __restrict__ qs,
        int nv4, const float* __restrict__ invp, float plus, float sc_in)
{
    // inv arrives by POINTER, not by value. The whole point of pre-enqueueing this kernel is that
    // it is submitted BEFORE the host reduction has produced 1/rms, so the value cannot be a
    // launch argument. One uniform scalar load; the address is wave-uniform so it lands in an
    // SGPR and costs nothing next to the 10.83 us launch this removes from the critical path.
    const float inv = *invp;

    // ONE float4 PER THREAD, not one float. x lives in pinned host memory, so a scalar load per
    // thread issues 5120 separate 4-byte PCIe reads -- the same pathology that made the outbound
    // zero-copy lose at 66 MB/s, in the read direction. 64 threads/block keeps 20 blocks live.
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nv4) return;
    const int k0 = t << 2;
    const float4 xv = x[t];

    const ushort4 wv = *(const ushort4*)(w + k0);
    ushort4 hv, yv;
    hv.x = q27_f2bf(xv.x); hv.y = q27_f2bf(xv.y);
    hv.z = q27_f2bf(xv.z); hv.w = q27_f2bf(xv.w);
    if (hid) *(ushort4*)(hid + k0) = hv;                     // residual stream for the next layer

    float n0 = xv.x * inv * (plus + q27_bf2f(wv.x));
    float n1 = xv.y * inv * (plus + q27_bf2f(wv.y));
    float n2 = xv.z * inv * (plus + q27_bf2f(wv.z));
    float n3 = xv.w * inv * (plus + q27_bf2f(wv.w));
    // y is the normalised bf16 vector. ONLY the gated-delta input projections read it
    // (q27_bf16_gemv2 on in_a/in_b); the MLP and the full-attention path consume xq/xs instead,
    // so on 80 of the 128 tail calls per token this store was written and never read.
    if (y) { yv.x = q27_f2bf(n0); yv.y = q27_f2bf(n1);
             yv.z = q27_f2bf(n2); yv.w = q27_f2bf(n3);
             *(ushort4*)(y + k0) = yv; }

    float v0 = n0, v1 = n1, v2 = n2, v3 = n3;
    if (MODE == 1) { v0 *= sc_in; v1 *= sc_in; v2 *= sc_in; v3 *= sc_in; }

    // a group of 16 elements is now 4 threads; reduce the max over those 4 lanes
    float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    mx = fmaxf(mx, __shfl_down(mx, 1, 4));
    mx = fmaxf(mx, __shfl_down(mx, 2, 4));
    mx = __shfl(mx, 0, 4);

    const int lane4 = t & 3;
    const int g     = t >> 2;
    float sc, qi;
    if (MODE == 1) { sc = mx * (1.f/127.f);            qi = (sc > 0.f) ? 1.0f/sc : 0.f; }
    else           { sc = mx * sc_in * (1.f/127.f);    qi = (sc > 0.f) ? sc_in/sc : 0.f; }
    if (lane4 == 0) qs[g] = sc;

    const float vv[4] = { v0, v1, v2, v3 };
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        int c = (int)rintf(vv[e] * qi);
        c = c < -127 ? -127 : (c > 127 ? 127 : c);
        const int lane16 = (lane4 << 2) + e;
        const int dp = (MODE == 1) ? ((lane16 & 1) ? (8 + (lane16 >> 1)) : (lane16 >> 1)) : lane16;
        q[(g << 4) + dp] = (signed char)c;
    }
}

// Pre-barrier: fold this card's share of the residual into its own partial. Runs in the producer
// stretch before barrier 1. Uniform work on all four cards is NOT free -- it moves the barrier
// release -- so this is one launch, and folding it into the projection epilogue is the follow-up.
__global__ __launch_bounds__(256) void q27_k_add_res_share(
        float* __restrict__ part, const unsigned short* __restrict__ hid, float rs, int n)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) part[i] += rs * q27_bf2f(hid[i]);
}

// What does a BARE post-barrier launch cost? The tail is ~10.6 us/call of real wall for a
// trivial map, and only ~1.25 of that is nominal launch overhead and ~2.65 the PCIe read. If an
// EMPTY kernel launched at the same point costs most of the remainder, the cost is dispatch into
// a pipeline the host barrier just drained, and the fix is merging post-barrier launches rather
// than making this kernel faster.
__global__ void q27_k_nothing(void) { }
extern "C" void q27_launch_nothing(hipStream_t s) {
    hipLaunchKernelGGL(q27_k_nothing, dim3(1), dim3(64), 0, s);
}

extern "C" void q27_add_res_share(float* part, const unsigned short* hid, float rs, int n,
                                  hipStream_t s) {
    hipLaunchKernelGGL(q27_k_add_res_share, dim3((n + 255) / 256), dim3(256), 0, s, part, hid, rs, n);
}

template<int MODE>
static int q27_rn_hostss_impl(const float* x, const unsigned short* wgt, unsigned short* hid,
                              unsigned short* y, signed char* xq, float* xs, int n,
                              int plus_one, float in_scale, const float* invp, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !x || !invp) return 0;
    const int nv4 = n >> 2;
    static const int rb = q27_rn_blk();
    hipLaunchKernelGGL((q27_k_rn_hostss<MODE>), dim3((nv4 + rb - 1) / rb), dim3(rb), 0, s,
                       (const float4*)x, wgt, hid, y, xq, xs, nv4, invp, plus_one ? 1.0f : 0.0f,
                       1.0f / in_scale);
    return 1;
}

// CHUNK-BATCHED host-sumsq tail: C positions in one launch (blockIdx.y = position), per-position
// pointers advanced by the slot strides; the arithmetic per position is q27_k_rn_hostss unchanged.
// The prefill sweep's chunk collective leaves C reduced vectors in C.acc ([C][HID], pinned) and
// C values of 1/rms in inv_hb; the tails were eight launches per chunk per layer.
template<int MODE>
__global__ __launch_bounds__(64) void q27_k_rn_hostss_b(
        const float4* __restrict__ x, const unsigned short* __restrict__ w,
        unsigned short* __restrict__ hid, unsigned short* __restrict__ y,
        signed char* __restrict__ q, float* __restrict__ qs,
        int nv4, const float* __restrict__ invp, float plus, float sc_in)
{
    const int c = blockIdx.y;
    x   += (size_t)c * nv4;                       // float4 units: HID/4
    if (hid) hid += (size_t)c * (nv4 << 2);
    if (y) y += (size_t)c * (nv4 << 2);
    q   += (size_t)c * (nv4 << 2);
    qs  += (size_t)c * (nv4 >> 2);                // HID/16 scales
    invp += c;
    const float inv = *invp;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nv4) return;
    const int k0 = t << 2;
    const float4 xv = x[t];
    const ushort4 wv = *(const ushort4*)(w + k0);
    ushort4 hv, yv;
    hv.x = q27_f2bf(xv.x); hv.y = q27_f2bf(xv.y);
    hv.z = q27_f2bf(xv.z); hv.w = q27_f2bf(xv.w);
    if (hid) *(ushort4*)(hid + k0) = hv;
    float n0 = xv.x * inv * (plus + q27_bf2f(wv.x));
    float n1 = xv.y * inv * (plus + q27_bf2f(wv.y));
    float n2 = xv.z * inv * (plus + q27_bf2f(wv.z));
    float n3 = xv.w * inv * (plus + q27_bf2f(wv.w));
    if (y) { yv.x = q27_f2bf(n0); yv.y = q27_f2bf(n1);
             yv.z = q27_f2bf(n2); yv.w = q27_f2bf(n3);
             *(ushort4*)(y + k0) = yv; }
    float v0 = n0, v1 = n1, v2 = n2, v3 = n3;
    if (MODE == 1) { v0 *= sc_in; v1 *= sc_in; v2 *= sc_in; v3 *= sc_in; }
    float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    mx = fmaxf(mx, __shfl_down(mx, 1, 4));
    mx = fmaxf(mx, __shfl_down(mx, 2, 4));
    mx = __shfl(mx, 0, 4);
    const int lane4 = t & 3;
    const int g     = t >> 2;
    float sc, qi;
    if (MODE == 1) { sc = mx * (1.f/127.f);            qi = (sc > 0.f) ? 1.0f/sc : 0.f; }
    else           { sc = mx * sc_in * (1.f/127.f);    qi = (sc > 0.f) ? sc_in/sc : 0.f; }
    if (lane4 == 0) qs[g] = sc;
    const float vv[4] = { v0, v1, v2, v3 };
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        int cq = (int)rintf(vv[e] * qi);
        cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
        const int lane16 = (lane4 << 2) + e;
        const int dp = (MODE == 1) ? ((lane16 & 1) ? (8 + (lane16 >> 1)) : (lane16 >> 1)) : lane16;
        q[(g << 4) + dp] = (signed char)cq;
    }
}
extern "C" int q27_rmsnorm_hostss_fp8_b(const float* x, const unsigned short* wgt,
        unsigned short* hid, unsigned short* y, signed char* xq, float* xs, int n,
        int plus_one, float in_scale, const float* invp, int C, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !x || !invp || C < 1) return 0;
    const int nv4 = n >> 2;
    static const int rb = q27_rn_blk_b();
    hipLaunchKernelGGL((q27_k_rn_hostss_b<0>), dim3((nv4 + rb - 1) / rb, C), dim3(rb), 0, s,
                       (const float4*)x, wgt, hid, y, xq, xs, nv4, invp, plus_one ? 1.0f : 0.0f,
                       1.0f / in_scale);
    return 1;
}
extern "C" int q27_rmsnorm_hostss_perm_b(const float* x, const unsigned short* wgt,
        unsigned short* hid, unsigned short* y, signed char* xq, float* xs, int n,
        int plus_one, float in_scale, const float* invp, int C, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !x || !invp || C < 1) return 0;
    const int nv4 = n >> 2;
    static const int rb = q27_rn_blk_b();
    hipLaunchKernelGGL((q27_k_rn_hostss_b<1>), dim3((nv4 + rb - 1) / rb, C), dim3(rb), 0, s,
                       (const float4*)x, wgt, hid, y, xq, xs, nv4, invp, plus_one ? 1.0f : 0.0f,
                       1.0f / in_scale);
    return 1;
}

extern "C" int q27_rmsnorm_hostss_fp8(const float* x, const unsigned short* wgt,
        unsigned short* hid, unsigned short* y, signed char* xq, float* xs, int n,
        int plus_one, float in_scale, const float* invp, hipStream_t s)
{ return q27_rn_hostss_impl<0>(x, wgt, hid, y, xq, xs, n, plus_one, in_scale, invp, s); }

extern "C" int q27_rmsnorm_hostss_perm(const float* x, const unsigned short* wgt,
        unsigned short* hid, unsigned short* y, signed char* xq, float* xs, int n,
        int plus_one, float in_scale, const float* invp, hipStream_t s)
{ return q27_rn_hostss_impl<1>(x, wgt, hid, y, xq, xs, n, plus_one, in_scale, invp, s); }

// ============================================================================
// Q27_PF_P2P: GPU-RESIDENT CROSS-CARD REDUCE.
// The prefill chunk collective left the GPU twice per chunk per layer: D2H of each card's
// full-width partial, host all-reduce + host sumsq, H2D of the assembled vector, then a
// host-sumsq rmsnorm tail. With peer access enabled (verified on gfx906: kernel peer loads
// and stores carry data both ways) ONE kernel per site does it all: it reads this card's
// partial and three PEER partials, assembles the full vector with the host's exact
// ((a+b)+(c+d)) tree, reduces the sumsq in-block, and applies the q27_k_rn_hostss_b<1>
// arithmetic (post-norm weights, plus=1, in_scale) with the same permuting fp8 quantize.
// The host never sees the payload; D2H, host reduce, H2D and the tail all disappear.
// 256 threads x 5 float4s = 1280 = HID/4 per position; one block per position (blockIdx.y).
// ============================================================================
__global__ __launch_bounds__(256) void q27_k_red_rn_perm_b(
        const float4* __restrict__ p0, const float4* __restrict__ p1,
        const float4* __restrict__ p2, const float4* __restrict__ p3,
        const unsigned short* __restrict__ wgt, unsigned short* __restrict__ hid,
        signed char* __restrict__ xq, float* __restrict__ xs,
        int nv4, float plus, float sc_in, int C)
{
    const int c = blockIdx.y;
    const int t = threadIdx.x;
    p0 += (size_t)c * nv4; p1 += (size_t)c * nv4; p2 += (size_t)c * nv4; p3 += (size_t)c * nv4;
    wgt += (size_t)c * (nv4 << 2);
    hid += (size_t)c * (nv4 << 2);
    xq  += (size_t)c * (nv4 << 2);
    xs  += (size_t)c * (nv4 >> 2);
    float4 v[5];
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    #pragma unroll
    for (int e = 0; e < 5; ++e) {
        const int i = e * 256 + t;
        const float4 a = p0[i], b = p1[i], cc = p2[i], d = p3[i];
        float4 r;
        r.x = (a.x + b.x) + (cc.x + d.x);
        r.y = (a.y + b.y) + (cc.y + d.y);
        r.z = (a.z + b.z) + (cc.z + d.z);
        r.w = (a.w + b.w) + (cc.w + d.w);
        v[e] = r;
        s0 = fmaf(r.x, r.x, s0); s1 = fmaf(r.y, r.y, s1);
        s2 = fmaf(r.z, r.z, s2); s3 = fmaf(r.w, r.w, s3);
    }
    float ss = (s0 + s1) + (s2 + s3);
    __shared__ float sm[4];
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_xor(ss, o, 64);
    const int w = t >> 6;
    if ((t & 63) == 0) sm[w] = ss;
    __syncthreads();
    if (t < 4) {
        float q4 = sm[t];
        q4 += __shfl_xor(q4, 2, 4);
        q4 += __shfl_xor(q4, 1, 4);
        if (t == 0) sm[0] = q4;
    }
    __syncthreads();
    ss = sm[0];
    const float inv = 1.0f / sqrtf(ss / (float)(nv4 << 2) + Q27_RMS_EPS);
    #pragma unroll
    for (int e = 0; e < 5; ++e) {
        const int i = e * 256 + t;
        const int k0 = i << 2;
        const float4 xv = v[e];
        const ushort4 wv = *(const ushort4*)(wgt + k0);
        ushort4 hv;
        hv.x = q27_f2bf(xv.x); hv.y = q27_f2bf(xv.y);
        hv.z = q27_f2bf(xv.z); hv.w = q27_f2bf(xv.w);
        *(ushort4*)(hid + k0) = hv;
        float n0 = xv.x * inv * (plus + q27_bf2f(wv.x));
        float n1 = xv.y * inv * (plus + q27_bf2f(wv.y));
        float n2 = xv.z * inv * (plus + q27_bf2f(wv.z));
        float n3 = xv.w * inv * (plus + q27_bf2f(wv.w));
        float v0 = n0 * sc_in, v1 = n1 * sc_in, v2 = n2 * sc_in, v3 = n3 * sc_in;
        float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        mx = fmaxf(mx, __shfl_down(mx, 1, 4));
        mx = fmaxf(mx, __shfl_down(mx, 2, 4));
        mx = __shfl(mx, 0, 4);
        const int lane4 = t & 3;
        const int g = i >> 2;
        float sc, qi;
        sc = mx * (1.f / 127.f);
        qi = (sc > 0.f) ? 1.0f / sc : 0.f;
        if (lane4 == 0) xs[g] = sc;
        const float vv[4] = { v0, v1, v2, v3 };
        #pragma unroll
        for (int ee = 0; ee < 4; ++ee) {
            int cq = (int)rintf(vv[ee] * qi);
            cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
            const int lane16 = (lane4 << 2) + ee;
            const int dp = (lane16 & 1) ? (8 + (lane16 >> 1)) : (lane16 >> 1);
            xq[(g << 4) + dp] = (signed char)cq;
        }
    }
}
extern "C" int q27_red_rn_perm_b(const float* p0, const float* p1, const float* p2, const float* p3,
        const unsigned short* wgt, unsigned short* hid, signed char* xq, float* xs, int n,
        int plus_one, float in_scale, int C, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !p0 || !p1 || !p2 || !p3 || !wgt || !hid || !xq || !xs || C < 1) return 0;
    const int nv4 = n >> 2;
    hipLaunchKernelGGL(q27_k_red_rn_perm_b, dim3(1, C), dim3(256), 0, s,
                       (const float4*)p0, (const float4*)p1, (const float4*)p2, (const float4*)p3,
                       wgt, hid, xq, xs, nv4, plus_one ? 1.0f : 0.0f, 1.0f / in_scale, C);
    return 1;
}

// ============================================================================
// Q27_NR_P2P: GPU-RESIDENT CROSS-CARD REDUCE FOR THE ROW-BATCHED (speculative verify) COLLECTIVE.
// The NR-row decode collective (NR = K+1 <= 8 rows x 5120 fp32 per card per layer) was host-staged:
// in-stream staging copy, host barrier, host slice reduce over cold pinned lines, host barrier,
// zero-copy read of the reduced rows by the norm kernel -- ~120-160 us per site at NR=8, 128 sites
// per round. Here the site is two device kernels and no host: (1) q27_k_red_nr_p2p, grid (5, NR),
// 256 threads, one float4 per thread: reads this card's partial and the three PEER partials (peer
// access, gated 12/12 on 2026-09-09; the 20 KB decode contract tied, this is the 8x contract),
// assembles the row with the host's exact ((a+b)+(c+d)) tree into a LOCAL fp32 buffer and writes
// its 1024-element sum of squares to ssp[row*5 + blk]; 40 blocks at NR=8 keep 40 CUs' worth of
// peer loads in flight instead of one block per row. (2) q27_k_rn_ssp_b<MODE>: the norm/quant tail
// of q27_k_rn_hostss_b<MODE> reading the local reduced rows, with 1/rms from the five partial sums
// combined in fixed order (identical on every card). MODE 0 = input norm (natural-order fp8 quant,
// optional bf16 normalized copy y), MODE 1 = post norm (permuted quant), MODE 2 = reduce only.
// ============================================================================
#define Q27_NRP_SB 5
__global__ __launch_bounds__(256) void q27_k_red_nr_p2p(
        const float4* __restrict__ p0, const float4* __restrict__ p1,
        const float4* __restrict__ p2, const float4* __restrict__ p3,
        float4* __restrict__ red, float* __restrict__ ssp, int nv4)
{
    const int c = blockIdx.y, bx = blockIdx.x, t = threadIdx.x;
    const int i = bx * 256 + t;
    if (i >= nv4) return;
    const size_t o = (size_t)c * nv4 + i;
    const float4 a = p0[o], b = p1[o], cc = p2[o], d = p3[o];
    float4 r;
    r.x = (a.x + b.x) + (cc.x + d.x);
    r.y = (a.y + b.y) + (cc.y + d.y);
    r.z = (a.z + b.z) + (cc.z + d.z);
    r.w = (a.w + b.w) + (cc.w + d.w);
    red[o] = r;
    float ss = fmaf(r.x, r.x, fmaf(r.y, r.y, fmaf(r.z, r.z, r.w * r.w)));
    __shared__ float sm[4];
    #pragma unroll
    for (int oo = 32; oo; oo >>= 1) ss += __shfl_xor(ss, oo, 64);
    const int w = t >> 6;
    if ((t & 63) == 0) sm[w] = ss;
    __syncthreads();
    if (t == 0) ssp[c * Q27_NRP_SB + bx] = (sm[0] + sm[1]) + (sm[2] + sm[3]);
}
template<int MODE>
__global__ __launch_bounds__(64) void q27_k_rn_ssp_b(
        const float4* __restrict__ x, const float* __restrict__ ssp, const unsigned short* __restrict__ w,
        unsigned short* __restrict__ hid, unsigned short* __restrict__ y,
        signed char* __restrict__ q, float* __restrict__ qs,
        int nv4, float plus, float sc_in)
{
    const int c = blockIdx.y;
    x   += (size_t)c * nv4;
    hid += (size_t)c * (nv4 << 2);
    if (y) y += (size_t)c * (nv4 << 2);
    q   += (size_t)c * (nv4 << 2);
    qs  += (size_t)c * (nv4 >> 2);
    const float* sp = ssp + c * Q27_NRP_SB;
    const float ss = ((sp[0] + sp[1]) + (sp[2] + sp[3])) + sp[4];          // fixed order: identical on every card
    const float inv = 1.0f / sqrtf(ss / (float)(nv4 << 2) + Q27_RMS_EPS);
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nv4) return;
    const int k0 = t << 2;
    const float4 xv = x[t];
    const ushort4 wv = *(const ushort4*)(w + k0);
    ushort4 hv;
    hv.x = q27_f2bf(xv.x); hv.y = q27_f2bf(xv.y);
    hv.z = q27_f2bf(xv.z); hv.w = q27_f2bf(xv.w);
    *(ushort4*)(hid + k0) = hv;
    float n0 = xv.x * inv * (plus + q27_bf2f(wv.x));
    float n1 = xv.y * inv * (plus + q27_bf2f(wv.y));
    float n2 = xv.z * inv * (plus + q27_bf2f(wv.z));
    float n3 = xv.w * inv * (plus + q27_bf2f(wv.w));
    if (y) { ushort4 yv; yv.x = q27_f2bf(n0); yv.y = q27_f2bf(n1); yv.z = q27_f2bf(n2); yv.w = q27_f2bf(n3);
             *(ushort4*)(y + k0) = yv; }
    float v0 = n0, v1 = n1, v2 = n2, v3 = n3;
    if (MODE == 1) { v0 *= sc_in; v1 *= sc_in; v2 *= sc_in; v3 *= sc_in; }
    float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    mx = fmaxf(mx, __shfl_down(mx, 1, 4));
    mx = fmaxf(mx, __shfl_down(mx, 2, 4));
    mx = __shfl(mx, 0, 4);
    const int lane4 = t & 3;
    const int g     = t >> 2;
    float sc, qi;
    if (MODE == 1) { sc = mx * (1.f/127.f);            qi = (sc > 0.f) ? 1.0f/sc : 0.f; }
    else           { sc = mx * sc_in * (1.f/127.f);    qi = (sc > 0.f) ? sc_in/sc : 0.f; }
    if (lane4 == 0) qs[g] = sc;
    const float vv[4] = { v0, v1, v2, v3 };
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        int cq = (int)rintf(vv[e] * qi);
        cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
        const int lane16 = (lane4 << 2) + e;
        const int dp = (MODE == 1) ? ((lane16 & 1) ? (8 + (lane16 >> 1)) : (lane16 >> 1)) : lane16;
        q[(g << 4) + dp] = (signed char)cq;
    }
}
// mode 0/1: reduce + norm/quant; mode 2: reduce only (red holds the fp32 rows). red: [C][n] fp32 local, ssp: [C][5] local.
extern "C" int q27_red_rn_nr_p2p(int mode, const float* p0, const float* p1, const float* p2, const float* p3,
        float* red, float* ssp, const unsigned short* wgt, unsigned short* hid, unsigned short* y,
        signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, hipStream_t s)
{
    if (n != Q27_HID || !p0 || !p1 || !p2 || !p3 || !red || !ssp || C < 1) return 0;
    const int nv4 = n >> 2;                                   // 1280 = 5 x 256
    hipLaunchKernelGGL(q27_k_red_nr_p2p, dim3(Q27_NRP_SB, C), dim3(256), 0, s,
                       (const float4*)p0, (const float4*)p1, (const float4*)p2, (const float4*)p3, (float4*)red, ssp, nv4);
    if (mode == 2) return 1;
    if (!(in_scale > 0.f) || !wgt || !hid || !xq || !xs) return 0;
    const dim3 grid((nv4 + 63) / 64, C), blk(64);
    if (mode == 1) hipLaunchKernelGGL(q27_k_rn_ssp_b<1>, grid, blk, 0, s, (const float4*)red, (const float*)ssp, wgt, hid, y, xq, xs, nv4, plus_one ? 1.0f : 0.0f, 1.0f / in_scale);
    else           hipLaunchKernelGGL(q27_k_rn_ssp_b<0>, grid, blk, 0, s, (const float4*)red, (const float*)ssp, wgt, hid, y, xq, xs, nv4, plus_one ? 1.0f : 0.0f, 1.0f / in_scale);
    return 1;
}
// Q27_NR_P2P v2: PUSH instead of PULL. The peer-READ reduce (v1) measured the layer pass at ~75 ms against 52 on
// the host path at NR=8: GPU-initiated peer loads over PCIe are latency-bound (uncached coarse-grained memory,
// 64-B requests), whereas peer STORES are posted writes (the 09-09 gate's kernel-store transport ran at the
// launch floor). So the producer card pushes its partial, packed to bf16, into the three peers' receive slots
// with one kernel, and every card reduces from LOCAL memory: own partial (rounded to bf16 in-kernel so all four
// cards sum identical inputs in the same ((0+1)+(2+3)) order) + three received bf16 rows.
__global__ __launch_bounds__(256) void q27_k_push_bf16_3(const float4* __restrict__ src,
                                                        uint2* __restrict__ d0, uint2* __restrict__ d1, uint2* __restrict__ d2, int n4)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n4) return;
    const float4 v = src[i];
    uint2 o;
    o.x = (unsigned)q27_f2bf(v.x) | ((unsigned)q27_f2bf(v.y) << 16);
    o.y = (unsigned)q27_f2bf(v.z) | ((unsigned)q27_f2bf(v.w) << 16);
    d0[i] = o; d1[i] = o; d2[i] = o;
}
extern "C" int q27_push_bf16_3(const float* src, unsigned short* d0, unsigned short* d1, unsigned short* d2, int n, hipStream_t s)
{
    if (!src || !d0 || !d1 || !d2 || (n & 3)) return 0;
    const int n4 = n >> 2;
    hipLaunchKernelGGL(q27_k_push_bf16_3, dim3((n4 + 255) / 256), dim3(256), 0, s,
                       (const float4*)src, (uint2*)d0, (uint2*)d1, (uint2*)d2, n4);
    return 1;
}
__device__ __forceinline__ float4 q27_bf16x4_to_f4(uint2 p) {
    float4 r;
    r.x = __uint_as_float(p.x << 16); r.y = __uint_as_float(p.x & 0xffff0000u);
    r.z = __uint_as_float(p.y << 16); r.w = __uint_as_float(p.y & 0xffff0000u);
    return r;
}
__device__ __forceinline__ float4 q27_round_bf16_f4(float4 v) {
    float4 r;
    r.x = __uint_as_float((unsigned)q27_f2bf(v.x) << 16); r.y = __uint_as_float((unsigned)q27_f2bf(v.y) << 16);
    r.z = __uint_as_float((unsigned)q27_f2bf(v.z) << 16); r.w = __uint_as_float((unsigned)q27_f2bf(v.w) << 16);
    return r;
}
// own = this card's fp32 partial (card index g), rk = the received bf16 rows of the other three cards in card
// order; the sum is formed as ((c0 + c1) + (c2 + c3)) by card index on every card.
__global__ __launch_bounds__(256) void q27_k_red_nr_local(const float4* __restrict__ own, int g,
        const uint2* __restrict__ ra, const uint2* __restrict__ rb, const uint2* __restrict__ rc,
        float4* __restrict__ red, float* __restrict__ ssp, int nv4)
{
    const int c = blockIdx.y, bx = blockIdx.x, t = threadIdx.x;
    const int i = bx * 256 + t;
    if (i >= nv4) return;
    const size_t o = (size_t)c * nv4 + i;
    const float4 mine = q27_round_bf16_f4(own[o]);
    const float4 a = q27_bf16x4_to_f4(ra[o]), b = q27_bf16x4_to_f4(rb[o]), cc = q27_bf16x4_to_f4(rc[o]);
    // place by card index: slots a/b/c hold the cards != g in ascending order
    float4 p[4];
    { int j = 0; for (int k = 0; k < 4; ++k) { if (k == g) p[k] = mine; else { p[k] = (j == 0) ? a : (j == 1) ? b : cc; ++j; } } }
    float4 r;
    r.x = (p[0].x + p[1].x) + (p[2].x + p[3].x);
    r.y = (p[0].y + p[1].y) + (p[2].y + p[3].y);
    r.z = (p[0].z + p[1].z) + (p[2].z + p[3].z);
    r.w = (p[0].w + p[1].w) + (p[2].w + p[3].w);
    red[o] = r;
    float ss = fmaf(r.x, r.x, fmaf(r.y, r.y, fmaf(r.z, r.z, r.w * r.w)));
    __shared__ float sm[4];
    #pragma unroll
    for (int oo = 32; oo; oo >>= 1) ss += __shfl_xor(ss, oo, 64);
    const int w = t >> 6;
    if ((t & 63) == 0) sm[w] = ss;
    __syncthreads();
    if (t == 0) ssp[c * Q27_NRP_SB + bx] = (sm[0] + sm[1]) + (sm[2] + sm[3]);
}
// mode 0/1: reduce + norm/quant; mode 2: reduce only. own fp32 [C][n] local; ra/rb/rc bf16 [C][n] local (received).
extern "C" int q27_red_rn_nr_local(int mode, const float* own, int g, const unsigned short* ra, const unsigned short* rb, const unsigned short* rc,
        float* red, float* ssp, const unsigned short* wgt, unsigned short* hid, unsigned short* y,
        signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, hipStream_t s)
{
    if (n != Q27_HID || !own || !ra || !rb || !rc || !red || !ssp || C < 1 || g < 0 || g > 3) return 0;
    const int nv4 = n >> 2;
    hipLaunchKernelGGL(q27_k_red_nr_local, dim3(Q27_NRP_SB, C), dim3(256), 0, s,
                       (const float4*)own, g, (const uint2*)ra, (const uint2*)rb, (const uint2*)rc, (float4*)red, ssp, nv4);
    if (mode == 2) return 1;
    if (!(in_scale > 0.f) || !wgt || !hid || !xq || !xs) return 0;
    const dim3 grid((nv4 + 63) / 64, C), blk(64);
    if (mode == 1) hipLaunchKernelGGL(q27_k_rn_ssp_b<1>, grid, blk, 0, s, (const float4*)red, (const float*)ssp, wgt, hid, y, xq, xs, nv4, plus_one ? 1.0f : 0.0f, 1.0f / in_scale);
    else           hipLaunchKernelGGL(q27_k_rn_ssp_b<0>, grid, blk, 0, s, (const float4*)red, (const float*)ssp, wgt, hid, y, xq, xs, nv4, plus_one ? 1.0f : 0.0f, 1.0f / in_scale);
    return 1;
}
// Q27_PF_P2P: the mixer collective (collective #2, deferred path): assemble the 4 partial MLP
// outputs in place of D2H+flush_c2+tp_reduce_b2+H2D. Writes the assembled fp32 vector to out and
// 1/rms per position to inv_out -- the exact two things the deferred path staged into pend_slots
// and inv_slots for the next layer's input norm.
__global__ __launch_bounds__(256) void q27_k_red_b(
        const float4* __restrict__ p0, const float4* __restrict__ p1,
        const float4* __restrict__ p2, const float4* __restrict__ p3,
        float4* __restrict__ out, float* __restrict__ inv_out, int nv4)
{
    const int c = blockIdx.y;
    const int t = threadIdx.x;
    p0 += (size_t)c * nv4; p1 += (size_t)c * nv4; p2 += (size_t)c * nv4; p3 += (size_t)c * nv4;
    out += (size_t)c * nv4;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    #pragma unroll
    for (int e = 0; e < 5; ++e) {
        const int i = e * 256 + t;
        const float4 a = p0[i], b = p1[i], cc = p2[i], d = p3[i];
        float4 r;
        r.x = (a.x + b.x) + (cc.x + d.x);
        r.y = (a.y + b.y) + (cc.y + d.y);
        r.z = (a.z + b.z) + (cc.z + d.z);
        r.w = (a.w + b.w) + (cc.w + d.w);
        out[i] = r;
        s0 = fmaf(r.x, r.x, s0); s1 = fmaf(r.y, r.y, s1);
        s2 = fmaf(r.z, r.z, s2); s3 = fmaf(r.w, r.w, s3);
    }
    float ss = (s0 + s1) + (s2 + s3);
    __shared__ float sm[4];
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_xor(ss, o, 64);
    const int w = t >> 6;
    if ((t & 63) == 0) sm[w] = ss;
    __syncthreads();
    if (t < 4) {
        float q4 = sm[t];
        q4 += __shfl_xor(q4, 2, 4);
        q4 += __shfl_xor(q4, 1, 4);
        if (t == 0) sm[0] = q4;
    }
    __syncthreads();
    ss = sm[0];
    if (t == 0) inv_out[c] = 1.0f / sqrtf(ss / (float)(nv4 << 2) + Q27_RMS_EPS);
}
extern "C" int q27_red_b(const float* p0, const float* p1, const float* p2, const float* p3,
        float* out, float* inv_out, int n, int C, hipStream_t s)
{
    if (n != Q27_HID || !p0 || !p1 || !p2 || !p3 || !out || !inv_out || C < 1) return 0;
    const int nv4 = n >> 2;
    hipLaunchKernelGGL(q27_k_red_b, dim3(1, C), dim3(256), 0, s,
                       (const float4*)p0, (const float4*)p1, (const float4*)p2, (const float4*)p3,
                       (float4*)out, inv_out, nv4);
    return 1;
}

// Q27_PF_P2P self-test: a kernel on the source card writes a marker into the TARGET card's
// buffer through peer access; the host reads it back and verifies. Covers every ordered pair.
__global__ void q27_k_p2p_probe(float* dst, float val) { *dst = val; }
extern "C" int q27_p2p_probe(float* dst, float val, hipStream_t s)
{ hipLaunchKernelGGL(q27_k_p2p_probe, dim3(1), dim3(64), 0, s, dst, val); return 1; }

// ============================================================================
// Q27_LAYER_SPLIT: single-card variants of the fused tails. With one card per layer the
// producer's output IS the full vector - there is no cross-card sum - but the sumsq + 1/rms
// still move on-device (the host-sumsq path would D2H the full 5120 floats per position).
// The rmsnorm+quant arithmetic is exactly q27_k_rn_hostss_b<1>; red_b1 also copies the
// mixer into pend_slots (the consumer's buffer).
// ============================================================================
__global__ __launch_bounds__(256) void q27_k_red_rn_perm_b1(
        const float4* __restrict__ x, const unsigned short* __restrict__ wgt,
        unsigned short* __restrict__ hid, float* __restrict__ hid32, signed char* __restrict__ xq, float* __restrict__ xs,
        int nv4, float plus, float sc_in)
{
    const int c = blockIdx.y;
    const int t = threadIdx.x;
    x   += (size_t)c * nv4;
    // post_norm is a PER-LAYER weight (the canonical q27_k_rn_hostss_b reads w + k0 with no
    // position offset). The position stride here was copied from the P2P 4-source kernel and
    // made every position c>0 read past the layer's weights into the arena: position 0 was
    // correct (the per-layer dumps matched the TP reference), positions 1..255 used fake
    // weights (L=2's read landed on fp8 scale bytes ~1e21 -> NaN cascade -> token 0).
    if (hid)   hid   += (size_t)c * (nv4 << 2);
    if (hid32) hid32 += (size_t)c * (nv4 << 2);
    xq  += (size_t)c * (nv4 << 2);
    xs  += (size_t)c * (nv4 >> 2);
    float4 v[5];
    #pragma unroll
    for (int e = 0; e < 5; ++e) {
        const int i = e * 256 + t;
        v[e] = x[i];
    }
    // 1/rms with the HOST's exact sumsq order (tp_reduce_bx at nd=1): one thread walks the
    // 5120 elements sequentially with the 4-accumulator fmaf cycle, so the inv matches the
    // reference bit-for-bit. A block-tree order differed in ulps and cascaded through the
    // fp8 quantization over the layers.
    __shared__ float sm[4];
    if (t < 4) {
        // four threads, each walking a contiguous 320-float4 segment with the 4-accumulator
        // cycle; the combine keeps segment order (s0+s1)+(s2+s3). 4x the throughput of the
        // single-thread walk (the summation order changes; the token gate is the authority).
        const int lo = t * (nv4 >> 2), hi = (t + 1) * (nv4 >> 2);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int i = lo; i < hi; ++i) {
            const float4 q = x[i];
            a0 = fmaf(q.x, q.x, a0);
            a1 = fmaf(q.y, q.y, a1);
            a2 = fmaf(q.z, q.z, a2);
            a3 = fmaf(q.w, q.w, a3);
        }
        sm[t] = (a0 + a1) + (a2 + a3);
    }
    __syncthreads();
    if (t == 0) sm[0] = (sm[0] + sm[1]) + (sm[2] + sm[3]);
    __syncthreads();
    const float ss = sm[0];
    const float inv = 1.0f / sqrtf(ss / (float)(nv4 << 2) + Q27_RMS_EPS);
    #pragma unroll
    for (int e = 0; e < 5; ++e) {
        const int i = e * 256 + t;
        const int k0 = i << 2;
        const float4 xv = v[e];
        const ushort4 wv = *(const ushort4*)(wgt + k0);
        if (hid32) *(float4*)(hid32 + k0) = xv;
        if (hid) { ushort4 hv; hv.x = q27_f2bf(xv.x); hv.y = q27_f2bf(xv.y); hv.z = q27_f2bf(xv.z); hv.w = q27_f2bf(xv.w); *(ushort4*)(hid + k0) = hv; }
        float n0 = xv.x * inv * (plus + q27_bf2f(wv.x));
        float n1 = xv.y * inv * (plus + q27_bf2f(wv.y));
        float n2 = xv.z * inv * (plus + q27_bf2f(wv.z));
        float n3 = xv.w * inv * (plus + q27_bf2f(wv.w));
        float v0 = n0 * sc_in, v1 = n1 * sc_in, v2 = n2 * sc_in, v3 = n3 * sc_in;
        float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        mx = fmaxf(mx, __shfl_down(mx, 1, 4));
        mx = fmaxf(mx, __shfl_down(mx, 2, 4));
        mx = __shfl(mx, 0, 4);
        const int lane4 = t & 3;
        const int g = i >> 2;
        float sc, qi;
        sc = mx * (1.f / 127.f);
        qi = (sc > 0.f) ? 1.0f / sc : 0.f;
        if (lane4 == 0) xs[g] = sc;
        const float vv[4] = { v0, v1, v2, v3 };
        #pragma unroll
        for (int ee = 0; ee < 4; ++ee) {
            int cq = (int)rintf(vv[ee] * qi);
            cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
            const int lane16 = (lane4 << 2) + ee;
            const int dp = (lane16 & 1) ? (8 + (lane16 >> 1)) : (lane16 >> 1);
            xq[(g << 4) + dp] = (signed char)cq;
        }
    }
}
extern "C" int q27_red_rn_perm_b1(const float* x, const unsigned short* wgt, unsigned short* hid, float* hid32,
        signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !x || !wgt || !xq || !xs || C < 1) return 0;
    const int nv4 = n >> 2;
    hipLaunchKernelGGL(q27_k_red_rn_perm_b1, dim3(1, C), dim3(256), 0, s,
                       (const float4*)x, wgt, hid, hid32, xq, xs, nv4, plus_one ? 1.0f : 0.0f,
                       1.0f / in_scale);
    return 1;
}

__global__ __launch_bounds__(256) void q27_k_red_b1(
        const float4* __restrict__ x, float4* __restrict__ out, float* __restrict__ inv_out, int nv4)
{
    const int c = blockIdx.y;
    const int t = threadIdx.x;
    x   += (size_t)c * nv4;
    out += (size_t)c * nv4;
    #pragma unroll
    for (int e = 0; e < 5; ++e) {
        const int i = e * 256 + t;
        out[i] = x[i];
    }
    // the host's exact sumsq order (see q27_k_red_rn_perm_b1)
    __shared__ float sm[4];
    if (t < 4) {
        // four threads, each walking a contiguous 320-float4 segment with the 4-accumulator
        // cycle; the combine keeps segment order (s0+s1)+(s2+s3). 4x the throughput of the
        // single-thread walk (the summation order changes; the token gate is the authority).
        const int lo = t * (nv4 >> 2), hi = (t + 1) * (nv4 >> 2);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int i = lo; i < hi; ++i) {
            const float4 q = x[i];
            a0 = fmaf(q.x, q.x, a0);
            a1 = fmaf(q.y, q.y, a1);
            a2 = fmaf(q.z, q.z, a2);
            a3 = fmaf(q.w, q.w, a3);
        }
        sm[t] = (a0 + a1) + (a2 + a3);
    }
    __syncthreads();
    if (t == 0) sm[0] = (sm[0] + sm[1]) + (sm[2] + sm[3]);
    __syncthreads();
    const float ss = sm[0];
    if (t == 0) inv_out[c] = 1.0f / sqrtf(ss / (float)(nv4 << 2) + Q27_RMS_EPS);
}
extern "C" int q27_red_b1(const float* x, float* out, float* inv_out, int n, int C, hipStream_t s)
{
    if (n != Q27_HID || !x || !out || !inv_out || C < 1) return 0;
    const int nv4 = n >> 2;
    hipLaunchKernelGGL(q27_k_red_b1, dim3(1, C), dim3(256), 0, s,
                       (const float4*)x, (float4*)out, inv_out, nv4);
    return 1;
}

template<int MODE>
static int q27_rn_split_impl(const unsigned short* x, const unsigned short* wgt, unsigned short* y,
                             signed char* xq, float* xs, int n, int plus_one, float in_scale,
                             const void* add, int abf, float* scratch, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !scratch) return 0;
    const int nv = n >> 2, ngrp = n >> 4;
    if (nv != Q27_RN_A_BLOCKS * 256) return 0;
    const float sc_in = 1.0f / in_scale;
    hipLaunchKernelGGL(q27_k_rn_partial, dim3(Q27_RN_A_BLOCKS), dim3(256), 0, s,
                       (ushort4*)x, add, abf, nv, scratch);
    hipLaunchKernelGGL((q27_k_rn_apply<MODE>), dim3(Q27_RN_B_BLOCKS), dim3(256), 0, s,
                       x, wgt, y, xq, xs, ngrp, 1.0f/(float)n, plus_one ? 1.0f : 0.0f,
                       sc_in, scratch);
    return 1;
}

extern "C" int q27_rmsnorm_quant_split_fp8(const unsigned short* x, const unsigned short* wgt,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const void* add, int abf, float* scratch, hipStream_t s)
{ return q27_rn_split_impl<0>(x, wgt, y, xq, xs, n, plus_one, in_scale, add, abf, scratch, s); }

extern "C" int q27_rmsnorm_quant_split_perm(const unsigned short* x, const unsigned short* wgt,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const void* add, int abf, float* scratch, hipStream_t s)
{ return q27_rn_split_impl<1>(x, wgt, y, xq, xs, n, plus_one, in_scale, add, abf, scratch, s); }

// Q27_LS_Q8: the post-attention/GDN norm + quantize with PER-64 scales in NATURAL order (the int8/64
// gate+up's input). Same fp32 sumsq and arithmetic as q27_k_red_rn_perm_b1; a 64-element group is
// 16 consecutive threads (float4 each), so the group max is a 16-wide shuffle ladder.
__global__ __launch_bounds__(256) void q27_k_red_rn_g64_b1(
        const float4* __restrict__ x, const unsigned short* __restrict__ wgt,
        float* __restrict__ hid32, signed char* __restrict__ xq, float* __restrict__ xs,
        int nv4, float plus, float sc_in)
{
    const int c = blockIdx.y;
    const int t = threadIdx.x;
    x   += (size_t)c * nv4;
    if (hid32) hid32 += (size_t)c * (nv4 << 2);
    xq  += (size_t)c * (nv4 << 2);
    xs  += (size_t)c * (nv4 >> 4);
    float4 v[5];
#pragma unroll
    for (int e = 0; e < 5; ++e) v[e] = x[e * 256 + t];
    __shared__ float sm[4];
    if (t < 4) {
        const int lo = t * (nv4 >> 2), hi = (t + 1) * (nv4 >> 2);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int i = lo; i < hi; ++i) {
            const float4 q = x[i];
            a0 = fmaf(q.x, q.x, a0); a1 = fmaf(q.y, q.y, a1); a2 = fmaf(q.z, q.z, a2); a3 = fmaf(q.w, q.w, a3);
        }
        sm[t] = (a0 + a1) + (a2 + a3);
    }
    __syncthreads();
    if (t == 0) sm[0] = (sm[0] + sm[1]) + (sm[2] + sm[3]);
    __syncthreads();
    const float inv = 1.0f / sqrtf(sm[0] / (float)(nv4 << 2) + Q27_RMS_EPS);
#pragma unroll
    for (int e = 0; e < 5; ++e) {
        const int i = e * 256 + t;
        const int k0 = i << 2;
        const float4 xv = v[e];
        const ushort4 wv = *(const ushort4*)(wgt + k0);
        if (hid32) *(float4*)(hid32 + k0) = xv;
        const float v0 = xv.x * inv * (plus + q27_bf2f(wv.x)) * sc_in;
        const float v1 = xv.y * inv * (plus + q27_bf2f(wv.y)) * sc_in;
        const float v2 = xv.z * inv * (plus + q27_bf2f(wv.z)) * sc_in;
        const float v3 = xv.w * inv * (plus + q27_bf2f(wv.w)) * sc_in;
        float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
        const float sc = mx * (1.f / 127.f);
        const float qi = (sc > 0.f) ? 1.0f / sc : 0.f;
        if ((t & 15) == 0) xs[i >> 4] = sc;
        int c0 = (int)rintf(v0 * qi), c1 = (int)rintf(v1 * qi), c2 = (int)rintf(v2 * qi), c3 = (int)rintf(v3 * qi);
        c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
        c2 = c2 < -127 ? -127 : (c2 > 127 ? 127 : c2); c3 = c3 < -127 ? -127 : (c3 > 127 ? 127 : c3);
        *(unsigned*)(xq + k0) = ((unsigned)c0 & 0xFFu) | (((unsigned)c1 & 0xFFu) << 8) | (((unsigned)c2 & 0xFFu) << 16) | (((unsigned)c3 & 0xFFu) << 24);
    }
}
extern "C" int q27_red_rn_g64_b1(const float* x, const unsigned short* wgt, float* hid32,
        signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !x || !wgt || !xq || !xs || C < 1) return 0;
    const int nv4 = n >> 2;
    hipLaunchKernelGGL(q27_k_red_rn_g64_b1, dim3(1, C), dim3(256), 0, s,
                       (const float4*)x, wgt, hid32, xq, xs, nv4, plus_one ? 1.0f : 0.0f, 1.0f / in_scale);
    return 1;
}

// Q27_LS_Q8 stage 2: the input norm with PER-64 activation scales, natural order (the int8/64 projections'
// input; the a/b GEMV reads the same buffer). Arithmetic as q27_k_rn_hostss_b<0>; a 64-group = 16 threads.
__global__ __launch_bounds__(64) void q27_k_rn_hostss_g64_b(
        const float4* __restrict__ x, const unsigned short* __restrict__ w,
        signed char* __restrict__ q, float* __restrict__ qs,
        int nv4, const float* __restrict__ invp, float plus, float sc_in)
{
    const int c = blockIdx.y;
    x   += (size_t)c * nv4;
    q   += (size_t)c * (nv4 << 2);
    qs  += (size_t)c * (nv4 >> 4);                // HID/64 scales
    invp += c;
    const float inv = *invp;
    const int t = blockIdx.x * 64 + threadIdx.x;
    const bool ok = t < nv4;
    const float4 xv = ok ? x[t] : make_float4(0.f, 0.f, 0.f, 0.f);
    const int k0 = (ok ? t : 0) << 2;
    const ushort4 wv = *(const ushort4*)(w + k0);
    const float v0 = xv.x * inv * (plus + q27_bf2f(wv.x)) * sc_in;
    const float v1 = xv.y * inv * (plus + q27_bf2f(wv.y)) * sc_in;
    const float v2 = xv.z * inv * (plus + q27_bf2f(wv.z)) * sc_in;
    const float v3 = xv.w * inv * (plus + q27_bf2f(wv.w)) * sc_in;
    float mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
    const float sc = mx * (1.f / 127.f);
    const float qi = (sc > 0.f) ? 1.0f / sc : 0.f;
    if (!ok) return;
    if ((t & 15) == 0) qs[t >> 4] = sc;
    int c0 = (int)rintf(v0 * qi), c1 = (int)rintf(v1 * qi), c2 = (int)rintf(v2 * qi), c3 = (int)rintf(v3 * qi);
    c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
    c2 = c2 < -127 ? -127 : (c2 > 127 ? 127 : c2); c3 = c3 < -127 ? -127 : (c3 > 127 ? 127 : c3);
    *(unsigned*)(q + k0) = ((unsigned)c0 & 0xFFu) | (((unsigned)c1 & 0xFFu) << 8) | (((unsigned)c2 & 0xFFu) << 16) | (((unsigned)c3 & 0xFFu) << 24);
}
extern "C" int q27_rmsnorm_hostss_g64_b(const float* x, const unsigned short* wgt, signed char* xq, float* xs, int n,
                                        int plus_one, float in_scale, const float* invp, int C, hipStream_t s)
{
    if (n != Q27_HID || !(in_scale > 0.f) || !x || !invp || C < 1) return 0;
    const int nv4 = n >> 2;
    hipLaunchKernelGGL(q27_k_rn_hostss_g64_b, dim3((nv4 + 63) / 64, C), dim3(64), 0, s,
                       (const float4*)x, wgt, xq, xs, nv4, invp, plus_one ? 1.0f : 0.0f, 1.0f / in_scale);
    return 1;
}
