// ============================================================================
// q27_attn.hip -- Qwen3.8-27B full attention decode, AMD gfx906 (MI50), wave64.
//
// Serves the 16 layers where Q27_IS_FULL(L).  Mathematics: CONTRACT section 3.
//   24 q heads / 4 kv heads / head_dim 256, GQA group = 6, softmax scale 1/16.
//   q_proj emits 12288 rows: head h owns [h*512, h*512+512) = 256 query dims
//   followed by 256 OUTPUT GATE dims.  The attention result is scaled by
//   sigmoid(gate) -- plain sigmoid, NOT swish (that is the gated-delta layer).
//   q_norm / k_norm: RMSNorm over the 256 dims of ONE head, eps 1e-6, weight
//   applied as (1+W), one shared [256] weight for all heads.  v is NOT normed.
//   RoPE on dims [0,64) only, rotate-half with half=32: pair = d^32.
//   KV cache [position][kv_head 4][head_dim 256] BF16; K post-norm+post-RoPE,
//   V raw.
//
// BUFFER CONTRACT (not stated in q27.h -- stated here once, used by both entries)
//   qkv_out    : BF16 [14336] = q_proj 12288 | k_proj 1024 | v_proj 1024.
//   q_out      : BF16 [6144]  = 24 heads x 256, post q_norm, post RoPE. This is
//                the `q` argument of q27_attn_decode.  The GATE is NOT copied
//                here; decode reads it from q_proj_raw (= the q_proj block of
//                qkv_out, 12288 rows) at [h*512 + 256 + d].
//   out        : BF16 [6144] = Q27_OROWS, the o_proj input.
//   kcache/vcache: this layer's base, i.e. full_index*max_context*Q27_KVROWS.
//
// MRoPE: with a scalar `pos` all three mrope axes coincide (text-only decode),
// so the position multiplier is simply `pos`.  A multimodal caller must pass
// per-axis positions; that is a signature change, not a kernel change.
//
// STRUCTURE -- head-group fusion (the lever):
//   ONE BLOCK (one wave64) owns one KV head and ALL SIX of its query heads.
//   Each K row and each V row is fetched ONCE per block and dotted / accumulated
//   against 6 query vectors held in registers, instead of being re-read 6x.
//   (FlashInfer decode bdy=GROUP_SIZE; exllamav3 attn_chunked q_reg[G]; the CUDA
//   oracle's PREFILL kernel -- its decode kernel does not, and neither did the
//   previous AMD kernel.)
//   Lane layout: 32 lanes cover the 256 head dims at 8 bf16 (one 16B dwordx4)
//   per lane, so a wave64 covers TWO KV rows per iteration (half-wave per row).
//   Score reduction is a 5-step xor butterfly inside each half-wave; the two
//   halves hold disjoint row subsets and are merged ONCE at the end.
//   No LDS anywhere: every operand has exactly one consumer per element per
//   block, which is precisely the case the estate measured LDS staging losing.
//   Sequence split across blocks (flash-decoding) restores CU occupancy; the
//   partials live in a static __device__ arena -- no allocation after READY.
//
// exp2 everywhere: v_exp_f32 is natively base-2 on gfx906, so log2(e) is folded
// into the softmax scale once and no exp() call ever appears in the row loop.
// Running max seeded with a finite -1e30 sentinel and denominator seeded 0, so
// an empty / all-masked split contributes exactly zero and cannot make a NaN.
// ============================================================================
#include "q27.h"
#include <cstdio>

#define Q27_G            6                          // GQA group: 24 q / 4 kv
#define Q27_VEC          8                          // bf16 per lane per row (16B)
#define Q27_LPR          (Q27_HDIM / Q27_VEC)       // 32 lanes per KV row
#define Q27_RPW          (64 / Q27_LPR)             // 2 rows per wave64
#define Q27_SPLITS_MAX   64                         // >= 4*64 = 256 waves resident
#define Q27_SPLIT_ROWS   64                         // rows per split target
#define Q27_NEG_BIG      (-1.0e30f)                 // finite sentinel, not -inf
#define Q27_SENT_TEST    (-1.0e29f)
#define Q27_SCALE_L2E    0.090168440055560215f      // 0.0625 * log2(e)
#define Q27_LOG2_THETA   23.253496664211534         // log2(1e7)

// q_proj block offsets inside qkv_out
#define Q27_QKV_Q        0
#define Q27_QKV_K        Q27_QROWS
#define Q27_QKV_V        (Q27_QROWS + Q27_KVROWS)

// ---------------- split-decode partials (static arena, module-load lifetime) --
// [split][q head][dim] fp32 accumulator, and (max, denominator) per split/head.
__device__ float q27_attn_part_o [8][Q27_SPLITS_MAX][Q27_NHEAD][Q27_HDIM];   // [row] for the Q27_SPEC multi-row launch (NR <= 8)
__device__ float q27_attn_part_ml[8][Q27_SPLITS_MAX][Q27_NHEAD][2];

// ---------------- small helpers ----------------
// eight BF16 (one 16B load) -> eight fp32. BF16->fp32 is a shift, not a cvt.
__device__ __forceinline__ void q27_unpack8(const uint4 u, float* f) {
    f[0] = __uint_as_float(u.x <<  16);
    f[1] = __uint_as_float(u.x & 0xFFFF0000u);
    f[2] = __uint_as_float(u.y <<  16);
    f[3] = __uint_as_float(u.y & 0xFFFF0000u);
    f[4] = __uint_as_float(u.z <<  16);
    f[5] = __uint_as_float(u.z & 0xFFFF0000u);
    f[6] = __uint_as_float(u.w <<  16);
    f[7] = __uint_as_float(u.w & 0xFFFF0000u);
}
__device__ __forceinline__ uint4 q27_ld8(const unsigned short* p) {
    return *reinterpret_cast<const uint4*>(p);      // 16B aligned by construction
}
// ds_bpermute takes a BYTE lane index and crosses the full wave64 (ds_swizzle
// would not).  For mask <= 16, lane^mask provably never leaves its own 32-lane
// half, so the clamp that the generic __shfl_xor(width=32) must emit is dead
// work -- and at this register pressure LLVM rematerializes it INSIDE the row
// loop (xor+cmp+cndmask+shl x5 per iteration, measured).  Precompute the five
// partner addresses once, spend 5 VGPRs, delete 20 instructions per iteration.
// Bare v_exp_f32.  gfx906's transcendental unit IS base-2, so with log2(e)
// folded into the softmax scale this is the whole exponential -- one instruction.
// libm exp2f wraps it in a denormal fixup (v_cmp + 2x v_cndmask + v_add + v_mul,
// measured 5 extra instructions x 12 call sites in the row loop) to keep results
// below 2^-126 exact.  A softmax weight that small is zero against the 1.0 the
// running max contributes, and the -1e30 sentinel gives 0 either way.  ROCm 5.7
// has neither __builtin_amdgcn_exp2f nor __ocml_native_exp2_f32; asm is the only
// route to the instruction.  Non-volatile, so the scheduler may still hoist it.
__device__ __forceinline__ float q27_exp2(float x) {
    float r;
    asm("v_exp_f32_e32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}
__device__ __forceinline__ float q27_bperm(int addr, float v) {
    return __uint_as_float((unsigned)__builtin_amdgcn_ds_bpermute(addr, (int)__float_as_uint(v)));
}
struct q27_bfly { int a[5]; };
__device__ __forceinline__ q27_bfly q27_bfly_init(int lane) {
    q27_bfly b;
#pragma unroll
    for (int i = 0; i < 5; ++i) b.a[i] = (lane ^ (16 >> i)) << 2;
    return b;
}
// all-lanes sum over one 32-lane half-wave. Butterfly => every lane ends with the
// SAME value bit-for-bit, so the softmax branch below is half-wave uniform.
__device__ __forceinline__ float q27_red32(float v, const q27_bfly& b) {
#pragma unroll
    for (int i = 0; i < 5; ++i) v += q27_bperm(b.a[i], v);
    return v;
}

// ============================================================================
// PREP: q_norm + k_norm + RoPE + KV cache write.
// One wave64 per head unit; lane t owns dims t, t+64, t+128, t+192.  All 64
// rotary dims therefore live in slot 0 (dim == t) and the rotate-half partner is
// exactly lane t^32 -- one cross-wave shuffle, no LDS, no second pass.
// grid.x = 24 q heads + 4 kv heads (the kv block also copies its raw V row).
// ============================================================================
// TEMPLATED ON THE SHARD GEOMETRY.  <24,4,12288,1024> is the whole-model shape and compiles to
// exactly what this kernel was before; <6,1,3072,256> is one card of the 4-way tensor-parallel
// split, where the GQA group is untouched (6 query heads still meet their own single KV head) so
// nothing about the mathematics changes -- only the buffer widths and the KV cache row stride.
// ---------------- INT8 KV CACHE (Q27_LS_Q8, 2026-09-11) ----------------
// K (post-norm, post-RoPE) and V (raw) are stored as int8 with one fp32 scale per 16 dims:
//   k8/v8 : [position][KVROWS] int8      ks/vs : [position][KVROWS/16] fp32
// The reader dequantizes 8 dims per lane in registers (one 8-byte load + one scale per row);
// the writer quantizes a head's 256 dims as 16 lane-groups of 16 (lane t owns dims t+64i, so
// dims [16m+64i, 16m+64i+16) are lanes 16m..16m+15 of slot i -- a 16-wide xor-max).
__device__ __forceinline__ uint2 q27_ld8i(const signed char* p) {
    return *reinterpret_cast<const uint2*>(p);
}
__device__ __forceinline__ void q27_unpack8i(const uint2 u, float s, float* f) {
    f[0] = (float)((int)(u.x << 24) >> 24) * s;
    f[1] = (float)((int)(u.x << 16) >> 24) * s;
    f[2] = (float)((int)(u.x <<  8) >> 24) * s;
    f[3] = (float)((int) u.x        >> 24) * s;
    f[4] = (float)((int)(u.y << 24) >> 24) * s;
    f[5] = (float)((int)(u.y << 16) >> 24) * s;
    f[6] = (float)((int)(u.y <<  8) >> 24) * s;
    f[7] = (float)((int) u.y        >> 24) * s;
}
// quantize one value per lane of a 16-lane group; returns the int8, writes the scale from lane 0
__device__ __forceinline__ signed char q27_q16(float v, float& sc_out) {
    float mx = fabsf(v);
    mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
    mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
    const float sc = mx * (1.0f / 127.0f);
    const float qi = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
    int c = (int)rintf(v * qi);
    c = c < -127 ? -127 : (c > 127 ? 127 : c);
    sc_out = sc;
    return (signed char)c;
}

// ============================================================================
// PREP: q_norm + k_norm + RoPE + KV cache write.
// One wave64 per head unit; lane t owns dims t, t+64, t+128, t+192.  All 64
// rotary dims therefore live in slot 0 (dim == t) and the rotate-half partner is
// exactly lane t^32 -- one cross-wave shuffle, no LDS, no second pass.
// grid.x = 24 q heads + 4 kv heads (the kv block also copies its raw V row).
// ============================================================================
template<int NQ, int NKV, int QROWS, int KVROWS>
__global__ __launch_bounds__(64)
void q27_attn_prep_k(const unsigned short* __restrict__ qkv,
                     const unsigned short* __restrict__ qnw,
                     const unsigned short* __restrict__ knw,
                     signed char* __restrict__ kcache, float* __restrict__ kscale,
                     signed char* __restrict__ vcache, float* __restrict__ vscale,
                     unsigned short* __restrict__ qout,
                     int pos, int kvstride, int hoff0, int qkvstride = 0, int qoutstride = 0, int boff = 0)
{
    const int row  = blockIdx.y;                       // Q27_SPEC: consecutive positions of one sequence
    qkv  += (size_t)row * qkvstride;
    qout += (size_t)row * qoutstride;
    pos  += row;
    const int t    = threadIdx.x;
    const int b    = blockIdx.x + boff;                // boff = NQ: k/v heads only (the draft layer's prompt pass)
    const bool isq = (b < NQ);
    const int head = isq ? b : (b - NQ);

    const unsigned short* src = isq ? (qkv + 0      + head * 512)
                                    : (qkv + QROWS  + head * Q27_HDIM);
    const unsigned short* wgt = isq ? qnw : knw;
    const unsigned short* vsrc =  qkv + (QROWS + KVROWS) + (isq ? 0 : head * Q27_HDIM);

    unsigned short xr[4], wr[4], vr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) xr[i] = src[t + i * 64];
#pragma unroll
    for (int i = 0; i < 4; ++i) wr[i] = wgt[t + i * 64];
#pragma unroll
    for (int i = 0; i < 4; ++i) vr[i] = isq ? (unsigned short)0 : vsrc[t + i * 64];

    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = q27_bf2f(xr[i]);

    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss = fmaf(x[i], x[i], ss);
#pragma unroll
    for (int m = 32; m > 0; m >>= 1) ss += __shfl_xor(ss, m, 64);   // 256 dims
    const float inv = rsqrtf(ss * (1.0f / (float)Q27_HDIM) + Q27_RMS_EPS);

    float y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = x[i] * inv * (1.0f + q27_bf2f(wr[i]));

    {
        const float pv   = __shfl_xor(y[0], Q27_ROT_HALF, 64);
        const float rot  = (t < Q27_ROT_HALF) ? -pv : pv;
        const int   freq = t & (Q27_ROT_HALF - 1);
        const double ifq = exp2(-((double)(2 * freq) / (double)Q27_ROT) * Q27_LOG2_THETA);
        double sn, cs;
        sincos((double)pos * ifq, &sn, &cs);
        y[0] = y[0] * (float)cs + rot * (float)sn;
    }

    if (isq) {
        unsigned short* dst = qout + head * Q27_HDIM;
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[t + i * 64] = q27_f2bf(y[i]);
    } else {
        const size_t off  = (size_t)pos * (size_t)kvstride + (size_t)hoff0 + (size_t)head * Q27_HDIM;
        const size_t soff = (size_t)pos * (size_t)(kvstride / 16) + (size_t)(hoff0 / 16) + (size_t)head * (Q27_HDIM / 16);
        signed char* kd = kcache + off;  float* ksd = kscale + soff;
        signed char* vd = vcache + off;  float* vsd = vscale + soff;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float sk, sv;
            const signed char qk = q27_q16(y[i], sk);
            const signed char qv = q27_q16(q27_bf2f(vr[i]), sv);
            kd[t + i * 64] = qk;
            vd[t + i * 64] = qv;
            if ((t & 15) == 0) { ksd[(t >> 4) + 4 * i] = sk; vsd[(t >> 4) + 4 * i] = sv; }
        }
    }
}

// ============================================================================
// DECODE stage 1: one wave64 == one (split, kv head), all 6 query heads fused.
// ============================================================================
template<int NQ, int NKV, int KVROWS>
__global__ __launch_bounds__(64)
void q27_attn_decode_k(const unsigned short* __restrict__ q,
                       const signed char* __restrict__ kc, const float* __restrict__ ks,
                       const signed char* __restrict__ vc, const float* __restrict__ vs,
                       const unsigned short* __restrict__ qraw,
                       unsigned short* __restrict__ out,
                       int L, int rps, int single, int kvstride, int hoff0, int qstride = 0, int ostride = 0, int qrawstride = 0)
{
    const int row = blockIdx.z;                        // Q27_SPEC: row r attends to L + r positions
    L    += row;
    q    += (size_t)row * qstride;
    out  += (size_t)row * ostride;
    qraw += (size_t)row * qrawstride;
    const int lane = threadIdx.x;
    const int hl   = lane >> 5;               // half-wave -> which of the 2 rows
    const int dg   = lane & 31;               // dim group
    const int d0   = dg * Q27_VEC;
    const int sp   = blockIdx.x;
    const int kvh  = blockIdx.y;
    const int h0   = kvh * Q27_G;
    const int KSR = kvstride / 16;

    const int r0 = sp * rps;
    const int r1 = (r0 + rps < L) ? (r0 + rps) : L;
    if (r0 >= r1) {                            // empty split: sentinel only
        if (lane < Q27_G) {
            q27_attn_part_ml[row][sp][h0 + lane][0] = Q27_NEG_BIG;
            q27_attn_part_ml[row][sp][h0 + lane][1] = 0.0f;
        }
        return;
    }

    uint4 qu[Q27_G];
#pragma unroll
    for (int g = 0; g < Q27_G; ++g) qu[g] = q27_ld8(q + (h0 + g) * Q27_HDIM + d0);
    float qv[Q27_G][Q27_VEC];
#pragma unroll
    for (int g = 0; g < Q27_G; ++g) q27_unpack8(qu[g], qv[g]);

    float acc[Q27_G][Q27_VEC], mm[Q27_G], ll[Q27_G];
#pragma unroll
    for (int g = 0; g < Q27_G; ++g) {
        mm[g] = Q27_NEG_BIG;
        ll[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < Q27_VEC; ++j) acc[g][j] = 0.0f;
    }

    const q27_bfly bf  = q27_bfly_init(lane);
    const size_t hoff  = (size_t)hoff0 + (size_t)kvh * Q27_HDIM + (size_t)d0;
    const int    soff  = (int)(hoff >> 4);
    const int    niter = (r1 - r0 + Q27_RPW - 1) / Q27_RPW;

    int    rp = r0 + hl;  if (rp >= r1) rp = r1 - 1;
    uint2  kb = q27_ld8i(kc + (size_t)rp * kvstride + hoff);
    uint2  vb = q27_ld8i(vc + (size_t)rp * kvstride + hoff);
    float  kbs = ks[(size_t)rp * KSR + soff];
    float  vbs = vs[(size_t)rp * KSR + soff];

    for (int it = 0; it < niter; ++it) {
        const int  r     = r0 + it * Q27_RPW + hl;
        const bool valid = (r < r1);
        const uint2 kcur = kb, vcur = vb;
        const float kcs = kbs, vcs = vbs;

        int rn = r0 + (it + 1) * Q27_RPW + hl;
        if (rn >= r1) rn = r1 - 1;                        // clamp: never OOB
        kb = q27_ld8i(kc + (size_t)rn * kvstride + hoff);
        vb = q27_ld8i(vc + (size_t)rn * kvstride + hoff);
        kbs = ks[(size_t)rn * KSR + soff];
        vbs = vs[(size_t)rn * KSR + soff];

        float kf[Q27_VEC];
        q27_unpack8i(kcur, kcs, kf);

        float sc[Q27_G];
#pragma unroll
        for (int g = 0; g < Q27_G; ++g) {
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < Q27_VEC; ++j) s = fmaf(qv[g][j], kf[j], s);
            sc[g] = s;
        }
#pragma unroll
        for (int g = 0; g < Q27_G; ++g) sc[g] = q27_red32(sc[g], bf);  // 6 independent chains

        if (valid) {
            float vf[Q27_VEC];
            q27_unpack8i(vcur, vcs, vf);
#pragma unroll
            for (int g = 0; g < Q27_G; ++g) {
                const float x = sc[g] * Q27_SCALE_L2E;      // log2 domain
                float p;
                if (x > mm[g]) {
                    const float c = q27_exp2(mm[g] - x);       // sentinel -> 0, no NaN
                    mm[g] = x;
                    ll[g] = ll[g] * c + 1.0f;
#pragma unroll
                    for (int j = 0; j < Q27_VEC; ++j) acc[g][j] *= c;
                    p = 1.0f;
                } else {
                    p = q27_exp2(x - mm[g]);
                    ll[g] += p;
                }
#pragma unroll
                for (int j = 0; j < Q27_VEC; ++j) acc[g][j] = fmaf(p, vf[j], acc[g][j]);
            }
        }
    }

#pragma unroll
    for (int g = 0; g < Q27_G; ++g) {
        const float mo = __shfl_xor(mm[g], 32, 64);
        const float lo = __shfl_xor(ll[g], 32, 64);
        const float M  = fmaxf(mm[g], mo);
        const float ca = q27_exp2(mm[g] - M), cb = q27_exp2(mo - M);
#pragma unroll
        for (int j = 0; j < Q27_VEC; ++j) {
            const float ao = __shfl_xor(acc[g][j], 32, 64);
            acc[g][j] = acc[g][j] * ca + ao * cb;
        }
        ll[g] = ll[g] * ca + lo * cb;
        mm[g] = M;
    }
    if (hl != 0) return;

    if (single) {
#pragma unroll
        for (int g = 0; g < Q27_G; ++g) {
            const float inv = (ll[g] > 0.0f) ? (1.0f / ll[g]) : 0.0f;
            float gt[Q27_VEC];
            q27_unpack8(q27_ld8(qraw + (h0 + g) * 512 + Q27_HDIM + d0), gt);
            uint4 o;
            unsigned* op = reinterpret_cast<unsigned*>(&o);
#pragma unroll
            for (int j = 0; j < Q27_VEC; j += 2) {
                const float s0 = 1.0f / (1.0f + __expf(-gt[j]));
                const float s1 = 1.0f / (1.0f + __expf(-gt[j + 1]));
                const unsigned a = q27_f2bf(acc[g][j]     * inv * s0);
                const unsigned b = q27_f2bf(acc[g][j + 1] * inv * s1);
                op[j >> 1] = a | (b << 16);
            }
            *reinterpret_cast<uint4*>(out + (h0 + g) * Q27_HDIM + d0) = o;
        }
    } else {
#pragma unroll
        for (int g = 0; g < Q27_G; ++g) {
            float* dst = &q27_attn_part_o[row][sp][h0 + g][d0];
            *reinterpret_cast<float4*>(dst)     = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
            *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
            if (dg == 0) {
                q27_attn_part_ml[row][sp][h0 + g][0] = mm[g];
                q27_attn_part_ml[row][sp][h0 + g][1] = ll[g];
            }
        }
    }
}

// ============================================================================
// DECODE stage 2: flash combine across splits + sigmoid output gate + BF16 out.
// One block per query head, one thread per head dim.
// ============================================================================
__global__ __launch_bounds__(Q27_HDIM)
void q27_attn_combine_k(const unsigned short* __restrict__ qraw,
                        unsigned short* __restrict__ out,
                        int nsplits,
                        signed char* __restrict__ Q, float* __restrict__ QS, float qgs, int ostride = 0, int qrawstride = 0)
{
    const int row = blockIdx.z;
    out  += (size_t)row * ostride;
    qraw += (size_t)row * qrawstride;
    const int h = blockIdx.x;
    const int d = threadIdx.x;

    float M = Q27_NEG_BIG;
    for (int s = 0; s < nsplits; ++s) M = fmaxf(M, q27_attn_part_ml[row][s][h][0]);

    float den = 0.0f, o = 0.0f;
    for (int s = 0; s < nsplits; ++s) {
        const float ms = q27_attn_part_ml[row][s][h][0];
        if (ms < Q27_SENT_TEST) continue;                 // empty split: skip
        const float w = q27_exp2(ms - M);
        den = fmaf(q27_attn_part_ml[row][s][h][1], w, den);
        o   = fmaf(q27_attn_part_o[row][s][h][d], w, o);
    }
    const float inv = (den > 0.0f) ? (1.0f / den) : 0.0f;
    const float gt  = q27_bf2f(qraw[h * 512 + Q27_HDIM + d]);
    const float sg  = 1.0f / (1.0f + __expf(-gt));        // plain sigmoid
    const float yv = o * inv * sg;
    out[h * Q27_HDIM + d] = q27_f2bf(yv);

    if (Q) {
        float mx = fabsf(yv);
        #pragma unroll
        for (int o16 = 8; o16; o16 >>= 1) mx = fmaxf(mx, __shfl_down(mx, o16, 16));
        mx = __shfl(mx, 0, 16);
        const float sc = mx * qgs * (1.0f / 127.0f);
        const float qi = (sc > 0.0f) ? (qgs / sc) : 0.0f;
        const int   k  = h * Q27_HDIM + d;
        if ((d & 15) == 0) QS[k >> 4] = sc;
        int c = (int)rintf(yv * qi);
        c = c < -127 ? -127 : (c > 127 ? 127 : c);
        Q[k] = (signed char)c;
    }
}

static inline void q27_attn_splits_host(int L, int& nsplits, int& rps, int split_rows = Q27_SPLIT_ROWS) {
    nsplits = (L + split_rows - 1) / split_rows;
    if (nsplits > Q27_SPLITS_MAX) nsplits = Q27_SPLITS_MAX;
    if (nsplits < 1) nsplits = 1;
    rps = (L + nsplits - 1) / nsplits;
    rps = (rps + Q27_RPW - 1) & ~(Q27_RPW - 1);
}
// ============================================================================
// CHUNK FORMS (Q27_PF_ATT_CHUNK): the three stages over M consecutive query positions in one
// launch each. Per query the arithmetic is the per-position kernels' (own split geometry from its
// own length, the same loops, the same combine). The chunk query is fp32 [c][NQ*HDIM] (an
// execution representation, loaded once per wave and held in registers); the KV cache is int8.
// ============================================================================
#define Q27_ATT_CHUNK_MAX Q27_PF_CH
#define Q27_POB(c, sp, h, d)  pob[(((size_t)(c) * Q27_SPLITS_MAX + (sp)) * NQ + (h)) * Q27_HDIM + (d)]
#define Q27_PML(c, sp, h, k)  pml[(((size_t)(c) * Q27_SPLITS_MAX + (sp)) * NQ + (h)) * 2 + (k)]
extern "C" size_t q27_attn_chunk_scratch_bytes(int ndev, int chunk_max, size_t* ml_bytes) {
    const int NQ = Q27_NHEAD / ndev;
    if (ml_bytes) *ml_bytes = (size_t)chunk_max * Q27_SPLITS_MAX * NQ * 2 * sizeof(float);
    return (size_t)chunk_max * Q27_SPLITS_MAX * NQ * Q27_HDIM * sizeof(float);
}

__device__ __forceinline__ void q27_attn_splits(int L, int& nsplits, int& rps, int split_rows = Q27_SPLIT_ROWS) {
    nsplits = (L + split_rows - 1) / split_rows;
    if (nsplits > Q27_SPLITS_MAX) nsplits = Q27_SPLITS_MAX;
    if (nsplits < 1) nsplits = 1;
    rps = (L + nsplits - 1) / nsplits;
    rps = (rps + Q27_RPW - 1) & ~(Q27_RPW - 1);
}

// shared tail of both chunk prep forms: norm, RoPE, fp32 query out, int8 K/V cache in
template<int NQ, int KVROWS>
__device__ __forceinline__ void q27_attn_prep_tail(const float* x, const unsigned short* wgt, const float* vf,
                                                   bool isq, int head, int t, int pos,
                                                   signed char* __restrict__ kcache, float* __restrict__ kscale,
                                                   signed char* __restrict__ vcache, float* __restrict__ vscale,
                                                   float* __restrict__ qout)
{
    unsigned short wr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) wr[i] = wgt[t + i * 64];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ss = fmaf(x[i], x[i], ss);
#pragma unroll
    for (int m = 32; m > 0; m >>= 1) ss += __shfl_xor(ss, m, 64);
    const float inv = rsqrtf(ss * (1.0f / (float)Q27_HDIM) + Q27_RMS_EPS);
    float y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = x[i] * inv * (1.0f + q27_bf2f(wr[i]));
    {
        const float pv   = __shfl_xor(y[0], Q27_ROT_HALF, 64);
        const float rot  = (t < Q27_ROT_HALF) ? -pv : pv;
        const int   freq = t & (Q27_ROT_HALF - 1);
        const double ifq = exp2(-((double)(2 * freq) / (double)Q27_ROT) * Q27_LOG2_THETA);
        double sn, cs;
        sincos((double)pos * ifq, &sn, &cs);
        y[0] = y[0] * (float)cs + rot * (float)sn;
    }
    if (isq) {
        float* dst = qout + head * Q27_HDIM;
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[t + i * 64] = y[i];
    } else {
        const size_t off  = (size_t)pos * (size_t)KVROWS + (size_t)head * Q27_HDIM;
        const size_t soff = (size_t)pos * (size_t)(KVROWS / 16) + (size_t)head * (Q27_HDIM / 16);
        signed char* kd = kcache + off;  float* ksd = kscale + soff;
        signed char* vd = vcache + off;  float* vsd = vscale + soff;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float sk, sv;
            const signed char qk = q27_q16(y[i], sk);
            const signed char qv = q27_q16(vf[i], sv);
            kd[t + i * 64] = qk;
            vd[t + i * 64] = qv;
            if ((t & 15) == 0) { ksd[(t >> 4) + 4 * i] = sk; vsd[(t >> 4) + 4 * i] = sv; }
        }
    }
}

// bf16 q/k/v producer (the TP prefill sweep's projections)
template<int NQ, int NKV, int QROWS, int KVROWS>
__global__ __launch_bounds__(64)
void q27_attn_prep_kb(const unsigned short* __restrict__ qkv_t, int qstride,
                      const unsigned short* __restrict__ qnw,
                      const unsigned short* __restrict__ knw,
                      signed char* __restrict__ kcache, float* __restrict__ kscale,
                      signed char* __restrict__ vcache, float* __restrict__ vscale,
                      float* __restrict__ qout_t,
                      int pos0)
{
    const int c   = blockIdx.y;
    const int pos = pos0 + c;
    const unsigned short* __restrict__ qkv = qkv_t + (size_t)c * qstride;
    float* __restrict__ qout = qout_t + (size_t)c * (NQ * Q27_HDIM);
    const int t    = threadIdx.x;
    const int b    = blockIdx.x;
    const bool isq = (b < NQ);
    const int head = isq ? b : (b - NQ);
    const unsigned short* src = isq ? (qkv + 0      + head * 512)
                                    : (qkv + QROWS  + head * Q27_HDIM);
    const unsigned short* wgt = isq ? qnw : knw;
    const unsigned short* vsrc =  qkv + (QROWS + KVROWS) + (isq ? 0 : head * Q27_HDIM);
    float x[4], vf[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = q27_bf2f(src[t + i * 64]);
#pragma unroll
    for (int i = 0; i < 4; ++i) vf[i] = isq ? 0.0f : q27_bf2f(vsrc[t + i * 64]);
    q27_attn_prep_tail<NQ, KVROWS>(x, wgt, vf, isq, head, t, pos, kcache, kscale, vcache, vscale, qout);
}

// int8 q/k/v producer (Q27_LS_Q8: the fused rb2p3 projection's int8 + per-16 scales)
template<int NQ, int NKV, int QROWS, int KVROWS>
__global__ __launch_bounds__(64)
void q27_attn_prep_q8b(const signed char* __restrict__ qkv8_t, const float* __restrict__ qkvs_t, int qstride,
                       const unsigned short* __restrict__ qnw,
                       const unsigned short* __restrict__ knw,
                       signed char* __restrict__ kcache, float* __restrict__ kscale,
                       signed char* __restrict__ vcache, float* __restrict__ vscale,
                       float* __restrict__ qout_t,
                       int pos0)
{
    const int c   = blockIdx.y;
    const int pos = pos0 + c;
    const signed char* __restrict__ qkv = qkv8_t + (size_t)c * qstride;
    const float* __restrict__ qsc = qkvs_t + (size_t)c * (qstride >> 4);
    float* __restrict__ qout = qout_t + (size_t)c * (NQ * Q27_HDIM);
    const int t    = threadIdx.x;
    const int b    = blockIdx.x;
    const bool isq = (b < NQ);
    const int head = isq ? b : (b - NQ);
    const int soff = isq ? (head * 512) : (QROWS + head * Q27_HDIM);
    const int voff = (QROWS + KVROWS) + (isq ? 0 : head * Q27_HDIM);
    const unsigned short* wgt = isq ? qnw : knw;
    float x[4], vf[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { const int k = soff + t + i * 64; x[i] = (float)qkv[k] * qsc[k >> 4]; }
#pragma unroll
    for (int i = 0; i < 4; ++i) { const int k = voff + t + i * 64; vf[i] = isq ? 0.0f : (float)qkv[k] * qsc[k >> 4]; }
    q27_attn_prep_tail<NQ, KVROWS>(x, wgt, vf, isq, head, t, pos, kcache, kscale, vcache, vscale, qout);
}

// HPW = query heads per wave (Q27_G = all six in one wave, as the decode kernel; 3 = two waves per
// (split, kv head, query), each with half the registers). Per-head arithmetic is unchanged, so the
// outputs are bit-identical for any HPW; K/V rows are loaded once per wave.
template<int NQ, int NKV, int KVROWS, int HPW, int PF>
__global__ __launch_bounds__(64)
void q27_attn_decode_kb(const float* __restrict__ q_t,
                        const signed char* __restrict__ kc, const float* __restrict__ ks,
                        const signed char* __restrict__ vc, const float* __restrict__ vs,
                        int pos0, float* __restrict__ pob, float* __restrict__ pml, int split_rows)
{
    static_assert(Q27_G % HPW == 0, "heads per wave must divide the GQA group");
    constexpr int WPH = Q27_G / HPW;           // waves per (split, kv head, query)
    constexpr int KSR = KVROWS / 16;
    const int c = blockIdx.z;
    const int L = pos0 + c + 1;
    int nsplits, rps; q27_attn_splits(L, nsplits, rps, split_rows);
    const int lane = threadIdx.x;
    const int hl   = lane >> 5;
    const int dg   = lane & 31;
    const int d0   = dg * Q27_VEC;
    const int sp   = blockIdx.x / WPH;
    const int hw   = blockIdx.x % WPH;         // which head group of this (split, kv head)
    const int kvh  = blockIdx.y;
    const int h0   = kvh * Q27_G + hw * HPW;
    if (sp >= nsplits) return;                 // this query has fewer splits than the launch grid
    const float* __restrict__ q = q_t + (size_t)c * (NQ * Q27_HDIM);
    const int r0 = sp * rps;
    const int r1 = (r0 + rps < L) ? (r0 + rps) : L;
    if (r0 >= r1) {
        if (lane < HPW) {
            Q27_PML(c, sp, h0 + lane, 0) = Q27_NEG_BIG;
            Q27_PML(c, sp, h0 + lane, 1) = 0.0f;
        }
        return;
    }
    float qv[HPW][Q27_VEC];
#pragma unroll
    for (int g = 0; g < HPW; ++g) {
        const float4 a = *reinterpret_cast<const float4*>(q + (h0 + g) * Q27_HDIM + d0);
        const float4 b = *reinterpret_cast<const float4*>(q + (h0 + g) * Q27_HDIM + d0 + 4);
        qv[g][0] = a.x; qv[g][1] = a.y; qv[g][2] = a.z; qv[g][3] = a.w;
        qv[g][4] = b.x; qv[g][5] = b.y; qv[g][6] = b.z; qv[g][7] = b.w;
    }
    float acc[HPW][Q27_VEC], mm[HPW], ll[HPW];
#pragma unroll
    for (int g = 0; g < HPW; ++g) {
        mm[g] = Q27_NEG_BIG;
        ll[g] = 0.0f;
#pragma unroll
        for (int j = 0; j < Q27_VEC; ++j) acc[g][j] = 0.0f;
    }
    const q27_bfly bf  = q27_bfly_init(lane);
    const size_t hoff  = (size_t)kvh * Q27_HDIM + (size_t)d0;
    const int    soff  = (int)(hoff >> 4);
    const int    niter = (r1 - r0 + Q27_RPW - 1) / Q27_RPW;
    int    rp = r0 + hl;  if (rp >= r1) rp = r1 - 1;
    uint2  kb = q27_ld8i(kc + (size_t)rp * KVROWS + hoff);
    uint2  vb = q27_ld8i(vc + (size_t)rp * KVROWS + hoff);
    float  kbs = ks[(size_t)rp * KSR + soff], vbs = vs[(size_t)rp * KSR + soff];
    uint2  kb2 = kb, vb2 = vb;
    float  kbs2 = kbs, vbs2 = vbs;
    if (PF >= 2) {
        int r2 = r0 + Q27_RPW + hl; if (r2 >= r1) r2 = r1 - 1;
        kb2 = q27_ld8i(kc + (size_t)r2 * KVROWS + hoff);
        vb2 = q27_ld8i(vc + (size_t)r2 * KVROWS + hoff);
        kbs2 = ks[(size_t)r2 * KSR + soff]; vbs2 = vs[(size_t)r2 * KSR + soff];
    }
    for (int it = 0; it < niter; ++it) {
        const int  r     = r0 + it * Q27_RPW + hl;
        const bool valid = (r < r1);
        const uint2 kcur = kb, vcur = vb;
        const float kcs = kbs, vcs = vbs;
        if (PF >= 2) {
            int rn = r0 + (it + 2) * Q27_RPW + hl;
            if (rn >= r1) rn = r1 - 1;
            kb = kb2; vb = vb2; kbs = kbs2; vbs = vbs2;
            kb2 = q27_ld8i(kc + (size_t)rn * KVROWS + hoff);
            vb2 = q27_ld8i(vc + (size_t)rn * KVROWS + hoff);
            kbs2 = ks[(size_t)rn * KSR + soff]; vbs2 = vs[(size_t)rn * KSR + soff];
        } else {
            int rn = r0 + (it + 1) * Q27_RPW + hl;
            if (rn >= r1) rn = r1 - 1;
            kb = q27_ld8i(kc + (size_t)rn * KVROWS + hoff);
            vb = q27_ld8i(vc + (size_t)rn * KVROWS + hoff);
            kbs = ks[(size_t)rn * KSR + soff]; vbs = vs[(size_t)rn * KSR + soff];
        }
        float kf[Q27_VEC];
        q27_unpack8i(kcur, kcs, kf);
        float sc[HPW];
#pragma unroll
        for (int g = 0; g < HPW; ++g) {
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < Q27_VEC; ++j) s = fmaf(qv[g][j], kf[j], s);
            sc[g] = s;
        }
#pragma unroll
        for (int g = 0; g < HPW; ++g) sc[g] = q27_red32(sc[g], bf);
        if (valid) {
            float vf[Q27_VEC];
            q27_unpack8i(vcur, vcs, vf);
#pragma unroll
            for (int g = 0; g < HPW; ++g) {
                const float x = sc[g] * Q27_SCALE_L2E;
                float p;
                if (x > mm[g]) {
                    const float cc = q27_exp2(mm[g] - x);
                    mm[g] = x;
                    ll[g] = ll[g] * cc + 1.0f;
#pragma unroll
                    for (int j = 0; j < Q27_VEC; ++j) acc[g][j] *= cc;
                    p = 1.0f;
                } else {
                    p = q27_exp2(x - mm[g]);
                    ll[g] += p;
                }
#pragma unroll
                for (int j = 0; j < Q27_VEC; ++j) acc[g][j] = fmaf(p, vf[j], acc[g][j]);
            }
        }
    }
#pragma unroll
    for (int g = 0; g < HPW; ++g) {
        const float mo = __shfl_xor(mm[g], 32, 64);
        const float lo = __shfl_xor(ll[g], 32, 64);
        const float M  = fmaxf(mm[g], mo);
        const float ca = q27_exp2(mm[g] - M), cb = q27_exp2(mo - M);
#pragma unroll
        for (int j = 0; j < Q27_VEC; ++j) {
            const float ao = __shfl_xor(acc[g][j], 32, 64);
            acc[g][j] = acc[g][j] * ca + ao * cb;
        }
        ll[g] = ll[g] * ca + lo * cb;
        mm[g] = M;
    }
    if (hl != 0) return;
#pragma unroll
    for (int g = 0; g < HPW; ++g) {
        float* dst = &Q27_POB(c, sp, h0 + g, d0);
        *reinterpret_cast<float4*>(dst)     = make_float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
        *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
        if (dg == 0) {
            Q27_PML(c, sp, h0 + g, 0) = mm[g];
            Q27_PML(c, sp, h0 + g, 1) = ll[g];
        }
    }
}

// combine + sigmoid gate + fused quantize; the bf16 `out` is optional (the LS path's o_proj
// consumes Q/QS only). Q8 selects the gate's source representation.

// ============================================================================================
// Q27_Q8_ATT_QPB=2: the decode-style chunk attention with TWO consecutive queries per block. Both queries
// share every K/V row load and int8 unpack (the dominant cost), doubling the FMAs per unpack. The split
// geometry of the PAIR is the odd query's (L1 = pos0 + c0 + 2); the even query masks rows >= its own L0.
// The combine must use the same pair geometry (pair2 = 1).
// ============================================================================================
template<int NQ, int NKV, int KVROWS, int HPW, int PF>
__global__ __launch_bounds__(64)
void q27_attn_decode_kb2(const float* __restrict__ q_t,
                         const signed char* __restrict__ kc, const float* __restrict__ ks,
                         const signed char* __restrict__ vc, const float* __restrict__ vs,
                         int pos0, int M, float* __restrict__ pob, float* __restrict__ pml, int split_rows)
{
    static_assert(Q27_G % HPW == 0, "heads per wave must divide the GQA group");
    constexpr int WPH = Q27_G / HPW;
    constexpr int KSR = KVROWS / 16;
    const int c0 = 2 * blockIdx.z;
    const bool has1 = (c0 + 1 < M);
    const int L0 = pos0 + c0 + 1, L1 = pos0 + c0 + 2;          // pair geometry from L1
    int nsplits, rps; q27_attn_splits(L1, nsplits, rps, split_rows);
    const int lane = threadIdx.x;
    const int hl   = lane >> 5;
    const int dg   = lane & 31;
    const int d0   = dg * Q27_VEC;
    const int sp   = blockIdx.x / WPH;
    const int hw   = blockIdx.x % WPH;
    const int kvh  = blockIdx.y;
    const int h0   = kvh * Q27_G + hw * HPW;
    if (sp >= nsplits) return;
    const int r0 = sp * rps;
    const int r1 = (r0 + rps < L1) ? (r0 + rps) : L1;
    if (r0 >= r1) {
        if (lane < HPW) {
            Q27_PML(c0, sp, h0 + lane, 0) = Q27_NEG_BIG; Q27_PML(c0, sp, h0 + lane, 1) = 0.0f;
            if (has1) { Q27_PML(c0 + 1, sp, h0 + lane, 0) = Q27_NEG_BIG; Q27_PML(c0 + 1, sp, h0 + lane, 1) = 0.0f; }
        }
        return;
    }
    float qv[2][HPW][Q27_VEC];
#pragma unroll
    for (int qq = 0; qq < 2; ++qq) {
        const float* __restrict__ q = q_t + (size_t)(c0 + ((qq && has1) ? 1 : 0)) * (NQ * Q27_HDIM);
#pragma unroll
        for (int g = 0; g < HPW; ++g) {
            const float4 a = *reinterpret_cast<const float4*>(q + (h0 + g) * Q27_HDIM + d0);
            const float4 b = *reinterpret_cast<const float4*>(q + (h0 + g) * Q27_HDIM + d0 + 4);
            qv[qq][g][0] = a.x; qv[qq][g][1] = a.y; qv[qq][g][2] = a.z; qv[qq][g][3] = a.w;
            qv[qq][g][4] = b.x; qv[qq][g][5] = b.y; qv[qq][g][6] = b.z; qv[qq][g][7] = b.w;
        }
    }
    float acc[2][HPW][Q27_VEC], mm[2][HPW], ll[2][HPW];
#pragma unroll
    for (int qq = 0; qq < 2; ++qq)
#pragma unroll
        for (int g = 0; g < HPW; ++g) { mm[qq][g] = Q27_NEG_BIG; ll[qq][g] = 0.0f;
#pragma unroll
            for (int j = 0; j < Q27_VEC; ++j) acc[qq][g][j] = 0.0f; }
    const q27_bfly bf  = q27_bfly_init(lane);
    const size_t hoff  = (size_t)kvh * Q27_HDIM + (size_t)d0;
    const int    soff  = (int)(hoff >> 4);
    const int    niter = (r1 - r0 + Q27_RPW - 1) / Q27_RPW;
    int    rp = r0 + hl;  if (rp >= r1) rp = r1 - 1;
    uint2  kb = q27_ld8i(kc + (size_t)rp * KVROWS + hoff);
    uint2  vb = q27_ld8i(vc + (size_t)rp * KVROWS + hoff);
    float  kbs = ks[(size_t)rp * KSR + soff], vbs = vs[(size_t)rp * KSR + soff];
    for (int it = 0; it < niter; ++it) {
        const int  r     = r0 + it * Q27_RPW + hl;
        const bool valid1 = (r < r1), valid0 = (r < L0);
        const uint2 kcur = kb, vcur = vb;
        const float kcs = kbs, vcs = vbs;
        {
            int rn = r0 + (it + 1) * Q27_RPW + hl;
            if (rn >= r1) rn = r1 - 1;
            kb = q27_ld8i(kc + (size_t)rn * KVROWS + hoff);
            vb = q27_ld8i(vc + (size_t)rn * KVROWS + hoff);
            kbs = ks[(size_t)rn * KSR + soff]; vbs = vs[(size_t)rn * KSR + soff];
        }
        float kf[Q27_VEC];
        q27_unpack8i(kcur, kcs, kf);
        float sc[2][HPW];
#pragma unroll
        for (int qq = 0; qq < 2; ++qq)
#pragma unroll
            for (int g = 0; g < HPW; ++g) {
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < Q27_VEC; ++j) s = fmaf(qv[qq][g][j], kf[j], s);
                sc[qq][g] = s;
            }
#pragma unroll
        for (int qq = 0; qq < 2; ++qq)
#pragma unroll
            for (int g = 0; g < HPW; ++g) sc[qq][g] = q27_red32(sc[qq][g], bf);
        if (valid1) {
            float vf[Q27_VEC];
            q27_unpack8i(vcur, vcs, vf);
#pragma unroll
            for (int qq = 0; qq < 2; ++qq) {
                if (qq == 0 && !valid0) continue;              // the even query does not see this row
                if (qq == 1 && !has1) continue;
#pragma unroll
                for (int g = 0; g < HPW; ++g) {
                    const float x = sc[qq][g] * Q27_SCALE_L2E;
                    float p;
                    if (x > mm[qq][g]) {
                        const float cc = q27_exp2(mm[qq][g] - x);
                        mm[qq][g] = x;
                        ll[qq][g] = ll[qq][g] * cc + 1.0f;
#pragma unroll
                        for (int j = 0; j < Q27_VEC; ++j) acc[qq][g][j] *= cc;
                        p = 1.0f;
                    } else {
                        p = q27_exp2(x - mm[qq][g]);
                        ll[qq][g] += p;
                    }
#pragma unroll
                    for (int j = 0; j < Q27_VEC; ++j) acc[qq][g][j] = fmaf(p, vf[j], acc[qq][g][j]);
                }
            }
        }
    }
#pragma unroll
    for (int qq = 0; qq < 2; ++qq)
#pragma unroll
        for (int g = 0; g < HPW; ++g) {
            const float mo = __shfl_xor(mm[qq][g], 32, 64);
            const float lo = __shfl_xor(ll[qq][g], 32, 64);
            const float Mx = fmaxf(mm[qq][g], mo);
            const float ca = (mm[qq][g] > Q27_SENT_TEST) ? q27_exp2(mm[qq][g] - Mx) : 0.0f, cb = (mo > Q27_SENT_TEST) ? q27_exp2(mo - Mx) : 0.0f;
#pragma unroll
            for (int j = 0; j < Q27_VEC; ++j) {
                const float ao = __shfl_xor(acc[qq][g][j], 32, 64);
                acc[qq][g][j] = acc[qq][g][j] * ca + ao * cb;
            }
            ll[qq][g] = ll[qq][g] * ca + lo * cb;
            mm[qq][g] = Mx;
        }
    if (hl != 0) return;
#pragma unroll
    for (int qq = 0; qq < 2; ++qq) {
        if (qq == 1 && !has1) break;
        const int c = c0 + qq;
#pragma unroll
        for (int g = 0; g < HPW; ++g) {
            float* dst = &Q27_POB(c, sp, h0 + g, d0);
            *reinterpret_cast<float4*>(dst)     = make_float4(acc[qq][g][0], acc[qq][g][1], acc[qq][g][2], acc[qq][g][3]);
            *reinterpret_cast<float4*>(dst + 4) = make_float4(acc[qq][g][4], acc[qq][g][5], acc[qq][g][6], acc[qq][g][7]);
            if (dg == 0) {
                Q27_PML(c, sp, h0 + g, 0) = mm[qq][g];
                Q27_PML(c, sp, h0 + g, 1) = ll[qq][g];
            }
        }
    }
}
template<int NQ, bool Q8>
__global__ __launch_bounds__(Q27_HDIM)
void q27_attn_combine_kb(const void* __restrict__ qraw_v, const float* __restrict__ qraws_t, int qstride,
                         unsigned short* __restrict__ out_t, int ostride,
                         int pos0,
                         signed char* __restrict__ Q_t, float* __restrict__ QS_t, float qgs,
                         const float* __restrict__ pob, const float* __restrict__ pml, int perm, int g64, int split_rows, int pair2)
{
    const int c = blockIdx.y;
    const int L = pos0 + (pair2 ? (c | 1) : c) + 1;   // pair2: the split geometry of the query PAIR (see q27_attn_decode_kb2)
    int nsplits, rps; q27_attn_splits(L, nsplits, rps, split_rows);
    unsigned short* __restrict__ out = out_t ? out_t + (size_t)c * ostride : nullptr;
    signed char* __restrict__ Q  = Q_t  ? Q_t  + (size_t)c * ostride : nullptr;
    float*       __restrict__ QS = QS_t ? QS_t + (size_t)c * (g64 ? (ostride >> 6) : (ostride >> 4)) : nullptr;
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    float M = Q27_NEG_BIG;
    for (int s = 0; s < nsplits; ++s) M = fmaxf(M, Q27_PML(c, s, h, 0));
    float den = 0.0f, o = 0.0f;
    for (int s = 0; s < nsplits; ++s) {
        const float ms = Q27_PML(c, s, h, 0);
        if (ms < Q27_SENT_TEST) continue;
        const float w = q27_exp2(ms - M);
        den = fmaf(Q27_PML(c, s, h, 1), w, den);
        o   = fmaf(Q27_POB(c, s, h, d), w, o);
    }
    const float inv = (den > 0.0f) ? (1.0f / den) : 0.0f;
    const int gk = h * 512 + Q27_HDIM + d;
    float gt;
    if (Q8) {
        const signed char* __restrict__ qraw = (const signed char*)qraw_v + (size_t)c * qstride;
        const float* __restrict__ qs = qraws_t + (size_t)c * (qstride >> 4);
        gt = (float)qraw[gk] * qs[gk >> 4];
    } else {
        const unsigned short* __restrict__ qraw = (const unsigned short*)qraw_v + (size_t)c * qstride;
        gt = q27_bf2f(qraw[gk]);
    }
    const float sg  = 1.0f / (1.0f + __expf(-gt));
    const float yv = o * inv * sg;
    if (out) out[h * Q27_HDIM + d] = q27_f2bf(yv);
    if (Q) {
        float mx = fabsf(yv);
        if (g64) { for (int o64 = 32; o64; o64 >>= 1) mx = fmaxf(mx, __shfl_down(mx, o64, 64)); mx = __shfl(mx, 0, 64); }
        else     { for (int o16 = 8; o16; o16 >>= 1) mx = fmaxf(mx, __shfl_down(mx, o16, 16)); mx = __shfl(mx, 0, 16); }
        const float sc = mx * qgs * (1.0f / 127.0f);
        const float qi = (sc > 0.0f) ? (qgs / sc) : 0.0f;
        const int   k  = h * Q27_HDIM + d;
        if (g64) { if ((d & 63) == 0) QS[k >> 6] = sc; } else if ((d & 15) == 0) QS[k >> 4] = sc;
        int cq = (int)rintf(yv * qi);
        cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
        // Q8: the NVFP4 o_proj consumes the even-then-odd order inside each group of 16 (q27.h contract)
        const int kq = perm ? ((k & ~15) | ((d & 1) ? (8 + ((d & 15) >> 1)) : ((d & 15) >> 1))) : k;
        Q[kq] = (signed char)cq;
    }
}

#define Q27_TP_CHUNK_DECODE(G) \
        if (hpw == 3) { if (attpf >= 2) hipLaunchKernelGGL((q27_attn_decode_kb<Q27_NHEAD/G, Q27_NKV/G, Q27_KVROWS/G, 3, 2>), \
                dim3(nsmax * 2, Q27_NKV/G, M), dim3(64), 0, s, qout_t, kcache, kscale, vcache, vscale, pos0, pob, pml, srows); \
            else hipLaunchKernelGGL((q27_attn_decode_kb<Q27_NHEAD/G, Q27_NKV/G, Q27_KVROWS/G, 3, 1>), \
                dim3(nsmax * 2, Q27_NKV/G, M), dim3(64), 0, s, qout_t, kcache, kscale, vcache, vscale, pos0, pob, pml, srows); } \
        else if (hpw == 2) { if (attpf >= 2) hipLaunchKernelGGL((q27_attn_decode_kb<Q27_NHEAD/G, Q27_NKV/G, Q27_KVROWS/G, 2, 2>), \
                dim3(nsmax * 3, Q27_NKV/G, M), dim3(64), 0, s, qout_t, kcache, kscale, vcache, vscale, pos0, pob, pml, srows); \
            else hipLaunchKernelGGL((q27_attn_decode_kb<Q27_NHEAD/G, Q27_NKV/G, Q27_KVROWS/G, 2, 1>), \
                dim3(nsmax * 3, Q27_NKV/G, M), dim3(64), 0, s, qout_t, kcache, kscale, vcache, vscale, pos0, pob, pml, srows); } \
        else hipLaunchKernelGGL((q27_attn_decode_kb<Q27_NHEAD/G, Q27_NKV/G, Q27_KVROWS/G, Q27_G, 1>), \
                dim3(nsmax, Q27_NKV/G, M), dim3(64), 0, s, qout_t, kcache, kscale, vcache, vscale, pos0, pob, pml, srows);

#define Q27_TP_CHUNK_CASE(G) \
    case G: { \
        hipLaunchKernelGGL((q27_attn_prep_kb<Q27_NHEAD/G, Q27_NKV/G, Q27_QROWS/G, Q27_KVROWS/G>), \
                dim3(Q27_NHEAD/G + Q27_NKV/G, M), dim3(64), 0, s, \
                qkv_t, qstride, q_norm_w, k_norm_w, kcache, kscale, vcache, vscale, qout_t, pos0); \
        Q27_TP_CHUNK_DECODE(G) \
        hipLaunchKernelGGL((q27_attn_combine_kb<Q27_NHEAD/G, false>), dim3(Q27_NHEAD/G, M), dim3(Q27_HDIM), 0, s, \
                (const void*)qkv_t, (const float*)nullptr, qstride, out_t, ostride, pos0, Q_t, QS_t, qgs, pob, pml, 0, 0, srows, 0); \
        return 1; }


// ============================================================================================
// Q27_Q8_ATTV2: TILED prefill attention for the Q8 chunk path (replaces decode_kb + combine).
// Block = TQ query positions x the G query heads of ONE kv head (48 rows), 4 threads per row (64 dims each).
// Key tiles of TK positions are staged in LDS (int8 + per-16 scales) once per block and reused by all rows;
// online softmax in exp2 space (scores pre-scaled by Q27_SCALE_L2E); output gated by sigmoid(raw gate) and
// quantized to int8 with per-64 (or per-16) scales straight into the o_proj input buffer.
// ============================================================================================
template<int NQ, int NKV, int KVROWS, int TQ, int TK, int LPR>
__global__ __launch_bounds__(TQ * (NQ / NKV) * LPR)
void q27_attn_tile_q8(const float* __restrict__ q_t,
                      const signed char* __restrict__ kc, const float* __restrict__ ks,
                      const signed char* __restrict__ vc, const float* __restrict__ vs,
                      const signed char* __restrict__ qraw_t, const float* __restrict__ qraws_t, int qstride,
                      int pos0, int M, signed char* __restrict__ Q_t, float* __restrict__ QS_t, int ostride, float qgs, int g64)
{
    constexpr int G = NQ / NKV, ROWS = TQ * G, NT = ROWS * LPR, DPT = Q27_HDIM / LPR, KSR = KVROWS / 16;
    static_assert(LPR == 4 || LPR == 8, "4 or 8 lanes per row");
    static_assert(NT <= 1024 && (NT % 64) == 0, "whole waves");
    __shared__ __align__(16) signed char sK[TK * Q27_HDIM];
    __shared__ __align__(16) signed char sV[TK * Q27_HDIM];
    __shared__ __align__(16) float sKs[TK * 16];
    __shared__ __align__(16) float sVs[TK * 16];
    const int tid = threadIdx.x, r = tid / LPR, lr = tid % LPR;
    const int kvh = blockIdx.y, qi = r / G, hg = r - qi * G;
    const int c = blockIdx.x * TQ + qi, h = kvh * G + hg;
    const bool live = (c < M);
    const int cc = live ? c : (M - 1);                    // clamped row for address math
    const int mypos = pos0 + cc;                          // causal limit: keys <= mypos
    const int tile_kmax = pos0 + min(blockIdx.x * TQ + TQ - 1, M - 1);   // last key any row of this block needs
    const int d0 = lr * DPT;
    float q[DPT];
    {
        const float* qp = q_t + (size_t)cc * (NQ * Q27_HDIM) + h * Q27_HDIM + d0;
#pragma unroll
        for (int i = 0; i < DPT; i += 4) { const float4 v = *(const float4*)(qp + i); q[i] = v.x * Q27_SCALE_L2E; q[i + 1] = v.y * Q27_SCALE_L2E; q[i + 2] = v.z * Q27_SCALE_L2E; q[i + 3] = v.w * Q27_SCALE_L2E; }
    }
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
    float m = Q27_NEG_BIG, l = 0.f;
    const size_t hoff = (size_t)kvh * Q27_HDIM;
    for (int k0 = 0; k0 <= tile_kmax; k0 += TK) {
        const int nk = min(TK, tile_kmax - k0 + 1);
        __syncthreads();                                                  // previous tile fully consumed
        for (int i = tid; i < (TK * Q27_HDIM) / 16; i += NT) {           // K and V tiles: int4 units
            const int row = i / (Q27_HDIM / 16), col = (i - row * (Q27_HDIM / 16)) * 16;
            const int key = k0 + row;
            int4 kk = make_int4(0, 0, 0, 0), vv = make_int4(0, 0, 0, 0);
            if (row < nk) { kk = *(const int4*)(kc + (size_t)key * KVROWS + hoff + col); vv = *(const int4*)(vc + (size_t)key * KVROWS + hoff + col); }
            *(int4*)(sK + row * Q27_HDIM + col) = kk; *(int4*)(sV + row * Q27_HDIM + col) = vv;
        }
        for (int i = tid; i < TK * 16; i += NT) {                         // per-16 scales of the kv head's 256 dims
            const int row = i >> 4, g16 = i & 15; const int key = k0 + row;
            float a = 0.f, b = 0.f;
            if (row < nk) { a = ks[(size_t)key * KSR + (hoff >> 4) + g16]; b = vs[(size_t)key * KSR + (hoff >> 4) + g16]; }
            sKs[i] = a; sVs[i] = b;
        }
        __syncthreads();
        if (!live) continue;
        for (int j0 = 0; j0 < nk; j0 += TK) {                            // score TK keys, then softmax-update with them
            const int nj = min(TK, nk - j0);
            float sc[TK];
#pragma unroll
            for (int j = 0; j < TK; ++j) {
                float s = 0.f;
                if (j < nj) {
                    const signed char* kr = sK + (j0 + j) * Q27_HDIM + d0;
                    const float* ksr = sKs + (j0 + j) * 16 + (d0 >> 4);
#pragma unroll
                    for (int gq = 0; gq < DPT / 16; ++gq) {
                        const int4 raw = *(const int4*)(kr + gq * 16);
                        const float kscale = ksr[gq];
                        const unsigned w[4] = { (unsigned)raw.x, (unsigned)raw.y, (unsigned)raw.z, (unsigned)raw.w };
                        float ps = 0.f;
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            ps = fmaf(q[gq * 16 + b * 4 + 0], (float)((int)(w[b] << 24) >> 24), ps);
                            ps = fmaf(q[gq * 16 + b * 4 + 1], (float)((int)(w[b] << 16) >> 24), ps);
                            ps = fmaf(q[gq * 16 + b * 4 + 2], (float)((int)(w[b] <<  8) >> 24), ps);
                            ps = fmaf(q[gq * 16 + b * 4 + 3], (float)((int) w[b]        >> 24), ps);
                        }
                        s = fmaf(ps, kscale, s);
                    }
                }
                s += __shfl_xor(s, 1, LPR); s += __shfl_xor(s, 2, LPR); if (LPR == 8) s += __shfl_xor(s, 4, LPR);   // full 256-dim dot in every lane of the row
                sc[j] = (j < nj && (k0 + j0 + j) <= mypos) ? s : Q27_NEG_BIG;
            }
            float tmax = Q27_NEG_BIG;
#pragma unroll
            for (int j = 0; j < TK; ++j) tmax = fmaxf(tmax, sc[j]);
            if (tmax <= Q27_SENT_TEST) continue;                         // every key of this half is masked
            const float mn = fmaxf(m, tmax);
            const float cf = q27_exp2(m - mn);
            l *= cf;
#pragma unroll
            for (int i = 0; i < DPT; ++i) acc[i] *= cf;
            m = mn;
#pragma unroll
            for (int j = 0; j < TK; ++j) {
                if (sc[j] <= Q27_SENT_TEST) continue;
                const float p = q27_exp2(sc[j] - m);
                l += p;
                const signed char* vr = sV + (j0 + j) * Q27_HDIM + d0;
                const float* vsr = sVs + (j0 + j) * 16 + (d0 >> 4);
#pragma unroll
                for (int gq = 0; gq < DPT / 16; ++gq) {
                    const int4 raw = *(const int4*)(vr + gq * 16);
                    const float pv = p * vsr[gq];
                    const unsigned w[4] = { (unsigned)raw.x, (unsigned)raw.y, (unsigned)raw.z, (unsigned)raw.w };
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        acc[gq * 16 + b * 4 + 0] = fmaf(pv, (float)((int)(w[b] << 24) >> 24), acc[gq * 16 + b * 4 + 0]);
                        acc[gq * 16 + b * 4 + 1] = fmaf(pv, (float)((int)(w[b] << 16) >> 24), acc[gq * 16 + b * 4 + 1]);
                        acc[gq * 16 + b * 4 + 2] = fmaf(pv, (float)((int)(w[b] <<  8) >> 24), acc[gq * 16 + b * 4 + 2]);
                        acc[gq * 16 + b * 4 + 3] = fmaf(pv, (float)((int) w[b]        >> 24), acc[gq * 16 + b * 4 + 3]);
                    }
                }
            }
        }
    }
    if (!live) return;
    const float inv = (l > 0.f) ? 1.f / l : 0.f;
    // gate = sigmoid(raw q row, second half of the head's 512 columns), then per-group int8 quantization
    const signed char* __restrict__ qraw = qraw_t + (size_t)c * qstride;
    const float* __restrict__ qs = qraws_t + (size_t)c * (qstride >> 4);
    float y[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) {
        const int gk = h * 512 + Q27_HDIM + d0 + i;
        const float gt = (float)qraw[gk] * qs[gk >> 4];
        y[i] = acc[i] * inv * (1.0f / (1.0f + __expf(-gt)));
    }
    signed char* __restrict__ Q = Q_t + (size_t)c * ostride + h * Q27_HDIM + d0;
    const int GS = g64 ? 64 : 16;
    float* __restrict__ QS = QS_t + (size_t)c * (ostride / GS) + (h * Q27_HDIM + d0) / GS;
#pragma unroll
    for (int gs0 = 0; gs0 < DPT; gs0 += 16) {
        // per-16 groups always computed; a per-64 group spans DPT/16 threads-worth when DPT < 64: reduce across lanes
        if (g64 && (gs0 > 0)) continue;
        const int span = g64 ? DPT : 16;
        float mx = 0.f;
        for (int i = 0; i < span; ++i) mx = fmaxf(mx, fabsf(y[gs0 + i]));
        if (g64 && DPT < 64) { mx = fmaxf(mx, __shfl_xor(mx, 1, 64 / DPT)); if (DPT < 32) mx = fmaxf(mx, __shfl_xor(mx, 2, 64 / DPT)); }
        const float sc = mx * qgs * (1.0f / 127.0f);
        const float qi = (sc > 0.f) ? (qgs / sc) : 0.f;
        if (!g64 || (d0 % 64) == 0) QS[gs0 / 16] = sc;
        for (int i = 0; i < span; i += 4) {
            int c0 = (int)rintf(y[gs0 + i] * qi), c1 = (int)rintf(y[gs0 + i + 1] * qi), c2 = (int)rintf(y[gs0 + i + 2] * qi), c3 = (int)rintf(y[gs0 + i + 3] * qi);
            c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
            c2 = c2 < -127 ? -127 : (c2 > 127 ? 127 : c2); c3 = c3 < -127 ? -127 : (c3 > 127 ? 127 : c3);
            *(unsigned*)(Q + gs0 + i) = ((unsigned)c0 & 0xFFu) | (((unsigned)c1 & 0xFFu) << 8) | (((unsigned)c2 & 0xFFu) << 16) | (((unsigned)c3 & 0xFFu) << 24);
        }
    }
}

// ---- ATTV2=32: fp32 LDS tiles. K/V (int8, per-16 scales) are dequantized ONCE per tile while staging (amortized
// over the block's 48 rows; the row-groups of a wave read the same key addresses -> LDS broadcast), so the inner
// loops are pure float4 loads + FMAs. TK = 16 keys per tile (32 KB of LDS), 8 lanes per row (32 dims each).
template<int NQ, int NKV, int KVROWS, int TQ, int TK>
__global__ __launch_bounds__(TQ * (NQ / NKV) * 8)
void q27_attn_tile32_q8(const float* __restrict__ q_t,
                        const signed char* __restrict__ kc, const float* __restrict__ ks,
                        const signed char* __restrict__ vc, const float* __restrict__ vs,
                        const signed char* __restrict__ qraw_t, const float* __restrict__ qraws_t, int qstride,
                        int pos0, int M, signed char* __restrict__ Q_t, float* __restrict__ QS_t, int ostride, float qgs, int g64)
{
    constexpr int LPR = 8, G = NQ / NKV, ROWS = TQ * G, NT = ROWS * LPR, DPT = Q27_HDIM / LPR, KSR = KVROWS / 16;
    __shared__ __align__(16) float sK[TK * Q27_HDIM];
    __shared__ __align__(16) float sV[TK * Q27_HDIM];
    const int tid = threadIdx.x, r = tid / LPR, lr = tid % LPR;
    const int kvh = blockIdx.y, qi = r / G, hg = r - qi * G;
    const int c = blockIdx.x * TQ + qi, h = kvh * G + hg;
    const bool live = (c < M);
    const int cc = live ? c : (M - 1);
    const int mypos = pos0 + cc;
    const int tile_kmax = pos0 + min(blockIdx.x * TQ + TQ - 1, M - 1);
    const int d0 = lr * DPT;
    float q[DPT];
    {
        const float* qp = q_t + (size_t)cc * (NQ * Q27_HDIM) + h * Q27_HDIM + d0;
#pragma unroll
        for (int i = 0; i < DPT; i += 4) { const float4 v = *(const float4*)(qp + i); q[i] = v.x * Q27_SCALE_L2E; q[i + 1] = v.y * Q27_SCALE_L2E; q[i + 2] = v.z * Q27_SCALE_L2E; q[i + 3] = v.w * Q27_SCALE_L2E; }
    }
    float acc[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) acc[i] = 0.f;
    float m = Q27_NEG_BIG, l = 0.f;
    const size_t hoff = (size_t)kvh * Q27_HDIM;
    for (int k0 = 0; k0 <= tile_kmax; k0 += TK) {
        const int nk = min(TK, tile_kmax - k0 + 1);
        __syncthreads();
        // stage: each thread converts 16-byte int8 units (one per-16 scale group) of K and V into fp32
        for (int i = tid; i < TK * (Q27_HDIM / 16); i += NT) {
            const int row = i / (Q27_HDIM / 16), g16 = i - row * (Q27_HDIM / 16), col = g16 * 16;
            const int key = k0 + row;
            float4 o[4], p[4];
            if (row < nk) {
                const int4 kk = *(const int4*)(kc + (size_t)key * KVROWS + hoff + col);
                const int4 vv = *(const int4*)(vc + (size_t)key * KVROWS + hoff + col);
                const float ksc = ks[(size_t)key * KSR + (hoff >> 4) + g16], vsc = vs[(size_t)key * KSR + (hoff >> 4) + g16];
                const unsigned kw[4] = { (unsigned)kk.x, (unsigned)kk.y, (unsigned)kk.z, (unsigned)kk.w };
                const unsigned vw[4] = { (unsigned)vv.x, (unsigned)vv.y, (unsigned)vv.z, (unsigned)vv.w };
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    o[b].x = (float)((int)(kw[b] << 24) >> 24) * ksc; o[b].y = (float)((int)(kw[b] << 16) >> 24) * ksc;
                    o[b].z = (float)((int)(kw[b] <<  8) >> 24) * ksc; o[b].w = (float)((int) kw[b]        >> 24) * ksc;
                    p[b].x = (float)((int)(vw[b] << 24) >> 24) * vsc; p[b].y = (float)((int)(vw[b] << 16) >> 24) * vsc;
                    p[b].z = (float)((int)(vw[b] <<  8) >> 24) * vsc; p[b].w = (float)((int) vw[b]        >> 24) * vsc;
                }
            } else {
#pragma unroll
                for (int b = 0; b < 4; ++b) { o[b] = make_float4(0.f, 0.f, 0.f, 0.f); p[b] = o[b]; }
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) { *(float4*)(sK + row * Q27_HDIM + col + 4 * b) = o[b]; *(float4*)(sV + row * Q27_HDIM + col + 4 * b) = p[b]; }
        }
        __syncthreads();
        if (!live) continue;
        float sc[TK];
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            float s = 0.f;
            if (j < nk) {
                const float* kr = sK + j * Q27_HDIM + d0;
#pragma unroll
                for (int i = 0; i < DPT; i += 4) { const float4 kv = *(const float4*)(kr + i); s = fmaf(q[i], kv.x, s); s = fmaf(q[i + 1], kv.y, s); s = fmaf(q[i + 2], kv.z, s); s = fmaf(q[i + 3], kv.w, s); }
            }
            s += __shfl_xor(s, 1, LPR); s += __shfl_xor(s, 2, LPR); s += __shfl_xor(s, 4, LPR);
            sc[j] = (j < nk && (k0 + j) <= mypos) ? s : Q27_NEG_BIG;
        }
        float tmax = Q27_NEG_BIG;
#pragma unroll
        for (int j = 0; j < TK; ++j) tmax = fmaxf(tmax, sc[j]);
        if (tmax <= Q27_SENT_TEST) continue;
        const float mn = fmaxf(m, tmax);
        const float cf = q27_exp2(m - mn);
        l *= cf;
#pragma unroll
        for (int i = 0; i < DPT; ++i) acc[i] *= cf;
        m = mn;
#pragma unroll
        for (int j = 0; j < TK; ++j) {
            if (sc[j] <= Q27_SENT_TEST) continue;
            const float p = q27_exp2(sc[j] - m);
            l += p;
            const float* vr = sV + j * Q27_HDIM + d0;
#pragma unroll
            for (int i = 0; i < DPT; i += 4) { const float4 vv = *(const float4*)(vr + i); acc[i] = fmaf(p, vv.x, acc[i]); acc[i + 1] = fmaf(p, vv.y, acc[i + 1]); acc[i + 2] = fmaf(p, vv.z, acc[i + 2]); acc[i + 3] = fmaf(p, vv.w, acc[i + 3]); }
        }
    }
    if (!live) return;
    const float inv = (l > 0.f) ? 1.f / l : 0.f;
    const signed char* __restrict__ qraw = qraw_t + (size_t)c * qstride;
    const float* __restrict__ qs = qraws_t + (size_t)c * (qstride >> 4);
    float y[DPT];
#pragma unroll
    for (int i = 0; i < DPT; ++i) {
        const int gk = h * 512 + Q27_HDIM + d0 + i;
        const float gt = (float)qraw[gk] * qs[gk >> 4];
        y[i] = acc[i] * inv * (1.0f / (1.0f + __expf(-gt)));
    }
    signed char* __restrict__ Q = Q_t + (size_t)c * ostride + h * Q27_HDIM + d0;
    const int GS = g64 ? 64 : 16;
    float* __restrict__ QS = QS_t + (size_t)c * (ostride / GS) + (h * Q27_HDIM + d0) / GS;
    // per-16 groups: 2 per thread; per-64: the 64-group spans 2 threads (32 dims each) -> one xor-1 exchange
    for (int gs0 = 0; gs0 < DPT; gs0 += 16) {
        if (g64 && gs0 > 0) continue;
        const int span = g64 ? DPT : 16;
        float mx = 0.f;
        for (int i = 0; i < span; ++i) mx = fmaxf(mx, fabsf(y[gs0 + i]));
        if (g64) mx = fmaxf(mx, __shfl_xor(mx, 1, LPR));
        const float sc = mx * qgs * (1.0f / 127.0f);
        const float qi = (sc > 0.f) ? (qgs / sc) : 0.f;
        if (!g64 || (lr & 1) == 0) QS[gs0 / 16] = sc;
        for (int i = 0; i < span; i += 4) {
            int c0 = (int)rintf(y[gs0 + i] * qi), c1 = (int)rintf(y[gs0 + i + 1] * qi), c2 = (int)rintf(y[gs0 + i + 2] * qi), c3 = (int)rintf(y[gs0 + i + 3] * qi);
            c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
            c2 = c2 < -127 ? -127 : (c2 > 127 ? 127 : c2); c3 = c3 < -127 ? -127 : (c3 > 127 ? 127 : c3);
            *(unsigned*)(Q + gs0 + i) = ((unsigned)c0 & 0xFFu) | (((unsigned)c1 & 0xFFu) << 8) | (((unsigned)c2 & 0xFFu) << 16) | (((unsigned)c3 & 0xFFu) << 24);
        }
    }
}
static int q27_attv2_on() { static const int v = q27_env_int("Q27_Q8_ATTV2", 0); return v; }   // 0 off; 4 = 4 lanes/row (1.83 ms/call at 1K); 8 = 8 lanes/row, 32 keys per pass
#define Q27_TP_CHUNK_Q8_CASE(G) \
    case G: { \
        hipLaunchKernelGGL((q27_attn_prep_q8b<Q27_NHEAD/G, Q27_NKV/G, Q27_QROWS/G, Q27_KVROWS/G>), \
                dim3(Q27_NHEAD/G + Q27_NKV/G, M), dim3(64), 0, s, \
                qkv8_t, qkvs_t, qstride, q_norm_w, k_norm_w, kcache, kscale, vcache, vscale, qout_t, pos0); \
        if (G == 1 && qpb2) { \
            hipLaunchKernelGGL((q27_attn_decode_kb2<Q27_NHEAD, Q27_NKV, Q27_KVROWS, 2, 1>), dim3(nsmax * 3, Q27_NKV, (M + 1) / 2), dim3(64), 0, s, \
                    qout_t, kcache, kscale, vcache, vscale, pos0, M, pob, pml, srows); \
            hipLaunchKernelGGL((q27_attn_combine_kb<Q27_NHEAD, true>), dim3(Q27_NHEAD, M), dim3(Q27_HDIM), 0, s, \
                    (const void*)qkv8_t, qkvs_t, qstride, out_t, ostride, pos0, Q_t, QS_t, qgs, pob, pml, perm, g64, srows, 1); \
            return 1; } \
        if (G == 1 && Q_t && QS_t && !perm && q27_attv2_on()) { \
            if (q27_attv2_on() == 32) hipLaunchKernelGGL((q27_attn_tile32_q8<Q27_NHEAD, Q27_NKV, Q27_KVROWS, 8, 16>), dim3((M + 7) / 8, Q27_NKV), dim3(384), 0, s, \
                    qout_t, kcache, kscale, vcache, vscale, qkv8_t, qkvs_t, qstride, pos0, M, Q_t, QS_t, ostride, qgs, g64); \
            else if (q27_attv2_on() == 8) hipLaunchKernelGGL((q27_attn_tile_q8<Q27_NHEAD, Q27_NKV, Q27_KVROWS, 8, 32, 8>), dim3((M + 7) / 8, Q27_NKV), dim3(384), 0, s, \
                    qout_t, kcache, kscale, vcache, vscale, qkv8_t, qkvs_t, qstride, pos0, M, Q_t, QS_t, ostride, qgs, g64); \
            else hipLaunchKernelGGL((q27_attn_tile_q8<Q27_NHEAD, Q27_NKV, Q27_KVROWS, 8, 16, 4>), dim3((M + 7) / 8, Q27_NKV), dim3(192), 0, s, \
                    qout_t, kcache, kscale, vcache, vscale, qkv8_t, qkvs_t, qstride, pos0, M, Q_t, QS_t, ostride, qgs, g64); \
            return 1; } \
        Q27_TP_CHUNK_DECODE(G) \
        hipLaunchKernelGGL((q27_attn_combine_kb<Q27_NHEAD/G, true>), dim3(Q27_NHEAD/G, M), dim3(Q27_HDIM), 0, s, \
                (const void*)qkv8_t, qkvs_t, qstride, out_t, ostride, pos0, Q_t, QS_t, qgs, pob, pml, perm, g64, srows, (G == 1 && qpb2) ? 1 : 0); \
        return 1; }

// Returns 1 when the chunk ran (the fused quantize ran iff in_scale > 0).
extern "C" int q27_attn_chunk_tp(const unsigned short* qkv_t, int qstride,
                                 const unsigned short* q_norm_w, const unsigned short* k_norm_w,
                                 signed char* kcache, float* kscale, signed char* vcache, float* vscale,
                                 float* qout_t, unsigned short* out_t, int ostride,
                                 int pos0, int M, int ndev,
                                 signed char* Q_t, float* QS_t, float in_scale, int hpw, int attpf,
                                 float* pob, float* pml, hipStream_t s)
{
    if (M < 1 || M > Q27_ATT_CHUNK_MAX || pos0 < 0 || !pob || !pml) return 0;
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { Q_t = nullptr; QS_t = nullptr; }
    const int srows = Q27_SPLIT_ROWS;
    int nsmax, rpsmax; q27_attn_splits_host(pos0 + M, nsmax, rpsmax, srows);
    switch (ndev) {
    Q27_TP_CHUNK_CASE(1)
    Q27_TP_CHUNK_CASE(2)
    Q27_TP_CHUNK_CASE(4)
    default:
        fprintf(stderr, "FATAL q27_attn_chunk_tp: ndev=%d is not an instantiated shard geometry.\n", ndev);
        abort();
    }
}
extern "C" int q27_attn_chunk_q8(const signed char* qkv8_t, const float* qkvs_t, int qstride,
                                 const unsigned short* q_norm_w, const unsigned short* k_norm_w,
                                 signed char* kcache, float* kscale, signed char* vcache, float* vscale,
                                 float* qout_t, unsigned short* out_t, int ostride,
                                 int pos0, int M, int ndev,
                                 signed char* Q_t, float* QS_t, float in_scale, int hpw, int attpf,
                                 float* pob, float* pml, int perm, int g64, hipStream_t s)
{
    if (M < 1 || M > Q27_ATT_CHUNK_MAX || pos0 < 0 || !pob || !pml || !qkv8_t || !qkvs_t || !qout_t) return 0;
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { Q_t = nullptr; QS_t = nullptr; }
    static const int srows = q27_env_int("Q27_Q8_ATT_SPLITROWS", 256);   // prefill: 256 queries already fill the GPU; fewer splits = fewer partials + smaller combine
    static const bool qpb2 = q27_env_int("Q27_Q8_ATT_QPB", 2) == 2;   // two queries per block (shared K/V loads + unpack)
    int nsmax, rpsmax; q27_attn_splits_host(pos0 + M, nsmax, rpsmax, srows);
    int nsmax_pair, rps_pair; q27_attn_splits_host(pos0 + M + 1, nsmax_pair, rps_pair, srows); if (qpb2 && nsmax_pair > nsmax) nsmax = nsmax_pair;
    switch (ndev) {
    Q27_TP_CHUNK_Q8_CASE(1)
    Q27_TP_CHUNK_Q8_CASE(2)
    Q27_TP_CHUNK_Q8_CASE(4)
    default:
        fprintf(stderr, "FATAL q27_attn_chunk_q8: ndev=%d is not an instantiated shard geometry.\n", ndev);
        abort();
    }
}
#undef Q27_TP_CHUNK_CASE
#undef Q27_TP_CHUNK_Q8_CASE
#undef Q27_TP_CHUNK_DECODE

// ---------------- host entry points ----------------
extern "C" void q27_attn_prep(const unsigned short* qkv_out,
                              const unsigned short* q_norm_w,
                              const unsigned short* k_norm_w,
                              signed char* kcache, float* kscale, signed char* vcache, float* vscale,
                              unsigned short* q_out, int pos, int kvstride, int hoff0, hipStream_t s)
{
    hipLaunchKernelGGL((q27_attn_prep_k<Q27_NHEAD, Q27_NKV, Q27_QROWS, Q27_KVROWS>),
                       dim3(Q27_NHEAD + Q27_NKV), dim3(64), 0, s,
                       qkv_out, q_norm_w, k_norm_w, kcache, kscale, vcache, vscale, q_out, pos, kvstride, hoff0);
}

// ---- tensor-parallel variants -------------------------------------------------
// Card g of G holds Q27_NHEAD/G query heads and Q27_NKV/G KV heads, and its q|k|v buffer and KV
// cache are that much narrower.  GQA group stays 6 at every G that divides 4 KV heads, which is
// why no collective is needed inside attention: head 6g..6g+5 meet KV head g, all on one card.
#define Q27_TP_ATTN_CASE(G) \
    case G: hipLaunchKernelGGL((q27_attn_prep_k<Q27_NHEAD/G, Q27_NKV/G, Q27_QROWS/G, Q27_KVROWS/G>), \
                dim3(Q27_NHEAD/G + Q27_NKV/G), dim3(64), 0, s, \
                qkv_out, q_norm_w, k_norm_w, kcache, kscale, vcache, vscale, q_out, pos, kvstride, hoff0); return;

extern "C" void q27_attn_prep_tp(const unsigned short* qkv_out,
                                 const unsigned short* q_norm_w,
                                 const unsigned short* k_norm_w,
                                 signed char* kcache, float* kscale, signed char* vcache, float* vscale,
                                 unsigned short* q_out, int pos, int ndev, int kvstride, int hoff0, hipStream_t s)
{
    switch (ndev) {
    Q27_TP_ATTN_CASE(1)
    Q27_TP_ATTN_CASE(2)
    Q27_TP_ATTN_CASE(4)
    default:
        fprintf(stderr, "FATAL q27_attn_prep_tp: ndev=%d is not an instantiated shard geometry "
                        "(1, 2 or 4). Refusing to run attention as a no-op.\n", ndev);
        abort();
    }
}
#undef Q27_TP_ATTN_CASE

extern "C" void q27_attn_decode(const unsigned short* q,
                                const signed char* kcache, const float* kscale,
                                const signed char* vcache, const float* vscale,
                                const unsigned short* q_proj_raw,
                                unsigned short* out, int pos, int kvstride, int hoff0, hipStream_t s)
{
    const int L = pos + 1;
    if (L <= 0) return;

    int nsplits = (L + Q27_SPLIT_ROWS - 1) / Q27_SPLIT_ROWS;
    if (nsplits > Q27_SPLITS_MAX) nsplits = Q27_SPLITS_MAX;
    if (nsplits < 1) nsplits = 1;
    int rps = (L + nsplits - 1) / nsplits;
    rps = (rps + Q27_RPW - 1) & ~(Q27_RPW - 1);           // whole wave row-pairs
    const int single = (nsplits == 1) ? 1 : 0;

    hipLaunchKernelGGL((q27_attn_decode_k<Q27_NHEAD, Q27_NKV, Q27_KVROWS>),
                       dim3(nsplits, Q27_NKV), dim3(64), 0, s,
                       q, kcache, kscale, vcache, vscale, q_proj_raw, out, L, rps, single, kvstride, hoff0);
    if (!single)
        hipLaunchKernelGGL(q27_attn_combine_k, dim3(Q27_NHEAD), dim3(Q27_HDIM), 0, s,
                           q_proj_raw, out, nsplits, nullptr, nullptr, 0.0f);
}

// ---- Q27_SPEC: NR consecutive positions of ONE sequence in one launch each -------------------
// prep: grid.y = row (every row's K/V lands in the cache before any row attends); decode: grid.z =
// row, row r sees L0 + r positions, split geometry from the longest row (shorter rows just leave
// their last split short or empty -- the combine skips sentinels). Bit-identical per row.
#define Q27_TP_ATTN_CASE_NR(G) \
    case G: hipLaunchKernelGGL((q27_attn_prep_k<Q27_NHEAD/G, Q27_NKV/G, Q27_QROWS/G, Q27_KVROWS/G>), \
                dim3(Q27_NHEAD/G + Q27_NKV/G, nr), dim3(64), 0, s, \
                qkv_out, q_norm_w, k_norm_w, kcache, kscale, vcache, vscale, q_out, pos0, kvstride, hoff0, qkvstride, qoutstride); return;
extern "C" void q27_attn_prep_tp_nr(const unsigned short* qkv_out, int qkvstride,
                                    const unsigned short* q_norm_w, const unsigned short* k_norm_w,
                                    signed char* kcache, float* kscale, signed char* vcache, float* vscale,
                                    unsigned short* q_out, int qoutstride, int pos0, int nr, int ndev,
                                    int kvstride, int hoff0, hipStream_t s)
{
    switch (ndev) {
    Q27_TP_ATTN_CASE_NR(1)
    Q27_TP_ATTN_CASE_NR(2)
    Q27_TP_ATTN_CASE_NR(4)
    default: fprintf(stderr, "FATAL q27_attn_prep_tp_nr: ndev=%d\n", ndev); abort();
    }
}
#undef Q27_TP_ATTN_CASE_NR
// K/V-only cache write for `nr` consecutive positions (no query output): the draft layer's prompt pass.
// kv_rows points at bf16 rows holding [k | v] (KVROWS each) at stride kvrstride; the kernel's q offsets are
// never dereferenced because only the NKV k/v blocks are launched (boff = NQ). ndev = 1 geometry only.
extern "C" void q27_attn_kvprep_nr(const unsigned short* kv_rows, int kvrstride, const unsigned short* k_norm_w,
                                   signed char* kcache, float* kscale, signed char* vcache, float* vscale,
                                   int pos0, int nr, hipStream_t s)
{
    const unsigned short* qkv_view = kv_rows - Q27_QROWS;   // so that qkv_view + QROWS == k rows (q blocks are not launched)
    hipLaunchKernelGGL((q27_attn_prep_k<Q27_NHEAD, Q27_NKV, Q27_QROWS, Q27_KVROWS>),
                       dim3(Q27_NKV, nr), dim3(64), 0, s,
                       qkv_view, k_norm_w, k_norm_w, kcache, kscale, vcache, vscale, (unsigned short*)nullptr,
                       pos0, Q27_KVROWS, 0, kvrstride, 0, Q27_NHEAD);
}

#define Q27_TP_DEC_CASE_NR(G) \
    case G: hipLaunchKernelGGL((q27_attn_decode_k<Q27_NHEAD/G, Q27_NKV/G, Q27_KVROWS/G>), \
                dim3(nsplits, Q27_NKV/G, nr), dim3(64), 0, s, \
                q, kcache, kscale, vcache, vscale, q_proj_raw, out, L0, rps, single, kvstride, hoff0, qstride, ostride, qrawstride); \
            if (!single) hipLaunchKernelGGL(q27_attn_combine_k, dim3(Q27_NHEAD/G, 1, nr), dim3(Q27_HDIM), \
                0, s, q_proj_raw, out, nsplits, (signed char*)nullptr, (float*)nullptr, 0.0f, ostride, qrawstride); \
            return;
extern "C" void q27_attn_decode_tp_nr(const unsigned short* q, int qstride,
                                      const signed char* kcache, const float* kscale,
                                      const signed char* vcache, const float* vscale,
                                      const unsigned short* q_proj_raw, int qrawstride,
                                      unsigned short* out, int ostride, int pos0, int nr, int ndev,
                                      int kvstride, int hoff0, hipStream_t s)
{
    const int L0 = pos0 + 1;                 // row 0's length; row r sees L0 + r
    const int Lmax = L0 + nr - 1;
    if (L0 <= 0 || nr < 1 || nr > 8) return;
    int nsplits = (Lmax + Q27_SPLIT_ROWS - 1) / Q27_SPLIT_ROWS;
    if (nsplits > Q27_SPLITS_MAX) nsplits = Q27_SPLITS_MAX;
    if (nsplits < 1) nsplits = 1;
    int rps = (Lmax + nsplits - 1) / nsplits;
    rps = (rps + Q27_RPW - 1) & ~(Q27_RPW - 1);
    const int single = (nsplits == 1) ? 1 : 0;
    switch (ndev) {
    Q27_TP_DEC_CASE_NR(1)
    Q27_TP_DEC_CASE_NR(2)
    Q27_TP_DEC_CASE_NR(4)
    default: fprintf(stderr, "FATAL q27_attn_decode_tp_nr: ndev=%d\n", ndev); abort();
    }
}
#undef Q27_TP_DEC_CASE_NR

#define Q27_TP_DEC_CASE(G) \
    case G: hipLaunchKernelGGL((q27_attn_decode_k<Q27_NHEAD/G, Q27_NKV/G, Q27_KVROWS/G>), \
                dim3(nsplits, Q27_NKV/G), dim3(64), 0, s, \
                q, kcache, kscale, vcache, vscale, q_proj_raw, out, L, rps, single, kvstride, hoff0); \
            if (!single) { hipLaunchKernelGGL(q27_attn_combine_k, dim3(Q27_NHEAD/G), dim3(Q27_HDIM), \
                0, s, q_proj_raw, out, nsplits, Q, QS, qgs); return Q ? 1 : 0; } \
            return 0;

// Returns 1 only if the fused quantize epilogue ACTUALLY RAN -- i.e. stage 2 executed AND a
// destination was supplied.
extern "C" int q27_attn_decode_tp(const unsigned short* q,
                                   const signed char* kcache, const float* kscale,
                                   const signed char* vcache, const float* vscale,
                                   const unsigned short* q_proj_raw,
                                   unsigned short* out, int pos, int ndev, int kvstride, int hoff0,
                                   signed char* Q, float* QS, float in_scale, hipStream_t s)
{
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { Q = nullptr; QS = nullptr; }
    const int L = pos + 1;
    if (L <= 0) return 0;
    int nsplits = (L + Q27_SPLIT_ROWS - 1) / Q27_SPLIT_ROWS;
    if (nsplits > Q27_SPLITS_MAX) nsplits = Q27_SPLITS_MAX;
    if (nsplits < 1) nsplits = 1;
    int rps = (L + nsplits - 1) / nsplits;
    rps = (rps + Q27_RPW - 1) & ~(Q27_RPW - 1);
    const int single = (nsplits == 1) ? 1 : 0;
    switch (ndev) {
    Q27_TP_DEC_CASE(1)
    Q27_TP_DEC_CASE(2)
    Q27_TP_DEC_CASE(4)
    default:
        fprintf(stderr, "FATAL q27_attn_decode_tp: ndev=%d is not an instantiated shard geometry "
                        "(1, 2 or 4). Refusing to run attention as a no-op.\n", ndev);
        abort();
    }
}
#undef Q27_TP_DEC_CASE
