// ============================================================================
// q27_gdn.hip - Qwen3.8-27B gated-delta (linear attention) core for AMD gfx906
// (MI50 / Radeon Pro VII, wave64, 60 CU, 64 KB LDS, ROCm 5.7.1, no MFMA).
//
// Covers 48 of the 64 layers.  Two entry points:
//   q27_gdn_conv  - causal depthwise conv1d + SiLU, in place over 10240 channels
//   q27_gdn_step  - the recurrent gated-delta update, one token
//
// Mathematics: gated delta rule, see the kernel comments below.
// Storage/ABI: include/q27.h.
//
// --------------------------------------------------------------------------
// H1 ALGEBRAIC RESTRUCTURING (contract section 8) - implemented here.
// The oracle form is two read passes over S plus a write pass.  Because
//     S_new[j][d] = decay*S_old[j][d] + k^[j]*delta[d]
// the second readout expands:
//     mixed[d] = sum_j S_new[j][d]*q^[j] = decay*sum_j S_old[j][d]*q^[j]
//                                        + delta[d]*(k^.q^)
// so ONE pass over S_old yields both quantities and the state write is a pure
// store.  Per token per layer: 3 MiB read + 3 MiB write instead of 6+3.
// Validated against the CUDA oracle over six sequential positions, rel L2
// <= 9e-5 (reference/ref_gdn_h1.py).  The traffic figure is proven only as an
// ALGEBRAIC IDENTITY here; the performance consequence is UNMEASURED.
//
// --------------------------------------------------------------------------
// TRAPS this file is written against (each silently yields plausible output):
//  T1 output gate is SWISH z*sigmoid(z), not sigmoid.  It is a compile-time
//     template parameter so the file cannot be quietly mis-served to the
//     sibling model family (Qwen4ExpText... sets output_gate_type sigmoid).
//  T2 q^ carries 1/sqrt(128) (Q27_GDN_QSCALE); k^ does not.
//  T3 both are L2 norms rsqrt(sum x^2 + 1e-6) - NO divide by 128.
//     The OUTPUT norm is a true RMS norm and DOES divide by 128.
//  T4 the output norm uses the RAW weight, no (1+W).
//  T5 SiLU is applied to the conv output for q, k AND v (all 10240 channels).
//  T6 the conv state is a SHIFT REGISTER, channel-major state[ch*4+tap],
//     tap 3 = current token, holding the RAW pre-conv value, BF16 each step.
//  T7 memory is read from the DECAYED state; the output from the UPDATED
//     state.  Both are preserved exactly by the H1 form above.
//
// --------------------------------------------------------------------------
// GEOMETRY / DEPENDENCY GRAPH (the thing being optimized, per the estate law)
//   S is [48 value heads][128 key dim][128 value dim] fp32, value-dim minor.
//   One block per value head (48 blocks).  Inside a block one LANE owns one
//   VALUE DIMENSION d, so a wave reads 64 consecutive floats of a key row =
//   256 B fully coalesced, and the whole O(K*V) inner product needs NO
//   cross-lane traffic.  Only four block-wide reductions exist per token:
//   sum q^2, sum k^2, sum q*k (fused, one barrier) and sum mixed^2.
//
//   The key dimension is split across G wave-groups of 128 lanes each so the
//   block is 128*G threads.  With G=1 a lane would hold all 128 of S[.][d] in
//   registers as the task sketches, but 48 blocks x 2 waves = 96 waves is far
//   too little parallelism to saturate HBM on 60 CUs.  G=4 holds 32 state
//   floats per lane, still one lane per value dim, still one global read pass
//   and one global write pass, and puts 8 waves on each of 48 CUs.  G is a
//   template parameter; retune it by measurement, not by argument.
//
//   COMPILE-TIME RESOURCE FACTS (ROCm 5.7.1, -Rpass-analysis=kernel-resource-usage;
//   these are compiler facts, NOT performance measurements):
//     G   block  JPT  VGPRs  scratch  waves/SIMD  VGPR ceiling for that block
//     1    128   128    194     0         1              256   ok
//     2    256    64    116     0         1              256   ok
//     4    512    32     77     0         2              128   ok  <- default
//     8   1024    16     56     0         4               64   ok, 8 to spare
//   Nothing spills at any G.  G=8 is on a cliff: a 1024-thread block must fit
//   64 VGPRs/lane or the LAUNCH FAILS at runtime, and it is using 56 of them.
//   G=4 is the default because it is not on that cliff and because G does NOT
//   change memory-level parallelism: the block issues 128*128/64 = 256
//   wave-load instructions per token whatever G is, so the outstanding-request
//   count per CU is identical.  G only trades VGPRs against reduction width.
//
//   Loads are issued as one unrolled burst of JPT independent global_load_dword
//   ahead of a single consumption loop (estate law: optimize the dependency
//   graph, several independent loads then one wait - not instruction count).
//   The loaded values stay live in VGPRs across the barrier so the store pass
//   never re-reads them.
//
//   LDS is used for exactly two things, both genuine broadcasts with 128
//   consumers per element: the normalized q^/k^ vectors and the cross-group
//   partial sums.  No activation is replicated into LDS.
// ============================================================================

#include "q27.h"

// ---------------------------------------------------------------- tunables
#ifndef Q27_GDN_G
// TESTED AND REJECTED: G=8 (1024 threads, 16 waves/block, JPT 32 -> 16) measured gdn_step
// 0.914 -> 1.024 ms and composed 19.439 -> 19.769. More waves in the same 12 blocks HURT, exactly
// as widening rmsnorm past 10 waves did. Both kernels are now known NOT to be occupancy-bound
// inside the block; the grid is the problem. gdn_step needs 12 -> 48 blocks by splitting the value
// dimension (legal: the state update is independent per value dim and the read reduction is over
// key dims for a FIXED value dim), and rmsnorm needs a multi-block reduction. Those are
// restructures, not constants.
#define Q27_GDN_G 4            // key-dim wave-groups per block; block = 128*G
#endif
#ifndef Q27_GDN_CONV_NT
#define Q27_GDN_CONV_NT 256    // threads per conv block
#endif

#define Q27_GDN_VDIM Q27_GDN_D                 // 128 value dims  (= lanes)
#define Q27_GDN_KDIM Q27_GDN_D                 // 128 key dims
#define Q27_GDN_HSTATE (Q27_GDN_D * Q27_GDN_D) // 16384 floats per value head

namespace {

// ---------------------------------------------------------------- helpers

// Full wave64 reduction.  __shfl_down with width 64 crosses the whole wave;
// ds_swizzle would only cross 32 lanes and is therefore NOT used here.
// Valid in lane 0 only; the ladder pollutes the upper lanes, which is fine
// because every caller broadcasts lane 0 through LDS.
__device__ __forceinline__ float q27_wave_sum(float x) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) x += __shfl_down(x, o, 64);
    return x;
}

__device__ __forceinline__ float q27_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));   // x <= -88: expf -> inf -> 0.  no NaN
}
__device__ __forceinline__ float q27_silu(float x) {
    return x / (1.0f + __expf(-x));      // x very negative -> -0.0, correct limit
}

// T1: the output gate.  SWISH for Qwen3.8-27B (output_gate_type: swish).
// SIGMOID exists only so the sibling family cannot be served by accident with
// the wrong constant baked in.
enum q27_gate_t { Q27_GATE_SWISH = 0, Q27_GATE_SIGMOID = 1 };

template <int GATE>
__device__ __forceinline__ float q27_out_gate(float z) {
    return (GATE == Q27_GATE_SWISH) ? q27_silu(z) : q27_sigmoid(z);
}

// ------------------------------------------------------------------ conv1d
// contract 4.1.  In place over all Q27_GDN_QKV channels.
//   shift : s0=s1 ; s1=s2 ; s2=s3 ; s3=x_t          (state holds RAW x, BF16)
//   conv  : acc = sum_tap state_new[tap]*w[ch*4+tap], fp32 accumulate
//   out   : qkv[ch] = bf16( SiLU(acc) )              (T5: q, k AND v)
// No bias tensor exists.
//
// One thread per channel.  state[ch*4+tap] and w[ch*4+tap] are four contiguous
// BF16 = 8 bytes, so a wave moves 512 contiguous bytes per access with
// global_load_dwordx2.  Requires conv_state / cw to be 8-byte aligned, which
// every hipMalloc'd tensor slice is (the layer stride is 10240*4 BF16).
__global__ __launch_bounds__(Q27_GDN_CONV_NT) void k_gdn_conv(
    unsigned short* __restrict__ qkv,
    unsigned short* __restrict__ conv_state,
    const unsigned short* __restrict__ cw)
{
    const int ch = blockIdx.x * Q27_GDN_CONV_NT + threadIdx.x;
    if (ch >= Q27_GDN_QKV) return;

    // three independent loads issued back to back, then one wait
    const uint2 st = reinterpret_cast<const uint2*>(conv_state)[ch];
    const uint2 w  = reinterpret_cast<const uint2*>(cw)[ch];
    const unsigned x = (unsigned)qkv[ch];          // already BF16: shift is exact

    // packed shift register.  st.x = t0|t1<<16, st.y = t2|t3<<16 (little endian)
    const unsigned n0 = (st.x >> 16) | (st.y << 16);   // t1, t2
    const unsigned n1 = (st.y >> 16) | (x << 16);      // t3, x_t

    float acc = q27_bf2f((unsigned short)(n0 & 0xFFFFu)) * q27_bf2f((unsigned short)(w.x & 0xFFFFu));
    acc = fmaf(q27_bf2f((unsigned short)(n0 >> 16)),      q27_bf2f((unsigned short)(w.x >> 16)),      acc);
    acc = fmaf(q27_bf2f((unsigned short)(n1 & 0xFFFFu)),  q27_bf2f((unsigned short)(w.y & 0xFFFFu)),  acc);
    acc = fmaf(q27_bf2f((unsigned short)(n1 >> 16)),      q27_bf2f((unsigned short)(w.y >> 16)),      acc);

    uint2 out; out.x = n0; out.y = n1;
    reinterpret_cast<uint2*>(conv_state)[ch] = out;    // T6: rounded BF16, raw x
    qkv[ch] = q27_f2bf(q27_silu(acc));                 // T5
}

// TENSOR-PARALLEL conv.  conv1d.weight stays REPLICATED whole (10240x4 BF16 = 80 KiB/layer), so
// the kernel maps this card's LOCAL channel onto the global weight channel.  The card's qkv shard
// is packed q | k | v with widths (2048/G, 2048/G, 6144/G) and its conv state is local, so only
// the weight lookup crosses into global numbering.
__global__ __launch_bounds__(Q27_GDN_CONV_NT) void k_gdn_conv_tp(
    unsigned short* __restrict__ qkv,
    unsigned short* __restrict__ conv_state,
    const unsigned short* __restrict__ cw,
    int nq, int nk, int nv, int qbase, int kbase, int vbase, int sfull)
{
    const int ch = blockIdx.x * Q27_GDN_CONV_NT + threadIdx.x;
    if (ch >= nq + nk + nv) return;
    const int gc = (ch < nq)      ? (qbase + ch)
                 : (ch < nq + nk) ? (kbase + ch - nq)
                                  : (vbase + ch - nq - nk);

    const int si = sfull ? gc : ch;               // sfull: the state is stored full-width (layer-split residency)
    const uint2 st = reinterpret_cast<const uint2*>(conv_state)[si];
    const uint2 w  = reinterpret_cast<const uint2*>(cw)[gc];
    const unsigned x = (unsigned)qkv[ch];

    const unsigned n0 = (st.x >> 16) | (st.y << 16);
    const unsigned n1 = (st.y >> 16) | (x << 16);

    float acc = q27_bf2f((unsigned short)(n0 & 0xFFFFu)) * q27_bf2f((unsigned short)(w.x & 0xFFFFu));
    acc = fmaf(q27_bf2f((unsigned short)(n0 >> 16)),      q27_bf2f((unsigned short)(w.x >> 16)),      acc);
    acc = fmaf(q27_bf2f((unsigned short)(n1 & 0xFFFFu)),  q27_bf2f((unsigned short)(w.y & 0xFFFFu)),  acc);
    acc = fmaf(q27_bf2f((unsigned short)(n1 >> 16)),      q27_bf2f((unsigned short)(w.y >> 16)),      acc);

    uint2 out; out.x = n0; out.y = n1;
    reinterpret_cast<uint2*>(conv_state)[si] = out;
    qkv[ch] = q27_f2bf(q27_silu(acc));
}

// -------------------------------------------------------------- gdn step
// contract 4.4 in the section 8 (H1) form, per value head h:
//   q^ = q * rsqrt(sum q^2 + 1e-6) * 1/sqrt(128)      (T2, T3)
//   k^ = k * rsqrt(sum k^2 + 1e-6)
//   beta  = sigmoid(b[h]) ; sp = softplus(a[h]+dt_bias[h])
//   decay = exp(-exp(A_log[h]) * sp)
//   A[d]  = sum_j S_old[j][d]*k^[j]   B[d] = sum_j S_old[j][d]*q^[j]  (one pass)
//   delta = (v[d] - decay*A[d]) * beta
//   mixed = decay*B[d] + delta*(k^.q^)                                    (T7)
//   S[j][d] = decay*S_old[j][d] + k^[j]*delta                     (pure store)
//   out   = mixed * rsqrt(sum mixed^2/128 + 1e-6) * norm_w[d] * gate(z[d])
//                                                                  (T4, T1)
// The body is a helper so the tensor-parallel variant is the SAME code, not a copy.  koff/voff
// are the k and v block offsets inside this card's qkv buffer (2048/4096 whole-model, 512/1024 at
// G=4) and hbase is the card's first GLOBAL value head, needed because A_log / dt_bias / in_proj_a
// / in_proj_b stay replicated at their full 48-head width.  The whole-model kernel below passes
// literals, so it inlines to exactly the code that was here before.
template <int G, int GATE>
__device__ __forceinline__ void gdn_step_body(
    const unsigned short* __restrict__ qkv_c,
    const unsigned short* __restrict__ zbuf,
    const unsigned short* __restrict__ abuf,
    const unsigned short* __restrict__ bbuf,
    const unsigned short* __restrict__ A_log,
    const unsigned short* __restrict__ dt_bias,
    const unsigned short* __restrict__ norm_w,
    float* __restrict__ S,
    unsigned short* __restrict__ out,
    int koff, int voff, int hbase,
    signed char* __restrict__ Q, float* __restrict__ QS, float qgs)
{
    constexpr int NT  = Q27_GDN_VDIM * G;    // threads per block
    constexpr int NW  = NT / 64;             // waves per block
    constexpr int JPT = Q27_GDN_KDIM / G;    // key rows (and state VGPRs) per lane

    const int tid  = threadIdx.x;
    const int d    = tid & (Q27_GDN_VDIM - 1);   // this lane's VALUE dimension
    const int g    = tid / Q27_GDN_VDIM;         // key-dim group (uniform per wave)
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int h    = blockIdx.x;                 // LOCAL value head
    const int kh   = h / 3;                      // contract 4.3: key_head = vh/3.  Exact locally
                                                 // too: with 12 value / 4 key heads per card,
                                                 // (12g+hl)/3 == 4g + hl/3.
    const int hg   = hbase + h;                  // global value head, for the replicated vectors

    __shared__ float sq[Q27_GDN_KDIM];       // q^, 128 consumers per element
    __shared__ float sk[Q27_GDN_KDIM];       // k^, 128 consumers per element
    __shared__ float pA[G][Q27_GDN_VDIM];    // cross-group partials of A
    __shared__ float pB[G][Q27_GDN_VDIM];    // cross-group partials of B
    __shared__ float r0[NW], r1[NW], r2[NW], r3[NW];

    // ---- phase 0 : q/k L2 norms and k^.q^, three reductions, one barrier ---
    // Only group 0 carries data; the others contribute 0 so the wave ladder
    // stays uniform (no divergent shuffles).
    float qraw = 0.0f, kraw = 0.0f;
    if (g == 0) {
        qraw = q27_bf2f(qkv_c[kh * Q27_GDN_D + d]);
        kraw = q27_bf2f(qkv_c[koff + kh * Q27_GDN_D + d]);
    }
    {
        const float a0 = q27_wave_sum(qraw * qraw);
        const float a1 = q27_wave_sum(kraw * kraw);
        const float a2 = q27_wave_sum(qraw * kraw);
        if (lane == 0) { r0[wv] = a0; r1[wv] = a1; r2[wv] = a2; }
    }
    __syncthreads();                                            // barrier 1

    float sumq2 = 0.0f, sumk2 = 0.0f, sumqk = 0.0f;
#pragma unroll
    for (int i = 0; i < NW; ++i) { sumq2 += r0[i]; sumk2 += r1[i]; sumqk += r2[i]; }

    // T2/T3: L2, eps inside the sqrt, no /128; 1/sqrt(128) on q ONLY.
    const float invq = rsqrtf(sumq2 + Q27_RMS_EPS) * Q27_GDN_QSCALE;
    const float invk = rsqrtf(sumk2 + Q27_RMS_EPS);
    const float kq   = sumqk * invq * invk;      // = k^ . q^ , one scalar

    if (g == 0) { sq[d] = qraw * invq; sk[d] = kraw * invk; }

    // Per-head scalars: uniform loads, recomputed per lane rather than
    // broadcast through another barrier.  These use the ACCURATE expf/log1pf,
    // not the native __expf used elementwise: decay multiplies the whole state
    // every step, so its relative error compounds over the sequence, while the
    // elementwise SiLU/gate errors are erased by the BF16 store that follows.
    // abuf/bbuf are SHARDED by value-head range, so they index LOCALLY; A_log and dt_bias are
    // 48-element vectors, still replicated, so they index globally. On the serial path hbase==0
    // makes hg==h and this is bit-identical to what it replaced.
    const float beta = 1.0f / (1.0f + expf(-q27_bf2f(bbuf[h])));
    const float t    = q27_bf2f(abuf[h]) + q27_bf2f(dt_bias[hg]);
    const float sp   = (t > 20.0f) ? t : log1pf(expf(t));
    const float decay = expf(-expf(q27_bf2f(A_log[hg])) * sp);

    const float vv = q27_bf2f(qkv_c[voff + h * Q27_GDN_D + d]);
    __syncthreads();                                            // barrier 2

    // ---- phase 1 : THE single read pass over S_old --------------------
    const int j0 = g * JPT;
    float* __restrict__ Sp = S + (size_t)h * Q27_GDN_HSTATE
                               + (size_t)j0 * Q27_GDN_VDIM + d;

    float st[JPT];
#pragma unroll
    for (int j = 0; j < JPT; ++j) st[j] = Sp[j * Q27_GDN_VDIM];   // burst, then one wait

    float accA = 0.0f, accB = 0.0f;
#pragma unroll
    for (int j = 0; j < JPT; ++j) {
        const float sv = st[j];                 // stays live for the store pass
        accA = fmaf(sv, sk[j0 + j], accA);      // LDS broadcast, index uniform per wave
        accB = fmaf(sv, sq[j0 + j], accB);
    }
    pA[g][d] = accA;
    pB[g][d] = accB;
    __syncthreads();                                            // barrier 3

    // ---- phase 2 : delta and mixed (identical in every group) -----------
    float A = 0.0f, B = 0.0f;
#pragma unroll
    for (int gg = 0; gg < G; ++gg) { A += pA[gg][d]; B += pB[gg][d]; }

    const float delta = (vv - decay * A) * beta;   // memory is the DECAYED readout
    const float mixed = fmaf(delta, kq, decay * B);// = sum_j S_new[j][d]*q^[j]

    // ---- phase 3 : the pure store pass ---------------------------------
#pragma unroll
    for (int j = 0; j < JPT; ++j)
        Sp[j * Q27_GDN_VDIM] = fmaf(sk[j0 + j], delta, decay * st[j]);

    // ---- phase 4 : output RMS norm (true RMS, /128) + gate --------------
    {
        const float m2 = (g == 0) ? mixed * mixed : 0.0f;
        const float a3 = q27_wave_sum(m2);
        if (lane == 0) r3[wv] = a3;
    }
    __syncthreads();                                            // barrier 4

    if (g == 0) {
        float summ = 0.0f;
#pragma unroll
        for (int i = 0; i < NW; ++i) summ += r3[i];
        const float inv = rsqrtf(summ * (1.0f / (float)Q27_GDN_D) + Q27_RMS_EPS);
        const float z   = q27_bf2f(zbuf[h * Q27_GDN_D + d]);
        // T4: RAW norm weight, no (1+W).   T1: gate is a template parameter.
        const float y = mixed * inv * q27_bf2f(norm_w[d]) * q27_out_gate<GATE>(z);
        out[h * Q27_GDN_D + d] = q27_f2bf(y);

        // FUSED QUANTIZE. `out` is consumed by nothing except q27_quant_fp8, which read it back
        // from HBM in a 1-block launch with 96 of 256 threads active, 48 times per token. Here the
        // value is already in a register, and a head's 128 outputs are one thread each with groups
        // of 16 landing on 16-lane boundaries, so the group max is a 16-lane shuffle. Natural order
        // (the fp8 contract has no permutation). qgs is 1/in_scale, matching q27_quant_fp8's
        // launcher, which passes the RECIPROCAL into its kernel.
        if (Q) {
            float mx = fabsf(y);
            #pragma unroll
            for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 16));
            mx = __shfl(mx, 0, 16);
            const float sc = mx * qgs * (1.0f / 127.0f);
            const float qi = (sc > 0.0f) ? (qgs / sc) : 0.0f;
            const int   k  = h * Q27_GDN_D + d;
            if ((d & 15) == 0) QS[k >> 4] = sc;
            int c = (int)rintf(y * qi);
            c = c < -127 ? -127 : (c > 127 ? 127 : c);
            Q[k] = (signed char)c;
        }
    }
}

template <int G, int GATE>
__global__ __launch_bounds__(Q27_GDN_VDIM * G) void k_gdn_step(
    const unsigned short* __restrict__ qkv_c, const unsigned short* __restrict__ zbuf,
    const unsigned short* __restrict__ abuf,  const unsigned short* __restrict__ bbuf,
    const unsigned short* __restrict__ A_log, const unsigned short* __restrict__ dt_bias,
    const unsigned short* __restrict__ norm_w, float* __restrict__ S,
    unsigned short* __restrict__ out)
{
    // Whole-model path keeps its separate quantizer; null disables the fused epilogue.
    gdn_step_body<G, GATE>(qkv_c, zbuf, abuf, bbuf, A_log, dt_bias, norm_w, S, out,
                           2048, 4096, 0, nullptr, nullptr, 0.0f);
}

template <int G, int GATE>
__global__ __launch_bounds__(Q27_GDN_VDIM * G) void k_gdn_step_tp(
    const unsigned short* __restrict__ qkv_c, const unsigned short* __restrict__ zbuf,
    const unsigned short* __restrict__ abuf,  const unsigned short* __restrict__ bbuf,
    const unsigned short* __restrict__ A_log, const unsigned short* __restrict__ dt_bias,
    const unsigned short* __restrict__ norm_w, float* __restrict__ S,
    unsigned short* __restrict__ out, int koff, int voff, int hbase,
        signed char* __restrict__ Q, float* __restrict__ QS, float qgs)
{
    gdn_step_body<G, GATE>(qkv_c, zbuf, abuf, bbuf, A_log, dt_bias, norm_w, S, out,
                           koff, voff, hbase, Q, QS, qgs);
}

// ============================================================================================
// PREFILL TILE FORMS (Q27_PF_GDN_SCAN). The sweep used the decode kernels once per position:
// 256 conv launches, 256 step launches and 256 a/b GEMVs per 256-position tile and layer, the
// state S (786 KB per head-set) round-tripping HBM every position. These three kernels do the
// tile in one launch each. Sequential semantics live only inside the scan loop; the per-position
// arithmetic (tap order, wave ladders, fmaf chains, the fused quantize) is gdn_step_body's and
// k_gdn_conv_tp's, so outputs are bit-identical.

// conv over a tile: one thread per channel walks the positions with the 3-value history in
// registers; in place over the tile slots (each position's raw input is read before it is
// overwritten, and later positions only need the RAW history, which the registers hold).
__global__ __launch_bounds__(Q27_GDN_CONV_NT) void k_gdn_conv_tp_tile(
    unsigned short* __restrict__ qkv, int qstride, int Ct,
    unsigned short* __restrict__ conv_state,
    const unsigned short* __restrict__ cw,
    int nq, int nk, int nv, int qbase, int kbase, int vbase, int sfull, unsigned short* __restrict__ snap, size_t snap_stride, int snap_all)
{
    const int ch = blockIdx.x * Q27_GDN_CONV_NT + threadIdx.x;
    if (ch >= nq + nk + nv) return;
    const int gc = (ch < nq)      ? (qbase + ch)
                 : (ch < nq + nk) ? (kbase + ch - nq)
                                  : (vbase + ch - nq - nk);
    const int si = sfull ? gc : ch;               // sfull: full-width state (layer-split residency)
    uint2 st = reinterpret_cast<const uint2*>(conv_state)[si];
    const uint2 w = reinterpret_cast<const uint2*>(cw)[gc];
    for (int t = 0; t < Ct; ++t) {
        unsigned short* p = qkv + (size_t)t * qstride + ch;
        const unsigned x = (unsigned)*p;
        const unsigned n0 = (st.x >> 16) | (st.y << 16);
        const unsigned n1 = (st.y >> 16) | (x << 16);
        float acc = q27_bf2f((unsigned short)(n0 & 0xFFFFu)) * q27_bf2f((unsigned short)(w.x & 0xFFFFu));
        acc = fmaf(q27_bf2f((unsigned short)(n0 >> 16)),      q27_bf2f((unsigned short)(w.x >> 16)),      acc);
        acc = fmaf(q27_bf2f((unsigned short)(n1 & 0xFFFFu)),  q27_bf2f((unsigned short)(w.y & 0xFFFFu)),  acc);
        acc = fmaf(q27_bf2f((unsigned short)(n1 >> 16)),      q27_bf2f((unsigned short)(w.y >> 16)),      acc);
        st.x = n0; st.y = n1;
        *p = q27_f2bf(q27_silu(acc));
        if (snap && t < Ct - 1 + snap_all) reinterpret_cast<uint2*>(snap + (size_t)t * snap_stride)[si] = st;   // Q27_SPEC bank t (snap_all: non-final chunk banks its last row too)
    }
    reinterpret_cast<uint2*>(conv_state)[si] = st;
}

extern "C" void q27_gdn_conv_tp_tile(unsigned short* qkv, int qstride, int Ct, unsigned short* conv_state,
                          const unsigned short* cw, int g, int ndev, hipStream_t s)
{
    const int nq = 2048 / ndev, nk = 2048 / ndev, nv = 6144 / ndev;
    const int n  = nq + nk + nv;
    const int nb = (n + Q27_GDN_CONV_NT - 1) / Q27_GDN_CONV_NT;
    hipLaunchKernelGGL(k_gdn_conv_tp_tile, dim3(nb), dim3(Q27_GDN_CONV_NT), 0, s,
                       qkv, qstride, Ct, conv_state, cw, nq, nk, nv, g * nq, 2048 + g * nk, 4096 + g * nv, 0, (unsigned short*)nullptr, (size_t)0, 0);
}

// Q27_SPEC: the tile conv with full-width state indexing and the per-token state snapshot.
extern "C" void q27_gdn_conv_tp_tile2(unsigned short* qkv, int qstride, int Ct, unsigned short* conv_state, int sfull,
                          const unsigned short* cw, int g, int ndev, unsigned short* snap, size_t snap_stride, int snap_all, hipStream_t s)
{
    const int nq = 2048 / ndev, nk = 2048 / ndev, nv = 6144 / ndev;
    const int n  = nq + nk + nv;
    const int nb = (n + Q27_GDN_CONV_NT - 1) / Q27_GDN_CONV_NT;
    hipLaunchKernelGGL(k_gdn_conv_tp_tile, dim3(nb), dim3(Q27_GDN_CONV_NT), 0, s,
                       qkv, qstride, Ct, conv_state, cw, nq, nk, nv, g * nq, 2048 + g * nk, 4096 + g * nv, sfull, snap, snap_stride, snap_all);
}

// the recurrence over a tile: one block per local value head, S resident in registers for the
// whole tile (one load before, one store after), gdn_step_body's phases per position.
template <int G, int GATE>
__global__ __launch_bounds__(Q27_GDN_VDIM * G) void k_gdn_scan_tp(
    const unsigned short* __restrict__ qkv_t, int qstride,
    const unsigned short* __restrict__ z_t, int zstride,
    const unsigned short* __restrict__ a_t, const unsigned short* __restrict__ b_t, int abstride,
    const unsigned short* __restrict__ A_log, const unsigned short* __restrict__ dt_bias,
    const unsigned short* __restrict__ norm_w, float* __restrict__ S,
    unsigned short* __restrict__ out_t, int ostride,
    signed char* __restrict__ Q_t, float* __restrict__ QS_t, float qgs,
    int koff, int voff, int hbase, int Ct, float* __restrict__ Ssnap, size_t snap_stride, int snap_all)
{
    constexpr int NT  = Q27_GDN_VDIM * G;
    constexpr int NW  = NT / 64;
    constexpr int JPT = Q27_GDN_KDIM / G;
    const int tid  = threadIdx.x;
    const int d    = tid & (Q27_GDN_VDIM - 1);
    const int g    = tid / Q27_GDN_VDIM;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int h    = blockIdx.x;
    const int kh   = h / 3;
    const int hg   = hbase + h;
    __shared__ float sq[Q27_GDN_KDIM];
    __shared__ float sk[Q27_GDN_KDIM];
    __shared__ float pA[G][Q27_GDN_VDIM];
    __shared__ float pB[G][Q27_GDN_VDIM];
    __shared__ float r0[NW], r1[NW], r2[NW], r3[NW];
    const int j0 = g * JPT;
    float* __restrict__ Sp = S + (size_t)h * Q27_GDN_HSTATE + (size_t)j0 * Q27_GDN_VDIM + d;
    float st[JPT];
#pragma unroll
    for (int j = 0; j < JPT; ++j) st[j] = Sp[j * Q27_GDN_VDIM];
    // per-head constants (position-independent)
    const float Aexp = expf(q27_bf2f(A_log[hg]));
    const float dtb  = q27_bf2f(dt_bias[hg]);
    const float nw   = q27_bf2f(norm_w[d]);
    for (int t = 0; t < Ct; ++t) {
        const unsigned short* __restrict__ qkv_c = qkv_t + (size_t)t * qstride;
        const unsigned short* __restrict__ zbuf  = z_t + (size_t)t * zstride;
        const unsigned short* __restrict__ abuf  = a_t + (size_t)t * abstride;
        const unsigned short* __restrict__ bbuf  = b_t + (size_t)t * abstride;
        unsigned short* __restrict__ out = out_t + (size_t)t * ostride;
        signed char* __restrict__ Q  = Q_t  ? Q_t  + (size_t)t * ostride : nullptr;
        float*       __restrict__ QS = QS_t ? QS_t + (size_t)t * (ostride >> 4) : nullptr;
        // ---- phase 0 ----
        float qraw = 0.0f, kraw = 0.0f;
        if (g == 0) {
            qraw = q27_bf2f(qkv_c[kh * Q27_GDN_D + d]);
            kraw = q27_bf2f(qkv_c[koff + kh * Q27_GDN_D + d]);
        }
        {
            const float a0 = q27_wave_sum(qraw * qraw);
            const float a1 = q27_wave_sum(kraw * kraw);
            const float a2 = q27_wave_sum(qraw * kraw);
            if (lane == 0) { r0[wv] = a0; r1[wv] = a1; r2[wv] = a2; }
        }
        __syncthreads();                                            // barrier 1
        float sumq2 = 0.0f, sumk2 = 0.0f, sumqk = 0.0f;
#pragma unroll
        for (int i = 0; i < NW; ++i) { sumq2 += r0[i]; sumk2 += r1[i]; sumqk += r2[i]; }
        const float invq = rsqrtf(sumq2 + Q27_RMS_EPS) * Q27_GDN_QSCALE;
        const float invk = rsqrtf(sumk2 + Q27_RMS_EPS);
        const float kq   = sumqk * invq * invk;
        if (g == 0) { sq[d] = qraw * invq; sk[d] = kraw * invk; }
        const float beta = 1.0f / (1.0f + expf(-q27_bf2f(bbuf[h])));
        const float tt   = q27_bf2f(abuf[h]) + dtb;
        const float sp   = (tt > 20.0f) ? tt : log1pf(expf(tt));
        const float decay = expf(-Aexp * sp);
        const float vv = q27_bf2f(qkv_c[voff + h * Q27_GDN_D + d]);
        __syncthreads();                                            // barrier 2
        // ---- phase 1 : the read pass over S_old, from registers ----
        float accA = 0.0f, accB = 0.0f;
#pragma unroll
        for (int j = 0; j < JPT; ++j) {
            const float sv = st[j];
            accA = fmaf(sv, sk[j0 + j], accA);
            accB = fmaf(sv, sq[j0 + j], accB);
        }
        pA[g][d] = accA;
        pB[g][d] = accB;
        __syncthreads();                                            // barrier 3
        // ---- phase 2 ----
        float A = 0.0f, B = 0.0f;
#pragma unroll
        for (int gg = 0; gg < G; ++gg) { A += pA[gg][d]; B += pB[gg][d]; }
        const float delta = (vv - decay * A) * beta;
        const float mixed = fmaf(delta, kq, decay * B);
        // ---- phase 3 : the store pass, into registers ----
#pragma unroll
        for (int j = 0; j < JPT; ++j)
            st[j] = fmaf(sk[j0 + j], delta, decay * st[j]);
        // ---- phase 4 ----
        {
            const float m2 = (g == 0) ? mixed * mixed : 0.0f;
            const float a3 = q27_wave_sum(m2);
            if (lane == 0) r3[wv] = a3;
        }
        __syncthreads();                                            // barrier 4
        if (g == 0) {
            float summ = 0.0f;
#pragma unroll
            for (int i = 0; i < NW; ++i) summ += r3[i];
            const float inv = rsqrtf(summ * (1.0f / (float)Q27_GDN_D) + Q27_RMS_EPS);
            const float z   = q27_bf2f(zbuf[h * Q27_GDN_D + d]);
            const float y = mixed * inv * nw * q27_out_gate<GATE>(z);
            out[h * Q27_GDN_D + d] = q27_f2bf(y);
            if (Q) {
                float mx = fabsf(y);
#pragma unroll
                for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 16));
                mx = __shfl(mx, 0, 16);
                const float sc = mx * qgs * (1.0f / 127.0f);
                const float qi = (sc > 0.0f) ? (qgs / sc) : 0.0f;
                const int   k  = h * Q27_GDN_D + d;
                if ((d & 15) == 0) QS[k >> 4] = sc;
                int c = (int)rintf(y * qi);
                c = c < -127 ? -127 : (c > 127 ? 127 : c);
                Q[k] = (signed char)c;
            }
        }
        if (Ssnap && t < Ct - 1 + snap_all) {                       // Q27_SPEC: the state after token t -> bank t (snap_all: non-final chunk)
            float* __restrict__ Sq = Ssnap + (size_t)t * snap_stride + (size_t)h * Q27_GDN_HSTATE + (size_t)j0 * Q27_GDN_VDIM + d;
#pragma unroll
            for (int j = 0; j < JPT; ++j) Sq[j * Q27_GDN_VDIM] = st[j];
        }
    }
#pragma unroll
    for (int j = 0; j < JPT; ++j) Sp[j * Q27_GDN_VDIM] = st[j];
}

extern "C" void q27_gdn_scan_tp(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                     const unsigned short* a_t, const unsigned short* b_t, int abstride,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S,
                     unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                     signed char* q_t, float* qs_t, float in_scale, hipStream_t s)
{
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { q_t = nullptr; qs_t = nullptr; }
    const int vh   = Q27_GDN_VH / ndev;
    const int koff = 2048 / ndev;
    const int voff = koff + 2048 / ndev;
    hipLaunchKernelGGL((k_gdn_scan_tp<Q27_GDN_G, Q27_GATE_SWISH>),
                       dim3(vh), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                       qkv_t, qstride, z_t, zstride, a_t, b_t, abstride, A_log, dt_bias, norm_w, S,
                       out_t, ostride, q_t, qs_t, qgs, koff, voff, g * vh, Ct, (float*)nullptr, (size_t)0, 0);
}

// Q27_SPEC: the same scan with the per-token state snapshot (bank t = the shard state after token t).
extern "C" void q27_gdn_scan_tp_snap(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                     const unsigned short* a_t, const unsigned short* b_t, int abstride,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S,
                     unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                     signed char* q_t, float* qs_t, float in_scale,
                     float* Ssnap, size_t snap_stride, int snap_all, hipStream_t s)
{
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { q_t = nullptr; qs_t = nullptr; }
    const int vh   = Q27_GDN_VH / ndev;
    const int koff = 2048 / ndev;
    const int voff = koff + 2048 / ndev;
    hipLaunchKernelGGL((k_gdn_scan_tp<Q27_GDN_G, Q27_GATE_SWISH>),
                       dim3(vh), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                       qkv_t, qstride, z_t, zstride, a_t, b_t, abstride, A_log, dt_bias, norm_w, S,
                       out_t, ostride, q_t, qs_t, qgs, koff, voff, g * vh, Ct, Ssnap, snap_stride, snap_all);
}

// ---- SCAN v2 (Q27_PF_GDN_SCAN2): the recurrence with its position-independent half moved out.
// The counter map of a real prefill put k_gdn_scan_tp at 626 us per layer-tile with 96 waves on 240 SIMDs,
// VALU busy 8.8% and ~620 cycles of LDS-issue wait per wave per position: every position paid four barriers,
// three wave-sum ladders for the q/k norms, and ~100 broadcast LDS reads of sq/sk per thread. None of the
// phase-0 quantities (q/k rms norms, kq, beta, decay) depend on S, so k_gdn_pre_tp computes them for the whole
// tile in parallel (one 128-thread block per position x key head; the two-wave sum reproduces the original
// r0[0]+r0[1]+0+... order exactly) and k_gdn_scan2_tp keeps only phases 1-4: sq/sk arrive through a
// double-buffered LDS stage as 16-byte broadcast reads (8+8 per thread instead of 96 scalar ones), the
// next position's values are staged while the current one computes, and two barriers remain. Every fmaf
// chain, every ladder and the output path are gdn_step_body's, in the same order: bit-identical.
template <int NKH>
__global__ __launch_bounds__(Q27_GDN_D) void k_gdn_pre_tp(
    const unsigned short* __restrict__ qkv_t, int qstride,
    const unsigned short* __restrict__ a_t, const unsigned short* __restrict__ b_t, int abstride,
    const unsigned short* __restrict__ A_log, const unsigned short* __restrict__ dt_bias,
    int koff, int hbase, int nvh,
    float* __restrict__ SQ, float* __restrict__ SK, float* __restrict__ SC)
{
    const int t = blockIdx.x / NKH, kh = blockIdx.x % NKH;
    const int d = threadIdx.x, lane = d & 63, wv = d >> 6;
    const unsigned short* __restrict__ qkv_c = qkv_t + (size_t)t * qstride;
    __shared__ float r0[2], r1[2], r2[2];
    const float qraw = q27_bf2f(qkv_c[kh * Q27_GDN_D + d]);
    const float kraw = q27_bf2f(qkv_c[koff + kh * Q27_GDN_D + d]);
    {
        const float a0 = q27_wave_sum(qraw * qraw);
        const float a1 = q27_wave_sum(kraw * kraw);
        const float a2 = q27_wave_sum(qraw * kraw);
        if (lane == 0) { r0[wv] = a0; r1[wv] = a1; r2[wv] = a2; }
    }
    __syncthreads();
    // the original sums r[0..NW-1] with waves 2..NW-1 contributing wave_sum(0) = 0: adding +0 is exact
    float sumq2 = 0.0f, sumk2 = 0.0f, sumqk = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) { sumq2 += r0[i]; sumk2 += r1[i]; sumqk += r2[i]; }
    const float invq = rsqrtf(sumq2 + Q27_RMS_EPS) * Q27_GDN_QSCALE;
    const float invk = rsqrtf(sumk2 + Q27_RMS_EPS);
    const float kq   = sumqk * invq * invk;
    SQ[((size_t)t * NKH + kh) * Q27_GDN_D + d] = qraw * invq;
    SK[((size_t)t * NKH + kh) * Q27_GDN_D + d] = kraw * invk;
    const int vper = nvh / NKH;                       // value heads per key head (3)
    if (d < vper) {
        const int h  = kh * vper + d;
        const int hg = hbase + h;
        const unsigned short* __restrict__ abuf = a_t + (size_t)t * abstride;
        const unsigned short* __restrict__ bbuf = b_t + (size_t)t * abstride;
        const float Aexp = expf(q27_bf2f(A_log[hg]));
        const float dtb  = q27_bf2f(dt_bias[hg]);
        const float beta = 1.0f / (1.0f + expf(-q27_bf2f(bbuf[h])));
        const float tt   = q27_bf2f(abuf[h]) + dtb;
        const float sp   = (tt > 20.0f) ? tt : log1pf(expf(tt));
        const float decay = expf(-Aexp * sp);
        float* sc = SC + ((size_t)t * nvh + h) * 4;
        sc[0] = kq; sc[1] = beta; sc[2] = decay; sc[3] = 0.0f;
    }
}
template <int G, int GATE, int NKH>
__global__ __launch_bounds__(Q27_GDN_VDIM * G) void k_gdn_scan2_tp(
    const unsigned short* __restrict__ qkv_t, int qstride,
    const unsigned short* __restrict__ z_t, int zstride,
    const float* __restrict__ SQ, const float* __restrict__ SK, const float* __restrict__ SC,
    const unsigned short* __restrict__ norm_w, float* __restrict__ S,
    unsigned short* __restrict__ out_t, int ostride,
    signed char* __restrict__ Q_t, float* __restrict__ QS_t, float qgs,
    int voff, int nvh, int Ct)
{
    constexpr int NT  = Q27_GDN_VDIM * G;
    constexpr int NW  = NT / 64;
    constexpr int JPT = Q27_GDN_KDIM / G;
    const int tid  = threadIdx.x;
    const int d    = tid & (Q27_GDN_VDIM - 1);
    const int g    = tid / Q27_GDN_VDIM;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int h    = blockIdx.x;
    const int kh   = h / (nvh / NKH);
    __shared__ __align__(16) float sqs[2][Q27_GDN_KDIM];
    __shared__ __align__(16) float sks[2][Q27_GDN_KDIM];
    __shared__ float pA[G][Q27_GDN_VDIM];
    __shared__ float pB[G][Q27_GDN_VDIM];
    __shared__ float r3[NW];
    const int j0 = g * JPT;
    float* __restrict__ Sp = S + (size_t)h * Q27_GDN_HSTATE + (size_t)j0 * Q27_GDN_VDIM + d;
    float st[JPT];
#pragma unroll
    for (int j = 0; j < JPT; ++j) st[j] = Sp[j * Q27_GDN_VDIM];
    const float nw = q27_bf2f(norm_w[d]);
    // stage position 0
    if (NT >= 2 * Q27_GDN_KDIM) {
        if (tid < Q27_GDN_KDIM) sqs[0][tid] = SQ[((size_t)0 * NKH + kh) * Q27_GDN_KDIM + tid];
        else if (tid < 2 * Q27_GDN_KDIM) sks[0][tid - Q27_GDN_KDIM] = SK[((size_t)0 * NKH + kh) * Q27_GDN_KDIM + (tid - Q27_GDN_KDIM)];
    } else { sqs[0][tid] = SQ[((size_t)0 * NKH + kh) * Q27_GDN_KDIM + tid]; sks[0][tid] = SK[((size_t)0 * NKH + kh) * Q27_GDN_KDIM + tid]; }   // G == 1: 128 threads load both
    __syncthreads();
    for (int t = 0; t < Ct; ++t) {
        const int cur = t & 1, nxt = cur ^ 1;
        const unsigned short* __restrict__ qkv_c = qkv_t + (size_t)t * qstride;
        const unsigned short* __restrict__ zbuf  = z_t + (size_t)t * zstride;
        unsigned short* __restrict__ out = out_t + (size_t)t * ostride;
        signed char* __restrict__ Q  = Q_t  ? Q_t  + (size_t)t * ostride : nullptr;
        float*       __restrict__ QS = QS_t ? QS_t + (size_t)t * (ostride >> 4) : nullptr;
        // stage position t+1 while t computes (consumed after barrier B)
        if (t + 1 < Ct) {
            if (NT >= 2 * Q27_GDN_KDIM) {
                if (tid < Q27_GDN_KDIM) sqs[nxt][tid] = SQ[((size_t)(t + 1) * NKH + kh) * Q27_GDN_KDIM + tid];
                else if (tid < 2 * Q27_GDN_KDIM) sks[nxt][tid - Q27_GDN_KDIM] = SK[((size_t)(t + 1) * NKH + kh) * Q27_GDN_KDIM + (tid - Q27_GDN_KDIM)];
            } else { sqs[nxt][tid] = SQ[((size_t)(t + 1) * NKH + kh) * Q27_GDN_KDIM + tid]; sks[nxt][tid] = SK[((size_t)(t + 1) * NKH + kh) * Q27_GDN_KDIM + tid]; }
        }
        const float* __restrict__ sc = SC + ((size_t)t * nvh + h) * 4;
        const float kq = sc[0], beta = sc[1], decay = sc[2];
        const float vv = q27_bf2f(qkv_c[voff + h * Q27_GDN_D + d]);
        // ---- phase 1 : the read pass over S_old, from registers (gdn_step_body's loop, sq/sk from the staged buffer) ----
        float accA = 0.0f, accB = 0.0f;
#pragma unroll
        for (int j = 0; j < JPT; ++j) {
            const float sv = st[j];
            accA = fmaf(sv, sks[cur][j0 + j], accA);
            accB = fmaf(sv, sqs[cur][j0 + j], accB);
        }
        pA[g][d] = accA;
        pB[g][d] = accB;
        __syncthreads();                                            // barrier A
        // ---- phase 2 ----
        float A = 0.0f, B = 0.0f;
#pragma unroll
        for (int gg = 0; gg < G; ++gg) { A += pA[gg][d]; B += pB[gg][d]; }
        const float delta = (vv - decay * A) * beta;
        const float mixed = fmaf(delta, kq, decay * B);
        // ---- phase 3 : the store pass, into registers ----
#pragma unroll
        for (int j = 0; j < JPT; ++j)
            st[j] = fmaf(sks[cur][j0 + j], delta, decay * st[j]);
        // ---- phase 4 ----
        {
            const float m2 = (g == 0) ? mixed * mixed : 0.0f;
            const float a3 = q27_wave_sum(m2);
            if (lane == 0) r3[wv] = a3;
        }
        __syncthreads();                                            // barrier B
        if (g == 0) {
            float summ = 0.0f;
#pragma unroll
            for (int i = 0; i < NW; ++i) summ += r3[i];
            const float inv = rsqrtf(summ * (1.0f / (float)Q27_GDN_D) + Q27_RMS_EPS);
            const float z   = q27_bf2f(zbuf[h * Q27_GDN_D + d]);
            const float y = mixed * inv * nw * q27_out_gate<GATE>(z);
            out[h * Q27_GDN_D + d] = q27_f2bf(y);
            if (Q) {
                float mx = fabsf(y);
#pragma unroll
                for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 16));
                mx = __shfl(mx, 0, 16);
                const float sc2 = mx * qgs * (1.0f / 127.0f);
                const float qi = (sc2 > 0.0f) ? (qgs / sc2) : 0.0f;
                const int   k  = h * Q27_GDN_D + d;
                if ((d & 15) == 0) QS[k >> 4] = sc2;
                int c = (int)rintf(y * qi);
                c = c < -127 ? -127 : (c > 127 ? 127 : c);
                Q[k] = (signed char)c;
            }
        }
    }
    // the write-back addresses are formed here, after the loop: an opaque copy of the base keeps the compiler
    // from precomputing sixteen 64-bit store addresses before the loop and holding them live through it (29
    // folded spills around the loop in the first build; none inside it)
    float* Sp2 = Sp;
    asm volatile("" : "+v"(Sp2));
#pragma unroll
    for (int j = 0; j < JPT; ++j) Sp2[j * Q27_GDN_VDIM] = st[j];
}
static float* s_scan2_pre[32] = {0};   // [device*2 + parity]: two chunks in flight per card (Q27_LS_Q8)
static size_t s_scan2_cap[32] = {0};
extern "C" void q27_gdn_scan2_tp(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                      const unsigned short* a_t, const unsigned short* b_t, int abstride,
                      const unsigned short* A_log, const unsigned short* dt_bias,
                      const unsigned short* norm_w, float* S,
                      unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                      signed char* q_t, float* qs_t, float in_scale, hipStream_t s)
{
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { q_t = nullptr; qs_t = nullptr; }
    const int vh   = Q27_GDN_VH / ndev;                 // 12 value heads per card
    const int koff = 2048 / ndev;
    const int voff = koff + 2048 / ndev;
    const int nkh  = koff / Q27_GDN_D;                  // 4 key heads per card
    const size_t need = ((size_t)Ct * nkh * Q27_GDN_D * 2 + (size_t)Ct * vh * 4) * sizeof(float);
    if (g < 0 || g >= 16) return;
    int slot = g;
    hipGetDevice(&slot);                                // scratch is per DEVICE, not per rank:
    if (slot < 0 || slot >= 16) slot = 0;               // the single-card (nd=1) path passes g=0
    slot = slot * 2;                                    // parity 0
    if (s_scan2_cap[slot] < need) {
        if (s_scan2_pre[slot]) hipFree(s_scan2_pre[slot]);
        if (hipMalloc((void**)&s_scan2_pre[slot], need) != hipSuccess) { s_scan2_pre[slot] = nullptr; s_scan2_cap[slot] = 0; return; }
        s_scan2_cap[slot] = need;
    }
    float* SQ = s_scan2_pre[slot];
    float* SK = SQ + (size_t)Ct * nkh * Q27_GDN_D;
    float* SC = SK + (size_t)Ct * nkh * Q27_GDN_D;
    if (getenv("Q27_LS_SDBG"))
        std::fprintf(stderr, "SCAN2 slot=%d SQ=%p need=%zu Ct=%d vh=%d koff=%d voff=%d S=%p out=%p q=%p\n",
                     slot, (const void*)SQ, need, Ct, vh, koff, voff, (const void*)S,
                     (const void*)out_t, (const void*)q_t);
    if (nkh == 4) {                                     // the 4-card layout this engine ships
        hipLaunchKernelGGL((k_gdn_pre_tp<4>), dim3(Ct * 4), dim3(Q27_GDN_D), 0, s,
                           qkv_t, qstride, a_t, b_t, abstride, A_log, dt_bias, koff, g * vh, vh, SQ, SK, SC);
        hipLaunchKernelGGL((k_gdn_scan2_tp<Q27_GDN_G, Q27_GATE_SWISH, 4>), dim3(vh), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                           qkv_t, qstride, z_t, zstride, SQ, SK, SC, norm_w, S, out_t, ostride, q_t, qs_t, qgs, voff, vh, Ct);
    } else if (nkh == 16) {                             // the single-card (layer-split) layout
        hipLaunchKernelGGL((k_gdn_pre_tp<16>), dim3(Ct * 16), dim3(Q27_GDN_D), 0, s,
                           qkv_t, qstride, a_t, b_t, abstride, A_log, dt_bias, koff, g * vh, vh, SQ, SK, SC);
        hipLaunchKernelGGL((k_gdn_scan2_tp<Q27_GDN_G, Q27_GATE_SWISH, 16>), dim3(vh), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                           qkv_t, qstride, z_t, zstride, SQ, SK, SC, norm_w, S, out_t, ostride, q_t, qs_t, qgs, voff, vh, Ct);
    } else return;
}

// ============================================================================================
// Q27_LS_Q8 -- the quantized GDN chain. in_qkv / in_z arrive as int8 + fp32 scales per 16
// channels ([t][ch], [t][ch/16]); the conv re-quantizes its SiLU output in place with the same
// grouping (16 consecutive channels = 16 consecutive lanes); the pre-kernel and the scan
// dequantize in registers. a/b arrive as fp32 [t][48] (the int8 GEMV's accumulation
// representation). The recurrent state S stays fp32; the 4-tap conv history stays bf16 at the
// chunk boundary only (the decode conv's ABI), fp32 inside the chunk. Arithmetic per position is
// gdn_step_body's.
// ============================================================================================
__global__ __launch_bounds__(Q27_GDN_CONV_NT) void k_gdn_conv_q8_tile(
    signed char* __restrict__ q8, float* __restrict__ qs, int qstride, int Ct,
    unsigned short* __restrict__ conv_state,
    const unsigned short* __restrict__ cw,
    int nq, int nk, int nv, int qbase, int kbase, int vbase)
{
    const int nch = nq + nk + nv;
    const int ch  = blockIdx.x * Q27_GDN_CONV_NT + threadIdx.x;
    const bool ok = ch < nch;
    const int chc = ok ? ch : (nch - 1);
    const int gc = (chc < nq)      ? (qbase + chc)
                 : (chc < nq + nk) ? (kbase + chc - nq)
                                   : (vbase + chc - nq - nk);
    const uint2 st = reinterpret_cast<const uint2*>(conv_state)[chc];
    const uint2 w  = reinterpret_cast<const uint2*>(cw)[gc];
    const float w0 = q27_bf2f((unsigned short)(w.x & 0xFFFFu)), w1 = q27_bf2f((unsigned short)(w.x >> 16));
    const float w2 = q27_bf2f((unsigned short)(w.y & 0xFFFFu)), w3 = q27_bf2f((unsigned short)(w.y >> 16));
    float h1 = q27_bf2f((unsigned short)(st.x >> 16));
    float h2 = q27_bf2f((unsigned short)(st.y & 0xFFFFu));
    float h3 = q27_bf2f((unsigned short)(st.y >> 16));
    const int sstride = qstride >> 4;
    const int grp = chc >> 4;
    for (int t = 0; t < Ct; ++t) {
        signed char* p = q8 + (size_t)t * qstride + chc;
        float* ps = qs + (size_t)t * sstride + grp;
        const float x = (float)(*p) * (*ps);
        float acc = h1 * w0;
        acc = fmaf(h2, w1, acc);
        acc = fmaf(h3, w2, acc);
        acc = fmaf(x,  w3, acc);
        h1 = h2; h2 = h3; h3 = x;
        const float y = q27_silu(acc);
        float mx = fabsf(y);
        mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
        const float sc = mx * (1.0f / 127.0f);
        const float qi = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        int c = (int)rintf(y * qi);
        c = c < -127 ? -127 : (c > 127 ? 127 : c);
        if (ok) {
            *p = (signed char)c;
            if ((chc & 15) == 0) *ps = sc;
        }
    }
    if (ok) {
        uint2 o;
        o.x = ((unsigned)q27_f2bf(h1) << 16) | (unsigned)q27_f2bf(h1);
        o.y = ((unsigned)q27_f2bf(h3) << 16) | (unsigned)q27_f2bf(h2);
        reinterpret_cast<uint2*>(conv_state)[chc] = o;
    }
}

extern "C" void q27_gdn_conv_q8_tile(signed char* q8, float* qs, int qstride, int Ct, unsigned short* conv_state,
                                     const unsigned short* cw, int g, int ndev, hipStream_t s)
{
    const int nq = 2048 / ndev, nk = 2048 / ndev, nv = 6144 / ndev;
    const int n  = nq + nk + nv;
    const int nb = (n + Q27_GDN_CONV_NT - 1) / Q27_GDN_CONV_NT;
    hipLaunchKernelGGL(k_gdn_conv_q8_tile, dim3(nb), dim3(Q27_GDN_CONV_NT), 0, s,
                       q8, qs, qstride, Ct, conv_state, cw, nq, nk, nv, g * nq, 2048 + g * nk, 4096 + g * nv);
}

template <int NKH>
__global__ __launch_bounds__(Q27_GDN_D) void k_gdn_pre_q8(
    const signed char* __restrict__ q8, const float* __restrict__ qs, int qstride,
    const float* __restrict__ a_t, const float* __restrict__ b_t, int abstride,
    const unsigned short* __restrict__ A_log, const unsigned short* __restrict__ dt_bias,
    int koff, int hbase, int nvh,
    float* __restrict__ SQ, float* __restrict__ SK, float* __restrict__ SC)
{
    const int t = blockIdx.x / NKH, kh = blockIdx.x % NKH;
    const int d = threadIdx.x, lane = d & 63, wv = d >> 6;
    const signed char* __restrict__ q8_c = q8 + (size_t)t * qstride;
    const float* __restrict__ qs_c = qs + (size_t)t * (qstride >> 4);
    __shared__ float r0[2], r1[2], r2[2];
    const int iq = kh * Q27_GDN_D + d, ik = koff + kh * Q27_GDN_D + d;
    const float qraw = (float)q8_c[iq] * qs_c[iq >> 4];
    const float kraw = (float)q8_c[ik] * qs_c[ik >> 4];
    {
        const float a0 = q27_wave_sum(qraw * qraw);
        const float a1 = q27_wave_sum(kraw * kraw);
        const float a2 = q27_wave_sum(qraw * kraw);
        if (lane == 0) { r0[wv] = a0; r1[wv] = a1; r2[wv] = a2; }
    }
    __syncthreads();
    float sumq2 = 0.0f, sumk2 = 0.0f, sumqk = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) { sumq2 += r0[i]; sumk2 += r1[i]; sumqk += r2[i]; }
    const float invq = rsqrtf(sumq2 + Q27_RMS_EPS) * Q27_GDN_QSCALE;
    const float invk = rsqrtf(sumk2 + Q27_RMS_EPS);
    const float kq   = sumqk * invq * invk;
    SQ[((size_t)t * NKH + kh) * Q27_GDN_D + d] = qraw * invq;
    SK[((size_t)t * NKH + kh) * Q27_GDN_D + d] = kraw * invk;
    const int vper = nvh / NKH;
    if (d < vper) {
        const int h  = kh * vper + d;
        const int hg = hbase + h;
        const float* __restrict__ abuf = a_t + (size_t)t * abstride;
        const float* __restrict__ bbuf = b_t + (size_t)t * abstride;
        const float Aexp = expf(q27_bf2f(A_log[hg]));
        const float dtb  = q27_bf2f(dt_bias[hg]);
        const float beta = 1.0f / (1.0f + expf(-bbuf[h]));
        const float tt   = abuf[h] + dtb;
        const float sp   = (tt > 20.0f) ? tt : log1pf(expf(tt));
        const float decay = expf(-Aexp * sp);
        float* sc = SC + ((size_t)t * nvh + h) * 4;
        sc[0] = kq; sc[1] = beta; sc[2] = decay; sc[3] = 0.0f;
    }
}

template <int G, int GATE, int NKH>
__global__ __launch_bounds__(Q27_GDN_VDIM * G) void k_gdn_scan2_q8(
    const signed char* __restrict__ q8, const float* __restrict__ qs, int qstride,
    const signed char* __restrict__ z8, const float* __restrict__ zs, int zstride,
    const float* __restrict__ SQ, const float* __restrict__ SK, const float* __restrict__ SC,
    const unsigned short* __restrict__ norm_w, float* __restrict__ S,
    unsigned short* __restrict__ out_t, int ostride,
    signed char* __restrict__ Q_t, float* __restrict__ QS_t, float qgs,
    int voff, int nvh, int Ct, int perm, int g64)
{
    constexpr int NT  = Q27_GDN_VDIM * G;
    constexpr int NW  = NT / 64;
    constexpr int JPT = Q27_GDN_KDIM / G;
    const int tid  = threadIdx.x;
    const int d    = tid & (Q27_GDN_VDIM - 1);
    const int g    = tid / Q27_GDN_VDIM;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int h    = blockIdx.x;
    const int kh   = h / (nvh / NKH);
    __shared__ __align__(16) float sqs[2][Q27_GDN_KDIM];
    __shared__ __align__(16) float sks[2][Q27_GDN_KDIM];
    __shared__ float pA[G][Q27_GDN_VDIM];
    __shared__ float pB[G][Q27_GDN_VDIM];
    __shared__ float r3[NW];
    const int j0 = g * JPT;
    float* __restrict__ Sp = S + (size_t)h * Q27_GDN_HSTATE + (size_t)j0 * Q27_GDN_VDIM + d;
    float st[JPT];
#pragma unroll
    for (int j = 0; j < JPT; ++j) st[j] = Sp[j * Q27_GDN_VDIM];
    const float nw = q27_bf2f(norm_w[d]);
    const int iv = voff + h * Q27_GDN_D + d;
    const int iz = h * Q27_GDN_D + d;
    const int qss = qstride >> 4, zss = zstride >> 4;
    if (tid < Q27_GDN_KDIM) sqs[0][tid] = SQ[((size_t)0 * NKH + kh) * Q27_GDN_KDIM + tid];
    else if (tid < 2 * Q27_GDN_KDIM) sks[0][tid - Q27_GDN_KDIM] = SK[((size_t)0 * NKH + kh) * Q27_GDN_KDIM + (tid - Q27_GDN_KDIM)];
    __syncthreads();
    // software prefetch: the per-position scalars (kq, beta, decay), v and z of position t+1 are loaded one
    // iteration ahead (the loop body is a dependent chain of two barriers; global-load latency was on it)
    float pf_kq, pf_beta, pf_decay, pf_vv, pf_z;
    {
        const float* __restrict__ sc0 = SC + ((size_t)0 * nvh + h) * 4;
        pf_kq = sc0[0]; pf_beta = sc0[1]; pf_decay = sc0[2];
        pf_vv = (float)q8[(size_t)0 * qstride + iv] * qs[(size_t)0 * qss + (iv >> 4)];
        pf_z  = (float)z8[(size_t)0 * zstride + iz] * zs[(size_t)0 * zss + (iz >> 4)];
    }
    for (int t = 0; t < Ct; ++t) {
        const int cur = t & 1, nxt = cur ^ 1;
        unsigned short* __restrict__ out = out_t ? out_t + (size_t)t * ostride : nullptr;
        signed char* __restrict__ Q  = Q_t  ? Q_t  + (size_t)t * ostride : nullptr;
        float*       __restrict__ QS = QS_t ? QS_t + (size_t)t * (g64 ? (ostride >> 6) : (ostride >> 4)) : nullptr;
        const float kq = pf_kq, beta = pf_beta, decay = pf_decay, vv = pf_vv, zcur = pf_z;
        if (t + 1 < Ct) {
            if (tid < Q27_GDN_KDIM) sqs[nxt][tid] = SQ[((size_t)(t + 1) * NKH + kh) * Q27_GDN_KDIM + tid];
            else if (tid < 2 * Q27_GDN_KDIM) sks[nxt][tid - Q27_GDN_KDIM] = SK[((size_t)(t + 1) * NKH + kh) * Q27_GDN_KDIM + (tid - Q27_GDN_KDIM)];
            const float* __restrict__ scn = SC + ((size_t)(t + 1) * nvh + h) * 4;
            pf_kq = scn[0]; pf_beta = scn[1]; pf_decay = scn[2];
            pf_vv = (float)q8[(size_t)(t + 1) * qstride + iv] * qs[(size_t)(t + 1) * qss + (iv >> 4)];
            pf_z  = (float)z8[(size_t)(t + 1) * zstride + iz] * zs[(size_t)(t + 1) * zss + (iz >> 4)];
        }
        float accA = 0.0f, accB = 0.0f;
#pragma unroll
        for (int j = 0; j < JPT; ++j) {
            const float sv = st[j];
            accA = fmaf(sv, sks[cur][j0 + j], accA);
            accB = fmaf(sv, sqs[cur][j0 + j], accB);
        }
        float A, B;
        if constexpr (G > 1) {
            pA[g][d] = accA;
            pB[g][d] = accB;
            __syncthreads();                                            // barrier A
            A = 0.0f; B = 0.0f;
#pragma unroll
            for (int gg = 0; gg < G; ++gg) { A += pA[gg][d]; B += pB[gg][d]; }
        } else { A = accA; B = accB; }                                    // G == 1: the thread owns the whole key dim
        const float delta = (vv - decay * A) * beta;
        const float mixed = fmaf(delta, kq, decay * B);
#pragma unroll
        for (int j = 0; j < JPT; ++j)
            st[j] = fmaf(sks[cur][j0 + j], delta, decay * st[j]);
        {
            const float m2 = (g == 0) ? mixed * mixed : 0.0f;
            const float a3 = q27_wave_sum(m2);
            if (lane == 0) r3[wv] = a3;
        }
        __syncthreads();                                            // barrier B
        if (g == 0) {
            float summ = 0.0f;
#pragma unroll
            for (int i = 0; i < NW; ++i) summ += r3[i];
            const float inv = rsqrtf(summ * (1.0f / (float)Q27_GDN_D) + Q27_RMS_EPS);
            const float y = mixed * inv * nw * q27_out_gate<GATE>(zcur);
            if (out) out[iz] = q27_f2bf(y);
            if (Q) {
                float mx = fabsf(y);
                if (g64) { for (int o = 32; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 64)); mx = __shfl(mx, 0, 64); }
                else     { for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 16)); mx = __shfl(mx, 0, 16); }
                const float sc2 = mx * qgs * (1.0f / 127.0f);
                const float qi = (sc2 > 0.0f) ? (qgs / sc2) : 0.0f;
                if (g64) { if ((d & 63) == 0) QS[iz >> 6] = sc2; } else if ((d & 15) == 0) QS[iz >> 4] = sc2;
                // the NVFP4 out_proj consumes the even-then-odd order inside each group of 16 (q27.h contract)
                const int izq = perm ? ((iz & ~15) | ((d & 1) ? (8 + ((d & 15) >> 1)) : ((d & 15) >> 1))) : iz;
                int c = (int)rintf(y * qi);
                c = c < -127 ? -127 : (c > 127 ? 127 : c);
                Q[izq] = (signed char)c;
            }
        }
    }
    float* Sp2 = Sp;
    asm volatile("" : "+v"(Sp2));
#pragma unroll
    for (int j = 0; j < JPT; ++j) Sp2[j * Q27_GDN_VDIM] = st[j];
}

extern "C" void q27_gdn_scan2_q8(const signed char* q8, const float* qs, int qstride,
                                 const signed char* z8, const float* zs, int zstride,
                                 const float* a_t, const float* b_t, int abstride,
                                 const unsigned short* A_log, const unsigned short* dt_bias,
                                 const unsigned short* norm_w, float* S,
                                 unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                                 signed char* q_t, float* qs_t, float in_scale, int perm, int g64, int par, hipStream_t s)
{
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { q_t = nullptr; qs_t = nullptr; }
    const int vh   = Q27_GDN_VH / ndev;
    const int koff = 2048 / ndev;
    const int voff = koff + 2048 / ndev;
    const int nkh  = koff / Q27_GDN_D;
    const size_t need = ((size_t)Ct * nkh * Q27_GDN_D * 2 + (size_t)Ct * vh * 4) * sizeof(float);
    if (g < 0 || g >= 16) return;
    int slot = g;
    hipGetDevice(&slot);
    if (slot < 0 || slot >= 16) slot = 0;
    slot = slot * 2 + (par & 1);                         // per (device, chunk parity) scratch
    if (s_scan2_cap[slot] < need) {
        if (s_scan2_pre[slot]) hipFree(s_scan2_pre[slot]);
        if (hipMalloc((void**)&s_scan2_pre[slot], need) != hipSuccess) { s_scan2_pre[slot] = nullptr; s_scan2_cap[slot] = 0; return; }
        s_scan2_cap[slot] = need;
    }
    float* SQ = s_scan2_pre[slot];
    float* SK = SQ + (size_t)Ct * nkh * Q27_GDN_D;
    float* SC = SK + (size_t)Ct * nkh * Q27_GDN_D;
    if (nkh == 4) {
        hipLaunchKernelGGL((k_gdn_pre_q8<4>), dim3(Ct * 4), dim3(Q27_GDN_D), 0, s,
                           q8, qs, qstride, a_t, b_t, abstride, A_log, dt_bias, koff, g * vh, vh, SQ, SK, SC);
        hipLaunchKernelGGL((k_gdn_scan2_q8<Q27_GDN_G, Q27_GATE_SWISH, 4>), dim3(vh), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                           q8, qs, qstride, z8, zs, zstride, SQ, SK, SC, norm_w, S, out_t, ostride, q_t, qs_t, qgs, voff, vh, Ct, perm, g64);
    } else if (nkh == 16) {
        hipLaunchKernelGGL((k_gdn_pre_q8<16>), dim3(Ct * 16), dim3(Q27_GDN_D), 0, s,
                           q8, qs, qstride, a_t, b_t, abstride, A_log, dt_bias, koff, g * vh, vh, SQ, SK, SC);
        static const int scang = q27_env_int("Q27_Q8_SCANG", Q27_GDN_G);   // threads per head = 128 x G; G=2 halves the barriers per position
        if (scang == 2)
            hipLaunchKernelGGL((k_gdn_scan2_q8<2, Q27_GATE_SWISH, 16>), dim3(vh), dim3(Q27_GDN_VDIM * 2), 0, s,
                               q8, qs, qstride, z8, zs, zstride, SQ, SK, SC, norm_w, S, out_t, ostride, q_t, qs_t, qgs, voff, vh, Ct, perm, g64);
        else if (scang == 1)
            hipLaunchKernelGGL((k_gdn_scan2_q8<1, Q27_GATE_SWISH, 16>), dim3(vh), dim3(Q27_GDN_VDIM * 1), 0, s,
                               q8, qs, qstride, z8, zs, zstride, SQ, SK, SC, norm_w, S, out_t, ostride, q_t, qs_t, qgs, voff, vh, Ct, perm, g64);
        else
            hipLaunchKernelGGL((k_gdn_scan2_q8<Q27_GDN_G, Q27_GATE_SWISH, 16>), dim3(vh), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                               q8, qs, qstride, z8, zs, zstride, SQ, SK, SC, norm_w, S, out_t, ostride, q_t, qs_t, qgs, voff, vh, Ct, perm, g64);
    } else return;
}

// VALUE-DIM SPLIT (Q27_GDN_DSPLIT). One block per value head (12 under TP) leaves 48 CUs idle
// and makes the kernel latency-bound; G=8 was measured and REJECTED (more waves in the same 12
// blocks hurt). This splits each head's 128 value dims across 4 blocks: 48 blocks of 128 threads
// (32 value dims x G=4). The state update and the A/B read reduction are per value dim, so they
// need NO cross-block traffic; only the two SCALAR norms (q/k/qk and the output mixed^2) cross
// blocks, through one global scratch per card with a generation flag that is never reset (unique
// per launch: gen = pos*LAYERS + L). All 48 blocks are resident on 60 CUs, so the spin progresses.
//   scratch layout per head, 12 heads: qk_part[4] mix_part[4] | invq invk kq invmix
//     | cnt_qk cnt_mix flag_qk flag_mix   (16 words per head)
template <int G, int GATE>
__global__ __launch_bounds__(Q27_GDN_VDIM / 4 * G) void k_gdn_step_tp_ds(
    const unsigned short* __restrict__ qkv_c, const unsigned short* __restrict__ zbuf,
    const unsigned short* __restrict__ abuf,  const unsigned short* __restrict__ bbuf,
    const unsigned short* __restrict__ A_log, const unsigned short* __restrict__ dt_bias,
    const unsigned short* __restrict__ norm_w, float* __restrict__ S,
    unsigned short* __restrict__ out, int koff, int voff, int hbase,
    unsigned* __restrict__ scr, int gen,
    signed char* __restrict__ Q, float* __restrict__ QS, float qgs)
{
    constexpr int DBLK = Q27_GDN_VDIM / 4;    // 32 value dims per block
    constexpr int NT   = DBLK * G;            // 128 threads
    constexpr int NW   = NT / 64;             // 2 waves
    constexpr int JPT  = Q27_GDN_KDIM / G;    // 32 key rows per lane

    const int tid  = threadIdx.x;
    const int d    = tid & (DBLK - 1);
    const int g    = tid / DBLK;              // 0..3
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    const int h    = blockIdx.x;              // LOCAL value head
    const int db   = blockIdx.y;              // value-dim block
    const int kh   = h / 3;
    const int hg   = hbase + h;
    const int dg   = db * DBLK + d;           // GLOBAL value dim

    __shared__ float sq[Q27_GDN_KDIM];
    __shared__ float sk[Q27_GDN_KDIM];
    __shared__ float pA[G][DBLK];
    __shared__ float pB[G][DBLK];
    __shared__ float r0[NW], r1[NW], r2[NW], r3[NW];

    // DETERMINISTIC cross-block reduction: each block stores its partial to a FIXED slot
    // (plain stores, ordered by the counter's release/acquire); the last arriver sums the
    // slots in fixed order. Arrival-order atomic adds measured NONdeterministic tokens.
    float*    const pqs   = (float*)(scr + (size_t)h * 16);           // [4] qk partials
    float*    const pmx   = (float*)(scr + (size_t)h * 16 + 4);       // [4] mix partials
    float*    const fin   = (float*)(scr + (size_t)h * 16 + 8);       // [4] invq invk kq invmix
    unsigned* const cntqk = scr + (size_t)h * 16 + 12;
    unsigned* const cntmx = scr + (size_t)h * 16 + 13;
    unsigned* const flgqk = scr + (size_t)h * 16 + 14;
    unsigned* const flgmx = scr + (size_t)h * 16 + 15;

    // ---- phase 0a: this block's q/k elements (4 per lane) + the cross-block qk reduction ----
    float qraw[4] = {0,0,0,0}, kraw[4] = {0,0,0,0};
    float sq2 = 0.0f, sk2 = 0.0f, sqk = 0.0f;
    if (g == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = d + i * DBLK;     // KEY dim: every block reads all 128 via 32 lanes x 4
            qraw[i] = q27_bf2f(qkv_c[kh * Q27_GDN_D + j]);
            kraw[i] = q27_bf2f(qkv_c[koff + kh * Q27_GDN_D + j]);
            sq2 = fmaf(qraw[i], qraw[i], sq2);
            sk2 = fmaf(kraw[i], kraw[i], sk2);
            sqk = fmaf(qraw[i], kraw[i], sqk);
        }
    }
    {
        const float a0 = q27_wave_sum(sq2), a1 = q27_wave_sum(sk2), a2 = q27_wave_sum(sqk);
        if (lane == 0) { r0[wv] = a0; r1[wv] = a1; r2[wv] = a2; }
    }
    __syncthreads();                                          // barrier 0a
    float psq2 = 0.0f, psk2 = 0.0f, psqk = 0.0f;
#pragma unroll
    for (int i = 0; i < NW; ++i) { psq2 += r0[i]; psk2 += r1[i]; psqk += r2[i]; }
    if (tid == 0) {                     // ONE thread per block: lane==0 is true for BOTH waves
        pqs[db * 3 + 0] = psq2;              // FIXED SLOT per block; the last arriver sums
        pqs[db * 3 + 1] = psk2;              // the slots in fixed order -> deterministic
        pqs[db * 3 + 2] = psqk;
        __threadfence();                     // order the slot stores before the arrival count
        const unsigned old = __atomic_fetch_add(cntqk, 1u, __ATOMIC_ACQ_REL);
        if (old == 3u) {                       // last of the 4 blocks
            const float s0 = pqs[0] + pqs[3] + pqs[6] + pqs[9];
            const float s1 = pqs[1] + pqs[4] + pqs[7] + pqs[10];
            const float s2 = pqs[2] + pqs[5] + pqs[8] + pqs[11];
            fin[0] = rsqrtf(s0 + Q27_RMS_EPS) * Q27_GDN_QSCALE;   // invq (carries QSCALE)
            fin[1] = rsqrtf(s1 + Q27_RMS_EPS);                    // invk
            fin[2] = s2 * fin[0] * fin[1];                        // kq = sumqk*invq*invk
            __atomic_store_n(cntqk, 0u, __ATOMIC_RELAXED);
            __threadfence();
            __atomic_store_n(flgqk, (unsigned)gen + 1u, __ATOMIC_RELEASE);   // +1: never the
        } else {                                             // initial zero, whatever gen is
            while (__atomic_load_n(flgqk, __ATOMIC_ACQUIRE) != (unsigned)gen + 1u) { }
        }
    }
    __syncthreads();                                          // barrier 0b
    const float invq = fin[0], invk = fin[1], kq = fin[2];

    if (g == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = d + i * DBLK;     // KEY dim, same indexing as the reads above
            sq[j] = qraw[i] * invq;
            sk[j] = kraw[i] * invk;
        }
    }
    const float beta = 1.0f / (1.0f + expf(-q27_bf2f(bbuf[h])));
    const float t    = q27_bf2f(abuf[h]) + q27_bf2f(dt_bias[hg]);
    const float sp   = (t > 20.0f) ? t : log1pf(expf(t));
    const float decay = expf(-expf(q27_bf2f(A_log[hg])) * sp);

    const float vv = q27_bf2f(qkv_c[voff + h * Q27_GDN_D + dg]);
    __syncthreads();                                          // barrier 2

    // ---- phase 1 : THE single read pass over S_old (unchanged per lane) ----
    const int j0 = g * JPT;
    float* __restrict__ Sp = S + (size_t)h * Q27_GDN_HSTATE
                               + (size_t)j0 * Q27_GDN_VDIM + dg;
    float st[JPT];
#pragma unroll
    for (int j = 0; j < JPT; ++j) st[j] = Sp[j * Q27_GDN_VDIM];
    float accA = 0.0f, accB = 0.0f;
#pragma unroll
    for (int j = 0; j < JPT; ++j) {
        const float sv = st[j];
        accA = fmaf(sv, sk[j0 + j], accA);
        accB = fmaf(sv, sq[j0 + j], accB);
    }
    pA[g][d] = accA;
    pB[g][d] = accB;
    __syncthreads();                                          // barrier 3

    float A = 0.0f, B = 0.0f;
#pragma unroll
    for (int gg = 0; gg < G; ++gg) { A += pA[gg][d]; B += pB[gg][d]; }
    const float delta = (vv - decay * A) * beta;
    const float mixed = fmaf(delta, kq, decay * B);
#pragma unroll
    for (int j = 0; j < JPT; ++j)
        Sp[j * Q27_GDN_VDIM] = fmaf(sk[j0 + j], delta, decay * st[j]);

    // ---- phase 4 : output RMS norm + gate, cross-block over the 128 value dims ----
    {
        const float m2 = (g == 0) ? mixed * mixed : 0.0f;
        const float a3 = q27_wave_sum(m2);
        if (lane == 0) r3[wv] = a3;
    }
    __syncthreads();                                          // barrier 4
    float pmix = 0.0f;
#pragma unroll
    for (int i = 0; i < NW; ++i) pmix += r3[i];
    if (tid == 0) {                     // ONE thread per block, as above
        pmx[db] = pmix;                      // fixed slot, deterministic fixed-order sum
        __threadfence();                     // order the slot store before the arrival count
        const unsigned old = __atomic_fetch_add(cntmx, 1u, __ATOMIC_ACQ_REL);
        if (old == 3u) {
            const float s3 = pmx[0] + pmx[1] + pmx[2] + pmx[3];
            fin[3] = rsqrtf(s3 * (1.0f / (float)Q27_GDN_D) + Q27_RMS_EPS);
            __atomic_store_n(cntmx, 0u, __ATOMIC_RELAXED);
            __threadfence();
            __atomic_store_n(flgmx, (unsigned)gen + 1u, __ATOMIC_RELEASE);
        } else {
            while (__atomic_load_n(flgmx, __ATOMIC_ACQUIRE) != (unsigned)gen + 1u) { }
        }
    }
    __syncthreads();                                          // barrier 4b
    const float invmix = fin[3];

    if (g == 0) {
        const float z   = q27_bf2f(zbuf[h * Q27_GDN_D + dg]);
        const float y   = mixed * invmix * q27_bf2f(norm_w[dg]) * q27_out_gate<GATE>(z);
        out[h * Q27_GDN_D + dg] = q27_f2bf(y);
        if (Q) {
            float mx = fabsf(y);
#pragma unroll
            for (int o = 8; o; o >>= 1) mx = fmaxf(mx, __shfl_down(mx, o, 16));
            mx = __shfl(mx, 0, 16);
            const float sc = mx * qgs * (1.0f / 127.0f);
            const float qi = (sc > 0.0f) ? (qgs / sc) : 0.0f;
            const int   k  = h * Q27_GDN_D + dg;
            if ((d & 15) == 0) QS[k >> 4] = sc;
            int c = (int)rintf(y * qi);
            c = c < -127 ? -127 : (c > 127 ? 127 : c);
            Q[k] = (signed char)c;
        }
    }
}

} // namespace

// ============================================================================
// host entry points.  extern "C" to match the q27.h ABI convention.
// Pointers are the CURRENT LAYER's slices:
//   conv_state -> conv_state_  + linear_index * Q27_GDN_QKV * Q27_GDN_CONV
//   S          -> recurrent_   + linear_index * 48*128*128
// ============================================================================
extern "C" {

void q27_gdn_conv(unsigned short* qkv, unsigned short* conv_state,
                  const unsigned short* cw, hipStream_t s)
{
    const int nb = (Q27_GDN_QKV + Q27_GDN_CONV_NT - 1) / Q27_GDN_CONV_NT;
    hipLaunchKernelGGL(k_gdn_conv, dim3(nb), dim3(Q27_GDN_CONV_NT), 0, s,
                       qkv, conv_state, cw);
}

// Qwen3.8-27B: output_gate_type = swish.
void q27_gdn_step(const unsigned short* qkv_c, const unsigned short* z,
                  const unsigned short* a, const unsigned short* b,
                  const unsigned short* A_log, const unsigned short* dt_bias,
                  const unsigned short* norm_w, float* S,
                  unsigned short* out, hipStream_t s)
{
    hipLaunchKernelGGL((k_gdn_step<Q27_GDN_G, Q27_GATE_SWISH>),
                       dim3(Q27_GDN_VH), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                       qkv_c, z, a, b, A_log, dt_bias, norm_w, S, out);
}

// ---- tensor-parallel entries -------------------------------------------------
// card `g` of `ndev`: 48/ndev value heads, 16/ndev key heads, qkv shard packed q|k|v.
void q27_gdn_conv_tp(unsigned short* qkv, unsigned short* conv_state, int sfull,
                     const unsigned short* cw, int g, int ndev, hipStream_t s)
{
    const int nq = 2048 / ndev, nk = 2048 / ndev, nv = 6144 / ndev;
    const int n  = nq + nk + nv;
    const int nb = (n + Q27_GDN_CONV_NT - 1) / Q27_GDN_CONV_NT;
    hipLaunchKernelGGL(k_gdn_conv_tp, dim3(nb), dim3(Q27_GDN_CONV_NT), 0, s,
                       qkv, conv_state, cw, nq, nk, nv, g * nq, 2048 + g * nk, 4096 + g * nv, sfull);
}

void q27_gdn_step_tp(const unsigned short* qkv_c, const unsigned short* z,
                     const unsigned short* a, const unsigned short* b,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S,
                     unsigned short* out, int g, int ndev,
                     signed char* q, float* qs, float in_scale, hipStream_t s)
{
    // in_scale > 0 enables the fused quantize epilogue and removes the standalone launch.
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { q = nullptr; qs = nullptr; }
    const int vh   = Q27_GDN_VH / ndev;          // value heads on this card == grid.x
    const int koff = 2048 / ndev;                // k block start inside the local qkv shard
    const int voff = koff + 2048 / ndev;         // v block start
    hipLaunchKernelGGL((k_gdn_step_tp<Q27_GDN_G, Q27_GATE_SWISH>),
                       dim3(vh), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                       qkv_c, z, a, b, A_log, dt_bias, norm_w, S, out,
                       koff, voff, g * vh, q, qs, qgs);
}

// Value-dim split: 4 blocks per head. scr is the per-card scratch (12 heads x 12 words,
// zeroed at allocation, self-cleaning counters); gen = pos*LAYERS + L, unique per launch.
void q27_gdn_step_tp_ds(const unsigned short* qkv_c, const unsigned short* z,
                     const unsigned short* a, const unsigned short* b,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S,
                     unsigned short* out, int g, int ndev, unsigned* scr, int gen,
                     signed char* q, float* qs, float in_scale, hipStream_t s)
{
    const float qgs = (in_scale > 0.0f) ? (1.0f / in_scale) : 0.0f;
    if (!(in_scale > 0.0f)) { q = nullptr; qs = nullptr; }
    const int vh   = Q27_GDN_VH / ndev;
    const int koff = 2048 / ndev;
    const int voff = koff + 2048 / ndev;
    hipLaunchKernelGGL((k_gdn_step_tp_ds<Q27_GDN_G, Q27_GATE_SWISH>),
                       dim3(vh, 4), dim3(Q27_GDN_VDIM / 4 * Q27_GDN_G), 0, s,
                       qkv_c, z, a, b, A_log, dt_bias, norm_w, S, out,
                       koff, voff, g * vh, scr, gen, q, qs, qgs);
}

// NOT for Qwen3.8-27B.  Exists so the swish/sigmoid choice is a compile-time
// instantiation rather than a constant someone can edit in place; the sibling
// Flash-Next family sets output_gate_type = sigmoid.  Calling this one on
// Qwen3.8-27B produces plausible, wrong output.
void q27_gdn_step_sigmoid_gate(const unsigned short* qkv_c, const unsigned short* z,
                               const unsigned short* a, const unsigned short* b,
                               const unsigned short* A_log, const unsigned short* dt_bias,
                               const unsigned short* norm_w, float* S,
                               unsigned short* out, hipStream_t s)
{
    hipLaunchKernelGGL((k_gdn_step<Q27_GDN_G, Q27_GATE_SIGMOID>),
                       dim3(Q27_GDN_VH), dim3(Q27_GDN_VDIM * Q27_GDN_G), 0, s,
                       qkv_c, z, a, b, A_log, dt_bias, norm_w, S, out);
}

} // extern "C"

// ============================================================================================
// Q27_Q8_CONVP: the causal 4-tap conv + silu + per-16 requant, PARALLEL over positions (out of place).
// The conv's "state" is only the previous three inputs, so a thread owns (channel, segment of SEG
// positions) and reads x[t-3..t] from the input buffer (or the conv_state for t < 3). The in-place kernel
// walked all Ct positions per thread with 40 blocks (0.21 ms per 256-token call); this one launches
// Ct/SEG times as many threads. Output int8 + per-16 scales go to a separate buffer (the scan's input);
// conv_state is updated with the last three INPUT positions.
// ============================================================================================
template <int SEG>
__global__ __launch_bounds__(256) void k_gdn_conv_q8_par(
    const signed char* __restrict__ q8, const float* __restrict__ qs, signed char* __restrict__ o8, float* __restrict__ os,
    int qstride, int Ct, unsigned short* __restrict__ conv_state, const unsigned short* __restrict__ cw,
    int nq, int nk, int nv, int qbase, int kbase, int vbase)
{
    const int nch = nq + nk + nv;
    const int nseg = (Ct + SEG - 1) / SEG;
    const int ch  = blockIdx.x * 256 + threadIdx.x;         // 256 consecutive channels per block (16 per requant group)
    const int seg = blockIdx.y;
    const bool ok = ch < nch;
    const int chc = ok ? ch : (nch - 1);
    const int gc = (chc < nq) ? (qbase + chc) : (chc < nq + nk) ? (kbase + chc - nq) : (vbase + chc - nq - nk);
    const uint2 w = reinterpret_cast<const uint2*>(cw)[gc];
    const float w0 = q27_bf2f((unsigned short)(w.x & 0xFFFFu)), w1 = q27_bf2f((unsigned short)(w.x >> 16));
    const float w2 = q27_bf2f((unsigned short)(w.y & 0xFFFFu)), w3 = q27_bf2f((unsigned short)(w.y >> 16));
    const int sstride = qstride >> 4, grp = chc >> 4;
    const int t0 = seg * SEG, t1 = min(Ct, t0 + SEG);
    // history x[t0-3], x[t0-2], x[t0-1]
    float h1, h2, h3;
    {
        const uint2 st = reinterpret_cast<const uint2*>(conv_state)[chc];
        const float s1 = q27_bf2f((unsigned short)(st.x >> 16)), s2 = q27_bf2f((unsigned short)(st.y & 0xFFFFu)), s3 = q27_bf2f((unsigned short)(st.y >> 16));
        auto xin = [&](int t) -> float { return (t < 0) ? ((t == -1) ? s3 : (t == -2) ? s2 : s1) : (float)q8[(size_t)t * qstride + chc] * qs[(size_t)t * sstride + grp]; };
        h1 = xin(t0 - 3); h2 = xin(t0 - 2); h3 = xin(t0 - 1);
    }
    for (int t = t0; t < t1; ++t) {
        const float x = (float)q8[(size_t)t * qstride + chc] * qs[(size_t)t * sstride + grp];
        float acc = h1 * w0; acc = fmaf(h2, w1, acc); acc = fmaf(h3, w2, acc); acc = fmaf(x, w3, acc);
        h1 = h2; h2 = h3; h3 = x;
        const float y = q27_silu(acc);
        float mx = fabsf(y);
        mx = fmaxf(mx, __shfl_xor(mx, 8, 16)); mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
        mx = fmaxf(mx, __shfl_xor(mx, 2, 16)); mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
        const float sc = mx * (1.0f / 127.0f);
        const float qi = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        int c = (int)rintf(y * qi); c = c < -127 ? -127 : (c > 127 ? 127 : c);
        if (ok) { o8[(size_t)t * qstride + chc] = (signed char)c; if ((chc & 15) == 0) os[(size_t)t * sstride + grp] = sc; }
    }
    if (ok && seg == nseg - 1) {                           // the last segment holds the last three inputs
        uint2 o;
        o.x = ((unsigned)q27_f2bf(h1) << 16) | (unsigned)q27_f2bf(h1);
        o.y = ((unsigned)q27_f2bf(h3) << 16) | (unsigned)q27_f2bf(h2);
        reinterpret_cast<uint2*>(conv_state)[chc] = o;
    }
}
extern "C" void q27_gdn_conv_q8_par(const signed char* q8, const float* qs, signed char* o8, float* os, int qstride, int Ct,
                                    unsigned short* conv_state, const unsigned short* cw, int g, int ndev, hipStream_t s)
{
    const int nq = 2048 / ndev, nk = 2048 / ndev, nv = 6144 / ndev;
    const int n  = nq + nk + nv;
    const int nb = (n + 255) / 256, nseg = (Ct + 15) / 16;
    hipLaunchKernelGGL((k_gdn_conv_q8_par<16>), dim3(nb, nseg), dim3(256), 0, s,
                       q8, qs, o8, os, qstride, Ct, conv_state, cw, nq, nk, nv, g * nq, 2048 + g * nk, 4096 + g * nv);
}
