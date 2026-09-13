// Qwen3.8-27B native engine for AMD gfx906 (MI50 / Radeon Pro VII), wave64, ROCm 5.7.1.
// THE ABI. Every kernel codes against this. Producer/consumer contracts are stated here once.
//
// Sources of truth:
//   architecture + numerics : this header and the per-kernel comments in src/
//   storage ABI             : below, derived from the checkpoint byte layout
//   execution representation: below, int8 per-64 weights / per-16 activations
//   gates                   : the oracle capture directory (layer vectors + golden generations), see tests/
#pragma once
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cstddef>

// ---------------- model geometry (config.json + verified tensor shapes) ----------------
#define Q27_HID        5120
#define Q27_INTER      17408
#define Q27_LAYERS     64
#define Q27_VOCAB      248320
#define Q27_RMS_EPS    1e-6f
// full attention on layers where (L & 3) == 3  -> 3,7,...,63 (16 layers)
#define Q27_IS_FULL(L) (((L) & 3) == 3)
// full attention
#define Q27_NHEAD      24
#define Q27_NKV        4
#define Q27_HDIM       256
#define Q27_QROWS      12288        // 24 heads x 512 = 256 query + 256 OUTPUT GATE per head
#define Q27_KVROWS     1024         // 4 x 256
#define Q27_OROWS      6144         // 24 x 256  (o_proj input width)
#define Q27_ROT        64           // partial_rotary_factor 0.25 -> first 64 dims only
#define Q27_ROT_HALF   32
#define Q27_ROPE_THETA 10000000.0f
#define Q27_ATTN_SCALE 0.0625f      // 1/sqrt(256)
// gated delta (linear attention), 48 layers
#define Q27_GDN_KH     16
#define Q27_GDN_VH     48
#define Q27_GDN_D      128
#define Q27_GDN_QKV    10240        // q 2048 | k 2048 | v 6144
#define Q27_GDN_Z      6144
#define Q27_GDN_CONV   4
#define Q27_GDN_QSCALE 0.08838834764831845f   // 1/sqrt(128), applied to q ONLY
#define Q27_PF_CAP     1024                  // prefill-sweep position slots (hid + pending)
#ifndef Q27_PF_CH
#define Q27_PF_CH      32                    // MAX prefill chunk width (sizes the chunk buffers); the runtime width is 8/16/32 by prompt length
#endif
#define Q27_PF_MC      8                     // MLP / q-k-v kernel width: these stay 8-wide (register shape) and run ceil(CH/8) times per chunk
// Minimum waves per SIMD for the fp8 projection kernels. They measured 233 VGPRs under a bare
// __launch_bounds__(256), i.e. ONE wave per SIMD: occupancy/latency bound, not arithmetic bound,
// which is why changing their M never helped (receipt v108). Forcing 2+ makes the compiler fit
// the register budget (spilling if it must) and is the cheapest test of that diagnosis.
#ifndef Q27_FP8_MINW
#define Q27_FP8_MINW 1   // 2 was measured WORSE (3705 vs 2613 ms): the spills cost more than the occupancy
#endif

#ifndef Q27_PF_TSLOT
#define Q27_PF_TSLOT   256                   // large-M tile slots (norm/act/qkv/z per position)
#endif   // overridable: the wide FFN wants the tile to span a whole layer (see receipt v103)

// ---------------- storage ABI (proven from the checkpoint bytes) ----------------
// NVFP4 weight  : U8 [R][K/2], byte j holds element 2j in the LOW nibble, 2j+1 in the HIGH.
//                 group scale : F8_E4M3 [R][K/16], dense row-major, NO swizzle on disk.
//                 weight_scale_2 : F32 scalar, a MULTIPLIER (= amax/(6*448)).
//   w[r][k] = E2M1(code) * e4m3(gs[r][k>>4]) * weight_scale_2
// FP8 weight    : F8_E4M3 [R][K], w = e4m3(byte) * weight_scale (scalar multiplier).
// Activation    : quantized with global = 1/input_scale, per-group-of-16 e4m3 scale:
//                 s[g] = e4m3(max|x| over the 16 * global / 6), q = e2m1(x * global / s[g]).
// GEMM epilogue : alpha = input_scale * weight_scale_2, applied ONCE.
//
// EXECUTION REPRESENTATION (measured, 16.8x over float dequant):
//   E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} x2 = {0,1,2,3,4,6,8,12}, all int8. So NVFP4 feeds
//   v_dot4_i32_i8 EXACTLY with the factor 1/2 folded into the group scale. Decode with
//   q27_dq4() below: two v_perm_b32 + one v_bfi_b32 per 4 nibbles. The bfi form is carry-safe;
//   do NOT use the (mag ^ mask) + sign form, which carries into the next byte on code 0x8 (-0).
//   gfx906 has ONLY the 4-operand VOP3 v_dot4_i32_i8 (Tensile AsmCaps (9,0,6)); there is no
//   v_dot4c and no v_dot2c_f32_f16. Use __builtin_amdgcn_sdot4(a,b,c,false).

#define Q27_MAG_HI 0x0C080604u
#define Q27_MAG_LO 0x03020100u
#define Q27_NEG_HI 0xF4F8FAFCu
#define Q27_NEG_LO 0xFDFEFF00u
#define Q27_SGN_TBL 0x0000FF00u

// four E2M1 nibbles (low half of each byte) -> four int8 lanes carrying 2x the value
__device__ __forceinline__ unsigned q27_dq4(unsigned d) {
    const unsigned mag = d & 0x07070707u, sel = (d >> 3) & 0x01010101u;
    const unsigned pos = __builtin_amdgcn_perm(Q27_MAG_HI, Q27_MAG_LO, mag);
    const unsigned neg = __builtin_amdgcn_perm(Q27_NEG_HI, Q27_NEG_LO, mag);
    const unsigned msk = __builtin_amdgcn_perm(0u, Q27_SGN_TBL, sel);
    return pos ^ ((pos ^ neg) & msk);
}
// e4m3 group scale. KEEP THE BRANCHES: e==0 and e==15&&m==7 are rare in real scale data, so
// s_cbranch_execz skips them wave-wide. Pangu measured de-branching WORSE (56.7 vs 46.0 us).
// SIGNED e4m3 (fp8 WEIGHT bytes: sign bit 7, exponent bits 6-3 bias 7, mantissa bits 2-0). q27_e4m3 below is the
// UNSIGNED group-scale decoder and must never be applied to a weight byte: a negative weight (bit 7 set) decodes
// there as a huge positive exponent -- which is exactly what turned every fp8->NVFP4 twin into garbage (2026-09-11).
__device__ __forceinline__ float q27_e4m3s(unsigned b) {
    const unsigned s = (b >> 7) & 1u, e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 0) v = (float)m * 0x1p-9f;
    else if (e == 15 && m == 7) v = INFINITY;
    else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
__device__ __forceinline__ float q27_e4m3(unsigned b) {
    const unsigned e = b >> 3, m = b & 7u;
    if (e == 0)  return (float)m * 0x1p-9f;
    if (e == 15 && m == 7) return INFINITY;
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}
// Select form of the same decode (identical values). The 8-row LDS kernels compiled the if-form into
// per-group s_and_saveexec regions; the single-row kernels keep the if-form: switching them cost 9% of
// plain decode (61.5 vs 67.3 tok/s; they sit at the 64-VGPR occupancy edge).
__device__ __forceinline__ float q27_e4m3_sel(unsigned b) {
    const unsigned e = b >> 3, m = b & 7u;
    const float nrm = __uint_as_float(((e + 120u) << 23) | (m << 20));
    const float sub = (float)m * 0x1p-9f;
    float v = (e == 0) ? sub : nrm;
    v = ((b & 0x7fu) == 0x7fu) ? INFINITY : v;
    return v;
}
__device__ __forceinline__ int q27_d4(unsigned w, unsigned x, int a) {
    return __builtin_amdgcn_sdot4((int)w, (int)x, a, false);
}
typedef unsigned q27_u32x4v __attribute__((ext_vector_type(4)));
#if defined(__HIPCC__) || defined(__HIP__)
__device__ __forceinline__ uint4 q27_ld16ntv(const void* p) {
    const q27_u32x4v v = __builtin_nontemporal_load((const q27_u32x4v*)p);
    return make_uint4(v.x, v.y, v.z, v.w);
}
#endif
__device__ __forceinline__ float q27_bf2f(unsigned short b) {
    return __uint_as_float(((unsigned)b) << 16);
}
__device__ __forceinline__ unsigned short q27_f2bf(float f) {   // round-to-nearest-even
    unsigned u = __float_as_uint(f);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// ---------------- ACTIVATION PERMUTATION CONTRACT ----------------
// Consumers of q27_dq4 require the int8 activation permuted EVEN-THEN-ODD within each group
// of 16: dst[(g<<4) + (j&1 ? 8+(j>>1) : (j>>1))] = src[g*16+j].
// This is produced ONCE, upstream, by the quantizer. NEVER in the consumer: Pangu measured a
// consumer-side permutation costing 2.95 MB of redundant LDS traffic per call.
// FP8 projections consume the NATURAL order. Do not contaminate the shared producer.

// ---------------- weight handles ----------------
typedef struct {                  // NVFP4 projection: rows x K
    const unsigned char* w;       // [rows][K/2] packed
    const unsigned char* gs;      // [rows][K/16] e4m3 group scales
    float ws2;                    // weight_scale_2 (multiplier)
    float in_scale;               // input_scale (activation static scale)
    int rows, K;
} q27_nvfp4_t;                    // (the optional int8 mirror of a tensor lives in a side table: q27_nvfp4_w8(); the
                                  //  struct keeps its release layout - an extra pointer here cost 3% of decode, receipt v93)

typedef struct {                  // INT8 per-64-group execution mirror (Q27_LS_Q8): rows x K int8, natural K order
    const signed char* w;         // [rows][K]
    const float* s;               // [rows][K/64] fp32 group scales
    float alpha;                  // in_scale * weight_scale_2 (applied once per output)
    int rows, K;
} q27_i8g_t;

typedef struct {                  // INT8 per-ROW execution mirror for the rocBLAS int8 GEMMs (Q27_LS_Q8 stage 3)
    const signed char* w;         // [rows][K] natural K order (the per-64 mirror requantized in place)
    const float* s;               // [rows][ng] fp32 scales per (row, 1024-group) (= max of the 16 per-64 scales)
    float alpha;                  // as q27_i8g_t
    int ng;                       // number of K groups (K / gs)
    int gs;                       // group size: 1024, or K (per-row / per-token scales)
    int rows, K;
} q27_i8r_t;

typedef struct {                  // FP8 projection: rows x K
    const unsigned char* w;       // [rows][K] e4m3
    float wscale, in_scale;
    int rows, K;
} q27_fp8_t;


// ---------------------------------------------------------------------------------------------
// ENV FLAGS WITH VALUE SEMANTICS. Four guards in this engine used `getenv("X") != nullptr`, i.e.
// PRESENCE semantics, so `X=0` ENABLED them -- the opposite of how every one of them reads, and
// directly contradicting q27_nvfp4.hip's own comment that "W32=1 restores the old geometry".
// No measurement taken on 2026-09-09 is affected (every arm used =1 or unset, never =0), but an
// A/B run with =0 expecting the default would silently have measured the wrong arm.
// FAILS CLOSED on an unparseable value rather than guessing, because a typo in a benchmark flag
// should stop the run, not quietly select an arm.
static inline bool q27_env_flag(const char* name, bool def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    if (!strcmp(v,"0") || !strcmp(v,"false") || !strcmp(v,"no")  || !strcmp(v,"off")) return false;
    if (!strcmp(v,"1") || !strcmp(v,"true")  || !strcmp(v,"yes") || !strcmp(v,"on"))  return true;
    fprintf(stderr, "FATAL %s=\"%s\" is not a boolean (use 0/1, false/true, no/yes, off/on). "
                    "Refusing to guess which arm you meant.\n", name, v);
    abort();
}
static inline int q27_env_int(const char* name, int def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    char* end = nullptr; const long x = strtol(v, &end, 10);
    if (end == v || *end) {
        fprintf(stderr, "FATAL %s=\"%s\" is not an integer.\n", name, v); abort();
    }
    return (int)x;
}

// ---------------- kernel entry points (host-callable, hipStream_t) ----------------
#ifdef __cplusplus
extern "C" {
#endif
// q27_nvfp4.hip
void q27_quant_perm(const unsigned short* x_bf16, signed char* xq_perm, float* xs,
                    int n, float in_scale, hipStream_t s);      // even-then-odd, per-16 scales
void q27_proj_nvfp4(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                    float* y, hipStream_t s);
int  q27_proj_nvfp4_bf16(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                         unsigned short* y, hipStream_t s);   // K=5120 only, BF16 epilogue
void q27_fp8_to_nvfp4_conv(const q27_fp8_t* src, q27_nvfp4_t* dst, hipStream_t st);  // at load
int  q27_nvfp4_make_w8(const q27_nvfp4_t* w, hipStream_t s);   // allocate + fill the tensor's int8 mirror (side table); 1 on success
const unsigned char* q27_nvfp4_w8(const q27_nvfp4_t* w);      // the tensor's int8 mirror or NULL
void q27_nvfp4_free_w8(const q27_nvfp4_t* w);
int  q27_nvfp4_make_w8s(const q27_nvfp4_t* w, hipStream_t s);   // PRESHUFFLED int8 mirror [kk][tile][tx][4r][16B]
const unsigned char* q27_nvfp4_w8s(const q27_nvfp4_t* w);
// Batched projection (prefill): C positions share every weight load; K=5120 only.
// xq_perm/xs are per-position arrays ([C][K] int8 / [C][K/16] f32); y is [C][rows].
void q27_proj_nvfp4_b(const q27_nvfp4_t* w, const signed char* xq_perm,
                      const float* xs, float* y, int C, hipStream_t s);
void q27_proj_nvfp4_gu_b(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                         const signed char* xq_perm, const float* xs,
                         float* yg, float* yu, int C, hipStream_t s);
void q27_proj_nvfp4_bt(const q27_nvfp4_t* w, const signed char* xq_perm,
                       const float* xs, float* y, int C, hipStream_t s);   // K=4352
// Batched fp8 (prefill): R=1, K=5120, BF16 epilogue; xq/xs per position, y is [C][rows].
void q27_proj_fp8_s1_b(const q27_fp8_t* w, const signed char* xq, const float* xs,
                       unsigned short* y, int C, int ystride, hipStream_t s);
// q27_fp8.hip
void q27_quant_fp8(const unsigned short* x_bf16, signed char* xq, float* xs,
                   int n, float in_scale, hipStream_t s);       // NATURAL order
// Forces rows-per-wave R (1, 2 or 4). For the shape census only; production goes through
// q27_proj_fp8, which picks R by the largest value whose wave population reaches 960.
void q27_mk_bench(void);
void q27_sync_tax(void);
void q27_mlp_bench(void);
void q27_mlpb_bench(void);
void q27_wide_bench(void);            // wide (GEMM-order) prefill kernels vs rung 3, Q27_WIDE_BENCH=1
int  q27_wide_gu(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                 float* yg, float* yu, int M, int ki, hipStream_t s);
int  q27_wide_down(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                   const unsigned short* radd, float rs, int ki, hipStream_t s);
int  q27_wide_gu_w8(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                    float* yg, float* yu, int M, int tcb, hipStream_t s);
int  q27_wide_gu_rbpf(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                      float* yg, float* yu, int M, int tcb, hipStream_t s);
int  q27_wide_gu_rb2p(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                      float* yg, float* yu, int M, int tcb, hipStream_t s);
int  q27_wide_gu_rb3p(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                      float* yg, float* yu, int M, int tcb, hipStream_t s);
int  q27_wide_down_rbpf(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                        const unsigned short* radd, float rs, int tcb, hipStream_t s);
int  q27_wide_down_b_rbpf(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                          unsigned short* y, int ystride, int M, int tcb, hipStream_t s);
int  q27_wide_down_rb2p_tr32(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                             const unsigned short* radd, float rs, hipStream_t s);
int  q27_wide_down_rb2p_sk2(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                            float* y0, float* y1, float* y2, float* y3, int splits, int M,
                            const unsigned short* radd, float rs, hipStream_t s);
void q27_add_f32(float* a, const float* b, int n, hipStream_t s);
int  q27_wide_down_b_rb2p3(const q27_nvfp4_t* wq, const q27_nvfp4_t* wk, const q27_nvfp4_t* wv,
                           const signed char* xq, const float* xs,
                           unsigned short* y0, unsigned short* y1, unsigned short* y2, int ystride, int M,
                           hipStream_t s);
void q27_red2_bf16(unsigned short* y, const float* a, const float* b,
                   const unsigned short* r, int n, hipStream_t s);
void q27_red2_f32(float* y, const float* a, const float* b,
                  const unsigned short* r, int n, hipStream_t s);
int  q27_wide_down_rb2p(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                        const unsigned short* radd, float rs, int tcb, hipStream_t s);
int  q27_wide_down_b_rb2p(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                          unsigned short* y, int ystride, int M, int tcb, hipStream_t s);
int  q27_wide_down_w8(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                      const unsigned short* radd, float rs, int tcb, hipStream_t s);
void q27_bw_probe(void);   // size sweep: is the census ceiling an L2 artifact?   // NVFP4 MLP census + drain tax   // drain-tax measurement: same kernel with/without a stream drain   // kernel-level A/B: s1_b vs the large-M v2 projection
void q27_fp8_census(void);   // Q27_FP8_CENSUS=1: the 3x3 shape table
// LARGE-M fp8 projection for the prefill chunk engine: M positions starting at pos0, block tile
// [4 rows x 64 positions], output written straight into the consumer slot [pos][ystride] bf16.
// Returns 0 when the shape is not the K=5120 form.
// v2 large-M projection: lanes cover K, the row weights are hoisted, positions walk in
// groups of 4. Same contract as q27_proj_fp8_m.
// v3: v2 + rows-per-wave (rw in {1,2,4}); one activation chunk feeds rw rows from registers.
// v4: hoists the e4m3->half2 weight DECODE into registers (once per tile, not per position).
// v6: consumer of a PREDECODED half2 activation representation (exact int8->half2).
void q27_i8_to_h2(const signed char* src, unsigned* dst, int n, hipStream_t s);
// v7: block-owned row tiling (8 rows/wave, C=32 group) -- cuts activation re-reads 8x.
// v9: row-tiled with TRANSIENT weights (isolates activation traffic from register pressure).
int  q27_proj_fp8_m9(const q27_fp8_t* w, const signed char* xq, const float* xs,
                    unsigned short* y, int M, int ystride, int pos0, int rw, hipStream_t s);
int  q27_proj_fp8_m7(const q27_fp8_t* w, const signed char* xq, const float* xs,
                    unsigned short* y, int M, int ystride, int pos0, hipStream_t s);
int  q27_proj_fp8_m6(const q27_fp8_t* w, const unsigned* xh2, const float* xs,
                    unsigned short* y, int M, int ystride, int pos0, hipStream_t s);
int  q27_proj_fp8_m4(const q27_fp8_t* w, const signed char* xq, const float* xs,
                    unsigned short* y, int M, int ystride, int pos0, hipStream_t s);
int  q27_proj_fp8_m3(const q27_fp8_t* w, const signed char* xq, const float* xs,
                    unsigned short* y, int M, int ystride, int pos0, int rw, hipStream_t s);
int  q27_proj_fp8_m2(const q27_fp8_t* w, const signed char* xq, const float* xs,
                     unsigned short* y, int M, int ystride, int pos0, hipStream_t s);
int  q27_proj_fp8_m(const q27_fp8_t* w, const signed char* xq, const float* xs,
                    unsigned short* y, int M, int ystride, int pos0, hipStream_t s);
void q27_proj_fp8_R(const q27_fp8_t* w, const signed char* xq, const float* xs,
                    float* y, int R, hipStream_t s);
void q27_proj_fp8(const q27_fp8_t* w, const signed char* xq, const float* xs,
                  float* y, hipStream_t s);
// q27_elem.hip
// Fused rmsnorm + quantize. Returns 1 if it ran, 0 if the shape is uncovered and the caller must
// issue the two kernels separately. y (the normalised bf16) is always written: the GDN layers'
// bf16 GEMVs read it. _perm produces the even-then-odd activation permutation, as the separate
// quantizer does -- the contract is unchanged, only the launch count is.
// `add` non-null folds a residual add (x += add, written back) in before the norm, deleting the
// separate q27_add_inplace launch. Pass nullptr for a plain norm.
int  q27_rmsnorm_quant_fp8(const unsigned short* x, const unsigned short* wgt, unsigned short* y,
                           signed char* xq, float* xs, int n, int plus_one, float in_scale,
                           const void* add, int abf, hipStream_t s);
int  q27_proj_fp8_res(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y,
        const unsigned short* radd, float rs, hipStream_t s);
void q27_nvfp4_set_res(const unsigned short* radd, float rs);
void q27_launch_nothing(hipStream_t s);
void q27_add_res_share(float* part, const unsigned short* hid, float rs, int n, hipStream_t s);
int  q27_rmsnorm_hostss_fp8(const float* x, const unsigned short* wgt, unsigned short* hid,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const float* invp, hipStream_t s);
int  q27_rmsnorm_hostss_perm(const float* x, const unsigned short* wgt, unsigned short* hid,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const float* invp, hipStream_t s);
int  q27_rmsnorm_quant_perm(const unsigned short* x, const unsigned short* wgt, unsigned short* y,
                            signed char* xq, float* xs, int n, int plus_one, float in_scale,
                            const void* add, int abf, hipStream_t s);
void q27_rmsnorm(const unsigned short* x, const unsigned short* wgt, unsigned short* y,
                 int n, int plus_one, hipStream_t s);           // plus_one=1 except linear_attn.norm
void q27_add_inplace(unsigned short* h, const float* m, int n, hipStream_t s);
// BF16 residual add, for when the collective transports BF16 instead of FP32.
void q27_add_inplace_bf16(unsigned short* h, const unsigned short* m, int n, hipStream_t s);
void q27_swiglu(const float* gate, const float* up, unsigned short* out, int n, hipStream_t s);
void q27_embed_gather(const unsigned short* tbl, unsigned tok, unsigned short* out, hipStream_t s);
void q27_argmax(const float* logits, unsigned* out, int n, hipStream_t s);
// argmax that also returns the winning value, for the tensor-parallel head: each card reduces its
// own logit shard and the four (value, global index) pairs settle the token.
void q27_argmax_val(const float* logits, unsigned* out_idx, float* out_val, int n, hipStream_t s);

// ---------------- tensor-parallel kernel entry points (tensor-parallel design) ----------
// Same mathematics, shard geometry.  `ndev` is the number of cards (1, 2 or 4 are instantiated);
// `g` is this card's index.  Attention needs no collective because the GQA group is never split.
void q27_attn_prep_tp(const unsigned short* qkv_out, const unsigned short* q_norm_w,
                      const unsigned short* k_norm_w, signed char* kcache, float* kscale,
                      signed char* vcache, float* vscale, unsigned short* q_out, int pos, int ndev, int kvstride, int hoff0,
                      hipStream_t s);
// Returns 1 if the fused quantize epilogue ran; 0 means the caller must still launch
// q27_quant_fp8 (the single-split path skips stage 2).
int  q27_attn_decode_tp(const unsigned short* q, const signed char* kcache, const float* kscale,
                        const signed char* vcache, const float* vscale, const unsigned short* q_proj_raw,
                        unsigned short* out, int pos, int ndev, int kvstride, int hoff0,
                        signed char* qout, float* qscl, float in_scale, hipStream_t s);
void q27_gdn_conv_tp_tile(unsigned short* qkv, int qstride, int Ct, unsigned short* conv_state,
                          const unsigned short* cw, int g, int ndev, hipStream_t s);
// ---- Q27_LS_Q8: the quantized producer/consumer prefill graph (2026-09-11) ----
// int8 + per-16 fp32 scale everywhere between operators; fp32 residual stream; int8 KV cache.
int  q27_wide_q8_rb2p(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                      signed char* yq, float* ys, int ystride, int M, hipStream_t s);
int  q27_wide_q8_rb2p3(const q27_nvfp4_t* wq, const q27_nvfp4_t* wk, const q27_nvfp4_t* wv,
                       const signed char* xq, const float* xs,
                       signed char* yq0, signed char* yq1, signed char* yq2,
                       float* ys0, float* ys1, float* ys2, int ystride, int M, hipStream_t s);
int  q27_wide_down_rb2p_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                             const float* radd, float rs, hipStream_t s);
int  q27_q8_gemv2_tile(const unsigned short* wa, const unsigned short* wb,
                       const signed char* xq, const float* xs, int xstride, float xmul, int perm, int xs64,
                       float* ya, float* yb, int rows, int K, int Ct, hipStream_t s);
void q27_gdn_conv_q8_tile(signed char* q8, float* qs, int qstride, int Ct, unsigned short* conv_state,
                          const unsigned short* cw, int g, int ndev, hipStream_t s);
void q27_gdn_conv_q8_par(const signed char* q8, const float* qs, signed char* o8, float* os, int qstride, int Ct, unsigned short* conv_state, const unsigned short* cw, int g, int ndev, hipStream_t s);   // out-of-place, position-parallel
void q27_gdn_scan2_q8(const signed char* q8, const float* qs, int qstride,
                      const signed char* z8, const float* zs, int zstride,
                      const float* a_t, const float* b_t, int abstride,
                      const unsigned short* A_log, const unsigned short* dt_bias,
                      const unsigned short* norm_w, float* S,
                      unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                      signed char* q_t, float* qs_t, float in_scale, int perm, int g64, int par, hipStream_t s);
int  q27_wide_down_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                        const float* radd, float rs, int ki, hipStream_t s);
void q27_fp8_to_i8g_conv(const q27_fp8_t* src, q27_nvfp4_t* dst, hipStream_t st);   // int8 dot4 execution mirror of an fp8 projection (at load)
int  q27_wide_down_rb_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                           const float* radd, float rs, hipStream_t s);
int  q27_wide_i8_rb_q8m(int n, const q27_nvfp4_t* const* ws, const signed char* xq, const float* xs,
                        signed char* const* yq, float* const* ys, const int* ystride, int M, hipStream_t s);
int  q27_wide_i8_rb_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                         const float* radd, float rs, hipStream_t s);
int  q27_fp8_to_i8g64_conv(const q27_fp8_t* src, q27_i8g_t* dst, hipStream_t st);   // int8 per-64 mirror of an fp8 projection (at load)
int  q27_fp8_to_i8g64_cat(const q27_fp8_t* const* srcs, int n, q27_i8g_t* dsts, hipStream_t st);   // stacked rows, one buffer (fused GEMM)
int  q27_rmsnorm_hostss_g64_b(const float* x, const unsigned short* wgt, signed char* xq, float* xs, int n, int plus_one, float in_scale, const float* invp, int C, hipStream_t s);
int  q27_wide_i8g64r8_q8m(int n, const q27_i8g_t* const* ws, const signed char* xq, const float* xs, signed char* const* yq, float* const* ys, const int* ystride, int M, hipStream_t s);
int  q27_wide_i8g64r8_f32r(const q27_i8g_t* w, const signed char* xq, const float* xs, float* y, int M, const float* radd, float rs, hipStream_t s);
int  q27_wide_gu_i8g64r8(const q27_i8g_t* wg, const q27_i8g_t* wu, const signed char* xq, const float* xs, float* yg, float* yu, int M, hipStream_t s);
int  q27_wide_i8g64_q8m(int n, const q27_i8g_t* const* ws, const signed char* xq, const float* xs, signed char* const* yq, float* const* ys, const int* ystride, int M, hipStream_t s);
int  q27_wide_i8g64_f32r(const q27_i8g_t* w, const signed char* xq, const float* xs, float* y, int M, const float* radd, float rs, hipStream_t s);
int  q27_nvfp4_to_i8g64_conv(const q27_nvfp4_t* src, q27_i8g_t* dst, hipStream_t st);   // int8 per-64 mirror of an NVFP4 tensor (at load)
int  q27_swiglu_quant_b_g64(const float* gate, const float* up, signed char* xq, float* xs, int n, float in_scale, int C, hipStream_t s);
// Q27_LS_Q8 stage 3: rocBLAS int8 GEMM path (src/q27_rb.cpp, src/q27_epi.hip)
int  q27_rb_init(int dev);
int  q27_rb_gemm_i8g(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, int GS, hipStream_t s);   // C[g][M][N] = X[M][group g] * W[N][group g]^T
int  q27_rb_tune_i8g(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, hipStream_t s);   // pick the fastest Tensile solution for (N,K,M) once; primes the shape
int  q27_i8g64_to_row_conv(q27_i8g_t* src, q27_i8r_t* dst, int gs, hipStream_t st);   // gs: 1024 or 0 = full K
int  q27_rmsnorm_hostss_tok_b(const float* x, const unsigned short* wgt, signed char* xq, float* xs, int n, int plus_one, float in_scale, const float* invp, int C, int gs, hipStream_t s);
int  q27_red_rn_tok_b1(const float* x, const unsigned short* wgt, float* hid32, signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, int gs, hipStream_t s);
int  q27_requant_g64_tok(const signed char* xq, const float* xs, signed char* yq, float* ys, int K, int xstride, int M, int gs, hipStream_t s);
int  q27_epi_q8(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha, signed char* yq, float* ys, int ystride, int M, hipStream_t s);
int  q27_epi_q8v2(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha, signed char* yq, float* ys, int ystride, int M, hipStream_t s);   // 4 columns per thread
int  q27_ab_to_i8r_conv(const unsigned short* wa, const unsigned short* wb, int rows, int K, q27_i8r_t* dst, hipStream_t st);   // GDN a/b bf16 -> int8 per-row [2*rows][K]
int  q27_epi_ab(const int* acc, int rows, const float* s, const float* stok, float xmul, float* ya, float* yb, int M, hipStream_t st);
int  q27_copy_f32(const float* src, float* dst, size_t n, hipStream_t s);   // fp32 copy kernel (peer-store handoff)
int  q27_bf16_to_f32(const unsigned short* src, float* dst, size_t n, hipStream_t s);   // bf16 rows -> fp32 rows
int  q27_epi_f32r(const int* acc, int N, const float* srow, int ng, const float* stok, float alpha, float* y, const float* radd, float rs, int M, hipStream_t s);
int  q27_epi_res_rn_tok(const int* acc, int N, const float* srow, int ng, const float* stok, float alpha, const float* pend, float* part, const unsigned short* wgt, float* hid32, signed char* xq, float* xs, int plus_one, float in_scale, int M, hipStream_t s);   // o/out_proj epilogue + residual + post-norm (per-token int8), fused
int  q27_epi_swiglu_tok(const int* ag, const int* au, int N, const float* sg, const float* su, int ng, const float* stok, float alpha_g, float alpha_u, float gscale, signed char* xq, float* xs, int M, int gs, hipStream_t s);
int  q27_red_rn_g64_b1(const float* x, const unsigned short* wgt, float* hid32, signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, hipStream_t s);
int  q27_wide_gu_i8g64(const q27_i8g_t* wg, const q27_i8g_t* wu, const signed char* xq, const float* xs, float* yg, float* yu, int M, hipStream_t s);
int  q27_wide_down_i8g64_f32r(const q27_i8g_t* w, const signed char* xq, const float* xs, float* y, int M, const float* radd, float rs, hipStream_t s);
int  q27_wide_i8_q8m(int n, const q27_nvfp4_t* const* ws, const signed char* xq, const float* xs,
                     signed char* const* yq, float* const* ys, const int* ystride, int M, int ki, hipStream_t s);
int  q27_wide_i8_q8(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                    signed char* yq, float* ys, int ystride, int M, int ki, hipStream_t s);
int  q27_wide_i8_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                      const float* radd, float rs, int ki, hipStream_t s);
int  q27_proj_fp8_wide_q8(const q27_fp8_t* w, const signed char* xq, const float* xs,
                          signed char* yq, float* ys, int ystride, int M, int ki, hipStream_t s);
int  q27_proj_fp8_wide_f32r(const q27_fp8_t* w, const signed char* xq, const float* xs,
                            float* y, int M, int ystride, const float* radd, int rstride, float rs,
                            int ki, hipStream_t s);
int  q27_attn_chunk_q8(const signed char* qkv8_t, const float* qkvs_t, int qstride,
                       const unsigned short* q_norm_w, const unsigned short* k_norm_w,
                       signed char* kcache, float* kscale, signed char* vcache, float* vscale,
                       float* qout_t, unsigned short* out_t, int ostride,
                       int pos0, int M, int ndev,
                       signed char* Q_t, float* QS_t, float in_scale, int hpw, int attpf,
                       float* pob, float* pml, int perm, int g64, hipStream_t s);
void q27_gdn_scan_tp(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                     const unsigned short* a_t, const unsigned short* b_t, int abstride,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S,
                     unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                     signed char* q_t, float* qs_t, float in_scale, hipStream_t s);
void q27_gdn_scan2_tp(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                     const unsigned short* a_t, const unsigned short* b_t, int abstride,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S,
                     unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                     signed char* q_t, float* qs_t, float in_scale, hipStream_t s);   // scan v2 (Q27_PF_GDN_SCAN2): phase 0 precomputed in parallel, two barriers per position

void q27_gdn_conv_tp(unsigned short* qkv, unsigned short* conv_state, int sfull, const unsigned short* cw,
                     int g, int ndev, hipStream_t s);
void q27_gdn_step_tp(const unsigned short* qkv_c, const unsigned short* z,
                     const unsigned short* a, const unsigned short* b,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S, unsigned short* out,
                     int g, int ndev,
                     // in_scale > 0 fuses the quantize into phase 4 and removes the standalone
                     // q27_quant_fp8 launch; pass 0 / nullptr to keep them separate.
                     signed char* q, float* qs, float in_scale, hipStream_t s);
// Value-dim split: 4 blocks per value head, cross-block scalar norms via per-card scratch.
void q27_gdn_step_tp_ds(const unsigned short* qkv_c, const unsigned short* z,
                     const unsigned short* a, const unsigned short* b,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S, unsigned short* out,
                     int g, int ndev, unsigned* scr, int gen,
                     signed char* q, float* qs, float in_scale, hipStream_t s);
#ifdef __cplusplus
}
#endif
