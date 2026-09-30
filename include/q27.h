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
// NORMAL-RANGE-ONLY group-scale decode: 2 instructions (v_add_u32 + v_lshlrev_b32), no exec mask.
//
// THIS IS A DELETION OF UNREACHABLE ARMS, NOT A DE-BRANCHING. De-branching was measured WORSE twice
// (the comments above q27_e4m3 and q27_e4m3_sel record both) because a select form still EVALUATES
// both arms and keeps the extra operands live. This form evaluates neither: the subnormal arm (e==0,
// i.e. byte < 8) and the NaN/Inf arm (byte == 127) are simply gone.
//
// IDENTITY (exact on 8 <= b <= 126): with b = (e<<3)|m, b<<20 == (e<<23)|(m<<20), and 960<<20 ==
// 120<<23, and m<<20 occupies bits 20-22 so it can never carry into bit 23. Therefore
// (b + 960) << 20 == ((e + 120) << 23) | (m << 20), which is q27_e4m3's normal path verbatim.
//
// PRECONDITION, MEASURED over this checkpoint 2026-09-17 -- 193 unsigned E4M3 group-scale planes,
// 1,149,009,920 bytes: 0 bytes < 8, 0 bytes == 127, 0 bytes >= 128, min 55, max 126. Note the NaN
// boundary is ONE code away (126 vs 127), so this is not a property to assume -- it is a property to
// VALIDATE per plane at load. q27_f32_to_e4m3 (the runtime producer under Q27_FP8X4=1, default off)
// has a live subnormal branch and returns 0 for an all-zero group, so a runtime-produced plane is NOT
// covered by any static scan of the checkpoint. Callers must fail closed to q27_e4m3_sel.
//
// NEVER apply this to a WEIGHT byte -- see q27_e4m3s above; that mistake is what turned every
// fp8->NVFP4 twin into garbage on 2026-09-11.
__device__ __forceinline__ float q27_e4m3_fast(unsigned b) {
    return __uint_as_float((b + 960u) << 20);
}
// OCP MXFP4 block scale: E8M0 is exponent-ONLY, 8 bits, bias 127, value = 2^(b-127). A float whose
// exponent field is b and whose mantissa is zero IS that value, so the whole decode is one shift.
// Compare q27_e4m3 (~6 ops + an exec region) and even q27_e4m3_fast (2 ops). b==0 and b==255 are the
// reserved/NaN encodings in OCP E8M0; measured over the downloaded amd/Qwen3.8-27B-Quark-AWQ-MXFP4
// gate_proj plane (2,785,280 bytes): zero of each, codes span 117..124 only (2^-10..2^-3), 8 distinct
// values. The loader still range-checks rather than trusting that.
__device__ __forceinline__ float q27_e8m0(unsigned b) {
    return __uint_as_float(b << 23);
}
__device__ __forceinline__ int q27_d4(unsigned w, unsigned x, int a) {
    return __builtin_amdgcn_sdot4((int)w, (int)x, a, false);
}
// Q27_GU_INT4: sign-extend four packed two's-complement int4 nibbles (low half of each byte)
// to four int8 lanes. SAME 3-perm+bfi structure as q27_dq4 but LINEAR tables (weights are
// already requantized to int4; no E2M1 magnitude decode). Carry-safe bfi form kept.
#define Q27_I4_POS_HI 0x07060504u   // mag 4..7 -> 4,5,6,7
#define Q27_I4_POS_LO 0x03020100u   // mag 0..3 -> 0,1,2,3
#define Q27_I4_NEG_HI 0xFCFDFEFFu   // mag 4..7 -> -4,-3,-2,-1
#define Q27_I4_NEG_LO 0xF8F9FAFBu   // mag 0..3 -> -8,-7,-6,-5
__device__ __forceinline__ unsigned q27_dq4_i4(unsigned d) {
    const unsigned mag = d & 0x07070707u, sel = (d >> 3) & 0x01010101u;
    const unsigned pos = __builtin_amdgcn_perm(Q27_I4_POS_HI, Q27_I4_POS_LO, mag);
    const unsigned neg = __builtin_amdgcn_perm(Q27_I4_NEG_HI, Q27_I4_NEG_LO, mag);
    const unsigned msk = __builtin_amdgcn_perm(0u, Q27_SGN_TBL, sel);
    return pos ^ ((pos ^ neg) & msk);
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

// ---------------- FP16 binary16 conversions (2026-09-16, KV scale ABI) ----------------
// House style, matching q27_f2bf: bit arithmetic only, no <hip/hip_fp16.h>, so the host syntax
// stub still parses these. q27_f2h rounds to nearest, ties to even -- the bound the KV scale ABI
// is priced against is 2^-11 = 0.049% worst-case relative error in the normal range.
__device__ __forceinline__ float q27_h2f(unsigned short h) {
    const unsigned s = (unsigned)h & 0x8000u;
    const unsigned e = (unsigned)(h >> 10) & 0x1Fu;
    unsigned m = (unsigned)h & 0x3FFu;
    if (e == 0) {
        if (m == 0) return __uint_as_float(s << 16);                    // +-0
        int sh = 0; while (!(m & 0x400u)) { m <<= 1; ++sh; }            // subnormal -> normal
        return __uint_as_float((s << 16) | ((unsigned)(113 - sh) << 23) | ((m & 0x3FFu) << 13));
    }
    if (e == 31) return __uint_as_float((s << 16) | 0x7F800000u | (m << 13));   // inf / nan
    return __uint_as_float((s << 16) | ((e + 112u) << 23) | (m << 13));
}
__device__ __forceinline__ unsigned short q27_f2h(float f) {            // round-to-nearest-even
    const unsigned x = __float_as_uint(f);
    const unsigned s = (x >> 16) & 0x8000u;
    const unsigned bem = (x >> 23) & 0xFFu;
    const unsigned m = x & 0x7FFFFFu;
    const int      e = (int)bem - 112;                                  // fp16 biased exponent
    if (bem == 0xFFu) return (unsigned short)(s | 0x7C00u | (m ? 0x200u : 0u));
    if (e >= 0x1F) return (unsigned short)(s | 0x7C00u);                // overflow -> inf
    if (e <= 0) {                                                       // subnormal or underflow
        if (e < -10) return (unsigned short)s;
        const unsigned mm = m | 0x800000u;
        const int      sh = 14 - e;
        unsigned short r  = (unsigned short)(mm >> sh);
        const unsigned rem = mm & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (unsigned short)(s | r);
    }
    unsigned short r  = (unsigned short)(((unsigned)e << 10) | (m >> 13));
    const unsigned rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;             // may carry into the exponent
    return (unsigned short)(s | r);
}

// ---------------- KV SCALE ABI (2026-09-16): FP16-OF-MAX ----------------
// The per-16-group KV scale used to be stored as fp32 of sc = mx/127: 256 B per position per card
// per plane, 20% of the quarter row. It is now FP16 of the group MAX mx itself.
//   writer : keeps computing the same fp32 sc = mx * (1.0f/127.0f) it always did and quantizes the
//            cached int8 with it -- so the int8 K/V bytes in the cache are UNCHANGED -- and stores
//            mx, which sits far inside fp16's normal range for this model's post-norm K and raw V.
//   reader : sc' = fp32(mx) * (1.0f/127.0f), the same fp32 expression, so the multiplier differs
//            from the writer's only by fp16's own rounding of mx (<= 0.049% relative).
// Half the scale-plane bytes and half the scale-plane read bandwidth, with no new quantizer and no
// change to the int8 bytes. e4m3 was rejected here: its 6.25% scale error would dominate the int8
// step's own 0.39% and multiply the cache's error budget by ~16x.
typedef unsigned short q27_kvs_t;
#define Q27_KVS_INV127 (1.0f / 127.0f)
__device__ __forceinline__ float q27_kvs_ld(q27_kvs_t v) { return q27_h2f(v) * Q27_KVS_INV127; }
__device__ __forceinline__ q27_kvs_t q27_kvs_st(float mx) { return q27_f2h(mx); }

// ---------------- V4: Hadamard-rotated int4 V (2026-09-16; Q27_V4, default OFF) ----------------
// A COMPILE-TIME switch, not a runtime flag: the V layout (uint32 bitplanes, half the row bytes,
// block-32 fp16 scales) differs from the int8 one, so writer and readers must be built together.
// Q27_V4=0 must leave the int8-V path byte-identical; build the A/B arm with `make V4=1`.
#ifndef Q27_V4
#define Q27_V4 0
#endif
// Mined from exllamav3's quant cache -- https://github.com/turboderp-org/exllamav3,
// exllamav3_ext/cache/q_cache_kernels.cuh. See V4_KV_MINING_20260916.md for the receipts: the
// rotation cuts V's relative L2 from 0.1786 (plain 4-bit, outlier-bearing blocks) to 0.0761, and
// is a no-op on clean Gaussian blocks -- i.e. its whole benefit is outlier suppression.
//
// Layout, per V row of `stride` dims (a 1024-dim row = 32 blocks):
//   vc4[pos][block][0..3]  uint32 bitplanes   -- (stride/32)*4 words = stride/2 BYTES per position
//   vs32[pos][block]       fp16 block max     -- (stride/32) halves  = stride/64 bytes
// Block b holds dims [32b, 32b+32). Bit index within a block is `dim % 32`, on BOTH sides.
//
// THE TWO SIDES HAVE DIFFERENT LANE OWNERSHIP, SO THEIR BUTTERFLIES DIFFER:
//   writer: lane t owns dim t inside each 64-dim slot -> all 5 stages are __shfl_xor (1,2,4,8,16)
//           and the 64-bit ballot carries BOTH halves (bits 0-31 = block 2i, 32-63 = block 2i+1).
//   reader: lane L owns 8 CONSECUTIVE dims [8L, 8L+8) -> block = L>>2, and inside the block the
//           index is 8*(L&3)+j, so the low 3 stages are IN-REGISTER and only the top 2 cross lanes.
#define Q27_VBLK     32
#define Q27_V4_SHIFT  3                     // 8 = 2^(bits-1): quantizer bias / reconstruction base
#define Q27_HAD32     0.17677669529663688110f   // 1/sqrt(32);  H*H = 32*I and K*K = 1/32

// HOST-SIDE PLANE OFFSETS -- the fix for the 2026-09-16 V4=1 first-job page fault.
// kv_off[] / kvs_off[] are built in FULL-WIDTH units because they are the K plane's offsets, and K
// is unchanged in both arms. Under V4 the V planes are allocated at HALF that size (kc_el/2 bytes,
// ks_fl/2 maxes), so every V base derived from those tables must be halved too. Eight call sites
// did not: the allocation was right, the layer stride inside it was not, so the first full-attention
// layer whose offset exceeded half the K plane addressed past the end of vc/vs. That is the
// `T6a_jobs_begin -> Memory access fault` on a 130-token prompt. Identity in the V4=0 arm.
#if Q27_V4
#define Q27_VOFF(x)  ((x) / 2)
#define Q27_VSOFF(x) ((x) / 2)
#else
#define Q27_VOFF(x)  (x)
#define Q27_VSOFF(x) (x)
#endif

// Draft-KV cost per position, for the affordability gate. V4 halves vc and quarters vs, so the
// int8 arm's 2,560 B/position over-prices the V4 draft by ~48% -- and over-pricing is exactly how
// speculation gets refused at depth (measured twice on 2026-09-15). Each is the true cost plus the
// same 256 B slack the int8 number has carried since the fp16-scale landing.
//   int8:  kc 1024 + ks 128 + vc 1024 + vs 128 = 2304   (+256)
//   V4:    kc 1024 + ks 128 + vc  512 + vs  64 = 1728   (+256)
#if Q27_V4
#define Q27_MTP_BPP 1984ull
#else
#define Q27_MTP_BPP 2560ull
#endif

// THE V4 BLOCK SCALE IS NOT AN INT8 CODE SCALE. q27_kvs_ld's 1/127 exists only because the int8
// path stores codes in [-127,127]. V4 stores 4-bit codes reconstructed as (q-8)/8, and the writer
// stores the RAW fp16 block max of the UNNORMALIZED transform x = H*v (H*H = 32*I). Reconstruction:
//     vhat_n = (q_n - 8) * (Q27_HAD32/8) * s = x_n * s / (sv * sqrt(32))
//     out    = H(vhat) = 32 * v * s / (sv * sqrt(32)) = sqrt(32) * (s/sv) * v
// so out == v requires s = sv / sqrt(32) = sv * Q27_HAD32. Loading the block max with q27_kvs_ld
// yields s = sv/127 and therefore out = (sqrt(32)/127) * v = 0.0445 * v -- a 22x UNIFORM collapse
// of the attention branch. Measured 2026-09-16: the engine boots, allocates, primes, serves a
// 12,386-token Harness turn with zero faults, and emits token salad. The offline round-trip check
// in V4_KV_MINING_20260916.md validated q27_v4_unpack32 GIVEN a correct sc; nothing validated how
// sc is produced from the stored half. That seam is where this lived.
__device__ __forceinline__ float q27_v4s_ld(q27_kvs_t v) { return q27_h2f(v) * Q27_HAD32; }

// WHAT THE V4 SAVING IS WORTH IN WINDOW.
// The window-aware MLP residency policy (q27_main.cpp, the "Q27_LS_MLP_I8K auto" block) is OPEN
// LOOP: it sheds one int8/Tensile MLP layer per `per` positions of window above a base of 9,216,
// and never reads free memory -- it calls hipMemGetInfo only to PRINT it. So a KV-layout change
// that hands every card ~0.985 GiB cannot reach it. Measured 2026-09-16: at a 12.4K window V4=1
// and V4=0 both printed "14 of 16 owned layers", with V4=1 sitting on 2.15 GiB free per card
// against V4=0's 1.16 -- a full gigabyte of recovered residency spent on nothing.
// Prefill slots cost 30,720 B per position per card, so 0.985 GiB is ~34,400 positions. Spend
// 32,768 of them and keep the remainder as the same working room the 9,216 base was tuned against.
// MEASURED 2026-09-16, and the answer was the opposite of the intent. Spending the V4 saving on
// window (bonus 32768) moved the 12.6K turn from "14 of 16" to "16 of 16" owned layers on
// int8/Tensile. Through the real Harness, same prompt, same session shape:
//     k=14   prefill 838 tok/s   warm decode 58.64 tok/s
//     k=16   prefill 936 tok/s   warm decode 51.67 tok/s
// The int8 mirror is 8 bits per weight; the NVFP4 authority it displaces is 4. Prefill is
// compute-bound and wants the Tensile GEMM; DECODE at M=1 is byte-bound and wants the SMALLER
// weight. So more int8 mirrors is a prefill/decode TRADE, not a win, and pointing the recovered
// gigabyte at it spends decode to buy prefill -- the wrong direction for this campaign. Default 0:
// the freed memory stays free. The knob remains so the trade can be taken deliberately.
#if Q27_V4
#define Q27_V4_WIN_BONUS 0        // MEASURED 0 IS RIGHT -- see below. Override at runtime with the env var of the same name.
#else
#define Q27_V4_WIN_BONUS 0
#endif

// V4 CREDIT AGAINST THE MIRROR BUDGET, IN LAYERS.
// With Q27_DEC_MLP_NV the decode reads the NVFP4 shard, so the shard is RETAINED and one int8
// mirror costs its full ~117 MB net instead of ~81 MB -- which is why the base drops 16 -> 13.
// That base is memory-calibrated, so it is exactly the budget V4's saving should be credited
// against: 0.985 GiB/card / 117 MB = ~8.6 mirrors of headroom. Take 3 (enough to restore the full
// 16 and no more), which is a credit against measured residency rather than a lowered safety
// margin -- the clamp below still caps k at Q27_LAYERS/ndev. Runtime-tunable for A/B.
#if Q27_V4
#define Q27_V4_KBONUS 3
#else
#define Q27_V4_KBONUS 0
#endif

// ---- writer side: all five stages cross lanes ----
__device__ __forceinline__ float q27_had32w(float v, const int lane) {
    #pragma unroll
    for (int i = 1; i < Q27_VBLK; i <<= 1) {
        const float pv = __shfl_xor(v, i, 64);
        unsigned u = __float_as_uint(v);
        if (lane & i) u ^= 0x80000000u;
        v = __uint_as_float(u) + pv;
    }
    return v;
}
// block max over this half-wave (lane 0-31 or 32-63); shfl_xor with i<32 never crosses the half
__device__ __forceinline__ float q27_v4_amax32(float a) {
    #pragma unroll
    for (int i = 1; i < Q27_VBLK; i <<= 1) a = fmaxf(a, __shfl_xor(a, i, 64));
    return a;
}
// biased 4-bit code of x with block max s: q = clamp(rint(x*8/s)+8, 0, 15)
__device__ __forceinline__ unsigned q27_v4_q4(const float x, const float s) {
    const float inv = (s > 0.0f) ? ((float)(1 << Q27_V4_SHIFT) / s) : 0.0f;
    const int q = (int)rintf(x * inv) + (1 << Q27_V4_SHIFT);
    return (unsigned)(q < 0 ? 0 : (q > 15 ? 15 : q));
}

// ---- reader side: 3 in-register stages, then 2 cross-lane stages ----
__device__ __forceinline__ void q27_had32r(float v[8], const int lane) {
    #pragma unroll
    for (int i = 1; i < 8; i <<= 1) {                      // stages for index bits 0,1,2
        float nv[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p = v[j ^ i];
            nv[j] = ((j & i) ? -v[j] : v[j]) + p;
        }
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = nv[j];
    }
    #pragma unroll
    for (int m = 1; m < 4; m <<= 1) {                      // stages for index bits 3,4 = lane bits 0,1
        float p[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[j] = __shfl_xor(v[j], m, 64);
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = ((lane & m) ? -v[j] : v[j]) + p[j];
    }
}
// ---- NIBBLE LAYOUT (replaced bitplanes 2026-09-17; see DECODE_READS_NVFP4_20260917.md §6) ----
// Bitplane packing is an artefact of the CUDA reference: `__ballot` pairs with cheap NVIDIA warp
// shuffles. On gfx906 it cost twice over. (1) ALU: assembling one 4-bit code from four plane words
// is 4 shifts + 4 ands + 3 ors = 11 ops, so 88 ops per lane per key row, against a nibble's
// 1 shift + 1 and = 16. (2) TRAFFIC: a lane needs all four plane words to recover ANY of its
// values, so all four lanes of a block load the same 16 B -- 512 B requested per 256-dim row for
// 128 B of distinct data, against int8's 256 B for 256 B.
// Nibbles fix both. Block b = 32 values = 16 B; nibble n lives at byte n>>1, low half when n is
// even. Lane L owns dims [8L, 8L+8), i.e. block L>>2 and nibbles [8*(L&3), +8) -- which is exactly
// BYTES [4*(L&3), +4): ONE aligned uint32 per lane, and the four lanes of a block read four
// DISJOINT, contiguous words. Same bytes on disk, same Hadamard, same block-32 fp16 scale.
// Reconstruct this lane's 8 dims from its one packed word + the block's max.
__device__ __forceinline__ void q27_v4_recon(const unsigned w, const float s, float v[8]) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned q = (w >> (4 * j)) & 0xFu;
        // (q - 8)/8 * (1/sqrt(32)) * s  ==  H*v/32, which the inverse transform maps back to v
        v[j] = ((float)q - (float)(1 << Q27_V4_SHIFT)) *
               (Q27_HAD32 / (float)(1 << Q27_V4_SHIFT)) * s;
    }
}

// THE INVERSE ROTATION DOES NOT BELONG IN THE PER-ROW LOOP.
// H is linear and the softmax weight p_r is a per-row SCALAR, so for any set of key rows
//     sum_r p_r * H(recon_r)  ==  H( sum_r p_r * recon_r ).
// The per-row block max s_r folds into recon_r (also a scalar), the online-softmax rescale
// acc *= c is a scalar multiply, and the half-wave merge is a weighted sum -- every operation
// between the reconstruct and the epilogue commutes with H. So a decode kernel reconstructs
// WITHOUT rotating, accumulates, and rotates the ACCUMULATOR once at the end.
__device__ __forceinline__ void q27_v4_unpack(const unsigned w, const float s,
                                              const int lane, float v[8]) {
    q27_v4_recon(w, s, v);
    q27_had32r(v, lane);
}
// One row of V held across the software-pipelined loop: ONE packed word + the block scale, which is
// FEWER registers than the int8 path's uint2 + fp32, so the pipeline shape is preserved.
typedef struct { unsigned w; q27_kvs_t s; } q27_vhold_t;

// `rowoff` is row_index * kvstride (dim units) and `hoff` is hoff0 + head*Q27_HDIM + d0. The lane's
// own 8 dims start at dim (rowoff + hoff), and the packed plane is half a byte per dim, so its word
// is at byte (rowoff + hoff)/2 -- which is 4-byte aligned because d0 is a multiple of 8.
__device__ __forceinline__ q27_vhold_t q27_v4_fetch(const signed char* __restrict__ vc,
                                                    const q27_kvs_t* __restrict__ vs,
                                                    const size_t rowoff, const int hoff) {
    q27_vhold_t h;
    h.w = *(const unsigned*)(vc + ((rowoff + (size_t)hoff) >> 1));
    h.s = vs[(rowoff >> 5) + (size_t)(hoff >> 5)];
    return h;
}
// `dg` is the lane's 32-lane dim group (d0 = dg*Q27_VEC), which is also the block-relative lane.
__device__ __forceinline__ void q27_v4_hold_unpack(const q27_vhold_t h, const int dg, float v[8]) {
    q27_v4_unpack(h.w, q27_v4s_ld(h.s), dg, v);
}
// Reconstruct only -- the caller rotates the accumulator once at its epilogue.
__device__ __forceinline__ void q27_v4_hold_recon(const q27_vhold_t h, const int dg, float v[8]) {
    (void)dg; q27_v4_recon(h.w, q27_v4s_ld(h.s), v);
}

// Fully in-register 32-value butterfly, for the kernels where one lane owns a whole block
// (q27_attn_tile_q8 / tile32_q8: LPR=8, DPT=32, so lane `lr` owns dims [32*lr, 32*lr+32)). No
// shuffles at all -- every stage is local. Same transform as q27_had32w/q27_had32r.
__device__ __forceinline__ void q27_had32_ir(float v[32]) {
    #pragma unroll
    for (int i = 1; i < Q27_VBLK; i <<= 1) {
        float nv[32];
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float p = v[j ^ i];
            nv[j] = ((j & i) ? -v[j] : v[j]) + p;
        }
        #pragma unroll
        for (int j = 0; j < 32; ++j) v[j] = nv[j];
    }
}
// Reconstruct one whole block (32 values) held by a single lane, scaled by the block max, rotated
// back. `p` may be folded in by the caller (the transform is linear and p is a scalar).
__device__ __forceinline__ void q27_v4_unpack32(const unsigned* __restrict__ w4, const float sc,
                                                float v[32]) {
    #pragma unroll
    for (int i = 0; i < Q27_VBLK; ++i) {
        const unsigned q = (w4[i >> 3] >> (4 * (i & 7))) & 0xFu;   // nibble layout: word i>>3, nibble i&7
        v[i] = ((float)q - (float)(1 << Q27_V4_SHIFT)) *
               (Q27_HAD32 / (float)(1 << Q27_V4_SHIFT)) * sc;
    }
    q27_had32_ir(v);
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
    // Q27_E4M3_FAST eligibility for THIS tensor's gs plane: 1 only after q27_nvfp4_gs_validate()
    // has seen every byte and found 8 <= b <= 126, i.e. neither the subnormal (b<8) nor the
    // NaN/Inf (b==127) arm of q27_e4m3 is reachable here. DEFAULTS TO 0 AND MUST: a plane that
    // was never validated -- including one manufactured at runtime by q27_f32_to_e4m3, which has
    // a live subnormal branch and returns 0 for an all-zero group -- falls back to the full
    // decoder. Fail closed; never infer eligibility from the checkpoint's provenance.
    unsigned char gs_fast = 0;
    // OCP MXFP4 (AMD Quark) instead of NVFP4: same E2M1 nibbles and the SAME [rows][K/2] K-contiguous
    // weight bytes -- only the scale plane differs. gs is then [rows][K/32] of E8M0 (one byte per 32
    // weights, exponent-only) rather than [rows][K/16] of E4M3, and there is no weight_scale_2, so
    // ws2 must be 1.0f. Set by the loader when it side-loads a Quark MXFP4 plane; 0 = our NVFP4.
    unsigned char gs_mx = 0;
} q27_nvfp4_t;                    // (the optional int8 mirror of a tensor lives in a side table: q27_nvfp4_w8(); the
                                  //  struct keeps its release layout - an extra pointer here cost 3% of decode, receipt v93)

typedef struct {                  // INT8 per-64-group execution mirror (Q27_LS_Q8): rows x K int8, natural K order
    // DEFAULTS ARE LOAD-BEARING. Every field carries one because these descriptors are built in
    // several places (conversion kernels, concatenating builders, view registration) and the
    // consumer kernels read ldw/lds unconditionally: an aggregate-initialised-nowhere stack local
    // used to leave the two stride fields as stack garbage, which the int8 row-batched kernels
    // multiply by the row index. That is a wild row stride -> GPU page fault, and it is how
    // Q27_ALIAS_QKV's row views (a shipped setting) broke the moment the consumers learned to
    // honour a stride. Never construct one of these without these two fields being 0 or a view.
    const signed char* w = nullptr;   // [rows][K]
    const float* s = nullptr;         // [rows][K/64] fp32 group scales
    float alpha = 0.0f;               // in_scale * weight_scale_2 (applied once per output)
    int rows = 0, K = 0;
    // K-VIEW STRIDES. 0 means "dense at this tensor's own width", which is what every existing
    // construction produces (they aggregate-initialise, so the fields are zero). A non-zero ldw
    // says the rows are a COLUMN RANGE of a wider resident row: the K columns start at the
    // caller's pointer offset and the physical row stride is ldw elements instead of K, with lds
    // (elements, 0 = K/64) the matching stride of the per-64 scale plane. This is how a TP decode
    // shard addresses the layer-split residency in place -- same bytes, same quantisation groups,
    // no materialised copy -- for the tensors the loader shards along K (Tp::fp8_row/nvfp4_row).
    int ldw = 0, lds = 0;
} q27_i8g_t;

typedef struct {                  // INT8 per-ROW execution mirror for the rocBLAS int8 GEMMs (Q27_LS_Q8 stage 3)
    const signed char* w;         // [rows][K] natural K order (the per-64 mirror requantized in place)
    const float* s;               // [rows][ng] fp32 scales per (row, 1024-group) (= max of the 16 per-64 scales)
    float alpha;                  // as q27_i8g_t
    int ng;                       // number of K groups (K / gs)
    int gs;                       // group size: 1024, or K (per-row / per-token scales)
    int rows, K;
} q27_i8r_t;

// Q27_PF_P2P: the four KV head-quarters of ONE layer, one per card. Under layer-split the owner
// used to keep a second, full-width copy of all four so its prefill attention could read them
// locally; every quarter already exists on the card that owns that head. With peer access on, the
// sweep reads the three remote quarters straight out of the peers and every card stores exactly the
// quarter TP decode reads. `on`=0 restores the single-base full-width addressing exactly.
typedef struct {
    const signed char* kc[4]; const q27_kvs_t* ks[4];
    const signed char* vc[4]; const q27_kvs_t* vs[4];
    int on;
} q27_kvquad_t;

typedef struct {                  // FP8 projection: rows x K
    const unsigned char* w;       // [rows][K] e4m3
    float wscale, in_scale;
    int rows, K;
    // BLOCK-SCALED FP8 (Qwen3.8-27B-FP8 / DeepSeek convention). When bs != nullptr the single
    // per-tensor wscale does not apply: the weight scale is a [rows/128][K/128] BF16 plane
    // (`weight_scale_inv`), so it varies along K WITHIN a row and the consumer must fold it per
    // 128-wide K block. Each wave owns one row, so the N-block index (row>>7) is wave-invariant
    // and only the K-block moves. bs_kblk is that plane's row stride, i.e. K_full/128.
    const unsigned short* bs = nullptr;
    int bs_kblk = 0;
} q27_fp8_t;


// ---------------------------------------------------------------------------------------------
// ENV FLAGS WITH VALUE SEMANTICS. Four guards in this engine used `getenv("X") != nullptr`, i.e.
// PRESENCE semantics, so `X=0` ENABLED them -- the opposite of how every one of them reads, and
// directly contradicting q27_nvfp4.hip's own comment that "W32=1 restores the old geometry".
// No measurement taken on 2026-09-09 is affected (every arm used =1 or unset, never =0), but an
// A/B run with =0 expecting the default would silently have measured the wrong arm.
// FAILS CLOSED on an unparseable value rather than guessing, because a typo in a benchmark flag
// should stop the run, not quietly select an arm.
// ---------------------------------------------------------------------------------------------
// Q27X_* -- THE NATIVE OVERRIDE NAMESPACE (2026-09-17). THE ENGINE HAS FINAL AUTHORITY.
//
// The serving front end is FROZEN infrastructure that hardcodes several Q27_* values into this
// process's environment (Q27_SPEC_K="2", Q27_NR_P2P_MIN="4", Q27_COLL_NRBF16="1", ...) rather than
// forwarding the shell's. A shell therefore cannot sweep them, and on 2026-09-17 that produced a
// PHANTOM ARM: a "Q27_NR_P2P_MIN=3" result was reported, written up, and nearly built on, when the
// front end had pinned the value at 4 the whole time and the arm never existed. The fix is not to
// edit the front end -- it is for the engine to own its own configuration:
//
//     resolution order:   Q27X_<NAME>   ->   Q27_<NAME>   ->   compiled default
//
// Q27X_* is not in the front end's dictionary, so it passes through from the shell untouched. This
// is generic in the two env helpers, so EVERY Q27_* knob in the engine is sweepable by construction;
// no per-flag override plumbing exists to fall out of date.
//
// Every override that fires is recorded for the boot receipt, because a REQUESTED value is not
// evidence of an EFFECTIVE one -- that distinction is the whole reason this namespace exists.
extern char q27_ovr_log[768];
extern int  q27_ovr_n;
static inline const char* q27_getenv_ovr(const char* name) {
    if (name[0]=='Q' && name[1]=='2' && name[2]=='7' && name[3]=='_') {
        char buf[128];
        const int n = snprintf(buf, sizeof buf, "Q27X_%s", name + 4);
        if (n > 0 && n < (int)sizeof buf) {
            const char* x = getenv(buf);
            if (x && *x) {
                if (!strstr(q27_ovr_log, buf)) {            // record once, not once per read
                    const char* base = getenv(name);
                    const size_t used = strlen(q27_ovr_log);
                    if (used + 96 < sizeof q27_ovr_log) {
                        snprintf(q27_ovr_log + used, sizeof(q27_ovr_log) - used, "%s%s=%s(fe:%s)",
                                 used ? "," : "", buf, x, (base && *base) ? base : "unset");
                        ++q27_ovr_n;
                    }
                }
                return x;
            }
        }
    }
    return getenv(name);
}
static inline bool q27_env_flag(const char* name, bool def) {
    const char* v = q27_getenv_ovr(name);
    if (!v || !*v) return def;
    if (!strcmp(v,"0") || !strcmp(v,"false") || !strcmp(v,"no")  || !strcmp(v,"off")) return false;
    if (!strcmp(v,"1") || !strcmp(v,"true")  || !strcmp(v,"yes") || !strcmp(v,"on"))  return true;
    fprintf(stderr, "FATAL %s=\"%s\" is not a boolean (use 0/1, false/true, no/yes, off/on). "
                    "Refusing to guess which arm you meant.\n", name, v);
    abort();
}
static inline int q27_env_int(const char* name, int def) {
    const char* v = q27_getenv_ovr(name);
    if (!v || !*v) return def;
    char* end = nullptr; const long x = strtol(v, &end, 10);
    if (end == v || *end) {
        fprintf(stderr, "FATAL %s=\"%s\" is not an integer.\n", name, v); abort();
    }
    return (int)x;
}

static inline void q27_mlp_wshare(int* off, int* len, int ndev);
// ---------------- Q27_TP3: ragged tensor-parallel shard plan (2026-09-28) ----------------
// One table describes what each TP worker owns. For ndev that divides every dimension it is the
// old uniform split (bit-identical geometry); for ndev == 3 it is the ragged plan that keeps
// whole KV groups and whole GDN key heads on one card:
//   attention q heads 12/6/6 (KV heads {0,1}/{2}/{3}; 24/q_hn selects the existing 2- and 4-way
//   attention kernel instantiations), GDN key heads 4/6/6 (value heads 12/18/18), MLP rows from
//   Q27_MLP_WSHARE (default 5888/5760/5760), lm_head rows 82944/82688/82688.
// Env overrides for ndev == 3 (comma lists): Q27_TP3_Q (q heads, multiples of 6), Q27_TP3_GK
// (GDN key heads), Q27_TP3_VOC (lm_head rows, multiples of 64).
#ifndef Q27_TP_MAXDEV
#define Q27_TP_MAXDEV 16   /* == q27_load.h Q27_MAX_DEVICES, which is not visible here */
#endif
typedef struct q27_tp_shard_s {
    int q_h0, q_hn;        // attention query heads   (of Q27_NHEAD)
    int kv_h0, kv_hn;      // KV heads                (of Q27_NKV), whole GQA groups
    int attn_g;            // Q27_NHEAD / q_hn: attention kernel geometry (1, 2 or 4)
    int gk_h0, gk_hn;      // GDN key heads           (of Q27_GDN_KH)
    int gv_h0, gv_hn;      // GDN value heads = 3 x key heads
    int mlp_off, mlp_len;  // gate/up rows == down K slice
    int voc_off, voc_len;  // lm_head rows
    int QL, KVL, OL, QKVL, ZL, VHL, QS, MIXL;   // derived row counts the decode paths use
} q27_tp_shard_t;
static inline int q27_tp_parse_list(const char* v, int* out, int n) {
    if (!v || !*v) return 0;
    int k = 0; const char* p = v;
    while (*p && k < n) { char* e = 0; long x = strtol(p, &e, 10); if (e == p) return 0; out[k++] = (int)x; p = e; while (*p == ',' || *p == ' ') ++p; }
    return (k == n && !*p) ? 1 : 0;
}
static inline int q27_tp_plan(int ndev, q27_tp_shard_t* S) {
    if (ndev < 1 || ndev > Q27_TP_MAXDEV) return 0;
    int q[Q27_TP_MAXDEV], gk[Q27_TP_MAXDEV], vo[Q27_TP_MAXDEV], mo[Q27_TP_MAXDEV], ml[Q27_TP_MAXDEV];
    if (ndev == 3) {
        static const int dq[3] = {12, 6, 6}, dgk[3] = {4, 6, 6}, dvo[3] = {82944, 82688, 82688};
        for (int g = 0; g < 3; ++g) { q[g] = dq[g]; gk[g] = dgk[g]; vo[g] = dvo[g]; }
        q27_tp_parse_list(q27_getenv_ovr("Q27_TP3_Q"), q, 3);
        q27_tp_parse_list(q27_getenv_ovr("Q27_TP3_GK"), gk, 3);
        q27_tp_parse_list(q27_getenv_ovr("Q27_TP3_VOC"), vo, 3);
    } else {
        if ((Q27_NHEAD % ndev) || (Q27_NKV % ndev) || (Q27_GDN_KH % ndev) || (Q27_VOCAB % ndev)) return 0;
        for (int g = 0; g < ndev; ++g) { q[g] = Q27_NHEAD / ndev; gk[g] = Q27_GDN_KH / ndev; vo[g] = Q27_VOCAB / ndev; }
    }
    if (ndev > 1) q27_mlp_wshare(mo, ml, ndev); else { mo[0] = 0; ml[0] = Q27_INTER; }   // ndev 1 = full width (the layer-split plan)
    int sq = 0, sgk = 0, svo = 0, sml = 0;
    for (int g = 0; g < ndev; ++g) { sq += q[g]; sgk += gk[g]; svo += vo[g]; sml += ml[g]; }
    if (sq != Q27_NHEAD || sgk != Q27_GDN_KH || svo != Q27_VOCAB || sml != Q27_INTER) return 0;
    int q0 = 0, gk0 = 0, vo0 = 0;
    for (int g = 0; g < ndev; ++g) {
        q27_tp_shard_t& T = S[g];
        const int grp = Q27_NHEAD / Q27_NKV;                 // 6 query heads per KV head
        if (q[g] <= 0 || (q[g] % grp) || (q0 % grp) || (Q27_NHEAD % q[g])) return 0;
        if (gk[g] <= 0 || vo[g] <= 0 || (vo[g] % 64)) return 0;
        T.q_h0 = q0; T.q_hn = q[g]; T.kv_h0 = q0 / grp; T.kv_hn = q[g] / grp; T.attn_g = Q27_NHEAD / q[g];
        if (T.attn_g != 1 && T.attn_g != 2 && T.attn_g != 4) return 0;
        T.gk_h0 = gk0; T.gk_hn = gk[g]; T.gv_h0 = 3 * gk0; T.gv_hn = 3 * gk[g];
        T.mlp_off = mo[g]; T.mlp_len = ml[g]; T.voc_off = vo0; T.voc_len = vo[g];
        T.QL = (Q27_QROWS / Q27_NHEAD) * T.q_hn;             // 512 rows per head (query + output gate)
        T.KVL = Q27_HDIM * T.kv_hn;
        T.OL = Q27_HDIM * T.q_hn;                            // o_proj K slice
        T.QKVL = 2 * Q27_GDN_D * T.gk_hn + Q27_GDN_D * T.gv_hn;   // q | k | v rows of in_proj_qkv
        T.ZL = Q27_GDN_D * T.gv_hn;                          // in_proj_z rows == out_proj K slice
        T.VHL = T.gv_hn;
        T.QS = T.QL + 2 * T.KVL;
        T.MIXL = (T.ZL > T.OL) ? T.ZL : T.OL;
        q0 += q[g]; gk0 += gk[g]; vo0 += vo[g];
    }
    return 1;
}
static q27_tp_shard_t q27_tp_tab[Q27_TP_MAXDEV];
static int q27_tp_tab_n = 0;
static inline int q27_tp_shard_init(int ndev) {
    if (q27_tp_tab_n == ndev) return 1;
    if (!q27_tp_plan(ndev, q27_tp_tab)) return 0;
    q27_tp_tab_n = ndev; return 1;
}
static inline const q27_tp_shard_t* q27_tp_shard(int g) {
    if (q27_tp_tab_n <= 0 || g < 0 || g >= q27_tp_tab_n) { fprintf(stderr, "FATAL q27_tp_shard(%d): table has %d workers\n", g, q27_tp_tab_n); abort(); }
    return &q27_tp_tab[g];
}
static inline int q27_tp_voc_max(void) { int m = 0; for (int g = 0; g < q27_tp_tab_n; ++g) if (q27_tp_tab[g].voc_len > m) m = q27_tp_tab[g].voc_len; return m ? m : Q27_VOCAB; }
// Layer-split block size. Q27_LS_OWN quotas (layers per card, multiples of Q27_LS_BLK, summing to
// Q27_LAYERS) are allowed even when Q27_LS_BLK does not divide LAYERS/ndev (ndev == 3: 20,20,24 at
// blk 4 = 5/5/6 blocks). Without valid quotas the old rule applies (blk must divide LAYERS/ndev).
static inline int q27_ls_blk_eff(int ndev) {
    int blk = q27_env_int("Q27_LS_BLK", 2);
    const int LPP = (ndev > 0) ? Q27_LAYERS / ndev : 0;
    if (blk >= 1 && blk <= Q27_LAYERS) {
        const char* v = q27_getenv_ovr("Q27_LS_OWN");
        int qv[Q27_TP_MAXDEV];
        if (v && *v && q27_tp_parse_list(v, qv, ndev)) {
            int sum = 0, ok = 1;
            for (int g = 0; g < ndev; ++g) { if (qv[g] <= 0 || (qv[g] % blk)) ok = 0; sum += qv[g]; }
            if (ok && sum == Q27_LAYERS) return blk;
        }
    }
    if (LPP <= 0 || blk < 1 || blk > LPP || (LPP % blk)) blk = (LPP > 0) ? LPP : 1;
    return blk;
}
// ---------------- Q27_MLP_WSHARE: per-card MLP row share (2026-09-18) ----------------
// The three MLP matrices (gate/up rows + down K) are 66 % of the 8-bit weight bytes, so the only
// way the 32 GiB card's spare residency becomes *authority* for the 16 GiB cards is to let card 0
// own a larger slice of the MLP. `Q27_MLP_WSHARE` is the gate/up ROW COUNT per card, comma
// separated, in device order; unset means the uniform Q27_INTER/ndev.
//
// Constraints, all enforced here rather than discovered as a corrupt shard:
//   * every entry > 0 and a multiple of 128 -- the resident FP8 plane is blocked
//     [rows/128][K/128] (q27_fp8_mlp_load), so a shard that is not a whole number of 128-row
//     blocks would read another card's scale block;
//   * the entries sum to Q27_INTER -- a short or long sum is a dropped or double-counted row of
//     the model, which is fluent and wrong;
//   * ndev entries exactly.
// down_proj's K-slice is this same vector (its K is Q27_INTER), so one vector defines both.
static inline void q27_mlp_wshare(int* off, int* len, int ndev) {
    if (ndev < 1) return;
    for (int g = 0; g < ndev; ++g) { len[g] = Q27_INTER / ndev; off[g] = g * (Q27_INTER / ndev); }
    const char* v = q27_getenv_ovr("Q27_MLP_WSHARE");
    if (!v || !*v) return;
    int got[32]; int n = 0;                 // literal cap: q27_load.h's Q27_MAX_DEVICES (16) is not
                                            // visible from here, and nothing above 32 is plausible
    const char* p = v;
    while (*p && n < 32) {
        char* e = nullptr;
        const long x = strtol(p, &e, 10);
        if (e == p) break;
        got[n++] = (int)x;
        p = e;
        while (*p == ' ' || *p == '\t' || *p == ',') ++p;   // skip separators
    }
    if (n != ndev) {
        fprintf(stderr, "FATAL Q27_MLP_WSHARE=\"%s\": expected %d comma-separated row counts, got %d.\n",
                v, ndev, n);
        abort();
    }
    long long sum = 0;
    for (int g = 0; g < ndev; ++g) {
        // 2026-09-28: 64-row granularity. The 128 rule protected the FP8 MLP plane's [rows/128][K/128]
        // blocking; the NVFP4 MLP (16-element groups, 64-row tiles) only needs 64, and the ragged down
        // consumer is instantiated for 3264/5440 (3/5 x 1024 + 192/320). Do not use non-128 shares
        // with Q27_FP8_MLP / Q27_LS_FP8_PREFILL.
        if (got[g] <= 0 || (got[g] % 64) != 0) {
            fprintf(stderr, "FATAL Q27_MLP_WSHARE=\"%s\": entry %d is %d, must be a positive multiple "
                            "of 128 (the FP8 plane is blocked [rows/128][K/128]).\n", v, g, got[g]);
            abort();
        }
        sum += got[g];
    }
    if (sum != Q27_INTER) {
        fprintf(stderr, "FATAL Q27_MLP_WSHARE=\"%s\": the %d entries sum to %lld, not Q27_INTER=%d. "
                        "A short sum drops model rows and a long sum double-counts them.\n",
                v, ndev, sum, Q27_INTER);
        abort();
    }
    int acc = 0;
    for (int g = 0; g < ndev; ++g) { off[g] = acc; len[g] = got[g]; acc += got[g]; }
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
                      const unsigned short* k_norm_w, signed char* kcache, q27_kvs_t* kscale,
                      signed char* vcache, q27_kvs_t* vscale, unsigned short* q_out, int pos, int ndev, int kvstride, int hoff0,
                      hipStream_t s);
// Returns 1 if the fused quantize epilogue ran; 0 means the caller must still launch
// q27_quant_fp8 (the single-split path skips stage 2).
int  q27_attn_decode_tp(const unsigned short* q, const signed char* kcache, const q27_kvs_t* kscale,
                        const signed char* vcache, const q27_kvs_t* vscale, const unsigned short* q_proj_raw,
                        unsigned short* out, int pos, int ndev, int kvstride, int hoff0,
                        signed char* qout, float* qscl, float in_scale, hipStream_t s,
                        int qost = 0);
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
int  q27_nvfp4_to_i8g64_into(const q27_nvfp4_t* src, q27_i8g_t* dst, signed char* w8, float* s64, hipStream_t st);   // same conversion into a caller-owned arena (layer-local staging)
int  q27_nvfp4_to_i8g64_conv(const q27_nvfp4_t* src, q27_i8g_t* dst, hipStream_t st);   // int8 per-64 mirror of an NVFP4 tensor (at load)
int  q27_swiglu_quant_b_g64(const float* gate, const float* up, signed char* xq, float* xs, int n, float in_scale, int C, hipStream_t s);
// Q27_LS_Q8 stage 3: rocBLAS int8 GEMM path (src/q27_rb.cpp, src/q27_epi.hip)
int  q27_rb_init(int dev);
int  q27_rb_gemm_i8g(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, int GS, hipStream_t s);   // C[g][M][N] = X[M][group g] * W[N][group g]^T
int  q27_rb_gemm_i8g_ld(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, int GS, int ldw, hipStream_t s);   // same, W rows are a column range of a wider resident tensor (ldw = physical row stride, 0 = K)
int  q27_rb_gemm_i8g_v(int dev, const q27_i8g_t* w, const signed char* X, int M, int* C, int GS, hipStream_t s);   // descriptor form: honours a K-view (ldw/lds)
int  q27_rb_tune_i8g(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, hipStream_t s);   // pick the fastest Tensile solution for (N,K,M) once; primes the shape
int  q27_i8g64_to_row_conv(q27_i8g_t* src, q27_i8r_t* dst, int gs, hipStream_t st);   // gs: 1024 or 0 = full K
int  q27_i8g64_rowify(q27_i8g_t* src, q27_i8r_t* dst, float* sgrp, hipStream_t st);  // in place, gs = K, caller-owned scales
int  q27_rmsnorm_hostss_tok_b(const float* x, const unsigned short* wgt, signed char* xq, float* xs, int n, int plus_one, float in_scale, const float* invp, int C, int gs, hipStream_t s);
int  q27_red_rn_tok_b1(const float* x, const unsigned short* wgt, float* hid32, signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, int gs, hipStream_t s);
int  q27_requant_g64_tok(const signed char* xq, const float* xs, signed char* yq, float* ys, int K, int xstride, int M, int gs, hipStream_t s);
int  q27_requant_g16_tok(const signed char* xq, const float* xs, signed char* yq, float* ys, int K, int xstride, int M, hipStream_t s);   // per-16 int8 -> per-token (1536-wide TP mixer; the g64 variant rejects K%1024)
int  q27_epi_q8(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha, signed char* yq, float* ys, int ystride, int M, hipStream_t s);
int  q27_epi_q8v2(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha, signed char* yq, float* ys, int ystride, int M, hipStream_t s);   // 4 columns per thread
int  q27_ab_to_i8r_conv(const unsigned short* wa, const unsigned short* wb, int rows, int K, q27_i8r_t* dst, hipStream_t st);   // GDN a/b bf16 -> int8 per-row [2*rows][K]
int  q27_epi_ab(const int* acc, int rows, const float* s, const float* stok, float xmul, float* ya, float* yb, int M, hipStream_t st);
int  q27_copy_f32(const float* src, float* dst, size_t n, hipStream_t s);   // fp32 copy kernel (peer-store handoff)
int  q27_bf16_to_f32(const unsigned short* src, float* dst, size_t n, hipStream_t s);   // bf16 rows -> fp32 rows
int  q27_epi_f32r(const int* acc, int N, const float* srow, int ng, const float* stok, float alpha, float* y, const float* radd, float rs, int M, hipStream_t s);
int  q27_epi_f32r_bf(const int* acc, int N, const float* srow, int ng, const float* stok, float alpha, float* y, const unsigned short* radd, float rs, int M, hipStream_t s);   // BF16 residual (the TP decode's residual stream)
int  q27_epi_res_rn_tok(const int* acc, int N, const float* srow, int ng, const float* stok, float alpha, const float* pend, float* part, const unsigned short* wgt, float* hid32, signed char* xq, float* xs, int plus_one, float in_scale, int M, hipStream_t s);   // o/out_proj epilogue + residual + post-norm (per-token int8), fused
int  q27_epi_swiglu_tok(const int* ag, const int* au, int N, const float* sg, const float* su, int ng, const float* stok, float alpha_g, float alpha_u, float gscale, signed char* xq, float* xs, int M, int gs, hipStream_t s);
int  q27_epi_swiglu_tok_small(const int* ag, const int* au, int N, const float* sg, const float* su, int ng, const float* stok, float alpha_g, float alpha_u, float gscale, signed char* xq, float* xs, int M, hipStream_t s);   // per-row scales, any N <= 5120 (the TP decode's 4352)
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
                       signed char* kcache, q27_kvs_t* kscale, signed char* vcache, q27_kvs_t* vscale,
                       float* qout_t, unsigned short* out_t, int ostride,
                       int pos0, int M, int ndev,
                       signed char* Q_t, float* QS_t, float in_scale, int hpw, int attpf,
                       float* pob, float* pml, int perm, int g64, hipStream_t s, const q27_kvquad_t* quad);
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

void q27_gdn_conv_tp_r(unsigned short* qkv, unsigned short* conv_state, int sfull, const unsigned short* cw, int kh0, int khn, hipStream_t s);
void q27_gdn_step_tp_r(const unsigned short* qkv_c, const unsigned short* z, const unsigned short* a, const unsigned short* b,
                       const unsigned short* A_log, const unsigned short* dt_bias, const unsigned short* norm_w, float* S,
                       unsigned short* out, int kh0, int khn, signed char* q, float* qs, float in_scale, hipStream_t s);
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
// ---- Q27_BUILDSTAMP (2026-09-17): the engine names the binary it is actually running --------
// Hashes /proc/self/exe, so the identity is a property of the RUNNING IMAGE: overwriting the file
// on disk cannot change what a resident boot reports. Printed once per engine process at start-up.
int  q27_md5_file(const char* path, char* hex, unsigned long long* bytes);
void q27_bootstamp(int spec, int spec_k, int ctx, int ndev, const char* cfg);
// q27_launch.cpp -- direct HIP launch counter (linker-wrapped hipLaunchKernel). See that file for
// why the chokepoint is the runtime symbol and not the hipLaunchKernelGGL macro.
void q27_lk_reset(void);
long q27_lk_total_get(void);
void q27_lk_dump(int req, long rounds, int nlayer, double resid_ms_per_round, int topn);
const char* q27_binstamp8(void);
#ifdef __cplusplus
}
#endif
