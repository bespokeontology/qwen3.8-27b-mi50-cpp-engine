#pragma once
// exl3_dequant_gfx906.h — gfx906 (MI50, ROCm 5.7, wave64) HIP port of the
// engine-side EXL3 trellis decode.
//
// Semantic source: src/exl3_dequant.cu — the PROVEN bit-exact CUDA port
// (receipt work/P1_R2_report.jsonl; negative control verified) of
// exllamav3 v0.0.43 pin c5d9c657:
//   exllamav3/exllamav3_ext/quant/reconstruct.cu   (kernel + launch)
//   exllamav3/exllamav3_ext/quant/exl3_dq.cuh      (bit unpack)
//   exllamav3/exllamav3_ext/quant/codebook.cuh     (procedural codebooks)
//   exllamav3/exllamav3_ext/util.cuh               (unions)
//
// Every constant and index expression keeps the "ref: <file>:<line>"
// citation of the CUDA source; where a PTX instruction had to be replaced
// for AMD the comment names the exl3_dequant.cu line it mirrors.
//
// The device decode templates live in THIS header (not the .hip) so the
// reconstruct kernel (exl3_dequant_gfx906.hip) and the X2 roofline probe
// (bench_exl3_bw_gfx906.hip) share ONE implementation of the bit math.

#include <cstdint>
#include <cstring>

#ifdef QF_HOST_CHECK
#include "hip_host_check_shim.h"
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#endif

// Decode a packed trellis tensor into the fp16 inner weight matrix.
// Same contract as src/exl3_dequant.h (cudaStream_t -> hipStream_t,
// half -> __half):
//   out        : device pointer, fp16, row-major (in_features, out_features)
//   packed     : device pointer, trellis bytes viewed as uint16,
//                logical shape (in_features/16, out_features/16, 256*K/16)
//   K          : bits per weight, = trellis.shape[-1] // 16
//                (exl3.py:57)
//   mcg/mul1   : codebook selector flags; mcg => multiplier 0xCBAC1FED
//                (reconstruct.cu:127-129)
//
// Constraints (from TORCH_CHECKs in reconstruct.cu:110-119):
//   in_features % 16 == 0, out_features % 128 == 0
//
// Produces the SAME element order and SAME fp16 values as
// ext.reconstruct (exl3.py:174-177) — bit-exact target.
void exl3_reconstruct_gfx906(
    __half* out,
    const uint16_t* packed,
    int K,
    bool mcg,
    bool mul1,
    int in_features,
    int out_features,
    hipStream_t stream
);

// Launch syntax collapses to a plain call for the host-only syntax check
// (qwenflash Makefile.hip4 "check" pattern: g++ -DQF_HOST_CHECK).
#ifdef QF_HOST_CHECK
#define EXL3_LAUNCH(k, g, b, s, ...) k(__VA_ARGS__)
#else
#define EXL3_LAUNCH(k, g, b, s, ...) k<<<g, b, 0, s>>>(__VA_ARGS__)
#endif

namespace exl3gfx
{

// ---------------------------------------------------------------------------
// Bit-level half/half2 constructors. The CUDA source uses unions over
// half2/half (ref: util.cuh:69-85, mirrored at exl3_dequant.cu:24-41);
// HIP's __half/__half2 are not guaranteed trivially unionable across ROCm
// versions, so the bitcast is spelled out via memcpy (strict-aliasing safe,
// compiles to a plain move). Layout is identical: little-endian uint16 /
// packed pair of uint16.
__device__ __forceinline__ __half half_from_bits(uint16_t u)
{
    __half h;
    memcpy(&h, &u, 2);
    return h;
}
__device__ __forceinline__ __half2 half2_from_bits(uint32_t u)
{
    __half2 h;
    memcpy(&h, &u, 4);
    return h;
}
__device__ __forceinline__ uint32_t half2_to_bits(__half2 h)
{
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

// ref: ptx.cuh:6-11 (Vec), ptx.cuh:14 (FragB = Vec<half2, 2>)
template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
};
using FragB = Vec<__half2, 2>;

// ---------------------------------------------------------------------------
// ref: codebook.cuh:5-23
// Reference's #else branch (plain x * w) applies, as on sm_121a
// (codebook.cuh:20-22, mirrored at exl3_dequant.cu:60-68).
template <uint32_t w>
__device__ __forceinline__
uint32_t mul_const_u32(uint32_t x)
{
    return x * w;
}

// lop3 replacement (no inline PTX on AMD):
//   PTX lop3.b32 d, a, b, c, 0x6a computes, per bit, lut[(a<<2)|(b<<1)|c].
//   lut 0x6a == 0b01101010 evaluates to d = c ^ (a & b), so the instruction
//   lop3.b32 x, x, 0x8fff8fff, 0x3b603b60, 0x6a
//   is exactly  x = (x & 0x8fff8fff) ^ 0x3b603b60  in plain C.
#define EXL3_LOP3_CB(x) ((x) = ((x) & 0x8fff8fffu) ^ 0x3b603b60u)

// vabsdiff4 replacement (no inline PTX on AMD):
//   PTX vabsdiff4.u32.u32.u32.add d, x, 0, acc sums the per-byte absolute
//   differences |x.b_i - 0| and adds acc, i.e. the plain byte sum of x + acc.
#define EXL3_VABSDIFF4_ADD0(x, acc) \
    (((x) & 0xffu) + (((x) >> 8) & 0xffu) + (((x) >> 16) & 0xffu) + ((x) >> 24) + (acc))

// ---------------------------------------------------------------------------
// V_SAD_U8: the byte sum in ONE instruction.
//
// sum_i |x.b_i - 0| + acc IS the plain byte sum plus acc, which is exactly what
// the macro above spells out in ~10 VALU (3 shifts, 4 masks, 4 adds) -- and it
// runs twice per half2, so it was ~10 VALU per WEIGHT and the dominant term in
// a decode measured at ~17 VALU/weight. V_SAD_U8 is a VOP3 instruction present
// since GCN3, so gfx906 has had it all along.
//
// Taken from vcruz305/exllamav3-amd (exllamav3_ext/quant/codebook.cuh), whose
// guard lists only RDNA because that port targets gfx11.5/gfx12 -- the
// instruction itself is not RDNA-specific. Their note: "the dp4a byte-sum is
// emulated with ~5 VALU ops per state, the SAD form is one instruction and the
// decode chain is instruction-issue-bound."
//
// The builtin is used rather than their inline asm on purpose: their form
// passes 0x6400 as an "n" immediate, and gfx906 VOP3 cannot encode a 32-bit
// literal (that arrived with gfx10). The builtin lets the compiler place the
// constant in a register and hoist it out of the loop.
//
// Bit-identical to the macro by construction: same sum, same addend.
#if !defined(EXL3_CB_NO_SAD) && (defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx942__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1200__) || defined(__gfx1201__))
#define EXL3_CB_HAVE_SAD 1
__device__ __forceinline__ uint32_t exl3_sad_u8_(uint32_t p, uint32_t addend)
{
    return __builtin_amdgcn_sad_u8(p, 0u, addend);
}
#endif

// ref: codebook.cuh:25-56
template <int cb>
__device__ inline __half decode_3inst(uint32_t x)
{
    if constexpr (cb == 0)
    {
        x *= 89226354u;                                  // codebook.cuh:30
        x += 64248484u;                                  // codebook.cuh:31
        EXL3_LOP3_CB(x);                                 // codebook.cuh:32 — mirrors exl3_dequant.cu:78 (lop3.b32, see above)
        __half2 xu = half2_from_bits(x);
        return __hadd(__low2half(xu), __high2half(xu));  // codebook.cuh:34
    }
    if constexpr (cb == 1)
    {
        x = mul_const_u32<0xCBAC1FEDu>(x);               // codebook.cuh:39 (MCG multiplier)
        EXL3_LOP3_CB(x);                                 // codebook.cuh:41 — mirrors exl3_dequant.cu:85 (lop3.b32)
        __half2 xu = half2_from_bits(x);
        return __hadd(__low2half(xu), __high2half(xu));  // codebook.cuh:43
    }
    if constexpr (cb == 2)
    {
        x *= 0x83DCD12Du;                                // codebook.cuh:47
        const uint32_t acc = 0x6400u;                    // codebook.cuh:49
        // vabsdiff4.u32.u32.u32.add sum, x, 0, acc == byte-sum(x) + acc;
        // explicit byte math — mirrors exl3_dequant.cu:94 (codebook.cuh:50).
#if defined(EXL3_CB_HAVE_SAD)
        uint32_t sum = exl3_sad_u8_(x, acc);
#else
        uint32_t sum = EXL3_VABSDIFF4_ADD0(x, acc);
#endif
        const __half k_inv_h = half_from_bits(0x1eee);   // codebook.cuh:51 (1/147.7; was __ushort_as_half, exl3_dequant.cu:95)
        const __half k_bias_h = half_from_bits(0xc931);  // codebook.cuh:52 (-10.39; exl3_dequant.cu:96)
        return __hfma(half_from_bits((uint16_t) sum), k_inv_h, k_bias_h);  // codebook.cuh:54
    }
    return half_from_bits((uint16_t)0);  // unreachable: cb in {0,1,2}
}

// ref: codebook.cuh:58-104
template <int cb>
__device__ inline __half2 decode_3inst_2(uint32_t x0, uint32_t x1)
{
    if constexpr (cb == 0)
    {
        x0 *= 89226354u;                                 // codebook.cuh:63
        x1 *= 89226354u;                                 // codebook.cuh:64
        x0 += 64248484u;                                 // codebook.cuh:65
        x1 += 64248484u;                                 // codebook.cuh:66
        EXL3_LOP3_CB(x0);                                // codebook.cuh:67 — mirrors exl3_dequant.cu:112 (lop3.b32)
        EXL3_LOP3_CB(x1);                                // codebook.cuh:68 — mirrors exl3_dequant.cu:113 (lop3.b32)
        __half2 xu0 = half2_from_bits(x0);
        __half2 xu1 = half2_from_bits(x1);
        __half2 d0 = __lows2half2(xu0, xu1);             // codebook.cuh:71
        __half2 d1 = __highs2half2(xu0, xu1);            // codebook.cuh:72
        return __hadd2(d0, d1);                          // codebook.cuh:73
    }
    if constexpr (cb == 1)
    {
        x0 = mul_const_u32<0xCBAC1FEDu>(x0);             // codebook.cuh:79
        x1 = mul_const_u32<0xCBAC1FEDu>(x1);             // codebook.cuh:80
        EXL3_LOP3_CB(x0);                                // codebook.cuh:81 — mirrors exl3_dequant.cu:124 (lop3.b32)
        EXL3_LOP3_CB(x1);                                // codebook.cuh:82 — mirrors exl3_dequant.cu:125 (lop3.b32)
        __half2 xu0 = half2_from_bits(x0);
        __half2 xu1 = half2_from_bits(x1);
        __half2 d0 = __lows2half2(xu0, xu1);             // codebook.cuh:85
        __half2 d1 = __highs2half2(xu0, xu1);            // codebook.cuh:86
        return __hadd2(d0, d1);                          // codebook.cuh:87
    }
    if constexpr (cb == 2)
    {
        x0 *= 0x83DCD12Du;                               // codebook.cuh:91
        x1 *= 0x83DCD12Du;                               // codebook.cuh:92
        const uint32_t acc = 0x6400u;                    // codebook.cuh:95
        // vabsdiff4 byte-sum replacements — mirrors exl3_dequant.cu:139-140
        // (codebook.cuh:96-97).
#if defined(EXL3_CB_HAVE_SAD)
        // decode_mul1_product_2_sad: two SADs, one packed add, one hfma2.
        // u + 1024 <= 2029, so the two 16-bit halves cannot carry into each other.
        const uint32_t sum1 = exl3_sad_u8_(x1, acc);
        const uint32_t sum0 = exl3_sad_u8_(x0, acc);
        const uint32_t packed = (sum1 << 16) + sum0;
        __half2 k_inv_h2 = __half2half2(half_from_bits(0x1eee));
        __half2 k_bias_h2 = __half2half2(half_from_bits(0xc931));
        return __hfma2(__halves2half2(half_from_bits((uint16_t) packed),
                                      half_from_bits((uint16_t)(packed >> 16))),
                       k_inv_h2, k_bias_h2);
#else
        uint32_t sum0 = EXL3_VABSDIFF4_ADD0(x0, acc);
        uint32_t sum1 = EXL3_VABSDIFF4_ADD0(x1, acc);
        __half2 k_inv_h2 = __half2half2(half_from_bits(0x1eee));   // codebook.cuh:98
        __half2 k_bias_h2 = __half2half2(half_from_bits(0xc931));  // codebook.cuh:99
        return __hfma2(__halves2half2(half_from_bits((uint16_t) sum0),
                                      half_from_bits((uint16_t) sum1)),
                       k_inv_h2, k_bias_h2);             // codebook.cuh:102
#endif
    }
    return __half2half2(half_from_bits((uint16_t)0));  // unreachable: cb in {0,1,2}
}

// ---------------------------------------------------------------------------
// ref: exl3_dq.cuh:5-13
// This helper is already plain C++; it is ALSO the replacement for every
// __funnelshift_r / shf.r.wrap.b32 below: CUDA __funnelshift_r(lo, hi, s)
// and PTX shf.r.wrap.b32 both compute low32({hi:lo} >> s), which is exactly
// fshift(lo, hi, s). No funnel-shift instruction is needed.
__device__ __forceinline__ uint32_t fshift(const uint32_t b, const uint32_t a, int shift)
{
     uint64_t merged = ((uint64_t)a << 32) | (uint64_t) b;   // exl3_dq.cuh:7
     return (uint32_t)(merged >> shift);                     // exl3_dq.cuh:8
}

// ref: exl3_dq.cuh:15-31
template <int bits, int cb>
__device__ __forceinline__ __half dq(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // exl3_dq.cuh:18
    int b1 = b0 + 16;                                   // exl3_dq.cuh:19
    int i0 = b0 / 32;                                   // exl3_dq.cuh:20
    int i1 = (b1 - 1) / 32;                             // exl3_dq.cuh:21
    int s0 = (i1 + 1) * 32 - b1;                        // exl3_dq.cuh:22

    uint32_t a = ptr[i0 % (bits * 256 / 32)];           // exl3_dq.cuh:25
    uint32_t b = ptr[i1 % (bits * 256 / 32)];           // exl3_dq.cuh:26

    // was __funnelshift_r(b, a, s0) — identical to fshift (exl3_dequant.cu:170)
    uint32_t w0 = fshift(b, a, s0) & 0xffff;            // exl3_dq.cuh:29
    return decode_3inst<cb>(w0);
}

// ref: exl3_dq.cuh:33-50
template <int bits, int cb>
__device__ __forceinline__ __half2 dq2(const uint32_t* ptr, int t_offset)
{
    int b0 = t_offset * bits + bits - 16 + 256 * bits;  // exl3_dq.cuh:36
    int b1 = b0 + 16;                                   // exl3_dq.cuh:37
    int i0 = b0 / 32;                                   // exl3_dq.cuh:38
    int i1 = (b1 - 1) / 32;                             // exl3_dq.cuh:39
    int s0 = (i1 + 1) * 32 - b1;                        // exl3_dq.cuh:40

    uint32_t a = ptr[i0 % (bits * 256 / 32)];           // exl3_dq.cuh:43
    uint32_t b = ptr[i1 % (bits * 256 / 32)];           // exl3_dq.cuh:44

    // were __funnelshift_r — identical to fshift (exl3_dequant.cu:187-188)
    uint32_t w1 = fshift(b, a, s0)        & 0xffff;     // exl3_dq.cuh:47
    uint32_t w0 = fshift(b, a, s0 + bits) & 0xffff;     // exl3_dq.cuh:48
    return decode_3inst_2<cb>(w0, w1);
}

// ref: exl3_dq.cuh:52-72
template <int bits, int cb>
__device__ __forceinline__ void dq4(const uint32_t* ptr, int t_offset, FragB& frag)
{
    int b0 = (t_offset + 257) * bits - 16;      // exl3_dq.cuh:55
    int b1 = b0 + 3 * bits;                     // exl3_dq.cuh:56
    int b2 = b1 + 16;                           // exl3_dq.cuh:57
    int i0 = b0 / 32;                           // exl3_dq.cuh:58
    int i2 = (b2 - 1) / 32;                     // exl3_dq.cuh:59
    int s2 = (i2 + 1) * 32 - b2;                // exl3_dq.cuh:60

    uint32_t a = ptr[i0 % (bits * 256 / 32)];   // exl3_dq.cuh:62
    uint32_t b = ptr[i2 % (bits * 256 / 32)];   // exl3_dq.cuh:63
    uint32_t w3 = fshift(b, a, s2)            & 0xffff;   // exl3_dq.cuh:64
    uint32_t w2 = fshift(b, a, s2 + bits)     & 0xffff;   // exl3_dq.cuh:65
    uint32_t w1 = fshift(b, a, s2 + bits * 2) & 0xffff;   // exl3_dq.cuh:66
    uint32_t w0 = fshift(b, a, s2 + bits * 3) & 0xffff;   // exl3_dq.cuh:67
    __half2 d0d1 = decode_3inst_2<cb>(w0, w1);  // exl3_dq.cuh:68
    __half2 d2d3 = decode_3inst_2<cb>(w2, w3);  // exl3_dq.cuh:69
    frag[0] = d0d1;
    frag[1] = d2d3;
}

// ref: exl3_dq.cuh:74-94
template <int bits, int cb>
__device__ __forceinline__ void dq2x2(const uint32_t* ptr, int t_offset, FragB& frag)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        int b0 = (t_offset + 2 * i + 257) * bits - 16;  // exl3_dq.cuh:80
        int b1 = b0 + 1 * bits;                         // exl3_dq.cuh:81
        int b2 = b1 + 16;                               // exl3_dq.cuh:82
        int i0 = b0 / 32;                               // exl3_dq.cuh:83
        int i2 = (b2 - 1) / 32;                         // exl3_dq.cuh:84
        int s2 = (i2 + 1) * 32 - b2;                    // exl3_dq.cuh:85

        uint32_t a = ptr[i0 % (bits * 256 / 32)];       // exl3_dq.cuh:87
        uint32_t b = ptr[i2 % (bits * 256 / 32)];       // exl3_dq.cuh:88
        uint32_t w1 = fshift(b, a, s2)        & 0xffff; // exl3_dq.cuh:89
        uint32_t w0 = fshift(b, a, s2 + bits) & 0xffff; // exl3_dq.cuh:90
        __half2 d0d1 = decode_3inst_2<cb>(w0, w1);
        frag[i] = d0d1;
    }
}

// ref: exl3_dq.cuh:96-161
template <int bits, int cb, int align>
__device__ __forceinline__ void dq8(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    int b1 = (t_offset + 257) * bits;               // exl3_dq.cuh:99
    int b0 = b1 - 16;                               // exl3_dq.cuh:100
    int b2 = b1 + bits * 7;                         // exl3_dq.cuh:101
    int i0 = b0 / 32;                               // exl3_dq.cuh:102
    int i2 = (b2 - 1) / 32;                         // exl3_dq.cuh:103
    int s2 = (i2 + 1) * 32 - b2;                    // exl3_dq.cuh:104

    uint32_t a = ptr[i0 % (bits * 256 / 32)];       // exl3_dq.cuh:106
    uint32_t b = ptr[i2 % (bits * 256 / 32)];       // exl3_dq.cuh:107
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (align == 1)                       // exl3_dq.cuh:109-119
    {
        w7 = fshift(b, a, s2);
        w6 = fshift(b, a, s2 + bits);
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = fshift(b, a, s2 + bits * 3);
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = fshift(b, a, s2 + bits * 5);
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = fshift(b, a, s2 + bits * 7);
    }
    if constexpr (align == 2)                       // exl3_dq.cuh:120-130
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = fshift(b, a, s2 + bits * 2);
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = fshift(b, a, s2 + bits * 6);
        w0 = w1 >> bits;
    }
    if constexpr (align == 4)                       // exl3_dq.cuh:131-141
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    if constexpr (align == 8)                       // exl3_dq.cuh:142-152
    {
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = w4 >> bits;
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
    }
    __half2 d0d1 = decode_3inst_2<cb>(w0 & 0xffff, w1 & 0xffff);  // exl3_dq.cuh:153
    __half2 d2d3 = decode_3inst_2<cb>(w2 & 0xffff, w3 & 0xffff);  // exl3_dq.cuh:154
    __half2 d4d5 = decode_3inst_2<cb>(w4 & 0xffff, w5 & 0xffff);  // exl3_dq.cuh:155
    __half2 d6d7 = decode_3inst_2<cb>(w6 & 0xffff, w7 & 0xffff);  // exl3_dq.cuh:156
    frag0[0] = d0d1;
    frag0[1] = d2d3;
    frag1[0] = d4d5;
    frag1[1] = d6d7;
}

// ref: exl3_dq.cuh:163-184  (the K=4 path used by this checkpoint)
template <int cb>
__device__ __forceinline__ void dq8_aligned_4bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, s, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 3;                             // exl3_dq.cuh:167
    i0 = (i1 + 31) & 31;                            // exl3_dq.cuh:168
    a = ptr[i0];                                    // exl3_dq.cuh:169
    b = ptr[i1];                                    // exl3_dq.cuh:170
    // was FSHF_IMM(s, b, a, 20) = shf.r.wrap.b32 — identical to fshift
    // (exl3_dequant.cu:54,315); exl3_dq.cuh:171
    s = fshift(b, a, 20);
    w7 = b & 0xffff;                                // exl3_dq.cuh:172
    // BFE16_IMM(dst, src, imm) was bfe.u32 dst, src, imm, 16
    // (exl3_dequant.cu:57,317-323); plain C: (src >> imm) & 0xffff.
    w6 = (b >> 4) & 0xffff;                         // exl3_dq.cuh:173
    w5 = (b >> 8) & 0xffff;                         // exl3_dq.cuh:174
    w4 = (b >> 12) & 0xffff;                        // exl3_dq.cuh:175
    w3 = (b >> 16) & 0xffff;                        // exl3_dq.cuh:176
    w2 = s & 0xffff;                                // exl3_dq.cuh:177
    w1 = (s >> 4) & 0xffff;                         // exl3_dq.cuh:178
    w0 = (s >> 8) & 0xffff;                         // exl3_dq.cuh:179
    frag0[0] = decode_3inst_2<cb>(w0, w1);          // exl3_dq.cuh:180
    frag0[1] = decode_3inst_2<cb>(w2, w3);          // exl3_dq.cuh:181
    frag1[0] = decode_3inst_2<cb>(w4, w5);          // exl3_dq.cuh:182
    frag1[1] = decode_3inst_2<cb>(w6, w7);          // exl3_dq.cuh:183
}

// ref: exl3_dq.cuh:186-207
template <int cb>
__device__ __forceinline__ void dq8_aligned_2bits(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 4;                             // exl3_dq.cuh:190
    i0 = (i1 + 15) & 15;                            // exl3_dq.cuh:191
    a = ptr[i0];                                    // exl3_dq.cuh:192
    b = ptr[i1];                                    // exl3_dq.cuh:193
    b = fshift(b, a, ((~t_offset) & 8) << 1);       // exl3_dq.cuh:194
    w7 = b & 0xffff;                                // exl3_dq.cuh:195
    w6 = (b >> 2) & 0xffff;                         // exl3_dq.cuh:196 (was bfe.u32, exl3_dequant.cu:341)
    w5 = (b >> 4) & 0xffff;                         // exl3_dq.cuh:197
    w4 = (b >> 6) & 0xffff;                         // exl3_dq.cuh:198
    w3 = (b >> 8) & 0xffff;                         // exl3_dq.cuh:199
    w2 = (b >> 10) & 0xffff;                        // exl3_dq.cuh:200
    w1 = (b >> 12) & 0xffff;                        // exl3_dq.cuh:201
    w0 = (b >> 14) & 0xffff;                        // exl3_dq.cuh:202
    frag0[0] = decode_3inst_2<cb>(w0, w1);          // exl3_dq.cuh:203
    frag0[1] = decode_3inst_2<cb>(w2, w3);          // exl3_dq.cuh:204
    frag1[0] = decode_3inst_2<cb>(w4, w5);          // exl3_dq.cuh:205
    frag1[1] = decode_3inst_2<cb>(w6, w7);          // exl3_dq.cuh:206
}

// ref: exl3_dq.cuh:209-230
template <int cb>
__device__ __forceinline__ void dq8_aligned_1bit(const uint32_t* ptr, int t_offset, FragB& frag0, FragB& frag1)
{
    uint32_t i0, i1, a, b, w0, w1, w2, w3, w4, w5, w6, w7;
    i1 = t_offset >> 5;                             // exl3_dq.cuh:213
    i0 = (i1 + 7) & 7;                              // exl3_dq.cuh:214
    a = ptr[i0];                                    // exl3_dq.cuh:215
    b = ptr[i1];                                    // exl3_dq.cuh:216
    b = fshift(b, a, ((~t_offset) & 24));           // exl3_dq.cuh:217
    w7 = b & 0xffff;                                // exl3_dq.cuh:218
    w6 = (b >> 1) & 0xffff;                         // exl3_dq.cuh:219 (was bfe.u32, exl3_dequant.cu:365)
    w5 = (b >> 2) & 0xffff;                         // exl3_dq.cuh:220
    w4 = (b >> 3) & 0xffff;                         // exl3_dq.cuh:221
    w3 = (b >> 4) & 0xffff;                         // exl3_dq.cuh:222
    w2 = (b >> 5) & 0xffff;                         // exl3_dq.cuh:223
    w1 = (b >> 6) & 0xffff;                         // exl3_dq.cuh:224
    w0 = (b >> 7) & 0xffff;                         // exl3_dq.cuh:225
    frag0[0] = decode_3inst_2<cb>(w0, w1);          // exl3_dq.cuh:226
    frag0[1] = decode_3inst_2<cb>(w2, w3);          // exl3_dq.cuh:227
    frag1[0] = decode_3inst_2<cb>(w4, w5);          // exl3_dq.cuh:228
    frag1[1] = decode_3inst_2<cb>(w6, w7);          // exl3_dq.cuh:229
}

// ref: exl3_dq.cuh:254-295
template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch(const uint32_t* ptr, int idx, FragB& frag0, FragB& frag1)
{
    if constexpr (bits == 1)
    {
        dq8_aligned_1bit<cb>(ptr, idx, frag0, frag1);       // exl3_dq.cuh:259
    }
    else if constexpr (bits == 2)
    {
        dq8_aligned_2bits<cb>(ptr, idx, frag0, frag1);      // exl3_dq.cuh:263
    }
    else if constexpr (bits == 3)
    {
        dq8<bits, cb, 4>(ptr, idx, frag0, frag1);           // exl3_dq.cuh:267
    }
    else if constexpr (bits == 4)
    {
        dq8_aligned_4bits<cb>(ptr, idx, frag0, frag1);      // exl3_dq.cuh:271
    }
    else if constexpr (bits == 5)
    {
        dq4<bits, cb>(ptr, idx, frag0);                     // exl3_dq.cuh:275
        dq4<bits, cb>(ptr, idx + 4, frag1);                 // exl3_dq.cuh:276
    }
    else if constexpr (bits == 6)
    {
        dq4<bits, cb>(ptr, idx, frag0);                     // exl3_dq.cuh:280
        dq4<bits, cb>(ptr, idx + 4, frag1);                 // exl3_dq.cuh:281
    }
    else if constexpr (bits == 7)
    {
        dq2x2<bits, cb>(ptr, idx, frag0);                   // exl3_dq.cuh:285
        dq2x2<bits, cb>(ptr, idx + 4, frag1);               // exl3_dq.cuh:286
    }
    else if constexpr (bits == 8)
    {
        dq4<bits, cb>(ptr, idx, frag0);                     // exl3_dq.cuh:290
        dq4<bits, cb>(ptr, idx + 4, frag1);                 // exl3_dq.cuh:291
    }
}

// ---------------------------------------------------------------------------
// ref: reconstruct.cu:43-73 (mirrored at exl3_dequant.cu:454-485)
// Tensor-core fragment layout -> row-major 16x128 tile staging via shuffle.
// Shared by the reconstruct kernel and the roofline probe's decode+dot
// kernel so the layout dance exists exactly once.
//
// wave64: CUDA ran 8 independent 32-lane warps with mask 0xFFFFFFFF. On
// gfx906 a wavefront is 64 lanes, so lane_id = t % 32 / warp_id = t / 32
// arithmetic puts TWO logical warps in one wavefront. The shuffle therefore
// uses the full 64-lane mask (~0ULL) with width = 32, which segments the
// wavefront into the same two 32-lane groups CUDA had; exchanges never
// cross the half-wave boundary, matching lane_id semantics exactly.
// The shuffle payload is the raw uint32 bits of each half2 (HIP ROCm 5.7
// guarantees integer shuffle overloads; half2 overloads are version-
// dependent), so the bits are shuffled and re-wrapped.
__device__ __forceinline__ void frags_to_tile(FragB frag[2], int lane_id, int warp_id,
                                              __half2 (*tile)[8][8])
{
    int n0b = __shfl_down((int)half2_to_bits(frag[0][0]), 4, 32);  // reconstruct.cu:45
    int n1b = __shfl_down((int)half2_to_bits(frag[0][1]), 4, 32);  // reconstruct.cu:46
    int n2b = __shfl_down((int)half2_to_bits(frag[1][0]), 4, 32);  // reconstruct.cu:47
    int n3b = __shfl_down((int)half2_to_bits(frag[1][1]), 4, 32);  // reconstruct.cu:48

    if (!(lane_id & 4))                                                 // reconstruct.cu:50
    {
        __half2 n0 = half2_from_bits((uint32_t)n0b);
        __half2 n1 = half2_from_bits((uint32_t)n1b);
        __half2 n2 = half2_from_bits((uint32_t)n2b);
        __half2 n3 = half2_from_bits((uint32_t)n3b);
        __half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));   // reconstruct.cu:52
        __half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0)); // reconstruct.cu:53
        __half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));   // reconstruct.cu:54
        __half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1)); // reconstruct.cu:55
        __half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));   // reconstruct.cu:56
        __half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2)); // reconstruct.cu:57
        __half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));   // reconstruct.cu:58
        __half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3)); // reconstruct.cu:59
        int r0 = (lane_id % 4) * 2;                     // reconstruct.cu:60
        int r1 = r0 + 1;                                // reconstruct.cu:61
        int r2 = r0 + 8;                                // reconstruct.cu:62
        int r3 = r0 + 9;                                // reconstruct.cu:63
        int c0 = lane_id / 8;                           // reconstruct.cu:64
        int c1 = c0 + 4;                                // reconstruct.cu:65
        tile[r0][warp_id][c0] = m0;                     // reconstruct.cu:66
        tile[r1][warp_id][c0] = m1;                     // reconstruct.cu:67
        tile[r2][warp_id][c0] = m2;                     // reconstruct.cu:68
        tile[r3][warp_id][c0] = m3;                     // reconstruct.cu:69
        tile[r0][warp_id][c1] = m4;                     // reconstruct.cu:70
        tile[r1][warp_id][c1] = m5;                     // reconstruct.cu:71
        tile[r2][warp_id][c1] = m6;                     // reconstruct.cu:72
        tile[r3][warp_id][c1] = m7;                     // reconstruct.cu:73
    }
}

}  // namespace exl3gfx
