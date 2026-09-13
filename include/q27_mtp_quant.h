// ============================================================================================
// MTP DRAFT-WEIGHT QUANTIZER (deliverable 3, step 2).
//
// The checkpoint stores the whole draft head UNQUANTIZED BF16 (hf_quant_config excludes "mtp*"),
// 849 MB. Streaming that per draft step would cost ~1-1.7 ms of pure weight traffic per drafted
// token, so the draft is quantized ONCE at load into the engine's own formats -- NVFP4 for the MLP
// and FP8 e4m3 for attention -- and then reuses the entire already-gated kernel family.
//
// This is legitimate, not a correctness compromise: correct speculative sampling uses the draft's
// ACTUAL proposal distribution q in min(1, p/q), so a quantized draft is a valid proposal; the
// target verifier stays exact, and storage representation != execution representation is the
// campaign's own law.
//
// Layouts are the checkpoint ABI (see q27.h):
//   NVFP4: U8 [R][K/2], byte j = element 2j in the LOW nibble, 2j+1 in the HIGH; group scale
//          F8_E4M3 [R][K/16], dense row-major; weight_scale_2 = a single F32 multiplier.
//   FP8  : U8 [R][K] e4m3 plus a single F32 wscale; the kernel decodes with a 2^-8 bias.
// The round-trip error is reported per matrix so the draft's quality is known before it is used.
// ============================================================================================
#pragma once
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

// E2M1 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6
static inline float q27_e2m1_val(int code) {
    static const float v[8] = { 0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
    return v[code & 7];
}
static inline int q27_e2m1_code(float x, float* out) {
    const float ax = std::fabs(x);
    int best = 0; float bd = 1e30f;
    for (int c = 0; c < 8; ++c) { const float d = std::fabs(ax - q27_e2m1_val(c)); if (d < bd) { bd = d; best = c; } }
    *out = q27_e2m1_val(best);
    return best | (x < 0.f ? 8 : 0);
}
// E4M3 encode (bias 7, 3-bit mantissa). Values above 448 clamp to the max code.
static inline unsigned char q27_e4m3_code(float x) {
    const int sign = (x < 0.f) ? 0x80 : 0x00;
    float ax = std::fabs(x);
    if (ax > 448.f) ax = 448.f;
    if (ax < 1e-10f) return (unsigned char)sign;
    int e = 0; float m = std::frexp(ax, &e);          // ax = m * 2^e, m in [0.5,1)
    int ex = e - 1 + 7;                               // exponent field
    if (ex < 1) {                                     // subnormal
        int mant = (int)std::lround(ax / std::ldexp(1.f, -6));
        if (mant > 7) mant = 7;
        return (unsigned char)(sign | mant);
    }
    if (ex > 15) { ex = 15; m = 1.75f; }
    int mant = (int)std::lround((m * 2.f - 1.f) * 8.f);
    if (mant > 7) { mant = 0; ++ex; if (ex > 15) { ex = 15; mant = 7; } }
    return (unsigned char)(sign | (ex << 3) | mant);
}
static inline float q27_e4m3_val(unsigned char c) {
    const int sign = (c & 0x80) ? -1 : 1;
    const int ex = (c >> 3) & 0x0F, mant = c & 7;
    if (ex == 0) return (float)sign * std::ldexp((float)mant / 8.f, -6);
    return (float)sign * std::ldexp(1.f + (float)mant / 8.f, ex - 7);
}
static inline float q27_bf16_to_f(unsigned short u) { unsigned t = (unsigned)u << 16; float f; std::memcpy(&f, &t, 4); return f; }
static inline unsigned short q27_f_to_bf16(float f) { unsigned t; std::memcpy(&t, &f, 4); return (unsigned short)((t + 0x8000u) >> 16); }

// BF16 [R][K] -> NVFP4 packed + group scales; returns relative L2 of the round trip.
static inline double q27_quant_nvfp4(const unsigned short* src, int R, int K,
                                     std::vector<unsigned char>& wpk,
                                     std::vector<unsigned char>& gs, float* ws2) {
    wpk.assign((size_t)R * (size_t)K / 2, 0);
    gs.assign((size_t)R * (size_t)(K / 16), 0);
    // ws2 normalises the group scales into E4M3's range. Raw group scales (~1e-3) fall below
    // E4M3's smallest NORMAL (2^-6 = 0.0156) and encode as zero, which silently zeroed whole
    // groups (relL2 0.98 on the real MTP matrices). ws2 = amax/(6*448) puts the largest group
    // at 448 and every other group well inside the normal range.
    float amax_all = 0.f;
    for (size_t i = 0; i < (size_t)R * (size_t)K; ++i) { const float a = std::fabs(q27_bf16_to_f(src[i])); if (a > amax_all) amax_all = a; }
    *ws2 = (amax_all > 0.f) ? (amax_all / (6.f * 448.f)) : 1.0f;
    double num = 0, den = 0;
    for (int r = 0; r < R; ++r) {
        for (int g = 0; g < K / 16; ++g) {
            float amax = 0.f;
            for (int i = 0; i < 16; ++i) { const float v = q27_bf16_to_f(src[(size_t)r * K + g * 16 + i]); const float a = std::fabs(v); if (a > amax) amax = a; }
            const float scale = ((amax > 0.f) ? (amax / 6.f) : 1.f) / (*ws2);   // relative to ws2
            gs[(size_t)r * (K / 16) + g] = q27_e4m3_code(scale);
            const float sdec = q27_e4m3_val(gs[(size_t)r * (K / 16) + g]) * (*ws2);
            for (int i = 0; i < 16; i += 2) {
                float d0 = 0, d1 = 0;
                const int c0 = q27_e2m1_code(q27_bf16_to_f(src[(size_t)r * K + g * 16 + i]) / sdec, &d0);
                const int c1 = q27_e2m1_code(q27_bf16_to_f(src[(size_t)r * K + g * 16 + i + 1]) / sdec, &d1);
                // BOTH NIBBLES ARE 4 BITS: bit 3 is the SIGN (q27_e2m1_code returns best|8 for
                // negatives). Masking with 7 stripped the sign from every negative weight, so the
                // projection summed |w|*x instead of w*x -- which preserves amax and rms exactly
                // (why every magnitude check passed) and inflates a near-cancelling sum by about
                // sqrt(K)*E|w|/rms(w) ~ 46-70x per projection, ~2-4e3x through a SwiGLU product and
                // ~1e5-4e5x through two more projections: the observed 400,000x.
                // The relL2 below restores the sign, so it measured the INTENDED quantization rather
                // than the STORED BYTES -- a check that agreed with itself. The engine's converter
                // packs c0|(c1<<4) with no mask, which is why every engine-built handle is correct.
                wpk[((size_t)r * K + g * 16 + i) / 2] = (unsigned char)((c0 & 0xF) | ((c1 & 0xF) << 4));
                const double orig = q27_bf16_to_f(src[(size_t)r * K + g * 16 + i]);
                const double o0 = (c0 & 8) ? -d0 : d0, o1 = (c1 & 8) ? -d1 : d1;
                num += (o0 * sdec - orig) * (o0 * sdec - orig);
                den += orig * orig;
                const double orig1 = q27_bf16_to_f(src[(size_t)r * K + g * 16 + i + 1]);
                num += (o1 * sdec - orig1) * (o1 * sdec - orig1);
                den += orig1 * orig1;
            }
        }
    }
    return (den > 0) ? std::sqrt(num / den) : 0.0;
}

// BF16 [R][K] -> e4m3 + wscale; returns relative L2 of the round trip.
static inline double q27_quant_fp8(const unsigned short* src, int R, int K,
                                   std::vector<unsigned char>& w, float* wscale) {
    w.assign((size_t)R * (size_t)K, 0);
    float amax = 0.f;
    for (size_t i = 0; i < (size_t)R * (size_t)K; ++i) { const float a = std::fabs(q27_bf16_to_f(src[i])); if (a > amax) amax = a; }
    const float scale = (amax > 0.f) ? (amax / 448.f) : 1.f;
    *wscale = scale;
    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)R * (size_t)K; ++i) {
        const float v = q27_bf16_to_f(src[i]) / scale;
        w[i] = q27_e4m3_code(v);
        const double back = (double)q27_e4m3_val(w[i]) * scale;
        num += (back - q27_bf16_to_f(src[i])) * (back - q27_bf16_to_f(src[i]));
        den += (double)q27_bf16_to_f(src[i]) * q27_bf16_to_f(src[i]);
    }
    return (den > 0) ? std::sqrt(num / den) : 0.0;
}
