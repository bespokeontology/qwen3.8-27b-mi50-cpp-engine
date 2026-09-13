// q27_wide.hip -- WIDE PREFILL (GEMM-order) kernels. Operator decision 2026-09-11: prefill may use a different fp32
// accumulation order than the frozen release (rung 3 stays the bit-identical exact mode); the acceptance gate is
// numerical quality (full-logit error, top-1/token agreement, continuation agreement), not an md5.
//
// Dataflow = llama.cpp MMQ / CK on gfx906 terms: OUTPUT-STATIONARY tiles of 128 weight rows x 64 tokens per 256-thread
// block; each thread owns 2 rows x 16 tokens = 32 fp32 accumulators and walks the WHOLE K inside the thread (no
// cross-lane reduction). Per K-iteration of KI 16-element groups the block decodes 128 rows of NVFP4 nibbles to int8
// ONCE into LDS (amortized over 64 tokens) and stages 64 tokens' int8 activations once (amortized over 128 rows).
// Per (row, token, group) the arithmetic is the reference's: four v_dot4 in the release pairing, one cvt, one
// (w_scale*x_scale) multiply, one fma -- only the ORDER in which the 320 groups accumulate differs (sequential per
// output instead of 64 lane-partials + a tree).
#include "q27.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

namespace {

template <int KI, int NW, int MINW>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_gu(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float ag, float au, int rows, int K, int M,
        float* __restrict__ YG, float* __restrict__ YU)
{
    constexpr int TR = 128, TC = 64;
    constexpr int WSTR = KI * 16 + 16;                     // bytes per staged weight row (16-B aligned, bank-shifted)
    constexpr int XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];    // decoded int8 weights
    __shared__ unsigned char swsc[TR * KI];                // e4m3 weight scales
    __shared__ __align__(16) signed char sx[TC * XSTR];    // int8 activations
    __shared__ float sxsc[TC * KI];                        // activation scales * 0.5
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int mat = blockIdx.z;
    const unsigned char* __restrict__ W = mat ? WU : WG;
    const unsigned char* __restrict__ S = mat ? SU : SG;
    float* __restrict__ Y = mat ? YU : YG;
    const float alpha = mat ? au : ag;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const uint2 v = *(const uint2*)(W + (size_t)rr * KP + (size_t)(g0 + g) * 8);
            uint4 o;
            o.x = q27_dq4( v.x       & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
            o.z = q27_dq4( v.y       & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
            *(uint4*)(sw + r * WSTR + g * 16) = o;
            swsc[it] = S[(size_t)rr * NG + g0 + g];
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
        __syncthreads();
#pragma unroll 2
        for (int g = 0; g < KI; ++g) {
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = *(const uint4*)(sw + (lane + 64) * WSTR + g * 16);
            const float s0 = q27_e4m3(swsc[lane * KI + g]);
            const float s1 = q27_e4m3(swsc[(lane + 64) * KI + g]);
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 x = *(const uint4*)(sx + t * XSTR + g * 16);
                const float xs = sxsc[t * KI + g];
                int a = q27_d4(w0.x, x.x, 0); a = q27_d4(w0.y, x.z, a); a = q27_d4(w0.z, x.y, a); a = q27_d4(w0.w, x.w, a);
                int b = q27_d4(w1.x, x.x, 0); b = q27_d4(w1.y, x.z, b); b = q27_d4(w1.z, x.y, b); b = q27_d4(w1.w, x.w, b);
                acc0[j] = fmaf((float)a, s0 * xs, acc0[j]);
                acc1[j] = fmaf((float)b, s1 * xs, acc1[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int t = tok_base + wave * TPW + j;
        if (t < M) {
            if (r0 < rows) Y[(size_t)t * rows + r0] = acc0[j] * alpha;
            if (r1 < rows) Y[(size_t)t * rows + r1] = acc1[j] * alpha;
        }
    }
}

// bf16-output sibling of q27_k_wide_down for the layer-split projection sites (q/k/v and
// GDN in_qkv/in_z): same staging/GEMM, writes q27_f2bf(v) into a caller-strided bf16 buffer.
template <int KI, int NW, int MINW>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_down_b(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, unsigned short* __restrict__ Y, int ystride)
{
    constexpr int TR = 128, TC = 64;
    constexpr int WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            const uint2 v = *(const uint2*)(W + (size_t)rr * KP + (size_t)gg * 8);
            uint4 o;
            o.x = q27_dq4( v.x       & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
            o.z = q27_dq4( v.y       & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
            *(uint4*)(sw + r * WSTR + g * 16) = o;
            swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
        }
        __syncthreads();
#pragma unroll 2
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = *(const uint4*)(sw + (lane + 64) * WSTR + g * 16);
            const float s0 = q27_e4m3(swsc[lane * KI + g]);
            const float s1 = q27_e4m3(swsc[(lane + 64) * KI + g]);
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 x = *(const uint4*)(sx + t * XSTR + g * 16);
                const float xs = sxsc[t * KI + g];
                int a = q27_d4(w0.x, x.x, 0); a = q27_d4(w0.y, x.z, a); a = q27_d4(w0.z, x.y, a); a = q27_d4(w0.w, x.w, a);
                int b = q27_d4(w1.x, x.x, 0); b = q27_d4(w1.y, x.z, b); b = q27_d4(w1.z, x.y, b); b = q27_d4(w1.w, x.w, b);
                acc0[j] = fmaf((float)a, s0 * xs, acc0[j]);
                acc1[j] = fmaf((float)b, s1 * xs, acc1[j]);
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

template <int KI, int NW, int MINW>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_down(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    constexpr int TR = 128, TC = 64;
    constexpr int WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);                    // K = 4352 -> 272 groups = 34 x 8 exactly; general guard
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            const uint2 v = *(const uint2*)(W + (size_t)rr * KP + (size_t)gg * 8);
            uint4 o;
            o.x = q27_dq4( v.x       & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
            o.z = q27_dq4( v.y       & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
            *(uint4*)(sw + r * WSTR + g * 16) = o;
            swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;   // masked groups contribute exact zeros
        }
        __syncthreads();
#pragma unroll 2
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = *(const uint4*)(sw + (lane + 64) * WSTR + g * 16);
            const float s0 = q27_e4m3(swsc[lane * KI + g]);
            const float s1 = q27_e4m3(swsc[(lane + 64) * KI + g]);
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 x = *(const uint4*)(sx + t * XSTR + g * 16);
                const float xs = sxsc[t * KI + g];
                int a = q27_d4(w0.x, x.x, 0); a = q27_d4(w0.y, x.z, a); a = q27_d4(w0.z, x.y, a); a = q27_d4(w0.w, x.w, a);
                int b = q27_d4(w1.x, x.x, 0); b = q27_d4(w1.y, x.z, b); b = q27_d4(w1.z, x.y, b); b = q27_d4(w1.w, x.w, b);
                acc0[j] = fmaf((float)a, s0 * xs, acc0[j]);
                acc1[j] = fmaf((float)b, s1 * xs, acc1[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int t = tok_base + wave * TPW + j;
        if (t < M) {
            if (r0 < rows) { float v = acc0[j] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rows + r0]); Y[(size_t)t * rows + r0] = v; }
            if (r1 < rows) { float v = acc1[j] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rows + r1]); Y[(size_t)t * rows + r1] = v; }
        }
    }
}


// Register-blocked microtile: tile 64 rows x 128 tokens, 256 threads as 16x16, each thread owns 4 rows x 8 tokens
// (32 fp32 accumulators). Per K-group each thread loads its 4 weight quads and 8 activation quads from LDS into
// registers ONCE, then does 4x8x4 v_dot4 out of registers -> LDS traffic ~1.5 B/dot4 (vs 2.5 for the row-major form).
// Same release dot4 pairing; GEMM accumulation order.
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_gu_rb(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float ag, float au, int rows, int K, int M, float* __restrict__ YG, float* __restrict__ YU) {
    constexpr int TR = 64, TC = 16 * TCB, WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int mat = blockIdx.z;
    const unsigned char* __restrict__ W = mat ? WU : WG;
    const unsigned char* __restrict__ S = mat ? SU : SG;
    float* __restrict__ Y = mat ? YU : YG;
    const float alpha = mat ? au : ag;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += 256) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const uint2 v = *(const uint2*)(W + (size_t)rr * KP + (size_t)(g0 + g) * 8);
            uint4 o; o.x = q27_dq4(v.x & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
            o.z = q27_dq4(v.y & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
            *(uint4*)(sw + r * WSTR + g * 16) = o; swsc[it] = S[(size_t)rr * NG + g0 + g];
        }
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            uint4 wr[4]; float ws[4]; uint4 xr[8]; float xsv[8];
#pragma unroll
            for (int r = 0; r < 4; ++r) { const int rw = tx * 4 + r; wr[r] = *(const uint4*)(sw + rw * WSTR + g * 16); ws[r] = q27_e4m3(swsc[rw * KI + g]); }
#pragma unroll
            for (int c = 0; c < TCB; ++c) { const int tc = ty * TCB + c; xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16); xsv[c] = sxsc[tc * KI + g]; }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr[r].x, xr[c].x, 0); d = q27_d4(wr[r].y, xr[c].z, d); d = q27_d4(wr[r].z, xr[c].y, d); d = q27_d4(wr[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws[r] * xsv[c], acc[r][c]);
                }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * rows + row] = acc[r][c] * alpha; }
    }
}
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_rb(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs) {
    constexpr int TR = 64, TC = 16 * TCB, WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TR * KI; it += 256) {
            const int r = it / KI, g = it - r * KI; const int rr = min(row_base + r, rows - 1); const int gg = min(g0 + g, NG - 1);
            const uint2 v = *(const uint2*)(W + (size_t)rr * KP + (size_t)gg * 8);
            uint4 o; o.x = q27_dq4(v.x & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
            o.z = q27_dq4(v.y & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
            *(uint4*)(sw + r * WSTR + g * 16) = o; swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 wr[4]; float ws[4]; uint4 xr[8]; float xsv[8];
#pragma unroll
            for (int r = 0; r < 4; ++r) { const int rw = tx * 4 + r; wr[r] = *(const uint4*)(sw + rw * WSTR + g * 16); ws[r] = q27_e4m3(swsc[rw * KI + g]); }
#pragma unroll
            for (int c = 0; c < TCB; ++c) { const int tc = ty * TCB + c; xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16); xsv[c] = sxsc[tc * KI + g]; }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr[r].x, xr[c].x, 0); d = q27_d4(wr[r].y, xr[c].z, d); d = q27_d4(wr[r].z, xr[c].y, d); d = q27_d4(wr[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws[r] * xsv[c], acc[r][c]);
                }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rows + row]); Y[(size_t)t * rows + row] = v; } }
    }
}

// rb + WEIGHT PREFETCH: the cold packed weights for K-group g+1 are loaded and decoded into
// registers while group g computes, so the dot4 chain never waits on a weight load. Activations
// keep the LDS staging (they are L2-hot: written by the fused tail right before this kernel).
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_gu_rbpf(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float ag, float au, int rows, int K, int M, float* __restrict__ YG, float* __restrict__ YU)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int mat = blockIdx.z;
    const unsigned char* __restrict__ W = mat ? WU : WG;
    const unsigned char* __restrict__ S = mat ? SU : SG;
    float* __restrict__ Y = mat ? YU : YG;
    const float alpha = mat ? au : ag;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    // prefetch group 0 weights
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int nxt0 = g0 + KI;
        __syncthreads();
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
        if (nxt0 < NG) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16);
                xsv[c] = sxsc[tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * rows + row] = acc[r][c] * alpha; }
    }
}

template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_rbpf(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
        }
        if (nxt0 < NG) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16);
                xsv[c] = sxsc[tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rows + row]); Y[(size_t)t * rows + row] = v; } }
    }
}

extern "C" int q27_wide_gu_rbpf(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                                float* yg, float* yu, int M, int tcb, hipStream_t s) {
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || (wg->K % 128) || M < 1) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((wg->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 2u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_gu_rbpf<8,4>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    else          hipLaunchKernelGGL((q27_k_wide_gu_rbpf<8,8>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    return 1;
}

template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_b_rbpf(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, unsigned short* __restrict__ Y, int ystride)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
        }
        if (nxt0 < NG) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16);
                xsv[c] = sxsc[tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * ystride + row] = q27_f2bf(acc[r][c] * alpha); }
    }
}

template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_b_rb2p(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, unsigned short* __restrict__ Y, int ystride)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(nxt0 + g, NG - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
                sxsc[cur ^ 1][it] = (nxt0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * ystride + row] = q27_f2bf(acc[r][c] * alpha); }
    }
}

extern "C" int q27_wide_down_b_rb2p(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                                    unsigned short* y, int ystride, int M, int tcb, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_b_rb2p<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, ystride);
    else          hipLaunchKernelGGL((q27_k_wide_down_b_rb2p<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, ystride);
    return 1;
}

extern "C" int q27_wide_down_b_rbpf(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                                    unsigned short* y, int ystride, int M, int tcb, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_b_rbpf<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, ystride);
    else          hipLaunchKernelGGL((q27_k_wide_down_b_rbpf<8,8>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, ystride);
    return 1;
}

extern "C" int q27_wide_down_rbpf(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                  const unsigned short* radd, float rs, int tcb, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_rbpf<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_down_rbpf<8,8>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    return 1;
}

// rb2p: prefetched weights (rbpf) PLUS double-buffered activation staging -- one
// __syncthreads per K-group instead of two, and the next group's global act loads overlap the
// current group's dot4 chain. TCB=4 only (the 2-buffer LDS fits 2 blocks/CU at that tile).
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_gu_rb2p(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float ag, float au, int rows, int K, int M, float* __restrict__ YG, float* __restrict__ YU)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int mat = blockIdx.z;
    const unsigned char* __restrict__ W = mat ? WU : WG;
    const unsigned char* __restrict__ S = mat ? SU : SG;
    float* __restrict__ Y = mat ? YU : YG;
    const float alpha = mat ? au : ag;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI;
                const int tt = min(tok_base + t, M - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(nxt0 + g) * 16);
                sxsc[cur ^ 1][it] = Xs[(size_t)tt * NG + nxt0 + g] * 0.5f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * rows + row] = acc[r][c] * alpha; }
    }
}

template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_rb2p(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(nxt0 + g, NG - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
                sxsc[cur ^ 1][it] = (nxt0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rows + row]); Y[(size_t)t * rows + row] = v; } }
    }
}

// Fused Q/K/V bf16 rb2p: one launch covers all three attention projections via grid.z
// (q rows 1536, k/v rows 256). Blocks whose row tile is entirely past their matrix's rows
// exit immediately, so the small k/v grids stop paying separate launch/sequencing overhead.
template <int KI>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_b_rb2p3(
        const unsigned char* __restrict__ WQ, const unsigned char* __restrict__ SQ,
        const unsigned char* __restrict__ WK, const unsigned char* __restrict__ SK,
        const unsigned char* __restrict__ WV, const unsigned char* __restrict__ SV,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float aq, float ak, float av, int K, int M,
        unsigned short* __restrict__ Y0, unsigned short* __restrict__ Y1, unsigned short* __restrict__ Y2, int ystride)
{
    constexpr int TR = 64, TCB = 4, TC = 64, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int z = blockIdx.z;
    const unsigned char* __restrict__ W = (z == 0) ? WQ : (z == 1) ? WK : WV;
    const unsigned char* __restrict__ S = (z == 0) ? SQ : (z == 1) ? SK : SV;
    unsigned short* __restrict__ Y = (z == 0) ? Y0 : (z == 1) ? Y1 : Y2;
    const float alpha = (z == 0) ? aq : (z == 1) ? ak : av;
    const int rows = (z == 0) ? Q27_QROWS : Q27_KVROWS;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    if (row_base >= rows) return;                    // k/v tiles beyond 4 rows: instant exit
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI;
                const int tt = min(tok_base + t, M - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(nxt0 + g) * 16);
                sxsc[cur ^ 1][it] = Xs[(size_t)tt * NG + nxt0 + g] * 0.5f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * ystride + row] = q27_f2bf(acc[r][c] * alpha); }
    }
}

// split-K bf16 reduce: y[i] = f2bf(a[i] + b[i]) (+ optional bf16 residual).
__global__ void q27_k_red2_bf16(unsigned short* __restrict__ y, const float* __restrict__ a,
                                const float* __restrict__ b, const unsigned short* __restrict__ r, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = a[i] + b[i]; if (r) v += q27_bf2f(r[i]); y[i] = q27_f2bf(v); }
}
extern "C" void q27_red2_bf16(unsigned short* y, const float* a, const float* b,
                              const unsigned short* r, int n, hipStream_t s) {
    hipLaunchKernelGGL(q27_k_red2_bf16, dim3((n + 255) / 256), dim3(256), 0, s, y, a, b, r, n);
}
// fp32 split-K reduce (for the out_proj, whose consumer expects fp32): y[i] = a[i]+b[i] (+bf16 r).
__global__ void q27_k_red2_f32(float* __restrict__ y, const float* __restrict__ a,
                               const float* __restrict__ b, const unsigned short* __restrict__ r, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = a[i] + b[i]; if (r) v += q27_bf2f(r[i]); y[i] = v; }
}
extern "C" void q27_red2_f32(float* y, const float* a, const float* b,
                             const unsigned short* r, int n, hipStream_t s) {
    hipLaunchKernelGGL(q27_k_red2_f32, dim3((n + 255) / 256), dim3(256), 0, s, y, a, b, r, n);
}

extern "C" int q27_wide_down_b_rb2p3(const q27_nvfp4_t* wq, const q27_nvfp4_t* wk, const q27_nvfp4_t* wv,
                                     const signed char* xq, const float* xs,
                                     unsigned short* y0, unsigned short* y1, unsigned short* y2, int ystride, int M,
                                     hipStream_t s) {
    if (!wq || !wk || !wv || (wq->K % 16) || M < 1) return 0;
    const float aq = wq->in_scale * wq->ws2, ak = wk->in_scale * wk->ws2, av = wv->in_scale * wv->ws2;
    const dim3 gr((unsigned)((Q27_QROWS + 63) / 64), (unsigned)((M + 63) / 64), 3u);
    hipLaunchKernelGGL((q27_k_wide_down_b_rb2p3<8>), gr, dim3(256), 0, s,
                       wq->w, wq->gs, wk->w, wk->gs, wv->w, wv->gs, xq, xs, aq, ak, av, wq->K, M, y0, y1, y2, ystride);
    return 1;
}

// gu rb3p: barrier-free register-pipelined gate+up. Both operands prefetched one K-group
// ahead (weights decoded in-register, activations loaded directly -- L2-hot from the fused
// tail). No LDS, no barriers: the K-loop (320 groups, shorter than down's 1088) relies on
// inter-block overlap for latency hiding instead of in-block syncs.
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_gu_rb3p(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float ag, float au, int rows, int K, int M, float* __restrict__ YG, float* __restrict__ YU)
{
    constexpr int TR = 64, TC = 16 * TCB;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int mat = blockIdx.z;
    const unsigned char* __restrict__ W = mat ? WU : WG;
    const unsigned char* __restrict__ S = mat ? SU : SG;
    float* __restrict__ Y = mat ? YU : YG;
    const float alpha = mat ? au : ag;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
    uint4 xr0[TCB], xr1[TCB]; float xsv0[TCB], xsv1[TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
#pragma unroll
    for (int c = 0; c < TCB; ++c) {
        const int tt = tok_base + ty * TCB + c;
        xr0[c] = *(const uint4*)(Xq + (size_t)tt * K + 0);
        xsv0[c] = Xs[(size_t)tt * NG + 0] * 0.5f;
    }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int nxt0 = g0 + KI;
        if (nxt0 < NG) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tt = tok_base + ty * TCB + c;
                xr1[c] = *(const uint4*)(Xq + (size_t)tt * K + (size_t)nxt0 * 16);
                xsv1[c] = Xs[(size_t)tt * NG + nxt0] * 0.5f;
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            const int kk = g0 + g;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tt = tok_base + ty * TCB + c;
                xr[c] = *(const uint4*)(Xq + (size_t)tt * K + (size_t)kk * 16);
                xsv[c] = Xs[(size_t)tt * NG + kk] * 0.5f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
#pragma unroll
        for (int c = 0; c < TCB; ++c) { xr0[c] = xr1[c]; xsv0[c] = xsv1[c]; }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * rows + row] = acc[r][c] * alpha; }
    }
}

extern "C" int q27_wide_gu_rb3p(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                                float* yg, float* yu, int M, int tcb, hipStream_t s) {
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || (wg->K % 128) || M < 1) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((wg->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 2u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_gu_rb3p<8,4>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    else          hipLaunchKernelGGL((q27_k_wide_gu_rb3p<8,4>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    return 1;
}

// rb2p-sk2: the rb2p down with K split across grid.z (2 halves). Each half stores its
// fp32 partial into its own output buffer (plain stores, no atomics); the residual folds into
// the z=0 half only; one tiny add kernel merges the halves afterwards. Doubles the block
// population (160 -> 320) to shrink the 1.33-wave tail.
template <int KI>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_rb2p_sk2(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y0, float* __restrict__ Y1,
        float* __restrict__ Y2, float* __restrict__ Y3, int khalf,
        const unsigned short* __restrict__ radd, float rs)
{
    constexpr int TR = 64, TCB = 4, TC = 64, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    const int k0 = blockIdx.z * khalf, k1 = k0 + khalf;
    float* __restrict__ Y = (blockIdx.z == 0) ? Y0 : (blockIdx.z == 1) ? Y1 : (blockIdx.z == 2) ? Y2 : Y3;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)k0 * 8);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + k0]);
    }
    {
        const int g0 = k0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = k0; g0 < k1; g0 += KI) {
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < k1) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI;
                const int tt = min(tok_base + t, M - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(nxt0 + g) * 16);
                sxsc[cur ^ 1][it] = Xs[(size_t)tt * NG + nxt0 + g] * 0.5f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
    const float zrs = blockIdx.z ? 0.f : rs;         // the residual folds into z=0 only
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) {
            const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (zrs && radd) v += zrs * q27_bf2f(radd[(size_t)t * rows + row]); Y[(size_t)t * rows + row] = v; }
        }
    }
}

__global__ void q27_k_add_f32(float* __restrict__ a, const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}
extern "C" void q27_add_f32(float* a, const float* b, int n, hipStream_t s) {
    hipLaunchKernelGGL(q27_k_add_f32, dim3((n + 255) / 256), dim3(256), 0, s, a, b, n);
}

extern "C" int q27_wide_down_rb2p_sk2(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                                      float* y0, float* y1, float* y2, float* y3, int splits, int M,
                                      const unsigned short* radd, float rs, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1 || !y1 || !y2 || !y3 || (splits != 2 && splits != 4)) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int khalf = (w->K >> 4) / splits;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + 63) / 64), (unsigned)splits);
    hipLaunchKernelGGL((q27_k_wide_down_rb2p_sk2<8>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y0, y1, y2, y3, khalf, radd, rs);
    return 1;
}

// TR=32 down specialization: the rb2p architecture with a 32-row tile and 128-thread
// blocks. Doubles grid.x (160 tiles) at the same block count per M-tile (2), so the block
// population doubles (160 -> 320): the 1.33-wave tail becomes 2.67 waves and the wasted
// partial wave shrinks ~4x in slot-time. 4 rows x 4 tokens per thread, same dot4 pairing.
template <int KI>
__global__ __launch_bounds__(128, 2) void q27_k_wide_down_rb2p_tr32(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    constexpr int TR = 32, TCB = 4, TC = 64, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 7, ty = tid >> 3;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 128) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 128) {
                const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(nxt0 + g, NG - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
                sxsc[cur ^ 1][it] = (nxt0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rows + row]); Y[(size_t)t * rows + row] = v; } }
    }
}

extern "C" int q27_wide_down_rb2p_tr32(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                       const unsigned short* radd, float rs, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 gr((unsigned)((w->rows + 31) / 32), (unsigned)((M + 63) / 64), 1u);
    hipLaunchKernelGGL((q27_k_wide_down_rb2p_tr32<8>), gr, dim3(128), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    return 1;
}

extern "C" int q27_wide_down_rb2p(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                  const unsigned short* radd, float rs, int tcb, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_rb2p<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_down_rb2p<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    return 1;
}

extern "C" int q27_wide_gu_rb2p(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                                float* yg, float* yu, int M, int tcb, hipStream_t s) {
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || (wg->K % 128) || M < 1) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((wg->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 2u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_gu_rb2p<8,4>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    else          hipLaunchKernelGGL((q27_k_wide_gu_rb2p<8,4>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    return 1;
}

// W8-BYPASS rb GEMM: weights come DIRECTLY from the load-time int8 mirror (q27_nvfp4_w8) --
// no LDS staging, no nibble decode, no weight barrier, 16-B nontemporal loads whose latency is
// hidden behind the activation staging + sync. Activations stay in LDS (single buffer, the same
// 2-sync structure as rb). Same dot4 pairing and accumulate order as rb: every partial sum is
// bit-identical.
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_gu_w8(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float ag, float au, int rows, int K, int M, float* __restrict__ YG, float* __restrict__ YU)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int mat = blockIdx.z;
    const unsigned char* __restrict__ W = mat ? WU : WG;
    const unsigned char* __restrict__ S = mat ? SU : SG;
    float* __restrict__ Y = mat ? YU : YG;
    const float alpha = mat ? au : ag;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, ntiles = rows >> 6;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            uint4 wr[4]; float ws[4]; uint4 xr[TCB]; float xsv[TCB];
            const int kk = g0 + g;
            const unsigned char* wbase = W + ((size_t)((size_t)kk * ntiles + blockIdx.x) * 16 + tx) * 64;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                wr[r] = *(const uint4*)(wbase + r * 16);
                ws[r] = q27_e4m3(S[(size_t)row * NG + kk]);
            }
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16);
                xsv[c] = sxsc[tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr[r].x, xr[c].x, 0); d = q27_d4(wr[r].y, xr[c].z, d);
                    d = q27_d4(wr[r].z, xr[c].y, d); d = q27_d4(wr[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws[r] * xsv[c], acc[r][c]);
                }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * rows + row] = acc[r][c] * alpha; }
    }
}

template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_w8(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const unsigned short* __restrict__ radd, float rs)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, ntiles = rows >> 6;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 wr[4]; float ws[4]; uint4 xr[TCB]; float xsv[TCB];
            const int kk = g0 + g;
            const unsigned char* wbase = W + ((size_t)((size_t)kk * ntiles + blockIdx.x) * 16 + tx) * 64;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                wr[r] = *(const uint4*)(wbase + r * 16);
                ws[r] = q27_e4m3(S[(size_t)row * NG + kk]);
            }
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16);
                xsv[c] = sxsc[tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr[r].x, xr[c].x, 0); d = q27_d4(wr[r].y, xr[c].z, d);
                    d = q27_d4(wr[r].z, xr[c].y, d); d = q27_d4(wr[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws[r] * xsv[c], acc[r][c]);
                }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * q27_bf2f(radd[(size_t)t * rows + row]); Y[(size_t)t * rows + row] = v; } }
    }
}

extern "C" int q27_wide_gu_w8(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                              float* yg, float* yu, int M, int tcb, hipStream_t s) {
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || (wg->K % 128) || M < 1) return 0;
    const unsigned char* g8 = q27_nvfp4_w8s(wg); const unsigned char* u8 = q27_nvfp4_w8s(wu);
    if (!g8 || !u8) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((wg->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 2u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_gu_w8<8,4>), gr, dim3(256), 0, s, g8, wg->gs, u8, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    else          hipLaunchKernelGGL((q27_k_wide_gu_w8<8,8>), gr, dim3(256), 0, s, g8, wg->gs, u8, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    return 1;
}

extern "C" int q27_wide_down_w8(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                const unsigned short* radd, float rs, int tcb, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1) return 0;
    const unsigned char* w8 = q27_nvfp4_w8s(w);
    if (!w8) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int TCv = 16 * tcb;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_w8<8,4>), gr, dim3(256), 0, s, w8, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_down_w8<8,8>), gr, dim3(256), 0, s, w8, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    return 1;
}

} // namespace

// gate+up over M tokens: grid (rows/128, M/64, 2). ki = 8 (30 KB LDS, two blocks per CU) or 16 (58 KB, one).
extern "C" int q27_wide_gu(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                           float* yg, float* yu, int M, int ki, hipStream_t s) {
    // ki encodes KI*10 + NW: e.g. 88 = KI 8 / 8 waves (512 threads), 84 = KI 8 / 4 waves (256), 168 = KI 16 / 8 waves
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || (wg->K % 128) || M < 1) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 g((unsigned)((wg->rows + 127) / 128), (unsigned)((M + 63) / 64), 2u);
    #define GU(KI,NW,MINW) hipLaunchKernelGGL((q27_k_wide_gu<KI,NW,MINW>), g, dim3(NW*64), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu)
    if (ki == 84)  { GU(8,4,1);  return 1; }
    if (ki == 88)  { GU(8,8,1);  return 1; }
    if (ki == 164) { GU(16,4,1); return 1; }
    if (ki == 168) { GU(16,8,1); return 1; }
    if (ki == 8000 || ki == 16000) {
        static const int tcb0 = q27_env_int("Q27_PF_TCB", 4);
        const int TCv = 16 * tcb0;                       // grid.y MUST follow TC or tokens are dropped
        const dim3 gr((unsigned)((wg->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 2u);
        // TCB = tokens per thread. TC = 16*TCB, and TC drives the LDS activation tile, which is
        // 57% of the 32 KB budget -- at TCB=8 only 2 blocks/CU fit (8 waves of 40 = 20% occupancy,
        // 2 per SIMD), so nothing hides LDS/global latency. TCB=4 cuts sx and sxsc and halves the
        // accumulator, which should fit 3-4 blocks. Q27_PF_TCB selects it.
        static const int tcb = q27_env_int("Q27_PF_TCB", 4);
        if (ki == 16000) { if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_gu_rb<16,4>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
                           else          hipLaunchKernelGGL((q27_k_wide_gu_rb<16,8>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu); }
        else             { if (tcb == 2) hipLaunchKernelGGL((q27_k_wide_gu_rb<8,2>),  gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
                           else if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_gu_rb<8,4>),  gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
                           else          hipLaunchKernelGGL((q27_k_wide_gu_rb<8,8>),  gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu); }
        return 1; }
    GU(8,8,1); return 1;
    #undef GU
}
extern "C" int q27_wide_down(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                             const unsigned short* radd, float rs, int ki, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64), 1u);
    #define DN(KI,NW,MINW) hipLaunchKernelGGL((q27_k_wide_down<KI,NW,MINW>), g, dim3(NW*64), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs)
    if (ki == 84)  { DN(8,4,1);  return 1; }
    if (ki == 88)  { DN(8,8,1);  return 1; }
    if (ki == 164) { DN(16,4,1); return 1; }
    if (ki == 168) { DN(16,8,1); return 1; }
    if (ki == 8000 || ki == 16000) {
        static const int tcb0 = q27_env_int("Q27_PF_TCB", 4);
        const int TCv = 16 * tcb0;
        const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
        static const int tcb = q27_env_int("Q27_PF_TCB", 4);
        if (ki == 16000) { if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_rb<16,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
                           else          hipLaunchKernelGGL((q27_k_wide_down_rb<16,8>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs); }
        else             { if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_rb<8,4>),  gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
                           else          hipLaunchKernelGGL((q27_k_wide_down_rb<8,8>),  gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs); }
        return 1; }
    DN(8,8,1); return 1;
    #undef DN
}
// bf16-output NVFP4 wide GEMM for the layer-split projection sites (q/k/v, GDN in_qkv/in_z).
extern "C" int q27_wide_down_b(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                               unsigned short* y, int ystride, int M, int ki, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64), 1u);
    #define DNB(KI,NW,MINW) hipLaunchKernelGGL((q27_k_wide_down_b<KI,NW,MINW>), g, dim3(NW*64), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, ystride)
    if (ki == 84)  { DNB(8,4,1);  return 1; }
    if (ki == 88)  { DNB(8,8,1);  return 1; }
    if (ki == 164) { DNB(16,4,1); return 1; }
    if (ki == 168) { DNB(16,8,1); return 1; }
    DNB(8,8,1); return 1;
    #undef DNB
}

// ---- census: wide kernels vs the rung-3 kernels (b13) on the real per-card shapes, M = 64 / 256 / 1024 ----
extern "C" int q27_proj_nvfp4_gu_b13(const q27_nvfp4_t*, const q27_nvfp4_t*, const signed char*, const float*, float*, float*, int, int, hipStream_t);
extern "C" int q27_proj_nvfp4_bt13(const q27_nvfp4_t*, const signed char*, const float*, float*, int, int, hipStream_t);

static void wide_fill(unsigned char* h, size_t n, unsigned seed) { unsigned s = seed; for (size_t i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; h[i] = (unsigned char)(s >> 24); } }
static void wide_fill_scales(unsigned char* h, size_t n, unsigned seed) { unsigned s = seed; for (size_t i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; h[i] = (unsigned char)(0x30 + ((s >> 24) & 0x1F)); } }  // e4m3 in a sane range
static void wide_compare(const char* tag, const float* ref, const float* got, size_t n, double* out) {
    double mx = 0, sum = 0, mxrel = 0; size_t same = 0, cnt = 0; double refabs = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)ref[i] - (double)got[i]); const double r = std::fabs((double)ref[i]);
        if (d > mx) mx = d; sum += d; refabs += r; if (ref[i] == got[i]) ++same; ++cnt;
        const double rel = d / (r > 1e-3 ? r : 1e-3); if (rel > mxrel) mxrel = rel;
    }
    std::printf("    %-22s max|d| %.3e  mean|d| %.3e  mean|ref| %.3e  max rel %.3e  bit-identical %.2f%%\n",
                tag, mx, sum / cnt, refabs / cnt, mxrel, 100.0 * same / cnt);
    if (out) { out[0] = mx; out[1] = sum / cnt; out[2] = mxrel; }
}
// One name per ki code, exhaustive. The old inline ternary mapped 84, 164 AND 168 to the same
// string "KI16NW8", so the down census printed three different configs under one label and the
// winner was transcribed as ki=88. A label that can be silently wrong is worse than no label.
static const char* wide_cfg_name(int ki) {
    switch (ki) {
        case 84:    return "row KI8 NW4";
        case 88:    return "row KI8 NW8";
        case 164:   return "row KI16 NW4";
        case 168:   return "row KI16 NW8";
        case 8000:  return "rb  KI8";
        case 16000: return "rb  KI16";
    }
    return "UNKNOWN";
}
#define WCK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { std::printf("  WIDE_BENCH hip error %d at line %d\n", (int)_e, __LINE__); return; } } while (0)
extern "C" void q27_wide_bench(void) {
    const int REPS = 20;
    hipEvent_t e0, e1; WCK(hipEventCreate(&e0)); WCK(hipEventCreate(&e1));
    auto timeit = [&](auto&& fn) -> double {
        fn(); (void)hipDeviceSynchronize();
        (void)hipEventRecord(e0, 0); for (int i = 0; i < REPS; ++i) fn();
        (void)hipEventRecord(e1, 0); (void)hipEventSynchronize(e1);
        float ms = 0; if (hipEventElapsedTime(&ms, e0, e1) != hipSuccess) return -1.0; return ms * 1000.0 / REPS;
    };
    const int MMAX = 1024;
    // ---- gate+up: rows 4352 (17408/4), K 5120 ----
    {
        const int rows = 4352, K = 5120, NG = K / 16;
        std::vector<unsigned char> hw((size_t)rows * K / 2), hs((size_t)rows * NG), hx((size_t)MMAX * K);
        std::vector<float> hxs((size_t)MMAX * NG);
        wide_fill(hw.data(), hw.size(), 1); wide_fill_scales(hs.data(), hs.size(), 2); wide_fill(hx.data(), hx.size(), 3);
        for (size_t i = 0; i < hxs.size(); ++i) hxs[i] = 0.01f + 0.05f * ((i * 2654435761u) % 97) / 97.0f;
        unsigned char *dWg, *dWu, *dSg, *dSu; signed char* dX; float *dXs, *dYg, *dYu, *dRg, *dRu;
        WCK(hipMalloc((void**)&dWg, hw.size())); WCK(hipMalloc((void**)&dWu, hw.size()));
        WCK(hipMalloc((void**)&dSg, hs.size())); WCK(hipMalloc((void**)&dSu, hs.size()));
        WCK(hipMalloc((void**)&dX, hx.size())); WCK(hipMalloc((void**)&dXs, hxs.size() * 4));
        const size_t ybytes = (size_t)MMAX * rows * 4;
        WCK(hipMalloc((void**)&dYg, ybytes)); WCK(hipMalloc((void**)&dYu, ybytes)); WCK(hipMalloc((void**)&dRg, ybytes)); WCK(hipMalloc((void**)&dRu, ybytes));
        WCK(hipMemcpy(dWg, hw.data(), hw.size(), hipMemcpyHostToDevice));
        wide_fill(hw.data(), hw.size(), 7); WCK(hipMemcpy(dWu, hw.data(), hw.size(), hipMemcpyHostToDevice));
        WCK(hipMemcpy(dSg, hs.data(), hs.size(), hipMemcpyHostToDevice));
        wide_fill_scales(hs.data(), hs.size(), 9); WCK(hipMemcpy(dSu, hs.data(), hs.size(), hipMemcpyHostToDevice));
        WCK(hipMemcpy(dX, hx.data(), hx.size(), hipMemcpyHostToDevice)); WCK(hipMemcpy(dXs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
        q27_nvfp4_t wg{}; wg.w = dWg; wg.gs = dSg; wg.ws2 = 1.f; wg.in_scale = 1.f; wg.rows = rows; wg.K = K;
        q27_nvfp4_t wu = wg; wu.w = dWu; wu.gs = dSu;
        std::printf("Q27_WIDE_BENCH gate+up  rows=%d K=%d (per card)  [us | TMAC/s per card | vs rung-3 b13 at the same M]\n", rows, K);
        for (int M : {64, 256, 1024}) {
            const double macs = 2.0 * rows * (double)K * M;
            const double tref = timeit([&]() { for (int t = 0; t < M; t += 8) q27_proj_nvfp4_gu_b13(&wg, &wu, dX + (size_t)t * K, dXs + (size_t)t * NG, dRg + (size_t)t * rows, dRu + (size_t)t * rows, 8, 5, 0); });
            std::printf("  M=%4d  rung-3 b13 x%d launches  %9.1f us  %6.2f TMAC/s\n", M, M / 8, tref, macs / tref / 1e6);
            std::vector<float> rg((size_t)M * rows), ru((size_t)M * rows), gg((size_t)M * rows), gu((size_t)M * rows);
            WCK(hipMemcpy(rg.data(), dRg, rg.size() * 4, hipMemcpyDeviceToHost)); WCK(hipMemcpy(ru.data(), dRu, ru.size() * 4, hipMemcpyDeviceToHost));
            for (int ki : {88, 168, 8000, 16000}) {
                WCK(hipMemset(dYg, 0, ybytes)); WCK(hipMemset(dYu, 0, ybytes));
                const double t = timeit([&]() { q27_wide_gu(&wg, &wu, dX, dXs, dYg, dYu, M, ki, 0); });
                std::printf("  M=%4d  wide %-12s %9.1f us  %6.2f TMAC/s  speedup %.2fx\n", M, wide_cfg_name(ki), t, macs / t / 1e6, tref / t);
                WCK(hipMemcpy(gg.data(), dYg, gg.size() * 4, hipMemcpyDeviceToHost)); WCK(hipMemcpy(gu.data(), dYu, gu.size() * 4, hipMemcpyDeviceToHost));
                wide_compare("gate vs b13", rg.data(), gg.data(), rg.size(), nullptr);
                wide_compare("up   vs b13", ru.data(), gu.data(), ru.size(), nullptr);
            }
        }
        hipFree(dWg); hipFree(dWu); hipFree(dSg); hipFree(dSu); hipFree(dX); hipFree(dXs); hipFree(dYg); hipFree(dYu); hipFree(dRg); hipFree(dRu);
    }
    // ---- down: rows 5120, K 4352 (17408/4) ----
    {
        const int rows = 5120, K = 4352, NG = K / 16;
        std::vector<unsigned char> hw((size_t)rows * K / 2), hs((size_t)rows * NG), hx((size_t)MMAX * K);
        std::vector<float> hxs((size_t)MMAX * NG);
        wide_fill(hw.data(), hw.size(), 11); wide_fill_scales(hs.data(), hs.size(), 12); wide_fill(hx.data(), hx.size(), 13);
        for (size_t i = 0; i < hxs.size(); ++i) hxs[i] = 0.01f + 0.05f * ((i * 2246822519u) % 89) / 89.0f;
        unsigned char *dW, *dS; signed char* dX; float *dXs, *dY, *dR;
        WCK(hipMalloc((void**)&dW, hw.size())); WCK(hipMalloc((void**)&dS, hs.size()));
        WCK(hipMalloc((void**)&dX, hx.size())); WCK(hipMalloc((void**)&dXs, hxs.size() * 4));
        const size_t ybytes = (size_t)MMAX * rows * 4;
        WCK(hipMalloc((void**)&dY, ybytes)); WCK(hipMalloc((void**)&dR, ybytes));
        WCK(hipMemcpy(dW, hw.data(), hw.size(), hipMemcpyHostToDevice)); WCK(hipMemcpy(dS, hs.data(), hs.size(), hipMemcpyHostToDevice));
        WCK(hipMemcpy(dX, hx.data(), hx.size(), hipMemcpyHostToDevice)); WCK(hipMemcpy(dXs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
        q27_nvfp4_t w{}; w.w = dW; w.gs = dS; w.ws2 = 1.f; w.in_scale = 1.f; w.rows = rows; w.K = K;
        std::printf("Q27_WIDE_BENCH down     rows=%d K=%d (per card)\n", rows, K);
        for (int M : {64, 256, 1024}) {
            const double macs = (double)rows * K * M;
            const double tref = timeit([&]() { for (int t = 0; t < M; t += 8) q27_proj_nvfp4_bt13(&w, dX + (size_t)t * K, dXs + (size_t)t * NG, dR + (size_t)t * rows, 8, 6, 0); });
            std::printf("  M=%4d  rung-3 b13 x%d launches  %9.1f us  %6.2f TMAC/s\n", M, M / 8, tref, macs / tref / 1e6);
            std::vector<float> r((size_t)M * rows), gsum((size_t)M * rows);
            WCK(hipMemcpy(r.data(), dR, r.size() * 4, hipMemcpyDeviceToHost));
            // The register-blocked forms exist for down (q27_k_wide_down_rb, ki 8000/16000) and were
            // never swept: the first census ran only the four row-major codes, while rb KI8 beat row-major
            // KI16 by 15% on gate+up at M=1024. The down cell has to be measured, not assumed.
            for (int ki : {84, 88, 164, 168, 8000, 16000}) {
                WCK(hipMemset(dY, 0, ybytes));
                const double t = timeit([&]() { q27_wide_down(&w, dX, dXs, dY, M, nullptr, 0.f, ki, 0); });
                std::printf("  M=%4d  wide %-12s %9.1f us  %6.2f TMAC/s  speedup %.2fx\n", M, wide_cfg_name(ki), t, macs / t / 1e6, tref / t);
                WCK(hipMemcpy(gsum.data(), dY, gsum.size() * 4, hipMemcpyDeviceToHost));
                wide_compare("down vs b13", r.data(), gsum.data(), r.size(), nullptr);
            }
        }
        hipFree(dW); hipFree(dS); hipFree(dX); hipFree(dXs); hipFree(dY); hipFree(dR);
    }
}

// ============================================================================================
// Q27_LS_Q8 -- the quantized producer/consumer graph (2026-09-11). The projection producers emit
// the representation their consumers execute on: int8 + one fp32 scale per 16 output rows,
// [token][row] / [token][row/16]. No bf16 materialization. The dq4/dot4 main loop is rb2p's;
// only the epilogue differs. A 16-row scale group is the 4 consecutive lanes tx&~3..tx|3 of
// one ty, each holding 4 rows, so the group max is two 4-wide shuffles.
// ============================================================================================
template <int TCB>
__device__ __forceinline__ void q27_q8_epilogue(float (&acc)[4][TCB], float alpha, int rows, int M,
        int row_base, int tok_base, int tx, int ty,
        signed char* __restrict__ Yq, float* __restrict__ Ys, int ystride, int sstride)
{
    const int row0 = row_base + tx * 4;
    const int grp  = row0 >> 4;
#pragma unroll
    for (int c = 0; c < TCB; ++c) {
        float v[4];
        float mx = 0.f;
#pragma unroll
        for (int r = 0; r < 4; ++r) { v[r] = acc[r][c] * alpha; mx = fmaxf(mx, fabsf(v[r])); }
        mx = fmaxf(mx, __shfl_down(mx, 1, 4));
        mx = fmaxf(mx, __shfl_down(mx, 2, 4));
        mx = __shfl(mx, 0, 4);
        const float sc = mx * (1.f / 127.f);
        const float qi = (sc > 0.f) ? (1.f / sc) : 0.f;
        const int t = tok_base + ty * TCB + c;
        if (t < M && row0 < rows) {
            unsigned packed = 0u;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                int cq = (int)rintf(v[r] * qi);
                cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
                packed |= ((unsigned)cq & 0xFFu) << (8 * r);
            }
            *(unsigned*)(Yq + (size_t)t * ystride + row0) = packed;
            if ((tx & 3) == 0) Ys[(size_t)t * sstride + grp] = sc;
        }
    }
}

// single matrix, int8+scale epilogue (GDN in_qkv / in_z producers)
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_q8_rb2p(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M,
        signed char* __restrict__ Yq, float* __restrict__ Ys, int ystride, int sstride)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = min(row_base + tx * 4 + r, rows - 1);
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(nxt0 + g, NG - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
                sxsc[cur ^ 1][it] = (nxt0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = min(row_base + tx * 4 + r, rows - 1);
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
    q27_q8_epilogue<TCB>(acc, alpha, rows, M, row_base, tok_base, tx, ty, Yq, Ys, ystride, sstride);
}

// fused q/k/v (grid.z), int8+scale epilogue; k/v live at row offsets inside the same token row
template <int KI>
__global__ __launch_bounds__(256, 2) void q27_k_wide_q8_rb2p3(
        const unsigned char* __restrict__ WQ, const unsigned char* __restrict__ SQ,
        const unsigned char* __restrict__ WK, const unsigned char* __restrict__ SK,
        const unsigned char* __restrict__ WV, const unsigned char* __restrict__ SV,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float aq, float ak, float av, int K, int M,
        signed char* __restrict__ Yq0, signed char* __restrict__ Yq1, signed char* __restrict__ Yq2,
        float* __restrict__ Ys0, float* __restrict__ Ys1, float* __restrict__ Ys2, int ystride, int sstride)
{
    constexpr int TR = 64, TCB = 4, TC = 64, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int z = blockIdx.z;
    const unsigned char* __restrict__ W = (z == 0) ? WQ : (z == 1) ? WK : WV;
    const unsigned char* __restrict__ S = (z == 0) ? SQ : (z == 1) ? SK : SV;
    signed char* __restrict__ Yq = (z == 0) ? Yq0 : (z == 1) ? Yq1 : Yq2;
    float* __restrict__ Ys = (z == 0) ? Ys0 : (z == 1) ? Ys1 : Ys2;
    const float alpha = (z == 0) ? aq : (z == 1) ? ak : av;
    const int rows = (z == 0) ? Q27_QROWS : Q27_KVROWS;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    if (row_base >= rows) return;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r;
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI;
                const int tt = min(tok_base + t, M - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(nxt0 + g) * 16);
                sxsc[cur ^ 1][it] = Xs[(size_t)tt * NG + nxt0 + g] * 0.5f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = row_base + tx * 4 + r;
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
    q27_q8_epilogue<TCB>(acc, alpha, rows, M, row_base, tok_base, tx, ty, Yq, Ys, ystride, sstride);
}

// fp32 output + fp32 residual (the residual stream never leaves fp32: o_proj/out_proj read the
// layer input, down reads the post-attention vector, both straight from their fp32 buffers)
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_rb2p_f32r(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const float* __restrict__ radd, float rs)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[4], wr1[4]; float ws0[4], ws1[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = min(row_base + tx * 4 + r, rows - 1);
        const uint2 v = *(const uint2*)(W + (size_t)row * KP + 0);
        wr0[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr0[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
        wr0[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr0[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
        ws0[r] = q27_e4m3(S[(size_t)row * NG + 0]);
    }
    {
        const int g0 = 0;
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            *(uint4*)(sx[0] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)(g0 + g) * 16);
            sxsc[0][it] = Xs[(size_t)tt * NG + g0 + g] * 0.5f;
        }
    }
    int cur = 0;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        const int nxt0 = g0 + KI;
        __syncthreads();
        if (nxt0 < NG) {
            for (int it = tid; it < TC * KI; it += 256) {
                const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(nxt0 + g, NG - 1);
                *(uint4*)(sx[cur ^ 1] + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
                sxsc[cur ^ 1][it] = (nxt0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = min(row_base + tx * 4 + r, rows - 1);
                const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)nxt0 * 8);
                wr1[r].x = q27_dq4( v.x       & 0x0F0F0F0Fu); wr1[r].y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
                wr1[r].z = q27_dq4( v.y       & 0x0F0F0F0Fu); wr1[r].w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
                ws1[r] = q27_e4m3(S[(size_t)row * NG + nxt0]);
            }
        }
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint4*)(sx[cur] + tc * XSTR + g * 16);
                xsv[c] = sxsc[cur][tc * KI + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[r].x, xr[c].x, 0); d = q27_d4(wr0[r].y, xr[c].z, d);
                    d = q27_d4(wr0[r].z, xr[c].y, d); d = q27_d4(wr0[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) { wr0[r] = wr1[r]; ws0[r] = ws1[r]; }
        cur ^= 1;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) {
            const int t = tok_base + ty * TCB + c;
            if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * radd[(size_t)t * rows + row]; Y[(size_t)t * rows + row] = v; }
        }
    }
}

extern "C" int q27_wide_q8_rb2p(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                                signed char* yq, float* ys, int ystride, int M, hipStream_t s) {
    if (!w || (w->K % 16) || (w->rows % 16) || (ystride % 16) || M < 1 || !xq || !xs || !yq || !ys) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + 63) / 64), 1u);
    hipLaunchKernelGGL((q27_k_wide_q8_rb2p<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M,
                       yq, ys, ystride, ystride >> 4);
    return 1;
}
extern "C" int q27_wide_q8_rb2p3(const q27_nvfp4_t* wq, const q27_nvfp4_t* wk, const q27_nvfp4_t* wv,
                                 const signed char* xq, const float* xs,
                                 signed char* yq0, signed char* yq1, signed char* yq2,
                                 float* ys0, float* ys1, float* ys2, int ystride, int M, hipStream_t s) {
    if (!wq || !wk || !wv || (wq->K % 16) || (ystride % 16) || M < 1 || !xq || !xs) return 0;
    if (wq->rows != Q27_QROWS || wk->rows != Q27_KVROWS || wv->rows != Q27_KVROWS) return 0;
    const float aq = wq->in_scale * wq->ws2, ak = wk->in_scale * wk->ws2, av = wv->in_scale * wv->ws2;
    const dim3 gr((unsigned)((Q27_QROWS + 63) / 64), (unsigned)((M + 63) / 64), 3u);
    hipLaunchKernelGGL((q27_k_wide_q8_rb2p3<8>), gr, dim3(256), 0, s,
                       wq->w, wq->gs, wk->w, wk->gs, wv->w, wv->gs, xq, xs, aq, ak, av, wq->K, M,
                       yq0, yq1, yq2, ys0, ys1, ys2, ystride, ystride >> 4);
    return 1;
}
extern "C" int q27_wide_down_rb2p_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                       const float* radd, float rs, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + 63) / 64), 1u);
    hipLaunchKernelGGL((q27_k_wide_down_rb2p_f32r<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    return 1;
}

// ============================================================================================
// Q27_LS_Q8 down: the validated row-major wide kernel (q27_k_wide_down, the ki 84/88/164/168
// forms) with an fp32 residual read straight from the residual stream. The rb2p family is NOT
// used on this path: measured against this kernel on the reference prompt (2026-09-11), the
// rb2p gate+up produced different pa/pb and wrong tokens.
// ============================================================================================
template <int KI, int NW, int MINW, int TR>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_down_f32r_row(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const float* __restrict__ radd, float rs)
{
    constexpr int TC = 64; static_assert(TR == 128 || TR == 64, "TR");
    constexpr int WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            const uint2 v = *(const uint2*)(W + (size_t)rr * KP + (size_t)gg * 8);
            uint4 o;
            o.x = q27_dq4( v.x       & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
            o.z = q27_dq4( v.y       & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
            *(uint4*)(sw + r * WSTR + g * 16) = o;
            swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
        }
        __syncthreads();
#pragma unroll 2
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = (TR == 128) ? *(const uint4*)(sw + (lane + 64) * WSTR + g * 16) : make_uint4(0u, 0u, 0u, 0u);
            const float s0 = q27_e4m3(swsc[lane * KI + g]);
            const float s1 = (TR == 128) ? q27_e4m3(swsc[(lane + 64) * KI + g]) : 0.f;
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 x = *(const uint4*)(sx + t * XSTR + g * 16);
                const float xs = sxsc[t * KI + g];
                int a = q27_d4(w0.x, x.x, 0); a = q27_d4(w0.y, x.z, a); a = q27_d4(w0.z, x.y, a); a = q27_d4(w0.w, x.w, a);
                int b = 0; if (TR == 128) { b = q27_d4(w1.x, x.x, 0); b = q27_d4(w1.y, x.z, b); b = q27_d4(w1.z, x.y, b); b = q27_d4(w1.w, x.w, b); }
                acc0[j] = fmaf((float)a, s0 * xs, acc0[j]);
                acc1[j] = fmaf((float)b, s1 * xs, acc1[j]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TPW; ++j) {
        const int t = tok_base + wave * TPW + j;
        if (t < M) {
            if (r0 < rows) { float v = acc0[j] * alpha; if (radd) v += rs * radd[(size_t)t * rows + r0]; Y[(size_t)t * rows + r0] = v; }
            if (TR == 128 && r1 < rows) { float v = acc1[j] * alpha; if (radd) v += rs * radd[(size_t)t * rows + r1]; Y[(size_t)t * rows + r1] = v; }
        }
    }
}

extern "C" int q27_wide_down_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                  const float* radd, float rs, int ki, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64), 1u);
    #define DNF(KI,NW,MINW,TR) hipLaunchKernelGGL((q27_k_wide_down_f32r_row<KI,NW,MINW,TR>), dim3((unsigned)((w->rows + TR - 1) / TR), (unsigned)((M + 63) / 64), 1u), dim3(NW*64), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs)
    if (ki == 88)  { DNF(8,8,2,64);  return 1; }     // small M: 64-row tiles, 2 blocks/CU
    if (ki == 84)  { DNF(8,4,1,128);  return 1; }
    if (ki == 168) { DNF(16,8,1,128); return 1; }
    DNF(16,8,1,128); return 1;
    #undef DNF
}
// rb2d: the CORRECTED weight cadence at the 8-group granularity. Activations stage 8
// k-values per buffer (14 KB LDS -> 4 blocks/CU), weights live in registers as the permuted
// even or odd half of each 16-group, one sync per iteration. Same dot4 pairing and release
// order as rb/rbpf; rb2p's defect (one weight group reused across eight activation groups)
// is gone. TCB=4.
template <int TCB>
__global__ __launch_bounds__(256, 4) void q27_k_wide_gu_rb2d(
        const unsigned char* __restrict__ WG, const unsigned char* __restrict__ SG,
        const unsigned char* __restrict__ WU, const unsigned char* __restrict__ SU,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float ag, float au, int rows, int K, int M, float* __restrict__ YG, float* __restrict__ YU)
{
    constexpr int TR = 64, TC = 16 * TCB, XSTR = 8 * 8 + 16;
    __shared__ __align__(16) signed char sx[2][TC * XSTR];
    __shared__ float sxsc[2][TC * 8];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int mat = blockIdx.z;
    const unsigned char* __restrict__ W = mat ? WU : WG;
    const unsigned char* __restrict__ S = mat ? SU : SG;
    float* __restrict__ Y = mat ? YU : YG;
    const float alpha = mat ? au : ag;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, NG8 = K >> 3, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    uint4 wr0[2]; float ws0[4];
    for (int it = tid; it < TC * 8; it += 256) {
        const int t = it / 8, g = it - t * 8;
        const int tt = min(tok_base + t, M - 1);
        const uint2 e = *(const uint2*)(Xq + (size_t)tt * K + 0);
        *(uint2*)(sx[0] + t * XSTR + g * 8) = e;
        sxsc[0][it] = Xs[(size_t)tt * NG + 0] * 0.5f;
    }
    int cur = 0;
    for (int g8 = 0; g8 < NG8; g8 += 8) {
        const int nxt8 = g8 + 8;
        __syncthreads();
        const int g16 = g8 >> 1;
        const int half = g8 & 1;
        if (nxt8 < NG8) {
            const int n16 = nxt8 >> 1, nhalf = nxt8 & 1;
            for (int it = tid; it < TC * 8; it += 256) {
                const int t = it / 8, g = it - t * 8;
                const int tt = min(tok_base + t, M - 1);
                const uint2 e = *(const uint2*)(Xq + (size_t)tt * K + (size_t)n16 * 16 + (size_t)nhalf * 8);
                *(uint2*)(sx[cur ^ 1] + t * XSTR + g * 8) = e;
                sxsc[cur ^ 1][it] = Xs[(size_t)tt * NG + n16] * 0.5f;
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = row_base + tx * 4 + r;
            const uint2 v = *(const uint2*)(W + (size_t)row * KP + (size_t)g16 * 8);
            if (half == 0) { wr0[0] = (uint4){ q27_dq4( v.x       & 0x0F0F0F0Fu), 0u, 0u, 0u };
                             wr0[1] = (uint4){ q27_dq4((v.x >> 4) & 0x0F0F0F0Fu), 0u, 0u, 0u }; }
            else           { wr0[0] = (uint4){ q27_dq4( v.y       & 0x0F0F0F0Fu), 0u, 0u, 0u };
                             wr0[1] = (uint4){ q27_dq4((v.y >> 4) & 0x0F0F0F0Fu), 0u, 0u, 0u }; }
            ws0[r] = q27_e4m3(S[(size_t)row * NG + g16]);
        }
#pragma unroll 1
        for (int g = 0; g < 8; ++g) {
            uint2 xr[TCB]; float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) {
                const int tc = ty * TCB + c;
                xr[c] = *(const uint2*)(sx[cur] + tc * XSTR + g * 8);
                xsv[c] = sxsc[cur][tc * 8 + g];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr0[0].x, xr[c].x, 0); d = q27_d4(wr0[1].x, xr[c].y, d);
                    acc[r][c] = fmaf((float)d, ws0[r] * xsv[c], acc[r][c]);
                }
        }
        cur ^= 1;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) Y[(size_t)t * rows + row] = acc[r][c] * alpha; }
    }
}

extern "C" int q27_wide_gu_rb2d(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
                                float* yg, float* yu, int M, hipStream_t s) {
    if (!wg || !wu || wg->rows != wu->rows || wg->K != wu->K || (wg->K % 128) || M < 1) return 0;
    const float ag = wg->in_scale * wg->ws2, au = wu->in_scale * wu->ws2;
    const dim3 gr((unsigned)((wg->rows + 63) / 64), (unsigned)((M + 63) / 64), 2u);
    hipLaunchKernelGGL((q27_k_wide_gu_rb2d<4>), gr, dim3(256), 0, s, wg->w, wg->gs, wu->w, wu->gs, xq, xs, ag, au, wg->rows, wg->K, M, yg, yu);
    return 1;
}


// ============================================================================================
// Q27_LS_Q8: dot4 GEMM on the int8 EXECUTION mirror of an fp8 projection (q27_fp8_to_i8g_conv):
// the validated row-major tile (128 rows x 64 tokens, K in-thread by groups of 16) with the weight
// stage reduced to one 16-byte load per (row, group) -- no nibble decode -- natural K order on both
// operands, per-group scale = e4m3(gs) * xs (no 0.5: the mirror is not magnitude-doubled).
//   OUTQ = 0 : fp32 out (+ fp32 residual)        o_proj / out_proj
//   OUTQ = 1 : int8 out + per-16-row fp32 scales  q/k/v, in_qkv, in_z
// ============================================================================================
template <int KI, int NW, int MINW, int OUTQ, int TR>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_i8(
        const signed char* __restrict__ W8, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M,
        float* __restrict__ Y, const float* __restrict__ radd, float rs,
        signed char* __restrict__ Yq, float* __restrict__ Ys, int ystride, int sstride)
{
    constexpr int TC = 64; static_assert(TR == 128 || TR == 64, "TR");
    constexpr int WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
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
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W8 + (size_t)rr * K + (size_t)gg * 16);
            swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] : 0.f;
        }
        __syncthreads();
#pragma unroll 2
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = (TR == 128) ? *(const uint4*)(sw + (lane + 64) * WSTR + g * 16) : make_uint4(0u, 0u, 0u, 0u);
            const float s0 = q27_e4m3(swsc[lane * KI + g]);
            const float s1 = (TR == 128) ? q27_e4m3(swsc[(lane + 64) * KI + g]) : 0.f;
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 x = *(const uint4*)(sx + t * XSTR + g * 16);
                const float xs = sxsc[t * KI + g];
                int a = q27_d4(w0.x, x.x, 0); a = q27_d4(w0.y, x.y, a); a = q27_d4(w0.z, x.z, a); a = q27_d4(w0.w, x.w, a);
                int b = 0; if (TR == 128) { b = q27_d4(w1.x, x.x, 0); b = q27_d4(w1.y, x.y, b); b = q27_d4(w1.z, x.z, b); b = q27_d4(w1.w, x.w, b); }
                acc0[j] = fmaf((float)a, s0 * xs, acc0[j]);
                acc1[j] = fmaf((float)b, s1 * xs, acc1[j]);
            }
        }
    }
    if (OUTQ == 0) {
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            const int t = tok_base + wave * TPW + j;
            if (t < M) {
                if (r0 < rows) { float v = acc0[j] * alpha; if (radd) v += rs * radd[(size_t)t * rows + r0]; Y[(size_t)t * rows + r0] = v; }
                if (TR == 128 && r1 < rows) { float v = acc1[j] * alpha; if (radd) v += rs * radd[(size_t)t * rows + r1]; Y[(size_t)t * rows + r1] = v; }
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            const int t = tok_base + wave * TPW + j;
            const float v0 = acc0[j] * alpha, v1 = acc1[j] * alpha;
            float m0 = fabsf(v0), m1 = fabsf(v1);
            m0 = fmaxf(m0, __shfl_xor(m0, 8, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 8, 16));
            m0 = fmaxf(m0, __shfl_xor(m0, 4, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 4, 16));
            m0 = fmaxf(m0, __shfl_xor(m0, 2, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 2, 16));
            m0 = fmaxf(m0, __shfl_xor(m0, 1, 16)); m1 = fmaxf(m1, __shfl_xor(m1, 1, 16));
            const float sc0 = m0 * (1.f / 127.f), sc1 = m1 * (1.f / 127.f);
            const float q0 = (sc0 > 0.f) ? (1.f / sc0) : 0.f, q1 = (sc1 > 0.f) ? (1.f / sc1) : 0.f;
            int c0 = (int)rintf(v0 * q0); c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0);
            int c1 = (int)rintf(v1 * q1); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
            if (t < M) {
                if (r0 < rows) { Yq[(size_t)t * ystride + r0] = (signed char)c0; if ((lane & 15) == 0) Ys[(size_t)t * sstride + (r0 >> 4)] = sc0; }
                if (TR == 128 && r1 < rows) { Yq[(size_t)t * ystride + r1] = (signed char)c1; if ((lane & 15) == 0) Ys[(size_t)t * sstride + (r1 >> 4)] = sc1; }
            }
        }
    }
}

extern "C" int q27_wide_i8_q8(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                              signed char* yq, float* ys, int ystride, int M, int ki, hipStream_t s) {
    if (!w || !w->w || (w->K % 16) || M < 1 || !xq || !xs || !yq || !ys || (ystride & 15)) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64), 1u);
    #define I8Q(KI,NW,MINW,TR) hipLaunchKernelGGL((q27_k_wide_i8<KI,NW,MINW,1,TR>), dim3((unsigned)((w->rows + TR - 1) / TR), (unsigned)((M + 63) / 64), 1u), dim3(NW*64), 0, s, (const signed char*)w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, (float*)nullptr, (const float*)nullptr, 0.f, yq, ys, ystride, ystride >> 4)
    if (ki == 88)  { I8Q(8,8,2,64);  return 1; }
    I8Q(16,8,1,128); return 1;
    #undef I8Q
}
extern "C" int q27_wide_i8_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                const float* radd, float rs, int ki, hipStream_t s) {
    if (!w || !w->w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    const float alpha = w->in_scale * w->ws2;
    const dim3 g((unsigned)((w->rows + 127) / 128), (unsigned)((M + 63) / 64), 1u);
    #define I8F(KI,NW,MINW,TR) hipLaunchKernelGGL((q27_k_wide_i8<KI,NW,MINW,0,TR>), dim3((unsigned)((w->rows + TR - 1) / TR), (unsigned)((M + 63) / 64), 1u), dim3(NW*64), 0, s, (const signed char*)w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs, (signed char*)nullptr, (float*)nullptr, 0, 0)
    if (ki == 88)  { I8F(8,8,2,64);  return 1; }
    I8F(16,8,1,128); return 1;
    #undef I8F
}

// ============================================================================================
// Q27_LS_Q8: FUSED int8-mirror projections sharing one activation chunk (q/k/v; in_qkv/in_z).
// grid.z selects the matrix; the k/v matrices (1024 rows) would otherwise launch 8-16 blocks
// alone on 60 CUs, so their blocks ride inside the q launch. Same body as q27_k_wide_i8<..,1>.
// ============================================================================================
struct q27_i8m3 {
    const signed char* w[3]; const unsigned char* s[3];
    signed char* yq[3]; float* ys[3];
    float alpha[3]; int rows[3]; int ystride[3]; int sstride[3]; int n;
};
template <int KI, int NW, int MINW, int TR>
__global__ __launch_bounds__(NW * 64, MINW) void q27_k_wide_i8m(
        q27_i8m3 P, const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        int K, int M)
{
    constexpr int TC = 64; static_assert(TR == 128 || TR == 64, "TR");
    constexpr int WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    constexpr int NT = NW * 64, TPW = TC / NW;
    const int z = blockIdx.z;
    const int rows = P.rows[z];
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    if (row_base >= rows) return;                       // whole block: no barrier reached
    const signed char* __restrict__ W8 = P.w[z];
    const unsigned char* __restrict__ S = P.s[z];
    signed char* __restrict__ Yq = P.yq[z];
    float* __restrict__ Ys = P.ys[z];
    const float alpha = P.alpha[z];
    const int ystride = P.ystride[z], sstride = P.sstride[z];
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int NG = K >> 4;
    const int r0 = row_base + lane, r1 = row_base + lane + 64;
    float acc0[TPW], acc1[TPW];
#pragma unroll
    for (int j = 0; j < TPW; ++j) { acc0[j] = 0.f; acc1[j] = 0.f; }
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TR * KI; it += NT) {
            const int r = it / KI, g = it - r * KI;
            const int rr = min(row_base + r, rows - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W8 + (size_t)rr * K + (size_t)gg * 16);
            swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += NT) {
            const int t = it / KI, g = it - t * KI;
            const int tt = min(tok_base + t, M - 1);
            const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] : 0.f;
        }
        __syncthreads();
#pragma unroll 2
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            const uint4 w0 = *(const uint4*)(sw + lane * WSTR + g * 16);
            const uint4 w1 = (TR == 128) ? *(const uint4*)(sw + (lane + 64) * WSTR + g * 16) : make_uint4(0u, 0u, 0u, 0u);
            const float s0 = q27_e4m3(swsc[lane * KI + g]);
            const float s1 = (TR == 128) ? q27_e4m3(swsc[(lane + 64) * KI + g]) : 0.f;
#pragma unroll
            for (int j = 0; j < TPW; ++j) {
                const int t = wave * TPW + j;
                const uint4 x = *(const uint4*)(sx + t * XSTR + g * 16);
                const float xs = sxsc[t * KI + g];
                int a = q27_d4(w0.x, x.x, 0); a = q27_d4(w0.y, x.y, a); a = q27_d4(w0.z, x.z, a); a = q27_d4(w0.w, x.w, a);
                int b = 0; if (TR == 128) { b = q27_d4(w1.x, x.x, 0); b = q27_d4(w1.y, x.y, b); b = q27_d4(w1.z, x.z, b); b = q27_d4(w1.w, x.w, b); }
                acc0[j] = fmaf((float)a, s0 * xs, acc0[j]);
                acc1[j] = fmaf((float)b, s1 * xs, acc1[j]);
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
        const float sc0 = m0 * (1.f / 127.f), sc1 = m1 * (1.f / 127.f);
        const float q0 = (sc0 > 0.f) ? (1.f / sc0) : 0.f, q1 = (sc1 > 0.f) ? (1.f / sc1) : 0.f;
        int c0 = (int)rintf(v0 * q0); c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0);
        int c1 = (int)rintf(v1 * q1); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
        if (t < M) {
            if (r0 < rows) { Yq[(size_t)t * ystride + r0] = (signed char)c0; if ((lane & 15) == 0) Ys[(size_t)t * sstride + (r0 >> 4)] = sc0; }
            if (TR == 128 && r1 < rows) { Yq[(size_t)t * ystride + r1] = (signed char)c1; if ((lane & 15) == 0) Ys[(size_t)t * sstride + (r1 >> 4)] = sc1; }
        }
    }
}

// n matrices (2 or 3) on one activation chunk; yq[i]/ys[i] point at each matrix's column block
// inside the shared [t][ystride] output row (e.g. q | k | v).
extern "C" int q27_wide_i8_q8m(int n, const q27_nvfp4_t* const* ws, const signed char* xq, const float* xs,
                               signed char* const* yq, float* const* ys, const int* ystride, int M, int ki, hipStream_t s) {
    if (n < 1 || n > 3 || !ws || M < 1 || !xq || !xs || !ystride) return 0;
    q27_i8m3 P; P.n = n;
    int rmax = 0; const int K = ws[0]->K;
    for (int i = 0; i < 3; ++i) {
        if (i < n) {
            const q27_nvfp4_t* w = ws[i];
            if (!w || !w->w || w->K != K || (w->K % 16) || !yq[i] || !ys[i]) return 0;
            P.w[i] = (const signed char*)w->w; P.s[i] = w->gs; P.yq[i] = yq[i]; P.ys[i] = ys[i];
            P.alpha[i] = w->in_scale * w->ws2; P.rows[i] = w->rows; P.ystride[i] = ystride[i]; P.sstride[i] = ystride[i] >> 4;
            if (w->rows > rmax) rmax = w->rows;
        } else { P.w[i] = nullptr; P.s[i] = nullptr; P.yq[i] = nullptr; P.ys[i] = nullptr; P.alpha[i] = 0.f; P.rows[i] = 0; P.ystride[i] = 0; P.sstride[i] = 0; }
    }
    const dim3 g((unsigned)((rmax + 127) / 128), (unsigned)((M + 63) / 64), (unsigned)n);
    if (ki == 88) { const dim3 g64((unsigned)((rmax + 63) / 64), (unsigned)((M + 63) / 64), (unsigned)n); hipLaunchKernelGGL((q27_k_wide_i8m<8,8,2,64>), g64, dim3(512), 0, s, P, xq, xs, K, M); return 1; }
    hipLaunchKernelGGL((q27_k_wide_i8m<16,8,1,128>), g, dim3(512), 0, s, P, xq, xs, K, M); return 1;
}

// ============================================================================================
// Q27_LS_Q8: REGISTER-BLOCKED forms (the validated q27_k_wide_gu_rb structure: 64 rows x 16*TCB
// tokens per block, each thread 4 rows x TCB tokens, LDS-staged weights AND activations, one
// weight/activation quad read per 4x/TCBx dot4 group) for
//   * the NVFP4 down with an fp32 residual (q27_k_wide_down_rb_f32r)
//   * the int8-mirror projections (q27_k_wide_i8_rb): int8 weights staged as raw 16-B copies,
//     natural K order on both operands, group scale e4m3(gs)*xs (no 0.5); grid.z = fused matrices;
//     OUTQ 0 = fp32 (+fp32 residual), 1 = int8 + per-16-row scales (q27_q8_epilogue).
// ============================================================================================
template <int KI, int TCB>
__global__ __launch_bounds__(256, 2) void q27_k_wide_down_rb_f32r(
        const unsigned char* __restrict__ W, const unsigned char* __restrict__ S,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float alpha, int rows, int K, int M, float* __restrict__ Y,
        const float* __restrict__ radd, float rs) {
    constexpr int TR = 64, TC = 16 * TCB, WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, KP = K >> 1;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TR * KI; it += 256) {
            const int r = it / KI, g = it - r * KI; const int rr = min(row_base + r, rows - 1); const int gg = min(g0 + g, NG - 1);
            const uint2 v = *(const uint2*)(W + (size_t)rr * KP + (size_t)gg * 8);
            uint4 o; o.x = q27_dq4(v.x & 0x0F0F0F0Fu); o.y = q27_dq4((v.x >> 4) & 0x0F0F0F0Fu);
            o.z = q27_dq4(v.y & 0x0F0F0F0Fu); o.w = q27_dq4((v.y >> 4) & 0x0F0F0F0Fu);
            *(uint4*)(sw + r * WSTR + g * 16) = o; swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] * 0.5f : 0.f;
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 wr[4]; float ws[4]; uint4 xr[8]; float xsv[8];
#pragma unroll
            for (int r = 0; r < 4; ++r) { const int rw = tx * 4 + r; wr[r] = *(const uint4*)(sw + rw * WSTR + g * 16); ws[r] = q27_e4m3(swsc[rw * KI + g]); }
#pragma unroll
            for (int c = 0; c < TCB; ++c) { const int tc = ty * TCB + c; xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16); xsv[c] = sxsc[tc * KI + g]; }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr[r].x, xr[c].x, 0); d = q27_d4(wr[r].y, xr[c].z, d); d = q27_d4(wr[r].z, xr[c].y, d); d = q27_d4(wr[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws[r] * xsv[c], acc[r][c]);
                }
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * radd[(size_t)t * rows + row]; Y[(size_t)t * rows + row] = v; } }
    }
}
extern "C" int q27_wide_down_rb_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                     const float* radd, float rs, hipStream_t s) {
    if (!w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    const float alpha = w->in_scale * w->ws2;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;   // 32-token tiles / 32-wide chunks measured 5x slower: never below 64
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_down_rb_f32r<8,4>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_down_rb_f32r<8,8>), gr, dim3(256), 0, s, w->w, w->gs, xq, xs, alpha, w->rows, w->K, M, y, radd, rs);
    return 1;
}

template <int KI, int TCB, int OUTQ>
__global__ __launch_bounds__(256, 2) void q27_k_wide_i8_rb(
        q27_i8m3 P, const signed char* __restrict__ Xq, const float* __restrict__ Xs, int K, int M,
        float* __restrict__ Y, const float* __restrict__ radd, float rs)
{
    constexpr int TR = 64, TC = 16 * TCB, WSTR = KI * 16 + 16, XSTR = KI * 16 + 16;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ unsigned char swsc[TR * KI];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxsc[TC * KI];
    const int z = blockIdx.z;
    const int rows = P.rows[z];
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    if (row_base >= rows) return;
    const signed char* __restrict__ W8 = P.w[z];
    const unsigned char* __restrict__ S = P.s[z];
    signed char* __restrict__ Yq = P.yq[z];
    float* __restrict__ Ys = P.ys[z];
    const float alpha = P.alpha[z];
    const int ystride = P.ystride[z], sstride = P.sstride[z];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int NG = K >> 4;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        const int gl = min(KI, NG - g0);
        __syncthreads();
        for (int it = tid; it < TR * KI; it += 256) {
            const int r = it / KI, g = it - r * KI; const int rr = min(row_base + r, rows - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W8 + (size_t)rr * K + (size_t)gg * 16);
            swsc[it] = S[(size_t)rr * NG + gg];
        }
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
            sxsc[it] = (g0 + g < NG) ? Xs[(size_t)tt * NG + gg] : 0.f;
        }
        __syncthreads();
#pragma unroll 1
        for (int g = 0; g < KI; ++g) {
            if (g >= gl) break;
            uint4 wr[4]; float ws[4]; uint4 xr[8]; float xsv[8];
#pragma unroll
            for (int r = 0; r < 4; ++r) { const int rw = tx * 4 + r; wr[r] = *(const uint4*)(sw + rw * WSTR + g * 16); ws[r] = q27_e4m3(swsc[rw * KI + g]); }
#pragma unroll
            for (int c = 0; c < TCB; ++c) { const int tc = ty * TCB + c; xr[c] = *(const uint4*)(sx + tc * XSTR + g * 16); xsv[c] = sxsc[tc * KI + g]; }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) {
                    int d = q27_d4(wr[r].x, xr[c].x, 0); d = q27_d4(wr[r].y, xr[c].y, d); d = q27_d4(wr[r].z, xr[c].z, d); d = q27_d4(wr[r].w, xr[c].w, d);
                    acc[r][c] = fmaf((float)d, ws[r] * xsv[c], acc[r][c]);
                }
        }
    }
    if (OUTQ == 1) {
        q27_q8_epilogue<TCB>(acc, alpha, rows, M, row_base, tok_base, tx, ty, Yq, Ys, ystride, sstride);
    } else {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
            for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * radd[(size_t)t * rows + row]; Y[(size_t)t * rows + row] = v; } }
        }
    }
}
extern "C" int q27_wide_i8_rb_q8m(int n, const q27_nvfp4_t* const* ws, const signed char* xq, const float* xs,
                                  signed char* const* yq, float* const* ys, const int* ystride, int M, hipStream_t s) {
    if (n < 1 || n > 3 || !ws || M < 1 || !xq || !xs || !ystride) return 0;
    q27_i8m3 P; P.n = n;
    int rmax = 0; const int K = ws[0]->K;
    for (int i = 0; i < 3; ++i) {
        if (i < n) {
            const q27_nvfp4_t* w = ws[i];
            if (!w || !w->w || w->K != K || (w->K % 16) || !yq[i] || !ys[i] || (w->rows % 16)) return 0;
            P.w[i] = (const signed char*)w->w; P.s[i] = w->gs; P.yq[i] = yq[i]; P.ys[i] = ys[i];
            P.alpha[i] = w->in_scale * w->ws2; P.rows[i] = w->rows; P.ystride[i] = ystride[i]; P.sstride[i] = ystride[i] >> 4;
            if (w->rows > rmax) rmax = w->rows;
        } else { P.w[i] = nullptr; P.s[i] = nullptr; P.yq[i] = nullptr; P.ys[i] = nullptr; P.alpha[i] = 0.f; P.rows[i] = 0; P.ystride[i] = 0; P.sstride[i] = 0; }
    }
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;   // 32-token tiles / 32-wide chunks measured 5x slower: never below 64
    const dim3 g((unsigned)((rmax + 63) / 64), (unsigned)((M + TCv - 1) / TCv), (unsigned)n);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8_rb<8,4,1>), g, dim3(256), 0, s, P, xq, xs, K, M, (float*)nullptr, (const float*)nullptr, 0.f);
    else          hipLaunchKernelGGL((q27_k_wide_i8_rb<8,8,1>), g, dim3(256), 0, s, P, xq, xs, K, M, (float*)nullptr, (const float*)nullptr, 0.f);
    return 1;
}
extern "C" int q27_wide_i8_rb_f32r(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y, int M,
                                   const float* radd, float rs, hipStream_t s) {
    if (!w || !w->w || (w->K % 16) || M < 1 || !xq || !xs || !y) return 0;
    q27_i8m3 P; P.n = 1;
    for (int i = 0; i < 3; ++i) { P.w[i] = nullptr; P.s[i] = nullptr; P.yq[i] = nullptr; P.ys[i] = nullptr; P.alpha[i] = 0.f; P.rows[i] = 0; P.ystride[i] = 0; P.sstride[i] = 0; }
    P.w[0] = (const signed char*)w->w; P.s[0] = w->gs; P.alpha[0] = w->in_scale * w->ws2; P.rows[0] = w->rows;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;   // 32-token tiles / 32-wide chunks measured 5x slower: never below 64
    const dim3 g((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8_rb<8,4,0>), g, dim3(256), 0, s, P, xq, xs, w->K, M, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_i8_rb<8,8,0>), g, dim3(256), 0, s, P, xq, xs, w->K, M, y, radd, rs);
    return 1;
}

// ============================================================================================
// Q27_LS_Q8: int8 PER-64-GROUP register-blocked GEMMs (the gate+up / down of the MLP on the
// q27_i8g_t mirrors). Tile 64 rows x 16*TCB tokens, each thread 4 rows x TCB tokens; per LDS
// step KI 16-groups = KI/4 super-groups of 64. Inside a super-group the 4 x 4 dot4 of every
// (row, token) accumulate in int32; ONE cvt + mul + fma per (row, token, super-group) applies the
// fp32 weight scale x activation scale. Natural K order on both operands.
// ============================================================================================
template <int KI, int TCB, int NMAT>
__global__ __launch_bounds__(256, 2) void q27_k_wide_i8g64(
        const signed char* __restrict__ W0, const float* __restrict__ S0,
        const signed char* __restrict__ W1, const float* __restrict__ S1,
        const signed char* __restrict__ Xq, const float* __restrict__ Xs,
        float a0, float a1, int rows, int K, int M,
        float* __restrict__ Y0, float* __restrict__ Y1,
        const float* __restrict__ radd, float rs)
{
    static_assert(KI % 4 == 0, "KI must be a multiple of 4 (a 64-group is 4 sixteen-groups)");
    constexpr int TR = 64, TC = 16 * TCB, WSTR = KI * 16 + 16, XSTR = KI * 16 + 16, NSG = KI / 4;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ float sws[TR * NSG];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxs[TC * NSG];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int mat = (NMAT == 2) ? (int)blockIdx.z : 0;
    const signed char* __restrict__ W = mat ? W1 : W0;
    const float* __restrict__ S = mat ? S1 : S0;
    float* __restrict__ Y = mat ? Y1 : Y0;
    const float alpha = mat ? a1 : a0;
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    const int NG = K >> 4, NS = K >> 6;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += 256) {
            const int r = it / KI, g = it - r * KI; const int rr = min(row_base + r, rows - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W + (size_t)rr * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TR * NSG; it += 256) {
            const int r = it / NSG, sg = it - r * NSG; const int rr = min(row_base + r, rows - 1); const int ss = min((g0 >> 2) + sg, NS - 1);
            sws[it] = S[(size_t)rr * NS + ss];
        }
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TC * NSG; it += 256) {
            const int t = it / NSG, sg = it - t * NSG; const int tt = min(tok_base + t, M - 1); const int ss = (g0 >> 2) + sg;
            sxs[it] = (ss < NS) ? Xs[(size_t)tt * NS + ss] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int sg = 0; sg < NSG; ++sg) {
            int ai[4][TCB];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) ai[r][c] = 0;
#pragma unroll
            for (int g4 = 0; g4 < 4; ++g4) {
                const int g = sg * 4 + g4;
                uint4 wr[4]; uint4 xr[TCB];
#pragma unroll
                for (int r = 0; r < 4; ++r) wr[r] = *(const uint4*)(sw + (tx * 4 + r) * WSTR + g * 16);
#pragma unroll
                for (int c = 0; c < TCB; ++c) xr[c] = *(const uint4*)(sx + (ty * TCB + c) * XSTR + g * 16);
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int c = 0; c < TCB; ++c) {
                        int d = q27_d4(wr[r].x, xr[c].x, ai[r][c]); d = q27_d4(wr[r].y, xr[c].y, d);
                        d = q27_d4(wr[r].z, xr[c].z, d); ai[r][c] = q27_d4(wr[r].w, xr[c].w, d);
                    }
            }
            float ws[4], xsv[TCB];
#pragma unroll
            for (int r = 0; r < 4; ++r) ws[r] = sws[(tx * 4 + r) * NSG + sg];
#pragma unroll
            for (int c = 0; c < TCB; ++c) xsv[c] = sxs[(ty * TCB + c) * NSG + sg];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) acc[r][c] = fmaf((float)ai[r][c], ws[r] * xsv[c], acc[r][c]);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
        for (int c = 0; c < TCB; ++c) {
            const int t = tok_base + ty * TCB + c;
            if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * radd[(size_t)t * rows + row]; Y[(size_t)t * rows + row] = v; }
        }
    }
}
extern "C" int q27_wide_gu_i8g64(const q27_i8g_t* wg, const q27_i8g_t* wu, const signed char* xq, const float* xs,
                                 float* yg, float* yu, int M, hipStream_t s) {
    if (!wg || !wu || !wg->w || !wu->w || wg->rows != wu->rows || wg->K != wu->K || (wg->K % 128) || M < 1 || !xq || !xs) return 0;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;   // 32-token tiles / 32-wide chunks measured 5x slower: never below 64
    const dim3 gr((unsigned)((wg->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 2u);
    if (tcb == 2) hipLaunchKernelGGL((q27_k_wide_i8g64<8,2,2>), gr, dim3(256), 0, s, wg->w, wg->s, wu->w, wu->s, xq, xs, wg->alpha, wu->alpha, wg->rows, wg->K, M, yg, yu, (const float*)nullptr, 0.f);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8g64<8,4,2>), gr, dim3(256), 0, s, wg->w, wg->s, wu->w, wu->s, xq, xs, wg->alpha, wu->alpha, wg->rows, wg->K, M, yg, yu, (const float*)nullptr, 0.f);
    else          hipLaunchKernelGGL((q27_k_wide_i8g64<8,8,2>), gr, dim3(256), 0, s, wg->w, wg->s, wu->w, wu->s, xq, xs, wg->alpha, wu->alpha, wg->rows, wg->K, M, yg, yu, (const float*)nullptr, 0.f);
    return 1;
}
extern "C" int q27_wide_down_i8g64_f32r(const q27_i8g_t* w, const signed char* xq, const float* xs, float* y, int M,
                                        const float* radd, float rs, hipStream_t s) {
    if (!w || !w->w || (w->K % 64) || M < 1 || !xq || !xs || !y) return 0;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;   // 32-token tiles / 32-wide chunks measured 5x slower: never below 64
    const dim3 gr((unsigned)((w->rows + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 2) hipLaunchKernelGGL((q27_k_wide_i8g64<8,2,1>), gr, dim3(256), 0, s, w->w, w->s, w->w, w->s, xq, xs, w->alpha, w->alpha, w->rows, w->K, M, y, y, radd, rs);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8g64<8,4,1>), gr, dim3(256), 0, s, w->w, w->s, w->w, w->s, xq, xs, w->alpha, w->alpha, w->rows, w->K, M, y, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_i8g64<8,8,1>), gr, dim3(256), 0, s, w->w, w->s, w->w, w->s, xq, xs, w->alpha, w->alpha, w->rows, w->K, M, y, y, radd, rs);
    return 1;
}

// ============================================================================================
// Q27_LS_Q8 stage 2: the fused int8 PER-64 projection GEMM (register-blocked, int32 accumulation
// inside a 64-group) on the q27_i8g_t mirrors of q/k/v, in_qkv/in_z, o/out_proj. grid.z = matrix.
//   OUTQ 1: int8 out + per-16-ROW fp32 scales (the attention / conv / scan consumers' representation)
//   OUTQ 0: fp32 out + fp32 residual
// ============================================================================================
struct q27_i8g3 {
    const signed char* w[3]; const float* s[3];
    signed char* yq[3]; float* ys[3];
    float alpha[3]; int rows[3]; int ystride[3]; int sstride[3]; int n;
};
template <int KI, int TCB, int OUTQ>
__global__ __launch_bounds__(256, 2) void q27_k_wide_i8g64m(
        q27_i8g3 P, const signed char* __restrict__ Xq, const float* __restrict__ Xs, int K, int M,
        float* __restrict__ Y, const float* __restrict__ radd, float rs)
{
    static_assert(KI % 4 == 0, "KI must be a multiple of 4");
    constexpr int TR = 64, TC = 16 * TCB, WSTR = KI * 16 + 16, XSTR = KI * 16 + 16, NSG = KI / 4;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ float sws[TR * NSG];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxs[TC * NSG];
    const int z = blockIdx.z;
    const int rows = P.rows[z];
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    if (row_base >= rows) return;
    const signed char* __restrict__ W = P.w[z];
    const float* __restrict__ S = P.s[z];
    const float alpha = P.alpha[z];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int NG = K >> 4, NS = K >> 6;
    float acc[4][TCB];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += 256) {
            const int r = it / KI, g = it - r * KI; const int rr = min(row_base + r, rows - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W + (size_t)rr * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TR * NSG; it += 256) {
            const int r = it / NSG, sg = it - r * NSG; const int rr = min(row_base + r, rows - 1); const int ss = min((g0 >> 2) + sg, NS - 1);
            sws[it] = S[(size_t)rr * NS + ss];
        }
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TC * NSG; it += 256) {
            const int t = it / NSG, sg = it - t * NSG; const int tt = min(tok_base + t, M - 1); const int ss = (g0 >> 2) + sg;
            sxs[it] = (ss < NS) ? Xs[(size_t)tt * NS + ss] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int sg = 0; sg < NSG; ++sg) {
            int ai[4][TCB];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) ai[r][c] = 0;
#pragma unroll
            for (int g4 = 0; g4 < 4; ++g4) {
                const int g = sg * 4 + g4;
                uint4 wr[4]; uint4 xr[TCB];
#pragma unroll
                for (int r = 0; r < 4; ++r) wr[r] = *(const uint4*)(sw + (tx * 4 + r) * WSTR + g * 16);
#pragma unroll
                for (int c = 0; c < TCB; ++c) xr[c] = *(const uint4*)(sx + (ty * TCB + c) * XSTR + g * 16);
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int c = 0; c < TCB; ++c) {
                        int d = q27_d4(wr[r].x, xr[c].x, ai[r][c]); d = q27_d4(wr[r].y, xr[c].y, d);
                        d = q27_d4(wr[r].z, xr[c].z, d); ai[r][c] = q27_d4(wr[r].w, xr[c].w, d);
                    }
            }
            float ws[4], xsv[TCB];
#pragma unroll
            for (int r = 0; r < 4; ++r) ws[r] = sws[(tx * 4 + r) * NSG + sg];
#pragma unroll
            for (int c = 0; c < TCB; ++c) xsv[c] = sxs[(ty * TCB + c) * NSG + sg];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) acc[r][c] = fmaf((float)ai[r][c], ws[r] * xsv[c], acc[r][c]);
        }
    }
    if (OUTQ == 1) {
        q27_q8_epilogue<TCB>(acc, alpha, rows, M, row_base, tok_base, tx, ty, P.yq[z], P.ys[z], P.ystride[z], P.sstride[z]);
    } else {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = row_base + tx * 4 + r; if (row >= rows) continue;
#pragma unroll
            for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * radd[(size_t)t * rows + row]; Y[(size_t)t * rows + row] = v; } }
        }
    }
}
static int q27_i8g3_fill(q27_i8g3& P, int n, const q27_i8g_t* const* ws, signed char* const* yq, float* const* ys, const int* ystride, int& rmax, int& K) {
    rmax = 0; K = ws[0]->K; P.n = n;
    for (int i = 0; i < 3; ++i) {
        if (i < n) {
            const q27_i8g_t* w = ws[i];
            if (!w || !w->w || w->K != K || (w->K % 64) || (w->rows % 16)) return 0;
            P.w[i] = w->w; P.s[i] = w->s; P.alpha[i] = w->alpha; P.rows[i] = w->rows;
            P.yq[i] = yq ? yq[i] : nullptr; P.ys[i] = ys ? ys[i] : nullptr;
            P.ystride[i] = ystride ? ystride[i] : 0; P.sstride[i] = ystride ? (ystride[i] >> 4) : 0;
            if (w->rows > rmax) rmax = w->rows;
        } else { P.w[i] = nullptr; P.s[i] = nullptr; P.yq[i] = nullptr; P.ys[i] = nullptr; P.alpha[i] = 0.f; P.rows[i] = 0; P.ystride[i] = 0; P.sstride[i] = 0; }
    }
    return 1;
}
extern "C" int q27_wide_i8g64_q8m(int n, const q27_i8g_t* const* ws, const signed char* xq, const float* xs,
                                  signed char* const* yq, float* const* ys, const int* ystride, int M, hipStream_t s) {
    if (n < 1 || n > 3 || !ws || M < 1 || !xq || !xs || !yq || !ys || !ystride) return 0;
    q27_i8g3 P; int rmax, K;
    if (!q27_i8g3_fill(P, n, ws, yq, ys, ystride, rmax, K)) return 0;
    for (int i = 0; i < n; ++i) if (ystride[i] & 15) return 0;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;   // 32-token tiles / 32-wide chunks measured 5x slower: never below 64
    const dim3 g((unsigned)((rmax + 63) / 64), (unsigned)((M + TCv - 1) / TCv), (unsigned)n);
    if (tcb == 2) hipLaunchKernelGGL((q27_k_wide_i8g64m<8,2,1>), g, dim3(256), 0, s, P, xq, xs, K, M, (float*)nullptr, (const float*)nullptr, 0.f);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8g64m<8,4,1>), g, dim3(256), 0, s, P, xq, xs, K, M, (float*)nullptr, (const float*)nullptr, 0.f);
    else          hipLaunchKernelGGL((q27_k_wide_i8g64m<8,8,1>), g, dim3(256), 0, s, P, xq, xs, K, M, (float*)nullptr, (const float*)nullptr, 0.f);
    return 1;
}
extern "C" int q27_wide_i8g64_f32r(const q27_i8g_t* w, const signed char* xq, const float* xs, float* y, int M,
                                   const float* radd, float rs, hipStream_t s) {
    if (!w || !w->w || M < 1 || !xq || !xs || !y) return 0;
    q27_i8g3 P; int rmax, K; const q27_i8g_t* ws[1] = { w };
    if (!q27_i8g3_fill(P, 1, ws, nullptr, nullptr, nullptr, rmax, K)) return 0;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;   // 32-token tiles / 32-wide chunks measured 5x slower: never below 64
    const dim3 g((unsigned)((rmax + 63) / 64), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 2) hipLaunchKernelGGL((q27_k_wide_i8g64m<8,2,0>), g, dim3(256), 0, s, P, xq, xs, K, M, y, radd, rs);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8g64m<8,4,0>), g, dim3(256), 0, s, P, xq, xs, K, M, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_i8g64m<8,8,0>), g, dim3(256), 0, s, P, xq, xs, K, M, y, radd, rs);
    return 1;
}

// ============================================================================================
// Q27_LS_Q8: TALLER register tile for the int8/64 GEMMs -- 128 rows x 64 tokens per block, each thread
// 8 rows x 4 tokens (two 4-row halves share one activation-quad load), int32 accumulation per 64-group.
// LDS reads per 128 dot4: 8 weight quads + 4 activation quads (1.5 B/dot4 vs 2.0 for the 4x4 tile).
// Same interface as q27_k_wide_i8g64m (NMAT via P.n / grid.z; OUTQ 1 = int8+per-16-row scales, 0 = fp32+residual).
// ============================================================================================
template <int KI, int TCB, int OUTQ>
__global__ __launch_bounds__(256, 2) void q27_k_wide_i8g64r8(
        q27_i8g3 P, const signed char* __restrict__ Xq, const float* __restrict__ Xs, int K, int M,
        float* __restrict__ Y, const float* __restrict__ radd, float rs)
{
    static_assert(KI % 4 == 0, "KI must be a multiple of 4");
    constexpr int TR = 128, TC = 16 * TCB, WSTR = KI * 16 + 16, XSTR = KI * 16 + 16, NSG = KI / 4;
    __shared__ __align__(16) signed char sw[TR * WSTR];
    __shared__ float sws[TR * NSG];
    __shared__ __align__(16) signed char sx[TC * XSTR];
    __shared__ float sxs[TC * NSG];
    const int z = blockIdx.z;
    const int rows = P.rows[z];
    const int row_base = blockIdx.x * TR, tok_base = blockIdx.y * TC;
    if (row_base >= rows) return;
    const signed char* __restrict__ W = P.w[z];
    const float* __restrict__ S = P.s[z];
    const float alpha = P.alpha[z];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int NG = K >> 4, NS = K >> 6;
    float acc[8][TCB];
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int c = 0; c < TCB; ++c) acc[r][c] = 0.f;
    for (int g0 = 0; g0 < NG; g0 += KI) {
        __syncthreads();
        for (int it = tid; it < TR * KI; it += 256) {
            const int r = it / KI, g = it - r * KI; const int rr = min(row_base + r, rows - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sw + r * WSTR + g * 16) = *(const uint4*)(W + (size_t)rr * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TR * NSG; it += 256) {
            const int r = it / NSG, sg = it - r * NSG; const int rr = min(row_base + r, rows - 1); const int ss = min((g0 >> 2) + sg, NS - 1);
            sws[it] = S[(size_t)rr * NS + ss];
        }
        for (int it = tid; it < TC * KI; it += 256) {
            const int t = it / KI, g = it - t * KI; const int tt = min(tok_base + t, M - 1); const int gg = min(g0 + g, NG - 1);
            *(uint4*)(sx + t * XSTR + g * 16) = *(const uint4*)(Xq + (size_t)tt * K + (size_t)gg * 16);
        }
        for (int it = tid; it < TC * NSG; it += 256) {
            const int t = it / NSG, sg = it - t * NSG; const int tt = min(tok_base + t, M - 1); const int ss = (g0 >> 2) + sg;
            sxs[it] = (ss < NS) ? Xs[(size_t)tt * NS + ss] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int sg = 0; sg < NSG; ++sg) {
            int ai[8][TCB];
#pragma unroll
            for (int r = 0; r < 8; ++r)
#pragma unroll
                for (int c = 0; c < TCB; ++c) ai[r][c] = 0;
#pragma unroll
            for (int g4 = 0; g4 < 4; ++g4) {
                const int g = sg * 4 + g4;
                uint4 xr[TCB];
#pragma unroll
                for (int c = 0; c < TCB; ++c) xr[c] = *(const uint4*)(sx + (ty * TCB + c) * XSTR + g * 16);
#pragma unroll
                for (int h = 0; h < 2; ++h) {                 // two 4-row halves reuse the activation quads
                    uint4 wr[4];
#pragma unroll
                    for (int r = 0; r < 4; ++r) wr[r] = *(const uint4*)(sw + (tx * 8 + h * 4 + r) * WSTR + g * 16);
#pragma unroll
                    for (int r = 0; r < 4; ++r)
#pragma unroll
                        for (int c = 0; c < TCB; ++c) {
                            int d = q27_d4(wr[r].x, xr[c].x, ai[h * 4 + r][c]); d = q27_d4(wr[r].y, xr[c].y, d);
                            d = q27_d4(wr[r].z, xr[c].z, d); ai[h * 4 + r][c] = q27_d4(wr[r].w, xr[c].w, d);
                        }
                }
            }
            float xsv[TCB];
#pragma unroll
            for (int c = 0; c < TCB; ++c) xsv[c] = sxs[(ty * TCB + c) * NSG + sg];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float ws = sws[(tx * 8 + r) * NSG + sg];
#pragma unroll
                for (int c = 0; c < TCB; ++c) acc[r][c] = fmaf((float)ai[r][c], ws * xsv[c], acc[r][c]);
            }
        }
    }
    if (OUTQ == 1) {
        // per-16-row scale groups: rows tx*8..tx*8+7 -> the 16-row group is two consecutive tx lanes
        signed char* __restrict__ Yq = P.yq[z]; float* __restrict__ Ys = P.ys[z];
        const int ystride = P.ystride[z], sstride = P.sstride[z];
        const int row0 = row_base + tx * 8;
#pragma unroll
        for (int c = 0; c < TCB; ++c) {
            float v[8]; float mx = 0.f;
#pragma unroll
            for (int r = 0; r < 8; ++r) { v[r] = acc[r][c] * alpha; mx = fmaxf(mx, fabsf(v[r])); }
            mx = fmaxf(mx, __shfl_xor(mx, 1, 2));
            const float sc = mx * (1.f / 127.f);
            const float qi = (sc > 0.f) ? (1.f / sc) : 0.f;
            const int t = tok_base + ty * TCB + c;
            if (t < M && row0 < rows) {
                unsigned p0 = 0u, p1 = 0u;
#pragma unroll
                for (int r = 0; r < 4; ++r) { int cq = (int)rintf(v[r] * qi); cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq); p0 |= ((unsigned)cq & 0xFFu) << (8 * r); }
#pragma unroll
                for (int r = 4; r < 8; ++r) { int cq = (int)rintf(v[r] * qi); cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq); p1 |= ((unsigned)cq & 0xFFu) << (8 * (r - 4)); }
                *(unsigned*)(Yq + (size_t)t * ystride + row0) = p0;
                *(unsigned*)(Yq + (size_t)t * ystride + row0 + 4) = p1;
                if ((tx & 1) == 0) Ys[(size_t)t * sstride + (row0 >> 4)] = sc;
            }
        }
    } else {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = row_base + tx * 8 + r; if (row >= rows) continue;
#pragma unroll
            for (int c = 0; c < TCB; ++c) { const int t = tok_base + ty * TCB + c; if (t < M) { float v = acc[r][c] * alpha; if (radd) v += rs * radd[(size_t)t * rows + row]; Y[(size_t)t * rows + row] = v; } }
        }
    }
}
extern "C" int q27_wide_i8g64r8_q8m(int n, const q27_i8g_t* const* ws, const signed char* xq, const float* xs,
                                    signed char* const* yq, float* const* ys, const int* ystride, int M, hipStream_t s) {
    if (n < 1 || n > 3 || !ws || M < 1 || !xq || !xs || !yq || !ys || !ystride) return 0;
    q27_i8g3 P; int rmax, K;
    if (!q27_i8g3_fill(P, n, ws, yq, ys, ystride, rmax, K)) return 0;
    for (int i = 0; i < n; ++i) if (ystride[i] & 15) return 0;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;
    const dim3 g((unsigned)((rmax + 127) / 128), (unsigned)((M + TCv - 1) / TCv), (unsigned)n);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8g64r8<8,4,1>), g, dim3(256), 0, s, P, xq, xs, K, M, (float*)nullptr, (const float*)nullptr, 0.f);
    else          hipLaunchKernelGGL((q27_k_wide_i8g64r8<8,8,1>), g, dim3(256), 0, s, P, xq, xs, K, M, (float*)nullptr, (const float*)nullptr, 0.f);
    return 1;
}
extern "C" int q27_wide_i8g64r8_f32r(const q27_i8g_t* w, const signed char* xq, const float* xs, float* y, int M,
                                     const float* radd, float rs, hipStream_t s) {
    if (!w || !w->w || M < 1 || !xq || !xs || !y) return 0;
    q27_i8g3 P; int rmax, K; const q27_i8g_t* ws[1] = { w };
    if (!q27_i8g3_fill(P, 1, ws, nullptr, nullptr, nullptr, rmax, K)) return 0;
    const int tcb = (M <= 64) ? 4 : 8, TCv = 16 * tcb;
    const dim3 g((unsigned)((rmax + 127) / 128), (unsigned)((M + TCv - 1) / TCv), 1u);
    if (tcb == 4) hipLaunchKernelGGL((q27_k_wide_i8g64r8<8,4,0>), g, dim3(256), 0, s, P, xq, xs, K, M, y, radd, rs);
    else          hipLaunchKernelGGL((q27_k_wide_i8g64r8<8,8,0>), g, dim3(256), 0, s, P, xq, xs, K, M, y, radd, rs);
    return 1;
}
// gate+up on the taller tile: two matrices = two z-slices of the fused struct (fp32 outputs into P.ys as Y? no:
// the fp32 epilogue writes Y, so gate and up need separate launches -- or one launch with Y selected by z).
extern "C" int q27_wide_gu_i8g64r8(const q27_i8g_t* wg, const q27_i8g_t* wu, const signed char* xq, const float* xs,
                                   float* yg, float* yu, int M, hipStream_t s) {
    if (!wg || !wu || !wg->w || !wu->w || wg->K != wu->K || M < 1 || !xq || !xs) return 0;
    if (!q27_wide_i8g64r8_f32r(wg, xq, xs, yg, M, nullptr, 0.f, s)) return 0;
    return q27_wide_i8g64r8_f32r(wu, xq, xs, yu, M, nullptr, 0.f, s);
}
