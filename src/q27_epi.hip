// Q27_LS_Q8 stage 3: producers/consumers around the rocBLAS int8 GEMMs (see src/q27_rb.cpp).
// K is split into 1024-wide GROUPS (Q27_Q8_GS = 1024 = Q8_GS below): weights carry one fp32 scale per
// (row, group), activations one per (token, group); one rocBLAS call per group writes its own int32 slab
// [M][N]; the epilogues sum the slabs with the product of the two scales. This keeps most of the per-64
// precision (a 1024-group spans 16 of the per-64 groups) at the rocBLAS rate.
//   * per-(row,group) int8 weights from the per-64 mirrors (in place; group scale = max of its 16 sub-scales)
//   * per-(token,group) int8 activations: input norm, post norm (+ residual capture), per-64 -> group requant
//   * int32 slab epilogues: -> int8 + per-16 scales (attention / GDN inputs), -> fp32 + residual, -> swiglu
#include <hip/hip_runtime.h>
#include <cstdio>
#include "q27.h"
#define Q8_GS 1024   // the kernels' unit group: 256 threads x float4; a runtime group = IPG units (IPG = 1 or K/1024 = "full K")

// ---- per-(row,group) int8 from the per-64 mirror (in place) --------------------------------------------
__global__ __launch_bounds__(256) void q27_k_i8g64_to_grp(signed char* __restrict__ w, const float* __restrict__ sg, int GS,
                                                          float* __restrict__ sgrp, int K)
{
    const int r = blockIdx.x, t = threadIdx.x, NS = K >> 6, NG = K / GS;
    __shared__ float red[256];
    signed char* row = w + (size_t)r * K;
    for (int g = 0; g < NG; ++g) {
        float mx = 0.f;
        for (int s = t; s < (GS >> 6); s += 256) mx = fmaxf(mx, sg[(size_t)r * NS + g * (GS >> 6) + s]);
        red[t] = mx; __syncthreads();
        for (int o = 128; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
        const float s = red[0];
        const float inv = (s > 0.f) ? 1.f / s : 0.f;
        if (t == 0) sgrp[(size_t)r * NG + g] = s;
        for (int k = g * GS + t; k < (g + 1) * GS; k += 256) {
            const float v = (float)row[k] * sg[(size_t)r * NS + (k >> 6)] * inv;
            int q = (int)rintf(v); q = q < -127 ? -127 : (q > 127 ? 127 : q);
            row[k] = (signed char)q;
        }
        __syncthreads();
    }
}
extern "C" int q27_i8g64_to_row_conv(q27_i8g_t* src, q27_i8r_t* dst, int gs, hipStream_t st)   // gs: 1024 or 0 = full K
{
    if (!src || !src->w || !src->s || !dst || src->rows < 1 || (src->K % Q8_GS)) return 0;
    const int GS = (gs > 0) ? gs : src->K;
    if (src->K % GS) return 0;
    const int NG = src->K / GS;
    float* sgrp = nullptr;
    if (hipMalloc((void**)&sgrp, (size_t)src->rows * NG * 4) != hipSuccess) return 0;
    hipLaunchKernelGGL(q27_k_i8g64_to_grp, dim3((unsigned)src->rows), dim3(256), 0, st, (signed char*)src->w, src->s, GS, sgrp, src->K);
    dst->w = src->w; dst->s = sgrp; dst->alpha = src->alpha; dst->rows = src->rows; dst->K = src->K; dst->ng = NG; dst->gs = GS;
    return 1;
}

// ---- helpers ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float q27_blockmax256(float v, float* red) {   // 256 threads; red[4]; two barriers
    v = fmaxf(v, __shfl_xor(v, 32, 64)); v = fmaxf(v, __shfl_xor(v, 16, 64)); v = fmaxf(v, __shfl_xor(v, 8, 64));
    v = fmaxf(v, __shfl_xor(v, 4, 64)); v = fmaxf(v, __shfl_xor(v, 2, 64)); v = fmaxf(v, __shfl_xor(v, 1, 64));
    if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = v;
    __syncthreads();
    const float m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    return m;
}
__device__ __forceinline__ unsigned q27_pack4(float v0, float v1, float v2, float v3, float qi) {
    int c0 = (int)rintf(v0 * qi), c1 = (int)rintf(v1 * qi), c2 = (int)rintf(v2 * qi), c3 = (int)rintf(v3 * qi);
    c0 = c0 < -127 ? -127 : (c0 > 127 ? 127 : c0); c1 = c1 < -127 ? -127 : (c1 > 127 ? 127 : c1);
    c2 = c2 < -127 ? -127 : (c2 > 127 ? 127 : c2); c3 = c3 < -127 ? -127 : (c3 > 127 ? 127 : c3);
    return ((unsigned)c0 & 0xFFu) | (((unsigned)c1 & 0xFFu) << 8) | (((unsigned)c2 & 0xFFu) << 16) | (((unsigned)c3 & 0xFFu) << 24);
}
// a 256-thread block holds one 1024-unit per float4-iteration; a group = IPG consecutive units (IPG = 1: 1024-groups,
// IPG = K/1024: one group per row = per-token scales). v[] holds the group's values; the max is a block max.
template <int IPG>
__device__ __forceinline__ void q27_quant_group(const float4* v, signed char* q, float* qs, int k0, int g, float* red) {
    float mx = 0.f;
#pragma unroll
    for (int i = 0; i < IPG; ++i) mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
    mx = q27_blockmax256(mx, red);
    const float sc = mx * (1.f / 127.f), qi = (sc > 0.f) ? 1.f / sc : 0.f;
    if (threadIdx.x == 0) qs[g] = sc;
#pragma unroll
    for (int i = 0; i < IPG; ++i) *(unsigned*)(q + k0 + i * Q8_GS) = q27_pack4(v[i].x, v[i].y, v[i].z, v[i].w, qi);
}


// ---- input norm, per-(token,group) int8 (host-computed inverse rms per token, as q27_k_rn_hostss_g64_b) ----
template <int IPG>
__global__ __launch_bounds__(256) void q27_k_rn_hostss_tok_b(
        const float4* __restrict__ x, const unsigned short* __restrict__ w,
        signed char* __restrict__ q, float* __restrict__ qs, int nv4, const float* __restrict__ invp, float plus, float sc_in)
{
    const int c = blockIdx.y, t = threadIdx.x, NU = (nv4 << 2) / Q8_GS, NG = NU / IPG;
    x += (size_t)c * nv4; q += (size_t)c * (nv4 << 2); qs += (size_t)c * NG;
    const float inv = invp[c];
    __shared__ float red[4];
    for (int g = 0; g < NG; ++g) {
        float4 v[IPG];
#pragma unroll
        for (int i = 0; i < IPG; ++i) {
            const int u = g * IPG + i, idx = u * 256 + t, k0 = idx << 2;
            const float4 xv = x[idx];
            const ushort4 wv = *(const ushort4*)(w + k0);
            v[i].x = xv.x * inv * (plus + q27_bf2f(wv.x)) * sc_in; v[i].y = xv.y * inv * (plus + q27_bf2f(wv.y)) * sc_in;
            v[i].z = xv.z * inv * (plus + q27_bf2f(wv.z)) * sc_in; v[i].w = xv.w * inv * (plus + q27_bf2f(wv.w)) * sc_in;
        }
        q27_quant_group<IPG>(v, q, qs, (g * IPG * 256 + t) << 2, g, red);
    }
}
extern "C" int q27_rmsnorm_hostss_tok_b(const float* x, const unsigned short* wgt, signed char* xq, float* xs, int n,
                                        int plus_one, float in_scale, const float* invp, int C, int gs, hipStream_t s)
{
    if (n != Q27_HID || (n % Q8_GS) || !(in_scale > 0.f) || !x || !invp || !xq || !xs || C < 1) return 0;
    if (gs == 1024) hipLaunchKernelGGL((q27_k_rn_hostss_tok_b<1>), dim3(1, C), dim3(256), 0, s, (const float4*)x, wgt, xq, xs, n >> 2, invp, plus_one ? 1.f : 0.f, 1.f / in_scale);
    else if (gs == 0 || gs == Q27_HID) hipLaunchKernelGGL((q27_k_rn_hostss_tok_b<Q27_HID / 1024>), dim3(1, C), dim3(256), 0, s, (const float4*)x, wgt, xq, xs, n >> 2, invp, plus_one ? 1.f : 0.f, 1.f / in_scale);
    else return 0;
    return 1;
}

// ---- post norm: fp32 residual (already summed) -> per-(token,group) int8 (+ fp32 residual capture) ---------
template <int IPG>
__global__ __launch_bounds__(256) void q27_k_red_rn_tok_b1(
        const float4* __restrict__ x, const unsigned short* __restrict__ wgt, float* __restrict__ hid32,
        signed char* __restrict__ xq, float* __restrict__ xs, int nv4, float plus, float sc_in)
{
    constexpr int NU = Q27_HID / Q8_GS, NG = NU / IPG;   // 5 units
    const int c = blockIdx.y, t = threadIdx.x;
    x += (size_t)c * nv4; xq += (size_t)c * (nv4 << 2); xs += (size_t)c * NG;
    if (hid32) hid32 += (size_t)c * (nv4 << 2);
    __shared__ float red[4];
    float4 v[NU]; float ss = 0.f;
#pragma unroll
    for (int e = 0; e < NU; ++e) { v[e] = x[e * 256 + t]; ss = fmaf(v[e].x, v[e].x, ss); ss = fmaf(v[e].y, v[e].y, ss); ss = fmaf(v[e].z, v[e].z, ss); ss = fmaf(v[e].w, v[e].w, ss); }
    ss += __shfl_xor(ss, 32, 64); ss += __shfl_xor(ss, 16, 64); ss += __shfl_xor(ss, 8, 64); ss += __shfl_xor(ss, 4, 64); ss += __shfl_xor(ss, 2, 64); ss += __shfl_xor(ss, 1, 64);
    if ((t & 63) == 0) red[t >> 6] = ss;
    __syncthreads();
    const float tot = (red[0] + red[1]) + (red[2] + red[3]);
    __syncthreads();
    const float inv = 1.0f / sqrtf(tot / (float)(nv4 << 2) + Q27_RMS_EPS);
#pragma unroll
    for (int e = 0; e < NU; ++e) {
        const int k0 = (e * 256 + t) << 2;
        const ushort4 wv = *(const ushort4*)(wgt + k0);
        if (hid32) *(float4*)(hid32 + k0) = v[e];
        float4 o;
        o.x = v[e].x * inv * (plus + q27_bf2f(wv.x)) * sc_in; o.y = v[e].y * inv * (plus + q27_bf2f(wv.y)) * sc_in;
        o.z = v[e].z * inv * (plus + q27_bf2f(wv.z)) * sc_in; o.w = v[e].w * inv * (plus + q27_bf2f(wv.w)) * sc_in;
        v[e] = o;
    }
#pragma unroll
    for (int g = 0; g < NG; ++g) q27_quant_group<IPG>(v + g * IPG, xq, xs, (g * IPG * 256 + t) << 2, g, red);
}
extern "C" int q27_red_rn_tok_b1(const float* x, const unsigned short* wgt, float* hid32, signed char* xq, float* xs, int n,
                                 int plus_one, float in_scale, int C, int gs, hipStream_t s)
{
    if (n != Q27_HID || n != 5 * Q8_GS || !(in_scale > 0.f) || !x || !wgt || !xq || !xs || C < 1) return 0;
    if (gs == 1024) hipLaunchKernelGGL((q27_k_red_rn_tok_b1<1>), dim3(1, C), dim3(256), 0, s, (const float4*)x, wgt, hid32, xq, xs, n >> 2, plus_one ? 1.f : 0.f, 1.f / in_scale);
    else if (gs == 0 || gs == Q27_HID) hipLaunchKernelGGL((q27_k_red_rn_tok_b1<5>), dim3(1, C), dim3(256), 0, s, (const float4*)x, wgt, hid32, xq, xs, n >> 2, plus_one ? 1.f : 0.f, 1.f / in_scale);
    else return 0;
    return 1;
}

// ---- per-64 int8 (xs[tok][K/64]) -> per-(token,group) int8 (attention / GDN outputs feeding o_proj / out_proj)
template <int IPG>
__global__ __launch_bounds__(256) void q27_k_requant_g64_tok(const signed char* __restrict__ xq, const float* __restrict__ xs,
                                                             signed char* __restrict__ yq, float* __restrict__ ys, int K, int xstride, int sstride)
{
    const int c = blockIdx.x, t = threadIdx.x, NU = K / Q8_GS, NG = NU / IPG;
    xq += (size_t)c * xstride; xs += (size_t)c * sstride; yq += (size_t)c * K; ys += (size_t)c * NG;
    __shared__ float red[4];
    for (int g = 0; g < NG; ++g) {
        float4 v[IPG];
#pragma unroll
        for (int i = 0; i < IPG; ++i) {
            const int k0 = (g * IPG + i) * Q8_GS + (t << 2);
            const unsigned p = *(const unsigned*)(xq + k0);
            const float s = xs[k0 >> 6];
            v[i].x = (float)(signed char)(p & 0xFF) * s; v[i].y = (float)(signed char)((p >> 8) & 0xFF) * s;
            v[i].z = (float)(signed char)((p >> 16) & 0xFF) * s; v[i].w = (float)(signed char)(p >> 24) * s;
        }
        q27_quant_group<IPG>(v, yq, ys, g * IPG * Q8_GS + (t << 2), g, red);
    }
}
extern "C" int q27_requant_g64_tok(const signed char* xq, const float* xs, signed char* yq, float* ys, int K, int xstride, int M, int gs, hipStream_t s)
{
    if (!xq || !xs || !yq || !ys || (K % Q8_GS) || M < 1 || (xstride & 63)) return 0;
    const int NU = K / Q8_GS;
    if (gs == 1024) hipLaunchKernelGGL((q27_k_requant_g64_tok<1>), dim3((unsigned)M), dim3(256), 0, s, xq, xs, yq, ys, K, xstride, xstride >> 6);
    else if ((gs == 0 || gs == K) && NU == 4) hipLaunchKernelGGL((q27_k_requant_g64_tok<4>), dim3((unsigned)M), dim3(256), 0, s, xq, xs, yq, ys, K, xstride, xstride >> 6);
    else if ((gs == 0 || gs == K) && NU == 6) hipLaunchKernelGGL((q27_k_requant_g64_tok<6>), dim3((unsigned)M), dim3(256), 0, s, xq, xs, yq, ys, K, xstride, xstride >> 6);
    else if ((gs == 0 || gs == K) && NU == 5) hipLaunchKernelGGL((q27_k_requant_g64_tok<5>), dim3((unsigned)M), dim3(256), 0, s, xq, xs, yq, ys, K, xstride, xstride >> 6);
    else return 0;
    return 1;
}
// ---- slab sum: y[t][n] = alpha * sum_g acc[g][t][n] * sw[n][g] * sx[t][g] ---------------------------------
__device__ __forceinline__ float4 q27_slab4(const int* __restrict__ acc, size_t MN, int ldc, int t, int n0,
                                            const float* __restrict__ sw, int ng, const float* __restrict__ sx, float alpha)
{
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int g = 0; g < ng; ++g) {
        const int4 q = *(const int4*)(acc + (size_t)g * MN + (size_t)t * ldc + n0);
        const float st = sx[(size_t)t * ng + g];
        o.x = fmaf((float)q.x, sw[(size_t)(n0    ) * ng + g] * st, o.x);
        o.y = fmaf((float)q.y, sw[(size_t)(n0 + 1) * ng + g] * st, o.y);
        o.z = fmaf((float)q.z, sw[(size_t)(n0 + 2) * ng + g] * st, o.z);
        o.w = fmaf((float)q.w, sw[(size_t)(n0 + 3) * ng + g] * st, o.w);
    }
    o.x *= alpha; o.y *= alpha; o.z *= alpha; o.w *= alpha;
    return o;
}

// ---- int32 slabs -> int8 + per-16 scales (attention / GDN input format) ---------------------------------------
__global__ __launch_bounds__(256) void q27_k_epi_q8(const int* __restrict__ acc, int ldc, int N, int M,
        const float* __restrict__ srow, int ng, const float* __restrict__ stok, float alpha,
        signed char* __restrict__ yq, float* __restrict__ ys, int ystride, int sstride)
{
    const int t = blockIdx.y, g16 = blockIdx.x * 256 + threadIdx.x;   // one 16-column group per thread
    if (g16 >= (N >> 4)) return;
    const int n0 = g16 << 4;
    const size_t MN = (size_t)M * ldc;
    float4 v[4]; float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = q27_slab4(acc, MN, ldc, t, n0 + 4 * j, srow, ng, stok, alpha);
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[j].x), fabsf(v[j].y)), fmaxf(fabsf(v[j].z), fabsf(v[j].w))));
    }
    const float sc = mx * (1.f / 127.f), qi = (sc > 0.f) ? 1.f / sc : 0.f;
    uint4 o; o.x = q27_pack4(v[0].x, v[0].y, v[0].z, v[0].w, qi); o.y = q27_pack4(v[1].x, v[1].y, v[1].z, v[1].w, qi);
    o.z = q27_pack4(v[2].x, v[2].y, v[2].z, v[2].w, qi); o.w = q27_pack4(v[3].x, v[3].y, v[3].z, v[3].w, qi);
    *(uint4*)(yq + (size_t)t * ystride + n0) = o;
    ys[(size_t)t * sstride + g16] = sc;
}
extern "C" int q27_epi_q8(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha,
                          signed char* yq, float* ys, int ystride, int M, hipStream_t s)
{
    if (!acc || !srow || !stok || !yq || !ys || (N & 15) || (ystride & 15) || (ldc & 3) || M < 1 || ng < 1) return 0;
    hipLaunchKernelGGL(q27_k_epi_q8, dim3((unsigned)(((N >> 4) + 255) / 256), (unsigned)M), dim3(256), 0, s, acc, ldc, N, M, srow, ng, stok, alpha, yq, ys, ystride, ystride >> 4);
    return 1;
}

// ---- int32 slabs -> fp32 [M][N] (+ rs * residual) -------------------------------------------------------------
__global__ __launch_bounds__(256) void q27_k_epi_f32r(const int* __restrict__ acc, int N, int M, const float* __restrict__ srow, int ng,
        const float* __restrict__ stok, float alpha, float* __restrict__ y, const float* __restrict__ radd, float rs)
{
    const int t = blockIdx.y, i = (blockIdx.x * 256 + threadIdx.x) << 2;
    if (i >= N) return;
    float4 o = q27_slab4(acc, (size_t)M * N, N, t, i, srow, ng, stok, alpha);
    if (radd) { const float4 r = *(const float4*)(radd + (size_t)t * N + i); o.x += rs * r.x; o.y += rs * r.y; o.z += rs * r.z; o.w += rs * r.w; }
    *(float4*)(y + (size_t)t * N + i) = o;
}
// Same epilogue with explicit leading dimensions: acc from a stacked (ldc-wide) GEMM output, y rows of a wider buffer.
__global__ __launch_bounds__(256) void q27_k_epi_f32r_ld(const int* __restrict__ acc, int ldc, int N, int M, const float* __restrict__ srow, int ng,
        const float* __restrict__ stok, float alpha, float* __restrict__ y, int ldy)
{
    const int t = blockIdx.y, i = (blockIdx.x * 256 + threadIdx.x) << 2;
    if (i >= N) return;
    const float4 o = q27_slab4(acc, (size_t)M * ldc, ldc, t, i, srow, ng, stok, alpha);
    *(float4*)(y + (size_t)t * ldy + i) = o;
}
extern "C" int q27_epi_f32r_ld(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha,
                               float* y, int ldy, int M, hipStream_t s)
{
    if (!acc || !srow || !stok || !y || (N & 3) || (ldy & 3) || M < 1 || ng != 1) return 0;
    hipLaunchKernelGGL(q27_k_epi_f32r_ld, dim3((unsigned)((N / 4 + 255) / 256), (unsigned)M), dim3(256), 0, s, acc, ldc, N, M, srow, ng, stok, alpha, y, ldy);
    return 1;
}
extern "C" int q27_epi_f32r(const int* acc, int N, const float* srow, int ng, const float* stok, float alpha, float* y,
                            const float* radd, float rs, int M, hipStream_t s)
{
    if (!acc || !srow || !stok || !y || (N & 3) || M < 1 || ng < 1) return 0;
    hipLaunchKernelGGL(q27_k_epi_f32r, dim3((unsigned)((N / 4 + 255) / 256), (unsigned)M), dim3(256), 0, s, acc, N, M, srow, ng, stok, alpha, y, radd, rs);
    return 1;
}

// ---- swiglu from two int32 slab sets -> per-(token,group) int8 (the down GEMM's input; N = 17 groups) ----------
template <int NU, int IPG>   // NU output units of 1024 (N = NU * 1024); a group = IPG units
__global__ __launch_bounds__(256) void q27_k_epi_swiglu_tok(const int* __restrict__ ag, const int* __restrict__ au, int N, int M,
        const float* __restrict__ sg, const float* __restrict__ su, int ng, const float* __restrict__ stok, float alpha_g, float alpha_u, float gscale,
        signed char* __restrict__ xq, float* __restrict__ xs)
{
    constexpr int NGO = NU / IPG;
    const int t = blockIdx.x, tid = threadIdx.x;
    xq += (size_t)t * N; xs += (size_t)t * NGO;
    const size_t MN = (size_t)M * N;
    __shared__ float red[4];
    for (int go = 0; go < NGO; ++go) {
        float4 h[IPG];
#pragma unroll
        for (int i = 0; i < IPG; ++i) {
            const int n0 = (go * IPG + i) * Q8_GS + (tid << 2);
            const float4 g = q27_slab4(ag, MN, N, t, n0, sg, ng, stok, alpha_g);
            const float4 u = q27_slab4(au, MN, N, t, n0, su, ng, stok, alpha_u);
            h[i].x = g.x / (1.0f + __expf(-g.x)) * u.x * gscale; h[i].y = g.y / (1.0f + __expf(-g.y)) * u.y * gscale;
            h[i].z = g.z / (1.0f + __expf(-g.z)) * u.z * gscale; h[i].w = g.w / (1.0f + __expf(-g.w)) * u.w * gscale;
        }
        q27_quant_group<IPG>(h, xq, xs, go * IPG * Q8_GS + (tid << 2), go, red);
    }
}
// full-K variant with 1024 threads per token: 4x the waves of the 256-thread kernel (latency-bound otherwise).
// NU4 = float4 count per token (N/4); each thread holds ceil(NU4/1024) float4.
template <int PER>
__global__ __launch_bounds__(1024) void q27_k_epi_swiglu_tok1k(const int* __restrict__ ag, const int* __restrict__ au, int N, int M,
        const float* __restrict__ sg, const float* __restrict__ su, const float* __restrict__ stok, float alpha_g, float alpha_u, float gscale,
        signed char* __restrict__ xq, float* __restrict__ xs)
{
    const int t = blockIdx.x, tid = threadIdx.x, NU4 = N >> 2;
    xq += (size_t)t * N;
    const int* g4 = ag + (size_t)t * N;
    const int* u4 = au + (size_t)t * N;
    const float stg = stok[t] * alpha_g, stu = stok[t] * alpha_u;
    __shared__ float red[16];
    float4 h[PER]; float mx = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = i * 1024 + tid;
        if (idx < NU4) {
            const int n0 = idx << 2;
            const int4 gq = *(const int4*)(g4 + n0), uq = *(const int4*)(u4 + n0);
            const float4 sgv = *(const float4*)(sg + n0), suv = *(const float4*)(su + n0);
            const float gx = (float)gq.x * sgv.x * stg, gy = (float)gq.y * sgv.y * stg, gz = (float)gq.z * sgv.z * stg, gw = (float)gq.w * sgv.w * stg;
            const float ux = (float)uq.x * suv.x * stu, uy = (float)uq.y * suv.y * stu, uz = (float)uq.z * suv.z * stu, uw = (float)uq.w * suv.w * stu;
            h[i].x = gx / (1.0f + __expf(-gx)) * ux * gscale; h[i].y = gy / (1.0f + __expf(-gy)) * uy * gscale;
            h[i].z = gz / (1.0f + __expf(-gz)) * uz * gscale; h[i].w = gw / (1.0f + __expf(-gw)) * uw * gscale;
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(h[i].x), fabsf(h[i].y)), fmaxf(fabsf(h[i].z), fabsf(h[i].w))));
        } else h[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    mx = fmaxf(mx, __shfl_xor(mx, 32, 64)); mx = fmaxf(mx, __shfl_xor(mx, 16, 64)); mx = fmaxf(mx, __shfl_xor(mx, 8, 64));
    mx = fmaxf(mx, __shfl_xor(mx, 4, 64)); mx = fmaxf(mx, __shfl_xor(mx, 2, 64)); mx = fmaxf(mx, __shfl_xor(mx, 1, 64));
    if ((tid & 63) == 0) red[tid >> 6] = mx;
    __syncthreads();
    if (tid < 16) { float v = red[tid]; v = fmaxf(v, __shfl_xor(v, 8, 16)); v = fmaxf(v, __shfl_xor(v, 4, 16)); v = fmaxf(v, __shfl_xor(v, 2, 16)); v = fmaxf(v, __shfl_xor(v, 1, 16)); if (tid == 0) red[0] = v; }
    __syncthreads();
    const float sc = red[0] * (1.f / 127.f), qi = (sc > 0.f) ? 1.f / sc : 0.f;
    if (tid == 0) xs[t] = sc;
#pragma unroll
    for (int i = 0; i < PER; ++i) { const int idx = i * 1024 + tid; if (idx < NU4) *(unsigned*)(xq + (idx << 2)) = q27_pack4(h[i].x, h[i].y, h[i].z, h[i].w, qi); }
}
extern "C" int q27_epi_swiglu_tok(const int* ag, const int* au, int N, const float* sg, const float* su, int ng, const float* stok,
                                  float alpha_g, float alpha_u, float gscale, signed char* xq, float* xs, int M, int gs, hipStream_t s)
{
    if (!ag || !au || !sg || !su || !stok || !xq || !xs || N < 1 || (N % Q8_GS) || M < 1 || ng < 1 || !(gscale > 0.f)) {
        std::fprintf(stderr, "q27_epi_swiglu_tok: N=%d M=%d ng=%d gscale=%g\n", N, M, ng, (double)gscale); return 0; }
    const int NU = N / Q8_GS;
    if (NU == 17 && gs == 1024) hipLaunchKernelGGL((q27_k_epi_swiglu_tok<17, 1>), dim3((unsigned)M), dim3(256), 0, s, ag, au, N, M, sg, su, ng, stok, alpha_g, alpha_u, gscale, xq, xs);
    else if (NU == 17 && (gs == 0 || gs == N) && ng == 1) hipLaunchKernelGGL((q27_k_epi_swiglu_tok1k<5>), dim3((unsigned)M), dim3(1024), 0, s, ag, au, N, M, sg, su, stok, alpha_g, alpha_u, gscale, xq, xs);   // 17408/4 = 4352 float4 = 4.25 per thread
    else if (NU == 17 && (gs == 0 || gs == N)) hipLaunchKernelGGL((q27_k_epi_swiglu_tok<17, 17>), dim3((unsigned)M), dim3(256), 0, s, ag, au, N, M, sg, su, ng, stok, alpha_g, alpha_u, gscale, xq, xs);
    else { std::fprintf(stderr, "q27_epi_swiglu_tok: unsupported N=%d gs=%d\n", N, gs); return 0; }
    return 1;
}

// ---- GDN a/b projection on rocBLAS: bf16 [48][5120] x2 -> one int8 per-row tensor [96][5120] (at load) ----------
__global__ __launch_bounds__(256) void q27_k_bf16_rows_to_i8r(const unsigned short* __restrict__ wa, const unsigned short* __restrict__ wb,
                                                              int rows, int K, signed char* __restrict__ q, float* __restrict__ s)
{
    const int r = blockIdx.x, t = threadIdx.x;   // r in [0, 2*rows): a rows then b rows
    const unsigned short* w = (r < rows) ? (wa + (size_t)r * K) : (wb + (size_t)(r - rows) * K);
    __shared__ float red[256];
    float mx = 0.f;
    for (int k = t; k < K; k += 256) mx = fmaxf(mx, fabsf(q27_bf2f(w[k])));
    red[t] = mx; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (t < o) red[t] = fmaxf(red[t], red[t + o]); __syncthreads(); }
    const float sc = red[0] * (1.f / 127.f), qi = (sc > 0.f) ? 1.f / sc : 0.f;
    if (t == 0) s[r] = sc;
    for (int k = t; k < K; k += 256) { int c = (int)rintf(q27_bf2f(w[k]) * qi); c = c < -127 ? -127 : (c > 127 ? 127 : c); q[(size_t)r * K + k] = (signed char)c; }
}
extern "C" int q27_ab_to_i8r_conv(const unsigned short* wa, const unsigned short* wb, int rows, int K, q27_i8r_t* dst, hipStream_t st)
{
    if (!wa || !wb || !dst || rows < 1 || (K & 63)) return 0;
    signed char* q = nullptr; float* s = nullptr;
    if (hipMalloc((void**)&q, (size_t)2 * rows * K) != hipSuccess) return 0;
    if (hipMalloc((void**)&s, (size_t)2 * rows * 4) != hipSuccess) { hipFree(q); return 0; }
    hipLaunchKernelGGL(q27_k_bf16_rows_to_i8r, dim3((unsigned)(2 * rows)), dim3(256), 0, st, wa, wb, rows, K, q, s);
    dst->w = q; dst->s = s; dst->alpha = 1.f; dst->rows = 2 * rows; dst->K = K; dst->ng = 1; dst->gs = K;
    return 1;
}
// int32 [M][2*rows] -> ya[M][rows], yb[M][rows] (fp32), y = acc * s[row] * stok[t] * xmul
__global__ __launch_bounds__(256) void q27_k_epi_ab(const int* __restrict__ acc, int rows, const float* __restrict__ s,
                                                    const float* __restrict__ stok, float xmul, float* __restrict__ ya, float* __restrict__ yb, int M)
{
    const int i = blockIdx.x * 256 + threadIdx.x, n2 = 2 * rows;
    const int t = i / n2, r = i - t * n2;
    if (t >= M) return;
    const float v = (float)acc[(size_t)t * n2 + r] * s[r] * stok[t] * xmul;
    if (r < rows) ya[(size_t)t * rows + r] = v; else yb[(size_t)t * rows + (r - rows)] = v;
}
extern "C" int q27_epi_ab(const int* acc, int rows, const float* s, const float* stok, float xmul, float* ya, float* yb, int M, hipStream_t st)
{
    if (!acc || !s || !stok || !ya || !yb || rows < 1 || M < 1) return 0;
    hipLaunchKernelGGL(q27_k_epi_ab, dim3((unsigned)((2 * rows * M + 255) / 256)), dim3(256), 0, st, acc, rows, s, stok, xmul, ya, yb, M);
    return 1;
}

// ---- int32 slabs -> int8 + per-16 scales, v2: 4 columns per thread, 4 lanes per 16-group (4x the waves of v1) ----
__global__ __launch_bounds__(256) void q27_k_epi_q8v2(const int* __restrict__ acc, int ldc, int N, int M,
        const float* __restrict__ srow, int ng, const float* __restrict__ stok, float alpha,
        signed char* __restrict__ yq, float* __restrict__ ys, int ystride, int sstride)
{
    const int t = blockIdx.y, c4 = blockIdx.x * 256 + threadIdx.x;   // 4 columns per thread
    if (c4 >= (N >> 2)) return;
    const int n0 = c4 << 2;
    const float4 v = q27_slab4(acc, (size_t)M * ldc, ldc, t, n0, srow, ng, stok, alpha);
    float mx = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    mx = fmaxf(mx, __shfl_xor(mx, 1, 4)); mx = fmaxf(mx, __shfl_xor(mx, 2, 4));
    const float sc = mx * (1.f / 127.f), qi = (sc > 0.f) ? 1.f / sc : 0.f;
    *(unsigned*)(yq + (size_t)t * ystride + n0) = q27_pack4(v.x, v.y, v.z, v.w, qi);
    if ((c4 & 3) == 0) ys[(size_t)t * sstride + (c4 >> 2)] = sc;
}
extern "C" int q27_epi_q8v2(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha,
                            signed char* yq, float* ys, int ystride, int M, hipStream_t s)
{
    if (!acc || !srow || !stok || !yq || !ys || (N & 15) || (ystride & 15) || (ldc & 3) || M < 1 || ng < 1) return 0;
    hipLaunchKernelGGL(q27_k_epi_q8v2, dim3((unsigned)(((N >> 2) + 255) / 256), (unsigned)M), dim3(256), 0, s, acc, ldc, N, M, srow, ng, stok, alpha, yq, ys, ystride, ystride >> 4);
    return 1;
}

// ---- plain fp32 copy kernel (peer-store handoff: the producer writes the block's residual straight into the next ring
// card's pend buffer through the peer mapping; replaces hipMemcpyPeerAsync + its runtime staging) -------------------
__global__ __launch_bounds__(256) void q27_k_copy_f32(const float4* __restrict__ src, float4* __restrict__ dst, size_t n4)
{
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}
extern "C" int q27_copy_f32(const float* src, float* dst, size_t n, hipStream_t s)
{
    if (!src || !dst || (n & 3) || (((uintptr_t)src | (uintptr_t)dst) & 15u)) return 0;
    const size_t n4 = n >> 2;
    hipLaunchKernelGGL(q27_k_copy_f32, dim3((unsigned)((n4 + 255) / 256)), dim3(256), 0, s, (const float4*)src, (float4*)dst, n4);
    return 1;
}

// ---- bf16 rows -> fp32 rows (the sweep embedding into pend_slots; the host only gathers bf16) ---------------------
__global__ __launch_bounds__(256) void q27_k_bf16_to_f32(const unsigned short* __restrict__ src, float* __restrict__ dst, size_t n)
{
    const size_t i = ((size_t)blockIdx.x * 256 + threadIdx.x) * 4;
    if (i + 3 < n) {
        const ushort4 v = *(const ushort4*)(src + i);
        float4 o; o.x = q27_bf2f(v.x); o.y = q27_bf2f(v.y); o.z = q27_bf2f(v.z); o.w = q27_bf2f(v.w);
        *(float4*)(dst + i) = o;
    } else for (size_t j = i; j < n; ++j) dst[j] = q27_bf2f(src[j]);
}
extern "C" int q27_bf16_to_f32(const unsigned short* src, float* dst, size_t n, hipStream_t s)
{
    if (!src || !dst || n == 0) return 0;
    hipLaunchKernelGGL(q27_k_bf16_to_f32, dim3((unsigned)((n / 4 + 255) / 256)), dim3(256), 0, s, src, dst, n);
    return 1;
}

// ---- FUSED: o/out_proj epilogue (+ residual) and the post-attention norm, one block per token -------------------
// part[t] = acc[t] * srow * stok[t] * alpha + pend[t]   (written: the down epilogue adds it back)
// xq1[t]  = int8 per-token quant of  part * rms_inv * (plus + gamma) * sc_in ;  xst1[t] = scale ; hid32 = part (last layer)
// Saves one launch and one 5 MB round trip of `part` per layer-chunk. ng == 1 (full-K scales) only.
__global__ __launch_bounds__(256) void q27_k_epi_res_rn_tok(
        const int* __restrict__ acc, const float* __restrict__ srow, const float* __restrict__ stok, float alpha,
        const float* __restrict__ pend, float* __restrict__ part, const unsigned short* __restrict__ wgt, float* __restrict__ hid32,
        signed char* __restrict__ xq, float* __restrict__ xs, float plus, float sc_in)
{
    constexpr int N = Q27_HID, NU = N / 1024;   // 5 units of 256 float4
    const int c = blockIdx.x, t = threadIdx.x;
    acc += (size_t)c * N; pend += (size_t)c * N; part += (size_t)c * N; xq += (size_t)c * N;
    if (hid32) hid32 += (size_t)c * N;
    const float st = stok[c] * alpha;
    __shared__ float red[4];
    float4 v[NU]; float ss = 0.f;
#pragma unroll
    for (int e = 0; e < NU; ++e) {
        const int k0 = (e * 256 + t) << 2;
        const int4 q = *(const int4*)(acc + k0);
        const float4 sr = *(const float4*)(srow + k0);
        const float4 pv = *(const float4*)(pend + k0);
        v[e].x = fmaf((float)q.x * sr.x, st, pv.x); v[e].y = fmaf((float)q.y * sr.y, st, pv.y);
        v[e].z = fmaf((float)q.z * sr.z, st, pv.z); v[e].w = fmaf((float)q.w * sr.w, st, pv.w);
        *(float4*)(part + k0) = v[e];
        if (hid32) *(float4*)(hid32 + k0) = v[e];
        ss = fmaf(v[e].x, v[e].x, ss); ss = fmaf(v[e].y, v[e].y, ss); ss = fmaf(v[e].z, v[e].z, ss); ss = fmaf(v[e].w, v[e].w, ss);
    }
    ss += __shfl_xor(ss, 32, 64); ss += __shfl_xor(ss, 16, 64); ss += __shfl_xor(ss, 8, 64); ss += __shfl_xor(ss, 4, 64); ss += __shfl_xor(ss, 2, 64); ss += __shfl_xor(ss, 1, 64);
    if ((t & 63) == 0) red[t >> 6] = ss;
    __syncthreads();
    const float tot = (red[0] + red[1]) + (red[2] + red[3]);
    __syncthreads();
    const float inv = 1.0f / sqrtf(tot / (float)N + Q27_RMS_EPS);
    float mx = 0.f;
#pragma unroll
    for (int e = 0; e < NU; ++e) {
        const int k0 = (e * 256 + t) << 2;
        const ushort4 wv = *(const ushort4*)(wgt + k0);
        v[e].x = v[e].x * inv * (plus + q27_bf2f(wv.x)) * sc_in; v[e].y = v[e].y * inv * (plus + q27_bf2f(wv.y)) * sc_in;
        v[e].z = v[e].z * inv * (plus + q27_bf2f(wv.z)) * sc_in; v[e].w = v[e].w * inv * (plus + q27_bf2f(wv.w)) * sc_in;
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[e].x), fabsf(v[e].y)), fmaxf(fabsf(v[e].z), fabsf(v[e].w))));
    }
    mx = q27_blockmax256(mx, red);
    const float sc = mx * (1.f / 127.f), qi = (sc > 0.f) ? 1.f / sc : 0.f;
    if (t == 0) xs[c] = sc;
#pragma unroll
    for (int e = 0; e < NU; ++e) { const int k0 = (e * 256 + t) << 2; *(unsigned*)(xq + k0) = q27_pack4(v[e].x, v[e].y, v[e].z, v[e].w, qi); }
}
extern "C" int q27_epi_res_rn_tok(const int* acc, int N, const float* srow, int ng, const float* stok, float alpha, const float* pend, float* part,
                                  const unsigned short* wgt, float* hid32, signed char* xq, float* xs, int plus_one, float in_scale, int M, hipStream_t s)
{
    if (N != Q27_HID || ng != 1 || !acc || !srow || !stok || !pend || !part || !wgt || !xq || !xs || !(in_scale > 0.f) || M < 1) return 0;
    hipLaunchKernelGGL(q27_k_epi_res_rn_tok, dim3((unsigned)M), dim3(256), 0, s, acc, srow, stok, alpha, pend, part, wgt, hid32, xq, xs, plus_one ? 1.f : 0.f, 1.f / in_scale);
    return 1;
}
