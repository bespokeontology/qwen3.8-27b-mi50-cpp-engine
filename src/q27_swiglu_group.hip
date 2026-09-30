// Fixed four-lane quantization groups for the native trunk only.
// Local donor: q27_epi_q8v2; keep the original SwiGLU expression and byte layout.
#include "q27.h"
#include "q27_nr.h"
#include <cstdio>
#include <cstdlib>

__global__ __launch_bounds__(64) void q27_k_swiglu_quant_group4(
        const float* __restrict__ G, const float* __restrict__ U,
        signed char* __restrict__ Q, float* __restrict__ XS,
        int n, float gscale) {
    const int c = blockIdx.y;
    const int t = blockIdx.x * 64 + threadIdx.x;
    const int g = t >> 2, lane = t & 3;
    if (g >= (n >> 4)) return;  // whole four-lane subgroups leave together
    const size_t row = (size_t)c * n;
    const size_t base = row + ((size_t)g << 4);
    float v[4], mx = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dp = lane * 4 + j;
        const int src = (dp < 8) ? 2 * dp : 2 * (dp - 8) + 1;
        const float a = G[base + src];
        const float b = U[base + src];
        const float sw = a / (1.0f + __expf(-a));
        v[j] = sw * b * gscale;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    mx = fmaxf(mx, __shfl_xor(mx, 1, 4));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 4));
    const float sc = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    if (lane == 0) XS[(size_t)c * (n >> 4) + g] = sc;
    unsigned packed = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int q = (int)rintf(v[j] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        packed |= (unsigned)(q & 255) << (j * 8);
    }
    *(unsigned*)(Q + base + lane * 4) = packed;
}

__global__ __launch_bounds__(64) void q27_k_quant_group4(
        const float* __restrict__ Y,
        signed char* __restrict__ Q, float* __restrict__ XS,
        int n, float gscale) {
    const int c = blockIdx.y;
    const int t = blockIdx.x * 64 + threadIdx.x;
    const int g = t >> 2, lane = t & 3;
    if (g >= (n >> 4)) return;
    const size_t row = (size_t)c * n;
    const size_t base = row + ((size_t)g << 4);
    float v[4], mx = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dp = lane * 4 + j;
        const int src = (dp < 8) ? 2 * dp : 2 * (dp - 8) + 1;
        v[j] = Y[base + src] * gscale;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    mx = fmaxf(mx, __shfl_xor(mx, 1, 4));
    mx = fmaxf(mx, __shfl_xor(mx, 2, 4));
    const float sc = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    if (lane == 0) XS[(size_t)c * (n >> 4) + g] = sc;
    unsigned packed = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int q = (int)rintf(v[j] * inv);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        packed |= (unsigned)(q & 255) << (j * 8);
    }
    *(unsigned*)(Q + base + lane * 4) = packed;
}

// Y is already silu(gate)*up (Q27_GU_SWILU); same permutation and scale as q27_swiglu_quant_group4.
extern "C" int q27_quant_group4(const float* y,
        signed char* xq, float* xs, int n, float in_scale, int C, hipStream_t s) {
    if (!y || !xq || !xs || n <= 0 || (n & 15) || !(in_scale > 0.f) || C < 1 || C > 4) return 0;
    if (((uintptr_t)y | (uintptr_t)xq) & 15u) return 0;
    hipLaunchKernelGGL(q27_k_quant_group4,
        dim3((unsigned)(((n >> 4) + 15) / 16), (unsigned)C), dim3(64), 0, s,
        y, xq, xs, n, 1.0f / in_scale);
    return 1;
}

extern "C" int q27_swiglu_quant_group4(const float* gate, const float* up,
        signed char* xq, float* xs, int n, float in_scale, int C, hipStream_t s) {
    if (!gate || !up || !xq || !xs || n <= 0 || (n & 15) ||
        !(in_scale > 0.f) || C < 1 || C > 4) return 0;
    if (((uintptr_t)gate | (uintptr_t)up | (uintptr_t)xq) & 15u) return 0;
    hipLaunchKernelGGL(q27_k_swiglu_quant_group4,
        dim3((unsigned)(((n >> 4) + 15) / 16), (unsigned)C), dim3(64), 0, s,
        gate, up, xq, xs, n, 1.0f / in_scale);
    return 1;
}

extern "C" int q27_swiglu_quant_trunk(const float* gate, const float* up,
        signed char* xq, float* xs, int n, float in_scale, int C, hipStream_t s) {
    static const bool enabled = [] { const char* e = std::getenv("Q27_SWIGLU_GROUP4"); return e && std::atoi(e) != 0; }();
    if (enabled && C >= 1 && C <= 4 &&
        q27_swiglu_quant_group4(gate, up, xq, xs, n, in_scale, C, s)) {
        static thread_local bool logged = false;
        if (!logged) {
            int device = -1; hipGetDevice(&device);
            std::fprintf(stderr, "Q27_SWIGLU_GROUP4 first_call device=%d n=%d NR=%d lanes=4 block=64 trunk=1\n", device, n, C);
            logged = true;
        }
        return 1;
    }
    return q27_swiglu_quant_b(gate, up, xq, xs, n, in_scale, C, s);
}
