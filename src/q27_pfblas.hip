// Default-off bounded-slab FP32 attention for the serialized layer-split prefill.
// No persistent expanded cache, context-sized allocation, decode or MTP changes.
#include "q27_pfblas.h"
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cmath>

static constexpr float Q27_PFBLAS_NEG = -1.0e30f;
static constexpr float Q27_PFBLAS_SCALE_L2E = 0.090168440055560215f; // Frozen Q27_SCALE_L2E.
static_assert(Q27_NHEAD == 24 && Q27_NKV == 4 && Q27_HDIM == 256 && Q27_KVROWS == 1024,
              "pfblas only supports the frozen full-width Qwen geometry");
static std::atomic<unsigned long long> pfblas_count[8];
struct q27_pfblas_receipt {
    ~q27_pfblas_receipt() {
        for (int d = 0; d < 8; ++d) {
            const auto n = pfblas_count[d].load(std::memory_order_relaxed);
            if (n) std::fprintf(stderr, "Q27_PF_BLAS receipt dev=%d enqueued_calls=%llu B=4096 fp32=1\n", d, n);
        }
    }
};
static q27_pfblas_receipt pfblas_receipt;

extern "C" int q27_pfblas_enabled() {
    static const bool on = q27_env_flag("Q27_PF_BLAS", false);
    return on;
}
extern "C" int q27_pfblas_force() {
    static const bool force = q27_env_flag("Q27_PF_BLAS_FORCE", false);
    return force;
}
extern "C" int q27_pfblas_should_run(int pos0, int M) {
    // 8192 = original prefill split_rows(256) * split cap(32). Past that
    // prefix, kb2 must lengthen each split's dependent loop. Full M preserves
    // the packed-GQA reuse (R1536). Force bypasses only this dispatch policy;
    // all pointer/shape/arena/stream eligibility checks still apply.
    return q27_pfblas_enabled() &&
           (q27_pfblas_force() || (M == Q27_PFBLAS_MAX_M && pos0 >= 8192));
}
static const char* q27_pfblas_mode() {
    return !q27_pfblas_enabled() ? "off" : (q27_pfblas_force() ? "forced" : "long");
}
extern "C" void q27_pfblas_log_config() {
    std::fprintf(stderr, "Q27_PF_BLAS config enabled=%d force=%d mode=%s normal_M=256 normal_min_pos0=8192\n",
                 q27_pfblas_enabled(), q27_pfblas_force(), q27_pfblas_mode());
}
extern "C" unsigned long long q27_pfblas_calls(int dev) {
    return (dev >= 0 && dev < 8) ? pfblas_count[dev].load(std::memory_order_relaxed) : 0;
}
extern "C" size_t q27_pfblas_scratch_bytes(int M) {
    if (M < 1 || M > Q27_PFBLAS_MAX_M) return 0;
    // Every region is a multiple of 256 bytes even at M=1. Eight statistics
    // arrays reserve 192 KiB at M256; four are used, four stay untouched.
    const size_t rows = (size_t)Q27_PFBLAS_H * Q27_PFBLAS_G * M;
    const size_t vec = rows * Q27_PFBLAS_D * sizeof(float);
    const size_t kv = (size_t)Q27_PFBLAS_H * Q27_PFBLAS_B * Q27_PFBLAS_D * sizeof(float);
    const size_t prob = rows * Q27_PFBLAS_B * sizeof(float);
    const size_t stats = (rows * sizeof(float) + 255) & ~(size_t)255;
    return 3 * vec + 2 * kv + prob + 8 * stats;
}
extern "C" int q27_pfblas_layout(void* arena, size_t bytes, int M, q27_pfblas_scratch* out) {
    const size_t need = q27_pfblas_scratch_bytes(M);
    if (!out || !arena || !need || bytes < need || ((uintptr_t)arena & 255u) ||
        (uintptr_t)arena > UINTPTR_MAX - need) return 0;
    q27_pfblas_scratch a{};
    a.M = M; a.R = Q27_PFBLAS_G * M; a.rows = Q27_PFBLAS_H * a.R; a.bytes = need;
    const size_t vec = (size_t)a.rows * Q27_PFBLAS_D * sizeof(float);
    const size_t kv = (size_t)Q27_PFBLAS_H * Q27_PFBLAS_B * Q27_PFBLAS_D * sizeof(float);
    const size_t prob = (size_t)a.rows * Q27_PFBLAS_B * sizeof(float);
    const size_t stats = ((size_t)a.rows * sizeof(float) + 255) & ~(size_t)255;
    char* p = (char*)arena;
    auto take = [&](size_t n) { float* v = (float*)p; p += n; return v; };
    a.q = take(vec); a.k = take(kv); a.v = take(kv); a.p = take(prob);
    a.u = take(vec); a.o = take(vec);
    a.m = take(stats); a.l = take(stats); a.bm = take(stats); a.bl = take(stats);
    p += 4 * stats;
    if ((size_t)(p - (char*)arena) != need) return 0;
    *out = a;
    return 1;
}

__device__ __forceinline__ float q27_pfblas_exp2(float x) {
    float y; asm("v_exp_f32_e32 %0, %1" : "=v"(y) : "v"(x)); return y;
}
template<bool Max>
__device__ __forceinline__ float q27_pfblas_reduce256(float v, float* red) {
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    for (int d = 32; d; d >>= 1) {
        const float b = __shfl_down(v, d, 64);
        v = Max ? fmaxf(v, b) : v + b;
    }
    if (!lane) red[wave] = v;
    __syncthreads();
    v = t < 4 ? red[t] : (Max ? Q27_PFBLAS_NEG : 0.0f);
    if (!wave) {
        for (int d = 32; d; d >>= 1) {
            const float b = __shfl_down(v, d, 64);
            v = Max ? fmaxf(v, b) : v + b;
        }
        if (!lane) red[0] = v;
    }
    __syncthreads();
    const float result = red[0];
    __syncthreads();
    return result;
}

__global__ void q27_pfblas_pack_kernel(q27_pfblas_scratch a, const float* q) {
    const int d = threadIdx.x, row = blockIdx.x;
    const int kh = row / a.R, r = row % a.R, c = r / Q27_PFBLAS_G;
    const int h = kh * Q27_PFBLAS_G + r % Q27_PFBLAS_G;
    const size_t off = (size_t)row * Q27_PFBLAS_D + d;
    a.q[off] = q[((size_t)c * Q27_NHEAD + h) * Q27_PFBLAS_D + d];
    a.o[off] = 0.0f;
    if (!d) { a.m[row] = Q27_PFBLAS_NEG; a.l[row] = 0.0f; }
}
__global__ void q27_pfblas_expand_kernel(q27_pfblas_scratch a,
        const signed char* kc, const q27_kvs_t* ks,
        const signed char* vc, const q27_kvs_t* vs, int k0, int b) {
#if Q27_V4
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q27_PFBLAS_H * b * 32) return;
    const int dg = i & 31, row = (i >> 5) % b, kh = (i >> 5) / b;
    const size_t src = ((size_t)(k0 + row) * Q27_PFBLAS_H + kh) * Q27_PFBLAS_D + dg * 8;
    const size_t dst = ((size_t)kh * Q27_PFBLAS_B + row) * Q27_PFBLAS_D + dg * 8;
    const float kscale = q27_kvs_ld(ks[src >> 4]);
    const unsigned vw = *(const unsigned*)(vc + (src >> 1));
    float vf[8]; q27_v4_recon(vw, q27_v4s_ld(vs[src >> 5]), vf);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        a.k[dst + j] = (float)kc[src + j] * kscale;
        a.v[dst + j] = vf[j];
    }
#endif
}
__global__ void q27_pfblas_softmax_kernel(q27_pfblas_scratch a, int pos0, int k0, int b) {
    __shared__ float red[4];
    const int row = blockIdx.x, t = threadIdx.x;
    const int c = (row % a.R) / Q27_PFBLAS_G;
    const int last = pos0 + c - k0;
    float score[16];
    float mx = Q27_PFBLAS_NEG;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int j = t + i * 256;
        score[i] = (j < b && j <= last) ? a.p[(size_t)row * Q27_PFBLAS_B + j] * Q27_PFBLAS_SCALE_L2E : Q27_PFBLAS_NEG;
        mx = fmaxf(mx, score[i]);
    }
    mx = q27_pfblas_reduce256<true>(mx, red);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int j = t + i * 256;
        const float p = (j < b && j <= last) ? q27_pfblas_exp2(score[i] - mx) : 0.0f;
        if (j < b) a.p[(size_t)row * Q27_PFBLAS_B + j] = p;
        sum += p;
    }
    sum = q27_pfblas_reduce256<false>(sum, red);
    if (!t) { a.bm[row] = mx; a.bl[row] = sum; }
}
__global__ void q27_pfblas_merge_kernel(q27_pfblas_scratch a) {
    __shared__ float factor[2];
    const int row = blockIdx.x, d = threadIdx.x;
    if (!d) {
        const float old_l = a.l[row], new_l = a.bl[row];
        const float mx = fmaxf(a.m[row], a.bm[row]);
        factor[0] = old_l > 0.0f ? q27_pfblas_exp2(a.m[row] - mx) : 0.0f;
        factor[1] = new_l > 0.0f ? q27_pfblas_exp2(a.bm[row] - mx) : 0.0f;
        a.m[row] = mx;
        a.l[row] = fmaf(old_l, factor[0], new_l * factor[1]);
    }
    __syncthreads();
    const size_t off = (size_t)row * Q27_PFBLAS_D + d;
    a.o[off] = fmaf(a.o[off], factor[0], a.u[off] * factor[1]);
}

// Shared by the product epilogue and standalone FP32 capture. Four contiguous
// lanes per block-32 must participate together in the two XOR shuffle stages.
__device__ __forceinline__ void q27_pfblas_normalize8(float v[8], const float* rotated, float den, int dg) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = rotated[dg * 8 + j];
#if Q27_V4
    q27_had32r(v, dg);
#endif
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] *= inv;
}
// All 256 dimensions participate with the same c/h, as in the frozen combine.
__device__ __forceinline__ void q27_pfblas_store_quant(float z,
        const signed char* raw, const float* raw_sc, int qstride,
        signed char* out, float* out_sc, int ostride,
        int c, int h, int d, float qgs, int g64) {
    const int gk = h * 512 + Q27_PFBLAS_D + d;
    const float gt = (float)raw[(size_t)c * qstride + gk] * raw_sc[(size_t)c * (qstride >> 4) + (gk >> 4)];
    const float sg = 1.0f / (1.0f + __expf(-gt));
    const float y = z * sg;
    float mx = fabsf(y);
    if (g64) { for (int off = 32; off; off >>= 1) mx = fmaxf(mx, __shfl_down(mx, off, 64)); mx = __shfl(mx, 0, 64); }
    else { for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_down(mx, off, 16)); mx = __shfl(mx, 0, 16); }
    const float sc = mx * qgs * (1.0f / 127.0f);
    const float qi = sc > 0.0f ? qgs / sc : 0.0f;
    const int k = h * Q27_PFBLAS_D + d;
    if (g64) { if (!(d & 63)) out_sc[(size_t)c * (ostride >> 6) + (k >> 6)] = sc; }
    else { if (!(d & 15)) out_sc[(size_t)c * (ostride >> 4) + (k >> 4)] = sc; }
    int q = (int)rintf(y * qi);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    out[(size_t)c * ostride + k] = (signed char)q;
}
__global__ void q27_pfblas_finish_kernel(q27_pfblas_scratch a,
        const signed char* raw, const float* raw_sc, int qstride,
        signed char* out, float* out_sc, int ostride, float qgs, int g64) {
    __shared__ float z[Q27_PFBLAS_D];
    const int row = blockIdx.x, t = threadIdx.x;
    const int kh = row / a.R, r = row % a.R, c = r / Q27_PFBLAS_G;
    const int h = kh * Q27_PFBLAS_G + r % Q27_PFBLAS_G;
    if (t < 64) {
        const int dg = t & 31;
        float v[8]; q27_pfblas_normalize8(v, a.o + (size_t)row * Q27_PFBLAS_D, a.l[row], dg);
        if (t < 32) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) z[dg * 8 + j] = v[j];
        }
    }
    __syncthreads();
    q27_pfblas_store_quant(z[t], raw, raw_sc, qstride, out, out_sc, ostride, c, h, t, qgs, g64);
}

static int q27_pfblas_launch_status(const char* what) {
    const hipError_t e = hipPeekAtLastError();
    if (e == hipSuccess) return 1;
    std::fprintf(stderr, "Q27_PF_BLAS %s launch failed: %s\n", what, hipGetErrorString(e));
    return -1;
}
extern "C" int q27_pfblas_pack(const q27_pfblas_scratch* a, const float* q, hipStream_t s) {
    hipLaunchKernelGGL(q27_pfblas_pack_kernel, dim3(a->rows), dim3(256), 0, s, *a, q);
    return q27_pfblas_launch_status("pack");
}
extern "C" int q27_pfblas_expand(const q27_pfblas_scratch* a,
        const signed char* kc, const q27_kvs_t* ks, const signed char* vc, const q27_kvs_t* vs,
        int k0, int b, hipStream_t s) {
    if (k0 < 0 || b < 1 || b > Q27_PFBLAS_B) return 0;
    hipLaunchKernelGGL(q27_pfblas_expand_kernel, dim3((Q27_PFBLAS_H * b * 32 + 255) / 256), dim3(256), 0, s, *a, kc, ks, vc, vs, k0, b);
    return q27_pfblas_launch_status("expand");
}
extern "C" int q27_pfblas_qk(int dev, const q27_pfblas_scratch* a, int b, hipStream_t s) {
    return q27_rb_pfblas_qk(dev, a->k, a->q, a->p, b, a->R, s) ? 1 : -1;
}
extern "C" int q27_pfblas_softmax(const q27_pfblas_scratch* a, int pos0, int k0, int b, hipStream_t s) {
    hipLaunchKernelGGL(q27_pfblas_softmax_kernel, dim3(a->rows), dim3(256), 0, s, *a, pos0, k0, b);
    return q27_pfblas_launch_status("softmax");
}
extern "C" int q27_pfblas_pv(int dev, const q27_pfblas_scratch* a, int b, hipStream_t s) {
    return q27_rb_pfblas_pv(dev, a->v, a->p, a->u, b, a->R, s) ? 1 : -1;
}
extern "C" int q27_pfblas_merge(const q27_pfblas_scratch* a, hipStream_t s) {
    hipLaunchKernelGGL(q27_pfblas_merge_kernel, dim3(a->rows), dim3(256), 0, s, *a);
    return q27_pfblas_launch_status("merge");
}
extern "C" int q27_pfblas_run(int dev, const float* q,
        const signed char* kc, const q27_kvs_t* ks, const signed char* vc, const q27_kvs_t* vs,
        int pos0, int M, void* arena, size_t bytes, hipStream_t s) {
    q27_pfblas_scratch a;
    if (!Q27_V4 || dev < 0 || dev >= 8 || !q || !kc || !ks || !vc || !vs || pos0 < 0 ||
        M < 1 || M > Q27_PFBLAS_MAX_M || pos0 > INT_MAX - M ||
        !q27_pfblas_layout(arena, bytes, M, &a)) return 0;
    if (q27_pfblas_pack(&a, q, s) != 1) return -1;
    const int length = pos0 + M;
    for (int k0 = 0; k0 < length;) {
        const int b = length - k0 < Q27_PFBLAS_B ? length - k0 : Q27_PFBLAS_B;
        if (q27_pfblas_expand(&a, kc, ks, vc, vs, k0, b, s) != 1 ||
            q27_pfblas_qk(dev, &a, b, s) != 1 ||
            q27_pfblas_softmax(&a, pos0, k0, b, s) != 1 ||
            q27_pfblas_pv(dev, &a, b, s) != 1 ||
            q27_pfblas_merge(&a, s) != 1) return -1;
        k0 += b;
    }
    const auto n = pfblas_count[dev].fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1) std::fprintf(stderr, "Q27_PF_BLAS first_call dev=%d M=%d R=%d pos0=%d context=%d B=4096 arena_used=%zu candidate=1 mode=%s\n",
                            dev, M, a.R, pos0, length, a.bytes, q27_pfblas_enabled() ? q27_pfblas_mode() : "direct");
    return 1;
}
extern "C" int q27_pfblas_finish(const q27_pfblas_scratch* a,
        const signed char* raw, const float* raw_sc, int qstride,
        signed char* out, float* out_sc, int ostride, float in_scale, int g64, hipStream_t s) {
    if (!a || !raw || !raw_sc || !out || !out_sc || qstride < Q27_QROWS || (qstride & 15) ||
        ostride < Q27_OROWS || (g64 != 0 && g64 != 1) || (ostride & (g64 ? 63 : 15)) ||
        !(in_scale > 0.0f) || !std::isfinite(in_scale) || !std::isfinite(1.0f / in_scale)) return 0;
    hipLaunchKernelGGL(q27_pfblas_finish_kernel, dim3(a->rows), dim3(256), 0, s, *a, raw, raw_sc, qstride, out, out_sc, ostride, 1.0f / in_scale, g64);
    return q27_pfblas_launch_status("finish");
}
