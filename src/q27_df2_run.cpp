// q27_df2_run.cpp — DFlash2 conditioning + KV injection (milestone 4). No BF16 hot-path weights:
// every GEMM rides q27_proj_i8g_rows on the int8 repack; only the tiny quant/merge helpers are new.
#include "q27_df2_run.h"
#include "q27_nr.h"
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "q27_df2_run:%d hip error %d\n", __LINE__, (int)e_); std::abort(); } } while (0)

namespace {
int fail(char* err, size_t cap, int code, const char* fmt, ...) {
    if (err && cap) { va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap); }
    return code;
}
__global__ void q27_k_f2bf(const float* __restrict__ x, unsigned short* __restrict__ y, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = q27_f2bf(x[i]);
}
// bf16 rows (srcstride) -> int8 per-16 groups + fp32 scales; dst rows at xqstride/xsstride
__global__ void q27_k_df2_quant(const unsigned short* __restrict__ src, int srcstride,
                                signed char* __restrict__ xq, int xqstride, float* __restrict__ xs, int xsstride,
                                int rows, int n) {
    const int r = blockIdx.y;
    const int ng = n >> 4;
    const unsigned short* sr = src + (size_t)r * srcstride;
    signed char* xr = xq + (size_t)r * xqstride;
    float* xsr = xs + (size_t)r * xsstride;
    for (int g = blockIdx.x; g < ng; g += gridDim.x) {
        const int b = g * 16;
        float amax = 1e-12f;
        for (int i = 0; i < 16; ++i) {
            unsigned u = (unsigned)sr[b + i] << 16; float f; std::memcpy(&f, &u, 4);
            float a = fabsf(f); if (a > amax) amax = a;
        }
        const float sc = amax / 127.f, isc = 127.f / amax;
        for (int i = 0; i < 16; ++i) {
            unsigned u = (unsigned)sr[b + i] << 16; float f; std::memcpy(&f, &u, 4);
            int v = (int)lrintf(f * isc); if (v > 127) v = 127; if (v < -127) v = -127;
            xr[b + i] = (signed char)v;
        }
        xsr[g] = sc;
    }
}
} // namespace

static void q27_df2_f2bf(const float* x, unsigned short* y, int n, hipStream_t s) {
    hipLaunchKernelGGL(q27_k_f2bf, dim3((n + 255) / 256), dim3(256), 0, s, x, y, n);
}
static void q27_df2_quant_bf16(const unsigned short* src, int srcstride, signed char* xq, int xqstride,
                               float* xs, int xsstride, int rows, int n, hipStream_t s) {
    const int ng = n >> 4;
    dim3 grid((ng < 64 ? ng : 64), rows), blk(128);
    hipLaunchKernelGGL(q27_k_df2_quant, grid, blk, 0, s, src, srcstride, xq, xqstride, xs, xsstride, rows, n);
}

int q27_df2_ctx_alloc(q27_df2_ctx_t* C, int dev, char* err, size_t errcap) {
    hipError_t e0 = hipSetDevice(dev); if (e0 != hipSuccess) return fail(err,errcap,1,"df2 ctx hipSetDevice %d: %d", dev, (int)e0);
    size_t bytes = 0;
    auto A = [&](void** p, size_t b) { if (hipMalloc(p, b) != hipSuccess) return false; hipMemset(*p, 0, b); bytes += b; return true; };
    if (!A((void**)&C->tap_xq, (size_t)8 * 25600) || !A((void**)&C->tap_xs, (size_t)8 * 1600 * 4) ||
        !A((void**)&C->g_part, (size_t)8 * 1280 * 4) || !A((void**)&C->g_full, (size_t)8 * 5120 * 4) ||
        !A((void**)&C->ssp, (size_t)8 * 4) || !A((void**)&C->g_bf16, (size_t)8 * 5120 * 2) ||
        !A((void**)&C->g_xq, (size_t)8 * 5120) || !A((void**)&C->g_xs, (size_t)8 * 320 * 4) ||
        !A((void**)&C->kvbf, (size_t)8 * 2048 * 2)) return fail(err,errcap,1,"df2 ctx alloc failed");
    for (int l = 0; l < 5; ++l) {
        if (!A((void**)&C->kc[l], (size_t)2048 * 1024) || !A((void**)&C->ks[l], (size_t)2048 * 64 * 4) ||
            !A((void**)&C->vc[l], (size_t)2048 * 1024) || !A((void**)&C->vs[l], (size_t)2048 * 64 * 4))
            return fail(err,errcap,1,"df2 kv cache alloc failed");
    }
    if (!A((void**)&C->hidden, (size_t)8*5120*2) || !A((void**)&C->norm, (size_t)8*5120*2) ||
        !A((void**)&C->dyn, (size_t)8*1280*2) || !A((void**)&C->conv, (size_t)8*5120*2) ||
        !A((void**)&C->n_xq, (size_t)8*5120) || !A((void**)&C->n_xs, (size_t)8*320*4) ||
        !A((void**)&C->c_xq, (size_t)8*5120) || !A((void**)&C->c_xs, (size_t)8*320*4) ||
        !A((void**)&C->qbf, (size_t)8*4096*2) || !A((void**)&C->kbf, (size_t)8*1024*2) ||
        !A((void**)&C->vbf, (size_t)8*1024*2) || !A((void**)&C->qp, (size_t)8*4096*2) ||
        !A((void**)&C->kp, (size_t)8*1024*2) || !A((void**)&C->attn, (size_t)8*4096*2) ||
        !A((void**)&C->a_xq, (size_t)8*4096) || !A((void**)&C->a_xs, (size_t)8*256*4) ||
        !A((void**)&C->mixer, (size_t)8*5120*2) || !A((void**)&C->mixer2, (size_t)8*5120*2) ||
        !A((void**)&C->pa, (size_t)8*17408*4) || !A((void**)&C->pb, (size_t)8*17408*4) ||
        !A((void**)&C->sw_xq, (size_t)8*17408) || !A((void**)&C->sw_xs, (size_t)8*1088*4) ||
        !A((void**)&C->down_part, (size_t)8*5120*4) || !A((void**)&C->mlp_full, (size_t)8*5120*4) ||
        !A((void**)&C->anchor, (size_t)5120*2) || !A((void**)&C->mask, (size_t)5120*2) ||
        !A((void**)&C->logits, (size_t)7*62080*4) ||
        !A((void**)&C->ws_val, (size_t)7*32*16*4) || !A((void**)&C->ws_id, (size_t)7*32*16*4) ||
        !A((void**)&C->cand_ids, (size_t)7*16*4) || !A((void**)&C->cand_val, (size_t)7*16*4) ||
        !A((void**)&C->sh, (size_t)7*256*2) || !A((void**)&C->path, (size_t)7*4))
        return fail(err,errcap,1,"df2 executor alloc failed");
    if (hipHostMalloc((void**)&C->cand_h, (size_t)7*16*4, hipHostMallocDefault) != hipSuccess ||
        hipHostMalloc((void**)&C->candv_h, (size_t)7*16*4, hipHostMallocDefault) != hipSuccess)
        return fail(err,errcap,1,"df2 selector host alloc failed");
    C->ctx_begin = 0; C->ctx_end = 0; C->bytes = bytes;
    std::printf("Q27_DFLASH2 ctx card %d: %.3f GiB buffers (KV caches %.1f MB)\n", dev, (double)bytes/1073741824.0,
                (double)(5 * ((size_t)2048*1024 + (size_t)2048*64*4 + (size_t)2048*1024 + (size_t)2048*64*4)) / 1048576.0);
    return 0;
}


__device__ float g_df2_attn_dbg[3];
// SwiGLU + per-16 int8 quantize in NATURAL order. q27_swiglu_quant_b (the main engine's) writes the
// PERMUTED layout that q27_proj_nvfp4 consumes (dp = (j&1) ? 8+(j>>1) : j>>1); q27_df2_gemm is fed
// everywhere else by q27_df2_quant_bf16 in natural order, so using the permuted one shuffled every
// operand inside each 16-group -> correct magnitude, destroyed direction (measured cos 0.28).
__global__ void q27_k_df2_swiglu_q16(const float* __restrict__ G, const float* __restrict__ U,
                                     signed char* __restrict__ Q, float* __restrict__ XS, int ngrp, int n) {
    const int c = blockIdx.y;
    const int g = blockIdx.x * 256 + threadIdx.x;
    if (g >= ngrp) return;
    const size_t base = (size_t)c * n + ((size_t)g << 4);
    float v[16]; float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float a = G[base + j], b = U[base + j];
        v[j] = (a / (1.0f + __expf(-a))) * b;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    const float sc = mx * (1.0f / 127.0f);
    const float inv = (sc > 0.f) ? (1.0f / sc) : 0.f;
    XS[(size_t)c * ngrp + g] = sc;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int cq = (int)rintf(v[j] * inv);
        cq = cq < -127 ? -127 : (cq > 127 ? 127 : cq);
        Q[base + j] = (signed char)cq;
    }
}
// Batched 8-row RMSNorm for the draft block. The forward called q27_rmsnorm in a per-row loop
// (8 launches x 2 norms x 5 layers + 8 for the final norm = 88 launches per block, all tiny).
// One launch, one block per row. w-only (plus_one=0), matching the reference's correction_weight=false.
__global__ __launch_bounds__(256) void q27_k_df2_rmsnorm_rows(const unsigned short* __restrict__ x,
        const unsigned short* __restrict__ w, unsigned short* __restrict__ y, int n) {
    const int row = blockIdx.x; const int t = threadIdx.x;
    const unsigned short* xr = x + (size_t)row * n; unsigned short* yr = y + (size_t)row * n;
    __shared__ float red[256];
    float s = 0.f;
    for (int i = t; i < n; i += 256) { const float v = q27_bf2f(xr[i]); s += v * v; }
    red[t] = s; __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (t < st) red[t] += red[t + st]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / (float)n + 1.0e-6f);
    for (int i = t; i < n; i += 256) yr[i] = q27_f2bf(q27_bf2f(xr[i]) * inv * q27_bf2f(w[i]));
}
__global__ void q27_k_df2_conv(const unsigned short* __restrict__ in, const unsigned short* __restrict__ dyn,
                               const float* __restrict__ base, unsigned short* __restrict__ out,
                               int rows, int hidden, int branch) {
    const int index = blockIdx.x * 256 + threadIdx.x;
    if (index >= rows * hidden) return;
    const int row = index / hidden, channel = index - row * hidden;
    const int groups = hidden >> 4, group = channel >> 4;
    float result = 0.0f;
#pragma unroll
    for (int off = 0; off < 2; ++off) {
        if (row < off) continue;
        const float v = q27_bf2f(in[(size_t)(row - off) * hidden + channel]);
        const float b = base[(size_t)(branch * 2 + off) * hidden + channel];
        const float d = q27_bf2f(dyn[(size_t)row * (4 * groups) + (size_t)branch * (2 * groups) + (size_t)off * groups + group]);
        result = fmaf(b + d, v, result);
    }
    out[index] = q27_f2bf(result);
}

__global__ void q27_k_df2_prepqk(const unsigned short* __restrict__ proj, const unsigned short* __restrict__ nw,
                                 unsigned short* __restrict__ out, int rows, int heads, int pos0) {
    const int row = blockIdx.x / heads, head = blockIdx.x - row * heads;
    const int dim = threadIdx.x;
    __shared__ float sh[128];
    const float raw = q27_bf2f(proj[((size_t)row * heads + head) * 128 + dim]);
    sh[dim] = raw * raw;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) { if (dim < st) sh[dim] += sh[dim + st]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / 128.0f + 1.0e-6f);
    float value = raw * inv * q27_bf2f(nw[dim]);
    const int half = 64, pair = dim < half ? dim + half : dim - half;
    const float pv = q27_bf2f(proj[((size_t)row * heads + head) * 128 + pair]) * inv * q27_bf2f(nw[pair]);
    const int freq = dim % half;
    const float invf = powf(10000000.0f, -2.0f * (float)freq / 128.0f);
    const float angle = (float)(pos0 + row) * invf;
    value = value * cosf(angle) + (dim < half ? -pv : pv) * sinf(angle);
    out[((size_t)row * heads + head) * 128 + dim] = q27_f2bf(value);
}

__global__ void q27_k_df2_add(unsigned short* __restrict__ a, const unsigned short* __restrict__ b, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) a[i] = q27_f2bf(q27_bf2f(a[i]) + q27_bf2f(b[i]));
}
// KV injection: the reference's normalize_rope_head EXACTLY (128-dim RMS, x w, NEOX base 1e7) + per-16
// int8 quantize into the persistent caches. Replaces q27_attn_kvprep_nr, whose target-model conventions
// (1+w norm, 256-dim scope, target rope base) put the context keys in a different basis than the block keys.
__global__ void q27_k_df2_kvinj(const unsigned short* __restrict__ kproj, const unsigned short* __restrict__ vproj,
                                const unsigned short* __restrict__ knw,
                                signed char* __restrict__ kc, float* __restrict__ ks,
                                signed char* __restrict__ vc, float* __restrict__ vs,
                                int pos0, int nr, int window) {
    const int row = blockIdx.x / 8, head = blockIdx.x - row * 8, dim = threadIdx.x;
    __shared__ float sh[128];
    __shared__ float gmax[8];
    const size_t src = ((size_t)row * 8 + head) * 128 + dim;
    const float raw = q27_bf2f(kproj[src]);
    sh[dim] = raw * raw;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) { if (dim < st) sh[dim] += sh[dim + st]; __syncthreads(); }
    const float inv = rsqrtf(sh[0] / 128.0f + 1.0e-6f);
    float value = raw * inv * q27_bf2f(knw[dim]);
    const int half = 64, pair = dim < half ? dim + half : dim - half;
    const float pv = q27_bf2f(kproj[((size_t)row * 8 + head) * 128 + pair]) * inv * q27_bf2f(knw[pair]);
    const int freq = dim % half;
    const float invf = powf(10000000.0f, -2.0f * (float)freq / 128.0f);
    const float angle = (float)(pos0 + row) * invf;
    value = value * cosf(angle) + (dim < half ? -pv : pv) * sinf(angle);
    const float vraw = q27_bf2f(vproj[src]);
    float kam = fabsf(value);
    for (int o = 8; o; o >>= 1) kam = fmaxf(kam, __shfl_xor(kam, o, 16));
    float vam = fabsf(vraw);
    for (int o = 8; o; o >>= 1) vam = fmaxf(vam, __shfl_xor(vam, o, 16));
    const int grp = dim >> 4;
    if ((dim & 15) == 0) { gmax[grp] = kam; }
    __syncthreads();
    const float ksc = gmax[grp] / 127.0f;
    const signed char qk = (signed char)__float2int_rn(value / ksc);
    // v: separate per-16 scale
    __shared__ float gmaxv[8];
    if ((dim & 15) == 0) gmaxv[grp] = vam;
    __syncthreads();
    const float vsc = gmaxv[grp] / 127.0f;
    const signed char qv = (signed char)__float2int_rn(vraw / vsc);
    const int p = (pos0 + row) % window;
    const size_t off = (size_t)p * 1024 + (size_t)head * 128 + dim;
    kc[off] = qk; vc[off] = qv;
    if ((dim & 15) == 0) { ks[(size_t)p * 64 + (size_t)head * 8 + grp] = ksc; vs[(size_t)p * 64 + (size_t)head * 8 + grp] = vsc; }
}
// tap capture: tap = bf16(fp32 residual-2 + bf16 residual-1) = the FULL residual entering the next layer
__global__ void q27_k_df2_tapadd_k(const float* __restrict__ acc, const unsigned short* __restrict__ hid,
                                   unsigned short* __restrict__ tap, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) tap[i] = q27_f2bf(acc[i] + q27_bf2f(hid[i]));
}
extern "C" void q27_k_df2_tapadd(const float* acc, const unsigned short* hid, unsigned short* tap, int n, hipStream_t s) {
    hipLaunchKernelGGL(q27_k_df2_tapadd_k, dim3((n + 255) / 256), dim3(256), 0, s, acc, hid, tap, n);
}
__global__ void q27_k_df2_tapadd_f_k(const float* __restrict__ a, const float* __restrict__ b, unsigned short* __restrict__ tap, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) tap[i] = q27_f2bf(a[i] + b[i]);
}
extern "C" void q27_k_df2_tapadd_f(const float* a, const float* b, unsigned short* tap, int n, hipStream_t s) {
    hipLaunchKernelGGL(q27_k_df2_tapadd_f_k, dim3((n + 255) / 256), dim3(256), 0, s, a, b, tap, n);
}
extern "C" void q27_k_df2_f2bfv(const float* src, unsigned short* dst, int n, hipStream_t s) {
    hipLaunchKernelGGL(q27_k_f2bf, dim3((n + 255) / 256), dim3(256), 0, s, src, dst, n);
}

__global__ void q27_k_df2_attn(const unsigned short* __restrict__ q, const unsigned short* __restrict__ bk,
                               const unsigned short* __restrict__ bv,
                               const signed char* __restrict__ ckc, const float* __restrict__ cks,
                               const signed char* __restrict__ cvc, const float* __restrict__ cvs,
                               unsigned short* __restrict__ out, int rows, int qpos0, int cbegin, int cend, int window,
                               int dbg_attn) {
    const int row = blockIdx.x / 32, qh = blockIdx.x - row * 32, kh = qh >> 2, dim = threadIdx.x;
    __shared__ float sh[128];
    __shared__ float sc_shared;
    const float qv = q27_bf2f(q[((size_t)row * 32 + qh) * 128 + dim]);
    const int qpos = qpos0 + row;
    float mx = -FLT_MAX, den = 0.0f, acc = 0.0f;
    float ctxw = 0.0f, blkw = 0.0f;   // context vs block softmax mass (debug)
    const int vbegin = cbegin > qpos - window + 1 ? cbegin : qpos - window + 1;
    for (int tok = vbegin; tok < cend; ++tok) {
        const size_t ci = (size_t)(tok % window) * 1024 + (size_t)kh * 128 + dim;
        const float ks = cks[(size_t)(tok % window) * 64 + (size_t)kh * 8 + (dim >> 4)];
        const float vs = cvs[(size_t)(tok % window) * 64 + (size_t)kh * 8 + (dim >> 4)];
        sh[dim] = qv * ((float)ckc[ci] * ks);
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) { if (dim < st) sh[dim] += sh[dim + st]; __syncthreads(); }
        if (dim == 0) sc_shared = sh[0] * 0.08838834764831845f;
        __syncthreads();
        const float sc = sc_shared, nm = fmaxf(mx, sc);
        const float pf = expf(mx - nm), cf = expf(sc - nm);
        acc = acc * pf + cf * ((float)cvc[ci] * vs);
        den = den * pf + cf; ctxw = ctxw * pf + cf; mx = nm;
    }
    for (int nr = 0; nr < rows; ++nr) {
        const size_t ni = ((size_t)nr * 8 + kh) * 128 + dim;
        sh[dim] = qv * q27_bf2f(bk[ni]);
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) { if (dim < st) sh[dim] += sh[dim + st]; __syncthreads(); }
        if (dim == 0) sc_shared = sh[0] * 0.08838834764831845f;
        __syncthreads();
        const float sc = sc_shared, nm = fmaxf(mx, sc);
        const float pf = expf(mx - nm), cf = expf(sc - nm);
        acc = acc * pf + cf * q27_bf2f(bv[ni]);
        den = den * pf + cf; blkw = blkw * pf + cf; mx = nm;
    }
    out[((size_t)row * 32 + qh) * 128 + dim] = q27_f2bf(acc / den);
    if (dbg_attn && dim == 0 && row == 1 && qh == 0) { g_df2_attn_dbg[0] = ctxw; g_df2_attn_dbg[1] = blkw; g_df2_attn_dbg[2] = mx; }
}

// block forward: returns a checksum (sum of |bf16| of the final norm rows 1..7) per call via *csum
int q27_df2_forward(q27_df2_ctx_t* C, int pos, q27_df2_reduce_fn reduce, void* uctx, int g, hipStream_t s,
                    float* csum, char* err, size_t errcap) {
    static const bool df2_noctx = q27_env_flag("Q27_DFLASH2_NOCTX", false);   // bisect: block with NO context at all
    const int cb_eff = C->ctx_begin, ce_eff = df2_noctx ? C->ctx_begin : C->ctx_end;
    (void)g;
    q27_df2_t& W = C->W;
    const int NR = 8;
    // block input: row 0 = anchor embed, rows 1..7 = mask embed
    CK(hipMemcpyAsync(C->hidden, C->anchor, 5120 * 2, hipMemcpyDeviceToDevice, s));
    for (int r = 1; r < 8; ++r) CK(hipMemcpyAsync(C->hidden + (size_t)r * 5120, C->mask, 5120 * 2, hipMemcpyDeviceToDevice, s));
    { CK(hipStreamSynchronize(s)); unsigned short a0[4], h0[4];
      CK(hipMemcpy(a0, C->anchor, 8, hipMemcpyDeviceToHost)); CK(hipMemcpy(h0, C->hidden, 8, hipMemcpyDeviceToHost));
      std::printf("Q27_DFLASH2_FWDIN anch=%04x %04x hid=%04x %04x\n", a0[0], a0[1], h0[0], h0[1]); }
    const bool dbg = q27_env_flag("Q27_DFLASH2_FWD_DBG", false);
    auto ck = [&](const char* tag, const unsigned short* p, size_t off) {
        if (!dbg) return;   // this lambda used to sync+D2H unconditionally: 30 blocking syncs per block forward
        CK(hipStreamSynchronize(s));
        unsigned short h[32]; CK(hipMemcpy(h, p + off, 64, hipMemcpyDeviceToHost));
        float cs = 0.f; for (int i = 0; i < 32; ++i) { unsigned u = (unsigned)h[i] << 16; float f; std::memcpy(&f, &u, 4); cs += fabsf(f); }
        if (dbg) std::printf("Q27_DFLASH2_DBG %s csum=%.4f\n", tag, cs);
    };
    ck("hidden_in", C->hidden, 0);
    static int df2_dump_done = 0;
    FILE* df2df = nullptr;
    if (!df2_dump_done && g == 0 && q27_env_flag("Q27_DFLASH2_DUMP", false)) {   // one block, layer by layer, for the host BF16 oracle
        df2df = fopen("/tmp/df2_block_dump.bin", "wb");
        if (df2df) {
            std::vector<unsigned short> t(5120);
            CK(hipStreamSynchronize(s));
            CK(hipMemcpy(t.data(), C->anchor, 5120*2, hipMemcpyDeviceToHost)); fwrite(t.data(), 2, 5120, df2df);
            CK(hipMemcpy(t.data(), C->mask,   5120*2, hipMemcpyDeviceToHost)); fwrite(t.data(), 2, 5120, df2df);
            std::printf("Q27_DFLASH2_DUMP pos=%d ctx=[%d,%d) writing /tmp/df2_block_dump.bin\n", pos, C->ctx_begin, C->ctx_end);
        }
    }
    for (int l = 0; l < 5; ++l) {
        hipLaunchKernelGGL(q27_k_df2_rmsnorm_rows, dim3(8), dim3(256), 0, s, C->hidden, (const unsigned short*)W.L[l].in_norm, C->norm, 5120);
        ck("norm", C->norm, 0);
        q27_df2_quant_bf16(C->norm, 5120, C->n_xq, 5120, C->n_xs, 320, 8, 5120, s);
        if (!q27_df2_gemm(&W.L[l].attn_conv_proj, C->n_xq, 5120, C->n_xs, (void*)C->dyn, 1280, nullptr, 0, 0.f, 8, -1, s)) return fail(err,errcap,1,"df2 acp declined L%d",l);
        q27_k_df2_conv<<<dim3((8*5120+255)/256), dim3(256), 0, s>>>(C->norm, C->dyn, W.L[l].attn_conv_base, C->conv, 8, 5120, 0);
        q27_df2_quant_bf16(C->conv, 5120, C->c_xq, 5120, C->c_xs, 320, 8, 5120, s);
        if (!q27_df2_gemm(&W.L[l].q, C->c_xq, 5120, C->c_xs, (void*)C->qbf, 4096, nullptr, 0, 0.f, 8, -1, s)) return fail(err,errcap,1,"df2 q declined L%d",l);
        if (!q27_df2_gemm(&W.L[l].k, C->c_xq, 5120, C->c_xs, (void*)C->kbf, 1024, nullptr, 0, 0.f, 8, -1, s)) return fail(err,errcap,1,"df2 k declined L%d",l);
        if (!q27_df2_gemm(&W.L[l].v, C->c_xq, 5120, C->c_xs, (void*)C->vbf, 1024, nullptr, 0, 0.f, 8, -1, s)) return fail(err,errcap,1,"df2 v declined L%d",l);
        q27_k_df2_prepqk<<<dim3(8*32), dim3(128), 0, s>>>(C->qbf, (const unsigned short*)W.L[l].q_norm, C->qp, 8, 32, pos);
        q27_k_df2_prepqk<<<dim3(8*8), dim3(128), 0, s>>>(C->kbf, (const unsigned short*)W.L[l].k_norm, C->kp, 8, 8, pos);
        q27_k_df2_attn<<<dim3(8*32), dim3(128), 0, s>>>(C->qp, C->kp, C->vbf, C->kc[l], C->ks[l], C->vc[l], C->vs[l], C->attn, 8, pos, cb_eff, ce_eff, 2048, dbg ? 1 : 0);
        // (stage checksums live behind Q27_DFLASH2_FWD_DBG in the PRE/POST/CONVFIN dumps; the old
        //  ck() block here read buffers BEFORE this layer's work and misled two debug rounds)
        if (dbg && l == 0) { CK(hipStreamSynchronize(s));
            float ad[3]; CK(hipMemcpyFromSymbol(ad, HIP_SYMBOL(g_df2_attn_dbg), 12, 0, hipMemcpyDeviceToHost));
            std::printf("Q27_DFLASH2_ATTNMASS ctxw=%.6f blkw=%.6f mx=%.4f frac_ctx=%.6f\n", ad[0], ad[1], ad[2], ad[0] + ad[1] > 0.f ? ad[0] / (ad[0] + ad[1]) : -1.f);
            // per-row attention profile: row 0 (anchor) vs rows 1-7 (mask)
            unsigned short hh[32]; float rr[8];
            for (int rr0 = 0; rr0 < 8; ++rr0) { CK(hipMemcpy(hh, C->attn + (size_t)rr0 * 4096, 64, hipMemcpyDeviceToHost));
                float cs = 0.f; for (int i = 0; i < 32; ++i) { unsigned u = (unsigned)hh[i] << 16; float f; std::memcpy(&f, &u, 4); cs += fabsf(f); } rr[rr0] = cs; }
            std::printf("Q27_DFLASH2_ATTNROWS %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
            unsigned short q0[4], k0[4];
            CK(hipMemcpy(q0, C->qp, 8, hipMemcpyDeviceToHost)); CK(hipMemcpy(k0, C->kp, 8, hipMemcpyDeviceToHost));
            signed char kc0[4]; float ks0[2];
            CK(hipMemcpy(kc0, C->kc[l] + (size_t)(pos % 2048) * 1024, 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(ks0, C->ks[l] + (size_t)(pos % 2048) * 64, 8, hipMemcpyDeviceToHost));
            std::printf("Q27_DFLASH2_APROBE qp0=%04x %04x kp0=%04x %04x kc=%d %d %d %d ks=%.6f %.6f\n", q0[0],q0[1],k0[0],k0[1],kc0[0],kc0[1],kc0[2],kc0[3],ks0[0],ks0[1]); }
        q27_df2_quant_bf16(C->attn, 4096, C->a_xq, 4096, C->a_xs, 256, 8, 4096, s);
        if (dbg) { CK(hipStreamSynchronize(s));
            signed char ax[16]; float as2[4]; signed char ow2[16]; float os2[4];
            CK(hipMemcpy(ax, C->a_xq, 16, hipMemcpyDeviceToHost)); CK(hipMemcpy(as2, C->a_xs, 16, hipMemcpyDeviceToHost));
            CK(hipMemcpy(ow2, W.L[l].o.w, 16, hipMemcpyDeviceToHost)); CK(hipMemcpy(os2, W.L[l].o.s, 16, hipMemcpyDeviceToHost));
            std::printf("Q27_DFLASH2_PRE L%d ax=%02x%02x%02x%02x%02x%02x%02x%02x as=%08x %08x ow=%02x%02x%02x%02x%02x%02x%02x%02x os=%08x %08x\n",
                l, ax[0],ax[1],ax[2],ax[3],ax[4],ax[5],ax[6],ax[7], ((unsigned*)as2)[0],((unsigned*)as2)[1],
                ow2[0],ow2[1],ow2[2],ow2[3],ow2[4],ow2[5],ow2[6],ow2[7], ((unsigned*)os2)[0],((unsigned*)os2)[1]); }
        if (!q27_df2_gemm(&W.L[l].o, C->a_xq, 4096, C->a_xs, (void*)C->mixer, 5120, nullptr, 0, 0.f, 8, -1, s)) return fail(err,errcap,1,"df2 o declined L%d",l);
        if (dbg) { CK(hipStreamSynchronize(s));
            unsigned short m0[8]; CK(hipMemcpy(m0, C->mixer, 16, hipMemcpyDeviceToHost));
            std::printf("Q27_DFLASH2_POST_LDS L%d m0=%04x %04x %04x %04x %04x %04x %04x %04x\n", l, m0[0],m0[1],m0[2],m0[3],m0[4],m0[5],m0[6],m0[7]); }
        {   // discriminator: the same o GEMM through the register path in two NR=4 chunks -> mixer2
            if (!q27_df2_gemm(&W.L[l].o, C->a_xq, 4096, C->a_xs, (void*)C->mixer2, 5120, nullptr, 0, 0.f, 4, -1, s)) return fail(err,errcap,1,"df2 o reg declined L%d",l);
            if (!q27_df2_gemm(&W.L[l].o, C->a_xq + 4 * 4096, 4096, C->a_xs + 4 * 256, (void*)(C->mixer2 + 4 * 5120), 5120, nullptr, 0, 0.f, 4, -1, s)) return fail(err,errcap,1,"df2 o reg2 declined L%d",l);
            if (dbg) { CK(hipStreamSynchronize(s));
                unsigned short m2[8]; CK(hipMemcpy(m2, C->mixer2, 16, hipMemcpyDeviceToHost));
                std::printf("Q27_DFLASH2_POST_REG L%d m2=%04x %04x %04x %04x %04x %04x %04x %04x\n", l, m2[0],m2[1],m2[2],m2[3],m2[4],m2[5],m2[6],m2[7]); }
        }
        if (df2df && l == 0) {   // layer-0 stage dumps for the oracle: norm, conv, attn, mixer
            CK(hipStreamSynchronize(s));
            std::vector<unsigned short> t((size_t)8*5120);
            CK(hipMemcpy(t.data(), C->norm,  (size_t)8*5120*2, hipMemcpyDeviceToHost)); fwrite(t.data(), 2, (size_t)8*5120, df2df);
            CK(hipMemcpy(t.data(), C->conv,  (size_t)8*5120*2, hipMemcpyDeviceToHost)); fwrite(t.data(), 2, (size_t)8*5120, df2df);
            CK(hipMemcpy(t.data(), C->attn,  (size_t)8*4096*2, hipMemcpyDeviceToHost)); fwrite(t.data(), 2, (size_t)8*4096, df2df);
            CK(hipMemcpy(t.data(), C->mixer, (size_t)8*5120*2, hipMemcpyDeviceToHost)); fwrite(t.data(), 2, (size_t)8*5120, df2df);
        }
        q27_k_df2_conv<<<dim3((8*5120+255)/256), dim3(256), 0, s>>>(C->mixer, C->dyn, W.L[l].attn_conv_base, C->norm, 8, 5120, 1);
        if (dbg) { CK(hipStreamSynchronize(s));
            unsigned short d0[8], n0[8]; float b0[4], b1[4];
            CK(hipMemcpy(d0, C->dyn, 16, hipMemcpyDeviceToHost)); CK(hipMemcpy(n0, C->norm, 16, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b0, W.L[l].attn_conv_base, 16, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b1, W.L[l].attn_conv_base + 2*5120, 16, hipMemcpyDeviceToHost));
            std::printf("Q27_DFLASH2_CONVFIN L%d dyn=%04x %04x %04x %04x norm=%04x %04x %04x %04x b0=%.4f %.4f %.4f %.4f b1=%.4f %.4f %.4f %.4f\n",
                l, d0[0],d0[1],d0[2],d0[3], n0[0],n0[1],n0[2],n0[3], b0[0],b0[1],b0[2],b0[3], b1[0],b1[1],b1[2],b1[3]); }
        q27_k_df2_add<<<dim3((8*5120+255)/256), dim3(256), 0, s>>>(C->hidden, C->norm, 8*5120);
        // MLP (sharded gate/up/down + one 8-row reduce)
        hipLaunchKernelGGL(q27_k_df2_rmsnorm_rows, dim3(8), dim3(256), 0, s, C->hidden, (const unsigned short*)W.L[l].post_norm, C->norm, 5120);
        q27_df2_quant_bf16(C->norm, 5120, C->n_xq, 5120, C->n_xs, 320, 8, 5120, s);
        if (!q27_df2_gemm(&W.L[l].mlp_conv_proj, C->n_xq, 5120, C->n_xs, (void*)C->dyn, 1280, nullptr, 0, 0.f, 8, -1, s)) return fail(err,errcap,1,"df2 mcp declined L%d",l);
        q27_k_df2_conv<<<dim3((8*5120+255)/256), dim3(256), 0, s>>>(C->norm, C->dyn, W.L[l].mlp_conv_base, C->conv, 8, 5120, 0);
        q27_df2_quant_bf16(C->conv, 5120, C->c_xq, 5120, C->c_xs, 320, 8, 5120, s);
        if (!q27_df2_gemm(&W.L[l].gate, C->c_xq, 5120, C->c_xs, (void*)C->pa, 4352, nullptr, 0, 0.f, 8, 0, s)) return fail(err,errcap,1,"df2 gate declined L%d",l);
        if (!q27_df2_gemm(&W.L[l].up,   C->c_xq, 5120, C->c_xs, (void*)C->pb, 4352, nullptr, 0, 0.f, 8, 0, s)) return fail(err,errcap,1,"df2 up declined L%d",l);
        if (df2df && l == 0) {   // gate/up (fp32, this card's 4352-row slice) before swiglu
            CK(hipStreamSynchronize(s));
            std::vector<float> f((size_t)8*4352);
            CK(hipMemcpy(f.data(), C->pa, (size_t)8*4352*4, hipMemcpyDeviceToHost)); fwrite(f.data(), 4, (size_t)8*4352, df2df);
            CK(hipMemcpy(f.data(), C->pb, (size_t)8*4352*4, hipMemcpyDeviceToHost)); fwrite(f.data(), 4, (size_t)8*4352, df2df);
        }
        hipLaunchKernelGGL(q27_k_df2_swiglu_q16, dim3((4352/16 + 255)/256, 8), dim3(256), 0, s, C->pa, C->pb, C->sw_xq, C->sw_xs, 4352/16, 4352);
        if (!q27_df2_gemm(&W.L[l].down, C->sw_xq, 4352, C->sw_xs, (void*)C->down_part, 5120, nullptr, 0, 0.f, 8, 0, s)) return fail(err,errcap,1,"df2 down declined L%d",l);
        if (!reduce(uctx, g, C->down_part, C->mlp_full, C->ssp, 8, s)) return fail(err,errcap,1,"df2 mlp reduce declined L%d",l);
        q27_df2_f2bf(C->mlp_full, C->mixer, 8 * 5120, s);
        if (df2df && l == 0) {   // MLP-branch stage dumps: post_norm output, mlp conv, reduced MLP output
            CK(hipStreamSynchronize(s));
            std::vector<unsigned short> t((size_t)8*5120);
            CK(hipMemcpy(t.data(), C->mixer, (size_t)8*5120*2, hipMemcpyDeviceToHost)); fwrite(t.data(), 2, (size_t)8*5120, df2df);
        }
        q27_k_df2_conv<<<dim3((8*5120+255)/256), dim3(256), 0, s>>>(C->mixer, C->dyn, W.L[l].mlp_conv_base, C->norm, 8, 5120, 1);
        q27_k_df2_add<<<dim3((8*5120+255)/256), dim3(256), 0, s>>>(C->hidden, C->norm, 8*5120);
        if (df2df) {   // all 8 rows of the residual after this layer, for the host oracle's layerwise cosine
            CK(hipStreamSynchronize(s));
            std::vector<unsigned short> t((size_t)8*5120);
            CK(hipMemcpy(t.data(), C->hidden, (size_t)8*5120*2, hipMemcpyDeviceToHost));
            fwrite(t.data(), 2, (size_t)8*5120, df2df);
        }
    }
    hipLaunchKernelGGL(q27_k_df2_rmsnorm_rows, dim3(8), dim3(256), 0, s, C->hidden, (const unsigned short*)W.final_norm, C->norm, 5120);
    if (!dbg && !df2df) { if (csum) *csum = 0.f; return 0; }   // the csum read is a blocking sync + D2H; only the debug print consumes it
    if (df2df) {   // final-norm rows 0..7 (row 1 is what feeds the lm_head), then close
        CK(hipStreamSynchronize(s));
        std::vector<unsigned short> t((size_t)8*5120);
        CK(hipMemcpy(t.data(), C->norm, (size_t)8*5120*2, hipMemcpyDeviceToHost));
        fwrite(t.data(), 2, (size_t)8*5120, df2df);
        fclose(df2df); df2df = nullptr; df2_dump_done = 1;
        std::printf("Q27_DFLASH2_DUMP complete: 2 embeds + 5 layer residuals + final norm\n");
    }
    // checksum: copy row 4's first 64 bf16 to host
    CK(hipStreamSynchronize(s));
    unsigned short h[64];
    CK(hipMemcpy(h, C->norm + (size_t)4 * 5120, 128, hipMemcpyDeviceToHost));
    float cs = 0.f; for (int i = 0; i < 64; ++i) { unsigned u = (unsigned)h[i] << 16; float f; std::memcpy(&f, &u, 4); cs += fabsf(f); }
    if (csum) *csum = cs;
    return 0;
}

// ============================ DFLASH2 CANDIDATES + SELECTOR (stage A) ============================
// Reference semantics (dflash_kernels.cu): two-stage top-16 over the lm_head logits (isfinite gate,
// higher value wins, tie -> smaller id); selector walk: score(cand) = unary + sum_d
// pred_codebook[prev][d] * hidden_projection[pos][d] * succ_codebook[cand][d], greedy per position.

__global__ void q27_k_df2_top16_s1(const float* __restrict__ logits, int rows, int vocab,
                                   float* __restrict__ ws_val, unsigned* __restrict__ ws_id) {
    const int row = blockIdx.y, part = blockIdx.x;
    __shared__ float sv[256]; __shared__ unsigned si[256]; __shared__ unsigned sel[16];
    const int begin = (int)((long long)vocab * part / 32);
    const int end   = (int)((long long)vocab * (part + 1) / 32);
    const float* rl = logits + (size_t)row * vocab;
    for (int rank = 0; rank < 16; ++rank) {
        float mx = -FLT_MAX; unsigned mi = 0xffffffffu;
        for (int idx = begin + threadIdx.x; idx < end; idx += 256) {
            bool used = false;
            for (int p = 0; p < rank; ++p) if (sel[p] == (unsigned)idx) { used = true; break; }
            if (used) continue;
            const float v = rl[idx];
            if (isfinite(v) && (v > mx || (v == mx && (unsigned)idx < mi))) { mx = v; mi = (unsigned)idx; }
        }
        sv[threadIdx.x] = mx; si[threadIdx.x] = mi;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (threadIdx.x < st && (sv[threadIdx.x + st] > sv[threadIdx.x] ||
                (sv[threadIdx.x + st] == sv[threadIdx.x] && si[threadIdx.x + st] < si[threadIdx.x]))) { sv[threadIdx.x] = sv[threadIdx.x + st]; si[threadIdx.x] = si[threadIdx.x + st]; }
            __syncthreads();
        }
        if (threadIdx.x == 0) { sel[rank] = si[0];
            ws_val[((size_t)row * 32 + part) * 16 + rank] = sv[0];
            ws_id [((size_t)row * 32 + part) * 16 + rank] = si[0]; }
        __syncthreads();
    }
}

__global__ void q27_k_df2_top16_merge(const float* __restrict__ ws_val, const unsigned* __restrict__ ws_id, int rows, int parts,
                                      unsigned* __restrict__ ids, float* __restrict__ vals, unsigned id_base) {
    const int row = blockIdx.x; if (threadIdx.x != 0) return;
    float bv[16]; unsigned bi[16];
    for (int r = 0; r < 16; ++r) { bv[r] = -FLT_MAX; bi[r] = 0xffffffffu; }
    for (int part = 0; part < parts; ++part) {
        for (int rank = 0; rank < 16; ++rank) {
            const float v = ws_val[((size_t)row * parts + part) * 16 + rank];
            const unsigned i = ws_id [((size_t)row * parts + part) * 16 + rank];
            int ins = 16;
            for (int s = 0; s < 16; ++s) if (v > bv[s] || (v == bv[s] && i < bi[s])) { ins = s; break; }
            if (ins == 16) continue;
            for (int s = 15; s > ins; --s) { bv[s] = bv[s - 1]; bi[s] = bi[s - 1]; }
            bv[ins] = v; bi[ins] = i;
        }
    }
    for (int r = 0; r < 16; ++r) { ids[(size_t)row * 16 + r] = bi[r] + id_base; vals[(size_t)row * 16 + r] = bv[r]; }   // GLOBAL vocab ids (the shard offset)
}

// card-0 final merge of the four cards' top-16 + the selector walk (one block, 512 threads)
__global__ void q27_k_df2_selector(const unsigned* __restrict__ cand_ids, const float* __restrict__ cand_val,
                                   const unsigned short* __restrict__ sh, const unsigned short* __restrict__ pred_cb,
                                   const unsigned short* __restrict__ succ_cb, unsigned anchor, unsigned* __restrict__ path,
                                   int positions) {
    __shared__ float scores[16];
    __shared__ unsigned pred;
    if (threadIdx.x == 0) pred = anchor;
    __syncthreads();
    const int cand = threadIdx.x / 32, lane = threadIdx.x & 31;
    for (int pos = 0; pos < positions; ++pos) {
        const unsigned cid = cand_ids[pos * 16 + cand];
        float score = 0.0f;
        for (int d = lane; d < 256; d += 32) {
            const float pv = q27_bf2f(pred_cb[(size_t)pred * 256 + d]);
            const float hv = q27_bf2f(sh[(size_t)pos * 256 + d]);
            const float sv = q27_bf2f(succ_cb[(size_t)cid * 256 + d]);
            score = fmaf(pv * hv, sv, score);
        }
        for (int off = 16; off > 0; off >>= 1) score += __shfl_down(score, off, 32);   // gfx906 wavefront is 64: the default width mixes TWO candidates(32 lanes each) into one score
        if (lane == 0) scores[cand] = score + cand_val[pos * 16 + cand];
        __syncthreads();
        if (threadIdx.x == 0) {
            int best = 0;
            for (int i = 1; i < 16; ++i)
                if (scores[i] > scores[best] || (scores[i] == scores[best] && cand_ids[pos * 16 + i] < cand_ids[pos * 16 + best])) best = i;
            pred = cand_ids[pos * 16 + best];
            path[pos] = pred;
        }
        __syncthreads();
    }
}

// select: lm_head shard logits for the 7 draft rows (final-norm output rows 1..7) -> local top-16
// -> pinned-host exchange (the round's barriers) -> card-0 merge of 4x16 -> selector walk.
int q27_df2_select(q27_df2_ctx_t* C, const q27_nvfp4_t* lmhead, unsigned anchor, int g,
                   q27_df2_exchange_fn exch, void* uctx, hipStream_t s, unsigned* out7, char* err, size_t errcap) {
    q27_df2_t& W = C->W;
    // norm rows 1..7 -> perm-quantized int8 (the lm_head contract)
    q27_df2_quant_bf16(C->norm + 5120, 5120, C->c_xq, 5120, C->c_xs, 320, 7, 5120, s);
    // NOTE: the lm_head wants the PERMUTED activation order (q27_quant_perm). The executor's final
    // norm is natural order; q27_proj_nvfp4_nr consumes the perm form. Use the perm quantizer.
    for (int r = 0; r < 7; ++r)
        q27_quant_perm(C->norm + (size_t)(r + 1) * 5120, C->c_xq + (size_t)r * 5120, C->c_xs + (size_t)r * 320, 5120, lmhead->in_scale, s);
    // the lm_head is FULLY REPLICATED (rows = 62080 on every card), so the projection covers the
    // whole vocab and the top-16 ids are ALREADY global. Adding any shard offset maps them out of
    // vocab; the earlier g*62080 id_base is exactly what poisoned every candidate (e.g. 220264).
    const int vsh = lmhead->rows;
    if (q27_env_flag("Q27_DFLASH2_SEL_DBG", false))
        std::fprintf(stderr, "Q27_DFLASH2_SELSH g=%d vsh=%d anch=%u\n", g, vsh, anchor);
    if (!q27_proj_nvfp4_nr(lmhead, C->c_xq, C->c_xs, C->logits, vsh, 7, s))
        return fail(err,errcap,1,"df2 lmhead declined");
    // local top-16 (two-stage)
    hipLaunchKernelGGL(q27_k_df2_top16_s1, dim3(32, 7), dim3(256), 0, s, C->logits, 7, vsh, C->ws_val, C->ws_id);
    hipLaunchKernelGGL(q27_k_df2_top16_merge, dim3(7), dim3(1), 0, s, C->ws_val, C->ws_id, 7, 32, C->cand_ids, C->cand_val, 0);
    CK(hipMemcpyAsync(C->cand_h, C->cand_ids, (size_t)7*16*4, hipMemcpyDeviceToHost, s));
    CK(hipMemcpyAsync(C->candv_h, C->cand_val, (size_t)7*16*4, hipMemcpyDeviceToHost, s));
    CK(hipStreamSynchronize(s));
    exch(uctx, g, C->cand_h, C->candv_h, 7);   // card 0 receives all four cards' 16s (or broadcast of 7 ids back)
    if (q27_env_flag("Q27_DFLASH2_SEL_DBG", false) && g == 0) {
        std::printf("Q27_DFLASH2_MERGED r0: %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\n",
            C->cand_h[0], C->cand_h[1], C->cand_h[2], C->cand_h[3], C->cand_h[4], C->cand_h[5], C->cand_h[6], C->cand_h[7],
            C->cand_h[8], C->cand_h[9], C->cand_h[10], C->cand_h[11], C->cand_h[12], C->cand_h[13], C->cand_h[14], C->cand_h[15]);
    }
    CK(hipMemcpyAsync(C->cand_ids, C->cand_h, (size_t)7*16*4, hipMemcpyHostToDevice, s));
    CK(hipMemcpyAsync(C->cand_val, C->candv_h, (size_t)7*16*4, hipMemcpyHostToDevice, s));
    if (g == 0) {
        if (q27_env_flag("Q27_DFLASH2_SEL_DBG", false)) { CK(hipStreamSynchronize(s));
            unsigned short gb[8]; float l0[4];
            CK(hipMemcpy(gb, C->g_bf16, 16, hipMemcpyDeviceToHost)); CK(hipMemcpy(l0, C->logits, 16, hipMemcpyDeviceToHost));
            std::printf("Q27_DFLASH2_GDGB g=%04x %04x %04x %04x %04x %04x %04x %04x logits0=%.2f %.2f %.2f %.2f\n",
                        gb[0],gb[1],gb[2],gb[3],gb[4],gb[5],gb[6],gb[7], l0[0],l0[1],l0[2],l0[3]); }
        // selector hidden: sel_hidden_proj over the final norm rows 1..7 (natural-order int8)
        q27_df2_quant_bf16(C->norm + 5120, 5120, C->n_xq, 5120, C->n_xs, 320, 7, 5120, s);
        if (!q27_df2_gemm(&W.sel_hidden_proj, C->n_xq, 5120, C->n_xs, (void*)C->sh, 256, nullptr, 0, 0.f, 7, -1, s))
            return fail(err,errcap,1,"df2 sel_hidden declined");
        hipLaunchKernelGGL(q27_k_df2_selector, dim3(1), dim3(512), 0, s, C->cand_ids, C->cand_val,
                           C->sh, (const unsigned short*)W.pred_codebook.w, (const unsigned short*)W.succ_codebook.w,
                           anchor, C->path, 7);
        CK(hipMemcpyAsync(C->cand_h, C->path, (size_t)7*4, hipMemcpyDeviceToHost, s));
        CK(hipStreamSynchronize(s));
        for (int j = 0; j < 7; ++j) out7[j] = C->cand_h[j];
    }
    // broadcast the 7 ids to every card (the round's mailbox pattern)
    exch(uctx, g, out7, nullptr, 7);
    return 0;
}

// Prime every cond_inject GEMM row count. cond_inject's rows = nacc+1 takes 8 distinct values at K=7,
// and each new shape pays a one-time rocBLAS/Tensile solution selection (measured: cond spiking to
// 10.5 and 46.3 ms on first use of a count, then <1 ms). Same discipline the prefill lane already uses.
int q27_df2_prime(q27_df2_ctx_t* C, hipStream_t s) {
    q27_df2_t& W = C->W;
    for (int r = 1; r <= 8; ++r) {
        q27_df2_gemm(&W.fc, C->tap_xq, 25600, C->tap_xs, (void*)C->g_full, 5120, nullptr, 0, 0.f, r, 0, s);
        for (int l = 0; l < 5; ++l) {
            q27_df2_gemm(&W.L[l].k, C->g_xq, 5120, C->g_xs, (void*)C->kvbf, 2048, nullptr, 0, 0.f, r, -1, s);
            q27_df2_gemm(&W.L[l].v, C->g_xq, 5120, C->g_xs, (void*)(C->kvbf + 1024), 2048, nullptr, 0, 0.f, r, -1, s);
        }
    }
    return (int)hipStreamSynchronize(s);
}
int q27_df2_cond_inject(q27_df2_ctx_t* C, const unsigned short* taps, int tap_stride, int nacc, int pos, int g,
                        q27_df2_reduce_fn reduce, void* uctx, hipStream_t s, char* err, size_t errcap) {
    if (nacc < 1) return 0;
    q27_df2_t& W = C->W;
    // 1. tap concat + quantize: taps[t][stride][5120] -> xq[r][25600] / xs[r][1600]
    for (int t = 0; t < 5; ++t)
        q27_df2_quant_bf16(taps + (size_t)t * tap_stride * 5120, 5120, C->tap_xq + (size_t)t * 5120, 25600,
                           C->tap_xs + (size_t)t * 320, 1600, nacc, 5120, s);
    // 2. fc FULL (replicated): [5120, 25600] -> g_full [nacc][5120] fp32, identical on every card
    (void)reduce; (void)uctx;
    if (!q27_df2_gemm(&W.fc, C->tap_xq, 25600, C->tap_xs, (void*)C->g_full, 5120, nullptr, 0, 0.f, nacc, 0, s))
        return fail(err,errcap,1,"df2 fc declined");
    // 4. fp32 -> bf16, then hidden_norm (bf16 in/out), then requantize for the k/v projections
    q27_df2_f2bf(C->g_full, C->g_bf16, nacc * 5120, s);
    hipLaunchKernelGGL(q27_k_df2_rmsnorm_rows, dim3(nacc), dim3(256), 0, s, C->g_bf16, (const unsigned short*)W.hidden_norm, C->g_bf16, 5120);
    q27_df2_quant_bf16(C->g_bf16, 5120, C->g_xq, 5120, C->g_xs, 320, nacc, 5120, s);
    if (q27_env_flag("Q27_DFLASH2_COND_DBG", false)) {   // does the committed row's conditioned feature actually change round to round?
        CK(hipStreamSynchronize(s));
        unsigned short h[16]; float ts[5] = {0,0,0,0,0};
        CK(hipMemcpy(h, C->g_bf16, 32, hipMemcpyDeviceToHost));
        float cs = 0.f; for (int i = 0; i < 16; ++i) { unsigned u = (unsigned)h[i] << 16; float f; std::memcpy(&f, &u, 4); cs += fabsf(f); }
        for (int t = 0; t < 5; ++t) { unsigned short th[16]; CK(hipMemcpy(th, taps + (size_t)t * tap_stride * 5120, 32, hipMemcpyDeviceToHost));
            for (int i = 0; i < 16; ++i) { unsigned u = (unsigned)th[i] << 16; float f; std::memcpy(&f, &u, 4); ts[t] += fabsf(f); } }
        std::printf("Q27_DFLASH2_CONDDBG pos=%d nrows=%d g_csum=%.4f tapcsum=%.3f %.3f %.3f %.3f %.3f\n", pos, nacc, cs, ts[0], ts[1], ts[2], ts[3], ts[4]);
    }
    // 5. per layer: k/v proj (full width) + kvprep injection at pos..pos+nacc-1
    for (int l = 0; l < 5; ++l) {
        if (!q27_df2_gemm(&W.L[l].k, C->g_xq, 5120, C->g_xs, (void*)C->kvbf, 2048, nullptr, 0, 0.f, nacc, -1, s))
            return fail(err,errcap,1,"df2 k_proj declined (layer %d)", l);
        if (!q27_df2_gemm(&W.L[l].v, C->g_xq, 5120, C->g_xs, (void*)(C->kvbf + 1024), 2048, nullptr, 0, 0.f, nacc, -1, s))
            return fail(err,errcap,1,"df2 v_proj declined (layer %d)", l);
        hipLaunchKernelGGL(q27_k_df2_kvinj, dim3(8 * nacc), dim3(128), 0, s, C->kvbf, C->kvbf + 1024,
                             (const unsigned short*)W.L[l].k_norm, C->kc[l], C->ks[l], C->vc[l], C->vs[l], pos, nacc, 2048);
    }
    if (C->ctx_end == 0) C->ctx_begin = pos;
    C->ctx_end = pos + nacc;
    if (C->ctx_end - C->ctx_begin > 2048) C->ctx_begin = C->ctx_end - 2048;
    return 0;
}
