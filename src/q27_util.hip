// Two small helpers the layer loop needs that are not part of any agent's file.
//   q27_bf16_gemv  - in_proj_a / in_proj_b: [48, 5120] BF16 against a BF16 activation.
//                    Tiny (0.5 MB per layer, 0.3% of decode bytes) so correctness beats cleverness:
//                    one wave64 per output row, fp32 accumulate, one shuffle reduction.
//   q27_f2bf_vec   - fp32 -> BF16 round-to-nearest-even, for projection outputs the mixer consumes
//                    as BF16.
#include "q27.h"

__global__ __launch_bounds__(256) void k_bf16_gemv(const unsigned short* __restrict__ w,
                                                   const unsigned short* __restrict__ x,
                                                   unsigned short* __restrict__ y,
                                                   int rows, int K) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const unsigned short* wr = w + (size_t)row * K;
    float acc = 0.f;
    for (int k = lane; k < K; k += 64) acc = fmaf(q27_bf2f(wr[k]), q27_bf2f(x[k]), acc);
    #pragma unroll
    for (int o = 32; o; o >>= 1) acc += __shfl_down(acc, o, 64);
    if (lane == 0) y[row] = q27_f2bf(acc);
}

__global__ __launch_bounds__(256) void k_f2bf(const float* __restrict__ src,
                                              unsigned short* __restrict__ dst, int n) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        dst[i] = q27_f2bf(src[i]);
}

extern "C" void q27_bf16_gemv(const unsigned short* w, const unsigned short* x, unsigned short* y,
                              int rows, int K, hipStream_t s) {
    hipLaunchKernelGGL(k_bf16_gemv, dim3((rows + 3) / 4), dim3(256), 0, s, w, x, y, rows, K);
}
extern "C" void q27_f2bf_vec(const float* src, unsigned short* dst, int n, hipStream_t s) {
    int blocks = (n + 255) / 256; if (blocks > 480) blocks = 480;
    hipLaunchKernelGGL(k_f2bf, dim3(blocks), dim3(256), 0, s, src, dst, n);
}

// =============================================================================================
// FUSED, VECTORISED in_proj_a/in_proj_b GEMV.
//
// The kernel this replaces was 15.7% of decode compute -- 3.783 ms/token over 96 calls, 39.4 us
// each, for two 12x5120 bf16 dot products whose weights are 123 KiB. That is ~30x off any byte
// floor, and the cause is not arithmetic: one wave per row, ONE 2-byte scalar load per lane per
// iteration, K/64 = 80 iterations, each dependent on the last. 80 serialised round trips at ~0.5 us
// is exactly the 40 us observed. Its own comment said "tiny, so correctness beats cleverness",
// which was true when it was 0.3% of decode bytes.
//
// THREE CHANGES, all mechanical:
//   1. uint2 loads -- 4 bf16 per lane per step, so a wave issues 512 B per transaction instead of
//      128. 512 B is the measured-best streaming pattern on these cards.
//   2. ALL loads issued before ANY wait. Each lane holds its whole slice in registers, so the
//      dependency graph is one wait instead of eighty.
//   3. a and b FUSED into one launch. They read the SAME activation, so the fused form halves the
//      launch count (96 -> 48/token) and reads x once instead of twice.
//
// Reduction order differs from the scalar version and the result is NOT bit-identical. That is
// allowed and intended: this is a native gfx906 engine, not a port. fp32 accumulation throughout,
// bf16 out, gated on model output quality rather than float identity.
#define Q27_GEMV2_NW 4                       // waves per block; block = 256 threads
__global__ __launch_bounds__(256) void k_bf16_gemv2(
        const unsigned short* __restrict__ wa, const unsigned short* __restrict__ wb,
        const unsigned short* __restrict__ x,
        unsigned short* __restrict__ ya, unsigned short* __restrict__ yb, int rows, int K)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int wv   = threadIdx.x >> 6;

    const int KV   = K >> 2;                       // uint2 along K (4 bf16 each)
    const int per  = KV / Q27_GEMV2_NW;            // uint2 this wave covers
    const int beg  = wv * per;
    const uint2* __restrict__ WA = (const uint2*)(wa + (size_t)row * K);
    const uint2* __restrict__ WB = (const uint2*)(wb + (size_t)row * K);
    const uint2* __restrict__ X  = (const uint2*)x;

    float aa = 0.f, bb = 0.f;
    for (int i = beg + lane; i < beg + per; i += 64) {
        const uint2 xv = X[i], av = WA[i], bv = WB[i];   // three independent loads, one wait
        const unsigned xp[2] = { xv.x, xv.y };
        const unsigned ap[2] = { av.x, av.y };
        const unsigned bp[2] = { bv.x, bv.y };
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float x0 = q27_bf2f((unsigned short)(xp[h] & 0xFFFFu));
            const float x1 = q27_bf2f((unsigned short)(xp[h] >> 16));
            aa = fmaf(q27_bf2f((unsigned short)(ap[h] & 0xFFFFu)), x0, aa);
            aa = fmaf(q27_bf2f((unsigned short)(ap[h] >> 16)),     x1, aa);
            bb = fmaf(q27_bf2f((unsigned short)(bp[h] & 0xFFFFu)), x0, bb);
            bb = fmaf(q27_bf2f((unsigned short)(bp[h] >> 16)),     x1, bb);
        }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) { aa += __shfl_down(aa, o, 64); bb += __shfl_down(bb, o, 64); }

    __shared__ float sa[Q27_GEMV2_NW], sb[Q27_GEMV2_NW];
    if (lane == 0) { sa[wv] = aa; sb[wv] = bb; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.f, tb = 0.f;
        #pragma unroll
        for (int i = 0; i < Q27_GEMV2_NW; ++i) { ta += sa[i]; tb += sb[i]; }
        ya[row] = q27_f2bf(ta);
        yb[row] = q27_f2bf(tb);
    }
}

// Returns 1 if the fused path ran. Falls back for shapes it does not cover, so an uncovered
// geometry degrades to the old kernel rather than producing wrong bytes.
// tile form (Q27_PF_GDN_SCAN): grid (rows, positions); every block is k_bf16_gemv2 for one row
// and one position (same wave split of K, same lane stride, same ladder and wave-order sum), so
// each output is bit-identical to the per-position launch; one launch per tile instead of Ct.
__global__ __launch_bounds__(256) void k_bf16_gemv2_tile(
        const unsigned short* __restrict__ wa, const unsigned short* __restrict__ wb,
        const unsigned short* __restrict__ xt, int xstride,
        unsigned short* __restrict__ ya, unsigned short* __restrict__ yb, int rows, int K)
{
    const int row = blockIdx.x, t = blockIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int wv   = threadIdx.x >> 6;
    const int KV   = K >> 2;
    const int per  = KV / Q27_GEMV2_NW;
    const int beg  = wv * per;
    const uint2* __restrict__ WA = (const uint2*)(wa + (size_t)row * K);
    const uint2* __restrict__ WB = (const uint2*)(wb + (size_t)row * K);
    const uint2* __restrict__ X  = (const uint2*)(xt + (size_t)t * xstride);
    float aa = 0.f, bb = 0.f;
    for (int i = beg + lane; i < beg + per; i += 64) {
        const uint2 xv = X[i], av = WA[i], bv = WB[i];
        const unsigned xp[2] = { xv.x, xv.y };
        const unsigned ap[2] = { av.x, av.y };
        const unsigned bp[2] = { bv.x, bv.y };
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float x0 = q27_bf2f((unsigned short)(xp[h] & 0xFFFFu));
            const float x1 = q27_bf2f((unsigned short)(xp[h] >> 16));
            aa = fmaf(q27_bf2f((unsigned short)(ap[h] & 0xFFFFu)), x0, aa);
            aa = fmaf(q27_bf2f((unsigned short)(ap[h] >> 16)),     x1, aa);
            bb = fmaf(q27_bf2f((unsigned short)(bp[h] & 0xFFFFu)), x0, bb);
            bb = fmaf(q27_bf2f((unsigned short)(bp[h] >> 16)),     x1, bb);
        }
    }
    #pragma unroll
    for (int o = 32; o; o >>= 1) { aa += __shfl_down(aa, o, 64); bb += __shfl_down(bb, o, 64); }
    __shared__ float sa[Q27_GEMV2_NW], sb[Q27_GEMV2_NW];
    if (lane == 0) { sa[wv] = aa; sb[wv] = bb; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.f, tb = 0.f;
        #pragma unroll
        for (int i = 0; i < Q27_GEMV2_NW; ++i) { ta += sa[i]; tb += sb[i]; }
        ya[(size_t)t * rows + row] = q27_f2bf(ta);
        yb[(size_t)t * rows + row] = q27_f2bf(tb);
    }
}

extern "C" int q27_bf16_gemv2_tile(const unsigned short* wa, const unsigned short* wb,
                                   const unsigned short* xt, int xstride,
                                   unsigned short* ya, unsigned short* yb, int rows, int K, int Ct,
                                   hipStream_t s)
{
    if (!wa || !wb || rows <= 0 || Ct <= 0 || (K & 15)) return 0;
    hipLaunchKernelGGL(k_bf16_gemv2_tile, dim3(rows, Ct), dim3(256), 0, s, wa, wb, xt, xstride, ya, yb, rows, K);
    return 1;
}

extern "C" int q27_bf16_gemv2(const unsigned short* wa, const unsigned short* wb,
                              const unsigned short* x, unsigned short* ya, unsigned short* yb,
                              int rows, int K, hipStream_t s)
{
    if (rows <= 0 || K <= 0 || (K & 7)) return 0;                 // need K/4 uint2, split NW ways
    if ((K >> 2) % Q27_GEMV2_NW) return 0;
    const uintptr_t mix = (uintptr_t)wa | (uintptr_t)wb | (uintptr_t)x;
    if (mix & 7u) return 0;                                        // uint2 needs 8-B alignment
    hipLaunchKernelGGL(k_bf16_gemv2, dim3(rows), dim3(256), 0, s, wa, wb, x, ya, yb, rows, K);
    return 1;
}

// ---------------------------------------------------------------------------------------------
// IN-STREAM STAGING COPY (Q27_COLL_KCOPY). The collective's outbound 20 KB partial left the card by
// hipMemcpyAsync D2H, i.e. the SDMA engine on a separate hardware queue. On the GPU timeline that
// cost ~25 us per collective (rocprof trace, 09-10: a 24-27 us gap between the producer's end and
// the flag-wait kernel's start, 127 times per token) although the copy itself is ~8 us: two
// cross-queue dependency handoffs bracket it. A kernel on the compute queue writing coalesced
// float4 into the pinned staging buffer (through its device alias) has no cross-queue handoff and
// the flag-wait follows it directly. The rejected zero-copy-out was a different mechanism -- 5120
// scattered 4-byte stores from the producer's epilogue.
__global__ __launch_bounds__(256) void k_copy_f4(const float4* __restrict__ src,
                                                float4* __restrict__ dst, int n4) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n4) dst[i] = src[i];
}
extern "C" void q27_copy_f4(float* dst_dev, const float* src, int n, hipStream_t s) {
    const int n4 = n >> 2;                                    // n is 5120: 1280 float4, 5 blocks
    hipLaunchKernelGGL(k_copy_f4, dim3((n4 + 255) / 256), dim3(256), 0, s,
                       (const float4*)src, (float4*)dst_dev, n4);
}

// ---------------------------------------------------------------------------------------------
// STAGING COPY + DONE FLAG (Q27_COLL_DFLAG). Same copy, one block, and after a system-scope fence
// thread 0 writes the staging generation into pinned host memory. The host spins on that word
// instead of hipEventSynchronize, and no event marker is recorded behind the copy: that removes one
// GPU-side marker (system release + signal) and the host's wake-up latency per collective.
__global__ __launch_bounds__(256) void k_copy_f4_flag(const float4* __restrict__ src,
                                                     float4* __restrict__ dst, int n4,
                                                     unsigned* __restrict__ done, unsigned gen) {
    for (int i = threadIdx.x; i < n4; i += 256) dst[i] = src[i];
    __threadfence_system();                      // this thread's stores visible system-wide
    __syncthreads();                             // every thread has fenced
    if (threadIdx.x == 0) { __threadfence_system(); *((volatile unsigned*)done) = gen; }
}
extern "C" void q27_copy_f4_flag(float* dst_dev, const float* src, int n, unsigned* done_dev,
                                 unsigned gen, hipStream_t s) {
    hipLaunchKernelGGL(k_copy_f4_flag, dim3(1), dim3(256), 0, s,
                       (const float4*)src, (float4*)dst_dev, n >> 2, done_dev, gen);
}

// Q27_LS_Q8: the GDN a/b projection straight from the input norm's int8 activations. The norm
// emits the NVFP4 consumers' order (even-then-odd inside each group of 16: bytes 0-7 hold
// elements 0,2,..,14 and bytes 8-15 hold 1,3,..,15) with one fp32 scale per group; xmul = the
// norm's in_scale so q*s*xmul is the normalized value. One lane owns whole 16-element groups:
// one 16-byte activation load + two 32-byte bf16 weight loads per matrix per group, fp32 fmaf
// chains, wave ladder + 4-wave sum as k_bf16_gemv2_tile. fp32 outputs [t][rows] (48 values,
// the pre-kernel's accumulation representation -- nothing to gain from quantizing 96 floats).
__global__ __launch_bounds__(256) void k_q8_gemv2_tile(
        const unsigned short* __restrict__ wa, const unsigned short* __restrict__ wb,
        const signed char* __restrict__ xq, const float* __restrict__ xs, int xstride, float xmul, int perm, int xs64,
        float* __restrict__ ya, float* __restrict__ yb, int rows, int K)
{
    const int row = blockIdx.x, t = blockIdx.y;
    if (row >= rows) return;
    const int lane = threadIdx.x & 63;
    const int wv   = threadIdx.x >> 6;
    const int NG   = K >> 4;                        // 16-element groups along K
    const int per  = NG / Q27_GEMV2_NW;
    const int beg  = wv * per;
    const uint4* __restrict__ WA = (const uint4*)(wa + (size_t)row * K);   // 8 bf16 per uint4
    const uint4* __restrict__ WB = (const uint4*)(wb + (size_t)row * K);
    const uint4* __restrict__ X  = (const uint4*)(xq + (size_t)t * xstride);   // 16 int8 per uint4
    const float* __restrict__ XS = xs + (size_t)t * (xstride >> 4);
    float aa = 0.f, bb = 0.f;
    for (int gi = beg + lane; gi < beg + per; gi += 64) {
        const uint4 xv = X[gi];
        const uint4 a0 = WA[2 * gi], a1 = WA[2 * gi + 1];
        const uint4 b0 = WB[2 * gi], b1 = WB[2 * gi + 1];
        const float sc = XS[xs64 ? (gi >> 2) : gi] * xmul;
        const unsigned xw[4] = { xv.x, xv.y, xv.z, xv.w };       // bytes 0-15 of the group
        const unsigned aw[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };   // elements 0-15, natural
        const unsigned bw[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const int e = perm ? ((b < 8) ? (2 * b) : (2 * (b - 8) + 1)) : b;   // element index of byte b
            const float x = (float)((int)(xw[b >> 2] << (24 - 8 * (b & 3))) >> 24) * sc;
            const unsigned short wea = (unsigned short)((e & 1) ? (aw[e >> 1] >> 16) : (aw[e >> 1] & 0xFFFFu));
            const unsigned short web = (unsigned short)((e & 1) ? (bw[e >> 1] >> 16) : (bw[e >> 1] & 0xFFFFu));
            aa = fmaf(q27_bf2f(wea), x, aa);
            bb = fmaf(q27_bf2f(web), x, bb);
        }
    }
#pragma unroll
    for (int o = 32; o; o >>= 1) { aa += __shfl_down(aa, o, 64); bb += __shfl_down(bb, o, 64); }
    __shared__ float sa[Q27_GEMV2_NW], sb[Q27_GEMV2_NW];
    if (lane == 0) { sa[wv] = aa; sb[wv] = bb; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.f, tb = 0.f;
#pragma unroll
        for (int i = 0; i < Q27_GEMV2_NW; ++i) { ta += sa[i]; tb += sb[i]; }
        ya[(size_t)t * rows + row] = ta;
        yb[(size_t)t * rows + row] = tb;
    }
}

extern "C" int q27_q8_gemv2_tile(const unsigned short* wa, const unsigned short* wb,
                                 const signed char* xq, const float* xs, int xstride, float xmul, int perm, int xs64,
                                 float* ya, float* yb, int rows, int K, int Ct, hipStream_t s)
{
    if (!wa || !wb || !xq || !xs || rows <= 0 || Ct <= 0 || (K & 63) || (xstride & 15)) return 0;
    hipLaunchKernelGGL(k_q8_gemv2_tile, dim3(rows, Ct), dim3(256), 0, s, wa, wb, xq, xs, xstride, xmul, perm, xs64, ya, yb, rows, K);
    return 1;
}

// ---------------------------------------------------------------------------------------------
// MULTI-BLOCK STAGING COPY + DONE FLAG (Q27_SPEC / Q27_COPY_MB). Same contract as
// q27_copy_f4_flag -- the host spins on `done` reaching `gen` -- but the copy is spread over
// nblk blocks: the single-block form moved the two-row 40 KB payload in ~9 us per call, PCIe
// write-latency bound with one block in flight. Every block fences its stores system-wide and
// bumps a device counter; the block that brings the counter to `target` (= nblk x calls so far
// on that counter) publishes the generation after a second system fence (threadFenceReduction).
__global__ __launch_bounds__(256) void k_copy_f4_flag_mb(const float4* __restrict__ src,
                                                        float4* __restrict__ dst, int n4,
                                                        unsigned* __restrict__ done, unsigned gen,
                                                        unsigned* __restrict__ cnt, unsigned target) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n4; i += gridDim.x * 256) dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(cnt, 1u);
        if (old + 1u == target) { __threadfence_system(); *((volatile unsigned*)done) = gen; }
    }
}
extern "C" void q27_copy_f4_flag_mb(float* dst_dev, const float* src, int n, unsigned* done_dev,
                                    unsigned gen, unsigned* cnt_dev, unsigned target, int nblk,
                                    hipStream_t s) {
    hipLaunchKernelGGL(k_copy_f4_flag_mb, dim3(nblk), dim3(256), 0, s,
                       (const float4*)src, (float4*)dst_dev, n >> 2, done_dev, gen, cnt_dev, target);
}
// Same flag protocol as k_copy_f4_flag_mb, but the payload is packed to bf16 on the way out (the
// NR-row collective payload is NR x 5120 fp32 = 160 KB per layer per card at NR=8; bf16 halves the
// D2H staging and the host reduce reads). n4 = number of float4 = number of 8-byte bf16x4 outputs.
__global__ __launch_bounds__(256) void k_copy_bf16_flag_mb(const float4* __restrict__ src,
                                                          uint2* __restrict__ dst, int n4,
                                                          unsigned* __restrict__ done, unsigned gen,
                                                          unsigned* __restrict__ cnt, unsigned target) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n4; i += gridDim.x * 256) {
        const float4 v = src[i];
        uint2 o;
        o.x = (unsigned)q27_f2bf(v.x) | ((unsigned)q27_f2bf(v.y) << 16);
        o.y = (unsigned)q27_f2bf(v.z) | ((unsigned)q27_f2bf(v.w) << 16);
        dst[i] = o;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = atomicAdd(cnt, 1u);
        if (old + 1u == target) { __threadfence_system(); *((volatile unsigned*)done) = gen; }
    }
}
extern "C" void q27_copy_bf16_flag_mb(unsigned short* dst_dev, const float* src, int n, unsigned* done_dev,
                                      unsigned gen, unsigned* cnt_dev, unsigned target, int nblk,
                                      hipStream_t s) {
    hipLaunchKernelGGL(k_copy_bf16_flag_mb, dim3(nblk), dim3(256), 0, s,
                       (const float4*)src, (uint2*)dst_dev, n >> 2, done_dev, gen, cnt_dev, target);
}
