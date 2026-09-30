// q27_exl3_fwd.hip — fused EXL3 decode + GEMM/GEMV and the EXL3 linear for gfx906.
//
// The weight bytes read by these kernels ARE the checkpoint bytes: the trellis
// is decoded IN LDS, used, and thrown away. No fp16 weight matrix is ever
// materialized, at load time or during execution.
//
// Both kernels are built on reconstruct_kernel's structure (stage 8 cells ->
// dq_dispatch<K,cb> -> frags_to_tile -> row-major 16x128 fp16 tile), which is
// K-generic and verified bit-exact against this store. Only the consumer of
// the tile differs: instead of storing it, we dot it. Reusing the staging and
// the element mapping verbatim is deliberate -- re-deriving that mapping is
// exactly how a wrong answer becomes plausible garbage, and it already cost a
// debugging round on the codebook index (mul1 selects cb 2, not cb 0).
//
// TWO CONSUMERS, ONE DECODE:
//   k_gemv_m1   M == 1 (decode). All 256 threads: 64 column-pairs x 4 k-groups,
//               each group summing 4 of the tile's 16 rows, then an LDS reduce.
//   k_gemm_mb   M > 1 (prefill). The tile is decoded ONCE per block and dotted
//               against MB rows at a time -- the reason a 256-row prefill costs
//               ONE weight sweep rather than 256.
// Both read the tile as __half2 (consecutive threads -> consecutive 4B words,
// no LDS bank conflict) rather than as scalar halves.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "exl3_dequant_gfx906.h"
#include "q27_exl3.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <atomic>

namespace q27exl3 {

using exl3gfx::FragB;
using exl3gfx::dq_dispatch;
using exl3gfx::frags_to_tile;

// ---------------------------------------------------------------------------
// M == 1. 64 column-pairs (t & 63) x 4 k-groups (t >> 6); k-group r covers tile
// rows 4r..4r+3. The 4 partials per column are reduced through LDS.
template <int K, int cb>
__global__ __launch_bounds__(256)
void k_gemv_m1(float* __restrict__ y,               // [out] fp32, pre-zeroed
               const __half* __restrict__ x,        // [in] fp16, already suh-scaled + rotated
               const uint16_t* __restrict__ g_packed,
               int packed_blocks_n)
{
    constexpr int packed_size = 256 * K / 16;
    const int t = threadIdx.x, lane_id = t % 32, warp_id = t / 32;
    const int k = blockIdx.y;
    const int n = blockIdx.x * 8;

    // LDS STAGE PADDING (their EXL3_HIP_STG_PAD, worth 31.7 -> 34.9 tok/s on
    // gfx1151). LDS is 32 banks x 1 dword, bank = dword_addr % 32. An unpadded
    // cell row is exactly packed_size/2 dwords -- 32 at K=4 -- so warp-group
    // w+1's base sits an exact multiple of 32 dwords after w's and lands on the
    // SAME banks. A wave64 spans two warp-groups, so every wave conflicted
    // 2-way on every decode read. One spare dword per row shifts each group by
    // one bank and the conflict disappears.
    __shared__ uint32_t s_packed[8][packed_size / 2 + 1];
    g_packed += (size_t)(k * packed_blocks_n + n) * packed_size;
    // ROW-AWARE STAGE WRITE. s_packed is [8][packed_size/2 + 1] -- PADDED -- and is READ as
    // s_packed[warp_id], i.e. at stride packed_size/2 + 1. A flat ((int4*)s_packed)[t] write
    // lays the rows out at stride packed_size/2, so cell w was written at w*32 dwords and read
    // at w*33 (K=4). Row 0 correct, rows 1-7 each shifted by w. That is the entire band-kernel
    // defect: -2.58 dB vs the verified M=1 path at every M. The M=1 kernels already wrote
    // row-aware, which is why only the batched path was wrong.
    if (t < packed_size) {
        constexpr int i4pc = packed_size / 8;
        *(int4*)&s_packed[t / i4pc][(t % i4pc) * 4] = ((const int4*)g_packed)[t];
    }
    __syncthreads();

    exl3gfx::FragB frag[2];
    exl3gfx::dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    __shared__ __half2 tile[16][8][8];              // == row-major [16][128] fp16
    exl3gfx::frags_to_tile(frag, lane_id, warp_id, tile);

    __shared__ float xs[16];
    if (t < 16) xs[t] = __half2float(x[k * 16 + t]);
    __syncthreads();

    const int cp = t & 63;                          // column pair -> columns 2cp, 2cp+1
    const int rg = t >> 6;                          // k group 0..3 -> rows 4rg..4rg+3
    const __half2* W2 = (const __half2*)tile;

    float a0 = 0.f, a1 = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = rg * 4 + i;
        const float2 w = __half22float2(W2[r * 64 + cp]);
        const float xr = xs[r];
        a0 = fmaf(xr, w.x, a0);
        a1 = fmaf(xr, w.y, a1);
    }

    __shared__ float red[4][128];
    red[rg][cp * 2]     = a0;
    red[rg][cp * 2 + 1] = a1;
    __syncthreads();
    if (t < 128) {
        const float s = red[0][t] + red[1][t] + red[2][t] + red[3][t];
        atomicAdd(&y[(size_t)n * 16 + t], s);
    }
}

// ---------------------------------------------------------------------------
// M > 1. Same decode, then MB rows per pass against the resident tile.
// 64 column-pairs x 4 row-groups: row-group g takes rows g, g+4, g+8, ... of
// the MB-row band, so all 256 threads carry dot work for MB >= 4.
template <int K, int cb, int MB>
__global__ __launch_bounds__(256)
void k_gemm_mb(float* __restrict__ y,               // [M][out] fp32, pre-zeroed
               const __half* __restrict__ x,        // [M][in] fp16
               const uint16_t* __restrict__ g_packed,
               int packed_blocks_n,
               int M, int out_features, int in_features)
{
    constexpr int packed_size = 256 * K / 16;
    const int t = threadIdx.x, lane_id = t % 32, warp_id = t / 32;
    const int k = blockIdx.y;
    const int n = blockIdx.x * 8;

    // LDS STAGE PADDING (their EXL3_HIP_STG_PAD, worth 31.7 -> 34.9 tok/s on
    // gfx1151). LDS is 32 banks x 1 dword, bank = dword_addr % 32. An unpadded
    // cell row is exactly packed_size/2 dwords -- 32 at K=4 -- so warp-group
    // w+1's base sits an exact multiple of 32 dwords after w's and lands on the
    // SAME banks. A wave64 spans two warp-groups, so every wave conflicted
    // 2-way on every decode read. One spare dword per row shifts each group by
    // one bank and the conflict disappears.
    __shared__ uint32_t s_packed[8][packed_size / 2 + 1];
    g_packed += (size_t)(k * packed_blocks_n + n) * packed_size;
    // ROW-AWARE STAGE WRITE. s_packed is [8][packed_size/2 + 1] -- PADDED -- and is READ as
    // s_packed[warp_id], i.e. at stride packed_size/2 + 1. A flat ((int4*)s_packed)[t] write
    // lays the rows out at stride packed_size/2, so cell w was written at w*32 dwords and read
    // at w*33 (K=4). Row 0 correct, rows 1-7 each shifted by w. That is the entire band-kernel
    // defect: -2.58 dB vs the verified M=1 path at every M. The M=1 kernels already wrote
    // row-aware, which is why only the batched path was wrong.
    if (t < packed_size) {
        constexpr int i4pc = packed_size / 8;
        *(int4*)&s_packed[t / i4pc][(t % i4pc) * 4] = ((const int4*)g_packed)[t];
    }
    __syncthreads();

    exl3gfx::FragB frag[2];
    exl3gfx::dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    __shared__ __half2 tile[16][8][8];
    exl3gfx::frags_to_tile(frag, lane_id, warp_id, tile);
    __syncthreads();

    const int cp = t & 63;
    const int rg = t >> 6;
    const __half2* W2 = (const __half2*)tile;
    const size_t ycol = (size_t)n * 16 + (size_t)cp * 2;

    __shared__ float xs[MB][16];
    for (int m0 = 0; m0 < M; m0 += MB) {
        const int mb = (M - m0 < MB) ? (M - m0) : MB;
        __syncthreads();                             // tile stays valid; xs is being rewritten
        for (int i = t; i < mb * 16; i += 256) {
            const int mm = i >> 4, rr = i & 15;
            xs[mm][rr] = __half2float(x[(size_t)(m0 + mm) * in_features + k * 16 + rr]);
        }
        __syncthreads();

        for (int mm = rg; mm < mb; mm += 4) {
            float a0 = 0.f, a1 = 0.f;
            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                const float2 w = __half22float2(W2[r * 64 + cp]);
                const float xr = xs[mm][r];
                a0 = fmaf(xr, w.x, a0);
                a1 = fmaf(xr, w.y, a1);
            }
            float* yr = y + (size_t)(m0 + mm) * out_features + ycol;
            atomicAdd(yr,     a0);
            atomicAdd(yr + 1, a1);
        }
    }
}


// ---------------------------------------------------------------------------
// BAND KERNEL -- one structure for decode (M=1) and prefill (M large).
//
// WHY IT REPLACED THE FIRST VERSION. v1 gave each block one 16x128 tile and
// atomicAdd'd its partial into y. Weights were read once, which looked right,
// but the atomic traffic is M*128 per block x (out/128)*(in/16) blocks: for
// gate_proj at M=256 that is 1.4e9 float atomics, and they, not the weights,
// set the time. Here each block instead OWNS a 128-column x MB-row patch of
// the output and walks a K RANGE, carrying the accumulator in registers; the
// only global writes are at the end. Split-k across blockIdx.z restores
// parallelism when M is small, at G atomics per output element instead of
// in/16 of them.
//
//   grid = (out/128, ceil(M/MB), G)
//   G == 1 -> plain stores, zero atomics.
//
// The dot is v_dot2_f32_f16 (two fp16 MACs per lane per instruction, fp32
// accumulate) rather than scalar fp32 FMA -- on gfx906 that is the difference
// between ~13.4 and ~26.8 TFLOPS, and prefill here is compute-bound, not
// weight-bound. dot2 needs its two fp16 lanes to lie along the REDUCTION axis,
// so the decoded tile is transposed in LDS to [col][r] first: ~8 LDS ops per
// thread per k-block against 128 dot2 of work, i.e. a few percent, and it buys
// the 2x. The transpose target is padded to 18 halves per column so the half2
// reads land on distinct banks ((c*9) mod 32 is a bijection).
static __device__ __forceinline__ float dot2acc(__half2 a, __half2 b, float c)
{
#if defined(__gfx906__) || defined(__gfx90a__) || defined(__gfx908__)
    typedef _Float16 h2v __attribute__((ext_vector_type(2)));
    h2v av, bv;
    __builtin_memcpy(&av, &a, 4);
    __builtin_memcpy(&bv, &b, 4);
    return __builtin_amdgcn_fdot2(av, bv, c, false);
#else
    // Correct fallback, never silently different in meaning -- only in speed.
    const float2 fa = __half22float2(a), fb = __half22float2(b);
    return fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, c));
#endif
}

template <int K, int cb, int MB>
__global__ __launch_bounds__(256)
void k_gemm_band(float* __restrict__ y,            // [M][out_f]
                 const __half* __restrict__ x,     // [M][in_f]
                 const uint16_t* __restrict__ g_packed,
                 int packed_blocks_n,
                 int M, int out_f, int in_f, int G, int ldx, int ldy)
{
    constexpr int packed_size = 256 * K / 16;
    constexpr int RH = (MB + 1) / 2;                 // rows per thread-half
    const int t = threadIdx.x, lane_id = t % 32, warp_id = t / 32;
    const int n  = blockIdx.x * 8;                   // first cell -> 128 columns
    const int m0 = blockIdx.y * MB;
    const int mb = (M - m0 < MB) ? (M - m0) : MB;
    if (mb <= 0) return;
    const int c = t & 127;                           // this thread's column
    const int h = t >> 7;                            // row half

    const int nk   = in_f / 16;
    const int kper = (nk + G - 1) / G;
    const int kbeg = blockIdx.z * kper;
    const int kend = (kbeg + kper < nk) ? (kbeg + kper) : nk;
    if (kbeg >= kend) return;

    // LDS STAGE PADDING (their EXL3_HIP_STG_PAD, worth 31.7 -> 34.9 tok/s on
    // gfx1151). LDS is 32 banks x 1 dword, bank = dword_addr % 32. An unpadded
    // cell row is exactly packed_size/2 dwords -- 32 at K=4 -- so warp-group
    // w+1's base sits an exact multiple of 32 dwords after w's and lands on the
    // SAME banks. A wave64 spans two warp-groups, so every wave conflicted
    // 2-way on every decode read. One spare dword per row shifts each group by
    // one bank and the conflict disappears.
    __shared__ uint32_t s_packed[8][packed_size / 2 + 1];
    __shared__ __half2  tile[16][8][8];              // frags_to_tile's row-major [16][128]
    __shared__ __half   tileT[128][18];              // [col][r], padded: bank-conflict free
    __shared__ __half2  xs[MB][8];                   // x pairs along r

    float acc[RH];
    #pragma unroll
    for (int i = 0; i < RH; ++i) acc[i] = 0.f;

    for (int kb = kbeg; kb < kend; ++kb) {
        __syncthreads();
        if (t < packed_size)
            { constexpr int i4pc = packed_size / 8;
              *(int4*)&s_packed[t / i4pc][(t % i4pc) * 4] =
                  ((const int4*)(g_packed + (size_t)(kb * packed_blocks_n + n) * packed_size))[t]; }
        __syncthreads();

        exl3gfx::FragB frag[2];
        exl3gfx::dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);
        exl3gfx::frags_to_tile(frag, lane_id, warp_id, tile);

        for (int i = t; i < mb * 8; i += 256) {
            const int mm = i >> 3, jj = i & 7;
            const __half* xr = x + (size_t)(m0 + mm) * ldx + kb * 16 + jj * 2;
            xs[mm][jj] = __halves2half2(xr[0], xr[1]);
        }
        __syncthreads();

        const __half* Wr = (const __half*)tile;
        for (int i = t; i < 2048; i += 256) tileT[i & 127][i >> 7] = Wr[i];
        __syncthreads();

        __half2 w[8];
        const __half2* wc = (const __half2*)&tileT[c][0];
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = wc[j];

        #pragma unroll
        for (int i = 0; i < RH; ++i) {
            const int mm = h * RH + i;
            if (mm < mb) {
                float a = acc[i];
                #pragma unroll
                for (int j = 0; j < 8; ++j) a = dot2acc(w[j], xs[mm][j], a);
                acc[i] = a;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < RH; ++i) {
        const int mm = h * RH + i;
        if (mm < mb) {
            float* yp = y + (size_t)(m0 + mm) * ldy + (size_t)n * 16 + c;
            if (G == 1) *yp = acc[i];
            else        atomicAdd(yp, acc[i]);
        }
    }
}


// ---------------------------------------------------------------------------
// M == 1 (single-stream decode). SEPARATE KERNEL, not the band kernel at MB=2.
//
// At M=1 the band kernel's LDS tile earns nothing: frags_to_tile writes 2048
// halves and the transpose moves them again, all to serve 2048 MACs -- one MAC
// per weight is the arithmetic intensity of a GEMV and no amount of staging
// improves it. Measured consequence: the earlier fused GEMV ran 269.7 G
// weights/s = 202 GB/s encoded, 24% of the 852 GB/s roofline and ~6x off the
// decode-ALU floor (dq4 is ~4 instructions per weight => ~14.5 ms/token
// against a 13.3 ms bandwidth floor).
//
// So: dot STRAIGHT FROM THE FRAGMENTS (no tile, no transpose) and carry the
// column partials in REGISTERS across a whole k-range (no per-k-block global
// atomic). The other fragment-direct kernel, gemv_kernel_frag, had the first
// half of that and not the second: one k-block per block meant 5.6M global
// atomics per projection.
//
// The lane->(row, column) mapping is frags_to_tile's own, VERBATIM including
// the __shfl_down(...,4,32) merge. It is not re-derived: re-deriving it is how
// a wrong element mapping becomes plausible garbage, which already cost one
// debugging round on this store (the mul1/cb-2 bug).
// ---------------------------------------------------------------------------
// M == 1, NO MERGE. frags_to_tile's __shfl_down(...,4,32) exists to assemble a
// TILE: it pairs lane L with lane L+4 so the two halves of a __half2 are
// adjacent COLUMNS of a row-major tile. A GEMV never materialises a tile, so
// the merge buys nothing and costs 4 ds_bpermute, 8 half2 packs and an
// exec-mask split that idles half the lanes in the FMA.
//
// The mapping below is DERIVED FROM the merge, not invented: in
// frags_to_tile/gemv_kernel_frag, m0 = (frag[0][0].low of lane L,
// frag[0][0].low of lane L+4) lands at row r0, columns b0 and b0+1. So lane L
// alone owns column b0 + (L>>2 & 1), with b0 = warp*16 + (L/8)*2 -- and
// (L & ~4)/8 == L/8, since clearing bit 2 cannot change the /8 quotient, which
// is why the same formula serves both halves of the pair. frag[1] is the same
// with b1 = b0 + 8. Rows are r0..r3 exactly as before.
//
// A re-derived element mapping is the one failure mode that produces plausible
// garbage rather than a crash (it already cost a round on the mul1/cb-2 bug),
// so bench_exl3_m1 --verify diffs this against the verbatim-merge kernel on the
// real store before it is allowed to serve.
template <int K, int cb>
__global__ __launch_bounds__(256)
void k_gemv_frag_m1_nm(float* __restrict__ y,
                       const __half* __restrict__ x,
                       const uint16_t* __restrict__ g_packed,
                       int packed_blocks_n, int in_f, int G, int ldx, int ldy)
{
    constexpr int packed_size = 256 * K / 16;
    const int t = threadIdx.x, lane_id = t % 32, warp_id = t / 32;
    const int n = blockIdx.x * 8;
    // ROW = blockIdx.y. blockIdx.x is the output tile and blockIdx.z the split-k band, so
    // y was free; carrying the row here lets ONE launch cover all M rows of a batched
    // projection instead of M launches, without touching the arithmetic below (which is
    // the path verified against BF16 at 21.59 dB). At M=1 gridDim.y is 1 and both offsets
    // are zero, so the decode path is bit-unchanged.
    y += (size_t)blockIdx.y * ldy;
    x += (size_t)blockIdx.y * ldx;

    const int nk   = in_f / 16;
    const int kper = (nk + G - 1) / G;
    const int kbeg = blockIdx.z * kper;
    const int kend = (kbeg + kper < nk) ? (kbeg + kper) : nk;
    if (kbeg >= kend) return;

    // LDS STAGE PADDING (their EXL3_HIP_STG_PAD, worth 31.7 -> 34.9 tok/s on
    // gfx1151). LDS is 32 banks x 1 dword, bank = dword_addr % 32. An unpadded
    // cell row is exactly packed_size/2 dwords -- 32 at K=4 -- so warp-group
    // w+1's base sits an exact multiple of 32 dwords after w's and lands on the
    // SAME banks. A wave64 spans two warp-groups, so every wave conflicted
    // 2-way on every decode read. One spare dword per row shifts each group by
    // one bank and the conflict disappears.
    __shared__ uint32_t s_packed[8][packed_size / 2 + 1];
    __shared__ float    xs[16];
    __shared__ float    acc_c[128];
    if (t < 128) acc_c[t] = 0.f;

    const int bit = (lane_id >> 2) & 1;                 // which of the merged pair this lane was
    const int cA  = warp_id * 16 + (lane_id / 8) * 2 + bit;
    const int cB  = cA + 8;
    const int r0 = (lane_id % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
    float pA = 0.f, pB = 0.f;

    const int4* gp4 = (const int4*)(g_packed + (size_t)(kbeg * packed_blocks_n + n) * packed_size);
    const size_t kstride4 = (size_t)packed_blocks_n * packed_size / 8;
    int4 nxt; __half xn;
    if (t < packed_size) nxt = gp4[t];
    if (t < 16) xn = x[kbeg * 16 + t];

    for (int kb = kbeg; kb < kend; ++kb) {
        __syncthreads();
        if (t < packed_size) {   // padded rows: place by (group, dword) not flat
            const int cell = t / (packed_size / 8), q = t % (packed_size / 8);
            *(int4*)&s_packed[cell][q * 4] = nxt;
        }
        if (t < 16) xs[t] = __half2float(xn);
        __syncthreads();
        if (kb + 1 < kend) {
            if (t < packed_size) nxt = gp4[(size_t)(kb + 1 - kbeg) * kstride4 + t];
            if (t < 16) xn = x[(kb + 1) * 16 + t];
        }

        exl3gfx::FragB frag[2];
        exl3gfx::dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

        const float x0 = xs[r0], x1 = xs[r1], x2 = xs[r2], x3 = xs[r3];
        pA = fmaf(x0, __low2float (frag[0][0]), pA);
        pA = fmaf(x1, __high2float(frag[0][0]), pA);
        pA = fmaf(x2, __low2float (frag[0][1]), pA);
        pA = fmaf(x3, __high2float(frag[0][1]), pA);
        pB = fmaf(x0, __low2float (frag[1][0]), pB);
        pB = fmaf(x1, __high2float(frag[1][0]), pB);
        pB = fmaf(x2, __low2float (frag[1][1]), pB);
        pB = fmaf(x3, __high2float(frag[1][1]), pB);
    }

    __syncthreads();
    atomicAdd(&acc_c[cA], pA);
    atomicAdd(&acc_c[cB], pB);
    __syncthreads();
    if (t < 128) atomicAdd(&y[(size_t)n * 16 + t], acc_c[t]);
}

template <int K, int cb>
__global__ __launch_bounds__(256)
void k_gemv_frag_m1(float* __restrict__ y,
                    const __half* __restrict__ x,
                    const uint16_t* __restrict__ g_packed,
                    int packed_blocks_n, int in_f, int G)
{
    constexpr int packed_size = 256 * K / 16;
    const int t = threadIdx.x, lane_id = t % 32, warp_id = t / 32;
    const int n = blockIdx.x * 8;

    const int nk   = in_f / 16;
    const int kper = (nk + G - 1) / G;
    const int kbeg = blockIdx.z * kper;
    const int kend = (kbeg + kper < nk) ? (kbeg + kper) : nk;
    if (kbeg >= kend) return;

    // LDS STAGE PADDING (their EXL3_HIP_STG_PAD, worth 31.7 -> 34.9 tok/s on
    // gfx1151). LDS is 32 banks x 1 dword, bank = dword_addr % 32. An unpadded
    // cell row is exactly packed_size/2 dwords -- 32 at K=4 -- so warp-group
    // w+1's base sits an exact multiple of 32 dwords after w's and lands on the
    // SAME banks. A wave64 spans two warp-groups, so every wave conflicted
    // 2-way on every decode read. One spare dword per row shifts each group by
    // one bank and the conflict disappears.
    __shared__ uint32_t s_packed[8][packed_size / 2 + 1];
    __shared__ float    xs[16];
    __shared__ float    acc_c[128];
    if (t < 128) acc_c[t] = 0.f;

    // Column indices this lane contributes to are fixed across the whole
    // k-range, so the partials live in registers for the entire loop.
    const int c0 = lane_id / 8, c1 = c0 + 4;
    const int b0 = warp_id * 16 + c0 * 2;
    const int b1 = warp_id * 16 + c1 * 2;
    const int r0 = (lane_id % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;

    // SOFTWARE PIPELINE. Without it every iteration stalls all 256 threads at
    // the barrier while the 64 loading threads wait out a full global-memory
    // latency: measured ~4400 cycles per iteration for ~50 instructions of
    // work, i.e. 22% of both the bandwidth and the decode-ALU floor. Issuing
    // the NEXT k-block's trellis load before decoding the current one overlaps
    // that latency with the decode+FMA. It costs one int4 of registers and no
    // extra LDS, which matters because the band kernel already showed this
    // kernel family is occupancy-bound.
    const int4* gp4 = (const int4*)(g_packed + (size_t)(kbeg * packed_blocks_n + n) * packed_size);
    const size_t kstride4 = (size_t)packed_blocks_n * packed_size / 8;   // int4s per k-block step
    int4 nxt;
    __half xn;
    if (t < packed_size) nxt = gp4[t];
    if (t < 16) xn = x[kbeg * 16 + t];

    for (int kb = kbeg; kb < kend; ++kb) {
        __syncthreads();
        if (t < packed_size) {   // padded rows: place by (group, dword) not flat
            const int cell = t / (packed_size / 8), q = t % (packed_size / 8);
            *(int4*)&s_packed[cell][q * 4] = nxt;
        }
        if (t < 16) xs[t] = __half2float(xn);
        __syncthreads();
        // Issue the next fetch BEFORE the decode, so its latency is paid under
        // the work below rather than at the next barrier.
        if (kb + 1 < kend) {
            if (t < packed_size) nxt = gp4[(size_t)(kb + 1 - kbeg) * kstride4 + t];
            if (t < 16) xn = x[(kb + 1) * 16 + t];
        }

        exl3gfx::FragB frag[2];
        exl3gfx::dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

        // --- frags_to_tile's merge, verbatim ---
        int n0b = __shfl_down((int)exl3gfx::half2_to_bits(frag[0][0]), 4, 32);
        int n1b = __shfl_down((int)exl3gfx::half2_to_bits(frag[0][1]), 4, 32);
        int n2b = __shfl_down((int)exl3gfx::half2_to_bits(frag[1][0]), 4, 32);
        int n3b = __shfl_down((int)exl3gfx::half2_to_bits(frag[1][1]), 4, 32);
        if (!(lane_id & 4)) {
            __half2 n0 = exl3gfx::half2_from_bits((uint32_t)n0b);
            __half2 n1 = exl3gfx::half2_from_bits((uint32_t)n1b);
            __half2 n2 = exl3gfx::half2_from_bits((uint32_t)n2b);
            __half2 n3 = exl3gfx::half2_from_bits((uint32_t)n3b);
            __half2 m0 = __halves2half2(__low2half (frag[0][0]), __low2half (n0));
            __half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            __half2 m2 = __halves2half2(__low2half (frag[0][1]), __low2half (n1));
            __half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            __half2 m4 = __halves2half2(__low2half (frag[1][0]), __low2half (n2));
            __half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            __half2 m6 = __halves2half2(__low2half (frag[1][1]), __low2half (n3));
            __half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            const float x0 = xs[r0], x1 = xs[r1], x2 = xs[r2], x3 = xs[r3];
            p0 = fmaf(x0, __low2float(m0), p0);  p1 = fmaf(x0, __high2float(m0), p1);
            p0 = fmaf(x1, __low2float(m1), p0);  p1 = fmaf(x1, __high2float(m1), p1);
            p0 = fmaf(x2, __low2float(m2), p0);  p1 = fmaf(x2, __high2float(m2), p1);
            p0 = fmaf(x3, __low2float(m3), p0);  p1 = fmaf(x3, __high2float(m3), p1);
            p2 = fmaf(x0, __low2float(m4), p2);  p3 = fmaf(x0, __high2float(m4), p3);
            p2 = fmaf(x1, __low2float(m5), p2);  p3 = fmaf(x1, __high2float(m5), p3);
            p2 = fmaf(x2, __low2float(m6), p2);  p3 = fmaf(x2, __high2float(m6), p3);
            p2 = fmaf(x3, __low2float(m7), p2);  p3 = fmaf(x3, __high2float(m7), p3);
        }
    }

    __syncthreads();
    if (!(lane_id & 4)) {
        atomicAdd(&acc_c[b0    ], p0);
        atomicAdd(&acc_c[b0 + 1], p1);
        atomicAdd(&acc_c[b1    ], p2);
        atomicAdd(&acc_c[b1 + 1], p3);
    }
    __syncthreads();
    if (t < 128) atomicAdd(&y[(size_t)n * 16 + t], acc_c[t]);
}


// Only K = 4 and K = 6 are instantiated: those are the two published stores'
// layer widths (and the lm_head's K=6). Anything else is a loud refusal rather
// than a silently wrong dispatch.
#define GF1(k, c) k_gemv_frag_m1<k, c>
constexpr auto frag1_inst = std::array{ GF1(4,0), GF1(4,1), GF1(4,2), GF1(6,0), GF1(6,1), GF1(6,2) };
#undef GF1
#define GN1(k, c) k_gemv_frag_m1_nm<k, c>
constexpr auto frag1nm_inst = std::array{ GN1(4,0), GN1(4,1), GN1(4,2), GN1(6,0), GN1(6,1), GN1(6,2) };
#undef GN1
#define GB32(k, c) k_gemm_band<k, c, 32>
constexpr auto band32_inst = std::array{ GB32(4,0), GB32(4,1), GB32(4,2), GB32(6,0), GB32(6,1), GB32(6,2) };
#undef GB32
// MB is the ROWS PER BLOCK, and it sets how many times the trellis is decoded:
// a column tile is decoded once per row band, so ceil(M/MB) decodes. At M=256
// MB=32 decodes every tile EIGHT times, and the decode -- not the weight traffic
// (68 GB/chunk against 852 GB/s) and not the dot -- is what sets prefill. MB=64
// halves that at 32 accumulators per thread.
#define GB64(k, c) k_gemm_band<k, c, 64>
constexpr auto band64_inst = std::array{ GB64(4,0), GB64(4,1), GB64(4,2), GB64(6,0), GB64(6,1), GB64(6,2) };
#undef GB64

static inline int slot(int K, int cbsel, const char* who) {
    const int kb = (K == 4) ? 0 : (K == 6 ? 1 : -1);
    const int cx = cbsel % 8;                        // cb() packs K-1 in the low bits
    const int cg = cbsel / 8;                        // 0 = plain, 1 = mcg, 2 = mul1
    (void)cx;
    if (kb < 0 || cg < 0 || cg > 2) {
        std::fprintf(stderr, "q27_exl3 %s: unsupported K=%d cb=%d (only K=4,6)\n", who, K, cbsel);
        return -1;
    }
    return kb * 3 + cg;
}

__global__ __launch_bounds__(256)
void k_h2f(float* __restrict__ o, const __half* __restrict__ i, int n)
{
    const int t = blockIdx.x * 256 + threadIdx.x;
    if (t < n) o[t] = __half2float(i[t]);
}

}  // namespace q27exl3

bool q27_exl3_gemm(float* y, const __half* x, const q27_exl3_t* w, int M, hipStream_t s,
                   int ldx, int ldy)
{
    if (ldx <= 0) ldx = w ? w->in : 0;
    if (ldy <= 0) ldy = w ? w->out : 0;
    if (!y || !x || !w || !w->live() || M < 1) return false;
    if (w->out % 128 || w->in % 16) return false;
    const int idx = q27exl3::slot(w->K, w->cb(), "gemm");
    if (idx < 0) return false;
    // Small M has too few (out/128) column tiles to fill the machine, so split
    // k across blockIdx.z; large M already has the row bands for parallelism
    // and takes G=1, which means ZERO atomics on the prefill path.
    if (M == 1) {
        // SINGLE-STREAM DECODE. Fragment-direct kernel, not the band kernel:
        // at M=1 the LDS tile + transpose move 2048 halves to serve 2048 MACs
        // and earn nothing (one MAC per weight is a GEMV's intensity).
        // G split-k because (out/128) alone is 136 blocks for gate_proj, which
        // does not fill 60 CUs; G=8 gives 1088 blocks and costs G atomics per
        // output element instead of in/16 of them.
        const int G = 8;
        const dim3 grid(w->out / 128, 1, G);
        static const int nm = [] { const char* e = getenv("Q27_EXL3_M1_MERGE");
                                  return (e && e[0] == '1') ? 0 : 1; }();   // default: no-merge
        if (nm) hipLaunchKernelGGL(q27exl3::frag1nm_inst[idx], grid, dim3(256), 0, s,
                                   y, x, w->trellis, w->out / 16, w->in, G, w->in, w->out);
        else    hipLaunchKernelGGL(q27exl3::frag1_inst[idx], grid, dim3(256), 0, s,
                                   y, x, w->trellis, w->out / 16, w->in, G);
    } else {
        // ---- band32 IS MEASURABLY WRONG. Q27_EXL3_BAND=1 opts back into it. ----
        // test_exl3_band feeds M IDENTICAL rows and compares each output row against the
        // M=1 path (itself verified against BF16 at 21.59 dB). band32 reads -2.49 dB on
        // gate_proj and -2.62 on down_proj -- every row the same wrong answer, so it is a
        // systematic error in the kernel, not a race or a stride overrun.
        //
        // Prefill is the only M>1 consumer (the chunk path still issues the attention
        // projections at M=1), so a CORRECT decode was rebuilding its state from a corrupt
        // prefill. That is exactly the salad this arm produced while every tensor, the
        // whole M=1 chain, and the live in-engine decode MLP all verified.
        //
        // Until it is re-derived, M>1 runs the VERIFIED fragment GEMV once per row.
        // Correct and slow beats fast and wrong. The donor for the re-derivation is
        // upstream exl3_gemm_inner.cuh; CarouselAether/rocm_exl3's ROCM_PORT_MAP.md rates
        // that file an extensive re-derivation (501 of 733 lines changed upstream), which
        // is why it is not attempted inline here.
        // FIXED 2026-09-19 (row-aware LDS stage write, above): band32 now measures 120.51 dB
        // against the verified M=1 path at M=8, 32 and 256 -- bit-exact bar atomicAdd ordering.
        // It is the default again because it decodes the trellis ONCE per block for MB=32 rows,
        // where the per-row GEMV re-decodes for every row. Q27_EXL3_BAND=0 forces the per-row
        // path back (correct, ~30%% slower prefill) if this is ever suspect again.
        static const int use_band = [] { const char* e = getenv("Q27_EXL3_BAND");
                                         return (e && e[0] == '0') ? 0 : 1; }();
        if (!use_band) {
            // ONE launch for all M rows (grid.y = M), not M launches. Split-k is dropped to
            // G=2 because M rows already give gridDim.y-fold parallelism -- G=8 on top of
            // that is 8x the atomics for occupancy the rows already supply.
            const int Gm = (M >= 8) ? 2 : 8;
            const dim3 gridm(w->out / 128, (unsigned)M, Gm);
            hipLaunchKernelGGL(q27exl3::frag1nm_inst[idx], gridm, dim3(256), 0, s,
                               y, x, w->trellis, w->out / 16, w->in, Gm, ldx, ldy);
            return hipPeekAtLastError() == hipSuccess;
        }
        // MEASURED 2026-09-19, 15360-token prefill through the Harness:
        //   MB=32  123.0 s      MB=64  129.3 s      MB=32 + k-pairing  144.3 s
        // Both "improvements" lost, and both raised LDS per block. That rules
        // out the decode AND the barriers and says the band kernel is
        // occupancy-bound: the lever is LDS per block, not work per block.
        const int G = 1;
        const dim3 grid(w->out / 128, (M + 31) / 32, G);
        hipLaunchKernelGGL(q27exl3::band32_inst[idx], grid, dim3(256), 0, s,
                           y, x, w->trellis, w->out / 16, M, w->out, w->in, G, ldx, ldy);
    }
    return hipPeekAtLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// Workspace: x' (M*in fp16) then the fp32 accumulator (M*out fp32), each 256B
// aligned. One arena sized for the widest projection serves every layer.
static inline size_t al256(size_t b) { return (b + 255) & ~(size_t)255; }

static inline size_t ws_off_acc(int M) { return al256((size_t)Q27_EXL3_WMAX * M * 2); }
static inline size_t ws_off_act(int M) { return ws_off_acc(M) + al256((size_t)Q27_EXL3_WMAX * M * 4); }
static inline size_t ws_off_nrm(int M) { return ws_off_act(M) + al256((size_t)Q27_EXL3_WMAX * M * 2); }

size_t q27_exl3_ws_bytes(int max_in, int max_out, int max_M)
{
    (void)max_in; (void)max_out;
    return ws_off_nrm(max_M) + al256((size_t)Q27_EXL3_WMAX * max_M * 2);
}
// Third region: the bf16 activation the EXL3 MLP hands from swiglu to down_proj.
unsigned short* q27_exl3_ws_act(void* ws, int in, int M, int out_acc)
{
    (void)in; (void)out_acc;
    return (unsigned short*)((char*)ws + ws_off_act(M));
}
unsigned short* q27_exl3_ws_norm(void* ws, int M)
{
    return (unsigned short*)((char*)ws + ws_off_nrm(M));
}

static bool exl3_run(const __half* x, const q27_exl3_t* w, int M, void* ws,
                     __half** xp_out, float** acc_out, hipStream_t s)
{
    if (!ws || !w->live() || M < 1) return false;
    __half* xp  = (__half*)ws;
    float*  acc = (float*)((char*)ws + ws_off_acc(M));

    // x' = H128(x (*) suh)
    if (q27_exl3_had128_fp16(xp, x, w->suh, nullptr, w->in, M, s) != hipSuccess) return false;
    if (hipMemsetAsync(acc, 0, (size_t)w->out * M * 4, s) != hipSuccess) return false;
    if (!q27_exl3_gemm(acc, xp, w, M, s)) return false;
    *xp_out = xp; *acc_out = acc;
    return true;
}

bool q27_exl3_linear(__half* y, const __half* x, const q27_exl3_t* w, int M, void* ws, hipStream_t s)
{
    __half* xp; float* acc;
    if (!exl3_run(x, w, M, ws, &xp, &acc, s)) return false;
    // y = H128(y') (*) svh
    return q27_exl3_had128_f32in(y, acc, w->svh, w->out, M, s) == hipSuccess;
}

bool q27_exl3_linear_f32(float* y, const __half* x, const q27_exl3_t* w, int M, void* ws, hipStream_t s)
{
    __half* xp; float* acc;
    if (!exl3_run(x, w, M, ws, &xp, &acc, s)) return false;
    // Narrow through fp16 (the authority's own output dtype) then widen: the
    // post-Hadamard is defined on fp16 in exllamav3, so doing it in fp32 here
    // would be a different function, not a more accurate one.
    __half* tmp = (__half*)acc;                      // reuse: reads complete before writes
    if (q27_exl3_had128_f32in(tmp, acc, w->svh, w->out, M, s) != hipSuccess) return false;
    const int n = w->out * M;
    hipLaunchKernelGGL(q27exl3::k_h2f, dim3((n + 255) / 256), dim3(256), 0, s, y, tmp, n);
    return hipPeekAtLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// ENGINE-FACING PROJECTIONS. These are the exact call shape the incumbent
// q27_proj_fp8_bf16 / q27_proj_fp8 have, except the input is the bf16 NORM
// VECTOR the engine already computes rather than a quantized activation: EXL3
// rotates the input itself (suh + H128), so there is nothing to pre-quantize.
//
//   ws layout: [ x' fp16 M*in ][ fp32 accumulator M*out ]
//
// Weight traffic per call is exactly in*out*K/8 bytes -- the checkpoint bytes,
// read once, decoded in LDS, never materialized.
extern hipError_t q27_exl3_had128_pre_bf16(__half*, const unsigned short*, const __half*, int, int, hipStream_t, int, int);
extern hipError_t q27_exl3_had128_post_bf16(unsigned short*, const float*, const __half*, int, int, hipStream_t, int, int);
extern hipError_t q27_exl3_had128_post_f32(float*, const float*, const __half*, int, int, hipStream_t, int, int);

// Proof-of-use counters. A dispatcher that silently falls back to the incumbent
// format looks exactly like one that ran EXL3 -- same output, same log -- so the
// only way to know which weights a measurement actually read is to count.
std::atomic<unsigned long long> g_exl3_calls{0}, g_exl3_rows{0}, g_exl3_calls_m1{0};

static bool exl3_pre_gemm(const q27_exl3_t* w, const unsigned short* xbf, int M, void* ws,
                          float** acc_out, hipStream_t s, int ldx)
{
    if (!ws || !w || !w->live() || M < 1) return false;
    __half* xp  = (__half*)ws;
    float*  acc = (float*)((char*)ws + ws_off_acc(M));
    // x' is written CONTIGUOUS even when x is strided, so the GEMM's x stride is
    // always w->in and only the Hadamard has to know about the caller's layout.
    if (q27_exl3_had128_pre_bf16(xp, xbf, w->suh, w->in, M, s, ldx, w->in) != hipSuccess) return false;
    // G=1 writes the accumulator outright; only the split-k decode path accumulates.
    // ACC MUST BE ZEROED FOR EVERY M. The "G=1 writes the accumulator outright" claim held
    // for the split-k GEMV only; band32 does not cover every element outright, so at M>1 the
    // accumulator kept the PREVIOUS projection's values out of the shared arena. Measured
    // 2026-09-19: band vs the verified M=1 path was -2.49 dB on gate_proj before this line.
    if (hipMemsetAsync(acc, 0, (size_t)w->out * M * 4, s) != hipSuccess) return false;
    if (!q27_exl3_gemm(acc, xp, w, M, s, w->in, w->out)) return false;
    g_exl3_calls.fetch_add(1, std::memory_order_relaxed);
    g_exl3_rows.fetch_add((unsigned long long)M, std::memory_order_relaxed);
    if (M == 1) g_exl3_calls_m1.fetch_add(1, std::memory_order_relaxed);
    *acc_out = acc;
    return true;
}

bool q27_exl3_proj_bf16(const q27_exl3_t* w, const unsigned short* x, unsigned short* y,
                        int M, void* ws, hipStream_t s, int ldx, int ldy)
{
    float* acc;
    if (!exl3_pre_gemm(w, x, M, ws, &acc, s, ldx)) return false;
    return q27_exl3_had128_post_bf16(y, acc, w->svh, w->out, M, s, w->out, ldy) == hipSuccess;
}

bool q27_exl3_proj_f32(const q27_exl3_t* w, const unsigned short* x, float* y,
                       int M, void* ws, hipStream_t s, int ldx, int ldy)
{
    float* acc;
    if (!exl3_pre_gemm(w, x, M, ws, &acc, s, ldx)) return false;
    return q27_exl3_had128_post_f32(y, acc, w->svh, w->out, M, s, w->out, ldy) == hipSuccess;
}

// ---------------------------------------------------------------------------
// Batched swiglu -> bf16. The incumbent swiglu variants all emit a QUANTIZED
// activation (int8 + scales) because their down_proj consumer wants one; EXL3's
// down rotates its own input, so what it needs is the plain bf16 activation in
// NATURAL order. gate/up here are the EXL3 projections' own fp32 output, so the
// even-then-odd permutation the NVFP4 chain carries does not apply.
namespace q27exl3 {
__global__ __launch_bounds__(256)
void k_swiglu_bf16(const float* __restrict__ a, const float* __restrict__ b,
                   unsigned short* __restrict__ o, int n, int rows, int lda, int ldo)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * rows) return;
    const int r = i / n, c = i - r * n;
    const float g = a[(size_t)r * lda + c];
    const float u = b[(size_t)r * lda + c];
    const float v = (g / (1.f + __expf(-g))) * u;     // silu(gate) * up
    unsigned bits; __builtin_memcpy(&bits, &v, 4);
    o[(size_t)r * ldo + c] = (unsigned short)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
}

namespace q27exl3 {
__global__ __launch_bounds__(256)
void k_add_res(float* __restrict__ part, const unsigned short* __restrict__ res,
               float share, int n, int rows, int ldp, int ldr)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * rows) return;
    const int r = i / n, c = i - r * n;
    unsigned u = (unsigned)res[(size_t)r * ldr + c] << 16; float f; __builtin_memcpy(&f, &u, 4);
    part[(size_t)r * ldp + c] += share * f;
}
}
// The incumbent projections fold the residual share into their epilogue
// (P_FP8_RES / P_DOWN take resid + share). EXL3's epilogue is the output
// Hadamard, which cannot carry it, so it rides as a separate pass. The share
// constant is MIRRORED from the incumbent call site rather than re-derived.
bool q27_exl3_add_res(float* part, const unsigned short* res, float share,
                      int n, int rows, int ldp, int ldr, hipStream_t s)
{
    if (n <= 0 || rows <= 0) return false;
    if (ldp <= 0) ldp = n; if (ldr <= 0) ldr = n;
    const long long tot = (long long)n * rows;
    hipLaunchKernelGGL(q27exl3::k_add_res, dim3((unsigned)((tot + 255) / 256)), dim3(256), 0, s,
                       part, res, share, n, rows, ldp, ldr);
    return hipPeekAtLastError() == hipSuccess;
}

namespace q27exl3 {
// The prefill sweep's post-norm emits a QUANTIZED activation (int8 + per-group
// scales, in the permuted order its NVFP4 consumer wants). EXL3 rotates its own
// input and needs the plain bf16 normalised vector in NATURAL order, so it gets
// its own norm off the same fp32 partial rather than trying to invert theirs --
// dequantizing a permuted activation and feeding it to a natural-order suh is
// exactly the kind of wrong-but-plausible wiring that costs a day.
// Contract 2: rsqrt(sum x^2 / N + 1e-6), weights applied as (1 + W).
__global__ __launch_bounds__(256)
void k_rmsnorm_f32_bf16(const float* __restrict__ x, const unsigned short* __restrict__ w,
                        unsigned short* __restrict__ y, int n, int ldx, int ldy)
{
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * ldx;
    __shared__ float sm[256 / 64];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) { const float v = xr[i]; ss = fmaf(v, v, ss); }
    #pragma unroll
    for (int o = 32; o; o >>= 1) ss += __shfl_down(ss, o, 64);
    if ((threadIdx.x & 63) == 0) sm[threadIdx.x >> 6] = ss;
    __syncthreads();
    if (threadIdx.x == 0) { float t = 0.f; for (int i = 0; i < 4; ++i) t += sm[i]; sm[0] = t; }
    __syncthreads();
    const float inv = rsqrtf(sm[0] / (float)n + 1e-6f);
    unsigned short* yr = y + (size_t)row * ldy;
    for (int i = threadIdx.x; i < n; i += 256) {
        unsigned u = (unsigned)w[i] << 16; float wf; __builtin_memcpy(&wf, &u, 4);
        const float v = xr[i] * inv * (1.f + wf);
        unsigned b; __builtin_memcpy(&b, &v, 4);
        yr[i] = (unsigned short)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
    }
}
__global__ __launch_bounds__(256)
void k_add_res_f32(float* __restrict__ part, const float* __restrict__ res,
                   float share, int n, int rows, int ldp, int ldr)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n * rows) return;
    const int r = i / n, c = i - r * n;
    part[(size_t)r * ldp + c] += share * res[(size_t)r * ldr + c];
}
}

bool q27_exl3_rmsnorm_bf16(const float* x, const unsigned short* w, unsigned short* y,
                           int n, int rows, int ldx, int ldy, hipStream_t s)
{
    if (n <= 0 || rows <= 0) return false;
    if (ldx <= 0) ldx = n; if (ldy <= 0) ldy = n;
    hipLaunchKernelGGL(q27exl3::k_rmsnorm_f32_bf16, dim3((unsigned)rows), dim3(256), 0, s, x, w, y, n, ldx, ldy);
    return hipPeekAtLastError() == hipSuccess;
}

bool q27_exl3_add_res_f32(float* part, const float* res, float share,
                          int n, int rows, int ldp, int ldr, hipStream_t s)
{
    if (n <= 0 || rows <= 0) return false;
    if (ldp <= 0) ldp = n; if (ldr <= 0) ldr = n;
    const long long tot = (long long)n * rows;
    hipLaunchKernelGGL(q27exl3::k_add_res_f32, dim3((unsigned)((tot + 255) / 256)), dim3(256), 0, s,
                       part, res, share, n, rows, ldp, ldr);
    return hipPeekAtLastError() == hipSuccess;
}

bool q27_exl3_swiglu_bf16(const float* a, const float* b, unsigned short* o,
                          int n, int rows, int lda, int ldo, hipStream_t s)
{
    if (n <= 0 || rows <= 0) return false;
    if (lda <= 0) lda = n; if (ldo <= 0) ldo = n;
    const long long tot = (long long)n * rows;
    hipLaunchKernelGGL(q27exl3::k_swiglu_bf16, dim3((unsigned)((tot + 255) / 256)), dim3(256), 0, s,
                       a, b, o, n, rows, lda, ldo);
    return hipPeekAtLastError() == hipSuccess;
}
