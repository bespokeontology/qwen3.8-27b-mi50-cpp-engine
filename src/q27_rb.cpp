// Q27_LS_Q8 stage 3: the int8 GEMMs of the prefill path on rocBLAS (Tensile int8x4 / v_dot4 kernels, gfx906).
// Weights int8 [N][K] (lda = K), activations int8 [M][K] (ldb = K), output int32 [M][N] row-major (ldc = N);
// the pack_int8x4 layout coincides with these natural layouts (verified against a host reference, full K and
// K-subranges). Per (N, K, M) shape the fastest Tensile SOLUTION is chosen at init (q27_rb_tune_i8g, beta API
// rocblas_gemm_ex_get_solutions) on the real weights and random activations; rocBLAS's default pick measured
// 9-15 TMAC/s in-engine on real data against 16-20 in a constant-data probe.
#define ROCBLAS_BETA_FEATURES_API 1
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mutex>
#include "q27.h"
#include "q27_pfblas.h"
enum { Q27_RB_MAXDEV = 8 };
static rocblas_handle g_rb[Q27_RB_MAXDEV];
struct q27_rb_sol { int dev, N, K, M; int sol; float ms_best, ms_def; };
static std::vector<q27_rb_sol> g_sols;
static std::mutex g_sol_mu;
static int q27_rb_lookup(int dev, int N, int K, int M) {
    std::lock_guard<std::mutex> lk(g_sol_mu);
    for (const auto& s : g_sols) if (s.dev == dev && s.N == N && s.K == K && s.M == M) return s.sol;
    return -1;
}
extern "C" int q27_rb_init(int dev) {
    if (dev < 0 || dev >= Q27_RB_MAXDEV) return 0;
    if (g_rb[dev]) return 1;
    rocblas_handle h = nullptr;
    if (rocblas_create_handle(&h) != rocblas_status_success) { std::fprintf(stderr, "q27_rb_init: rocblas_create_handle failed (device %d)\n", dev); return 0; }
    rocblas_set_pointer_mode(h, rocblas_pointer_mode_host);
    g_rb[dev] = h;
    return 1;
}
// Fixed batch4 FP32 GEMMs for bounded prefill. They share the existing handle,
// are ordered on the caller's serialized stream, and never invoke i8 tuning.
extern "C" int q27_rb_pfblas_qk(int dev, const float* k, const float* q, float* p, int b, int R, hipStream_t s) {
    if (dev < 0 || dev >= Q27_RB_MAXDEV || !g_rb[dev] || !k || !q || !p ||
        b < 1 || b > Q27_PFBLAS_B || R < 1 || R > Q27_PFBLAS_G * Q27_PFBLAS_MAX_M) return 0;
    rocblas_handle h = g_rb[dev];
    if (rocblas_set_stream(h, s) != rocblas_status_success) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    const rocblas_status st = rocblas_sgemm_strided_batched(h,
        rocblas_operation_transpose, rocblas_operation_none, b, R, Q27_PFBLAS_D,
        &alpha, k, Q27_PFBLAS_D, (rocblas_stride)Q27_PFBLAS_B * Q27_PFBLAS_D,
        q, Q27_PFBLAS_D, (rocblas_stride)R * Q27_PFBLAS_D,
        &beta, p, Q27_PFBLAS_B, (rocblas_stride)R * Q27_PFBLAS_B, Q27_PFBLAS_H);
    if (st != rocblas_status_success) std::fprintf(stderr, "Q27_PF_BLAS QK dev=%d b=%d R=%d status=%d\n", dev, b, R, (int)st);
    return st == rocblas_status_success;
}
extern "C" int q27_rb_pfblas_pv(int dev, const float* v, const float* p, float* u, int b, int R, hipStream_t s) {
    if (dev < 0 || dev >= Q27_RB_MAXDEV || !g_rb[dev] || !v || !p || !u ||
        b < 1 || b > Q27_PFBLAS_B || R < 1 || R > Q27_PFBLAS_G * Q27_PFBLAS_MAX_M) return 0;
    rocblas_handle h = g_rb[dev];
    if (rocblas_set_stream(h, s) != rocblas_status_success) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    const rocblas_status st = rocblas_sgemm_strided_batched(h,
        rocblas_operation_none, rocblas_operation_none, Q27_PFBLAS_D, R, b,
        &alpha, v, Q27_PFBLAS_D, (rocblas_stride)Q27_PFBLAS_B * Q27_PFBLAS_D,
        p, Q27_PFBLAS_B, (rocblas_stride)R * Q27_PFBLAS_B,
        &beta, u, Q27_PFBLAS_D, (rocblas_stride)R * Q27_PFBLAS_D, Q27_PFBLAS_H);
    if (st != rocblas_status_success) std::fprintf(stderr, "Q27_PF_BLAS PV dev=%d b=%d R=%d status=%d\n", dev, b, R, (int)st);
    return st == rocblas_status_success;
}
static rocblas_status q27_rb_call(rocblas_handle h, const signed char* W, int N, int K, int GS, const signed char* X, int M, int* C, int sol, int lda = 0) {
    const int alpha = 1, beta = 0;
    // lda == 0 keeps the historical behaviour (stride = the logical K). A caller that hands us a
    // COLUMN RANGE of a wider resident row -- the K-view case -- passes the full physical row
    // stride instead, which is the whole point: the resident full-K tensor serves both phases with
    // no duplicate copy.
    const int lw = lda ? lda : K;
    return rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, N, M, GS, &alpha,
            W, rocblas_datatype_i8_r, lw, X, rocblas_datatype_i8_r, K, &beta,
            C, rocblas_datatype_i32_r, N, C, rocblas_datatype_i32_r, N,
            rocblas_datatype_i32_r, sol >= 0 ? rocblas_gemm_algo_solution_index : rocblas_gemm_algo_standard,
            sol >= 0 ? sol : 0, rocblas_gemm_flags_pack_int8x4);
}
// C[g][M][N] (int32) = X[M][g*GS .. +GS] * W[N][g*GS .. +GS]^T for g in [0, K/GS); slab stride M*N
extern "C" int q27_rb_gemm_i8g(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, int GS, hipStream_t s) {
    return q27_rb_gemm_i8g_ld(dev, W, N, K, X, M, C, GS, 0, s);
}
// K-VIEW VARIANT. Same GEMM, but the weight rows may be a COLUMN RANGE of a wider resident tensor:
// ldw is the physical row stride in elements (the resident full K), K the logical shard length, and
// W already offset to the shard's first column. Used to let decode read the layer-split residency
// in place instead of keeping a second, K-sharded copy of the same weights.
extern "C" int q27_rb_gemm_i8g_ld(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, int GS, int ldw, hipStream_t s) {
    if (dev < 0 || dev >= Q27_RB_MAXDEV || !g_rb[dev] || !W || !X || !C || N < 1 || K < 4 || M < 1 || GS < 4 || (K % GS) || (GS & 3) || (N & 3)) return 0;
    rocblas_handle h = g_rb[dev];
    if (rocblas_set_stream(h, s) != rocblas_status_success) return 0;
    const int NG = K / GS;
    const int sol = (NG == 1) ? q27_rb_lookup(dev, N, K, M) : -1;
    for (int g = 0; g < NG; ++g) {
        rocblas_status st = q27_rb_call(h, W + (size_t)g * GS, N, K, GS, X + (size_t)g * GS, M, C + (size_t)g * M * N, sol, ldw);
        if (st != rocblas_status_success && sol >= 0) st = q27_rb_call(h, W + (size_t)g * GS, N, K, GS, X + (size_t)g * GS, M, C + (size_t)g * M * N, -1, ldw);
        if (st != rocblas_status_success) { std::fprintf(stderr, "q27_rb_gemm_i8g: rocblas_gemm_ex status %d (N=%d K=%d GS=%d M=%d g=%d)\n", (int)st, N, K, GS, M, g); return 0; }
    }
    return 1;
}
// Tune (N, K, M) once per device: enumerate the Tensile solutions, time each (3 reps, min), keep the fastest.
// DESCRIPTOR FORM OF THE SAME GEMM: takes a q27_i8g_t that may be a K-VIEW (ldw/lds non-zero), so a
// caller can hand the consumer a column range of a wider resident tensor without materialising it.
extern "C" int q27_rb_gemm_i8g_v(int dev, const q27_i8g_t* w, const signed char* X, int M, int* C, int GS, hipStream_t s) {
    if (!w || !w->w) return 0;
    return q27_rb_gemm_i8g_ld(dev, w->w, w->rows, w->K, X, M, C, GS, w->ldw, s);
}
// Returns the chosen index (-1 = rocBLAS default). Also serves as the priming call for the shape.
extern "C" int q27_rb_tune_i8g(int dev, const signed char* W, int N, int K, const signed char* X, int M, int* C, hipStream_t s) {
    if (dev < 0 || dev >= Q27_RB_MAXDEV || !g_rb[dev] || !W || !X || !C || N < 1 || K < 4 || M < 1 || (K & 3) || (N & 3)) return -1;
    { std::lock_guard<std::mutex> lk(g_sol_mu); for (const auto& e : g_sols) if (e.dev == dev && e.N == N && e.K == K && e.M == M) return e.sol; }
    rocblas_handle h = g_rb[dev];
    if (rocblas_set_stream(h, s) != rocblas_status_success) return -1;
    static const bool tune = q27_env_flag("Q27_RB_TUNE", true);
    static const int reps = q27_env_int("Q27_RB_TUNE_REPS", 2);
    const int alpha = 1, beta = 0;
    rocblas_int nsol = 0;
    std::vector<rocblas_int> list;
    if (tune) {
        rocblas_status st = rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &alpha,
                W, rocblas_datatype_i8_r, K, X, rocblas_datatype_i8_r, K, &beta, C, rocblas_datatype_i32_r, N, C, rocblas_datatype_i32_r, N,
                rocblas_datatype_i32_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_pack_int8x4, nullptr, &nsol);
        if (st == rocblas_status_success && nsol > 0) {
            list.assign((size_t)nsol, 0);
            st = rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, N, M, K, &alpha,
                W, rocblas_datatype_i8_r, K, X, rocblas_datatype_i8_r, K, &beta, C, rocblas_datatype_i32_r, N, C, rocblas_datatype_i32_r, N,
                rocblas_datatype_i32_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_pack_int8x4, list.data(), &nsol);
            if (st != rocblas_status_success) { list.clear(); nsol = 0; }
        } else nsol = 0;
    }
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    auto time_sol = [&](int sol) -> float {   // ms per call, min over reps; -1 if the solution fails
        if (q27_rb_call(h, W, N, K, K, X, M, C, sol) != rocblas_status_success) return -1.f;   // warm
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            hipEventRecord(e0, s);
            if (q27_rb_call(h, W, N, K, K, X, M, C, sol) != rocblas_status_success) return -1.f;
            hipEventRecord(e1, s); hipEventSynchronize(e1);
            float ms = 0.f; hipEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
        }
        return best;
    };
    const float ms_def = time_sol(-1);
    int best = -1; float ms_best = ms_def;
    for (rocblas_int i = 0; i < nsol; ++i) {
        const float ms = time_sol((int)list[(size_t)i]);
        if (ms > 0.f && ms < ms_best * 0.995f) { ms_best = ms; best = (int)list[(size_t)i]; }
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    { std::lock_guard<std::mutex> lk(g_sol_mu); g_sols.push_back({dev, N, K, M, best, ms_best, ms_def}); }
    if (q27_env_flag("Q27_RB_TUNE_LOG", false))
        std::fprintf(stderr, "Q27_RB_TUNE dev=%d N=%d K=%d M=%d: %d solutions, default %.3f ms (%.1f TMAC/s) -> sol %d %.3f ms (%.1f TMAC/s)\n",
                     dev, N, K, M, (int)nsol, (double)ms_def, (double)N * K * M / ms_def / 1e9, best, (double)ms_best, (double)N * K * M / ms_best / 1e9);
    return best;
}
