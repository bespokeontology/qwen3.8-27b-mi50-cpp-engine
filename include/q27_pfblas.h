#pragma once
#include "q27.h"
#include <stddef.h>

// One fixed, bounded FP32 prefill geometry; persistent KV and decode are unchanged.
enum { Q27_PFBLAS_B = 4096, Q27_PFBLAS_D = 256, Q27_PFBLAS_H = 4,
       Q27_PFBLAS_G = 6, Q27_PFBLAS_MAX_M = 256 };
struct q27_pfblas_scratch {
    int M, R, rows;
    size_t bytes;
    float *q, *k, *v, *p, *u, *o;
    float *m, *l, *bm, *bl;
};

extern "C" {
// Layout refusal is CPU-only and writes no arena bytes. Base must be 256-byte aligned.
size_t q27_pfblas_scratch_bytes(int M);
int q27_pfblas_layout(void* arena, size_t bytes, int M, q27_pfblas_scratch* out);
int q27_pfblas_enabled();
int q27_pfblas_force();
int q27_pfblas_should_run(int pos0, int M);
void q27_pfblas_log_config();
unsigned long long q27_pfblas_calls(int dev);

// Stage entry points permit a standalone test to inspect/poison boundaries using
// the actual implementation, with no output-capture hooks in the product path.
int q27_pfblas_pack(const q27_pfblas_scratch* a, const float* q, hipStream_t s);
int q27_pfblas_expand(const q27_pfblas_scratch* a,
                     const signed char* kc, const q27_kvs_t* ks,
                     const signed char* vc, const q27_kvs_t* vs,
                     int k0, int b, hipStream_t s);
int q27_pfblas_qk(int dev, const q27_pfblas_scratch* a, int b, hipStream_t s);
int q27_pfblas_softmax(const q27_pfblas_scratch* a, int pos0, int k0, int b, hipStream_t s);
int q27_pfblas_pv(int dev, const q27_pfblas_scratch* a, int b, hipStream_t s);
int q27_pfblas_merge(const q27_pfblas_scratch* a, hipStream_t s);

// 0 = refused before enqueue, 1 = enqueued successfully, -1 = execution/API failure.
// Retains rotated O and denominator l in the descriptor for the normal epilogue.
int q27_pfblas_run(int dev, const float* q,
                  const signed char* kc, const q27_kvs_t* ks,
                  const signed char* vc, const q27_kvs_t* vs,
                  int pos0, int M, void* arena, size_t bytes, hipStream_t s);
int q27_pfblas_finish(const q27_pfblas_scratch* a,
                     const signed char* raw, const float* raw_sc, int qstride,
                     signed char* out, float* out_sc, int ostride,
                     float in_scale, int g64, hipStream_t s);

// LS-only opt-in wrapper: unchanged original prep followed by bounded attention.
// Caller falls back only for 0, and fails the request for -1.
int q27_attn_chunk_q8_pfblas(int dev, int serialized,
                    const signed char* raw, const float* raw_sc, int qstride,
                    const unsigned short* q_norm, const unsigned short* k_norm,
                    signed char* kc, q27_kvs_t* ks, signed char* vc, q27_kvs_t* vs,
                    float* q, int ostride, int pos0, int M,
                    signed char* out, float* out_sc, float in_scale,
                    void* arena, size_t bytes, int g64, hipStream_t s,
                    const q27_kvquad_t* quad);
// Narrow rocBLAS wrappers; use the existing per-device handle and selected stream.
int q27_rb_pfblas_qk(int dev, const float* k, const float* q, float* p, int b, int R, hipStream_t s);
int q27_rb_pfblas_pv(int dev, const float* v, const float* p, float* u, int b, int R, hipStream_t s);
}
