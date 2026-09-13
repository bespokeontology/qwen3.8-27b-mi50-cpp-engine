// q27_df2_run.h — DFlash2 conditioning + KV injection runner (milestone 4).
// Per round, per committed row r: g = hidden_norm(fc([tap5,tap19,tap33,tap47,tap61])) ; then per
// drafter layer: k/v = k/v_proj(g), k_norm + RoPE + int8-cache quantize into the persistent KV.
#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>
#include "q27.h"
#include "q27_dflash2.h"

// 4-card reduce of fp32 partial rows (the round's collective machinery, via callback)
typedef int (*q27_df2_reduce_fn)(void* u, int g, const float* own, float* red, float* ssp, int NR, hipStream_t s);

struct q27_df2_ctx_t {
    q27_df2_t W;                       // this card's sharded weights + arena
    signed char* tap_xq = nullptr;     // [8][25600] int8 per-16
    float*       tap_xs = nullptr;     // [8][1600]
    float*       g_part = nullptr;     // [8][1280] fp32 fc shard partials
    float*       g_full = nullptr;     // [8][5120] fp32 after the 4-card reduce
    float*       ssp = nullptr;        // [8] sumsq scratch for the reduce
    unsigned short* g_bf16 = nullptr;  // [8][5120]
    signed char* g_xq = nullptr;       // [8][5120] int8 per-16
    float*       g_xs = nullptr;       // [8][320]
    unsigned short* kvbf = nullptr;    // [8][2048] k|v rows (kvrstride 2048)
    signed char* kc[5] = {0}; float* ks[5] = {0}; signed char* vc[5] = {0}; float* vs[5] = {0};   // [2048][1024] int8 + [2048][64] fp32
    // block-executor scratch
    unsigned short* hidden = nullptr;   // [8][5120]
    unsigned short* norm = nullptr;     // [8][5120]
    unsigned short* dyn = nullptr;      // [8][1280]
    unsigned short* conv = nullptr;     // [8][5120]
    signed char* n_xq = nullptr; float* n_xs = nullptr;   // [8][5120] / [8][320]
    signed char* c_xq = nullptr; float* c_xs = nullptr;   // [8][5120] / [8][320]
    unsigned short* qbf = nullptr;      // [8][4096]
    unsigned short* kbf = nullptr;      // [8][1024]
    unsigned short* vbf = nullptr;      // [8][1024]
    unsigned short* qp = nullptr;       // [8][4096]
    unsigned short* kp = nullptr;       // [8][1024]
    unsigned short* attn = nullptr;     // [8][4096]
    signed char* a_xq = nullptr; float* a_xs = nullptr;   // [8][4096] / [8][256]
    unsigned short* mixer = nullptr;    // [8][5120]
    unsigned short* mixer2 = nullptr;   // [8][5120] diagnostic scratch (register-path o result)
    float* pa = nullptr;                // [8][4352] gate shard fp32
    float* pb = nullptr;                // [8][4352] up shard fp32
    signed char* sw_xq = nullptr; float* sw_xs = nullptr; // [8][4352] / [8][272]
    float* down_part = nullptr;         // [8][5120]
    float* mlp_full = nullptr;          // [8][5120]
    unsigned short* anchor = nullptr;   // [5120]
    unsigned short* mask = nullptr;     // [5120]
    // candidates + selector
    float* logits = nullptr;            // [7][62080] lm_head shard logits
    float* ws_val = nullptr; unsigned* ws_id = nullptr;   // [7][32][16] top-16 stage-1 workspace
    unsigned* cand_ids = nullptr; float* cand_val = nullptr;   // [7][16] local top-16
    unsigned short* sh = nullptr;       // [7][256] selector hidden
    unsigned* path = nullptr;           // [7] selected ids
    unsigned* cand_h = nullptr; float* candv_h = nullptr;      // pinned host exchange staging [7][16]
    int ctx_begin = 0, ctx_end = 0;
    size_t bytes = 0;
};
// candidates + selector: lm_head shard logits -> local top-16 -> exchange callback -> card-0 merge + walk.
// On return, path[0..6] = the 7 selected draft ids (valid on all cards if the exchange broadcasts them).
typedef void (*q27_df2_exchange_fn)(void* u, int g, const unsigned* ids, const float* vals, int NR);
int q27_df2_select(q27_df2_ctx_t* C, const q27_nvfp4_t* lmhead, unsigned anchor, int g,
                   q27_df2_exchange_fn exch, void* uctx, hipStream_t s, unsigned* out7, char* err, size_t errcap);
extern "C" void q27_k_df2_tapadd(const float* acc, const unsigned short* hid, unsigned short* tap, int n, hipStream_t s);
extern "C" void q27_k_df2_tapadd_f(const float* a, const float* b, unsigned short* tap, int n, hipStream_t s);
extern "C" void q27_k_df2_f2bfv(const float* src, unsigned short* dst, int n, hipStream_t s);
int q27_df2_forward(q27_df2_ctx_t* C, int pos, q27_df2_reduce_fn reduce, void* uctx, int g,
                    hipStream_t s, float* csum, char* err, size_t errcap);   // full 8-row block forward (anchor/mask must be preloaded into C->anchor/C->mask)

// allocate the ctx buffers for a card (call after q27_df2_upload_sharded for that card)
int q27_df2_ctx_alloc(q27_df2_ctx_t* C, int dev, char* err, size_t errcap);
// one round of conditioning + KV injection for rows 0..nacc-1 at positions pos..pos+nacc-1.
// taps: [5][8][HID] bf16 as captured into Dev::nr_taps. Returns 0 on success.
int q27_df2_prime(q27_df2_ctx_t* C, hipStream_t s);
int q27_df2_cond_inject(q27_df2_ctx_t* C, const unsigned short* taps, int tap_stride, int nacc, int pos, int g,
                        q27_df2_reduce_fn reduce, void* uctx, hipStream_t s, char* err, size_t errcap);
