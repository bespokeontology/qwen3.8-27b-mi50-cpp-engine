// Q27_SPEC: row-batched decode forms for the speculative verify step (NR rows of one sequence).
#ifndef Q27_NR_H
#define Q27_NR_H
#include <hip/hip_runtime.h>
#include "q27.h"
extern "C" {
int q27_proj_fp8_bf16_nr(const q27_fp8_t* w, const signed char* xq, const float* xs,
                         unsigned short* y, int ystride, int NR, hipStream_t s);
// Block-scaled FP8 MLP (Qwen3.8-27B-FP8): weight scale is a [rows/128][K/128] BF16 plane, shard-local.
int q27_proj_fp8b_gu_nr(const q27_fp8_t* g, const q27_fp8_t* u,
                        const signed char* xq, const float* xs, float* yg, float* yu,
                        int ystride, int NR, hipStream_t stream);
int q27_proj_fp8b_nr(const q27_fp8_t* w, const signed char* xq, const float* xs,
                     float* y, int ystride, const unsigned short* radd, int rstride,
                     float rs, int NR, hipStream_t s);
int q27_proj_fp8_res_nr(const q27_fp8_t* w, const signed char* xq, const float* xs,
                        float* y, int ystride, const unsigned short* radd, int rstride,
                        float rs, int NR, hipStream_t s);
int q27_proj_nvfp4_nr(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                      float* y, int ystride, int NR, hipStream_t s);
// Q27_E4M3_FAST: scan this tensor's resident e4m3 group-scale plane and set w->gs_fast=1 only if
// every byte is 8 <= b <= 126. Returns the flag. Fails closed (0) on any error.
int q27_nvfp4_gs_validate(q27_nvfp4_t* w, hipStream_t s);
int q27_proj_nvfp4_gu_nr(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                         const signed char* xq_perm, const float* xs,
                         float* yg, float* yu, int ystride, int NR, hipStream_t s);
// Direct K-widened arm for component A/B; trunk dispatch is Q27_GU_WIDE inside gu_nr.
int q27_proj_nvfp4_gu_wide(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                         const signed char* xq_perm, const float* xs,
                         float* yg, float* yu, int ystride, int NR, hipStream_t s);
// Q27_GU_ILEAVE: interleaved gate/up (WPERM-style). Pack is one-time at load.
int q27_gu_ileave_pack(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                       unsigned char* gu, unsigned char* gs, hipStream_t s);
int q27_proj_nvfp4_gu_ileave(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                             const unsigned char* gu, const unsigned char* gs,
                             const signed char* xq_perm, const float* xs,
                             float* yg, float* yu, int ystride, int NR, hipStream_t s);
// Q27_GU_SWILU: GEMV writes silu(g)*u into yg only.
int q27_proj_nvfp4_gu_swilu(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                            const signed char* xq_perm, const float* xs,
                            float* yg, int ystride, int NR, hipStream_t s);
int q27_quant_group4(const float* y, signed char* xq, float* xs, int n, float in_scale, int C, hipStream_t s);
// Q27_QUANT_DOWN: quant_group4 + down GEMV in one launch (TP3 down shards).
int q27_quant_down(const float* y, signed char* xq, float* xs, int n, float in_scale,
                   const q27_nvfp4_t* w, const unsigned* xp, const float* xsc,
                   float* out, int ystride, const unsigned short* radd,
                   int rstride, float rs, int xst, hipStream_t s);
// Q27_GU_SWILU_Q: GEMV+SwiGLU+quant in one launch (NR=1).
int q27_gu_swilu_q16(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                     const signed char* xq_perm, const float* xs,
                     signed char* xq, float* xss, int n, float in_scale, hipStream_t s);
// Q27_GU_INT4: offline/load-time E2M1->int4 requant + linear-decode GEMV.
int q27_nvfp4_to_int4(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                      unsigned char* wg4, float* sg4, unsigned char* wu4, float* su4, hipStream_t s);
// Q27_GU_E8M0: one-time e4m3->e8m0 scale pack (per-32). Sets gs_mx for MX=1 kernels.
int q27_nvfp4_to_e8m0(const q27_nvfp4_t* w, unsigned char* s8, hipStream_t s);
int q27_proj_int4_gu_nr(const unsigned char* wg4, const float* sg4,
                        const unsigned char* wu4, const float* su4,
                        const signed char* xq_perm, const float* xs,
                        float* yg, float* yu, int rows, int K,
                        float ag, float au, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_down_nr(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                           float* y, int ystride, const unsigned short* radd, int rstride,
                           float rs, int NR, hipStream_t s);
int q27_proj_fp8_bf16_lds(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y, int ystride, int NR, hipStream_t s);
int q27_proj_fp8_res_lds(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s);
int q27_proj_nvfp4_lds(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_gu_lds(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_down_lds(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s);
int q27_proj_fp8_bf16_hg(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y, int ystride, int NR, hipStream_t s);
int q27_proj_fp8_res_hg(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s);
int q27_proj_nvfp4_hg(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_gu_hg(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, const float* xs, float* yg, float* yu, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_down_hg(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s);
int q27_proj_fp8_bf16_nrs(const q27_fp8_t* w, const signed char* xq, int xst, const float* xs, unsigned short* y, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_gu_nrs(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm, int xst, const float* xs, float* yg, float* yu, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_down_nrs(const q27_nvfp4_t* w, const signed char* xq_perm, int xst, const float* xs, float* y, int ystride, const unsigned short* radd, int rstride, float rs, int NR, hipStream_t s);
int q27_rmsnorm_hostss_fp8_b(const float* x, const unsigned short* wgt, unsigned short* hid,
                             unsigned short* y, signed char* xq, float* xs, int n, int plus_one,
                             float in_scale, const float* invp, int C, hipStream_t s);
int q27_rmsnorm_hostss_perm_b(const float* x, const unsigned short* wgt, unsigned short* hid,
                              unsigned short* y, signed char* xq, float* xs, int n, int plus_one,
                              float in_scale, const float* invp, int C, hipStream_t s);
void q27_copy_f4_flag_mb(float* dst_dev, const float* src, int n, unsigned* done_dev, unsigned gen,
                         unsigned* cnt_dev, unsigned target, int nblk, hipStream_t s);
void q27_copy_bf16_flag_mb(unsigned short* dst_dev, const float* src, int n, unsigned* done_dev, unsigned gen,
                           unsigned* cnt_dev, unsigned target, int nblk, hipStream_t s);   // bf16-packed payload (NR collective)
int  q27_red_rn_nr_p2p(int mode, const float* p0, const float* p1, const float* p2, const float* p3, float* red, float* ssp,
                      const unsigned short* wgt, unsigned short* hid, unsigned short* y, signed char* xq, float* xs,
                      int n, int plus_one, float in_scale, int C, hipStream_t s);   // Q27_NR_P2P: peer-read reduce (+ norm/quant) of NR rows; mode 0 input norm, 1 post norm perm, 2 reduce only
int  q27_fp8_make_i8g(const q27_fp8_t* w, hipStream_t s);          // Q27_DEC_I8: build + register the int8 per-64 mirror of an fp8 projection (side table by weight pointer)
int  q27_df2_gemm(const void* m, const signed char* xq, int xst, const float* xs, void* y, int ystride,
                  const unsigned short* radd, int rstride, float rs, int NR, int outbf, hipStream_t s);   // DFlash2 drafter GEMM (same dispatch)
int  q27_fp8_set_i8g(const q27_fp8_t* w, const q27_i8g_t* m);   // register an EXISTING i8g (or a view into one) as this handle's mirror
int  q27_fp8_i8g_rekey(const void* oldw, const void* neww);   // Q27_DEC_I8: move a mirror's side-table key (fp8 freed -> sentinel)
const q27_i8g_t* q27_fp8_i8g(const q27_fp8_t* w);                  // the mirror or NULL
void q27_fp8_freed_set(int v);      // Q27_DEC_I8: mark the layer fp8 arena freed (consumers MUST route to mirrors)
int  q27_fp8_freed(void);           // 1 once the layer fp8 arena is freed
int  q27_push_bf16_3(const float* src, unsigned short* d0, unsigned short* d1, unsigned short* d2, int n, hipStream_t s);   // Q27_NR_P2P v2: push bf16 rows to three peers
int  q27_red_rn_nr_local(int mode, const float* own, int g, const unsigned short* ra, const unsigned short* rb, const unsigned short* rc,
                        float* red, float* ssp, const unsigned short* wgt, unsigned short* hid, unsigned short* y, signed char* xq, float* xs,
                        int n, int plus_one, float in_scale, int C, hipStream_t s);   // Q27_NR_P2P v2: reduce own + three received rows locally (+ norm)
void q27_ll_hint_arm(unsigned* stage_words, unsigned seq);   // Q27_LL_EPI: arm the producer-epilogue LL pack for the next site
int  q27_ll_hint_take(unsigned** p, unsigned* seq);          // (kernels' host wrappers) consume the armed hint
int  q27_ll_hint_done(void);                                 // 1 if a producer took the hint (no pack kernel needed); clears it
int  q27_proj_i8g_bf16_seg(const q27_fp8_t* const* ws, unsigned short* const* ys, const int* ystrides, int nseg, const signed char* xq, const float* xs, int NR, hipStream_t s);   // Q27_PROJ_SEG: q/k/v or in_qkv/in_z in one launch
int  q27_pack_ll(const float* src, unsigned* dst, int n, int C, unsigned seq, hipStream_t s);   // Q27_LL_SDMA: pack C rows of LL words into a local staging block
int  q27_push_ll(const float* src, unsigned* d0, unsigned* d1, unsigned* d2, int n, int C, unsigned seq, hipStream_t s);   // Q27_NR_P2P=4: LL push of C rows (bf16 pairs + seq per 8-byte word) into 2-3 peers' slots
int  q27_red_rn_nr_ll(int mode, const float* own, int g, int ndev, const unsigned* ra, const unsigned* rb, const unsigned* rc,
                     unsigned seq, float* red, float* ssp, const unsigned short* wgt, unsigned short* hid, unsigned short* y,
                     signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, unsigned* err, unsigned* cnt, hipStream_t s);   // Q27_NR_P2P=4: spin on seq, reduce own + received (+ norm); cnt = Q27_LL_LB row counters
int  q27_red_rn_nr_ll_trunk(int mode, const float* own, int g, int ndev, const unsigned* ra, const unsigned* rb, const unsigned* rc,
                     unsigned seq, float* red, float* ssp, const unsigned short* wgt, unsigned short* hid, unsigned short* y,
                     signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, unsigned* err, unsigned* cnt, hipStream_t s);
int  q27_ll_probe_wait(const unsigned* slot, unsigned seq, unsigned* out, hipStream_t s);    // LL self-test: spinning reader (out[0]=data|~0, out[1]=stale seq, out[2]=spins)
int  q27_ll_probe_write(unsigned* slot, unsigned data, unsigned seq, hipStream_t s);          // LL self-test: single-word peer writer
// Native trunk opt-in; all other callers retain q27_swiglu_quant_b.
int q27_swiglu_quant_group4(const float* gate, const float* up, signed char* xq, float* xs,
                          int n, float in_scale, int C, hipStream_t s);
int q27_swiglu_quant_trunk(const float* gate, const float* up, signed char* xq, float* xs,
                         int n, float in_scale, int C, hipStream_t s);
int q27_swiglu_quant_b(const float* gate, const float* up, signed char* xq, float* xs,
                       int n, float in_scale, int C, hipStream_t s);
void q27_quant_fp8_b(const unsigned short* x_bf16, size_t xstride, signed char* xq, size_t qstride,
                     float* xs, size_t sstride, int n, float in_scale, int C, hipStream_t s);
int  q27_bf16_gemv2_tile(const unsigned short* wa, const unsigned short* wb, const unsigned short* xt, int xstride,
                         unsigned short* ya, unsigned short* yb, int rows, int K, int Ct, hipStream_t s);
void q27_gdn_conv_tp_tile2_r(unsigned short* qkv, int qstride, int Ct, unsigned short* conv_state, int sfull,
                          const unsigned short* cw, int kh0, int khn, unsigned short* snap, size_t snap_stride, int snap_all, hipStream_t s);
// Q27_GDN_AB_CONV: fused a/b GEMV + conv in one launch.
void q27_gdn_ab_conv_tp_tile2_r(const unsigned short* wa, const unsigned short* wb,
                                const unsigned short* xt, int xstride,
                                unsigned short* ya, unsigned short* yb, int ab_rows, int ab_K,
                                unsigned short* qkv, int qstride, int Ct, unsigned short* conv_state, int sfull,
                                const unsigned short* cw, int kh0, int khn, unsigned short* snap, size_t snap_stride, int snap_all, hipStream_t s);
void q27_gdn_scan_tp_snap_r(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                     const unsigned short* a_t, const unsigned short* b_t, int abstride,
                     const unsigned short* A_log, const unsigned short* dt_bias,
                     const unsigned short* norm_w, float* S,
                     unsigned short* out_t, int ostride, int kh0, int khn, int Ct,
                     signed char* q_t, float* qs_t, float in_scale,
                     float* Ssnap, size_t snap_stride, int snap_all, hipStream_t s);
void q27_gdn_conv_tp_tile2(unsigned short* qkv, int qstride, int Ct, unsigned short* conv_state, int sfull,
                           const unsigned short* cw, int g, int ndev, unsigned short* snap, size_t snap_stride, int snap_all, hipStream_t s);
void q27_gdn_scan_tp_snap(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                          const unsigned short* a_t, const unsigned short* b_t, int abstride,
                          const unsigned short* A_log, const unsigned short* dt_bias, const unsigned short* norm_w, float* S,
                          unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                          signed char* q_t, float* qs_t, float in_scale, float* Ssnap, size_t snap_stride, int snap_all, hipStream_t s);
void q27_attn_prep_tp_nr(const unsigned short* qkv_out, int qkvstride, const unsigned short* q_norm_w, const unsigned short* k_norm_w,
                         signed char* kcache, q27_kvs_t* kscale, signed char* vcache, q27_kvs_t* vscale, unsigned short* q_out, int qoutstride,
                         int pos0, int nr, int ndev, int kvstride, int hoff0, hipStream_t s);
int q27_attn_decode_tp_nr(const unsigned short* q, int qstride, const signed char* kcache, const q27_kvs_t* kscale,
                           const signed char* vcache, const q27_kvs_t* vscale, const unsigned short* q_proj_raw, int qrawstride,
                           unsigned short* out, int ostride, int pos0, int nr, int ndev, int kvstride, int hoff0, hipStream_t s,
                           signed char* Q = nullptr, float* QS = nullptr, float in_scale = 0.f, int qost = 0);
void q27_attn_kvprep_nr(const unsigned short* kv_rows, int kvrstride, const unsigned short* k_norm_w, signed char* kcache, q27_kvs_t* kscale,
                        signed char* vcache, q27_kvs_t* vscale, int pos0, int nr, hipStream_t s);
int  q27_epi_f32r_ld(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha, float* y, int ldy, int M, hipStream_t s);
}
#endif
