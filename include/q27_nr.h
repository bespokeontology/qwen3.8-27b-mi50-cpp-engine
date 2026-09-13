// Q27_SPEC: row-batched decode forms for the speculative verify step (NR rows of one sequence).
#ifndef Q27_NR_H
#define Q27_NR_H
#include <hip/hip_runtime.h>
#include "q27.h"
extern "C" {
int q27_proj_fp8_bf16_nr(const q27_fp8_t* w, const signed char* xq, const float* xs,
                         unsigned short* y, int ystride, int NR, hipStream_t s);
int q27_proj_fp8_res_nr(const q27_fp8_t* w, const signed char* xq, const float* xs,
                        float* y, int ystride, const unsigned short* radd, int rstride,
                        float rs, int NR, hipStream_t s);
int q27_proj_nvfp4_nr(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs,
                      float* y, int ystride, int NR, hipStream_t s);
int q27_proj_nvfp4_gu_nr(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu,
                         const signed char* xq_perm, const float* xs,
                         float* yg, float* yu, int ystride, int NR, hipStream_t s);
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
int  q27_fp8_i8g_rekey(const void* oldw, const void* neww);   // Q27_DEC_I8: move a mirror's side-table key (fp8 freed -> sentinel)
const q27_i8g_t* q27_fp8_i8g(const q27_fp8_t* w);                  // the mirror or NULL
void q27_fp8_freed_set(int v);      // Q27_DEC_I8: mark the layer fp8 arena freed (consumers MUST route to mirrors)
int  q27_fp8_freed(void);           // 1 once the layer fp8 arena is freed
int  q27_push_bf16_3(const float* src, unsigned short* d0, unsigned short* d1, unsigned short* d2, int n, hipStream_t s);   // Q27_NR_P2P v2: push bf16 rows to three peers
int  q27_red_rn_nr_local(int mode, const float* own, int g, const unsigned short* ra, const unsigned short* rb, const unsigned short* rc,
                        float* red, float* ssp, const unsigned short* wgt, unsigned short* hid, unsigned short* y, signed char* xq, float* xs,
                        int n, int plus_one, float in_scale, int C, hipStream_t s);   // Q27_NR_P2P v2: reduce own + three received rows locally (+ norm)
int q27_swiglu_quant_b(const float* gate, const float* up, signed char* xq, float* xs,
                       int n, float in_scale, int C, hipStream_t s);
void q27_quant_fp8_b(const unsigned short* x_bf16, size_t xstride, signed char* xq, size_t qstride,
                     float* xs, size_t sstride, int n, float in_scale, int C, hipStream_t s);
int  q27_bf16_gemv2_tile(const unsigned short* wa, const unsigned short* wb, const unsigned short* xt, int xstride,
                         unsigned short* ya, unsigned short* yb, int rows, int K, int Ct, hipStream_t s);
void q27_gdn_conv_tp_tile2(unsigned short* qkv, int qstride, int Ct, unsigned short* conv_state, int sfull,
                           const unsigned short* cw, int g, int ndev, unsigned short* snap, size_t snap_stride, int snap_all, hipStream_t s);
void q27_gdn_scan_tp_snap(const unsigned short* qkv_t, int qstride, const unsigned short* z_t, int zstride,
                          const unsigned short* a_t, const unsigned short* b_t, int abstride,
                          const unsigned short* A_log, const unsigned short* dt_bias, const unsigned short* norm_w, float* S,
                          unsigned short* out_t, int ostride, int g, int ndev, int Ct,
                          signed char* q_t, float* qs_t, float in_scale, float* Ssnap, size_t snap_stride, int snap_all, hipStream_t s);
void q27_attn_prep_tp_nr(const unsigned short* qkv_out, int qkvstride, const unsigned short* q_norm_w, const unsigned short* k_norm_w,
                         signed char* kcache, float* kscale, signed char* vcache, float* vscale, unsigned short* q_out, int qoutstride,
                         int pos0, int nr, int ndev, int kvstride, int hoff0, hipStream_t s);
void q27_attn_decode_tp_nr(const unsigned short* q, int qstride, const signed char* kcache, const float* kscale,
                           const signed char* vcache, const float* vscale, const unsigned short* q_proj_raw, int qrawstride,
                           unsigned short* out, int ostride, int pos0, int nr, int ndev, int kvstride, int hoff0, hipStream_t s);
void q27_attn_kvprep_nr(const unsigned short* kv_rows, int kvrstride, const unsigned short* k_norm_w, signed char* kcache, float* kscale,
                        signed char* vcache, float* vscale, int pos0, int nr, hipStream_t s);
int  q27_epi_f32r_ld(const int* acc, int ldc, int N, const float* srow, int ng, const float* stok, float alpha, float* y, int ldy, int M, hipStream_t s);
}
#endif
