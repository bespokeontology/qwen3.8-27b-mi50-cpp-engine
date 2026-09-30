// q27_exl3.h — EXL3 trellis weights for the Qwen3.8-27B native gfx906 engine.
//
// ONE resident representation per projection: the packed trellis, plus the two
// fp16 Hadamard scale vectors. Nothing is dequantized at load time and nothing
// is converted during decode -- the weight bytes the kernel reads ARE the
// checkpoint bytes. That is the entire point: at K=4 a projection costs
// in*out/2 bytes, at K=6 in*out*3/4, versus in*out*2 for fp16.
//
// FORWARD CONTRACT (exllamav3 pin c5d9c657, exl3_ext/quant/hadamard.cu:88-139
// + linear.py): an EXL3 linear is NOT just a GEMM against decoded weights --
// the quantizer rotated the space, so the runtime must rotate the same way:
//
//      x' = H128( x (*) suh )        pre-scale, THEN blockwise Hadamard
//      y' = x' @ W_inner             W_inner decoded from the trellis
//      y  = H128( y' ) (*) svh       blockwise Hadamard, THEN post-scale
//
// H128 is blockwise over each contiguous 128 elements (had_r_128), so every
// Qwen3.8-27B dimension works with no padding: 1024, 5120=40*128,
// 6144=48*128, 10240=80*128, 12288=96*128, 17408=136*128.
// Exactly one of pre/post per Hadamard call -- the authority's dispatch law.
//
// K IS PER TENSOR, NOT PER STORE. Both published stores put the lm_head at
// K=6; the "4bpw" store's LAYERS are K=4 and the "6bpw" store's are K=6. The
// loader therefore derives K from each trellis's own trailing dim
// (K = shape[2]*16/256) and never from a global setting.
#pragma once
#include <cstdint>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// A device-resident EXL3 projection. trellis==nullptr means "not present", and
// that is the flag every dispatcher keys off -- the same convention the FP8 and
// MXFP4 paths already use (a null w leaves the incumbent path untouched).
struct q27_exl3_t {
    const uint16_t* trellis = nullptr;   // (in/16, out/16, 256*K/16) u16
    const __half*   suh     = nullptr;   // [in]   fp16 pre-scale
    const __half*   svh     = nullptr;   // [out]  fp16 post-scale
    int in = 0, out = 0, K = 0;
    int mcg = 0, mul1 = 0;               // codebook selector (see cb())
    // reconstruct.cu:127-129 -- mul1 selects codebook 2, NOT 0. Getting this
    // wrong yields a tile wrong in 2046/2048 elements that still looks like
    // plausible weights; it cost a debugging round on 2026-09-19.
    int  cb()   const { return (K - 1) + (mcg ? 8 : (mul1 ? 16 : 0)); }
    bool live() const { return trellis && suh && svh && in > 0 && out > 0 && K > 0; }
    size_t trellis_bytes() const { return (size_t)(in / 16) * (out / 16) * (256 * K / 16) * 2; }
    size_t bytes() const { return trellis_bytes() + (size_t)(in + out) * 2; }
};

// ---------------------------------------------------------------------------
// Store: the mmapped EXL3 safetensors directory. Any shard count (the main
// loader is hardcoded to the NVFP4 checkpoint's three; these stores have two).
struct q27_exl3_store_t;

int  q27_exl3_open(const char* dir, q27_exl3_store_t** out, char* err, size_t errcap);
void q27_exl3_close(q27_exl3_store_t* S);

// Upload one projection, named by its EXL3 prefix (e.g.
// "model.language_model.layers.7.mlp.gate_proj"). Allocates on the CURRENT
// device; K, in and out come from the checkpoint's own shapes.
int  q27_exl3_upload(q27_exl3_store_t* S, const char* prefix, q27_exl3_t* w,
                     char* err, size_t errcap);

// Column slice [n0, n0+n) of the output dim -- the trellis's second axis, so a
// TP column shard is a contiguous row range of cells per k block. n0 and n must
// be multiples of 128 (the Hadamard block, and the kernel's column tile).
int  q27_exl3_upload_cols(q27_exl3_store_t* S, const char* prefix, int n0, int n,
                          q27_exl3_t* w, char* err, size_t errcap);

int  q27_exl3_upload_shard_col(q27_exl3_store_t* S, const char* prefix, const long long* rbase,
                               const long long* rlen, int nrange, int g, int ndev,
                               q27_exl3_t* w, char* err, size_t errcap);
int  q27_exl3_upload_shard_row(q27_exl3_store_t* S, const char* prefix, int g, int ndev,
                               q27_exl3_t* w, char* err, size_t errcap);
void q27_exl3_free(q27_exl3_t* w);
size_t q27_exl3_store_bytes(const q27_exl3_store_t* S);   // host mmap footprint

// ---------------------------------------------------------------------------
// Execution. The Hadamard stages need scratch: in*M halves for x' and out*M
// floats for the fp32 accumulator. q27_exl3_ws_bytes gives the size for the
// largest projection the caller will run; one arena serves every layer.
// FIXED workspace layout. The regions are sized at the widest projection
// (Q27_EXL3_WMAX) rather than per-call, so every projection's x' / accumulator /
// activation land at the SAME offsets. Sizing them per-projection is what makes
// the MLP's activation scratch (written by swiglu, read by down) overlap
// down_proj's own accumulator -- a silent corruption, not a crash.
#define Q27_EXL3_WMAX 17408
size_t q27_exl3_ws_bytes(int max_in, int max_out, int max_M);
unsigned short* q27_exl3_ws_act(void* ws, int in, int M, int out_acc);
unsigned short* q27_exl3_ws_norm(void* ws, int M);
bool q27_exl3_rmsnorm_bf16(const float* x, const unsigned short* w, unsigned short* y,
                           int n, int rows, int ldx, int ldy, hipStream_t s);
bool q27_exl3_add_res_f32(float* part, const float* res, float share,
                          int n, int rows, int ldp, int ldr, hipStream_t s);

// y[M][out] fp16 = H128(H128(x (*) suh) @ W_inner) (*) svh
// x may alias nothing in ws. ws must be q27_exl3_ws_bytes-sized and 16B aligned.
bool q27_exl3_linear(__half* y, const __half* x, const q27_exl3_t* w, int M,
                     void* ws, hipStream_t s);

// Same, fp32 output (the engine's projection consumers take fp32 partials).
bool q27_exl3_linear_f32(float* y, const __half* x, const q27_exl3_t* w, int M,
                         void* ws, hipStream_t s);

// The Hadamard stages on their own (exposed for fusion experiments + the
// load-time self check).
hipError_t q27_exl3_had128_fp16(__half* y, const __half* x, const __half* pre_scale,
                                const __half* post_scale, int n, int rows, hipStream_t s, int lin = 0, int lout = 0);
hipError_t q27_exl3_had128_f32in(__half* y, const float* x, const __half* post_scale,
                                 int n, int rows, hipStream_t s, int lin = 0, int lout = 0);

// ENGINE-FACING: same call shape as q27_proj_fp8_bf16 / q27_proj_fp8, but the
// input is the bf16 NORM VECTOR, not a quantized activation -- EXL3 rotates the
// input itself (suh + H128), so there is nothing to pre-quantize and no
// in_scale to carry. y is the engine's bf16 activation or fp32 partial.
// ldx / ldy are the caller's ROW STRIDES (0 = dense). The engine's multi-row
// prefill buffers are strided -- q_proj writes into a QS-wide qkva row, o_proj
// reads a MIXL-wide mixer row -- so a dense-only kernel would silently read and
// write the wrong rows.
bool q27_exl3_proj_bf16(const q27_exl3_t* w, const unsigned short* x, unsigned short* y,
                        int M, void* ws, hipStream_t s, int ldx = 0, int ldy = 0);
bool q27_exl3_proj_f32 (const q27_exl3_t* w, const unsigned short* x, float* y,
                        int M, void* ws, hipStream_t s, int ldx = 0, int ldy = 0);
bool q27_exl3_add_res(float* part, const unsigned short* res, float share,
                      int n, int rows, int ldp, int ldr, hipStream_t s);
bool q27_exl3_swiglu_bf16(const float* a, const float* b, unsigned short* o,
                          int n, int rows, int lda, int ldo, hipStream_t s);
hipError_t q27_exl3_had128_pre_bf16(__half* y, const unsigned short* x, const __half* pre_scale,
                                    int n, int rows, hipStream_t s, int lin = 0, int lout = 0);
hipError_t q27_exl3_had128_post_bf16(unsigned short* y, const float* x, const __half* post_scale,
                                     int n, int rows, hipStream_t s, int lin = 0, int lout = 0);
hipError_t q27_exl3_had128_post_f32(float* y, const float* x, const __half* post_scale,
                                    int n, int rows, hipStream_t s, int lin = 0, int lout = 0);

// Fused decode + M-band GEMM: y[M][out] fp32 += x[M][in] @ W_inner.
// y MUST be pre-zeroed (split-k accumulation is atomicAdd). MBAND is chosen
// internally from M; the weight tile is decoded ONCE per block for all M rows,
// which is the whole reason prefill does not cost M weight sweeps.
bool q27_exl3_gemm(float* y, const __half* x, const q27_exl3_t* w, int M, hipStream_t s,
                   int ldx = 0, int ldy = 0);
