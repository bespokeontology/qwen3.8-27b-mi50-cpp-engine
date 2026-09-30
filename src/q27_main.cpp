
// Qwen3.8-27B native decode engine for gfx906. The layer loop and the multi-card pipeline.
//
// Decomposition: 4-card layer/pipeline split, 16 layers per card, serial for M=1.
// Chosen over tensor parallel for the FIRST build because it needs 3 host crossings per token
// instead of 128, and because it gives clean causal accounting. Both crossing costs are measured:
//   pipeline crossing 0.0147 ms x3     = 0.0441 ms/token
//   4-card all-reduce 0.0735 ms x128   = 9.40   ms/token   (what TP would cost)
// TP is the better single-stream LATENCY design (22.5 vs 52.6 ms projected) and is worth building
// second; the pipeline plus token pipelining is the better THROUGHPUT design, which is the standing
// objective. See the roofline discussion in README.md.
#include <execinfo.h>
#include <dlfcn.h>
static void q27_memory_receipt(const char* label) {
    static auto fn = (void(*)(const char*))dlsym(RTLD_DEFAULT,"q27_memory_report");
    if(fn)fn(label);
}
static void q27_transfer_receipt(int req,bool begin) {
    static auto first = (void(*)(int))dlsym(RTLD_DEFAULT,"q27_xfer_begin");
    static auto last = (void(*)(int))dlsym(RTLD_DEFAULT,"q27_xfer_report");
    if(begin){if(first)first(req);}else{if(last)last(req);}
}

#include <signal.h>
#include <unistd.h>
#include "q27.h"
#include "q27_candidate_flags.h"
#include "q27_pfblas.h"
#include "q27_load.h"
extern "C" void q27_finite_check(const void*,size_t,int,unsigned*,int,hipStream_t);
#include "q27_dflash2.h"
#include "q27_df2_run.h"
#include "q27_nr.h"
#include "q27_sr.h"   // Q27_TPSR: natural-order swiglu quantiser + block-scaled FP8 -> per-row int8 converters (src/q27_sr.hip)
extern "C" int q27_proj_fp8b_nrs(const q27_fp8_t* w, const signed char* xq, int xst, const float* xs,
                                 float* y, int ystride, const unsigned short* radd, int rstride,
                                 float rs, int NR, hipStream_t s);   // q27_fp8.hip: block-scaled consumer with an explicit activation row stride
#include "../include/q27_mtp_quant.h"
#include "../tests/q27_gate.h"
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include "hf_tokenizer.h"
#include "q27_chat.h"
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <random>

#define CK(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ \
    std::fprintf(stderr,"Q27_HIP %s at %s:%d\n",hipGetErrorString(e_),__FILE__,__LINE__); \
    std::exit(1);} }while(0)

// kernels supplied by the other translation units
extern "C" void q27_vgate_read(float* mx, float* ref, int reset);   // attention-combine numerics gate (Q27_VGATE=1)
extern "C" {
void q27_gdn_conv(unsigned short* qkv, unsigned short* conv_state, const unsigned short* cw, hipStream_t s);
void q27_gdn_step(const unsigned short* qkv_c, const unsigned short* z, const unsigned short* a,
                  const unsigned short* b, const unsigned short* A_log, const unsigned short* dt_bias,
                  const unsigned short* norm_w, float* S, unsigned short* out, hipStream_t s);
void q27_attn_prep(const unsigned short* qkv_out, const unsigned short* q_norm_w,
                   const unsigned short* k_norm_w, signed char* kcache, q27_kvs_t* kscale, signed char* vcache, q27_kvs_t* vscale,
                   unsigned short* q_out, int pos, int kvstride, int hoff0, hipStream_t s);
void q27_attn_decode(const unsigned short* q, const signed char* kcache, const q27_kvs_t* kscale,
                     const signed char* vcache, const q27_kvs_t* vscale, const unsigned short* q_proj_raw,
                     unsigned short* out, int pos, int kvstride, int hoff0, hipStream_t s);
// Fused in_proj_a/in_proj_b GEMV: one launch for both, uint2 loads, every load issued before any
// wait. Returns 1 if it ran, 0 if the shape is uncovered so the caller falls back to two scalar
// calls. Reduction order differs; the output is NOT bit-identical, by design.
// Writes BF16 directly, replacing q27_proj_fp8 + q27_f2bf_vec at the five per-layer sites whose
// consumer wants BF16. Returns 1 if it ran; 0 leaves the two-kernel path.
// Fused SwiGLU + permuting quantizer: never materialises the activation buffer.
// Fused gate+up nvfp4 projection: one launch, activation decoded once for both.
// gate+up -> SwiGLU -> permuting quantize as ONE operation. Materialises neither gate/up outputs
// nor the activation buffer.
int  q27_mlp_gu_swiglu_q(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
                         const float* xs, signed char* dq, float* ds, float down_in_scale,
                         hipStream_t s);
int  q27_proj_nvfp4_gu(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
                       const float* xs, float* yg, float* yu, hipStream_t s);
// Small-M prefill MLP (Q27_PF_MLP2): LDS-staged activations, R rows per wave, weights decoded once.
int  q27_proj_nvfp4_gu_b2(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
                          const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);
int  q27_proj_nvfp4_gu_b3(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);   // hoisted-weight MLP (Q27_PF_MLP3)
int  q27_proj_nvfp4_bt3(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int R, hipStream_t s);
int  q27_proj_nvfp4_gu_b14(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int minw, hipStream_t s);   // v13 on the int8 mirror (Q27_PF_W8 + Q27_PF_MLP14 = 5|6)
int  q27_proj_nvfp4_bt14(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int minw, hipStream_t s);
int  q27_proj_nvfp4_gu_b13(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int minw, hipStream_t s);   // v12 + uniform addressing (Q27_PF_MLP13 = 4|5 waves)
int  q27_proj_nvfp4_bt13(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int minw, hipStream_t s);
// q27_wide.hip: the GEMM-order wide FFN (Q27_PF_WIDE). Output-stationary tiles, K in the thread,
// NVFP4 decoded once per tile into LDS. NOT a drop-in for the b13 family in geometry, but it IS one in
// layout: it stages the same 16-byte activation group and pairs it q27_d4(w.x,x.x) (w.y,x.z) (w.z,x.y)
// (w.w,x.w) exactly as b13 does, so it consumes the identical permuted activation buffer and writes
// the identical Y[t*rows+row] epilogue. ki encodes the tile form (see q27_wide.hip):
// 84/88/164/168 = row-major KI8|KI16 x NW4|NW8, 8000/16000 = register-blocked KI8|KI16.
int  q27_wide_gu(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq, const float* xs,
        float* yg, float* yu, int M, int ki, hipStream_t s);
int  q27_wide_down(const q27_nvfp4_t* w, const signed char* xq, const float* xs, float* y,
        int M, const unsigned short* radd, float rs, int ki, hipStream_t s);
int  q27_proj_nvfp4_gu_b12(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int gb, hipStream_t s);   // rebuilt epilogue (Q27_PF_MLP12 = 4|8|16)
int  q27_proj_nvfp4_bt12(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int gb, hipStream_t s);
int  q27_proj_nvfp4_gu_b11(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int depth, hipStream_t s);   // LDS reads pipelined (Q27_PF_MLP11 = depth 2|4|8)
int  q27_proj_nvfp4_bt11(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int depth, hipStream_t s);
int  q27_proj_nvfp4_gu_b10(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);   // weights fetched before the staging (Q27_PF_MLP10)
int  q27_proj_nvfp4_bt10(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int R, hipStream_t s);
int  q27_proj_nvfp4_gu_b9(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);   // weight prefetch one chunk ahead (Q27_PF_MLP9)
int  q27_proj_nvfp4_bt9(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int R, hipStream_t s);
int  q27_proj_nvfp4_gu_b8(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);   // double-buffered staging without register caches (Q27_PF_MLP8)
int  q27_proj_nvfp4_gu_b7(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int rpw, hipStream_t s);   // persistent-row MLP (Q27_PF_MLP7)
int  q27_proj_nvfp4_bt7(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int rpw, hipStream_t s);
int  q27_proj_nvfp4_gu_b6(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);   // two k-chunks per stage (Q27_PF_MLP6)
int  q27_proj_nvfp4_bt6(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int R, hipStream_t s);
int  q27_proj_nvfp4_gu_b5(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);   // no activation register cache (Q27_PF_MLP5)
int  q27_proj_nvfp4_bt5(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int R, hipStream_t s);
int  q27_proj_nvfp4_gu_b4(const q27_nvfp4_t* wg, const q27_nvfp4_t* wu, const signed char* xq_perm,
        const float* xs, float* yg, float* yu, int C, int R, hipStream_t s);   // software-pipelined MLP (Q27_PF_MLP4)
int  q27_proj_nvfp4_bt4(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
        int C, int R, hipStream_t s);
int  q27_proj_nvfp4_bt2(const q27_nvfp4_t* w, const signed char* xq_perm, const float* xs, float* y,
                        int C, int R, hipStream_t s);
// Chunk out_proj: M positions' quantized mixer -> fp32 partials with the residual share folded.
int  q27_proj_fp8_m2_res(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y, int M,
                         int ystride, const unsigned short* radd, int rstride, float rs, hipStream_t s);
// Multi-block rmsnorm+quant: stage A (5 blocks) folds the residual and reduces partial sums of
// squares; stage B (20 blocks) applies the norm and quantizes. Replaces the single-block form.
// Chunk tail (prefill): C positions' host-sumsq norm + permuting quantize in one launch.
int  q27_rmsnorm_hostss_perm_b(const float* x, const unsigned short* wgt, unsigned short* hid,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const float* invp, int C, hipStream_t s);
// Q27_PF_P2P: GPU-resident cross-card collective. The fused reduce+rmsnorm+perm-quant tail
// reads this card's partial and three PEER partials in one kernel; the plain reduce assembles
// the mixer and writes 1/rms. Both replace the D2H/host-reduce/H2D round trip.
int  q27_red_rn_perm_b(const float* p0, const float* p1, const float* p2, const float* p3,
        const unsigned short* wgt, unsigned short* hid, signed char* xq, float* xs, int n,
        int plus_one, float in_scale, int C, hipStream_t s);
int  q27_red_b(const float* p0, const float* p1, const float* p2, const float* p3,
        float* out, float* inv_out, int n, int C, hipStream_t s);
int  q27_p2p_probe(float* dst, float val, hipStream_t s);
// Q27_LAYER_SPLIT: single-card fused tails (sumsq + 1/rms on device, rmsnorm+quant, or the
// mixer assemble into pend_slots). No peer sources.
int  q27_red_rn_perm_b1(const float* x, const unsigned short* wgt, unsigned short* hid, float* hid32,
        signed char* xq, float* xs, int n, int plus_one, float in_scale, int C, hipStream_t s);
int  q27_red_b1(const float* x, float* out, float* inv_out, int n, int C, hipStream_t s);
int  q27_fp8_to_h2(const q27_fp8_t* w, void* w16, hipStream_t s);
int  q27_proj_fp8_m2r2(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y,
        int M, int ystride, int pos0, int R, hipStream_t s);   // m2r, reduction epilogue rebuilt (Q27_PF_MK2R2)
int  q27_proj_fp8_m2r(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y,
        int M, int ystride, int pos0, int R, hipStream_t s);   // tile projection, R rows/wave, no LDS (Q27_PF_MK2R)
int  q27_proj_fp8_m2w16(const q27_fp8_t* w, const void* w16, const signed char* xq, const float* xs, unsigned short* y,
        int M, int ystride, int pos0, hipStream_t s);   // tile projection on pre-converted fp16 weights (Q27_PF_MK16)
int  q27_proj_fp8_m3l(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y,
        int M, int ystride, int pos0, int R, hipStream_t s);   // tile projection v3 (LDS-staged, R rows/wave)
int  q27_proj_fp8_m2_res3(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y, int M,
        int ystride, const unsigned short* radd, int rstride, float rs, int R, hipStream_t s);   // m2f2, epilogue rebuilt (Q27_PF_M2F3)
int  q27_proj_fp8_m2_res2(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y, int M,
        int ystride, const unsigned short* radd, int rstride, float rs, int R, hipStream_t s);   // R rows/wave, no LDS
int  q27_proj_fp8_m3_res(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y, int M,
        int ystride, const unsigned short* radd, int rstride, float rs, int R, hipStream_t s);
int  q27_rmsnorm_hostss_fp8_b(const float* x, const unsigned short* wgt, unsigned short* hid,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const float* invp, int C, hipStream_t s);   // tile input norm (GDN layers), one launch
int  q27_rmsnorm_quant_split_fp8(const unsigned short* x, const unsigned short* wgt,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const void* add, int abf, float* scratch, hipStream_t s);
int  q27_rmsnorm_quant_split_perm(const unsigned short* x, const unsigned short* wgt,
        unsigned short* y, signed char* xq, float* xs, int n, int plus_one, float in_scale,
        const void* add, int abf, float* scratch, hipStream_t s);
int  q27_swiglu_quant(const float* gate, const float* up, signed char* xq, float* xs,
                      int n, float in_scale, hipStream_t s);
int  q27_swiglu_quant_b(const float* gate, const float* up, signed char* xq, float* xs,
                        int n, float in_scale, int C, hipStream_t s);   // C positions, one launch
int  q27_proj_fp8_bf16(const q27_fp8_t* w, const signed char* xq, const float* xs,
                       unsigned short* y, hipStream_t s);
// k_proj and v_proj in ONE launch (256+256 rows/card under 4-way TP), one alpha per matrix.
// Returns 1 if it ran. neg=1 is the firing control: perturbs matrix B's alpha, MUST change tokens.
int  q27_proj_fp8_s1_b3(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y,
                        int C, int ystride, int R, hipStream_t s);   // s1_b2, epilogue rebuilt (Q27_PF_QKV3)
int  q27_proj_fp8_s1_b2(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y,
                        int C, int ystride, int R, hipStream_t s);   // small-M q/k/v (prefill)
// q27_fp8.hip: the WIDE fp8 projection GEMM (Q27_PF_X4W). MMQ's execution law adapted to fp8:
// 128 rows x 64 tokens per block, grid over BOTH rows and tokens (M in the grid, one launch per
// matrix per chunk), K in-thread, int8->half2 conversion hoisted to LDS staging instead of being
// redone per (row, k-step) as the m2 family does. Float epilogue with optional bf16 residual, and a
// bf16-output twin for consumers that expect the fp8 path's BF16 shape.
int  q27_proj_fp8_wide(const q27_fp8_t* w, const signed char* xq, const float* xs, float* y,
                       int M, int ystride, const unsigned short* radd, int rstride, float rs,
                       int ki, hipStream_t s);
int  q27_proj_fp8_wideb(const q27_fp8_t* w, const signed char* xq, const float* xs, unsigned short* y,
                        int M, int ystride, int ki, hipStream_t s);
int  q27_proj_fp8_bf16_2(const q27_fp8_t* wa, const q27_fp8_t* wb, const signed char* xq,
                         const float* xs, unsigned short* ya, unsigned short* yb, int neg,
                         hipStream_t s);
int  q27_attn_chunk_tp(const unsigned short* qkv_t, int qstride, const unsigned short* q_norm_w, const unsigned short* k_norm_w,
        signed char* kcache, q27_kvs_t* kscale, signed char* vcache, q27_kvs_t* vscale, float* qout_t, unsigned short* out_t, int ostride,
        int pos0, int M, int ndev, signed char* Q_t, float* QS_t, float in_scale, int hpw, int attpf,
        float* pob, float* pml, hipStream_t s, const q27_kvquad_t* quad = nullptr);   // chunk attention
size_t q27_attn_chunk_scratch_bytes(int ndev, int chunk_max, size_t* ml_bytes);
int q27_wide_down_b(const q27_nvfp4_t* w, const signed char* xq, const float* xs,
                    unsigned short* y, int ystride, int M, int ki, hipStream_t s);
int  q27_bf16_gemv2_tile(const unsigned short* wa, const unsigned short* wb, const unsigned short* xt, int xstride,
        unsigned short* ya, unsigned short* yb, int rows, int K, int Ct, hipStream_t s);   // tile form (GDN scan)
int  q27_bf16_gemv2(const unsigned short* wa, const unsigned short* wb, const unsigned short* x,
                    unsigned short* ya, unsigned short* yb, int rows, int K, hipStream_t s);
void q27_bf16_gemv(const unsigned short* w, const unsigned short* x, unsigned short* y,
                   int rows, int K, hipStream_t s);   // in_proj_a / in_proj_b
// In-stream staging copy: n floats from a device buffer into pinned host memory through its
// device alias, on the compute queue (replaces the SDMA hipMemcpyAsync D2H when Q27_COLL_KCOPY=1).
void q27_copy_f4(float* dst_dev, const float* src, int n, hipStream_t s);
void q27_copy_f4_flag(float* dst_dev, const float* src, int n, unsigned* done_dev, unsigned gen,
                      hipStream_t s);
void q27_f2bf_vec(const float* src, unsigned short* dst, int n, hipStream_t s);
}

// ---------------- per-device scratch ----------------
// Q27_LS_Q8: one chunk's scratch. Two sets per card (chunk parity) so two chunks can be in flight on
// two streams; set 0 aliases the single-stream buffers below, set 1 is its own allocation.
struct Q8Scr {
    signed char* xqin = nullptr; float* xsin = nullptr;
    signed char* qkva8 = nullptr; float* qkvas = nullptr;
    float* qh = nullptr; float* pob = nullptr; float* pml = nullptr;
    size_t pob_bytes = 0;              // Actual allocation capacity for bounded alternate attention.
    signed char* mixq_slots = nullptr; float* mixs_slots = nullptr;
    signed char* qkv8 = nullptr; float* qkvs = nullptr; signed char* z8 = nullptr; float* zs = nullptr;
    float* a32 = nullptr; float* b32 = nullptr;
    signed char* mixq_tile = nullptr; float* mixs_tile = nullptr;
    float* part = nullptr; signed char* xq1 = nullptr; float* xs1 = nullptr; float* pa = nullptr; float* pb = nullptr;
    signed char* xq2 = nullptr; float* xs2 = nullptr; float* mixer = nullptr; float* hid32c = nullptr;
    int* acc = nullptr;                    // rocBLAS int32 GEMM output [M][N] (up to 2 x INTER columns)
    signed char* xqt = nullptr; float* xst = nullptr;     // per-token int8 input activations + scales
    float* xst1 = nullptr; float* xst2 = nullptr;          // per-token scales of xq1 (post-norm) and xq2 (swiglu)
    signed char* mixq_t = nullptr; float* mixs_t = nullptr; // per-token requant of the attention / GDN outputs
    signed char* qkvc8 = nullptr; float* qkvcs = nullptr;   // conv output (position-parallel conv is out of place)
};
static std::vector<hipEvent_t> g_ls_evdone[Q27_MAX_DEVICES];   // per card: chunk-k done events (Q27_LS_Q8 schedule)
// 2026-09-18 DIAGNOSTIC: the layer-split sweep dies here with a NULL/garbage hipEvent_t
// (dmesg: "segfault at 10/18 ... in libamdhip64", i.e. a bad handle dereferenced INSIDE HIP,
// not a bad return address). g_ls_evdone[src] is populated by card `src`'s OWN thread; this
// read is from card g. Name the offender instead of faulting: an empty peer vector is a
// missing barrier, an out-of-range pos is a jobpos/njobs disagreement, and a null entry is a
// failed hipEventCreate. All three are silent today.
#define Q27_LS_EVCHK(tag, card, idx) \
    do { const size_t _n = g_ls_evdone[(card)].size(); \
         if ((size_t)(idx) >= _n || g_ls_evdone[(card)][(size_t)(idx)] == nullptr) { \
             std::fprintf(stderr, "Q27_LS_EVBAD %s: card %d reading peer %d event %d of %zu%s\n", \
                          tag, g, (card), (int)(idx), _n, \
                          ((size_t)(idx) < _n) ? " -- entry is NULL" : " -- INDEX OUT OF RANGE"); \
             std::fflush(stderr); std::exit(1); } } while (0)
// rows of a group of int8 mirrors when they are ONE contiguous buffer (stacked at load), else 0
static int q27_cat_rows(const q27_i8r_t* const* w, int n) {
    if (!w || n < 1 || !w[0] || !w[0]->w) return 0;
    int tot = w[0]->rows; const signed char* next = w[0]->w + (size_t)w[0]->rows * w[0]->K;
    for (int i = 1; i < n; ++i) { if (!w[i] || w[i]->w != next || w[i]->K != w[0]->K || w[i]->gs != w[0]->gs) return 0; tot += w[i]->rows; next += (size_t)w[i]->rows * w[i]->K; }
    return tot;
}
struct Dev {
    int id = -1;
    hipStream_t stream = nullptr;
    hipStream_t hs[4] = {nullptr, nullptr, nullptr, nullptr};   // Q27_PF_HSTREAMS: the 8-position halves of a chunk on concurrent streams
    hipStream_t q8hi = nullptr, q8lo = nullptr;   // Q27_LS_Q8 schedule: stage A (latency-critical chain) high priority, MLP low priority
    hipEvent_t ev_state_ready = nullptr;
    hipEvent_t  ev_fork = nullptr, ev_join[4] = {nullptr, nullptr, nullptr, nullptr};
    unsigned short* hidden = nullptr;      // [5120] BF16, the residual stream
    unsigned short* norm   = nullptr;      // [5120] BF16
    float*  mixer  = nullptr;              // [5120] f32
    signed char* xq = nullptr;             // [17408] int8 activation (max width)
    float*  xs     = nullptr;              // [1088]  per-16 activation scales (max width)
    float*  pa     = nullptr;              // [17408] f32
    float*  pb     = nullptr;              // [17408] f32
    unsigned short* act = nullptr;         // [17408] BF16 (swiglu output)
    // gated delta
    unsigned short* qkv = nullptr;         // [10240] BF16
    unsigned short* zbuf= nullptr;         // [6144]  BF16
    unsigned short* ab  = nullptr;         // [48]    BF16
    unsigned short* bb  = nullptr;         // [48]    BF16
    unsigned short* mix6= nullptr;         // [6144]  BF16, mixer output before out_proj
    // Q27_EXL3 scratch: [ x' fp16 M*in ][ fp32 accumulator M*out ]. Grown on
    // demand rather than sized in dev_alloc_tp, because the EXL3 residency is
    // attached AFTER dev_alloc runs -- a value latched at alloc time would be
    // read before it was written (the failure mode that made the first rewind
    // arm inert on 2026-09-19).
    void*  exws = nullptr;
    size_t exws_bytes = 0;
    // The head is 248320 wide -- 14x the MLP intermediate -- so it gets its own
    // buffer rather than inflating the per-projection one.
    void*  exws_head = nullptr;
    size_t exws_head_bytes = 0;
    // PER-SLOT state. A serial layer pipeline leaves 3 of 4 cards idle at every instant; the only
    // way to use the machine is to keep several sequences in flight, which makes the recurrent
    // state, the conv shift register, the KV cache and the residual all per-slot.
    float*  S      = nullptr;              // [slots][nlayer_gdn][48][128][128] f32
    unsigned short* conv = nullptr;        // [slots][nlayer_gdn][10240][4] BF16
    // full attention
    unsigned short* qkva = nullptr;        // [14336] BF16 = q 12288 | k 1024 | v 1024
    unsigned short* qh   = nullptr;        // [6144]  BF16 normed+roped query
    signed char* kc = nullptr;             // [slots][nlayer_full][ctx][1024] int8 (Q27_LS_Q8 KV ABI)
    signed char* vc = nullptr;
    q27_kvs_t* ks = nullptr;               // [slots][nlayer_full][ctx][1024/16] FP16 group maxima
    q27_kvs_t* vs = nullptr;               // (KV scale ABI: mx/127 is reconstructed on read)
    void* pinned = nullptr;                // crossing staging
    unsigned short* hidden_slots = nullptr;// [slots][5120] BF16 residual, one per sequence
    unsigned short* emb_pin = nullptr; size_t emb_pin_cap = 0;   // pinned host staging for the sweep embedding (allocated at init: lazy pinning inside the sweep cost ~30 ms of barrier skew)
    void* pin_out = nullptr;               // [slots][5120] BF16 pinned, this card's output
    unsigned* dtok = nullptr;              // argmax scratch, per card (no per-token hipMalloc)
    int slots = 1;
    size_t s_stride = 0, conv_stride = 0, kv_stride = 0;   // elements per slot
    // ---- tensor-parallel only (Q27_TP=1); untouched by the serial and pipeline paths ----
    float* dval = nullptr;                 // winning logit value, for the 4-way head max
    unsigned* gdn_scr = nullptr;           // gdn d-split cross-block scratch: 12 heads x 12 words
    float* pend_slots = nullptr;           // [Q27_PF_CAP][Q27_HID] per-position pending residuals
    float* pend_stage = nullptr;           // pinned fp32 staging for the non-zc pending hop
    float* inv_slots = nullptr;            // pinned [Q27_PF_CAP] per-position hostss 1/rms
    signed char* xq_slots = nullptr;       // [8][Q27_HID] batched-final activations
    float* xs_slots = nullptr;             // [8][Q27_HID/16] batched-final activation scales
    float* pa_slots = nullptr;             // [8][Q27_VOCAB/4] batched lm_head outputs
    signed char* xq1_slots = nullptr;      // [8][Q27_HID] batched post-norm activations (MLP)
    float* xs1_slots = nullptr;            // [8][Q27_HID/16]
    signed char* xq2_slots = nullptr;      // [8][Q27_INTER/ndev] batched swiglu activations
    float* xs2_slots = nullptr;            // [8][(Q27_INTER/ndev)/16]
    signed char* mixq_slots = nullptr;     // [8][Q27_GDN_Z/ndev] chunk mixer quantized (Q27_PF_OPROJ_B)
    float* mixs_slots = nullptr;           // [8][(Q27_GDN_Z/ndev)/16]
    float* pa_mlp = nullptr, * pb_mlp = nullptr;   // [8][Q27_INTER/ndev] batched gate/up outputs
    float* mixer_slots = nullptr;          // [8][Q27_HID] batched down outputs (collective #2)
    float* mix2 = nullptr;                 // split-K partials 2..4 (layer-split)
    float* mix3 = nullptr;
    float* mix4 = nullptr;
    float* qkv2a = nullptr; float* qkv2b = nullptr;   // split-K fp32 partials (in_qkv)
    float* zb2a  = nullptr; float* zb2b  = nullptr;   // split-K fp32 partials (in_z)
    float* op2a  = nullptr; float* op2b  = nullptr;   // split-K fp32 partials (out_proj)
// Q27_PF_WIDE: the TILE-WIDE MLP chain. Separate Q27_PF_TSLOT-sized buffers, allocated only when the
// flag is on: the chunk structures above stay exactly the rung-3 size, which is the whole point of the
// change. Layout is the wide kernels' native one ([position][row]), which is also b13's.
signed char* xq1_t = nullptr;          // [TSLOT][Q27_HID]            tile MLP input (post-norm, permuted)
float*       xs1_t = nullptr;          // [TSLOT][Q27_HID/16]         its per-16 scales
float*       pa_t  = nullptr;          // [TSLOT][Q27_INTER/ndev]     gate output
float*       pb_t  = nullptr;          // [TSLOT][Q27_INTER/ndev]     up output
signed char* xq2_t = nullptr;          // [TSLOT][Q27_INTER/ndev]     swiglu-quantized down input
float*       xs2_t = nullptr;          // [TSLOT][(Q27_INTER/ndev)/16]
float*       mix_t = nullptr;          // [TSLOT][Q27_HID]            down output -> collective #2
    signed char* xqin_slots = nullptr;    // [8][Q27_HID] batched input-norm activations (fp8 natural)
    float* xsin_slots = nullptr;          // [8][Q27_HID/16]
    unsigned short* qkva_slots = nullptr; // [8][QL+2*KVL] batched q/k/v outputs (bf16)
    unsigned short* norm_slots = nullptr;  // [8][Q27_HID] batched input-norm outputs (bf16)
    unsigned short* qkv_slots = nullptr;   // [8][QKV/ndev] batched GDN in_qkv outputs (bf16)
    unsigned short* zbuf_slots = nullptr;  // [8][Z/ndev] batched GDN in_z outputs (bf16)
    float* part_slots = nullptr;           // [8][Q27_HID] per-position partials (batched coll#1)
    float* qh_slots = nullptr;             // [chw][NQ*HDIM] chunk attention queries, fp32 (Q27_PF_ATT_CHUNK)
    float* attn_pob = nullptr;             // chunk attention split partials [CH][splits][NQ][HDIM] (per card, hipMalloc)
    size_t attn_pob_bytes = 0;
    float* attn_pml = nullptr;             // [CH][splits][NQ][2]
    void* w16_scratch = nullptr;           // [2560][5120] fp16 pairs: one projection's weights, pre-converted per tile (Q27_PF_MK16)
    unsigned short* ab_tile = nullptr;     // [TSLOT][VH/ndev] GDN a, tile scan (Q27_PF_GDN_SCAN)
    unsigned short* bb_tile = nullptr;     // [TSLOT][VH/ndev] GDN b
    unsigned short* mix_tile = nullptr;    // [TSLOT][Z/ndev] scan mixer (bf16)
    signed char* mixq_tile = nullptr;      // [TSLOT][Z/ndev] scan mixer, fused-quantized
    float* mixs_tile = nullptr;            // [TSLOT][(Z/ndev)/16]
    float* rnpart = nullptr;      // 5-float partial sums for the split rmsnorm
    size_t tp_s = 0, tp_conv = 0, tp_kv = 0;               // PER-LAYER strides of the shard state
    // ---- PER-LAYER STATE PLACEMENT (what lets layer-split reach ctx=262144) ------------------
    // Layer-split allocated EVERY layer full width on EVERY card (dev_alloc_tp is called with
    // ndev=1) and then broadcast each owner's whole row to all four. That is 4x redundant by
    // construction: a card computes full-width state only for the layers it OWNS -- its prefill
    // attention reads every head of those -- and for every other layer the ONLY reader is TP
    // decode, which addresses exactly one head-quarter (kvstride/hoff0). So non-owned layers are
    // stored quarter-width and the export copies just that quarter. At ctx=262144 that is 4.70 GB
    // of KV per card instead of 10.74 GB, which is the difference between fitting and not.
    // In TP mode these are filled with the flat shard layout (off = fs*tp_kv, stride = KVROWS/ndev,
    // hoff = 0), i.e. addressing identical to before this existed.
    size_t kv_off[Q27_LAYERS] = {0};       // element offset of full-attn layer fs within kc / vc
    size_t kvs_off[Q27_LAYERS] = {0};      // float offset of that layer within ks / vs
    int    kv_rowstride[Q27_LAYERS] = {0}; // KV row stride in elements for that layer
    int    kv_hoff[Q27_LAYERS] = {0};      // this card's head offset inside its OWN stored row
    int    kv_hoff_peer[Q27_LAYERS] = {0};// and inside the OWNER's full-width row (export source)
    size_t kv_bytes[Q27_LAYERS] = {0};     // bytes actually stored (export size)
    size_t kvs_bytes[Q27_LAYERS] = {0};
    size_t s_off[Q27_LAYERS] = {0};        // float offset of GDN layer gs within S
    size_t s_qoff[Q27_LAYERS] = {0};       // this card's quarter offset inside that layer's S
    size_t s_bytes[Q27_LAYERS] = {0};
    size_t conv_off[Q27_LAYERS] = {0};     // element offset of GDN layer gs within conv
    size_t s_bytes_total = 0, conv_bytes_total = 0;   // whole-allocation sizes, for the serve reset
    // ---- Q27_LS_MLP_STAGE: layer-local int8 MLP arena --------------------------------------
    // The fast MLP consumers (q27_wide_gu_i8g64 / the Tensile int8 GEMMs) need int8 per-64
    // weights. Materialising those for all 16 owned layers is 271 MiB x 16 = 4.23 GiB and does
    // not fit at ctx=262144, which is why Q27_LS_MLP_NV falls back to decoding NVFP4 inside every
    // compute tile. But layer-split prefill walks its layers SEQUENTIALLY: only the CURRENT
    // layer's gate/up/down have to exist in int8 at any instant. One reusable 271 MiB arena buys
    // the fast consumer without the residency.
    signed char* mlp8_w = nullptr;      // gate | up | down, int8 [rows][K]
    float*       mlp8_s = nullptr;      // their per-64 fp32 group scales
    // Q27_LS_MLP_RB: the same arena, requantized IN PLACE to the per-ROW form the rocBLAS/Tensile
    // int8 GEMMs want (w [rows][K] unchanged, s replaced by one fp32 scale per row). The banked
    // evidence for this row: the 1263-class engine ran the MLP through int8 mirrors + Tensile; the
    // per-64 arena lost only because its consumer was the bespoke wide i8g64 kernel (2.3x slower),
    // and the per-1024 arena lost because GS=1024 sliced down into 17 K=1024 GEMMs. Per-ROW weights
    // keep K whole: gate/up/down each become ONE int8 GEMM per chunk, which is the fast shape.
    float*       mlp8_rs = nullptr;     // one per-row fp32 scale per row per tensor (tiny)
    q27_i8r_t    mlp8_gate_r{}, mlp8_up_r{}, mlp8_down_r{};
    // FOUR SLOTS, NOT ONE. The Q8 ring walks its layers in blocks of Q27_LS_BLK=4, so a 1-layer
    // arena misses on every revisit: 16 re-expansions per chunk, which is what made the first
    // staging attempt a loss. Keying the arena by (L & 3) converts each layer once per block per
    // chunk instead -- 4 expansions per chunk, ~40 GiB of extra traffic over a whole prefill.
    static constexpr int MLP8_SLOTS = 4;
    int          mlp8_layer[MLP8_SLOTS] = {-1, -1, -1, -1};   // layer held in each slot
    q27_i8g_t    mlp8_gate{}, mlp8_up{}, mlp8_down{};
    size_t conv_qoff[Q27_LAYERS] = {0};
    size_t conv_bytes[Q27_LAYERS] = {0};
    int    conv_wide[Q27_LAYERS] = {0};    // 1 = the stored conv block is full width (this card owns it)
    int mtp_cap = 0;                       // positions the draft KV is currently sized for (Q27_MTP_PAGED)
    signed char* mtp_kc = nullptr;         // MTP draft layer's own KV slice (one tp_kv each)
    q27_kvs_t* mtp_ks = nullptr; q27_kvs_t* mtp_vs = nullptr;
    size_t tp_kvs = 0;                     // per-layer stride of the KV scale planes (= tp_kv/16)
    // ---- Q27_LS_Q8: quantized producer/consumer buffers (full width: the layer-split rank) ----
    signed char* qkva8 = nullptr;          // [chw][QROWS+2*KVROWS] int8 q/k/v producer output
    float* qkvas = nullptr;                // [chw][(QROWS+2*KVROWS)/16]
    signed char* qkv8 = nullptr;           // [chw][GDN_QKV] int8 in_qkv output, re-quantized in place by the conv
    float* qkvs = nullptr;                 // [chw][GDN_QKV/16]
    signed char* z8 = nullptr;             // [chw][GDN_Z] int8 in_z output
    float* zs = nullptr;                   // [chw][GDN_Z/16]
    float* a32 = nullptr; float* b32 = nullptr;   // [chw][48] GDN a/b (fp32)
    float* hid32c = nullptr;               // [chw][HID] fp32 h' of the LAST layer's chunk (the head's residual)
    Q8Scr q8s[2];                          // Q27_LS_Q8 chunk scratch by parity (see Q8Scr)
    signed char* mtp_vc = nullptr;
    // ---- Q27_SPEC: row-batched verify (NR rows of one sequence) + speculative bookkeeping ----
    unsigned short* nr_norm = nullptr;  signed char* nr_xq = nullptr;  float* nr_xs = nullptr;
    unsigned short* nr_qkva = nullptr;  unsigned short* nr_qh = nullptr;
    unsigned short* nr_qkv = nullptr;   unsigned short* nr_zbuf = nullptr;
    unsigned short* nr_ab = nullptr;    unsigned short* nr_bb = nullptr;  unsigned short* nr_mix6 = nullptr;
    float* nr_pa = nullptr;  float* nr_pb = nullptr;  float* nr_mixer = nullptr;
    int*   nr_acc = nullptr;   // Q27_DEC_MLP_I8: int32 accumulator for the decode's rocBLAS MLP GEMMs
    unsigned* nr_tok = nullptr;  float* nr_val = nullptr;   // one 64-B device buffer: tok[8] then val[8]
    float* nr_mail_h = nullptr; float* nr_mail_d = nullptr;   // pinned mailbox (2*NR <= 16 floats) the flag copy lands in
    float* nr_Ssnap = nullptr;  unsigned short* nr_csnap = nullptr;   // [3][n_gdn] state-shard / conv banks (lazy)
    // Q27_REWIND_CK: request-boundary checkpoints of the WHOLE recurrent state, so a divergent
    // window can rewind to a position instead of resetting to 0. Same two regions the reset
    // memsets (d.S, d.conv) and the same D2D copy the speculation banks already use -- the KV
    // cache needs nothing, it is position-indexed and is overwritten by the next prefill.
    float* ck_S = nullptr;  unsigned short* ck_conv = nullptr;   // [q27_rewind_ck()] full (S, conv) banks
    hipStream_t s_cap = nullptr; hipEvent_t ev_cap = nullptr;   // DFlash2 tap capture: side stream, so the capture never stalls the next layer
    unsigned short* nr_taps = nullptr;    // [5][NRM][HID] bf16 residual taps after target layers 5/19/33/47/61 (DFlash2 conditioning)
    unsigned short* df2_pf_taps = nullptr; // [5][cap][HID] bf16 prompt-position taps (prefill capture -> prompt conditioning)
    unsigned short* df2_pf_taps_h = nullptr;   // pinned host mirror for the cross-card gather
    unsigned short* nr_hprev = nullptr;   // [HID] final residual of the last committed position (the draft's hidden)
    unsigned short* nr_hcatch = nullptr;  // [8][HID] hidden rows the draft layer still has to consume
    unsigned short* nr_ebounce = nullptr; int nr_eslot = 0;   // pinned [32][HID] bounce ring: embedding rows go host->pinned->device async
    unsigned nr_catch_tok[8] = {0u,0u,0u,0u,0u,0u,0u,0u}; int nr_catch_pos[8] = {0,0,0,0,0,0,0,0}; int nr_ncatch = 0;
    int pf_released = 0;   // prefill-sweep scratch (hidden_slots/pend_slots/emb_pin/inv_slots) freed at the first decode step; hidden_slots shrinks to the NR row buffer
    int nr_have_h = 0; int nr_mtp_base = 0;  long nr_drafts = 0, nr_hits = 0;  long nr_acc_hist[8] = {0,0,0,0,0,0,0,0};   // rounds with >= j accepted drafts
    double sp_t[8] = {0,0,0,0,0,0,0,0};   // per-phase wall (ms): draft, draft-xchg, embed, layers, head-sync, head-xchg, tail, total
    // ---- Q27_LS_SR: single residency (src/q27_sr_main.inc) ----
    hipEvent_t sr_ev[4] = {nullptr, nullptr, nullptr, nullptr};   // decode hop events: one per block visit per round
    long sr_round = 0;                                            // rounds this card has run (hop-counter base)
    int  sr_gdn_gs[Q27_LAYERS] = {0};  int sr_ngdn = 0;           // global gdn slot of each OWNED gdn layer, in layer order
    int  sr_gdn_local[Q27_LAYERS] = {0};                          // global gdn slot -> local index (-1 if not owned)
    signed char* sr_xqw = nullptr; float* sr_xsw = nullptr;       // [NRM][INTER] int8 + per-16 scales: o/out_proj input, swiglu output
    float* sr_pd[4] = {nullptr, nullptr, nullptr, nullptr};       // [NRM][HID] fp32 down partials, one per K-slice
    float* sr_part = nullptr;  float* sr_part2 = nullptr;         // [NRM][HID] fp32 mixer / MLP outputs (pending residual)
    signed char* sr_pw[2] = {nullptr, nullptr}; float* sr_prs[2] = {nullptr, nullptr}; int sr_player[2] = {-1, -1};   // prefill ring: per-row projection mirrors
    signed char* sr_mw[2] = {nullptr, nullptr}; float* sr_ms[2] = {nullptr, nullptr}; float* sr_mrs[2] = {nullptr, nullptr}; int sr_mlayer[2] = {-1, -1};   // prefill ring: MLP per-64 (+ per-row scales)
    q27_layer_t sr_view[2];                                       // the layer handle the sweep reads: resident fields + this slot's mirrors
    hipEvent_t sr_te[12] = {nullptr};                             // Q27_SR_TIME: timing events -- [v] block start, [8+v] block end, [4]/[5] draft, [6]/[7] head
    double sr_ms_blk[4] = {0, 0, 0, 0}; double sr_ms_draft = 0, sr_ms_head = 0, sr_ms_wall = 0; long sr_rounds_t = 0;
    hipEvent_t sr_ste[16 * 7] = {nullptr};                        // Q27_SR_TIME=2: per owned layer (local index 0..15), 7 stage boundaries
    double sr_ms_st[6] = {0, 0, 0, 0, 0, 0};                      // norm+in-proj, attention/GDN, out-proj, post-norm+gate/up, swiglu+down, sum/add
    int  sr_lidx = 0;                                             // local layer counter inside a round (0..15)
    // ---- Q27_TPSR: transient per-ROW int8 MLP shard (gate|up rows, down K-slice) for the batched TP prefill's rocBLAS GEMMs ----
    signed char* tpsr_mw = nullptr; float* tpsr_ms = nullptr;     // one layer: [2*INTER/ndev][HID] + [HID][INTER/ndev] int8, per-row fp32 scales
    float* tpsr_nv_s64 = nullptr; // reusable donor NVFP4 -> int8/64 -> row conversion scales
    int tpsr_mlayer = -1; int tpsr_primed = 0;                    // layer currently expanded; rocBLAS shapes tuned once
    q27_i8r_t tpsr_g8r{}, tpsr_u8r{}, tpsr_d8r{};                 // the per-row views the GEMMs read
    // ---- Q27_TPSR_PROJ_RB (prefill v2): transient per-ROW int8 projection shards for the rocBLAS GEMMs.
    // Attention: q|k|v stacked [3584][HID] + o [HID][1536]. GDN: in_qkv|in_z stacked [4096][HID] + out [HID][1536].
    signed char* tpsr_pw = nullptr; float* tpsr_ps = nullptr;
    int tpsr_pmlayer = -1;
    q27_i8r_t tpsr_q8r{}, tpsr_k8r{}, tpsr_v8r{}, tpsr_o8r{}, tpsr_iqkv8r{}, tpsr_iz8r{}, tpsr_op8r{};
};

static void dev_alloc(Dev& d, int id, int n_gdn, int n_full, int ctx, int slots) {
    d.id = id; d.slots = slots;
    CK(hipSetDevice(id));
    CK(hipStreamCreate(&d.stream));
    for (int k = 0; k < 4; ++k) { CK(hipStreamCreateWithFlags(&d.hs[k], hipStreamNonBlocking)); CK(hipEventCreateWithFlags(&d.ev_join[k], hipEventDisableTiming)); }
    CK(hipEventCreateWithFlags(&d.ev_fork, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&d.ev_state_ready, hipEventDisableTiming));
    CK(hipStreamCreateWithFlags(&d.s_cap, hipStreamNonBlocking)); CK(hipEventCreateWithFlags(&d.ev_cap, hipEventDisableTiming));
    { int lo = 0, hi = 0; CK(hipDeviceGetStreamPriorityRange(&lo, &hi));
      CK(hipStreamCreateWithPriority(&d.q8hi, hipStreamNonBlocking, hi));
      CK(hipStreamCreateWithPriority(&d.q8lo, hipStreamNonBlocking, lo)); }
    if (!q27_rb_init(d.id)) { std::fprintf(stderr, "rocBLAS init failed on device %d\n", d.id); std::exit(1); }
    auto A=[&](void** p, size_t b){ CK(hipMalloc(p,b)); CK(hipMemset(*p,0,b)); };
    A((void**)&d.hidden, Q27_HID*2);      A((void**)&d.norm, Q27_HID*2);
    A((void**)&d.mixer,  Q27_HID*4);      A((void**)&d.xq, Q27_INTER);
    A((void**)&d.xs, (Q27_INTER/16)*4);
    // pa/pb must ALSO hold the lm_head logits: 248320 floats, 14x wider than the MLP intermediate.
    // Sizing them at Q27_INTER overflowed the allocation by 923,648 bytes on every single token.
    // It appeared to work because the overflow landed in the next hipMalloc arena and argmax read
    // it straight back; the defect surfaced only when a hipMemcpy of the true logit width failed
    // and returned an all-zero buffer.
    const size_t PA_ELEMS = (size_t)((Q27_INTER > Q27_VOCAB) ? Q27_INTER : Q27_VOCAB);
    A((void**)&d.pa, PA_ELEMS*4);
    A((void**)&d.pb, PA_ELEMS*4);
    A((void**)&d.act, (size_t)Q27_INTER*2);
    A((void**)&d.qkv, Q27_GDN_QKV*2);     A((void**)&d.zbuf, Q27_GDN_Z*2);
    A((void**)&d.ab, 48*2);               A((void**)&d.bb, 48*2);
    A((void**)&d.mix6, Q27_GDN_Z*2);
    A((void**)&d.qkva, (Q27_QROWS+2*Q27_KVROWS)*2);  A((void**)&d.qh, Q27_OROWS*2);
    d.s_stride    = (size_t)n_gdn*Q27_GDN_VH*Q27_GDN_D*Q27_GDN_D;
    d.conv_stride = (size_t)n_gdn*Q27_GDN_QKV*Q27_GDN_CONV;
    d.kv_stride   = (size_t)n_full*ctx*Q27_KVROWS;
    if (n_gdn > 0) {
        A((void**)&d.S,    d.s_stride*(size_t)slots*4);
        A((void**)&d.conv, d.conv_stride*(size_t)slots*2);
    }
    if (n_full > 0) {
        A((void**)&d.kc, d.kv_stride*(size_t)slots); A((void**)&d.ks, d.kv_stride/16*(size_t)slots*2);
#if Q27_V4
        A((void**)&d.vc, d.kv_stride/2*(size_t)slots); A((void**)&d.vs, d.kv_stride/32*(size_t)slots*2);
#else
        A((void**)&d.vc, d.kv_stride*(size_t)slots); A((void**)&d.vs, d.kv_stride/16*(size_t)slots*2);
#endif
    }
    A((void**)&d.hidden_slots, (size_t)slots*Q27_HID*2);
    CK(hipHostMalloc(&d.pin_out, (size_t)slots*Q27_HID*2, hipHostMallocDefault));
    A((void**)&d.dtok, 4);
    CK(hipHostMalloc(&d.pinned, Q27_HID*2, hipHostMallocDefault));
}

// MTP draft head: types + declarations, included HERE so run_tp (much further down) can see them.
#include "q27_mtp_defs.h"

// ---------------- one layer ----------------
// Topology, contract §1: pre-norm, exactly two residual adds, no residual scaling.
// ---- per-stage profile. Enabled by Q27_PROFILE=1; adds a sync per stage so it is a RANKING
// instrument, never a timing path. Steady-state only: the caller resets it after prefill.
enum { PF_NORM, PF_QUANT, PF_PROJ_FP8, PF_ATTN, PF_GDN, PF_PROJ_NVFP4, PF_SWIGLU, PF_ADD, PF_N };
static const char* PF_NAME[PF_N] = {"rmsnorm","quant","proj_fp8","attention","gated_delta",
                                    "proj_nvfp4","swiglu","residual"};
static double g_pf[PF_N] = {0};
static int    g_profile  = 0;
static inline double pf_now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                               return t.tv_sec*1e3 + t.tv_nsec/1e6; }
#define PF(bucket, stream, code) do { \
    if (!g_profile) { code; } else { double _t = pf_now(); code; \
        hipStreamSynchronize(stream); g_pf[bucket] += pf_now() - _t; } } while(0)

static const char* g_orc = nullptr;      // oracle dir when the sub-layer bisect is on
static int  g_sub_layer = -1;            // which layer to bisect
static int  g_sub_pos   = -1;

// Bisect INSIDE one layer. The oracle capture carries pre_norm / mixer_out / post_mixer /
// post_norm / post_mlp precisely so a layer-boundary mismatch can be attributed to a stage.
static void sub(Dev& d, const char* tag, const void* buf, int layer, int pos, int is_f32, int n) {
    if (!g_orc || layer != g_sub_layer || pos != g_sub_pos) return;
    hipStreamSynchronize(d.stream);
    std::vector<float> got = is_f32 ? q27_d2h_f32(buf, n) : q27_d2h_bf16(buf, n);
    char nm[64]; std::snprintf(nm, sizeof nm, "%s L%02d P%d", tag, layer, pos);
    q27_gate(nm, got, q27_oracle(g_orc, tag, layer, pos, n), 1e30);   // report, never fail
}

static void run_layer(Dev& d, const q27_layer_t* L, int pos, int gdn_slot, int full_slot, int ctx,
                      int slot) {
    hipStream_t s = d.stream;
    unsigned short* hid = d.hidden_slots + (size_t)slot*Q27_HID;
    // --- mixer ---
    PF(PF_NORM, s, q27_rmsnorm(hid, L->input_norm, d.norm, Q27_HID, 1, s));
    sub(d, "pre_norm", d.norm, L->layer, pos, 0, Q27_HID);
    if (L->is_full) {
        PF(PF_QUANT, s, q27_quant_fp8(d.norm, d.xq, d.xs, Q27_HID, L->q_proj.in_scale, s));
        // q_proj [12288] | k_proj [1024] | v_proj [1024] laid out contiguously, which is the
        // buffer layout q27_attn_prep documents. q/k/v share one input_scale (contract §3), so the
        // single q27_quant_fp8 above feeds all three.
        PF(PF_PROJ_FP8, s, q27_proj_fp8(&L->q_proj, d.xq, d.xs, d.pa, s));
        q27_f2bf_vec(d.pa, d.qkva, Q27_QROWS, s);
        PF(PF_PROJ_FP8, s, q27_proj_fp8(&L->k_proj, d.xq, d.xs, d.pa, s));
        q27_f2bf_vec(d.pa, d.qkva + Q27_QROWS, Q27_KVROWS, s);
        PF(PF_PROJ_FP8, s, q27_proj_fp8(&L->v_proj, d.xq, d.xs, d.pa, s));
        q27_f2bf_vec(d.pa, d.qkva + Q27_QROWS + Q27_KVROWS, Q27_KVROWS, s);
        signed char* kc = d.kc + (size_t)slot*d.kv_stride + (size_t)full_slot*ctx*Q27_KVROWS;
        q27_kvs_t* ks = d.ks + ((size_t)slot*d.kv_stride + (size_t)full_slot*ctx*Q27_KVROWS) / 16;
#if Q27_V4
        // V4 planes are packed: half the bytes per row, one block max per 32 dims.
        signed char* vc = d.vc + ((size_t)slot*d.kv_stride + (size_t)full_slot*ctx*Q27_KVROWS) / 2;
        q27_kvs_t* vs = d.vs + ((size_t)slot*d.kv_stride + (size_t)full_slot*ctx*Q27_KVROWS) / 32;
#else
        signed char* vc = d.vc + (size_t)slot*d.kv_stride + (size_t)full_slot*ctx*Q27_KVROWS;
        q27_kvs_t* vs = d.vs + ((size_t)slot*d.kv_stride + (size_t)full_slot*ctx*Q27_KVROWS) / 16;
#endif
        PF(PF_ATTN, s, { q27_attn_prep(d.qkva, L->q_norm, L->k_norm, kc, ks, vc, vs, d.qh, pos, Q27_KVROWS, 0, s);
                         q27_attn_decode(d.qh, kc, ks, vc, vs, d.qkva, d.mix6, pos, Q27_KVROWS, 0, s); });
        q27_quant_fp8(d.mix6, d.xq, d.xs, Q27_OROWS, L->o_proj.in_scale, s);
        PF(PF_PROJ_FP8, s, q27_proj_fp8(&L->o_proj, d.xq, d.xs, d.mixer, s));
    } else {
        q27_quant_fp8(d.norm, d.xq, d.xs, Q27_HID, L->in_qkv.in_scale, s);
        PF(PF_PROJ_FP8, s, q27_proj_fp8(&L->in_qkv, d.xq, d.xs, d.pa, s));  q27_f2bf_vec(d.pa, d.qkv,  Q27_GDN_QKV, s);
        PF(PF_PROJ_FP8, s, q27_proj_fp8(&L->in_z,   d.xq, d.xs, d.pb, s));  q27_f2bf_vec(d.pb, d.zbuf, Q27_GDN_Z,   s);
        q27_bf16_gemv(L->in_a, d.norm, d.ab, Q27_GDN_VH, Q27_HID, s);
        q27_bf16_gemv(L->in_b, d.norm, d.bb, Q27_GDN_VH, Q27_HID, s);
        PF(PF_GDN, s, { q27_gdn_conv(d.qkv, d.conv + (size_t)slot*d.conv_stride + (size_t)gdn_slot*Q27_GDN_QKV*Q27_GDN_CONV, L->conv1d, s);
                        q27_gdn_step(d.qkv, d.zbuf, d.ab, d.bb, L->A_log, L->dt_bias, L->gdn_norm,
                                     d.S + (size_t)slot*d.s_stride + (size_t)gdn_slot*Q27_GDN_VH*Q27_GDN_D*Q27_GDN_D, d.mix6, s); });
        q27_quant_fp8(d.mix6, d.xq, d.xs, Q27_GDN_Z, L->out_proj.in_scale, s);
        PF(PF_PROJ_FP8, s, q27_proj_fp8(&L->out_proj, d.xq, d.xs, d.mixer, s));
    }
    sub(d, "mixer_out", d.mixer, L->layer, pos, 1, Q27_HID);
    PF(PF_ADD, s, q27_add_inplace(hid, d.mixer, Q27_HID, s));            // residual 1
    sub(d, "post_mixer", hid, L->layer, pos, 0, Q27_HID);

    // --- MLP ---
    PF(PF_NORM, s, q27_rmsnorm(hid, L->post_norm, d.norm, Q27_HID, 1, s));
    sub(d, "post_norm", d.norm, L->layer, pos, 0, Q27_HID);
    // ONE activation quantization serves gate AND up; they share input_scale (contract §1)
    PF(PF_QUANT, s, q27_quant_perm(d.norm, d.xq, d.xs, Q27_HID, L->gate.in_scale, s));
    PF(PF_PROJ_NVFP4, s, { q27_proj_nvfp4(&L->gate, d.xq, d.xs, d.pa, s);
                           q27_proj_nvfp4(&L->up,   d.xq, d.xs, d.pb, s); });
    PF(PF_SWIGLU, s, q27_swiglu(d.pa, d.pb, d.act, Q27_INTER, s));
    PF(PF_QUANT, s, q27_quant_perm(d.act, d.xq, d.xs, Q27_INTER, L->down.in_scale, s));
    PF(PF_PROJ_NVFP4, s, q27_proj_nvfp4(&L->down, d.xq, d.xs, d.mixer, s));
    PF(PF_ADD, s, q27_add_inplace(hid, d.mixer, Q27_HID, s));            // residual 2
}

// ================= TENSOR PARALLEL (C1), behind Q27_TP=1 =================
// Every card runs ALL 64 layers on its own shard of every large matrix. Both RMSNorms are
// computed redundantly on all four cards (a 5120-wide norm is cheaper than the extra collective a
// sharded residual would need), so every card always holds the FULL residual h and the FULL
// normalized n. Two collectives per layer, 128 per token, plus one 4-way max for the head.
//
// TRANSPORT: the frozen host-staged reduce-scatter of reference/allred3.hip, measured 40.61 us.
// One host thread per GPU; atomic-spin rendezvous; each worker D2Hs its 5120-float partial into
// its own pinned buffer, then reduces the DISJOINT 1/ndev slice it owns out of all ndev pinned
// buffers, then H2Ds the assembled result.
//
// NO P2P, AND THIS IS A SAFETY PROPERTY, NOT AN OPTIMIZATION CHOICE. On this box (5.15.0-190,
// hostile permuted-offset gate, five negative controls, run twice byte-identical) P2P fails 0/12
// directed pairs on ALL THREE transports (hipMemcpyPeer, peer-pointer store, peer-pointer load),
// and it fails DESTRUCTIVELY: for the six ordered pairs among cards 0,1,3 the write lands in the
// SENDER's own VRAM at the destination offsets (1,089,536 words clobbered). Pairs involving card 2
// write to a sender-local shadow the target never sees. An earlier reading of this as a harmless
// "silent no-op" was an artifact of an identity-offset test, under which a self-alias and a no-op
// are byte-indistinguishable; it is retracted.
//
// Consequence: a collective that accidentally took a peer path would not merely compute the wrong
// answer, it would silently corrupt the sending card's weights mid-token while still looking
// coherent. This file therefore uses ONLY hipMemcpyHostToDevice / hipMemcpyDeviceToHost through
// pinned host memory. `make audit` fails the build if any peer or D2D API appears in the tree.
//
// The rendezvous here is a sense-reversing barrier across the same worker threads rather than
// allred3's driver-thread phase counter; the shape is identical (two atomic-spin meeting points
// around the slice reduction) and it saves a thread that would otherwise only spin.

static inline double tp_now() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// Wall-clock deadline for the rendezvous. Generous: the collective resolves in ~35 us, so any
// wait beyond seconds is a failure, not slowness. Q27_TP_BAR_MS overrides.
static double g_bar_deadline_ms = 5000.0;
// NEGATIVE CONTROL for the deadline: Q27_TP_BAR_STRAND=<card> makes that card never join
// b1, so its peers must time out and abort. A deadline that has never fired is unproven.
static int    g_bar_strand = -1;
// BF16 collective transport: producers emit bf16, payload halves. Q27_COLL_BF16=0 reverts.
static bool   g_coll_bf16  = true;
static bool   g_coll_nrbf16 = true;    // Q27_COLL_NRBF16=0 reverts: the NR-row (speculative verify) collective stages bf16 partials (measured +6-8% at K=1..3)
static int    g_nr_p2p     = 1;       // Q27_NR_P2P: 1 = bf16 store-kernel push + local reduce (default), 2 = peer-read pull, 3 = SDMA push, 0 = host-staged
static int    g_nr_p2p_min = 4;       // Q27_NR_P2P_MIN: rows from which the GPU-resident collective is used (measured: wins at NR=8, ties at 4, loses at 2)
static int    g_devid_tab[Q27_MAX_DEVICES] = {0};   // HIP device ordinal per card index (filled at the NR_P2P init; hipMemcpyPeerAsync needs ids)
#define Q27_LL_SLOTS 4                          // Q27_NR_P2P=4: receive slots per (owner, source), indexed by site sequence & 3
static int    g_ll_sdma    = 1;       // Q27_LL_SDMA: 1 = pack locally + copy-engine writes into the peers' uncached slots (default), 0 = shader stores
static int    g_proj_seg   = 0;       // Q27_PROJ_SEG: q/k/v (attention) and in_qkv/in_z (GDN) int8-mirror projections in ONE launch per layer
static int    g_ll_epi     = 0;       // Q27_LL_EPI: 1 = the site producers (int8-mirror o/out_proj, NVFP4 down) write the LL words in their epilogue (no pack kernel)
static long   g_ll_epi_taken[Q27_MAX_DEVICES] = {0}, g_ll_epi_armed[Q27_MAX_DEVICES] = {0};   // receipt: sites whose producer packed the LL words / sites armed
// ---- RUNTIME TRANSPORT RECEIPT (2026-09-17) -------------------------------------------------
// WHAT THE CONFIGURATION SAID IS NOT WHAT THE ENGINE DID. `p2p_` is recomputed per pass as
// (g_nr_p2p && NR >= g_nr_p2p_min), so a threshold, a row count and a mode interact to choose a
// transport 128 times per round, and none of that is visible from the environment. These counters
// are incremented INSIDE the two transport implementations themselves -- tp_allreduce_nr (host
// staging) and nrp_site (GPU-resident) -- never at the decision point, so they report execution
// and not intent. 64 layers x 2 sites means a decode request must show sites summing to
// 128 x rounds; anything else is the receipt refusing to corroborate the arm.
// Per-phase host-collective wall, accumulated per card over the process lifetime and differenced
// per request. [0] stage_wait (blocks on the GPU finishing the preceding segment -> prices that
// segment), [1] b1 skew, [2] reduce arithmetic, [3] b2 skew, [4] invN tail.
static double g_collt[Q27_MAX_DEVICES][6] = {{0}};
static long g_coll_host_sites[Q27_MAX_DEVICES] = {0};
static long g_coll_p2p_sites [Q27_MAX_DEVICES][5] = {{0}};   // [mode]: 1 push, 2 pull, 3 SDMA, 4 LL (flag-in-data)
static int  g_coll_nr_eff    [Q27_MAX_DEVICES] = {0};        // the row count the last site actually saw
static bool   g_coll_side  = false;  // Q27_COLL_SIDE: stage the D2H off the main stream
static bool   g_rn_split   = true;
static bool   g_gdn_fq     = true;
static bool   g_coll_prof  = false;
static bool   g_coll_zc    = false;  // Q27_COLL_ZC=1: consumer reads the pinned result in place
static float* g_coll_part[Q27_MAX_DEVICES] = {nullptr}; // device alias of C.hp[g]
static bool   g_coll_kcopy = false;   // Q27_COLL_KCOPY=1: stage the partial with an in-stream kernel
static int    g_coll_dflag = 0;       // Q27_COLL_DFLAG: 1 = host spins on a device-written done flag
static int    g_copy_mb    = 0;       // Q27_COPY_MB: multi-block staging copy on the plain decode path
static int    g_nr_lds     = 0;       // Q27_NR_LDS: LDS-staged row-batched GEMVs (measured: 112-145 VGPRs as compiled, slower; forcing 64 VGPRs spills)
static int    g_nr_hg      = 0;       // Q27_NR_HG: split-row (two waves per row) row-batched GEMVs (measured: no gain at NR=2/3 despite 4 waves/SIMD; -5% on plain at NR=1)
static int    g_nr_fam     = 0;       // Q27_NR_FAM: bisect mask -> split-row kernels per family (1 fp8 K5120, 2 fp8 K1536, 4 gate/up, 8 down, 16 lm_head)
static int    g_nr_att1    = 0;       // Q27_NR_ATT1=1: per-row attention launches (bisect)
static int    g_nr_gdn1    = 0;       // Q27_NR_GDN1=1: per-row conv/step + memcpy snapshots (bisect)
static int    g_nr_qb      = 1;       // Q27_NR_QB=0: per-row quantize launches (bisect)
// Q27_NR_DUP: THE SHIPPED PATH HAS NO LEDGER, so price a stage the way the combine was priced --
// run it TWICE and read d(wall)/d(stage). Every stage below stores (no accumulation into a shared
// buffer across launches), so a duplicate recomputes the same bytes and the token stream is
// unchanged; the decode tok/s delta is the stage's cost. Bitmask:
//   1 = attention decode (+ combine), 2 = MLP gate/up + swiglu, 4 = lm_head,
//   8 = fp8 projections (qkv / in_qkv / in_z via P_FP8_BF16), 16 = fp8 result projections
//   64 = down_proj alone (P_DOWN). MEASURED IN-BOOT 2026-09-17; see the note at the call site --
//   P_DOWN IS a pure store and the 08:21 "it accumulates" diagnosis below is retracted.
//   (o_proj / out_proj via P_FP8_RES -- VALIDATE BY HASH: same two-output+alpha shape as P_DOWN),
//   32 = the per-row quantize of the attention output and of z.
// ANY arm whose tokhash moves is INVALID: that is how the whole-trio MLP duplicate was caught
// (2026-09-17 08:21, text garbage from byte 9, so P_DOWN accumulates and is not idempotent).
static int    g_nr_dup     = 0;
// E4 (2026-09-17): PER-REQUEST ARM CONTROL. Q27_NR_DUP is read once at boot, so pricing N stages
// cost N boots -- and this box scatters +-1.6% across boots against +-0.2% WITHIN one boot
// (measured: same binary, same prompt 14360, same tokhash, ms_per_round 28.3143 / 28.2570 in one
// boot, 27.8728 in the next). A per-stage attribution table built one-boot-per-arm therefore
// carries 8x the noise of the effect it is trying to resolve. Re-reading the arm at the start of
// every request collapses the whole table into ONE boot, where the floor is 0.2%.
// The file holds two integers: "<nr_dup> <ab_arm>". Card 0 reads it between C.b1 and C.b2, the
// same ordering the request itself uses, so every thread sees one value for the whole request.
// It selects which already-compiled path runs; it does not originate a request. Measurement still
// comes from a Harness turn and nothing else.
static const char* g_ab_file = nullptr;
static int    g_ab_arm     = 0;       // generic kernel-variant selector, reported as arm= on Q27_REQ
static void q27_ab_reread() {
    if (!g_ab_file) return;
    FILE* f = std::fopen(g_ab_file, "r");
    if (!f) return;
    int dup = 0, arm = 0;
    const int got = std::fscanf(f, "%d %d", &dup, &arm);
    std::fclose(f);
    if (got >= 1) g_nr_dup = dup;
    if (got >= 2) g_ab_arm = arm;
}
static int    g_nr_ch      = 4;       // Q27_NR_CH=1..4: rows per chunk of the NR>4 verify (4-row form 39.5 ms vs 3-row 26.4 at 8K)
                                      // instead of hipEventSynchronize; 2 = FIRING CONTROL, no wait
static std::atomic<long long> g_dflag_waits{0}, g_dflag_skips{0};
static std::atomic<long long> g_kcopy_calls{0};   // executed-call proof (expect 127-129/token)
static unsigned g_coll_hostflag = hipHostMallocDefault; // Q27_COLL_COARSE=1 -> coarse-grained (cacheable host reads)
// CRITICAL-PATH SENSITIVITY PROBE. Profiled kernel milliseconds are not wall milliseconds: an
// operation before a barrier can be made faster and give the time straight back as barrier wait.
// Rather than build an optimisation to find out, make the stage SLOWER by a known amount and
// watch the wall. dWall/dStage ~ 1 means critical path; ~ 0 means slack. Both probed stages are
// idempotent (pure functions of inputs the repeat does not modify), so tokens must not change --
// which is itself the check that the probe is repeating the real work.
static bool   g_text  = false;   // Q27_TEXT=1: text in, text out (declared early: the markers route by it)
static int    g_slow_quant = 0;     // extra redundant runs of the standalone pre-barrier quantizer
// MILESTONE CLOCK. A run that exceeds its budget must say WHERE the time went; "probably
// contention" is not a diagnosis. Prints elapsed-since-process-start at every phase boundary so
// LOAD / ALLOCATE / RESIDENCY / PREFILL / DECODE / EXIT are separable from one line of output.
static double g_t0_wall = 0;
static void MS(const char* tag) {
    static const bool early_text = q27_env_flag("Q27_TEXT", false);   // T0 fires before argv is parsed
    std::fprintf((g_text || early_text) ? stderr : stdout, "Q27_MS %-18s %8.2f s\n", tag, (tp_now() - g_t0_wall) / 1000.0);
    std::fflush(stdout);
}
static unsigned g_host_tail_mask = 0; // logical workers sharing a physical GPU may use host-submitted consumers
static bool   g_tail_preenq = false;   // enqueue the post-barrier tail BEFORE the barrier
static bool   g_preenq_ok    = false;  // device actually supports hipStreamWaitValue32
static int    g_wait_kernel  = 0;      // Q27_WAIT_KERNEL: spin kernel (q27_wait_flag_eq) instead of hipStreamWaitValue32
extern "C" void q27_wait_flag_eq(const unsigned* flag_dev, unsigned want, hipStream_t s);
static bool   g_rn_dropy = true;    // skip the tail's normalised-vector store where nothing reads it
static int    g_slow_null  = 0;     // extra EMPTY launches at the post-barrier point
static int    g_slow_mixq  = 0;     // extra runs of the pre-barrier mixer fp8 quantizer
static int    g_slow_tail  = 0;     // extra redundant runs of the post-barrier normalize/quantize
static bool   g_rn_epi = true;      // fold the residual share into the projection epilogue
static bool   g_hostss_neg = false; // firing control: drop the pre-add, residual must vanish
static bool   g_rn_hostss = false;  // host computes sum(x^2) during the reduce it already does
static bool   g_null_red = false;   // CEILING PROBE ONLY: skip reduce+b2, output is garbage
static bool   g_ar1_skip = false;   // CEILING PROBE ONLY: delete all-reduce #1 wholesale, output is garbage
// Counts how often the fused rmsnorm+quantize is DECLINED and the two-kernel fallback runs. The
// fused variants exist for exactly this pairing, so a nonzero count means 2-3 extra launches per
// layer per token are being paid for a shape the fused kernel does not cover.
static std::atomic<long long> g_rnq_fb{0};

static bool   g_tp_affinity = false;  // pin each card's host thread near that card's PCIe root
static bool   g_fd_neg = false;   // firing control: claim the copy happened without enqueuing it
static bool   g_coll_pf = false;   // prefetch the reduce's 320 lines during barrier-1 slack
static bool   g_fp8x4 = false;    // route the fp8 projections through their load-time NVFP4 twins
static int    g_pf_x4w = 0;       // Q27_PF_X4W=<ki>: route the BATCHED prefill projections through the same
                                  // load-time NVFP4 twins with the WIDE kernels (MMQ geometry), WITHOUT setting
                                  // g_fp8x4 -- which is wired into the serial path and disables attb / the GDN
                                  // tile path / o_proj batching. Value = wide ki code.
static int    g_kv_fuse = 0;      // Q27_KV_FUSE: 1 = k_proj+v_proj in one launch, 2 = firing control
static int    g_pf_mlp2 = 0;      // Q27_PF_MLP2=1: small-M prefill MLP kernels; Q27_PF_MLP_R = rows per wave (2|4)
static int    g_pf_mlp_r = 2;
static int    g_pf_oproj_b = 0;   // Q27_PF_OPROJ_B=1: chunk out_proj for GDN layers in the prefill sweep
static int    g_pf_tail_b  = 0;   // Q27_PF_TAIL_B=1: one launch for the chunk's C tails (needs the chunk collective)
static int    g_pf_qkv2    = 0;   // Q27_PF_QKV2=1: small-M q/k/v kernels for the attention layers' prefill
static int    g_pf_qkv_r   = 2;   // Q27_PF_QKV_R rows per wave (1|2|4)
static int    g_pf_gdn_fq  = 0;   // Q27_PF_GDN_FQ=1: prefill mode 6 quantizes the mixer in the step epilogue
static int    g_pf_pipe    = 0;   // Q27_PF_PIPE=1: pipeline chunk collective #2 behind the next chunk
static int    g_pf_pipe1   = 0;   // Q27_PF_PIPE1=1: pipeline chunk collective #1 (GDN tile path; needs OPROJ_B+TAIL_B)
static int    g_pf_norm_b  = 0;   // Q27_PF_NORM_B=1: the GDN tile input norm+quantize in one launch per tile
static int    g_pf_mk3     = 0;   // Q27_PF_MK3=2|4: tile in_qkv/in_z via the LDS-staged R-rows/wave kernel (bit-identical to m2)
static int    g_pf_m3f     = 0;   // Q27_PF_M3F=2|4: chunk out_proj/o_proj via the LDS-staged R-rows/wave kernel (bit-identical to m2f)
static int    g_pf_att_chunk = 0; // Q27_PF_ATT_CHUNK=1: attention prep/decode/combine(+quant) over the chunk in three launches (bit-identical)
static int    g_pf_gdn_scan = 0;  // Q27_PF_GDN_SCAN=1: GDN a/b, conv and the recurrence as three launches per tile (bit-identical)
static int    g_att_pf     = 1;   // Q27_ATT_PF=2: chunk attention keeps two K/V row pairs in flight (bit-identical)
static int    g_att_hpw    = 6;   // Q27_ATT_HPW=3: chunk attention with 3 heads per wave (two waves per split; ~half the registers)
static int    g_pf_chw     = 0;   // Q27_PF_CHW: runtime chunk width (<= Q27_PF_CH); 0 = auto by prompt length
static int    g_pf_chw_auto = 256; // Q27_PF_CHW_AUTO: prompts shorter than this use 8-position chunks
static int    g_pf_chw_auto2 = 4096; // Q27_PF_CHW_AUTO2: prompts shorter than this use 16; longer use the compile-time width
static int    g_pf_kcopy   = 0;   // Q27_PF_KCOPY=1: the chunk collectives stage through the in-stream copy kernel instead of SDMA D2H
static int    g_pf_look    = 1;   // Q27_PF_LOOK=1..3: collective #1 lookahead depth in chunks (staging ring of NB=4)
static int    g_pf_mk2r    = 0;   // Q27_PF_MK2R=2|4: tile in_qkv/in_z with R rows per wave, no LDS (bit-identical to m2)
static int    g_pf_mk16    = 0;   // Q27_PF_MK16=1: tile in_qkv/in_z via fp16 pre-converted weights (bit-identical to m2)
static int    g_pf_hstreams = 0;   // Q27_PF_HSTREAMS=1: the 8-position halves of a chunk's MLP and q/k/v run on concurrent streams (fork/join events; same kernels, same data)
// Q27_PF_WIDE: the GEMM-order wide FFN (milestone 2). Replaces the ceil(Cch/8) 8-wide b13 MLP launches
// with ONE wide GEMM per matrix per chunk, which is why the chunk has to be wide: the wide gate+up LOSES
// to b13 below M=128 (the register-blocked tile is 128 tokens wide) and only wins from M=256 (window 80).
static int    g_pf_wide    = 0;   // Q27_PF_WIDE=1: wide FFN on the MLP only; rung 3 stays the frozen mode
static int    g_pf_wide_min = 128; // Q27_PF_WIDE_MIN: below this many positions the wide kernel loses to b13, so fall back
static int    g_mlp_tl     = 0;   // Q27_MLP_TIMELINE=1: host+GPU timeline of the MLP/collective boundary. Instrumentation only.
static int    g_mlp_timing = 0;   // Q27_MLP_TIMING=1: bracket the b13 MLP too, so the wide FFN's GPU time has a
                                  // same-binary, same-stream counterpart instead of a share quoted from an old profile
static int    g_pf_wide_lag = 6;   // Q27_PF_WIDE_LAG: chunks to wait before draining a staged tile
static int    g_pf_wide_slice = 0; // Q27_PF_WIDE_SLICE: positions per commit slice (0 = Q27_PF_MC). Set to
                                   // Q27_PF_TSLOT to reproduce the single-block commit for bisection.
static int    g_pf_wide_sdma = 1;  // Q27_PF_WIDE_SDMA=1 (default): stage the tile output with the copy engine
                                   // (hipMemcpyAsync D2H) exactly as rung 3 stages a chunk. 0 = the in-stream
                                   // q27_copy_f4 kernel, which writes 5.24 MB into PINNED host memory from the
                                   // CUs over PCIe -- latency-bound and it occupies CUs rung 3 leaves free.
static int    g_pf_wide_noffn = 0; // Q27_PF_WIDE_NOFFN=1: DIAGNOSTIC ONLY -- run the wide graph (deferred MLP,
                                   // tile collective) but skip the three FFN launches. Separates "the host cannot
                                   // feed the GPU" from "the FFN costs this much" without touching the kernels.
static int    g_pf_wide_kg = 8000; // Q27_PF_WIDE_KG: gate+up ki code. 8000 = register-blocked KI8 (best at M>=128)
static int    g_pf_wide_kd = 168;  // Q27_PF_WIDE_KD: down ki code. 168 = row-major KI16 NW8 (best at M=256; 8000 wins at M=1024)
static int    g_pf_gdn_scan2 = 1;  // Q27_PF_GDN_SCAN2=1 (default, rung 3): tile recurrence with phase 0 precomputed in parallel, two barriers per position (bit-identical)
static int    g_pf_mlp14   = 0;   // Q27_PF_MLP14=5|6: gate+up on the int8 mirror (needs Q27_PF_W8=1 at load; bit-identical; measured no faster than v13 at 2x bytes; off)
static int    g_pf_mlp14d  = 0;   // Q27_PF_MLP14D=5|6: down likewise
static int    g_pf_mk2r2   = 1;   // Q27_PF_MK2R2=1 (default, rung 3): tile projection m2r with the rebuilt reduction epilogue (bit-identical)
static int    g_pf_m2f3    = 1;   // Q27_PF_M2F3=1 (default, rung 3): chunk out_proj m2f2 likewise
// Q27_PF_KVP2P=1: peer-quarter KV (EXPERIMENTAL, DEFAULT OFF, CURRENTLY INCORRECT).
// Distinct from Q27_PF_P2P below -- an earlier version of this reused THAT flag, which silently
// turned on the cross-card collective as well; that collective's self-test fails on this box
// ("peer access unavailable/failed; reverting to the host collective"), so the peer pointers this
// path writes through were never mapped and the KV was corrupted. The model could not see its own
// prompt. Do not enable until peer access self-tests clean AND this path is re-validated.
static int    g_pf_kvp2p   = 0;
static int    g_pf_p2p     = 0;   // Q27_PF_P2P=1: GPU-resident cross-card collective (peer reads in one
                                  // fused kernel; no D2H/host reduce/H2D). Requires Q27_PF_X4W>0.
static int    g_pf_p2p_rn  = 1;   // Q27_PF_P2P_RN=0: keep collective #1 sites (red_rn_perm_b) on the host (bisect)
static int    g_pf_p2p_mix = 1;   // Q27_PF_P2P_MIX=0: keep the mixer collective on the host (bisect)
static bool   g_ls_q8      = true; // Q27_LS_Q8=1 (default): the quantized producer/consumer prefill graph (int8 KV, int8 GDN/attention handoffs, fp32 residual)
static int    g_ls          = 0;  // Q27_LAYER_SPLIT=1: layer-split residency -- card g owns FULL layers
static int    g_sr          = 0;  // Q27_LS_SR=1: SINGLE residency -- the layer-split copy is the only copy; decode runs sequentially through the cards (src/q27_sr_main.inc)
static int    g_tpsr        = 0;  // Q27_TPSR=1: TENSOR-PARALLEL single residency (2026-09-17 night) -- Q27_LAYER_SPLIT=0, every card holds 1/ndev of every layer ONCE in 8-bit
                                  // (block-scaled FP8 gate/up row shards + FP8 down K-slice, int8/64 mirrors of the fp8 projection shards), quarter KV; the shipped TP
                                  // decode round reads the FP8 down slice through the natural-order swiglu; the batched TP prefill runs its MLP on rocBLAS int8 from a
                                  // transient per-row shard converted once per layer per sweep, and conditions the MTP draft on the prompt like the layer-split sweep does.
static int    g_tpsr_proj_rb = 0; // Q27_TPSR_PROJ_RB=1: prefill v2 -- the batched TP sweep's projections (q/k/v, o_proj, in_qkv/in_z, out_proj) on rocBLAS int8 from
                                  // transient per-row mirrors, instead of the fp8 chunk kernels on the originals. Off by default; opt-in, fail-closed (tdie) at the seam.
static int    g_sr_head     = -1; // Q27_LS_SR: the card that owns layer 63 -- final norm, lm_head, MTP draft and draft KV live there
                                  // [g*16,(g+1)*16); the prefill sweep is chunk-major per stage with per-chunk
                                  // handoff copies and NO intra-layer collectives. Decode keeps the TP layout.
static int    g_pf_qkv3    = 0;   // Q27_PF_QKV3=1: q/k/v s1_b2 likewise (measured neutral at 4 waves; off)
static int    g_pf_mlp13   = 5;   // Q27_PF_MLP13=4|5 (default 5, rung 3): v12 + wave-uniform weight addressing, launch bound = that many waves (bit-identical)
static int    g_pf_mlp13d  = 6;   // Q27_PF_MLP13D=5|6 (default 6, rung 3): down likewise
static int    g_pf_mlp12   = 0;   // Q27_PF_MLP12=4|8|16: gate+up with the interleaved bpermute+DPP epilogue (bit-identical)
static int    g_pf_mlp12d  = 0;   // Q27_PF_MLP12D=4|8: down likewise
static int    g_pf_mlp11   = 0;   // Q27_PF_MLP11=2|4|8: gate+up with that many LDS reads in flight (bit-identical)
static int    g_pf_mlp11d  = 0;   // Q27_PF_MLP11D=2|4|8: down likewise
static int    g_pf_mlp10   = 0;   // Q27_PF_MLP10=1: gate+up with the weight fetch before the staging (bit-identical)
static int    g_pf_mlp10d  = 0;   // Q27_PF_MLP10D=1|2: down likewise
static int    g_pf_mlp9    = 0;   // Q27_PF_MLP9=1: gate+up with the next chunk's weights prefetched (bit-identical)
static int    g_pf_mlp9d   = 0;   // Q27_PF_MLP9D=1|2: down with the weight prefetch
static int    g_pf_mlp8    = 0;   // Q27_PF_MLP8=1: gate+up with double-buffered staging, no register caches (bit-identical)
static int    g_pf_mlp7    = 0;   // Q27_PF_MLP7=1: persistent-row gate+up (bit-identical)
static int    g_pf_mlp7d   = 0;   // Q27_PF_MLP7D=1: persistent-row down
static int    g_pf_mlp7_rpw = 0;  // Q27_PF_MLP7_RPW: rows per wave (0 = one round over 60 CUs)
static int    g_pf_mlp6    = 0;   // Q27_PF_MLP6=1: gate+up with two k-chunks per stage (bit-identical)
static int    g_pf_mlp6d   = 0;   // Q27_PF_MLP6D=1|2: down with two k-chunks per stage
static int    g_pf_mlp5    = 0;   // Q27_PF_MLP5=1|2: MLP kernels without the activation register cache, occupancy hint (bit-identical)
static int    g_pf_mlp4    = 0;   // Q27_PF_MLP4=1|2 (rows/wave): software-pipelined MLP kernels (bit-identical to MLP2)
static int    g_pf_m2f2    = 0;   // Q27_PF_M2F2=2|4: chunk out_proj with R rows per wave, no LDS (bit-identical to m2f)
static int    g_pf_mlp3    = 0;   // Q27_PF_MLP3=1|2 (rows/wave): MLP kernels with every k-chunk's weights fetched up front (bit-identical to MLP2)
static int    g_pf_noredo  = 1;   // Q27_PF_NOREDO=0 re-enables mode 6's dead input norm + K=5120 projections (control arm)
static int    g_pf_att_b   = 0;   // Q27_PF_ATT_B=1: attention layers' chunk o_proj + chunk collective + chunk tail (pipelined with PIPE1)
// PREFILL SLOT CAP, sized at run time. The sweep keeps one hidden/pending/1-over-rms slot per
// PROMPT POSITION (30 KB per position per card), and those arrays were a compile-time 1024, so any
// longer prompt was refused. The prompt length is known before allocation, so size them to it.
static int    g_pf_cap = Q27_PF_CAP;
// ---- Q27_SERVE_GROW: in-place growth of the per-position prefill scratch ------------------------
// g_pf_cap was sized once, at cold start, from whichever prompt arrived first -- in serve mode that
// is routinely a ~130-token title request, so the engine sat at the floor and REFUSED every window
// that later grew past it. A refusal costs the whole conversation: the server kills the engine, a
// replacement cold-starts (~60 s) and re-prefills the entire history (34 s on a 17.8 k window,
// measured, made worse by restart pollution). The slot buffers are pure per-window scratch, so they
// can be freed and re-allocated in place; the KV cache, the recurrent state, the weights and
// pos_cur are all separate allocations and survive. Growth goes through a bucket ladder, is capped
// by the tightest card's free memory on all devices WITH a safety margin, and is announced.
static std::atomic<int> g_pf_grow_seq{0};    // leader publishes, waiters observe
static std::atomic<int> g_pf_grow_ok{0};
static int    g_df2_draft = 0;   // Q27_DFLASH2: the integrated DFlash2 block-drafter path (declared early: dev_alloc_tp sizes the prompt tap buffers)
// SERVE MODE (Q27_SERVE=1): the model stays resident and requests arrive on stdin.
static bool   g_serve = false;
static bool   g_eos_im_end = false;   // treat <|im_end|> (248046) as a stop token (default: on in serve mode)
static inline bool q27_is_eos(unsigned t) { return t == 248044u || (g_eos_im_end && t == 248046u); }
// pf_only: SWEEP THE IDS, GENERATE NOTHING. The serving path appends an oversized window in
// arena-sized steps, and a step must land the prefix EXACTLY where the caller expects it: a step
// that decodes (even one EOS token) appends tokens the caller never asked for, so the next step's
// ids are swept at the wrong absolute positions AND the model's prefix stops being the prompt.
// Measured before this existed: a 9,109-token window served as 8960 + 149 had the remainder swept
// 19 positions late, behind 19 generated tokens, and the Harness turn came back with no answer.
struct ReqState { std::vector<unsigned> ids; int maxn = 0; bool eof = false; bool pf_only = false;
                  int rewind_pos = -1; };   // Q27_SERVE_REWIND (maxn == -3): absolute target position
static ReqState g_req;
// Q27_REWIND_CK: how many request-boundary (S, conv) checkpoints to keep. Each bank costs the
// full recurrent state -- measured 64.6 MiB/card at ctx=262144 (S 63.0 + conv 1.6) -- so the
// ring is bounded by the TIGHTEST card, not the 32. 0 disables and restores the old reset-only
// behaviour exactly.
// ORDER-INDEPENDENT ON PURPOSE. Device allocation runs long before the serve-loop init, so a
// plain global set in that init reads 0 at the allocation site and the banks are silently never
// allocated -- an inert arm that still logs the env var as requested. Caught by VRAM not moving.
static int q27_rewind_ck(void) {
    static int v = -1;
    if (v < 0) { v = q27_env_int("Q27_REWIND_CK", 0); if (v < 0) v = 0; if (v > 16) v = 16; }
    return v;
}
static int g_ck_pos[16] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };  // absolute position each bank holds; -1 = empty
// TEXT MODE (Q27_TEXT=1): argv carries the user's text, stdin turns are text, output is streamed as
// text. The tokenizer is the generic HF tokenizer.json engine in src/tok (PCRE2 pre-tokenizer);
// the chat template is the Qwen3.5 text-only subset in q27_chat.h.
static hf::Tokenizer* g_tok = nullptr;
static FILE*  g_text_out = stdout;         // text mode: the real stdout; diagnostics move to stderr
static std::string g_first_text;           // request 0's user text, echoed when generation starts
static bool   g_think = false;             // Q27_THINK=1 leaves the think block to the model
static std::string g_system;               // Q27_SYSTEM="..." optional system turn
static std::vector<unsigned> q27_encode(const std::string& text) {
    std::vector<unsigned> ids;
    for (std::int32_t t : g_tok->encode(text)) if (t >= 0 && t < Q27_VOCAB) ids.push_back((unsigned)t);
    return ids;
}
// STREAMING ON THE TOKEN WIRE (Q27_SERVE_STREAM=1, default ON in serve mode).
// Serve mode printed Q27_TOKENS only AFTER the whole generation finished, so a client could not
// show anything until the turn ended -- the Harness sat on "Deep diving..." and then dumped the
// reasoning and the answer at once. Text mode streams per token but its line-based wire destroys
// the newlines in file contents, diffs and shell output that tool results carry, so it is not an
// option here. Emitting the token ID as it is produced gives the same streaming without touching
// the bytes: the client decodes incrementally and the final Q27_TOKENS line still arrives
// unchanged, so a non-streaming reader keeps working exactly as before.
// Emission rides inside q27_emit(), which the decode loops call ONLY for committed tokens --
// including the speculative path, where q27_main.cpp:6796 emits exactly sp_ncommit of them after
// verify. So a rejected draft token can never reach the wire and nothing has to be retracted.
static bool g_stream_ids = false;
static void q27_emit_id(unsigned t) {
    if (!g_stream_ids) return;
    if (t == 248044u || t == 248045u || t == 248046u) return;   // chat specials are not content
    std::printf("Q27_TOK %u\n", t); std::fflush(stdout);
}
// Stream one generated token as text (thread 0 only). The three chat specials are not printed;
// everything else, including <think> tags when thinking is on, is shown as the model wrote it.
static void q27_emit(unsigned t) {
    q27_emit_id(t);
    if (!g_text || !g_tok) return;
    if (t == 248044u || t == 248045u || t == 248046u) return;
    const std::string piece = g_tok->decode_token((std::int32_t)t);
    std::fwrite(piece.data(), 1, piece.size(), g_text_out); std::fflush(g_text_out);
}
// One request per line: "<maxn> <id> <id> ...". Blank lines are skipped; EOF ends serving. Ids are
// range-checked against the vocabulary (an out-of-range id is a silent garbage prompt otherwise).
static void serve_read_request(ReqState& R) {
    R.ids.clear(); R.maxn = 0; R.eof = false; R.pf_only = false;
    std::string line;
    for (;;) {
        if (!std::getline(std::cin, line)) { R.eof = true; return; }
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        break;
    }
    if (g_text) {   // one user turn per line; the template closes the previous assistant turn
        R.ids = q27_encode(q27::chat_next_turn(line, g_think));
        R.maxn = q27_env_int("Q27_SERVE_MAXN", 512);
        std::fprintf(g_text_out, "\n[user] %s\n[assistant] ", line.c_str()); std::fflush(g_text_out);
        return;
    }
    std::istringstream is(line);
    long long v; bool first = true;
    while (is >> v) {
        if (first) { R.maxn = (int)v; first = false;
                     // maxn == -3 is REWIND and the NEXT field is an absolute POSITION, not a token
                     // id. It must bypass the vocabulary range check below: ctx is 262144 while
                     // Q27_VOCAB is 248320, so any position past the vocabulary would be silently
                     // dropped and the rewind would land at 0 -- i.e. look exactly like the reset it
                     // is meant to replace.
                     if (R.maxn == -3) { long long pp = -1; if (is >> pp) R.rewind_pos = (int)pp; break; }
                     continue; }
        if (v < 0 || v >= (long long)Q27_VOCAB) { std::fprintf(stderr, "Q27_REQ: id %lld out of range, dropped\n", v); continue; }
        R.ids.push_back((unsigned)v);
    }
    // maxn == -1 is the RESET command, not a token budget. maxn == -2 is PREFILL-ONLY (the
    // serving path's chunk step, see ReqState). Only a zero/absent budget defaults.
    if (R.maxn == -2) { R.pf_only = true; R.maxn = 0; }
    else if (R.maxn == 0) R.maxn = 256;
}
// EXECUTED-CALL PROOF FOR THE CROSS-CARD COLLECTIVE.
// Q27_PF_P2P is a GPU-resident collective that replaces the host D2H/reduce/H2D rendezvous. It
// has a startup self-test, but a self-test only proves peer access WORKS -- not that the prefill
// actually took the peer path. That distinction is not academic here: the peer-enable check used
// to treat hipErrorPeerAccessAlreadyEnabled as failure, so the flag silently demoted itself to
// host staging on every run ever measured. An inert arm that looks like a measurement is exactly
// what these counters exist to make impossible, so count both paths and print them.
static std::atomic<long long> g_p2p_rn_calls{0};    // collective #1 on the GPU (red_rn_perm_b)
static std::atomic<long long> g_p2p_mix_calls{0};   // collective #2 on the GPU (red_b, mixer)
static std::atomic<long long> g_host_coll_calls{0}; // host-staged reduce (tp_reduce_bx)
static std::atomic<long long> g_kv_fused{0};   // executed-call proof for the fused k|v path
static bool   g_preenq_mlp = false; // pre-enqueue the whole MLP chain behind the flag-gated tail
static bool   g_preenq_prologue = false; // pre-enqueue the NEXT layer's prologue behind its input norm
static bool   g_dsplit = false;    // GDN step value-dim split: 4 blocks per head, 48 blocks per card
static bool   g_coll_fusedrain = true;  // enqueue the D2H behind the producer, one sync for both
static int    g_zcout_neg   = 0;       // negative control: keep the VRAM destination, still skip the D2H
static int    g_coll_zcout  = 1;       // producers store the partial straight into pinned host memory
static float* g_coll_acc   = nullptr; // device-visible pointer to the pinned reduced vector  // Q27_COLL_PROF=1: per-phase collective breakdown   // Q27_GDN_FQ=0 -> separate quantize launch   // Q27_RN_SPLIT=0 -> single-block rmsnorm

// WHICH CARD STOPPED. TpBar is a counter, so "arrived 3" names a NUMBER, not a card, and the three
// spinning threads are usually parked in the ring's ls_rec spin (which has no deadline) rather than
// in the barrier at all. Each worker stamps the generation it is entering before every wait; on
// timeout the handler prints all of them plus the ring's per-card job record, and the cells that
// are behind name the card that never arrived and the job it stopped on. Relaxed stores on a cell
// nobody reads on the fast path cost nothing measurable.
static std::atomic<unsigned> g_tp_seen[Q27_MAX_DEVICES];
static std::atomic<int>*     g_tp_lsrec = nullptr;      // points at the shared C.ls_rec once it exists
static int                   g_tp_ndev_dbg = 0;
static thread_local int      t_tp_card = -1;

// ---- Q27_MLP_WSHARE (2026-09-18): the MLP rows each card owns -------------------------------
// gate/up are column-parallel (a card owns rows [off,off+len) of the 17408-row intermediate) and
// down_proj is row-parallel over the same 17408 K, so ONE vector describes all three. Uniform
// Q27_INTER/ndev unless `Q27_MLP_WSHARE` says otherwise; parsed and validated once in q27.h.
// This is the lever that turns the 32 GiB card's spare residency into authority for the 16s, and
// it is the only place the MLP shard length is allowed to come from.
static int  g_mlp_off[Q27_MAX_DEVICES] = {0};
static int  g_mlp_len[Q27_MAX_DEVICES] = {0};
static int  g_mlp_max = 0;
static bool g_mlp_ready = false;
static void mlp_wshare_init(int ndev) {
    if (g_mlp_ready) return;
    q27_mlp_wshare(g_mlp_off, g_mlp_len, ndev);
    g_mlp_max = 0;
    for (int g = 0; g < ndev; ++g) if (g_mlp_len[g] > g_mlp_max) g_mlp_max = g_mlp_len[g];
    g_mlp_ready = true;
}
static inline int mlp_len(int card) { return g_mlp_ready ? g_mlp_len[card] : (Q27_INTER / 4); }
// Layer-split prefill collapses the divisor to 1: the owner card holds the WHOLE MLP.
static inline int mlp_len_div(int card, int div) { return div == 1 ? Q27_INTER : mlp_len(card); }
struct TpBar {
    std::atomic<int>      cnt;
    std::atomic<unsigned> gen;
    int                   n;
    void init(int nd) { cnt.store(0); gen.store(0); n = nd; }
    // DEADLINE, NOT A SPIN COUNT. This spun unbounded: a worker that died or a card that wedged
    // would hang the engine forever holding four GPUs, and the collective gate's own negative
    // control drives this path by construction. A spin count is not a timeout -- it is a duration
    // that changes with clock speed and contention. The clock is read once per 4096 spins so the
    // fast path (the barrier resolves in tens of microseconds) pays essentially nothing.
    // idle=true: the peers are parked waiting for card 0 to receive the NEXT REQUEST, which is a
    // human/Harness think-time wait, not a wedged card. Carrying the compute deadline here aborted
    // a perfectly healthy engine 60 s after every reply ("FATAL TpBar ... arrived 3") and destroyed
    // residency, so every turn paid a full model reload. Idle waits have no deadline and sleep
    // instead of spinning, so three cards do not burn three cores between turns.
    void wait(bool idle = false) {
        const unsigned g = gen.load(std::memory_order_acquire);
        if (t_tp_card >= 0) g_tp_seen[t_tp_card].store(g, std::memory_order_relaxed);
        if (cnt.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            cnt.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
        } else {
            const double t0 = tp_now();
            unsigned spins = 0;
            while (gen.load(std::memory_order_acquire) == g) {
                if (idle) { if ((++spins & 255u) == 0u) usleep(50); continue; }
                if ((++spins & 4095u) == 0u && tp_now() - t0 > g_bar_deadline_ms) {
                    std::fprintf(stderr,
                        "FATAL TpBar: waited %.0f ms for %d participants at generation %u "
                        "(arrived %d). A peer thread is gone or a card has wedged; aborting rather "
                        "than holding the GPUs forever.\n",
                        tp_now() - t0, n, g, cnt.load(std::memory_order_relaxed));
                    char who[256]; int wn = 0;
                    for (int q = 0; q < g_tp_ndev_dbg && wn < 200; ++q)
                        wn += std::snprintf(who + wn, sizeof(who) - (size_t)wn, " card%d:gen=%u,job=%d", q,
                                            g_tp_seen[q].load(std::memory_order_relaxed),
                                            g_tp_lsrec ? g_tp_lsrec[q].load(std::memory_order_relaxed) : -99);
                    std::fprintf(stderr, "FATAL TpBar detail:%s  (a card whose gen is BEHIND %u never "
                                         "reached this barrier; its job index says where the ring left it)\n",
                                 who, g);
                    std::abort();
                }
            }
        }
    }
};

// ARRIVAL SKEW IS TOPOLOGY, NOT ORDER. Cards 2 and 3 are last to the barrier 95% of the time and
// are never overtaken, so this is not the executor rotating work: the four cards sit on separate
// PCIe roots of the EPYC IO die and the distant two pay more per launch and per transfer. Ask the
// kernel which CPUs are local to each card and keep that card's driving thread there. If the list
// is not a strict subset (NPS1 exposes every CPU as local) fall back to a fixed disjoint block,
// which at least stops the four threads from migrating across CCXs and sharing caches.
static void tp_pin_thread(int devid, int g, int ndev) {
    if (!g_tp_affinity) return;
    cpu_set_t set; CPU_ZERO(&set);
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    int n = 0;
    char bdf[64] = {0};
    if (hipDeviceGetPCIBusId(bdf, sizeof bdf, devid) == hipSuccess) {
        for (char* q = bdf; *q; ++q) *q = (char)tolower((unsigned char)*q);
        char path[256];
        std::snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/local_cpulist", bdf);
        if (FILE* f = std::fopen(path, "r")) {
            char buf[512] = {0};
            if (std::fgets(buf, sizeof buf, f)) {
                for (char* tok = std::strtok(buf, ",\n"); tok; tok = std::strtok(nullptr, ",\n")) {
                    int a2, b2;
                    if (std::sscanf(tok, "%d-%d", &a2, &b2) == 2) { for (int c = a2; c <= b2; ++c) { CPU_SET(c, &set); ++n; } }
                    else if (std::sscanf(tok, "%d", &a2) == 1)    { CPU_SET(a2, &set); ++n; }
                }
            }
            std::fclose(f);
        }
    }
    if (n == 0 || n >= (int)ncpu) {         // no usable locality info -- disjoint blocks instead
        CPU_ZERO(&set);
        const int per = (int)ncpu / (ndev > 0 ? ndev : 1);
        for (int c = g * per; c < (g + 1) * per && c < (int)ncpu; ++c) CPU_SET(c, &set);
    }
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

struct TpColl {
    std::atomic<int> ls_rec[Q27_MAX_DEVICES];   // Q27_LS_Q8: highest chunk index whose done-event card g has recorded
    int    ndev = 0, n = 0;
    float* hp[Q27_MAX_DEVICES] = {nullptr};   // per-card pinned staging
    float* acc = nullptr;                     // pinned, assembled result
    // Per-card arrival instrumentation. t_arr[g] is when card g finished its producer and entered
    // the collective; the spread across cards is real skew, not transport. Tallied between the two
    // barriers, which is the only race-free window: a card cannot reach the next collective's
    // t_arr write without first passing b2, and b2 cannot release until the tally is done.
    double t_arr[Q27_MAX_DEVICES] = {0};
    float  ss[Q27_MAX_DEVICES] = {0};    // per-slice sum of squares, published by b2
    // PRE-ENQUEUE. A bare launch at the post-barrier point costs 10.83 us of WALL (measured:
    // 512 empty launches per token cost 5.546 ms, -24.4%, 0/4) against a nominal 1.245 us launch
    // overhead, because the host must submit it onto a pipeline the barrier has just drained.
    // The submission itself can happen BEFORE the barrier, in the slack the barrier law says is
    // free, if the GPU waits on a flag instead of on the host. What is left on the critical path
    // is a flag store of a few hundred nanoseconds.
    unsigned* flag_h = nullptr;          // pinned: host writes it after the reduction
    unsigned* flag_d = nullptr;          // the same storage, addressed from the device
    unsigned  gen[Q27_MAX_DEVICES] = {0};
    int       pre_in[Q27_MAX_DEVICES] = {0};   // next layer's input norm already enqueued
    unsigned* done_h = nullptr;                // pinned: the staging copy kernel writes its generation
    unsigned* done_d[Q27_MAX_DEVICES] = {nullptr};   // device alias of done_h, taken per card
    unsigned  cgen[Q27_MAX_DEVICES] = {0};     // staging generation per card
    unsigned  pre_gen[Q27_MAX_DEVICES] = {0};  // generation of a copy that rode pre-barrier
    hipEvent_t ev_copy[Q27_MAX_DEVICES] = {nullptr};
    float* inv_h = nullptr;              // pinned: 1/rms per card, written by the host after b2
    float* inv_d = nullptr;              // the same storage, addressed from the device
    float* invN_h = nullptr; float* invN_d = nullptr;     // Q27_SPEC: [card][8] per-row 1/rms (pinned)
    float ssN[Q27_MAX_DEVICES][8] = {{0}};
    unsigned hidxN[Q27_MAX_DEVICES][8] = {{0}}; float hvalN[Q27_MAX_DEVICES][8] = {{0}};
    unsigned* cp_cnt[Q27_MAX_DEVICES] = {nullptr};   // device counter of the multi-block staging copy (per card)
    unsigned  cp_calls[Q27_MAX_DEVICES] = {0};
    // BATCHED COLLECTIVE #2 (prefill): one rendezvous reduces a whole Cch-position chunk, so the
    // fixed rendezvous cost (two barriers + flag round trip) amortises over Cch payloads while the
    // DMA grows linearly. ssb/inv_hb carry the per-position sum of squares / 1-over-rms.
    static const int BC = Q27_PF_CH;          // max positions per batched collective (= the chunk width)
    // SIZED BY THE STRIDE THAT IS ACTUALLY USED. These were Q27_MAX_DEVICES * 8 -- a leftover from when
    // the chunk width was 8 -- while every access is ssb[g * TpColl::BC + c] with BC = Q27_PF_CH. At
    // CH=32 with a 16-wide chunk that wrote 316 bytes past ssb2 straight into inv_hb2 and ev_c2[],
    // corrupting the event handles; at CH=64 it went further and hipEventRecord died with SIGBUS.
    // A pre-existing corruption present in frozen rung 3, not introduced by the wide work.
    float ssb[Q27_MAX_DEVICES * Q27_PF_CH] = {0};     // host-only
    // PIPELINED coll#2 (Q27_PF_PIPE): a second staging/result pair so the previous chunk's reduce
    // can run on the host while the GPU computes the current chunk's MLP.
    float* hp2[Q27_MAX_DEVICES] = {nullptr};
    float* hp2_d[Q27_MAX_DEVICES] = {nullptr};   // device aliases of hp2 (Q27_PF_KCOPY: in-stream staging copy)
    float* acc2 = nullptr;
    float  ssb2[Q27_MAX_DEVICES * Q27_PF_CH] = {0};
    float  inv_hb2[Q27_MAX_DEVICES * Q27_PF_CH] = {0};
    hipEvent_t ev_c2[Q27_MAX_DEVICES] = {nullptr};
    // Q27_PF_WIDE: TILE-WIDE collective #2. The chunk staging above is Q27_PF_CH positions wide; the
    // wide FFN's consumer is a whole Q27_PF_TSLOT tile, so it needs its own staging sized by the tile
    // and its own BC. Allocated only when the flag is on, so the rung-3 path is untouched.
    float* hp2w[Q27_MAX_DEVICES] = {nullptr};
    float* hp2w_d[Q27_MAX_DEVICES] = {nullptr};
    float* acc2w = nullptr;
    float  ssb2w[Q27_MAX_DEVICES * Q27_PF_TSLOT] = {0};
    float  inv_hb2w[Q27_MAX_DEVICES * Q27_PF_TSLOT] = {0};
    hipEvent_t ev_w0[Q27_MAX_DEVICES] = {nullptr};   // bracket the tile-wide FFN launches
    hipEvent_t ev_w1[Q27_MAX_DEVICES] = {nullptr};
    double t_wide_ffn[Q27_MAX_DEVICES] = {0};        // accumulated GPU time in the wide FFN, not enqueue time
    // The SAME bracket around the b13 MLP loop, so old and new FFN GPU time are comparable numbers from
    // the same binary on the same stream rather than a share quoted from an older profile.
    hipEvent_t ev_b0[2][Q27_MAX_DEVICES] = {{nullptr}};
    hipEvent_t ev_b1[2][Q27_MAX_DEVICES] = {{nullptr}};
    double t_b13_ffn[Q27_MAX_DEVICES] = {0};
    // ---- Q27_MLP_TIMELINE: INSTRUMENTATION ONLY, no scheduling change anywhere -------------------
    // Ping-pong event pairs so the bracket for round N is still intact when round N+1 reads it.
    hipEvent_t ev_pre[2][Q27_MAX_DEVICES] = {{nullptr}};
    hipEvent_t ev_ml[2][Q27_MAX_DEVICES] = {{nullptr}};
    double   r_sync[Q27_MAX_DEVICES] = {0};   // host: waiting for the staging D2H/copy to land
    double   r_b1[Q27_MAX_DEVICES] = {0};     // host: barrier #1 wait (cards arriving together?)
    double   r_loop[Q27_MAX_DEVICES] = {0};   // host: the reduction arithmetic itself
    double   r_b2[Q27_MAX_DEVICES] = {0};     // host: barrier #2 wait
    double   r_h2d[Q27_MAX_DEVICES] = {0};    // host: enqueue of the H2D of the result
    double   r_bubble[Q27_MAX_DEVICES] = {0}; // GPU-observed idle: last queued work end -> MLP start
    double   h_flush[Q27_MAX_DEVICES] = {0};  // host: whole flush_* block
    double   h_enq[Q27_MAX_DEVICES] = {0};    // host: enqueue of the MLP launches
    double   h_stage[Q27_MAX_DEVICES] = {0};  // host: enqueue of the result staging
    long long r_n[Q27_MAX_DEVICES] = {0};     // reductions
    long long r_pos[Q27_MAX_DEVICES] = {0};   // positions reduced (=> bytes = pos * Q27_HID * 4)
    long long r_bub_n[Q27_MAX_DEVICES] = {0};
    long long tl_round[Q27_MAX_DEVICES] = {0};
    long long r_bub_fail[Q27_MAX_DEVICES] = {0};
    // HOST CPU BREAKDOWN (Q27_MLP_TIMELINE): where the host's own time goes. If the host is the critical
    // path, summing these must approach the sweep wall.
    double h_proatt[Q27_MAX_DEVICES] = {0};   // prologue host work (norms, qkv, attn, o_proj launches)
    double h_norm[Q27_MAX_DEVICES] = {0};     // the post-attention norm launch
    double h_red1[Q27_MAX_DEVICES] = {0};     // collective #1 reduce, host arithmetic
    double h_red2[Q27_MAX_DEVICES] = {0};     // collective #2 reduces, host arithmetic (chunk + tile)
    double h_sync[Q27_MAX_DEVICES] = {0};     // hipEventSynchronize on the coll#1 staging event
    double h_loop[Q27_MAX_DEVICES] = {0};
    double h_pm[Q27_MAX_DEVICES] = {0};       // post_mlp total     // THE WHOLE CHUNK LOOP, both arms: if this approaches the
                                              // prefill wall the host is the span, and no GPU-side change helps
    bool ev_pair_ok = false;   // set by a controlled record/record/sync/elapsed at startup
    struct TLRec { int layer, tile, card; double beg, sync_end, red_end, h2d_end, ffn_end, stage_end, bubble; };
    TLRec tl[Q27_MAX_DEVICES][64] = {};
    int   tl_n[Q27_MAX_DEVICES] = {0};
    // PIPELINED coll#1 (Q27_PF_PIPE1): two staging/result sets so chunk c+1's prologue can be on the
    // GPU while the host reduces chunk c, and chunk c's tail reads its own parity result.
    static const int NB = 4;                  // collective #1 staging ring (lookahead depth + 1 <= NB)
    float* hpb[NB][Q27_MAX_DEVICES] = {{nullptr}};
    float* hpb_d[NB][Q27_MAX_DEVICES] = {{nullptr}};   // device aliases (Q27_PF_KCOPY)
    float* accb[NB] = {nullptr};
    float* accb_d[NB] = {nullptr};    // device aliases (the chunk tail reads them in place)
    float  ssbb[NB][Q27_MAX_DEVICES * Q27_PF_CH] = {{0}};
    float* inv_hbb[NB] = {nullptr};   // pinned: the chunk tail reads 1/rms per position in place
    hipEvent_t ev1[NB][Q27_MAX_DEVICES] = {{nullptr}};
    // Q27_PF_P2P: device ring of the o_proj/out_proj partials (the producers write the ring slot
    // directly; the fused collective reads own + peers), and the cross-card event chain:
    //   ev_rd[par][g]  = fused collective #1 done on card g (slot par safe to overwrite),
    //   ev_mix[i][g]   = wide down (mixer producer) done, chunk-index ring (3 deep),
    //   ev_mix_rd[i][g]= mixer collective done (mixer_slots safe to overwrite).
    // The waits replace the host barriers b1/b2: a card's kernel waits on the peer's record.
    float* part_d[NB][Q27_MAX_DEVICES] = {{nullptr}};
    hipEvent_t ev_rd[NB][Q27_MAX_DEVICES] = {{nullptr}};
    float* mixp[Q27_MAX_DEVICES] = {nullptr};          // peer device pointers to d.mixer_slots
    // Q27_NR_P2P: per-card device partial slots (2, alternating per site), local reduced rows + sumsq partials, and the
    // cross-card event chain (done = the slot holds this site's partial; rd = this card has finished reading the peers' slot)
    float* nrp_part[2][Q27_MAX_DEVICES] = {{nullptr}};
    float* nrp_red[Q27_MAX_DEVICES] = {nullptr};
    float* nrp_ssp[Q27_MAX_DEVICES] = {nullptr};
    hipEvent_t ev_nrp_done[2][Q27_MAX_DEVICES] = {{nullptr}};
    hipEvent_t ev_nrp_rd[2][Q27_MAX_DEVICES] = {{nullptr}};
    unsigned short* nrp_recv[2][Q27_MAX_DEVICES][Q27_MAX_DEVICES] = {{{nullptr}}};   // v2 push: [slot][owner][source] bf16 rows, allocated on the owner
    unsigned short* nrp_stage[2][Q27_MAX_DEVICES] = {{nullptr}};   // v3 (Q27_NR_P2P=3): local bf16 staging rows, then hipMemcpyPeerAsync (SDMA) into the peers' slots
    // Q27_NR_P2P=4 (LL, 2026-09-28): [slot][owner][source] 8-byte words {2 x bf16, seq}, allocated UNCACHED/fine-grained on the
    // owner (its spin-reads bypass L2; the source's peer stores bypass its L2 too). No events, no host barrier: the sequence
    // number inside every word is the only synchronization. nrp_err: 4 x u32 per card, pinned, written only on a spin timeout.
    unsigned* nrp_ll[Q27_LL_SLOTS][Q27_MAX_DEVICES][Q27_MAX_DEVICES] = {{{nullptr}}};
    unsigned* nrp_llstage[Q27_LL_SLOTS][Q27_MAX_DEVICES] = {{nullptr}};   // Q27_LL_SDMA: local coarse staging block (packed LL words) per slot
    int       nrp_prepacked[Q27_MAX_DEVICES] = {0};    // Q27_LL_EPI: the producer already wrote this site's LL words (skip the pack kernel)
    unsigned* nrp_err[Q27_MAX_DEVICES] = {nullptr};
    unsigned* nrp_cnt[Q27_MAX_DEVICES] = {nullptr};      // Q27_LL_LB: 8 row counters per card (device memory, zero at rest)
    unsigned  nrp_seq[Q27_MAX_DEVICES] = {0};     // per-card site sequence; every card runs the same site order (SPMD), so they agree
    hipEvent_t ev_mix[3][Q27_MAX_DEVICES] = {{nullptr}};
    hipEvent_t ev_mix_rd[3][Q27_MAX_DEVICES] = {{nullptr}};
    // PINNED, because the tails read it FROM THE DEVICE. A plain host array inside this struct
    // looked identical at the call site and faulted the GPU ("page not present" at 0x7fff...).
    float* inv_hb = nullptr;
    // SIDE-STREAM STAGING (Q27_COLL_SIDE=1): the D2H leg runs on its own stream so the MAIN
    // stream is never drained. hipStreamSynchronize(st) waits for everything queued -- including
    // flag-gated work that cannot run yet -- and leaves the GPU idle through the host reduce;
    // that idle is what costs every following kernel ~1.6x (measured: Q27_SYNC_TAX).
    hipStream_t side[Q27_MAX_DEVICES] = {nullptr};
    hipEvent_t  ev_side[Q27_MAX_DEVICES] = {nullptr};
    double skew_sum = 0; long skew_n = 0;
    long   last_cnt[Q27_MAX_DEVICES] = {0};
    double slot_skew[136] = {0}; long slot_n[136] = {0};
    TpBar  b1, b2;
    // head: one (value, index) pair per card, settled by a 4-way max
    float    hval[Q27_MAX_DEVICES] = {0};
    unsigned hidx[Q27_MAX_DEVICES] = {0};
    // per-card instrumentation
    // Fine-grained collective decomposition (Q27_COLL_PROF=1). The collective is 24% of the token
    // and 39 us/call has never been broken down. Five phases: D2H, wait at barrier 1, host reduce,
    // wait at barrier 2, H2D. Barrier waits are SKEW -- time this card spends idle waiting for a
    // peer -- and are charged separately from transport.
    double   c_d2h[Q27_MAX_DEVICES]  = {0};
    double   c_b1[Q27_MAX_DEVICES]   = {0};
    double   c_red[Q27_MAX_DEVICES]  = {0};
    double   c_b2[Q27_MAX_DEVICES]   = {0};
    double   c_h2d[Q27_MAX_DEVICES]  = {0};
    double   t_wall[Q27_MAX_DEVICES] = {0};   // per-card wall: the critical path is a REAL card
    double   t_coll[Q27_MAX_DEVICES] = {0};
    double   t_comp[Q27_MAX_DEVICES] = {0};
    long long n_coll[Q27_MAX_DEVICES] = {0};
    // ---- Q27_LS_SR: the sequential decode chain's publish counters and the two per-round exchanges ----
    std::atomic<long> sr_pub[Q27_MAX_DEVICES];   // blocks this card has published (monotonic; base = 4 * rounds)
    unsigned sr_dt[8] = {0u,0u,0u,0u,0u,0u,0u,0u};   // the head card's drafts for this round
    unsigned sr_nx[8] = {0u,0u,0u,0u,0u,0u,0u,0u};   // the head card's verified tokens
    int      sr_nacc = 0;
};


// reduce-scatter all-reduce of a 5120-float device buffer, in place on every card
// NEGATIVE CONTROL for the changing-input gate. Q27_TP_COLLGATE_BREAK=1 DELETES the first
// rendezvous, reproducing exactly the defect Window 2 found in allred3's 40.61 us variant: a
// worker may then consume a peer's host buffer before that peer's D2H has landed, i.e. read
// generation N-1 while producing N. The gate MUST fail under this flag. If it passes, the gate
// has no detection power and its clean runs mean nothing. Never set this outside the control.
static bool tp_break_barrier() {
    static const bool b = q27_env_flag("Q27_TP_COLLGATE_BREAK", false);
    return b;
}

// BF16 TRANSPORT. The collective is strictly serial -- allreduce, residual, norm, MLP, allreduce,
// residual -- so nothing overlaps it within a layer or across layers, and the only lever is the cost
// of one collective. P2P cannot supply that: 30 KB out per card is the measured floor for a 4-card
// all-reduce leaving the full vector everywhere, which lands level with host staging. So: bytes.
// The producers emit BF16 instead of FP32 and the payload halves, 20480 -> 10240 B each way.
// Partials are rounded to BF16 before summing; the residual stream is BF16 anyway, so the only loss
// is in the partials. Not bit-identical, by design; the quality gate decides. Q27_COLL_BF16=0 reverts.
static inline float q27_h_bf2f(unsigned short b) { union { unsigned u; float f; } c; c.u = (unsigned)b << 16; return c.f; }
static inline unsigned short q27_h_f2bf(float f) {
    union { float f; unsigned u; } c; c.f = f;
    return (unsigned short)((c.u + 0x7FFFu + ((c.u >> 16) & 1u)) >> 16);
}

static void tp_allreduce_bf16(TpColl& C, int g, unsigned short* dmixer, hipStream_t st) {
    const size_t B = (size_t)C.n * 2;                      // HALF of the fp32 payload
    unsigned short* hp = (unsigned short*)C.hp[g];
    CK(hipMemcpyAsync(hp, dmixer, B, hipMemcpyDeviceToHost, st));
    CK(hipStreamSynchronize(st));
    if (g_coll_pf) { const int slice = C.n / C.ndev + ((g == C.ndev - 1) ? C.n % C.ndev : 0), off = g * (C.n / C.ndev);
        for (int k = 0; k < C.ndev; ++k) { const unsigned short* p = (const unsigned short*)C.hp[k] + off;
            for (int i = 0; i < slice; i += 32) __builtin_prefetch(p + i); } }   // 32 halves = 64 B
    if (!tp_break_barrier() && g != g_bar_strand) C.b1.wait();

    const int slice = C.n / C.ndev + ((g == C.ndev - 1) ? C.n % C.ndev : 0);   // Q27_TP3: 5120 % 3 == 2 -> the last worker takes the remainder
    const int off   = g * (C.n / C.ndev);
    unsigned short* __restrict o = (unsigned short*)C.acc + off;
    const unsigned short* __restrict a0 = (const unsigned short*)C.hp[0] + off;
    const unsigned short* __restrict a1 = (const unsigned short*)C.hp[1] + off;
    const unsigned short* __restrict a2 = (const unsigned short*)C.hp[2] + off;
    const unsigned short* __restrict a3 = (const unsigned short*)C.hp[3] + off;
    for (int i = 0; i < slice; ++i)                        // sum in fp32, store bf16
        o[i] = q27_h_f2bf((q27_h_bf2f(a0[i]) + q27_h_bf2f(a1[i]))
                        + (q27_h_bf2f(a2[i]) + q27_h_bf2f(a3[i])));
    C.b2.wait();
    CK(hipMemcpyAsync(dmixer, (const unsigned short*)C.acc, B, hipMemcpyHostToDevice, st));
    CK(hipStreamSynchronize(st));
}

// 90 tok/s is an 11.11 ms/token budget. At 5.013 ms this collective alone is 45% of it before any
// compute runs, so it is an architectural target, not a cleanup item. These are the cheap
// host-staged fixes; the structural question (whether 129 separately synchronised collectives per
// token need to exist at all) is separate.
//
//  * STREAM-ORDERED copies. These were synchronous hipMemcpy with no stream, i.e. the NULL stream,
//    which carries implicit synchronisation against every other stream on the device. The layer
//    already drains its own stream before calling, so ordering never needed the null stream.
//  * VECTORISABLE host reduction. The inner loop was `for k in ndev` with a stride, which does not
//    vectorise. Four __restrict pointers summed pairwise does.
// The outbound staging leg, one place. Either the SDMA D2H (shipped) or the in-stream copy kernel.
// Returns the staging generation when the copy carries a done flag (Q27_COLL_DFLAG), else 0.
static inline unsigned tp_stage_out(TpColl& C, int g, const float* src, hipStream_t st) {
    if (g_coll_kcopy && g_coll_part[g]) {
        if (g_coll_dflag && C.done_d[g]) {
            const unsigned gen = ++C.cgen[g];
            if (g_copy_mb) {   // Q27_COPY_MB: 8-block staging copy (same done-flag contract)
                const unsigned target = 8u * (++C.cp_calls[g]);
                q27_copy_f4_flag_mb(g_coll_part[g], src, C.n, C.done_d[g] + g, gen, C.cp_cnt[g], target, 8, st);
            } else
            q27_copy_f4_flag(g_coll_part[g], src, C.n, C.done_d[g] + g, gen, st);
            g_kcopy_calls.fetch_add(1, std::memory_order_relaxed);
            return gen;
        }
        q27_copy_f4(g_coll_part[g], src, C.n, st);
        g_kcopy_calls.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    CK(hipMemcpyAsync(C.hp[g], src, (size_t)C.n * 4, hipMemcpyDeviceToHost, st));
    return 0;
}
// Wait for the staged partial to be in host memory: spin on the done flag when the copy carried
// one, otherwise the event. Mode 2 is the FIRING CONTROL: it does not wait, so the host reduce can
// read a stale partial and the tokens / collective gate MUST break.
static inline void tp_stage_wait(TpColl& C, int g, unsigned gen) {
    if (g_coll_dflag && gen) {
        // FIRING CONTROL (=2): wait only for the PREVIOUS generation, so the host may reduce a
        // stale but FINITE partial. Perturb, don't break: a no-wait control fed unfinished data
        // into the reduce and drove the engine into the NaN/denormal slow path (12 minutes for
        // 24 tokens, 09-10) -- a probe that destroys correctness cannot time anything.
        const unsigned want = (g_coll_dflag == 2) ? gen - 1u : gen;
        if (g_coll_dflag == 2) g_dflag_skips.fetch_add(1, std::memory_order_relaxed);
        const double t0 = tp_now(); unsigned spins = 0;
        while ((int)(__atomic_load_n(&C.done_h[g], __ATOMIC_ACQUIRE) - want) < 0) {
            if ((++spins & 1023u) == 0u && tp_now() - t0 > g_bar_deadline_ms) {
                std::fprintf(stderr, "FATAL: staging done-flag on card %d never reached generation %u "
                             "(saw %u) after %.0f ms; aborting rather than holding the GPUs.\n",
                             g, want, C.done_h[g], tp_now() - t0);
                std::abort();
            }
        }
        g_dflag_waits.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    CK(hipEventSynchronize(C.ev_copy[g]));
}

struct Q27CollectivePoint { int req, pos, site; double enter, staged, arrived, reduced, published, returned; };
static thread_local std::vector<Q27CollectivePoint> g_coll_points;
static thread_local int g_timeline_req = 0, g_timeline_pos = 0;
static thread_local bool g_timeline_sample = false;
static void tp_allreduce(TpColl& C, int g, float* dmixer, hipStream_t st, bool copied = false) {
    const size_t B = (size_t)C.n * 4;
    const bool prof = g_coll_prof || g_timeline_sample;
    double p0 = prof ? tp_now() : 0.0, p1, p2, p3, p4;
    if (!g_coll_zcout && !copied) {   // `copied` = the caller already enqueued it behind the
        if (g_coll_side) {            // staged on a side stream: the main stream keeps its queue
            CK(hipEventRecord(C.ev_side[g], st));
            CK(hipStreamWaitEvent(C.side[g], C.ev_side[g], 0));
            CK(hipMemcpyAsync(C.hp[g], dmixer, B, hipMemcpyDeviceToHost, C.side[g]));
            CK(hipEventRecord(C.ev_copy[g], C.side[g]));
            CK(hipEventSynchronize(C.ev_copy[g]));   // wait for the COPY, not the stream
        } else {
            tp_stage_out(C, g, dmixer, st);
            CK(hipStreamSynchronize(st));
        }
    }
    // REDUCE WORKING-SET PREFETCH, in barrier-1 SLACK. The host reduce below reads exactly the
    // four slice regions the D2H just wrote, and the D2H invalidated those 320 cache lines, so the
    // first pass pays cold-line latency on every one of them (measured 9.51 us/call reduce on the
    // critical card vs 5.59 on the fastest -- the same arithmetic, cold lines). Each card then
    // waits ~12 us at barrier 1 for the straggler's D2H: warm the lines during that wait. The
    // prefetch ISSUE cost is uniform pre-b1 work (~0.3 us); the DATA lands in slack for free.
    // If the pinned mapping turns out non-cacheable this is a no-op and the A/B shows it.
    if (g_coll_pf) {
        const int slice = C.n / C.ndev + ((g == C.ndev - 1) ? C.n % C.ndev : 0);   // Q27_TP3: last worker takes the remainder
        const int off   = g * (C.n / C.ndev);
        for (int k = 0; k < C.ndev; ++k) {
            const float* p = C.hp[k] + off;
            for (int i = 0; i < slice; i += 16) __builtin_prefetch(p + i);   // 16 floats = 64 B line
        }
    }
    if (prof) { p1 = tp_now(); C.c_d2h[g] += p1 - p0; }
    if (prof) C.t_arr[g] = p1;
    if (!tp_break_barrier() && g != g_bar_strand) C.b1.wait();
    if (prof) { p2 = tp_now(); C.c_b1[g] += p2 - p1; }
    if (prof && g == 0) {   // one strand tallies; all four arrivals are published by b1
        double lo = C.t_arr[0], hi = C.t_arr[0]; int late = 0;
        for (int k = 1; k < C.ndev; ++k) {
            if (C.t_arr[k] < lo) lo = C.t_arr[k];
            if (C.t_arr[k] > hi) { hi = C.t_arr[k]; late = k; }
        }
        C.skew_sum += hi - lo; C.skew_n++; C.last_cnt[late]++;
        const int slot = (int)(C.n_coll[0] % 136);
        C.slot_skew[slot] += hi - lo; C.slot_n[slot]++;
    }
    // CEILING PROBE. Two earlier forms were confounded and are recorded so they are not retried:
    // returning early skips b2, which is the same barrier object the per-token step sync uses, so
    // the generation counts desynchronise (6.7x slower -- a broken protocol, not a deleted phase);
    // and leaving C.acc unwritten runs the whole engine on garbage floats, whose denormals/NaNs
    // make the ARITHMETIC slow (also 6.7x). A probe that destroys correctness cannot time anything.
    // This form keeps b2, keeps values in range, and drops 3/4 of the reduction's memory traffic.

    const int slice = C.n / C.ndev + ((g == C.ndev - 1) ? C.n % C.ndev : 0);   // Q27_TP3: 5120 % 3 == 2 -> the last worker takes the remainder
    const int off   = g * (C.n / C.ndev);
    float* __restrict o = C.acc + (size_t)off;
    if (C.ndev == 4) {
        const float* __restrict a0 = C.hp[0] + off;
        const float* __restrict a1 = C.hp[1] + off;
        const float* __restrict a2 = C.hp[2] + off;
        const float* __restrict a3 = C.hp[3] + off;
        if (g_null_red) { for (int i = 0; i < slice; ++i) o[i] = a0[i]; }
        else if (g_rn_hostss) {
            // The sum of squares rides in the SAME traversal -- these floats are already in
            // registers, so the norm's reduction costs the host essentially nothing here, while
            // on the GPU it was forcing a one-block kernel onto the post-barrier critical path.
            // FOUR INDEPENDENT ACCUMULATORS. A single ss chain is a loop-carried FP dependency,
            // and without fast-math the compiler may not reassociate it -- which de-vectorises the
            // whole traversal, including the stores it was already doing. Measured cost of the
            // naive form: reduce 4.78 -> 7.24 us/call, on the critical path between the barriers.
            // Reassociation is free here; there is no bit-exactness requirement.
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
            int i = 0;
            for (; i + 3 < slice; i += 4) {
                const float v0 = (a0[i  ] + a1[i  ]) + (a2[i  ] + a3[i  ]);
                const float v1 = (a0[i+1] + a1[i+1]) + (a2[i+1] + a3[i+1]);
                const float v2 = (a0[i+2] + a1[i+2]) + (a2[i+2] + a3[i+2]);
                const float v3 = (a0[i+3] + a1[i+3]) + (a2[i+3] + a3[i+3]);
                o[i] = v0; o[i+1] = v1; o[i+2] = v2; o[i+3] = v3;
                s0 = fmaf(v0, v0, s0); s1 = fmaf(v1, v1, s1);
                s2 = fmaf(v2, v2, s2); s3 = fmaf(v3, v3, s3);
            }
            for (; i < slice; ++i) {
                const float v = (a0[i] + a1[i]) + (a2[i] + a3[i]);
                o[i] = v; s0 = fmaf(v, v, s0);
            }
            C.ss[g] = (s0 + s1) + (s2 + s3);
        }
        else for (int i = 0; i < slice; ++i) o[i] = (a0[i] + a1[i]) + (a2[i] + a3[i]);
    } else {
        for (int i = 0; i < slice; ++i) {
            float s = 0.0f;
            for (int k = 0; k < C.ndev; ++k) s += C.hp[k][off + i];
            o[i] = s;
        }
    }
    if (prof) { p3 = tp_now(); C.c_red[g] += p3 - p2; }
    C.b2.wait();
    // b2 is the publication barrier, so every card's slice sum is visible here and only here.
    if (g_rn_hostss) {
        float tot = 0.f;
        for (int k = 0; k < C.ndev; ++k) tot += C.ss[k];
        C.inv_h[g] = 1.0f / sqrtf(tot / (float)C.n + Q27_RMS_EPS);
    }
    if (prof) { p4 = tp_now(); C.c_b2[g] += p4 - p3; }
    // ZERO-COPY RETURN. The phase breakdown says the H2D is the LARGEST phase of the collective at
    // 13.8 us/call -- larger than the outbound D2H of the same 20 KB, and 35% of the whole
    // collective. Every card DMAs the full assembled vector back into VRAM and exactly one kernel
    // reads it, once. C.acc is hipHostMalloc'd, so it is already device-addressable; the consumer
    // can read it in place over PCIe instead. 129 calls/token x 13.8 us = 1.78 ms/token at stake.
    // Q27_COLL_ZC=0 restores the copy.
    if (!g_coll_zc) {
        CK(hipMemcpyAsync(dmixer, C.acc, B, hipMemcpyHostToDevice, st));
        CK(hipStreamSynchronize(st));
    }
    if (prof) C.c_h2d[g] += tp_now() - p4;
    if (g_timeline_sample) g_coll_points.push_back({g_timeline_req,g_timeline_pos,
        int(C.n_coll[g] % 129),p0,p1,p2,p3,p4,tp_now()});
}

// ---- Q27_SPEC: the NR-row collective (rows = consecutive positions of ONE sequence) ------------
// Same rendezvous as tp_allreduce on its shipped path (in-stream staging copy + done flag, b1,
// per-card slice reduce carrying the sum of squares, b2, 1/rms), for NR contiguous 5120-float
// rows. The pinned staging/result buffers are BC rows deep, so NR <= 4 needs no new allocation.
#define Q27_CP_NBLK 8
static inline unsigned tp_stage_out_nr(TpColl& C, int g, const float* src, int NR, hipStream_t st) {
    const unsigned gen = ++C.cgen[g];
    const unsigned target = (unsigned)Q27_CP_NBLK * (++C.cp_calls[g]);
    if (g_coll_nrbf16) q27_copy_bf16_flag_mb((unsigned short*)g_coll_part[g], src, C.n * NR, C.done_d[g] + g, gen, C.cp_cnt[g], target, Q27_CP_NBLK, st);
    else               q27_copy_f4_flag_mb(g_coll_part[g], src, C.n * NR, C.done_d[g] + g, gen, C.cp_cnt[g], target, Q27_CP_NBLK, st);
    return gen;
}
// Q27_COLL_AVX2 (2026-09-28): the bf16 NR site reduce, 8 lanes wide. Measured scalar cost was
// 0.5-0.8 ms/token (4-6 us/site at NR=1) sitting between b1 and b2 on every card, i.e. ALU-bound at
// scalar speed, not transport-bound as the 09-18 note assumed for NR=3. The element values o[i] keep
// the exact pairwise association (b0+b1)+(b2+b3); only the sum-of-squares accumulation order changes.
#if !defined(__HIP_DEVICE_COMPILE__)
#include <immintrin.h>
__attribute__((target("avx2,fma")))
static void tp_reduce4_bf16_avx2(const unsigned short* __restrict b0, const unsigned short* __restrict b1,
                                 const unsigned short* __restrict b2, const unsigned short* __restrict b3,
                                 float* __restrict o, int n, float* ss_out) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        #define Q27_BF16X8(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(p))), 16))
        const __m256 f0 = _mm256_add_ps(_mm256_add_ps(Q27_BF16X8(b0 + i), Q27_BF16X8(b1 + i)), _mm256_add_ps(Q27_BF16X8(b2 + i), Q27_BF16X8(b3 + i)));
        const __m256 f1 = _mm256_add_ps(_mm256_add_ps(Q27_BF16X8(b0 + i + 8), Q27_BF16X8(b1 + i + 8)), _mm256_add_ps(Q27_BF16X8(b2 + i + 8), Q27_BF16X8(b3 + i + 8)));
        #undef Q27_BF16X8
        _mm256_storeu_ps(o + i, f0); _mm256_storeu_ps(o + i + 8, f1);
        acc0 = _mm256_fmadd_ps(f0, f0, acc0); acc1 = _mm256_fmadd_ps(f1, f1, acc1);
    }
    float t[8]; _mm256_storeu_ps(t, _mm256_add_ps(acc0, acc1));
    float s = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
    for (; i < n; ++i) {
        const float v = (q27_h_bf2f(b0[i]) + q27_h_bf2f(b1[i])) + (q27_h_bf2f(b2[i]) + q27_h_bf2f(b3[i]));
        o[i] = v; s = fmaf(v, v, s);
    }
    *ss_out = s;
}
static int tp_coll_avx2_on() {
    static int v = -1;
    if (v < 0) { v = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && q27_env_int("Q27_COLL_AVX2", 1)) ? 1 : 0; }
    return v;
}
#else
static int tp_coll_avx2_on() { return 0; }
static void tp_reduce4_bf16_avx2(const unsigned short*, const unsigned short*, const unsigned short*, const unsigned short*, float*, int, float*) {}
#endif
static void tp_allreduce_nr(TpColl& C, int g, int NR) {
    ++g_coll_host_sites[g]; g_coll_nr_eff[g] = NR;   // runtime receipt: this site took the HOST path
    // PHASE DECOMPOSITION OF ONE HOST SITE (2026-09-17). Host-side wall is REAL here, unlike in the
    // enqueue-only stretches between sites: this path blocks. The split is what decides whether the
    // bf16 reduce is worth vectorising, and it is the only way to tell arithmetic from rendezvous:
    //   [1] b1     -- cross-card arrival skew (cards waiting for the slowest peer to stage)
    //   [2] reduce -- the host bf16->fp32 sum + sumsq over this card's 1/ndev slice  <-- AVX2 target
    //   [3] b2     -- second barrier: everyone's slice reduced before the norms are combined
    //   [4] invN   -- the rsqrt tail over ndev partial sumsqs
    // tp_stage_wait ([0]) is bracketed at the CALL SITES, because it blocks on the GPU finishing the
    // preceding segment and therefore prices that segment's execution, not the collective.
    const double t_a = tp_now();
    C.b1.wait();
    const double t_b = tp_now();
    const int slice = C.n / C.ndev + ((g == C.ndev - 1) ? C.n % C.ndev : 0);   // Q27_TP3: 5120 % 3 == 2 -> the last worker takes the remainder
    const int off   = g * (C.n / C.ndev);
    for (int r = 0; r < NR; ++r) {
        const size_t ro = (size_t)r * C.n + off;
        float* __restrict o = C.acc + ro;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        if (C.ndev == 4 || C.ndev == 3) {   // Q27_TP3: slot 3 is zero-filled by Q27_COLL_PAD, so the 4-slot sum is exact
            if (g_coll_nrbf16) {   // bf16 partials (packed by k_copy_bf16_flag_mb), summed in fp32, result fp32 in C.acc
                const unsigned short* __restrict b0 = (const unsigned short*)C.hp[0] + ro;  const unsigned short* __restrict b1 = (const unsigned short*)C.hp[1] + ro;
                const unsigned short* __restrict b2 = (const unsigned short*)C.hp[2] + ro;  const unsigned short* __restrict b3 = (const unsigned short*)C.hp[3] + ro;
                if (tp_coll_avx2_on()) { tp_reduce4_bf16_avx2(b0, b1, b2, b3, o, slice, &s0); C.ssN[g][r] = s0; continue; }
                for (int i = 0; i < slice; ++i) {
                    const float v = (q27_h_bf2f(b0[i]) + q27_h_bf2f(b1[i])) + (q27_h_bf2f(b2[i]) + q27_h_bf2f(b3[i]));
                    o[i] = v; s0 = fmaf(v, v, s0);
                }
                C.ssN[g][r] = s0;
                continue;
            }
            const float* __restrict a0 = C.hp[0] + ro;  const float* __restrict a1 = C.hp[1] + ro;
            const float* __restrict a2 = C.hp[2] + ro;  const float* __restrict a3 = C.hp[3] + ro;
            int i = 0;
            for (; i + 3 < slice; i += 4) {
                const float v0 = (a0[i  ] + a1[i  ]) + (a2[i  ] + a3[i  ]);
                const float v1 = (a0[i+1] + a1[i+1]) + (a2[i+1] + a3[i+1]);
                const float v2 = (a0[i+2] + a1[i+2]) + (a2[i+2] + a3[i+2]);
                const float v3 = (a0[i+3] + a1[i+3]) + (a2[i+3] + a3[i+3]);
                o[i] = v0; o[i+1] = v1; o[i+2] = v2; o[i+3] = v3;
                s0 = fmaf(v0, v0, s0); s1 = fmaf(v1, v1, s1);
                s2 = fmaf(v2, v2, s2); s3 = fmaf(v3, v3, s3);
            }
            for (; i < slice; ++i) { const float v = (a0[i] + a1[i]) + (a2[i] + a3[i]); o[i] = v; s0 = fmaf(v, v, s0); }
        } else {
            for (int i = 0; i < slice; ++i) {
                float v = 0.f;
                for (int k = 0; k < C.ndev; ++k) v += C.hp[k][ro + i];
                o[i] = v; s0 = fmaf(v, v, s0);
            }
        }
        C.ssN[g][r] = (s0 + s1) + (s2 + s3);
    }
    const double t_c = tp_now();
    C.b2.wait();
    const double t_d = tp_now();
    for (int r = 0; r < NR; ++r) {
        float tot = 0.f;
        for (int k = 0; k < C.ndev; ++k) tot += C.ssN[k][r];
        C.invN_h[g * 8 + r] = 1.0f / sqrtf(tot / (float)C.n + Q27_RMS_EPS);
    }
    g_collt[g][1] += t_b - t_a;      // b1 skew
    g_collt[g][2] += t_c - t_b;      // reduce arithmetic
    g_collt[g][3] += t_d - t_c;      // b2 skew
    g_collt[g][4] += tp_now() - t_d; // invN tail
    if(g_timeline_sample)g_coll_points.push_back({g_timeline_req,g_timeline_pos,
        int(C.n_coll[g] % 128),t_a,t_a,t_b,t_c,t_d,tp_now()});
}

// BATCHED ALL-REDUCE. Cch positions of the same layer share ONE rendezvous: the payload is
// Cch*5120 floats each way, and the two barriers plus the flag round trip are paid once instead
// of Cch times. The host reduce keeps the EXACT per-position arithmetic of tp_allreduce (each
// card sums its own 1/ndev slice, pairwise, in fp32) so the reduced vector is bit-identical to
// Cch separate collectives -- the shared cost is the rendezvous, not the arithmetic.
// src must be a device buffer holding Cch*C.n floats; on the zero-copy path the assembled result
// stays in C.acc, otherwise it is copied back into src (same contract as tp_allreduce).
static void tp_allreduce_b(TpColl& C, int g, float* src, int Cch, hipStream_t st) {
    const size_t B = (size_t)Cch * (size_t)C.n * 4;
    CK(hipMemcpyAsync(C.hp[g], src, B, hipMemcpyDeviceToHost, st));
    CK(hipStreamSynchronize(st));
    if (!tp_break_barrier() && g != g_bar_strand) C.b1.wait();

    const int slice = C.n / C.ndev + ((g == C.ndev - 1) ? C.n % C.ndev : 0);   // Q27_TP3: 5120 % 3 == 2 -> the last worker takes the remainder
    const int off   = g * (C.n / C.ndev);
    const float* __restrict a0 = C.hp[0];
    const float* __restrict a1 = C.hp[1];
    const float* __restrict a2 = C.hp[2];
    const float* __restrict a3 = C.hp[3];
    for (int c = 0; c < Cch; ++c) {
        const size_t base = (size_t)c * (size_t)C.n;
        float* __restrict o = C.acc + base;
        // + off: card g assembles ITS OWN 1/ndev slice out of all four stagings (C.acc is shared
        // pinned memory, and every card fills only its slice). Dropping this offset made every card
        // re-assemble slice 0 into all four slices.
        const float* __restrict b0 = a0 + base + off;
        const float* __restrict b1 = a1 + base + off;
        const float* __restrict b2 = a2 + base + off;
        const float* __restrict b3 = a3 + base + off;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        int i = 0;
        for (; i + 3 < slice; i += 4) {
            const float v0 = (b0[i  ] + b1[i  ]) + (b2[i  ] + b3[i  ]);
            const float v1 = (b0[i+1] + b1[i+1]) + (b2[i+1] + b3[i+1]);
            const float v2 = (b0[i+2] + b1[i+2]) + (b2[i+2] + b3[i+2]);
            const float v3 = (b0[i+3] + b1[i+3]) + (b2[i+3] + b3[i+3]);
            o[off+i] = v0; o[off+i+1] = v1; o[off+i+2] = v2; o[off+i+3] = v3;
            s0 = fmaf(v0, v0, s0); s1 = fmaf(v1, v1, s1);
            s2 = fmaf(v2, v2, s2); s3 = fmaf(v3, v3, s3);
        }
        for (; i < slice; ++i) {
            const float v = (b0[i] + b1[i]) + (b2[i] + b3[i]);
            o[off+i] = v; s0 = fmaf(v, v, s0);
        }
        C.ssb[g * TpColl::BC + c] = (s0 + s1) + (s2 + s3);
    }
    C.b2.wait();
    if (g_rn_hostss) {
        for (int c = 0; c < Cch; ++c) {
            float tot = 0.f;
            for (int k = 0; k < C.ndev; ++k) tot += C.ssb[k * TpColl::BC + c];
            C.inv_hb[g * TpColl::BC + c] = 1.0f / sqrtf(tot / (float)C.n + Q27_RMS_EPS);
        }
    }
    if (!g_coll_zc) {
        CK(hipMemcpyAsync(src, C.acc, B, hipMemcpyHostToDevice, st));
        CK(hipStreamSynchronize(st));
    }
}

// Deferred half of the pipelined chunk collective: the D2H into hp2 was enqueued earlier (with
// ev_c2 behind it); this reduces hp2 -> acc2 with the same per-position arithmetic as
// tp_allreduce_b and leaves 1/rms per position in inv_hb2. Caller syncs ev_c2 first.
// bc is the staging stride (positions per staging row). It is TpColl::BC for every chunk path; the
// wide tile path passes Q27_PF_TSLOT because its collective covers a whole tile, not a chunk.
// ---- SAFE CROSS-CARD COPY. 2026-09-18. -----------------------------------------------------
// The layer-split ring handoff called hipMemcpyPeerAsync unconditionally and HIP dereferenced a
// null inside it: SIGSEGV at 0x18, backtrace hipMemcpyPeerAsync <- run_tp's worker lambda.
// hipMemcpyPeerAsync requires PEER ACCESS to be enabled for the (src,dst) pair. The engine enables
// it in exactly two places and BOTH are gated -- the NR_P2P block additionally by `ndev == 4`, and
// Q27_PF_P2P is off in the shipped recipe. So on any arm where neither gate fired, the ring handed
// HIP a pair it had never mapped.
// This does the peer copy when the pair is actually mapped and falls back to an ordinary
// device-to-device async copy otherwise (the runtime stages that itself and needs no mapping).
// It also refuses to call HIP with a null pointer or stream and says which one was null.
static int q27_xcopy(void* dst, int ddev, const void* src, int sdev, size_t bytes,
                     hipStream_t st, const char* tag) {
    if (!dst || !src || !st) {
        std::fprintf(stderr, "Q27_XCOPY %s: NULL arg (dst=%p src=%p stream=%p) d%d<-d%d %zu B\n",
                     tag, dst, src, (void*)st, ddev, sdev, bytes);
        std::fflush(stderr); return 0;
    }
    if (ddev == sdev) return hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, st) == hipSuccess;
    static int peer_ok[Q27_MAX_DEVICES][Q27_MAX_DEVICES];
    static std::atomic<int> probed{0};
    static std::mutex probe_mu;
    if (!probed.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(probe_mu);
        if (!probed.load(std::memory_order_relaxed)) {
            // CLAUSE 9: this helper touches other cards, so it MUST restore the caller's device on
            // every exit path. Not doing so left the worker thread pointed at the wrong device and
            // the next HIP call faulted inside the runtime.
            int caller = 0; hipGetDevice(&caller);
            int nd = 0; hipGetDeviceCount(&nd);
            if (nd > Q27_MAX_DEVICES) nd = Q27_MAX_DEVICES;
            for (int i = 0; i < nd; ++i) for (int j = 0; j < nd; ++j) {
                if (i == j) { peer_ok[i][j] = 1; continue; }
                int can = 0;
                if (hipDeviceCanAccessPeer(&can, i, j) != hipSuccess || !can) { peer_ok[i][j] = 0;
                    std::fprintf(stderr, "Q27_XCOPY probe: %d->%d canAccessPeer=%d => unmapped\n", i, j, can);
                    continue; }
                hipSetDevice(i);
                hipError_t pe = hipDeviceEnablePeerAccess(j, 0);
                peer_ok[i][j] = (pe == hipSuccess || pe == hipErrorPeerAccessAlreadyEnabled);
                std::fprintf(stderr, "Q27_XCOPY probe: %d->%d can=%d enable=%s => %s\n", i, j, can,
                             hipGetErrorName(pe), peer_ok[i][j] ? "MAPPED" : "unmapped");
                (void)hipGetLastError();
            }
            hipSetDevice(caller);   // RESTORE, unconditionally
            std::fprintf(stderr, "Q27_XCOPY: peer map probed, %d device(s), caller device %d restored\n", nd, caller);
            std::fflush(stderr);
            probed.store(1, std::memory_order_release);
        }
    }
    if (ddev >= 0 && ddev < Q27_MAX_DEVICES && sdev >= 0 && sdev < Q27_MAX_DEVICES && peer_ok[ddev][sdev])
        return hipMemcpyPeerAsync(dst, ddev, src, sdev, bytes, st) == hipSuccess;
    // NO PEER MAPPING: stage through the host EXPLICITLY. hipMemcpyDeviceToDevice and even
    // hipMemcpyDefault still fault across an unmapped pair on this runtime (SIGSEGV inside
    // hipMemcpyAsync at +0x18). Two single-device copies need no peer mapping and no cross-device
    // pointer resolution at all, so this cannot fault on an unmapped pair. It is synchronous and
    // therefore slower -- correctness first; this is the fallback, not the fast path.
    // CLAUSE 9: restore the caller's device on EVERY exit path.
    {
        int caller = 0; hipGetDevice(&caller);
        // The caller queued the producer event on st. A synchronous copy on HIP's
        // default stream does not honor a nonblocking stream's pending wait.
        // Finish that dependency before the host reads the source allocation.
        if (hipStreamSynchronize(st) != hipSuccess) {
            std::fprintf(stderr, "Q27_XCOPY %s: handoff stream wait failed\n", tag);
            return 0;
        }
        void* hbuf = nullptr;
        if (hipHostMalloc(&hbuf, bytes, hipHostMallocDefault) != hipSuccess) {
            std::fprintf(stderr, "Q27_XCOPY %s: host staging alloc of %zu B failed\n", tag, bytes);
            std::fflush(stderr); hipSetDevice(caller); return 0;
        }
        int ok = 1;
        if (hipSetDevice(sdev) != hipSuccess) ok = 0;
        if (ok && hipMemcpy(hbuf, src, bytes, hipMemcpyDeviceToHost) != hipSuccess) ok = 0;
        if (ok && hipSetDevice(ddev) != hipSuccess) ok = 0;
        if (ok && hipMemcpy(dst, hbuf, bytes, hipMemcpyHostToDevice) != hipSuccess) ok = 0;
        hipHostFree(hbuf);
        hipSetDevice(caller);
        if (!ok) { std::fprintf(stderr, "Q27_XCOPY %s: host-staged d%d<-d%d %zu B failed\n", tag, ddev, sdev, bytes); std::fflush(stderr); }
        static int said = 0;
        if (ok && !said) { said = 1; std::fprintf(stderr, "Q27_XCOPY: pair d%d<-d%d has no peer mapping; using host staging\n", ddev, sdev); std::fflush(stderr); }
        return ok;
    }
}
static void tp_reduce_bx_count() { g_host_coll_calls.fetch_add(1, std::memory_order_relaxed); }
static void tp_reduce_bx(TpColl& C, int g, int Cch, float* const* hp, float* acc, float* ssb, float* inv_hb,
                         int bc = TpColl::BC) {
    tp_reduce_bx_count();
    const bool tl = (C.ev_pre[0][g] != nullptr);
    const double q0 = tl ? tp_now() : 0.0;
    if (!tp_break_barrier() && g != g_bar_strand) C.b1.wait();
    const double q1 = tl ? tp_now() : 0.0;
    const int slice = C.n / C.ndev + ((g == C.ndev - 1) ? C.n % C.ndev : 0);   // Q27_TP3: 5120 % 3 == 2 -> the last worker takes the remainder
    const int off   = g * (C.n / C.ndev);
    const float* __restrict a0 = hp[0];
    const float* __restrict a1 = hp[1];
    const float* __restrict a2 = hp[2];
    const float* __restrict a3 = hp[3];
    for (int c = 0; c < Cch; ++c) {
        const size_t base = (size_t)c * (size_t)C.n;
        float* __restrict o = acc + base;
        const float* __restrict b0 = a0 + base + off;
        const float* __restrict b1 = a1 + base + off;
        const float* __restrict b2 = a2 + base + off;
        const float* __restrict b3 = a3 + base + off;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        int i = 0;
        for (; i + 3 < slice; i += 4) {
            const float v0 = (b0[i  ] + b1[i  ]) + (b2[i  ] + b3[i  ]);
            const float v1 = (b0[i+1] + b1[i+1]) + (b2[i+1] + b3[i+1]);
            const float v2 = (b0[i+2] + b1[i+2]) + (b2[i+2] + b3[i+2]);
            const float v3 = (b0[i+3] + b1[i+3]) + (b2[i+3] + b3[i+3]);
            o[off+i] = v0; o[off+i+1] = v1; o[off+i+2] = v2; o[off+i+3] = v3;
            s0 = fmaf(v0, v0, s0); s1 = fmaf(v1, v1, s1);
            s2 = fmaf(v2, v2, s2); s3 = fmaf(v3, v3, s3);
        }
        for (; i < slice; ++i) {
            const float v = (b0[i] + b1[i]) + (b2[i] + b3[i]);
            o[off+i] = v; s0 = fmaf(v, v, s0);
        }
        ssb[g * bc + c] = (s0 + s1) + (s2 + s3);
    }
    C.b2.wait();
    const double q2 = tl ? tp_now() : 0.0;
    for (int c = 0; c < Cch; ++c) {
        float tot = 0.f;
        for (int k = 0; k < C.ndev; ++k) tot += ssb[k * bc + c];
        inv_hb[g * bc + c] = 1.0f / sqrtf(tot / (float)C.n + Q27_RMS_EPS);
    }
    if (tl) { C.r_b1[g] += q1 - q0; C.r_loop[g] += q2 - q1; C.r_b2[g] += tp_now() - q2;
              C.r_n[g]++; C.r_pos[g] += Cch; }
}
static void tp_reduce_b2(TpColl& C, int g, int Cch) { tp_reduce_bx(C, g, Cch, C.hp2, C.acc2, C.ssb2, C.inv_hb2); }

// per-card scratch for TP. The residual/norm/activation buffers stay full width (they are a few
// MB in total and every card holds the whole vector); only the STATE is shard-sized.
// rndev = the REAL card count. ndev = the allocation width divisor, which layer-split sets to 1
// so the scratch and the OWNED layers are full width. ls distinguishes the two regimes.
static void dev_alloc_tp(Dev& d, int id, int ndev, int ctx, int rndev, bool ls) {
    d.id = id; d.slots = 1;
    CK(hipSetDevice(id));
    // Avoid implicit default-stream dependencies between co-resident logical
    // workers. Dependencies on this worker's own data are explicit.
    CK(hipStreamCreateWithFlags(&d.stream, hipStreamNonBlocking));
    for (int k = 0; k < 4; ++k) { CK(hipStreamCreateWithFlags(&d.hs[k], hipStreamNonBlocking)); CK(hipEventCreateWithFlags(&d.ev_join[k], hipEventDisableTiming)); }
    CK(hipEventCreateWithFlags(&d.ev_fork, hipEventDisableTiming));
    CK(hipEventCreateWithFlags(&d.ev_state_ready, hipEventDisableTiming));
    CK(hipStreamCreateWithFlags(&d.s_cap, hipStreamNonBlocking)); CK(hipEventCreateWithFlags(&d.ev_cap, hipEventDisableTiming));
    { int lo = 0, hi = 0; CK(hipDeviceGetStreamPriorityRange(&lo, &hi));
      CK(hipStreamCreateWithPriority(&d.q8hi, hipStreamNonBlocking, hi));
      CK(hipStreamCreateWithPriority(&d.q8lo, hipStreamNonBlocking, lo)); }
    if (!q27_rb_init(d.id)) { std::fprintf(stderr, "rocBLAS init failed on device %d\n", d.id); std::exit(1); }
    const bool dbg_alloc = q27_env_flag("Q27_DEVALLOC_DBG", false);
    // "Q27_HIP out of memory at q27_main.cpp:<this line>" names the SHARED lambda, so every
    // allocation in this function reports the same file:line and identifies nothing. On failure
    // say what was asked for and what was left -- that is the difference between a budget you can
    // act on and a line number you cannot.
    size_t a_total = 0;
    auto A_impl = [&](void** p, size_t b, const char* name) {
        const hipError_t e = hipMalloc(p, b);
        if (e != hipSuccess) {
            size_t fr = 0, tt = 0; hipMemGetInfo(&fr, &tt);
            std::fprintf(stderr, "Q27_DEVALLOC_FAIL card %d requested=%zu (%.3f GiB) free=%.3f GiB "
                                 "of %.3f GiB; granted so far=%.3f GiB\n",
                         id, b, (double)b / 1073741824.0, (double)fr / 1073741824.0,
                         (double)tt / 1073741824.0, (double)a_total / 1073741824.0);
            std::fflush(stderr);
            CK(e);
        }
        a_total += b;
        CK(hipMemset(*p, 0, b));
        if (dbg_alloc) std::fprintf(stderr,"Q27_ALLOC logical=%d ctx=%d name=%s bytes=%zu\n",id,ctx,name,b); };
#define A(P,B) A_impl(P,B,#P)
    const size_t PA_ELEMS = (size_t)((Q27_INTER > Q27_VOCAB) ? Q27_INTER : Q27_VOCAB);
    A((void**)&d.hidden, Q27_HID * 2);      A((void**)&d.norm, Q27_HID * 2);
    A((void**)&d.mixer,  Q27_HID * 4);      A((void**)&d.xq, Q27_INTER);
    A((void**)&d.rnpart, 64);               // 5 floats, padded
    A((void**)&d.xs, (Q27_INTER / 16) * 4);
    A((void**)&d.pa, PA_ELEMS * 4);         A((void**)&d.pb, PA_ELEMS * 4);
    A((void**)&d.act, (size_t)Q27_INTER * 2);
    A((void**)&d.qkv, Q27_GDN_QKV * 2);     A((void**)&d.zbuf, Q27_GDN_Z * 2);
    A((void**)&d.ab, Q27_GDN_VH * 2);       A((void**)&d.bb, Q27_GDN_VH * 2);
    A((void**)&d.mix6, Q27_GDN_Z * 2);
    A((void**)&d.qkva, (Q27_QROWS + 2 * Q27_KVROWS) * 2);
    A((void**)&d.qh, Q27_OROWS * 2);
    A((void**)&d.dtok, 4);
    A((void**)&d.dval, 4);
    A((void**)&d.gdn_scr, 12 * 16 * 4);    // zeroed: counters and generation flags start clean
    CK(hipHostMalloc(&d.pend_stage, Q27_HID * 4, hipHostMallocDefault));   // pinned, fp32 wide
    // A tile-slot buffer can be written with M = Qch (chunk width) by the attention chunk path AND with
    // M = Ct (tile width) by the GDN tile path, so size by whichever is larger. Sizing by TSLOT alone
    // was a live out-of-bounds write for chw > Q27_PF_TSLOT (GPU page fault at chw=512).
    #define Q27_PF_SLOTS (Q27_PF_CH > Q27_PF_TSLOT ? Q27_PF_CH : Q27_PF_TSLOT)
    // Q27_LAYER_SPLIT runs the chunk kernels at the TILE width (256) on one FULL-width rank.
    // The chunk buffers below were sized for the TP geometry (32 positions x width/ndev), so the
    // LS sweep wrote ~4x past every one of them; positions 32..255 read spilled garbage (NaN
    // finals, argmax token 0). pf_chw/pf_div widen them only under layer-split.
    const bool pf_ls = q27_env_flag("Q27_LAYER_SPLIT", true);
    const int  pf_chw = pf_ls ? Q27_PF_TSLOT : Q27_PF_CH;
    const int  pf_div = pf_ls ? 1 : ndev;
    A((void**)&d.xq_slots, (size_t)Q27_PF_CH * Q27_HID);
    A((void**)&d.xs_slots, (size_t)Q27_PF_CH * (Q27_HID / 16) * 4);
    // The lm_head SHARD's rows. ndev is the ALLOCATION divisor, which layer-split sets to 1 so the
    // scratch and the owned layers come out full width -- but Q27_VOCAB/1 is 4x the lm_head's actual
    // row count (62080), and the batched head only ever writes Q27_PF_CH rows of it. Size by the REAL
    // card count: 190 MB/card back at the 262K layer-split edge, where every card is the binding one.
    // Only the card that OWNS the last layer runs the head in the layer-split sweep; every other card's
    // pa_slots is 63 MB nobody writes. Ownership is derived exactly as q27_upload_ls and the sweep do.
    // 2026-09-18: was an inline copy of the ownership modulo. It is now the oracle, so
    // Q27_LS_OWN's uneven quotas move the head with the last layer instead of leaving
    // pa_slots on a card that no longer owns it.
    const bool pa_head = !ls || (q27_ls_owned(Q27_LAYERS - 1, id, rndev) != 0);
    A((void**)&d.pa_slots, pa_head ? (size_t)Q27_PF_CH * (size_t)q27_tp_voc_max() * 4 : 16);
    A((void**)&d.xq1_slots, (size_t)pf_chw * Q27_HID);
    A((void**)&d.xs1_slots, (size_t)pf_chw * (Q27_HID / 16) * 4);
    A((void**)&d.xq2_slots, (size_t)pf_chw * mlp_len_div(id, pf_div));
    A((void**)&d.xs2_slots, (size_t)pf_chw * (mlp_len_div(id, pf_div) / 16) * 4);
    A((void**)&d.mixq_slots, (size_t)pf_chw * (Q27_GDN_Z / pf_div));
    A((void**)&d.mixs_slots, (size_t)pf_chw * ((Q27_GDN_Z / pf_div) / 16) * 4);
    A((void**)&d.pa_mlp, (size_t)pf_chw * mlp_len_div(id, pf_div) * 4);
    A((void**)&d.pb_mlp, (size_t)pf_chw * mlp_len_div(id, pf_div) * 4);
    A((void**)&d.mixer_slots, (size_t)pf_chw * Q27_HID * 4);
    if (g_tpsr && !ls) {   // Q27_TPSR: one layer's per-row int8 MLP shard (~67 MB at ndev=4) + its per-row scales
        const size_t nr = (size_t)mlp_len(id);
        A((void**)&d.tpsr_mw, 2 * nr * Q27_HID + (size_t)Q27_HID * nr);
        A((void**)&d.tpsr_ms, (2 * nr + Q27_HID) * 4);
        if (q27_env_flag("Q27_TPSR_NV", false))
            A((void**)&d.tpsr_nv_s64, nr * Q27_HID / 64 * 4);
        // prefill v2 projection transient: max(attn q|k|v+o, gdn in_qkv|in_z+out) int8 + per-row scales
        A((void**)&d.tpsr_pw, ((size_t)(Q27_GDN_QKV + Q27_GDN_Z) / ndev) * Q27_HID + (size_t)Q27_HID * (Q27_OROWS / ndev));
        A((void**)&d.tpsr_ps, ((size_t)(Q27_GDN_QKV + Q27_GDN_Z) / ndev + Q27_HID) * 4);
    }
    A((void**)&d.mix2, (size_t)pf_chw * Q27_HID * 4);
    A((void**)&d.mix3, (size_t)pf_chw * Q27_HID * 4);
    A((void**)&d.mix4, (size_t)pf_chw * Q27_HID * 4);
    A((void**)&d.qkv2a, (size_t)pf_chw * Q27_GDN_QKV * 4);
    A((void**)&d.qkv2b, (size_t)pf_chw * Q27_GDN_QKV * 4);
    A((void**)&d.zb2a,  (size_t)pf_chw * Q27_GDN_Z * 4);
    A((void**)&d.zb2b,  (size_t)pf_chw * Q27_GDN_Z * 4);
    A((void**)&d.op2a,  (size_t)pf_chw * Q27_HID * 4);
    A((void**)&d.op2b,  (size_t)pf_chw * Q27_HID * 4);
    if (g_pf_wide) {   // Q27_PF_WIDE: the tile-wide MLP chain (~17 MB), separate from every chunk buffer
        A((void**)&d.xq1_t, (size_t)Q27_PF_TSLOT * Q27_HID);
        A((void**)&d.xs1_t, (size_t)Q27_PF_TSLOT * (Q27_HID / 16) * 4);
        A((void**)&d.pa_t,  (size_t)Q27_PF_TSLOT * mlp_len(id) * 4);
        A((void**)&d.pb_t,  (size_t)Q27_PF_TSLOT * mlp_len(id) * 4);
        A((void**)&d.xq2_t, (size_t)Q27_PF_TSLOT * mlp_len(id));
        A((void**)&d.xs2_t, (size_t)Q27_PF_TSLOT * (mlp_len(id) / 16) * 4);
        A((void**)&d.mix_t, (size_t)Q27_PF_TSLOT * Q27_HID * 4);
    }
    // Q27_PF_TSLOT positions per tile: the large-M prefill tile (PF_TILE) needs the norm /
    // activation / projection slots to span a whole tile, not one 8-position chunk.
    A((void**)&d.xqin_slots, (size_t)Q27_PF_SLOTS * Q27_HID);
    A((void**)&d.xsin_slots, (size_t)Q27_PF_SLOTS * (Q27_HID / 16) * 4);
    A((void**)&d.qkva_slots, (size_t)pf_chw * (Q27_QROWS / pf_div + 2 * (Q27_KVROWS / pf_div)) * 2);
    A((void**)&d.norm_slots, (size_t)Q27_PF_SLOTS * Q27_HID * 2);
    A((void**)&d.qkv_slots, (size_t)pf_chw * (Q27_GDN_QKV / pf_div) * 2);
    A((void**)&d.zbuf_slots, (size_t)pf_chw * (Q27_GDN_Z / pf_div) * 2);
    A((void**)&d.qh_slots, (size_t)pf_chw * (Q27_OROWS / pf_div) * 4);
    // Q27_LS_Q8 producer/consumer buffers (full width: the layer-split rank)
    A((void**)&d.qkva8, (size_t)pf_chw * (Q27_QROWS + 2 * Q27_KVROWS));
    A((void**)&d.qkvas, (size_t)pf_chw * ((Q27_QROWS + 2 * Q27_KVROWS) / 16) * 4);
    A((void**)&d.qkv8,  (size_t)pf_chw * Q27_GDN_QKV);
    A((void**)&d.qkvs,  (size_t)pf_chw * (Q27_GDN_QKV / 16) * 4);
    A((void**)&d.z8,    (size_t)pf_chw * Q27_GDN_Z);
    A((void**)&d.zs,    (size_t)pf_chw * (Q27_GDN_Z / 16) * 4);
    A((void**)&d.a32,   (size_t)Q27_PF_SLOTS * Q27_GDN_VH * 4);
    A((void**)&d.b32,   (size_t)Q27_PF_SLOTS * Q27_GDN_VH * 4);
    A((void**)&d.hid32c,(size_t)pf_chw * Q27_HID * 4);
    { size_t mlb = 0; const size_t ob = q27_attn_chunk_scratch_bytes(pf_div, pf_chw, &mlb); A((void**)&d.attn_pob, ob); A((void**)&d.attn_pml, mlb); d.attn_pob_bytes = ob; }
    // Q27_PF_MK16 stages one projection's weights as fp16 pairs here; with the flag off (the default)
    // nothing ever touches it, so do not spend 105 MB/card of the 262K layer-split budget on it.
    A((void**)&d.w16_scratch, g_pf_mk16 ? (size_t)(Q27_GDN_QKV / ndev) * Q27_HID * 2 : 16);
    A((void**)&d.ab_tile, (size_t)Q27_PF_SLOTS * (Q27_GDN_VH / pf_div) * 2);
    A((void**)&d.bb_tile, (size_t)Q27_PF_SLOTS * (Q27_GDN_VH / pf_div) * 2);
    // mix_tile is written by the ATTENTION chunk path with M = Qch <= Q27_PF_CH, and by the GDN tile path
    // with Ct <= Q27_PF_TSLOT, so it must be sized by whichever is larger. Sizing it by TSLOT alone was a
    // live out-of-bounds write for any chw > Q27_PF_TSLOT (GPU page fault at chw=512).
    A((void**)&d.mix_tile, (size_t)(Q27_PF_CH > Q27_PF_TSLOT ? Q27_PF_CH : Q27_PF_TSLOT) * (Q27_GDN_Z / pf_div) * 2);
    A((void**)&d.mixq_tile, (size_t)Q27_PF_SLOTS * (Q27_GDN_Z / pf_div));
    A((void**)&d.mixs_tile, (size_t)Q27_PF_SLOTS * ((Q27_GDN_Z / pf_div) / 16) * 4);
    A((void**)&d.part_slots, (size_t)pf_chw * Q27_HID * 4);   // per-position partials for batched coll#1
    {   // Q27_LS_Q8 chunk scratch sets: set 0 = the buffers above, set 1 = a second allocation (two chunks in flight)
        Q8Scr& s0 = d.q8s[0];
        s0.xqin = d.xqin_slots; s0.xsin = d.xsin_slots; s0.qkva8 = d.qkva8; s0.qkvas = d.qkvas; s0.qh = d.qh_slots;
        s0.pob = d.attn_pob; s0.pml = d.attn_pml; s0.mixq_slots = d.mixq_slots; s0.mixs_slots = d.mixs_slots;
        s0.pob_bytes = d.attn_pob_bytes;
        s0.qkv8 = d.qkv8; s0.qkvs = d.qkvs; s0.z8 = d.z8; s0.zs = d.zs; s0.a32 = d.a32; s0.b32 = d.b32;
        s0.mixq_tile = d.mixq_tile; s0.mixs_tile = d.mixs_tile; s0.part = d.part_slots; s0.xq1 = d.xq1_slots; s0.xs1 = d.xs1_slots;
        s0.pa = d.pa_mlp; s0.pb = d.pb_mlp; s0.xq2 = d.xq2_slots; s0.xs2 = d.xs2_slots; s0.mixer = d.mixer_slots; s0.hid32c = d.hid32c;
        // Q27_LS_ONE_SET: ONE chunk-scratch set instead of two. The second set exists so two ring
        // jobs can be in flight; at ctx=262144 it costs 680 MB/card (pob 403 MB + acc 178 MB + the
        // rest), and 680 MB is the difference between the layer-split product fitting on the head
        // card -- which carries the full lm_head on top of everything else -- and not fitting. Set 1
        // ALIASES set 0 rather than being dropped, so every existing `q8s[k & 1]` reference stays
        // valid and the ring simply serializes: correct at any width, just without the overlap.
        const bool ls_one = ls && q27_env_flag("Q27_LS_ONE_SET", false);
        // NOTE: the alias happens AFTER the stage-3 loop below. Copying set 0 here would capture it
        // while its rocBLAS scratch (acc/xqt/xst/...) is still null, and every odd ring chunk would
        // then decline the input norm ("Q27_LS_Q8: input norm tok declined") the moment the rocBLAS
        // path is live -- which is exactly the fast int8 Tensile configuration.
        if (!ls_one) {
        Q8Scr& s1 = d.q8s[1];
        A((void**)&s1.xqin, (size_t)Q27_PF_SLOTS * Q27_HID);
        A((void**)&s1.xsin, (size_t)Q27_PF_SLOTS * (Q27_HID / 16) * 4);
        A((void**)&s1.qkva8, (size_t)pf_chw * (Q27_QROWS + 2 * Q27_KVROWS));
        A((void**)&s1.qkvas, (size_t)pf_chw * ((Q27_QROWS + 2 * Q27_KVROWS) / 16) * 4);
        A((void**)&s1.qh, (size_t)pf_chw * (Q27_OROWS / pf_div) * 4);
        { size_t mlb = 0; const size_t ob = q27_attn_chunk_scratch_bytes(pf_div, pf_chw, &mlb); A((void**)&s1.pob, ob); A((void**)&s1.pml, mlb); s1.pob_bytes = ob; }
        A((void**)&s1.mixq_slots, (size_t)pf_chw * (Q27_GDN_Z / pf_div));
        A((void**)&s1.mixs_slots, (size_t)pf_chw * ((Q27_GDN_Z / pf_div) / 16) * 4);
        A((void**)&s1.qkv8,  (size_t)pf_chw * Q27_GDN_QKV);
        A((void**)&s1.qkvs,  (size_t)pf_chw * (Q27_GDN_QKV / 16) * 4);
        A((void**)&s1.z8,    (size_t)pf_chw * Q27_GDN_Z);
        A((void**)&s1.zs,    (size_t)pf_chw * (Q27_GDN_Z / 16) * 4);
        A((void**)&s1.a32,   (size_t)Q27_PF_SLOTS * Q27_GDN_VH * 4);
        A((void**)&s1.b32,   (size_t)Q27_PF_SLOTS * Q27_GDN_VH * 4);
        A((void**)&s1.mixq_tile, (size_t)Q27_PF_SLOTS * (Q27_GDN_Z / pf_div));
        A((void**)&s1.mixs_tile, (size_t)Q27_PF_SLOTS * ((Q27_GDN_Z / pf_div) / 16) * 4);
        A((void**)&s1.part,  (size_t)pf_chw * Q27_HID * 4);
        A((void**)&s1.xq1,   (size_t)pf_chw * Q27_HID);
        A((void**)&s1.xs1,   (size_t)pf_chw * (Q27_HID / 16) * 4);
        A((void**)&s1.pa,    (size_t)pf_chw * mlp_len_div(id, pf_div) * 4);
        A((void**)&s1.pb,    (size_t)pf_chw * mlp_len_div(id, pf_div) * 4);
        A((void**)&s1.xq2,   (size_t)pf_chw * mlp_len_div(id, pf_div));
        A((void**)&s1.xs2,   (size_t)pf_chw * (mlp_len_div(id, pf_div) / 16) * 4);
        A((void**)&s1.mixer, (size_t)pf_chw * Q27_HID * 4);
        A((void**)&s1.hid32c,(size_t)pf_chw * Q27_HID * 4);
        }
        for (int si = 0; si < (ls_one ? 1 : 2); ++si) {   // stage 3 (rocBLAS) scratch, both sets
            Q8Scr& sx = d.q8s[si];
            A((void**)&sx.acc, (size_t)pf_chw * 2 * Q27_INTER * (Q27_HID / 1024) * 4); A((void**)&sx.xqt, (size_t)pf_chw * Q27_HID);   // int32 slabs: 2 matrices x K/1024 groups
            A((void**)&sx.xst, (size_t)pf_chw * (Q27_HID / 1024) * 4); A((void**)&sx.xst1, (size_t)pf_chw * (Q27_HID / 1024) * 4); A((void**)&sx.xst2, (size_t)pf_chw * (Q27_INTER / 1024) * 4);
            A((void**)&sx.mixq_t, (size_t)pf_chw * (Q27_GDN_Z > Q27_OROWS ? Q27_GDN_Z : Q27_OROWS)); A((void**)&sx.mixs_t, (size_t)pf_chw * ((Q27_GDN_Z > Q27_OROWS ? Q27_GDN_Z : Q27_OROWS) / 1024) * 4);
            A((void**)&sx.qkvc8, (size_t)pf_chw * Q27_GDN_QKV); A((void**)&sx.qkvcs, (size_t)pf_chw * (Q27_GDN_QKV / 16) * 4);
        }
        // Set 1 ALIASES set 0, now that set 0 is fully populated: every existing q8s[k & 1]
        // reference stays valid and the ring simply serializes instead of overlapping.
        if (ls_one) d.q8s[1] = d.q8s[0];
    }
    // ALLOCATED BEFORE THE CTX-SCALED STATE, ON PURPOSE. Allocation order decides whether a
    // buffer lands in HBM or GTT -- the engine already banks that "once the fp8 arena is released
    // the banks/slots land in HBM instead of GTT (the p1k K=7 collapse)". Allocating this arena
    // last put 271 MiB of MLP weights in host memory, and every GEMM then read them over PCIe:
    // a 135-token prefill took 305 s and decode fell to 1.27 tok/s. It has to be HBM.
    if (q27_env_int("Q27_LS_MLP_STAGE", 0)) {
        // gate [INTER][HID] | up [INTER][HID] | down [HID][INTER], int8 + per-64 fp32 scales.
        // Mode 1: gate|up|down (271 MiB). Mode 2: down ONLY (90 MiB).
        // The headroom is NOT the 1.04 GiB that hipMemGetInfo reports as free -- measured, the
        // driver collapses into continuous eviction somewhere between 0.78 and 1.04 GiB free
        // (135-token prefill 89 s, decode 1.77 tok/s). 271 MiB does not fit inside that margin;
        // 90 MiB plausibly does, and down is the projection most likely to dominate.
        const int stage_mode = q27_env_int("Q27_LS_MLP_STAGE", 0);
        const size_t wb = ((stage_mode == 2) ? (size_t)Q27_HID * Q27_INTER
                                            : (size_t)Q27_INTER * Q27_HID * 2 + (size_t)Q27_HID * Q27_INTER)
                          * (size_t)Dev::MLP8_SLOTS;
        const size_t sb = ((stage_mode == 2) ? (size_t)Q27_HID * (Q27_INTER / 64)
                                             : (size_t)Q27_INTER * (Q27_HID / 64) * 2 + (size_t)Q27_HID * (Q27_INTER / 64))
                          * 4 * (size_t)Dev::MLP8_SLOTS;
        if (hipMalloc((void**)&d.mlp8_w, wb) != hipSuccess || hipMalloc((void**)&d.mlp8_s, sb) != hipSuccess) {
            std::fprintf(stderr, "Q27_LS_MLP_STAGE card %d: arena allocation failed (%.1f MiB); "
                                 "falling back to in-kernel NVFP4\n", id, (double)(wb + sb) / 1048576.0);
            d.mlp8_w = nullptr; d.mlp8_s = nullptr;
        } else {
            size_t fr = 0, tt = 0; CK(hipMemGetInfo(&fr, &tt));
            std::fprintf(stderr, "Q27_LS_MLP_STAGE card %d: %.1f MiB layer-local int8 MLP arena, "
                                 "free after %.2f GiB\n", id, (double)(wb + sb) / 1048576.0,
                         (double)fr / 1073741824.0);
        }
        // Q27_LS_MLP_RB: the row-form scales. One float per row per tensor per slot, so this is
        // 4 slots x (2*INTER + HID) floats = 624 KiB -- noise next to the 1.07 GiB of weights it
        // describes, and it is what makes gate/up/down single full-K int8 GEMMs for Tensile.
        if (q27_env_int("Q27_LS_MLP_RB", 0) && stage_mode != 2 && d.mlp8_w && d.mlp8_s) {
            const size_t rs = (size_t)(2 * Q27_INTER + Q27_HID) * Dev::MLP8_SLOTS * sizeof(float);
            if (hipMalloc((void**)&d.mlp8_rs, rs) != hipSuccess) {
                std::fprintf(stderr, "Q27_LS_MLP_RB card %d: row-scale arena allocation failed (%.1f KiB); "
                                     "falling back to in-kernel NVFP4\n", id, (double)rs / 1024.0);
                d.mlp8_rs = nullptr;
            } else {
                size_t fr = 0, tt = 0; CK(hipMemGetInfo(&fr, &tt));
                std::fprintf(stderr, "Q27_LS_MLP_RB card %d: %.1f MiB arena + %.1f KiB row scales, "
                                     "free after %.2f GiB\n", id, (double)(wb + sb) / 1048576.0,
                             (double)rs / 1024.0, (double)fr / 1073741824.0);
            }
        }
        std::fflush(stderr);
    }
    // shard-sized recurrent state / KV cache: 48/ndev value heads, 1024/ndev KV rows per position
    const int n_gdn = Q27_LAYERS - Q27_LAYERS / 4, n_full = Q27_LAYERS / 4;   // 48 / 16
    // Q27_TP3: ndev here is the allocation divisor (1 under layer-split = full width); the
    // worker's narrow shard sizes come from the plan table.
    const q27_tp_shard_t* const TS = q27_tp_shard(id);
    d.tp_s    = (size_t)(ndev == 1 ? Q27_GDN_VH : TS->VHL) * Q27_GDN_D * Q27_GDN_D;   // per layer
    d.tp_conv = (size_t)(ndev == 1 ? Q27_GDN_QKV : TS->QKVL) * Q27_GDN_CONV;          // per layer
    d.tp_kv   = (size_t)ctx * (ndev == 1 ? Q27_KVROWS : TS->KVL);                     // per layer
    d.tp_kvs  = d.tp_kv / 16;
    // OWNERSHIP, derived exactly as q27_upload_ls and the sweep derive it: same env vars, same
    // defaults, same clamp. It MUST agree with both, or a card would address a layer it does not
    // hold -- that agreement is the whole precondition for storing non-owned layers narrow.
    const int ls_blk_A = q27_ls_blk_eff(rndev);   // Q27_TP3: quota-aware block size (same oracle as the loader)
    (void)q27_env_int("Q27_LS_BAL", 0);   // read+applied by q27_ls_block_owner; kept for the boot-time env audit
    // 2026-09-18: the oracle, not a fourth copy of the modulo. ls_bal_A is now unused here --
    // Q27_LS_BAL is applied inside q27_ls_block_owner so the uneven-quota path cannot disagree.
    auto owner_A = [&](int L) { return q27_ls_block_owner(L / ls_blk_A, rndev); };
    const int KVL_R = TS->KVL;                     // the KV-head range TP decode reads (ragged under Q27_TP3)
    const int KVH_OFF = Q27_HDIM * TS->kv_h0;      // its column offset inside a full-width row
    size_t kc_el = 0, ks_fl = 0, s_fl = 0, cv_el = 0;
    for (int L = 0; L < Q27_LAYERS; ++L) {
        const bool own = !ls || owner_A(L) == id;
        if (Q27_IS_FULL(L)) {
            const int fs = L >> 2;
            // Q27_PF_P2P: the owner's FULL-WIDTH copy of its own layers is the last duplicate in
            // the KV. Every non-owned quarter of an owned layer already exists on the card that
            // owns that head -- the owner keeps a second copy of all four quarters purely so its
            // prefill attention can read them locally. With peer access enabled (it already is,
            // hipDeviceEnablePeerAccess at ~3710) the sweep can read the three remote quarters
            // straight out of the peers, and every card stores exactly the quarter TP decode
            // reads. Measured at ctx=262144, LS_BLK=4: kc+vc+scales 4.375 -> 2.500 GiB/card.
            static const bool p2pkv = q27_env_flag("Q27_PF_KVP2P", false);
            // Q27_LS_SR: a layer this card does not own is not decoded here either, so it stores
            // NOTHING -- the 12 head-quarters (1.27 GiB/card at 262K) were the TP decode's.
            const int stride = (g_sr && !own) ? 0 : ((!ls || p2pkv) ? KVL_R : (own ? Q27_KVROWS : KVL_R));
            d.kv_rowstride[fs] = stride;
            d.kv_hoff[fs]      = (!ls || !own || p2pkv) ? 0 : KVH_OFF;
            d.kv_hoff_peer[fs] = ls ? KVH_OFF : 0;      // always this card's heads in a full-width row
            d.kv_off[fs]    = kc_el;
            d.kvs_off[fs]   = ks_fl;
            d.kv_bytes[fs]  = (size_t)ctx * stride;
            d.kvs_bytes[fs] = (size_t)ctx * (stride / 16) * (int)sizeof(q27_kvs_t);
            kc_el += (size_t)ctx * stride;
            ks_fl += (size_t)ctx * (stride / 16);
        } else {
            const int gs = L - (L >> 2);
            const size_t s_narrow = (size_t)TS->VHL * Q27_GDN_D * Q27_GDN_D, c_narrow = (size_t)TS->QKVL * Q27_GDN_CONV;
            const size_t sw = (g_sr && !own) ? 0 : ((!ls || own) ? d.tp_s    : s_narrow);
            const size_t cw = (g_sr && !own) ? 0 : ((!ls || own) ? d.tp_conv : c_narrow);
            d.s_off[gs]      = s_fl;
            d.conv_off[gs]   = cv_el;
            d.s_qoff[gs]     = (!ls || !own) ? 0 : (size_t)TS->gv_h0 * Q27_GDN_D * Q27_GDN_D;
            d.conv_qoff[gs]  = 0;   // conv is packed q|k|v: no contiguous quarter (unused)
            d.conv_wide[gs]  = (ls && own) ? 1 : 0;
            d.s_bytes[gs]    = sw * 4;
            d.conv_bytes[gs] = cw * 2;
            s_fl  += sw;
            cv_el += cw;
        }
    }
    A((void**)&d.S,    s_fl * 4);
    A((void**)&d.conv, cv_el * 2);
    d.s_bytes_total = s_fl * 4; d.conv_bytes_total = cv_el * 2;
    // Q27_REWIND_CK banks of the WHOLE recurrent state. Priced before allocating (the
    // affordability rule): the tightest card decides, and a bank is s_bytes_total +
    // conv_bytes_total. Allocated here so a failure is reported by the same A() that names the
    // request and the remaining free VRAM, not as a mystery OOM later.
    if (q27_rewind_ck() > 0) {
        A((void**)&d.ck_S,    (size_t)q27_rewind_ck() * d.s_bytes_total);
        A((void**)&d.ck_conv, (size_t)q27_rewind_ck() * d.conv_bytes_total);
        if (id == 0) std::fprintf(stderr, "Q27_REWIND_CK %d banks x %.1f MiB = %.2f GiB/card "
                                          "(S %.1f + conv %.1f MiB)\n", q27_rewind_ck(),
                     (double)(d.s_bytes_total + d.conv_bytes_total) / 1048576.0,
                     (double)q27_rewind_ck() * (d.s_bytes_total + d.conv_bytes_total) / 1073741824.0,
                     (double)d.s_bytes_total / 1048576.0, (double)d.conv_bytes_total / 1048576.0);
    }
    A((void**)&d.kc,   kc_el);
    A((void**)&d.ks,   ks_fl * sizeof(q27_kvs_t));
#if Q27_V4
    A((void**)&d.vc,   kc_el / 2);                       // packed bitplanes: stride/2 bytes per row
    A((void**)&d.vs,   ks_fl / 2 * sizeof(q27_kvs_t));   // one block max per 32 dims, not per 16
#else
    A((void**)&d.vc,   kc_el);
    A((void**)&d.vs,   ks_fl * sizeof(q27_kvs_t));
#endif
    {   // BOUNDS, CHECKED AT ALLOCATION. kv_off[]/kvs_off[] are FULL-WIDTH offsets (they are K's),
        // and V4 halves both V planes, so every V base must pass through Q27_VOFF/Q27_VSOFF. Eight
        // call sites did not, and the engine's first dispatched prefill job walked off vc on a
        // 130-token prompt (2026-09-16). A page fault names an address; this names the layer. The
        // check is arithmetic on ~16 entries at boot and is identity-true in the V4=0 arm.
        const size_t vcb = (size_t)Q27_VOFF(kc_el), vsb = (size_t)Q27_VSOFF(ks_fl) * sizeof(q27_kvs_t);
        for (int fsb = 0; fsb < n_full; ++fsb) {
            const size_t ve = (size_t)Q27_VOFF(d.kv_off[fsb]) + (size_t)Q27_VOFF(d.kv_bytes[fsb]);
            const size_t se = ((size_t)Q27_VSOFF(d.kvs_off[fsb])
                             + (size_t)Q27_VSOFF(d.kvs_bytes[fsb] / sizeof(q27_kvs_t))) * sizeof(q27_kvs_t);
            if (ve > vcb || se > vsb) {
                std::fprintf(stderr, "Q27_KVBOUNDS FATAL card %d full-layer %d: vc end %zu of %zu, "
                                     "vs end %zu of %zu -- a V base is missing Q27_VOFF/Q27_VSOFF\n",
                             id, fsb, ve, vcb, se, vsb);
                std::fflush(stderr); std::exit(1);
            }
        }
    }
    if (q27_env_flag("Q27_KVPLACE", false))
#if Q27_V4
        std::fprintf(stderr, "Q27_KVPLACE card %d ls=%d kc=%.3f GiB vc=%.3f GiB ks=%.3f GiB vs=%.3f GiB "
                             "S=%.1f MiB conv=%.1f MiB\n", id, (int)ls,
                     (double)kc_el / 1073741824.0, (double)(kc_el / 2) / 1073741824.0,
                     (double)(ks_fl * sizeof(q27_kvs_t)) / 1073741824.0,
                     (double)(ks_fl / 2 * sizeof(q27_kvs_t)) / 1073741824.0,
                     (double)(s_fl * 4) / 1048576.0, (double)(cv_el * 2) / 1048576.0);
#else
        std::fprintf(stderr, "Q27_KVPLACE card %d ls=%d kc/vc=%.3f GiB each ks/vs=%.3f GiB each "
                             "S=%.1f MiB conv=%.1f MiB\n", id, (int)ls,
                     (double)kc_el / 1073741824.0, (double)(ks_fl * sizeof(q27_kvs_t)) / 1073741824.0,
                     (double)(s_fl * 4) / 1048576.0, (double)(cv_el * 2) / 1048576.0);
#endif
    if (q27_env_flag("Q27_KCDBG", false)) std::fprintf(stderr, "Q27_KCDBG alloc id=%d ndev=%d ctx=%d tp_kv=%zu n_full=%d kc=%p\n", id, ndev, ctx, d.tp_kv, n_full, (const void*)d.kc);
    // MTP draft KV: the draft layer is REPLICATED and runs at FULL width, so its cache is a full
    // ctx*Q27_KVROWS per layer -- NOT the TP shard stride tp_kv, which is 4x smaller at ndev=4.
    // Carving it out of d.kc with the shard stride made the full-width attention kernel write past
    // the end of the allocation: a GPU page fault at the SAME address on every run, which is what
    // identified it (a staging-buffer bug would have moved when the staging allocation changed).
    // GROW ON DEMAND. Sized for the full context this is 640 MiB per card at ctx=262144 -- and it
    // is REPLICATED, so it is 640 MiB on all four. For a 9K turn ~97% of that is dead capacity:
    // the draft only ever addresses positions that exist. Start small and grow as the conversation
    // does (q27_mtp_kv_reserve). The 262144 capability is unchanged -- capacity is reached by
    // growing, not by reserving it up front. No math change, no communication change, no change to
    // the draft's full-width replication.
    d.mtp_cap = q27_env_int("Q27_MTP_KV_CAP0", 16384);
    if (d.mtp_cap > ctx) d.mtp_cap = ctx;
    if (d.mtp_cap < 1024) d.mtp_cap = 1024;
    if (!q27_env_int("Q27_MTP_PAGED", 1)) d.mtp_cap = ctx;      // =0 restores the up-front reservation
    if (g_sr && !pa_head) d.mtp_cap = 0;   // Q27_LS_SR: the draft layer and its KV live on the head card only
    if (d.mtp_cap > 0) {
    A((void**)&d.mtp_kc, (size_t)d.mtp_cap * Q27_KVROWS);
    A((void**)&d.mtp_ks, (size_t)d.mtp_cap * (Q27_KVROWS / 16) * sizeof(q27_kvs_t));
#if Q27_V4
    A((void**)&d.mtp_vc, (size_t)d.mtp_cap * (Q27_KVROWS / 2));
    A((void**)&d.mtp_vs, (size_t)d.mtp_cap * (Q27_KVROWS / 32) * sizeof(q27_kvs_t));
#else
    A((void**)&d.mtp_vc, (size_t)d.mtp_cap * Q27_KVROWS);
    A((void**)&d.mtp_vs, (size_t)d.mtp_cap * (Q27_KVROWS / 16) * sizeof(q27_kvs_t));
#endif
    }   // d.mtp_cap > 0
    if (q27_env_flag("Q27_SPEC", true)) {   // Q27_SPEC (shipped default on): NR-row verify buffers, full width (fits any TP degree)
        const size_t NRM = 8;   // Q27_SPEC: scratch holds K+1 <= 8 verify rows (the NR kernels run chunks of <= 4)
        A((void**)&d.nr_norm, NRM * Q27_HID * 2);
        A((void**)&d.nr_xq,   NRM * Q27_HID);
        A((void**)&d.nr_xs,   NRM * (Q27_HID / 16) * 4);
        A((void**)&d.nr_qkva, NRM * (Q27_QROWS + 2 * Q27_KVROWS) * 2);
        A((void**)&d.nr_qh,   NRM * Q27_OROWS * 2);
        A((void**)&d.nr_qkv,  NRM * Q27_GDN_QKV * 2);
        A((void**)&d.nr_zbuf, NRM * Q27_GDN_Z * 2);
        A((void**)&d.nr_ab,   NRM * Q27_GDN_VH * 2);  A((void**)&d.nr_bb, NRM * Q27_GDN_VH * 2);
        A((void**)&d.nr_mix6, NRM * Q27_GDN_Z * 2);
        A((void**)&d.nr_pa,   NRM * PA_ELEMS * 4);    A((void**)&d.nr_pb, NRM * PA_ELEMS * 4);
        A((void**)&d.nr_acc,  NRM * Q27_HID * 4);
        A((void**)&d.nr_mixer, NRM * Q27_HID * 4);
        A((void**)&d.nr_tok, 64); d.nr_val = (float*)(d.nr_tok + 8);
        CK(hipHostMalloc((void**)&d.nr_mail_h, 64, hipHostMallocDefault)); std::memset(d.nr_mail_h, 0, 64);
        CK(hipHostGetDevicePointer((void**)&d.nr_mail_d, d.nr_mail_h, 0));
        A((void**)&d.nr_hprev, Q27_HID * 2);  A((void**)&d.nr_hcatch, NRM * Q27_HID * 2);
        A((void**)&d.nr_taps, (size_t)5 * NRM * Q27_HID * 2);
        if (g_df2_draft) { A((void**)&d.df2_pf_taps, (size_t)5 * (size_t)g_pf_cap * Q27_HID * 2);
                           if (hipHostMalloc((void**)&d.df2_pf_taps_h, (size_t)5 * (size_t)g_pf_cap * Q27_HID * 2, hipHostMallocDefault) != hipSuccess) d.df2_pf_taps_h = nullptr; }
        CK(hipHostMalloc((void**)&d.nr_ebounce, (size_t)32 * Q27_HID * 2, hipHostMallocDefault));
    }
#undef A

}

// EXPORT ONE FULL-ATTENTION LAYER'S KV FROM ITS OWNER TO A CARD THAT DOES NOT OWN IT.
// The owner ran layer-split prefill on this layer and holds it FULL WIDTH: rows of Q27_KVROWS
// with this card's heads at kv_hoff. The destination stores only the head-quarter TP decode
// reads, contiguous at stride KVROWS/ndev. So this is a strided 2D transfer of the quarter over
// the npos positions the sweep actually produced -- not the whole row, and not the whole ctx.
// Both of those were in the original: it copied every head to every card for every layer, and
// the post-sweep path copied all ctx positions rather than the prefix.
static void q27_export_kv_quarter(Dev& dst, Dev& src, int fs, int beg, int end, hipStream_t st) {
    if (!q27_env_flag("Q27_KV_DELTA", false)) beg = 0;
    const int npos = end - beg;
    if (npos <= 0) return;
    const int w  = dst.kv_rowstride[fs];             // destination row = the quarter
    const int sw = src.kv_rowstride[fs];             // source row = the owner's full width
    const int hoff = dst.kv_hoff_peer[fs];        // this card's heads inside the owner's row
    CK(hipMemcpy2DAsync(dst.kc + (dst.kv_off[fs] + (size_t)beg * w), (size_t)w,
                        src.kc + (src.kv_off[fs] + (size_t)beg * sw) + hoff, (size_t)sw,
                        (size_t)w, (size_t)npos, hipMemcpyDeviceToDevice, st));
#if Q27_V4
    // V4: half a byte per dim and one max per 32 dims, so both the row stride and the head offset
    // halve; the K copies above and the K scales below are untouched.
    CK(hipMemcpy2DAsync(dst.vc + (dst.kv_off[fs] + (size_t)beg * w) / 2, (size_t)(w / 2),
                        src.vc + (src.kv_off[fs] + (size_t)beg * sw) / 2 + hoff / 2, (size_t)(sw / 2),
                        (size_t)(w / 2), (size_t)npos, hipMemcpyDeviceToDevice, st));
#else
    CK(hipMemcpy2DAsync(dst.vc + (dst.kv_off[fs] + (size_t)beg * w), (size_t)w,
                        src.vc + (src.kv_off[fs] + (size_t)beg * sw) + hoff, (size_t)sw,
                        (size_t)w, (size_t)npos, hipMemcpyDeviceToDevice, st));
#endif
    CK(hipMemcpy2DAsync(dst.ks + (dst.kvs_off[fs] + (size_t)beg * (w / 16)), (size_t)(w / 16) * sizeof(q27_kvs_t),
                        src.ks + (src.kvs_off[fs] + (size_t)beg * (sw / 16)) + hoff / 16, (size_t)(sw / 16) * sizeof(q27_kvs_t),
                        (size_t)(w / 16) * sizeof(q27_kvs_t), (size_t)npos, hipMemcpyDeviceToDevice, st));
#if Q27_V4
    CK(hipMemcpy2DAsync(dst.vs + (dst.kvs_off[fs] + (size_t)beg * (w / 16)) / 2, (size_t)(w / 32) * sizeof(q27_kvs_t),
                        src.vs + (src.kvs_off[fs] + (size_t)beg * (sw / 16)) / 2 + hoff / 32, (size_t)(sw / 32) * sizeof(q27_kvs_t),
                        (size_t)(w / 32) * sizeof(q27_kvs_t), (size_t)npos, hipMemcpyDeviceToDevice, st));
#else
    CK(hipMemcpy2DAsync(dst.vs + (dst.kvs_off[fs] + (size_t)beg * (w / 16)), (size_t)(w / 16) * sizeof(q27_kvs_t),
                        src.vs + (src.kvs_off[fs] + (size_t)beg * (sw / 16)) + hoff / 16, (size_t)(sw / 16) * sizeof(q27_kvs_t),
                        (size_t)(w / 16) * sizeof(q27_kvs_t), (size_t)npos, hipMemcpyDeviceToDevice, st));
#endif
}

// Conv state is packed q|k|v, not four contiguous worker quarters. Each channel
// has CONV BF16 history values. Preserve the three segment boundaries in both directions.
static void q27_copy_conv_quarter(Dev& owner, Dev& shard, int gs, int worker,
                                  int ndev, bool gather, hipStream_t st) {
    const int widths[3] = {Q27_GDN_KH * Q27_GDN_D, Q27_GDN_KH * Q27_GDN_D,
                           Q27_GDN_VH * Q27_GDN_D};
    // Q27_TP3: the worker's key-head range inside each packed segment (q, k: 128 channels per key
    // head; v: 384), from the plan table rather than an equal split.
    const q27_tp_shard_t* const W = q27_tp_shard(worker);
    const size_t seg_off[3] = {(size_t)Q27_GDN_D * W->gk_h0, (size_t)Q27_GDN_D * W->gk_h0, (size_t)3 * Q27_GDN_D * W->gk_h0};
    const size_t seg_len[3] = {(size_t)Q27_GDN_D * W->gk_hn, (size_t)Q27_GDN_D * W->gk_hn, (size_t)3 * Q27_GDN_D * W->gk_hn};
    (void)ndev;
    size_t full = 0, narrow = 0;
    for (int i = 0; i < 3; ++i) {
        const int n = widths[i];
        const size_t count = seg_len[i] * Q27_GDN_CONV;
        auto* op = owner.conv + owner.conv_off[gs] + full + seg_off[i] * Q27_GDN_CONV;
        auto* sp = shard.conv + shard.conv_off[gs] + narrow;
        if (gather) CK(hipMemcpyPeerAsync(op, owner.id, sp, shard.id, count * 2, st));
        else CK(hipMemcpyPeerAsync(sp, shard.id, op, owner.id, count * 2, st));
        full += (size_t)n * Q27_GDN_CONV; narrow += count;
    }
}

// Refresh only newly decoded KV positions in the full layer owner's cache.
static void q27_gather_kv_quarter(Dev& dst, Dev& src, int fs, int beg, int end, hipStream_t st) {
    if (end <= beg) return;
    const size_t dw = dst.kv_rowstride[fs], sw = src.kv_rowstride[fs];
    const size_t hoff = src.kv_hoff_peer[fs], height = end - beg;
    CK(hipMemcpy2DAsync(dst.kc + dst.kv_off[fs] + (size_t)beg*dw + hoff, dw,
        src.kc + src.kv_off[fs] + (size_t)beg*sw, sw, sw, height, hipMemcpyDeviceToDevice, st));
    CK(hipMemcpy2DAsync(dst.vc + Q27_VOFF(dst.kv_off[fs] + (size_t)beg*dw + hoff), Q27_VOFF(dw),
        src.vc + Q27_VOFF(src.kv_off[fs] + (size_t)beg*sw), Q27_VOFF(sw), Q27_VOFF(sw), height, hipMemcpyDeviceToDevice, st));
    CK(hipMemcpy2DAsync(dst.ks + dst.kvs_off[fs] + ((size_t)beg*dw + hoff)/16, dw/16*sizeof(q27_kvs_t),
        src.ks + src.kvs_off[fs] + (size_t)beg*sw/16, sw/16*sizeof(q27_kvs_t), sw/16*sizeof(q27_kvs_t), height, hipMemcpyDeviceToDevice, st));
    CK(hipMemcpy2DAsync(dst.vs + Q27_VSOFF(dst.kvs_off[fs] + ((size_t)beg*dw + hoff)/16), Q27_VSOFF(dw/16)*sizeof(q27_kvs_t),
        src.vs + Q27_VSOFF(src.kvs_off[fs] + (size_t)beg*sw/16), Q27_VSOFF(sw/16)*sizeof(q27_kvs_t), Q27_VSOFF(sw/16)*sizeof(q27_kvs_t), height, hipMemcpyDeviceToDevice, st));
}

// GROW THE DRAFT KV TO COVER `need` POSITIONS.
// Called at the sweep boundary, where the stream is quiescent for this card and every consumer
// re-reads d.mtp_* (nothing caches the pointers across a request). Doubling keeps the number of
// grows logarithmic; the live prefix is copied so already-written draft KV survives. Returns
// false only if the device cannot satisfy the larger allocation, in which case the caller keeps
// the smaller one and the engine refuses the window as it would have before.
static bool q27_mtp_kv_reserve(Dev& d, int need, int ctx) {
    if (need <= d.mtp_cap) return true;
    // GROW TO WHAT THE WINDOW NEEDS, NOT TO THE NEXT ROUND NUMBER. The draft KV is addressed by
    // ABSOLUTE position (the conditioning kernels write row p0 with no mask), so the capacity never
    // had to be a power of two -- doubling was a habit, and at depth it was the whole problem: from
    // 131,072 the next power of two is 262,144, so a conversation at 138,199 asked for 671 MB when
    // 3 MB of extra capacity was what it needed (measured 2026-09-16; that refusal is why
    // speculation died and decode sat at 21 tok/s where it was 46-55). Stepping by a fixed 16,384
    // fixed the first case but still overshot the top: at 248,793 positions it asked for the
    // 262,144 buffer (671 MB) and was refused by SEVEN MEGABYTES against 0.21 GiB free plus
    // 545 MB of reclaimable old cache. Now the target is the need itself plus a small slack, which
    // is also the smallest allocation the window can be served from.
    const int step = 1024;                     // slack over the need, rounded up to 256 below
    int cap = d.mtp_cap;
    if (cap < need) {
        long long want = (long long)need + step;
        if (want > ctx) want = ctx;
        cap = (int)((want + 255LL) & ~255LL);
        if (cap < need) cap = ctx;
    }
    if (cap < need) return false;
    CK(hipSetDevice(d.id));
#if Q27_V4
    // The replicated draft KV follows the target planes: V halves and its maxes go per-32 dims.
    const size_t wb = (size_t)cap * (Q27_KVROWS / 2), sb = (size_t)cap * (Q27_KVROWS / 32) * sizeof(q27_kvs_t), kwb = (size_t)cap * Q27_KVROWS, ksb = (size_t)cap * (Q27_KVROWS / 16) * sizeof(q27_kvs_t);   // K IS NOT HALVED
    const size_t owb = (size_t)d.mtp_cap * (Q27_KVROWS / 2), osb = (size_t)d.mtp_cap * (Q27_KVROWS / 32) * sizeof(q27_kvs_t), okwb = (size_t)d.mtp_cap * Q27_KVROWS, oksb = (size_t)d.mtp_cap * (Q27_KVROWS / 16) * sizeof(q27_kvs_t);
#else
    const size_t wb = (size_t)cap * Q27_KVROWS, sb = (size_t)cap * (Q27_KVROWS / 16) * sizeof(q27_kvs_t), kwb = wb, ksb = sb;
    const size_t owb = (size_t)d.mtp_cap * Q27_KVROWS, osb = (size_t)d.mtp_cap * (Q27_KVROWS / 16) * sizeof(q27_kvs_t), okwb = owb, oksb = osb;
#endif
    signed char *kc = nullptr, *vc = nullptr; q27_kvs_t *ks = nullptr, *vs = nullptr;
    auto alloc_new = [&]() -> bool {
        if (hipMalloc((void**)&kc, kwb) != hipSuccess) { kc = nullptr; return false; }
        if (hipMalloc((void**)&vc, wb) != hipSuccess) { hipFree(kc); kc = vc = nullptr; return false; }
        if (hipMalloc((void**)&ks, ksb) != hipSuccess) { hipFree(kc); hipFree(vc); kc = vc = nullptr; return false; }
        if (hipMalloc((void**)&vs, sb) != hipSuccess) { hipFree(kc); hipFree(vc); hipFree(ks); kc = vc = nullptr; ks = nullptr; return false; }
        return true;
    };
    if (!alloc_new()) {
        // FREE BEFORE ALLOCATE. The copy below needs source and destination alive at once, so the
        // direct path's true peak is OLD + NEW -- while this function's caller prices only NEW
        // ("THE PEAK IS THE WHOLE NEW BUFFER, NOT THE DIFFERENCE"), which is true only if the old
        // buffers are released first. Stage the old cache through the host so they can be: the peak
        // becomes max(old, new) plus pageable host memory, and the caller's price becomes correct.
        const size_t tot = okwb + owb + oksb + osb;
        std::vector<signed char> stage(tot);
        signed char* sp = stage.data();
        if (hipMemcpyAsync(sp,          d.mtp_kc, okwb, hipMemcpyDeviceToHost, d.stream) != hipSuccess ||
            hipMemcpyAsync(sp + okwb,   d.mtp_vc, owb, hipMemcpyDeviceToHost, d.stream) != hipSuccess ||
            hipMemcpyAsync(sp + okwb + owb, d.mtp_ks, oksb, hipMemcpyDeviceToHost, d.stream) != hipSuccess ||
            hipMemcpyAsync(sp + okwb + owb + oksb, d.mtp_vs, osb, hipMemcpyDeviceToHost, d.stream) != hipSuccess) {
            return false;                                     // the old cache is still intact
        }
        CK(hipStreamSynchronize(d.stream));
        CK(hipFree(d.mtp_kc)); CK(hipFree(d.mtp_vc)); CK(hipFree(d.mtp_ks)); CK(hipFree(d.mtp_vs));
        d.mtp_kc = nullptr; d.mtp_vc = nullptr; d.mtp_ks = nullptr; d.mtp_vs = nullptr;
        const int lost = d.mtp_cap; d.mtp_cap = 0;             // if the new alloc fails there is no
                                                              // draft cache: within_cap reports 0
        if (!alloc_new()) return false;
        CK(hipMemsetAsync(kc, 0, kwb, d.stream)); CK(hipMemsetAsync(vc, 0, wb, d.stream));
        CK(hipMemsetAsync(ks, 0, ksb, d.stream)); CK(hipMemsetAsync(vs, 0, sb, d.stream));
        CK(hipMemcpyAsync(kc, sp,          okwb, hipMemcpyHostToDevice, d.stream));
        CK(hipMemcpyAsync(vc, sp + okwb,   owb, hipMemcpyHostToDevice, d.stream));
        CK(hipMemcpyAsync(ks, sp + okwb + owb, oksb, hipMemcpyHostToDevice, d.stream));
        CK(hipMemcpyAsync(vs, sp + okwb + owb + oksb, osb, hipMemcpyHostToDevice, d.stream));
        CK(hipStreamSynchronize(d.stream));
        d.mtp_kc = kc; d.mtp_vc = vc; d.mtp_ks = ks; d.mtp_vs = vs;
        d.mtp_cap = cap;
        std::fprintf(stderr, "Q27_MTP_HOSTSTAGE card %d: draft KV %d -> %d positions; old cache "
                             "staged through the host so the peak is the NEW buffer, not old+new\n",
                     d.id, lost, cap);
        std::fflush(stderr);
        return true;
    }
    CK(hipMemsetAsync(kc, 0, kwb, d.stream)); CK(hipMemsetAsync(vc, 0, wb, d.stream));
    CK(hipMemsetAsync(ks, 0, ksb, d.stream)); CK(hipMemsetAsync(vs, 0, sb, d.stream));
    CK(hipMemcpyAsync(kc, d.mtp_kc, okwb, hipMemcpyDeviceToDevice, d.stream));
    CK(hipMemcpyAsync(vc, d.mtp_vc, owb, hipMemcpyDeviceToDevice, d.stream));
    CK(hipMemcpyAsync(ks, d.mtp_ks, oksb, hipMemcpyDeviceToDevice, d.stream));
    CK(hipMemcpyAsync(vs, d.mtp_vs, osb, hipMemcpyDeviceToDevice, d.stream));
    CK(hipStreamSynchronize(d.stream));
    signed char* okc = d.mtp_kc; signed char* ovc = d.mtp_vc; q27_kvs_t* oks = d.mtp_ks; q27_kvs_t* ovs = d.mtp_vs;
    CK(hipFree(d.mtp_kc)); CK(hipFree(d.mtp_vc)); CK(hipFree(d.mtp_ks)); CK(hipFree(d.mtp_vs));
    d.mtp_kc = kc; d.mtp_vc = vc; d.mtp_ks = ks; d.mtp_vs = vs;
    const int old = d.mtp_cap; d.mtp_cap = cap;
    if (d.id == 0) {
        // Log the POINTER TRANSITION, not just the sizes: it is the proof that the realloc+copy
        // actually happened rather than some dormant branch reporting a plan it never executed.
        std::fprintf(stderr, "Q27_MTP_PAGED card %d: draft KV %d -> %d positions (%.0f -> %.0f MiB) "
                             "kc %p->%p vc %p->%p ks %p->%p vs %p->%p\n",
                     d.id, old, cap, (double)(okwb + owb + oksb + osb) / 1048576.0,
                     (double)(kwb + wb + ksb + sb) / 1048576.0,
                     (void*)okc, (void*)kc, (void*)ovc, (void*)vc, (void*)oks, (void*)ks, (void*)ovs, (void*)vs);
        std::fflush(stderr);
    }
    return true;
}

// Prefill-sweep scratch, allocated LAST (after the mirrors, the fp8 free, mtp_init and the spec
// snapshot banks): at 32K these cost ~1 GiB/card, and allocating them inside dev_alloc_tp pushed the
// decode-resident state over the HBM edge (the 32K OOM at the MTP upload). They are needed only from
// the prefill sweep onward; hidden_slots is reused as the NR row buffer after prefill (rows 0..NR-1),
// legal because the sweep has consumed its positions by then.
static void dev_alloc_pfslots(Dev& d) {
    CK(hipSetDevice(d.id));
    CK(hipMalloc((void**)&d.hidden_slots, (size_t)g_pf_cap * Q27_HID * 2));  CK(hipMemset(d.hidden_slots, 0, (size_t)g_pf_cap * Q27_HID * 2));
    // No full-width pinned staging: the prefill grows a per-window pinned buffer lazily (the
    // g_pf_cap-sized pin exhausted the ROCm pinned window at 32K and killed inv_slots below).
    d.emb_pin = nullptr; d.emb_pin_cap = 0;
    CK(hipMalloc((void**)&d.pend_slots, (size_t)g_pf_cap * Q27_HID * 4));    CK(hipMemset(d.pend_slots, 0, (size_t)g_pf_cap * Q27_HID * 4));
    // inv_slots is passed as the invp DEVICE pointer by prefill norm variants (SPEC=0 paths read
    // it on-device), so it must stay pinned/device-visible. The 32K pinned-window exhaustion that
    // motivated the pageable attempt is banked as a separate prefill-architecture item.
    CK(hipHostMalloc(&d.inv_slots, (size_t)g_pf_cap * 4, hipHostMallocDefault));   // per-position 1/rms
}

// One layer on one card. Identical topology to run_layer(); the two differences are that every
// projection runs on a shard and that the two mixer outputs are PARTIAL SUMS that must be
// all-reduced before the residual add.

// ---- TP PER-STAGE ATTRIBUTION (Q27_TP_PROFILE=1) --------------------------------------------
// The TP layer issues ~20 async kernels per layer onto one stream, so wall time alone cannot say
// which of them owns the ~21.3 ms/token compute term. This inserts a stream drain and a timer per
// stage. That DISTORTS absolute wall (every stage pays a full drain instead of pipelining), so the
// profile is for ATTRIBUTION ONLY and its total must never be quoted as a performance number.
// down_proj is bucketed separately because TP makes it K=17408/4=4352, a shape no measured kernel
// geometry covers.
enum { TPF_NORM, TPF_QUANT, TPF_PROJ_FP8, TPF_FP8_QKV, TPF_FP8_Z, TPF_FP8_O,
        TPF_PROJ_NVFP4, TPF_DOWN, TPF_ATTN,
       TPF_GDN_STEP, TPF_GDN_CONV, TPF_GDN_GEMV, TPF_ELEM,
       TPF_DR_PRE, TPF_DR_FC, TPF_DR_QKV, TPF_DR_ATTN, TPF_DR_O, TPF_DR_MLP, TPF_DR_HEAD,   // draft pass (2026-09-28)
       TPF_TR_MLP, TPF_TR_ATTN, TPF_TR_GDN, TPF_TR_CONS,                                   // trunk NR path stages
       TPF_LL_PUSH, TPF_LL_RED,                                                              // Q27_NR_P2P=4 site halves (2026-09-28)
       TPF_N };
static const char* TPF_NAME[TPF_N] = { "rmsnorm", "quantize", "proj_fp8",
                                       "fp8_in_qkv", "fp8_in_z", "fp8_out_proj", "proj_nvfp4",
                                       "down_proj(K=4352)", "attention",
                                       "gdn_step", "gdn_conv", "gdn_gemv", "elementwise",
                                       "draft_pre", "draft_fc", "draft_qkv", "draft_attn", "draft_o", "draft_mlp", "draft_head",
                                       "trunk_mlp", "trunk_attn", "trunk_gdn", "trunk_consumer", "ll_push", "ll_spin_reduce" };
static double g_tpf[Q27_MAX_DEVICES][TPF_N] = {{0}};
// ---- Q27_PF_PROFILE (2026-09-28): prefill-sweep attribution per OUTER CHUNK. The sweep's existing tr(L, ph) phase hook
// records one hipEvent on the compute stream per phase boundary (no sync); at the chunk end (stream idle) consecutive events
// are differenced: full layer 0->1 qkv proj, 1->2 attention, 2->3 o_proj, 3->next-layer-0/job-end MLP; GDN layer 0->5
// in_qkv/z proj, 5->6 a/b, 6->7 conv, 7->8 scan, 8->9 out_proj, 9->next MLP; job-end->next-job-0 = the ring boundary
// (input wait + P2P copy + host lateness); export phase bracketed on its own stream. ATTRIBUTION ONLY, no drains.
struct PfProf {
    static const int CAP = 16384;
    hipEvent_t ev[CAP]; short L[CAP]; short ph[CAP]; int n = 0; bool made = false; long dropped = 0;
    void ensure() { if (made) return; for (int i = 0; i < CAP; ++i) CK(hipEventCreateWithFlags(&ev[i], 0)); made = true; }
    void mark(hipStream_t s, int l, int p) { if (n >= CAP) { ++dropped; return; } ensure(); CK(hipEventRecord(ev[n], s)); L[n] = (short)l; ph[n] = (short)p; ++n; }
};
static PfProf g_pfp[Q27_MAX_DEVICES];
static int g_pf_profile = -1;
static inline int pf_profile_on() { if (g_pf_profile < 0) g_pf_profile = q27_env_int("Q27_PF_PROFILE", 0); return g_pf_profile; }
static void pf_profile_report(int g, int sw_beg, int sw_end) {   // call with the card's streams idle
    PfProf& P = g_pfp[g]; if (P.n < 2) { P.n = 0; return; }
    enum { B_QKV, B_ATTN, B_O, B_MLP, B_GPROJ, B_AB, B_CONV, B_SCAN, B_OUT, B_GAP, B_EXPORT, B_OTHER, B_N };
    static const char* NM[B_N] = { "proj_qkv", "attention", "o_proj", "mlp", "gdn_proj", "gdn_ab", "gdn_conv", "gdn_scan", "out_proj", "ring_gap", "export", "other" };
    double acc[B_N] = {0}; long cnt[B_N] = {0};
    for (int i = 1; i < P.n; ++i) {
        float ms = 0.f; if (hipEventElapsedTime(&ms, P.ev[i - 1], P.ev[i]) != hipSuccess) continue;
        const int a = P.ph[i - 1], b = P.ph[i]; int k = B_OTHER;
        if (b == 1) k = B_QKV; else if (b == 2) k = B_ATTN; else if (b == 3) k = B_O;
        else if (b == 5) k = B_GPROJ; else if (b == 6) k = B_AB; else if (b == 7) k = B_CONV; else if (b == 8) k = B_SCAN; else if (b == 9) k = B_OUT;
        else if (b == 0 && (a == 3 || a == 9)) k = B_MLP; else if (b == 99 && (a == 3 || a == 9)) k = B_MLP;
        else if (b == 0 && a == 99) k = B_GAP; else if (b == 97 && a == 98) k = B_EXPORT;
        acc[k] += ms; ++cnt[k];
    }
    float wall = 0.f; hipEventElapsedTime(&wall, P.ev[0], P.ev[P.n - 1]);
    double busy = 0; for (int k = 0; k < B_N; ++k) if (k != B_GAP) busy += acc[k];
    std::fprintf(stderr, "Q27_PF_PROFILE card=%d chunk=[%d,%d) events=%d dropped=%ld wall_ms=%.1f busy_ms=%.1f", g, sw_beg, sw_end, P.n, P.dropped, wall, busy);
    for (int k = 0; k < B_N; ++k) if (cnt[k]) std::fprintf(stderr, " %s=%.1f/%ld", NM[k], acc[k], cnt[k]);
    std::fprintf(stderr, "\n"); std::fflush(stderr);
    P.n = 0; P.dropped = 0;
}
static bool   g_tp_profile = false;
// Q27_TP_LEAD=1 (implies Q27_TP_PROFILE): per stage, how far AHEAD of the GPU the host was when it
// submitted. lead = (GPU time at which the stage's beg marker completed, from a token-start
// reference event) - (host time of the submit, from the same instant). A lead of only a few us
// means the queue was EMPTY when the host got there: the stage then pays dispatch latency in the
// open. The event-based profile alone cannot tell that idle from kernel time; this can.
static bool   g_tp_lead    = false;
// Fused rmsnorm+quantize. ON by default: measured +5.45% (27.749 -> 26.314 ms/token, three
// interleaved pairs, distributions non-overlapping) with tokens IDENTICAL to the oracle in all
// four arms. Q27_FUSE=0 rolls back to the separate kernels; 2 = fp8 pair only, 3 = perm pair only
// (that split is what located the fp8 scale bug in one run and is worth keeping).
static int    g_tp_fuse    = 1;

// EVENT-BASED, NOT DRAIN-BASED. A drain per stage serialises the whole layer and inflated the
// measured compute from ~21.3 to 32.6 ms/token, which does not distort every bucket equally: a
// bucket's inflation is proportional to its STAGE COUNT, so the many-small-stage buckets
// (elementwise, quantize) were over-weighted against the few-large-stage ones. hipEvents are
// recorded in-stream and never block the host, so the stream still pipelines and the shares are
// real. Events are read back once per token, after the drain the layer loop already performs.
struct TpEv {
    static const int CAP = 2048;                 // ~21 stages x 64 layers = ~1344 per token
    hipEvent_t beg[CAP], end[CAP];
    int        bucket[CAP];
    int        n = 0;
    long       dropped = 0, peak = 0;   // calls past CAP are SILENTLY untimed -- count them
    bool       made = false;
    // ---- Q27_TP_LEAD state
    hipEvent_t ref;                     // token-start reference, recorded on an idle stream
    double     host_ref = 0;            // tp_now() at that instant (ms)
    double     host_sub[CAP];           // tp_now() just before each beg record (ms)
    double     lead_sum[TPF_N] = {0};
    long       lead_n[TPF_N] = {0}, lead_starved[TPF_N] = {0};
    long       lead_hist[6] = {0};      // <5, 5-12, 12-30, 30-100, 100-1000, >=1000 us
    void ensure() {
        if (made) return;
        for (int i = 0; i < CAP; ++i) { CK(hipEventCreate(&beg[i])); CK(hipEventCreate(&end[i])); }
        CK(hipEventCreate(&ref));
        made = true;
    }
    void token_start(hipStream_t s) {           // call when the stream is idle, before any stage
        ensure(); CK(hipEventRecord(ref, s)); host_ref = tp_now();
    }
    void drain_into(double* acc) {               // call AFTER the stream is known idle
        if (n > peak) peak = n;
        for (int i = 0; i < n; ++i) {
            float ms = 0; CK(hipEventElapsedTime(&ms, beg[i], end[i]));
            acc[bucket[i]] += ms;
            if (g_tp_lead) {
                float gb = 0; CK(hipEventElapsedTime(&gb, ref, beg[i]));       // GPU: ref -> beg
                const double lead_us = ((double)gb - (host_sub[i] - host_ref)) * 1e3;
                const int b = bucket[i];
                lead_sum[b] += lead_us; lead_n[b]++;
                if (lead_us < 12.0) lead_starved[b]++;
                const int h = lead_us < 5 ? 0 : lead_us < 12 ? 1 : lead_us < 30 ? 2 :
                              lead_us < 100 ? 3 : lead_us < 1000 ? 4 : 5;
                lead_hist[h]++;
            }
        }
        n = 0;
    }
};
static TpEv g_tpev[Q27_MAX_DEVICES];

// NB: the parameter is `bkt`, not `bucket` -- naming it `bucket` made the expansion rewrite the
// struct member `_e.bucket[...]` into `_e.<enumerator>[...]`. Classic macro capture.
#define TPF(bkt, stream, call) do {                                           \
    if (!g_tp_profile) { call; }                                              \
    else { TpEv& _e = g_tpev[g];                                              \
           if (_e.n >= TpEv::CAP) { _e.dropped++; call; }                     \
           else { const int _i = _e.n++;                                      \
                  _e.bucket[_i] = (bkt);                                      \
                  if (g_tp_lead) _e.host_sub[_i] = tp_now();                  \
                  CK(hipEventRecord(_e.beg[_i], stream));                     \
                  call;                                                       \
                  CK(hipEventRecord(_e.end[_i], stream)); } }                 \
} while (0)

// `pending` is the PREVIOUS layer's residual-2 add, folded into this layer's input norm instead of
// being its own launch. Residual 1 folds into post_norm the same way. That deletes all 128
// q27_add_inplace launches per token; the caller applies the final pending add before final_norm.
static thread_local bool g_pf_sweep = false;   // prefill sweep: preenq machinery OFF
// The first plain decode pass makes all kernel modules resident before any
// host-published wait is enqueued. HIP first-use loading may synchronize a
// device, which cannot finish behind a flag this same host has yet to publish.
static thread_local bool g_plain_warmed = false;
static thread_local bool g_nr_warmed = false;
// ---- Q27_TPSR: expand this layer's block-scaled FP8 MLP shards into the card's transient per-ROW int8
// shard (out of place, ~67 MB of HBM traffic) and stamp the three q27_i8r_t views the rocBLAS prefill
// GEMMs read. Called once per layer at the batched sweep's layer head, on the compute stream, so it is
// stream-ordered before the first chunk's MLP and after the previous layer's last one. Per-row scale of
// the down K-slice is slice-local: the GEMM's int32 partial is scaled on this card before collective #2
// sums the four partials, so slice-local rows are the right units (the decode's K-view precedent). ----
static bool tpsr_mlp_view(Dev& d, const q27_layer_t* L, hipStream_t s) {
    if (!L) return false;
    if (d.tpsr_mlayer == L->layer && d.tpsr_g8r.w) return true;
    if (q27_env_flag("Q27_TPSR_NV", false)) {
        if (!d.tpsr_mw || !d.tpsr_ms || !d.tpsr_nv_s64) return false;
        const q27_nvfp4_t* src[3] = {&L->gate, &L->up, &L->down};
        q27_i8r_t* dst[3] = {&d.tpsr_g8r, &d.tpsr_u8r, &d.tpsr_d8r};
        signed char* w = d.tpsr_mw; float* sc = d.tpsr_ms;
        for (int i = 0; i < 3; ++i) {
            q27_i8g_t g64{};
            if (!src[i]->w || !q27_nvfp4_to_i8g64_into(src[i], &g64, w, d.tpsr_nv_s64, s) ||
                !q27_i8g64_rowify(&g64, dst[i], sc, s)) return false;
            w += (size_t)src[i]->rows * src[i]->K; sc += src[i]->rows;
        }
        d.tpsr_mlayer = L->layer; return true;
    }
    if (!d.tpsr_mw || !d.tpsr_ms || !L->f8_gate.w || !L->f8_up.w || !L->f8_down.w || !L->f8_gate.bs || !L->f8_down.bs) return false;
    const int nr = L->f8_gate.rows, K = L->f8_gate.K, dr = L->f8_down.rows, dk = L->f8_down.K;
    if (L->f8_up.rows != nr || L->f8_up.K != K) return false;
    signed char* wg = d.tpsr_mw; signed char* wu = wg + (size_t)nr * K; signed char* wd = wu + (size_t)nr * K;
    float* sg = d.tpsr_ms; float* su = sg + nr; float* sd = su + nr;
    if (!q27_sr_fp8bs_to_i8row(&L->f8_gate, wg, (size_t)K, sg, s) ||
        !q27_sr_fp8bs_to_i8row(&L->f8_up,   wu, (size_t)K, su, s) ||
        !q27_sr_fp8bs_to_i8row(&L->f8_down, wd, (size_t)dk, sd, s)) { d.tpsr_mlayer = -1; return false; }
    auto mk = [](q27_i8r_t* r, signed char* w, float* sc, float alpha, int rows, int Kk) {
        *r = q27_i8r_t{}; r->w = w; r->s = sc; r->alpha = alpha; r->rows = rows; r->K = Kk; r->ng = 1; r->gs = Kk; };
    mk(&d.tpsr_g8r, wg, sg, L->f8_gate.in_scale * L->f8_gate.wscale, nr, K);
    mk(&d.tpsr_u8r, wu, su, L->f8_up.in_scale   * L->f8_up.wscale,   nr, K);
    mk(&d.tpsr_d8r, wd, sd, L->f8_down.in_scale * L->f8_down.wscale, dr, dk);
    d.tpsr_mlayer = L->layer;
    return true;
}

// ---- Q27_TPSR_PROJ_RB (prefill v2): expand this layer's int8 per-64 projection MIRRORS into the card's
// transient per-ROW int8 stack (out of place) and stamp the q27_i8r_t views the rocBLAS GEMMs read. Same
// once-per-layer-per-sweep discipline as tpsr_mlp_view; the sweep is layer-major so the conversion is
// amortised over every chunk. Attention layers stack q|k|v and carry o; GDN layers stack in_qkv|in_z and
// carry out_proj. alpha = mirror alpha = in_scale*wscale (no *256: that is the fp8 kernels' e4m3 artefact).
static bool tpsr_proj_view(Dev& d, const q27_layer_t* L, int is_full, hipStream_t s) {
    if (!L) return false;
    if (d.tpsr_pmlayer == L->layer) return is_full ? (d.tpsr_q8r.w != nullptr) : (d.tpsr_iqkv8r.w != nullptr);
    if (!d.tpsr_pw || !d.tpsr_ps) return false;
    auto mk = [](q27_i8r_t* r, const signed char* w, const float* sc, float alpha, int rows, int Kk) {
        *r = q27_i8r_t{}; r->w = w; r->s = sc; r->alpha = alpha; r->rows = rows; r->K = Kk; r->ng = 1; r->gs = Kk; };
    d.tpsr_q8r = d.tpsr_k8r = d.tpsr_v8r = d.tpsr_o8r = d.tpsr_iqkv8r = d.tpsr_iz8r = d.tpsr_op8r = q27_i8r_t{};
    d.tpsr_pmlayer = L->layer;
    if (is_full) {
        const q27_i8g_t* mq = q27_fp8_i8g(&L->q_proj);
        const q27_i8g_t* mk_ = q27_fp8_i8g(&L->k_proj);
        const q27_i8g_t* mv = q27_fp8_i8g(&L->v_proj);
        const q27_i8g_t* mo = q27_fp8_i8g(&L->o_proj);
        if (!mq || !mk_ || !mv || !mo || !mq->w || !mk_->w || !mv->w || !mo->w) { d.tpsr_pmlayer = -1; return false; }
        const int QLb = mq->rows, KVLb = mk_->rows, HID = mq->K, OLb = mo->K;
        signed char* wq = d.tpsr_pw; signed char* wk = wq + (size_t)QLb * HID; signed char* wv = wk + (size_t)KVLb * HID;
        signed char* wo = d.tpsr_pw + (size_t)(QLb + 2 * KVLb) * HID;
        float* sq = d.tpsr_ps; float* sk = sq + QLb; float* sv = sk + KVLb; float* so = d.tpsr_ps + (QLb + 2 * KVLb);
        if (!q27_sr_i8g64_to_i8row(mq, wq, (size_t)HID, sq, s) ||
            !q27_sr_i8g64_to_i8row(mk_, wk, (size_t)HID, sk, s) ||
            !q27_sr_i8g64_to_i8row(mv, wv, (size_t)HID, sv, s) ||
            !q27_sr_i8g64_to_i8row(mo, wo, (size_t)OLb, so, s)) { d.tpsr_pmlayer = -1; return false; }
        mk(&d.tpsr_q8r, wq, sq, mq->alpha, QLb, HID);
        mk(&d.tpsr_k8r, wk, sk, mk_->alpha, KVLb, HID);
        mk(&d.tpsr_v8r, wv, sv, mv->alpha, KVLb, HID);
        mk(&d.tpsr_o8r, wo, so, mo->alpha, mo->rows, OLb);
    } else {
        const q27_i8g_t* miq = q27_fp8_i8g(&L->in_qkv);
        const q27_i8g_t* miz = q27_fp8_i8g(&L->in_z);
        const q27_i8g_t* mop = q27_fp8_i8g(&L->out_proj);
        if (!miq || !miz || !mop || !miq->w || !miz->w || !mop->w) { d.tpsr_pmlayer = -1; return false; }
        const int QKVLb = miq->rows, ZLb = miz->rows, HID = miq->K, OPLb = mop->K;
        signed char* wiq = d.tpsr_pw; signed char* wiz = wiq + (size_t)QKVLb * HID;
        signed char* wop = d.tpsr_pw + (size_t)(QKVLb + ZLb) * HID;
        float* siq = d.tpsr_ps; float* siz = siq + QKVLb; float* sop = d.tpsr_ps + (QKVLb + ZLb);
        if (!q27_sr_i8g64_to_i8row(miq, wiq, (size_t)HID, siq, s) ||
            !q27_sr_i8g64_to_i8row(miz, wiz, (size_t)HID, siz, s) ||
            !q27_sr_i8g64_to_i8row(mop, wop, (size_t)OPLb, sop, s)) { d.tpsr_pmlayer = -1; return false; }
        mk(&d.tpsr_iqkv8r, wiq, siq, miq->alpha, QKVLb, HID);
        mk(&d.tpsr_iz8r, wiz, siz, miz->alpha, ZLb, HID);
        mk(&d.tpsr_op8r, wop, sop, mop->alpha, mop->rows, OPLb);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Q27_EXL3 dispatch. A live trellis handle wins over every other format at the
// site; a null one leaves the incumbent branch byte-for-byte untouched, which
// is what keeps the 8-bit arm intact while this is brought up.
//
// EXL3 takes the bf16 NORM VECTOR, not the quantized activation every other
// format here consumes: it rotates its own input (suh then blockwise H128), so
// there is no in_scale and nothing to pre-quantize. The weight bytes it reads
// are the checkpoint bytes -- in*out*K/8 per call, decoded in LDS, never
// materialized.
extern std::atomic<unsigned long long> g_exl3_calls, g_exl3_rows, g_exl3_calls_m1;
// PER-SITE M=1 CENSUS. 95.5%% of prefill EXL3 calls are M=1 (983,044 of 1,029,124 on a
// 15360-token prompt = one per position per layer) and they carry only 7.7%% of the rows.
// Which SITE issues them decides the fix, so each per-position site is counted by name.
static int g_exl3_dup_o = 0;   // Q27_EXL3_DUP_O: extra idempotent repeats of the o_proj stage
static std::atomic<unsigned long long> g_ex_site_o{0}, g_ex_site_op{0}, g_ex_site_qkv{0}, g_ex_site_z{0};

static void* exl3_ws(Dev& d, int in, int out, int M) {
    (void)in; (void)out;                       // fixed layout at the widest projection
    const size_t need = q27_exl3_ws_bytes(Q27_EXL3_WMAX, Q27_EXL3_WMAX, M);
    if (need > d.exws_bytes) {
        if (d.exws) (void)hipFree(d.exws);
        if (hipMalloc(&d.exws, need) != hipSuccess) { d.exws = nullptr; d.exws_bytes = 0; return nullptr; }
        d.exws_bytes = need;
    }
    return d.exws;
}
static std::atomic<unsigned long long> g_exl3_n{0};
static inline void exl3_note(const char* who, int M) {
    const unsigned long long n = g_exl3_n.fetch_add(1, std::memory_order_relaxed);
    if (n == 0 || n == 1000 || n == 100000)
        std::fprintf(stderr, "Q27_EXL3_RUN #%llu %s M=%d\n", (unsigned long long)n, who, M), std::fflush(stderr);
}

static void* exl3_head_ws(Dev& d, int out, int M) {
    const size_t need = ((size_t)Q27_EXL3_WMAX * M * 2 + 255 & ~(size_t)255) + (size_t)out * M * 4 + 4096;
    if (need > d.exws_head_bytes) {
        if (d.exws_head) (void)hipFree(d.exws_head);
        if (hipMalloc(&d.exws_head, need) != hipSuccess) { d.exws_head = nullptr; d.exws_head_bytes = 0; return nullptr; }
        d.exws_head_bytes = need;
    }
    return d.exws_head;
}
// EXL3 lm_head. Same call shape as q27_proj_nvfp4 on the head, except it takes
// the bf16 NORM the quantize right above it consumes -- EXL3 rotates its own
// input, so the quantize is simply not part of this path.
static inline bool exl3_head(Dev& d, const q27_globals_t* G, const unsigned short* xbf,
                             float* y, int M, hipStream_t s) {
    if (!G || !G->ex_head.live() || !xbf) return false;
    void* ws = exl3_head_ws(d, G->ex_head.out, M);
    if (!ws) return false;
    const bool ok = q27_exl3_proj_f32(&G->ex_head, xbf, y, M, ws, s);
    if (ok) exl3_note("lm_head", M);
    return ok;
}

// bf16 out (activation) and fp32 out (partial). Both return false when the
// handle is dead so the caller falls through to the incumbent path.
// PROOF OF USE. A dispatcher that silently falls back to the incumbent format
// is indistinguishable from one that ran EXL3 -- same log, same shape of output
// -- so every EXL3 site announces itself the first time it fires. Without this
// a "measurement of EXL3" can easily be a measurement of NVFP4.

static inline bool exl3_bf16(Dev& d, const q27_exl3_t& w, const unsigned short* x,
                             unsigned short* y, int M, hipStream_t s, int ldx = 0, int ldy = 0) {
    if (!w.live()) return false;
    void* ws = exl3_ws(d, w.in, w.out, M);
    const bool ok = ws && q27_exl3_proj_bf16(&w, x, y, M, ws, s, ldx, ldy);
    if (ok) exl3_note("proj_bf16", M);
    return ok;
}
static inline bool exl3_f32(Dev& d, const q27_exl3_t& w, const unsigned short* x,
                            float* y, int M, hipStream_t s, int ldx = 0, int ldy = 0) {
    if (!w.live()) return false;
    void* ws = exl3_ws(d, w.in, w.out, M);
    const bool ok = ws && q27_exl3_proj_f32(&w, x, y, M, ws, s, ldx, ldy);
    if (ok) exl3_note("proj_f32", M);
    return ok;
}
// The EXL3 MLP for a row band: gate/up -> natural bf16 swiglu -> down. The
// activation lives in the EXL3 workspace because every incumbent swiglu emits a
// QUANTIZED activation, which is exactly what EXL3 does not want.
static inline bool exl3_mlp(Dev& d, const q27_layer_t* L, const unsigned short* xnorm, int ldx,
                            float* pa, float* pb, int ld_ab, float* part, int ldpart,
                            int IL, int M, hipStream_t s) {
    if (!L->ex_gate.live() || !L->ex_up.live() || !L->ex_down.live()) return false;
    if (!exl3_f32(d, L->ex_gate, xnorm, pa, M, s, ldx, ld_ab)) return false;
    if (!exl3_f32(d, L->ex_up,   xnorm, pb, M, s, ldx, ld_ab)) return false;
    void* ws = exl3_ws(d, L->ex_down.in, L->ex_down.out, M);
    if (!ws) return false;
    unsigned short* act = q27_exl3_ws_act(ws, L->ex_down.in, M, L->ex_down.out);
    if (!q27_exl3_swiglu_bf16(pa, pb, act, IL, M, ld_ab, IL, s)) return false;
    return exl3_f32(d, L->ex_down, act, part, M, s, IL, ldpart);
}

static void run_layer_tp(Dev& d, const q27_layer_t* L, int pos, int gsraw, int fsraw, int ctx,
                         int g, int ndev, TpColl& C, double* mark, const void* pending,
                         const q27_layer_t* Lnext = nullptr, int slot = 0,
                         const float* invin = nullptr,
                         signed char* xqout = nullptr, float* xsout = nullptr,
                         bool stop = false, int pfmode = 0,
                         signed char* xqin = nullptr, float* xsin = nullptr,
                         unsigned short* qkvaslot = nullptr,
                         unsigned short* normout = nullptr,
                         const unsigned short* qkvin = nullptr,
                         const unsigned short* zbin = nullptr,
                         float* partout = nullptr,
                         const float* accin = nullptr,
                         const float* invin2 = nullptr,
                         signed char* mixq_out = nullptr, float* mixs_out = nullptr) {
    hipStream_t s = d.stream;
    const float rshare = 1.0f / (float)ndev;   // Q27_TP3: each worker folds 1/ndev of the residual into its partial (was a hard-coded 0.25f)
    unsigned short* hid = d.hidden_slots + (size_t)slot * Q27_HID;
    // The caller passes the RAW layer-type counters; masking here keeps the rest of the body
    // unchanged and lets the pe2 pre-enqueue compute Lnext's slot indices exactly as the caller
    // would after its own increment.
    const int gslot = L->is_full ? 0 : gsraw;
    const int fslot = L->is_full ? fsraw : 0;
    const q27_tp_shard_t* const T = q27_tp_shard(g);   // Q27_TP3: this worker's shard geometry
    const int QL  = T->QL, KVL = T->KVL, OL = T->OL;
    const int QKVL = T->QKVL, ZL = T->ZL, IL = mlp_len(d.id);
    // Q27_TPSR: the MLP is the block-scaled FP8 shard (gate/up rows + down K-slice) -- natural-order
    // activations, so the post-norm quantises in natural order and the NVFP4 chain below is bypassed.
    const bool f8mlp = (L->f8_gate.w && L->f8_up.w && L->f8_down.w && L->f8_gate.bs && L->f8_up.bs && L->f8_down.bs);

    // Destination of the three PARTIAL projections. Under zero-copy-out this is card g's pinned
    // reduction slot, so the only bytes crossing PCIe are the final epilogue stores -- the weights,
    // activations and accumulators all stay in VRAM. The bf16 collective keeps the VRAM path.
    // Q27_ZCOUT_NEGCTL=1 keeps the producer writing VRAM while the D2H stays deleted, so the
    // reduction consumes a pinned buffer nobody filled. It MUST corrupt the output. If it does
    // not, the producers were never redirected and a passing token match proves nothing.
    // partout: the batched-collective #1 arm lands each position's PARTIAL in its own chunk slot
    // (pfmode 4) so one rendezvous can reduce the whole chunk.
    float* const part = partout ? partout
                                : ((g_coll_zcout && !g_coll_bf16 && !g_zcout_neg) ? g_coll_part[g]
                                                                                : (float*)d.mixer);
    // Each card folds its share of the residual into its own partial: all four hold the same
    // residual r, so reducing (p_i + r/4) gives sum(p_i) + r, the exact post-residual vector.
    // It rides in the projection epilogue rather than a separate launch, because a launch on all
    // four cards is uniform work that moves the barrier release instead of being absorbed.
    const unsigned short* const rad =
        (g_rn_hostss && !g_hostss_neg) ? (const unsigned short*)hid : nullptr;

    // ---- mixer, no collective inside: the GQA / value-head groups are never split ----
    // Both branches quantize the SAME normalised vector immediately after the norm; only the
    // scale differs. Hoist the scale so the pair can be fused into one launch.
    const float in_s = L->is_full ? L->q_proj.in_scale : L->in_qkv.in_scale;
    // pro_in: the PREVIOUS layer pre-enqueued this layer's input norm AND its whole prologue
    // (Q27_PREENQ_PROLOGUE), including the collective-1 D2H copy. Everything up to the barrier
    // is already in the stream queue; the iteration starts at the event sync.
    const bool pro_in = (C.pre_in[g] != 0);
    // Batched-prologue destinations. When the driver supplies per-position slots (pfmode 1) the
    // norm vector and the quantized activation MUST land there: the batched projections read the
    // slots, and every one of the four fused norm paths below used to hardcode d.norm/d.xq/d.xs.
    // Layer 0 has no pending, so it takes the FUSED fallback -- the one that hardcoded them.
    unsigned short* const normdst = normout ? normout : d.norm;
    signed char* const xqdst = xqin ? xqin : d.xq;
    float* const xsdst = xsin ? xsin : d.xs;
    // 1 = norm+quantize only;  2/3 = full/GDN body with the norm already done;  4 = full-layer
    // prologue -> partial, no barrier (batched collective #1).
    // The post-norm tail, callable from the normal flow (after collective #1) and from the
    // batched-collective arm: pfmode 5 calls it with THIS position's slice of the chunk-level
    // reduce (a_in) and its 1/rms (iv_in) instead of the single-slot g_coll_acc / C.inv_d.
    auto tail_fn = [&](bool pe1_done, const float* a_in, const float* iv_in) {
        const float* const p_acc = a_in;
        const float* const p_inv = iv_in;
        signed char* const xq_t = xqout ? xqout : d.xq;         // the batched sweep routes the tail's
        float* const xs_t = xsout ? xsout : d.xs;               // quantize into per-position slots
        TPF(TPF_NORM, s, {
            if (pe1_done) { /* already enqueued before the barrier */ }
            else if (g_rn_hostss) {
                if (f8mlp) q27_rmsnorm_hostss_fp8(p_acc, L->post_norm, hid,   // Q27_TPSR: natural order for the FP8 gate/up
                                        g_rn_dropy ? nullptr : d.norm, xq_t, xs_t,
                                        Q27_HID, 1, L->f8_gate.in_scale, p_inv, s);
                else
                q27_rmsnorm_hostss_perm(p_acc, L->post_norm, hid,
                                        g_rn_dropy ? nullptr : d.norm, xq_t, xs_t,
                                        Q27_HID, 1, L->gate.in_scale, p_inv, s);
                for (int rp = 0; rp < g_slow_tail; ++rp)               // sensitivity probe, idempotent
                    q27_rmsnorm_hostss_perm(p_acc, L->post_norm, hid,
                                            g_rn_dropy ? nullptr : d.norm, xq_t, xs_t,
                                            Q27_HID, 1, L->gate.in_scale, p_inv, s);
                for (int rp = 0; rp < g_slow_null; ++rp) q27_launch_nothing(s);   // bare dispatch cost
            } else
            if (!(g_tp_fuse == 1 || g_tp_fuse == 3) ||
                !(  (g_rn_split && q27_rmsnorm_quant_split_perm(hid, L->post_norm, d.norm, xq_t, xs_t,
                            Q27_HID, 1, L->gate.in_scale, d.mixer, g_coll_bf16 ? 1 : 0, d.rnpart, s))
                 || q27_rmsnorm_quant_perm(hid, L->post_norm, d.norm, xq_t, xs_t,
                            Q27_HID, 1, L->gate.in_scale, g_coll_zc ? (const void*)g_coll_acc : (const void*)d.mixer, g_coll_bf16 ? 1 : 0, s))) {
                if (g_coll_bf16) q27_add_inplace_bf16(hid, (const unsigned short*)d.mixer, Q27_HID, s);
                else             q27_add_inplace(hid, d.mixer, Q27_HID, s);   // residual 1 fallback
                q27_rmsnorm(hid, L->post_norm, d.norm, Q27_HID, 1, s);
                q27_quant_perm(d.norm, xq_t, xs_t, Q27_HID, L->gate.in_scale, s);
            }
        });

    };

    if (pfmode == 5) {   // batched collective #1: tail only, acc from the chunk
        C.pre_in[g] = 0;
        tail_fn(false, accin ? accin : g_coll_acc, invin2 ? invin2 : (C.inv_d + g));
        return;
    }
    if (pfmode == 2 || pfmode == 3 || pfmode == 4 || (pfmode == 6 && g_pf_noredo)) { C.pre_in[g] = 0; goto pf_skip_norm; }
    TPF(TPF_NORM, s, {
        if (pro_in) { C.pre_in[g] = 0; /* enqueued before the previous barrier */ }
        else if (g_rn_hostss && pending) {
            if (g_fp8x4) {
                q27_rmsnorm_hostss_perm(g_coll_acc, L->input_norm, hid,
                                   (g_rn_dropy && L->is_full) ? nullptr : (normout ? normout : d.norm), d.xq, d.xs,
                                   Q27_HID, 1, in_s, C.inv_d + g, s);
                for (int rp = 0; rp < g_slow_tail; ++rp)           // sensitivity probe, idempotent
                    q27_rmsnorm_hostss_perm(g_coll_acc, L->input_norm, hid,
                                       (g_rn_dropy && L->is_full) ? nullptr : (normout ? normout : d.norm),
                                       xqin ? xqin : d.xq, xsin ? xsin : d.xs,
                                       Q27_HID, 1, in_s, invin ? invin : (C.inv_d + g), s);
            } else {
                // The mixer source is the PENDING residual (this position's previous layer).
                // Position-major it equals g_coll_acc's content; the layer-major prefill sweep
                // must read the position's own slot, or it folds the previous position's mixer.
                // Same for invin: C.inv_d is a single slot written by EVERY collective, so the
                // sweep passes the per-position 1/rms the previous layer's collective computed.
                q27_rmsnorm_hostss_fp8((const float*)pending, L->input_norm, hid,
                                   (g_rn_dropy && L->is_full) ? nullptr : (normout ? normout : d.norm),
                                   xqin ? xqin : d.xq, xsin ? xsin : d.xs,
                                   Q27_HID, 1, in_s, invin ? invin : (C.inv_d + g), s);
                for (int rp = 0; rp < g_slow_tail; ++rp)           // sensitivity probe, idempotent
                    q27_rmsnorm_hostss_fp8((const float*)pending, L->input_norm, hid,
                                       (g_rn_dropy && L->is_full) ? nullptr : (normout ? normout : d.norm),
                                       xqin ? xqin : d.xq, xsin ? xsin : d.xs,
                                       Q27_HID, 1, in_s, invin ? invin : (C.inv_d + g), s);
            }
            for (int rp = 0; rp < g_slow_null; ++rp) q27_launch_nothing(s);   // bare dispatch cost
        } else
        if (!(g_tp_fuse == 1 || g_tp_fuse == 2) ||
            !(  (g_rn_split && (g_fp8x4
                    ? q27_rmsnorm_quant_split_perm(hid, L->input_norm, normdst, xqdst, xsdst,
                        Q27_HID, 1, in_s, pending, g_coll_bf16 ? 1 : 0, d.rnpart, s)
                    : q27_rmsnorm_quant_split_fp8(hid, L->input_norm, normdst, xqdst, xsdst,
                        Q27_HID, 1, in_s, pending, g_coll_bf16 ? 1 : 0, d.rnpart, s)))
             || (g_fp8x4
                    ? q27_rmsnorm_quant_perm(hid, L->input_norm, normdst, xqdst, xsdst,
                        Q27_HID, 1, in_s, pending, g_coll_bf16 ? 1 : 0, s)
                    : q27_rmsnorm_quant_fp8(hid, L->input_norm, normdst, xqdst, xsdst,
                        Q27_HID, 1, in_s, pending, g_coll_bf16 ? 1 : 0, s)))) {
            if (pending) {
                if (g_coll_bf16) q27_add_inplace_bf16(hid, (const unsigned short*)pending, Q27_HID, s);
                else             q27_add_inplace(hid, (const float*)pending, Q27_HID, s);
            }
            g_rnq_fb.fetch_add(1, std::memory_order_relaxed);
            q27_rmsnorm(hid, L->input_norm, normdst, Q27_HID, 1, s);
            if (g_fp8x4) q27_quant_perm(normdst, xqdst, xsdst, Q27_HID, in_s, s);
            else         q27_quant_fp8(normdst, xqdst, xsdst, Q27_HID, in_s, s);
        }
    });
    if (pfmode == 1) return;          // the batched driver takes over: q/k/v run per chunk
pf_skip_norm:
    auto enq_pro = [&](const q27_layer_t* Lx, int fsx, int gsx) {
        if (g_nr_p2p == 2) for (int k = 0; k < ndev; ++k) if (k != g) CK(hipStreamWaitEvent(s, C.ev_nrp_rd[0][k], 0));   // pull: slot 0 free on every peer
    if (Lx->is_full) {
        // mode 2: q/k/v were batched by the driver, into qkvaslot. Mode 4 (the attention chunk form)
        // supplies the same slot; until this line it re-ran all three per position into d.qkva and
        // the attention read the slot (the default-build trace: 48 dead K=5120 launches/position).
        if (pfmode != 2 && !(g_pf_noredo && pfmode == 4 && qkvaslot)) {
        const unsigned short* nrmp_a = normout ? normout : d.norm;
        TPF(TPF_PROJ_FP8, s, {
            if (exl3_bf16(d, Lx->ex_q, nrmp_a, d.qkva, 1, s)) { g_ex_site_qkv.fetch_add(1, std::memory_order_relaxed); }
            else if (g_fp8x4) q27_proj_nvfp4_bf16(&Lx->q4, d.xq, d.xs, d.qkva, s);
            else if (!q27_proj_fp8_bf16(&Lx->q_proj, d.xq, d.xs, d.qkva, s)) {
                q27_proj_fp8(&Lx->q_proj, d.xq, d.xs, d.pa, s);
                q27_f2bf_vec(d.pa, d.qkva, QL, s);
            } });
        // Q27_KV_FUSE: k_proj and v_proj are 256-row shards whose launches cost ~25 us each
        // in-engine against 5.5 us of work; one launch for both removes 16 launches/token.
        bool kv_done = false;
        if (Lx->ex_k.live() && Lx->ex_v.live())
            kv_done = exl3_bf16(d, Lx->ex_k, nrmp_a, d.qkva + QL, 1, s) &&
                      exl3_bf16(d, Lx->ex_v, nrmp_a, d.qkva + QL + KVL, 1, s);
        if (!kv_done && g_kv_fuse && !g_fp8x4)
            TPF(TPF_PROJ_FP8, s, {
                kv_done = q27_proj_fp8_bf16_2(&Lx->k_proj, &Lx->v_proj, d.xq, d.xs,
                                              d.qkva + QL, d.qkva + QL + KVL,
                                              g_kv_fuse == 2 ? 1 : 0, s) != 0;
                if (kv_done) g_kv_fused.fetch_add(1, std::memory_order_relaxed); });
        if (!kv_done) {
        TPF(TPF_PROJ_FP8, s, {
            if (g_fp8x4) q27_proj_nvfp4_bf16(&Lx->k4, d.xq, d.xs, d.qkva + QL, s);
            else if (!q27_proj_fp8_bf16(&Lx->k_proj, d.xq, d.xs, d.qkva + QL, s)) {
                q27_proj_fp8(&Lx->k_proj, d.xq, d.xs, d.pa, s);
                q27_f2bf_vec(d.pa, d.qkva + QL, KVL, s);
            } });
        TPF(TPF_PROJ_FP8, s, {
            if (g_fp8x4) q27_proj_nvfp4_bf16(&Lx->v4, d.xq, d.xs, d.qkva + QL + KVL, s);
            else if (!q27_proj_fp8_bf16(&Lx->v_proj, d.xq, d.xs, d.qkva + QL + KVL, s)) {
                q27_proj_fp8(&Lx->v_proj, d.xq, d.xs, d.pa, s);
                q27_f2bf_vec(d.pa, d.qkva + QL + KVL, KVL, s);
            } });
        }
        }
        signed char* kc = d.kc + d.kv_off[fsx];   q27_kvs_t* ks = d.ks + d.kvs_off[fsx];
        signed char* vc = d.vc + Q27_VOFF(d.kv_off[fsx]);   q27_kvs_t* vs = d.vs + Q27_VSOFF(d.kvs_off[fsx]);
        // A layer is stored full width only on the card that OWNS it (its prefill reads every
        // head); every other card holds just the head-quarter decode reads. Stride and head offset
        // are therefore per LAYER, not per residency mode. TP fills these with KVL / 0.
        const int kvstride = d.kv_rowstride[fsx];
        const int hoff0    = d.kv_hoff[fsx];
        TPF(TPF_ATTN, s, q27_attn_prep_tp(qkvaslot ? qkvaslot : d.qkva, Lx->q_norm, Lx->k_norm, kc, ks, vc, vs, d.qh, pos, T->attn_g, kvstride, hoff0, s));
        int attn_fq = 0;
        // q_proj_raw is the q_proj block of the qkv buffer: decode reads the output GATE
        // from it at [h*512+256+d]. In pfmode 2 the projections landed in the per-position
        // qkvaslot, and d.qkva is untouched (zeros -> sigmoid(0)=0.5 scaled the attention).
        TPF(TPF_ATTN, s, attn_fq = q27_attn_decode_tp(d.qh, kc, ks, vc, vs, qkvaslot ? qkvaslot : d.qkva, d.mix6, pos, T->attn_g, kvstride, hoff0,
                        (g_gdn_fq && !g_fp8x4) ? d.xq : nullptr, (g_gdn_fq && !g_fp8x4) ? d.xs : nullptr,
                        (g_gdn_fq && !g_fp8x4) ? Lx->o_proj.in_scale : 0.0f, s));
        // Q27_PF_ATT_B (prefill, mode 4): quantize the attention mixer into this position's CHUNK
        // slot and return; the caller runs ONE o_proj over the chunk into part_slots.
        if (Lx->ex_o.live()) { static std::atomic<int> t1{0}; if (t1.fetch_add(1) == 0)
            std::fprintf(stderr, "Q27_OTRACE full-attn o_proj: pfmode=%d mixq_out=%p attn_fq=%d fp8x4=%d layer=%d\n",
                         pfmode, (void*)mixq_out, (int)attn_fq, (int)g_fp8x4, Lx->layer), std::fflush(stderr); }
        if (mixq_out && pfmode == 4 && !attn_fq && !g_fp8x4) {
            TPF(TPF_QUANT, s, q27_quant_fp8(d.mix6, mixq_out, mixs_out, OL, Lx->o_proj.in_scale, s));
            return;
        }
        if (!attn_fq)
            TPF(TPF_QUANT, s, {
                if (g_fp8x4) {
                    q27_quant_perm(d.mix6, d.xq, d.xs, OL, Lx->o_proj.in_scale, s);
                    for (int rp = 0; rp < g_slow_mixq; ++rp)
                        q27_quant_perm(d.mix6, d.xq, d.xs, OL, Lx->o_proj.in_scale, s);
                } else {
                    q27_quant_fp8(d.mix6, d.xq, d.xs, OL, Lx->o_proj.in_scale, s);
                    for (int rp = 0; rp < g_slow_mixq; ++rp)           // idempotent
                        q27_quant_fp8(d.mix6, d.xq, d.xs, OL, Lx->o_proj.in_scale, s);
                } });
        TPF(TPF_PROJ_FP8, s, {
            if (exl3_f32(d, Lx->ex_o, d.mix6, part, 1, s)) { g_ex_site_o.fetch_add(1, std::memory_order_relaxed);
                // Q27_EXL3_DUP_O=1: price THIS stage as d(wall). The projection writes `part`
                // outright (the residual share is a separate pass below), so running it twice is
                // idempotent and the wall delta is the stage's recoverable cost -- per
                // recoverable-wall-not-profiled-ms. 983,040 of 1,029,124 prefill EXL3 calls are
                // this site; the question is whether that is TIME or just COUNT.
                for (int rp = 0; rp < g_exl3_dup_o; ++rp) (void)exl3_f32(d, Lx->ex_o, d.mix6, part, 1, s);
                if (rad) q27_add_res_share(part, hid, rshare, Q27_HID, s);
            } else if (g_fp8x4) {
                q27_nvfp4_set_res((rad && g_rn_epi) ? rad : nullptr, rshare);
                q27_proj_nvfp4(&Lx->o4, d.xq, d.xs, part, s);
                q27_nvfp4_set_res(nullptr, 0.f);
                if (rad && !g_rn_epi) q27_add_res_share(part, hid, rshare, Q27_HID, s);
            } else if (!g_coll_bf16 || !q27_proj_fp8_bf16(&Lx->o_proj, d.xq, d.xs,
                                                   (unsigned short*)d.mixer, s)) {
                if (!rad || !g_rn_epi || !q27_proj_fp8_res(&Lx->o_proj, d.xq, d.xs, part, rad, rshare, s)) {
                    q27_proj_fp8(&Lx->o_proj, d.xq, d.xs, part, s);
                    if (rad) q27_add_res_share(part, hid, rshare, Q27_HID, s);
                }
            }
        });   // PARTIAL, 1536 cols
    } else {
        // pfmode 3 (batched GDN prologue): the input norm + quantize ran in mode 1 and the two
        // K=5120 projections were batched across the chunk by the driver; this pass reads the
        // per-position slots instead of d.qkv/d.zbuf and skips its own projections.
        unsigned short* qkvp = qkvin ? const_cast<unsigned short*>(qkvin) : d.qkv;
        const unsigned short* zbp = zbin ? zbin : d.zbuf;
        const unsigned short* nrmp = normout ? normout : d.norm;
        // Mode 6 (chunk out_proj / pipelined paths) also reads the tile slots; until 4303082 it
        // re-ran both K=5120 projections per position into d.qkv/d.zbuf and never read them
        // (the 128-token trace: 96 dead launches x 16.7 us per position, ~18% of GPU busy).
        if (pfmode != 3 && !(g_pf_noredo && qkvin && zbin)) {
        TPF(TPF_FP8_QKV, s, {
            if (Lx->ex_iqkv[0].live()) {   // q | k | v, three handles, written at their offsets
                int off = 0;
                for (int r = 0; r < 3; ++r) { exl3_bf16(d, Lx->ex_iqkv[r], nrmp, d.qkv + off, 1, s); off += Lx->ex_iqkv[r].out; }
                g_ex_site_qkv.fetch_add(3, std::memory_order_relaxed);
            }
            else if (g_fp8x4) q27_proj_nvfp4_bf16(&Lx->iqkv4, d.xq, d.xs, d.qkv, s);
            else if (!q27_proj_fp8_bf16(&Lx->in_qkv, d.xq, d.xs, d.qkv, s)) {
                q27_proj_fp8(&Lx->in_qkv, d.xq, d.xs, d.pa, s);
                q27_f2bf_vec(d.pa, d.qkv, QKVL, s);
            } });
        TPF(TPF_FP8_Z, s, {
            if (exl3_bf16(d, Lx->ex_iz, nrmp, d.zbuf, 1, s)) { g_ex_site_z.fetch_add(1, std::memory_order_relaxed); }
            else if (g_fp8x4) q27_proj_nvfp4_bf16(&Lx->iz4, d.xq, d.xs, d.zbuf, s);
            else if (!q27_proj_fp8_bf16(&Lx->in_z, d.xq, d.xs, d.zbuf, s)) {
                q27_proj_fp8(&Lx->in_z, d.xq, d.xs, d.pb, s);
                q27_f2bf_vec(d.pb, d.zbuf, ZL, s);
            } });
        }
        // Only this card's value heads. Was Q27_GDN_VH (all 48) on every card: 4x redundant.
        const int VHL = T->VHL;
        TPF(TPF_GDN_GEMV, s, {
            if (!q27_bf16_gemv2(Lx->in_a, Lx->in_b, nrmp, d.ab, d.bb, VHL, Q27_HID, s)) {
                q27_bf16_gemv(Lx->in_a, nrmp, d.ab, VHL, Q27_HID, s);
                q27_bf16_gemv(Lx->in_b, nrmp, d.bb, VHL, Q27_HID, s);
            }
        });
        TPF(TPF_GDN_CONV, s, q27_gdn_conv_tp_r(qkvp, d.conv + d.conv_off[gsx], d.conv_wide[gsx], Lx->conv1d, T->gk_h0, T->gk_hn, s));
        // Quantize fused into the step's phase-4 epilogue: d.mix6 was consumed by nothing else.
        // Q27_PF_GDN_FQ (prefill mode 6 with the chunk out_proj): the step's fused quantize epilogue
        // writes the mixer straight into this position's chunk slot; no separate quantize launch.
        const bool fq_slot = (mixq_out && pfmode == 6 && g_pf_gdn_fq && !g_fp8x4);
        signed char* fq_q  = fq_slot ? mixq_out : (g_gdn_fq ? d.xq : nullptr);
        float*       fq_qs = fq_slot ? mixs_out : (g_gdn_fq ? d.xs : nullptr);
        const float  fq_is = (fq_slot || g_gdn_fq) ? Lx->out_proj.in_scale : 0.0f;
        TPF(TPF_GDN_STEP, s,
            g_dsplit
              ? q27_gdn_step_tp_ds(qkvp, zbp, d.ab, d.bb, Lx->A_log, Lx->dt_bias, Lx->gdn_norm,
                        d.S + d.s_off[gsx] + d.s_qoff[gsx], d.mix6, g, ndev, d.gdn_scr,
                        pos * Q27_LAYERS + Lx->layer, fq_q, fq_qs, fq_is, s)
              : q27_gdn_step_tp_r(qkvp, zbp, d.ab, d.bb, Lx->A_log, Lx->dt_bias, Lx->gdn_norm,
                        d.S + d.s_off[gsx] + d.s_qoff[gsx], d.mix6, T->gk_h0, T->gk_hn, fq_q, fq_qs, fq_is, s));
        if (fq_slot) return;
        // Q27_PF_OPROJ_B (prefill, mode 6): quantize the mixer into this position's CHUNK slot and
        // return; the caller runs ONE out_proj over the chunk (q27_proj_fp8_m2_res) into part_slots.
        if (mixq_out && pfmode == 6 && !g_gdn_fq && !g_fp8x4) {
            { static std::atomic<int> o1{0}; if (o1.fetch_add(1) == 0)
                std::fprintf(stderr, "Q27_OPROJ_TRACE chunk-quantize arm TAKEN (pfmode=6) ex_op.live=%d out_proj.w=%p\n",
                             (int)Lx->ex_op.live(), (void*)Lx->out_proj.w), std::fflush(stderr); }
            TPF(TPF_QUANT, s, q27_quant_fp8(d.mix6, mixq_out, mixs_out, ZL, Lx->out_proj.in_scale, s));
            return;
        }
        { static std::atomic<int> o2{0}; if (o2.fetch_add(1) == 0)
            std::fprintf(stderr, "Q27_OPROJ_TRACE PER-POSITION arm (pfmode=%d mixq_out=%p gdn_fq=%d fp8x4=%d) ex_op.live=%d\n",
                         pfmode, (void*)mixq_out, (int)g_gdn_fq, (int)g_fp8x4, (int)Lx->ex_op.live()), std::fflush(stderr); }
        if (!g_gdn_fq)
            TPF(TPF_QUANT, s, {
                if (g_fp8x4) {
                    q27_quant_perm(d.mix6, d.xq, d.xs, ZL, Lx->out_proj.in_scale, s);
                    for (int rp = 0; rp < g_slow_mixq; ++rp)
                        q27_quant_perm(d.mix6, d.xq, d.xs, ZL, Lx->out_proj.in_scale, s);
                } else {
                    q27_quant_fp8(d.mix6, d.xq, d.xs, ZL, Lx->out_proj.in_scale, s);
                    for (int rp = 0; rp < g_slow_mixq; ++rp)           // idempotent
                        q27_quant_fp8(d.mix6, d.xq, d.xs, ZL, Lx->out_proj.in_scale, s);
                } });
        TPF(TPF_FP8_O, s, {
            if (exl3_f32(d, Lx->ex_op, d.mix6, part, 1, s)) { g_ex_site_op.fetch_add(1, std::memory_order_relaxed);
                // EXL3 writes the partial directly; the residual share rides the
                // same epilogue the fp8 fallback below uses.
                if (rad) q27_add_res_share(part, hid, rshare, Q27_HID, s);
            } else if (g_fp8x4) {
                q27_nvfp4_set_res((rad && g_rn_epi) ? rad : nullptr, rshare);
                q27_proj_nvfp4(&Lx->op4, d.xq, d.xs, part, s);
                q27_nvfp4_set_res(nullptr, 0.f);
                if (rad && !g_rn_epi) q27_add_res_share(part, hid, rshare, Q27_HID, s);
            } else if (!g_coll_bf16 || !q27_proj_fp8_bf16(&Lx->out_proj, d.xq, d.xs,
                                                   (unsigned short*)d.mixer, s)) {
                if (!rad || !g_rn_epi || !q27_proj_fp8_res(&Lx->out_proj, d.xq, d.xs, part, rad, rshare, s)) {
                    q27_proj_fp8(&Lx->out_proj, d.xq, d.xs, part, s);
                    if (rad) q27_add_res_share(part, hid, rshare, Q27_HID, s);
                }
            }
        }); // PARTIAL, 1536 cols
    }
    };
    if (!pro_in || !g_preenq_prologue) enq_pro(L, fslot, gslot);   // flag off: prologue runs here
    // 4 (full layers) / 6 (GDN): the batched-collective #1 arm -- the per-position PARTIAL is in
    // partout and the tail waits for the chunk-level rendezvous.
    if (pfmode == 4 || pfmode == 6) return;
    // PRE-ENQUEUE the post-barrier tail. It is submitted HERE, before the barrier, and the GPU
    // waits on a flag rather than on the host, so the 10.83 us submission lands in pre-barrier
    // slack instead of on the critical path. The collective must then wait on an event covering
    // ONLY the copy: a full hipStreamSynchronize would also wait on the tail we just queued,
    // which cannot run until the host sets the flag -- deadlock.
    const bool pe1 = g_tail_preenq && g_preenq_ok && g_rn_hostss && !g_coll_bf16 && !g_coll_zcout
                     && !g_pf_sweep && g_plain_warmed && !(g_host_tail_mask & (1u << g));
    const bool fd1 = (g_coll_fusedrain || pe1) && !g_coll_bf16 && !g_coll_zcout;
    const bool mlp_pre = (!L->ex_gate.live()) && (pe1 && g_preenq_mlp && !f8mlp);   // EXL3 does the MLP in the branch below; the pre-enqueued copy would
                                 // run it a SECOND time, and on dropped NVFP4 weights.   // the whole MLP chain rides behind the flag-gated tail (NVFP4 chain only)
    bool mlp_fused = false;
    unsigned cg1 = 0;
    if (fd1 && !g_fd_neg && !(pro_in && g_preenq_prologue))   // copy skipped only when it rode pre-barrier
        cg1 = tp_stage_out(C, g, d.mixer, s);
    else if (fd1 && !g_fd_neg && pro_in) cg1 = C.pre_gen[g];   // it rode pre-barrier: its generation
    unsigned gn1 = 0;
    if (pe1) {
        if (!(g_coll_dflag && cg1)) CK(hipEventRecord(C.ev_copy[g], s));
        gn1 = ++C.gen[g];
        CK(hipStreamWaitValue32(s, (void*)(C.flag_d + g), gn1, hipStreamWaitValueEq, 0xffffffffu));
        if (f8mlp) q27_rmsnorm_hostss_fp8(g_coll_acc, L->post_norm, hid,   // Q27_TPSR: natural order
                                g_rn_dropy ? nullptr : d.norm, d.xq, d.xs,
                                Q27_HID, 1, L->f8_gate.in_scale, C.inv_d + g, s);
        else
        q27_rmsnorm_hostss_perm(g_coll_acc, L->post_norm, hid,
                                g_rn_dropy ? nullptr : d.norm, d.xq, d.xs,
                                Q27_HID, 1, L->gate.in_scale, C.inv_d + g, s);
        if (mlp_pre) {   // ---- MLP pre-enqueued behind the flag-gated tail: the host then has
            TPF(TPF_PROJ_NVFP4, s, {      // nothing post-b2 to submit but the reduce itself ----
                mlp_fused = q27_mlp_gu_swiglu_q(&L->gate, &L->up, d.xq, d.xs,
                                                d.xq, d.xs, L->down.in_scale, s) != 0;
                if (!mlp_fused) {
                    if (!q27_proj_nvfp4_gu(&L->gate, &L->up, d.xq, d.xs, d.pa, d.pb, s)) {
                        q27_proj_nvfp4(&L->gate, d.xq, d.xs, d.pa, s);
                        q27_proj_nvfp4(&L->up,   d.xq, d.xs, d.pb, s);
                    }
                } });
            TPF(TPF_QUANT, s, {
                if (!mlp_fused) {
                    if (!q27_swiglu_quant(d.pa, d.pb, d.xq, d.xs, IL, L->down.in_scale, s)) {
                        q27_swiglu(d.pa, d.pb, d.act, IL, s);
                        q27_quant_perm(d.act, d.xq, d.xs, IL, L->down.in_scale, s);
                    } else {
                        for (int rp = 0; rp < g_slow_quant; ++rp)          // idempotent
                            q27_swiglu_quant(d.pa, d.pb, d.xq, d.xs, IL, L->down.in_scale, s);
                    }
                } });
            TPF(TPF_DOWN, s, {
                if (rad && g_rn_epi) q27_nvfp4_set_res(rad, rshare);
                q27_proj_nvfp4(&L->down, d.xq, d.xs, part, s);              // PARTIAL, K=4352
                q27_nvfp4_set_res(nullptr, 0.f);
                if (rad && !g_rn_epi) q27_add_res_share(part, hid, rshare, Q27_HID, s);
            });
        }
        tp_stage_wait(C, g, cg1);
    } else {
        CK(hipStreamSynchronize(s));
    }
    C.t_comp[g] += tp_now() - *mark;                               // drain counts as compute
    { const double t0 = tp_now();
      if (g_ar1_skip) C.b2.wait();                            // ---- ALL-REDUCE #1 SKIPPED (ceiling probe) ----
      else if (g_coll_bf16) tp_allreduce_bf16(C, g, (unsigned short*)d.mixer, s);
      else                  tp_allreduce(C, g, d.mixer, s, fd1);
      C.t_coll[g] += tp_now() - t0; C.n_coll[g]++; *mark = tp_now(); }
    // RELEASE: the reduced vector and 1/rms are both written by now; publishing the flag after
    // them, with release ordering, is what makes the pre-enqueued kernel safe to run.
    if (pe1) __atomic_store_n(&C.flag_h[g], gn1, __ATOMIC_RELEASE);

    // ---- MLP ----
    // accin/invin2: pfmode 5 reads THIS position's slice of the chunk-level collective #1 result
    // (and its 1/rms) instead of the single-slot g_coll_acc / C.inv_d.
    tail_fn(pe1, g_coll_acc, C.inv_d + g);
    if (stop) return;               // the batched sweep takes over: gu/swiglu/down/coll#2 outside
    // ---- Q27_EXL3 (M=1): gate/up straight off the bf16 post-norm, natural-order
    // swiglu, down into the partial. No quantize step anywhere in the chain --
    // EXL3 rotates its own input at each projection, so the int8/fp8 activation
    // currency (and its in_scale) simply does not appear.
    if (L->ex_gate.live() && L->ex_up.live() && L->ex_down.live()) {
        // DIAGNOSTIC 2026-09-19. The post-attention norm that fills d.norm -- the ONLY
        // input this chain has -- is written by a kernel that refuses to launch unless
        // in_scale > 0, and both call sites DISCARD its return value. With the NVFP4
        // MLP dropped, L->gate.in_scale is 0, so d.norm would keep the PRE-attention
        // norm the prologue left there. Print what the scale actually is.
        { static std::atomic<int> once{0};
          if (once.fetch_add(1) == 0)
            std::fprintf(stderr, "Q27_EXL3_MLPDIAG L=%d gate.in_scale=%.6g f8_gate.in_scale=%.6g "
                                 "pe1=%d rn_hostss=%d rn_dropy=%d f8mlp=%d\n",
                         L->layer, (double)L->gate.in_scale, (double)L->f8_gate.in_scale,
                         (int)pe1, (int)g_rn_hostss, (int)g_rn_dropy, (int)f8mlp), std::fflush(stderr); }
        if (!exl3_f32(d, L->ex_gate, d.norm, d.pa, 1, s) ||
            !exl3_f32(d, L->ex_up,   d.norm, d.pb, 1, s)) {
            std::fprintf(stderr, "Q27_EXL3: gate/up declined (M=1, L=%d in=%d out=%d K=%d)\n",
                         L->layer, L->ex_gate.in, L->ex_gate.out, L->ex_gate.K); std::abort(); }
        q27_swiglu(d.pa, d.pb, d.act, IL, s);      // natural order, bf16: EXL3 down rotates it itself
        if (!exl3_f32(d, L->ex_down, d.act, part, 1, s)) {
            std::fprintf(stderr, "Q27_EXL3: down declined (M=1, L=%d)\n", L->layer); std::abort(); }
        if (rad) q27_add_res_share(part, hid, rshare, Q27_HID, s);
    } else
    if (f8mlp) {   // ---- Q27_TPSR (M=1): FP8 gate/up row shard, natural swiglu, FP8 down K-slice with the residual share ----
        if (!q27_proj_fp8b_nr(&L->f8_gate, d.xq, d.xs, d.pa, IL, nullptr, 0, 0.f, 1, s) ||
            !q27_proj_fp8b_nr(&L->f8_up,   d.xq, d.xs, d.pb, IL, nullptr, 0, 0.f, 1, s)) {
            std::fprintf(stderr, "Q27_TPSR: FP8 gate/up declined (M=1, L=%d rows=%d K=%d)\n", L->layer, L->f8_gate.rows, L->f8_gate.K); std::abort(); }
        if (!q27_sr_swiglu_quant_nat_b(d.pa, d.pb, d.xq, d.xs, IL, L->f8_down.in_scale, 1, s)) {
            std::fprintf(stderr, "Q27_TPSR: natural swiglu declined (M=1, L=%d)\n", L->layer); std::abort(); }
        if (!q27_proj_fp8b_nrs(&L->f8_down, d.xq, IL, d.xs, part, Q27_HID,
                               (rad && g_rn_epi) ? rad : nullptr, Q27_HID, rshare, 1, s)) {
            std::fprintf(stderr, "Q27_TPSR: FP8 down slice declined (M=1, L=%d rows=%d K=%d)\n", L->layer, L->f8_down.rows, L->f8_down.K); std::abort(); }
        if (rad && !g_rn_epi) q27_add_res_share(part, hid, rshare, Q27_HID, s);
    } else
    if (!mlp_pre) {
    TPF(TPF_PROJ_NVFP4, s, {
        mlp_fused = q27_mlp_gu_swiglu_q(&L->gate, &L->up, d.xq, d.xs,
                                        d.xq, d.xs, L->down.in_scale, s) != 0;
        if (!mlp_fused) {
            if (!q27_proj_nvfp4_gu(&L->gate, &L->up, d.xq, d.xs, d.pa, d.pb, s)) {
                q27_proj_nvfp4(&L->gate, d.xq, d.xs, d.pa, s);
                q27_proj_nvfp4(&L->up,   d.xq, d.xs, d.pb, s);
            }
        } });
    TPF(TPF_QUANT, s, {
        if (!mlp_fused) {
            if (!q27_swiglu_quant(d.pa, d.pb, d.xq, d.xs, IL, L->down.in_scale, s)) {
                q27_swiglu(d.pa, d.pb, d.act, IL, s);
                q27_quant_perm(d.act, d.xq, d.xs, IL, L->down.in_scale, s);
            } else {
                // Probe the kernel that ACTUALLY runs. The first placement sat on the
                // q27_quant_perm fallback beneath the fused path and never fired, which reads
                // exactly like "this stage is free" -- the firing check caught it.
                for (int rp = 0; rp < g_slow_quant; ++rp)          // idempotent
                    q27_swiglu_quant(d.pa, d.pb, d.xq, d.xs, IL, L->down.in_scale, s);
            }
        } });
    TPF(TPF_DOWN, s, {
        if (rad && g_rn_epi) q27_nvfp4_set_res(rad, rshare);
        q27_proj_nvfp4(&L->down, d.xq, d.xs, part, s);                          // PARTIAL, K=4352
        q27_nvfp4_set_res(nullptr, 0.f);
        if (rad && !g_rn_epi) q27_add_res_share(part, hid, rshare, Q27_HID, s);
    });
    }   // !mlp_pre
    // Same transformation across the layer boundary: the consumer of ALL-REDUCE #2 is the NEXT
    // layer's input norm, so pre-enqueueing it needs that layer's weights, which is why Lnext is
    // threaded in. The next layer skips its own launch via C.pre_in[g].
    const bool pe2 = g_tail_preenq && g_preenq_ok && g_rn_hostss && !g_coll_bf16 && !g_coll_zcout
                     && Lnext != nullptr && !g_pf_sweep && g_plain_warmed && !(g_host_tail_mask & (1u << g));
    const bool fd2 = (g_coll_fusedrain || pe2) && !g_coll_bf16 && !g_coll_zcout;
    unsigned cg2 = 0;
    if (fd2 && !g_fd_neg) cg2 = tp_stage_out(C, g, d.mixer, s);
    unsigned gn2 = 0;
    if (pe2) {
        if (!(g_coll_dflag && cg2)) CK(hipEventRecord(C.ev_copy[g], s));
        gn2 = ++C.gen[g];
        CK(hipStreamWaitValue32(s, (void*)(C.flag_d + g), gn2, hipStreamWaitValueEq, 0xffffffffu));
        const float in_s2 = Lnext->is_full ? Lnext->q_proj.in_scale : Lnext->in_qkv.in_scale;
        if (g_fp8x4)
            q27_rmsnorm_hostss_perm(g_coll_acc, Lnext->input_norm, hid,
                               (g_rn_dropy && Lnext->is_full) ? nullptr : d.norm, d.xq, d.xs,
                               Q27_HID, 1, in_s2, C.inv_d + g, s);
        else
            q27_rmsnorm_hostss_fp8(g_coll_acc, Lnext->input_norm, hid,
                               (g_rn_dropy && Lnext->is_full) ? nullptr : d.norm, d.xq, d.xs,
                               Q27_HID, 1, in_s2, C.inv_d + g, s);
        // Q27_PREENQ_PROLOGUE: the next layer's whole prologue (q/k/v/attn/quant/o_proj or the
        // GDN equivalent) plus its collective-1 D2H copy ride pre-barrier behind the flag-gated
        // input norm, so the host's post-b2 submissions shrink to the pe1 tail+MLP enqueues and
        // the reduce. Stream order is the only dependency: prologue after the tail, copy after
        // o_proj. The next layer's own iteration starts at the event sync.
        if (g_preenq_prologue && g_preenq_mlp) {
            const int fsx = Lnext->is_full ? (fsraw + (L->is_full ? 1 : 0)) : 0;
            const int gsx = Lnext->is_full ? 0 : (gsraw + (L->is_full ? 0 : 1));
            enq_pro(Lnext, fsx, gsx);
            if (!g_coll_bf16 && !g_coll_zcout && !g_fd_neg)
                C.pre_gen[g] = tp_stage_out(C, g, d.mixer, s);
        }
        C.pre_in[g] = 1;
        tp_stage_wait(C, g, cg2);
    } else {
        C.pre_in[g] = 0; C.pre_gen[g] = 0;
        CK(hipStreamSynchronize(s));
    }
    C.t_comp[g] += tp_now() - *mark;
    { const double t0 = tp_now(); if (g_coll_bf16) tp_allreduce_bf16(C, g, (unsigned short*)d.mixer, s);
      else             tp_allreduce(C, g, d.mixer, s, fd2);   // ---- ALL-REDUCE #2 ----
      C.t_coll[g] += tp_now() - t0; C.n_coll[g]++; *mark = tp_now(); }
    if (pe2) __atomic_store_n(&C.flag_h[g], gn2, __ATOMIC_RELEASE);
    // residual 2 is NOT applied here: it becomes the next layer's `pending`, folded into that
    // layer's input norm. The caller applies the last one explicitly before final_norm.
}

// =============================================================================================
// Q27_SPEC: ONE LAYER FOR NR CONSECUTIVE POSITIONS OF ONE SEQUENCE (the speculative verify).
// This is run_layer_tp on its shipped defaults (host-sumsq norms, flag-gated pre-enqueue of the
// tail / MLP / next prologue, in-stream staging copy + done flag, zero-copy consumer reads) with
// every weight-streaming projection replaced by its row-batched form, so the NR rows share ONE
// pass over the weights. Attention, the GDN recurrence and the small kernels run per row: row r
// attends to positions [0, pos0+r] and the conv / recurrent state advance in row order. After
// each non-final row's GDN step this card's state shard and conv block are snapshotted to bank r,
// so a rejected draft is undone by restoring that bank (the KV cache is position-indexed and the
// next round's row 0 simply overwrites the rejected position: no rollback there).
// Per-row arithmetic is that of the single-row kernels, so a verified token equals the token the
// plain step would have produced: the spec run's token stream must be IDENTICAL to plain greedy.
// Q27_SPEC K=7: the register-resident NR kernels cap at 4 rows (NR > 4 would spill); rows 5..8
// run as a second in-stream chunk of <= 4. The split is semantics-preserving: chunk c's attention
// decode only reads K/V up to its own last position (row r attends pos0..pos0+r), and the GDN
// conv/scan recurrence continues from the state chunk c-1 left in place; the snapshot bank
// pointers just offset by c0 rows. NR <= 4 runs exactly one chunk (bit-identical to the old path).
// ---- Q27_NR_P2P helpers: the other three card indices in ascending order, and one collective site's transport ----
static int g_nrp_ndev = 4;   // set at the NR_P2P init; 3 cards use a two-peer table with -1 in the third slot
static inline const int* nrp_peers(int g) {
    static const int T[4][3] = { {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2} };
    static const int T3[3][3] = { {1, 2, -1}, {0, 2, -1}, {0, 1, -1} };
    return (g_nrp_ndev == 3) ? T3[g % 3] : T[g & 3];
}
// the receive-slot pointer for peer index j (null past the last peer), so the 4-card call sites work unchanged at 3 cards
#define NRP_RECV(slot, owner, pk, j) (((pk)[j] >= 0) ? C.nrp_recv[slot][owner][(pk)[j]] : nullptr)
// v2 (g_nr_p2p == 1): wait until every peer has finished reading what this card pushed into their slot last time,
// push this card's fp32 partial rows as bf16 into the three peers' receive slots (posted stores), record "done",
// order the enqueue with a host barrier (so every card's record precedes any card's wait), then wait on the
// peers' done events. v1 (g_nr_p2p == 2): the peers read this card's partial in place; only done/wait here.
static inline void nrp_site(TpColl& C, int g, int ndev, int slot, hipStream_t s, const float* part, int NR, int mode) {
    // runtime receipt: this site took the GPU-RESIDENT path, and which of the three transports
    ++g_coll_p2p_sites[g][(mode >= 1 && mode <= 4) ? mode : 0]; g_coll_nr_eff[g] = NR;
    if (mode == 4) {   // LL: one push kernel into the peers' slot for this sequence number; nothing else crosses the cards
        (void)slot;
        const unsigned seq = ++C.nrp_seq[g];
        const int sl = (int)(seq & (Q27_LL_SLOTS - 1));
        int pk[3] = {-1, -1, -1}, np = 0; for (int k = 0; k < ndev; ++k) if (k != g) pk[np++] = k;
        int okp = 0;
        if (g_ll_sdma) {
            TPF(TPF_LL_PUSH, s, {
                if (C.nrp_prepacked[g]) { okp = 1; C.nrp_prepacked[g] = 0; }   // Q27_LL_EPI: the producer's epilogue packed this site
                else okp = q27_pack_ll(part, C.nrp_llstage[sl][g], Q27_HID, NR, seq, s);
                for (int j = 0; j < np; ++j)
                    CK(hipMemcpyPeerAsync(C.nrp_ll[sl][pk[j]][g], g_devid_tab[pk[j]], C.nrp_llstage[sl][g], g_devid_tab[g], (size_t)NR * Q27_HID * 4, s));
            });
        } else {
            TPF(TPF_LL_PUSH, s, okp = q27_push_ll(part, C.nrp_ll[sl][pk[0]][g], C.nrp_ll[sl][pk[1]][g], (np > 2) ? C.nrp_ll[sl][pk[2]][g] : nullptr, Q27_HID, NR, seq, s));
        }
        if (!okp) { std::fprintf(stderr, "Q27_NR_P2P=4: push declined\n"); std::abort(); }
        return;
    }
    if (mode != 2) {
        for (int k = 0; k < ndev; ++k) if (k != g) CK(hipStreamWaitEvent(s, C.ev_nrp_rd[slot][k], 0));
        const int* pk = nrp_peers(g);
        if (mode == 3) {   // pack locally, then three SDMA copies into the peers' slots (stream-ordered on s)
            q27_f2bf_vec(part, C.nrp_stage[slot][g], (size_t)NR * Q27_HID, s);
            for (int j = 0; j < 3 && pk[j] >= 0; ++j)
                CK(hipMemcpyPeerAsync(C.nrp_recv[slot][pk[j]][g], g_devid_tab[pk[j]], C.nrp_stage[slot][g], g_devid_tab[g], (size_t)NR * Q27_HID * 2, s));
        } else if (!q27_push_bf16_3(part, C.nrp_recv[slot][pk[0]][g], C.nrp_recv[slot][pk[1]][g], (pk[2] >= 0) ? C.nrp_recv[slot][pk[2]][g] : nullptr, NR * Q27_HID, s)) {
            std::fprintf(stderr, "Q27_NR_P2P: push declined\n"); std::abort(); }
    }
    CK(hipEventRecord(C.ev_nrp_done[slot][g], s));
    if (slot == 0) C.b1.wait(); else C.b2.wait();          // host order only: no GPU idle behind it
    for (int k = 0; k < ndev; ++k) if (k != g) CK(hipStreamWaitEvent(s, C.ev_nrp_done[slot][k], 0));
}

// Q27_NR_P2P=4: the consumer half of an LL site -- spin on this card's receive slots for the current sequence number, then
// reduce own + received in card order (+ the norm/quant tail per mode). Runs on the SAME stream right behind the push.
static inline int nrp_ll_reduce(TpColl& C, int g, int ndev, int mode, const float* own, const unsigned short* wgt, unsigned short* hid,
                                unsigned short* y, signed char* xq, float* xs, int plus_one, float in_scale, int NR, hipStream_t s, bool trunk = false) {
    const unsigned seq = C.nrp_seq[g];
    const int sl = (int)(seq & (Q27_LL_SLOTS - 1));
    int pk[3] = {-1, -1, -1}, np = 0; for (int k = 0; k < ndev; ++k) if (k != g) pk[np++] = k;
    int ok = 0;
    const auto reduce = trunk ? q27_red_rn_nr_ll_trunk : q27_red_rn_nr_ll;
    TPF(TPF_LL_RED, s, ok = reduce(mode, own, g, ndev, C.nrp_ll[sl][g][pk[0]], C.nrp_ll[sl][g][pk[1]], (np > 2) ? C.nrp_ll[sl][g][pk[2]] : nullptr, seq,
                                             C.nrp_red[g], C.nrp_ssp[g], wgt, hid, y, xq, xs, Q27_HID, plus_one, in_scale, NR, C.nrp_err[g], C.nrp_cnt[g], s));
    return ok;
}

// ---- DFlash2 conditioning diagnostic state ----
static q27_df2_ctx_t g_df2c[Q27_MAX_DEVICES];
static int g_df2_cond = 0;
static int g_df2_fwd = 0;
static int g_df2_sel = 0;
static int g_df2_d7_ready[Q27_MAX_DEVICES] = {0,0,0,0,0,0,0,0};
static unsigned g_df2_d7[Q27_MAX_DEVICES][7] = {{0}};
static int g_df2_nprompt = 0;
static int g_df2_prompt_done[Q27_MAX_DEVICES] = {0,0,0,0,0,0,0,0};
static unsigned g_df2_cands[7][16] = {{0}};   // card 0: the merged top-16 per position of the last block
static long g_df2_cand_hits = 0, g_df2_cand_rounds = 0;
static int df2_reduce_cb(void* u, int g, const float* own, float* red, float* ssp, int NR, hipStream_t s) {
    TpColl* C = (TpColl*)u;
    nrp_site(*C, g, C->ndev, 0, s, own, NR, 1);   // bf16 push + host barrier + wait peers' done
    const int* pk = nrp_peers(g);
    if (!q27_red_rn_nr_local(2, own, g, C->nrp_recv[0][g][pk[0]], C->nrp_recv[0][g][pk[1]], (pk[2] >= 0) ? C->nrp_recv[0][g][pk[2]] : nullptr,
                             red, ssp, nullptr, nullptr, nullptr, nullptr, nullptr, Q27_HID, 0, 0.f, NR, s)) return 0;
    CK(hipEventRecord(C->ev_nrp_rd[0][g], s));   // peers' next push must wait for this read
    return 1;
}
static void df2_exch_cb(void* u, int g, const unsigned* ids, const float* vals, int NR) {
    (void)NR;
    TpColl* C = (TpColl*)u;
    static unsigned ex_ids[4][7 * 16]; static float ex_vals[4][7 * 16];
    if (ids && vals) {   // phase 1: stage each card's local top-16, card 0 merges 4x16 -> top-16
        std::memcpy(ex_ids[g], ids, (size_t)7 * 16 * 4); std::memcpy(ex_vals[g], vals, (size_t)7 * 16 * 4);
        C->b1.wait(); C->b2.wait();
        if (g == 0) {
            for (int r = 0; r < 7; ++r) {
                unsigned best[16]; float bv[16];
                for (int k = 0; k < 16; ++k) { bv[k] = -FLT_MAX; best[k] = 0xffffffffu; }
                for (int c = 0; c < 4; ++c) for (int k = 0; k < 16; ++k) {
                    const float v = ex_vals[c][r * 16 + k]; const unsigned i = ex_ids[c][r * 16 + k];
                    { int dup = 0; for (int s = 0; s < 16; ++s) if (best[s] == i) { dup = 1; break; } if (dup) continue; }   // lm_head is REPLICATED (rows=62080/card): all 4 cards return identical top-16; without this the 16 slots hold only 4 distinct ids
                    int ins = 16;
                    for (int s = 0; s < 16; ++s) if (v > bv[s] || (v == bv[s] && i < best[s])) { ins = s; break; }
                    if (ins == 16) continue;
                    for (int s = 15; s > ins; --s) { bv[s] = bv[s - 1]; best[s] = best[s - 1]; }
                    bv[ins] = v; best[ins] = i;
                }
                std::memcpy(ex_ids[0] + r * 16, best, 64); std::memcpy(ex_vals[0] + r * 16, bv, 64);
            }
        }
        C->b1.wait(); C->b2.wait();
        std::memcpy((void*)ids, ex_ids[0], (size_t)7 * 16 * 4);
        std::memcpy((void*)vals, ex_vals[0], (size_t)7 * 16 * 4);
        if (g == 0) std::memcpy(g_df2_cands, ex_ids[0], (size_t)7 * 16 * 4);
    } else if (ids) {   // phase 2: broadcast card 0's 7 selected ids
        if (g == 0) std::memcpy(ex_ids[0], ids, 28);
        C->b1.wait(); C->b2.wait();
        std::memcpy((void*)ids, ex_ids[0], 28);
    }
}
// chunk width = g_nr_ch (Q27_NR_CH, default 4)
template <class F>
static inline void nr_chunks(int NR, F&& f) {
    for (int c0 = 0; c0 < NR; c0 += g_nr_ch) f(c0, (NR - c0 < g_nr_ch) ? (NR - c0) : g_nr_ch);
}
// The pre-enqueued consumer's wait on the host-published flag: CP barrier-value packet (shipped)
// or, under Q27_WAIT_KERNEL=1, a spinning wave that only stalls its own stream.
static inline void tp_flag_wait(hipStream_t s, TpColl& C, int g, unsigned gn) {
    if (g_wait_kernel) q27_wait_flag_eq((const unsigned*)(C.flag_d + g), gn, s);
    else CK(hipStreamWaitValue32(s, (void*)(C.flag_d + g), gn, hipStreamWaitValueEq, 0xffffffffu));
}
static void run_layer_tp_nr(Dev& d, const q27_layer_t* L, int pos0, int NR, int gsraw, int fsraw,
                            int g, int ndev, TpColl& C, double* mark, bool first,
                            const q27_layer_t* Lnext, const q27_globals_t* G) {
    hipStream_t s = d.stream;
    const float rshare = 1.0f / (float)ndev;   // Q27_TP3: residual share per worker
    const bool prequeue = g_nr_warmed && !(g_host_tail_mask & (1u << g));
    const int HID = Q27_HID;
    const int gslot = L->is_full ? 0 : gsraw, fslot = L->is_full ? fsraw : 0;
    // Q27_TP3 (2026-09-28): this worker's shard geometry comes from the plan table (uniform at
    // ndev 1/2/4, ragged whole-KV-group / whole-key-head ranges at ndev 3).
    const q27_tp_shard_t* const T = q27_tp_shard(g);
    const int QL = T->QL, KVL = T->KVL, OL = T->OL;
    const int QKVL = T->QKVL, ZL = T->ZL, IL = mlp_len(d.id), VHL = T->VHL;
    const int QS = T->QS;                                                         // qkva row stride
    const int MIXL = T->MIXL;                                                     // mixer row stride
    const int n_gdn = Q27_LAYERS - Q27_LAYERS / 4;
    const size_t sshard = (size_t)VHL * Q27_GDN_D * Q27_GDN_D;
    (void)ndev;
    unsigned short* const hid = d.hidden_slots;                                   // rows 0..NR-1, stride HID
    const int p2p_ = (g_nr_p2p && NR >= g_nr_p2p_min) ? g_nr_p2p : 0;   // transport for this pass (row-count conditional)
    // Q27_LL_EPI: arm the producer-epilogue LL pack for the NEXT site (sequence C.nrp_seq[g]+1, its slot) for the chunk starting at row c0
    auto ll_arm = [&](int c0) {
        if (p2p_ != 4 || !g_ll_epi) return;
        const unsigned seq = C.nrp_seq[g] + 1u; const int sl = (int)(seq & (Q27_LL_SLOTS - 1));
        q27_ll_hint_arm(C.nrp_llstage[sl][g] + (size_t)c0 * Q27_HID, seq);
    };
    auto ll_settle = [&]() { if (p2p_ == 4 && g_ll_epi) { ++g_ll_epi_armed[g]; if (q27_ll_hint_done()) { C.nrp_prepacked[g] = 1; ++g_ll_epi_taken[g]; } else { C.nrp_prepacked[g] = 0; } } };
    float* const part  = p2p_ ? C.nrp_part[0][g] : d.nr_mixer;                  // [NR][HID] partial payload, site 1 (peer-visible slot 0 under Q27_NR_P2P)
    float* const part2 = p2p_ ? C.nrp_part[1][g] : d.nr_mixer;                  // site 2 (slot 1)
    const float in_s = L->is_full ? L->q_proj.in_scale : L->in_qkv.in_scale;
    const bool pro_in = (C.pre_in[g] != 0);
    if (!g_preenq_ok) { std::fprintf(stderr, "Q27_SPEC: hipStreamWaitValue32 unsupported on this device\n"); std::abort(); }
    // Q27_NR_LDS: LDS-staged forms (activations staged once per block) vs the register-resident forms
    auto P_FP8_BF16 = (g_nr_hg || (g_nr_fam & 1))  ? q27_proj_fp8_bf16_hg : (g_nr_lds ? q27_proj_fp8_bf16_lds : q27_proj_fp8_bf16_nr);
    auto P_FP8_RES  = (g_nr_hg || (g_nr_fam & 2))  ? q27_proj_fp8_res_hg  : (g_nr_lds ? q27_proj_fp8_res_lds  : q27_proj_fp8_res_nr);
    auto P_GU       = (g_nr_hg || (g_nr_fam & 4))  ? q27_proj_nvfp4_gu_hg : (g_nr_lds ? q27_proj_nvfp4_gu_lds : q27_proj_nvfp4_gu_nr);
    auto P_DOWN     = (g_nr_hg || (g_nr_fam & 8))  ? q27_proj_nvfp4_down_hg : (g_nr_lds ? q27_proj_nvfp4_down_lds : q27_proj_nvfp4_down_nr);
    auto P_LM       = (g_nr_hg || (g_nr_fam & 16)) ? q27_proj_nvfp4_hg    : (g_nr_lds ? q27_proj_nvfp4_lds    : q27_proj_nvfp4_nr);

    // ---- input norm (layer 0: from the embedding rows; else pre-enqueued by the previous layer) ----
    if (first) {
        for (int r = 0; r < NR; ++r)
            if (!q27_rmsnorm_quant_fp8(hid + (size_t)r * HID, L->input_norm, d.nr_norm + (size_t)r * HID,
                                       d.nr_xq + (size_t)r * HID, d.nr_xs + (size_t)r * (HID / 16),
                                       HID, 1, in_s, nullptr, 0, s)) {
                q27_rmsnorm(hid + (size_t)r * HID, L->input_norm, d.nr_norm + (size_t)r * HID, HID, 1, s);
                q27_quant_fp8(d.nr_norm + (size_t)r * HID, d.nr_xq + (size_t)r * HID, d.nr_xs + (size_t)r * (HID / 16), HID, in_s, s);
            }
    } else if (!pro_in) {
        nr_chunks(NR, [&](int c0, int cn) {
            q27_rmsnorm_hostss_fp8_b(g_coll_acc + (size_t)c0 * HID, L->input_norm, hid + (size_t)c0 * HID,
                                     d.nr_norm + (size_t)c0 * HID, d.nr_xq + (size_t)c0 * HID,
                                     d.nr_xs + (size_t)c0 * (HID / 16),
                                     HID, 1, in_s, C.invN_d + g * 8 + c0, cn, s);
        });
    }
    C.pre_in[g] = 0;

    auto enq_pro_impl = [&](const q27_layer_t* Lx, int fsx, int gsx) {
        if (Lx->is_full) {
            nr_chunks(NR, [&](int c0, int cn) {
                if (Lx->ex_q.live() && Lx->ex_k.live() && Lx->ex_v.live()) {
                    exl3_bf16(d, Lx->ex_q, d.nr_norm + (size_t)c0 * HID, d.nr_qkva + (size_t)c0 * QS,            cn, s, HID, QS);
                    exl3_bf16(d, Lx->ex_k, d.nr_norm + (size_t)c0 * HID, d.nr_qkva + (size_t)c0 * QS + QL,       cn, s, HID, QS);
                    exl3_bf16(d, Lx->ex_v, d.nr_norm + (size_t)c0 * HID, d.nr_qkva + (size_t)c0 * QS + QL + KVL, cn, s, HID, QS);
                } else {
                const q27_fp8_t* sw[3] = { &Lx->q_proj, &Lx->k_proj, &Lx->v_proj };
                unsigned short* sy[3] = { d.nr_qkva + (size_t)c0 * QS, d.nr_qkva + (size_t)c0 * QS + QL, d.nr_qkva + (size_t)c0 * QS + QL + KVL };
                const int sst[3] = { QS, QS, QS };
                if (!(g_proj_seg && q27_proj_i8g_bf16_seg(sw, sy, sst, 3, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), cn, s))) {   // Q27_PROJ_SEG: one launch
                P_FP8_BF16(&Lx->q_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS,            QS, cn, s);
                P_FP8_BF16(&Lx->k_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS + QL,       QS, cn, s);
                P_FP8_BF16(&Lx->v_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS + QL + KVL, QS, cn, s);
                }
                }
                if (g_nr_dup & 8) {   // duplicate: q/k/v projections, same stores
                    P_FP8_BF16(&Lx->q_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS,            QS, cn, s);
                    P_FP8_BF16(&Lx->k_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS + QL,       QS, cn, s);
                    P_FP8_BF16(&Lx->v_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS + QL + KVL, QS, cn, s);
                }
            });
            signed char* kc = d.kc + d.kv_off[fsx];   q27_kvs_t* ks = d.ks + d.kvs_off[fsx];
            signed char* vc = d.vc + Q27_VOFF(d.kv_off[fsx]);   q27_kvs_t* vs = d.vs + Q27_VSOFF(d.kvs_off[fsx]);
            const int kvstride = d.kv_rowstride[fsx];
            const int hoff0    = d.kv_hoff[fsx];
            static const bool attn_q = q27_env_flag("Q27_ATTN_Q", false);
            int attn_q_ok = attn_q ? 1 : 0;
            // all NR rows in one prep launch (every row.s K/V lands before any row attends) and one
            // decode launch (row r attends to pos0 + r + 1 positions)
            if (g_nr_att1) {   // bisect: per-row launches
                for (int r = 0; r < NR; ++r)
                    q27_attn_prep_tp(d.nr_qkva + (size_t)r * QS, Lx->q_norm, Lx->k_norm, kc, ks, vc, vs,
                                     d.nr_qh + (size_t)r * Q27_OROWS, pos0 + r, T->attn_g, kvstride, hoff0, s);
                for (int r = 0; r < NR; ++r) {
                    if (attn_q)
                        attn_q_ok &= q27_attn_decode_tp(d.nr_qh + (size_t)r * Q27_OROWS, kc, ks, vc, vs, d.nr_qkva + (size_t)r * QS,
                                           d.nr_mix6 + (size_t)r * MIXL, pos0 + r, T->attn_g, kvstride, hoff0,
                                           d.nr_xq + (size_t)r * OL, d.nr_xs + (size_t)r * (OL / 16), Lx->o_proj.in_scale, s, OL);
                    else
                        q27_attn_decode_tp(d.nr_qh + (size_t)r * Q27_OROWS, kc, ks, vc, vs, d.nr_qkva + (size_t)r * QS,
                                           d.nr_mix6 + (size_t)r * MIXL, pos0 + r, T->attn_g, kvstride, hoff0, nullptr, nullptr, 0.0f, s);
                }
            } else {
            TPF(TPF_TR_ATTN, s, nr_chunks(NR, [&](int c0, int cn) {
            q27_attn_prep_tp_nr(d.nr_qkva + (size_t)c0 * QS, QS, Lx->q_norm, Lx->k_norm, kc, ks, vc, vs,
                                d.nr_qh + (size_t)c0 * Q27_OROWS, Q27_OROWS, pos0 + c0, cn, T->attn_g, kvstride, hoff0, s);
            if (attn_q)
                attn_q_ok &= q27_attn_decode_tp_nr(d.nr_qh + (size_t)c0 * Q27_OROWS, Q27_OROWS, kc, ks, vc, vs, d.nr_qkva + (size_t)c0 * QS, QS,
                                  d.nr_mix6 + (size_t)c0 * MIXL, MIXL, pos0 + c0, cn, T->attn_g, kvstride, hoff0, s,
                                  d.nr_xq + (size_t)c0 * OL, d.nr_xs + (size_t)c0 * (OL / 16), Lx->o_proj.in_scale, OL);
            else
                q27_attn_decode_tp_nr(d.nr_qh + (size_t)c0 * Q27_OROWS, Q27_OROWS, kc, ks, vc, vs, d.nr_qkva + (size_t)c0 * QS, QS,
                                  d.nr_mix6 + (size_t)c0 * MIXL, MIXL, pos0 + c0, cn, T->attn_g, kvstride, hoff0, s);
            }));
            }
            if (g_nr_dup & 1) {   // duplicate: idempotent stores -> d(wall)/d(attention) on the SHIPPED path
                if (g_nr_att1) {
                    for (int r = 0; r < NR; ++r)
                        q27_attn_decode_tp(d.nr_qh + (size_t)r * Q27_OROWS, kc, ks, vc, vs, d.nr_qkva + (size_t)r * QS,
                                           d.nr_mix6 + (size_t)r * MIXL, pos0 + r, T->attn_g, kvstride, hoff0, nullptr, nullptr, 0.0f, s);
                } else {
                    nr_chunks(NR, [&](int c0, int cn) {
                        q27_attn_decode_tp_nr(d.nr_qh + (size_t)c0 * Q27_OROWS, Q27_OROWS, kc, ks, vc, vs,
                                              d.nr_qkva + (size_t)c0 * QS, QS, d.nr_mix6 + (size_t)c0 * MIXL, MIXL,
                                              pos0 + c0, cn, T->attn_g, kvstride, hoff0, s);
                    });
                }
            }
            if (!attn_q_ok) {
            if (g_nr_qb) nr_chunks(NR, [&](int c0, int cn) { q27_quant_fp8_b(d.nr_mix6 + (size_t)c0 * MIXL, (size_t)MIXL, d.nr_xq + (size_t)c0 * OL, (size_t)OL, d.nr_xs + (size_t)c0 * (OL / 16), (size_t)(OL / 16), OL, Lx->o_proj.in_scale, cn, s); });
            else for (int r = 0; r < NR; ++r) q27_quant_fp8(d.nr_mix6 + (size_t)r * MIXL, d.nr_xq + (size_t)r * OL, d.nr_xs + (size_t)r * (OL / 16), OL, Lx->o_proj.in_scale, s);
            }
            nr_chunks(NR, [&](int c0, int cn) {
                ll_arm(c0);   // Q27_LL_EPI: this producer feeds site 1 (the next site)
                if (exl3_f32(d, Lx->ex_o, d.nr_mix6 + (size_t)c0 * MIXL, part + (size_t)c0 * HID, cn, s, MIXL, HID))
                    q27_exl3_add_res(part + (size_t)c0 * HID, hid + (size_t)c0 * HID, rshare, HID, cn, HID, HID, s);
                else P_FP8_RES(&Lx->o_proj, d.nr_xq + (size_t)c0 * OL, d.nr_xs + (size_t)c0 * (OL / 16), part + (size_t)c0 * HID, HID, hid + (size_t)c0 * HID, HID, rshare, cn, s); });
            ll_settle();
            if (g_nr_dup & 16) nr_chunks(NR, [&](int c0, int cn) {   // duplicate: o_proj (validate by hash)
                P_FP8_RES(&Lx->o_proj, d.nr_xq + (size_t)c0 * OL, d.nr_xs + (size_t)c0 * (OL / 16), part + (size_t)c0 * HID, HID, hid + (size_t)c0 * HID, HID, rshare, cn, s);
            });
            if (g_nr_dup & 32) {   // duplicate: the attention-output quantize (store-only)
                if (g_nr_qb) nr_chunks(NR, [&](int c0, int cn) { q27_quant_fp8_b(d.nr_mix6 + (size_t)c0 * MIXL, (size_t)MIXL, d.nr_xq + (size_t)c0 * OL, (size_t)OL, d.nr_xs + (size_t)c0 * (OL / 16), (size_t)(OL / 16), OL, Lx->o_proj.in_scale, cn, s); });
                else for (int r = 0; r < NR; ++r) q27_quant_fp8(d.nr_mix6 + (size_t)r * MIXL, d.nr_xq + (size_t)r * OL, d.nr_xs + (size_t)r * (OL / 16), OL, Lx->o_proj.in_scale, s);
            }
        } else {
            nr_chunks(NR, [&](int c0, int cn) {
                if (Lx->ex_iqkv[0].live() && Lx->ex_iz.live()) {
                    { int off = 0;
                      for (int r = 0; r < 3; ++r) {
                          exl3_bf16(d, Lx->ex_iqkv[r], d.nr_norm + (size_t)c0 * HID,
                                    d.nr_qkv + (size_t)c0 * QKVL + off, cn, s, HID, QKVL);
                          off += Lx->ex_iqkv[r].out; } }
                    exl3_bf16(d, Lx->ex_iz,   d.nr_norm + (size_t)c0 * HID, d.nr_zbuf + (size_t)c0 * ZL,   cn, s, HID, ZL);
                } else {
                const q27_fp8_t* sw[2] = { &Lx->in_qkv, &Lx->in_z };
                unsigned short* sy[2] = { d.nr_qkv + (size_t)c0 * QKVL, d.nr_zbuf + (size_t)c0 * ZL };
                const int sst[2] = { QKVL, ZL };
                if (!(g_proj_seg && q27_proj_i8g_bf16_seg(sw, sy, sst, 2, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), cn, s))) {   // Q27_PROJ_SEG: one launch
                P_FP8_BF16(&Lx->in_qkv, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkv + (size_t)c0 * QKVL,  QKVL, cn, s);
                P_FP8_BF16(&Lx->in_z,   d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_zbuf + (size_t)c0 * ZL,   ZL,   cn, s);
                }
                }
                if (g_nr_dup & 8) {   // duplicate: in_qkv / in_z, same stores
                    P_FP8_BF16(&Lx->in_qkv, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkv + (size_t)c0 * QKVL,  QKVL, cn, s);
                    P_FP8_BF16(&Lx->in_z,   d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_zbuf + (size_t)c0 * ZL,   ZL,   cn, s);
                }
            });
            // a/b per chunk (tile form, identical arithmetic); rows at stride VHL
            const bool gdn_ab_conv = q27_env_flag("Q27_GDN_AB_CONV", false) && !g_nr_gdn1;
            const bool g_dn_scan_q = q27_env_flag("Q27_GDN_SCAN_Q", false) && !g_nr_gdn1;
            if (!gdn_ab_conv) nr_chunks(NR, [&](int c0, int cn) {
                q27_bf16_gemv2_tile(Lx->in_a, Lx->in_b, d.nr_norm + (size_t)c0 * HID, HID,
                                    d.nr_ab + (size_t)c0 * VHL, d.nr_bb + (size_t)c0 * VHL, VHL, HID, cn, s);
            });
            unsigned short* convp = d.conv + d.conv_off[gsx];
            float* Sp = d.S + d.s_off[gsx] + d.s_qoff[gsx];
            // conv and the recurrence over the NR rows in ONE launch each (the prefill scan/tile forms),
            // both writing bank t = the state after row t for t < NR-1 (no copies).
            if (g_nr_gdn1) {   // bisect: per-row conv + step with memcpy snapshots
                for (int r = 0; r < NR; ++r) {
                    q27_gdn_conv_tp_r(d.nr_qkv + (size_t)r * QKVL, convp, d.conv_wide[gsx], Lx->conv1d, T->gk_h0, T->gk_hn, s);
                    q27_gdn_step_tp_r(d.nr_qkv + (size_t)r * QKVL, d.nr_zbuf + (size_t)r * ZL, d.nr_ab + (size_t)r * VHL, d.nr_bb + (size_t)r * VHL,
                                    Lx->A_log, Lx->dt_bias, Lx->gdn_norm, Sp, d.nr_mix6 + (size_t)r * MIXL, T->gk_h0, T->gk_hn, nullptr, nullptr, 0.0f, s);
                    if (r < NR - 1) {
                        CK(hipMemcpyAsync(d.nr_Ssnap + ((size_t)r * n_gdn + gsx) * sshard, Sp, sshard * 4, hipMemcpyDeviceToDevice, s));
                        CK(hipMemcpyAsync(d.nr_csnap + ((size_t)r * n_gdn + gsx) * d.tp_conv, convp, d.conv_bytes[gsx], hipMemcpyDeviceToDevice, s));
                    }
                }
            } else {
            TPF(TPF_TR_GDN, s, nr_chunks(NR, [&](int c0, int cn) {
            if (gdn_ab_conv) {
                q27_gdn_ab_conv_tp_tile2_r(Lx->in_a, Lx->in_b, d.nr_norm + (size_t)c0 * HID, HID,
                    d.nr_ab + (size_t)c0 * VHL, d.nr_bb + (size_t)c0 * VHL, VHL, HID,
                    d.nr_qkv + (size_t)c0 * QKVL, QKVL, cn, convp, d.conv_wide[gsx], Lx->conv1d, T->gk_h0, T->gk_hn,
                    d.nr_csnap + (size_t)(gsx + c0 * n_gdn) * d.tp_conv, (size_t)n_gdn * d.tp_conv, (c0 + cn < NR) ? 1 : 0, s);
            } else {
            q27_gdn_conv_tp_tile2_r(d.nr_qkv + (size_t)c0 * QKVL, QKVL, cn, convp, d.conv_wide[gsx], Lx->conv1d, T->gk_h0, T->gk_hn,
                                  d.nr_csnap + (size_t)(gsx + c0 * n_gdn) * d.tp_conv, (size_t)n_gdn * d.tp_conv, (c0 + cn < NR) ? 1 : 0, s);
            }
            q27_gdn_scan_tp_snap_r(d.nr_qkv + (size_t)c0 * QKVL, QKVL, d.nr_zbuf + (size_t)c0 * ZL, ZL,
                                 d.nr_ab + (size_t)c0 * VHL, d.nr_bb + (size_t)c0 * VHL, VHL,
                                 Lx->A_log, Lx->dt_bias, Lx->gdn_norm, Sp, d.nr_mix6 + (size_t)c0 * MIXL, MIXL, T->gk_h0, T->gk_hn, cn,
                                 // Q27_GDN_SCAN_Q: scan's epilogue quantizes into nr_xq/nr_xs (same
                                 // arithmetic as q27_quant_fp8_b), so the separate quant launch is skipped.
                                 g_dn_scan_q ? d.nr_xq + (size_t)c0 * ZL : nullptr,
                                 g_dn_scan_q ? d.nr_xs + (size_t)c0 * (ZL / 16) : nullptr,
                                 g_dn_scan_q ? Lx->out_proj.in_scale : 0.0f,
                                 d.nr_Ssnap + (size_t)(gsx + c0 * n_gdn) * sshard, (size_t)n_gdn * sshard, (c0 + cn < NR) ? 1 : 0, s);
            }));
            }
            if (!g_dn_scan_q) {
            if (g_nr_qb) nr_chunks(NR, [&](int c0, int cn) { q27_quant_fp8_b(d.nr_mix6 + (size_t)c0 * MIXL, (size_t)MIXL, d.nr_xq + (size_t)c0 * ZL, (size_t)ZL, d.nr_xs + (size_t)c0 * (ZL / 16), (size_t)(ZL / 16), ZL, Lx->out_proj.in_scale, cn, s); });
            else for (int r = 0; r < NR; ++r) q27_quant_fp8(d.nr_mix6 + (size_t)r * MIXL, d.nr_xq + (size_t)r * ZL, d.nr_xs + (size_t)r * (ZL / 16), ZL, Lx->out_proj.in_scale, s);
            }
            nr_chunks(NR, [&](int c0, int cn) { ll_arm(c0); P_FP8_RES(&Lx->out_proj, d.nr_xq + (size_t)c0 * ZL, d.nr_xs + (size_t)c0 * (ZL / 16), part + (size_t)c0 * HID, HID, hid + (size_t)c0 * HID, HID, rshare, cn, s); });
            ll_settle();
            if (g_nr_dup & 16) nr_chunks(NR, [&](int c0, int cn) {   // duplicate: out_proj (validate by hash)
                P_FP8_RES(&Lx->out_proj, d.nr_xq + (size_t)c0 * ZL, d.nr_xs + (size_t)c0 * (ZL / 16), part + (size_t)c0 * HID, HID, hid + (size_t)c0 * HID, HID, rshare, cn, s);
            });
        }
    };
    auto enq_pro = [&](const q27_layer_t* Lx, int fsx, int gsx) {
        TPF(TPF_PROJ_FP8, s, enq_pro_impl(Lx, fsx, gsx));
    };
    if (!pro_in) enq_pro(L, fslot, gslot);

    // ---- collective #1: the tail and the whole MLP ride behind the flag (host path) or behind the peers' events (Q27_NR_P2P) ----
    unsigned cg1 = 0, gn1 = 0;
    if (p2p_) {
        nrp_site(C, g, ndev, 0, s, part, NR, p2p_);      // v2: wait peers rd, push bf16 rows to the peers, record done; barrier; wait peers done
        if (p2p_ == 4) {
            if (!nrp_ll_reduce(C, g, ndev, 1, part, L->post_norm, hid, nullptr, d.nr_xq, d.nr_xs, 1, L->gate.in_scale, NR, s, true)) { std::fprintf(stderr, "Q27_NR_P2P=4: site 1 declined\n"); std::abort(); }
        } else if (p2p_ == 2) {
            if (!q27_red_rn_nr_p2p(1, C.nrp_part[0][0], C.nrp_part[0][1], C.nrp_part[0][2], C.nrp_part[0][3], C.nrp_red[g], C.nrp_ssp[g],
                                   L->post_norm, hid, nullptr, d.nr_xq, d.nr_xs, HID, 1, L->gate.in_scale, NR, s)) { std::fprintf(stderr, "Q27_NR_P2P: site 1 declined\n"); std::abort(); }
        } else {
            const int* pk = nrp_peers(g);
            if (!q27_red_rn_nr_local(1, part, g, NRP_RECV(0, g, pk, 0), NRP_RECV(0, g, pk, 1), NRP_RECV(0, g, pk, 2), C.nrp_red[g], C.nrp_ssp[g],
                                     L->post_norm, hid, nullptr, d.nr_xq, d.nr_xs, HID, 1, L->gate.in_scale, NR, s)) { std::fprintf(stderr, "Q27_NR_P2P: site 1 declined\n"); std::abort(); }
        }
        if (p2p_ != 4) CK(hipEventRecord(C.ev_nrp_rd[0][g], s));
    } else {
        cg1 = pro_in ? C.pre_gen[g] : tp_stage_out_nr(C, g, part, NR, s);
        gn1 = ++C.gen[g];
        if (prequeue) tp_flag_wait(s, C, g, gn1);
        else { tp_stage_wait(C,g,cg1);tp_allreduce_nr(C,g,NR);C.n_coll[g]++; }
    }
    TPF(TPF_TR_MLP, s, nr_chunks(NR, [&](int c0, int cn) {
        // ---- Q27_DEC_MLP_I8: this layer's MLP is the persistent int8 mirror, so run the SAME
        // rocBLAS/Tensile path the prefill uses and let the NVFP4 MLP shard stay unallocated. The
        // decode's intermediate is per-card (IL = INTER/ndev) and contiguous, so gate/up are one
        // full-K GEMM each on a ROW view of the mirror and down is one GEMM on a COLUMN view of it
        // (ldw = the full INTER), with every per-row scale applied after the collective reduction --
        // a K-shard's int32 partial is in the same units, so the K-view is sound here exactly as it
        // is in the prefill. ----
        if (L->dc_gate8r.w && L->dc_up8r.w && L->dc_down8r.w) {
            const q27_i8r_t* G8 = &L->dc_gate8r; const q27_i8r_t* U8 = &L->dc_up8r;
            const q27_i8r_t* D8 = &L->dc_down8r;
            int* ag = (int*)(d.nr_pa + (size_t)c0 * IL);
            int* au = (int*)(d.nr_pb + (size_t)c0 * IL);
            int* ad = d.nr_acc + (size_t)c0 * HID;
            if (!p2p_) q27_rmsnorm_hostss_tok_b(g_coll_acc + (size_t)c0 * HID, L->post_norm,
                                                d.nr_xq + (size_t)c0 * HID, d.nr_xs + c0, HID, 1,
                                                L->gate.in_scale, C.invN_d + g * 8 + c0, cn, 0, s);
            if (!q27_rb_gemm_i8g(d.id, G8->w, G8->rows, G8->K, d.nr_xq + (size_t)c0 * HID, cn, ag, G8->gs, s) ||
                !q27_rb_gemm_i8g(d.id, U8->w, U8->rows, U8->K, d.nr_xq + (size_t)c0 * HID, cn, au, U8->gs, s)) {
                std::fprintf(stderr, "Q27_DEC_MLP_I8: gate/up GEMM declined (L=%d N=%d K=%d M=%d)\n",
                             L->layer, G8->rows, G8->K, cn); std::abort();
            }
            if (!q27_epi_swiglu_tok_small(ag, au, G8->rows, G8->s, U8->s, G8->ng, d.nr_xs + c0,
                                          G8->alpha, U8->alpha, 1.0f / L->down.in_scale,
                                          d.nr_xq + (size_t)c0 * IL, d.nr_xs + c0, cn, s)) {
                std::fprintf(stderr, "Q27_DEC_MLP_I8: swiglu epilogue declined\n"); std::abort();
            }
            if (!q27_rb_gemm_i8g_ld(d.id, D8->w, D8->rows, D8->K, d.nr_xq + (size_t)c0 * IL, cn, ad,
                                    D8->gs, Q27_INTER, s)) {
                std::fprintf(stderr, "Q27_DEC_MLP_I8: down GEMM declined (L=%d N=%d K=%d ldw=%d M=%d)\n",
                             L->layer, D8->rows, D8->K, Q27_INTER, cn); std::abort();
            }
            if (!q27_epi_f32r_bf(ad, D8->rows, D8->s, D8->ng, d.nr_xs + c0, D8->alpha,
                                 part2 + (size_t)c0 * HID, hid + (size_t)c0 * HID, rshare, cn, s)) {
                std::fprintf(stderr, "Q27_DEC_MLP_I8: down epilogue declined\n"); std::abort();
            }
            return;
        }
        // ---- Q27_FP8_MLP: block-scaled FP8 gate/up (Qwen3.8-27B-FP8). FIRST-BOOT SCOPE: gate/up
        // only. The FP8 consumers take NATURAL-order int8 activations (q27.h activation contract:
        // "FP8 projections consume the NATURAL order"), which is exactly what
        // q27_rmsnorm_hostss_fp8_b produces; q27_swiglu_quant_b then emits the EVEN-THEN-ODD
        // permuted order that the NVFP4 down consumer requires, so down stays on its existing path
        // rather than being fed an ordering it cannot read. Extending down needs a natural-order
        // swiglu quantiser, which is the next step, not a first-boot one. ----
        if (L->f8_gate.w && L->f8_up.w && L->f8_gate.bs && L->f8_up.bs) {
            if (!p2p_) q27_rmsnorm_hostss_fp8_b(g_coll_acc + (size_t)c0 * HID, L->post_norm, hid + (size_t)c0 * HID, nullptr,
                                      d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                                      HID, 1, L->f8_gate.in_scale, C.invN_d + g * 8 + c0, cn, s);
            int gu_ok = 0;
            TPF(TPF_PROJ_NVFP4, s, gu_ok = q27_proj_fp8b_gu_nr(&L->f8_gate, &L->f8_up,
                                  d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                                  d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, IL, cn, s));
            if (!gu_ok) {
                std::fprintf(stderr, "Q27_FP8_MLP: gate/up declined (L=%d rows=%d K=%d NR=%d)\n",
                             L->layer, L->f8_gate.rows, L->f8_gate.K, cn); std::abort();
            }
            if (L->f8_down.w && L->f8_down.bs) {
                // ---- Q27_TPSR: the down_proj is this card's block-scaled FP8 K-SLICE ([HID][INTER/ndev],
                // the geometry SR's f8_dn[c] consumer already runs), fed in NATURAL order by the SR swiglu
                // quantiser; the 0.25 residual share rides in the epilogue exactly as P_DOWN's did, so
                // collective #2 is unchanged. The NVFP4 down shard is not resident on this arm. ----
                if (!q27_sr_swiglu_quant_nat_b(d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, d.nr_xq + (size_t)c0 * IL,
                                               d.nr_xs + (size_t)c0 * (IL / 16), IL, L->f8_down.in_scale, cn, s)) {
                    std::fprintf(stderr, "Q27_TPSR: natural swiglu declined (L=%d NR=%d)\n", L->layer, cn); std::abort(); }
                if (p2p_ == 2 && c0 == 0) for (int k = 0; k < ndev; ++k) if (k != g) CK(hipStreamWaitEvent(s, C.ev_nrp_rd[1][k], 0));
                int dn_ok = 0;
                TPF(TPF_DOWN, s, dn_ok = q27_proj_fp8b_nrs(&L->f8_down, d.nr_xq + (size_t)c0 * IL, IL, d.nr_xs + (size_t)c0 * (IL / 16),
                                       part2 + (size_t)c0 * HID, HID, hid + (size_t)c0 * HID, HID, rshare, cn, s));
                if (!dn_ok) {
                    std::fprintf(stderr, "Q27_TPSR: FP8 down slice declined (L=%d rows=%d K=%d NR=%d)\n",
                                 L->layer, L->f8_down.rows, L->f8_down.K, cn); std::abort(); }
                return;
            }
            q27_swiglu_quant_b(d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, d.nr_xq + (size_t)c0 * IL,
                               d.nr_xs + (size_t)c0 * (IL / 16), IL, L->down.in_scale, cn, s);
            if (p2p_ == 2 && c0 == 0) for (int k = 0; k < ndev; ++k) if (k != g) CK(hipStreamWaitEvent(s, C.ev_nrp_rd[1][k], 0));
            P_DOWN(&L->down, d.nr_xq + (size_t)c0 * IL, d.nr_xs + (size_t)c0 * (IL / 16), part2 + (size_t)c0 * HID, HID,
                   hid + (size_t)c0 * HID, HID, rshare, cn, s);
            return;
        }
        if (!p2p_) q27_rmsnorm_hostss_perm_b(g_coll_acc + (size_t)c0 * HID, L->post_norm, hid + (size_t)c0 * HID, nullptr,
                                  d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                                  HID, 1, L->gate.in_scale, C.invN_d + g * 8 + c0, cn, s);
        const bool ex_mlp_done =
            exl3_mlp(d, L, d.nr_norm + (size_t)c0 * HID, HID,
                     d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, IL,
                     part2 + (size_t)c0 * HID, HID, IL, cn, s) &&
            q27_exl3_add_res(part2 + (size_t)c0 * HID, hid + (size_t)c0 * HID, rshare, HID, cn, HID, HID, s);
        if (!ex_mlp_done) {
        static const bool gu_swilu = q27_env_flag("Q27_GU_SWILU", false);
        static const bool gu_swilu_q = q27_env_flag("Q27_GU_SWILU_Q", false);
        int down_done = 0;
        if (gu_swilu_q && cn == 1 && IL == (int)L->down.K) {
            // one launch: GEMV + SwiGLU + group-4 quant. Writes nr_pb (int8s) and nr_pa (scales)
            // so input nr_xq (HID) / nr_xs (HID/16) are never overwritten mid-kernel.
            if (!q27_gu_swilu_q16(L->mx_gate.gs_mx ? &L->mx_gate : &L->gate, L->mx_up.gs_mx ? &L->mx_up : &L->up,
                    d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                    (signed char*)(d.nr_pb + (size_t)c0 * IL), d.nr_pa + (size_t)c0 * IL, IL, L->down.in_scale, s)) {
                std::fprintf(stderr, "FATAL Q27_TP3: gu_swilu_q16 declined (L=%d rows=%d)\n", L->layer, L->gate.rows); std::abort();
            }
        } else if (gu_swilu) {
            // silu(gate)*up lands in nr_pa; quant_group4 consumes it (same permutation/scale).
            if (!q27_proj_nvfp4_gu_swilu(L->mx_gate.gs_mx ? &L->mx_gate : &L->gate, L->mx_up.gs_mx ? &L->mx_up : &L->up,
                    d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                    d.nr_pa + (size_t)c0 * IL, IL, cn, s)) {
                std::fprintf(stderr, "FATAL Q27_TP3: gate/up swilu declined (L=%d rows=%d NR=%d)\n", L->layer, L->gate.rows, cn); std::abort();
            }
            static const bool quant_down = q27_env_flag("Q27_QUANT_DOWN", false);
            if (quant_down && cn == 1 && (L->down.K == 5760 || L->down.K == 5888)) {
                if (!q27_quant_down(d.nr_pa + (size_t)c0 * IL, d.nr_xq + (size_t)c0 * IL,
                        d.nr_xs + (size_t)c0 * (IL / 16), IL, L->down.in_scale,
                        L->mx_down.gs_mx ? &L->mx_down : &L->down,
                        (const unsigned*)(d.nr_xq + (size_t)c0 * IL), d.nr_xs + (size_t)c0 * (IL / 16),
                        part2 + (size_t)c0 * HID, HID, hid + (size_t)c0 * HID, HID, rshare, IL, s)) {
                    std::fprintf(stderr, "FATAL Q27_TP3: quant_down declined (L=%d)\n", L->layer); std::abort();
                }
                down_done = 1;
            } else {
                q27_quant_group4(d.nr_pa + (size_t)c0 * IL, d.nr_xq + (size_t)c0 * IL,
                                 d.nr_xs + (size_t)c0 * (IL / 16), IL, L->down.in_scale, cn, s);
            }
        } else {
        if (!P_GU(L->mx_gate.gs_mx ? &L->mx_gate : &L->gate, L->mx_up.gs_mx ? &L->mx_up : &L->up, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
             d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, IL, cn, s)) {
            std::fprintf(stderr, "FATAL Q27_TP3: gate/up declined (L=%d rows=%d NR=%d)\n", L->layer, L->gate.rows, cn); std::abort();
        }
        q27_swiglu_quant_trunk(d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, d.nr_xq + (size_t)c0 * IL,
                           d.nr_xs + (size_t)c0 * (IL / 16), IL, L->down.in_scale, cn, s);
        }
        if (p2p_ == 2 && c0 == 0) for (int k = 0; k < ndev; ++k) if (k != g) CK(hipStreamWaitEvent(s, C.ev_nrp_rd[1][k], 0));   // pull: slot 1 free on every peer
        ll_arm(c0);   // Q27_LL_EPI: this producer feeds site 2
        if (!down_done) {
        static const bool gu_swilu_q = q27_env_flag("Q27_GU_SWILU_Q", false);
        const bool fused_q = gu_swilu_q && cn == 1 && IL == (int)L->down.K;
        const signed char* down_xq = fused_q
            ? (const signed char*)(d.nr_pb + (size_t)c0 * IL)
            : d.nr_xq + (size_t)c0 * IL;
        const float* down_xs = fused_q
            ? (d.nr_pa + (size_t)c0 * IL)
            : (d.nr_xs + (size_t)c0 * (IL / 16));
        if (!P_DOWN(L->mx_down.gs_mx ? &L->mx_down : &L->down, down_xq, down_xs, part2 + (size_t)c0 * HID, HID,
               hid + (size_t)c0 * HID, HID, rshare, cn, s)) {
            std::fprintf(stderr, "FATAL Q27_TP3: down_proj declined (L=%d K=%d NR=%d) -- no kernel instantiation for this shard width\n", L->layer, L->down.K, cn); std::abort();
        }
        }
        }   // !ex_mlp_done
        // BIT 64 (2026-09-17): DOWN_PROJ ALONE. The 08:21 note below blamed P_DOWN for the invalid
        // whole-trio arm. That diagnosis is WRONG and it cost the estate the largest unpriced GPU
        // stage. q27_proj_nvfp4_down_nr epilogue is v = acc*alpha; if (radd) v += rs*bf2f(radd[..]);
        // Y[..] = v -- ONE output pointer (part2), one READ-ONLY bf16 residual (radd = hid). A pure
        // store, the same shape bit 16 already duplicates. What actually broke the trio is ALIASING:
        // d.nr_xq is a single activation scratch viewed at stride HID=5120 by gate/up and at stride
        // IL=4352 by swiglu, and at c0=0 the two views overlap byte-for-byte -- so a duplicated
        // swiglu overwrites the gate/up input it is about to be re-fed. Bit 2 survives only because
        // its duplicate sits AFTER this call. Bit 64 therefore goes HERE, before bit 2 clobbers
        // nr_xq, where the inputs are still the ones P_DOWN just consumed.
        if (g_nr_dup & 64) P_DOWN(L->mx_down.gs_mx ? &L->mx_down : &L->down, d.nr_xq + (size_t)c0 * IL, d.nr_xs + (size_t)c0 * (IL / 16),
                                  part2 + (size_t)c0 * HID, HID, hid + (size_t)c0 * HID, HID, rshare, cn, s);
        // BIT 2 DUPLICATES ONLY THE STORE-ONLY PART OF THE MLP. Receipt: duplicating the whole trio
        // (2026-09-17 08:21, Q27_NR_DUP=2 with down included) CHANGED THE TOKEN STREAM -- 8,192-token
        // repetition cap, tokhash b65f6e92e04f54c8 vs the stashed 8b04cd30fa708350 -- so P_DOWN is
        // not a pure store: it fuses something additive (its two output pointers plus alpha=1/ndev
        // are the tell). An idempotent-stage discriminator is only valid where the stage stores.
        if (g_nr_dup & 2) {   // duplicate: gate/up + swiglu only (both store their own outputs)
            P_GU(L->mx_gate.gs_mx ? &L->mx_gate : &L->gate, L->mx_up.gs_mx ? &L->mx_up : &L->up, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                 d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, IL, cn, s);
            q27_swiglu_quant_b(d.nr_pa + (size_t)c0 * IL, d.nr_pb + (size_t)c0 * IL, d.nr_xq + (size_t)c0 * IL,
                               d.nr_xs + (size_t)c0 * (IL / 16), IL, L->down.in_scale, cn, s);
        }
    }));
    ll_settle();
    if (!p2p_ && prequeue) {
        { const double tw = tp_now(); tp_stage_wait(C, g, cg1); g_collt[g][0] += tp_now() - tw; }
        C.t_comp[g] += tp_now() - *mark;
        { const double t0 = tp_now(); tp_allreduce_nr(C, g, NR);
          C.t_coll[g] += tp_now() - t0; C.n_coll[g]++; *mark = tp_now(); }
        __atomic_store_n(&C.flag_h[g], gn1, __ATOMIC_RELEASE);
    } else { C.t_comp[g] += tp_now() - *mark; *mark = tp_now(); }

    // ---- collective #2: the next layer's input norm + prologue (or the head) ride behind the flag (host path) or the peers' events ----
    unsigned cg2 = 0, gn2 = 0;
    if (p2p_) {
        nrp_site(C, g, ndev, 1, s, part2, NR, p2p_);
    } else {
        cg2 = tp_stage_out_nr(C, g, part2, NR, s);
        gn2 = ++C.gen[g];
        if (prequeue) tp_flag_wait(s, C, g, gn2);
        else { tp_stage_wait(C,g,cg2);tp_allreduce_nr(C,g,NR);C.n_coll[g]++; }
    }
    if (Lnext) {
        const float in_s2 = Lnext->is_full ? Lnext->q_proj.in_scale : Lnext->in_qkv.in_scale;
        if (p2p_) {
            int ok2 = 0;
            if (p2p_ == 4) ok2 = nrp_ll_reduce(C, g, ndev, 0, part2, Lnext->input_norm, hid, d.nr_norm, d.nr_xq, d.nr_xs, 1, in_s2, NR, s, true);
            else if (p2p_ == 2) ok2 = q27_red_rn_nr_p2p(0, C.nrp_part[1][0], C.nrp_part[1][1], C.nrp_part[1][2], C.nrp_part[1][3], C.nrp_red[g], C.nrp_ssp[g],
                                                     Lnext->input_norm, hid, d.nr_norm, d.nr_xq, d.nr_xs, HID, 1, in_s2, NR, s);
            else { const int* pk = nrp_peers(g);
                   ok2 = q27_red_rn_nr_local(0, part2, g, NRP_RECV(1, g, pk, 0), NRP_RECV(1, g, pk, 1), NRP_RECV(1, g, pk, 2), C.nrp_red[g], C.nrp_ssp[g],
                                             Lnext->input_norm, hid, d.nr_norm, d.nr_xq, d.nr_xs, HID, 1, in_s2, NR, s); }
            if (!ok2) { std::fprintf(stderr, "Q27_NR_P2P: site 2 declined\n"); std::abort(); }
            if (p2p_ != 4) CK(hipEventRecord(C.ev_nrp_rd[1][g], s));
        } else TPF(TPF_TR_CONS, s, nr_chunks(NR, [&](int c0, int cn) {
            q27_rmsnorm_hostss_fp8_b(g_coll_acc + (size_t)c0 * HID, Lnext->input_norm, hid + (size_t)c0 * HID,
                                     d.nr_norm + (size_t)c0 * HID, d.nr_xq + (size_t)c0 * HID,
                                     d.nr_xs + (size_t)c0 * (HID / 16),
                                     HID, 1, in_s2, C.invN_d + g * 8 + c0, cn, s);
        }));
        const int fsx = Lnext->is_full ? (fsraw + (L->is_full ? 1 : 0)) : 0;
        const int gsx = Lnext->is_full ? 0 : (gsraw + (L->is_full ? 0 : 1));
        enq_pro(Lnext, fsx, gsx);
        C.pre_gen[g] = p2p_ ? 0u : tp_stage_out_nr(C, g, part, NR, s);
        C.pre_in[g] = 1;
    } else {
        // head, behind the same flag: last residual add, final norm, lm_head shard, per-row argmax
        const float* accp = g_coll_acc;
        if (p2p_) {   // reduce only: the same fp32 rows the host path leaves in g_coll_acc, local to this card
            int okh = 0;
            if (p2p_ == 4) okh = nrp_ll_reduce(C, g, ndev, 2, part2, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1.0f, NR, s, true);
            else if (p2p_ == 2) okh = q27_red_rn_nr_p2p(2, C.nrp_part[1][0], C.nrp_part[1][1], C.nrp_part[1][2], C.nrp_part[1][3], C.nrp_red[g], C.nrp_ssp[g],
                                                     nullptr, nullptr, nullptr, nullptr, nullptr, HID, 0, 1.0f, NR, s);
            else { const int* pk = nrp_peers(g);
                   okh = q27_red_rn_nr_local(2, part2, g, NRP_RECV(1, g, pk, 0), NRP_RECV(1, g, pk, 1), NRP_RECV(1, g, pk, 2), C.nrp_red[g], C.nrp_ssp[g],
                                             nullptr, nullptr, nullptr, nullptr, nullptr, HID, 0, 1.0f, NR, s); }
            if (!okh) { std::fprintf(stderr, "Q27_NR_P2P: head reduce declined\n"); std::abort(); }
            if (p2p_ != 4) CK(hipEventRecord(C.ev_nrp_rd[1][g], s));
            accp = C.nrp_red[g];
        }
        for (int r = 0; r < NR; ++r) q27_add_inplace(hid + (size_t)r * HID, accp + (size_t)r * HID, HID, s);
        for (int r = 0; r < NR; ++r) q27_rmsnorm(hid + (size_t)r * HID, G->final_norm, d.nr_norm + (size_t)r * HID, HID, 1, s);
        for (int r = 0; r < NR; ++r) q27_quant_perm(d.nr_norm + (size_t)r * HID, d.nr_xq + (size_t)r * HID,
                                                    d.nr_xs + (size_t)r * (HID / 16), HID, G->lm_head.in_scale, s);
        nr_chunks(NR, [&](int c0, int cn) {
            if (exl3_head(d, G, d.nr_norm + (size_t)c0 * HID,
                          d.nr_pa + (size_t)c0 * G->ex_head.out, cn, s)) return;
            P_LM(&G->lm_head, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                 d.nr_pa + (size_t)c0 * G->lm_head.rows, G->lm_head.rows, cn, s);
        });
        if (g_nr_dup & 4) nr_chunks(NR, [&](int c0, int cn) {   // duplicate: lm_head, same stores
            P_LM(&G->lm_head, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                 d.nr_pa + (size_t)c0 * G->lm_head.rows, G->lm_head.rows, cn, s);
        });
        for (int r = 0; r < NR; ++r) q27_argmax_val(d.nr_pa + (size_t)r * G->lm_head.rows, d.nr_tok + r, d.nr_val + r, G->lm_head.rows, s);
        C.pre_in[g] = 0; C.pre_gen[g] = 0;
    }
    if (!p2p_ && prequeue) {
        { const double tw = tp_now(); tp_stage_wait(C, g, cg2); g_collt[g][0] += tp_now() - tw; }
        C.t_comp[g] += tp_now() - *mark;
        { const double t0 = tp_now(); tp_allreduce_nr(C, g, NR);
          C.t_coll[g] += tp_now() - t0; C.n_coll[g]++; *mark = tp_now(); }
        __atomic_store_n(&C.flag_h[g], gn2, __ATOMIC_RELEASE);
    } else { C.t_comp[g] += tp_now() - *mark; *mark = tp_now(); }
    if (!Lnext) g_nr_warmed = true;
}

// =============================================================================================
// Q27_SPEC: ONE SPECULATIVE ROUND. The MTP head drafts ONE token from (h_prev, emb(tok)) -- the
// pairing the module was trained on: the trunk's final residual at position pos-1 and the
// embedding of the token at pos -- then the trunk verifies [tok @ pos, draft @ pos+1] in ONE
// row-batched pass and the draft is accepted iff it equals the trunk's own token for position
// pos+1 (greedy). Returns the number of committed tokens (1 or 2) in outtok[]. All four cards run
// the identical control flow (the tokens are agreed by the 4-way exchanges), so the decode
// loop's barriers stay in lockstep. The draft layer's own KV cache is indexed by the hidden's
// position relative to the first drafted position, so it attends only to positions it has
// actually seen (the prompt is not in it; a zero-filled prefix would dilute its attention).
// =============================================================================================
// Q27_SPEC: NR CONSECUTIVE ROWS THROUGH THE MTP DRAFT LAYER IN ONE PASS. Row r = (hidden h_r,
// token t_r) at draft position dpos0 + r: the catch-up rows (accepted positions the layer has not
// consumed yet) and the round's draft, in one weight pass with the same row-batched kernels the
// verify step uses; per-row norms; both rows through one attention prep/decode (row r attends to
// its predecessors). Only head_row's logits are produced (S.dpa). The draft's numerics differ
// from the single-row mtp_draft_step in fp32 add order only; the trunk verifies every draft.
struct MtpNr {
    unsigned short *cat = nullptr, *emb = nullptr, *nrm = nullptr, *hid = nullptr, *qkva = nullptr, *qh = nullptr, *mix6 = nullptr;
    signed char *xqa = nullptr, *xqb = nullptr, *xqo = nullptr, *xq2 = nullptr;
    float *xsa = nullptr, *xsb = nullptr, *xso = nullptr, *xs2 = nullptr, *pa = nullptr, *pb = nullptr, *pd = nullptr;
    int cap = 0;
};
static MtpNr g_mtpnr[Q27_MAX_DEVICES];
// Q27_MTP_PP: PROMPT CONDITIONING of the draft layer. After the prompt sweep, every card runs the MTP layer over
// the prompt positions (fc of [norm_emb(next token) | norm_hid(final residual)] -> in_norm -> q|k|v -> chunk
// attention), so the draft's own KV cache holds the prompt instead of starting speculation context-blind. Only
// the K/V write matters: the layer's attention/MLP outputs are never consumed for prompt positions, so the pass
// stops after the attention kernel. Inputs: the head card exports each chunk's final residual (h' + pending,
// exactly the finals' arithmetic) as bf16 into pinned host memory during the sweep; the next tokens are the prompt.
struct MtpPP {
    q27_i8g_t kv8g[2], fc8g[2]; q27_i8r_t kv8r[2], fc8r[2];     // int8 per-row mirrors (rocBLAS shapes): k|v stacked, fc halves
    float *h32 = nullptr, *e32 = nullptr, *res = nullptr, *fin32 = nullptr, *kv32 = nullptr;
    unsigned short *hbf = nullptr, *ebf = nullptr, *finbf = nullptr, *estage = nullptr, *kvbf = nullptr;   // estage: pinned host
    int ready = 0; double t_ms = 0; long npos = 0;
};
static MtpPP g_mtppp[Q27_MAX_DEVICES];
static unsigned short* g_mtp_hall = nullptr; static size_t g_mtp_hall_cap = 0;   // pinned [cap][HID] bf16: final residual per prompt slot
static int g_mtp_pp = 1;   // Q27_MTP_PP=0 disables prompt conditioning (context-blind draft, the rc-20260912b behaviour)
// Embedding row host->device without a blocking pageable memcpy: host copies the 10 KB row into a pinned
// bounce slot and enqueues an async H2D behind the stream. Slots cycle per call; a round uses at most
// 10 of the 16 and every round ends with a stream-level wait, so a slot is never reused in flight.
static inline void spec_emb_h2d(Dev& d, unsigned short* dst, const unsigned short* emb_tbl, unsigned tok, hipStream_t s) {
    const unsigned et = tok < (unsigned)Q27_VOCAB ? tok : 0u;
    if (!d.nr_ebounce) { CK(hipMemcpy(dst, emb_tbl + (size_t)et * Q27_HID, (size_t)Q27_HID * 2, hipMemcpyHostToDevice)); return; }
    unsigned short* b = d.nr_ebounce + (size_t)(d.nr_eslot & 31) * Q27_HID; d.nr_eslot++;
    std::memcpy(b, emb_tbl + (size_t)et * Q27_HID, (size_t)Q27_HID * 2);
    CK(hipMemcpyAsync(dst, b, (size_t)Q27_HID * 2, hipMemcpyHostToDevice, s));
}
static void mtp_nr_alloc(int id) {
    MtpNr& B = g_mtpnr[id]; if (B.cap) return;
    const size_t N = 8, HID = Q27_HID, QS = Q27_QROWS + 2 * Q27_KVROWS;   // K=7: catch-up rows + the draft row, one pass
    auto A = [&](void** p, size_t b) { CK(hipMalloc(p, b)); CK(hipMemset(*p, 0, b)); };
    A((void**)&B.cat, N * 2 * HID * 2); A((void**)&B.emb, N * HID * 2); A((void**)&B.nrm, N * HID * 2); A((void**)&B.hid, N * HID * 2);
    A((void**)&B.qkva, N * QS * 2); A((void**)&B.qh, N * Q27_OROWS * 2); A((void**)&B.mix6, N * Q27_OROWS * 2);
    A((void**)&B.xqa, N * HID); A((void**)&B.xqb, N * HID); A((void**)&B.xqo, N * Q27_OROWS); A((void**)&B.xq2, N * Q27_INTER);
    A((void**)&B.xsa, N * (HID / 16) * 4); A((void**)&B.xsb, N * (HID / 16) * 4); A((void**)&B.xso, N * (Q27_OROWS / 16) * 4); A((void**)&B.xs2, N * (Q27_INTER / 16) * 4);
    A((void**)&B.pa, N * Q27_INTER * 4); A((void**)&B.pb, N * Q27_INTER * 4); A((void**)&B.pd, N * HID * 4);
    B.cap = (int)N;
}
// ---- Q27_MTP_GRAPH: the draft pass as two captured graphs ---------------------------------
// The draft block is a pure per-card kernel sequence -- q27_mtp_defs.h: "with NO communication at
// all" -- so it is the one part of the step that graphs cleanly. Only TWO of its ~60 launches
// depend on the token position (attn_prep / attn_decode take pos0), so the block is captured
// AROUND them: graph A up to the q/k/v projections, the two position-dependent launches issued
// normally, graph B from the attention epilogue to the head. No kernel ABI change, and no
// per-launch node patching.
// Because those two calls are also the only users of d.mtp_kc/vc/ks/vs, neither graph captures a
// draft-KV pointer -- so Q27_MTP_PAGED growth cannot invalidate them. Keyed by (card, NR,
// head_row): NR sets every buffer offset and head_row selects which row feeds the head.
struct MtpGraph { hipGraphExec_t a = nullptr, b = nullptr; };
static MtpGraph g_mtp_graph[Q27_MAX_DEVICES][9][10];
static int g_mtp_graph_on = 0;

static void mtp_pass_nr(Dev& d, MtpState& S, const q27_globals_t* G, int NR, const unsigned short* const* hrows,
                        const unsigned* toks, const unsigned short* emb_tbl, int dpos0, int head_row) {
    hipStream_t s = d.stream;
    mtp_nr_alloc(d.id);
    MtpNr& B = g_mtpnr[d.id];
    const float IS = g_mtp_is;
    const int g = d.id;   // TPF bracket index (logical worker)
    const int HID = Q27_HID, QS = Q27_QROWS + 2 * Q27_KVROWS, OR = Q27_OROWS, IN = Q27_INTER;
    for (int r = 0; r < NR; ++r) spec_emb_h2d(d, B.emb + (size_t)r * HID, emb_tbl, toks[r], s);
    const bool gon = g_mtp_graph_on && NR >= 1 && NR <= 8 && head_row < 9;
    MtpGraph& GR = g_mtp_graph[d.id][NR][head_row + 1];
    auto capture = [&](void (*unused)(), hipGraphExec_t* out, const std::function<void()>& fn) {
        (void)unused;
        CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
        fn();
        hipGraph_t gg = nullptr;
        CK(hipStreamEndCapture(s, &gg));
        CK(hipGraphInstantiate(out, gg, nullptr, nullptr, 0));
        CK(hipGraphDestroy(gg));
    };
    std::function<void()> bodyA = [&]() {
    TPF(TPF_DR_PRE, s, ([&]{ for (int r = 0; r < NR; ++r) {
        unsigned short* cat = B.cat + (size_t)r * 2 * HID;
        q27_rmsnorm(B.emb + (size_t)r * HID, S.W.pre_emb, cat + mtp_emb_off(), HID, g_mtp_po, s);
        q27_rmsnorm(hrows[r], S.W.pre_hid, cat + mtp_hid_off(), HID, g_mtp_po, s);
        q27_quant_perm(cat,       B.xqa + (size_t)r * HID, B.xsa + (size_t)r * (HID / 16), HID, IS, s);
        q27_quant_perm(cat + HID, B.xqb + (size_t)r * HID, B.xsb + (size_t)r * (HID / 16), HID, IS, s);
    } })());
    TPF(TPF_DR_FC, s, nr_chunks(NR, [&](int c0, int cn) {
        q27_proj_nvfp4_nr(&S.W.fc[0], B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.pa + (size_t)c0 * HID, HID, cn, s);
        q27_proj_nvfp4_nr(&S.W.fc[1], B.xqb + (size_t)c0 * HID, B.xsb + (size_t)c0 * (HID / 16), B.pb + (size_t)c0 * HID, HID, cn, s);
    }));
    TPF(TPF_DR_PRE, s, ([&]{ for (int r = 0; r < NR; ++r) {
        q27_f2bf_vec(B.pa + (size_t)r * HID, B.nrm + (size_t)r * HID, HID, s);          // residual stream
        q27_add_inplace(B.nrm + (size_t)r * HID, B.pb + (size_t)r * HID, HID, s);
        if (!q27_rmsnorm_quant_fp8(B.nrm + (size_t)r * HID, S.W.in_norm, B.hid + (size_t)r * HID, B.xqa + (size_t)r * HID,
                                   B.xsa + (size_t)r * (HID / 16), HID, g_mtp_po, IS, nullptr, 0, s)) {
            q27_rmsnorm(B.nrm + (size_t)r * HID, S.W.in_norm, B.hid + (size_t)r * HID, HID, g_mtp_po, s);
            q27_quant_fp8(B.hid + (size_t)r * HID, B.xqa + (size_t)r * HID, B.xsa + (size_t)r * (HID / 16), HID, IS, s);
        }
    } })());
    TPF(TPF_DR_QKV, s, nr_chunks(NR, [&](int c0, int cn) {
        q27_proj_fp8_bf16_nr(&S.W.q, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.qkva + (size_t)c0 * QS, QS, cn, s);
        q27_proj_fp8_bf16_nr(&S.W.k, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.qkva + (size_t)c0 * QS + Q27_QROWS, QS, cn, s);
        q27_proj_fp8_bf16_nr(&S.W.v, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.qkva + (size_t)c0 * QS + Q27_QROWS + Q27_KVROWS, QS, cn, s);
    }));
    };   // bodyA
    if (gon) { if (!GR.a) capture(nullptr, &GR.a, bodyA); CK(hipGraphLaunch(GR.a, s)); } else bodyA();
    TPF(TPF_DR_ATTN, s, nr_chunks(NR, [&](int c0, int cn) {
        q27_attn_prep_tp_nr(B.qkva + (size_t)c0 * QS, QS, S.W.q_norm, S.W.k_norm, d.mtp_kc, d.mtp_ks, d.mtp_vc, d.mtp_vs,
                            B.qh + (size_t)c0 * OR, OR, dpos0 + c0, cn, 1, Q27_KVROWS, 0, s);
        q27_attn_decode_tp_nr(B.qh + (size_t)c0 * OR, OR, d.mtp_kc, d.mtp_ks, d.mtp_vc, d.mtp_vs, B.qkva + (size_t)c0 * QS, QS,
                              B.mix6 + (size_t)c0 * OR, OR, dpos0 + c0, cn, 1, Q27_KVROWS, 0, s);
    }));
    std::function<void()> bodyB = [&]() {
    TPF(TPF_DR_O, s, nr_chunks(NR, [&](int c0, int cn) { q27_quant_fp8_b(B.mix6 + (size_t)c0 * OR, (size_t)OR, B.xqo + (size_t)c0 * OR, (size_t)OR, B.xso + (size_t)c0 * (OR / 16), (size_t)(OR / 16), OR, IS, cn, s); }));
    TPF(TPF_DR_O, s, nr_chunks(NR, [&](int c0, int cn) { q27_proj_fp8_bf16_nrs(&S.W.o, B.xqo + (size_t)c0 * OR, OR, B.xso + (size_t)c0 * (OR / 16), B.hid + (size_t)c0 * HID, HID, cn, s); }));   // K = 6144
    TPF(TPF_DR_O, s, ([&]{ for (int r = 0; r < NR; ++r) {
        q27_add_inplace_bf16(B.nrm + (size_t)r * HID, B.hid + (size_t)r * HID, HID, s);
        if (!q27_rmsnorm_quant_perm(B.nrm + (size_t)r * HID, S.W.post_norm, B.hid + (size_t)r * HID, B.xqa + (size_t)r * HID,
                                    B.xsa + (size_t)r * (HID / 16), HID, g_mtp_po, IS, nullptr, 0, s)) {
            q27_rmsnorm(B.nrm + (size_t)r * HID, S.W.post_norm, B.hid + (size_t)r * HID, HID, g_mtp_po, s);
            q27_quant_perm(B.hid + (size_t)r * HID, B.xqa + (size_t)r * HID, B.xsa + (size_t)r * (HID / 16), HID, IS, s);
        }
    } })());
    TPF(TPF_DR_MLP, s, nr_chunks(NR, [&](int c0, int cn) {
        q27_proj_nvfp4_gu_nr(&S.W.gate, &S.W.up, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16),
                             B.pa + (size_t)c0 * IN, B.pb + (size_t)c0 * IN, IN, cn, s);
        q27_swiglu_quant_b(B.pa + (size_t)c0 * IN, B.pb + (size_t)c0 * IN, B.xq2 + (size_t)c0 * IN,
                           B.xs2 + (size_t)c0 * (IN / 16), IN, IS, cn, s);
    }));
    TPF(TPF_DR_MLP, s, ([&]{ for (int j = 0; j < 4; ++j) {
        nr_chunks(NR, [&](int c0, int cn) {
            q27_proj_nvfp4_down_nrs(&S.W.down[j], B.xq2 + (size_t)c0 * IN + (size_t)j * 4352, IN,
                                    B.xs2 + (size_t)c0 * (IN / 16) + (size_t)j * 272, B.pd + (size_t)c0 * HID, HID, nullptr, 0, 0.f, cn, s);
        });
        for (int r = 0; r < NR; ++r) q27_add_inplace(B.nrm + (size_t)r * HID, B.pd + (size_t)r * HID, HID, s);
    } })());
    if (head_row >= 0) TPF(TPF_DR_HEAD, s, ([&]{
        q27_rmsnorm(B.nrm + (size_t)head_row * HID, S.W.mtp_norm, B.hid, HID, 1, s);
        q27_quant_perm(B.hid, B.xqa, B.xsa, HID, G->lm_head.in_scale, s);
        if (!exl3_head(d, G, B.hid, S.dpa, 1, s)) {
            // 2026-09-28: the draft's logit shard through the row-batched NR form (the trunk tail's P_LM), NR=1;
            // Q27_DRAFT_HEAD_PLAIN=1 restores the plain q27_proj_nvfp4 launch for A/B.
            static const int plain = q27_env_int("Q27_DRAFT_HEAD_PLAIN", 0);
            if (plain || !q27_proj_nvfp4_nr(&G->lm_head, B.xqa, B.xsa, S.dpa, G->lm_head.rows, 1, s))
                q27_proj_nvfp4(&G->lm_head, B.xqa, B.xsa, S.dpa, s);                            // this card's logit shard
        }
    })());
    };   // bodyB
    if (gon) { if (!GR.b) capture(nullptr, &GR.b, bodyB); CK(hipGraphLaunch(GR.b, s)); } else bodyB();
}

// Q27_MTP_TP (2026-09-28): the draft layer as a TP-sharded full-attention layer. Every card runs fc + the pre-norms on
// the full hidden (replicated, cheap), then ITS q heads / kv heads / o K-slice / MLP rows, and the two LL sites carry the
// two partial sums exactly as a trunk layer's do (residual folded into the producers with rs = 1/ndev; site 1 consumer
// = post_norm + permuted quant for gate/up, site 2 consumer = mtp.norm + permuted quant for the lm_head shard).
// The draft KV cache stays full-width (the prompt pass fills every head from k_full/v_full); decode writes and reads only
// this card's heads at their offset, as the trunk does for a shard reading a wider stored row.
static void mtp_pass_nr_tp(Dev& d, MtpState& S, const q27_globals_t* G, int NR, const unsigned short* const* hrows,
                           const unsigned* toks, const unsigned short* emb_tbl, int dpos0, int head_row, TpColl& C, int ndev) {
    hipStream_t s = d.stream;
    mtp_nr_alloc(d.id);
    MtpNr& B = g_mtpnr[d.id];
    const float IS = g_mtp_is;
    const int g = d.id;   // TPF bracket index (logical worker)
    const q27_tp_shard_t* T = q27_tp_shard(g);
    const int HID = Q27_HID, QL = T->QL, KVL = T->KVL, QS = QL + 2 * KVL, OR = T->OL, IN = T->mlp_len;
    const int hoff = T->kv_h0 * Q27_HDIM;
    const float rshare = 1.0f / (float)ndev;
    float* const part  = C.nrp_part[0][g];   // [NR][HID] fp32 partials (8 rows deep)
    float* const part2 = C.nrp_part[1][g];
    for (int r = 0; r < NR; ++r) spec_emb_h2d(d, B.emb + (size_t)r * HID, emb_tbl, toks[r], s);
    TPF(TPF_DR_PRE, s, ([&]{ for (int r = 0; r < NR; ++r) {
        unsigned short* cat = B.cat + (size_t)r * 2 * HID;
        q27_rmsnorm(B.emb + (size_t)r * HID, S.W.pre_emb, cat + mtp_emb_off(), HID, g_mtp_po, s);
        q27_rmsnorm(hrows[r], S.W.pre_hid, cat + mtp_hid_off(), HID, g_mtp_po, s);
        q27_quant_perm(cat,       B.xqa + (size_t)r * HID, B.xsa + (size_t)r * (HID / 16), HID, IS, s);
        q27_quant_perm(cat + HID, B.xqb + (size_t)r * HID, B.xsb + (size_t)r * (HID / 16), HID, IS, s);
    } })());
    TPF(TPF_DR_FC, s, nr_chunks(NR, [&](int c0, int cn) {
        q27_proj_nvfp4_nr(&S.W.fc[0], B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.pa + (size_t)c0 * HID, HID, cn, s);
        q27_proj_nvfp4_nr(&S.W.fc[1], B.xqb + (size_t)c0 * HID, B.xsb + (size_t)c0 * (HID / 16), B.pb + (size_t)c0 * HID, HID, cn, s);
    }));
    TPF(TPF_DR_PRE, s, ([&]{ for (int r = 0; r < NR; ++r) {
        q27_f2bf_vec(B.pa + (size_t)r * HID, B.nrm + (size_t)r * HID, HID, s);          // residual stream
        q27_add_inplace(B.nrm + (size_t)r * HID, B.pb + (size_t)r * HID, HID, s);
        if (!q27_rmsnorm_quant_fp8(B.nrm + (size_t)r * HID, S.W.in_norm, B.hid + (size_t)r * HID, B.xqa + (size_t)r * HID,
                                   B.xsa + (size_t)r * (HID / 16), HID, g_mtp_po, IS, nullptr, 0, s)) {
            q27_rmsnorm(B.nrm + (size_t)r * HID, S.W.in_norm, B.hid + (size_t)r * HID, HID, g_mtp_po, s);
            q27_quant_fp8(B.hid + (size_t)r * HID, B.xqa + (size_t)r * HID, B.xsa + (size_t)r * (HID / 16), HID, IS, s);
        }
    } })());
    TPF(TPF_DR_QKV, s, nr_chunks(NR, [&](int c0, int cn) {   // this card's q heads and kv heads only
        const q27_fp8_t* sw[3] = { &S.W.q, &S.W.k, &S.W.v };
        unsigned short* sy[3] = { B.qkva + (size_t)c0 * QS, B.qkva + (size_t)c0 * QS + QL, B.qkva + (size_t)c0 * QS + QL + KVL };
        const int sst[3] = { QS, QS, QS };
        if (!(g_proj_seg && q27_proj_i8g_bf16_seg(sw, sy, sst, 3, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), cn, s))) {
            q27_proj_fp8_bf16_nr(&S.W.q, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.qkva + (size_t)c0 * QS, QS, cn, s);
            q27_proj_fp8_bf16_nr(&S.W.k, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.qkva + (size_t)c0 * QS + QL, QS, cn, s);
            q27_proj_fp8_bf16_nr(&S.W.v, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16), B.qkva + (size_t)c0 * QS + QL + KVL, QS, cn, s);
        }
    }));
    TPF(TPF_DR_ATTN, s, nr_chunks(NR, [&](int c0, int cn) {
        q27_attn_prep_tp_nr(B.qkva + (size_t)c0 * QS, QS, S.W.q_norm, S.W.k_norm, d.mtp_kc, d.mtp_ks, d.mtp_vc, d.mtp_vs,
                            B.qh + (size_t)c0 * OR, OR, dpos0 + c0, cn, T->attn_g, Q27_KVROWS, hoff, s);
        q27_attn_decode_tp_nr(B.qh + (size_t)c0 * OR, OR, d.mtp_kc, d.mtp_ks, d.mtp_vc, d.mtp_vs, B.qkva + (size_t)c0 * QS, QS,
                              B.mix6 + (size_t)c0 * OR, OR, dpos0 + c0, cn, T->attn_g, Q27_KVROWS, hoff, s);
    }));
    // ---- site 1: o_proj K-slice with the residual folded in (rs = 1/ndev), then the LL all-reduce + post_norm + permuted quant ----
    TPF(TPF_DR_O, s, nr_chunks(NR, [&](int c0, int cn) { q27_quant_fp8_b(B.mix6 + (size_t)c0 * OR, (size_t)OR, B.xqo + (size_t)c0 * OR, (size_t)OR, B.xso + (size_t)c0 * (OR / 16), (size_t)(OR / 16), OR, IS, cn, s); }));
    TPF(TPF_DR_O, s, nr_chunks(NR, [&](int c0, int cn) {
        if (!q27_proj_fp8_res_nr(&S.W.o, B.xqo + (size_t)c0 * OR, B.xso + (size_t)c0 * (OR / 16), part + (size_t)c0 * HID, HID, B.nrm + (size_t)c0 * HID, HID, rshare, cn, s)) {
            std::fprintf(stderr, "Q27_MTP_TP: o_proj (K=%d) declined\n", S.W.o.K); std::abort(); }
    }));
    nrp_site(C, g, ndev, 0, s, part, NR, 4);
    if (!nrp_ll_reduce(C, g, ndev, 1, part, S.W.post_norm, B.nrm, nullptr, B.xqa, B.xsa, g_mtp_po, S.W.gate.in_scale, NR, s)) { std::fprintf(stderr, "Q27_MTP_TP: site 1 declined\n"); std::abort(); }
    // ---- MLP rows [mlp_off, +mlp_len): gate/up, swiglu, down K-slice with the residual folded in; site 2: LL + mtp.norm + head quant ----
    TPF(TPF_DR_MLP, s, nr_chunks(NR, [&](int c0, int cn) {
        q27_proj_nvfp4_gu_nr(&S.W.gate, &S.W.up, B.xqa + (size_t)c0 * HID, B.xsa + (size_t)c0 * (HID / 16),
                             B.pa + (size_t)c0 * IN, B.pb + (size_t)c0 * IN, IN, cn, s);
        q27_swiglu_quant_b(B.pa + (size_t)c0 * IN, B.pb + (size_t)c0 * IN, B.xq2 + (size_t)c0 * IN,
                           B.xs2 + (size_t)c0 * (IN / 16), IN, IS, cn, s);
        if (!q27_proj_nvfp4_down_nr(&S.W.down[0], B.xq2 + (size_t)c0 * IN, B.xs2 + (size_t)c0 * (IN / 16), part2 + (size_t)c0 * HID, HID, B.nrm + (size_t)c0 * HID, HID, rshare, cn, s)) {
            std::fprintf(stderr, "Q27_MTP_TP: down (K=%d) declined\n", S.W.down[0].K); std::abort(); }
    }));
    nrp_site(C, g, ndev, 1, s, part2, NR, 4);
    if (!nrp_ll_reduce(C, g, ndev, 1, part2, S.W.mtp_norm, B.nrm, nullptr, B.xqa, B.xsa, 1, G->lm_head.in_scale, NR, s)) { std::fprintf(stderr, "Q27_MTP_TP: site 2 declined\n"); std::abort(); }
    if (head_row >= 0) TPF(TPF_DR_HEAD, s, ([&]{
        const signed char* xq = B.xqa + (size_t)head_row * HID; const float* xs = B.xsa + (size_t)head_row * (HID / 16);
        if (!q27_proj_nvfp4_nr(&G->lm_head, xq, xs, S.dpa, G->lm_head.rows, 1, s))
            q27_proj_nvfp4(&G->lm_head, xq, xs, S.dpa, s);                            // this card's logit shard
    })());
}

static int mtp_prompt_init(Dev& d, int g, int M) {
    MtpPP& P = g_mtppp[g]; if (P.ready) return 1;
    MtpState& S = g_mtp[g];
    const q27_fp8_t* s2[2] = { g_mtp_tp ? &S.W.k_full : &S.W.k, g_mtp_tp ? &S.W.v_full : &S.W.v };   // only k and v feed the cache (full width: the prompt pass fills every head)
    if (!q27_fp8_to_i8g64_cat(s2, 2, P.kv8g, 0)) { std::fprintf(stderr, "Q27_MTP_PP: k|v int8 mirror failed\n"); return 0; }
    for (int i = 0; i < 2; ++i) if (!q27_i8g64_to_row_conv(&P.kv8g[i], &P.kv8r[i], 0, 0)) { std::fprintf(stderr, "Q27_MTP_PP: k|v row view failed\n"); return 0; }
    for (int j = 0; j < 2; ++j) {
        if (!q27_nvfp4_to_i8g64_conv(&S.W.fc[j], &P.fc8g[j], 0)) { std::fprintf(stderr, "Q27_MTP_PP: fc int8 mirror failed\n"); return 0; }
        if (!q27_i8g64_to_row_conv(&P.fc8g[j], &P.fc8r[j], 0, 0)) { std::fprintf(stderr, "Q27_MTP_PP: fc row view failed\n"); return 0; }
    }
    auto A = [&](void** p, size_t b) { CK(hipMalloc(p, b)); CK(hipMemset(*p, 0, b)); };
    const size_t HID = Q27_HID;
    A((void**)&P.h32, (size_t)M * HID * 4); A((void**)&P.e32, (size_t)M * HID * 4); A((void**)&P.res, (size_t)M * HID * 4); A((void**)&P.fin32, (size_t)M * HID * 4);
    A((void**)&P.hbf, (size_t)M * HID * 2); A((void**)&P.ebf, (size_t)M * HID * 2); A((void**)&P.finbf, (size_t)M * HID * 2);
    A((void**)&P.kv32, (size_t)M * 2 * Q27_KVROWS * 4); A((void**)&P.kvbf, (size_t)M * 2 * Q27_KVROWS * 2);
    CK(hipHostMalloc((void**)&P.estage, (size_t)M * HID * 2, hipHostMallocDefault));
    CK(hipDeviceSynchronize());
    // rocBLAS solutions for the pass's two shapes at the chunk width (the q|k|v shape is the trunk's, already tuned)
    const Q8Scr& Q = d.q8s[0];
    const q27_i8r_t* w2[2] = { &P.kv8r[0], &P.kv8r[1] };
    const int ntot = q27_cat_rows(w2, 2);
    if (ntot > 0) q27_rb_tune_i8g(d.id, w2[0]->w, ntot, w2[0]->K, Q.xqt, M, Q.acc, d.stream);
    else for (int i = 0; i < 2; ++i) q27_rb_tune_i8g(d.id, w2[i]->w, w2[i]->rows, w2[i]->K, Q.xqt, M, Q.acc, d.stream);
    q27_rb_tune_i8g(d.id, P.fc8r[0].w, P.fc8r[0].rows, P.fc8r[0].K, Q.xqt, M, Q.acc, d.stream);
    CK(hipStreamSynchronize(d.stream));
    P.ready = 1;
    return 1;
}
// The pass over positions [pbeg, pend): position p consumes (final residual at p, embedding of the token at p+1).
static void mtp_prompt_pass(Dev& d, int g, int pbeg, int pend, int sb, const unsigned* sw_tok, int sw_lo,
                            const std::vector<unsigned>& prompt, const unsigned short* emb_tbl, int M) {
    MtpPP& P = g_mtppp[g]; MtpState& S = g_mtp[g]; hipStream_t s = d.stream; const Q8Scr& Q = d.q8s[0];
    const int HID = Q27_HID, QS = Q27_QROWS + 2 * Q27_KVROWS; const float IS = g_mtp_is;
    static const bool q8g64 = q27_env_flag("Q27_Q8_G64", true);
    const int e_idx = (mtp_emb_off() == 0) ? 0 : 1, h_idx = 1 - e_idx;   // which fc half consumes which concat half
    auto die = [&](const char* w) { std::fprintf(stderr, "Q27_MTP_PP: %s declined; aborting\n", w); std::exit(1); };
    for (int p0 = pbeg; p0 < pend; p0 += M) {
        const int Cp = (pend - p0 < M) ? (pend - p0) : M;
        CK(hipMemcpyAsync(P.hbf, g_mtp_hall + (size_t)(p0 - sb) * HID, (size_t)Cp * HID * 2, hipMemcpyHostToDevice, s));
        for (int c = 0; c < Cp; ++c) {
            const int p = p0 + c + 1;
            const unsigned t = sw_tok ? sw_tok[p - sw_lo] : prompt[(size_t)p];
            const unsigned et = t < (unsigned)Q27_VOCAB ? t : 0u;
            std::memcpy(P.estage + (size_t)c * HID, emb_tbl + (size_t)et * HID, (size_t)HID * 2);
        }
        CK(hipMemcpyAsync(P.ebf, P.estage, (size_t)Cp * HID * 2, hipMemcpyHostToDevice, s));
        q27_bf16_to_f32(P.hbf, P.h32, (size_t)Cp * HID, s);
        q27_bf16_to_f32(P.ebf, P.e32, (size_t)Cp * HID, s);
        if (!q27_red_rn_tok_b1(P.h32, S.W.pre_hid, nullptr, Q.xq1, Q.xst1, HID, g_mtp_po, IS, Cp, 0, s)) die("pre_hid norm");
        if (!q27_red_rn_tok_b1(P.e32, S.W.pre_emb, nullptr, Q.xqt, Q.xst,  HID, g_mtp_po, IS, Cp, 0, s)) die("pre_emb norm");
        {   const q27_i8r_t* we = &P.fc8r[e_idx]; const q27_i8r_t* wh = &P.fc8r[h_idx];
            if (!q27_rb_gemm_i8g(d.id, we->w, we->rows, we->K, Q.xqt, Cp, Q.acc, we->gs, s)) die("fc gemm (emb)");
            if (!q27_epi_f32r(Q.acc, we->rows, we->s, we->ng, Q.xst, we->alpha, P.res, nullptr, 0.f, Cp, s)) die("fc epi (emb)");
            if (!q27_rb_gemm_i8g(d.id, wh->w, wh->rows, wh->K, Q.xq1, Cp, Q.acc, wh->gs, s)) die("fc gemm (hid)");
            if (!q27_epi_f32r(Q.acc, wh->rows, wh->s, wh->ng, Q.xst1, wh->alpha, P.res, P.res, 1.0f, Cp, s)) die("fc epi (hid)");
        }
        if (!q27_red_rn_tok_b1(P.res, S.W.in_norm, nullptr, Q.xqt, Q.xst, HID, g_mtp_po, IS, Cp, 0, s)) die("in_norm");
        {   // k|v (N = 2048) -> fp32 -> bf16 rows [Cp][2*KVROWS] -> K/V-only cache write (k_norm, RoPE, int8, per-16 scales)
            const q27_i8r_t* w2[2] = { &P.kv8r[0], &P.kv8r[1] };
            const int ntot = q27_cat_rows(w2, 2);
            if (ntot > 0) {
                if (!q27_rb_gemm_i8g(d.id, w2[0]->w, ntot, w2[0]->K, Q.xqt, Cp, Q.acc, w2[0]->gs, s)) die("kv gemm");
                int off = 0;
                for (int i = 0; i < 2; ++i) { const q27_i8r_t* w = w2[i];
                    if (!q27_epi_f32r_ld(Q.acc + off, ntot, w->rows, w->s, w->ng, Q.xst, w->alpha, P.kv32 + (size_t)i * Q27_KVROWS, 2 * Q27_KVROWS, Cp, s)) die("kv epi");
                    off += w->rows; }
            } else die("kv mirror not contiguous");
            q27_f2bf_vec(P.kv32, P.kvbf, Cp * 2 * Q27_KVROWS, s);
            q27_attn_kvprep_nr(P.kvbf, 2 * Q27_KVROWS, S.W.k_norm, d.mtp_kc, d.mtp_ks, d.mtp_vc, d.mtp_vs, p0, Cp, s);
        }
        CK(hipStreamSynchronize(s));   // the pinned embedding staging is reused by the next chunk
    }
}
// Q27_SPEC: the argmax result (tok[8], val[8]) travels to the host through a pinned mailbox behind the
// staging done-flag (one in-stream copy kernel; the host spins on the flag) instead of a stream
// synchronize plus two blocking memcpys.
static inline void spec_mail_wait(Dev& d, TpColl& C, int g, hipStream_t s, int n) {
    const unsigned gen = ++C.cgen[g];
    const unsigned target = 8u * (++C.cp_calls[g]);
    q27_copy_f4_flag_mb(d.nr_mail_d, (const float*)d.nr_tok, n, C.done_d[g] + g, gen, C.cp_cnt[g], target, 8, s);
    tp_stage_wait(C, g, gen);
}
#include "q27_sr_main.inc"   // Q27_LS_SR: single residency -- sequential decode round, prefill mirror ring, boot helpers
// `K` IS A PARAMETER, NOT g_spec_k, AND THAT IS LOAD-BEARING. When a decode walks out of the draft
// KV's reach the round must still run -- with K=0, i.e. one trunk row through run_layer_tp_nr, which
// is the path the product actually runs (int8 MLP mirror, K-views, shipped collectives). Falling into
// the PLAIN step instead is not a slower step, it is a crash: the plain path has no Q27_DEC_MLP_I8
// dispatch and dereferences the NVFP4 MLP shards that DEC_MLP_I8 deliberately never allocates. That is
// the root cause recorded in DECODE_AUTHORITY_AND_LS_PREFILL_GATE_20260915.md for Q27_SPEC=0
// (`gate=(nil)` -> "Memory access fault by GPU node-1"). Measured again 2026-09-15 22:28:07: a Harness
// turn at pos 65,388 generating past the 65,536 draft-KV cap took the plain step and the engine died
// mid-generation. K=0 is the documented working non-draft configuration; use it.
static int spec_round(Dev& d, q27_model_t* m, const unsigned short* emb, int nlayer, int pos, unsigned tok,
                      int g, int ndev, TpColl& C, double* mark, unsigned* outtok, int K) {
    hipStream_t s = d.stream;
    const double sp0 = tp_now(); double spa = sp0;
    d.nr_eslot = 0;
    auto lap = [&](int k) { const double t = tp_now(); d.sp_t[k] += t - spa; spa = t; };
    MtpState& S = g_mtp[g];
    const q27_globals_t* G = q27_globals(m, d.id);
    const int NR = K + 1, HID = Q27_HID;
    const int n_gdn = Q27_LAYERS - Q27_LAYERS / 4;
    const size_t sshard = (size_t)q27_tp_shard(g)->VHL * Q27_GDN_D * Q27_GDN_D;
    if (K > 0 && !d.nr_Ssnap) {   // snapshot banks (normally allocated at init; lazy fallback), sized by the SHARD
        CK(hipMalloc((void**)&d.nr_Ssnap, (size_t)K * n_gdn * sshard * 4));
        CK(hipMalloc((void**)&d.nr_csnap, (size_t)K * n_gdn * d.tp_conv * 2));
    }
    if (g_df2_draft && !g_df2_prompt_done[g]) {   // DFlash2 prompt conditioning: inject the full prompt's features once
        char derr[128] = {0};
        for (int c0 = 0; c0 < g_df2_nprompt; c0 += 8) {
            const int cn = (g_df2_nprompt - c0 < 8) ? g_df2_nprompt - c0 : 8;
            if (q27_df2_cond_inject(&g_df2c[g], d.df2_pf_taps + (size_t)c0 * Q27_HID, g_pf_cap, cn, c0, g,
                                    df2_reduce_cb, (void*)&C, s, derr, sizeof derr)) {
                std::fprintf(stderr, "Q27_DFLASH2_PFINJ_FAIL %s\n", derr); std::abort();
            }
        }
        g_df2_prompt_done[g] = 1;
        CK(hipStreamSynchronize(s));
        if (g == 0) std::printf("Q27_DFLASH2_PROMPT_INJ %d positions\n", g_df2_nprompt);
    }
    // ---- 1. one MTP pass over the catch-up rows (accepted positions the draft layer has not consumed)
    //         plus the round's draft, then K-1 chained single-row drafts ----
    unsigned dt[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
    double tc0 = 0.0;
    const unsigned short* hchain = nullptr;
    if (!g_df2_draft || !g_df2_d7_ready[g]) {   // df2 mode: the block draft replaces the chain (MTP bootstrap on the first round)
    for (int j = 0; j < K; ++j) {
        if (j == 0) {
            const unsigned short* hrows[8]; unsigned toks[8]; int nrm = 0;
            for (int c = 0; c < d.nr_ncatch; ++c) { hrows[nrm] = d.nr_hcatch + (size_t)c * HID; toks[nrm] = d.nr_catch_tok[c]; ++nrm; }
            hrows[nrm] = d.nr_hprev; toks[nrm] = tok; ++nrm;
            const int dpos0 = (d.nr_ncatch ? d.nr_catch_pos[0] : pos - 1) - d.nr_mtp_base;   // consecutive by construction
            if (g_mtp_tp) mtp_pass_nr_tp(d, S, G, nrm, hrows, toks, emb, dpos0, nrm - 1, C, ndev); else mtp_pass_nr(d, S, G, nrm, hrows, toks, emb, dpos0, nrm - 1);
            d.nr_ncatch = 0;
            hchain = g_mtpnr[d.id].nrm + (size_t)(nrm - 1) * HID;   // the draft row's residual: the next draft's hidden
        } else {
            // draft j consumes the MTP layer's OWN residual output for draft j-1 and emb(draft j-1)
            const unsigned short* hr[1] = { hchain }; const unsigned tk[1] = { dt[j - 1] };
            if (g_mtp_tp) mtp_pass_nr_tp(d, S, G, 1, hr, tk, emb, pos - 1 - d.nr_mtp_base + j, 0, C, ndev); else mtp_pass_nr(d, S, G, 1, hr, tk, emb, pos - 1 - d.nr_mtp_base + j, 0);
            hchain = g_mtpnr[d.id].nrm;
        }
        q27_argmax_val(S.dpa, d.nr_tok, d.nr_val, G->lm_head.rows, s);
        spec_mail_wait(d, C, g, s, 16);   // whole 64-B mailbox: tok[8] | val[8]
        const unsigned li = ((const unsigned*)d.nr_mail_h)[0]; const float lv = d.nr_mail_h[8];   // val[0] lives at word 8 in the tok[8] | val[8] layout
        lap(0);
        C.t_comp[g] += tp_now() - *mark;
        tc0 = tp_now();
        C.hval[g] = lv; C.hidx[g] = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)li);
        C.b1.wait();
        unsigned bt = C.hidx[0]; float bv = C.hval[0];
        for (int k = 1; k < ndev; ++k)
            if (C.hval[k] > bv || (C.hval[k] == bv && C.hidx[k] < bt)) { bv = C.hval[k]; bt = C.hidx[k]; }
        C.b2.wait();
        dt[j] = bt;
        C.t_coll[g] += tp_now() - tc0; C.n_coll[g]++; *mark = tp_now();
        lap(1);
    }
    } else {   // df2: the previous round's block drafts
        for (int j = 0; j < K; ++j) dt[j] = g_df2_d7[g][j];
        hchain = nullptr;
    }
    // ---- 2. verify [tok @ pos, dt[0] @ pos+1, ..., dt[K-1] @ pos+K] in one NR-row pass over the trunk ----
    for (int r = 0; r < NR; ++r) spec_emb_h2d(d, d.hidden_slots + (size_t)r * HID, emb, (r == 0) ? tok : dt[r - 1], s);
    lap(2);
    int gs = 0, fs = 0;
    lap(2);
    for (int L = 0; L < nlayer; ++L) {
        const q27_layer_t* lay = q27_layer_tp(m, L, d.id);
        const q27_layer_t* nxt = (L + 1 < nlayer) ? q27_layer_tp(m, L + 1, d.id) : nullptr;
        run_layer_tp_nr(d, lay, pos, NR, gs, fs, g, ndev, C, mark, L == 0, nxt, G);
        // DFlash2 tap capture: the residual ENTERING layer L+1 (hid rows after this layer's
        // residual add) at the five conditioning boundaries. Committed rows only are consumed later.
        static const int TAP_L[5] = {5, 19, 33, 47, 61};
        for (int t = 0; t < 5; ++t) if (L == TAP_L[t]) {
            CK(hipEventRecord(d.ev_cap, s));
            CK(hipStreamWaitEvent(d.s_cap, d.ev_cap, 0));
            for (int r = 0; r < NR; ++r)   // the FULL residual entering layer L+1: g_coll_acc (fp32 residual-2) + hid (bf16 residual-1)
                q27_k_df2_tapadd(g_coll_acc + (size_t)r * Q27_HID, d.hidden_slots + (size_t)r * Q27_HID,
                                 d.nr_taps + ((size_t)t * 8 + r) * Q27_HID, Q27_HID, d.s_cap);
        }
        if (Q27_IS_FULL(L)) fs++; else gs++;
    }
    lap(3);
    spec_mail_wait(d, C, g, s, 16);               // the head ran behind the last collective's flag; its result lands in the mailbox
    unsigned lis[8]; float lvs[8];
    for (int r = 0; r < NR; ++r) { lis[r] = ((const unsigned*)d.nr_mail_h)[r]; lvs[r] = d.nr_mail_h[8 + r]; }   // val[r] at word 8 + r for every NR
    lap(4);
    C.t_comp[g] += tp_now() - *mark;
    tc0 = tp_now();
    for (int r = 0; r < NR; ++r) { C.hvalN[g][r] = lvs[r]; C.hidxN[g][r] = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)lis[r]); }
    C.b1.wait();
    unsigned nx[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
    for (int r = 0; r < NR; ++r) {
        unsigned bi = C.hidxN[0][r]; float bv = C.hvalN[0][r];
        for (int k = 1; k < ndev; ++k)
            if (C.hvalN[k][r] > bv || (C.hvalN[k][r] == bv && C.hidxN[k][r] < bi)) { bv = C.hvalN[k][r]; bi = C.hidxN[k][r]; }
        nx[r] = bi;
    }
    C.b2.wait();
    C.t_coll[g] += tp_now() - tc0; C.n_coll[g]++; *mark = tp_now();
    // ---- 3. greedy acceptance of the longest matching draft prefix ----
    int nacc = 0;
    while (nacc < K && nx[nacc] == dt[nacc]) ++nacc;
    if (g_df2_draft && g == 0 && g_df2_d7_ready[g]) {   // top-16 overlap: is the target's true next token among the drafter's candidates?
        const unsigned truth = nx[0]; int hit = 0;   // cands[0] drafts position pos+1; nx[0] is the target's token there. nx[1] was the continuation of the REJECTED branch (it conditions on dt[0]), so the old instrument could never match.
        for (int c = 0; c < 16; ++c) if (g_df2_cands[0][c] == truth) { hit = 1; break; }
        if (hit) ++g_df2_cand_hits; ++g_df2_cand_rounds;
        if ((g_df2_cand_rounds & 7) == 0)
            std::printf("Q27_DFLASH2_CAND overlap=%ld/%ld truth=%u cands=%u %u %u %u %u\n",
                        g_df2_cand_hits, g_df2_cand_rounds, truth, g_df2_cands[0][0], g_df2_cands[0][1], g_df2_cands[0][2], g_df2_cands[0][3], g_df2_cands[0][4]);
    }
    static const bool sp_trace = q27_env_flag("Q27_SPEC_TRACE", false);
    if (sp_trace && g == 0) {
        std::printf("Q27_SPEC_TR pos=%d tok=%u nacc=%d dt=", pos, tok, nacc);
        for (int j = 0; j < K; ++j) std::printf("%u%c", dt[j], (j + 1 < K) ? ',' : ' ');
        std::printf("nx=");
        for (int r = 0; r < NR; ++r) std::printf("%u%c", nx[r], (r + 1 < NR) ? ',' : '\n');
        std::fflush(stdout);
    }
    if (g_df2_draft && q27_env_flag("Q27_DFLASH2_SELTEST", false)) {
        // ORACLE: push the TARGET's OWN final-norm row 0 through the DRAFT select path
        // (perm quantize -> shared lm_head -> two-stage top-16 -> 4-card merge). The target's own
        // greedy token nx[0] MUST appear in that top-16 if the select path is sound. If it does not,
        // the defect is in select, not in the drafter's hidden.
        if (g == 0) {   // magnitude of the DRAFTER's own row-1 final-norm vs the TARGET's row-0, same units
            CK(hipStreamSynchronize(s));
            unsigned short a[32], b[32]; float ca = 0.f, cb = 0.f;
            CK(hipMemcpy(a, g_df2c[g].norm + 5120, 64, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b, d.nr_norm, 64, hipMemcpyDeviceToHost));
            for (int i = 0; i < 32; ++i) { unsigned u = (unsigned)a[i] << 16; float f; std::memcpy(&f, &u, 4); ca += fabsf(f);
                                           unsigned v = (unsigned)b[i] << 16; float h; std::memcpy(&h, &v, 4); cb += fabsf(h); }
            std::printf("Q27_DFLASH2_MAG draft_row1=%.4f target_row0=%.4f ratio=%.3f\n", ca, cb, cb > 0.f ? ca / cb : -1.f);
        }
        CK(hipMemcpyAsync(g_df2c[g].norm + 5120, d.nr_norm, (size_t)Q27_HID * 2, hipMemcpyDeviceToDevice, s));
        CK(hipStreamSynchronize(s));
        unsigned t7[7] = {0,0,0,0,0,0,0}; char derr2[128] = {0};
        if (q27_df2_select(&g_df2c[g], &G->lm_head, nx[0], g, df2_exch_cb, (void*)&C, s, t7, derr2, sizeof derr2)) {
            std::fprintf(stderr, "Q27_DFLASH2_SELTEST_FAIL %s\n", derr2); std::abort();
        }
        if (g == 0) {
            int hit = 0; for (int c = 0; c < 16; ++c) if (g_df2_cands[0][c] == nx[0]) { hit = 1; break; }
            std::printf("Q27_DFLASH2_SELTEST truth=%u hit=%d cands=%u %u %u %u\n", nx[0], hit,
                        g_df2_cands[0][0], g_df2_cands[0][1], g_df2_cands[0][2], g_df2_cands[0][3]);
        }
    }
    if (g_df2_draft) {   // DFlash2 block production for the NEXT round (cond on rows 0..nacc, block at pos+nacc, select 7)
        const double t_pre = tp_now();
        CK(hipStreamSynchronize(d.s_cap));
        const double t2 = tp_now();
        char derr[128] = {0};
        if (q27_df2_cond_inject(&g_df2c[g], d.nr_taps, 8, nacc + 1, pos, g, df2_reduce_cb, (void*)&C, s, derr, sizeof derr)) {
            std::fprintf(stderr, "Q27_DFLASH2_COND_FAIL %s\n", derr); std::abort();
        }
        static const bool df2_split = q27_env_flag("Q27_DFLASH2_SPLIT", false);
        double ts_capsync = t2 - t_pre;
        double ts_cond = 0, ts_fwd = 0, ts_sel = 0;
        if (df2_split) { CK(hipStreamSynchronize(s)); ts_cond = tp_now(); }
        const unsigned anch = nx[nacc];
        const int mask_id = q27_env_int("Q27_DFLASH2_MASK", 248070);   // from the DFlash2 config (vllm.cpp test: tokenizer.ggml.mask_token_id = 248070)
        std::memcpy((void*)d.nr_ebounce, (const char*)emb + (size_t)anch * 5120 * 2, 5120 * 2);
        std::memcpy((void*)(d.nr_ebounce + 5120), (const char*)emb + (size_t)mask_id * 5120 * 2, 5120 * 2);
        CK(hipMemcpyAsync(g_df2c[g].anchor, d.nr_ebounce, 5120 * 2, hipMemcpyHostToDevice, s));
        CK(hipMemcpyAsync(g_df2c[g].mask, d.nr_ebounce + 5120, 5120 * 2, hipMemcpyHostToDevice, s));
        float csum = 0.f;
        if (df2_split) { CK(hipStreamSynchronize(s)); ts_fwd = tp_now(); }
        if (q27_df2_forward(&g_df2c[g], g_df2c[g].ctx_end, df2_reduce_cb, (void*)&C, g, s, &csum, derr, sizeof derr)) {
            std::fprintf(stderr, "Q27_DFLASH2_FWD_FAIL %s\n", derr); std::abort();
        }
        if (df2_split) { CK(hipStreamSynchronize(s)); ts_sel = tp_now(); }
        if (q27_df2_select(&g_df2c[g], &G->lm_head, anch, g, df2_exch_cb, (void*)&C, s, g_df2_d7[g], derr, sizeof derr)) {
            std::fprintf(stderr, "Q27_DFLASH2_SEL_FAIL %s\n", derr); std::abort();
        }
        g_df2_d7_ready[g] = 1;
        if (df2_split && g == 0) { CK(hipStreamSynchronize(s)); std::printf("Q27_DFLASH2_SPLIT capsync=%.3f cond=%.3f emb=%.3f fwd=%.3f sel=%.3f total=%.3f ms\n", ts_capsync, ts_cond - t2,
            ts_fwd-ts_cond, ts_sel-ts_fwd, tp_now()-ts_sel, tp_now()-t_pre); }
        CK(hipStreamSynchronize(s));
        if (g == 0) std::printf("Q27_DFLASH2_BLK pos=%d anch=%u nacc=%d d7=%u %u %u %u %u %u %u ms=%.3f\n",
                                pos, anch, nacc, g_df2_d7[g][0], g_df2_d7[g][1], g_df2_d7[g][2], g_df2_d7[g][3],
                                g_df2_d7[g][4], g_df2_d7[g][5], g_df2_d7[g][6], tp_now() - t2);
    }
    if (g_df2_cond && !g_df2_draft) {   // DFlash2 conditioning diagnostic: run on the real taps of this round
        CK(hipStreamSynchronize(d.s_cap));
        const double t2 = tp_now();
        char derr[128] = {0};
        if (q27_df2_cond_inject(&g_df2c[g], d.nr_taps, 8, nacc, pos, g, df2_reduce_cb, (void*)&C, s, derr, sizeof derr)) {
            std::fprintf(stderr, "Q27_DFLASH2_COND_FAIL %s\n", derr); std::abort();
        }
        CK(hipStreamSynchronize(s));
        if (g == 0) std::printf("Q27_DFLASH2_COND pos=%d nacc=%d ms=%.3f\n", pos, nacc, tp_now() - t2);
    }
    if (g_df2_fwd && nacc > 0 && !g_df2_draft) {   // DFlash2 block forward diagnostic: 8-row forward after the injection
        const int mask_id = q27_env_int("Q27_DFLASH2_MASK", 151643);   // UNPROVEN: incoai config not local; env override
        const unsigned anch = nx[nacc - 1];
        if (g == 0) std::printf("Q27_DFLASH2_EMB anch=%u mask=%d emb0=%04x %04x %04x %04x\n", anch, mask_id,
            emb[(size_t)anch * 5120], emb[(size_t)anch * 5120 + 1], emb[(size_t)anch * 5120 + 2], emb[(size_t)anch * 5120 + 3]);
        std::memcpy((void*)d.nr_ebounce, (const char*)emb + (size_t)anch * 5120 * 2, 5120 * 2);
        std::memcpy((void*)(d.nr_ebounce + 5120), (const char*)emb + (size_t)mask_id * 5120 * 2, 5120 * 2);
        CK(hipMemcpyAsync(g_df2c[g].anchor, d.nr_ebounce, 5120 * 2, hipMemcpyHostToDevice, s));
        CK(hipMemcpyAsync(g_df2c[g].mask, d.nr_ebounce + 5120, 5120 * 2, hipMemcpyHostToDevice, s));
        CK(hipStreamSynchronize(s));
        if (g == 0) { unsigned short chk[4]; CK(hipMemcpy(chk, g_df2c[g].anchor, 8, hipMemcpyDeviceToHost));
            std::printf("Q27_DFLASH2_ANCH_D %04x %04x %04x %04x\n", chk[0], chk[1], chk[2], chk[3]); }
        const double t3 = tp_now();
        float csum = 0.f; char derr[128] = {0};
        if (q27_df2_forward(&g_df2c[g], pos, df2_reduce_cb, (void*)&C, g, s, &csum, derr, sizeof derr)) {
            std::fprintf(stderr, "Q27_DFLASH2_FWD_FAIL %s\n", derr); std::abort();
        }
        CK(hipStreamSynchronize(s));
        if (g == 0) std::printf("Q27_DFLASH2_FWD pos=%d anchor=%u mask=%d ms=%.3f csum=%.4f\n", pos, anch, mask_id, tp_now() - t3, csum);
    }
    if (g_df2_sel && nacc > 0 && !g_df2_draft) {   // candidates + selector walk on the just-produced block
        const double t4 = tp_now();
        unsigned d7[7] = {0,0,0,0,0,0,0}; char derr[128] = {0};
        if (q27_df2_select(&g_df2c[g], &G->lm_head, nx[nacc - 1], g, df2_exch_cb, (void*)&C, s, d7, derr, sizeof derr)) {
            std::fprintf(stderr, "Q27_DFLASH2_SEL_FAIL %s\n", derr); std::abort();
        }
        if (g == 0) std::printf("Q27_DFLASH2_SEL pos=%d anchor=%u ids=%u %u %u %u %u %u %u ms=%.3f\n",
                                pos, nx[nacc - 1], d7[0], d7[1], d7[2], d7[3], d7[4], d7[5], d7[6], tp_now() - t4);
    }
    const int ncommit = nacc + 1;
    for (int j = 0; j < ncommit; ++j) outtok[j] = nx[j];
    d.nr_drafts++; d.nr_hits += nacc;
    for (int j = 1; j <= nacc; ++j) d.nr_acc_hist[j]++;
    if (nacc < K) {   // rows beyond nacc advanced the recurrent state with wrong tokens: restore bank nacc
        const float* Sb = d.nr_Ssnap + (size_t)nacc * n_gdn * sshard;
        const unsigned short* Cb = d.nr_csnap + (size_t)nacc * n_gdn * d.tp_conv;
        for (int j = 0; j < n_gdn; ++j) {
            float* Sp = d.S + d.s_off[j] + d.s_qoff[j];
            CK(hipMemcpyAsync(Sp, Sb + (size_t)j * sshard, sshard * 4, hipMemcpyDeviceToDevice, s));
            CK(hipMemcpyAsync(d.conv + d.conv_off[j], Cb + (size_t)j * d.tp_conv, d.conv_bytes[j], hipMemcpyDeviceToDevice, s));
        }
    }
    // ---- 4. hand-off: h_prev = final residual of the last committed position; when row 1 was
    //         accepted the draft layer still has to consume position pos (row 0) first ----
    // (hand-off)
    for (int j = 0; j < nacc; ++j) {   // positions pos..pos+nacc-1 still owe the draft layer a pass
        CK(hipMemcpyAsync(d.nr_hcatch + (size_t)j * HID, d.hidden_slots + (size_t)j * HID, (size_t)HID * 2, hipMemcpyDeviceToDevice, s));
        d.nr_catch_tok[j] = nx[j]; d.nr_catch_pos[j] = pos + j;
    }
    d.nr_ncatch = nacc;
    CK(hipMemcpyAsync(d.nr_hprev, d.hidden_slots + (size_t)nacc * HID, (size_t)HID * 2, hipMemcpyDeviceToDevice, s));
    lap(6); d.sp_t[7] += tp_now() - sp0;
    if (g == 0 && (d.nr_drafts % 16) == 0) {
        std::printf("Q27_SPEC rounds=%ld accepted=%ld per-round=%.3f", d.nr_drafts, d.nr_hits, (double)d.nr_hits / (double)d.nr_drafts);
        for (int j = 1; j <= K; ++j) std::printf("  P(>=%d)=%.3f", j, (double)d.nr_acc_hist[j] / d.nr_drafts);
        std::printf("\n");
    }
    return ncommit;
}

// ---------------- HOST SAMPLER ----------------
// Mechanism: plain
// C++, std::mt19937, temperature / top-k / top-p over the FULL logit vector. The only change this
// engine needs is the gather, because here the logits are SHARDED -- each card owns 62080 of 248320
// -- so each card writes its shard into a shared pinned buffer and the existing b1 barrier publishes
// all four. Every card then samples from the SAME vector with the SAME seed and therefore agrees on
// the token with no extra communication.
//
// Q27_TEMP=0 keeps the banked greedy path bit-for-bit, so the authority stream stays reproducible.
static float g_temp = 0.f;
static int   g_topk = 0;          // 0 => no top-k
static float g_topp = 1.f;        // 1 => no top-p
static int   g_sampler = 0;       // 1 => the gather+sample path is active
static float* g_logitbuf = nullptr;   // pinned [Q27_VOCAB], shared by all four cards

static unsigned q27_sample_logits(const float* lg, int n, float temp, int topk, float topp,
                                  unsigned long long seq) {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull ^ (seq * 0xBF58476D1CE4E5B9ull));
    if (!(temp > 0.f)) {                       // greedy: exactly the banked argmax behaviour
        unsigned b = 0; float bv = lg[0];
        for (int i = 1; i < n; ++i) if (lg[i] > bv) { bv = lg[i]; b = (unsigned)i; }
        return b;
    }
    int k = (topk > 0 && topk < n) ? topk : n;
    std::vector<int> idx((size_t)n);
    for (int i = 0; i < n; ++i) idx[(size_t)i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [lg](int a, int b) { return lg[a] > lg[b]; });
    std::vector<double> p((size_t)k);
    const double mx = (double)lg[idx[0]];
    double s = 0;
    for (int i = 0; i < k; ++i) {
        p[(size_t)i] = std::exp(((double)lg[idx[(size_t)i]] - mx) / (double)temp);
        s += p[(size_t)i];
    }
    for (int i = 0; i < k; ++i) p[(size_t)i] /= s;
    int keep = k; double acc = 0;
    if (topp > 0.f && topp < 1.f)
        for (int i = 0; i < k; ++i) {
            acc += p[(size_t)i];
            if (acc >= (double)topp) { keep = i + 1; break; }
        }
    double s2 = 0; for (int i = 0; i < keep; ++i) s2 += p[(size_t)i];
    std::uniform_real_distribution<double> U(0.0, s2);
    const double r = U(rng); double c = 0;
    for (int i = 0; i < keep; ++i) { c += p[(size_t)i]; if (r <= c) return (unsigned)idx[(size_t)i]; }
    return (unsigned)idx[(size_t)(keep - 1)];
}

// ---- Q27_SERVE_GROW: grow the per-position prefill scratch IN PLACE -----------------------------
// Called when a request's window exceeds g_pf_cap, instead of refusing it. Only the three
// slot-dependent buffers move (hidden_slots, pend_slots, inv_slots) plus the pinned MTP prompt hall;
// the weights, the int8 mirrors, the target/draft KV, the recurrent state, pos_cur and the tokenizer
// session are untouched, so the same resident conversation continues and only the delta is ever
// prefilled. Capacity goes up a bucket ladder (1.5x, 1024-aligned) and is capped by the TIGHTEST
// card's free memory, keeping a driver safety margin -- the cliff is real and measured: 0.46-0.61 GiB
// free is fast, 0.06 GiB took a 5.9 s prefill to 32 s.
// Shed ONE int8 MLP mirror on THIS CARD (its own ordinal in the ownership order) to make room for
// something else this card needs -- the draft KV here, the slot arena in q27_pfslots_grow. Per-card
// only: the memory is per-card, so there is nothing to coordinate, and the caller may retry.
static int q27_evict_one_mlp_own_card(q27_model_t* m, int ndev, int card) {
    static int s_ord[Q27_MAX_DEVICES];
    // LATCHED PER CARD, AND IT IS THE POINT OF THIS FUNCTION'S CONTRACT: an eviction that does not
    // return memory must not be attempted a second time. Two callers use this -- the draft-KV grow
    // loop and q27_keep_headroom -- and a guard in either one only bounds that caller. Measured
    // 2026-09-15 21:49:15-19: keep_headroom stopped after one round per card, but the draft-KV loop
    // then took ordinal 1 on card 3 on the very next growth attempt, 164 MB more expensive each time,
    // walking every card to `no fast layer left to shed` on a 30 k conversation. Returning 0 here
    // makes the caller fall back (smaller draft KV, speculation off), which is recoverable;
    // stripped residency is not.
    static int s_useless[Q27_MAX_DEVICES];
    if (!m || card < 0 || card >= Q27_MAX_DEVICES || ndev < 1) return 0;
    if (s_useless[card]) return 0;
    const int lpp = Q27_LAYERS;   // Q27_TP3: ordinal loop bounded by actual ownership (quotas may be uneven)
    for (int ord = s_ord[card]; ord < lpp; ++ord) {
        int seen = 0;
        for (int L = 0; L < Q27_LAYERS; ++L) {
            if (!q27_ls_owned(L, card, ndev)) continue;
            if (seen++ != ord) continue;
            size_t fr_before = 0, fr_after = 0, tt = 0;
            if (hipSetDevice(card) == hipSuccess) hipMemGetInfo(&fr_before, &tt);
            if (q27_ls_mlp_evict(m, L, card, ndev)) {
                s_ord[card] = ord + 1;
                if (hipSetDevice(card) == hipSuccess && hipMemGetInfo(&fr_after, &tt) == hipSuccess &&
                    fr_after + 32ull * 1048576ull < fr_before) {
                    s_useless[card] = 1;
                    std::fprintf(stderr, "Q27_SERVE_EVICT card %d layer %d (ordinal %d): converted, "
                                         "but the card LOST %.0f MB (%.2f -> %.2f GiB free) -- latching "
                                         "this card: no further residency will be spent on it\n",
                                 card, L, ord, (double)(fr_before - fr_after) / 1048576.0,
                                 (double)fr_before / 1073741824.0, (double)fr_after / 1073741824.0);
                    std::fflush(stderr);
                    return 0;       // caller must fall back, not try the next ordinal
                }
                std::fprintf(stderr, "Q27_SERVE_EVICT card %d layer %d (ordinal %d): int8 MLP mirror "
                                     "released for the draft KV\n", card, L, ord);
                std::fflush(stderr);
                return 1;
            }
            break;      // this layer is already on NVFP4: move to the next ordinal
        }
        s_ord[card] = ord + 1;
    }
    return 0;
}

// KEEP THE DRIVER'S WORKING ROOM ON THIS CARD. Below the margin the driver evicts continuously and
// every kernel slows by an order of magnitude; the engine's own growth code documents the cliff
// (0.46-0.61 GiB free is fast, 0.06 GiB took a 5.9 s prefill to 32 s). A growing conversation walks
// into it without touching the slot arena at all: KV, draft KV and recurrent state all keep growing
// while fast-layer residency stays where it was. Each card sheds its OWN mirrors (owner card only,
// nothing to coordinate -- the peers never read this card's layer weights), and only until the margin
// is back, so the layers that can be kept are kept.
static void q27_keep_headroom(q27_model_t* m, int ndev, Dev& d) {
    // THE MARGIN IS A DRIVER WORKING-ROOM TARGET, NOT A RESIDENCY TARGET. The fast floor measured on
    // this host is 0.46-0.61 GiB free (see the pfslots comment above: 0.06 GiB took a 5.9 s prefill
    // to 32 s, and every point in 0.46-0.61 was fast). The old 1.00 GiB target is above what a loaded
    // card ever has, so the policy fired on every draft-KV grow and spent int8 MLP residency -- the
    // residency the whole 1554 prefill speedup consists of -- to chase a number it could not reach.
    // 384 MB is BELOW the measured fast band (0.46-0.61 GiB), on purpose. The loaded engine sits at
    // 0.47 GiB free before the draft KV is built, so a 512 MB target evicted one layer per card to
    // chase 33 MB it could not gain -- and each of those four layers is prefill at ~6% of the total
    // (measured: 1,488 tok/s with the four gone, 1,577 with them kept). The cliff this guards is the
    // 0.06 GiB point in the pfslots comment, which is a long way below 384 MB.
    const size_t margin = (size_t)q27_env_int("Q27_SERVE_HEADROOM_MB", 384) * 1048576ull;
    if (hipSetDevice(d.id) != hipSuccess) return;
    for (int i = 0; i < Q27_LAYERS; ++i) {
        size_t fr = 0, tt = 0;
        if (hipMemGetInfo(&fr, &tt) != hipSuccess || fr >= margin) return;
        const size_t before = fr;
        if (!q27_evict_one_mlp_own_card(m, ndev, d.id)) {
            std::fprintf(stderr, "Q27_SERVE_HEADROOM card %d: %.2f GiB free, no fast layer left to "
                                 "shed (margin %.2f GiB)\n", d.id, (double)fr / 1073741824.0,
                         (double)margin / 1073741824.0);
            std::fflush(stderr);
            return;
        }
        hipMemGetInfo(&fr, &tt);
        // AN EVICTION THAT DOES NOT RETURN MEMORY MUST STOP THE LOOP. The swap allocates the layer's
        // NVFP4 authority BEFORE releasing the int8 mirror, and the mirror's buffers are not always
        // separately freeable, so a layer can come out NET NEGATIVE: measured 2026-09-15 21:21:13,
        // "card 1: 0.61 -> 0.45 GiB free" after one eviction -- 164 MB LOST, not gained. The loop
        // then shed all 16 layers per card (1,481 Q27_SERVE_EVICT lines in twelve minutes) without
        // ever reaching the margin: it gave away the fast prefill path (1,577 -> 1,170 tok/s on the
        // same 9.1 k Harness turn) and bought nothing. Stopping here keeps the layers that can be
        // kept and lets the caller fall back to a smaller draft KV / speculation-off, which is
        // recoverable; stripped residency is not.
        if (fr + 32ull * 1048576ull < before) {
            std::fprintf(stderr, "Q27_SERVE_HEADROOM card %d: eviction returned %.0f MB LESS free "
                                 "(%.2f -> %.2f GiB) -- stopping instead of shedding more residency\n",
                         d.id, (double)(before - fr) / 1048576.0,
                         (double)before / 1073741824.0, (double)fr / 1073741824.0);
            std::fflush(stderr);
            return;
        }
        std::fprintf(stderr, "Q27_SERVE_HEADROOM card %d: %.2f -> %.2f GiB free (margin %.2f GiB)\n",
                     d.id, (double)before / 1073741824.0, (double)fr / 1073741824.0,
                     (double)margin / 1073741824.0);
        std::fflush(stderr);
    }
}

// Is EVERY card's draft KV capacity able to hold positions up to `need`? Speculation touches the
// draft cache on all four cards, so the decision has to be the same on all four: one card
// short means either a write past its cache (a GPU fault) or a verify round that never
// completes. Cheap enough to re-evaluate per decode step.
static bool q27_mtp_within_cap(const std::vector<Dev>& D, int ndev, int need) {
    if (g_sr) return D[g_sr_head].mtp_cap >= need;   // Q27_LS_SR: one draft KV, on the head card
    for (int q = 0; q < ndev; ++q) if (D[q].mtp_cap < need) return false;
    return true;
}

static int q27_pfslots_grow(q27_model_t* m, std::vector<Dev>& D, int ndev, int need) {
    // HIP VALIDATES STREAM, EVENT AND MEMORY HANDLES AGAINST THE CALLING THREAD'S CURRENT DEVICE.
    // This helper is run by card 0's thread and allocates on EVERY card, so it leaves that thread on
    // whichever device it touched last; the thread's next hipEventRecord on its OWN stream then fails
    // with "invalid resource handle". Measured 2026-09-15: a 15,616-position chunk step died at the
    // first event of the layer-split ring schedule with a handle that had just been created
    // successfully -- the handle was fine, the current device was not. Restore on every exit path,
    // including the early returns below.
    int q27_saved_dev = 0;
    CK(hipGetDevice(&q27_saved_dev));
    struct DevRestore {
        int d; ~DevRestore() { hipSetDevice(d); }
    } q27_dev_restore{q27_saved_dev};
    int want = g_pf_cap + g_pf_cap / 2;
    if (want < need) want = need;
    want = ((want + 1023) / 1024) * 1024;
    const size_t per_pos = (size_t)Q27_HID * 6;                       // hidden (bf16) + pend (fp32)
    // THE GROWTH MARGIN IS THE FAST/CLIFF BOUNDARY AND IT MUST SIT ABOVE THE CLIFF, NOT BELOW IT.
    // Measured 2026-09-15 on one Harness session: growing the arena to 14,336 slots for an 11,297-token
    // window took a card from 0.47 to 0.32 GiB free and prefill from 1,571 to 809 tok/s; the draft-KV
    // grow then took it to 0.10 GiB and decode collapsed to 10.1 tok/s on a 45-token append. 0.46-0.61
    // GiB is the fast band (see the pfslots comment); 256 MB left the engine growing INTO the cliff.
    // Refusing to grow and appending in arena chunks instead keeps the headroom and the fast path.
    const size_t margin  = (size_t)q27_env_int("Q27_SERVE_MARGIN_MB", 512) * 1048576ull;
    auto cap_at_margin = [&]() {
        int cap = want;
        for (int gg = 0; gg < ndev; ++gg) {
            size_t fr = 0, tt = 0;
            if (hipSetDevice(D[gg].id) != hipSuccess) return g_pf_cap;
            if (hipMemGetInfo(&fr, &tt) != hipSuccess || fr <= margin) return g_pf_cap;
            const int extra = (int)((fr - margin) / per_pos);
            if (g_pf_cap + extra < cap) cap = g_pf_cap + extra;
            cap = (cap / 1024) * 1024;
        }
        return cap;
    };
    int cap = cap_at_margin();
    // BEFORE SPENDING ANY RESIDENCY: can eviction possibly pay for this window? A failed growth
    // attempt is NOT free. The int8 mirrors are released before the new buffers are allocated, and
    // a released layer is never restored -- it stays on its NVFP4 authority. Without this check a
    // hopeless window (a 249,310-token probe against a 16,256-position arena) would strip every fast
    // layer and STILL fail, which is the worst of both: the request is served by the chunk fallback
    // afterwards and runs the whole way on the slow path for no reason.
    // One eviction is worth about 7/16 of a layer's int8 payload per card: the mirror is
    // 3*INTER*HID int8 bytes and the NVFP4 authority restored in its place is 1/2 + 1/16 of that.
    if (cap < want) {
        const long long net_per_layer = (long long)3 * Q27_INTER * Q27_HID * 7 / 16;
        int owned = 0;
        for (int L = 0; L < Q27_LAYERS; ++L) if (q27_ls_owned(L, 0, ndev)) ++owned;
        const long long gain = net_per_layer * owned / (long long)per_pos;
        if ((long long)cap + gain < want) {
            std::fprintf(stderr, "Q27_SERVE_GROW need=%d unreachable: %d owned fast layers are worth at "
                                 "most +%lld positions on top of %d -- refusing without evicting\n",
                         need, owned, gain, cap);
            std::fflush(stderr);
            return 0;
        }
    }
    // Not enough on free memory alone? Buy the rest with fast-layer residency: release int8 MLP
    // mirrors (owner card only -- the other cards' NVFP4 shards were never dropped) until the
    // tightest card can pay for the bucket at the margin. Release BEFORE allocate, always: the new
    // buffers are taken only after the memory to hold them exists.
    int evicted = 0;
    int shed[Q27_MAX_DEVICES] = {0};   // per-card ordinal cursor: how many of ITS OWN layers went
    while (cap < want && q27_env_flag("Q27_SERVE_EVICT", true)) {
        const int prev_cap = cap;
        int did = 0;
        // SHED THE BINDING CARD ONLY. cap_at_margin() takes the MINIMUM over cards, so a fast layer
        // released on a card that is not the minimum cannot buy a single position -- it only destroys
        // residency, and q27_ls_mlp_evict never restores it (the layer stays on its NVFP4 authority
        // for the life of the process). The loop used to spend one layer on EVERY card and only then
        // ask whether the round had bought anything, so a hopeless round cost four layers instead of
        // one. Measured 2026-09-17 11:45:48 on the live serve path: card 0 L0, card 1 L4, card 2 L8,
        // card 3 L12 all released, immediately followed by "round 1 bought no positions (cap still
        // 33792)". Whole-log: 1616 Q27_SERVE_EVICT lines against 24 such wasted rounds. A converted
        // layer is worth ~203 ms of prefill on a 9K window (FREEZE_q27-prefill-1372-20260914.md,
        // linear 8->12 layers = 813 ms), and stripping all 16 is the measured 1557 -> 1026 tok/s
        // (-34%), or 935 -> 703 (-25%) on this V4=1 build. The guard below still stops the NEXT
        // round; this stops the round that is about to be wasted from costing 4x what it must.
        int bind = -1; size_t bind_free = (size_t)-1;
        for (int g = 0; g < ndev; ++g) {
            size_t fr = 0, tt = 0;
            if (hipSetDevice(D[g].id) != hipSuccess) continue;
            if (hipMemGetInfo(&fr, &tt) != hipSuccess) continue;
            if (fr < bind_free) { bind_free = fr; bind = g; }
        }
        if (bind < 0) break;
        {
            const int g = bind;
            // the card's layer of ordinal shed[g] in its OWN ownership order, as at load time
            int ord = 0;
            for (int L = 0; L < Q27_LAYERS; ++L) {
                if (!q27_ls_owned(L, g, ndev)) continue;
                if (ord++ != shed[g]) continue;
                if (q27_ls_mlp_evict(m, L, g, ndev)) {
                    ++did; ++shed[g];
                    std::fprintf(stderr, "Q27_SERVE_EVICT card %d layer %d (ordinal %d, binding card, "
                                         "%.0f MB free): int8 MLP mirror released, layer back on "
                                         "NVFP4 authority\n", g, L, shed[g] - 1,
                                 (double)bind_free / 1048576.0);
                    std::fflush(stderr);
                }
                break;
            }
        }
        if (!did) break;
        ++evicted;
        cap = cap_at_margin();
        if (cap <= prev_cap) {
            // THE ROUND BOUGHT NOTHING, AND THAT WILL NOT CHANGE. q27_ls_mlp_evict rebuilds a
            // layer's NVFP4 authority BEFORE releasing its int8 mirror, so a card without that much
            // free memory gives up nothing at all -- which is what card 3 did on 2026-09-15 (63 MB
            // free after an earlier attempt leaked) while every round kept spending cards 0-2.
            // Without this check the loop keeps spending residency for a window it can never reach.
            std::fprintf(stderr, "Q27_SERVE_GROW need=%d: round %d bought no positions (cap still %d)"
                                 " -- stopping without spending more residency\n", need, evicted, cap);
            std::fflush(stderr);
            return 0;
        }
    }
    if (cap <= g_pf_cap) return 0;
    // A GROW THAT CANNOT REACH THE WINDOW IS NOT A GROW. Measured 2026-09-17 on real agentic traffic:
    // slots 23,552 -> 52,224 (-840 MB/card) toward a 55,280-token window, then Q27_REQ_REFUSED sweep
    // anyway and the window was appended in 8,960-token chunks -- which the OLD arena already held.
    // The server's chunk step never exceeds Q27_SERVE_STEP, so an arena short of `need` buys nothing
    // and costs the headroom the draft KV needs later. Refuse whole; the chunk fallback is exact.
    if (cap < need) {
        std::fprintf(stderr, "Q27_SERVE_GROW need=%d: reachable cap %d is short of the window -- refusing the partial grow "
                             "(slots stay %d; the window is appended in chunks)\n", need, cap, g_pf_cap);
        std::fflush(stderr);
        return 0;
    }
    // Pinned MTP prompt hall first: if the pinned window cannot take the new size we keep the old one
    // and simply stop priming the draft beyond it (a degraded draft is not a correctness problem --
    // every drafted token is still verified against the target).
    size_t hall_old = g_mtp_hall_cap;
    if (g_spec && g_mtp_pp && g_mtp_hall) {
        void* nh = nullptr;
        if (hipHostMalloc(&nh, (size_t)cap * Q27_HID * 2, hipHostMallocDefault) == hipSuccess) {
            hipHostFree(g_mtp_hall); g_mtp_hall = (unsigned short*)nh; g_mtp_hall_cap = (size_t)cap;
        }
    }
    const size_t before = [&]() { size_t fr = 0, tt = 0; hipSetDevice(D[0].id); hipMemGetInfo(&fr, &tt); return fr; }();
    // TRANSACTIONAL GROW. 2026-09-18. This loop used to free each card's sweep scratch and then
    // allocate the replacement, returning 0 the moment any allocation failed -- which left every
    // card after the failure with pend_slots/hidden_slots/inv_slots FREED AND NULL while the engine
    // carried on. The layer-split ring handoff then passed D[src].pend_slots straight into
    // hipMemcpyPeerAsync, and HIP dereferenced the null: SIGSEGV at 0x18, backtrace
    // hipMemcpyPeerAsync <- run_tp's worker lambda. The engine was not out of memory when it died;
    // it died on a pointer a FAILED grow had already thrown away.
    //
    // ALLOCATE EVERYTHING FIRST, FREE THE OLD ONLY ONCE ALL OF IT SUCCEEDED. A grow that cannot be
    // afforded now leaves the arena exactly as it was and the caller falls back to chunked append,
    // which is the documented behaviour ("refuse whole; the chunk fallback is exact").
    {
        std::vector<unsigned short*> nh_(ndev, nullptr);
        std::vector<float*>          np_(ndev, nullptr);
        std::vector<float*>          ni_(ndev, nullptr);
        int ok = 1;
        for (int gg = 0; gg < ndev && ok; ++gg) {
            if (hipSetDevice(D[gg].id) != hipSuccess) { ok = 0; break; }
            if (hipMalloc((void**)&nh_[gg], (size_t)cap * Q27_HID * 2) != hipSuccess) { ok = 0; break; }
            if (hipMalloc((void**)&np_[gg], (size_t)cap * Q27_HID * 4) != hipSuccess) { ok = 0; break; }
            if (hipHostMalloc((void**)&ni_[gg], (size_t)cap * 4, hipHostMallocDefault) != hipSuccess) { ok = 0; break; }
        }
        if (!ok) {   // roll back: nothing old was touched, so the engine is exactly as it was
            for (int gg = 0; gg < ndev; ++gg) {
                if (nh_[gg]) { hipSetDevice(D[gg].id); hipFree(nh_[gg]); }
                if (np_[gg]) { hipSetDevice(D[gg].id); hipFree(np_[gg]); }
                if (ni_[gg]) hipHostFree(ni_[gg]);
            }
            std::fprintf(stderr, "Q27_SERVE_GROW need=%d cap=%d: allocation failed -- ROLLED BACK, arena stays "
                                 "%d slots and the window is appended in chunks (no scratch was freed)\n",
                         need, cap, g_pf_cap);
            std::fflush(stderr);
            return 0;
        }
        for (int gg = 0; gg < ndev; ++gg) {   // every replacement exists: now retire the old
            Dev& d = D[gg];
            hipSetDevice(d.id);
            if (d.hidden_slots) hipFree(d.hidden_slots);
            if (d.pend_slots)   hipFree(d.pend_slots);
            if (d.inv_slots)    hipHostFree(d.inv_slots);
            d.hidden_slots = nh_[gg]; d.pend_slots = np_[gg]; d.inv_slots = ni_[gg];
            hipMemset(d.hidden_slots, 0, (size_t)cap * Q27_HID * 2);
            hipMemset(d.pend_slots,   0, (size_t)cap * Q27_HID * 4);
        }
    }
    size_t after = 0, tt2 = 0; hipSetDevice(D[0].id); hipMemGetInfo(&after, &tt2);
    const int old = g_pf_cap;
    g_pf_cap = cap;
    std::fprintf(stderr, "Q27_SERVE_GROW slots %d -> %d (need %d)  device %+.0f MB/card  "
                         "free_after %.2f GiB  draft_hall %zu -> %zu  pos_cur preserved\n",
                 old, cap, need, (double)((long long)after - (long long)before) / 1048576.0,
                 (double)after / 1073741824.0, hall_old, g_mtp_hall_cap);
    std::fflush(stderr);
    return cap;
}

static int run_tp(q27_model_t* m, int ndev, int ctx, int maxn,
                  const std::vector<unsigned>& prompt, const char* orc) {
    // ARGV TOKEN BUDGET. maxn == -2 means "request 0 is PREFILL-ONLY": the caller booted this engine
    // on a BOUNDED prefix of a longer prompt and will append the remainder (argv cannot carry it),
    // so request 0 must not generate -- an answer generated there is written into the KV and the
    // appended remainder then lands at the wrong positions behind it.
    const bool arg_pf_only = (maxn == -2);
    const int  gen_budget = arg_pf_only ? 0 : (maxn > 0 ? maxn : 0);   // draft-KV growth horizon
    const int nlayer = q27_env_int("Q27_TP_LAYERS", Q27_LAYERS);
    // Size the prefill slots to the prompt (never below the historical 1024, so the MTP verifier's
    // small windows keep their headroom). 30 KB per position per card: 8K = 245 MB, 32K = 1 GB.
    if ((int)prompt.size() > g_pf_cap) g_pf_cap = (int)prompt.size();
    // Q27_SERVE_MAXPROMPT is an INGEST LIMIT. The prefill slots cost g_pf_cap * 30,720 B and every
    // card pays it, so inflating g_pf_cap to the ceiling spends ~1 GiB of the 262K layer-split
    // budget on positions that are never swept -- and on the head card, which also carries the full
    // lm_head, that is the difference between fitting and not. The engine is cold-started on the
    // prompt it will actually serve and the turns that follow are appends, so size the slots to that
    // prompt plus headroom; anything longer still fails closed at the g_pf_cap check.
    if (q27_env_flag("Q27_SERVE", false)) {
        // FLOOR, then cap. Sizing purely from the first prompt is wrong in a server: the engine
        // is cold-started by WHICHEVER request arrives first, and that is routinely a 200-token
        // title generation. The slots then sit at the 1024 floor and the real 9K conversation
        // cannot be swept at all -- it has to throw the engine away and reload 27B of weights.
        // Q27_SERVE_SLOTS is the number of positions we intend to be able to sweep; the prompt
        // raises it further if it is longer. 30,720 B per position per card, so 12288 ~= 360 MB.
        const int floor_ = q27_env_int("Q27_SERVE_SLOTS", q27_env_int("Q27_SERVE_MAXPROMPT", 8192));
        int cap = (int)prompt.size() + 256;
        if (cap < floor_) cap = floor_;
        if (cap > g_pf_cap) g_pf_cap = cap;
    }
    if (nlayer < 1 || nlayer > Q27_LAYERS) { std::fprintf(stderr, "Q27_TP_LAYERS out of range\n"); return 1; }

    // the embedding table stays on the HOST: 2.37 GiB x 4 cards buys nothing when the row is
    // 10 KiB and every card needs the same row.
    size_t emb_bytes = 0; long long esh[Q27_MAX_DIMS];
    int edt = 0, endim = 0;
    const unsigned short* emb = (const unsigned short*)q27_host_ptr(
        m, "model.language_model.embed_tokens.weight", &emb_bytes, &edt, &endim, esh);
    if (!emb || edt != Q27_DT_BF16 || endim != 2 || esh[0] != Q27_VOCAB || esh[1] != Q27_HID) {
        std::fprintf(stderr, "Q27_TP: embed_tokens is missing or has the wrong shape\n"); return 1;
    }
    if (q27_env_flag("Q27_EMB_HOST", true)) {   // resident copy: the mmap page-faulted ~0.25 ms per new token row on the decode path
        void* eb = std::malloc(emb_bytes);
        if (eb) { std::memcpy(eb, emb, emb_bytes); emb = (const unsigned short*)eb; }
    }

    std::vector<Dev> D(ndev);
    // Validate the shard geometry BEFORE a single byte is uploaded. ndev arrives from argv[2]
    // unchecked, and only 1, 2 and 4 have instantiated attention kernels; anything else used to
    // make attention a silent no-op deep inside the run.
    g_tp_profile = q27_env_flag("Q27_TP_PROFILE", false);
    g_exl3_dup_o = q27_env_int("Q27_EXL3_DUP_O", 0);
    g_tp_lead    = q27_env_flag("Q27_TP_LEAD", false);
    if (g_tp_lead) g_tp_profile = true;          // the lead rides on the profile's events
    g_bar_deadline_ms = (double)q27_env_int("Q27_TP_BAR_MS", 5000);
    // Host sampler. Q27_TEMP unset or 0 => the banked device-argmax path, bit-for-bit.
    if (const char* v = std::getenv("Q27_TEMP")) g_temp = (float)std::atof(v);
    if (const char* v = std::getenv("Q27_TOPP")) g_topp = (float)std::atof(v);
    g_topk    = q27_env_int("Q27_TOPK", 0);
    g_sampler = (g_temp > 0.f) ? 1 : 0;
    g_sr = q27_env_flag("Q27_LS_SR", false) ? 1 : 0;
    if (g_sr) {
        g_sr_head = q27_ls_sr_head_card(m);
        g_sr_rb   = q27_env_int("Q27_SR_MLP_RB", 1);
        g_sr_time = q27_env_int("Q27_SR_TIME", 0);   // 1 = per-card block timing, 2 = + per-stage timing inside the layer step
        sr_check_geometry(ndev);
        if (g_sr_head != ndev - 1) { std::fprintf(stderr, "FATAL Q27_LS_SR: head card %d is not the last card\n", g_sr_head); return 4; }
        if (g_sampler) { std::fprintf(stderr, "FATAL Q27_LS_SR: the sequential decode is greedy-only (Q27_TEMP must be unset)\n"); return 4; }
        std::fprintf(stderr, "Q27_LS_SR: single residency, sequential decode, head card %d, prefill MLP %s\n",
                     g_sr_head, g_sr_rb ? "rocBLAS per-row" : "int8/64 wide");
    }
    g_tpsr = q27_env_flag("Q27_TPSR", false) ? 1 : 0;
    if (g_tpsr) {   // Q27_TPSR: tensor-parallel single residency -- every precondition checked at the seam, before any card allocates
        if (q27_env_flag("Q27_LAYER_SPLIT", true)) { std::fprintf(stderr, "FATAL Q27_TPSR needs Q27_LAYER_SPLIT=0 (one residency: the TP shards)\n"); return 4; }
        if (g_sr) { std::fprintf(stderr, "FATAL Q27_TPSR and Q27_LS_SR are exclusive\n"); return 4; }
        if (!q27_env_int("Q27_DEC_I8", 0)) { std::fprintf(stderr, "FATAL Q27_TPSR: decode reads the int8 per-64 projection mirrors, which need Q27_DEC_I8=1\n"); return 4; }
        if (q27_env_flag("Q27_ALIAS_MLP", false) || (q27_env_flag("Q27_DEC_MLP_I8", false) && !q27_env_flag("Q27_DEC_MLP_NV", false))) {
            std::fprintf(stderr, "FATAL Q27_TPSR: Q27_ALIAS_MLP / Q27_DEC_MLP_I8 drop TP MLP shards that only a layer-split copy could replace\n"); return 4; }
        int missing = 0;
        for (int gg = 0; gg < ndev; ++gg) for (int L = 0; L < nlayer; ++L) {
            const q27_layer_t* lay = q27_layer_tp(m, L, gg);
            if (q27_env_flag("Q27_TPSR_NV", false)) {
                if (!lay || !lay->gate.w || !lay->up.w || !lay->down.w) ++missing;
            } else if (!lay || !lay->f8_gate.w || !lay->f8_up.w || !lay->f8_down.w || !lay->f8_gate.bs || !lay->f8_up.bs || !lay->f8_down.bs) ++missing;
        }
        if (missing) { std::fprintf(stderr, "FATAL Q27_TPSR: %d (layer, card) handles lack the FP8 gate/up/down shards (Q27_FP8_MLP + Q27_FP8_MLP_DOWN=1)\n", missing); return 4; }
        std::fprintf(stderr, "Q27_TPSR: tensor-parallel single residency -- FP8 gate/up row shards + FP8 down K-slice, int8/64 projection mirrors, "
                             "quarter KV on every card; batched TP prefill with the MLP on rocBLAS int8; draft prompt-conditioned\n");
        std::fflush(stderr);
    }
    g_tpsr_proj_rb = q27_env_flag("Q27_TPSR_PROJ_RB", false) ? 1 : 0;
    if (g_tpsr_proj_rb && !g_tpsr) { std::fprintf(stderr, "FATAL Q27_TPSR_PROJ_RB needs Q27_TPSR=1\n"); return 4; }
    if (g_sampler) {
        CK(hipHostMalloc((void**)&g_logitbuf, (size_t)Q27_VOCAB * 4, hipHostMallocDefault));
        std::printf("Q27_SAMPLER temp=%.3f topk=%d topp=%.3f  (gather of 4 x %d logits per token)\n",
                    g_temp, g_topk, g_topp, (int)0);
    }
    g_bar_strand      = q27_env_int("Q27_TP_BAR_STRAND", -1);
    // REJECTED on measurement, retained as evidence. Halving the collective payload (fp32 -> bf16,
    // 20480 -> 10240 B each way) measured +0.932 / -1.530 / +0.636 ms, mean +0.013: NEUTRAL.
    // That is the useful result: the collective is LATENCY-bound, not bandwidth-bound, so bytes are
    // not a lever on it. Three levers are now closed -- host-side fixes (neutral), P2P (tie at the
    // 30 KB/card floor), and payload size (neutral). Q27_COLL_BF16=1 re-enables.
    g_coll_bf16       = q27_env_flag("Q27_COLL_BF16", false);
    g_coll_nrbf16     = q27_env_flag("Q27_COLL_NRBF16", true);
    g_nr_p2p          = q27_env_int("Q27_NR_P2P", 1);
    g_mtp_tp          = q27_env_int("Q27_MTP_TP", 0);
    if (g_mtp_tp && g_nr_p2p != 4) { std::fprintf(stderr, "Q27_MTP_TP needs the LL collective (Q27_NR_P2P=4); draft head stays replicated\n"); g_mtp_tp = 0; }
    g_nr_p2p_min      = q27_env_int("Q27_NR_P2P_MIN", 4);
    // REJECTED on measurement, retained as evidence. Multi-block rmsnorm (stage A 5 blocks
    // reducing partial sums, stage B 20 blocks applying + quantizing) measured rmsnorm
    // 1.962 -> 2.584 ms and composed 19.377 -> 20.186. More parallelism did NOT pay, because the
    // single-block form keeps the normalized values IN REGISTERS between the reduction and the
    // quantize, while the split must write the residual stream out in stage A and read the whole
    // vector back in stage B, plus a second launch 128x/token. Q27_RN_SPLIT=1 re-enables.
    g_rn_split        = q27_env_flag("Q27_RN_SPLIT", false);
    // REJECTED on measurement, retained as evidence. Fusing the mixer quantize into the two
    // producers' epilogues (gdn_step phase 4 and attention stage 2) deletes 64 launches/token and
    // 64 write-then-read round trips of a buffer nothing else reads. It still LOSES:
    //   CONTROL   52.274 tok/s  19.130 ms/token  range [18.717, 20.167]
    //   CANDIDATE 50.979 tok/s  19.616 ms/token  range [18.386, 20.031]
    //   DELTA -2.48%, candidate faster in 3/6  -> REJECT
    // The epilogue adds a 16-lane shuffle and per-thread quantize work inside two already-complex
    // kernels, and it does not even save the bf16 write (out[] is still produced). Deleting
    // launches is not free when the launch being deleted is cheaper than what replaces it.
    // Q27_GDN_FQ=1 re-enables.
    g_gdn_fq          = q27_env_flag("Q27_GDN_FQ", false);
    g_coll_prof       = q27_env_flag("Q27_COLL_PROF", false);
    // ON. Twelve of twelve paired wins across two independent six-pair blocks:
    //   first six   CONTROL 51.434  CANDIDATE 55.098  +7.12%  6/6
    //   second six  CONTROL 50.292  CANDIDATE 54.570  +8.51%  6/6
    // Predicted ceiling from the phase breakdown was 129 x 13.8 us = 1.78 ms/token; measured gain
    // ~1.4 ms. Predicted and measured magnitudes agree, which no rejected candidate could claim.
    // Q27_COLL_ZC=0 restores the H2D copy.
    g_coll_zc         = q27_env_flag("Q27_COLL_ZC", true);
    // Q27_COLL_ZCOUT=0 restores the outbound D2H copy.
    // REJECTED ON MEASUREMENT, kept as evidence. Making the projections store their partial
    // straight into pinned host memory deletes the outbound DMA but costs 3.2x overall:
    //   CONTROL 56.346 tok/s 17.747 ms  CANDIDATE 17.351 tok/s 57.633 ms  -69.21%, 0/6 paired.
    // The return path and the outbound path are NOT mirror images. The return is a GPU READ of
    // 20 KB that the consuming kernel pipelines and hides. The outbound is 5120 scattered 4-byte
    // WRITES that each cross PCIe and must fully drain before the host may read: ~309 us/call,
    // about 66 MB/s. Coalescing the epilogue would win maybe 4x and still lose to the 10.6 us DMA.
    g_coll_zcout      = q27_env_flag("Q27_COLL_ZCOUT", false) ? 1 : 0;
    // KEPT 2026-09-10: 6 interleaved pairs at 200 tokens, control 16.852 -> candidate 14.620 ms/token
    // (+15.26%, 6/6, ranges disjoint); tokens identical, collective gate 4/4 with its negative
    // control failing 4/4, zero-card control 4/4. Mechanism: the SDMA D2H of the 20 KB partial
    // cost ~25 us of GPU idle per collective (127/token) as two cross-queue handoffs; the
    // in-stream copy kernel has none. Q27_COLL_KCOPY=0 restores the SDMA path.
    g_coll_kcopy      = q27_env_flag("Q27_COLL_KCOPY", true);
    g_coll_dflag      = q27_env_int("Q27_COLL_DFLAG", 1);   // promoted 09-10 (v67): +2.2% at 18 pairs, all gates
    g_copy_mb         = q27_env_flag("Q27_COPY_MB", false) ? 1 : 0;
    g_nr_lds          = q27_env_flag("Q27_NR_LDS", false) ? 1 : 0;
    g_nr_hg           = q27_env_flag("Q27_NR_HG", false) ? 1 : 0;
    g_nr_fam          = q27_env_int("Q27_NR_FAM", 0);
    g_nr_att1         = q27_env_flag("Q27_NR_ATT1", false) ? 1 : 0;
    g_nr_gdn1         = q27_env_flag("Q27_NR_GDN1", false) ? 1 : 0;
    g_nr_qb           = q27_env_flag("Q27_NR_QB", true) ? 1 : 0;
    g_nr_dup          = q27_env_int("Q27_NR_DUP", 0);
    g_ab_arm          = q27_env_int("Q27_AB_ARM", 0);
    g_ab_file         = std::getenv("Q27_AB_FILE");     // null -> per-request re-read disabled
    g_nr_ch           = q27_env_int("Q27_NR_CH", 4); if (g_nr_ch < 1) g_nr_ch = 1; if (g_nr_ch > 8) g_nr_ch = 8;   // 8 = native 8-row kernels (no chunking)
    g_serve           = q27_env_flag("Q27_SERVE", false);
    g_eos_im_end      = q27_env_flag("Q27_EOS_IM_END", g_serve);
    g_zcout_neg       = q27_env_flag("Q27_ZCOUT_NEGCTL", false) ? 1 : 0;
    // The failed zero-copy-out experiment localized the real cost: not the 20 KB copy, but the
    // synchronization around it. The engine already drains the stream to attribute compute, then
    // tp_allreduce enqueues its copy and drains AGAIN -- two full stream syncs per collective for
    // one dependency. Enqueue the copy behind the producer and one sync covers both.
    // REJECTED: -0.33%, 4/6 paired -- inside the noise floor. Merging the two stream syncs into
    // one saves real host time on every card, and buys nothing, because the D2H phase sits BEFORE
    // barrier 1: time returned to a card there is simply absorbed into the wait for the slowest.
    // This is why the H2D deletion paid and these did not -- it was the only phase AFTER the last
    // barrier, a pure serial tail. Phases before a barrier are absorbed by skew; phases after the
    // last one are critical path. The reduce, between the two barriers, is the remaining target.
    g_coll_fusedrain  = q27_env_flag("Q27_COLL_FUSEDRAIN", false);
    // Q27_FD_NEGCTL=1 tells tp_allreduce the copy was already made while never enqueuing it, so
    // the reduction reads a stale staging buffer. It MUST corrupt. Without this a null A/B result
    // cannot be told apart from a change that never fired.
    g_fd_neg          = q27_env_flag("Q27_FD_NEGCTL", false);
    // Q27_COLL_PF=1: prefetch the four slice regions the host reduce reads, during barrier-1
    // slack. The D2H invalidates exactly those 320 lines and the reduce pays cold-line latency
    // on every one; the wait for the straggler is free time to warm them.
    g_coll_pf         = q27_env_flag("Q27_COLL_PF", false);
    // Q27_FP8X4=1: the loader re-quantized the fp8 projections into NVFP4 twins at load; the
    // dispatcher below and the activation quantizers (permuted order) must agree with it.
    g_fp8x4           = q27_env_flag("Q27_FP8X4", false);
    g_pf_x4w          = q27_env_int("Q27_PF_X4W", 0);
    g_kv_fuse         = q27_env_int("Q27_KV_FUSE", 0);
    g_pf_wide         = q27_env_int("Q27_PF_WIDE", 0);
    g_pf_wide_min     = q27_env_int("Q27_PF_WIDE_MIN", 128);
    g_mlp_tl          = q27_env_int("Q27_MLP_TIMELINE", 0);
    g_mlp_timing      = q27_env_int("Q27_MLP_TIMING", 0);
    g_pf_wide_lag     = q27_env_int("Q27_PF_WIDE_LAG", 6);
    g_pf_wide_slice   = q27_env_int("Q27_PF_WIDE_SLICE", 0);
    g_pf_wide_sdma    = q27_env_int("Q27_PF_WIDE_SDMA", 1);
    g_pf_wide_noffn   = q27_env_int("Q27_PF_WIDE_NOFFN", 0);
    g_pf_wide_kg      = q27_env_int("Q27_PF_WIDE_KG", 8000);
    g_pf_wide_kd      = q27_env_int("Q27_PF_WIDE_KD", 168);
    // FAIL CLOSED, because this arm cannot engage and an inert arm that looks like a measurement is
    // the one failure mode this engine's flags are supposed to make impossible. The wide FFN needs
    // Q27_PF_WIDE_MIN positions per tile (below that its own census says it LOSES to b13); the chunk
    // is capped at the compile-time Q27_PF_CH, and raising that cap does not work: a pristine rung-3
    // tree built with -DQ27_PF_CH=64 dies at the first hipEventRecord of the sweep with SIGBUS, the
    // same signature as the 256-wide build. So the chunk is 32, the wide branch can never be taken,
    // and asking for it must stop the run rather than quietly produce rung-3 numbers under a wide
    // label. The real fix is a tile-wide MLP PATH (its own TSLOT-sized slots and a deferred
    // collective #2), not a wider chunk; see receipt v99.
    // The wide FFN needs a TILE of g_pf_wide_min positions, and the tile is Q27_PF_TSLOT, independent of
    // the chunk width -- so unlike the first attempt this precondition is satisfiable at Q27_PF_CH=32.
    // It is still checked, because a build whose tile is too small would leave the branch unreachable and
    // report rung-3 numbers under a wide label.
    if (g_pf_wide && Q27_PF_TSLOT < g_pf_wide_min) {
        std::fprintf(stderr,
            "FATAL Q27_PF_WIDE=%d needs tiles of >= %d positions but Q27_PF_TSLOT is %d. The wide FFN\n"
            "      branch would never be taken and this run would report rung-3 numbers under a wide\n"
            "      label. Refusing.\n",
            g_pf_wide, g_pf_wide_min, Q27_PF_TSLOT);
        std::abort();
    }
    g_pf_mlp2         = q27_env_int("Q27_PF_MLP2", 1);   // default ON 09-10: 1K prefill 14.39 -> 13.01 s (3 reps), tokens and 3-turn chat identical
    g_pf_mlp_r        = q27_env_int("Q27_PF_MLP_R", 2);
    g_pf_oproj_b      = q27_env_int("Q27_PF_OPROJ_B", 1);
    g_pf_tail_b       = q27_env_int("Q27_PF_TAIL_B", 1);
    g_pf_qkv2         = q27_env_int("Q27_PF_QKV2", 1);
    g_pf_qkv_r        = q27_env_int("Q27_PF_QKV_R", 1);
    g_pf_gdn_fq       = q27_env_int("Q27_PF_GDN_FQ", 0);
    g_pf_pipe         = q27_env_int("Q27_PF_PIPE", 1);
    g_pf_pipe1        = q27_env_int("Q27_PF_PIPE1", 1);
    g_pf_norm_b       = q27_env_int("Q27_PF_NORM_B", 1);
    g_pf_att_b        = q27_env_int("Q27_PF_ATT_B", 1);
    g_pf_noredo       = q27_env_int("Q27_PF_NOREDO", 1);
    g_pf_mk3          = q27_env_int("Q27_PF_MK3", 0);
    g_pf_m3f          = q27_env_int("Q27_PF_M3F", 0);
    g_pf_mlp3         = q27_env_int("Q27_PF_MLP3", 0);
    g_pf_kvp2p        = q27_env_int("Q27_PF_KVP2P", 0);
    g_pf_p2p          = q27_env_int("Q27_PF_P2P", 0);
    g_pf_p2p_rn       = q27_env_int("Q27_PF_P2P_RN", 1);
    g_pf_p2p_mix      = q27_env_int("Q27_PF_P2P_MIX", 1);
    g_ls              = q27_env_flag("Q27_LAYER_SPLIT", true) ? 1 : 0;
    g_ls_q8           = q27_env_flag("Q27_LS_Q8", true);
    g_pf_mlp4         = q27_env_int("Q27_PF_MLP4", 0);
    g_pf_mlp5         = q27_env_int("Q27_PF_MLP5", 1);
    g_pf_mlp6         = q27_env_int("Q27_PF_MLP6", 0);
    g_pf_mlp7         = q27_env_int("Q27_PF_MLP7", 0);
    g_pf_mlp8         = q27_env_int("Q27_PF_MLP8", 0);
    g_pf_mlp9         = q27_env_int("Q27_PF_MLP9", 0);
    g_pf_mlp10        = q27_env_int("Q27_PF_MLP10", 0);
    g_pf_mlp11        = q27_env_int("Q27_PF_MLP11", 0);
    g_pf_mlp12        = q27_env_int("Q27_PF_MLP12", 0);
    g_pf_mlp13        = q27_env_int("Q27_PF_MLP13", 5);   // rung 3 default: v13 gate+up, 5 waves
    g_pf_mk2r2        = q27_env_int("Q27_PF_MK2R2", 1);   // rung 3 default: tile projection with the rebuilt epilogue
    g_pf_mlp14        = q27_env_int("Q27_PF_MLP14", 0);
    g_pf_gdn_scan2    = q27_env_int("Q27_PF_GDN_SCAN2", 1);
    g_pf_hstreams     = q27_env_int("Q27_PF_HSTREAMS", 0);   // rung 3 default: tile recurrence with phase 0 precomputed (window 64: identical, 1K -2.5..3.4%, 8K -2.6%)
    g_pf_mlp14d       = q27_env_int("Q27_PF_MLP14D", 0);
    g_pf_m2f3         = q27_env_int("Q27_PF_M2F3", 1);    // rung 3 default: chunk out_proj with the rebuilt epilogue
    g_pf_qkv3         = q27_env_int("Q27_PF_QKV3", 0);
    g_pf_mlp13d       = q27_env_int("Q27_PF_MLP13D", 6);  // rung 3 default: v13 down, 6 waves
    g_pf_mlp12d       = q27_env_int("Q27_PF_MLP12D", 0);
    g_pf_mlp11d       = q27_env_int("Q27_PF_MLP11D", 0);
    g_pf_mlp10d       = q27_env_int("Q27_PF_MLP10D", 0);
    g_pf_mlp9d        = q27_env_int("Q27_PF_MLP9D", 0);
    g_pf_mlp7d        = q27_env_int("Q27_PF_MLP7D", 0);
    g_pf_mlp7_rpw     = q27_env_int("Q27_PF_MLP7_RPW", 0);
    g_pf_mlp6d        = q27_env_int("Q27_PF_MLP6D", 0);
    g_pf_mk16         = q27_env_int("Q27_PF_MK16", 0);
    g_pf_mk2r         = q27_env_int("Q27_PF_MK2R", 4);
    g_pf_look         = q27_env_int("Q27_PF_LOOK", 1);
    g_pf_kcopy        = q27_env_int("Q27_PF_KCOPY", 0);
    g_pf_chw          = q27_env_int("Q27_PF_CHW", 0);
    g_pf_chw_auto     = q27_env_int("Q27_PF_CHW_AUTO", 256);
    g_pf_chw_auto2    = q27_env_int("Q27_PF_CHW_AUTO2", 4096);
    g_att_hpw         = q27_env_int("Q27_ATT_HPW", 2);
    g_att_pf          = q27_env_int("Q27_ATT_PF", 1);
    g_pf_m2f2         = q27_env_int("Q27_PF_M2F2", 2);
    g_pf_gdn_scan     = q27_env_int("Q27_PF_GDN_SCAN", 1);
    g_pf_att_chunk    = q27_env_int("Q27_PF_ATT_CHUNK", 1);
    // Q27_PREENQ_MLP=1: the MLP chain (gate/up -> swiglu-quant -> down) is enqueued pre-barrier
    // right behind the flag-gated post-norm tail, so the host's post-b2 submissions shrink to
    // the reduce itself. Requires the pe1 tail pre-enqueue to be active (it is, by default).
    // Q27_PREENQ_PROLOGUE=1: stacks on Q27_PREENQ_MLP -- the next layer's q/k/v/attn/quant/
    // o_proj (or GDN equivalent) and its collective-1 D2H copy are enqueued in the current
    // layer's pe2 section, behind the flag-gated input norm. A/B against prologue-post-b2.
    // SHIPPED 2026-09-10: re-measured after the sync-tax result. The flag was banked when the
    // collective was believed to cost only launch overhead; the drain measurement showed every
    // kernel after a drain runs 1.58x slower, and this keeps the prologue in the pre-barrier
    // queue. Decode A/B: +2.20% throughput (16.310 -> 15.960 ms/token), 8/12 paired wins,
    // tokens identical in both arms; profiled stages -6.8% (in_qkv -9.2%, rmsnorm -31%).
    g_preenq_prologue  = q27_env_flag("Q27_PREENQ_PROLOGUE", true);
    g_coll_side        = q27_env_flag("Q27_COLL_SIDE", false);
    // Q27_GDN_DSPLIT=1: the documented GDN restructure -- 12 -> 48 blocks per card by splitting
    // each head's value dims across 4 blocks; only the two scalar norms cross blocks.
    g_dsplit           = q27_env_flag("Q27_GDN_DSPLIT", false);
    if (g_dsplit && q27_env_int("Q27_NDEV", 4) == 3) { std::fprintf(stderr, "Q27_GDN_DSPLIT has no ragged (Q27_TP3) form; disabled\n"); g_dsplit = false; }
    g_preenq_mlp       = q27_env_flag("Q27_PREENQ_MLP", true);   // KEPT: 200-tok blocks +3.20/+1.22%
                                                        // (3/6 each), 500-tok block +4.35% (6/6); tokens identical
    // REJECTED end-to-end: +2.72% then -2.21%, 6/12 paired, medians opposite in sign -- while the
    // instrumentation says it worked (skew 12.38 -> 6.69 us/call, last-to-arrive redistributed from
    // 2/2/39/56% to 27/21/28/24%, b1 waits collapsed). It equalises by LEVELLING DOWN: mean D2H
    // goes 12.05 -> 12.21 us/call, the fast cards slowed to the straggler's pace. Barrier wait is
    // slack, not cost. Only making the LAST card arrive earlier can pay.
    g_tp_affinity     = q27_env_flag("Q27_TP_AFFINITY", false);
    // Q27_NULLRED=1 answers "is reduce+b2 worth attacking at all" before any kernel is written.
    // It deletes both and CORRUPTS the output; it is a ceiling probe, never a candidate.
    g_null_red        = q27_env_flag("Q27_NULLRED", false);
    // Q27_AR1_SKIP=1 answers "is deleting one of the two per-layer all-reduces worth designing
    // a deferred residual for at all" before any kernel is written. It skips the ENTIRE all-reduce
    // #1 rendezvous+reduce and lets the layer run on its rank-local partial (the tail reads the
    // stale reduced vector, which stays finite and in range). b2 is still waited, so the per-token
    // step sync's generation parity holds on all four cards. It CORRUPTS the output by
    // construction; it is a ceiling probe, never a candidate, and must never pass the token gate.
    g_ar1_skip        = q27_env_flag("Q27_AR1_SKIP", false);
    // Q27_RN_HOSTSS=1: residual share folded in pre-barrier, sum of squares fused into the host
    // reduce, and the post-barrier tail becomes a reduction-free elementwise map.
    // KEPT: +3.04% (5/6) then +2.43% (5/6), 10/12 paired, both blocks the same sign and the
    // magnitude the mechanism predicts (rmsnorm 2.653 -> 1.878 ms/token = 4.3% of an 18 ms token).
    g_rn_hostss       = q27_env_flag("Q27_RN_HOSTSS", true);
    g_hostss_neg      = q27_env_flag("Q27_HOSTSS_NEG", false);
    g_rn_epi          = q27_env_flag("Q27_RN_EPI", true);
    g_slow_quant      = q27_env_int("Q27_SLOW_QUANT", 0);
    g_slow_tail       = q27_env_int("Q27_SLOW_TAIL", 0);
    g_slow_mixq       = q27_env_int("Q27_SLOW_MIXQ", 0);
    g_slow_null       = q27_env_int("Q27_SLOW_NULL", 0);
    g_rn_dropy        = q27_env_flag("Q27_RN_DROPY", true);
    g_tail_preenq     = q27_env_flag("Q27_TAIL_PREENQ", true);
    g_host_tail_mask = (unsigned)q27_env_int("Q27_HOST_TAIL_MASK", 0);
    g_wait_kernel    = q27_env_int("Q27_WAIT_KERNEL", 0);   // 1: spin kernel instead of the CP barrier-value wait at the two NR sites
    // REJECTED. Two blocks of six disagree in sign (+3.70% then -1.43%, 8/12 paired) and the
    // mechanism check is decisive: the reduce phase is 6.56/6.35/5.91/6.55 us/call with this off
    // and 6.78/6.29/6.76/5.57 with it on. Allocation mode does not touch the reduction cost, so
    // the first block's win was noise wearing a plausible story. The reduce is cold-line latency
    // (320 lines the DMA just invalidated), not coherence and not host bandwidth.
    g_coll_hostflag   = q27_env_flag("Q27_COLL_COARSE", false) ? hipHostMallocNonCoherent
                                                              : hipHostMallocDefault;
    g_tp_fuse    = q27_env_int("Q27_FUSE", 1);

    // FOUR-CARD CONTRACT, EXPLICIT. Attention instantiates G=1,2,4, but only G=4 has been
    // designed and reasoned end-to-end: the loader shard map (in_proj_qkv as three contiguous
    // ranges), the GDN TP kernel offsets, the KV-cache split and the collective slice were all
    // derived for four shards. 1 and 2 would COMPILE and run, which is exactly the hazard.
    // Accept only what is actually implemented; do not generalise on the strength of the CLI.
    //
    // 2026-09-18 -- Q27_TP_NDEV_ANY=1 OPENS THIS DELIBERATELY. Operator order: run the model on
    // the single 32 GiB card. The refusal above is CAUTION, not a missing implementation -- its
    // own comment says 1 and 2 compile and run. At ndev=1 every quantity this comment worries
    // about degenerates to the identity: each "shard" is the full tensor, every /ndev is /1, the
    // KV split is the whole cache, and a collective over one card is a no-op. That is the least
    // dangerous member of the family, not the most.
    // Still OPT-IN and still loud: an unreviewed ndev=2 really does carry the shard-map risk the
    // comment describes (half-width in_proj_qkv ranges and GDN offsets nobody has checked), so
    // this prints what it is accepting and why, every boot.
    {
        const bool ndev_any = q27_env_flag("Q27_TP_NDEV_ANY", false);
        if (!q27_tp_shard_init(ndev)) { std::fprintf(stderr, "FATAL: no valid TP shard plan for ndev=%d\n", ndev); return 1; }
        for (int gg = 0; gg < ndev; ++gg) { const q27_tp_shard_t* T = q27_tp_shard(gg);
            std::fprintf(stderr, "Q27_TP_SHARD worker %d: q_heads [%d,+%d) kv_heads [%d,+%d) attn_g=%d gdn_key_heads [%d,+%d) value_heads [%d,+%d) mlp [%d,+%d) vocab [%d,+%d) | QL=%d KVL=%d OL=%d QKVL=%d ZL=%d\n",
                         gg, T->q_h0, T->q_hn, T->kv_h0, T->kv_hn, T->attn_g, T->gk_h0, T->gk_hn, T->gv_h0, T->gv_hn, T->mlp_off, T->mlp_len, T->voc_off, T->voc_len, T->QL, T->KVL, T->OL, T->QKVL, T->ZL); }
        if (ndev != 4 && !ndev_any) {
            std::fprintf(stderr, "FATAL run_tp: this TP implementation has a FOUR-CARD contract; "
                                 "ndev=%d is not implemented (attention has G=1,2 kernels but the "
                                 "shard map, GDN offsets and KV split were derived for 4 only). "
                                 "Set Q27_TP_NDEV_ANY=1 to run it anyway.\n", ndev);
            return 4;
        }
        if (ndev != 4) {
            std::fprintf(stderr, "Q27_TP_NDEV_ANY: running the FOUR-CARD TP path at ndev=%d by "
                                 "explicit opt-in.%s\n", ndev,
                         ndev == 1 ? "  ndev=1: all shards are full width, every collective is a"
                                     " no-op -- the degenerate case."
                                   : "  ndev!=1: the in_proj_qkv shard ranges and GDN offsets were"
                                     " NEVER reviewed for this width. Verify output before trusting it.");
            std::fflush(stderr);
        }
    }
    if (!q27_tp_shard_init(ndev)) {   // Q27_TP3: a valid (possibly ragged) shard plan replaces the divisibility rule
        std::fprintf(stderr, "FATAL run_tp: no shard plan for ndev=%d "
                             "(heads %d, kv %d, hidden %d).\n",
                             ndev, Q27_NHEAD, Q27_NKV, Q27_HID);
        return 4;
    }
    // ---- Q27_DEC_I8 (layers): int8 per-64 execution mirrors of every TP shard's fp8 projections, built BEFORE any
    //      context-scaled device allocation (KV cache, GDN snapshot banks, P2P slots) so (a) the transient fp8+int8
    //      peak never competes with the KV cache (the 8K/32K mirror-build OOMs) and (b) once the fp8 arena is
    //      released the banks/slots land in HBM instead of GTT (the p1k K=7 collapse). Q27_DEC_I8_PLAIN=1 routes the
    //      single-row (plain) consumers to the mirrors too; only then is the fp8 arena freed (Q27_KEEP_FP8=1 keeps it,
    //      diagnostic). bad > 0 => fp8 stays resident (the routers fall back per tensor). Draft-head mirrors: below. ----
    // ---- Q27_ALIAS_QKV: address the layer-split residency instead of duplicating it -----------
    // The model is resident TWICE: q27_upload_tp (all 64 layers, each sharded to 1/4 width, for
    // decode) and q27_upload_ls (this card's 16 layers at FULL width, for the fast prefill). That
    // is ~34.6 GiB of weights for a ~20 GB checkpoint, and it is why only ~0.9 GiB/card is left
    // and the int8 MLP mirrors do not fit at 262144.
    //
    // They do not have to be two copies. q27_i8g_t is w=[rows][K] with s=[rows][K/64] -- both
    // row-contiguous, and the group scales are PER ROW, so a TP output-row boundary can never
    // split a quantization group. q_proj/k_proj/v_proj are output-row sharded, so card g's TP
    // shard of a layer it ALSO owns in LS is exactly a row block of the LS full-width mirror:
    // same bytes, same scales, a different starting pointer and row count. Point the decode
    // handle's side-table entry at that view and the TP copy has no reader.
    //
    // Not in_qkv: the loader shards it as THREE contiguous ranges (q|k|v inside one tensor), so
    // its TP shard is not a single row block. It needs three views and is left alone here.
    //
    // RUNS BEFORE THE Q27_DEC_I8 MIRROR BUILD ON PURPOSE. q27_fp8_make_i8g() early-returns when a
    // handle already has a side-table entry, so registering the views first makes the build loop
    // skip exactly the mirrors we are aliasing -- the duplicate is never allocated at all, which
    // is the point. Registering afterwards would overwrite them and save nothing.
    if (g_ls && !g_sr && q27_env_int("Q27_ALIAS_QKV", 0)) {
        int aliased = 0, skipped = 0;
        for (int gg = 0; gg < ndev; ++gg) {
            for (int L = 0; L < nlayer; ++L) {
                if (!Q27_IS_FULL(L)) continue;
                const q27_layer_t* lsl = q27_layer_ls(m, L, gg);
                const q27_layer_t* tpl = q27_layer_tp(m, L, gg);
                if (!lsl || !tpl || !lsl->q8g.w) { ++skipped; continue; }   // gg does not own L in LS
                const q27_i8g_t* src[3] = { &lsl->q8g, &lsl->k8g, &lsl->v8g };
                const q27_fp8_t* dst[3] = { &tpl->q_proj, &tpl->k_proj, &tpl->v_proj };
                for (int i = 0; i < 3; ++i) {
                    if (!src[i]->w || !src[i]->s) { ++skipped; continue; }
                    q27_i8g_t v = *src[i];
                    const q27_tp_shard_t* TS = q27_tp_shard(gg);   // Q27_TP3: the card's head range, not an equal slice
                    const int nr = (i == 0) ? TS->QL : TS->KVL;
                    const int r0 = (i == 0) ? (Q27_QROWS / Q27_NHEAD) * TS->q_h0 : Q27_HDIM * TS->kv_h0;
                    v.w    = v.w + (size_t)r0 * (size_t)v.K;
                    v.s    = v.s + (size_t)r0 * (size_t)(v.K / 64);
                    v.rows = nr;
                    if (dst[i]->rows != nr) { ++skipped; continue; }   // shard geometry must agree
                    aliased += q27_fp8_set_i8g(dst[i], &v);
                }
            }
        }
        // in_z is the one GDN projection the loader shards as a SINGLE row range (q/k/v of in_qkv
        // are three separate ranges and cannot be expressed as one descriptor). Its column-parallel
        // shard is therefore rows [g*Z/ndev, +Z/ndev) of the card's own full-width in_z mirror --
        // the same row-view trick as q/k/v, applied to the other half of the GDN prologue.
        for (int gg = 0; gg < ndev; ++gg) {
            for (int L = 0; L < nlayer; ++L) {
                if (Q27_IS_FULL(L)) continue;
                const q27_layer_t* lsl = q27_layer_ls(m, L, gg);
                const q27_layer_t* tpl = q27_layer_tp(m, L, gg);
                if (!lsl || !tpl || !lsl->iz8g.w || !lsl->iz8g.s) { ++skipped; continue; }
                q27_i8g_t v = lsl->iz8g;
                const q27_tp_shard_t* TS = q27_tp_shard(gg);   // Q27_TP3
                const int nr = TS->ZL, r0 = Q27_GDN_D * TS->gv_h0;
                v.w    = v.w + (size_t)r0 * (size_t)v.K;
                v.s    = v.s + (size_t)r0 * (size_t)(v.K / 64);
                v.rows = nr;
                if (tpl->in_z.rows != nr) { ++skipped; continue; }
                if (v.alpha != tpl->in_z.in_scale * tpl->in_z.wscale) { ++skipped; continue; }
                aliased += q27_fp8_set_i8g(&tpl->in_z, &v);
            }
        }
        std::fprintf(stderr, "Q27_ALIAS_QKV: %d decode handles now address the layer-split copy "
                             "(%d skipped)\n", aliased, skipped);
        std::fflush(stderr);
    }
    // ---- Q27_ALIAS_K: the same trick for the tensors the loader shards along K --------------
    // Tp::fp8_row splits o_proj and out_proj by INPUT columns, so card g's TP shard is
    // [rows][K/ndev] whose rows are a COLUMN RANGE of the card's own full-K layer-split tensor.
    // Same bytes, same per-64 groups (K/ndev is group-aligned), different starting pointer and
    // row stride. Register the view before the mirror build so the duplicate TP mirror is never
    // allocated -- which is the memory this is for. The consumer side is q27_proj_i8g_rows ->
    // q27_k_proj_i8g_snr, which now takes the physical row stride and scale-plane stride.
    if (g_ls && !g_sr && q27_env_int("Q27_ALIAS_K", 0)) {
        int aliased = 0, skipped = 0;
        const int kdbg = q27_env_int("Q27_ALIAS_K_DBG", 0);
        for (int gg = 0; gg < ndev; ++gg) {
            for (int L = 0; L < nlayer; ++L) {
                const q27_layer_t* lsl = q27_layer_ls(m, L, gg);
                const q27_layer_t* tpl = q27_layer_tp(m, L, gg);
                if (!lsl || !tpl) { ++skipped; continue; }
                // (LS mirror, decode fp8 handle) for each K-sharded family. o_proj/out_proj only;
                // in_qkv is output-row sharded as three discrete ranges and needs row views, not
                // a K-view, so it is deliberately not in this list.
                const q27_i8g_t* src[2] = { lsl->o8g.w ? &lsl->o8g : nullptr, lsl->op8g.w ? &lsl->op8g : nullptr };
                const q27_fp8_t* dst[2] = { &tpl->o_proj, &tpl->out_proj };
                for (int i = 0; i < 2; ++i) {
                    if (!src[i] || !src[i]->w || !src[i]->s) { ++skipped; continue; }
                    const int Kfull = src[i]->K;
                    const int Ksh   = dst[i]->K;
                    if (Ksh <= 0 || Kfull <= 0 || (Kfull % (Ksh * ndev))) { ++skipped; continue; }
                    const int k0 = gg * Ksh;                     // this card's contiguous K range
                    if ((k0 % 64) || (Ksh % 64)) { ++skipped; continue; }   // must not split a group
                    q27_i8g_t v = *src[i];
                    v.w    = v.w + (size_t)k0;
                    v.s    = v.s + (size_t)(k0 >> 6);
                    v.ldw  = Kfull;                              // physical row stride of the resident row
                    v.lds  = Kfull >> 6;                         // matching scale-plane row stride
                    v.K    = Ksh;                                // logical stride of the shard
                    if (v.alpha != dst[i]->in_scale * dst[i]->wscale) { ++skipped; continue; }
                    if (kdbg) std::fprintf(stderr, "Q27_ALIAS_K_DBG view card=%d L=%d %s rows=%d K=%d k0=%d "
                                                    "ldw=%d lds=%d w=%p s=%p (tp fp8 w=%p rows=%d K=%d)\n",
                                            gg, L, i ? "out_proj" : "o_proj", v.rows, v.K, k0,
                                            v.ldw, v.lds, (const void*)v.w, (const void*)v.s,
                                            (const void*)dst[i]->w, dst[i]->rows, dst[i]->K);
                    aliased += q27_fp8_set_i8g(dst[i], &v);
                }
            }
        }
        std::fprintf(stderr, "Q27_ALIAS_K: %d decode K-sharded handles now address the layer-split "
                             "copy (%d skipped)\n", aliased, skipped);
        std::fflush(stderr);
    }
    if (g_sr) sr_build_mirrors(m, ndev);   // Q27_LS_SR: per-64 mirrors on the layer-split handles, then the fp8 originals go
    else if (q27_env_int("Q27_DEC_I8", 0) &&
             (q27_env_flag("Q27_SPEC", true) || q27_env_flag("Q27_DEC_I8_PLAIN", false) || g_tpsr)) {
        size_t bytes = 0; int nt = 0, bad = 0;
        for (int gg = 0; gg < ndev; ++gg) {
            CK(hipSetDevice(gg));
            for (int L = 0; L < nlayer; ++L) {
                const q27_layer_t* lay = q27_layer_tp(m, L, gg);
                if (!lay) continue;
                const q27_fp8_t* hs[4] = { nullptr, nullptr, nullptr, nullptr }; int nh = 0;
                if (lay->is_full) { hs[0] = &lay->q_proj; hs[1] = &lay->k_proj; hs[2] = &lay->v_proj; hs[3] = &lay->o_proj; nh = 4; }
                else              { hs[0] = &lay->in_qkv; hs[1] = &lay->in_z; hs[2] = &lay->out_proj; nh = 3; }
                for (int i = 0; i < nh; ++i) { if (q27_fp8_make_i8g(hs[i], 0)) { ++nt; bytes += (size_t)hs[i]->rows * hs[i]->K; } else ++bad; }
            }
            CK(hipDeviceSynchronize());
            q27_storage_owner_ready(gg);
        }
        std::printf("Q27_DEC_I8 on: %d int8 per-64 layer mirrors, %.2f GiB over %d cards (%d failed)\n", nt, (double)bytes / (1024.0 * 1024.0 * 1024.0), ndev, bad);
        for (int gg = 0; gg < ndev; ++gg) { size_t fr = 0, tt = 0; CK(hipSetDevice(gg)); CK(hipMemGetInfo(&fr, &tt));
            std::printf("Q27_VRAM after_i8_mirrors card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0); }
        if (bad == 0 && q27_env_int("Q27_DEC_I8_PLAIN", 0) && !q27_env_flag("Q27_KEEP_FP8", false)) {
            const size_t frb = q27_tp_free_fp8_proj(m, q27_fp8_i8g_rekey);
            std::printf("Q27_DEC_I8: freed %.2f GiB of fp8 layer projections per card (mirrors carry every consumer)\n", (double)frb / (1073741824.0 * (double)ndev));
            q27_fp8_freed_set(1);
            for (int gg = 0; gg < ndev; ++gg) { size_t fr = 0, tt = 0; CK(hipSetDevice(gg)); CK(hipMemGetInfo(&fr, &tt));
                std::printf("Q27_VRAM after_fp8_free card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0); }
        } else if (bad) std::printf("Q27_DEC_I8: %d mirrors failed, fp8 projections kept resident (per-tensor fallback)\n", bad);
        CK(hipSetDevice(0));
    }
    q27_memory_receipt("weights_resident"); MS("T3_devalloc_begin");
    for (int g = 0; g < ndev; ++g) dev_alloc_tp(D[g], g, g_ls ? 1 : ndev, ctx, ndev, g_ls != 0);
    if (g_sr) { g_sr_D = D.data(); g_sr_ndev = ndev; }
    MS("T4_devalloc_done");
    for (int gg = 0; gg < ndev; ++gg) { size_t fr = 0, tt = 0; CK(hipSetDevice(gg)); CK(hipMemGetInfo(&fr, &tt));
        std::printf("Q27_VRAM after_devalloc card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0); }
    // ---- GDN STATE SNAPSHOT/RESTORE COST (commit/recovery, which the objective requires measured
    // separately). 48 of 64 layers are gated-delta with POSITION-SERIAL recurrent state, so a
    // speculative verify that advances T positions and accepts only k must roll the state back to
    // k. Device-to-device is banned on this box (the p2p-poison audit fails the build), so the
    // snapshot goes through host staging -- the same mechanism the collectives already use.
    if (q27_env_flag("Q27_GDN_SNAP", false)) {
        const int n_gdn_p = Q27_LAYERS - Q27_LAYERS / 4;
        for (int g = 0; g < ndev; ++g) {
            Dev& d = D[g];
            const size_t nb = d.tp_s * (size_t)n_gdn_p * 4;
            void* pin = nullptr;
            CK(hipSetDevice(d.id));
            CK(hipHostMalloc(&pin, nb, hipHostMallocDefault));
            CK(hipStreamSynchronize(d.stream));
            const double t0 = tp_now();
            CK(hipMemcpy(pin, d.S, nb, hipMemcpyDeviceToHost));
            const double t1 = tp_now();
            CK(hipMemcpy(d.S, pin, nb, hipMemcpyHostToDevice));
            const double t2 = tp_now();
            std::printf("Q27_GDN_SNAP card %d  %.2f MB  snapshot=%.3f ms  restore=%.3f ms  round=%.3f ms\n",
                        d.id, nb / 1e6, (t1 - t0), (t2 - t1), (t2 - t0));
            std::fflush(stdout);
            CK(hipHostFree(pin));
        }
    }
    g_mtp_graph_on = q27_env_int("Q27_MTP_GRAPH", 0);
    g_spec = q27_env_flag("Q27_SPEC", true) ? 1 : 0;   // shipped default: speculative decoding (greedy); Q27_SPEC=0 = plain
    // K is the number of DRAFT tokens the round proposes; NR = K+1 is the number of trunk rows the
    // verify pass runs. K=0 is legal and is the NON-SPECULATIVE measurement configuration: the draft
    // loop does not execute, spec_round runs exactly one trunk row through run_layer_tp_nr, no
    // acceptance test can fire, and ncommit is 1 by construction -- i.e. a plain target decode step
    // built from the SHIPPED kernels. That matters because the alternative (Q27_SPEC=0) is not a
    // working configuration: the persistent int8 MLP mirror deliberately leaves the NVFP4 TP MLP
    // shards unallocated and the int8 dispatch exists ONLY inside run_layer_tp_nr, so the plain
    // Q27_SPEC=0 path dereferences a null gate (measured: "layer_in L=1 gate=(nil)", fault at the
    // sweep->decode boundary). K=0 reaches back into the shipped code instead of resurrecting it.
    g_spec_k = q27_env_int("Q27_SPEC_K", 1); if (g_spec_k < 0) g_spec_k = 0; if (g_spec_k > 7) g_spec_k = 7;   // K<=7: NR=K+1 verify rows run as chunks of <= 4
    if (q27_env_flag("Q27_MTP_DRAFT", false) || g_spec) {
        // DEFAULT IS THE EMBEDDING-FIRST ORDER. Settled by the golden test, not by taste: feeding
        // the oracle's h_0 and emb(t_1)=8678, embedding-first predicts 198 (= the authority stream's
        // third token) and hidden-first predicts 63191. The earlier 'both orders fail' reading was
        // taken while the NVFP4 sign bug was still present and was therefore meaningless.
        g_mtp_swap = q27_env_flag("Q27_MTP_SWAP", true) ? 1 : 0;
        if (const char* v = std::getenv("Q27_MTP_IS")) g_mtp_is = (float)std::atof(v);
        g_mtp_po = q27_env_int("Q27_MTP_PO", 1);
        g_mtp_chain = q27_env_int("Q27_MTP_CHAIN", 1);
        if (g_mtp_chain < 1) g_mtp_chain = 1;
        if (g_mtp_chain > 8) g_mtp_chain = 8;
        g_mtp_dbg = q27_env_flag("Q27_MTP_DBG", false) ? 1 : 0;
        g_lmref = q27_globals(m, ndev - 1)->lm_head;
        for (int gg = 0; gg < ndev; ++gg) { size_t fr = 0, tt = 0; CK(hipSetDevice(gg)); CK(hipMemGetInfo(&fr, &tt));
            std::printf("Q27_VRAM pre_mtp_init card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0); }
        // The K=0 target-only row decoder never consumes draft weights. Keep
        // its existing scratch/protocol setup below, but avoid loading and
        // converting the replicated MTP head when every draft consumer is off.
        const bool need_draft_weights = g_spec_k > 0 || q27_env_flag("Q27_MTP_DRAFT", false)
            || q27_env_flag("Q27_MTP_PP", true) || g_sr
            || std::getenv("Q27_MTP_GOLD_H") || q27_env_flag("Q27_MTP_KERNBENCH", false);
        if (!need_draft_weights) std::fprintf(stderr,"Q27_MTP weights skipped: target-only K=0, draft and prompt-head disabled\n");
        if (need_draft_weights && mtp_init(m, ndev)) {
            if (std::getenv("Q27_SPEC") || std::getenv("Q27_MTP_DRAFT")) { std::fprintf(stderr, "Q27_MTP_INIT_FAIL\n"); return 1; }
            std::fprintf(stderr, "Q27_MTP_INIT_FAIL: speculative decoding disabled, plain decode\n"); g_spec = 0;
        }
        g_mtp_on = q27_env_flag("Q27_MTP_DRAFT", false) ? 1 : 0;
        g_mtp_pp = q27_env_flag("Q27_MTP_PP", true) ? 1 : 0;
        if (g_spec && g_mtp_pp) { g_mtp_hall_cap = (size_t)g_pf_cap; CK(hipHostMalloc((void**)&g_mtp_hall, g_mtp_hall_cap * Q27_HID * 2, hipHostMallocDefault)); }
        if (g_sr) {   // Q27_LS_SR: full-width banks for this card's OWN gdn layers, the decode scratch and the prefill mirror ring
            for (int gg = 0; gg < ndev; ++gg) sr_alloc_card(D[gg], ndev, g_spec_k);
            CK(hipSetDevice(0));
        } else
        if (g_spec && g_spec_k > 0) {   // GDN state / conv snapshot banks (K banks per card) reserved now, not at the first round when VRAM is at its peak (32K OOM)
            // K=0 needs no banks: the snapshot is written only for rows that a later acceptance test
            // could reject (c0 + cn < NR), and NR = 1 means there is no such row. Sizing this at K=0
            // would be a zero-byte hipMalloc.
            const int n_gdn = Q27_LAYERS - Q27_LAYERS / 4;
            for (int gg = 0; gg < ndev; ++gg) { CK(hipSetDevice(D[gg].id));
                const size_t sshard = (size_t)q27_tp_shard(gg)->VHL * Q27_GDN_D * Q27_GDN_D;
                CK(hipMalloc((void**)&D[gg].nr_Ssnap, (size_t)g_spec_k * n_gdn * sshard * 4));
                CK(hipMalloc((void**)&D[gg].nr_csnap, (size_t)g_spec_k * n_gdn * D[gg].tp_conv * 2)); }
        }
        if (g_spec && !(g_rn_hostss && g_tail_preenq && g_coll_kcopy && g_coll_dflag && g_coll_zc && !g_coll_bf16 && !g_coll_zcout)) {
            std::fprintf(stderr, "Q27_SPEC requires the shipped collective protocol (RN_HOSTSS, TAIL_PREENQ, COLL_KCOPY, COLL_DFLAG, COLL_ZC; no COLL_BF16 / COLL_ZCOUT)\n");
            return 1;
        }
        // AFTER the upload: the gold head test needs the handles and the host embed table.
        if (const char* gh = std::getenv("Q27_MTP_GOLD_H")) {
            const char* gv = std::getenv("Q27_MTP_GOLD_TOK");
            g_gold_mode = q27_env_int("Q27_MTP_GOLD_MODE", 0);
            mtp_gold(m, D, ndev, gh, gv ? (unsigned)strtoul(gv, nullptr, 10) : 0u);
            return 0;   // probe mode
        }
    }
    // WHAT SPECULATION ACTUALLY RESOLVED TO. Q27_SPEC_K=0 (no draft, one trunk row per round) is
    // the non-speculative measurement configuration, and it looks identical to K=2 in every other
    // log line, so the engine states it once at start-up rather than leaving it to be inferred.
    std::fprintf(stderr, "Q27_SPECCFG spec=%d K=%d verify_rows=%d draft=%d mtp_pp=%d paged=%d\n",
                 g_spec, g_spec_k, g_spec ? g_spec_k + 1 : 1, g_mtp_on, g_mtp_pp,
                 q27_env_int("Q27_MTP_PAGED", 1));
    std::fflush(stderr);
    {   // E2 (2026-09-17): the boot names the binary it is running. Reported BEFORE the first
        // request so every arm of an A/B is attributable from the log alone. See q27_buildstamp.cpp.
        // nr / nr_p2p / nr_p2p_min come from the PARSED GLOBALS (set ~500 lines above, same
        // function), not from a re-read of the environment: the receipt has to name what the
        // engine will actually execute with. nr = the speculative verify row count, which is what
        // the collective's transport threshold is compared against -- printing it next to
        // nr_p2p_min makes the transport decision readable off the boot line alone.
        char bcfg[288];
        std::snprintf(bcfg, sizeof bcfg,
                      "v4=%d pf_ch=%d pf_tslot=%d ls=%d kvplace=%d dec_mlp_nv=%d "
                      "nr=%d nr_p2p=%d nr_p2p_min=%d coll_nrbf16=%d tpsr=%d",
                      (int)Q27_V4, (int)Q27_PF_CH, (int)Q27_PF_TSLOT,
                      q27_env_int("Q27_LAYER_SPLIT", 0), q27_env_int("Q27_KVPLACE", 0),
                      q27_env_int("Q27_DEC_MLP_NV", 0),
                      g_spec ? g_spec_k + 1 : 1, g_nr_p2p, g_nr_p2p_min, (int)g_coll_nrbf16, g_tpsr);
        q27_bootstamp(g_spec, g_spec_k, ctx, ndev, bcfg);
    }

    TpColl C; C.ndev = ndev; C.n = Q27_HID; C.b1.init(ndev); C.b2.init(ndev);
    for (int g = 0; g < Q27_MAX_DEVICES; ++g) C.sr_pub[g].store(0);   // Q27_LS_SR hop counters (atomics do not self-initialise)
    // ALLOCATION MODE IS PART OF THE ALGORITHM HERE. ROCm's default host allocation is
    // fine-grained coherent, which keeps the GPU coherent with no explicit flush and makes CPU
    // reads slow -- and this buffer's whole purpose is a GPU write followed by a CPU read. The
    // engine already drains the stream before the host touches it, so coarse-grained
    // (non-coherent) memory is legal and is cacheable on the host side.
    CK(hipHostMalloc((void**)&C.flag_h, (size_t)Q27_MAX_DEVICES * 4, hipHostMallocDefault));
    CK(hipHostGetDevicePointer((void**)&C.flag_d, C.flag_h, 0));
    for (int k = 0; k < Q27_MAX_DEVICES; ++k) C.flag_h[k] = 0;
    CK(hipHostMalloc((void**)&C.inv_h, (size_t)Q27_MAX_DEVICES * 4, hipHostMallocDefault));
    CK(hipHostMalloc((void**)&C.inv_hb, (size_t)Q27_MAX_DEVICES * TpColl::BC * 4, hipHostMallocDefault));
    for (int k = 0; k < Q27_MAX_DEVICES; ++k) { CK(hipStreamCreateWithFlags(&C.side[k], hipStreamNonBlocking)); CK(hipEventCreate(&C.ev_side[k])); }
    for (int k = 0; k < Q27_MAX_DEVICES * TpColl::BC; ++k) C.inv_hb[k] = 1.0f;
    CK(hipHostGetDevicePointer((void**)&C.inv_d, C.inv_h, 0));
    CK(hipHostMalloc((void**)&C.invN_h, (size_t)Q27_MAX_DEVICES * 8 * 4, hipHostMallocDefault));
    for (int k = 0; k < Q27_MAX_DEVICES * 8; ++k) C.invN_h[k] = 1.0f;
    CK(hipHostGetDevicePointer((void**)&C.invN_d, C.invN_h, 0));
    for (int k = 0; k < Q27_MAX_DEVICES; ++k) C.inv_h[k] = 1.0f;
    if (g_tail_preenq) {
        int cap = 0;
        CK(hipDeviceGetAttribute(&cap, hipDeviceAttributeCanUseStreamWaitValue, D[0].id));
        g_preenq_ok = (cap != 0);
        if (!g_preenq_ok)
            std::fprintf(stderr, "Q27_TAIL_PREENQ: device reports no hipStreamWaitValue32 "
                                 "support; falling back to host-submitted tail\n");
        for (int g = 0; g < ndev && g_preenq_ok; ++g) {
            CK(hipSetDevice(D[g].id));
            CK(hipEventCreateWithFlags(&C.ev_copy[g], hipEventDisableTiming));
            // Initialize the stream-wait path with an already-satisfied value,
            // before workers can enqueue an unsatisfied host-published flag.
            CK(hipStreamWaitValue32(D[g].stream, (void*)(C.flag_d + g), 0,
                                    hipStreamWaitValueEq, 0xffffffffu));
            CK(hipStreamSynchronize(D[g].stream));
        }
    }

    CK(hipHostMalloc((void**)&C.acc, (size_t)C.n * TpColl::BC * 4, g_coll_hostflag));
    CK(hipHostMalloc((void**)&C.acc2, (size_t)C.n * TpColl::BC * 4, g_coll_hostflag));
    if (g_pf_wide) CK(hipHostMalloc((void**)&C.acc2w, (size_t)C.n * Q27_PF_TSLOT * 4, g_coll_hostflag));
    for (int q = 0; q < TpColl::NB; ++q) {
        CK(hipHostMalloc((void**)&C.accb[q], (size_t)C.n * TpColl::BC * 4, g_coll_hostflag));
        CK(hipHostGetDevicePointer((void**)&C.accb_d[q], C.accb[q], 0));
        CK(hipHostMalloc((void**)&C.inv_hbb[q], (size_t)Q27_MAX_DEVICES * TpColl::BC * 4, hipHostMallocDefault));
        for (int k = 0; k < Q27_MAX_DEVICES * TpColl::BC; ++k) C.inv_hbb[q][k] = 1.0f;
    }
    CK(hipHostGetDevicePointer((void**)&g_coll_acc, C.acc, 0));
    CK(hipHostMalloc((void**)&C.done_h, (size_t)Q27_MAX_DEVICES * 4, hipHostMallocDefault));
    std::memset(C.done_h, 0, (size_t)Q27_MAX_DEVICES * 4);
    for (int g = 0; g < ndev; ++g)
    {   CK(hipHostMalloc((void**)&C.hp[g], (size_t)C.n * TpColl::BC * 4, g_coll_hostflag));
        CK(hipHostMalloc((void**)&C.hp2[g], (size_t)C.n * TpColl::BC * 4, g_coll_hostflag));
        CK(hipSetDevice(D[g].id)); CK(hipEventCreateWithFlags(&C.ev_c2[g], hipEventDisableTiming));
        if (g_pf_wide || g_mlp_timing) for (int q = 0; q < 2; ++q) {
            CK(hipEventCreateWithFlags(&C.ev_b0[q][g], 0));
            CK(hipEventCreateWithFlags(&C.ev_b1[q][g], 0));
        }
        if (g_mlp_tl) for (int q = 0; q < 2; ++q) CK(hipEventCreateWithFlags(&C.ev_pre[q][g], 0));
        if (g_mlp_tl) {
            // CONTROLLED VALIDATION OF THE MEASUREMENT ITSELF: two events recorded around a known sleep
            // on the compute stream, then queried. If this pair cannot be measured, every bubble number
            // downstream is meaningless, so say so instead of reporting 0.
            hipEvent_t a = nullptr, b = nullptr;
            CK(hipEventCreateWithFlags(&a, 0)); CK(hipEventCreateWithFlags(&b, 0));
            CK(hipEventRecord(a, D[g].stream));
            CK(hipStreamSynchronize(D[g].stream));      // ~0 GPU work, so elapsed should be ~0 but VALID
            CK(hipMemcpyAsync(C.hp2 ? C.hp2[g] : nullptr, C.hp2 ? C.hp2[g] : nullptr, 0, hipMemcpyDeviceToDevice, D[g].stream));
            CK(hipEventRecord(b, D[g].stream));
            CK(hipEventSynchronize(b));
            float ms = -1.f;
            C.ev_pair_ok = (hipEventElapsedTime(&ms, a, b) == hipSuccess) && (ms >= 0.f);
            hipEventDestroy(a); hipEventDestroy(b);
        }
        if (g_pf_wide) {
            CK(hipHostMalloc((void**)&C.hp2w[g], (size_t)C.n * Q27_PF_TSLOT * 4, g_coll_hostflag));
            // the device alias must be taken with card g current (a mapping is per-device)
            CK(hipHostGetDevicePointer((void**)&C.hp2w_d[g], C.hp2w[g], 0));
            CK(hipEventCreateWithFlags(&C.ev_w0[g], 0));
            CK(hipEventCreateWithFlags(&C.ev_w1[g], 0));
        }
        for (int q = 0; q < TpColl::NB; ++q) {
            CK(hipHostMalloc((void**)&C.hpb[q][g], (size_t)C.n * TpColl::BC * 4, g_coll_hostflag));
            CK(hipEventCreateWithFlags(&C.ev1[q][g], hipEventDisableTiming));
            // Q27_PF_P2P: the producer writes the ring slot directly; the fused collective reads
            // own + peers through peer access (enabled below, behind the runtime self-test).
            CK(hipSetDevice(D[g].id));
            CK(hipMalloc((void**)&C.part_d[q][g], (size_t)C.n * TpColl::BC * 4));
            CK(hipEventCreateWithFlags(&C.ev_rd[q][g], hipEventDisableTiming));
            CK(hipEventRecord(C.ev_rd[q][g], D[g].stream));   // prime: slot is free at start
        }
        CK(hipSetDevice(D[g].id));
        C.mixp[g] = D[g].mixer_slots;                         // peer pointer for the mixer collective
        for (int i = 0; i < 3; ++i) {
            CK(hipEventCreateWithFlags(&C.ev_mix[i][g], hipEventDisableTiming));
            CK(hipEventCreateWithFlags(&C.ev_mix_rd[i][g], hipEventDisableTiming));
            CK(hipEventRecord(C.ev_mix_rd[i][g], D[g].stream));  // prime: mixer_slots free at start
        }
        // Mirror of the return-path fix: the host reduction is the next consumer of this buffer,
        // so the producer can store into it directly and the D2H DMA disappears. Take the device
        // pointer with card g current -- a mapping is per-device, and a pointer taken under the
        // wrong context is a silently wrong address, not an error.
        CK(hipSetDevice(D[g].id));
        CK(hipHostGetDevicePointer((void**)&g_coll_part[g], C.hp[g], 0));
        CK(hipMalloc((void**)&C.cp_cnt[g], 64)); CK(hipMemset(C.cp_cnt[g], 0, 64));   // multi-block staging copy counter
        CK(hipHostGetDevicePointer((void**)&C.done_d[g], C.done_h, 0));
        CK(hipHostGetDevicePointer((void**)&C.hp2_d[g], C.hp2[g], 0));
        for (int q = 0; q < TpColl::NB; ++q) CK(hipHostGetDevicePointer((void**)&C.hpb_d[q][g], C.hpb[q][g], 0)); }

    // ---- ndev < 4: PAD THE UNUSED COLLECTIVE SLOTS WITH ZEROS, DO NOT LEAVE THEM NULL. 2026-09-18.
    // FIVE host-reduce sites (tp_reduce_bx and four siblings around :1285/:1410/:1503/:1565) are
    // hand-unrolled over hp[0..3] and read all four UNCONDITIONALLY. At ndev=2 slots 2 and 3 were
    // null and the adds dereferenced them: both card threads faulted at address 0 and 0x2800 inside
    // tp_reduce_bx (dmesg, 2026-09-18 12:42:30). That is what "the shard map was derived for 4 only"
    // amounted to in practice -- not a shard map at all, an unrolled loop.
    // Zero is the identity for this reduction, so padding is numerically EXACT: every unrolled fast
    // path keeps running at every width and no reduce site has to change. The ndev==4 arm allocates
    // nothing here and is untouched.
    for (int k = ndev; k < 4; ++k) {
        const size_t nb = (size_t)C.n * TpColl::BC * 4;
        CK(hipHostMalloc((void**)&C.hp[k],  nb, g_coll_hostflag)); std::memset(C.hp[k],  0, nb);
        CK(hipHostMalloc((void**)&C.hp2[k], nb, g_coll_hostflag)); std::memset(C.hp2[k], 0, nb);
        for (int q = 0; q < TpColl::NB; ++q) {
            CK(hipHostMalloc((void**)&C.hpb[q][k], nb, g_coll_hostflag)); std::memset(C.hpb[q][k], 0, nb);
        }
        if (g_pf_wide) {
            const size_t wb = (size_t)C.n * Q27_PF_TSLOT * 4;
            CK(hipHostMalloc((void**)&C.hp2w[k], wb, g_coll_hostflag)); std::memset(C.hp2w[k], 0, wb);
        }
    }
    if (ndev < 4) {
        std::fprintf(stderr, "Q27_COLL_PAD: ndev=%d -- collective slots %d..3 zero-filled "
                             "(the host reduces are unrolled over hp[0..3])\n", ndev, ndev);
        std::fflush(stderr);
    }

    // ---- Q27_PF_P2P: peer access behind a runtime self-test. A kernel writes a marker into every
    // peer's buffer; the host reads each back and verifies. Any failure (or a missing wide producer)
    // drops P2P and the sweep runs the untouched host collective -- both the producer routing and
    // the collective site revert together, so no stale-device-ring read is possible.
    // ---- Q27_NR_P2P: peer access for the row-batched collective (all ordered pairs, runtime self-test), the per-card
    //      partial slots / reduced rows / sumsq partials, and the cross-card events (rd primed: every slot starts free) ----
    // ---- Q27_DEC_I8 (draft head): int8 per-64 mirrors of the MTP head's q/k/v/o (uploaded by mtp_init above); the
    //      head fp8 tensors are individual allocations, freed one by one under the same gate as the layer arena. ----
    if (g_spec && q27_env_int("Q27_DEC_I8", 0)) {
        size_t bytes = 0; int nt = 0, bad = 0;
        for (int gg = 0; gg < ndev; ++gg) {
            CK(hipSetDevice(D[gg].id));
            const q27_fp8_t* hh[4] = { &g_mtp[gg].W.q, &g_mtp[gg].W.k, &g_mtp[gg].W.v, &g_mtp[gg].W.o };
            for (int i = 0; i < 4; ++i) { if (!hh[i]->w) continue; if (q27_fp8_make_i8g(hh[i], D[gg].stream)) { ++nt; bytes += (size_t)hh[i]->rows * hh[i]->K; } else ++bad; }
            CK(hipStreamSynchronize(D[gg].stream));
        }
        std::printf("Q27_DEC_I8 on: %d int8 per-64 draft-head mirrors, %.3f GiB over %d cards (%d failed)\n", nt, (double)bytes / (1024.0 * 1024.0 * 1024.0), ndev, bad);
        if (bad == 0 && nt > 0 && q27_env_int("Q27_DEC_I8_PLAIN", 0) && !q27_env_flag("Q27_KEEP_FP8", false)) {
            size_t frb = 0;
            for (int gg = 0; gg < ndev; ++gg) {
                CK(hipSetDevice(D[gg].id));
                q27_fp8_t* hw[4] = { &g_mtp[gg].W.q, &g_mtp[gg].W.k, &g_mtp[gg].W.v, &g_mtp[gg].W.o };
                // k and v stay resident under Q27_MTP_PP: mtp_prompt_init() builds its concatenated k|v int8 prompt mirror
                // from the fp8 head at prime time (inside the worker threads, after this point).
                for (int i = 0; i < 4; ++i) { if (!hw[i]->w) continue; if (g_mtp_pp && (i == 1 || i == 2)) continue;
                    frb += (size_t)hw[i]->rows * hw[i]->K; CK(hipFree((void*)hw[i]->w));
                    const unsigned char* sent = (const unsigned char*)(uintptr_t)(0xDEAD10000000ULL + (uint64_t)(gg * 4 + i + 1) * 4096ULL);   // unmapped: loud fault
                    q27_fp8_i8g_rekey((const void*)hw[i]->w, (const void*)sent); hw[i]->w = sent; }
            }
            std::printf("Q27_DEC_I8: freed %.3f GiB of draft-head fp8 projections (all cards%s)\n", (double)frb / 1073741824.0, g_mtp_pp ? "; k/v kept for Q27_MTP_PP" : "");
        }
        for (int gg = 0; gg < ndev; ++gg) { size_t fr = 0, tt = 0; CK(hipSetDevice(D[gg].id)); CK(hipMemGetInfo(&fr, &tt));
            std::printf("Q27_VRAM after_head_mirrors card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0); }
        CK(hipSetDevice(D[0].id));
    }
    if (g_spec && g_nr_p2p) {
        int ok = (ndev == 4 || ndev == 3) ? 1 : 0;
        for (int i = 0; i < ndev && ok; ++i) for (int j = 0; j < ndev; ++j) if (i != j) {
            int can = 0; CK(hipDeviceCanAccessPeer(&can, D[i].id, D[j].id)); if (!can) ok = 0; }
        for (int i = 0; i < ndev && ok; ++i) { CK(hipSetDevice(D[i].id));
            for (int j = 0; j < ndev; ++j) if (i != j) { hipError_t pe = hipDeviceEnablePeerAccess(D[j].id, 0);
                if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) ok = 0; (void)hipGetLastError(); } }
        if (ok) {
            float* probe[Q27_MAX_DEVICES] = {nullptr};
            for (int i = 0; i < ndev; ++i) { CK(hipSetDevice(D[i].id)); CK(hipMalloc((void**)&probe[i], 16)); }
            for (int i = 0; i < ndev && ok; ++i) for (int j = 0; j < ndev; ++j) if (i != j) {
                CK(hipSetDevice(D[i].id));
                const float want = 2000.0f + (float)i * 10.0f + (float)j;
                q27_p2p_probe(probe[j], want, D[i].stream);
                CK(hipStreamSynchronize(D[i].stream));
                float got = -1.0f; CK(hipSetDevice(D[j].id)); CK(hipMemcpy(&got, probe[j], 4, hipMemcpyDeviceToHost));
                if (got != want) { ok = 0; std::fprintf(stderr, "Q27_NR_P2P self-test FAIL %d->%d got %g want %g\n", i, j, (double)got, (double)want); }
            }
            for (int i = 0; i < ndev; ++i) { CK(hipSetDevice(D[i].id)); CK(hipFree(probe[i])); }
        }
        if (ok) {
            for (int gg = 0; gg < ndev; ++gg) g_devid_tab[gg] = D[gg].id;
            g_nrp_ndev = ndev;
            if (ndev == 3) {   // pull mode reads four partial slots: the fourth is a zero pad on card 0 (never written)
                CK(hipSetDevice(D[0].id));
                for (int q = 0; q < 2; ++q) { CK(hipMalloc((void**)&C.nrp_part[q][3], (size_t)8 * Q27_HID * 4)); CK(hipMemset(C.nrp_part[q][3], 0, (size_t)8 * Q27_HID * 4)); }
            }
            for (int gg = 0; gg < ndev; ++gg) {
                CK(hipSetDevice(D[gg].id));
                for (int q = 0; q < 2; ++q) {
                    CK(hipMalloc((void**)&C.nrp_part[q][gg], (size_t)8 * Q27_HID * 4));
                    CK(hipMemset(C.nrp_part[q][gg], 0, (size_t)8 * Q27_HID * 4));
                    CK(hipEventCreateWithFlags(&C.ev_nrp_done[q][gg], hipEventDisableTiming));
                    CK(hipEventCreateWithFlags(&C.ev_nrp_rd[q][gg], hipEventDisableTiming));
                    CK(hipEventRecord(C.ev_nrp_rd[q][gg], D[gg].stream));   // prime: the slot is free at start
                }
                CK(hipMalloc((void**)&C.nrp_red[gg], (size_t)8 * Q27_HID * 4));
                CK(hipMalloc((void**)&C.nrp_ssp[gg], (size_t)8 * 5 * 4));
                for (int q = 0; q < 2; ++q) for (int k = 0; k < ndev; ++k) if (k != gg) {
                    CK(hipMalloc((void**)&C.nrp_recv[q][gg][k], (size_t)8 * Q27_HID * 2));
                    CK(hipMemset(C.nrp_recv[q][gg][k], 0, (size_t)8 * Q27_HID * 2)); }
                for (int q = 0; q < 2; ++q) CK(hipMalloc((void**)&C.nrp_stage[q][gg], (size_t)8 * Q27_HID * 2));
            }
            if (g_nr_p2p == 4) {
                // LL slots: 8 rows x HID/2 words x 8 B per (owner, source, slot), UNCACHED if the runtime offers it, else fine-grained.
                // Both map MTYPE UC on gfx9 (amdgpu get_pte_flags: coherent -> UC), for the owner's spin-reads AND the peers' stores.
                const size_t llb = (size_t)8 * (Q27_HID / 2) * 8;
                const char* llkind = "uncached";
                int llok = 1;
                for (int gg = 0; gg < ndev && llok; ++gg) {
                    CK(hipSetDevice(D[gg].id));
                    for (int q = 0; q < Q27_LL_SLOTS && llok; ++q) for (int k = 0; k < ndev && llok; ++k) if (k != gg) {
                        void* p = nullptr;
                        hipError_t e = hipExtMallocWithFlags(&p, llb, hipDeviceMallocUncached);
                        if (e != hipSuccess) { (void)hipGetLastError(); llkind = "finegrained"; e = hipExtMallocWithFlags(&p, llb, hipDeviceMallocFinegrained); }
                        if (e != hipSuccess) { (void)hipGetLastError(); std::fprintf(stderr, "Q27_NR_P2P=4: no uncached/fine-grained device allocation on card %d (%s)\n", gg, hipGetErrorString(e)); llok = 0; break; }
                        CK(hipMemset(p, 0, llb));
                        C.nrp_ll[q][gg][k] = (unsigned*)p;
                    }
                    for (int q = 0; q < Q27_LL_SLOTS; ++q) { CK(hipMalloc((void**)&C.nrp_llstage[q][gg], llb)); CK(hipMemset(C.nrp_llstage[q][gg], 0, llb)); }
                    CK(hipMalloc((void**)&C.nrp_cnt[gg], 64)); CK(hipMemset(C.nrp_cnt[gg], 0, 64));
                    CK(hipHostMalloc((void**)&C.nrp_err[gg], 64, hipHostMallocMapped));
                    std::memset(C.nrp_err[gg], 0, 64);
                    C.nrp_seq[gg] = 0;
                }
                g_ll_sdma = q27_env_int("Q27_LL_SDMA", 1);
                g_ll_epi  = g_ll_sdma ? q27_env_int("Q27_LL_EPI", 0) : 0;
                g_proj_seg = q27_env_int("Q27_PROJ_SEG", 0);
                // Self-test, every ordered pair, the protocol itself: the OWNER's spinning reader is launched FIRST (it reads the
                // stale word once, then polls), THEN the source stores {data, seq} through peer access. The reader must see the new
                // word (proves the peer store lands and the owner's re-read is not served from a stale cache) and report the stale
                // seq it saw first (proves the reader was really waiting, not reading a leftover).
                int llpass = 0, lltot = 0;
                for (int i = 0; i < ndev && llok; ++i) for (int j = 0; j < ndev; ++j) if (i != j) {
                    ++lltot;
                    unsigned* out = C.nrp_err[j];                   // pinned scratch on the owner side
                    std::memset(out, 0, 16);
                    unsigned* slot = C.nrp_ll[0][j][i];             // owner j, source i
                    const unsigned seq = 0x51000000u + (unsigned)(i * 16 + j), data = 0xA5000000u + (unsigned)(j * 16 + i);
                    CK(hipSetDevice(D[j].id)); CK(hipMemset(slot, 0, 64)); CK(hipStreamSynchronize(D[j].stream));
                    q27_ll_probe_wait(slot, seq, out, D[j].stream);               // owner spins
                    usleep(2000);
                    CK(hipSetDevice(D[i].id)); q27_ll_probe_write(slot, data, seq, D[i].stream); CK(hipStreamSynchronize(D[i].stream));
                    CK(hipSetDevice(D[j].id)); CK(hipStreamSynchronize(D[j].stream));
                    const int ok1 = (out[0] == data) && (out[1] == 0u);
                    if (ok1) ++llpass;
                    else std::fprintf(stderr, "Q27_NR_P2P=4 self-test FAIL %d->%d: got %08x want %08x (stale-before %08x, spins %u)\n", i, j, out[0], data, out[1], out[2]);
                    std::memset(out, 0, 16);
                    CK(hipSetDevice(D[j].id)); CK(hipMemset(slot, 0, 64)); CK(hipStreamSynchronize(D[j].stream));
                }
                CK(hipSetDevice(D[0].id));
                if (!llok || llpass != lltot) {
                    std::fprintf(stderr, "Q27_NR_P2P=4: LL transport unavailable (%d/%d pairs); host-staged collective kept.\n", llpass, lltot);
                    g_nr_p2p = 0;
                } else {
                    std::printf("Q27_NR_P2P=4 on: LL flag-in-data %s + local spin reduce for the %d-card row-batched collective (%s slots, %d deep, self-test %d/%d)\n",
                                g_ll_sdma ? "pack + copy-engine push" : "shader-store push", ndev, llkind, Q27_LL_SLOTS, llpass, lltot);
                }
            } else
            std::printf("Q27_NR_P2P=%d on: %s for the %d-card row-batched collective (self-test %d/%d)\n", g_nr_p2p,
                        g_nr_p2p == 2 ? "peer-read (pull) reduce" : (g_nr_p2p == 3 ? "bf16 SDMA push (hipMemcpyPeerAsync) + local reduce" : "bf16 store-kernel push + local reduce"), ndev, ndev * (ndev - 1), ndev * (ndev - 1));
        } else {
            std::fprintf(stderr, "Q27_NR_P2P: peer access unavailable or self-test failed; host-staged collective kept.\n");
            g_nr_p2p = 0;
        }
        CK(hipSetDevice(D[0].id));
    }
    // ---- Q27_VRAM: device memory headroom after every allocation phase (a late allocation that lands in
    //      GTT/host-coherent memory makes every kernel touching it PCIe-bound: print, do not guess) ----
    for (int gg = 0; gg < ndev; ++gg) {
        size_t fr = 0, tt = 0; CK(hipSetDevice(D[gg].id)); CK(hipMemGetInfo(&fr, &tt));
        std::printf("Q27_VRAM after_spec_init card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0);
    }
    for (int g = 0; g < ndev; ++g) dev_alloc_pfslots(D[g]);
    // The ACTUAL slot capacity, announced where it becomes true. Q27_READY's `slots=` is a
    // different quantity (an argv/env ingest knob that defaults to 1), so a server that wants to
    // pace work against the slot arena has to read this line -- and the Q27_SERVE_GROW line after
    // every in-place growth.
    q27_memory_receipt("runtime_resident");
    std::printf("Q27_SLOTS cap=%d\n", g_pf_cap);
    std::fflush(stdout);
    for (int gg = 0; gg < ndev; ++gg) {
        size_t fr = 0, tt = 0; CK(hipSetDevice(D[gg].id)); CK(hipMemGetInfo(&fr, &tt));
        std::printf("Q27_VRAM after_pfslots card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0);
    }
    CK(hipSetDevice(D[0].id));
    if (g_pf_p2p) {
        int p2p_ok = 1;
        for (int i = 0; i < ndev && p2p_ok; ++i) for (int j = 0; j < ndev; ++j) if (i != j) {
            int can = 0; CK(hipDeviceCanAccessPeer(&can, i, j));
            if (!can) p2p_ok = 0;
        }
        for (int i = 0; i < ndev && p2p_ok; ++i) {
            CK(hipSetDevice(i));
            for (int j = 0; j < ndev; ++j) if (i != j) {
                // hipDeviceEnablePeerAccess returns hipErrorPeerAccessAlreadyEnabled when the
                // pair is already enabled, and Q27_NR_P2P (on in the shipped recipe) enables every
                // pair earlier in startup. Treating that as failure declared peer access
                // "unavailable" on a box where it works perfectly well, which silently demoted the
                // collective to host staging -- and made the peer-quarter KV path read and write
                // through pointers it believed were mapped. Clear the sticky error and continue.
                hipError_t pe = hipDeviceEnablePeerAccess(j, 0);
                if (pe == hipErrorPeerAccessAlreadyEnabled) { (void)hipGetLastError(); pe = hipSuccess; }
                if (pe != hipSuccess) {
                    std::fprintf(stderr, "Q27_PF_P2P: enable %d->%d failed: %s\n", i, j, hipGetErrorName(pe));
                    p2p_ok = 0;
                }
            }
        }
        if (p2p_ok) {
            float* probe[Q27_MAX_DEVICES] = {nullptr};
            for (int i = 0; i < ndev; ++i) { CK(hipSetDevice(i)); CK(hipMalloc(&probe[i], 16)); }
            for (int i = 0; i < ndev && p2p_ok; ++i) for (int j = 0; j < ndev; ++j) if (i != j) {
                CK(hipSetDevice(i));
                const float want = 1000.0f + (float)i * 10.0f + (float)j;
                q27_p2p_probe(probe[j], want, D[i].stream);      // the write fully overwrites the slot
                CK(hipStreamSynchronize(D[i].stream));
                float got = -1.0f;
                CK(hipSetDevice(j));
                CK(hipMemcpy(&got, probe[j], 4, hipMemcpyDeviceToHost));
                if (got != want) { p2p_ok = 0;
                    std::fprintf(stderr, "Q27_PF_P2P self-test FAIL %d->%d got %f want %f\n", i, j, (double)got, (double)want); }
            }
            for (int i = 0; i < ndev; ++i) { CK(hipSetDevice(i)); CK(hipFree(probe[i])); }
        }
        if (!p2p_ok) {
            std::fprintf(stderr, "Q27_PF_P2P: peer access unavailable/failed; reverting to the host collective.\n");
            g_pf_p2p = 0;
        } else if (g_pf_x4w < 1 || !g_pf_att_chunk || !g_pf_gdn_scan || !g_pf_norm_b || !g_pf_att_b) {
            std::fprintf(stderr, "Q27_PF_P2P: requires Q27_PF_X4W>0 plus the batched attn/GDN/norm producers "
                                 "(Q27_PF_ATT_CHUNK, Q27_PF_GDN_SCAN, Q27_PF_NORM_B, Q27_PF_ATT_B); reverting.\n");
            g_pf_p2p = 0;
        } else {
            std::printf("Q27_PF_P2P: peer access self-test passed on %d cards.\n", ndev);
        }
    }

    // ---- NEGATIVE CONTROL: zero one card's shard of the two ROW-PARALLEL matrices of layer 0,
    // i.e. exactly the two contributions that only reach the answer through an all-reduce. If the
    // gate still passes, the collective is not carrying that card and the gate proves nothing.
    const int zero_card = q27_env_int("Q27_TP_ZERO", -1);
    // Q27_TP_ZERO_LAYERS=n zeroes the shard in the first n layers (default: every layer).
    // Perturbing only layer 0 made this control depend on a small numeric nudge propagating far
    // enough to flip an argmax, and it did not always survive the trip: with the host-sumsq norm
    // off, card 0 went undetected for 80 tokens; with it on, card 1 did -- a DIFFERENT card in
    // each configuration, which is the signature of an insensitive instrument rather than a
    // dropped contribution. A control that silently fails to fire is the thing this gate exists
    // to prevent, so it now removes the card's contribution everywhere.
    const int zero_layers = q27_env_int("Q27_TP_ZERO_LAYERS", nlayer);
    if (zero_card >= 0 && zero_card < ndev) {
        CK(hipSetDevice(D[zero_card].id));
        size_t zb = 0, db = 0; int nz = 0;
        for (int Lz = 0; Lz < nlayer && Lz < zero_layers; ++Lz) {
            const q27_layer_t* L0 = q27_layer_tp(m, Lz, D[zero_card].id);
            if (!L0) { std::fprintf(stderr, "Q27_TP_ZERO: no layer-%d handle on card %d\n", Lz, zero_card); return 1; }
            if (L0->is_full) { zb = (size_t)L0->o_proj.rows * (size_t)L0->o_proj.K;
                               CK(hipMemset(const_cast<unsigned char*>(L0->o_proj.w), 0, zb)); }
            else             { zb = (size_t)L0->out_proj.rows * (size_t)L0->out_proj.K;
                               CK(hipMemset(const_cast<unsigned char*>(L0->out_proj.w), 0, zb)); }
            db = (size_t)L0->down.rows * (size_t)(L0->down.K / 2);
            CK(hipMemset(const_cast<unsigned char*>(L0->down.w), 0, db));
            ++nz;
        }
        std::printf("Q27_TP_NEGATIVE_CONTROL card %d: %d layers, mixer shard zeroed (%zu B/layer) "
                    "and down_proj shard zeroed (%zu B/layer)\n", zero_card, nz, zb, db);
    }

    const bool ladder = (orc != nullptr);
    std::atomic<int> fails{0};
    std::vector<std::vector<float>> snap(ndev);          // cross-card residual equality check
    TpBar snapbar; snapbar.init(ndev);

    // shared sequence state; every card computes it from the same 4-way max, so no broadcast
    std::vector<unsigned> out;
    unsigned tok = prompt[0];
    size_t pi = 1;
    bool done = false;
    double t_decode = 0, t_prefill = 0;
    int n_decode = 0;
    // E3 (2026-09-17): SPECULATIVE DECOMPOSITION OF THE DECODE RATE.
    //   decode_tok_s == 1000 * (tokens/round) / (ms/round)
    // Only the denominator is kernel speed. The numerator is speculative ACCEPTANCE, which moves
    // with the token stream -- so two arms that generate different text cannot be compared on
    // decode_tok_s at all. Receipt for why this matters: prompt 13,665 on 37a74d01 read 72.73 tok/s
    // and on b34ccefd read 81.69, with the SAME tokhash and the SAME 2,130 tokens -- 12.3% apart
    // for an identical computation. A round counter separates the two terms: ms_per_round is what
    // a kernel change moves, tok_per_round is what a stream change moves.
    int  n_rounds = 0;                     // forward passes (draft+verify rounds), not committed tokens
    long rq_drafts0 = 0, rq_hits0 = 0, rq_acc1_0 = 0, rq_acc2_0 = 0;
    // E5 (2026-09-17): the per-phase speculative wall (draft / draft-xchg / embed / layers /
    // head-sync / head-xchg / tail) is ALREADY accumulated in d.sp_t[] every round, but it is only
    // printed after the decode loop joins -- which in serve mode never happens. Snapshotting it per
    // request turns it into a free per-round ledger, and it is the only instrument that separates
    // the replicated MTP drafter from the trunk without duplicating anything.
    double rq_spt0[8] = {0,0,0,0,0,0,0,0};
    // Per-request baselines for the transport receipt. The counters themselves are process-lifetime
    // (incremented inside the two transport implementations); differencing them per request is what
    // makes "this arm ran P2P" a measurement rather than a claim about the environment.
    long rq_collh0 = 0, rq_collp0[5] = {0,0,0,0,0};
    double rq_collt0[6] = {0,0,0,0,0,0};
    // The launch table is reset at T7_decode_begin, which fires AFTER the first decode step has
    // already run -- so exactly one round's launches are missing from the count. On a 532-round
    // request that is 0.19%, below the +-1.1% noise floor, but the denominator is corrected anyway
    // so launches/round is not quietly biased low.
    int lk_skipped = 0;

    // WARM-UP, before the timed region. The first launch of each kernel shape on each device loads
    // its code object, which costs tens of ms per device and otherwise lands INSIDE TTFT -- measured
    // as ~170 ms between T5_threads_begin and T6_prefill_begin, against 19.2 ms of actual prefill
    // work for a 7-token prompt. That is one-time initialisation, not per-request work, so it is
    // paid here rather than billed to the first prompt.
    if (q27_env_flag("Q27_WARMUP", true)) {
        for (int g = 0; g < ndev; ++g) {
            CK(hipSetDevice(D[g].id));
            q27_launch_nothing(D[g].stream);
            CK(hipStreamSynchronize(D[g].stream));
        }
    }

    MS("T5_threads_begin");
    g_tp_lsrec = C.ls_rec; g_tp_ndev_dbg = ndev;      // so the barrier's timeout handler can name the card
    std::vector<std::thread> th;
    for (int g = 0; g < ndev; ++g) th.emplace_back([&, g] {
        t_tp_card = g;                                  // this thread's card, for the barrier timeout report
        CK(hipSetDevice(D[g].id));
        tp_pin_thread(D[g].id, g, ndev);
        Dev& d = D[g];
        if (g_tp_profile) g_tpev[g].ensure();

        // ---- CHANGING-INPUT COLLECTIVE GATE (Q27_TP_COLLGATE=<iters>) --------------------
        // WHY THIS EXISTS. Window 2 proved on 2026-09-09 that allred3's 40.61 us "parallel"
        // all-reduce was RACY: it dropped the rendezvous, so a worker could consume generation
        // N-1 of a peer's host buffer while producing generation N. The benchmark could not see
        // it because input(N) == input(N-1), making stale data numerically identical to fresh.
        // A fixed-input collective test is therefore VACUOUS by construction.
        //
        // This gate gives every (iteration, card) a distinguishable deterministic payload, so a
        // stale, missing, duplicated or wrong-source contribution all change the sum EXACTLY.
        // Values stay integral and below 2^24, so fp32 equality is exact and any mismatch is a
        // real defect, never rounding. Reading the code is not evidence; this is.
        if (const char* cg = getenv("Q27_TP_COLLGATE")) {
            int iters = atoi(cg); if (iters <= 0) iters = 2000;
            if (iters > 4000) iters = 4000;                    // keep coef*(i%251+1) < 2^24
            std::vector<float> hsend(Q27_HID), hback(Q27_HID);
            long long bad = 0; int first_bad = -1;
            for (int it = 0; it < iters; ++it) {
                for (int i = 0; i < Q27_HID; ++i)
                    hsend[i] = (float)((long long)(it * ndev + g + 1) * (long long)(i % 251 + 1));
                // Feed the payload in wherever the PRODUCER would have left it. Under zero-copy-out
                // nothing copies d.mixer any more, so a gate that stages there would reduce stale
                // pinned memory -- and would keep passing while carrying none of its own payload.
                if (g_coll_zcout) memcpy(C.hp[g], hsend.data(), (size_t)Q27_HID * 4);
                else CK(hipMemcpy(d.mixer, hsend.data(), (size_t)Q27_HID * 4, hipMemcpyHostToDevice));
                tp_allreduce(C, g, (float*)d.mixer, d.stream);
                // Read the result from where the collective ACTUALLY leaves it. Under zero-copy the
                // reduced vector stays in pinned host memory and d.mixer is never written back, so
                // reading d.mixer here checked a stale buffer and reported 0/4 while the engine was
                // correct. A gate that tests the old contract after the contract changes is worse
                // than no gate: it fails on correct code and would pass on a broken zero-copy path.
                if (g_coll_zc) memcpy(hback.data(), C.acc, (size_t)Q27_HID * 4);
                else CK(hipMemcpy(hback.data(), d.mixer, (size_t)Q27_HID * 4, hipMemcpyDeviceToHost));
                // sum over k of (it*ndev + k + 1) = ndev*ndev*it + ndev*(ndev+1)/2
                const long long coef = (long long)ndev * ndev * it + (long long)ndev * (ndev + 1) / 2;
                for (int i = 0; i < Q27_HID; ++i)
                    if (hback[i] != (float)(coef * (long long)(i % 251 + 1))) {
                        if (first_bad < 0) first_bad = it;
                        ++bad;
                    }
            }
            std::printf("Q27_TP_COLLGATE card %d: %d iters x %d floats, %lld mismatches%s\n",
                        g, iters, Q27_HID, bad,
                        bad ? "  ***RACY/INCORRECT COLLECTIVE***" : "  PASS");
            if (bad) std::printf("  card %d first bad iteration %d\n", g, first_bad);
            if (bad) fails.fetch_add(1);
            return;                                            // gate only; generate nothing
        }
        // ---------------------------------------------------------------------------------

        // ---- PREFILL SWEEP (Q27_PREFILL_SWEEP=1): layer-major over the prompt positions. ----
        // The per-position math is IDENTICAL to the serial prefill (same kernels, same order per
        // position); only the iteration order changes, so tokens must match exactly. Positions
        // live in their own hid slots; each position's residual-2 rides a per-position pending
        // slot (the copy is stream-ordered and the next collective's event sync covers it before
        // the host rewrites acc). The preenq machinery is OFF during the sweep: this is the
        // correctness skeleton the batched kernels will replace stage by stage.
        int pos0 = 0;
        // ---- REQUEST LOOP (Q27_SERVE=1). Request 0 is the argv prompt; later requests arrive on
        // stdin as "<maxn> <id> <id> ...", are prefilled by the same sweep at the CURRENT absolute
        // position (KV cache and GDN state are position-indexed, so the conversation simply
        // continues), and decode until EOS or maxn. Without Q27_SERVE the loop runs once.
        const unsigned* sw_tok = nullptr;   // nullptr => read prompt[]
        int sw_lo = 0, sw_n = -1;           // -1 => the whole prompt
        int pos_cur = 0;                    // where the next request's prompt starts
        std::vector<unsigned> req_ids;      // this thread's copy of the current request
        int  req_maxn = arg_pf_only ? 0 : maxn;
        int  req_rewind = -1;                   // Q27_SERVE_REWIND target, per request
        bool req_pf_only = arg_pf_only;     // request 0 is prefill-only iff argv said so
        // Q27_LCP_DBG: RESIDENT-PREFIX DIVERGENCE DIAGNOSIS. The server's reuse test is
        // all-or-nothing (exact whole-prefix match or reset to 0), so when a Harness re-render stops
        // being an exact extension the whole conversation is re-prefilled -- measured at ~4 minutes
        // on a 77K window. Nothing on either side records WHERE the two token streams part, so the
        // engine keeps its own resident history and reports the longest common prefix across a
        // reset. Card 0 only; pure logging, no behaviour change, no GPU work.
        const bool lcp_on = q27_env_flag("Q27_LCP_DBG", true);
        std::vector<unsigned> lcp_hist;     // tokens this engine actually holds, by position
        std::vector<unsigned> lcp_prev;     // what it held immediately before the last reset
        bool lcp_pending = false;           // a reset happened; compare the next window against prev
        for (int req = 0;; ++req) {
        if (req == 0 && g == 0 && g_text) { std::fprintf(g_text_out, "[user] %s\n[assistant] ", g_first_text.c_str()); std::fflush(g_text_out); }
        if (req > 0) {
            if (g == 0) { serve_read_request(g_req); out.clear(); done = false; n_decode = 0; t_decode = 0; t_prefill = 0;
                           q27_ab_reread();
                           n_rounds = 0; rq_drafts0 = d.nr_drafts; rq_hits0 = d.nr_hits;
                           rq_acc1_0 = d.nr_acc_hist[1]; rq_acc2_0 = d.nr_acc_hist[2];
                           for (int j = 0; j < 8; ++j) rq_spt0[j] = d.sp_t[j];
                           rq_collh0 = g_coll_host_sites[g];
                           for (int j = 0; j < 5; ++j) rq_collp0[j] = g_coll_p2p_sites[g][j];
                           for (int j = 0; j < 6; ++j) rq_collt0[j] = g_collt[g][j]; }
            C.b1.wait(true);                                // request published (or EOF) -- IDLE: card 0 is
                                                            // blocked reading the next request; no deadline.
            const bool eof = g_req.eof;
            req_ids = g_req.ids; req_maxn = g_req.maxn; req_pf_only = g_req.pf_only;
            req_rewind = g_req.rewind_pos;
            C.b2.wait();                                    // every thread holds its copy
            d.nr_have_h = 0; d.nr_ncatch = 0;                  // Q27_SPEC: the first step of a new request is plain (it seeds h_prev)
            if (eof) break;
            // RESET: start a fresh conversation on the RESIDENT engine.
            // The serve protocol prefills at the current absolute position, so a request that is
            // not a prefix extension of what we hold cannot be appended. The server was handling
            // that by killing this process and reloading 27B of weights (~70 s), or by refusing
            // the request with a 400. A serving backend does neither: it serves every request and
            // pays a prefill when there is no prefix to reuse. Rewinding to position 0 costs a
            // memset of the recurrent state -- the KV cache is position-indexed and is simply
            // overwritten by whatever gets prefilled next.
            // REWIND (maxn == -3): truncate the resident conversation to an absolute POSITION
            // instead of discarding it. The KV cache needs NO work -- it is position-indexed, so
            // every row at or beyond the target is simply overwritten by the next prefill, exactly
            // as the reset comment below says. Only the recurrent state must be restored, because
            // S/conv are running summaries with no per-position structure; that is what the
            // Q27_REWIND_CK banks hold. When no bank covers the target this falls back to the
            // position-0 reset, so the behaviour is never worse than before it existed.
            if (req_maxn == -3) {
                CK(hipSetDevice(d.id));
                int pick = -1, pick_pos = -1;
                if (q27_rewind_ck() > 0 && req_rewind > 0)
                    for (int i = 0; i < q27_rewind_ck(); ++i)
                        if (g_ck_pos[i] >= 0 && g_ck_pos[i] <= req_rewind && g_ck_pos[i] > pick_pos)
                            { pick = i; pick_pos = g_ck_pos[i]; }
                // Every worker must choose from the same checkpoint table before
                // thread 0 invalidates future banks. Otherwise a fast worker can
                // remove a bank while another worker is still selecting it.
                C.b1.wait(); C.b2.wait();
                if (pick >= 0) {
                    CK(hipMemcpyAsync(d.S, d.ck_S + (size_t)pick * (d.s_bytes_total / sizeof(float)),
                                      d.s_bytes_total, hipMemcpyDeviceToDevice, d.stream));
                    CK(hipMemcpyAsync(d.conv, d.ck_conv + (size_t)pick * (d.conv_bytes_total / sizeof(unsigned short)),
                                      d.conv_bytes_total, hipMemcpyDeviceToDevice, d.stream));
                    CK(hipStreamSynchronize(d.stream));
                    pos_cur = pick_pos;
                } else {
                    CK(hipMemsetAsync(d.S, 0, d.s_bytes_total, d.stream));
                    CK(hipMemsetAsync(d.conv, 0, d.conv_bytes_total, d.stream));
                    CK(hipStreamSynchronize(d.stream));
                    pos_cur = 0;
                }
                d.nr_have_h = 0; d.nr_ncatch = 0;
                // Any bank at or past the new head describes a future that no longer exists.
                if (g == 0) for (int i = 0; i < q27_rewind_ck(); ++i) if (g_ck_pos[i] > pos_cur) g_ck_pos[i] = -1;
                if (g == 0 && lcp_on && (int)lcp_hist.size() > pos_cur) lcp_hist.resize(pos_cur);
                C.b1.wait(); C.b2.wait();           // every card at the new head before card 0 acks
                if (g == 0) { std::printf("Q27_SERVE_REWIND ok %d asked %d\n", pos_cur, req_rewind); std::fflush(stdout); }
                continue;
            }
            if (req_maxn == -1) {
                CK(hipSetDevice(d.id));
                CK(hipMemsetAsync(d.S, 0, d.s_bytes_total, d.stream));
                CK(hipMemsetAsync(d.conv, 0, d.conv_bytes_total, d.stream));
                CK(hipStreamSynchronize(d.stream));
                pos_cur = 0;
                d.nr_have_h = 0; d.nr_ncatch = 0;
                // Keep what we held: the next window is the one that failed the server's exact-prefix
                // test, and it is the only chance to see where the two streams diverged.
                if (g == 0 && lcp_on) { lcp_prev.swap(lcp_hist); lcp_pending = true; }
                lcp_hist.clear();
                if (g == 0) for (int i = 0; i < q27_rewind_ck(); ++i) g_ck_pos[i] = -1;
                C.b1.wait(); C.b2.wait();           // every card clean before card 0 acknowledges
                if (g == 0) { std::printf("Q27_SERVE_RESET ok\n"); std::fflush(stdout); }
                continue;
            }
            if (req_ids.empty() || pos_cur + (int)req_ids.size() + req_maxn > ctx) {
                if (g == 0) { std::printf("Q27_REQ_REFUSED %d: pos=%d prompt=%d maxn=%d ctx=%d\n", req, pos_cur,
                                          (int)req_ids.size(), req_maxn, ctx); std::fflush(stdout); }
                continue;
            }
            // Q27_REWIND_CK: checkpoint the recurrent state as it stands BEFORE this request's
            // prefill, tagged with the position it corresponds to, so a later divergent window can
            // rewind HERE instead of to 0. Same D2D copy the speculation banks already use: one
            // 64.6 MiB/card copy per request (~0.1 ms at D2D bandwidth) and nothing at all on the
            // decode path. Every card writes the same value into the same slot -- `req` is identical
            // on all of them -- so the ring needs no lock.
            if (q27_rewind_ck() > 0) {
                const int ck_slot = (req * 2) % q27_rewind_ck();
                CK(hipSetDevice(d.id));
                CK(hipMemcpyAsync(d.ck_S + (size_t)ck_slot * (d.s_bytes_total / sizeof(float)), d.S,
                                  d.s_bytes_total, hipMemcpyDeviceToDevice, d.stream));
                CK(hipMemcpyAsync(d.ck_conv + (size_t)ck_slot * (d.conv_bytes_total / sizeof(unsigned short)),
                                  d.conv, d.conv_bytes_total, hipMemcpyDeviceToDevice, d.stream));
                if (g == 0) g_ck_pos[ck_slot] = pos_cur;
            }
            sw_tok = req_ids.data(); sw_lo = pos_cur; sw_n = (int)req_ids.size();
            if (g == 0 && lcp_on) {
                if (lcp_pending) {      // this window is the one the server could not reuse
                    lcp_pending = false;
                    const size_t R = lcp_prev.size(), N = req_ids.size();
                    const size_t lim = (R < N) ? R : N;
                    size_t l = 0; while (l < lim && lcp_prev[l] == req_ids[l]) ++l;
                    std::printf("Q27_LCP resident=%zu incoming=%zu lcp=%zu retainable=%.2f%% "
                                "first_mismatch=%zu resident_tok=%d incoming_tok=%d\n",
                                R, N, l, R ? 100.0 * (double)l / (double)R : 0.0, l,
                                (l < R) ? (int)lcp_prev[l] : -1, (l < N) ? (int)req_ids[l] : -1);
                    const size_t a = (l > 12) ? l - 12 : 0;
                    std::printf("Q27_LCP_RES @%zu:", a);
                    for (size_t i = a; i < l + 12 && i < R; ++i) std::printf(" %u", lcp_prev[i]);
                    std::printf("\nQ27_LCP_INC @%zu:", a);
                    for (size_t i = a; i < l + 12 && i < N; ++i) std::printf(" %u", req_ids[i]);
                    std::printf("\n"); std::fflush(stdout);
                }
                lcp_hist.resize((size_t)sw_lo);     // this request starts at sw_lo; drop anything past it
                lcp_hist.insert(lcp_hist.end(), req_ids.begin(), req_ids.end());
            }
        }
        if (req == 0 && g == 0 && lcp_on) lcp_hist.assign(prompt.begin(), prompt.end());
        // SWEEP WINDOW. For the prompt this is [0, npre) and behaviour is bit-identical. The MTP
        // verifier reuses the SAME body for a chunk of drafted tokens at an arbitrary base position,
        // which is why the base and length are variables rather than the prompt length: the KV cache
        // and the GDN state are position-indexed, so a chunk must be addressed by ABSOLUTE position,
        // not renumbered from zero.
        // PRODUCTION PREFILL POLICY. The layer-major sweep with batched kernels measured 2.34 ms per
        // prompt token against 17.98 serial (41-token prompt, tokens identical, handoff verified),
        // so it is the default for any prompt of Q27_PF_AUTO_MIN (8) tokens or more.
        // Q27_PREFILL_SWEEP=0 forces the serial per-position path; =1 forces the sweep.
        const bool pf_auto = !prompt.empty() && (int)prompt.size() >= q27_env_int("Q27_PF_AUTO_MIN", 8);
        if (req > 0 || (q27_env_flag("Q27_PREFILL_SWEEP", pf_auto) && !prompt.empty())) {
            const int npre = (int)prompt.size();
            // Window: [0,npre) for the prompt (sw_n < 0), else the requested chunk at an ABSOLUTE base.
            const int sw_beg = (sw_n < 0) ? 0 : sw_lo;
            const int sw_end = (sw_n < 0) ? npre : sw_lo + sw_n;
            const int sb = sw_beg;               // slot index = position - window base
            const int nwin = sw_end - sw_beg;
            // The draft writes KV for every position this window produces, plus whatever decode
            // generates after it. Grow before the sweep touches them.
            // THEN KEEP THE DRIVER'S WORKING ROOM. The headroom note in q27_pfslots_grow is the
            // measured one (0.06 GiB free took a 5.9 s prefill to 32 s), and the serve path can walk
            // the cards into that state without ever growing the slot arena: the KV, the draft KV
            // and the recurrent state all keep growing with the conversation while fast-layer
            // residency stays put. Measured 2026-09-15 in a growing session: 8,960-position appends
            // went 7.8 s -> 110 s -> 189 s with the cards at ~0.2 GiB free and four MORE fast layers
            // resident than a comparable run that was fast at the same positions with ~1.5 GiB free.
            // Residency is only worth keeping while the driver has room to work, so each card sheds
            // its OWN mirrors (owner card only, no coordination) until the margin is back.
            q27_keep_headroom(m, ndev, d);
            // K=0 never reads the draft KV (the draft loop does not execute), so growing it is pure
            // cost: 2,560 B/position of allocation and a copy of the live prefix for a buffer no
            // consumer touches. The shipped path keeps the grow unchanged.
            if (g_spec && g_spec_k > 0 && (!g_sr || g == g_sr_head)) {   // Q27_LS_SR: only the head card holds a draft KV
                // THE GROWTH HORIZON IS THIS REQUEST'S BUDGET, NOT THE ENGINE'S ARGV. `gen_budget`
                // is computed ONCE at boot (q27_main.cpp:4008) from the argv maxn, and the serve
                // path boots the engine with maxn == -2 (prefill-only) -- so arg_pf_only is true and
                // gen_budget is PERMANENTLY 0. The draft KV was therefore sized to sw_end with only
                // q27_mtp_kv_reserve's 1024-position slack as generation headroom, whatever the
                // request actually asked to generate. Decode then walks out of the cap,
                // q27_mtp_within_cap() fails, eff_k drops to 0 (q27_main.cpp:8127) and the round
                // degrades to a 1-row trunk with no draft -- the documented safe fallback, firing
                // for the wrong reason. Measured 2026-09-17 on REAL agentic traffic:
                //   REQ 10  gen 1003 tok (the 1024 slack)  acc_p1 0.050  nr_eff 1  decode 36.87
                //   REQ 21  17 positions of slack left     acc_p1 0.018  nr_eff 1  decode 30.78
                // against acc_p1 ~0.85-0.95 and 56-80 tok/s on every neighbouring request, i.e. a
                // 45-55% decode loss whenever a generation outran the slack. req_maxn is this
                // request's real budget (set per request by the serve loop); 8192 positions of draft
                // KV is 21 MB, and the affordability rule below still governs whether it is taken.
                const int gen_horizon = req_pf_only ? 0 : (req_maxn > 0 ? req_maxn : gen_budget);
                const int mtp_need = (sw_end + gen_horizon < ctx) ? (sw_end + gen_horizon) : ctx;
                // AFFORDABILITY FIRST. q27_mtp_kv_reserve frees the old draft buffers and allocates
                // the new size; when the card cannot pay, hipMalloc does not return an error -- it
                // stalls inside the driver, leaving this card's thread spinning and its three peers
                // parked on the idle barrier (measured 2026-09-15: a 234,223-token window against a
                // 15,616-position arena hung the 249,583-token probe exactly there, one card in R
                // and three in S, temperatures at idle). Ask the arithmetic question first: 2,560 B
                // per draft position (kc/vc/ks/vs, confirmed by the engine's own grow lines: 40 MiB
                // at 16,384, 320 MiB at 131,072, 640 MiB at 262,144) against free memory minus the
                // margin this file already keeps. Not affording it is fine: the cap stays where it
                // is, q27_mtp_within_cap keeps speculation off beyond it, and every drafted token
                // would have been verified anyway.
                // THE PEAK IS THE WHOLE NEW BUFFER, NOT THE DIFFERENCE. q27_mtp_kv_reserve
                // allocates the new kc/vc/ks/vs before it frees the old ones, and rounds the
                // capacity UP TO A POWER OF TWO. Asking for (need - cap) is how the first version
                // of this check still let the 249,583-token probe stall: 558 MB of "delta" against a
                // reserve that must find 640 MB at once.
                int mtp_cap_t = d.mtp_cap;
                const int mtp_step = 1024;               // keep in lockstep with q27_mtp_kv_reserve
                if (mtp_cap_t < mtp_need) {
                    long long want = (long long)mtp_need + mtp_step;
                    if (want > ctx) want = ctx;
                    mtp_cap_t = (int)((want + 255LL) & ~255LL);
                    if (mtp_cap_t < mtp_need) mtp_cap_t = ctx;
                }
                const size_t mtp_need_bytes = (size_t)mtp_cap_t * Q27_MTP_BPP;
                // THE OLD CACHE IS PART OF THE BUDGET. q27_mtp_kv_reserve releases the current
                // draft buffers before it takes the new ones (host-staged when the direct path
                // cannot fit), so what has to fit is the NEW buffer against free + the old one.
                // Pricing the new buffer against free alone is what refused the 138 k growth while
                // 335 MB of old cache sat there reclaimable.
                const size_t mtp_reclaim = (size_t)d.mtp_cap * Q27_MTP_BPP;
                // THE DRAFT KV GETS ITS OWN, SMALLER MARGIN THAN THE ARENA. The arena costs 30,720 B
                // per position and is worth protecting hard; the draft KV costs 2,560 B per position,
                // so its next bucket is +40-80 MB -- and paying for it is what keeps speculation ON,
                // which is worth roughly 2x decode (96 tok/s speculation vs 47 native, both measured
                // on this host). Sharing the arena's 512 MB margin made every grow above 16,384
                // positions fail at the operating point: measured 2026-09-15 22:02, "draft needs
                // 17854 positions (+80 MB) but 0.56 GiB free -- keeping 16384, speculation stays off
                // beyond it". 256 MB is still far above the 0.06 GiB cliff this guard exists for.
                // ... AND IT IS TIGHTER THAN THE ARENA'S, BECAUSE IT BUYS 2x DECODE FOR 2,560 B/POSITION.
                // Measured 2026-09-15 22:37:06: the draft KV needed one more bucket (131,072 positions,
                // +320 MB) with 0.56 GiB free, and a 256 MB margin refused it by THREE MEGABYTES --
                // speculation went off and decode fell 64 -> 28.7 tok/s for the rest of the conversation.
                // 192 MB still sits well above the driver cliff (0.06 GiB measured) and is spent only on
                // the draft cache, never on the arena, which keeps its hard 512 MB.
                const size_t mtp_margin =
                    (size_t)q27_env_int("Q27_SERVE_DRAFT_MARGIN_MB", 192) * 1048576ull;
                size_t mtp_free = 0, mtp_total = 0;
                hipMemGetInfo(&mtp_free, &mtp_total);
                const bool mtp_afford = mtp_cap_t >= mtp_need &&
                    mtp_need_bytes <= ((mtp_free + mtp_reclaim) > mtp_margin
                                       ? (mtp_free + mtp_reclaim) - mtp_margin : 0);
                if (!mtp_afford) {
                    // FREE BEFORE ALLOCATE, PRICE THE PEAK. The gate above is right to refuse an
                    // allocation it has not proven affordable. What it did NOT do was consult the
                    // one wallet this card actually holds: its own fast-layer residency. The
                    // eviction path in the next branch was reachable only when the allocation was
                    // ALREADY affordable -- i.e. exactly when shedding was not needed -- so the
                    // draft cache stopped growing the moment free memory fell below the whole new
                    // buffer. Measured 2026-09-16 at 138 k positions: the draft needed the next
                    // bucket with 0.41-0.47 GiB free, 13-16 int8 MLP layers worth ~120 MB each sat
                    // unspent, speculation stayed off for the rest of the conversation, and decode
                    // sat at 21 tok/s in a conversation that had decoded at 55 earlier.
                    //
                    // So: shed this card's own fast layers FIRST (each hipFree raises free), re-test
                    // affordability against the same peak rule after every shed, and allocate only
                    // once it is genuinely affordable. A card that never becomes affordable still
                    // prints a refusal and keeps its old capacity -- a refusal must remain a message,
                    // never a kill.
                    for (int attempt = 0; attempt < Q27_LAYERS; ++attempt) {
                        hipMemGetInfo(&mtp_free, &mtp_total);
                        if (mtp_cap_t >= mtp_need &&
                            mtp_need_bytes <= ((mtp_free + mtp_reclaim) > mtp_margin
                                               ? (mtp_free + mtp_reclaim) - mtp_margin : 0)) break;
                        if (!q27_evict_one_mlp_own_card(m, ndev, d.id)) break;
                    }
                    hipMemGetInfo(&mtp_free, &mtp_total);
                    const bool mtp_afford_now = mtp_cap_t >= mtp_need &&
                        mtp_need_bytes <= ((mtp_free + mtp_reclaim) > mtp_margin
                                           ? (mtp_free + mtp_reclaim) - mtp_margin : 0);
                    if (mtp_afford_now && q27_mtp_kv_reserve(d, mtp_need, ctx)) {
                        std::fprintf(stderr, "Q27_MTP_FUND card %d: draft KV %d -> %d positions "
                                             "(shed fast layers to afford the whole buffer; %.2f GiB free)\n",
                                     d.id, d.mtp_cap, mtp_cap_t, (double)mtp_free / 1073741824.0);
                    } else {
                        std::fprintf(stderr, "Q27_MTP_SKIP card %d: draft needs %d positions (+%zu MB) but "
                                             "%.2f GiB free even after shedding -- keeping %d, "
                                             "speculation stays off beyond it\n",
                                     d.id, mtp_need, mtp_need_bytes >> 20,
                                     (double)mtp_free / 1073741824.0, d.mtp_cap);
                    }
                    std::fflush(stderr);
                } else if (!q27_mtp_kv_reserve(d, mtp_need, ctx)) {
                    // FUND IT from fast-layer residency, exactly like slot growth: the draft KV is
                    // per-card memory, so each card may shed its own int8 MLP mirrors (freeing
                    // ~120 MB each) without any cross-card coordination -- no symmetry requirement,
                    // unlike the collective paths. Measured failure this repairs: card 0 grew its
                    // draft cache to 131,072 positions while the tight card refused and kept
                    // 65,536, and the verify sweep then wrote past the end of the tight card's
                    // draft cache -> GPU page fault (node-4).
                    for (int attempt = 0; attempt < Q27_LAYERS; ++attempt) {
                        if (!q27_evict_one_mlp_own_card(m, ndev, d.id)) break;
                        if (q27_mtp_kv_reserve(d, mtp_need, ctx)) break;
                    }
                    if (!q27_mtp_kv_reserve(d, mtp_need, ctx))
                        std::fprintf(stderr, "Q27_MTP_PAGED card %d: grow to %d refused; keeping %d "
                                             "(no more fast layers to shed)\n",
                                     d.id,  sw_end + gen_horizon, d.mtp_cap);
                }
            }
            if (nwin > g_pf_cap) {
                // GROW, DON'T REFUSE. Every card reaches this with the same nwin, so card 0 is the
                // leader: it resizes the slot buffers on ALL devices (a host thread may allocate on
                // any device) and publishes the result; the others wait and then continue. Refusing
                // here used to cost the whole conversation -- server kills the engine, a replacement
                // cold-starts and re-prefills the entire history.
                // THE GROW IS COLLECTIVE, SO IT MUST BE BRACKETED BY BARRIERS -- NOT POLLED.
                // Every card reaches here with the same nwin, but only card 0 grows; the others used
                // to POLL `g_pf_grow_ok < 0` with a bounded spin, which has TWO independent doors out
                // of the round and both were open on every session:
                //   Door A -- the waiter arrives BEFORE the leader announces. g_pf_grow_ok still holds
                //     its previous value (0 at boot, 1 after any earlier grow), `1 < 0` is false, and
                //     the waiter falls straight through. Card 0 is systematically LAST to arrive here
                //     (it has just run q27_keep_headroom and the whole MTP eviction block above), so
                //     this is the common case, not the rare one.
                //   Door B -- the bounded 2,000,001-yield spin (~1.2 s on this box) expires while the
                //     grow is still running (~2 s measured).
                // Either way the escaping card re-reads the STALE g_pf_cap below, prints
                // Q27_SWEEP_TOO_LONG, and `continue`s past the sweep -- skipping barrier pairs its
                // peers do execute. Because the inter-request idle rendezvous reuses the same C.b1/C.b2
                // objects, TpBar counts the desynced card's next arrival as a valid one and launders
                // the divergence, so the failure surfaces 60 s later at an unrelated barrier with the
                // other threads parked in the two waits that have no deadline (the bare ring spin at
                // `while (C.ls_rec[src].load(acquire) < pos) { }` and the idle barrier). That is the
                // "FATAL TpBar ... arrived 1/2/3" on the first big turn of a session.
                // MEASURED 2026-09-16, and it is NOT a V4 defect: the identical signature fired at
                // 20:13:42-20:14:45 on the V4=0 binary (its boot prints the int8 arm's
                // "kc/vc=1.750 GiB each"), with Q27_SWEEP_TOO_LONG logged at 20:13:42 and the leader's
                // Q27_SERVE_GROW only at 20:13:44 -- a card refusing two seconds before the grow it was
                // supposed to be waiting for. A prior handoff recorded this same event as a
                // VRAM-settling false negative caused by start sequencing; it is neither.
                // b1 = nobody starts the round early. b2 = g_pf_cap is published before anyone re-reads
                // it, so all four cards take the SAME branch below. A barrier also has a deadline, so a
                // genuinely lost leader aborts loudly here instead of desyncing silently.
                C.b1.wait();
                if (g == 0) {
                    const int seq = g_pf_grow_seq.load() + 1;
                    g_pf_grow_ok.store(-1, std::memory_order_seq_cst);        // round in progress
                    g_pf_grow_seq.store(seq, std::memory_order_seq_cst);      // announce
                    const int got = q27_pfslots_grow(m, D, ndev, nwin);
                    g_pf_grow_ok.store(got > 0 ? 1 : 0, std::memory_order_seq_cst);  // outcome
                }
                C.b2.wait();
            }
            // BELT AND BRACES: every card re-asserts its own device after a grow decision the leader
            // may have made on its behalf (see the guard inside q27_pfslots_grow).
            CK(hipSetDevice(d.id));
            if (nwin > g_pf_cap) {
                std::fprintf(stderr, "Q27_SWEEP_TOO_LONG window=%d > slots=%d -- refusing\n",
                             nwin, g_pf_cap);
                // SAY IT ON THE SERVE WIRE TOO. The prefill slots are sized once, at cold start,
                // from the first prompt; a later window can exceed them (a reset re-prefills the
                // whole conversation). This used to be a stderr line only, so the server saw a
                // normal end-of-generation with zero tokens and reported a successful empty turn
                // -- the client just sat there. A refusal has to be visible as a refusal.
                if (g == 0) {
                    std::printf("Q27_REQ_REFUSED sweep: window=%d slots=%d\n", nwin, g_pf_cap);
                    std::fflush(stdout);
                }
                fails.fetch_add(1);
                // AND IT MUST NOT KILL THE ENGINE. This was `return`: it ended this card's thread,
                // so an un-growable window destroyed the resident conversation (all four cards
                // return, the threads join, the process exits). The serve path is built to answer
                // such a request by appending the window in arena-sized chunks through the slots we
                // already have, and it can only do that if the engine survives to serve them.
                // Nothing has been swept at this point and pos_cur is untouched, so the refusal is
                // exactly as recoverable as the ctx refusal above.
                continue;
            }
            g_pf_sweep = true;
            if (g_tpsr && !d.tpsr_primed && q27_env_flag("Q27_Q8_PRIME", true)) {
                // ---- Q27_TPSR PRIME (once per card, outside the timed sweep): tune the two rocBLAS shapes of the
                // sharded MLP -- gate/up [INTER/ndev][HID] and down [HID][INTER/ndev] -- at the chunk width. An
                // untuned shape runs rocblas_gemm_algo_standard, measured 9-15 vs 16-20 TMAC/s (src/q27_rb.cpp). ----
                const q27_layer_t* l0 = q27_layer_tp(m, 0, d.id);
                const int Mp = (g_pf_chw > 0 && g_pf_chw <= Q27_PF_CH) ? g_pf_chw : Q27_PF_CH;
                const Q8Scr& S0 = d.q8s[0];
                if (l0 && tpsr_mlp_view(d, l0, d.stream)) {
                    // Q27_MLP_WSHARE: xq2 is the DOWN input, whose width is this card's MLP slice --
                    // NOT HID. It is > HID whenever a card owns more than 5 k-blocks, and reading
                    // Mp*mlp_len bytes out of an Mp*HID host buffer is a host overread that the DMA
                    // engine reports as "Memory access fault by GPU node-N ... Page not present"
                    // ~52 s into the boot (measured 19:13:32, card 0 with an 8192-row slice).
                    const int mlw = mlp_len(d.id);
                    const size_t nrand = (size_t)Mp * (Q27_HID > mlw ? Q27_HID : mlw);
                    std::vector<signed char> rnd(nrand); unsigned st = 0x9E3779B9u ^ (unsigned)g;
                    for (size_t i = 0; i < rnd.size(); ++i) { st = st * 1664525u + 1013904223u; rnd[i] = (signed char)((int)(st >> 24) - 128); }
                    CK(hipMemcpy(S0.xqt, rnd.data(), (size_t)Mp * Q27_HID, hipMemcpyHostToDevice));
                    CK(hipMemcpy(S0.xq2, rnd.data(), (size_t)Mp * mlw, hipMemcpyHostToDevice));
                    (void)q27_rb_tune_i8g(d.id, d.tpsr_g8r.w, d.tpsr_g8r.rows, d.tpsr_g8r.K, S0.xqt, Mp, S0.acc, d.stream);
                    (void)q27_rb_tune_i8g(d.id, d.tpsr_d8r.w, d.tpsr_d8r.rows, d.tpsr_d8r.K, S0.xq2, Mp, S0.acc, d.stream);
                    CK(hipStreamSynchronize(d.stream));
                    if (g == 0) std::printf("Q27_TPSR: rocBLAS shapes tuned at M=%d (gate/up %dx%d, down %dx%d)\n", Mp,
                                            d.tpsr_g8r.rows, d.tpsr_g8r.K, d.tpsr_d8r.rows, d.tpsr_d8r.K);
                    if (g_tpsr_proj_rb) {
                        // ---- Q27_TPSR_PROJ_RB: tune the three projection shapes too, or they run algo_standard
                        // (9-15 vs 16-20 TMAC/s). Layer 0 is GDN (in_qkv|in_z + out_proj); layer 3 is the first
                        // full-attention layer (q|k|v + o_proj). o_proj and out_proj share the [5120][1536] key.
                        const q27_layer_t* l3 = q27_layer_tp(m, 3, d.id);
                        if (l0 && tpsr_proj_view(d, l0, 0, d.stream)) {
                            const int niq = d.tpsr_iqkv8r.rows + d.tpsr_iz8r.rows;
                            (void)q27_rb_tune_i8g(d.id, d.tpsr_iqkv8r.w, niq, Q27_HID, S0.xqt, Mp, S0.acc, d.stream);
                            (void)q27_rb_tune_i8g(d.id, d.tpsr_op8r.w, d.tpsr_op8r.rows, d.tpsr_op8r.K, S0.mixq_t, Mp, S0.acc, d.stream);
                        }
                        if (l3 && tpsr_proj_view(d, l3, 1, d.stream)) {
                            const int nqkv = d.tpsr_q8r.rows + d.tpsr_k8r.rows + d.tpsr_v8r.rows;
                            (void)q27_rb_tune_i8g(d.id, d.tpsr_q8r.w, nqkv, Q27_HID, S0.xqt, Mp, S0.acc, d.stream);
                            (void)q27_rb_tune_i8g(d.id, d.tpsr_o8r.w, d.tpsr_o8r.rows, d.tpsr_o8r.K, S0.mixq_t, Mp, S0.acc, d.stream);
                        }
                        CK(hipStreamSynchronize(d.stream));
                        if (g == 0) std::printf("Q27_TPSR_PROJ_RB: projection shapes tuned at M=%d (q|k|v %dx%d, iqkv|iz %dx%d, o/out %dx%d)\n", Mp,
                                                (Q27_QROWS + 2 * Q27_KVROWS) / ndev, Q27_HID,
                                                (Q27_GDN_QKV + Q27_GDN_Z) / ndev, Q27_HID,
                                                Q27_HID, Q27_OROWS / ndev);
                    }
                } else { std::fprintf(stderr, "Q27_TPSR: prime skipped on card %d (no layer-0 FP8 shard view)\n", g); }
                d.tpsr_primed = 1;
                C.b1.wait(); C.b2.wait();
            }
            if (g_ls && g_ls_q8 && q27_env_flag("Q27_Q8_PRIME", true)) {
                // ---- PRIME (outside the timed prefill): run every owned layer's rocBLAS GEMMs once on dummy input.
                // First-touch of the ~6.7 GB of int8 weights per card + rocBLAS first-call setup made the first
                // chunk's blocks take 36 ms instead of 18, i.e. the whole 4-card pipeline fill (3 block-times).
                const int LPPp = nlayer / ndev;
                int blkp = q27_env_int("Q27_LS_BLK", 2); if (blkp < 1 || blkp > LPPp || (LPPp % blkp)) blkp = LPPp;
                const int Mp = q27_env_int("Q27_LS_CHW", g_ls_q8 ? 256 : Q27_PF_TSLOT);   // tune at the chunk width (M) the sweep uses
                const Q8Scr& S0 = d.q8s[0];
                hipStream_t sp = d.hs[0];
                {   // random int8 activations for tuning/priming (constant data under-reports the dot4 power draw)
                    std::vector<signed char> rnd((size_t)Mp * Q27_INTER); unsigned st = 0x9E3779B9u ^ (unsigned)g;
                    for (size_t i = 0; i < rnd.size(); ++i) { st = st * 1664525u + 1013904223u; rnd[i] = (signed char)((int)(st >> 24) - 128); }
                    CK(hipMemcpy(S0.xq2, rnd.data(), rnd.size(), hipMemcpyHostToDevice));
                    CK(hipMemcpy(S0.xqt, rnd.data(), (size_t)Mp * Q27_HID, hipMemcpyHostToDevice));
                    CK(hipMemcpy(S0.xq1, rnd.data(), (size_t)Mp * Q27_HID, hipMemcpyHostToDevice));
                }
                auto prime1 = [&](const q27_i8r_t* w, const signed char* x) {
                    if (!(w && w->w && w->s)) return;
                    if (w->gs == w->K) (void)q27_rb_tune_i8g(d.id, w->w, w->rows, w->K, x, Mp, S0.acc, sp);   // once per shape: tune + prime
                    else (void)q27_rb_gemm_i8g(d.id, w->w, w->rows, w->K, x, Mp, S0.acc, w->gs, sp);
                };
                for (int L = 0; L < nlayer; ++L) {
                    { if (q27_ls_block_owner(L / blkp, ndev) != g) continue; }   // 2026-09-18: the oracle (was a sixth inline copy of the modulo)
                    const q27_layer_t* lay = g_sr ? sr_layer_view(d, m, L, sp) : q27_layer_ls(m, L, d.id);   // Q27_LS_SR: tune on the transient mirrors the sweep will read
                    if (!lay) continue;
                    if (lay->is_full) { const q27_i8r_t* w3[3] = { &lay->q8r, &lay->k8r, &lay->v8r }; const int n3 = q27_cat_rows(w3, 3);
                        if (n3 > 0 && lay->q8r.gs == lay->q8r.K) (void)q27_rb_tune_i8g(d.id, lay->q8r.w, n3, lay->q8r.K, S0.xqt, Mp, S0.acc, sp); else { prime1(&lay->q8r, S0.xqt); prime1(&lay->k8r, S0.xqt); prime1(&lay->v8r, S0.xqt); }
                        prime1(&lay->o8r, S0.xqt); }
                    else { const q27_i8r_t* w2[2] = { &lay->iqkv8r, &lay->iz8r }; const int n2 = q27_cat_rows(w2, 2);
                        if (n2 > 0 && lay->iqkv8r.gs == lay->iqkv8r.K) (void)q27_rb_tune_i8g(d.id, lay->iqkv8r.w, n2, lay->iqkv8r.K, S0.xqt, Mp, S0.acc, sp); else { prime1(&lay->iqkv8r, S0.xqt); prime1(&lay->iz8r, S0.xqt); }
                        prime1(&lay->op8r, S0.xqt); prime1(&lay->ab8r, S0.xqt); }
                    prime1(&lay->gate8r, S0.xq1); prime1(&lay->up8r, S0.xq1); prime1(&lay->down8r, S0.xq2);
                }
                CK(hipStreamSynchronize(sp));
                if (g == 0) MS("T5b_primed");
            }
            if (g_ls && g_ls_q8) { C.b1.wait(); C.b2.wait(); }   // every card primed/tuned before anyone starts the timed sweep
            if (g == 0) MS("T5c_all_primed");
            if (g_spec && g_mtp_pp && ((g_ls && g_ls_q8) || g_tpsr) && g_mtp_hall) {   // Q27_MTP_PP: mirrors + solutions before the clock starts (Q27_TPSR: every card, on the batched sweep)
                if ((!g_sr || g == g_sr_head) && !mtp_prompt_init(d, g, Q27_PF_TSLOT)) { std::fprintf(stderr, "Q27_MTP_PP init failed on card %d\n", g); std::exit(1); }
                C.b1.wait(); C.b2.wait();
            }
            const double t_sw0 = tp_now();
            if (g == 0) q27_transfer_receipt(req,true);
            C.b1.wait(); C.b2.wait();
            if (g == 0) std::fprintf(stderr, "Q27_PHASE req=%d phase=prefill edge=begin monotonic_ms=%.6f\n", req, t_sw0);
            if (g == 0) MS("T6_sweep_begin");
            const bool pf_batch = q27_env_flag("Q27_PF_BATCH", true) && nwin > 1 && g_rn_epi;
            const bool swpl = q27_env_flag("Q27_SWEEP_LADDER", false) && (orc != nullptr);
            {   // embedding for the sweep window: the host gathers bf16 rows into a PINNED staging buffer (grown once),
                // one H2D copy per card into hidden_slots (bf16), and card 0 expands it to fp32 pend_slots on the device.
                // (1023 synchronous per-row copies + a host double sumsq cost ~40 ms; pageable 31 MB copies + a host
                //  scalar convert cost 60 ms.)
                static const bool noinv_pre = q27_env_flag("Q27_Q8_NOINV", true);
                const size_t nsw = (size_t)(sw_end - sw_beg), nel = nsw * (size_t)Q27_HID;
                static const bool q8_old_res = (q27_env_int("Q27_Q8_OLD", 0) & 4) != 0;
                const int first_owner = q27_ls_block_owner(0, ndev);
                const bool need_bf16 = !(g_ls && g_ls_q8 && !q8_old_res) || g == first_owner;
                unsigned short* pin = d.emb_pin; static thread_local unsigned short* lazy = nullptr; static thread_local size_t lazy_cap = 0;
                if (!pin || d.emb_pin_cap < nel) { if (lazy_cap < nel) { if (lazy) CK(hipHostFree(lazy)); CK(hipHostMalloc((void**)&lazy, nel * 2, hipHostMallocDefault)); lazy_cap = nel; } pin = lazy; }
                if (need_bf16) {
                for (int p = sw_beg; p < sw_end; ++p) {
                    const unsigned ptok_src = sw_tok ? sw_tok[p - sw_lo] : prompt[p];
                    const unsigned etok = (ptok_src < (unsigned)Q27_VOCAB) ? ptok_src : (unsigned)(Q27_VOCAB - 1);
                    std::memcpy(pin + (size_t)(p - sw_beg) * Q27_HID, emb + (size_t)etok * Q27_HID, (size_t)Q27_HID * 2);
                }
                CK(hipMemcpyAsync(d.hidden_slots + (size_t)(sw_beg - sb) * Q27_HID, pin, nel * 2, hipMemcpyHostToDevice, d.stream));
                }
                if (g_ls && g_ls_q8 && g == first_owner) {
                    if (!q27_bf16_to_f32(d.hidden_slots + (size_t)(sw_beg - sb) * Q27_HID, d.pend_slots + (size_t)(sw_beg - sb) * Q27_HID, nel, d.stream)) { std::fprintf(stderr, "embedding expand declined\n"); std::exit(1); }
                    if (!noinv_pre) {   // the device-side input norm does not read inv_slots
                        CK(hipStreamSynchronize(d.stream));
                        for (int p = sw_beg; p < sw_end; ++p) {
                            const unsigned short* r = pin + (size_t)(p - sw_beg) * Q27_HID; double ss = 0.0;
                            for (int i = 0; i < Q27_HID; ++i) { const unsigned u = ((unsigned)r[i]) << 16; float f; std::memcpy(&f, &u, 4); ss += (double)f * (double)f; }
                            d.inv_slots[p - sb] = (float)(1.0 / std::sqrt(ss / (double)Q27_HID + (double)Q27_RMS_EPS));
                        }
                    }
                }
                CK(hipStreamSynchronize(d.stream));   // the ring loop's first job runs on hs[0]: the embedding must have landed
            }
            if (g == 0) MS("T6_embed_done");
            int gs = 0, fs = 0;
            if (g_ls) {
                // ---- LAYER-SPLIT SWEEP (Q27_LAYER_SPLIT): chunk-major, one card per layer group.
                // Card g owns FULL layers [L0,L1) (q27_layer_ls); no shard, no intra-layer
                // collective. The only cross-card data is the activation handoff: the previous
                // stage's mixer output (pend_slots + inv_slots) per chunk, copied with
                // hipMemcpyPeer right after the barrier round that follows the producer's ev_h
                // sync. Chunks == tiles (chw = Q27_PF_TSLOT) so the GDN tile prologue consumes
                // exactly one chunk's handoff. The consumer runs one chunk behind the producer
                // (the pipeline fill), hence nch+1 barrier rounds.
                const int LPP = nlayer / ndev;
                int q8blk = q27_ls_blk_eff(ndev);   // Q27_TP3: the same quota-aware oracle as the loader (4 at 20,20,24); rotating layer blocks (see q27_upload_ls)
                const int L0 = g * q8blk, L1 = (q8blk < LPP) ? L0 + q8blk : L0 + LPP;   // contiguous only when q8blk == LPP (the block loop below owns the general case)
                static const int q8bal = q27_env_int("Q27_LS_BAL", 0); (void)q8bal;   // applied inside q27_ls_block_owner
                // 2026-09-18: THE ORACLE, not a fifth copy of the modulo. This is the ring the
                // comment above is about, so it is the site that most needs to agree with the
                // residency planner -- if it does not, a card fetches a layer it does not hold.
                // Q27_LS_OWN's quota-skipping deal still advances one card per block, so the
                // conflict-free constant-shift property the ring depends on is preserved.
                auto blk_owner = [&](int b) { return q27_ls_block_owner(b, ndev); };
                auto lay_owner = [&](int L) { return blk_owner(L / q8blk); };
                const int last_card = blk_owner((nlayer - 1) / q8blk);                  // the card holding the last layer (finals)
                // Q27_LS_CHW: runtime chunk width (<= the 256-position buffer sizing). Smaller
                // chunks shrink the (nch+ndev-1)/nch pipeline fill at some GEMM M-efficiency cost.
                int chw = q27_env_int("Q27_LS_CHW", g_ls_q8 ? 256 : Q27_PF_TSLOT);   // Q8: 256-wide chunks with 2-layer ring blocks: 612 ms at 1K (128-wide: 648)
                if (chw < 1 || chw > Q27_PF_TSLOT) chw = Q27_PF_TSLOT;
                // Hybrid schedule: Q27_LS_CHW0 sets a smaller width for the first/last
                // Q27_LS_CHW0_N chunks. The pipeline fill and drain ride the edge chunks, so
                // shrinking them cuts the 3-stage tax without paying the M<128 GEMM penalty on
                // the steady-state middle.
                const int chw0 = q27_env_int("Q27_LS_CHW0", 0);
                const int nedge = (chw0 > 0 && chw0 < chw) ? q27_env_int("Q27_LS_CHW0_N", 2) : 0;
                const int nlead = nedge, ntail = q27_env_flag("Q27_LS_LEAD_ONLY", false) ? 0 : nedge;   // LEAD_ONLY: narrow first chunk(s) only (the fill is the first chunk's block time)
                const int tailw = q27_env_int("Q27_LS_TAILW", 0);   // narrow LAST chunk (drain = 3 x its time in theory): measured 1019 vs 964 ms at 1K -> off
                const int ntail2 = (tailw > 0 && tailw < chw && nwin > tailw + 2 * nedge * chw0) ? 1 : 0;
                const long long mid_tok = (long long)nwin - (long long)(nlead + ntail) * chw0 - (long long)ntail2 * tailw;
                const int nchmid = (mid_tok > 0) ? (int)((mid_tok + chw - 1) / chw) : 0;
                const int nch = nlead + nchmid + ntail + ntail2;
                auto lschw = [&](int k) {   // lead edges, mid chunks (the last mid may be partial), tail edges, then the narrow tail
                    if (nedge && k < nlead) return chw0;
                    if (ntail2 && k == nch - 1) return tailw;
                    if (nedge && k >= nch - ntail2 - ntail) return chw0;
                    const long long rem = mid_tok - (long long)(k - nlead) * chw;
                    return (int)((rem < chw) ? rem : chw);
                };
                auto lsoff = [&](int k) { int off = 0; for (int i = 0; i < k; ++i) off += lschw(i); return off; };
                const int QLb = Q27_QROWS, KVLb = Q27_KVROWS;      // nd=1: full projection widths
                if (q27_env_flag("Q27_LS_ADDRS", false)) {
                    std::fprintf(stderr, "LS_ADDRS g=%d kc=%p vc=%p tp_kv=%zu dev=%d\n", g, (const void*)d.kc, (const void*)d.vc, d.tp_kv, d.id);
                    std::fprintf(stderr, "LS_ADDRS g=%d pend=%p S=%p mixer=%p part=%p qkv=%p zbuf=%p ab=%p bb=%p mix=%p mixq=%p mixs=%p mixqs=%p xqin=%p xq1=%p xq2=%p pa=%p pb=%p kc=%p vc=%p conv=%p nslots=%p\n",
                        g, (const void*)d.pend_slots, (const void*)d.S, (const void*)d.mixer_slots,
                        (const void*)d.part_slots, (const void*)d.qkv_slots, (const void*)d.zbuf_slots,
                        (const void*)d.ab_tile, (const void*)d.bb_tile, (const void*)d.mix_tile,
                        (const void*)d.mixq_tile, (const void*)d.mixs_tile, (const void*)d.mixq_slots,
                        (const void*)d.xqin_slots, (const void*)d.xq1_slots, (const void*)d.xq2_slots,
                        (const void*)d.pa_mlp, (const void*)d.pb_mlp, (const void*)d.kc, (const void*)d.vc,
                        (const void*)d.conv, (const void*)d.norm_slots);
                }
                if (q27_env_int("Q27_LS_SCANPROBE", 0) > 0) {
                    // in-engine probe: one nd=1 scan at Ct=N on dummy data, before the sweep
                    const int Cprobe = q27_env_int("Q27_LS_SCANPROBE", 256);
                    const q27_layer_t* lay0 = q27_layer_ls(m, L0, d.id);
                    CK(hipMemset(d.qkv_slots, 0x3c, (size_t)Q27_PF_CH * Q27_GDN_QKV * 2));
                    CK(hipMemset(d.zbuf_slots, 0x3c, (size_t)Q27_PF_CH * Q27_GDN_Z * 2));
                    CK(hipMemset(d.ab_tile, 0x3c, (size_t)Q27_PF_SLOTS * Q27_GDN_VH * 2));
                    CK(hipMemset(d.bb_tile, 0x3c, (size_t)Q27_PF_SLOTS * Q27_GDN_VH * 2));
                    CK(hipMemset(d.S, 0, d.s_off[47] + d.s_bytes[47] / 4 * 4));   // the actual S allocation, which is no longer 48 * tp_s
                    CK(hipMemset(d.norm_slots, 0x3c, (size_t)Q27_PF_SLOTS * Q27_HID * 2));
                    q27_bf16_gemv2_tile(lay0->in_a, lay0->in_b, d.norm_slots, Q27_HID, d.ab_tile, d.bb_tile,
                                        Q27_GDN_VH, Q27_HID, Cprobe, d.stream);
                    q27_gdn_conv_tp_tile(d.qkv_slots, Q27_GDN_QKV, Cprobe, d.conv, lay0->conv1d, 0, 1, d.stream);
                    q27_gdn_scan2_tp(d.qkv_slots, Q27_GDN_QKV, d.zbuf_slots, Q27_GDN_Z,
                                     d.ab_tile, d.bb_tile, Q27_GDN_VH, lay0->A_log, lay0->dt_bias, lay0->gdn_norm,
                                     d.S, d.mix_tile, Q27_GDN_Z, 0, 1, Cprobe, d.mixq_tile, d.mixs_tile,
                                     lay0->out_proj.in_scale, d.stream);
                    CK(hipStreamSynchronize(d.stream));
                    std::printf("Q27_LS_SCANPROBE ok (nd=1 Ct=%d)\n", Cprobe);
                }
                hipEvent_t ev_h[2];
                CK(hipEventCreateWithFlags(&ev_h[0], hipEventDisableTiming));
                CK(hipEventCreateWithFlags(&ev_h[1], hipEventDisableTiming));
                // Barrier protocol: the producer's publish of chunk k (ev_h sync after the
                // stage) precedes round k+2; the consumer's copy of chunk k follows round k+2 and
                // precedes ITS stage of chunk k. With one round per iteration per thread this is
                // the same round for both edges: thread g's chunk k lives at iteration k+1+g.
                // Q27_LS_SERIAL=1 (diagnostic): stages run fully sequentially -- thread g waits g
                // rounds, runs its whole sweep, then closes the remaining rounds.
                const bool ls_serial = q27_env_flag("Q27_LS_SERIAL", false);
    const bool g_ls_w8    = q27_env_flag("Q27_LS_W8", false);   // w8-bypass GEMMs for gate/up/down
    const bool g_ls_pf    = q27_env_flag("Q27_LS_PF", false);   // weight-prefetch rb GEMMs
    const bool g_ls_pf2   = q27_env_flag("Q27_LS_PF2", false);  // + double-buffered act staging (gu)
    const bool g_ls_pf5   = q27_env_flag("Q27_LS_PF5", false);  // TR=32 down specialization
    const bool g_ls_pf9   = q27_env_flag("Q27_LS_PF9", false);  // barrier-free register-pipelined gu
    const bool g_ls_pf7   = q27_env_flag("Q27_LS_PF7", false);  // fused q/k/v launch
    const int  g_ls_pf8   = q27_env_int("Q27_LS_PF8", 0);     // split-K (2 or 4) for in_qkv/in_z/out_proj
    const int  g_ls_pf6   = q27_env_int("Q27_LS_PF6", 0);     // split-K down: 2 or 4 partials
    const int  g_ls_w8_tcb = q27_env_int("Q27_LS_W8_TCB", 8);
    const int  g_ls_pf_tcbd = q27_env_int("Q27_LS_PF_TCBD", g_ls_w8_tcb);   // down's token column tile
    const bool g_ls_nv    = q27_env_flag("Q27_LS_NV", false);   // NVFP4 twins for the fp8 projection sites
    const bool g_ls_nv_qkv = q27_env_flag("Q27_LS_NV_QKV", g_ls_nv);   // attention q/k/v
    const bool g_ls_nv_iq  = q27_env_flag("Q27_LS_NV_IQKV", g_ls_nv);  // GDN in_qkv/in_z
    const bool g_ls_nv_op  = q27_env_flag("Q27_LS_NV_OP", g_ls_nv);    // out_proj + o_proj
    const int  g_ls_nv_ki = q27_env_int("Q27_LS_NV_KI", g_pf_wide_kd);
                if (ls_serial) for (int i = 0; i < g; ++i) { C.b1.wait(); C.b2.wait(); }
                const int nit = nch + ndev;
                // Q27_LS_TIME: per-phase attribution (drain-based, diagnostic only -- the
                // stream drains distort absolute wall, so totals are never performance numbers).
                static double ls_ph[Q27_MAX_DEVICES][16];
                static bool ls_time = q27_env_flag("Q27_LS_TIME", false);
                double ls_t0 = tp_now();
                auto lsp = [&](int ph) { if (ls_time) { CK(hipStreamSynchronize(d.stream));
                    const double t1 = tp_now(); ls_ph[g][ph] += t1 - ls_t0; ls_t0 = t1; } };
                // ---- Q27_LS_Q8: one layer of one chunk on the quantized producer/consumer graph ----
                // Representations: fp32 residual stream (pend_slots in, part_slots after attention/GDN,
                // pend_slots out); int8 + per-16 fp32 scales for every inter-operator activation
                // (input norm -> projections -> conv/scan/attention -> out/o_proj; post-norm -> gate/up;
                // swiglu -> down); int8 KV cache with per-16 scales; fp32 GDN state S and fp32 a/b; the
                // conv's 4-tap history bf16 only at the chunk boundary (the decode conv's ABI). No bf16
                // activation is written anywhere in this body.
                // Q27_Q8_OLD (diagnostic bisect mask): 1 = legacy bf16 attention block, 2 = legacy bf16 GDN
                // block, 4 = legacy bf16 residual copies + radd, 8 = legacy finals (needs 4), 16 = legacy
                // per-position layer-0 prologue. Every combination produces the same stage-boundary
                // representations, so any mix runs.
                static const int q8old = q27_env_int("Q27_Q8_OLD", 0);
                static const int q8ki = q27_env_int("Q27_I8_KI", 168);   // int8-mirror projection tile (64-row/KI8 = 88 measured worse: LDS-bound)
                static const int q8kd = q27_env_int("Q27_Q8_KD", 168);   // Q8 down tile (row-major fallback)
                static const bool q8rb = q27_env_flag("Q27_Q8_RB", true);       // register-blocked int8 projections (default; measured -98 ms/card of GPU time at 1K)
                static const bool q8rbd = q27_env_flag("Q27_Q8_RB_DOWN", false); // register-blocked down: measured SLOWER than the row-major KI16 form (+78 ms/card), off
                static const bool q8g64 = q27_env_flag("Q27_Q8_G64", true);     // int8 per-64-group MLP with int32 accumulation inside the group (default)
                // Q27_LS_MLP_NV: run the MLP on the card's NVFP4 MLP weights instead of per-64 int8
                // mirrors. The mirrors cost 4.27 GB/card (16 owned layers x gate/up/down at FULL width);
                // the nvfp4 originals they replace cost 2.14 GB/card, so the layer-split residency drops
                // ~2.1 GB/card -- which is what lets the layer-split KV fit at ctx=262144. Only the MLP is
                // decoupled: the projections keep their int8 mirrors (0.42 GB) because the wide int8
                // GEMMs are what makes the layer-split sweep fast. The MLP runs the nvfp4 kernels the TP
                // prefill already uses (q27_wide_gu / q27_wide_down).
                static const bool q8mlp = q8g64 && !q27_env_flag("Q27_LS_MLP_NV", false);
                // Q27_LS_MLP_STAGE: expand ONLY the current layer's MLP into the reusable arena,
                // then run the same int8 consumers the frozen fast path used. Neither extreme:
                // not 4.23 GiB of permanent mirrors, and not decoding NVFP4 inside every tile.
                static const int  mstage_mode = q27_env_int("Q27_LS_MLP_STAGE", 0);
                static const bool mlpstage = mstage_mode != 0;
                // Q27_LS_MLP_RB: run THIS layer's MLP through the rocBLAS/Tensile int8 GEMMs, with
                // the weights in the layer-local arena requantized in place to per-ROW scales. Same
                // consumer and same representation as the persistent-mirror bl_mlp path, which is
                // the only thing the 1263-class engine ever needed that the 262144 budget could not
                // hold: 4.23 GiB/card of mirrors. The arena is 1.07 GiB and holds four layers, which
                // is exactly the ring's reuse distance, so each layer is expanded once per prefill.
                static const bool mlp_rb = q27_env_int("Q27_LS_MLP_RB", 0) != 0;
                static const bool q8r8 = q27_env_flag("Q27_Q8_R8", false);       // 8x4 register tile: measured 2214 vs 1498 ms (register pressure); off
                static const bool q8bl = q27_env_flag("Q27_Q8_RB", true);        // stage 3: rocBLAS int8 GEMMs (1024-group scales both operands, int32 slabs) + fused epilogues
                static const int q8blm = q27_env_int("Q27_Q8_RBM", 7);            // which GEMM families take the rocBLAS path: 1 attention q/k/v/o, 2 GDN in/out, 4 MLP
                // bl_mlp (the rocBLAS-blocked MLP) reads the PER-ROW int8 MLP mirrors, which only exist on
                // the full-int8-mirror configuration. With Q27_LS_MLP_NV the MLP runs on the nvfp4 originals,
                // so the blocked path must be off with it -- otherwise the sweep declines its first gemm and
                // aborts ("rb gate gemm declined (L0 Cch=8)").
                const bool bl_att = q8bl && (q8blm & 1), bl_gdn = q8bl && (q8blm & 2), bl_mlp = q8mlp && q8bl && (q8blm & 4);
                static const bool q8noinv = q27_env_flag("Q27_Q8_NOINV", true);   // input norm computes its sum of squares on the device: no host inv handoff, no second norm launch
                static const bool q8redb1 = q27_env_flag("Q27_Q8_REDB1", false);  // keep the red_b1 copy+sumsq kernel (control); default: the down epilogue writes pend directly
                static const int q8gs = q27_env_int("Q27_Q8_GS", 0);              // K-group size of the rocBLAS path: 0 = full K (per-row / per-token scales, one call per GEMM: 1002 ms) or 1024 (1255 ms, same coherence)
                static const bool q8trace = q27_env_flag("Q27_Q8_TRACE", false);
                // host reference for out_proj row 0 / position 0 from the dumped operands (Q27_Q8_DBG, L==0)
                auto opref = [&](int L, const q27_nvfp4_t* w4, const q27_fp8_t* w8, const signed char* mq, const float* ms, const float* part, const float* pendv) {
                    if (!q27_env_flag("Q27_Q8_DBG", false) || g != 0 || L != 0) return;
                    CK(hipStreamSynchronize(d.stream));
                    const int K = w4->K, NG = K >> 4;
                    std::vector<signed char> xq(K); std::vector<float> xs(NG);
                    std::vector<unsigned char> wp((size_t)K >> 1), gs(NG), w8r(K);
                    CK(hipMemcpy(xq.data(), mq, K, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(xs.data(), ms, (size_t)NG * 4, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(wp.data(), w4->w, (size_t)K >> 1, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(gs.data(), w4->gs, NG, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(w8r.data(), w8->w, K, hipMemcpyDeviceToHost));
                    auto e4m3 = [](unsigned b) { const int s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7; float v;
                        if (e == 0) v = (float)m / 8.0f * std::ldexp(1.0f, -6); else v = (1.0f + (float)m / 8.0f) * std::ldexp(1.0f, e - 7);
                        return s ? -v : v; };
                    static const float e2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
                    double y4 = 0.0, y8 = 0.0, yperm = 0.0;
                    for (int gi = 0; gi < NG; ++gi) {
                        const float gsv = e4m3(gs[gi]);
                        for (int j = 0; j < 16; ++j) {
                            const unsigned nib = (j & 1) ? (wp[gi * 8 + (j >> 1)] >> 4) : (wp[gi * 8 + (j >> 1)] & 15);
                            const float wv = e2m1[nib & 7] * ((nib & 8) ? -1.f : 1.f) * gsv * w4->ws2;
                            const int pj = (j & 1) ? (8 + (j >> 1)) : (j >> 1);        // where element j sits in the permuted activation
                            const float xperm = (float)xq[gi * 16 + pj] * xs[gi] * w4->in_scale;
                            const float xnat  = (float)xq[gi * 16 + j]  * xs[gi] * w4->in_scale;
                            y4 += (double)wv * (double)xperm;
                            yperm += (double)wv * (double)xnat;
                            y8 += (double)e4m3(w8r[gi * 16 + j]) * (double)w8->wscale * (double)xperm;
                        }
                    }
                    float p0, pd0; CK(hipMemcpy(&p0, part, 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(&pd0, pendv, 4, hipMemcpyDeviceToHost));
                    std::fprintf(stderr, "Q8DBG opref K=%d rows=%d in_scale=%g/%g ws2=%g wscale=%g | twin(perm x)=%.6g twin(nat x)=%.6g fp8w(perm x)=%.6g | kernel part0-pend0=%.6g\n",
                                 K, w4->rows, (double)w4->in_scale, (double)w8->in_scale, (double)w4->ws2, (double)w8->wscale, y4, yperm, y8, (double)(p0 - pd0));
                };
                static const bool q8dbg = q27_env_flag("Q27_Q8_DBG", false);
                auto dumpf = [&](const char* tag, int L, const float* p, int n) {
                    if (!q8dbg || g != 0 || L > 3) return;
                    CK(hipStreamSynchronize(d.stream));
                    float v[8]; if (n > 8) n = 8;
                    CK(hipMemcpy(v, p, (size_t)n * 4, hipMemcpyDefault));
                    std::fprintf(stderr, "Q8DBG L=%d %-10s", L, tag);
                    for (int i = 0; i < n; ++i) std::fprintf(stderr, " %.5g", (double)v[i]);
                    std::fprintf(stderr, "\n");
                };
                auto dumpq = [&](const char* tag, int L, const signed char* p, const float* ps, int n) {
                    if (!q8dbg || g != 0 || L > 3) return;
                    CK(hipStreamSynchronize(d.stream));
                    signed char v[8]; float s0 = 0.f; if (n > 8) n = 8;
                    CK(hipMemcpy(v, p, (size_t)n, hipMemcpyDeviceToHost));
                    if (ps) CK(hipMemcpy(&s0, ps, 4, hipMemcpyDeviceToHost));
                    std::fprintf(stderr, "Q8DBG L=%d %-10s", L, tag);
                    for (int i = 0; i < n; ++i) std::fprintf(stderr, " %d", (int)v[i]);
                    std::fprintf(stderr, "  scale0=%.5g\n", (double)s0);
                };
                auto ls_q8_layer = [&](int L, int p0, int Cch, const Q8Scr& S, hipStream_t sk, int par, hipEvent_t evA, int stage) {   // stage: 0 = A (norm..o/out_proj), 1 = B (post-norm..red_b1), 2 = both
                    const int fsl = L >> 2;
                    const int gsl = L - fsl;
                    const q27_layer_t* lay = g_sr ? sr_layer_view(d, m, L, sk) : q27_layer_ls(m, L, d.id);   // Q27_LS_SR: this layer's transient mirrors, materialised on the compute stream
                    const q27_layer_t* nxt = (L + 1 < L1) ? q27_layer_ls(m, L + 1, d.id) : nullptr;
                    const int isf = Q27_IS_FULL(L);
                    const float in_scl = isf ? lay->q_proj.in_scale : lay->in_qkv.in_scale;
                    float* const pend = d.pend_slots + (size_t)(p0 - sb) * Q27_HID;   // fp32 layer input = the residual
                    unsigned short* const hid_bf = d.hidden_slots + (size_t)(p0 - sb) * Q27_HID;   // legacy bf16 residual copy (Q27_Q8_OLD&4)
                    auto p2dump = [&](const Q8Scr& S) {
                        if (!q8trace || g != 1) return;
                        const int fsl = L >> 2, gsl = L - fsl;
                        std::fprintf(stderr, "Q8P1 L=%d in_norm=%p post_norm=%p in_a=%p in_b=%p iqkv8r=%p iz8r=%p ab8r=%p op8r=%p gate8r=%p up8r=%p down8r=%p q8r=%p o8r=%p iqkv8g=%p iz8g=%p gate8g=%p down8g=%p q_proj.w=%p\n",
                            L, (const void*)lay->input_norm, (const void*)lay->post_norm, (const void*)lay->in_a, (const void*)lay->in_b,
                            (const void*)lay->iqkv8r.w, (const void*)lay->iz8r.w, (const void*)lay->ab8r.w, (const void*)lay->op8r.w,
                            (const void*)lay->gate8r.w, (const void*)lay->up8r.w, (const void*)lay->down8r.w,
                            (const void*)lay->q8r.w, (const void*)lay->o8r.w, (const void*)lay->iqkv8g.w, (const void*)lay->iz8g.w,
                            (const void*)lay->gate8.w, (const void*)lay->down8.w, (const void*)lay->q_proj.w);
                    };
                    const bool old_attn = (q8old & 1), old_gdn = (q8old & 2), old_res = (q8old & 4), old_l0 = (q8old & 16), old_mlp = (q8old & 32);
                    // OWNERSHIP IS PART OF THE DIAGNOSIS. "declined" on a layer this card does
                    // not own is not a kernel fault, it is the sweep enumerating layers by the
                    // SYMMETRIC range [g*q8blk, ...) while residency was dealt by the oracle.
                    auto die = [&](const char* what) {
                        std::fprintf(stderr, "Q27_LS_Q8: %s declined (L%d Cch=%d) card=%d owner=%d owned=%d; aborting\n",
                                     what, L, Cch, g, q27_ls_block_owner(L / q8blk, ndev),
                                     q27_ls_owned(L, g, ndev));
                        std::exit(1); };
                    auto tr = [&](int L2, int ph) { if (pf_profile_on()) g_pfp[g].mark(sk, L2, ph); if (q8trace) { CK(hipStreamSynchronize(sk)); std::fprintf(stderr, "Q8T g=%d L=%d ph=%d ok\n", g, L2, ph); } };
                    // stage codes: 0/2 = A (whole), 1 = MLP, 10 = A1 (norm + projections), 11 = B (attention / conv + scan), 12 = A2 (o/out_proj + MLP)
                    const bool doA1 = (stage == 0 || stage == 2 || stage == 10), doB = (stage == 0 || stage == 2 || stage == 11);
                    const bool doA2 = (stage == 0 || stage == 2 || stage == 12), doMLP = (stage == 1 || stage == 2 || stage == 12);
                    static const bool q8fusepn = q27_env_flag("Q27_Q8_FUSEPN", true);
                    const bool fuse_pn = q8fusepn && stage == 2 && bl_mlp && q8noinv && q8gs == 0 && (isf ? bl_att : bl_gdn) && !old_res;   // fused o/out_proj epilogue + post-norm: whole-layer execution only
                    if (doA1 || doB || doA2) {
                    if (doA1) {
                    // input norm: fp32 residual in, int8 natural-order activations + scales out (no bf16 copies)
                    if (old_l0 && L == 0) {
                        unsigned short* ynorm = isf ? nullptr : d.norm_slots;
                        for (int c = 0; c < Cch; ++c) {
                            const int p = p0 + c;
                            double mark = tp_now();
                            run_layer_tp(d, lay, p, gsl, fsl, ctx, 0, 1, C, &mark, nullptr, nxt, p - sb,
                                         d.inv_slots + (p - sb), nullptr, nullptr, false, 1,
                                         S.xqin + (size_t)c * Q27_HID,
                                         S.xsin + (size_t)c * (Q27_HID / 16),
                                         nullptr, ynorm ? (ynorm + (size_t)c * Q27_HID) : nullptr);
                        }
                        if (old_res) {   // the legacy residual copy for this layer is the bf16 embedding already in hidden_slots
                        } else {
                            // the fp32 residual for the rest of the graph: pend must hold the embedding (it does: the host wrote it)
                        }
                    } else
                    // input norm: fp32 residual in; int8 NATURAL-order activations + per-16 scales out (the wide
                    // fp8 GEMM family's order; the a/b GEMV reads the same buffer). No bf16 copies unless a
                    // diagnostic legacy stage needs its bf16 residual copy / normalized vector.
                    if (q8g64 && !old_res && !old_gdn && !old_attn) {   // stage 2: per-64 natural-order activations for the int8/64 projections
                        if (!(q8noinv && bl_att && bl_gdn && lay->ab8r.w) && !q27_rmsnorm_hostss_g64_b(pend, lay->input_norm, S.xqin, S.xsin, Q27_HID, 1, in_scl, d.inv_slots + (p0 - sb), Cch, sk)) die("input norm g64");   // dead when every consumer is on rocBLAS
                        if ((bl_att || bl_gdn) && !(q8noinv ? q27_red_rn_tok_b1(pend, lay->input_norm, nullptr, S.xqt, S.xst, Q27_HID, 1, in_scl, Cch, q8gs, sk) : q27_rmsnorm_hostss_tok_b(pend, lay->input_norm, S.xqt, S.xst, Q27_HID, 1, in_scl, d.inv_slots + (p0 - sb), Cch, q8gs, sk))) die("input norm tok");   // q8noinv: sumsq on the device, no host inv
                    } else
                    if (!q27_rmsnorm_hostss_fp8_b(pend, lay->input_norm, old_res ? hid_bf : nullptr, old_gdn ? d.norm_slots : nullptr,
                                                  S.xqin, S.xsin, Q27_HID, 1, in_scl, d.inv_slots + (p0 - sb), Cch, sk)) die("input norm");
                    }   // doA1: input norm
                    tr(L, 0); dumpf("pend_in", L, pend, 4); dumpf("inv", L, d.inv_slots + (p0 - sb), 1); dumpq("xqin", L, S.xqin, S.xsin, 8);
                    if (isf) {
                        if (q8trace && g == 1) {
                            std::fprintf(stderr, "Q8P1F L=%d q8r=%p k8r=%p v8r=%p o8r=%p q8=%p k8=%p v8=%p o8=%p q8g=%p k8g=%p v8g=%p o8g=%p q_proj.w=%p o_proj.w=%p q_norm=%p k_norm=%p qkva8=%p qkvas=%p qh=%p\n",
                                L, (const void*)lay->q8r.w, (const void*)lay->k8r.w, (const void*)lay->v8r.w, (const void*)lay->o8r.w,
                                (const void*)lay->q8.w, (const void*)lay->k8.w, (const void*)lay->v8.w, (const void*)lay->o8.w,
                                (const void*)lay->q8g.w, (const void*)lay->k8g.w, (const void*)lay->v8g.w, (const void*)lay->o8g.w,
                                (const void*)lay->q_proj.w, (const void*)lay->o_proj.w, (const void*)lay->q_norm, (const void*)lay->k_norm,
                                (const void*)S.qkva8, (const void*)S.qkvas, (const void*)S.qh);
                        }
                        // OLb IS A STRIDE, NOT A WIDTH. It is passed as `ostride` to the attention
                        // chunk, which writes the mixer output at `Q_t + c * ostride` (q8 branch) and
                        // at `out_t + c * ostride` (legacy branch). Both of those buffers are sized
                        // `Q27_GDN_Z / pf_div` wide, and pf_div is 1 under layer-split but **ndev**
                        // without it (q27_main.cpp:1571) -- so with Q27_LAYER_SPLIT=0 the full-width
                        // 6144 strides a 1536-wide allocation and walks off the end of it. That is
                        // the boot-prefix `Memory access fault … on address 0x66000` (measured
                        // 2026-09-16 01:33, batched + KVP2P + KEEP_FP8): 4x overrun per chunk row, on
                        // the very buffer the o_proj consumes next. The sibling batched path at
                        // :6797 documents the same identity -- `OLb == (Q27_GDN_Z / ndev)`.
                        // Q27_OROWS == Q27_GDN_Z == 6144, so this is unchanged under layer-split.
                        const int OLb = g_ls ? Q27_OROWS : (Q27_OROWS / ndev);
                        const int qstride = QLb + 2 * KVLb;
                        if (old_attn) {   // legacy: fp8 wide q/k/v with bf16 out -> bf16 chunk attention
                            if (!(q27_proj_fp8_wideb(&lay->q_proj, S.xqin, S.xsin, d.qkva_slots, Cch, qstride, 8, sk) &&
                                  q27_proj_fp8_wideb(&lay->k_proj, S.xqin, S.xsin, d.qkva_slots + QLb, Cch, qstride, 8, sk) &&
                                  q27_proj_fp8_wideb(&lay->v_proj, S.xqin, S.xsin, d.qkva_slots + QLb + KVLb, Cch, qstride, 8, sk))) die("q/k/v (legacy fp8)");
                            tr(L, 1);
                            if (!q27_attn_chunk_tp(d.qkva_slots, qstride, lay->q_norm, lay->k_norm,
                                                   d.kc + d.kv_off[fsl], d.ks + d.kvs_off[fsl],
                                                   d.vc + Q27_VOFF(d.kv_off[fsl]), d.vs + Q27_VSOFF(d.kvs_off[fsl]),
                                                   S.qh, d.mix_tile, OLb, p0, Cch, 1,
                                                   S.mixq_slots, S.mixs_slots, lay->o_proj.in_scale, g_att_hpw, g_att_pf,
                                                   S.pob, S.pml, sk)) die("attention chunk (legacy)");
                        } else {          // q/k/v -> int8 + per-16 scales; attention consumes them and the int8 KV cache directly
                            if (doA1) {   // q/k/v: ONE launch; the 1024-row k/v matrices ride inside the q grid (grid.z)
                                const q27_nvfp4_t* wq3[3] = { &lay->q8, &lay->k8, &lay->v8 };
                                signed char* yq3[3] = { S.qkva8, S.qkva8 + QLb, S.qkva8 + QLb + KVLb };
                                float* ys3[3] = { S.qkvas, S.qkvas + QLb / 16, S.qkvas + (QLb + KVLb) / 16 };
                                const int st3[3] = { qstride, qstride, qstride };
                                const q27_i8g_t* wq3g[3] = { &lay->q8g, &lay->k8g, &lay->v8g };
                                if (bl_att) { const q27_i8r_t* w3[3] = { &lay->q8r, &lay->k8r, &lay->v8r }; int* accp = S.acc; const int w_gs = lay->q8r.gs;
                                    const int ntot3 = q27_cat_rows(w3, 3);   // > 0: the three mirrors are one contiguous buffer -> ONE GEMM
                                    if (ntot3 > 0 && w_gs == lay->q8r.K) {
                                        if (!q27_rb_gemm_i8g(d.id, w3[0]->w, ntot3, w3[0]->K, S.xqt, Cch, S.acc, w_gs, sk)) die("rb qkv gemm (fused)");
                                        int off = 0; for (int i = 0; i < 3; ++i) { const q27_i8r_t* w = w3[i];
                                            if (!q27_epi_q8v2(S.acc + off, ntot3, w->rows, w->s, 1, S.xst, w->alpha, yq3[i], ys3[i], st3[i], Cch, sk)) die("rb qkv epi (fused)");
                                            off += w->rows; }
                                    } else
                                    for (int i = 0; i < 3; ++i) { const q27_i8r_t* w = w3[i];
                                        if (!q27_rb_gemm_i8g(d.id, w->w, w->rows, w->K, S.xqt, Cch, accp, w_gs, sk)) die("rb qkv gemm");
                                        if (!q27_epi_q8v2(accp, w->rows, w->rows, w->s, w->ng, S.xst, w->alpha, yq3[i], ys3[i], st3[i], Cch, sk)) die("rb qkv epi");
                                        accp += (size_t)Cch * w->rows * w->ng; } }
                                else                                 if (!(q8g64 ? (q8r8 ? q27_wide_i8g64r8_q8m(3, wq3g, S.xqin, S.xsin, yq3, ys3, st3, Cch, sk) : q27_wide_i8g64_q8m(3, wq3g, S.xqin, S.xsin, yq3, ys3, st3, Cch, sk))
                                           : q8rb ? q27_wide_i8_rb_q8m(3, wq3, S.xqin, S.xsin, yq3, ys3, st3, Cch, sk)
                                           : q27_wide_i8_q8m(3, wq3, S.xqin, S.xsin, yq3, ys3, st3, Cch, q8ki, sk))) die("q/k/v");
                            }
                            tr(L, 1);
                            if (doB) {
                            // Q27_PF_P2P: the four KV head-quarters of this layer, one per card.
                            // Peer access is already enabled, so the sweep reads the three remote
                            // quarters in place instead of the owner keeping a full-width copy.
                            q27_kvquad_t QQ; QQ.on = 0;
                            for (int h = 0; h < 4; ++h) { QQ.kc[h] = nullptr; QQ.ks[h] = nullptr; QQ.vc[h] = nullptr; QQ.vs[h] = nullptr; }
                            if (g_pf_kvp2p) { QQ.on = 1;
                                for (int h = 0; h < ndev; ++h) {
                                    QQ.kc[h] = D[h].kc + D[h].kv_off[fsl]; QQ.ks[h] = D[h].ks + D[h].kvs_off[fsl];
                                    QQ.vc[h] = D[h].vc + Q27_VOFF(D[h].kv_off[fsl]); QQ.vs[h] = D[h].vs + Q27_VSOFF(D[h].kvs_off[fsl]); } }
                            int pfblas_rc = 0;
                            if (q27_pfblas_should_run(p0, Cch)) {
                                static const bool pfblas_serial = !q27_env_flag("Q27_Q8_SPLIT", false);
                                pfblas_rc = q27_attn_chunk_q8_pfblas(d.id, pfblas_serial,
                                    S.qkva8, S.qkvas, qstride, lay->q_norm, lay->k_norm,
                                    d.kc + d.kv_off[fsl], d.ks + d.kvs_off[fsl],
                                    d.vc + Q27_VOFF(d.kv_off[fsl]), d.vs + Q27_VSOFF(d.kvs_off[fsl]),
                                    S.qh, OLb, p0, Cch, S.mixq_slots, S.mixs_slots, lay->o_proj.in_scale,
                                    S.pob, S.pob_bytes, q8g64 ? 1 : 0, sk, &QQ);
                                if (pfblas_rc < 0) die("bounded FP32 attention");
                            }
                            if (!pfblas_rc && !q27_attn_chunk_q8(S.qkva8, S.qkvas, qstride, lay->q_norm, lay->k_norm,
                                                   d.kc + d.kv_off[fsl], d.ks + d.kvs_off[fsl],
                                                   d.vc + Q27_VOFF(d.kv_off[fsl]), d.vs + Q27_VSOFF(d.kvs_off[fsl]),
                                                   S.qh, nullptr, OLb, p0, Cch, 1,
                                                   S.mixq_slots, S.mixs_slots, lay->o_proj.in_scale, g_att_hpw, g_att_pf,
                                                   S.pob, S.pml, 0, q8g64 ? 1 : 0, sk, &QQ)) die("attention chunk");
                            }   // doB: attention
                        }
                        tr(L, 2);
                        // o_proj on the fp8 weights; residual = the fp32 layer input (or the bf16 copy under the legacy switch)
                        if (doA2) {
                        if (old_res) { if (!q27_proj_fp8_wide(&lay->o_proj, S.mixq_slots, S.mixs_slots, S.part, Cch, Q27_HID, hid_bf, Q27_HID, 1.0f, 8, sk)) die("o_proj (bf16 radd)"); }
                        else if (bl_att) { const q27_i8r_t* w = &lay->o8r; const int w_gs = w->gs;
                            if (!q27_requant_g64_tok(S.mixq_slots, S.mixs_slots, S.mixq_t, S.mixs_t, w->K, w->K, Cch, q8gs, sk)) die("rb o requant");
                            if (!q27_rb_gemm_i8g(d.id, w->w, w->rows, w->K, S.mixq_t, Cch, S.acc, w_gs, sk)) die("rb o gemm");
                            if (fuse_pn) { if (!q27_epi_res_rn_tok(S.acc, w->rows, w->s, w->ng, S.mixs_t, w->alpha, pend, S.part, lay->post_norm, (L == nlayer - 1) ? S.hid32c : nullptr, S.xq1, S.xst1, 1, lay->gate.in_scale, Cch, sk)) die("rb o epi+norm"); }
                            else if (!q27_epi_f32r(S.acc, w->rows, w->s, w->ng, S.mixs_t, w->alpha, S.part, pend, 1.0f, Cch, sk)) die("rb o epi"); }
                        else if (q8g64) { if (!(q8r8 ? q27_wide_i8g64r8_f32r(&lay->o8g, S.mixq_slots, S.mixs_slots, S.part, Cch, pend, 1.0f, sk) : q27_wide_i8g64_f32r(&lay->o8g, S.mixq_slots, S.mixs_slots, S.part, Cch, pend, 1.0f, sk))) die("o_proj g64"); }
                        else if (!(q8rb ? q27_wide_i8_rb_f32r(&lay->o8, S.mixq_slots, S.mixs_slots, S.part, Cch, pend, 1.0f, sk)
                                        : q27_wide_i8_f32r(&lay->o8, S.mixq_slots, S.mixs_slots, S.part, Cch, pend, 1.0f, q8ki, sk))) die("o_proj");
                        }   // doA2: o_proj
                        tr(L, 3); dumpf("part", L, S.part, 4);
                    } else {
                        const int QKVLb = Q27_GDN_QKV, ZLb = Q27_GDN_Z;
                        if (old_gdn) {    // legacy: fp8 wide in_qkv/in_z (bf16 out), bf16 gemv/conv/scan
                            if (!(q27_proj_fp8_wideb(&lay->in_qkv, S.xqin, S.xsin, d.qkv_slots, Cch, QKVLb, 8, sk) &&
                                  q27_proj_fp8_wideb(&lay->in_z,   S.xqin, S.xsin, d.zbuf_slots, Cch, ZLb, 8, sk))) die("in_qkv/in_z (legacy fp8)");
                            tr(L, 5);
                            if (!q27_bf16_gemv2_tile(lay->in_a, lay->in_b, d.norm_slots, Q27_HID, d.ab_tile, d.bb_tile, Q27_GDN_VH, Q27_HID, Cch, sk)) die("a/b gemv (legacy)");
                            tr(L, 6);
                            q27_gdn_conv_tp_tile(d.qkv_slots, QKVLb, Cch, d.conv + d.conv_off[gsl], lay->conv1d, 0, 1, sk);
                            tr(L, 7);
                            q27_gdn_scan2_tp(d.qkv_slots, QKVLb, d.zbuf_slots, ZLb, d.ab_tile, d.bb_tile, Q27_GDN_VH,
                                             lay->A_log, lay->dt_bias, lay->gdn_norm, d.S + d.s_off[gsl], d.mix_tile, ZLb,
                                             0, 1, Cch, S.mixq_tile, S.mixs_tile, lay->out_proj.in_scale, sk);
                            tr(L, 8);
                        } else {          // in_qkv/in_z -> int8 + scales; int8 a/b GEMV, conv (in place), scan
                            p2dump(S);
                            if (doA1) {   // in_qkv + in_z: ONE launch (grid.z), each with its own output stride
                                const q27_nvfp4_t* wi2[2] = { &lay->iqkv8, &lay->iz8 };
                                signed char* yi2[2] = { S.qkv8, S.z8 };
                                float* si2[2] = { S.qkvs, S.zs };
                                const int st2[2] = { QKVLb, ZLb };
                                const q27_i8g_t* wi2g[2] = { &lay->iqkv8g, &lay->iz8g };
                                if (bl_gdn) { const q27_i8r_t* w2[2] = { &lay->iqkv8r, &lay->iz8r }; int* accp = S.acc; const int w_gs = lay->iqkv8r.gs;
                                    const int ntot2 = q27_cat_rows(w2, 2);
                                    if (ntot2 > 0 && w_gs == lay->iqkv8r.K) {
                                        if (!q27_rb_gemm_i8g(d.id, w2[0]->w, ntot2, w2[0]->K, S.xqt, Cch, S.acc, w_gs, sk)) die("rb gdn gemm (fused)");
                                        int off = 0; for (int i = 0; i < 2; ++i) { const q27_i8r_t* w = w2[i];
                                            if (!q27_epi_q8v2(S.acc + off, ntot2, w->rows, w->s, 1, S.xst, w->alpha, yi2[i], si2[i], st2[i], Cch, sk)) die("rb gdn epi (fused)");
                                            off += w->rows; }
                                    } else
                                    for (int i = 0; i < 2; ++i) { const q27_i8r_t* w = w2[i];
                                        if (!q27_rb_gemm_i8g(d.id, w->w, w->rows, w->K, S.xqt, Cch, accp, w_gs, sk)) die("rb gdn gemm");
                                        if (!q27_epi_q8v2(accp, w->rows, w->rows, w->s, w->ng, S.xst, w->alpha, yi2[i], si2[i], st2[i], Cch, sk)) die("rb gdn epi");
                                        accp += (size_t)Cch * w->rows * w->ng; } }
                                else                                 if (!(q8g64 ? (q8r8 ? q27_wide_i8g64r8_q8m(2, wi2g, S.xqin, S.xsin, yi2, si2, st2, Cch, sk) : q27_wide_i8g64_q8m(2, wi2g, S.xqin, S.xsin, yi2, si2, st2, Cch, sk))
                                           : q8rb ? q27_wide_i8_rb_q8m(2, wi2, S.xqin, S.xsin, yi2, si2, st2, Cch, sk)
                                           : q27_wide_i8_q8m(2, wi2, S.xqin, S.xsin, yi2, si2, st2, Cch, q8ki, sk))) die("in_qkv/in_z");
                            }
                            tr(L, 5); dumpq("qkv8", L, S.qkv8, S.qkvs, 8); dumpq("z8", L, S.z8, S.zs, 8);
                            if (doA1) {
                            if (bl_gdn && lay->ab8r.w) {   // a/b as one int8 per-row GEMM [M][96] on the per-token input
                                if (!q27_rb_gemm_i8g(d.id, lay->ab8r.w, lay->ab8r.rows, lay->ab8r.K, S.xqt, Cch, S.acc, lay->ab8r.gs, sk)) die("rb a/b gemm");
                                if (!q27_epi_ab(S.acc, Q27_GDN_VH, lay->ab8r.s, S.xst, in_scl, S.a32, S.b32, Cch, sk)) die("rb a/b epi");
                            } else
                            if (!q27_q8_gemv2_tile(lay->in_a, lay->in_b, S.xqin, S.xsin, Q27_HID, in_scl, 0, q8g64 ? 1 : 0,
                                                   S.a32, S.b32, Q27_GDN_VH, Q27_HID, Cch, sk)) die("a/b gemv");
                            }   // doA1: a/b
                            if (doB) {
                            tr(L, 6); dumpf("a32", L, S.a32, 4); dumpf("b32", L, S.b32, 4);
                            static const bool q8convp = q27_env_flag("Q27_Q8_CONVP", true);   // position-parallel conv, out of place
                            const signed char* scan_q8 = q8convp ? S.qkvc8 : S.qkv8; const float* scan_qs = q8convp ? S.qkvcs : S.qkvs;
                            if (q8convp) q27_gdn_conv_q8_par(S.qkv8, S.qkvs, S.qkvc8, S.qkvcs, QKVLb, Cch, d.conv + d.conv_off[gsl], lay->conv1d, 0, 1, sk);
                            else q27_gdn_conv_q8_tile(S.qkv8, S.qkvs, QKVLb, Cch, d.conv + d.conv_off[gsl], lay->conv1d, 0, 1, sk);
                            tr(L, 7); dumpq("conv8", L, S.qkv8, S.qkvs, 8);
                            p2dump(S);
                            q27_gdn_scan2_q8(scan_q8, scan_qs, QKVLb, S.z8, S.zs, ZLb, S.a32, S.b32, Q27_GDN_VH,
                                             lay->A_log, lay->dt_bias, lay->gdn_norm, d.S + d.s_off[gsl],
                                             nullptr, ZLb, 0, 1, Cch, S.mixq_tile, S.mixs_tile, lay->out_proj.in_scale, 0, q8g64 ? 1 : 0, par, sk);
                            }   // doB: conv + scan
                            tr(L, 8); dumpq("mixq", L, S.mixq_tile, S.mixs_tile, 8); dumpf("S", L, d.S + d.s_off[gsl], 4);
                        }
                        // out_proj on the fp8 weights; residual = the fp32 layer input (or the bf16 copy under the legacy switch)
                        if (doA2) {
                        if (old_res) { if (!q27_proj_fp8_wide(&lay->out_proj, S.mixq_tile, S.mixs_tile, S.part, Cch, Q27_HID, hid_bf, Q27_HID, 1.0f, 8, sk)) die("out_proj (bf16 radd)"); }
                        else if (bl_gdn) { const q27_i8r_t* w = &lay->op8r; const int w_gs = w->gs;
                            if (!q27_requant_g64_tok(S.mixq_tile, S.mixs_tile, S.mixq_t, S.mixs_t, w->K, w->K, Cch, q8gs, sk)) die("rb op requant");
                            if (!q27_rb_gemm_i8g(d.id, w->w, w->rows, w->K, S.mixq_t, Cch, S.acc, w_gs, sk)) die("rb op gemm");
                            if (fuse_pn) { if (!q27_epi_res_rn_tok(S.acc, w->rows, w->s, w->ng, S.mixs_t, w->alpha, pend, S.part, lay->post_norm, (L == nlayer - 1) ? S.hid32c : nullptr, S.xq1, S.xst1, 1, lay->gate.in_scale, Cch, sk)) die("rb op epi+norm"); }
                            else if (!q27_epi_f32r(S.acc, w->rows, w->s, w->ng, S.mixs_t, w->alpha, S.part, pend, 1.0f, Cch, sk)) die("rb op epi"); }
                        else if (q8g64) { if (!(q8r8 ? q27_wide_i8g64r8_f32r(&lay->op8g, S.mixq_tile, S.mixs_tile, S.part, Cch, pend, 1.0f, sk) : q27_wide_i8g64_f32r(&lay->op8g, S.mixq_tile, S.mixs_tile, S.part, Cch, pend, 1.0f, sk))) die("out_proj g64"); }
                        else if (!(q8rb ? q27_wide_i8_rb_f32r(&lay->op8, S.mixq_tile, S.mixs_tile, S.part, Cch, pend, 1.0f, sk)
                                        : q27_wide_i8_f32r(&lay->op8, S.mixq_tile, S.mixs_tile, S.part, Cch, pend, 1.0f, q8ki, sk))) die("out_proj");
                        }   // doA2: out_proj
                        tr(L, 9); dumpf("part", L, S.part, 4);
                    }
                    if (evA) CK(hipEventRecord(evA, sk));   // stage A done: the next chunk may enter this layer
                    }   // stage A
                    if (doMLP) {
                    // LAYER-LOCAL INT8 MLP. The arena holds one layer; re-expand when it does not
                    // already hold this one. The conversion is ~423 MiB of traffic against a chunk
                    // of MLP GEMMs that reads the same weights for every token in the chunk, so it
                    // amortises -- and it is the whole reason the fast int8 consumer is reachable
                    // at ctx=262144 at all.
                    const bool mstage = mlpstage && d.mlp8_w && d.mlp8_s && lay->gate.w && lay->up.w && lay->down.w;
                    const int mslot = L & (Dev::MLP8_SLOTS - 1);
                    // This layer's MLP on the rocBLAS/Tensile path, from the arena (rb8) or from the
                    // persistent mirrors (bl_mlp). Everything downstream keys off BL so a layer whose
                    // representation came from the arena is indistinguishable from one whose
                    // representation was loaded -- which is the entire point of the arena.
                    const bool rb8 = mlp_rb && mstage_mode != 2 && mstage && d.mlp8_rs;
                    // PER-LAYER rocBLAS MLP: bl_mlp is the engine-wide "the mirrors are the
                    // configured representation" flag, which is false on the NVFP4 configuration.
                    // A layer is on the rocBLAS path whenever its PER-ROW mirror actually exists --
                    // which is the persistent mirror set (Q27_LS_MLP_I8K layers) or the arena slot.
                    const bool bl_l = q8bl && (q8blm & 4) && lay->gate8r.w && lay->up8r.w && lay->down8r.w;
                    const bool L_ex_mlp = lay->ex_gate.live() && lay->ex_up.live() && lay->ex_down.live();
                    const bool BL  = (bl_mlp || bl_l || rb8) && !old_res && !old_mlp;
                    const q27_i8r_t* G8 = rb8 ? &d.mlp8_gate_r : &lay->gate8r;
                    const q27_i8r_t* U8 = rb8 ? &d.mlp8_up_r   : &lay->up8r;
                    const q27_i8r_t* D8 = rb8 ? &d.mlp8_down_r : &lay->down8r;
                    if (mstage && d.mlp8_layer[mslot] != L) {
                        const size_t gw = (size_t)Q27_INTER * Q27_HID, gs_ = (size_t)Q27_INTER * (Q27_HID / 64);
                        const size_t dw_slot = (mstage_mode == 2) ? gw : (2 * gw + gw);
                        const size_t ds_slot = (mstage_mode == 2) ? gs_ : (2 * gs_ + gs_);
                        signed char* wp = d.mlp8_w + (size_t)mslot * dw_slot;
                        float*       sp = d.mlp8_s + (size_t)mslot * ds_slot;
                        if (mstage_mode == 2) {
                            if (!q27_nvfp4_to_i8g64_into(&lay->down, &d.mlp8_down, wp, sp, sk))
                                die("MLP stage expand (down)");
                        } else if (!q27_nvfp4_to_i8g64_into(&lay->gate, &d.mlp8_gate, wp,          sp,           sk) ||
                                   !q27_nvfp4_to_i8g64_into(&lay->up,   &d.mlp8_up,   wp + gw,     sp + gs_,     sk) ||
                                   !q27_nvfp4_to_i8g64_into(&lay->down, &d.mlp8_down, wp + 2 * gw, sp + 2 * gs_, sk))
                            die("MLP stage expand");
                        if (rb8) {   // same slot, requantized in place to the per-ROW form
                            float* rs = d.mlp8_rs + (size_t)mslot * (size_t)(2 * Q27_INTER + Q27_HID);
                            if (!q27_i8g64_rowify(&d.mlp8_gate, &d.mlp8_gate_r, rs,               sk) ||
                                !q27_i8g64_rowify(&d.mlp8_up,   &d.mlp8_up_r,   rs + Q27_INTER,     sk) ||
                                !q27_i8g64_rowify(&d.mlp8_down, &d.mlp8_down_r, rs + 2*Q27_INTER,   sk))
                                die("MLP stage rowify");
                        }
                        d.mlp8_layer[mslot] = L;
                    }
                    if (rb8 && !d.mlp8_gate_r.w) die("MLP arena row handles are empty");
                    // post-norm: fp32 h' in, int8 permuted MLP activations out; h' itself (part_slots) is the
                    // down's residual; only the last layer keeps an fp32 copy for the head
                    // NOTE: bl_l MUST be in this condition, not just q8mlp. When the int8 MLP mirrors
                    // are present on only SOME layers (Q27_LS_MLP_I8K), q8mlp is false engine-wide, so
                    // a rocBLAS layer that fell through to the permuted form would feed the GEMM
                    // permuted activations. That is fluent-looking garbage -- it degenerates into
                    // repetition, not into a fault.
                    if (BL) {
                        if (fuse_pn) { /* post norm already produced by the fused o/out_proj epilogue */ }
                        else if (!q27_red_rn_tok_b1(S.part, lay->post_norm, (L == nlayer - 1) ? S.hid32c : nullptr, S.xq1, S.xst1, Q27_HID, 1, lay->gate.in_scale, Cch, q8gs, sk)) die("post norm tok");
                    } else
                    if (q8mlp && !old_res && !old_mlp) {
                        if (fuse_pn) { /* fused epilogue already produced it */ }
                        else if (!q27_red_rn_g64_b1(S.part, lay->post_norm, (L == nlayer - 1) ? S.hid32c : nullptr, S.xq1, S.xs1, Q27_HID, 1, lay->gate.in_scale, Cch, sk)) die("post norm g64");
                    } else
                    if (!q27_red_rn_perm_b1(S.part, lay->post_norm, old_res ? hid_bf : nullptr, (L == nlayer - 1) ? S.hid32c : nullptr,
                                            S.xq1, S.xs1, Q27_HID, 1, lay->gate.in_scale, Cch, sk)) die("post norm");
                    tr(L, 10); dumpq("xq1", L, S.xq1, S.xs1, 8);
                    // ---- Q27_EXL3: the MLP on the resident trellis ----
                    // This is THE 4-bit surface of prefill: in the incumbent the
                    // attention projections are already q27_fp8_t (8-bit) and only
                    // gate/up/down are NVFP4. The post-norm above is left in place --
                    // it still produces S.hid32c for the head on the last layer -- and
                    // EXL3 takes its own bf16 norm off the same fp32 partial rather
                    // than trying to un-permute the quantized one.
                    bool ex_mlp_done = false;
                    if (L_ex_mlp) {
                        void* xw = exl3_ws(d, Q27_EXL3_WMAX, Q27_EXL3_WMAX, Cch);
                        if (!xw) die("EXL3 workspace");
                        unsigned short* nb  = q27_exl3_ws_norm(xw, Cch);
                        unsigned short* act = q27_exl3_ws_act(xw, lay->ex_down.in, Cch, lay->ex_down.out);
                        if (!q27_exl3_rmsnorm_bf16(S.part, lay->post_norm, nb, Q27_HID, Cch, Q27_HID, Q27_HID, sk)) die("EXL3 post norm");
                        // Q27_INTER IS THE WRONG WIDTH HERE. S.pa/S.pb are d.pa_mlp/d.pb_mlp,
                        // allocated with a PER-CARD row stride (mlp_len = Q27_INTER/ndev). Passing
                        // the full 17408 wrote each row 4x too far apart on ndev=4 -- past the end
                        // of the allocation -- and the swiglu then read 17408 columns out of a
                        // 4352-wide row. The sibling batched site (the chunk MLP) already used the
                        // shard width; this one did not, and nothing caught it because every
                        // TENSOR verifies and the M=1 decode MLP is exact (measured 2026-09-19:
                        // gate 22.82 dB, down 19.53 dB against the live input). Prefill corrupted
                        // the state the correct decode then read.
                        const int EXIL = lay->ex_gate.out;
                        if (EXIL != lay->ex_up.out || EXIL != lay->ex_down.in || EXIL != mlp_len(d.id))
                            die("EXL3 MLP shard width disagrees with the chunk buffers");
                        if (!exl3_f32(d, lay->ex_gate, nb, S.pa, Cch, sk, Q27_HID, EXIL)) die("EXL3 gate");
                        if (!exl3_f32(d, lay->ex_up,   nb, S.pb, Cch, sk, Q27_HID, EXIL)) die("EXL3 up");
                        if (!q27_exl3_swiglu_bf16(S.pa, S.pb, act, EXIL, Cch, EXIL, EXIL, sk)) die("EXL3 swiglu");
                        if (!exl3_f32(d, lay->ex_down, act, S.mixer, Cch, sk, EXIL, Q27_HID)) die("EXL3 down");
                        // the incumbent down folds the fp32 residual (S.part) into its
                        // epilogue with share 1.0; EXL3's epilogue is the output
                        // Hadamard, so it rides as its own pass.
                        if (!q27_exl3_add_res_f32(S.mixer, S.part, 1.0f, Q27_HID, Cch, Q27_HID, Q27_HID, sk)) die("EXL3 down residual");
                        ex_mlp_done = true;
                    }
                    if (!ex_mlp_done) {
                    if (old_mlp) { if (!q27_wide_gu(&lay->gate, &lay->up, S.xq1, S.xs1, S.pa, S.pb, Cch, g_pf_wide_kg, sk)) die("gate+up (legacy)"); }
                    else if (BL) { const q27_i8r_t* wg = G8; const q27_i8r_t* wu = U8; const int w_gs = wg->gs; int* ag = S.acc; int* au = S.acc + (size_t)Cch * wg->rows * wg->ng;
                        if (!q27_rb_gemm_i8g(d.id, wg->w, wg->rows, wg->K, S.xq1, Cch, ag, w_gs, sk)) die("rb gate gemm");
                        if (!q27_rb_gemm_i8g(d.id, wu->w, wu->rows, wu->K, S.xq1, Cch, au, w_gs, sk)) die("rb up gemm");
                        if (!q27_epi_swiglu_tok(ag, au, wg->rows, wg->s, wu->s, wg->ng, S.xst1, wg->alpha, wu->alpha, 1.0f / lay->down.in_scale, S.xq2, S.xst2, Cch, q8gs, sk)) die("rb swiglu"); }
                    else if (mstage && mstage_mode != 2) { if (!q27_wide_gu_i8g64(&d.mlp8_gate, &d.mlp8_up, S.xq1, S.xs1, S.pa, S.pb, Cch, sk)) die("gate+up i8g64 (staged)"); }
                    else if (q8mlp && !old_res) { if (!(q8r8 ? q27_wide_gu_i8g64r8(&lay->gate8, &lay->up8, S.xq1, S.xs1, S.pa, S.pb, Cch, sk) : q27_wide_gu_i8g64(&lay->gate8, &lay->up8, S.xq1, S.xs1, S.pa, S.pb, Cch, sk))) die("gate+up i8g64"); }
                    else if (!q27_wide_gu(&lay->gate, &lay->up, S.xq1, S.xs1, S.pa, S.pb, Cch, g_pf_wide_kg, sk)) die("gate+up");
                    tr(L, 11); dumpf("pa", L, S.pa, 4); dumpf("pb", L, S.pb, 4);
                    if (BL) { /* the gate/up epilogue (q27_epi_swiglu_tok) already wrote the per-token S.xq2 */ }
                    else if (mstage) { if (!q27_swiglu_quant_b_g64(S.pa, S.pb, S.xq2, S.xs2, Q27_INTER, lay->down.in_scale, Cch, sk)) die("swiglu g64 (staged)"); }   // g64 form feeds the i8g down in both modes
                    else if (!bl_mlp && q8mlp && !old_res && !old_mlp) { if (!q27_swiglu_quant_b_g64(S.pa, S.pb, S.xq2, S.xs2, Q27_INTER, lay->down.in_scale, Cch, sk)) die("swiglu g64"); }
                    else if (!bl_mlp && !q27_swiglu_quant_b(S.pa, S.pb, S.xq2, S.xs2, Q27_INTER, lay->down.in_scale, Cch, sk)) die("swiglu");
                    tr(L, 12); dumpq("xq2", L, S.xq2, S.xs2, 8);
                    if (old_mlp) { if (!old_res) die("Q27_Q8_OLD=32 needs 4"); if (!q27_wide_down(&lay->down, S.xq2, S.xs2, S.mixer, Cch, hid_bf, 1.0f, g_pf_wide_kd, sk)) die("down (legacy wide)"); }
                    else if (old_res) { if (!q27_wide_down_rb2p(&lay->down, S.xq2, S.xs2, S.mixer, Cch, hid_bf, 1.0f, 4, sk)) die("down (legacy)"); }
                    else if (BL) { const q27_i8r_t* w = D8; const int w_gs = w->gs;
                        if (!q27_rb_gemm_i8g(d.id, w->w, w->rows, w->K, S.xq2, Cch, S.acc, w_gs, sk)) die("rb down gemm");
                        if (!q27_epi_f32r(S.acc, w->rows, w->s, w->ng, S.xst2, w->alpha, (q8noinv && !q8redb1) ? pend : S.mixer, S.part, 1.0f, Cch, sk)) die("rb down epi"); }   // NOINV: the new residual goes straight to pend (no red_b1 copy, inv unused)
                    else if (mstage) { if (!q27_wide_down_i8g64_f32r(&d.mlp8_down, S.xq2, S.xs2, S.mixer, Cch, S.part, 1.0f, sk)) die("down i8g64 (staged)"); }
                    else if (q8mlp) { if (!(q8r8 ? q27_wide_i8g64r8_f32r(&lay->down8, S.xq2, S.xs2, S.mixer, Cch, S.part, 1.0f, sk) : q27_wide_down_i8g64_f32r(&lay->down8, S.xq2, S.xs2, S.mixer, Cch, S.part, 1.0f, sk))) die("down i8g64"); }
                    else if (!(q8rbd ? q27_wide_down_rb_f32r(&lay->down, S.xq2, S.xs2, S.mixer, Cch, S.part, 1.0f, sk)
                                    : q27_wide_down_f32r(&lay->down, S.xq2, S.xs2, S.mixer, Cch, S.part, 1.0f, q8kd, sk))) die("down");
                    }   // !ex_mlp_done
                    tr(L, 13); dumpf("mixer", L, S.mixer, 4);
                    if (!(BL && q8noinv && !q8redb1)) q27_red_b1(S.mixer, pend, d.inv_slots + (p0 - sb), Q27_HID, Cch, sk);
                    tr(L, 14);
                    }   // stage B
                };
                static const int q8sched = q27_env_int("Q27_Q8_SCHED", 1);   // 1 = per-chunk two-stream (1498 ms); 2 = stage-A high-priority / MLP low-priority chunk pairs (1553 ms)
                static const bool q8one = q27_env_flag("Q27_Q8_1S", true);    // v1 schedule on ONE stream: 933 ms vs 964 with two streams (co-scheduled GEMMs ran 1.5x slower)
                bool export_done = false;   // set by the overlapped export inside the ring schedule
                std::vector<hipEvent_t> ring_late;   // ring-schedule events destroyed after the timing stop
                int last_set = (nch - 1) & 1;   // scratch set holding the last chunk's hid32c (the finals); the paired schedule uses job parity
                if (g_ls_q8 && q8blk < LPP) {
                    // ---- Q27_LS_Q8 SCHEDULE v3: RING OF LAYER BLOCKS. Card g owns blocks b = r*ndev + g
                    // (r = 0..nround-1) of q8blk layers; a chunk visits blocks 0..nblk-1 in order, so it
                    // hops the ring card 0 -> 1 -> ... -> ndev-1 -> 0 ... (nblk handoffs per chunk).
                    // The pipeline fill is (ndev-1) block-times instead of (ndev-1) chunk-times.
                    // One stream per card, jobs (k, r) ordered by readiness key k + r*ndev (ties: lower k).
                    // Cross-card: the producer records evdone[pos] when it ENQUEUES job pos and publishes
                    // pos in ls_rec; the consumer spins for that (short) and waits on the device.
                    // nround WAS `LPP / q8blk` -- the UNIFORM blocks-per-card -- while cblk is
                    // built from the ownership oracle. Under Q27_LS_OWN those disagree: at
                    // 28,12,12,12 with blk=4 the oracle deals 7,3,3,3, so card 0 silently DROPPED
                    // 3 of its blocks and cards 1-3 indexed cblk[g][3] on a 3-element vector --
                    // an out-of-bounds read whose garbage block sent every one of them to layer 0
                    // ("input norm g64 declined (L0) card=3 owner=0 owned=0"). That is the crash
                    // the 09-19 pool note records as the blocker for uneven ownership.
                    // The per-card count is cblk[g].size(); nothing else may stand in for it.
                    const int nblk = nlayer / q8blk;
                    std::vector<std::vector<int>> cblk((size_t)ndev);
                    for (int b = 0; b < nblk; ++b) cblk[(size_t)blk_owner(b)].push_back(b);
                    const int myblk = (int)cblk[(size_t)g].size();   // THIS card's blocks, not LPP/q8blk
                    const int njobs = nch * myblk;
                    if (myblk <= 0) { std::fprintf(stderr, "Q27_LS_OWN: card %d owns no layer blocks; aborting\n", g); std::exit(1); }
                    auto jobkey = [&](int k, int b) { return (long long)(k + b) * nch + k; };   // readiness: stage k + b, ties by chunk
                    std::vector<std::pair<long long, int>> ord; ord.reserve(njobs);   // this card: second = k * myblk + i (i = index into cblk[g])
                    for (int k = 0; k < nch; ++k) for (int i = 0; i < myblk; ++i) ord.push_back({jobkey(k, cblk[(size_t)g][(size_t)i]), k * myblk + i});
                    std::sort(ord.begin(), ord.end());
                    auto jobpos = [&](int card, int k, int b) {   // position of job (k, b) in card's (sorted) order
                        const long long key = jobkey(k, b); int pos = 0;
                        for (int kk = 0; kk < nch; ++kk) for (size_t i = 0; i < cblk[(size_t)card].size(); ++i) if (jobkey(kk, cblk[(size_t)card][i]) < key) ++pos;
                        return pos; };
                    std::vector<hipEvent_t>& evd = g_ls_evdone[g];
                    evd.assign((size_t)njobs, nullptr);
                    for (int j = 0; j < njobs; ++j) CK(hipEventCreateWithFlags(&evd[j], hipEventDisableTiming));
                    if (q8trace) std::fprintf(stderr, "Q27_Q8_RING g=%d sw=[%d,%d) nwin=%d sb=%d nch=%d myblk=%d njobs=%d ev0=%p\n",
                                 g, sw_beg, sw_end, nwin, sb, nch, myblk, njobs, (void*)evd[0]);
                    if (q8trace) std::fprintf(stderr, "Q27_Q8_EV g=%d n=%zu e0=%p e1=%p\n", g, evd.size(),
                                 (void*)evd[0], njobs > 1 ? (void*)evd[1] : nullptr);
                    if (g == 0) MS("T6_events_done");
                    // THE RING RECORD MUST BE RESET BETWEEN TWO BARRIERS, NOT BEFORE BOTH OF THEM.
                    // ls_rec[g] is a MONOTONIC per-card job counter that peers spin on: `while (ls_rec[src] < pos) {}`.
                    // Resetting it to -1 before b1 lets a card that has finished its round drive the counter BACKWARDS
                    // while a peer is still spinning on it for the OLD round -- that peer's condition can then never
                    // be satisfied and it never reaches any barrier again. Measured 2026-09-16, deterministic on the
                    // SERVE_RESET -> SERVE_GROW -> sweep path: "FATAL TpBar ... arrived 2", with card2/card3 at gen=3318
                    // job=-1 (already reset, one round ahead) and card0/card1 at gen=3317 job=3 (still consuming).
                    // b1 = every card has left the previous round, so nobody is still reading these cells.
                    // b2 = every reset is visible before any producer publishes, so nobody reads a STALE-HIGH cell.
                    // Both hazards are real; the code had only the second fence.
                    C.b1.wait();
                    C.ls_rec[g].store(-1, std::memory_order_release);
                    C.b2.wait();                                   // every card's event table exists
                    hipStream_t sk = d.hs[0];
                    // peer-store handoff (Q27_Q8_PEERW): the producer writes the block residual into the NEXT ring card's
                    // pend buffer with a copy kernel through the peer mapping; the consumer only waits on the event.
                    // Enabled per card after a self-test (a kernel store to the peer, read back).
                    static const bool q8peerw_env = q27_env_flag("Q27_Q8_PEERW", false);
                    bool peerw = false;
                    if (q8peerw_env && ndev > 1) {
                        const int nxt = (g + 1) % ndev; int can = 0;
                        CK(hipDeviceCanAccessPeer(&can, d.id, D[nxt].id));
                        if (can) { hipError_t pe = hipDeviceEnablePeerAccess(D[nxt].id, 0); if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) can = 0; (void)hipGetLastError(); }
                        if (can) {
                            const float want = 4200.0f + (float)g;
                            q27_p2p_probe(D[nxt].pend_slots, want, sk);   // pend_slots[0] of the peer: overwritten by the embedding later
                            CK(hipStreamSynchronize(sk));
                            float got = -1.f; CK(hipMemcpy(&got, D[nxt].pend_slots, 4, hipMemcpyDeviceToHost));
                            peerw = (got == want);
                            if (!peerw) std::fprintf(stderr, "Q27_Q8_PEERW: peer store self-test %d->%d failed (got %g), using hipMemcpyPeerAsync\n", g, nxt, (double)got);
                        } else std::fprintf(stderr, "Q27_Q8_PEERW: no peer access %d->%d, using hipMemcpyPeerAsync\n", g, (g + 1) % ndev);
                    }
                    if (g == 0) MS("T6_setup_done");
                    C.b1.wait(); C.b2.wait();                       // peer probes done before anyone writes pend
                    if (g == 0) MS("T6a_jobs_begin");
                    // ---- Q27_Q8_SPLIT: per-card two-stream software pipeline over PAIRS of consecutive jobs (different
                    // chunks). Per layer: sk (GEMM stream): A1(j) A1(j') | A2(j) A2(j');  sB (high priority): B(j) B(j').
                    // The attention / conv+scan segment of one job overlaps the projections of the other; GEMMs never
                    // share the GPU with GEMMs. Scratch set = job parity (consecutive jobs alternate).
                    static const bool q8split = q27_env_flag("Q27_Q8_SPLIT", false);   // measured 902 vs 567 ms: off
                    static const bool q8pfin = q27_env_flag("Q27_Q8_PFIN", true);   // prefetch each job's input on the copy stream (overlaps the previous job)
                    std::vector<hipEvent_t> evin((size_t)njobs, nullptr);
                    for (int j = 0; j < njobs; ++j) CK(hipEventCreateWithFlags(&evin[j], hipEventDisableTiming));
                    std::vector<hipEvent_t> evA1, evB;
                    hipStream_t sB = d.q8hi;
                    if (q8split) {
                        evA1.assign((size_t)njobs, nullptr); evB.assign((size_t)njobs, nullptr);
                        for (int j = 0; j < njobs; ++j) { CK(hipEventCreateWithFlags(&evA1[j], hipEventDisableTiming)); CK(hipEventCreateWithFlags(&evB[j], hipEventDisableTiming)); }
                        auto handoff = [&](int k, int b, int p0, int Cch) {
                            if (b <= 0) return;
                            const int src = blk_owner(b - 1);
                            const int pos = jobpos(src, k, b - 1);
                            while (C.ls_rec[src].load(std::memory_order_acquire) < pos) { }
                            Q27_LS_EVCHK("handoff", src, pos);
                            CK(hipStreamWaitEvent(sk, g_ls_evdone[src][pos], 0));
                            if (!q27_xcopy(d.pend_slots + (size_t)(p0 - sb) * Q27_HID, d.id,
                                              D[src].pend_slots + (size_t)(p0 - sb) * Q27_HID, D[src].id,
                                              (size_t)Cch * Q27_HID * 4, sk, "ring")) { std::fprintf(stderr, "Q27_XCOPY ring failed g=%d src=%d\n", g, src); std::fflush(stderr); std::exit(1); }
                            if (!q8noinv) CK(hipMemcpyAsync(d.inv_slots + (p0 - sb), D[src].inv_slots + (p0 - sb), (size_t)Cch * 4, hipMemcpyHostToHost, sk));
                        };
                        for (int j0 = 0; j0 < njobs; j0 += 2) {
                            const int nj = (j0 + 1 < njobs) ? 2 : 1;
                            int kk[2] = {0, 0}, bb[2] = {0, 0}, pp[2] = {0, 0}, cc[2] = {0, 0};
                            for (int t = 0; t < nj; ++t) {
                                const int j = j0 + t; kk[t] = ord[j].second / myblk; bb[t] = cblk[(size_t)g][(size_t)(ord[j].second % myblk)];
                                pp[t] = sw_beg + lsoff(kk[t]); const int cwk = lschw(kk[t]); cc[t] = (sw_end - pp[t] < cwk) ? (sw_end - pp[t]) : cwk;
                            }
                            handoff(kk[0], bb[0], pp[0], cc[0]);
                            for (int l = 0; l < q8blk; ++l) {
                                for (int t = 0; t < nj; ++t) {
                                    const int j = j0 + t, L = bb[t] * q8blk + l;
                                    if (t == 1 && l == 0) handoff(kk[1], bb[1], pp[1], cc[1]);   // after job j0's first work is enqueued
                                    ls_q8_layer(L, pp[t], cc[t], d.q8s[j & 1], sk, j & 1, nullptr, 10);
                                    CK(hipEventRecord(evA1[j], sk)); CK(hipStreamWaitEvent(sB, evA1[j], 0));
                                    ls_q8_layer(L, pp[t], cc[t], d.q8s[j & 1], sB, j & 1, nullptr, 11);
                                    CK(hipEventRecord(evB[j], sB));
                                }
                                for (int t = 0; t < nj; ++t) {
                                    const int j = j0 + t, L = bb[t] * q8blk + l;
                                    CK(hipStreamWaitEvent(sk, evB[j], 0));
                                    ls_q8_layer(L, pp[t], cc[t], d.q8s[j & 1], sk, j & 1, nullptr, 12);
                                    if (l == q8blk - 1) { CK(hipEventRecord(evd[j], sk)); C.ls_rec[g].store(j, std::memory_order_release); }
                                }
                            }
                        }
                        last_set = (njobs - 1) & 1;
                    } else
                    for (int j = 0; j < njobs; ++j) {
                        const int k = ord[j].second / myblk, b = cblk[(size_t)g][(size_t)(ord[j].second % myblk)];
                        const int p0 = sw_beg + lsoff(k);
                        const int cwk = lschw(k);
                        const int Cch = (sw_end - p0 < cwk) ? (sw_end - p0) : cwk;
                        if (b > 0) {                                // input = block b-1's output, on the previous ring card
                            const int src = blk_owner(b - 1);
                            const int pos = jobpos(src, k, b - 1);
                            while (C.ls_rec[src].load(std::memory_order_acquire) < pos) { }
                            // Q27_Q8_PFIN: the input copy runs on the COPY stream as soon as the producer signals, so it
                            // overlaps the previous job's compute; the compute stream only waits for the copy's event.
                            hipStream_t sc = (q8pfin && !peerw) ? d.hs[1] : sk;
                            Q27_LS_EVCHK("jobin", src, pos);
                            CK(hipStreamWaitEvent(sc, g_ls_evdone[src][pos], 0));
                            if (!peerw) if (!q27_xcopy(d.pend_slots + (size_t)(p0 - sb) * Q27_HID, d.id,
                                              D[src].pend_slots + (size_t)(p0 - sb) * Q27_HID, D[src].id,
                                              (size_t)Cch * Q27_HID * 4, sc, "ring")) { std::fprintf(stderr, "Q27_XCOPY ring failed g=%d src=%d\n", g, src); std::fflush(stderr); std::exit(1); }
                            if (!q8noinv) CK(hipMemcpyAsync(d.inv_slots + (p0 - sb), D[src].inv_slots + (p0 - sb), (size_t)Cch * 4,
                                              hipMemcpyHostToHost, sc));
                            if (sc != sk) { CK(hipEventRecord(evin[j], sc)); CK(hipStreamWaitEvent(sk, evin[j], 0)); }
                        }
                        if (q8trace && g == 1) { CK(hipStreamSynchronize(sk)); std::fprintf(stderr, "Q8T g=%d JOBIN k=%d b=%d p0=%d Cch=%d ok\n", g, k, b, p0, Cch); }
                        for (int L = b * q8blk; L < (b + 1) * q8blk; ++L)
                            ls_q8_layer(L, p0, Cch, d.q8s[k & 1], sk, k & 1, nullptr, 2);
                        if (g_df2_draft) {   // DFlash2 prompt taps: the five boundary layers are the LAST of their 2-layer block
                            const int lastL = (b + 1) * q8blk - 1;
                            static const int PTAP[5] = {5, 19, 33, 47, 61};
                            for (int t = 0; t < 5; ++t) if (lastL == PTAP[t]) {
                                q27_k_df2_f2bfv(d.pend_slots + (size_t)(p0 - sb) * Q27_HID,
                                                 d.df2_pf_taps + ((size_t)t * g_pf_cap + (p0 - sb)) * Q27_HID, Cch * Q27_HID, sk);   // the prefill's pend IS the full residual stream (hid32c only holds the finals)
                            }
                        }
                        if (g_spec && g_mtp_pp && g_mtp_hall && g == last_card && b == nblk - 1) {   // Q27_MTP_PP: this chunk's final residual -> host
                            MtpPP& P = g_mtppp[g]; const Q8Scr& Sk = d.q8s[k & 1];
                            q27_copy_f32(Sk.hid32c, P.fin32, (size_t)Cch * Q27_HID, sk);
                            q27_add_f32(P.fin32, d.pend_slots + (size_t)(p0 - sb) * Q27_HID, Cch * Q27_HID, sk);   // h' + pending = the finals' arithmetic
                            q27_f2bf_vec(P.fin32, P.finbf, Cch * Q27_HID, sk);
                            CK(hipMemcpyAsync(g_mtp_hall + (size_t)(p0 - sb) * Q27_HID, P.finbf, (size_t)Cch * Q27_HID * 2, hipMemcpyDeviceToHost, sk));
                        }
                        if (peerw && b + 1 < nblk)   // push the block residual to the next ring card
                            if (!q27_copy_f32(d.pend_slots + (size_t)(p0 - sb) * Q27_HID, D[(g + 1) % ndev].pend_slots + (size_t)(p0 - sb) * Q27_HID, (size_t)Cch * Q27_HID, sk)) { std::fprintf(stderr, "Q27_Q8_PEERW: peer copy declined\n"); std::exit(1); }
                        if (q8trace && j < 2) std::fprintf(stderr, "Q27_Q8_REC g=%d j=%d n=%zu e=%p\n", g, j, evd.size(), (void*)evd[j]);
                        if (pf_profile_on()) g_pfp[g].mark(sk, -1, 99);   // job end
                        CK(hipEventRecord(evd[j], sk));
                        C.ls_rec[g].store(j, std::memory_order_release);
                        if (q8trace && g == 1) { CK(hipStreamSynchronize(sk)); std::fprintf(stderr, "Q8T g=%d JOBEND k=%d b=%d ok\n", g, k, b); }
                    }
                    // ---- decode-state export, OVERLAPPED: pull every non-owned layer's KV prefix / S / conv on the
                    // second stream as soon as the owner's last-chunk job for that block is enqueued (device-side
                    // event wait), instead of after the barrier on the finals' stream. Only the filled KV prefix.
                    static const bool q8exp = q27_env_flag("Q27_Q8_EXPOVL", true);
                    if (q8exp && !g_sr) {   // Q27_LS_SR: nothing to export -- no card decodes a layer it does not own
                        hipStream_t sx = d.hs[1];
                        if (pf_profile_on()) g_pfp[g].mark(sx, -1, 98);   // export begin
                        for (int L = 0; L < nlayer; ++L) {
                            const int owner = lay_owner(L);
                            if (owner == g) continue;
                            // EXPORT OF A LAYER THIS CARD DOES NOT OWN. The owner holds the layer
                            // full width; this card stores -- and decode reads -- only head-quarter
                            // g. So the transfer is the quarter, strided out of the owner's rows,
                            // not the whole row. Three quarters of the old copy were bytes nothing
                            // ever read.
                            const int pos = jobpos(owner, nch - 1, L / q8blk);
                            while (C.ls_rec[owner].load(std::memory_order_acquire) < pos) { }
                            Q27_LS_EVCHK("export", owner, pos);
                            CK(hipStreamWaitEvent(sx, g_ls_evdone[owner][pos], 0));
                            const int fsc = L >> 2, gsc = L - fsc;
                            // Q27_PF_P2P: prep already wrote every head straight into the card that
                            // OWNS it, so there is nothing to export -- and exporting would be WRONG:
                            // the owner no longer has full-width rows, so the strided read would copy
                            // data shifted by this card's head offset over its own correct quarter.
                            // The GDN state below is still owner-resident and must still be copied.
                            if (Q27_IS_FULL(L)) {
                                if (!g_pf_kvp2p) q27_export_kv_quarter(d, D[owner], fsc, sw_beg, sw_end, sx);
                            } else {
                                CK(hipMemcpyPeerAsync(d.S + d.s_off[gsc], d.id,
                                                      D[owner].S + D[owner].s_off[gsc] + (size_t)q27_tp_shard(g)->gv_h0 * Q27_GDN_D * Q27_GDN_D,
                                                      D[owner].id, d.s_bytes[gsc], sx));
                                q27_copy_conv_quarter(D[owner], d, gsc, g, ndev, false, sx);
                            }
                        }
                        if (pf_profile_on()) g_pfp[g].mark(sx, -1, 97);   // export end
                        export_done = true;
                    }
                    CK(hipStreamSynchronize(sk));
                    if (q8split) { CK(hipStreamSynchronize(sB)); }
                    if (q8pfin) { CK(hipStreamSynchronize(d.hs[1])); }
                    if (pf_profile_on()) { CK(hipStreamSynchronize(d.hs[1])); pf_profile_report(g, sw_beg, sw_end); }
                    if (g == last_card) MS("T6b_loop_done");
                    if (g_df2_draft) {   // DFlash2: gather the prompt taps (each card owns some boundaries) -> every card gets the full set
                        static const int PTAP2[5] = {5, 19, 33, 47, 61};
                        // d.df2_pf_taps_h is PER-DEVICE: the old gather wrote four private buffers and no card
                        // ever saw a peer's boundaries, so 4 of the 5 fc taps stayed zero for every prompt row.
                        static unsigned short* tapshare = nullptr;
                        const size_t trow = (size_t)g_pf_cap * Q27_HID;       // device/host layout is [5][g_pf_cap][HID]
                        const size_t tbytes = (size_t)(sw_end) * Q27_HID * 2;
                        if (g == 0 && !tapshare && hipHostMalloc((void**)&tapshare, (size_t)5 * trow * 2, hipHostMallocDefault) != hipSuccess) tapshare = nullptr;
                        C.b1.wait(); C.b2.wait();
                        if (!tapshare) { if (g == 0) std::fprintf(stderr, "Q27_DFLASH2_PFTAPS share alloc FAILED\n"); std::abort(); }
                        for (int t = 0; t < 5; ++t) if (blk_owner(PTAP2[t] / q8blk) == g)
                            CK(hipMemcpyAsync(tapshare + (size_t)t * trow, d.df2_pf_taps + (size_t)t * trow, tbytes, hipMemcpyDeviceToHost, sk));
                        CK(hipStreamSynchronize(sk));
                        C.b1.wait(); C.b2.wait();
                        for (int t = 0; t < 5; ++t)                            // per-boundary stride, not one contiguous 5*sw_end run
                            CK(hipMemcpyAsync(d.df2_pf_taps + (size_t)t * trow, tapshare + (size_t)t * trow, tbytes, hipMemcpyHostToDevice, sk));
                        CK(hipStreamSynchronize(sk));
                        C.b1.wait(); C.b2.wait();
                        if (g == 0) std::printf("Q27_DFLASH2_PFTAPS gathered %d positions (shared buffer, all 5 boundaries)\n", sw_end);
                    }
                    C.b1.wait(); C.b2.wait();                       // nobody frees an event a peer may still wait on
                    // event destruction is deferred until after the finals / timing stop (64+ hipEventDestroy calls cost ~6 ms here)
                    ring_late.insert(ring_late.end(), evA1.begin(), evA1.end()); ring_late.insert(ring_late.end(), evB.begin(), evB.end());
                    ring_late.insert(ring_late.end(), evin.begin(), evin.end());
                } else
                if (g_ls_q8 && q8sched == 2) {
                    // ---- Q27_LS_Q8 SCHEDULE (v2): chunk PAIRS interleaved layer by layer. Every stage A
                    // (norm, projections, attention/GDN, o/out_proj -- the latency-critical chain whose
                    // recurrent state / KV rows gate the next chunk) runs on the HIGH-priority stream; every
                    // stage B (post-norm, MLP, red_b1) on the LOW-priority stream. Within a pair: A(k0,L) A(k1,L)
                    // on hi, B(k0,L) B(k1,L) on lo, with events A->B (same chunk, same layer) and B->A (same
                    // chunk, next layer). Cross-chunk state order is the hi stream's enqueue order. Scratch
                    // set = chunk parity; the next pair's A(0) waits for this pair's B(last) on the same set.
                    // Cards hand chunks over with cross-device stream events (no barrier rounds, no drains).
                    std::vector<hipEvent_t>& evd = g_ls_evdone[g];
                    evd.assign((size_t)nch, nullptr);
                    for (int k = 0; k < nch; ++k) CK(hipEventCreateWithFlags(&evd[k], hipEventDisableTiming));
                    std::vector<hipEvent_t> evA((size_t)2 * LPP, nullptr), evB((size_t)2 * LPP, nullptr);
                    for (int i = 0; i < 2 * LPP; ++i) { CK(hipEventCreateWithFlags(&evA[i], hipEventDisableTiming)); CK(hipEventCreateWithFlags(&evB[i], hipEventDisableTiming)); }
                    hipStream_t hi = d.q8hi, lo = d.q8lo;
                    // THE RING RECORD MUST BE RESET BETWEEN TWO BARRIERS, NOT BEFORE BOTH OF THEM.
                    // ls_rec[g] is a MONOTONIC per-card job counter that peers spin on: `while (ls_rec[src] < pos) {}`.
                    // Resetting it to -1 before b1 lets a card that has finished its round drive the counter BACKWARDS
                    // while a peer is still spinning on it for the OLD round -- that peer's condition can then never
                    // be satisfied and it never reaches any barrier again. Measured 2026-09-16, deterministic on the
                    // SERVE_RESET -> SERVE_GROW -> sweep path: "FATAL TpBar ... arrived 2", with card2/card3 at gen=3318
                    // job=-1 (already reset, one round ahead) and card0/card1 at gen=3317 job=3 (still consuming).
                    // b1 = every card has left the previous round, so nobody is still reading these cells.
                    // b2 = every reset is visible before any producer publishes, so nobody reads a STALE-HIGH cell.
                    // Both hazards are real; the code had only the second fence.
                    C.b1.wait();
                    C.ls_rec[g].store(-1, std::memory_order_release);
                    C.b2.wait();                                   // every card's event table exists
                    auto handoff = [&](int k) {                     // this card's input for chunk k, on hi
                        if (g == 0) return;
                        const int p0 = sw_beg + lsoff(k);
                        const int cwk = lschw(k);
                        const int Cch = (sw_end - p0 < cwk) ? (sw_end - p0) : cwk;
                        while (C.ls_rec[g - 1].load(std::memory_order_acquire) < k) { }
                        CK(hipStreamWaitEvent(hi, g_ls_evdone[g - 1][k], 0));
                        CK(hipMemcpyPeerAsync(d.pend_slots + (size_t)(p0 - sb) * Q27_HID, d.id,
                                              D[g - 1].pend_slots + (size_t)(p0 - sb) * Q27_HID, D[g - 1].id,
                                              (size_t)Cch * Q27_HID * 4, hi));
                        CK(hipMemcpyAsync(d.inv_slots + (p0 - sb), D[g - 1].inv_slots + (p0 - sb), (size_t)Cch * 4,
                                          hipMemcpyHostToHost, hi));
                    };
                    for (int kp = 0; kp < nch; kp += 2) {
                        const int nk = (kp + 1 < nch) ? 2 : 1;
                        int p0s[2] = {0, 0}, Cchs[2] = {0, 0};
                        for (int j = 0; j < nk; ++j) {
                            const int k = kp + j;
                            p0s[j] = sw_beg + lsoff(k);
                            const int cwk = lschw(k);
                            Cchs[j] = (sw_end - p0s[j] < cwk) ? (sw_end - p0s[j]) : cwk;
                            if (kp >= 2) CK(hipStreamWaitEvent(hi, evB[(size_t)j * LPP + (LPP - 1)], 0));   // scratch set j is free
                            handoff(k);
                        }
                        for (int L = L0; L < L1; ++L) {
                            for (int j = 0; j < nk; ++j) {
                                const int k = kp + j;
                                if (L > L0) CK(hipStreamWaitEvent(hi, evB[(size_t)j * LPP + (L - 1 - L0)], 0));
                                ls_q8_layer(L, p0s[j], Cchs[j], d.q8s[j], hi, j, nullptr, 0);
                                CK(hipEventRecord(evA[(size_t)j * LPP + (L - L0)], hi));
                                CK(hipStreamWaitEvent(lo, evA[(size_t)j * LPP + (L - L0)], 0));
                                ls_q8_layer(L, p0s[j], Cchs[j], d.q8s[j], lo, j, nullptr, 1);
                                CK(hipEventRecord(evB[(size_t)j * LPP + (L - L0)], lo));
                                if (L == L1 - 1) { CK(hipEventRecord(evd[k], lo)); C.ls_rec[g].store(k, std::memory_order_release); }
                            }
                        }
                    }
                    CK(hipStreamSynchronize(hi));
                    CK(hipStreamSynchronize(lo));
                    C.b1.wait(); C.b2.wait();                       // nobody frees an event a peer may still wait on
                    for (int i = 0; i < 2 * LPP; ++i) { CK(hipEventDestroy(evA[i])); CK(hipEventDestroy(evB[i])); }
                    for (int k = 0; k < nch; ++k) CK(hipEventDestroy(evd[k]));
                    evd.clear();
                } else if (g_ls_q8) {
                    // ---- Q27_LS_Q8 SCHEDULE: no barrier rounds, no host drains. Chunk k runs on stream
                    // hs[k&1] with scratch set k&1. Ordering is device-side: chunk k+1 enters layer L only
                    // after chunk k's stage A (attention/GDN, the KV rows / recurrent state) of layer L
                    // (event evA); the next card starts chunk k after this card's done-event for chunk k
                    // (hipStreamWaitEvent across devices) plus the pend/inv copies on its own stream. The
                    // host only spins on a per-card "recorded up to k" counter, so every card keeps two
                    // chunks in flight and the cross-card fill is the only idle left.
                    std::vector<hipEvent_t>& evd = g_ls_evdone[g];
                    evd.assign((size_t)nch, nullptr);
                    for (int k = 0; k < nch; ++k) CK(hipEventCreateWithFlags(&evd[k], hipEventDisableTiming));
                    std::vector<hipEvent_t> evA((size_t)2 * LPP, nullptr);
                    for (int i = 0; i < 2 * LPP; ++i) CK(hipEventCreateWithFlags(&evA[i], hipEventDisableTiming));
                    // THE RING RECORD MUST BE RESET BETWEEN TWO BARRIERS, NOT BEFORE BOTH OF THEM.
                    // ls_rec[g] is a MONOTONIC per-card job counter that peers spin on: `while (ls_rec[src] < pos) {}`.
                    // Resetting it to -1 before b1 lets a card that has finished its round drive the counter BACKWARDS
                    // while a peer is still spinning on it for the OLD round -- that peer's condition can then never
                    // be satisfied and it never reaches any barrier again. Measured 2026-09-16, deterministic on the
                    // SERVE_RESET -> SERVE_GROW -> sweep path: "FATAL TpBar ... arrived 2", with card2/card3 at gen=3318
                    // job=-1 (already reset, one round ahead) and card0/card1 at gen=3317 job=3 (still consuming).
                    // b1 = every card has left the previous round, so nobody is still reading these cells.
                    // b2 = every reset is visible before any producer publishes, so nobody reads a STALE-HIGH cell.
                    // Both hazards are real; the code had only the second fence.
                    C.b1.wait();
                    C.ls_rec[g].store(-1, std::memory_order_release);
                    C.b2.wait();                                   // every card's event table exists
                    for (int k = 0; k < nch; ++k) {
                        const int par = k & 1;
                        hipStream_t sk = q8one ? d.hs[0] : d.hs[par];
                        const int p0 = sw_beg + lsoff(k);
                        const int cwk = lschw(k);
                        const int Cch = (sw_end - p0 < cwk) ? (sw_end - p0) : cwk;
                        if (g > 0) {
                            while (C.ls_rec[g - 1].load(std::memory_order_acquire) < k) { }
                            CK(hipStreamWaitEvent(sk, g_ls_evdone[g - 1][k], 0));
                            CK(hipMemcpyPeerAsync(d.pend_slots + (size_t)(p0 - sb) * Q27_HID, d.id,
                                                  D[g - 1].pend_slots + (size_t)(p0 - sb) * Q27_HID, D[g - 1].id,
                                                  (size_t)Cch * Q27_HID * 4, sk));
                            CK(hipMemcpyAsync(d.inv_slots + (p0 - sb), D[g - 1].inv_slots + (p0 - sb), (size_t)Cch * 4,
                                              hipMemcpyHostToHost, sk));
                        }
                        for (int L = L0; L < L1; ++L) {
                            if (k > 0) CK(hipStreamWaitEvent(sk, evA[(size_t)(par ^ 1) * LPP + (L - L0)], 0));
                            ls_q8_layer(L, p0, Cch, d.q8s[par], sk, par, evA[(size_t)par * LPP + (L - L0)], 2);
                        }
                        CK(hipEventRecord(evd[k], sk));
                        C.ls_rec[g].store(k, std::memory_order_release);
                    }
                    CK(hipStreamSynchronize(d.hs[0]));
                    CK(hipStreamSynchronize(d.hs[1]));
                    C.b1.wait(); C.b2.wait();                       // nobody frees an event a peer may still wait on
                    for (int i = 0; i < 2 * LPP; ++i) CK(hipEventDestroy(evA[i]));
                    for (int k = 0; k < nch; ++k) CK(hipEventDestroy(evd[k]));
                    evd.clear();
                } else
                for (int it = 0; it < nit; ++it) {
                    const int k = it - 1 - g;             // this thread's chunk index (-1 or >= nch = none)
                    if (!ls_serial) { C.b1.wait(); C.b2.wait(); }
                    if (k >= 0 && k < nch && g > 0 && !q27_env_flag("Q27_LS_NOCOPY", false)) {
                        const int q0 = sw_beg + lsoff(k);
                        const int cwk = lschw(k);
                        const int Cc = (sw_end - q0 < cwk) ? (sw_end - q0) : cwk;
                        CK(hipMemcpyPeerAsync(d.pend_slots + (size_t)(q0 - sb) * Q27_HID, d.id,
                                              D[g - 1].pend_slots + (size_t)(q0 - sb) * Q27_HID, D[g - 1].id,
                                              (size_t)Cc * Q27_HID * 4, d.stream));
                        // inv_slots is PINNED HOST memory: a plain host memcpy after the
                        // barrier round (the producer's ev_h sync made the zero-copy writes visible).
                        memcpy(d.inv_slots + (q0 - sb), D[g - 1].inv_slots + (q0 - sb), (size_t)Cc * 4);
                    }
                    if (k < 0 || k >= nch) continue;
                    const int p0 = sw_beg + lsoff(k);
                    const int cwk2 = lschw(k);
                    const int Cch = (sw_end - p0 < cwk2) ? (sw_end - p0) : cwk2;
                    lsp(15);   // barrier wait + handoff copy since the previous chunk's tail
                    for (int L = L0; L < L1; ++L) {
                        if (q27_env_flag("Q27_LS_DBG", false))
                            std::fprintf(stderr, "LS_DBG start g=%d k=%d L=%d\n", g, k, L);
                        // Per-layer state slots are PURE FUNCTIONS of L (Q27_IS_FULL(L) == L%4==3).
                        // Full layers are L=3,7,11,...: their KV slot is L>>2 (3->0, 63->15).
                        // GDN layers count the full layers below them: slot = L - (L>>2)
                        // (0->0, 4->3, 48->36, 62->47). No running counters anywhere: the
                        // chunk-major schedule must never change a state address.
                        const int fsl = L >> 2;              // full-attention state slot (L%4==3)
                        const int gsl = L - fsl;             // GDN state slot (L%4!=3)
                        const q27_layer_t* lay = q27_layer_ls(m, L, d.id);
                        const q27_layer_t* nxt = (L + 1 < L1) ? q27_layer_ls(m, L + 1, d.id) : nullptr;
                        const int isf = Q27_IS_FULL(L);
                        if (g_ls_q8) { ls_q8_layer(L, p0, Cch, d.q8s[0], d.stream, 0, nullptr, 2); } else {
                        signed char* pq1 = d.xq1_slots;
                        float*       ps1 = d.xs1_slots;
                        // ---- prologue: input norm -> xqin/xsin; hidden_slots gets the bf16 h ----
                        // GDN layers also need norm_slots (the a/b GEMV input); full layers do not.
                        unsigned short* ynorm = isf ? nullptr : d.norm_slots;
                        const float in_scl = isf ? lay->q_proj.in_scale : lay->in_qkv.in_scale;
                        if (q27_env_flag("Q27_LS_DBG", false))
                            std::fprintf(stderr, "LS_DBG norm g=%d k=%d L=%d x=%p w=%p hid=%p inv=%p Cch=%d\n",
                                g, k, L, (const void*)(d.pend_slots + (size_t)(p0 - sb) * Q27_HID),
                                (const void*)lay->input_norm,
                                (const void*)(d.hidden_slots + (size_t)(p0 - sb) * Q27_HID),
                                (const void*)(d.inv_slots + (p0 - sb)), Cch);
                        if (!(g_pf_norm_b && L > 0 &&
                              q27_rmsnorm_hostss_fp8_b(d.pend_slots + (size_t)(p0 - sb) * Q27_HID, lay->input_norm,
                                    d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, ynorm,
                                    d.xqin_slots, d.xsin_slots, Q27_HID, 1, in_scl,
                                    d.inv_slots + (p0 - sb), Cch, d.stream)))
                        for (int c = 0; c < Cch; ++c) {
                            const int p = p0 + c;
                            double mark = tp_now();
                            const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                            run_layer_tp(d, lay, p, gsl, fsl, ctx, 0, 1, C, &mark, pendp, nxt, p - sb,
                                         d.inv_slots + (p - sb), nullptr, nullptr, false, 1,
                                         d.xqin_slots + (size_t)c * Q27_HID,
                                         d.xsin_slots + (size_t)c * (Q27_HID / 16),
                                         nullptr, ynorm ? (ynorm + (size_t)c * Q27_HID) : nullptr);
                        }
                        lsp(0);
                        if (isf) {
                            // ---- full attention: q/k/v, attention over the chunk, o_proj, fused tail ----
                            const int OLb = Q27_OROWS;                    // nd=1: the whole o projection
                            const int qstride = QLb + 2 * KVLb;
                            // The 8-wide s1_b family declines at Cch > Q27_PF_MC (it silently
                            // returns 0, which left qkva_slots zeroed at nd=1 -- the L=3 divide).
                            // The chunk-wide bf16 fp8 GEMM (wideb) is the right geometry here;
                            // it already serves in_qkv/in_z and writes the same [M][R] layout the
                            // attention chunk reads with qstride.
                            if (!(g_ls_nv_qkv &&
                                  ((g_ls_pf7 && q27_wide_down_b_rb2p3(&lay->q4, &lay->k4, &lay->v4, d.xqin_slots, d.xsin_slots,
                                                                      d.qkva_slots, d.qkva_slots + QLb, d.qkva_slots + QLb + KVLb, qstride, Cch, d.stream)) ||
                                   (g_ls_pf2 && q27_wide_down_b_rb2p(&lay->q4, d.xqin_slots, d.xsin_slots, d.qkva_slots, qstride, Cch, g_ls_w8_tcb, d.stream)) ||
                                   (g_ls_pf && q27_wide_down_b_rbpf(&lay->q4, d.xqin_slots, d.xsin_slots, d.qkva_slots, qstride, Cch, g_ls_w8_tcb, d.stream)) ||
                                   q27_wide_down_b(&lay->q4, d.xqin_slots, d.xsin_slots, d.qkva_slots, qstride, Cch, g_ls_nv_ki, d.stream)) &&
                                  ((g_ls_pf2 && !g_ls_pf7 && q27_wide_down_b_rb2p(&lay->k4, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb, qstride, Cch, g_ls_w8_tcb, d.stream)) ||
                                   (g_ls_pf && q27_wide_down_b_rbpf(&lay->k4, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb, qstride, Cch, g_ls_w8_tcb, d.stream)) ||
                                   q27_wide_down_b(&lay->k4, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb, qstride, Cch, g_ls_nv_ki, d.stream)) &&
                                  ((g_ls_pf2 && !g_ls_pf7 && q27_wide_down_b_rb2p(&lay->v4, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb + KVLb, qstride, Cch, g_ls_w8_tcb, d.stream)) ||
                                   (g_ls_pf && q27_wide_down_b_rbpf(&lay->v4, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb + KVLb, qstride, Cch, g_ls_w8_tcb, d.stream)) ||
                                   q27_wide_down_b(&lay->v4, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb + KVLb, qstride, Cch, g_ls_nv_ki, d.stream))) &&
                                !(g_pf_x4w &&
                                  q27_proj_fp8_wideb(&lay->q_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots, Cch, qstride, g_pf_x4w, d.stream) &&
                                  q27_proj_fp8_wideb(&lay->k_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb, Cch, qstride, g_pf_x4w, d.stream) &&
                                  q27_proj_fp8_wideb(&lay->v_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb + KVLb, Cch, qstride, g_pf_x4w, d.stream)))
                                { std::fprintf(stderr, "Q27_LS: wide q/k/v declined (L%d) qK=%d qrows=%d qw=%p x4w=%d M=%d; aborting\n", L, lay->q_proj.K, lay->q_proj.rows, (const void*)lay->q_proj.w, g_pf_x4w, Cch); std::exit(1); }
                            lsp(1);
                            // Q27_PF_KVP2P: hand the sweep the four cards' KV head-quarters for this
                            // layer. Each card stores exactly the quarter its TP decode reads, so the
                            // owner's attention reads the three remote quarters out of the peers
                            // instead of keeping a duplicate full-width row (worth 1.875 GiB/card at
                            // ctx=262144). Null when the flag is off -- then this is the old kernel.
                            q27_kvquad_t QQ; QQ.on = 0;
                            for (int i = 0; i < 4; ++i) { QQ.kc[i] = nullptr; QQ.ks[i] = nullptr; QQ.vc[i] = nullptr; QQ.vs[i] = nullptr; }
                            if (g_pf_kvp2p) {
                                QQ.on = 1;
                                for (int h = 0; h < ndev; ++h) {
                                    QQ.kc[h] = D[h].kc + D[h].kv_off[fsl]; QQ.ks[h] = D[h].ks + D[h].kvs_off[fsl];
                                    QQ.vc[h] = D[h].vc + Q27_VOFF(D[h].kv_off[fsl]); QQ.vs[h] = D[h].vs + Q27_VSOFF(D[h].kvs_off[fsl]);
                                }
                            }
                            if (!(g_pf_att_chunk &&
                                  q27_attn_chunk_tp(d.qkva_slots, qstride, lay->q_norm, lay->k_norm,
                                                    d.kc + d.kv_off[fsl], d.ks + d.kvs_off[fsl], d.vc + Q27_VOFF(d.kv_off[fsl]), d.vs + Q27_VSOFF(d.kvs_off[fsl]),
                                                    d.qh_slots, d.mix_tile, OLb, p0, Cch, 1,
                                                    d.mixq_slots, d.mixs_slots, lay->o_proj.in_scale, g_att_hpw, g_att_pf, d.attn_pob, d.attn_pml, d.stream,
                                                    g_pf_kvp2p ? &QQ : nullptr)))
                                { std::fprintf(stderr, "Q27_LS: attn chunk declined (L%d); aborting\n", L); std::exit(1); }
                            lsp(2);
                            // o_proj -> part_slots; nd=1 so the residual share is 1.0, not 1/ndev
                            if (!(g_ls_nv_op && ((g_ls_pf2 && q27_wide_down_rb2p(&lay->o4, d.mixq_slots, d.mixs_slots, d.part_slots, Cch,
                                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, 1.0f, g_ls_w8_tcb, d.stream)) ||
                                                   (g_ls_pf && q27_wide_down_rbpf(&lay->o4, d.mixq_slots, d.mixs_slots, d.part_slots, Cch,
                                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, 1.0f, g_ls_w8_tcb, d.stream)) ||
                                                   q27_wide_down(&lay->o4, d.mixq_slots, d.mixs_slots, d.part_slots, Cch,
                                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, 1.0f, g_ls_nv_ki, d.stream))) &&
                                !(g_pf_x4w && q27_proj_fp8_wide(&lay->o_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, Q27_HID, 1.0f, g_pf_x4w, d.stream)) &&
                                !(g_pf_m2f2 && (g_pf_m2f3 ? q27_proj_fp8_m2_res3 : q27_proj_fp8_m2_res2)(&lay->o_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, Q27_HID, 1.0f, g_pf_m2f2, d.stream)) &&
                                !(g_pf_m3f && q27_proj_fp8_m3_res(&lay->o_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, Q27_HID, 1.0f, g_pf_m3f, d.stream)))
                                q27_proj_fp8_m2_res(&lay->o_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                    d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, Q27_HID, 1.0f, d.stream);
                            lsp(3);
                            if (q27_env_flag("Q27_LS_DBG", false) && L == 3) {
                                CK(hipStreamSynchronize(d.stream));
                                std::vector<float> qv = q27_d2h_bf16(d.qkva_slots, 4);
                                std::vector<float> kv = q27_d2h_bf16(d.qkva_slots + QLb, 2);
                                float mq[4];
                                CK(hipMemcpy(mq, d.mixq_slots, 16, hipMemcpyDeviceToHost));
                                std::fprintf(stderr, "LS_DBG L3 post-attn q=%.3f %.3f k=%.3f %.3f mq=%.1f %.1f %.1f %.1f\n",
                                    (double)qv[0],(double)qv[1],(double)kv[0],(double)kv[1],
                                    (double)mq[0],(double)mq[1],(double)mq[2],(double)mq[3]);
                            }
                            q27_red_rn_perm_b1(d.part_slots, lay->post_norm,
                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, nullptr,
                                        pq1, ps1, Q27_HID, 1, lay->gate.in_scale, Cch, d.stream);
                            lsp(4);
                        } else {
                            // ---- GDN: in_qkv/in_z, conv + recurrence over the chunk-as-tile, out_proj, fused tail ----
                            const int QKVLb = Q27_GDN_QKV, ZLb = Q27_GDN_Z;   // nd=1: full widths
                            if (q27_env_flag("Q27_LS_DBG", false) && L == 5) {
                                CK(hipStreamSynchronize(d.stream));
                                float tv[4];
                                CK(hipMemcpy(tv, d.xqin_slots, 16, hipMemcpyDeviceToHost));
                                std::vector<float> nv = q27_d2h_bf16(d.norm_slots, 4);
                                std::fprintf(stderr, "LS_DBG L5 pre-iqkv xq=%.3f %.3f %.3f %.3f norm=%.3f %.3f %.3f %.3f\n",
                                    (double)tv[0],(double)tv[1],(double)tv[2],(double)tv[3],
                                    (double)nv[0],(double)nv[1],(double)nv[2],(double)nv[3]);
                            }
                            if (g_ls_pf8) {
                                if (!q27_wide_down_rb2p_sk2(&lay->iqkv4, d.xqin_slots, d.xsin_slots, d.qkv2a, d.qkv2b, d.mix3, d.mix4, g_ls_pf8, Cch, nullptr, 0.f, d.stream))
                                    std::fprintf(stderr, "Q27_LS: sk2 in_qkv declined (L%d)\n", L);
                                if (g_ls_pf8 >= 4) { q27_add_f32(d.qkv2a, d.mix3, Cch * QKVLb, d.stream); q27_add_f32(d.qkv2a, d.mix4, Cch * QKVLb, d.stream); }
                                q27_red2_bf16(d.qkv_slots, d.qkv2a, d.qkv2b, nullptr, Cch * QKVLb, d.stream);
                                if (!q27_wide_down_rb2p_sk2(&lay->iz4, d.xqin_slots, d.xsin_slots, d.zb2a, d.zb2b, d.mix3, d.mix4, g_ls_pf8, Cch, nullptr, 0.f, d.stream))
                                    std::fprintf(stderr, "Q27_LS: sk2 in_z declined (L%d)\n", L);
                                if (g_ls_pf8 >= 4) { q27_add_f32(d.zb2a, d.mix3, Cch * ZLb, d.stream); q27_add_f32(d.zb2a, d.mix4, Cch * ZLb, d.stream); }
                                q27_red2_bf16(d.zbuf_slots, d.zb2a, d.zb2b, nullptr, Cch * ZLb, d.stream);
                            } else
                            if (!(g_ls_nv_iq && ((g_ls_pf2 && q27_wide_down_b_rb2p(&lay->iqkv4, d.xqin_slots, d.xsin_slots, d.qkv_slots, QKVLb, Cch, g_ls_pf_tcbd, d.stream)) ||
                                                    (g_ls_pf && q27_wide_down_b_rbpf(&lay->iqkv4, d.xqin_slots, d.xsin_slots, d.qkv_slots, QKVLb, Cch, g_ls_pf_tcbd, d.stream)) ||
                                                    q27_wide_down_b(&lay->iqkv4, d.xqin_slots, d.xsin_slots, d.qkv_slots, QKVLb, Cch, g_ls_nv_ki, d.stream))
                                                        && ((g_ls_pf2 && q27_wide_down_b_rb2p(&lay->iz4,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, ZLb, Cch, g_ls_pf_tcbd, d.stream)) ||
                                                            (g_ls_pf && q27_wide_down_b_rbpf(&lay->iz4,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, ZLb, Cch, g_ls_pf_tcbd, d.stream)) ||
                                                            q27_wide_down_b(&lay->iz4,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, ZLb, Cch, g_ls_nv_ki, d.stream))) &&
                                !(g_pf_x4w && q27_proj_fp8_wideb(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, g_pf_x4w, d.stream)
                                                        && q27_proj_fp8_wideb(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, g_pf_x4w, d.stream)) &&
                                !(g_pf_mk2r && (g_pf_mk2r2 ? q27_proj_fp8_m2r2 : q27_proj_fp8_m2r)(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, 0, g_pf_mk2r, d.stream)
                                            && (g_pf_mk2r2 ? q27_proj_fp8_m2r2 : q27_proj_fp8_m2r)(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, 0, g_pf_mk2r, d.stream)) &&
                                !(g_pf_mk16 && q27_fp8_to_h2(&lay->in_qkv, d.w16_scratch, d.stream)
                                            && q27_proj_fp8_m2w16(&lay->in_qkv, d.w16_scratch, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, 0, d.stream)
                                            && q27_fp8_to_h2(&lay->in_z, d.w16_scratch, d.stream)
                                            && q27_proj_fp8_m2w16(&lay->in_z, d.w16_scratch, d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, 0, d.stream)) &&
                                !(g_pf_mk3 && q27_proj_fp8_m3l(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, 0, g_pf_mk3, d.stream)
                                           && q27_proj_fp8_m3l(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, 0, g_pf_mk3, d.stream))) {
                                q27_proj_fp8_m2(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, 0, d.stream);
                                q27_proj_fp8_m2(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, 0, d.stream);
                            }
                            lsp(5);
                            if (g_pf_gdn_scan) {
                                q27_bf16_gemv2_tile(lay->in_a, lay->in_b, d.norm_slots, Q27_HID, d.ab_tile, d.bb_tile,
                                                    Q27_GDN_VH, Q27_HID, Cch, d.stream);
                                if (q27_env_flag("Q27_LS_DBG", false) && L == 5) {
                                    CK(hipStreamSynchronize(d.stream));
                                    float tv[4], vv2[4], sv[4];
                                    CK(hipMemcpy(tv, d.qkv_slots, 16, hipMemcpyDeviceToHost));
                                    CK(hipMemcpy(vv2, d.qkv_slots + 4096, 16, hipMemcpyDeviceToHost));
                                    std::vector<float> av = q27_d2h_bf16(d.ab_tile, 4);
                                    std::vector<float> zv = q27_d2h_bf16(d.zbuf_slots, 4);
                                    CK(hipMemcpy(sv, d.S + d.s_off[gsl], 16, hipMemcpyDeviceToHost));
                                    std::vector<float> kv = q27_d2h_bf16(d.qkv_slots + 2048, 2);
                                    std::vector<float> al = q27_d2h_bf16(lay->A_log, 2);
                                    std::vector<float> db = q27_d2h_bf16(lay->dt_bias, 2);
                                    std::vector<float> gn = q27_d2h_bf16(lay->gdn_norm, 2);
                                    std::vector<float> bv2 = q27_d2h_bf16(d.bb_tile, 2);
                                    std::fprintf(stderr, "LS_DBG L5 post-gemv q=%.3f k=%.3f v=%.3f ab=%.3f bb=%.3f z=%.3f S=%.3f Alog=%.3f dtb=%.3f gn=%.3f oscale=%g\n",
                                        (double)tv[0],(double)kv[0],(double)vv2[0],
                                        (double)av[0],(double)bv2[0],(double)zv[0],
                                        (double)sv[0],(double)al[0],(double)db[0],
                                        (double)gn[0],(double)lay->out_proj.in_scale);
                                }
                                if (q27_env_flag("Q27_LS_DBG", false)) { CK(hipStreamSynchronize(d.stream)); std::fprintf(stderr, "LS_DBG ph gemv g=%d k=%d L=%d\n", g, k, L); }
                                lsp(6);
                                q27_gdn_conv_tp_tile(d.qkv_slots, QKVLb, Cch, d.conv + d.conv_off[gsl], lay->conv1d, 0, 1, d.stream);
                                if (q27_env_flag("Q27_LS_DBG", false) && (L == 1 || L == 2)) {
                                    CK(hipStreamSynchronize(d.stream));
                                    float tv[12];
                                    const int pe = p0 + Cch - 1 - sb;
                                    CK(hipMemcpy(tv, d.qkv_slots + (size_t)pe * QKVLb, 48, hipMemcpyDeviceToHost));
                                    CK(hipMemcpy(tv + 4, d.S + d.s_off[gsl], 16, hipMemcpyDeviceToHost));
                                    CK(hipMemcpy(tv + 8, d.S + d.s_off[gsl] + 4000, 16, hipMemcpyDeviceToHost));
                                    std::fprintf(stderr, "LS_DBG L%d ph-conv qE=%.3f %.3f %.3f %.3f S0=%.4g %.4g %.4g %.4g S4k=%.4g %.4g %.4g %.4g\n",
                                        L, (double)tv[0],(double)tv[1],(double)tv[2],(double)tv[3],
                                        (double)tv[4],(double)tv[5],(double)tv[6],(double)tv[7],
                                        (double)tv[8],(double)tv[9],(double)tv[10],(double)tv[11]);
                                }
                                if (q27_env_flag("Q27_LS_DBG", false)) { CK(hipStreamSynchronize(d.stream)); std::fprintf(stderr, "LS_DBG ph conv g=%d k=%d L=%d\n", g, k, L); }
                                lsp(7);
                                if (q27_env_flag("Q27_LS_SDBG", false))
                                    std::fprintf(stderr, "LS_SDBG gs=%d fs=%d tp_s=%zu tp_kv=%zu L=%d\n",
                                                 gs, fs, d.tp_s, d.tp_kv, L);
                                (g_pf_gdn_scan2 ? q27_gdn_scan2_tp : q27_gdn_scan_tp)(d.qkv_slots, QKVLb, d.zbuf_slots, ZLb, d.ab_tile, d.bb_tile, Q27_GDN_VH,
                                                lay->A_log, lay->dt_bias, lay->gdn_norm, d.S + d.s_off[gsl], d.mix_tile, ZLb,
                                                0, 1, Cch, d.mixq_tile, d.mixs_tile, lay->out_proj.in_scale, d.stream);
                                if (q27_env_flag("Q27_LS_DBG", false) && (L == 1 || L == 2)) {
                                    CK(hipStreamSynchronize(d.stream));
                                    float tv[8]; signed char tq[4];
                                    const int pe = p0 + Cch - 1 - sb;
                                    CK(hipMemcpy(tv, d.mix_tile + (size_t)pe * ZLb, 32, hipMemcpyDeviceToHost));
                                    CK(hipMemcpy(tq, d.mixq_tile + (size_t)pe * ZLb, 4, hipMemcpyDeviceToHost));
                                    CK(hipMemcpy(tv + 4, d.mixs_tile + (size_t)pe * (ZLb / 16), 16, hipMemcpyDeviceToHost));
                                    std::fprintf(stderr, "LS_DBG L%d ph-scan mixE=%.3f %.3f %.3f %.3f mixqE=%d %d %d %d mixsE=%.4g %.4g %.4g %.4g\n",
                                        L, (double)tv[0],(double)tv[1],(double)tv[2],(double)tv[3],
                                        (int)tq[0],(int)tq[1],(int)tq[2],(int)tq[3],
                                        (double)tv[4],(double)tv[5],(double)tv[6],(double)tv[7]);
                                }
                                if (q27_env_flag("Q27_LS_DBG", false)) { CK(hipStreamSynchronize(d.stream)); std::fprintf(stderr, "LS_DBG ph scan g=%d k=%d L=%d\n", g, k, L); }
                                lsp(8);
                                if (q27_env_flag("Q27_LS_DBG", false) && L == 5) {
                                    CK(hipStreamSynchronize(d.stream));
                                    float tv[4];
                                    CK(hipMemcpy(tv, d.mixq_tile, 16, hipMemcpyDeviceToHost));
                                    std::vector<float> mv = q27_d2h_bf16(d.mix_tile, 4);
                                    std::fprintf(stderr, "LS_DBG L5 post-scan mixq=%.3f %.3f %.3f %.3f mix=%.3f %.3f %.3f %.3f\n",
                                        (double)tv[0],(double)tv[1],(double)tv[2],(double)tv[3],
                                        (double)mv[0],(double)mv[1],(double)mv[2],(double)mv[3]);
                                }
                                C.pre_in[g] = 0;
                            } else
                            for (int c = 0; c < Cch; ++c) {
                                // per-position serial GDN body (the validated ndev-parametric path):
                                // gemv/conv/step + fused quantize into the chunk slots
                                const int p = p0 + c;
                                double mark = tp_now();
                                run_layer_tp(d, lay, p, gsl, fsl, ctx, 0, 1, C, &mark, nullptr, nxt, p - sb,
                                             d.inv_slots + (p - sb), nullptr, nullptr, false, 6,
                                             nullptr, nullptr, nullptr,
                                             d.norm_slots + (size_t)c * Q27_HID,
                                             d.qkv_slots + (size_t)c * QKVLb,
                                             d.zbuf_slots + (size_t)c * ZLb,
                                             d.part_slots + (size_t)c * Q27_HID, nullptr, nullptr,
                                             d.mixq_slots + (size_t)c * ZLb, d.mixs_slots + (size_t)c * (ZLb / 16));
                            }
                            const signed char* mq = g_pf_gdn_scan ? d.mixq_tile : d.mixq_slots;
                            const float*       ms = g_pf_gdn_scan ? d.mixs_tile : d.mixs_slots;
                            if (g_ls_pf8) {
                                const unsigned short* oradd = (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr;
                                if (!q27_wide_down_rb2p_sk2(&lay->op4, mq, ms, d.op2a, d.op2b, d.mix3, d.mix4, g_ls_pf8, Cch, nullptr, 0.f, d.stream))
                                    std::fprintf(stderr, "Q27_LS: sk2 out_proj declined (L%d)\n", L);
                                if (g_ls_pf8 >= 4) { q27_add_f32(d.op2a, d.mix3, Cch * Q27_HID, d.stream); q27_add_f32(d.op2a, d.mix4, Cch * Q27_HID, d.stream); }
                                q27_red2_f32(d.part_slots, d.op2a, d.op2b, oradd, Cch * Q27_HID, d.stream);
                            } else
                            if (!(g_ls_nv_op && ((g_ls_pf2 && q27_wide_down_rb2p(&lay->op4, mq, ms, d.part_slots, Cch,
                                                        (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                        1.0f, g_ls_pf_tcbd, d.stream)) ||
                                                   (g_ls_pf && q27_wide_down_rbpf(&lay->op4, mq, ms, d.part_slots, Cch,
                                                        (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                        1.0f, g_ls_pf_tcbd, d.stream)) ||
                                                   q27_wide_down(&lay->op4, mq, ms, d.part_slots, Cch,
                                                        (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                        1.0f, g_ls_nv_ki, d.stream))) &&
                                !(g_pf_x4w && q27_proj_fp8_wide(&lay->out_proj, mq, ms, d.part_slots, Cch, Q27_HID,
                                                        (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                        Q27_HID, 1.0f, g_pf_x4w, d.stream)) &&
                                !(g_pf_m2f2 && (g_pf_m2f3 ? q27_proj_fp8_m2_res3 : q27_proj_fp8_m2_res2)(&lay->out_proj, mq, ms, d.part_slots, Cch, Q27_HID,
                                                        (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                        Q27_HID, 1.0f, g_pf_m2f2, d.stream)) &&
                                !(g_pf_m3f && q27_proj_fp8_m3_res(&lay->out_proj, mq, ms, d.part_slots, Cch, Q27_HID,
                                                        (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                        Q27_HID, 1.0f, g_pf_m3f, d.stream)))
                                q27_proj_fp8_m2_res(&lay->out_proj, mq, ms, d.part_slots, Cch, Q27_HID,
                                                    (!g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                    Q27_HID, 1.0f, d.stream);
                            lsp(9);
                            if (q27_env_flag("Q27_LS_DBG", false) && L == 2) {
                                CK(hipStreamSynchronize(d.stream));
                                float t4[16];
                                const int pe = p0 + Cch - 1 - sb;
                                CK(hipMemcpy(t4, d.part_slots + (size_t)pe * Q27_HID, 64, hipMemcpyDeviceToHost));
                                std::fprintf(stderr, "LS_DBG L2 ph-op partE=%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f\n",
                                    (double)t4[0],(double)t4[1],(double)t4[2],(double)t4[3],
                                    (double)t4[4],(double)t4[5],(double)t4[6],(double)t4[7],
                                    (double)t4[8],(double)t4[9],(double)t4[10],(double)t4[11],
                                    (double)t4[12],(double)t4[13],(double)t4[14],(double)t4[15]);
                            }
                            q27_red_rn_perm_b1(d.part_slots, lay->post_norm,
                                        d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, nullptr,
                                        pq1, ps1, Q27_HID, 1, lay->gate.in_scale, Cch, d.stream);
                            lsp(10);
                            if (q27_env_flag("Q27_LS_DBG", false) && L == 2) {
                                CK(hipStreamSynchronize(d.stream));
                                signed char x1[8]; float s1[8];
                                const int pe = p0 + Cch - 1 - sb;
                                CK(hipMemcpy(x1, d.xq1_slots + (size_t)pe * Q27_HID, 8, hipMemcpyDeviceToHost));
                                CK(hipMemcpy(s1, d.xs1_slots + (size_t)pe * (Q27_HID / 16), 32, hipMemcpyDeviceToHost));
                                std::fprintf(stderr, "LS_DBG L2 ph-rn xq1E=%d %d %d %d | %d %d %d %d xs1E=%.4g %.4g %.4g %.4g | %.4g %.4g %.4g %.4g\n",
                                    (int)x1[0],(int)x1[1],(int)x1[2],(int)x1[3],
                                    (int)x1[4],(int)x1[5],(int)x1[6],(int)x1[7],
                                    (double)s1[0],(double)s1[1],(double)s1[2],(double)s1[3],
                                    (double)s1[4],(double)s1[5],(double)s1[6],(double)s1[7]);
                            }
                        }
                        // ---- MLP half: wide FFN at the full chunk (nd=1) + fused mixer tail ----
                        if (!(g_ls_w8 && q27_wide_gu_w8(&lay->gate, &lay->up, d.xq1_slots, d.xs1_slots, d.pa_mlp, d.pb_mlp, Cch, g_ls_w8_tcb, d.stream)) &&
                            !(g_ls_pf9 && q27_wide_gu_rb3p(&lay->gate, &lay->up, d.xq1_slots, d.xs1_slots, d.pa_mlp, d.pb_mlp, Cch, g_ls_w8_tcb, d.stream)) &&
                            !(g_ls_pf2 && q27_wide_gu_rb2p(&lay->gate, &lay->up, d.xq1_slots, d.xs1_slots, d.pa_mlp, d.pb_mlp, Cch, g_ls_w8_tcb, d.stream)) &&
                            !(g_ls_pf && q27_wide_gu_rbpf(&lay->gate, &lay->up, d.xq1_slots, d.xs1_slots, d.pa_mlp, d.pb_mlp, Cch, g_ls_w8_tcb, d.stream)) &&
                            !q27_wide_gu(&lay->gate, &lay->up, d.xq1_slots, d.xs1_slots, d.pa_mlp, d.pb_mlp, Cch, g_pf_wide_kg, d.stream))
                            std::fprintf(stderr, "Q27_LS: wide gate+up declined (L%d Cch=%d)\n", L, Cch);
                        lsp(11);
                        if (q27_env_flag("Q27_LS_DBG", false) && L == 2) {
                            CK(hipStreamSynchronize(d.stream));
                            float t4[4];
                            const int pe = p0 + Cch - 1 - sb;
                            CK(hipMemcpy(t4, d.pa_mlp + (size_t)pe * Q27_INTER, 16, hipMemcpyDeviceToHost));
                            std::fprintf(stderr, "LS_DBG L2 ph-gu paE=%.3f %.3f %.3f %.3f\n",
                                (double)t4[0],(double)t4[1],(double)t4[2],(double)t4[3]);
                            CK(hipMemcpy(t4, d.pb_mlp + (size_t)pe * Q27_INTER, 16, hipMemcpyDeviceToHost));
                            std::fprintf(stderr, "LS_DBG L2 ph-gu pbE=%.3f %.3f %.3f %.3f\n",
                                (double)t4[0],(double)t4[1],(double)t4[2],(double)t4[3]);
                        }
                        if (!q27_swiglu_quant_b(d.pa_mlp, d.pb_mlp, d.xq2_slots, d.xs2_slots, Q27_INTER, lay->down.in_scale, Cch, d.stream))
                            std::fprintf(stderr, "Q27_LS: wide swiglu declined (L%d)\n", L);
                        lsp(12);
                        if (q27_env_flag("Q27_LS_DBG", false) && L == 2) {
                            CK(hipStreamSynchronize(d.stream));
                            signed char t4[4]; float t5[4];
                            const int pe = p0 + Cch - 1 - sb;
                            CK(hipMemcpy(t4, d.xq2_slots + (size_t)pe * Q27_INTER, 4, hipMemcpyDeviceToHost));
                            CK(hipMemcpy(t5, d.xs2_slots + (size_t)pe * (Q27_INTER / 16), 16, hipMemcpyDeviceToHost));
                            std::fprintf(stderr, "LS_DBG L2 ph-sw xq2E=%d %d %d %d xs2E=%.4g %.4g %.4g %.4g\n",
                                (int)t4[0],(int)t4[1],(int)t4[2],(int)t4[3],
                                (double)t5[0],(double)t5[1],(double)t5[2],(double)t5[3]);
                        }
                        const unsigned short* lradd = g_hostss_neg ? nullptr
                            : (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID);
                        if (!(g_ls_w8 && q27_wide_down_w8(&lay->down, d.xq2_slots, d.xs2_slots, d.mixer_slots, Cch, lradd, 1.0f, g_ls_w8_tcb, d.stream)) &&
                            !(g_ls_pf6 && (q27_wide_down_rb2p_sk2(&lay->down, d.xq2_slots, d.xs2_slots, d.mixer_slots, d.mix2, d.mix3, d.mix4, g_ls_pf6, Cch, lradd, 1.0f, d.stream) &&
                                          (q27_add_f32(d.mixer_slots, d.mix2, Cch * Q27_HID, d.stream), 1) &&
                                          (g_ls_pf6 < 4 || (q27_add_f32(d.mixer_slots, d.mix3, Cch * Q27_HID, d.stream), 1)) &&
                                          (g_ls_pf6 < 4 || (q27_add_f32(d.mixer_slots, d.mix4, Cch * Q27_HID, d.stream), 1)))) &&
                            !(g_ls_pf5 && q27_wide_down_rb2p_tr32(&lay->down, d.xq2_slots, d.xs2_slots, d.mixer_slots, Cch, lradd, 1.0f, d.stream)) &&
                            !(g_ls_pf2 && q27_wide_down_rb2p(&lay->down, d.xq2_slots, d.xs2_slots, d.mixer_slots, Cch, lradd, 1.0f, g_ls_pf_tcbd, d.stream)) &&
                            !(g_ls_pf && q27_wide_down_rbpf(&lay->down, d.xq2_slots, d.xs2_slots, d.mixer_slots, Cch, lradd, 1.0f, g_ls_pf_tcbd, d.stream)) &&
                            !q27_wide_down(&lay->down, d.xq2_slots, d.xs2_slots, d.mixer_slots, Cch, lradd, 1.0f, g_pf_wide_kd, d.stream))
                            std::fprintf(stderr, "Q27_LS: wide down declined (L%d)\n", L);
                        lsp(13);
                        if (q27_env_flag("Q27_LS_DBG", false) && L == 2) {
                            CK(hipStreamSynchronize(d.stream));
                            float t4[4];
                            const int pe = p0 + Cch - 1 - sb;
                            CK(hipMemcpy(t4, d.mixer_slots + (size_t)pe * Q27_HID, 16, hipMemcpyDeviceToHost));
                            std::fprintf(stderr, "LS_DBG L2 ph-dn mixE=%.3f %.3f %.3f %.3f\n",
                                (double)t4[0],(double)t4[1],(double)t4[2],(double)t4[3]);
                        }
                        q27_red_b1(d.mixer_slots, d.pend_slots + (size_t)(p0 - sb) * Q27_HID,
                                   d.inv_slots + (p0 - sb), Q27_HID, Cch, d.stream);
                        lsp(14);
                        }
                        CK(hipEventRecord(ev_h[k & 1], d.stream));
                        if (q27_env_flag("Q27_LS_DBG", false)) {
                            CK(hipStreamSynchronize(d.stream));
                            float pv[4], pve[4];
                            CK(hipMemcpy(pv, d.pend_slots + (size_t)(p0 - sb) * Q27_HID, 16, hipMemcpyDeviceToHost));
                            CK(hipMemcpy(pve, d.pend_slots + (size_t)(p0 + Cch - 1 - sb) * Q27_HID, 16, hipMemcpyDeviceToHost));
                            std::vector<float> hv = q27_d2h_bf16(d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, 4);
                            std::vector<float> hve = q27_d2h_bf16(d.hidden_slots + (size_t)(p0 + Cch - 1 - sb) * Q27_HID, 4);
                            std::fprintf(stderr, "LS_DBG g=%d k=%d L=%d pend0=%.3f %.3f %.3f %.3f hid0=%.3f %.3f %.3f %.3f | pendE=%.3f %.3f %.3f %.3f hidE=%.3f %.3f %.3f %.3f\n",
                                g, k, L, (double)pv[0], (double)pv[1], (double)pv[2], (double)pv[3],
                                (double)hv[0], (double)hv[1], (double)hv[2], (double)hv[3],
                                (double)pve[0], (double)pve[1], (double)pve[2], (double)pve[3],
                                (double)hve[0], (double)hve[1], (double)hve[2], (double)hve[3]);
                        }

                    }
                    if (g < ndev - 1) CK(hipEventSynchronize(ev_h[k & 1]));   // publish chunk k's mixer
                    lsp(15);
                    if (q27_env_flag("Q27_LS_DBG", false) && k == nch - 1) {
                        CK(hipStreamSynchronize(d.stream));
                        float p0[8];
                        CK(hipMemcpy(p0, d.pend_slots + (size_t)(sw_end - 1 - sb) * Q27_HID, 32, hipMemcpyDeviceToHost));
                        std::vector<float> h0 = q27_d2h_bf16(d.hidden_slots + (size_t)(sw_end - 1 - sb) * Q27_HID, 8);
                        std::printf("DBG_LS_STAGE g=%d pend=%.6f %.6f %.6f %.6f | hid=%.6f %.6f %.6f %.6f\n",
                            g, (double)p0[0],(double)p0[1],(double)p0[2],(double)p0[3],
                            (double)h0[0],(double)h0[1],(double)h0[2],(double)h0[3]);
                    }
                }
                if (ls_time) {
                    CK(hipStreamSynchronize(d.stream));
                    std::printf("Q27_LS_TIME card=%d norm=%.1f attqkv=%.1f attn=%.1f oproj=%.1f atttail=%.1f "
                                "iqkv=%.1f gemv=%.1f conv=%.1f scan=%.1f outproj=%.1f gdntail=%.1f "
                                "gu=%.1f swi=%.1f down=%.1f redb1=%.1f bar=%.1f\n",
                                g, ls_ph[g][0], ls_ph[g][1], ls_ph[g][2], ls_ph[g][3], ls_ph[g][4],
                                ls_ph[g][5], ls_ph[g][6], ls_ph[g][7], ls_ph[g][8], ls_ph[g][9],
                                ls_ph[g][10], ls_ph[g][11], ls_ph[g][12], ls_ph[g][13], ls_ph[g][14], ls_ph[g][15]);
                }
                // ---- sweep done: one more round so every stage's last chunk is published, then
                // drain our own stream (the copies below read the OWNER's buffers via SDMA).
                if (ls_serial) { for (int i = 0; i < ndev - 1 - g; ++i) { C.b1.wait(); C.b2.wait(); } }
                else { C.b1.wait(); C.b2.wait(); }
                CK(hipStreamSynchronize(d.stream));
                // ---- hand the prefill state to the decode path, which keeps the TP layout:
                // copy each layer's full-width KV / GDN-S / conv state from its owner to every
                // other card at the same (full-size, nd=1-allocated) offsets. One-time ~200 MB.
                for (int L = 0; L < ((export_done || g_sr) ? 0 : nlayer); ++L) {   // skipped when the ring schedule exported early; Q27_LS_SR: no card decodes a layer it does not own, nothing to hand over
                    const int owner = lay_owner(L);   // rotating layer blocks (Q27_LS_BLK), balanced ownership (Q27_LS_BAL)
                    if (owner == g) continue;
                    const int fsc = L >> 2, gsc = L - fsc;   // pure functions of L
                    if (Q27_IS_FULL(L)) {
                        // the sweep's prefix only: positions past sw_end hold nothing yet, and the
                        // old copy moved the whole ctx-sized plane on every request.
                        // Q27_PF_P2P: nothing to export (see the ring path above).
                        if (!g_pf_kvp2p) q27_export_kv_quarter(d, D[owner], fsc, sw_beg, sw_end, d.stream);
                    } else {
                        CK(hipMemcpyPeerAsync(d.S + d.s_off[gsc], d.id,
                                              D[owner].S + D[owner].s_off[gsc] + (size_t)q27_tp_shard(g)->gv_h0 * Q27_GDN_D * Q27_GDN_D,
                                              D[owner].id, d.s_bytes[gsc], d.stream));
                        q27_copy_conv_quarter(D[owner], d, gsc, g, ndev, false, d.stream);
                    }
                }
                // ---- LS finals: the whole head runs on the LAST card; one barrier round shares
                // the token. (The other cards' TP-layout lm_head is for the decode path.)
                const q27_globals_t* GL = q27_globals_ls(m, d.id);
                const bool head_all = swpl || (orc != nullptr);
                for (int p = head_all ? sw_beg : sw_end - 1; p < sw_end; ++p) {
                    const double t0 = tp_now();
                    if (g == last_card) {
                        if (g_ls_q8 && !(q8old & 8)) {
                            // Q27_LS_Q8 finals: the release's arithmetic (h' of the last layer + its output vector,
                            // then final_norm and the NVFP4 head) on fp32 values; the only positions with an fp32
                            // h' are the last chunk's (hid32c), which is all the one-shot head needs.
                            if (head_all) { std::fprintf(stderr, "Q27_LS_Q8: the all-position head (ladder/oracle) is not wired; aborting\n"); std::exit(1); }
                            const int c_last = p - (sw_beg + lsoff(nch - 1));
                            if (c_last < 0 || c_last >= lschw(nch - 1)) { std::fprintf(stderr, "Q27_LS_Q8: head position outside the last chunk; aborting\n"); std::exit(1); }
                            float* hrow = d.q8s[last_set].hid32c + (size_t)c_last * Q27_HID;
                            const float* pendp32 = d.pend_slots + (size_t)(p - sb) * Q27_HID;
                            if (q27_env_flag("Q27_LS_FINDBG", false) && g == last_card) {
                                // The finals read set `last_set`, whose hid32c is written only by the
                                // job that ran the LAST layer of the LAST chunk. The comment at
                                // `last_set = (nch-1)&1` says the paired schedule keys it on job
                                // PARITY, so the two need not agree. Dump BOTH sets: the residual the
                                // head consumes is h' + pend, so a set that was never written shows up
                                // as hrow == 0 while its pend is not.
                                float a[4] = {0,0,0,0}, b4[4] = {0,0,0,0}, c4[4] = {0,0,0,0};
                                CK(hipMemcpy(a, hrow, sizeof a, hipMemcpyDeviceToHost));
                                CK(hipMemcpy(b4, pendp32, sizeof b4, hipMemcpyDeviceToHost));
                                const float* oth = d.q8s[last_set ^ 1].hid32c
                                                   ? d.q8s[last_set ^ 1].hid32c + (size_t)c_last * Q27_HID : nullptr;
                                if (oth) CK(hipMemcpy(c4, oth, sizeof c4, hipMemcpyDeviceToHost));
                                std::fprintf(stderr, "Q27_LS_FINDBG g=%d nch=%d last_set=%d c_last=%d p=%d sw_beg=%d sw_end=%d sb=%d "
                                             "hid32c_set%d=[%.6f %.6f %.6f %.6f] hid32c_set%d=[%.6f %.6f %.6f %.6f] pend=[%.6f %.6f %.6f %.6f]\n",
                                             g, nch, last_set, c_last, p, sw_beg, sw_end, sb,
                                             last_set, (double)a[0],(double)a[1],(double)a[2],(double)a[3],
                                             last_set ^ 1, (double)c4[0],(double)c4[1],(double)c4[2],(double)c4[3],
                                             (double)b4[0],(double)b4[1],(double)b4[2],(double)b4[3]);
                                std::fflush(stderr);
                            }
                            // Q27_LS_FIN_BF16=1: reproduce the NON-layer-split finals arithmetic --
                            // h' rounded to BF16, then BF16 += fp32 pend, then q27_rmsnorm +
                            // q27_quant_perm -- instead of feeding the fp32 residual straight into
                            // q27_red_b1 + q27_rmsnorm_hostss_perm. The TP path keeps its residual in
                            // hidden_slots (BF16) while the LS path keeps it fp32, so the two pipelines
                            // are NOT the same numerical function; the authoritative token stream
                            // (tests/reference_tokens.txt) was recorded on the BF16 one.
                            if (q27_env_flag("Q27_LS_FINAL_COMPLETE", true)) {
                                // Every LS down epilogue includes S.part with share
                                // 1.0. pend is therefore the complete layer output.
                                // Normalize it once, using the existing fp32 RMS
                                // producer, without another residual add or a host
                                // round trip for the inverse RMS.
                                if (!q27_red_rn_perm_b1(pendp32, GL->final_norm, d.norm, nullptr,
                                        d.xq, d.xs, Q27_HID, 1, GL->lm_head.in_scale, 1, d.stream)) {
                                    std::fprintf(stderr, "Q27_LS complete final norm declined\n"); std::abort();
                                }
                            } else if (q27_env_flag("Q27_LS_FIN_BF16", false)) {
                                q27_f2bf_vec(hrow, d.norm, Q27_HID, d.stream);
                                q27_add_inplace(d.norm, pendp32, Q27_HID, d.stream);
                                q27_rmsnorm(d.norm, GL->final_norm, d.hidden, Q27_HID, 1, d.stream);
                                q27_quant_perm(d.hidden, d.xq, d.xs, Q27_HID, GL->lm_head.in_scale, d.stream);
                            } else {
                            if (q27_env_flag("Q27_FIN_DUMP", false) && g == last_card) {
                                // the two SUMMANDS, not the sum: h' of the last layer (hid32c) and the
                                // pending residual, so the discrepancy can be attributed to one of them
                                static std::vector<float> fh(Q27_HID), fpp(Q27_HID);
                                CK(hipMemcpy(fh.data(), hrow, (size_t)Q27_HID * 4, hipMemcpyDeviceToHost));
                                CK(hipMemcpy(fpp.data(), pendp32, (size_t)Q27_HID * 4, hipMemcpyDeviceToHost));
                                FILE* fp = std::fopen("/tmp/q27_fin_ls_h.bin", "wb");
                                if (fp) { std::fwrite(fh.data(), 4, Q27_HID, fp); std::fclose(fp); }
                                fp = std::fopen("/tmp/q27_fin_ls_p.bin", "wb");
                                if (fp) { std::fwrite(fpp.data(), 4, Q27_HID, fp); std::fclose(fp); }
                            }
                            q27_add_f32(hrow, pendp32, Q27_HID, d.stream);
                            q27_red_b1(hrow, d.mix3, d.inv_slots + (p - sb), Q27_HID, 1, d.stream);
                            if (!q27_rmsnorm_hostss_perm(hrow, GL->final_norm, nullptr, d.norm, d.xq, d.xs, Q27_HID, 1,
                                                         GL->lm_head.in_scale, d.inv_slots + (p - sb), d.stream))
                                { std::fprintf(stderr, "Q27_LS_Q8: final norm declined; aborting\n"); std::exit(1); }
                            }
                            // EXL3 needs the bf16 norm, so the hostss call above now emits it
                            // (it was dropping it into nullptr) instead of only the quantized copy.
                            if (!exl3_head(d, GL, d.norm, d.pa, 1, d.stream))
                            q27_proj_nvfp4(&GL->lm_head, d.xq, d.xs, d.pa, d.stream);
                            q27_argmax_val(d.pa, d.dtok, d.dval, GL->lm_head.rows, d.stream);
                            CK(hipStreamSynchronize(d.stream));
                            unsigned li; float lv;
                            CK(hipMemcpy(&li, d.dtok, 4, hipMemcpyDeviceToHost));
                            CK(hipMemcpy(&lv, d.dval, 4, hipMemcpyDeviceToHost));
                            C.hval[g] = lv;
                            C.hidx[g] = (unsigned)li;
                        } else {
                        unsigned short* hidp = d.hidden_slots + (size_t)(p - sb) * Q27_HID;
                        const void* pendp = (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                        if (q27_env_flag("Q27_LS_DBG", false)) {
                            std::vector<float> h0 = q27_d2h_bf16(hidp, 8);
                            float p0[8];
                            CK(hipMemcpy(p0, pendp, 32, hipMemcpyDeviceToHost));
                            std::printf("DBG_LS_FIN hid=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f | pend=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                (double)h0[0],(double)h0[1],(double)h0[2],(double)h0[3],
                                (double)h0[4],(double)h0[5],(double)h0[6],(double)h0[7],
                                (double)p0[0],(double)p0[1],(double)p0[2],(double)p0[3],
                                (double)p0[4],(double)p0[5],(double)p0[6],(double)p0[7]);
                        }
                        if (g_coll_bf16) q27_add_inplace_bf16(hidp, (const unsigned short*)pendp, Q27_HID, d.stream);
                        else             q27_add_inplace(hidp, (const float*)pendp, Q27_HID, d.stream);
                        q27_rmsnorm(hidp, GL->final_norm, d.norm, Q27_HID, 1, d.stream);
                        q27_quant_perm(d.norm, d.xq, d.xs, Q27_HID, GL->lm_head.in_scale, d.stream);
                        if (!exl3_head(d, GL, d.norm, d.pa, 1, d.stream))
                        q27_proj_nvfp4(&GL->lm_head, d.xq, d.xs, d.pa, d.stream);
                        q27_argmax_val(d.pa, d.dtok, d.dval, GL->lm_head.rows, d.stream);
                        CK(hipStreamSynchronize(d.stream));
                        unsigned li; float lv;
                        CK(hipMemcpy(&li, d.dtok, 4, hipMemcpyDeviceToHost));
                        CK(hipMemcpy(&lv, d.dval, 4, hipMemcpyDeviceToHost));
                        C.hval[g] = lv;
                        C.hidx[g] = (unsigned)li;
                        }
                    }
                    C.b1.wait();
                    C.b2.wait();
                    const unsigned nx = C.hidx[g_ls ? last_card : ndev - 1];   // the head runs on the card owning the last layer
                    if (g == 0) {
                        if ((!sw_tok || g_serve) && p == sw_end - 1) {   // the first decode token
                            tok = nx; out.push_back(nx); q27_emit(nx);
                            if (q27_is_eos(nx) || (int)out.size() >= req_maxn) done = true;
                        }
                        if (t_prefill == 0) MS("T6_prefill_begin");
                        t_prefill += tp_now() - t0;
                    }
                    C.b1.wait();
                    C.b2.wait();
                }
                if (g == last_card) MS("T6b_finals_done");
                if (export_done) CK(hipStreamSynchronize(d.hs[1]));   // the overlapped decode-state export must land before decode
                // THE PROMPT CONDITIONING WRITES DRAFT-KV ROWS FOR [sw_beg, sw_end), SO THE DRAFT
                // CAPACITY HAS TO COVER THE WINDOW. mtp_prompt_pass calls q27_attn_kvprep_nr at
                // absolute position p0 with no cap check of its own, and d.mtp_cap is PER CARD: a
                // card that could not fund its paged draft grow keeps a shorter cache than its peers
                // (measured: card 3 held 65,536 while the others reached 131,072 and the next window
                // ended at 133,465). Without this gate that card writes ~15 MB past its draft cache
                // -- silently corrupting whatever the allocator put there and hanging all four ring
                // threads with no error at all. Skipping the conditioning costs the draft quality for
                // this window (decode falls back to plain target rounds) and nothing else.
                const bool mtp_pp_fits = q27_mtp_within_cap(D, ndev, sw_end);
                // K=0 (non-speculative): the draft layer is never read, so its prompt conditioning is
                // pure overhead and is skipped -- but only for K=0; the shipped path is unchanged.
                if (g_spec && g_spec_k > 0 && g_mtp_pp && g_ls_q8 && g_mtppp[g_sr ? g_sr_head : g].ready && sw_end - 1 > sw_beg && mtp_pp_fits) {   // Q27_MTP_PP: prompt-condition the draft layer
                    const double tpp0 = tp_now();
                    C.b1.wait(); C.b2.wait();               // the head card's exports landed before its stream sync above; publish to every card
                    if (!g_sr || g == g_sr_head) {   // Q27_LS_SR: one draft layer, on the head card
                    mtp_prompt_pass(d, g, sw_beg, sw_end - 1, sb, sw_tok, sw_lo, prompt, emb, Q27_PF_TSLOT);
                    CK(hipMemcpy(d.nr_hprev, g_mtp_hall + (size_t)(sw_end - 1 - sb) * Q27_HID, (size_t)Q27_HID * 2, hipMemcpyHostToDevice));
                    }
                    d.nr_have_h = 1; d.nr_mtp_base = 0; d.nr_ncatch = 0;   // the first decode step is a speculative round
                    C.b1.wait(); C.b2.wait();
                    g_mtppp[g].t_ms += tp_now() - tpp0; g_mtppp[g].npos += (sw_end - 1 - sw_beg);
                    if (g == 0) std::printf("Q27_MTP_PREFILL positions=%d ms=%.1f\n", sw_end - 1 - sw_beg, tp_now() - tpp0);
                }
                else if (g_spec && g_spec_k > 0 && g_mtp_pp && g_ls_q8 && g_mtppp[g_sr ? g_sr_head : g].ready && sw_end - 1 > sw_beg && g == 0) {
                    int worst = 0; for (int q = 0; q < ndev; ++q) if (D[q].mtp_cap < sw_end) { worst = q; break; }
                    std::fprintf(stderr, "Q27_MTP_PP skipped: card %d draft cap %d < window end %d -- "
                                         "no draft conditioning for this window (speculation stays off)\n",
                                 worst, D[worst].mtp_cap, sw_end);
                    std::fflush(stderr);
                }
                if (g == 0) {
                    std::fprintf(stderr,"Q27_PHASE req=%d phase=prefill edge=end monotonic_ms=%.6f\n",req,tp_now());
                    MS("T6b_sweep_end"); t_prefill = tp_now() - t_sw0;
                }
                if (g_ls_q8 && q8blk < LPP) {   // deferred ring-schedule teardown (after the timing stop; the peers finished their waits at the barrier above)
                    C.b1.wait(); C.b2.wait();
                    if (q8trace) std::fprintf(stderr, "Q27_Q8_TEARDOWN g=%d nev=%zu ring_late=%zu\n", g, g_ls_evdone[g].size(), ring_late.size());
                    for (size_t i = 0; i < g_ls_evdone[g].size(); ++i) {
                        if (q8trace && i < 2) std::fprintf(stderr, "Q27_Q8_DEST g=%d i=%zu e=%p\n", g, i, (void*)g_ls_evdone[g][i]);
                        CK(hipEventDestroy(g_ls_evdone[g][i]));
                    }
                    g_ls_evdone[g].clear();
                    for (size_t i = 0; i < ring_late.size(); ++i) CK(hipEventDestroy(ring_late[i]));
                    ring_late.clear();
                }
                if (g == 0) std::printf("Q27_LS_SWEEP positions=%d host_coll_ms=0.0 (layer-split: no intra-layer collectives)\n", nwin);
                C.t_comp[g] = 0; C.t_coll[g] = 0; C.n_coll[g] = 0; C.t_wall[g] = 0;
                g_pf_sweep = false;
                if (!sw_tok || g_serve) { pos0 = sw_end; pi = prompt.size(); }
            } else {
            // Q27_PF_PIPE: the previous chunk's collective #2 is reduced on the host while the GPU
            // runs the current chunk's MLP. Its consumer is the NEXT layer's input norm, so deferring
            // it by one chunk (and across the layer boundary) changes nothing but the overlap.
            int c2_p0 = -1, c2_n = 0;
            auto flush_c2 = [&]() {
                if (c2_n <= 0) return;
                { const double tw = tp_now(); CK(hipEventSynchronize(C.ev_c2[g])); C.c_d2h[g] += tp_now() - tw; }
                // ev_c2 was recorded after ev_b1 on the same stream, so the sync above covers the bracket
                // Ping-ponged parity: the bracket for the round being read is the one NOT holding this
                // round's just-recorded events. Reading the live pair here is what made the first attempt
                // report 20.54 ms for 4096 chunk-MLPs -- it was reading a bracket that had been overwritten.
                if (C.ev_b0[0][g]) { const int pv = (int)((C.tl_round[g] + 1) & 1); float ms = 0.f;
                  C.h_red2[g] += 0.0;   // (attribution happens at the call sites)
                  if (hipEventElapsedTime(&ms, C.ev_b0[pv][g], C.ev_b1[pv][g]) == hipSuccess) C.t_b13_ffn[g] += (double)ms;
                  if (hipEventElapsedTime(&ms, C.ev_pre[pv][g], C.ev_b0[pv][g]) == hipSuccess) { C.r_bubble[g] += (double)ms; C.r_bub_n[g]++; } }
                { const double t0 = tp_now(); tp_reduce_b2(C, g, c2_n); const double dt = tp_now() - t0;
                  C.t_coll[g] += dt; C.c_red[g] += dt; C.n_coll[g] += c2_n; }
                for (int c = 0; c < c2_n; ++c) d.inv_slots[c2_p0 - sb + c] = C.inv_hb2[g * TpColl::BC + c];
                CK(hipMemcpyAsync(d.pend_slots + (size_t)(c2_p0 - sb) * Q27_HID, (const void*)C.acc2,
                                  (size_t)c2_n * Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                c2_n = 0;
            };
            // Q27_PF_WIDE: the TILE-WIDE collective #2, its counters, and the FFN timing bracket.
            int c2w_p0 = -1, c2w_n = 0, wide_coll_w = 0;
            // ---- CHUNK-GRAIN COMMIT OF A TILE-WIDE RESULT (Q27_PF_WIDE_SLICE, default on) ----------
            // The tile output is STAGED once (one D2H), then REDUCED IN Q27_PF_MC-position slices, one
            // slice per subsequent chunk. Same total bytes and same total host arithmetic as the
            // 256-position block, but spread across 16 chunk boundaries instead of landing as one
            // contiguous host stall. Measured reason: with the FFN removed from both arms the two graphs
            // do identical GPU work (1422.6 vs 1404.8 ms) yet the wide one had 276 gaps >1 ms (1008 ms)
            // against rung 3's 25 (260 ms) -- roughly one ~3.7 ms stall per tile boundary, which is
            // exactly this block. Compute width and communication grain are independent:
            // the MLP stays M=256, the commit goes back to 16.
            int ws_p0 = -1, ws_n = 0, ws_off = 0;   // pending staged tile, in positions
            // CHUNK LAG before a staged tile is drained. The staging sits behind that tile whole
            // attention and its three FFN kernels, so a check one chunk after the close finds nothing
            // ready, the slices back up, and the layer boundary force-drains the backlog in one block --
            // which measured WORSE than simply blocking. Waiting Q27_PF_WIDE_LAG chunks costs nothing
            // (the result has a whole layer of slack before it is consumed).
            long long chunk_no = 0; long long ws_arm = 0;
            // The commit grain must MATCH the chunk width, not Q27_PF_MC. Slicing at 8 doubled the
            // number of tp_reduce_bx calls (12288 against rung 3 8192) because each call carries two
            // barrier round trips; at the chunk width the call count is identical to rung 3 and the
            // communication grain is the one rung 3 was tuned at.
            int ws_sw = 16;
            // blocking=false: take a slice only if the staged tile has ALREADY landed (hipEventQuery),
            // otherwise leave it for a later chunk and never stall the enqueue path. blocking=true is used
            // only at the layer boundary, where the slices must be in pend_slots before the next layer.
            //
            // This is the fix for the 1427 ms of host block measured at sync: the tile's staging sits
            // behind the whole tile's attention AND the three wide FFN kernels, so a blocking sync one
            // chunk after the tile close waits on almost the entire tile's GPU work while the host has
            // only ~one chunk of prologue queued ahead of it -- the GPU drains and the enqueue path stops.
            auto drain_slice = [&](bool blocking) {
                if (ws_n <= 0 || ws_off >= ws_n) return;
                if (!blocking && chunk_no - ws_arm < g_pf_wide_lag) return;   // not expected yet
                const int sw = (g_pf_wide_slice > 0) ? g_pf_wide_slice : ws_sw;
                const int n = (ws_n - ws_off < sw) ? (ws_n - ws_off) : sw;
                if (ws_off == 0) {
                                   if (!blocking) { if (hipEventQuery(C.ev_c2[g]) != hipSuccess) return; }
                                   const double tw = tp_now();
                                   if (blocking) CK(hipEventSynchronize(C.ev_c2[g]));
                                   const double td = tp_now() - tw; C.c_d2h[g] += td; C.h_sync[g] += td;
                                   // ev_c2 was recorded after ev_w1, so this sync also completes the FFN bracket
                                   float fms = 0.f;
                                   if (hipEventElapsedTime(&fms, C.ev_w0[g], C.ev_w1[g]) == hipSuccess) C.t_wide_ffn[g] += (double)fms;
                                   const int pv = (int)((C.tl_round[g] + 1) & 1);
                                   if (C.ev_pre[0][g] && hipEventElapsedTime(&fms, C.ev_pre[pv][g], C.ev_w0[g]) == hipSuccess) {
                                       C.r_bubble[g] += (double)fms; C.r_bub_n[g]++; } }
                float* hps[Q27_MAX_DEVICES];
                for (int k = 0; k < ndev; ++k) hps[k] = C.hp2w[k] + (size_t)ws_off * C.n;
                const double t0 = tp_now();
                tp_reduce_bx(C, g, n, hps, C.acc2w + (size_t)ws_off * C.n, C.ssb2w, C.inv_hb2w, Q27_PF_TSLOT);
                C.h_red2[g] += tp_now() - t0;
                for (int c = 0; c < n; ++c) d.inv_slots[ws_p0 + ws_off + c - sb] = C.inv_hb2w[g * Q27_PF_TSLOT + c];
                CK(hipMemcpyAsync(d.pend_slots + (size_t)(ws_p0 + ws_off - sb) * Q27_HID,
                                  (const void*)(C.acc2w + (size_t)ws_off * C.n),
                                  (size_t)n * Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                ws_off += n;
                if (ws_off >= ws_n) { ws_n = 0; ws_off = 0; }
            };
            long wide_tiles = 0, wide_gu_calls = 0, wide_dn_calls = 0, b13_calls = 0;
            auto flush_c2w = [&]() {
                if (c2w_n <= 0) return;
                { const double tw = tp_now(); CK(hipEventSynchronize(C.ev_c2[g])); C.c_d2h[g] += tp_now() - tw;
                  C.r_sync[g] += tp_now() - tw;
                  // GPU-observed bubble at the PREVIOUS tile boundary. ev_w0 still holds that round's
                  // record (this round's is recorded after this call) and ev_pre[pv] is that round's
                  // marker, so the pair is complete and not yet overwritten.
                  const int pv = (int)((C.tl_round[g] + 1) & 1);
                  if (C.ev_pre[0][g]) { float bms = 0.f;
                    if (hipEventElapsedTime(&bms, C.ev_pre[pv][g], C.ev_w0[g]) == hipSuccess) { C.r_bubble[g] += (double)bms; C.r_bub_n[g]++; }
                    else ++C.r_bub_fail[g]; } }
                { float ms = 0.f; if (hipEventElapsedTime(&ms, C.ev_w0[g], C.ev_w1[g]) == hipSuccess) C.t_wide_ffn[g] += (double)ms; }
                { const double t0 = tp_now();
                  tp_reduce_bx(C, g, c2w_n, C.hp2w, C.acc2w, C.ssb2w, C.inv_hb2w, Q27_PF_TSLOT);
                  const double dt = tp_now() - t0; C.t_coll[g] += dt; C.c_red[g] += dt; C.n_coll[g] += c2w_n; }
                for (int c = 0; c < c2w_n; ++c) d.inv_slots[c2w_p0 - sb + c] = C.inv_hb2w[g * Q27_PF_TSLOT + c];
                { const double t0 = tp_now();
                  CK(hipMemcpyAsync(d.pend_slots + (size_t)(c2w_p0 - sb) * Q27_HID, (const void*)C.acc2w,
                                    (size_t)c2w_n * Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                  C.r_h2d[g] += tp_now() - t0; }
                c2w_n = 0;
            };
            for (int L = 0; L < nlayer; ++L) {
                const q27_layer_t* lay = q27_layer_tp(m, L, d.id);
                if (!lay) { std::fprintf(stderr, "Q27_TP_NOT_RESIDENT layer %d card %d\n", L, g); std::exit(1); }
                // Q27_TPSR: this layer's MLP shard -> per-row int8 (once per layer per sweep), stream-ordered before its first chunk
                if (g_tpsr && !tpsr_mlp_view(d, lay, d.stream)) { std::fprintf(stderr, "Q27_TPSR: MLP shard view failed (L%d card %d)\n", L, g); std::exit(1); }
                // The previous layer's last chunk collective #2 must land in pend_slots before ANY
                // prologue of this layer reads it: a prompt that fits in one tile has that chunk in
                // the first tile. One chunk of overlap lost per layer, ordering kept.
                flush_c2();
                flush_c2w();
                // The tile slices drain one per chunk, so a layer's LAST tile still has up to
                // Q27_PF_TSLOT/Q27_PF_MC slices outstanding when its chunk loop ends. They must land
                // before ANY layer reads pend_slots for those positions -- the same contract flush_c2
                // honours for the chunk path. Draining only after the whole layer loop (the first
                // version) left layers 1..63 reading stale pend_slots and broke the tokens.
                while (ws_n > 0) drain_slice(true);   // the boundary where they MUST be in pend_slots
                const int isf = Q27_IS_FULL(L);
                if (g_tpsr_proj_rb && !tpsr_proj_view(d, lay, isf, d.stream)) { std::fprintf(stderr, "Q27_TPSR_PROJ_RB: projection mirror view failed (L%d card %d)\n", L, g); std::exit(1); }
                const q27_layer_t* nxt = (L + 1 < nlayer) ? q27_layer_tp(m, L + 1, d.id) : nullptr;
                if (pf_batch && g_rn_epi) {
                    // Q27_PF_WIDE: WHERE THIS CHUNK'S MLP INPUT LIVES. With the tile-wide FFN the
                    // post-attention norm writes STRAIGHT INTO the tile buffer at this chunk's offset, so
                    // there is no per-chunk staging copy. The b13 fallback reads the SAME pointer, so the two
                    // paths read one location rather than two -- and a declined wide launch degrades to b13
                    // over live data instead of over a stale chunk buffer. Declared out here because the
                    // lambda below is defined before the chunk loop that assigns them.
                    int wtb0 = sw_beg;
                    bool wide_tile = false;
                    bool wide_local = false;
                    signed char* pq1 = nullptr;   // this chunk's MLP input, tile-relative when wide
                    float*       ps1 = nullptr;   // its per-16 scales
                    // the shared MLP + collective#2 tail for one chunk
                    auto post_mlp = [&](const q27_layer_t* layx, int p0, int Cch_all, int ci) {
                      // The MLP kernels are Q27_PF_MC (8) wide by register shape; a wider chunk runs them per 8-position half.
                      // Q27_PF_HSTREAMS: the halves are independent (disjoint positions, disjoint scratch), so halves 1.. run on
                      // side streams forked from d.stream after the tail norm and joined before the collective.
                      if (g_pf_wide_noffn) return;   // DIAGNOSTIC: cost of the CALL vs the BODY
                      const int nhalf = (Cch_all + Q27_PF_MC - 1) / Q27_PF_MC;
                      const bool hstr = g_pf_hstreams && nhalf > 1;
                      if (hstr) { CK(hipEventRecord(d.ev_fork, d.stream)); for (int k = 1; k < nhalf && k <= 4; ++k) CK(hipStreamWaitEvent(d.hs[k - 1], d.ev_fork, 0)); }
                      // ---- Q27_PF_WIDE: the TILE-WIDE DEFERRED MLP ----------------------------------
                      // This chunk's MLP input is RETAINED in tile-wide storage instead of being consumed
                      // by ceil(Cch/8) 8-wide b13 launches. At the tile boundary ONE gate+up and ONE down
                      // run at M = the tile width, followed by ONE collective #2 over the whole tile.
                      //
                      // Deferral is legal, and that is a claim about the execution graph rather than a
                      // hope: the only consumer of collective #2's output is the NEXT layer's input norm.
                      // pend_slots is read either as the layer input norm's source
                      // (q27_rmsnorm_hostss_fp8_b(d.pend_slots + ...)) or as the pendp argument to
                      // run_layer_tp, and in both cases the data it holds is layer L-1's output. Nothing
                      // inside layer L reads pend_slots for layer L's own positions -- which is exactly
                      // why flush_c2 already carries one chunk's collective across the layer boundary.
                      // Holding chunks to the tile close therefore only moves work later within one
                      // layer, and layer L+1 cannot start before layer L ends regardless.
                      //
                      // The width is a precondition, not a preference: window 80 measured the wide
                      // gate+up LOSING to b13 below M=128 (0.48-0.85x at M=64; its tile is 64/128 tokens
                      // wide). A tile too short to pay falls through to the b13 loop below -- per TILE,
                      // so the deferred xq1_t is never left half-consumed.
                      //
                      // If either wide launch declines, the b13 loop below still runs over the chunk's
                      // live buffers, so a refusal degrades to rung-3 geometry rather than to silence.
                      // NO STAGING COPY. The post-norm above already wrote this chunk's MLP input into
                      // d.xq1_t at this chunk's offset (pq1/ps1), so the tile simply accumulates as the
                      // chunks complete. The first version copied xq1_slots -> xq1_t here, once per chunk;
                      // a rocprof trace of the 1K prefill put that at 1281 ms of GPU time over 33792
                      // k_copy_f4 launches -- more than the 2022 ms the wide kernels saved -- and it was
                      // pure overhead on the critical path. Removing the copy removes the whole cost.
                      if (wide_local) {
                        // ONE wide GEMM per matrix for THIS chunk, on the chunk buffers. Q27_PF_CH must be
                        // >= Q27_PF_WIDE_MIN and the runtime width set to match (Q27_PF_CHW).
                        if (g_pf_p2p && g_pf_p2p_mix) { const int mq = ci % 3;
                            for (int k = 0; k < ndev; ++k) if (k != g)
                                CK(hipStreamWaitEvent(d.stream, C.ev_mix_rd[mq][k], 0)); }   // mixer_slots free to overwrite
                        if (!g_pf_wide_noffn) {
                          CK(hipEventRecord(C.ev_w0[g], d.stream));
                          if (!q27_wide_gu(&layx->gate, &layx->up, d.xq1_slots, d.xs1_slots,
                                           d.pa_mlp, d.pb_mlp, Cch_all, g_pf_wide_kg, d.stream))
                            std::fprintf(stderr, "Q27_PF_WIDE: gate+up declined (L%d Cch=%d)\n", L, Cch_all);
                          ++wide_gu_calls;
                          if (!q27_swiglu_quant_b(d.pa_mlp, d.pb_mlp, d.xq2_slots, d.xs2_slots,
                                                  mlp_len(d.id), layx->down.in_scale, Cch_all, d.stream))
                            std::fprintf(stderr, "Q27_PF_WIDE: swiglu declined (L%d Cch=%d)\n", L, Cch_all);
                          const unsigned short* lradd = g_hostss_neg ? nullptr
                              : (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID);
                          if (!q27_wide_down(&layx->down, d.xq2_slots, d.xs2_slots, d.mixer_slots, Cch_all,
                                             lradd, 0.25f, g_pf_wide_kd, d.stream))
                            std::fprintf(stderr, "Q27_PF_WIDE: down declined (L%d Cch=%d)\n", L, Cch_all);
                          ++wide_dn_calls; ++wide_tiles;
                          CK(hipEventRecord(C.ev_w1[g], d.stream));
                          if (g_pf_p2p && g_pf_p2p_mix) {
                              g_p2p_mix_calls.fetch_add(1, std::memory_order_relaxed);
                              // GPU-resident mixer collective #2: assemble the 4 partial MLP outputs
                              // right after the down kernel. The old path D2H'd here and reduced on the
                              // host one chunk later; pend_slots/inv_slots feed the next layer's input
                              // norm, exactly as the deferred flush did.
                              const int mq = ci % 3;
                              CK(hipEventRecord(C.ev_mix[mq][g], d.stream));
                              for (int k = 0; k < ndev; ++k) if (k != g)
                                  CK(hipStreamWaitEvent(d.stream, C.ev_mix[mq][k], 0));
                              q27_red_b(C.mixp[0], C.mixp[1], C.mixp[2], C.mixp[3],
                                        d.pend_slots + (size_t)(p0 - sb) * Q27_HID,
                                        d.inv_slots + (p0 - sb), Q27_HID, Cch_all, d.stream);
                              CK(hipEventRecord(C.ev_mix_rd[mq][g], d.stream));
                          }
                          if (g == 0 && wide_tiles == 1)
                            std::printf("MLP_M=%d gateup_calls=%d down_calls=%d collective2_width=%d\n",
                                        Cch_all, 1, 1, Cch_all);
                        }
                      } else if (wide_tile) {
                        const int wend = p0 + Cch_all;
                        if (wend >= wtb0 + Q27_PF_TSLOT || wend >= sw_end) {
                          const int Ct = wend - wtb0;
                          // Q27_MLP_TIMELINE (instrumentation only): host clock around the tile close, and
                          // ev_pre marks the point in the STREAM where the last queued work ends. The gap
                          // from ev_pre to the gu launch is the GPU-observed bubble at this boundary.
                          const bool TL = (C.ev_pre[0][g] != nullptr);
                          const int tlpar = (int)(C.tl_round[g] & 1);
                          const double th0 = TL ? tp_now() : 0.0;
                          const double sb_s = C.r_sync[g], sb_1 = C.r_b1[g], sb_l = C.r_loop[g], sb_2 = C.r_b2[g];
                          if (TL) CK(hipEventRecord(C.ev_pre[tlpar][g], d.stream));
                          flush_c2();                      // a single-chunk result still pending
                          const double th1 = TL ? tp_now() : 0.0;
                          if (TL) C.h_flush[g] += th1 - th0;
                          CK(hipEventRecord(C.ev_w0[g], d.stream));
                          if (!g_pf_wide_noffn) {
                          if (!q27_wide_gu(&layx->gate, &layx->up, d.xq1_t, d.xs1_t, d.pa_t, d.pb_t,
                                           Ct, g_pf_wide_kg, d.stream))
                            std::fprintf(stderr, "Q27_PF_WIDE: gate+up declined (L%d Ct=%d kg=%d)\n", L, Ct, g_pf_wide_kg);
                          ++wide_gu_calls;
                          if (!q27_swiglu_quant_b(d.pa_t, d.pb_t, d.xq2_t, d.xs2_t, mlp_len(d.id),
                                                  layx->down.in_scale, Ct, d.stream))
                            std::fprintf(stderr, "Q27_PF_WIDE: swiglu declined (L%d Ct=%d)\n", L, Ct);
                          const unsigned short* wradd = g_hostss_neg ? nullptr
                              : (d.hidden_slots + (size_t)(wtb0 - sb) * Q27_HID);
                          if (!q27_wide_down(&layx->down, d.xq2_t, d.xs2_t, d.mix_t, Ct,
                                             wradd, 0.25f, g_pf_wide_kd, d.stream))
                            std::fprintf(stderr, "Q27_PF_WIDE: down declined (L%d Ct=%d kd=%d)\n", L, Ct, g_pf_wide_kd);
                          }   // end Q27_PF_WIDE_NOFFN diagnostic
                          ++wide_dn_calls; ++wide_tiles;
                          CK(hipEventRecord(C.ev_w1[g], d.stream));
                          const double th2 = TL ? tp_now() : 0.0;
                          // THE PREVIOUS TILE'S REDUCE RUNS HERE, NOT BEFORE THE LAUNCHES. It used to sit
                          // above the gu launch, which put its hipEventSynchronize on the critical path:
                          // the host blocked waiting for tile T-1's staging copy and only then launched
                          // tile T's FFN, leaving the GPU idle. Behind the launches the same sync waits on
                          // a copy that has had a whole FFN's worth of time to land. Ordering is unchanged
                          // -- this still writes pend_slots for T-1, and its consumer is the NEXT layer.
                          flush_c2w();
                          // unconditionally in-stream here: hp2w_d exists whenever Q27_PF_WIDE is on, and a
                          // 5.24 MB D2H per tile is 8x coarser than the chunk path's staging
                          if (!g_pf_wide_sdma && C.hp2w_d[g]) q27_copy_f4(C.hp2w_d[g], d.mix_t, Ct * Q27_HID, d.stream);
                          else CK(hipMemcpyAsync(C.hp2w[g], d.mix_t, (size_t)Ct * Q27_HID * 4,
                                                 hipMemcpyDeviceToHost, d.stream));
                          CK(hipEventRecord(C.ev_c2[g], d.stream));
                          const double th3 = TL ? tp_now() : 0.0;
                          if (TL) { C.h_enq[g] += th2 - th1; C.h_stage[g] += th3 - th2;
                            if (L == 0 && C.tl_n[g] < 64) { TpColl::TLRec& R = C.tl[g][C.tl_n[g]++];
                              R.layer = L; R.tile = (int)wide_tiles; R.card = g; R.ffn_end = th2; R.stage_end = th3; R.beg = th0;
                              R.sync_end = th0 + (C.r_sync[g] - sb_s);
                              R.red_end  = R.sync_end + (C.r_loop[g] - sb_l);
                              R.bubble   = (C.r_b1[g] - sb_1) + (C.r_b2[g] - sb_2); }
                            C.tl_round[g]++; }
                          // Stage the whole tile once; the reduce is spread over the next chunks.
                          ws_p0 = wtb0; ws_n = Ct; ws_off = 0; ws_arm = chunk_no;
                          wide_coll_w = Ct;
                          if (g == 0 && wide_tiles == 1)
                            std::printf("MLP_M=%d gateup_calls=%d down_calls=%d collective2_width=%d\n",
                                        Ct, 1, 1, Ct);
                        }
                      } else if (layx->ex_gate.live() && layx->ex_up.live() && layx->ex_down.live()) {
                        // ---- Q27_EXL3: THE BATCHED PREFILL MLP ON THE RESIDENT TRELLIS ----
                        // This is the FOURTH MLP path in the engine (M=1 decode, the nr verify
                        // rows, the layer-split wide sweep, and this batched chunk path), and the
                        // one the codebase already names at q27_main.cpp's upload comment: a
                        // dropped NVFP4 gate/up leaves q27_proj_nvfp4_gu_b13 reading w=NULL on the
                        // first chunk of layer 0 -- the "batched path's small-address fault".
                        // Structure mirrors the TPSR arm below exactly: expand the bf16 residual,
                        // norm it, gate/up, swiglu, down, fold the 0.25 residual share. Only the
                        // three GEMMs and the activation currency differ -- EXL3 rotates its own
                        // input, so nothing here is quantized.
                        const Q8Scr& Q = d.q8s[0];
                        const int Cch = Cch_all;
                        const int IL = mlp_len(d.id);
                        const unsigned short* hrow = d.hidden_slots + (size_t)(p0 - sb) * Q27_HID;
                        auto xdie = [&](const char* w) { std::fprintf(stderr, "Q27_EXL3 prefill MLP: %s declined (L%d card %d M=%d)\n", w, L, g, Cch); std::fflush(stderr); std::exit(1); };
                        if (!q27_bf16_to_f32(hrow, Q.hid32c, (size_t)Cch * Q27_HID, d.stream)) xdie("residual expand");
                        void* xw = exl3_ws(d, Q27_EXL3_WMAX, Q27_EXL3_WMAX, Cch);
                        if (!xw) xdie("workspace");
                        unsigned short* nb  = q27_exl3_ws_norm(xw, Cch);
                        unsigned short* act = q27_exl3_ws_act(xw, layx->ex_down.in, Cch, layx->ex_down.out);
                        if (!q27_exl3_rmsnorm_bf16(Q.hid32c, layx->post_norm, nb, Q27_HID, Cch, Q27_HID, Q27_HID, d.stream)) xdie("post norm");
                        if (!exl3_f32(d, layx->ex_gate, nb, d.pa_mlp, Cch, d.stream, Q27_HID, IL)) xdie("gate");
                        if (!exl3_f32(d, layx->ex_up,   nb, d.pb_mlp, Cch, d.stream, Q27_HID, IL)) xdie("up");
                        if (!q27_exl3_swiglu_bf16(d.pa_mlp, d.pb_mlp, act, IL, Cch, IL, IL, d.stream)) xdie("swiglu");
                        if (!exl3_f32(d, layx->ex_down, act, d.mixer_slots, Cch, d.stream, IL, Q27_HID)) xdie("down");
                        if (g_rn_hostss && !g_hostss_neg)
                            q27_exl3_add_res(d.mixer_slots, hrow, 0.25f, Q27_HID, Cch, Q27_HID, Q27_HID, d.stream);
                        ++b13_calls;
                      } else if (g_tpsr && d.tpsr_mlayer == L && d.tpsr_g8r.w && d.tpsr_u8r.w && d.tpsr_d8r.w) {
                        // ---- Q27_TPSR: THE MLP ON rocBLAS INT8 FROM THE TRANSIENT PER-ROW SHARD ----------
                        // The same three GEMMs + two epilogues the layer-split sweep runs (ls_q8_layer), on this
                        // card's 1/ndev rows of gate/up and its K-slice of down, expanded once per layer per sweep
                        // from the resident block-scaled FP8 (tpsr_mlp_view at the layer head). The per-token
                        // natural int8 of the post-norm is re-derived from the bf16 residual the post-norm above
                        // wrote into hidden_slots (VRAM), so the pinned collective result is read once, not twice.
                        // The down output is this card's K-partial with the 0.25 residual share folded in, exactly
                        // what the NVFP4 kernels handed collective #2 -- the rest of post_mlp runs unchanged.
                        const Q8Scr& Q = d.q8s[0];
                        const int Cch = Cch_all;
                        const q27_i8r_t* G8 = &d.tpsr_g8r; const q27_i8r_t* U8 = &d.tpsr_u8r; const q27_i8r_t* D8 = &d.tpsr_d8r;
                        const unsigned short* hrow = d.hidden_slots + (size_t)(p0 - sb) * Q27_HID;
                        auto tdie = [&](const char* w) { std::fprintf(stderr, "Q27_TPSR prefill MLP: %s declined (L%d card %d M=%d)\n", w, L, g, Cch); std::fflush(stderr); std::exit(1); };
                        if (!q27_bf16_to_f32(hrow, Q.hid32c, (size_t)Cch * Q27_HID, d.stream)) tdie("residual expand");
                        const float gate_scale = layx->f8_gate.w ? layx->f8_gate.in_scale : layx->gate.in_scale;
                        const float down_scale = layx->f8_down.w ? layx->f8_down.in_scale : layx->down.in_scale;
                        if (!q27_red_rn_tok_b1(Q.hid32c, layx->post_norm, nullptr, Q.xqt, Q.xst, Q27_HID, 1, gate_scale, Cch, 0, d.stream)) tdie("post norm tok");
                        int* ag = Q.acc; int* au = Q.acc + (size_t)Cch * G8->rows;
                        if (!q27_rb_gemm_i8g(d.id, G8->w, G8->rows, G8->K, Q.xqt, Cch, ag, G8->gs, d.stream)) tdie("gate gemm");
                        if (!q27_rb_gemm_i8g(d.id, U8->w, U8->rows, U8->K, Q.xqt, Cch, au, U8->gs, d.stream)) tdie("up gemm");
                        if (!q27_epi_swiglu_tok_small(ag, au, G8->rows, G8->s, U8->s, 1, Q.xst, G8->alpha, U8->alpha,
                                                      1.0f / down_scale, Q.xq2, Q.xst2, Cch, d.stream)) tdie("swiglu epilogue");
                        if (!q27_rb_gemm_i8g(d.id, D8->w, D8->rows, D8->K, Q.xq2, Cch, Q.acc, D8->gs, d.stream)) tdie("down gemm");   // acc reused: the swiglu epilogue consumed ag/au (stream order)
                        if (!q27_epi_f32r_bf(Q.acc, D8->rows, D8->s, 1, Q.xst2, D8->alpha, d.mixer_slots,
                                             (g_rn_hostss && !g_hostss_neg) ? hrow : nullptr, 0.25f, Cch, d.stream)) tdie("down epilogue");
                        ++b13_calls;
                      } else if (!g_pf_wide_noffn) {
                      const bool TLb = (C.ev_pre[0][g] != nullptr);
                      const int tlparb = (int)(C.tl_round[g] & 1);
                      if (TLb) CK(hipEventRecord(C.ev_pre[tlparb][g], d.stream));
                      if (C.ev_b0[0][g]) CK(hipEventRecord(C.ev_b0[tlparb][g], d.stream));
                      ++b13_calls;
                      for (int h0 = 0; h0 < Cch_all; h0 += Q27_PF_MC) {
                        const int Cch = (Cch_all - h0 < Q27_PF_MC) ? (Cch_all - h0) : Q27_PF_MC;
                        const int hidx = h0 / Q27_PF_MC;
                        hipStream_t hs_ = (hstr && hidx >= 1 && hidx <= 4) ? d.hs[hidx - 1] : d.stream;
                        signed char* xq1h = pq1 + (size_t)h0 * Q27_HID;   float* xs1h = ps1 + (size_t)h0 * (Q27_HID / 16);
                        signed char* xq2h = d.xq2_slots + (size_t)h0 * mlp_len(d.id);   float* xs2h = d.xs2_slots + (size_t)h0 * (mlp_len(d.id) / 16);
                        float* pah = d.pa_mlp + (size_t)h0 * mlp_len(d.id);   float* pbh = d.pb_mlp + (size_t)h0 * mlp_len(d.id);
                        float* mixh = d.mixer_slots + (size_t)h0 * Q27_HID;
                        if (!(g_pf_mlp14 && q27_proj_nvfp4_gu_b14(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp14, hs_)) &&
                            !(g_pf_mlp13 && q27_proj_nvfp4_gu_b13(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp13, hs_)) &&
                            !(g_pf_mlp12 && q27_proj_nvfp4_gu_b12(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp12, hs_)) &&
                            !(g_pf_mlp11 && q27_proj_nvfp4_gu_b11(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp11, hs_)) &&
                            !(g_pf_mlp10 && q27_proj_nvfp4_gu_b10(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, 1, hs_)) &&
                            !(g_pf_mlp9 && q27_proj_nvfp4_gu_b9(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, 1, hs_)) &&
                            !(g_pf_mlp8 && q27_proj_nvfp4_gu_b8(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, 1, hs_)) &&
                            !(g_pf_mlp7 && q27_proj_nvfp4_gu_b7(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp7_rpw, hs_)) &&
                            !(g_pf_mlp6 && q27_proj_nvfp4_gu_b6(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp6, hs_)) &&
                            !(g_pf_mlp5 && q27_proj_nvfp4_gu_b5(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp5, hs_)) &&
                            !(g_pf_mlp4 && q27_proj_nvfp4_gu_b4(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp4, hs_)) &&
                            !(g_pf_mlp3 && q27_proj_nvfp4_gu_b3(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp3, hs_)) &&
                            !(g_pf_mlp2 && q27_proj_nvfp4_gu_b2(&layx->gate, &layx->up, xq1h, xs1h,
                                                                 pah, pbh, Cch, g_pf_mlp_r, hs_)))
                        q27_proj_nvfp4_gu_b(&layx->gate, &layx->up, xq1h, xs1h,
                                            pah, pbh, Cch, hs_);
                        if (!(g_pf_mlp2 && q27_swiglu_quant_b(pah, pbh, xq2h, xs2h,
                                                              mlp_len(d.id), layx->down.in_scale, Cch, hs_)))
                        for (int c = 0; c < Cch; ++c) {
                            q27_swiglu_quant(pah + (size_t)c * mlp_len(d.id),
                                             pbh + (size_t)c * mlp_len(d.id),
                                             xq2h + (size_t)c * mlp_len(d.id),
                                             xs2h + (size_t)c * (mlp_len(d.id) / 16),
                                             mlp_len(d.id), layx->down.in_scale, hs_);
                        }
                        q27_nvfp4_set_res((g_rn_hostss && !g_hostss_neg)
                                              ? (d.hidden_slots + (size_t)(p0 - sb + h0) * Q27_HID) : nullptr, 0.25f);
                        if (!(g_pf_mlp14d && q27_proj_nvfp4_bt14(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp14d, hs_)) &&
                            !(g_pf_mlp13d && q27_proj_nvfp4_bt13(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp13d, hs_)) &&
                            !(g_pf_mlp12d && q27_proj_nvfp4_bt12(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp12d, hs_)) &&
                            !(g_pf_mlp11d && q27_proj_nvfp4_bt11(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp11d, hs_)) &&
                            !(g_pf_mlp10d && q27_proj_nvfp4_bt10(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp10d, hs_)) &&
                            !(g_pf_mlp9d && q27_proj_nvfp4_bt9(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp9d, hs_)) &&
                            !(g_pf_mlp7d && q27_proj_nvfp4_bt7(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp7_rpw, hs_)) &&
                            !(g_pf_mlp6d && q27_proj_nvfp4_bt6(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp6d, hs_)) &&
                            !(g_pf_mlp5 && q27_proj_nvfp4_bt5(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp5, hs_)) &&
                            !(g_pf_mlp4 && q27_proj_nvfp4_bt4(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp4, hs_)) &&
                            !(g_pf_mlp3 && q27_proj_nvfp4_bt3(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp3, hs_)) &&
                            !(g_pf_mlp2 && q27_proj_nvfp4_bt2(&layx->down, xq2h, xs2h, mixh, Cch, g_pf_mlp_r, hs_)))
                        q27_proj_nvfp4_bt(&layx->down, xq2h, xs2h, mixh, Cch, hs_);
                        q27_nvfp4_set_res(nullptr, 0.f);
                      }
                      if (C.ev_b0[0][g]) CK(hipEventRecord(C.ev_b1[tlparb][g], d.stream));
                      if (TLb) C.tl_round[g]++;
                      }
                      if (hstr) { for (int k = 1; k < nhalf && k <= 4; ++k) { CK(hipEventRecord(d.ev_join[k - 1], d.hs[k - 1])); CK(hipStreamWaitEvent(d.stream, d.ev_join[k - 1], 0)); } }
                      const int Cch = Cch_all;
                        const bool coll_b = q27_env_flag("Q27_COLL_BATCH", true) && g_rn_hostss
                                            && !g_coll_bf16 && !g_coll_zcout && Cch > 1;
                        // Q27_PF_WIDE: SKIP THE CHUNK COLLECTIVE ENTIRELY when this chunk's MLP was
                        // deferred into a wide tile. Its producer never ran, so d.mixer_slots still holds
                        // the PREVIOUS chunk's output -- staging and reducing it published stale data into
                        // pend_slots and cost a full extra collective per chunk. v100's tokens survived
                        // only because flush_c2w() runs right after flush_c2() at the tile close and
                        // overwrote the same ranges. That was luck, not ordering by design.
                        // THE COMMIT SLICE GOES HERE, NOT AT THE TOP OF post_mlp. Rung 3 does its equivalent
                        // 16-position reduce AFTER ~10 launches have been enqueued for the chunk, so the GPU
                        // has a deep queue while the host blocks. Running the same reduce at the TOP -- before
                        // this chunk's work is enqueued -- measured 1377 ms of GPU idle between the
                        // post-attention norm and the next chunk's o_proj, which was the whole regression.
                        // Drain AS MANY slices as have landed, not one. One-per-chunk exactly matches the
                        // production rate, so any chunk where the staging has not landed leaves a slice
                        // outstanding and the backlog is force-drained in a block at the layer boundary --
                        // which is where the stall reappeared. Draining all available keeps the backlog at
                        // zero without ever blocking the enqueue path.
                        { long guard = 0; const int before = ws_off;
                          while (ws_n > 0 && ws_off == before && guard++ < (Q27_PF_TSLOT / Q27_PF_MC) + 2)
                              drain_slice(false);
                          for (int g2 = 0; ws_n > 0 && ws_off < ws_n && g2 < 64; ++g2) {
                              const int o0 = ws_off; drain_slice(false); if (ws_off == o0) break; } }
                        if (coll_b && g_pf_pipe && !wide_tile && !(g_pf_p2p && g_pf_p2p_mix)) {
                            flush_c2();                              // previous chunk, overlapped with this MLP
                            if (g_pf_kcopy && C.hp2_d[g]) q27_copy_f4(C.hp2_d[g], d.mixer_slots, Cch * Q27_HID, d.stream);   // in-stream: no SDMA round trip
                            else CK(hipMemcpyAsync(C.hp2[g], d.mixer_slots, (size_t)Cch * Q27_HID * 4, hipMemcpyDeviceToHost, d.stream));
                            CK(hipEventRecord(C.ev_c2[g], d.stream));
                            c2_p0 = p0; c2_n = Cch;
                        } else
                        if (coll_b && !(g_pf_p2p && g_pf_p2p_mix)) {
                            // ONE rendezvous for the whole chunk: same per-position arithmetic as
                            // the per-position loop (each card sums its own 1/ndev slice), so the
                            // reduced vectors are bit-identical; only the fixed cost is shared.
                            { const double t0 = tp_now();
                              tp_allreduce_b(C, g, d.mixer_slots, Cch, d.stream);
                              C.t_coll[g] += tp_now() - t0; C.n_coll[g] += Cch; }
                            for (int c = 0; c < Cch; ++c)
                                d.inv_slots[p0 - sb + c] = C.inv_hb[g * TpColl::BC + c];
                            // pend_slots is [pos][Q27_HID] contiguous, so the chunk lands in one DMA.
                            // The source is ALWAYS C.acc: it holds the assembled vector on both paths
                            // (with zc off the copy-back to src only reads it). Using d.mixer_slots
                            // here fed device memory to an H2D copy and produced garbage.
                            CK(hipMemcpyAsync(d.pend_slots + (size_t)(p0 - sb) * Q27_HID, (const void*)C.acc,
                                              (size_t)Cch * Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                        } else if (!wide_tile && !(g_pf_p2p && g_pf_p2p_mix))
                        for (int c = 0; c < Cch; ++c) {
                            const int p = p0 + c;
                            { const double t0 = tp_now();
                              if (g_coll_bf16) tp_allreduce_bf16(C, g, (unsigned short*)(d.mixer_slots + (size_t)c * Q27_HID), d.stream);
                              else             tp_allreduce(C, g, d.mixer_slots + (size_t)c * Q27_HID, d.stream);
                              C.t_coll[g] += tp_now() - t0; C.n_coll[g]++; }
                            d.inv_slots[p - sb] = C.inv_h[g];
                            if (g_coll_zc)
                                CK(hipMemcpyAsync(d.pend_slots + (size_t)(p - sb) * Q27_HID, g_coll_acc,
                                                  (size_t)Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                            else {
                                CK(hipMemcpyAsync(d.pend_stage, d.mixer_slots + (size_t)c * Q27_HID,
                                                  (size_t)Q27_HID * 4, hipMemcpyDeviceToHost, d.stream));
                                CK(hipMemcpyAsync(d.pend_slots + (size_t)(p - sb) * Q27_HID, d.pend_stage,
                                                  (size_t)Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                            }
                        }
                    };
                    const int QLb = Q27_QROWS / ndev, KVLb = Q27_KVROWS / ndev;
                    // Runtime chunk width: Q27_PF_CHW (<= the compile-time Q27_PF_CH that sizes the buffers); 0 = auto
                    // (8 below Q27_PF_CHW_AUTO prompt positions, else the compile-time width). Fewer, wider chunks halve
                    // the per-chunk collectives; the smallest prompts pay their fixed per-chunk work instead.
                    const int npos_w = sw_end - sw_beg;
                    // Q27_PF_WIDE widens the chunk to Q27_PF_CH (build with -DQ27_PF_CH>=256): the wide FFN needs
                    // >=256 positions per tile to beat b13, and a wider chunk is also what collapses the per-chunk
                    // collectives. With wide OFF the long-prompt width stays the rung-3 value (32) even in a wide
                    // build, so a Q27_PF_WIDE=0 run of this binary must be bit-identical to rung 3 -- that is the
                    // control that says the bigger buffers changed nothing.
                    // The chunk ladder is LEFT EXACTLY AT RUNG 3 even when Q27_PF_WIDE is on. The wide MLP
                    // aggregates chunks into a Q27_PF_TSLOT tile, so the chunk width is not what gives it
                    // its M -- and moving two things at once would make the wall-clock delta uninterpretable.
                    // Keeping it also means the attention, q/k/v, o_proj and collective #1 geometry under
                    // test here is bit-for-bit the geometry every rung was measured on.
                    int chw = (g_pf_chw > 0) ? g_pf_chw
                            : (npos_w < g_pf_chw_auto) ? 8 : (npos_w < g_pf_chw_auto2) ? 16 : Q27_PF_CH;   // measured: 8 wins <256, 16 to ~2K, 32 at 8K
                    if (chw > Q27_PF_CH) chw = Q27_PF_CH; if (chw < 1) chw = 1;
                    ws_sw = chw;
                    int pipe1_done = -1;                     // Q27_PF_PIPE1: furthest chunk whose prologue is enqueued (per layer)
                    const int look = (g_pf_look < 1) ? 1 : (g_pf_look > TpColl::NB - 1) ? (TpColl::NB - 1) : g_pf_look;
                    auto ring = [&](int q) { return ((q - sw_beg) / chw) % TpColl::NB; };   // staging slot of chunk q
                    const double tloop0 = tp_now();
                    for (int p0 = sw_beg; p0 < sw_end; p0 += chw) {
                        const int Cch = (sw_end - p0 < chw) ? (sw_end - p0) : chw;
                        wtb0 = sw_beg + ((p0 - sw_beg) / Q27_PF_TSLOT) * Q27_PF_TSLOT;
                        // LOCAL: the chunk itself already carries >= Q27_PF_WIDE_MIN positions, so the
                        // wide FFN can use the CHUNK buffers directly -- no tile, no deferral, no commit
                        // machinery, and the existing chunk collective #2 runs unchanged on d.mixer_slots.
                        // This is the design the CH>32 overflow bug had been blocking all along.
                        const bool wide_ok = g_pf_wide && g_rn_hostss && !g_coll_bf16 && !g_coll_zcout && g_pf_pipe;
                        wide_local = wide_ok && (Cch >= g_pf_wide_min);
                        wide_tile = wide_ok && !wide_local && (sw_end - wtb0) >= g_pf_wide_min;
                        pq1 = wide_tile ? (d.xq1_t + (size_t)(p0 - wtb0) * Q27_HID) : d.xq1_slots;
                        ps1 = wide_tile ? (d.xs1_t + (size_t)(p0 - wtb0) * (Q27_HID / 16)) : d.xs1_slots;
                        // Q27_PF_FULL_SERIAL: full layers take the per-position serial prologue
                        // (round-6 behaviour) while the batched MLP stays; isolates the batched q/k/v.
                        const bool full_serial = q27_env_flag("Q27_PF_FULL_SERIAL", false);
                        auto gate_pn = [&](int p) {
                            if (p >= 6 || g != 0) return;
                            CK(hipStreamSynchronize(d.stream));
                            char nm[64]; std::snprintf(nm, sizeof nm, "swp post_mixer L%02d P%d", L, p);
                            std::vector<float> got = q27_d2h_bf16(d.hidden_slots + (size_t)(p - sb) * Q27_HID, Q27_HID);
                            std::vector<float> ref = q27_oracle(orc, "post_mixer", L, p);
                            if (!q27_gate(nm, got, ref, 2.5e-2)) {
                                fails.fetch_add(1);
                                char dp[96]; std::snprintf(dp, sizeof dp, "/tmp/swp_pm_L%02d_P%d.txt", L, p);
                                FILE* f = std::fopen(dp, "w");
                                if (f) {
                                    for (int i = 0; i < Q27_HID; ++i)
                                        std::fprintf(f, "%.6e %.6e\n", (double)got[i], (double)ref[i]);
                                    std::fclose(f);
                                }
                            }
                        };
                        const int OLb = Q27_OROWS / ndev;
                        const bool attb = g_pf_att_b && lay->is_full && !full_serial && !g_fp8x4 && !g_gdn_fq && g_rn_hostss
                                          && g_rn_dropy && !g_coll_bf16 && !g_coll_zcout && !g_hostss_neg && !swpl
                                          && OLb == (Q27_GDN_Z / ndev);   // no per-chunk opt-out: see pipe1
                        if (attb) {
                            // ATTENTION LAYERS, CHUNK FORM: input norms, batched q/k/v, per-position attention
                            // with the mixer quantized into chunk slots, ONE o_proj (residual share folded),
                            // ONE collective #1 (parity-staged; the next chunk's prologue is on the GPU while
                            // this chunk reduces when Q27_PF_PIPE1), ONE chunk tail. The MLP follows as today.
                            const int qstride = QLb + 2 * KVLb;
                            auto pro_att = [&](int q0, int par) {
                                const int Qch = (sw_end - q0 < chw) ? (sw_end - q0) : chw;
                                if (g_pf_p2p && g_pf_p2p_rn) for (int k = 0; k < ndev; ++k)
                                    CK(hipStreamWaitEvent(d.stream, C.ev_rd[par][k], 0));   // slot par free to overwrite
                                float* pout = (g_pf_p2p && g_pf_p2p_rn) ? C.part_d[par][g] : d.part_slots;
                                if (!(g_pf_norm_b && L > 0 &&
                                      q27_rmsnorm_hostss_fp8_b(d.pend_slots + (size_t)(q0 - sb) * Q27_HID, lay->input_norm,
                                            d.hidden_slots + (size_t)(q0 - sb) * Q27_HID, nullptr,
                                            d.xqin_slots, d.xsin_slots, Q27_HID, 1, lay->q_proj.in_scale,
                                            d.inv_slots + (q0 - sb), Qch, d.stream)))
                                for (int c = 0; c < Qch; ++c) {
                                    const int p = q0 + c;
                                    double mark = tp_now();
                                    const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                                    run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, pendp, nxt, p - sb,
                                                 d.inv_slots + (p - sb), nullptr, nullptr, false, 1,
                                                 d.xqin_slots + (size_t)c * Q27_HID,
                                                 d.xsin_slots + (size_t)c * (Q27_HID / 16));
                                }
                                // Q27_PF_X4W: q/k/v through the WIDE fp8 projection GEMM -- one launch per
                                // projection at the whole chunk M (128 rows x 64 tokens per block, M in the
                                // grid), replacing ceil(Qch/8) 8-wide launches.
                                // This is the ONE wide-fp8 call site that did not read its return
                                // value. The others are written as `!(g_pf_x4w && wide(...)) &&
                                // fallback`, so a decline falls through correctly; here a decline
                                // left q/k/v silently uncomputed. Guard the branch instead, so a
                                // card whose fp8 originals are gone takes the 8-wide path rather
                                // than launching a kernel on a null weight pointer.
                                if (g_tpsr_proj_rb) {
                                    // ---- Q27_TPSR_PROJ_RB: q/k/v on rocBLAS int8 from the transient per-row
                                    // mirror stack. The per-token input is re-derived from the bf16 raw residual
                                    // hidden_slots (exactly post_mlp's precedent), one stacked GEMM (N = qstride)
                                    // then three per-shard epilogues (each its own mirror alpha/row scales) into
                                    // an fp32 scratch and one f2bf into qkva_slots. Set 1 scratch, so the MLP's
                                    // set 0 is untouched under the pipe1 lookahead.
                                    const Q8Scr& Q1 = d.q8s[1];
                                    auto tdie = [&](const char* w) { std::fprintf(stderr, "Q27_TPSR_PROJ_RB qkv: %s declined (L%d card %d M=%d)\n", w, L, g, Qch); std::fflush(stderr); std::exit(1); };
                                    if (!q27_bf16_to_f32(d.hidden_slots + (size_t)(q0 - sb) * Q27_HID, Q1.hid32c, (size_t)Qch * Q27_HID, d.stream)) tdie("residual expand");
                                    if (!q27_red_rn_tok_b1(Q1.hid32c, lay->input_norm, nullptr, Q1.xqt, Q1.xst, Q27_HID, 1, lay->q_proj.in_scale, Qch, 0, d.stream)) tdie("input norm tok");
                                    if (!q27_rb_gemm_i8g(d.id, d.tpsr_q8r.w, qstride, Q27_HID, Q1.xqt, Qch, Q1.acc, Q27_HID, d.stream)) tdie("qkv gemm");
                                    float* F = Q1.mixer;
                                    if (!q27_epi_f32r_ld(Q1.acc, qstride, QLb, d.tpsr_q8r.s, 1, Q1.xst, d.tpsr_q8r.alpha, F, qstride, Qch, d.stream)) tdie("q epilogue");
                                    if (!q27_epi_f32r_ld(Q1.acc + QLb, qstride, KVLb, d.tpsr_k8r.s, 1, Q1.xst, d.tpsr_k8r.alpha, F + QLb, qstride, Qch, d.stream)) tdie("k epilogue");
                                    if (!q27_epi_f32r_ld(Q1.acc + QLb + KVLb, qstride, KVLb, d.tpsr_v8r.s, 1, Q1.xst, d.tpsr_v8r.alpha, F + QLb + KVLb, qstride, Qch, d.stream)) tdie("v epilogue");
                                    q27_f2bf_vec(F, d.qkva_slots, (size_t)Qch * qstride, d.stream);
                                } else if (g_pf_x4w && Qch > Q27_PF_MC &&
                                    lay->q_proj.w && lay->k_proj.w && lay->v_proj.w) {
                                    q27_proj_fp8_wideb(&lay->q_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots, Qch, qstride, g_pf_x4w, d.stream);
                                    q27_proj_fp8_wideb(&lay->k_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb, Qch, qstride, g_pf_x4w, d.stream);
                                    q27_proj_fp8_wideb(&lay->v_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb + KVLb, Qch, qstride, g_pf_x4w, d.stream);
                                } else {
                                const int nqh = (Qch + Q27_PF_MC - 1) / Q27_PF_MC;
                                const bool qstr = g_pf_hstreams && nqh > 1;
                                if (qstr) { CK(hipEventRecord(d.ev_fork, d.stream)); for (int k = 1; k < nqh && k <= 4; ++k) CK(hipStreamWaitEvent(d.hs[k - 1], d.ev_fork, 0)); }
                                for (int h0 = 0; h0 < Qch; h0 += Q27_PF_MC) {   // q/k/v kernels are 8 wide by register shape
                                    const int Qh = (Qch - h0 < Q27_PF_MC) ? (Qch - h0) : Q27_PF_MC;
                                    const int qi = h0 / Q27_PF_MC;
                                    hipStream_t qs_ = (qstr && qi >= 1 && qi <= 4) ? d.hs[qi - 1] : d.stream;
                                    const signed char* xqh = d.xqin_slots + (size_t)h0 * Q27_HID; const float* xsh = d.xsin_slots + (size_t)h0 * (Q27_HID / 16);
                                    unsigned short* qkh = d.qkva_slots + (size_t)h0 * qstride;
                                    if (g_pf_qkv2) {
                                        (g_pf_qkv3 ? q27_proj_fp8_s1_b3 : q27_proj_fp8_s1_b2)(&lay->q_proj, xqh, xsh, qkh, Qh, qstride, g_pf_qkv_r, qs_);
                                        (g_pf_qkv3 ? q27_proj_fp8_s1_b3 : q27_proj_fp8_s1_b2)(&lay->k_proj, xqh, xsh, qkh + QLb, Qh, qstride, g_pf_qkv_r, qs_);
                                        (g_pf_qkv3 ? q27_proj_fp8_s1_b3 : q27_proj_fp8_s1_b2)(&lay->v_proj, xqh, xsh, qkh + QLb + KVLb, Qh, qstride, g_pf_qkv_r, qs_);
                                    } else {
                                        q27_proj_fp8_s1_b(&lay->q_proj, xqh, xsh, qkh, Qh, qstride, d.stream);
                                        q27_proj_fp8_s1_b(&lay->k_proj, xqh, xsh, qkh + QLb, Qh, qstride, d.stream);
                                        q27_proj_fp8_s1_b(&lay->v_proj, xqh, xsh, qkh + QLb + KVLb, Qh, qstride, d.stream);
                                    }
                                }
                                if (qstr) { for (int k = 1; k < nqh && k <= 4; ++k) { CK(hipEventRecord(d.ev_join[k - 1], d.hs[k - 1])); CK(hipStreamWaitEvent(d.stream, d.ev_join[k - 1], 0)); } }
                                }
                                if (g_pf_att_chunk &&
                                    q27_attn_chunk_tp(d.qkva_slots, qstride, lay->q_norm, lay->k_norm,
                                                      d.kc + d.kv_off[fs], d.ks + d.kvs_off[fs], d.vc + Q27_VOFF(d.kv_off[fs]), d.vs + Q27_VSOFF(d.kvs_off[fs]),
                                                      d.qh_slots, d.mix_tile, OLb, q0, Qch, ndev,
                                                      d.mixq_slots, d.mixs_slots, lay->o_proj.in_scale, g_att_hpw, g_att_pf, d.attn_pob, d.attn_pml, d.stream)) {
                                    // Q27_PF_ATT_CHUNK: prep / decode / combine(+fused quantize into the chunk slots)
                                    // for the chunk's Qch queries in three launches; per-query arithmetic unchanged.
                                    C.pre_in[g] = 0;
                                } else
                                for (int c = 0; c < Qch; ++c) {
                                    const int p = q0 + c;
                                    double mark = tp_now();
                                    run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, nullptr, nxt, p - sb,
                                                 d.inv_slots + (p - sb), nullptr, nullptr, false, 4,
                                                 nullptr, nullptr, d.qkva_slots + (size_t)c * qstride,
                                                 nullptr, nullptr, nullptr,
                                                 pout + (size_t)c * Q27_HID, nullptr, nullptr,
                                                 d.mixq_slots + (size_t)c * OLb, d.mixs_slots + (size_t)c * (OLb / 16));
                                }
                                if (g_tpsr_proj_rb) {
                                    // ---- Q27_TPSR_PROJ_RB: o_proj on rocBLAS int8. Requant the per-16 mixer to
                                    // per-token (the one new kernel), GEMM (5120,1536), then the bf16-residual-fold
                                    // epilogue into pout exactly where the fp8 m2_res3 wrote it.
                                    const Q8Scr& Q1 = d.q8s[1];
                                    auto tdie = [&](const char* w) { std::fprintf(stderr, "Q27_TPSR_PROJ_RB o_proj: %s declined (L%d card %d M=%d)\n", w, L, g, Qch); std::fflush(stderr); std::exit(1); };
                                    if (!q27_requant_g16_tok(d.mixq_slots, d.mixs_slots, Q1.mixq_t, Q1.mixs_t, OLb, OLb, Qch, d.stream)) tdie("mixer requant");
                                    if (!q27_rb_gemm_i8g(d.id, d.tpsr_o8r.w, d.tpsr_o8r.rows, d.tpsr_o8r.K, Q1.mixq_t, Qch, Q1.acc, d.tpsr_o8r.gs, d.stream)) tdie("o gemm");
                                    if (!q27_epi_f32r_bf(Q1.acc, d.tpsr_o8r.rows, d.tpsr_o8r.s, 1, Q1.mixs_t, d.tpsr_o8r.alpha, pout,
                                                         d.hidden_slots + (size_t)(q0 - sb) * Q27_HID, 0.25f, Qch, d.stream)) tdie("o epilogue");
                                } else if (!(g_pf_x4w && q27_proj_fp8_wide(&lay->o_proj, d.mixq_slots, d.mixs_slots, pout, Qch, Q27_HID,
                                                    d.hidden_slots + (size_t)(q0 - sb) * Q27_HID, Q27_HID, 0.25f, g_pf_x4w, d.stream)) &&
                                    !(g_pf_m2f2 && (g_pf_m2f3 ? q27_proj_fp8_m2_res3 : q27_proj_fp8_m2_res2)(&lay->o_proj, d.mixq_slots, d.mixs_slots, pout, Qch, Q27_HID,
                                                    d.hidden_slots + (size_t)(q0 - sb) * Q27_HID, Q27_HID, 0.25f, g_pf_m2f2, d.stream)) &&
                                            !(g_pf_m3f && q27_proj_fp8_m3_res(&lay->o_proj, d.mixq_slots, d.mixs_slots, pout, Qch, Q27_HID,
                                                    d.hidden_slots + (size_t)(q0 - sb) * Q27_HID, Q27_HID, 0.25f, g_pf_m3f, d.stream)))
                                        q27_proj_fp8_m2_res(&lay->o_proj, d.mixq_slots, d.mixs_slots, pout, Qch, Q27_HID,
                                                    d.hidden_slots + (size_t)(q0 - sb) * Q27_HID, Q27_HID, 0.25f, d.stream);
                                if (!(g_pf_p2p && g_pf_p2p_rn)) {
                                    if (g_pf_kcopy && C.hpb_d[par][g]) q27_copy_f4(C.hpb_d[par][g], d.part_slots, Qch * Q27_HID, d.stream);
                                    else CK(hipMemcpyAsync(C.hpb[par][g], d.part_slots, (size_t)Qch * Q27_HID * 4, hipMemcpyDeviceToHost, d.stream));
                                }
                                CK(hipEventRecord(C.ev1[par][g], d.stream));
                            };
                            { const double tp0 = tp_now();
                              if (pipe1_done < p0) { pro_att(p0, ring(p0)); pipe1_done = p0; }
                              C.h_proatt[g] += tp_now() - tp0; }
                            const int par = ring(p0);
                            if (g_pf_pipe1) for (int la = 1; la <= look; ++la) {
                                const int q = p0 + la * chw;
                                if (q < sw_end && pipe1_done < q) { pro_att(q, ring(q)); pipe1_done = q; }
                            }
                            { const double tw = tp_now(); CK(hipEventSynchronize(C.ev1[par][g]));
                              const double td = tp_now() - tw; C.c_d2h[g] += td; C.h_sync[g] += td; }
                            if (q27_env_flag("Q27_P2P_DBG", false) && L == 3 && p0 == sw_beg && g == 0) {
                                CK(hipStreamSynchronize(d.stream));
                                std::vector<float> qv = q27_d2h_bf16(d.qkva_slots, 4);
                                std::vector<float> kv = q27_d2h_bf16(d.qkva_slots + QLb, 2);
                                float mq[4];
                                CK(hipMemcpy(mq, d.mixq_slots, 16, hipMemcpyDeviceToHost));
                                std::fprintf(stderr, "TP_DBG L3 post-attn q=%.3f %.3f k=%.3f %.3f mq=%.1f %.1f %.1f %.1f\n",
                                    (double)qv[0],(double)qv[1],(double)kv[0],(double)kv[1],
                                    (double)mq[0],(double)mq[1],(double)mq[2],(double)mq[3]);
                            }
                            if (g_pf_p2p && g_pf_p2p_rn) {
                                g_p2p_rn_calls.fetch_add(1, std::memory_order_relaxed);
                                // GPU-resident collective #1: own + peer partials, fused rmsnorm+quant.
                                for (int k = 0; k < ndev; ++k) if (k != g)
                                    CK(hipStreamWaitEvent(d.stream, C.ev1[par][k], 0));
                                q27_red_rn_perm_b(C.part_d[par][0], C.part_d[par][1], C.part_d[par][2], C.part_d[par][3],
                                        lay->post_norm, d.hidden_slots + (size_t)(p0 - sb) * Q27_HID,
                                        pq1, ps1, Q27_HID, 1, lay->gate.in_scale, Cch, d.stream);
                                CK(hipEventRecord(C.ev_rd[par][g], d.stream));
                                if (q27_env_flag("Q27_P2P_DBG", false) && L < 64 && p0 == sw_beg) {
                                    CK(hipStreamSynchronize(d.stream));
                                    for (int k = 0; k < ndev; ++k) {
                                        float tmp[8];
                                        CK(hipMemcpy(tmp, C.part_d[par][k], 32, hipMemcpyDeviceToHost));
                                        std::printf("DBG_RN card=%d L%d peer=%d p=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                            g, L, k, (double)tmp[0],(double)tmp[1],(double)tmp[2],(double)tmp[3],
                                            (double)tmp[4],(double)tmp[5],(double)tmp[6],(double)tmp[7]);
                                    }
                                    std::vector<float> hd = q27_d2h_bf16(d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, 8);
                                    std::printf("DBG_RN card=%d L%d hid=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                        g, L, (double)hd[0],(double)hd[1],(double)hd[2],(double)hd[3],
                                        (double)hd[4],(double)hd[5],(double)hd[6],(double)hd[7]);
                                }
                            } else {
                                { const double t0 = tp_now();
                                  tp_reduce_bx(C, g, Cch, C.hpb[par], C.accb[par], C.ssbb[par], C.inv_hbb[par]);
                                  const double dt = tp_now() - t0; C.t_coll[g] += dt; C.c_red[g] += dt; C.n_coll[g] += Cch;
                                  C.h_red1[g] += dt; }
                                if (q27_env_flag("Q27_P2P_DBG", false) && L < 64 && p0 == sw_beg) {
                                    for (int k = 0; k < ndev; ++k)
                                        std::printf("DBG_HST card=%d L%d peer=%d p=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                            g, L, k, (double)C.hpb[par][k][0],(double)C.hpb[par][k][1],(double)C.hpb[par][k][2],(double)C.hpb[par][k][3],
                                            (double)C.hpb[par][k][4],(double)C.hpb[par][k][5],(double)C.hpb[par][k][6],(double)C.hpb[par][k][7]);
                                    std::printf("DBG_HST card=%d L%d acc=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                        g, L, (double)C.accb[par][0],(double)C.accb[par][1],(double)C.accb[par][2],(double)C.accb[par][3],
                                        (double)C.accb[par][4],(double)C.accb[par][5],(double)C.accb[par][6],(double)C.accb[par][7]);
                                }
                                q27_rmsnorm_hostss_perm_b(C.accb_d[par], lay->post_norm,
                                            d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, nullptr,
                                            pq1, ps1, Q27_HID, 1, lay->gate.in_scale,
                                            C.inv_hbb[par] + (size_t)g * TpColl::BC, Cch, d.stream);
                    }
                } else
                        if (lay->is_full && !full_serial) {                 // batched q/k/v on top of the batched MLP
                            for (int c = 0; c < Cch; ++c) {
                                const int p = p0 + c;
                                double mark = tp_now();
                                const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                                run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, pendp, nxt, p - sb,
                                             d.inv_slots + (p - sb), nullptr, nullptr, false, 1,
                                             d.xqin_slots + (size_t)c * Q27_HID,
                                             d.xsin_slots + (size_t)c * (Q27_HID / 16));
                            }
                            const int qstride = QLb + 2 * KVLb;
                            if (g_pf_qkv2) {
                                (g_pf_qkv3 ? q27_proj_fp8_s1_b3 : q27_proj_fp8_s1_b2)(&lay->q_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots, Cch, qstride, g_pf_qkv_r, d.stream);
                                (g_pf_qkv3 ? q27_proj_fp8_s1_b3 : q27_proj_fp8_s1_b2)(&lay->k_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb, Cch, qstride, g_pf_qkv_r, d.stream);
                                (g_pf_qkv3 ? q27_proj_fp8_s1_b3 : q27_proj_fp8_s1_b2)(&lay->v_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb + KVLb, Cch, qstride, g_pf_qkv_r, d.stream);
                            } else {
                            q27_proj_fp8_s1_b(&lay->q_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots, Cch, qstride, d.stream);
                            q27_proj_fp8_s1_b(&lay->k_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb, Cch, qstride, d.stream);
                            q27_proj_fp8_s1_b(&lay->v_proj, d.xqin_slots, d.xsin_slots, d.qkva_slots + QLb + KVLb, Cch, qstride, d.stream);
                            }
                            if (q27_env_flag("Q27_QKV_GATE", false) && g == 0 && L < 12) {
                                // diagnostic: serial per-position q/k/v into d.qkva, compare vs
                                // the batched slot. NOTE: the extra input norm re-folds the
                                // pending, so hid is corrupted afterwards -- diagnosis only.
                                for (int c = 0; c < Cch; ++c) {
                                    const int p = p0 + c;
                                    double mark = tp_now();
                                    run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark,
                                                 (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID), nxt, p - sb,
                                                 d.inv_slots + (p - sb), nullptr, nullptr, false, 1);
                                    q27_proj_fp8_bf16(&lay->q_proj, d.xq, d.xs, d.qkva, d.stream);
                                    q27_proj_fp8_bf16(&lay->k_proj, d.xq, d.xs, d.qkva + QLb, d.stream);
                                    q27_proj_fp8_bf16(&lay->v_proj, d.xq, d.xs, d.qkva + QLb + KVLb, d.stream);
                                    CK(hipStreamSynchronize(d.stream));
                                    std::vector<float> a = q27_d2h_bf16(d.qkva, QLb + 2 * KVLb);
                                    std::vector<float> b = q27_d2h_bf16(d.qkva_slots + (size_t)c * qstride, QLb + 2 * KVLb);
                                    double mx = 0; int mi = -1, nbad = 0;
                                    for (int i = 0; i < QLb + 2 * KVLb; ++i) {
                                        const double dd = std::fabs((double)a[i] - (double)b[i]);
                                        if (dd > 0.01) { ++nbad; if (dd > mx) { mx = dd; mi = i; } }
                                    }
                                    std::printf("  QKV_GATE L%02d P%d: serial vs batched maxdiff=%.4f at %d, nbad=%d %s\n",
                                                L, p, mx, mi, nbad, nbad ? "***MISMATCH***" : "(identical)");
                                }
                            }
                            for (int c = 0; c < Cch; ++c) {
                                const int p = p0 + c;
                                double mark = tp_now();
                                run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, nullptr, nxt, p - sb,
                                             d.inv_slots + (p - sb),
                                             d.xq1_slots + (size_t)c * Q27_HID,
                                             d.xs1_slots + (size_t)c * (Q27_HID / 16),
                                             true, 2, nullptr, nullptr,
                                             d.qkva_slots + (size_t)c * (QLb + 2 * KVLb));
                                if (q27_env_flag("Q27_MIX6_DUMP", false) && g == 0 && L < 12 && p < 6) {
                                    CK(hipStreamSynchronize(d.stream));
                                    char dp[96]; std::snprintf(dp, sizeof dp, "/tmp/mix6_bat_L%02d_P%d.txt", L, p);
                                    std::vector<float> m6 = q27_d2h_bf16(d.mix6, Q27_OROWS / ndev);
                                    FILE* f = std::fopen(dp, "w");
                                    if (f) { for (int i = 0; i < (int)m6.size(); ++i) std::fprintf(f, "%.6e\n", (double)m6[i]); std::fclose(f); }
                                }
                                if (swpl) gate_pn(p);
                    }
                    } else {
                            // BATCHED GDN PROLOGUE (deliverable 2): mode 1 writes the per-position
                            // norm + quantized activation into chunk slots, ONE weight stream per
                            // chunk runs the two K=5120 projections (in_qkv, in_z), then mode 3 runs
                            // the GDN body (gemv/conv/step/quant/out_proj) reading the per-position
                            // slots. The step stays position-serial -- the recurrent state demands it.
                            // Q27_PF_GDN_SERIAL=1 is the control arm (per-position prologue).
                            const int QKVLb = Q27_GDN_QKV / ndev, ZLb = Q27_GDN_Z / ndev;
                            // BANKED 2026-09-10: the batched GDN prologue gated 12/12 IDENTICAL
                            // but measured 0.00% prefill wall (12 interleaved pairs, 4/12 paired
                            // wins) -- the fp8 family is wave-bound, so the weight reuse it buys is
                            // not the constraint. Shipped config = per-position prologue;
                            // Q27_PF_GDN_BATCH=1 opts back in. Full layers routed here by
                            // Q27_PF_FULL_SERIAL must always take the serial prologue.
                            // SHIPPED 2026-09-10: the batched GDN path is the default now that it
                            // carries the PF_TILE large-M projections (C=8 alone measured 0.00%).
                            const bool gdn_ser = !q27_env_flag("Q27_PF_GDN_BATCH", true) || lay->is_full;
                            // ---- Q27_PF_TILE: the large-M tile ----------------------------------
                            // The two K=5120 GDN projections run at M=PT (one weight broadcast per
                            // 64 positions) instead of M=8. The tile-wide norm+quant and BOTH
                            // projections happen once at each tile boundary; the per-8-chunk loop
                            // then runs the bodies (which stay position-serial: the GDN state is
                            // recurrent) followed by the already-gated 8-wide MLP + coll#2. Nothing
                            // else changes shape, so the tile only moves the projection arithmetic.
                            const int mk = q27_env_int("Q27_PF_MK", 2);   // 0 = s1_b, 1 = M-kernel v1, 2 = v2
                            int PT = q27_env_int("Q27_PF_TILE", 256);
                            PT = ((PT + 7) / 8) * 8;                  // whole 8-chunks only
                            if (PT > Q27_PF_TSLOT) PT = Q27_PF_TSLOT;
                            if (!gdn_ser && PT > 8) {
                                // TILE BASE IS WINDOW-RELATIVE. With an absolute base a window starting at
                                // position 341 never met a tile boundary, the GDN projections of the new
                                // turn's tokens were never computed, and the model continued its previous
                                // answer (48 of 64 layers never saw the new turn). Found by the 3-turn chat.
                                const int tb = sw_beg + ((p0 - sw_beg) / PT) * PT;
                                // The pipelined decision must be constant across a layer's chunks: the lookahead
                                // has ALREADY run the next chunk's prologue (GDN conv+step included), so a
                                // per-chunk opt-out (the old `Cch > 1`) made a one-position last chunk run its
                                // GDN body twice -- turn 3 of the serve gate (33 tokens) answered <|im_end|>.
                                const bool pipe1 = g_pf_pipe1 && g_pf_oproj_b && g_pf_tail_b && g_rn_dropy && g_rn_hostss
                                                   && !g_gdn_fq && !g_fp8x4 && !g_coll_bf16 && !g_coll_zcout && mk == 2;
                                if (!pipe1 && p0 == tb) {              // tile boundary
                                    const int Ct = (sw_end - tb < PT) ? (sw_end - tb) : PT;
                                    // Q27_PF_NORM_B: the tile's input norm+quantize (host-sumsq form) in ONE
                                    // launch; layer 0 has no pending residual and keeps the per-position form.
                                    if (!(g_pf_norm_b && L > 0 && g_rn_hostss && !g_hostss_neg &&
                                          q27_rmsnorm_hostss_fp8_b(d.pend_slots + (size_t)(tb - sb) * Q27_HID, lay->input_norm,
                                                d.hidden_slots + (size_t)(tb - sb) * Q27_HID, d.norm_slots,
                                                d.xqin_slots, d.xsin_slots, Q27_HID, 1, lay->in_qkv.in_scale,
                                                d.inv_slots + (tb - sb), Ct, d.stream)))
                                    for (int c = 0; c < Ct; ++c) {
                                        const int p = tb + c;
                                        double mark = tp_now();
                                        const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                                        run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, pendp, nxt, p - sb,
                                                     d.inv_slots + (p - sb), nullptr, nullptr, false, 1,
                                                     d.xqin_slots + (size_t)c * Q27_HID,
                                                     d.xsin_slots + (size_t)c * (Q27_HID / 16),
                                                     nullptr, d.norm_slots + (size_t)c * Q27_HID);
                                    }
                                    if (mk == 2) {
                                        if (!(g_pf_x4w && q27_proj_fp8_wideb(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, g_pf_x4w, d.stream)
                                                            && q27_proj_fp8_wideb(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, g_pf_x4w, d.stream)) &&
                                            !(g_pf_mk2r && (g_pf_mk2r2 ? q27_proj_fp8_m2r2 : q27_proj_fp8_m2r)(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, g_pf_mk2r, d.stream)
                                                            && (g_pf_mk2r2 ? q27_proj_fp8_m2r2 : q27_proj_fp8_m2r)(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, g_pf_mk2r, d.stream)) &&
                                                !(g_pf_mk16 && q27_fp8_to_h2(&lay->in_qkv, d.w16_scratch, d.stream)
                                                            && q27_proj_fp8_m2w16(&lay->in_qkv, d.w16_scratch, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, d.stream)
                                                            && q27_fp8_to_h2(&lay->in_z, d.w16_scratch, d.stream)
                                                            && q27_proj_fp8_m2w16(&lay->in_z, d.w16_scratch, d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, d.stream)) &&
                                                !(g_pf_mk3 && q27_proj_fp8_m3l(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, g_pf_mk3, d.stream)
                                                       && q27_proj_fp8_m3l(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, g_pf_mk3, d.stream))) {
                                        q27_proj_fp8_m2(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, d.stream);
                                        q27_proj_fp8_m2(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, d.stream);
                                        }
                                    } else {
                                        q27_proj_fp8_m(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, d.stream);
                                        q27_proj_fp8_m(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, d.stream);
                                    }
                                }
                                // BATCHED COLLECTIVE #1: every position's prologue lands its PARTIAL
                                // in a per-position slot (mode 6, no barrier), ONE rendezvous reduces
                                // the whole chunk, and the tails then read their own slice (mode 5).
                                // The GDN state stays position-serial -- only the collective is
                                // amortised, which is the half of the 128 collectives/token that
                                // coll#2's batching did not cover.
                                const bool oprb = g_pf_oproj_b && !g_gdn_fq && !g_fp8x4;
                                const bool cb1 = (q27_env_flag("Q27_COLL1_BATCH", false) || oprb) && Cch > 1
                                                 && g_rn_hostss && !g_coll_bf16 && !g_coll_zcout;
                                if (pipe1) {
                                    // PIPELINED COLLECTIVE #1. The next chunk's prologue (tile prologue if it
                                    // opens a tile, mode-6 bodies, chunk out_proj, D2H staging into the other
                                    // parity) is on the GPU while the host reduces this chunk; the chunk tail
                                    // then reads this chunk's parity result in place. Stream order carries
                                    // every producer->consumer pair; only the host wait moved.
                                    auto pro_chunk = [&](int q0, int par) {
                                        const int tbq = sw_beg + ((q0 - sw_beg) / PT) * PT;
                                        const int Qch = (sw_end - q0 < chw) ? (sw_end - q0) : chw;
                                        if (g_pf_p2p && g_pf_p2p_rn) for (int k = 0; k < ndev; ++k)
                                            CK(hipStreamWaitEvent(d.stream, C.ev_rd[par][k], 0));
                                        float* pout = (g_pf_p2p && g_pf_p2p_rn) ? C.part_d[par][g] : d.part_slots;
                                        if (q0 == tbq) {
                                            const int Ct = (sw_end - tbq < PT) ? (sw_end - tbq) : PT;
                                            if (!(g_pf_norm_b && L > 0 && !g_hostss_neg &&
                                                  q27_rmsnorm_hostss_fp8_b(d.pend_slots + (size_t)(tbq - sb) * Q27_HID, lay->input_norm,
                                                        d.hidden_slots + (size_t)(tbq - sb) * Q27_HID, d.norm_slots,
                                                        d.xqin_slots, d.xsin_slots, Q27_HID, 1, lay->in_qkv.in_scale,
                                                        d.inv_slots + (tbq - sb), Ct, d.stream)))
                                            for (int c = 0; c < Ct; ++c) {
                                                const int p = tbq + c;
                                                double mark = tp_now();
                                                const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                                                run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, pendp, nxt, p - sb,
                                                             d.inv_slots + (p - sb), nullptr, nullptr, false, 1,
                                                             d.xqin_slots + (size_t)c * Q27_HID,
                                                             d.xsin_slots + (size_t)c * (Q27_HID / 16),
                                                             nullptr, d.norm_slots + (size_t)c * Q27_HID);
                                            }
                                            if (g_tpsr_proj_rb) {
                                                // ---- Q27_TPSR_PROJ_RB: GDN in_qkv/in_z on rocBLAS int8 (stacked [4096][HID]).
                                                const Q8Scr& Q1 = d.q8s[1];
                                                auto tdie = [&](const char* w) { std::fprintf(stderr, "Q27_TPSR_PROJ_RB gdn in: %s declined (L%d card %d M=%d)\n", w, L, g, Ct); std::fflush(stderr); std::exit(1); };
                                                if (!q27_bf16_to_f32(d.hidden_slots + (size_t)(tbq - sb) * Q27_HID, Q1.hid32c, (size_t)Ct * Q27_HID, d.stream)) tdie("residual expand");
                                                if (!q27_red_rn_tok_b1(Q1.hid32c, lay->input_norm, nullptr, Q1.xqt, Q1.xst, Q27_HID, 1, lay->in_qkv.in_scale, Ct, 0, d.stream)) tdie("input norm tok");
                                                const int ntot = d.tpsr_iqkv8r.rows + d.tpsr_iz8r.rows;
                                                if (!q27_rb_gemm_i8g(d.id, d.tpsr_iqkv8r.w, ntot, Q27_HID, Q1.xqt, Ct, Q1.acc, Q27_HID, d.stream)) tdie("iqkv/iz gemm");
                                                const int iqR = d.tpsr_iqkv8r.rows, izR = d.tpsr_iz8r.rows;
                                                float* Fq = Q1.mixer; float* Fz = Q1.mixer + (size_t)Ct * iqR;
                                                if (!q27_epi_f32r_ld(Q1.acc, ntot, iqR, d.tpsr_iqkv8r.s, 1, Q1.xst, d.tpsr_iqkv8r.alpha, Fq, iqR, Ct, d.stream)) tdie("iqkv epilogue");
                                                if (!q27_epi_f32r_ld(Q1.acc + iqR, ntot, izR, d.tpsr_iz8r.s, 1, Q1.xst, d.tpsr_iz8r.alpha, Fz, izR, Ct, d.stream)) tdie("iz epilogue");
                                                q27_f2bf_vec(Fq, d.qkv_slots, (size_t)Ct * iqR, d.stream);
                                                q27_f2bf_vec(Fz, d.zbuf_slots, (size_t)Ct * izR, d.stream);
                                            } else if (!(g_pf_x4w && q27_proj_fp8_wideb(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, g_pf_x4w, d.stream)
                                                                && q27_proj_fp8_wideb(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, g_pf_x4w, d.stream)) &&
                                                !(g_pf_mk2r && (g_pf_mk2r2 ? q27_proj_fp8_m2r2 : q27_proj_fp8_m2r)(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, g_pf_mk2r, d.stream)
                                                            && (g_pf_mk2r2 ? q27_proj_fp8_m2r2 : q27_proj_fp8_m2r)(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, g_pf_mk2r, d.stream)) &&
                                                !(g_pf_mk16 && q27_fp8_to_h2(&lay->in_qkv, d.w16_scratch, d.stream)
                                                            && q27_proj_fp8_m2w16(&lay->in_qkv, d.w16_scratch, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, d.stream)
                                                            && q27_fp8_to_h2(&lay->in_z, d.w16_scratch, d.stream)
                                                            && q27_proj_fp8_m2w16(&lay->in_z, d.w16_scratch, d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, d.stream)) &&
                                                !(g_pf_mk3 && q27_proj_fp8_m3l(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, g_pf_mk3, d.stream)
                                                           && q27_proj_fp8_m3l(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, g_pf_mk3, d.stream))) {
                                            q27_proj_fp8_m2(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Ct, QKVLb, 0, d.stream);
                                            q27_proj_fp8_m2(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Ct, ZLb, 0, d.stream);
                                            }
                                            if (g_pf_gdn_scan) {
                                                // Q27_PF_GDN_SCAN: a/b for the tile, conv over the tile (in place), and the recurrence
                                                // for the tile with S resident in registers -- three launches instead of 3 x Ct; the
                                                // scan's fused quantize lands the mixers in the tile-wide chunk-slot layout.
                                                q27_bf16_gemv2_tile(lay->in_a, lay->in_b, d.norm_slots, Q27_HID, d.ab_tile, d.bb_tile,
                                                                    Q27_GDN_VH / ndev, Q27_HID, Ct, d.stream);
                                                q27_gdn_conv_tp_tile(d.qkv_slots, QKVLb, Ct, d.conv + d.conv_off[gs], lay->conv1d, g, ndev, d.stream);
                                                (g_pf_gdn_scan2 ? q27_gdn_scan2_tp : q27_gdn_scan_tp)(d.qkv_slots, QKVLb, d.zbuf_slots, ZLb, d.ab_tile, d.bb_tile, Q27_GDN_VH / ndev,
                                                                lay->A_log, lay->dt_bias, lay->gdn_norm, d.S + d.s_off[gs], d.mix_tile, ZLb,
                                                                g, ndev, Ct, d.mixq_tile, d.mixs_tile, lay->out_proj.in_scale, d.stream);
                                                C.pre_in[g] = 0;
                                            }
                                        }
                                        if (!g_pf_gdn_scan)
                                        for (int c = 0; c < Qch; ++c) {
                                            const int p = q0 + c, idx = p - tbq;
                                            double mark = tp_now();
                                            run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, nullptr, nxt, p - sb,
                                                         d.inv_slots + (p - sb), nullptr, nullptr, false, 6,
                                                         nullptr, nullptr, nullptr,
                                                         d.norm_slots + (size_t)idx * Q27_HID,
                                                         d.qkv_slots + (size_t)idx * QKVLb,
                                                         d.zbuf_slots + (size_t)idx * ZLb,
                                                         pout + (size_t)c * Q27_HID, nullptr, nullptr,
                                                         d.mixq_slots + (size_t)c * ZLb, d.mixs_slots + (size_t)c * (ZLb / 16));
                                        }
                                        const signed char* mq = g_pf_gdn_scan ? d.mixq_tile + (size_t)(q0 - tbq) * ZLb : d.mixq_slots;
                                        const float*       ms = g_pf_gdn_scan ? d.mixs_tile + (size_t)(q0 - tbq) * (ZLb / 16) : d.mixs_slots;
                                        if (g_tpsr_proj_rb) {
                                            // ---- Q27_TPSR_PROJ_RB: GDN out_proj on rocBLAS int8 (same pattern as o_proj).
                                            const Q8Scr& Q1 = d.q8s[1];
                                            auto tdie = [&](const char* w) { std::fprintf(stderr, "Q27_TPSR_PROJ_RB gdn out: %s declined (L%d card %d M=%d)\n", w, L, g, Qch); std::fflush(stderr); std::exit(1); };
                                            if (!q27_requant_g16_tok(mq, ms, Q1.mixq_t, Q1.mixs_t, ZLb, ZLb, Qch, d.stream)) tdie("mixer requant");
                                            if (!q27_rb_gemm_i8g(d.id, d.tpsr_op8r.w, d.tpsr_op8r.rows, d.tpsr_op8r.K, Q1.mixq_t, Qch, Q1.acc, d.tpsr_op8r.gs, d.stream)) tdie("out gemm");
                                            if (!q27_epi_f32r_bf(Q1.acc, d.tpsr_op8r.rows, d.tpsr_op8r.s, 1, Q1.mixs_t, d.tpsr_op8r.alpha, pout,
                                                                 (!g_hostss_neg) ? (d.hidden_slots + (size_t)(q0 - sb) * Q27_HID) : nullptr, 0.25f, Qch, d.stream)) tdie("out epilogue");
                                        } else if (!(g_pf_x4w && q27_proj_fp8_wide(&lay->out_proj, mq, ms, pout, Qch, Q27_HID,
                                                            (!g_hostss_neg) ? (d.hidden_slots + (size_t)(q0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, g_pf_x4w, d.stream)) &&
                                            !(g_pf_m2f2 && (g_pf_m2f3 ? q27_proj_fp8_m2_res3 : q27_proj_fp8_m2_res2)(&lay->out_proj, mq, ms, pout, Qch, Q27_HID,
                                                            (!g_hostss_neg) ? (d.hidden_slots + (size_t)(q0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, g_pf_m2f2, d.stream)) &&
                                            !(g_pf_m3f && q27_proj_fp8_m3_res(&lay->out_proj, mq, ms, pout, Qch, Q27_HID,
                                                            (!g_hostss_neg) ? (d.hidden_slots + (size_t)(q0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, g_pf_m3f, d.stream)))
                                        q27_proj_fp8_m2_res(&lay->out_proj, mq, ms, pout, Qch, Q27_HID,
                                                            (!g_hostss_neg) ? (d.hidden_slots + (size_t)(q0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, d.stream);
                                        if (!(g_pf_p2p && g_pf_p2p_rn)) {
                                            if (g_pf_kcopy && C.hpb_d[par][g]) q27_copy_f4(C.hpb_d[par][g], d.part_slots, Qch * Q27_HID, d.stream);
                                            else CK(hipMemcpyAsync(C.hpb[par][g], d.part_slots, (size_t)Qch * Q27_HID * 4, hipMemcpyDeviceToHost, d.stream));
                                        }
                                        CK(hipEventRecord(C.ev1[par][g], d.stream));
                                    };
                                    if (pipe1_done < p0) { pro_chunk(p0, ring(p0)); pipe1_done = p0; }
                                    const int par = ring(p0);
                                    for (int la = 1; la <= look; ++la) {
                                        const int q = p0 + la * chw;
                                        if (q < sw_end && pipe1_done < q) { pro_chunk(q, ring(q)); pipe1_done = q; }
                                    }
                                    { const double tw = tp_now(); CK(hipEventSynchronize(C.ev1[par][g])); C.c_d2h[g] += tp_now() - tw; }
                                    if (g_pf_p2p && g_pf_p2p_rn) {
                                        // GPU-resident collective #1 (GDN): own + peer partials, fused rmsnorm+quant.
                                        for (int k = 0; k < ndev; ++k) if (k != g)
                                            CK(hipStreamWaitEvent(d.stream, C.ev1[par][k], 0));
                                        q27_red_rn_perm_b(C.part_d[par][0], C.part_d[par][1], C.part_d[par][2], C.part_d[par][3],
                                                lay->post_norm, d.hidden_slots + (size_t)(p0 - sb) * Q27_HID,
                                                pq1, ps1, Q27_HID, 1, lay->gate.in_scale, Cch, d.stream);
                                        CK(hipEventRecord(C.ev_rd[par][g], d.stream));
                                        if (q27_env_flag("Q27_P2P_DBG", false) && L < 64 && (p0 == sw_beg || p0 + Cch >= sw_end)) {
                                            CK(hipStreamSynchronize(d.stream));
                                            for (int k = 0; k < ndev; ++k) {
                                                float tmp[8];
                                                CK(hipMemcpy(tmp, C.part_d[par][k], 32, hipMemcpyDeviceToHost));
                                                std::printf("DBG_RNG card=%d L%d peer=%d p=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                                    g, L, k, (double)tmp[0],(double)tmp[1],(double)tmp[2],(double)tmp[3],
                                                    (double)tmp[4],(double)tmp[5],(double)tmp[6],(double)tmp[7]);
                                            }
                                            std::vector<float> hd = q27_d2h_bf16(d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, 8);
                                            std::printf("DBG_RNG card=%d L%d hid=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                                g, L, (double)hd[0],(double)hd[1],(double)hd[2],(double)hd[3],
                                                (double)hd[4],(double)hd[5],(double)hd[6],(double)hd[7]);
                                        }
                                    } else {
                                        { const double t0 = tp_now();
                                          tp_reduce_bx(C, g, Cch, C.hpb[par], C.accb[par], C.ssbb[par], C.inv_hbb[par]);
                                          const double dt = tp_now() - t0; C.t_coll[g] += dt; C.c_red[g] += dt; C.n_coll[g] += Cch;
                                  C.h_red1[g] += dt; }
                                        if (q27_env_flag("Q27_P2P_DBG", false) && L < 64 && (p0 == sw_beg || p0 + Cch >= sw_end)) {
                                            for (int k = 0; k < ndev; ++k)
                                                std::printf("DBG_HSTG card=%d L%d peer=%d p=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                                    g, L, k, (double)C.hpb[par][k][0],(double)C.hpb[par][k][1],(double)C.hpb[par][k][2],(double)C.hpb[par][k][3],
                                                    (double)C.hpb[par][k][4],(double)C.hpb[par][k][5],(double)C.hpb[par][k][6],(double)C.hpb[par][k][7]);
                                            std::printf("DBG_HSTG card=%d L%d acc=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                                                g, L, (double)C.accb[par][0],(double)C.accb[par][1],(double)C.accb[par][2],(double)C.accb[par][3],
                                                (double)C.accb[par][4],(double)C.accb[par][5],(double)C.accb[par][6],(double)C.accb[par][7]);
                                        }
                                        q27_rmsnorm_hostss_perm_b(C.accb_d[par], lay->post_norm,
                                                    d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, nullptr,
                                                    pq1, ps1, Q27_HID, 1, lay->gate.in_scale,
                                                    C.inv_hbb[par] + (size_t)g * TpColl::BC, Cch, d.stream);
                                    }
                                } else
                                if (cb1) {
                                    for (int c = 0; c < Cch; ++c) {
                                        const int p = p0 + c, idx = p - tb;
                                        double mark = tp_now();
                                        run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, nullptr, nxt, p - sb,
                                                     d.inv_slots + (p - sb), nullptr, nullptr, false, 6,
                                                     nullptr, nullptr, nullptr,
                                                     d.norm_slots + (size_t)idx * Q27_HID,
                                                     d.qkv_slots + (size_t)idx * QKVLb,
                                                     d.zbuf_slots + (size_t)idx * ZLb,
                                                     d.part_slots + (size_t)c * Q27_HID, nullptr, nullptr,
                                                     (oprb && cb1) ? d.mixq_slots + (size_t)c * ZLb : nullptr,
                                                     (oprb && cb1) ? d.mixs_slots + (size_t)c * (ZLb / 16) : nullptr);
                                        if (swpl) gate_pn(p);
                                    }
                                    if (oprb && cb1)       // ONE out_proj for the chunk, residual share folded per position
                                        if (!(g_pf_x4w && q27_proj_fp8_wide(&lay->out_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                            (g_rn_hostss && !g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, g_pf_x4w, d.stream)) &&
                                            !(g_pf_m2f2 && (g_pf_m2f3 ? q27_proj_fp8_m2_res3 : q27_proj_fp8_m2_res2)(&lay->out_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                            (g_rn_hostss && !g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, g_pf_m2f2, d.stream)) &&
                                            !(g_pf_m3f && q27_proj_fp8_m3_res(&lay->out_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                            (g_rn_hostss && !g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, g_pf_m3f, d.stream)))
                                        q27_proj_fp8_m2_res(&lay->out_proj, d.mixq_slots, d.mixs_slots, d.part_slots, Cch, Q27_HID,
                                                            (g_rn_hostss && !g_hostss_neg) ? (d.hidden_slots + (size_t)(p0 - sb) * Q27_HID) : nullptr,
                                                            Q27_HID, 0.25f, d.stream);
                                    { const double t0 = tp_now();
                                      tp_allreduce_b(C, g, d.part_slots, Cch, d.stream);
                                      C.t_coll[g] += tp_now() - t0; C.n_coll[g] += Cch; }
                                    // Q27_PF_TAIL_B: the C tails (host-sumsq norm + permuting quantize into the
                                    // xq1/xs1 chunk slots) in ONE launch; y (d.norm) is only written when
                                    // g_rn_dropy is off, which would race across positions -> per-position then.
                                    if (!(g_pf_tail_b && g_rn_dropy && g_rn_hostss &&
                                          q27_rmsnorm_hostss_perm_b(C.acc, lay->post_norm,
                                                d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, nullptr,
                                                pq1, ps1, Q27_HID, 1, lay->gate.in_scale,
                                                C.inv_hb + (size_t)g * TpColl::BC, Cch, d.stream)))
                                    for (int c = 0; c < Cch; ++c) {
                                        const int p = p0 + c;
                                        double mark = tp_now();
                                        run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, nullptr, nxt, p - sb,
                                                     d.inv_slots + (p - sb),
                                                     d.xq1_slots + (size_t)c * Q27_HID,
                                                     d.xs1_slots + (size_t)c * (Q27_HID / 16),
                                                     true, 5, nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr, nullptr,
                                                     C.acc + (size_t)c * Q27_HID,
                                                     C.inv_hb + (size_t)g * TpColl::BC + c);
                                    }
                                } else
                                for (int c = 0; c < Cch; ++c) {        // bodies for this 8-chunk
                                    const int p = p0 + c, idx = p - tb;
                                    double mark = tp_now();
                                    run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, nullptr, nxt, p - sb,
                                                 d.inv_slots + (p - sb),
                                                 d.xq1_slots + (size_t)c * Q27_HID,
                                                 d.xs1_slots + (size_t)c * (Q27_HID / 16),
                                                 true, 3, nullptr, nullptr, nullptr,
                                                 d.norm_slots + (size_t)idx * Q27_HID,
                                                 d.qkv_slots + (size_t)idx * QKVLb,
                                                 d.zbuf_slots + (size_t)idx * ZLb);
                                    if (swpl) gate_pn(p);
                                }
                            }
                            if (!gdn_ser && PT <= 8) {
                                for (int c = 0; c < Cch; ++c) {
                                    const int p = p0 + c;
                                    double mark = tp_now();
                                    const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                                    run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, pendp, nxt, p - sb,
                                                 d.inv_slots + (p - sb), nullptr, nullptr, false, 1,
                                                 d.xqin_slots + (size_t)c * Q27_HID,
                                                 d.xsin_slots + (size_t)c * (Q27_HID / 16),
                                                 nullptr, d.norm_slots + (size_t)c * Q27_HID);
                                }
                                // Q27_PF_MK=1 routes the same two projections through the LARGE-M kernel
                                // (block tile [4 x 64], one weight broadcast per 64 positions). At the
                                // C=8 chunk this only exercises its arithmetic; the PF_TILE=128 chunk is
                                // what it is built for.
                                if (mk == 2) {
                                    q27_proj_fp8_m2(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, 0, d.stream);
                                    q27_proj_fp8_m2(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, 0, d.stream);
                                } else if (mk == 1) {
                                    q27_proj_fp8_m(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, 0, d.stream);
                                    q27_proj_fp8_m(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, 0, d.stream);
                                } else {
                                q27_proj_fp8_s1_b(&lay->in_qkv, d.xqin_slots, d.xsin_slots, d.qkv_slots, Cch, QKVLb, d.stream);
                                q27_proj_fp8_s1_b(&lay->in_z,   d.xqin_slots, d.xsin_slots, d.zbuf_slots, Cch, ZLb, d.stream);
                                }
                                for (int c = 0; c < Cch; ++c) {
                                    const int p = p0 + c;
                                    double mark = tp_now();
                                    run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, nullptr, nxt, p - sb,
                                                 d.inv_slots + (p - sb),
                                                 d.xq1_slots + (size_t)c * Q27_HID,
                                                 d.xs1_slots + (size_t)c * (Q27_HID / 16),
                                                 true, 3, nullptr, nullptr, nullptr,
                                                 d.norm_slots + (size_t)c * Q27_HID,
                                                 d.qkv_slots + (size_t)c * QKVLb,
                                                 d.zbuf_slots + (size_t)c * ZLb);
                                    if (swpl) gate_pn(p);
                                }
                            }
                            if (gdn_ser)
                            for (int c = 0; c < Cch; ++c) {
                                const int p = p0 + c;
                                double mark = tp_now();
                                const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                                run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, pendp, nxt, p - sb,
                                             d.inv_slots + (p - sb),
                                             d.xq1_slots + (size_t)c * Q27_HID,
                                             d.xs1_slots + (size_t)c * (Q27_HID / 16),
                                             true);       // stop after the post-norm tail
                                if (q27_env_flag("Q27_MIX6_DUMP", false) && g == 0 && L < 12 && p < 6) {
                                    CK(hipStreamSynchronize(d.stream));
                                    char dp[96]; std::snprintf(dp, sizeof dp, "/tmp/mix6_ser_L%02d_P%d.txt", L, p);
                                    std::vector<float> m6 = q27_d2h_bf16(d.mix6, lay->is_full ? (Q27_OROWS / ndev) : (Q27_GDN_Z / ndev));
                                    FILE* f = std::fopen(dp, "w");
                                    if (f) { for (int i = 0; i < (int)m6.size(); ++i) std::fprintf(f, "%.6e\n", (double)m6[i]); std::fclose(f); }
                                }
                                if (swpl) gate_pn(p);
                                if (p == 0 && L == 0) {
                                    CK(hipStreamSynchronize(d.stream));
                                    snap[g] = q27_d2h_bf16(d.hidden_slots, Q27_HID);
                                    snapbar.wait();
                                    if (g == 0) {
                                        for (int k = 1; k < ndev; ++k) {
                                            double mx = 0;
                                            for (int i = 0; i < Q27_HID; ++i)
                                                mx = std::max(mx, (double)std::fabs(snap[k][i] - snap[0][i]));
                                            std::printf("  card %d vs card 0 residual after layer 0: max|diff| = "
                                                        "%.3e %s\n", k, mx, mx == 0.0 ? "(identical)" : "");
                                        }
                                    }
                                    snapbar.wait();
                                }
                            }
                        }
                        { const double tp_ = tp_now(); post_mlp(lay, p0, Cch, (int)((p0 - sw_beg) / chw)); C.h_pm[g] += tp_now() - tp_; }
                        ++chunk_no;
                    }
                    C.h_loop[g] += tp_now() - tloop0;   // the host's whole chunk loop for this layer
                }

                else {
                for (int p = sw_beg; p < sw_end; ++p) {
                    double mark = tp_now();
                    const void* pendp = (L == 0) ? nullptr : (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                    run_layer_tp(d, lay, p, gs, fs, ctx, g, ndev, C, &mark, pendp, nxt, p - sb,
                                 d.inv_slots + (p - sb));
                    d.inv_slots[p - sb] = C.inv_h[g];
                    if (g_coll_zc)
                        CK(hipMemcpyAsync(d.pend_slots + (size_t)(p - sb) * Q27_HID, g_coll_acc,
                                          (size_t)Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                    else {
                        CK(hipMemcpyAsync(d.pend_stage, d.mixer, (size_t)Q27_HID * 4,
                                          hipMemcpyDeviceToHost, d.stream));
                        CK(hipMemcpyAsync(d.pend_slots + (size_t)(p - sb) * Q27_HID, d.pend_stage,
                                          (size_t)Q27_HID * 4, hipMemcpyHostToDevice, d.stream));
                    }
                    if (p == 0 && L == 0) {
                        CK(hipStreamSynchronize(d.stream));
                        snap[g] = q27_d2h_bf16(d.hidden_slots, Q27_HID);
                        snapbar.wait();
                        if (g == 0) {
                            for (int k = 1; k < ndev; ++k) {
                                double mx = 0;
                                for (int i = 0; i < Q27_HID; ++i)
                                    mx = std::max(mx, (double)std::fabs(snap[k][i] - snap[0][i]));
                                std::printf("  card %d vs card 0 residual after layer 0: max|diff| = "
                                            "%.3e %s\n", k, mx, mx == 0.0 ? "(identical)" : "");
                            }
                        }
                        snapbar.wait();
                    }

                }
                }
                if (isf) fs++; else gs++;
            }
            flush_c2();                                   // the last chunk's collective #2 (Q27_PF_PIPE)
            flush_c2w();                                  // the last tile's, and its measured FFN time
            while (ws_n > 0) drain_slice(true);           // finish the last tile's slices before the finals
            if (g == 0 && (wide_tiles || b13_calls))
                std::printf("Q27_MLP summary: wide_tiles=%ld wide_gateup=%ld wide_down=%ld coll2_width=%d "
                            "wide_ffn_gpu_ms=%.2f | b13_calls=%ld b13_ffn_gpu_ms=%.2f\n",
                            wide_tiles, wide_gu_calls, wide_dn_calls, wide_coll_w, C.t_wide_ffn[g],
                            b13_calls, C.t_b13_ffn[g]);
            if (g_mlp_tl) {
                // per-card CSV for the representative layer (L=0), times relative to that card's first tile
                char tlp[128]; std::snprintf(tlp, sizeof tlp, "/tmp/q27_tl_card%d.csv", g);
                FILE* tlf = std::fopen(tlp, "w");
                if (tlf) {
                    std::fprintf(tlf, "layer,tile,card,beg_ms,flush_end_ms,reduce_end_ms,ffn_enq_end_ms,"
                                      "stage_end_ms,bubble_ms\n");
                    const double t0l = (C.tl_n[g] > 0) ? C.tl[g][0].beg : 0.0;
                    for (int i = 0; i < C.tl_n[g]; ++i) { const TpColl::TLRec& R = C.tl[g][i];
                        std::fprintf(tlf, "%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", R.layer, R.tile, R.card,
                                     R.beg - t0l, R.sync_end - t0l, R.red_end - t0l,
                                     R.ffn_end - t0l, R.stage_end - t0l, R.bubble); }
                    std::fclose(tlf);
                }
                std::printf("Q27_MLP_TL card=%d reductions=%lld positions=%lld bytes=%lld "
                            "sync_ms=%.1f b1_ms=%.1f reduce_ms=%.1f b2_ms=%.1f h2d_ms=%.1f | "
                            "flush_ms=%.1f enq_ms=%.1f stage_ms=%.1f | gpu_bubble_ms=%.1f (n=%lld)\n",
                            g, C.r_n[g], C.r_pos[g], C.r_pos[g] * (long long)Q27_HID * 4,
                            C.r_sync[g], C.r_b1[g], C.r_loop[g], C.r_b2[g], C.r_h2d[g],
                            C.h_flush[g], C.h_enq[g], C.h_stage[g], C.r_bubble[g], C.r_bub_n[g]);
                std::printf("Q27_HOST card=%d loop=%.1f pm=%.1f proatt=%.1f red1=%.1f red2=%.1f sync=%.1f flushw=%.1f "
                            "enqFFN=%.1f stage=%.1f bubble_fail=%lld evpair=%s\n",
                            g, C.h_loop[g], C.h_pm[g], C.h_proatt[g], C.h_red1[g], C.h_red2[g], C.h_sync[g], C.h_flush[g],
                            C.h_enq[g], C.h_stage[g], C.r_bub_fail[g],
                            C.ev_pair_ok ? "PASS(controlled pair elapsed works)" : "FAIL(controlled pair elapsed fails)");
            }
            // per-position finals: residual-2 add + final norm + lm_head + argmax + 4-way max.
            // ONLY THE LAST POSITION'S head is needed for generation; the others were 2.33 ms per
            // prompt position (lm_head + argmax + two syncs + two barriers, 16% of a 1K prefill and
            // 76 s of a 32K one). Every position runs only when the oracle ladder or the sweep gate
            // asks for it.
            const q27_globals_t* G = q27_globals(m, d.id);
            const bool head_all = swpl || (orc != nullptr);
            if (g_tpsr && g_spec && g_spec_k > 0 && g_mtp_pp && g_mtp_hall && g_mtppp[g].ready) {
                // ---- Q27_TPSR / Q27_MTP_PP: every prompt position's FINAL residual -> the pinned hall, with the
                // finals' own arithmetic (bf16 hidden of layer 63 + its collective-#2 result), BEFORE the finals
                // below fold the last position in place. The layer-split ring exports the same quantity per chunk. ----
                MtpPP& P = g_mtppp[g];
                for (int p0 = sw_beg; p0 < sw_end; p0 += Q27_PF_TSLOT) {
                    const int Cp = (sw_end - p0 < Q27_PF_TSLOT) ? (sw_end - p0) : Q27_PF_TSLOT;
                    if (!q27_bf16_to_f32(d.hidden_slots + (size_t)(p0 - sb) * Q27_HID, P.fin32, (size_t)Cp * Q27_HID, d.stream)) { std::fprintf(stderr, "Q27_TPSR: hall expand declined\n"); std::exit(1); }
                    q27_add_f32(P.fin32, d.pend_slots + (size_t)(p0 - sb) * Q27_HID, Cp * Q27_HID, d.stream);
                    q27_f2bf_vec(P.fin32, P.finbf, Cp * Q27_HID, d.stream);
                    CK(hipMemcpyAsync(g_mtp_hall + (size_t)(p0 - sb) * Q27_HID, P.finbf, (size_t)Cp * Q27_HID * 2, hipMemcpyDeviceToHost, d.stream));
                    CK(hipStreamSynchronize(d.stream));   // fin32/finbf are reused by the next tile
                }
            }
            for (int p = head_all ? sw_beg : sw_end - 1; p < sw_end; ++p) {
                const double t0 = tp_now();
                unsigned short* hidp = d.hidden_slots + (size_t)(p - sb) * Q27_HID;
                const void* pendp = (const void*)(d.pend_slots + (size_t)(p - sb) * Q27_HID);
                if (q27_env_flag("Q27_LS_DBG", false) && g == 0) {
                    std::vector<float> h0 = q27_d2h_bf16(hidp, 8);
                    float p0[8];
                    CK(hipMemcpy(p0, pendp, 32, hipMemcpyDeviceToHost));
                    std::printf("DBG_TP_FIN hid=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f | pend=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                        (double)h0[0],(double)h0[1],(double)h0[2],(double)h0[3],
                        (double)h0[4],(double)h0[5],(double)h0[6],(double)h0[7],
                        (double)p0[0],(double)p0[1],(double)p0[2],(double)p0[3],
                        (double)p0[4],(double)p0[5],(double)p0[6],(double)p0[7]);
                }
                if (q27_env_flag("Q27_FIN_DUMP", false) && g == 0) {
                    // the same two summands on the authoritative (BF16) arm
                    std::vector<float> fh = q27_d2h_bf16(hidp, Q27_HID);
                    static std::vector<float> fpp(Q27_HID);
                    CK(hipMemcpy(fpp.data(), pendp, (size_t)Q27_HID * 4, hipMemcpyDeviceToHost));
                    FILE* fp = std::fopen("/tmp/q27_fin_tp_h.bin", "wb");
                    if (fp) { std::fwrite(fh.data(), 4, Q27_HID, fp); std::fclose(fp); }
                    fp = std::fopen("/tmp/q27_fin_tp_p.bin", "wb");
                    if (fp) { std::fwrite(fpp.data(), 4, Q27_HID, fp); std::fclose(fp); }
                }
                if (g_coll_bf16) q27_add_inplace_bf16(hidp, (const unsigned short*)pendp, Q27_HID, d.stream);
                else             q27_add_inplace(hidp, (const float*)pendp, Q27_HID, d.stream);
                q27_rmsnorm(hidp, G->final_norm, d.norm, Q27_HID, 1, d.stream);
                if (pf_batch && head_all) {
                    // batch the lm_head across 8-position chunks: one weight stream, C outputs.
                    // Per-position accumulation order is identical to the serial kernel, so the
                    // tokens stay bit-identical; the argmax and the head max stay per position.
                    const int Cch = ((sw_end - p) < 8) ? (sw_end - p) : 8;
                    for (int c = 0; c < Cch; ++c) {
                        unsigned short* hidc = d.hidden_slots + (size_t)(p + c - sb) * Q27_HID;
                        const void* pendc = (const void*)(d.pend_slots + (size_t)(p + c - sb) * Q27_HID);
                        if (c > 0) {
                            if (g_coll_bf16) q27_add_inplace_bf16(hidc, (const unsigned short*)pendc, Q27_HID, d.stream);
                            else             q27_add_inplace(hidc, (const float*)pendc, Q27_HID, d.stream);
                            q27_rmsnorm(hidc, G->final_norm, d.norm, Q27_HID, 1, d.stream);
                        }
                        // d.norm is reused per slot, so the bf16 EXL3 wants is copied
                        // into the per-slot buffer the batched head reads.
                        if (G->ex_head.live())
                            CK(hipMemcpyAsync(d.norm_slots + (size_t)c * Q27_HID, d.norm, (size_t)Q27_HID * 2, hipMemcpyDeviceToDevice, d.stream));
                        q27_quant_perm(d.norm, d.xq_slots + (size_t)c * Q27_HID,
                                       d.xs_slots + (size_t)c * (Q27_HID / 16),
                                       Q27_HID, G->lm_head.in_scale, d.stream);
                    }
                    if (!exl3_head(d, G, d.norm_slots, d.pa_slots, Cch, d.stream))
                    q27_proj_nvfp4_b(&G->lm_head, d.xq_slots, d.xs_slots, d.pa_slots, Cch, d.stream);
                    for (int c = 0; c < Cch; ++c) {
                        q27_argmax_val(d.pa_slots + (size_t)c * G->lm_head.rows, d.dtok, d.dval,
                                       G->lm_head.rows, d.stream);
                        CK(hipStreamSynchronize(d.stream));
                        unsigned li; float lv;
                        CK(hipMemcpy(&li, d.dtok, 4, hipMemcpyDeviceToHost));
                        CK(hipMemcpy(&lv, d.dval, 4, hipMemcpyDeviceToHost));
                        C.hval[g] = lv;
                        C.hidx[g] = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)li);
                        C.b1.wait();
                        unsigned nx = C.hidx[0]; float bv = C.hval[0];
                        for (int k = 1; k < ndev; ++k)
                            if (C.hval[k] > bv || (C.hval[k] == bv && C.hidx[k] < nx)) { bv = C.hval[k]; nx = C.hidx[k]; }
                        C.b2.wait();
                        if (g == 0) {
                            if ((!sw_tok || g_serve) && p + c == sw_end - 1) {   // the first decode token (prompt sweep only)
                                tok = nx; out.push_back(nx); q27_emit(nx);
                                if (q27_is_eos(nx) || (int)out.size() >= req_maxn) done = true;
                            }
                            if (t_prefill == 0) MS("T6_prefill_begin");
                            t_prefill += tp_now() - t0;
                        }
                        C.b1.wait();   // every card leaves the position with the same tok/pi/done
                        C.b2.wait();
                    }
                    p += Cch - 1;        // the chunk consumed Cch positions
                    continue;
                }
                q27_quant_perm(d.norm, d.xq, d.xs, Q27_HID, G->lm_head.in_scale, d.stream);
                if (!exl3_head(d, G, d.norm, d.pa, 1, d.stream))
                        q27_proj_nvfp4(&G->lm_head, d.xq, d.xs, d.pa, d.stream);
                q27_argmax_val(d.pa, d.dtok, d.dval, G->lm_head.rows, d.stream);
                CK(hipStreamSynchronize(d.stream));
                unsigned li; float lv;
                CK(hipMemcpy(&li, d.dtok, 4, hipMemcpyDeviceToHost));
                CK(hipMemcpy(&lv, d.dval, 4, hipMemcpyDeviceToHost));
                C.hval[g] = lv;
                C.hidx[g] = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)li);
                C.b1.wait();
                unsigned nx = C.hidx[0]; float bv = C.hval[0];
                for (int k = 1; k < ndev; ++k)
                    if (C.hval[k] > bv || (C.hval[k] == bv && C.hidx[k] < nx)) { bv = C.hval[k]; nx = C.hidx[k]; }
                C.b2.wait();
                if (g == 0) {
                    if ((!sw_tok || g_serve) && p == sw_end - 1) {   // the first decode token (prompt sweep only)
                        tok = nx; out.push_back(nx); q27_emit(nx);
                        if (q27_is_eos(nx) || (int)out.size() >= req_maxn) done = true;
                    }
                    if (t_prefill == 0) MS("T6_prefill_begin");
                    t_prefill += tp_now() - t0;
                }
                C.b1.wait();       // every card leaves the position with the same tok/pi/done
                C.b2.wait();
            }
            if (g_tpsr && g_spec && g_spec_k > 0 && g_mtp_pp && g_mtp_hall && g_mtppp[g].ready && sw_end - 1 > sw_beg) {
                // ---- Q27_TPSR / Q27_MTP_PP: prompt-condition the (replicated) draft layer on every card from the hall,
                // exactly as the layer-split sweep does after its ring; gated on the draft KV cap of EVERY card. ----
                const bool mtp_pp_fits = q27_mtp_within_cap(D, ndev, sw_end);
                if (mtp_pp_fits) {
                    const double tpp0 = tp_now();
                    C.b1.wait(); C.b2.wait();
                    mtp_prompt_pass(d, g, sw_beg, sw_end - 1, sb, sw_tok, sw_lo, prompt, emb, Q27_PF_TSLOT);
                    CK(hipMemcpy(d.nr_hprev, g_mtp_hall + (size_t)(sw_end - 1 - sb) * Q27_HID, (size_t)Q27_HID * 2, hipMemcpyHostToDevice));
                    d.nr_have_h = 1; d.nr_mtp_base = 0; d.nr_ncatch = 0;   // the first decode step is a speculative round
                    C.b1.wait(); C.b2.wait();
                    g_mtppp[g].t_ms += tp_now() - tpp0; g_mtppp[g].npos += (sw_end - 1 - sw_beg);
                    if (g == 0) std::printf("Q27_MTP_PREFILL positions=%d ms=%.1f\n", sw_end - 1 - sw_beg, tp_now() - tpp0);
                } else if (g == 0) {
                    int worst = 0; for (int q = 0; q < ndev; ++q) if (D[q].mtp_cap < sw_end) { worst = q; break; }
                    std::fprintf(stderr, "Q27_MTP_PP skipped: card %d draft cap %d < window end %d -- "
                                         "no draft conditioning for this window (speculation stays off)\n",
                                 worst, D[worst].mtp_cap, sw_end);
                    std::fflush(stderr);
                }
            }
            if (g == 0) {
                std::fprintf(stderr, "Q27_PHASE req=%d phase=prefill edge=end monotonic_ms=%.6f\n", req, tp_now());
                MS("T6b_sweep_end"); t_prefill = tp_now() - t_sw0;   // the WHOLE prompt phase
                // WHICH COLLECTIVE ACTUALLY RAN. host=0 with p2p>0 means the prefill was
                // GPU-resident end to end; host>0 with p2p=0 means the flag was inert and any
                // throughput quoted against it is the host-staged number wearing a P2P label.
                std::printf("Q27_COLL_PROOF p2p_rn=%lld p2p_mix=%lld host=%lld\n",
                            g_p2p_rn_calls.load(), g_p2p_mix_calls.load(),
                            g_host_coll_calls.load());
                std::fflush(stdout);
            }
            if (g == 0) std::printf("Q27_SWEEP_COLL positions=%d collectives=%lld host_coll_ms=%.1f  "
                                    "(pipelined: event_wait %.1f reduce %.1f ms; unpipelined phases need Q27_COLL_PROF=1; card 0)\n",
                                    nwin, (long long)C.n_coll[g], C.t_coll[g], C.c_d2h[g], C.c_red[g]);
            C.t_comp[g] = 0; C.t_coll[g] = 0; C.n_coll[g] = 0; C.t_wall[g] = 0;
            C.c_d2h[g]=C.c_b1[g]=C.c_red[g]=C.c_b2[g]=C.c_h2d[g]=0;
            if (g == 0) { C.skew_sum = 0; C.skew_n = 0;
                for (int k = 0; k < ndev; ++k) C.last_cnt[k] = 0;
                for (int k = 0; k < 136; ++k) { C.slot_skew[k]=0; C.slot_n[k]=0; } }
            g_pf_sweep = false;
            // A VERIFY must not move the handoff: only the prompt sweep finishes the prompt.
            if (!sw_tok || g_serve) { pos0 = sw_end; pi = prompt.size(); }
            }   // !g_ls: the layer-major TP sweep above stays the frozen default
        }

        // PREFILL-ONLY REQUEST: the sweep above has written KV and recurrent state for exactly these
        // positions, and the caller's next request is the REST OF THE SAME PROMPT. Skipping decode
        // here is what keeps pos_cur equal to the number of positions the caller handed over -- no
        // generated tokens inside the prompt, so every later step lands where it belongs. `done` is
        // shared by all four card threads and every card reaches this line with the same flag.
        if (req_pf_only) done = true;
        int pos = pos0;
        bool ck_pf_done = false;            // Q27_REWIND_CK: post-prefill bank taken once per request
        for (; pos < ctx && !done; ++pos) {
            const bool is_decode = (pi >= prompt.size());
            g_timeline_req = req; g_timeline_pos = pos;
            g_timeline_sample = getenv("Q27_TIMELINE_ROOT") && is_decode && n_rounds >= 2 && n_rounds < 5;
            // Q27_REWIND_CK: THE PREFILL->DECODE BOUNDARY IS THE CHECKPOINT THAT MATTERS. A client
            // that re-renders the previous assistant turn -- the reasoning-block case this server
            // already works to avoid -- diverges exactly HERE: at the position where the prompt
            // ended and generation began. A bank taken at the request's start is therefore always
            // too early (measured: divergence at 6122 with the only bank at 0) and one taken at its
            // end is always too late, because the generated tokens are what the client replaced.
            // Queued on d.stream without a sync: stream order already puts this copy ahead of the
            // decode kernels that overwrite d.S, and a rewind does its own synchronise.
            if (is_decode && !ck_pf_done && q27_rewind_ck() > 0) {
                ck_pf_done = true;
                const int cks = (req * 2 + 1) % q27_rewind_ck();
                CK(hipMemcpyAsync(d.ck_S + (size_t)cks * (d.s_bytes_total / sizeof(float)), d.S,
                                  d.s_bytes_total, hipMemcpyDeviceToDevice, d.stream));
                CK(hipMemcpyAsync(d.ck_conv + (size_t)cks * (d.conv_bytes_total / sizeof(unsigned short)),
                                  d.conv, d.conv_bytes_total, hipMemcpyDeviceToDevice, d.stream));
                if (g == 0) g_ck_pos[cks] = pos;
            }
            const double t0 = tp_now();
            if (g == 0) std::fprintf(stderr, "Q27_PHASE req=%d phase=%s edge=begin pos=%d monotonic_ms=%.6f\n", req, is_decode ? "decode" : "prefill-tail", pos, t0);
            double mark = t0;
            if (g_tp_lead) g_tpev[g].token_start(d.stream);   // stream idle: previous token drained
            // ---- Q27_SPEC: a speculative round (draft 1, verify 2 rows) replaces the plain step ----
            unsigned nx = 0;
            int sp_ncommit = 0; unsigned sp_tok[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
            // K=0 (Q27_SPEC_K=0, the NON-SPECULATIVE measurement configuration): a round still runs,
            // but it is one trunk row with no draft, so it must NOT depend on draft state. nr_have_h
            // is the "h_prev captured" flag the draft consumes, and q27_mtp_within_cap is a draft-KV
            // bound; neither is reachable at K=0, and requiring them would silently drop K=0 back
            // into the plain Q27_SPEC=0 path (which is not a working configuration).
            // LEAVE THE DRAFT BEHIND WITHOUT LEAVING THE ROW PATH. The draft KV is a per-card bound, so
            // a long generation walks out of it (measured: cap 65,536, position 65,388). Dropping to
            // the plain step there is what faulted the GPU -- see the spec_round comment. Degrade to
            // K=0 instead: no draft, no MTP pass, NR=1 trunk row through the shipped int8 row path.
            int eff_k = g_spec_k;
            if (g_spec && !g_sampler && is_decode && eff_k > 0 &&
                !(d.nr_have_h && q27_mtp_within_cap(D, ndev, pos + eff_k)))
                eff_k = 0;
            const bool sp_round = (g_spec && !g_sampler && is_decode && (eff_k == 0 || d.nr_have_h)
                                   && pos + eff_k < ctx);   // greedy only, and only where every card's draft cache can hold these positions
            if (sp_round) {
                sp_ncommit = g_sr ? spec_round_sr(d, m, emb, nlayer, pos, tok, g, ndev, C, &mark, sp_tok, eff_k)
                                  : spec_round(d, m, emb, nlayer, pos, tok, g, ndev, C, &mark, sp_tok, eff_k);
                nx = sp_tok[sp_ncommit - 1];
            } else {
            // embedding: the same 10 KiB row H2D on every card, straight out of the mmap
            const unsigned etok = (tok < (unsigned)Q27_VOCAB) ? tok : (unsigned)(Q27_VOCAB - 1);
            if (d.nr_ebounce) spec_emb_h2d(d, d.hidden_slots, emb, etok, d.stream);   // async through the pinned bounce ring
            else CK(hipMemcpy(d.hidden_slots, emb + (size_t)etok * Q27_HID, Q27_HID * 2, hipMemcpyHostToDevice));
            if (ladder && pos < 6 && g == 0) {
                if (!q27_gate("embed", q27_d2h_bf16(d.hidden_slots, Q27_HID),
                              q27_oracle(orc, "embed", -1, pos), 2.5e-2)) fails.fetch_add(1);
                mark = tp_now();
            }
            int gs = 0, fs = 0;
            // Residual 2 of layer L is not applied inside layer L; it rides here as `pending` and
            // is folded into layer L+1's input norm. SAFE because the input norm consumes d.mixer
            // before anything in that layer writes it, and stream order guarantees that. The last
            // one is applied explicitly below, before final_norm.
            const void* pending = nullptr;
            for (int L = 0; L < nlayer; ++L) {
                const q27_layer_t* lay = q27_layer_tp(m, L, d.id);
                if (!lay) { std::fprintf(stderr, "Q27_TP_NOT_RESIDENT layer %d card %d\n", L, g); std::exit(1); }
                const int isf = Q27_IS_FULL(L);
                const q27_layer_t* nxt = (L + 1 < nlayer) ? q27_layer_tp(m, L + 1, d.id) : nullptr;
                run_layer_tp(d, lay, pos, gs, fs, ctx, g, ndev, C, &mark, pending, nxt);
                pending = g_coll_zc ? (const void*)g_coll_acc : (const void*)d.mixer;
                if (isf) fs++; else gs++;
                if (pos == 0 && L == 0) {           // do all four cards agree on the residual?
                    CK(hipStreamSynchronize(d.stream));
                    snap[g] = q27_d2h_bf16(d.hidden_slots, Q27_HID);
                    snapbar.wait();
                    if (g == 0) {
                        for (int k = 1; k < ndev; ++k) {
                            double mx = 0;
                            for (int i = 0; i < Q27_HID; ++i)
                                mx = std::max(mx, (double)std::fabs(snap[k][i] - snap[0][i]));
                            std::printf("  card %d vs card 0 residual after layer 0: max|diff| = "
                                        "%.3e %s\n", k, mx, mx == 0.0 ? "(identical)" : "");
                        }
                    }
                    snapbar.wait();
                    mark = tp_now();
                }
                if (ladder && pos < 6 && g == 0) {
                    CK(hipStreamSynchronize(d.stream));
                    char nm[64]; std::snprintf(nm, sizeof nm, "post_mlp L%02d P%d", L, pos);
                    std::vector<float> got = q27_d2h_bf16(d.hidden_slots, Q27_HID);
                    std::vector<float> ref = q27_oracle(orc, "post_mlp", L, pos);
                    if (!q27_gate(nm, got, ref, 2.5e-2)) fails.fetch_add(1);
                    if (q27_env_flag("Q27_DUMP_LADDER", false)) {
                        char dp[96]; std::snprintf(dp, sizeof dp, "/tmp/ser_dump_L%02d_P%d.txt", L, pos);
                        FILE* f = std::fopen(dp, "w");
                        if (f) {
                            for (int i = 0; i < Q27_HID; ++i)
                                std::fprintf(f, "%.6e %.6e\n", (double)got[i], (double)ref[i]);
                            std::fclose(f);
                        }
                    }
                    mark = tp_now();
                }
            }
            // ---- head: local argmax over this card's 62080 logits, then a 4-way max ----
            const q27_globals_t* G = q27_globals(m, d.id);
            // The last layer's residual 2 was never applied inside the loop; apply it here.
            if (pending) {
                if (g_coll_bf16) q27_add_inplace_bf16(d.hidden_slots, (const unsigned short*)pending,
                                                      Q27_HID, d.stream);
                else             q27_add_inplace(d.hidden_slots, (const float*)pending, Q27_HID, d.stream);
            }
            q27_rmsnorm(d.hidden_slots, G->final_norm, d.norm, Q27_HID, 1, d.stream);
            if (g_spec) {   // Q27_SPEC: keep this position's final residual as the next draft's hidden
                if (!d.nr_have_h) d.nr_mtp_base = pos;
                CK(hipMemcpyAsync(d.nr_hprev, d.hidden_slots, Q27_HID * 2, hipMemcpyDeviceToDevice, d.stream));
                d.nr_have_h = 1;
            }
            q27_quant_perm(d.norm, d.xq, d.xs, Q27_HID, G->lm_head.in_scale, d.stream);
            if (!exl3_head(d, G, d.norm, d.pa, 1, d.stream))
                        q27_proj_nvfp4(&G->lm_head, d.xq, d.xs, d.pa, d.stream);
            // nx: declared before the Q27_SPEC branch
            if (g_sampler) {
                // Gather this card's 62080-logit shard into the shared pinned buffer, then let b1
                // publish all four. Every card samples the SAME 248320-vector with the SAME seed
                // (seeded by pos), so they agree on the token with no extra communication. The
                // barrier order is unchanged: b1 then b2, once per token, by every card.
                CK(hipStreamSynchronize(d.stream));
                C.t_comp[g] += tp_now() - mark;
                const double tc0 = tp_now();
                CK(hipMemcpy(g_logitbuf + (size_t)g * (size_t)G->lm_head.rows, d.pa,
                             (size_t)G->lm_head.rows * 4, hipMemcpyDeviceToHost));
                C.hval[g] = 0.f; C.hidx[g] = 0u;
                C.b1.wait();
                nx = q27_sample_logits(g_logitbuf, Q27_VOCAB, g_temp, g_topk, g_topp,
                                       (unsigned long long)pos);
                C.b2.wait();
                C.t_coll[g] += tp_now() - tc0; C.n_coll[g]++;
            } else {
            q27_argmax_val(d.pa, d.dtok, d.dval, G->lm_head.rows, d.stream);
            CK(hipStreamSynchronize(d.stream));
            unsigned li; float lv;
            CK(hipMemcpy(&li, d.dtok, 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(&lv, d.dval, 4, hipMemcpyDeviceToHost));
            C.t_comp[g] += tp_now() - mark;
            const double tc0 = tp_now();
            C.hval[g] = lv;
            C.hidx[g] = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)li);   // GLOBAL id
            C.b1.wait();
            nx = C.hidx[0]; float bv = C.hval[0];
            for (int k = 1; k < ndev; ++k)                       // ties -> lowest global index
                if (C.hval[k] > bv || (C.hval[k] == bv && C.hidx[k] < nx)) { bv = C.hval[k]; nx = C.hidx[k]; }
            C.b2.wait();
            C.t_coll[g] += tp_now() - tc0; C.n_coll[g]++;
            }
            // ---- MTP DRAFT (deliverable 3) -------------------------------------------------
            // The draft for THIS position consumes cat[0:5120] = norm_h(h_{pos-1}) and S.emb =
            // emb(t_pos), both left behind by the previous position, and predicts t_{pos+1} --
            // which the target has just produced as `nx`. So the comparison is same-step: no
            // cross-step bookkeeping and no stale register. Between the previous position's write
            // and this read sit 64 layers of collectives, which are hardware barriers that already
            // exist, so the embedding hop needs no new synchronization.
            if (g_mtp_on && is_decode) {
                MtpState& S = g_mtp[g];
                const bool dbg = (pos <= (int)prompt.size() + 3);
                if (dbg) { std::fprintf(stderr, "Q27_MTP_DBG g=%d pos=%d enter\n", g, pos); std::fflush(stderr); }
                if (pos == 0) std::fprintf(stderr,
                    "Q27_MTP_PTR g=%d hid=%p cat=%p emb=%p pre_hid=%p pre_emb=%p mtp_norm=%p in_norm=%p "
                    "qnorm=%p knorm=%p qkva=%p qh=%p mix6=%p act=%p nrm=%p hidb=%p xq=%p xs=%p "
                    "mtp_kc=%p mtp_vc=%p pa=%p pb=%p lm=%p rows=%d\n",
                    g, (void*)d.hidden_slots, (void*)S.cat, (void*)S.emb, (void*)S.W.pre_hid,
                    (void*)S.W.pre_emb, (void*)S.W.mtp_norm, (void*)S.W.in_norm, (void*)S.W.q_norm,
                    (void*)S.W.k_norm, (void*)S.qkva, (void*)S.qh, (void*)S.mix6, (void*)S.act,
                    (void*)S.nrm, (void*)S.hid, (void*)S.xq, (void*)S.xs, (void*)d.mtp_kc,
                    (void*)d.mtp_vc, (void*)d.pa, (void*)d.pb, (void*)G->lm_head.w, G->lm_head.rows);
                const double t_a = tp_now();
                if (pos >= 1) {
                    if (g != 0) {
                        if (dbg) { std::fprintf(stderr, "Q27_MTP_DBG g=%d pos=%d h2d_begin\n", g, pos); std::fflush(stderr); }
                        CK(hipMemcpy(S.emb, g_mtp[0].host_emb, (size_t)Q27_HID * 2, hipMemcpyHostToDevice));
                        if (dbg) { std::fprintf(stderr, "Q27_MTP_DBG g=%d pos=%d h2d_done\n", g, pos); std::fflush(stderr); }
                    }
                    const double t_b = tp_now();
                    mtp_draft_step(d, S, (q27_globals_t*)G, pos - 1, 1);
                    const double t_c = tp_now();
                    if (dbg) { std::fprintf(stderr, "Q27_MTP_DBG g=%d pos=%d draft_done\n", g, pos); std::fflush(stderr); }
                    q27_argmax_val(S.dpa, d.dtok, d.dval, G->lm_head.rows, d.stream);
                    CK(hipStreamSynchronize(d.stream));
                    unsigned li; float lv;
                    CK(hipMemcpy(&li, d.dtok, 4, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(&lv, d.dval, 4, hipMemcpyDeviceToHost));
                    const double t_d = tp_now();
                    C.hval[g] = lv;
                    C.hidx[g] = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)li);
                    C.b1.wait();
                    unsigned dt = C.hidx[0]; float dv = C.hval[0];
                    for (int k = 1; k < ndev; ++k)
                        if (C.hval[k] > dv || (C.hval[k] == dv && C.hidx[k] < dt)) { dv = C.hval[k]; dt = C.hidx[k]; }
                    C.b2.wait();
                    if (g == 0 && (S.n_draft % 8) == 0)
                        std::printf("Q27_MTP_T h2d=%.3f draft=%.3f settle=%.3f bar=%.3f total=%.3f ms\n",
                                    (t_b - t_a) * 1000, (t_c - t_b) * 1000, (t_d - t_c) * 1000,
                                    (tp_now() - t_d) * 1000, (tp_now() - t_a) * 1000);
                    if (g == 0 && g_mtp_dbg) {
                        std::vector<float> lg = q27_d2h_f32(S.dpa, G->lm_head.rows);
                        int bi[5] = {-1,-1,-1,-1,-1}; float bv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
                        for (int i2 = 0; i2 < (int)lg.size(); ++i2)
                            for (int k = 0; k < 5; ++k)
                                if (lg[i2] > bv[k]) { for (int j = 4; j > k; --j) { bv[j]=bv[j-1]; bi[j]=bi[j-1]; }
                                                      bv[k]=lg[i2]; bi[k]=i2; break; }
                        std::printf("Q27_MTP_TOP5");
                        for (int k = 0; k < 5; ++k) std::printf(" %d:%.2f", bi[k], bv[k]);
                        std::printf("  target=%u", nx);
                        if ((int)nx < (int)lg.size()) {
                            int rank = 0; for (int i2 = 0; i2 < (int)lg.size(); ++i2) if (lg[i2] > lg[nx]) ++rank;
                            std::printf(" target_rank=%d(/%d) target_logit=%.2f", rank, (int)lg.size(), lg[nx]);
                        }
                        std::printf("\n"); std::fflush(stdout);
                    }
                    S.last_draft = (int)dt;
                    // Resolve any older chain position whose target step is now.
                    for (int jj = 1; jj < 8; ++jj) {
                        const int idx = (pos - jj) & 15;
                        if (pos - jj >= 0 && S.q_iss[jj][idx] == pos - jj) {
                            ++S.ch_n[jj];
                            if (S.q_tok[jj][idx] == (int)nx) ++S.ch_hit[jj];
                            S.q_iss[jj][idx] = -1;
                        }
                    }
                    // Chain position 0 is resolved IMMEDIATELY and in this same step. A draft issued at
                    // step p predicts t_{p+1}, and this step's `nx` IS t_{p+1} -- which is exactly the
                    // comparison the aggregate counter makes. The earlier code registered it for step
                    // p+1, where nx is t_{p+2}: one position too late, which is why the per-position
                    // table read 0.129 while the aggregate read 0.5625. With this, ch_n[0]/ch_hit[0]
                    // are the aggregate by construction and can no longer disagree with it.
                    ++S.ch_n[0];
                    if ((int)dt == (int)nx) ++S.ch_hit[0];
                    if (g_mtp_chain > 1) {
                        // CHAINED drafts. No copy is needed: the next draft's normed hidden input can
                        // be written straight from the previous draft's residual stream (S.nrm) before
                        // that buffer is reused. Each draft occupies the next MTP KV slot.
                        int dprev = (int)dt;
                        for (int j2 = 1; j2 < g_mtp_chain; ++j2) {
                            if (j2 > 1) {
                                // resolve nothing here; chains beyond 0 are registered below
                            }
                            q27_rmsnorm(S.nrm, S.W.pre_hid, S.cat + mtp_hid_off(), Q27_HID, g_mtp_po, d.stream);
                            const unsigned tk2 = ((unsigned)dprev < (unsigned)Q27_VOCAB) ? (unsigned)dprev : 0u;
                            // g_mtp[0]'s staging and table: only card 0 allocates them (host memory,
                            // readable by every card), exactly as the publish path above does.
                            std::memcpy(g_mtp[0].host_emb,
                                        (const char*)g_mtp[0].emb_tbl + (size_t)tk2 * Q27_HID * 2,
                                        (size_t)Q27_HID * 2);
                            CK(hipMemcpy(S.emb, g_mtp[0].host_emb, (size_t)Q27_HID * 2, hipMemcpyHostToDevice));
                            mtp_draft_step(d, S, (q27_globals_t*)G, pos - 1 + j2, 0);
                            q27_argmax_val(S.dpa, d.dtok, d.dval, G->lm_head.rows, d.stream);
                            CK(hipStreamSynchronize(d.stream));
                            unsigned li2; float lv2;
                            CK(hipMemcpy(&li2, d.dtok, 4, hipMemcpyDeviceToHost));
                            CK(hipMemcpy(&lv2, d.dval, 4, hipMemcpyDeviceToHost));
                            C.hval[g] = lv2;
                            C.hidx[g] = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)li2);
                            C.b1.wait();
                            unsigned dt2 = C.hidx[0]; float dv2 = C.hval[0];
                            for (int k2 = 1; k2 < ndev; ++k2)
                                if (C.hval[k2] > dv2 || (C.hval[k2] == dv2 && C.hidx[k2] < dt2)) { dv2 = C.hval[k2]; dt2 = C.hidx[k2]; }
                            C.b2.wait();
                            dprev = (int)dt2;
                            // Chain position j>=1 predicts t_{p+1+j} and is produced by the target at
                            // step p+j, so it resolves there -- `pos + j2`, not `pos + 1 + j2`.
                            { S.q_tok[j2][pos & 15] = (int)dt2; S.q_iss[j2][pos & 15] = pos; }
                            if (g == 0 && S.ch_n[j2] < 3)
                                std::printf("Q27_MTP_CHAIN j=%d pos=%d draft=%u\n", j2, pos, dt2);
                        }
                    }
                    if (g == 0 && (S.n_draft % 16) == 0) {
                        std::printf("Q27_MTP_CHPOS");
                        for (int j2 = 0; j2 < 8; ++j2)
                            if (S.ch_n[j2] > 0) std::printf("  j%d=%d/%d(%.3f)", j2, S.ch_hit[j2], S.ch_n[j2],
                                                            (double)S.ch_hit[j2] / S.ch_n[j2]);
                        std::printf("\n"); std::fflush(stdout);
                    }
                    if ((int)dt == (int)nx) ++S.n_hit;      // hit = draft == what the target chose
                    if (g == 0 && S.n_draft < 10)
                        std::printf("Q27_MTP_CMP pos=%d draft=%u target=%u\n", pos, dt, nx);
                    ++S.n_draft;
                    if (g == 0 && (S.n_draft & 7) == 0)
                        std::printf("Q27_MTP_ACC drafts=%d hits=%d rate=%.4f\n",
                                    S.n_draft, S.n_hit, (double)S.n_hit / (double)S.n_draft);
                } else {
                    q27_rmsnorm(d.hidden_slots, S.W.pre_hid, S.cat + mtp_hid_off(), Q27_HID, 1, d.stream);
                    hipError_t e_st = hipDeviceSynchronize();
                    std::fprintf(stderr, "Q27_MTP_DBG g=%d pos=0 stash_done err=%d\n", g, (int)e_st);
                    std::fflush(stderr);
                }
                if (g == 0) {   // publish emb(t_{pos+1}) for the NEXT position's draft
                    const unsigned tk = (nx < (unsigned)Q27_VOCAB) ? nx : 0u;
                    std::memcpy(S.host_emb, (const char*)S.emb_tbl + (size_t)tk * Q27_HID * 2,
                                (size_t)Q27_HID * 2);
                    CK(hipMemcpy(S.emb, S.host_emb, (size_t)Q27_HID * 2, hipMemcpyHostToDevice));
                }
            }

            }   // !sp_round
            g_plain_warmed = true;
            if (g_tp_profile) g_tpev[g].drain_into(g_tpf[g]);   // stream is idle here
            const double dt = tp_now() - t0;
            if (g == 0) std::fprintf(stderr, "Q27_PHASE req=%d phase=%s edge=end pos=%d monotonic_ms=%.6f\n", req, is_decode ? "decode" : "prefill-tail", pos, t0 + dt);
            C.t_wall[g] += dt;                       // EVERY card, not just card 0
            if (g == 0) {
                if (is_decode) {
                    if (n_decode == 0) { MS("T7_decode_begin"); q27_lk_reset(); lk_skipped = 1; }
                    t_decode += dt; n_decode += sp_round ? sp_ncommit : 1; ++n_rounds;
                    if ((n_decode & 7) == 0) {          // cadence every 8 tokens
                        char tg[32]; std::snprintf(tg, sizeof tg, "T8_decode_%d", n_decode);
                        MS(tg);
                    }
                } else {
                    if (t_prefill == 0) MS("T6_prefill_begin");
                    t_prefill += dt;
                }
                if (!g_serve && !q27_env_flag("Q27_KEEP_PF_BUFFERS", false)
                    && pi >= prompt.size() && !d.pf_released) {   // optional standalone scratch shrink; the service retains these buffers
                    CK(hipStreamSynchronize(d.stream));
                    if (d.pend_slots) { CK(hipFree(d.pend_slots)); d.pend_slots = nullptr; }
                    if (d.hidden_slots) { CK(hipFree(d.hidden_slots)); d.hidden_slots = nullptr; }
                    if (d.emb_pin) { CK(hipHostFree(d.emb_pin)); d.emb_pin = nullptr; d.emb_pin_cap = 0; }
                    if (d.inv_slots) { CK(hipHostFree(d.inv_slots)); d.inv_slots = nullptr; }
                    CK(hipMalloc((void**)&d.hidden_slots, (size_t)8 * Q27_HID * 2));
                    CK(hipMemset(d.hidden_slots, 0, (size_t)8 * Q27_HID * 2));
                    d.pf_released = 1;
                    if (g == 0) { size_t fr = 0, tt = 0; CK(hipMemGetInfo(&fr, &tt));
                        std::printf("Q27_VRAM after_pf_release card %d: free %.2f GiB of %.2f GiB\n", g, (double)fr / 1073741824.0, (double)tt / 1073741824.0); }
                }
                if (pi < prompt.size()) tok = prompt[pi++];
                else if (sp_round) {
                    for (int i = 0; i < sp_ncommit; ++i) { out.push_back(sp_tok[i]); q27_emit(sp_tok[i]);
                        if (q27_is_eos(sp_tok[i]) || (int)out.size() >= req_maxn) { done = true; break; } }
                    tok = nx;
                } else { tok = nx; out.push_back(nx); q27_emit(nx);
                       if (q27_is_eos(nx) || (int)out.size() >= req_maxn) done = true; }
            }
            if (!is_decode) { C.t_comp[g] = 0; C.t_coll[g] = 0; C.n_coll[g] = 0; C.t_wall[g] = 0;
                              C.c_d2h[g]=C.c_b1[g]=C.c_red[g]=C.c_b2[g]=C.c_h2d[g]=0;
                              if (g == 0) { C.skew_sum = 0; C.skew_n = 0;
                                  for (int k = 0; k < ndev; ++k) C.last_cnt[k] = 0;
                                  for (int k = 0; k < 136; ++k) { C.slot_skew[k]=0; C.slot_n[k]=0; } }
                              for (int b = 0; b < TPF_N; ++b) g_tpf[g][b] = 0; }
            C.b1.wait();       // every card leaves the step with the same tok/pi/done
            C.b2.wait();
            if (sp_round) pos += sp_ncommit - 1;   // Q27_SPEC: the round committed sp_ncommit positions
        }
        // Decode updates TP quarters while LS prefill reads full owner state. Join the
        // producers, then gather updated quarters once at the request boundary. No device
        // synchronize; the adapter maps co-resident workers to D2D on nonblocking streams.
        if (g_ls && !g_sr && pos > pos0) {
            CK(hipEventRecord(d.ev_state_ready, d.stream));
            C.b1.wait(); C.b2.wait();
            for (int peer = 0; peer < ndev; ++peer) if (peer != g) {
                CK(hipStreamWaitEvent(d.stream, D[peer].ev_state_ready, 0));
                for (int L = 0; L < nlayer; ++L) {
                    if (!q27_ls_owned(L, g, ndev)) continue;
                    const int fs = L >> 2, gs = L - fs;
                    if (Q27_IS_FULL(L)) {
                        if (!g_pf_kvp2p) q27_gather_kv_quarter(d, D[peer], fs, pos0, pos, d.stream);
                    } else {
                        const size_t count = (size_t)q27_tp_shard(peer)->VHL * Q27_GDN_D * Q27_GDN_D;
                        CK(hipMemcpyPeerAsync(d.S + d.s_off[gs] + (size_t)q27_tp_shard(peer)->gv_h0 * Q27_GDN_D * Q27_GDN_D, d.id,
                            D[peer].S + D[peer].s_off[gs], D[peer].id, count * sizeof(float), d.stream));
                        q27_copy_conv_quarter(d, D[peer], gs, peer, ndev, true, d.stream);
                    }
                }
            }
            CK(hipStreamSynchronize(d.stream));
            C.b1.wait(); C.b2.wait();
        }
        pos_cur = pos;
        C.b1.wait();
        if(g==0){q27_transfer_receipt(req,false);char label[64];snprintf(label,sizeof label,"req%d-pos%d",req,pos_cur);q27_memory_receipt(label);}
        C.b2.wait();
        if(q27_env_flag("Q27_VALIDATE_STATE",false)) {
            unsigned* flags=nullptr; unsigned bad[5]={0};
            CK(hipMalloc((void**)&flags,sizeof bad));CK(hipMemsetAsync(flags,0,sizeof bad,d.stream));
            q27_finite_check(d.S,d.s_bytes_total/4,0,flags,0,d.stream);
            q27_finite_check(d.conv,d.conv_bytes_total/2,1,flags,1,d.stream);
            for(int fs=0;fs<Q27_LAYERS/4;++fs) {
                const size_t count=(size_t)pos_cur*(size_t)(d.kv_rowstride[fs]/16);
                q27_finite_check(d.ks+d.kvs_off[fs],count,2,flags,2,d.stream);
                q27_finite_check(d.vs+Q27_VSOFF(d.kvs_off[fs]),Q27_VSOFF(count),2,flags,2,d.stream);
            }
            q27_finite_check(d.nr_pa,(size_t)8*Q27_VOCAB,0,flags,3,d.stream);
            q27_finite_check(d.pa,Q27_VOCAB,0,flags,3,d.stream);
            q27_finite_check(d.hidden_slots,(size_t)8*Q27_HID,1,flags,4,d.stream);
            CK(hipMemcpyAsync(bad,flags,sizeof bad,hipMemcpyDeviceToHost,d.stream));
            CK(hipStreamSynchronize(d.stream));CK(hipFree(flags));
            std::fprintf(stderr,"Q27_FINITE req=%d logical=%d pos=%d state=%u conv=%u kvscale=%u logits_scratch=%u hidden=%u\n",req,g,pos_cur,bad[0],bad[1],bad[2],bad[3],bad[4]);
            if(bad[0]||bad[1]||bad[2]||bad[3]||bad[4]){std::fprintf(stderr,"FATAL nonfinite state\n");std::abort();}
        }
        C.b1.wait();C.b2.wait();
        g_timeline_sample = false;
        if (!g_coll_points.empty()) {
            char name[2048]; std::snprintf(name,sizeof name,"%s/collective-req%d-logical%d.tsv",getenv("Q27_TIMELINE_ROOT"),req,g);
            FILE* f=std::fopen(name,"w");
            if (!f) { std::fprintf(stderr,"Cannot write collective timeline\n");std::abort(); }
            std::fprintf(f,"request\tposition\tsite\tproducer_observed_ms\tstaged_ms\tall_arrived_ms\treduced_ms\tpublished_ms\treturned_ms\n");
            for (const auto& v:g_coll_points) std::fprintf(f,"%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",v.req,v.pos,v.site,v.enter,v.staged,v.arrived,v.reduced,v.published,v.returned);
            std::fclose(f);g_coll_points.clear();
        }
        // the tokens this request GENERATED are resident too, and they are part of what the next
        // render has to match; without them the LCP would falsely stop at the end of the prompt
        if (g == 0 && lcp_on && !out.empty()) lcp_hist.insert(lcp_hist.end(), out.begin(), out.end() - 1);
        if (!g_serve) break;
        if (g == 0) {
            if (g_text) { std::printf("\n"); std::fflush(stdout); }
            const double ttft = t_prefill;      // the sweep computes the first token at its last position
            // PER-REQUEST DECODE LEDGER (Q27_TP_PROFILE=1). The end-of-run report at the bottom of
            // this function only fires when the engine EXITS, so in serve mode it never printed and
            // the ledger was unreachable from real Harness traffic. Dump it per request instead and
            // zero the accumulators, so every Harness turn yields its own attribution.
            // ATTRIBUTION ONLY: each stage is measured with its own hipEvent pair, the stream still
            // pipelines, and overlapping stages double-count -- the SHARES rank the work, the total
            // is not a token time. Never quote a tok/s from this.
            // COMBINE NUMERICS GATE (Q27_VGATE=1). Reports what a bf16 partial payload WOULD cost,
            // measured on this request's real data: the combine accumulated the same sum twice,
            // once from the fp32 partials and once from those partials round-tripped through bf16,
            // and kept the worst relative deviation of the final gated output. Run this BEFORE
            // narrowing q27_attn_part_o, not after.
            if (q27_env_flag("Q27_VGATE", false) && n_decode > 0) {
                float vmx = 0.f, vref = 0.f; q27_vgate_read(&vmx, &vref, 1);
                std::fprintf(stderr, "Q27_VGATE req=%d steps=%d max_abs_dev=%.3e scale=%.3e rel_linf=%.3e "
                                     "(bf16 partials vs fp32, real data, card 0)\n",
                             req, n_decode, (double)vmx, (double)vref,
                             vref > 0.f ? (double)(vmx / vref) : 0.0);
            }
            if (g_tp_profile && n_decode > 0) {
                int crit = 0; double best = -1;
                for (int gg = 0; gg < ndev; ++gg) { double s = 0;
                    for (int b = 0; b < TPF_N; ++b) s += g_tpf[gg][b];
                    if (s > best) { best = s; crit = gg; } }
                std::fprintf(stderr, "Q27_DECLEDGER req=%d steps=%d card=%d (attribution only, shares rank the work)\n",
                            req, n_decode, crit);
                for (int b = 0; b < TPF_N; ++b)
                    std::fprintf(stderr, "Q27_DECLEDGER   %-20s %8.4f ms/token  %5.1f%%\n", TPF_NAME[b],
                                g_tpf[crit][b] / n_decode, best > 0 ? 100.0 * g_tpf[crit][b] / best : 0.0);
                std::fprintf(stderr, "Q27_DECLEDGER   %-20s %8.4f ms/token\n", "profiled sum", best / n_decode);
                for (int gg = 0; gg < ndev; ++gg) for (int b = 0; b < TPF_N; ++b) g_tpf[gg][b] = 0;
            }
            {   // E2 (2026-09-17): the generated-token stream's identity, on the request line.
                // FNV-1a over the ids this request produced. A TELEMETRY IDENTIFIER, NOT A GATE:
                // two requests with the same prompt and the same engine state must report the same
                // tokhash, because the shipped decode is greedy (Q27_TEMP unset -> device argmax).
                // A differing hash is the observation that localizes a state defect; it never
                // decides whether an answer is right.
                unsigned long long th = 1469598103934665603ULL;
                for (unsigned t : out) { th ^= (unsigned long long)t; th *= 1099511628211ULL; }
                const char* fin = req_pf_only ? "prefill"
                                : ((int)out.size() >= req_maxn ? "length" : "stop");
                const long rq_dr = d.nr_drafts - rq_drafts0;
                // TRANSPORT RECEIPT. host/p2p are this request's site counts, differenced from the
                // counters the two transport implementations increment themselves. nr is the row
                // count a site actually saw. These three make the arm self-evidencing: a "P2P arm"
                // whose p2p= stays 0 never happened, whatever the environment claimed.
                const long cs_h = g_coll_host_sites[g] - rq_collh0;
                const long cs_1 = g_coll_p2p_sites[g][1] - rq_collp0[1];
                const long cs_2 = g_coll_p2p_sites[g][2] - rq_collp0[2];
                const long cs_3 = g_coll_p2p_sites[g][3] - rq_collp0[3];
                const long cs_4 = g_coll_p2p_sites[g][4] - rq_collp0[4];
                if (g_nr_p2p == 4 && C.nrp_err[g] && C.nrp_err[g][0]) {   // an LL word never arrived: the reduce proceeded on stale data
                    std::fprintf(stderr, "FATAL Q27_NR_P2P=4 card %d: spin timeout at seq %u (word %u, peer slot %u, saw seq %u); the request's output is invalid. Aborting.\n",
                                 g, C.nrp_err[g][0], C.nrp_err[g][1], C.nrp_err[g][2], C.nrp_err[g][3]);
                    std::abort();
                }
                std::printf("Q27_REQ %d prompt_tokens=%d prefill_ms=%.1f ttft_ms=%.1f gen_tokens=%d decode_ms_per_tok=%.3f "
                            "decode_tok_s=%.2f rounds=%d ms_per_round=%.4f tok_per_round=%.4f acc_p1=%.3f acc_p2=%.3f dup=%d arm=%d "
                            "coll_host=%ld coll_p2p=%ld coll_push=%ld coll_pull=%ld coll_sdma=%ld coll_ll=%ld epi=%ld/%ld nr_eff=%d p2p_min=%d "
                            // exl3_calls / exl3_rows were already being PASSED here with no format
                            // specifiers for them -- computed, appended, and silently dropped by
                            // printf. They are the only receipt that says how much of a request
                            // actually ran on the trellis, so they are printed now. m1= is the
                            // share that went through the M=1 GEMV rather than the banded GEMM.
                            "pos_end=%d spec=%d spec_k=%d finish=%s tokhash=%016llx bin=%s "
                            "exl3_calls=%llu exl3_rows=%llu exl3_m1=%llu "
                            "ex_o=%llu ex_op=%llu ex_qkv=%llu ex_z=%llu\n",
                            req, (int)(req == 0 ? prompt.size() : req_ids.size()),
                            t_prefill, ttft, (int)out.size(), n_decode ? t_decode / n_decode : 0.0,
                            (n_decode && t_decode > 0) ? 1000.0 * n_decode / t_decode : 0.0,
                            n_rounds, n_rounds ? t_decode / n_rounds : 0.0,
                            n_rounds ? (double)n_decode / (double)n_rounds : 0.0,
                            rq_dr ? (double)(d.nr_acc_hist[1] - rq_acc1_0) / (double)rq_dr : 0.0,
                            rq_dr ? (double)(d.nr_acc_hist[2] - rq_acc2_0) / (double)rq_dr : 0.0,
                            g_nr_dup, g_ab_arm,
                            cs_h, cs_1 + cs_2 + cs_3 + cs_4, cs_1, cs_2, cs_3, cs_4, g_ll_epi_taken[g], g_ll_epi_armed[g], g_coll_nr_eff[g], g_nr_p2p_min,
                            pos_cur, g_spec, g_spec_k, fin, th, q27_binstamp8(),
                (unsigned long long)g_exl3_calls.load(std::memory_order_relaxed),
                (unsigned long long)g_exl3_rows.load(std::memory_order_relaxed),
                (unsigned long long)g_exl3_calls_m1.load(std::memory_order_relaxed),
                (unsigned long long)g_ex_site_o.load(std::memory_order_relaxed),
                (unsigned long long)g_ex_site_op.load(std::memory_order_relaxed),
                (unsigned long long)g_ex_site_qkv.load(std::memory_order_relaxed),
                (unsigned long long)g_ex_site_z.load(std::memory_order_relaxed));
                (void)rq_hits0;
                if (cs_h > 0 && n_rounds > 0) {
                    // HOST-COLLECTIVE PHASE LEDGER. us/site is the per-rendezvous cost; ms/round is
                    // that times sites/round, i.e. the share of the round each phase actually owns.
                    // stage_wait blocks on the GPU finishing the PRECEDING segment, so it prices
                    // that segment's execution (attention / GDN / MLP / norms), NOT the collective:
                    // the collective proper is b1 + reduce + b2 + invN. Reading stage_wait as
                    // rendezvous overhead would overstate the collective by an order of magnitude.
                    const double S = (double)cs_h, Rr = (double)n_rounds;
                    double ph[5];
                    for (int j = 0; j < 5; ++j) ph[j] = g_collt[g][j] - rq_collt0[j];
                    const double coll = ph[1] + ph[2] + ph[3] + ph[4];
                    std::fprintf(stderr,
                        "Q27_COLLT %d sites=%ld sites/round=%.1f | us/site: stage_wait=%.2f b1=%.2f "
                        "reduce=%.2f b2=%.2f invN=%.2f | ms/round: stage_wait=%.4f collective=%.4f "
                        "(b1=%.4f reduce=%.4f b2=%.4f invN=%.4f)\n",
                        req, cs_h, S / Rr,
                        1000.0 * ph[0] / S, 1000.0 * ph[1] / S, 1000.0 * ph[2] / S,
                        1000.0 * ph[3] / S, 1000.0 * ph[4] / S,
                        ph[0] / Rr, coll / Rr,
                        ph[1] / Rr, ph[2] / Rr, ph[3] / Rr, ph[4] / Rr);
                    std::fflush(stderr);
                    // The residual the ledger cannot attribute: layers minus the GPU execution it
                    // blocked on minus the collective it performed. This is the quantity the launch
                    // counter is testing, so it is computed here and handed straight to the dump.
                    const double layers_ms = (d.sp_t[3] - rq_spt0[3]) / Rr;
                    const double resid = layers_ms - ph[0] / Rr - coll / Rr;
                    q27_lk_dump(req, n_rounds - lk_skipped, Q27_LAYERS, resid, 12);
                }
                if (g_spec && rq_dr > 0) {
                    const double R = (double)rq_dr;
                    // stderr, NOT stdout: the server stdout pump logs only a whitelist of prefixes
                    // (Q27_REQ / Q27_MS / Q27_VRAM / Q27_TOKENS ...) and silently DROPS everything
                    // else -- which is why the engine has been accumulating a per-phase speculative
                    // ledger for weeks that never reached a single log line. stderr is pumped whole.
                    std::fprintf(stderr, "Q27_SPECT %d rounds=%ld ms/round: draft=%.4f draft_xchg=%.4f embed=%.4f "
                                "layers=%.4f head_sync=%.4f head_xchg=%.4f tail=%.4f total=%.4f\n",
                                req, rq_dr,
                                (d.sp_t[0] - rq_spt0[0]) / R, (d.sp_t[1] - rq_spt0[1]) / R,
                                (d.sp_t[2] - rq_spt0[2]) / R, (d.sp_t[3] - rq_spt0[3]) / R,
                                (d.sp_t[4] - rq_spt0[4]) / R, (d.sp_t[5] - rq_spt0[5]) / R,
                                (d.sp_t[6] - rq_spt0[6]) / R, (d.sp_t[7] - rq_spt0[7]) / R);
                }
                if (g_sr && g_sr_time && d.sr_rounds_t > 0) {   // Q27_SR_TIME: GPU time of this card's blocks per round (event-timed), every card
                    const double Rt = (double)d.sr_rounds_t;
                    std::fprintf(stderr, "Q27_SRT %d card %d rounds=%ld ms/round: blk=%.3f+%.3f+%.3f+%.3f=%.3f draft=%.3f head=%.3f wall=%.3f\n",
                                 req, g, d.sr_rounds_t, d.sr_ms_blk[0] / Rt, d.sr_ms_blk[1] / Rt, d.sr_ms_blk[2] / Rt, d.sr_ms_blk[3] / Rt,
                                 (d.sr_ms_blk[0] + d.sr_ms_blk[1] + d.sr_ms_blk[2] + d.sr_ms_blk[3]) / Rt,
                                 d.sr_ms_draft / Rt, d.sr_ms_head / Rt, d.sr_ms_wall / Rt);
                    if (g_sr_time >= 2)
                        std::fprintf(stderr, "Q27_SRT %d card %d stages ms/round (16 layers): norm+inproj=%.3f attn/gdn=%.3f outproj=%.3f postnorm+gateup=%.3f swiglu+down=%.3f sum=%.3f\n",
                                     req, g, d.sr_ms_st[0] / Rt, d.sr_ms_st[1] / Rt, d.sr_ms_st[2] / Rt, d.sr_ms_st[3] / Rt, d.sr_ms_st[4] / Rt, d.sr_ms_st[5] / Rt);
                    for (int v = 0; v < 4; ++v) d.sr_ms_blk[v] = 0; d.sr_ms_draft = d.sr_ms_head = d.sr_ms_wall = 0; d.sr_rounds_t = 0;
                    for (int k = 0; k < 6; ++k) d.sr_ms_st[k] = 0;
                    std::fflush(stderr);
                }
            }
            std::printf("Q27_TOKENS"); for (unsigned t : out) std::printf(" %u", t); std::printf("\n");
            std::fflush(stdout);
            if (g_text) { std::fprintf(g_text_out, "\n(%d tokens | prompt %d tok, prefill %.0f ms | decode %.1f tok/s | context %d)\n",
                                       (int)out.size(), (int)(req == 0 ? prompt.size() : req_ids.size()), t_prefill,
                                       (n_decode && t_decode > 0) ? 1000.0 * n_decode / t_decode : 0.0, pos_cur);
                          std::fflush(g_text_out); }
        }
        }   // for req
    });
    for (auto& t : th) t.join();
    MS("T9_decode_end");

    if (g_text && !g_serve) std::printf("\n");
    std::printf("Q27_TOKENS");
    for (unsigned t : out) std::printf(" %u", t);
    std::printf("\n");
    if (ladder) std::printf("Q27_LADDER fails=%d\n", fails.load());

    // ---- the five numbers, and nothing else ----
    if (n_decode > 0) {
        const int steps = n_decode;      // accumulators were reset after the last prompt step
        if (g_spec) std::printf("Q27_SPEC K=%d rounds=%ld accepted_drafts=%ld tokens_per_round=%.3f P(>=1)=%.3f P(>=2)=%.3f P(>=3)=%.3f committed_tokens=%d  (committed tok/s = decode_tok_s)\n",
                                g_spec_k, D[0].nr_drafts, D[0].nr_hits, D[0].nr_drafts ? 1.0 + (double)D[0].nr_hits / (double)D[0].nr_drafts : 0.0,
                                D[0].nr_drafts ? (double)D[0].nr_acc_hist[1] / D[0].nr_drafts : 0.0, D[0].nr_drafts ? (double)D[0].nr_acc_hist[2] / D[0].nr_drafts : 0.0,
                                D[0].nr_drafts ? (double)D[0].nr_acc_hist[3] / D[0].nr_drafts : 0.0, n_decode);
        if (g_spec && D[0].nr_drafts) { const double R = (double)D[0].nr_drafts;
            std::printf("Q27_SPEC_T per round (ms, card 0): draft %.3f  draft-xchg %.3f  embed %.3f  layers %.3f  head-sync %.3f  head-xchg %.3f  tail %.3f  total %.3f\n",
                        D[0].sp_t[0] / R, D[0].sp_t[1] / R, D[0].sp_t[2] / R, D[0].sp_t[3] / R, D[0].sp_t[4] / R, D[0].sp_t[5] / R, D[0].sp_t[6] / R, D[0].sp_t[7] / R); }
        std::printf("Q27_TP_PERF cards=%d layers=%d prompt=%zu decode_tokens=%d\n",
                    ndev, nlayer, prompt.size(), n_decode);

        // PER-CARD DECOMPOSITION IS THE AUTHORITATIVE FORM. Taking max(compute) and max(collective)
        // independently builds a SYNTHETIC card that represents no execution that ever occurred:
        // the maxima can come from different cards and double-count barrier skew, which can drive
        // the residual negative. Report four real rows, then the real critical-path card.
        std::printf("  per-card (ms/token):   wall   compute  collective  residual  collectives\n");
        int crit = 0;
        for (int g = 0; g < ndev; ++g) {
            const double w = C.t_wall[g] / steps, cp = C.t_comp[g] / steps, cl = C.t_coll[g] / steps;
            std::printf("    card %d          %8.3f %8.3f %11.3f %9.3f %12lld\n",
                        g, w, cp, cl, w - cp - cl, C.n_coll[g] / steps);
            if (C.t_wall[g] > C.t_wall[crit]) crit = g;
        }
        for (int g = 0; g < ndev; ++g)   // host-site phases (tp_allreduce_nr brackets), ms/token, per card
            std::printf("  Q27_COLLT_RUN card %d ms/token: stage_wait %.3f b1 %.3f reduce %.3f b2 %.3f invN %.3f\n", g,
                        g_collt[g][0] / steps, g_collt[g][1] / steps, g_collt[g][2] / steps, g_collt[g][3] / steps, g_collt[g][4] / steps);
        // The critical path is the card with the largest wall time. Its own compute/collective/
        // residual is a decomposition of an execution that actually happened.
        const double wall = C.t_wall[crit] / steps;
        const double comp = C.t_comp[crit] / steps;
        const double coll = C.t_coll[crit] / steps;
        std::printf("  CRITICAL PATH = card %d (largest wall)\n", crit);
        std::printf("    wall          %8.3f ms/token\n", wall);
        std::printf("    raw           %8.3f tok/s\n", 1000.0 / wall);
        std::printf("    compute       %8.3f ms/token\n", comp);
        std::printf("    collective    %8.3f ms/token\n", coll);
        std::printf("    collectives   %8lld per token  (128 payload all-reduces + 1 head max)\n",
                    C.n_coll[crit] / steps);
        // Pre-registered by the operator BEFORE this binary was ever run.
        std::printf("    residual      %8.3f ms/token   (= wall - compute - collective, %.1f%%)\n",
                    wall - comp - coll, 100.0 * (wall - comp - coll) / wall);
        if (g_coll_prof) {
            std::printf("  COLLECTIVE PHASE BREAKDOWN, per card, ms/token and us/call over %lld calls\n",
                        C.n_coll[crit] / steps);
            std::printf("    %-6s %9s %9s %9s %9s %9s\n", "card", "D2H", "wait b1", "reduce", "wait b2", "H2D");
            for (int gg = 0; gg < ndev; ++gg) {
                const double nc = (double)(C.n_coll[gg] / steps) * steps;
                std::printf("    %-6d %9.3f %9.3f %9.3f %9.3f %9.3f   ms/token\n", gg,
                            C.c_d2h[gg]/steps, C.c_b1[gg]/steps, C.c_red[gg]/steps,
                            C.c_b2[gg]/steps, C.c_h2d[gg]/steps);
                std::printf("    %-6s %9.2f %9.2f %9.2f %9.2f %9.2f   us/call\n", "",
                            C.c_d2h[gg]*1e3/nc, C.c_b1[gg]*1e3/nc, C.c_red[gg]*1e3/nc,
                            C.c_b2[gg]*1e3/nc, C.c_h2d[gg]*1e3/nc);
            }
            // WHO IS LATE, AND IS IT ALWAYS THE SAME CARD. A mean barrier wait says only that
            // skew exists. If one card owns "last" the cause is that card or its work; if "last"
            // is spread evenly the host executor is manufacturing the skew; if it tracks the slot
            // it belongs to particular producers.
            if (C.skew_n > 0) {
                std::printf("  ARRIVAL SKEW  mean %.2f us/call over %ld calls\n",
                            C.skew_sum * 1e3 / (double)C.skew_n, C.skew_n);
                std::printf("    last-to-arrive:");
                for (int gg = 0; gg < ndev; ++gg)
                    std::printf("  card %d %5.1f%%", gg, 100.0*(double)C.last_cnt[gg]/(double)C.skew_n);
                std::printf("\n");
                int ws[6] = {0,0,0,0,0,0};
                for (int k = 0; k < 136; ++k) if (C.slot_n[k]) {
                    for (int r = 0; r < 6; ++r) {
                        const int c = ws[r];
                        if (!C.slot_n[c] || C.slot_skew[k]/C.slot_n[k] > C.slot_skew[c]/C.slot_n[c]) {
                            for (int q = 5; q > r; --q) ws[q] = ws[q-1];
                            ws[r] = k; break;
                        }
                    }
                }
                std::printf("    worst collective slots (index within token, us/call):");
                for (int r = 0; r < 6; ++r) if (C.slot_n[ws[r]])
                    std::printf("  #%d %.1f", ws[r], C.slot_skew[ws[r]]*1e3/C.slot_n[ws[r]]);
                std::printf("\n");
            }
        }
        if (g_tp_profile) {
            // ATTRIBUTION ONLY. Every stage paid a full stream drain, so this total is inflated and
            // must never be quoted as a performance number. The SHARES are what it is for.
            double tot = 0;
            for (int b = 0; b < TPF_N; ++b) tot += g_tpf[crit][b];
            // A profile that silently drops stages past CAP under-reports exactly the late
            // layers, which is the same class of defect as a timer that re-records one event pair
            // per stage and measures only the last layer. Say so rather than assume it.
            std::printf("  event slots: peak %ld of %d per token, %ld calls UNTIMED%s\n",
                        g_tpev[crit].peak, TpEv::CAP, g_tpev[crit].dropped,
                        g_tpev[crit].dropped ? "   *** PROFILE TRUNCATED ***" : "");
            std::printf("  PER-STAGE ATTRIBUTION, card %d (hipEvent GPU time, "
                        "events, stream still pipelines)\n", crit);
            for (int b = 0; b < TPF_N; ++b)
                std::printf("    %-20s %8.3f ms/token  %5.1f%%\n",
                            TPF_NAME[b], g_tpf[crit][b] / steps, tot > 0 ? 100.0 * g_tpf[crit][b] / tot : 0.0);
            std::printf("    %-20s %8.3f ms/token  (sums GPU busy time per stage)\n", "profiled total", tot / steps);
            if (g_tp_lead) {
                TpEv& E = g_tpev[crit];
                std::printf("  SUBMIT LEAD, card %d: GPU start of each stage minus host submit time.\n"
                            "    lead < 12 us = the queue was EMPTY at submit (dispatch latency exposed)\n"
                            "    %-20s %9s %10s %9s\n", crit, "stage", "n/token", "mean us", "starved");
                for (int b = 0; b < TPF_N; ++b) if (E.lead_n[b])
                    std::printf("    %-20s %9.1f %10.1f %8.1f%%\n", TPF_NAME[b],
                                (double)E.lead_n[b] / steps, E.lead_sum[b] / E.lead_n[b],
                                100.0 * E.lead_starved[b] / E.lead_n[b]);
                static const char* HL[6] = {"<5", "5-12", "12-30", "30-100", "100-1000", ">=1000"};
                std::printf("    lead histogram (stages/token):");
                for (int h = 0; h < 6; ++h) std::printf("  %s:%.0f", HL[h], (double)E.lead_hist[h] / steps);
                std::printf("\n");
            }
        }
        std::printf("Q27_TP_TIMING prefill_ms=%.1f (%zu tokens)  decode_ms=%.1f (%d tokens)  epi=%ld/%ld\n",
                    t_prefill, prompt.size() - 1, t_decode, n_decode, g_ll_epi_taken[0], g_ll_epi_armed[0]);
        std::printf("Q27_RNQ_FALLBACK total=%lld  (0 = the fused norm+quant ran everywhere)\n",
                    (long long)g_rnq_fb.load());
        std::printf("Q27_KV_FUSE mode=%d fused_calls=%lld  (expect 16/token when on)\n",
                    g_kv_fuse, (long long)g_kv_fused.load());
        std::printf("Q27_COLL_KCOPY on=%d kernel_copies=%lld  (expect ~127-129/token when on)\n",
                    (int)g_coll_kcopy, (long long)g_kcopy_calls.load());
        std::printf("Q27_COLL_DFLAG mode=%d flag_waits=%lld skipped=%lld  (expect ~128/token/card waits when on)\n",
                    g_coll_dflag, (long long)g_dflag_waits.load(), (long long)g_dflag_skips.load());
    }
    // Per-position acceptance, printed ONCE at the end. The earlier periodic print was gated on
    // n_draft % 16 and fired BEFORE the increment, so a mid-run snapshot (j0=2/16 at step 16) was
    // being read as a final rate and wrongly disagreed with the aggregate. The resolution logic
    // itself was correct: a draft issued at step p for chain position j predicts t_{p+1+j} and is
    // resolved at step p+j against that step's own target token.
    if (g_mtp_on) {
        const MtpState& M = g_mtp[0];
        std::printf("Q27_MTP_POSITION");
        for (int j = 0; j < 8; ++j)
            if (M.ch_n[j] > 0)
                std::printf("  j%d=%d/%d(%.4f)", j, M.ch_hit[j], M.ch_n[j],
                            (double)M.ch_hit[j] / M.ch_n[j]);
        std::printf("\n");
        double mean_acc = 0.0, runp = 1.0;
        for (int j = 0; j < g_mtp_chain; ++j) {
            const double p = (M.ch_n[j] > 0) ? (double)M.ch_hit[j] / M.ch_n[j] : 0.0;
            runp *= p; mean_acc += runp;
        }
        std::printf("Q27_MTP_ROUND chain=%d mean_accepted=%.3f mean_committed=%.3f "
                    "(greedy, from per-position rates)\n", g_mtp_chain, mean_acc, 1.0 + mean_acc);
        std::printf("Q27_MTP_ACC drafts=%d hits=%d rate=%.4f\n", M.n_draft, M.n_hit,
                    M.n_draft ? (double)M.n_hit / M.n_draft : 0.0);
    }
    return fails.load() ? 4 : 0;
}


// ============================================================================================
// MTP PROBE (deliverable 3, step 1). The draft head is in the checkpoint but the loader skips
// it, so nothing has ever confirmed it is addressable from inside the engine. This fetches each
// mtp tensor through the same q27_host_ptr path the embedding uses and reports name, dtype, shape
// and bytes -- the input the upload needs. It touches NO upload state, so it cannot break a load.
// The checkpoint's hf_quant_config lists exclude_modules ["mtp*", "mtp.layers.0*"], i.e. the draft
// layer is stored UNQUANTIZED BF16, which is why its shapes are the raw ones.
//   usage: Q27_MTP_PROBE=1 ./q27_gen ./model
static const char* Q27_MTP_NAMES[15] = {
    "mtp.fc.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight"
};
static int q27_mtp_probe(q27_model_t* m) {
    long long total = 0;
    int found = 0;
    std::printf("  MTP_PROBE: %d candidate names, prefix tried as-is then 'model.' + name\n", 15);
    for (int i = 0; i < 15; ++i) {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, Q27_MTP_NAMES[i], &bytes, &dt, &nd, sh);
        char full[160];
        if (!p) {
            std::snprintf(full, sizeof full, "model.%s", Q27_MTP_NAMES[i]);
            p = q27_host_ptr(m, full, &bytes, &dt, &nd, sh);
        }
        if (!p) { std::printf("    %-46s MISSING\n", Q27_MTP_NAMES[i]); continue; }
        ++found; total += (long long)bytes;
        std::printf("    %-46s dt=%-3d ndim=%d shape=[%lld%s%lld] %.1f MB\n",
                    Q27_MTP_NAMES[i], dt, nd, sh[0], nd > 1 ? "," : "", nd > 1 ? sh[1] : 0LL,
                    bytes / 1e6);
    }
    std::printf("  MTP_PROBE: %d/15 present, %.1f MB total (BF16, unquantized by hf_quant_config)\n",
                found, total / 1e6);
    return found == 15 ? 0 : 1;
}


// MTP quantization probe: run the load-time quantizer on the REAL draft-head matrices and report
// the round-trip error and the packed sizes, so the draft's quality is known before it is used.
static int q27_mtp_quant_probe(q27_model_t* m) {
    const char* nm[5] = {
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight" };
    const int conv[5] = { 0, 0, 0, 1, 1 };            // 0 = nvfp4 (MLP/fc), 1 = fp8 (attention)
    for (int j = 0; j < 5; ++j) {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, nm[j], &bytes, &dt, &nd, sh);
        if (!p) { std::printf("    %-40s MISSING\n", nm[j]); continue; }
        const int R = (int)sh[0], K = (int)sh[1];
        std::vector<unsigned short> src((size_t)R * (size_t)K);
        std::memcpy(src.data(), p, (size_t)R * (size_t)K * 2);
        if (conv[j] == 0) {
            std::vector<unsigned char> wpk, gs; float ws2 = 1.f;
            const double err = q27_quant_nvfp4(src.data(), R, K, wpk, gs, &ws2);
            std::printf("    %-40s nvfp4 %5dx%-5d packed=%.1f MB + scales %.1f MB  relL2=%.4f\n",
                        nm[j], R, K, wpk.size() / 1e6, gs.size() / 1e6, err);
        } else {
            std::vector<unsigned char> w; float wscale = 1.f;
            const double err = q27_quant_fp8(src.data(), R, K, w, &wscale);
            std::printf("    %-40s fp8   %5dx%-5d packed=%.1f MB  wscale=%.6f  relL2=%.4f\n",
                        nm[j], R, K, w.size() / 1e6, wscale, err);
        }
    }
    std::printf("  (reference: the checkpoint's own model weights are stored at these formats;\n");
    std::printf("   relL2 by itself is not a pass/fail -- the gate is token agreement through the\n");
   std::printf("   verifier, which stays exact.)\n");
    return 0;
}


// ============================================================================================
// MTP WEIGHT-MAGNITUDE CROSS-CHECK.
// The draft's MLP output measured 5445 against the oracle's 0.026 for the SAME layer type at the
// same position -- a 200,000x contradiction on an input whose RMS is sane. Either the loaded MTP
// weights are the wrong tensor, or the projection is. This settles which, by putting the MTP's raw
// BF16 magnitudes beside the ENGINE's own lm_head dequantized exactly as its kernel dequantizes it.
static int q27_mtp_wmag_probe(q27_model_t* m) {
    const char* nm[4] = {"mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.down_proj.weight",
                         "mtp.layers.0.self_attn.q_proj.weight", "mtp.fc.weight"};
    const int   rr[4] = {17408, 5120, 12288, 5120};
    const int   kk[4] = {5120, 17408, 5120, 10240};
    std::printf("Q27_WMAG MTP matrices as loaded (true BF16 units):\n");
    for (int i = 0; i < 4; ++i) {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, nm[i], &bytes, &dt, &nd, sh);
        if (!p) { std::printf("  %-40s MISSING\n", nm[i]); continue; }
        const unsigned short* b = (const unsigned short*)p;
        const size_t n = (size_t)rr[i] * (size_t)kk[i];
        double ss = 0; float amax = 0;
        for (size_t j = 0; j < n; ++j) { const float v = q27_bf16_to_f(b[j]);
            ss += (double)v * v; if (std::fabs(v) > amax) amax = std::fabs(v); }
        std::printf("  %-40s rows=%-6d K=%-6d amax=%.6f rms=%.6f hostbytes=%zu\n",
                    nm[i], rr[i], kk[i], amax, std::sqrt(ss / n), bytes);
    }
    // ---- activation-layout equivalence: full-width (17408) vs 4 x 4352 sliced quantization ----
    // Two mathematically equivalent decompositions of the SAME projection disagreed by 78x. Energy is
    // permutation-invariant, so comparing reconstructed energy isolates LAYOUT from ORDERING: if the
    // two layouts carry the same energy, the activation quantization is not the culprit and the
    // weight slice is; if they do not, the layout is.
    {
        CK(hipSetDevice(0));
        const int N = 17408, SL = 4352;
        std::vector<unsigned short> h((size_t)N);
        for (int i = 0; i < N; ++i)
            h[i] = q27_f_to_bf16(3.0f * std::sin(0.01f * i) + 0.7f * std::cos(0.13f * i));
        unsigned short* dx = nullptr; signed char *q1 = nullptr, *q2 = nullptr;
        float *s1 = nullptr, *s2 = nullptr;
        CK(hipMalloc((void**)&dx, (size_t)N * 2));
        CK(hipMalloc((void**)&q1, N)); CK(hipMalloc((void**)&q2, N));
        CK(hipMalloc((void**)&s1, (size_t)(N / 16) * 4)); CK(hipMalloc((void**)&s2, (size_t)(N / 16) * 4));
        CK(hipMemcpy(dx, h.data(), (size_t)N * 2, hipMemcpyHostToDevice));
        q27_quant_perm(dx, q1, s1, N, 0.02f, 0);
        for (int j = 0; j < 4; ++j)
            q27_quant_perm(dx + (size_t)j * SL, q2 + (size_t)j * SL, s2 + (size_t)j * 272, SL, 0.02f, 0);
        CK(hipDeviceSynchronize());
        std::vector<signed char> Q1(N), Q2(N);
        std::vector<float> S1(N / 16), S2(N / 16);
        CK(hipMemcpy(Q1.data(), q1, N, hipMemcpyDeviceToHost));
        CK(hipMemcpy(Q2.data(), q2, N, hipMemcpyDeviceToHost));
        CK(hipMemcpy(S1.data(), s1, (size_t)(N / 16) * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(S2.data(), s2, (size_t)(N / 16) * 4, hipMemcpyDeviceToHost));
        double e1 = 0, e2 = 0, ex = 0; int nz1 = 0;
        for (int i = 0; i < N; ++i) {
            const double r1 = (double)Q1[i] * S1[i >> 4], r2 = (double)Q2[i] * S2[i >> 4];
            e1 += r1 * r1; e2 += r2 * r2; if (Q1[i]) ++nz1;
            const double x = q27_bf16_to_f(h[i]) / 0.02; ex += x * x;
        }
        std::printf("Q27_QT energy  target=%.6e  full=%.6e (%.4f x)  sliced=%.6e (%.4f x)  q_nonzero=%d/%d\n",
                    ex, e1, e1 / ex, e2, e2 / ex, nz1, N);
        std::fflush(stdout);
    }
    if (!q27_env_flag("Q27_MTP_WMAG_ALL", false)) return 0;   // engine-built part needs the upload
    CK(hipSetDevice(0));
    const q27_nvfp4_t* H = &q27_globals(m, 0)->lm_head;
    if (!H->w || !H->gs) { std::printf("  (globals not resident yet; run under Q27_MTP_DRAFT)\n"); return 0; }
    const size_t WB = (size_t)H->rows * (size_t)(H->K >> 1), SB = (size_t)H->rows * (size_t)(H->K >> 4);
    std::vector<unsigned char> w(WB), g(SB);
    CK(hipMemcpy(w.data(), H->w, WB, hipMemcpyDeviceToHost));
    CK(hipMemcpy(g.data(), H->gs, SB, hipMemcpyDeviceToHost));
    double ss = 0; float amax = 0;
    for (int r = 0; r < H->rows; ++r)
        for (int k = 0; k < H->K; k += 2) {
            const unsigned char by = w[(size_t)r * (H->K >> 1) + (k >> 1)];
            const float s = q27_e4m3_val(g[(size_t)r * (H->K >> 4) + (k >> 4)]) * H->ws2;
            const float v0 = q27_e2m1_val(by & 7) * s, v1 = q27_e2m1_val((by >> 4) & 7) * s;
            ss += (double)v0 * v0 + (double)v1 * v1;
            if (std::fabs(v0) > amax) amax = std::fabs(v0);
            if (std::fabs(v1) > amax) amax = std::fabs(v1);
        }
    const size_t n2 = (size_t)H->rows * (size_t)H->K;
    std::printf("  %-40s rows=%-6d K=%-6d amax=%.6f rms=%.6f  ws2=%.4e  <-- ENGINE-BUILT nvfp4\n",
                "lm_head (dequantized by its own rule)", H->rows, H->K, amax, std::sqrt(ss / n2), H->ws2);
    return 0;
}

// MTP DRAFT HEAD UPLOAD (deliverable 3, step 3).
//
// The draft layer is REPLICATED on every card rather than TP-sharded. Every card already holds the
// full hidden state after the collectives, and the draft's inputs (that state and the sampled
// token's embedding) are identical on all four, so a replicated draft produces identical drafts
// with NO communication at all -- the only cross-card op left is the existing 4-way head max.
// 285 MB quantized per card is nothing against 4.1 GiB of resident model.
//
// Weights arrive BF16 and are quantized here (q27_mtp_quant.h): NVFP4 for fc and the MLP, FP8 for
// attention, BF16 passthrough for the six 5120-wide norms. in_scale for the fp8/nvfp4 handles is a
// fixed activation scale (0.02 => activations up to ~9 stay inside e4m3); the draft's quality only
// moves the acceptance rate, never correctness, so a fixed scale is acceptable and measurable.
// struct MtpW now lives in q27_mtp_defs.h (included above) so that run_tp, which is defined
// BEFORE this point in the file, can hold one by value.

static int q27_mtp_upload(q27_model_t* m, MtpW* W, int id) {
    CK(hipSetDevice(id));
    const float IN_SCALE = g_mtp_is;
    // helper: BF16 host matrix -> quantized device buffers, fill the handle
    auto load_fp8 = [&](const char* name, q27_fp8_t* h, int rows, int K) -> bool {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, name, &bytes, &dt, &nd, sh);
        if (!p) { std::fprintf(stderr, "MTP_UPLOAD missing %s\n", name); return false; }
        std::vector<unsigned short> src((size_t)rows * (size_t)K);
        std::memcpy(src.data(), p, src.size() * 2);
        std::vector<unsigned char> w; float wscale = 1.f;
        q27_quant_fp8(src.data(), rows, K, w, &wscale);
        unsigned char* dw = nullptr;
        CK(hipMalloc((void**)&dw, w.size()));
        CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        h->w = dw; h->wscale = wscale; h->in_scale = IN_SCALE; h->rows = rows; h->K = K;
        W->bytes += (long long)w.size();
        return true;
    };
    auto load_nv = [&](const char* name, q27_nvfp4_t* h, int rows, int K) -> bool {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, name, &bytes, &dt, &nd, sh);
        if (!p) { std::fprintf(stderr, "MTP_UPLOAD missing %s\n", name); return false; }
        std::vector<unsigned short> src((size_t)rows * (size_t)K);
        std::memcpy(src.data(), p, src.size() * 2);
        // REVERTED to the host NVFP4 encoder after measurement. Routing through the engine's
        // q27_fp8_to_nvfp4_conv was tried in round 31 on the theory that the engine's own layout
        // must be right by construction; the gold stage trace then showed fc_out rms 134 and gate
        // rms 2951 where the host encoder gives 0.65 and 3.75. The converter sets ws2 = src->wscale,
        // a convention the engine's checkpoint satisfies and this host FP8 quantizer does not, so the
        // converted weights came out ~100-3000x too large. Convention compatibility is not the same
        // as "uses the engine's code".
        std::vector<unsigned char> wpk, gs; float ws2 = 1.f;
        q27_quant_nvfp4(src.data(), rows, K, wpk, gs, &ws2);
        unsigned char *dw = nullptr, *dg = nullptr;
        CK(hipMalloc((void**)&dw, wpk.size()));
        CK(hipMalloc((void**)&dg, gs.size()));
        CK(hipMemcpy(dw, wpk.data(), wpk.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dg, gs.data(), gs.size(), hipMemcpyHostToDevice));
        h->w = dw; h->gs = dg; h->ws2 = ws2; h->in_scale = IN_SCALE; h->rows = rows; h->K = K;
        W->bytes += (long long)(wpk.size() + gs.size());
        return true;
    };
    // Column slice of a [rows][Ktot] matrix -> a standalone [rows][Ksl] nvfp4 handle. Ksl must be
    // a multiple of 16 so the per-16 group scales stay aligned to the slice.
    auto load_nv_slice = [&](const char* name, q27_nvfp4_t* h, int rows, int Ktot, int koff, int Ksl) -> bool {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, name, &bytes, &dt, &nd, sh);
        if (!p) { std::fprintf(stderr, "MTP_UPLOAD missing %s\n", name); return false; }
        std::vector<unsigned short> src((size_t)rows * (size_t)Ktot);
        std::memcpy(src.data(), p, src.size() * 2);
        std::vector<unsigned short> sub((size_t)rows * (size_t)Ksl);
        for (int r = 0; r < rows; ++r)
            std::memcpy(&sub[(size_t)r * Ksl], &src[(size_t)r * Ktot + koff], (size_t)Ksl * 2);
        std::vector<unsigned char> wpk, gs; float ws2 = 1.f;
        q27_quant_nvfp4(sub.data(), rows, Ksl, wpk, gs, &ws2);
        unsigned char *dw = nullptr, *dg = nullptr;
        CK(hipMalloc((void**)&dw, wpk.size()));
        CK(hipMalloc((void**)&dg, gs.size()));
        CK(hipMemcpy(dw, wpk.data(), wpk.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dg, gs.data(), gs.size(), hipMemcpyHostToDevice));
        h->w = dw; h->gs = dg; h->ws2 = ws2; h->in_scale = IN_SCALE; h->rows = rows; h->K = Ksl;
        W->bytes += (long long)(wpk.size() + gs.size());
        return true;
    };
    // Q27_MTP_TP slices: a ROW range [row0, row0+rows) of a [rtot][K] matrix (fp8 or nvfp4), and a COLUMN range of an fp8 matrix
    auto load_fp8_rows = [&](const char* name, q27_fp8_t* h, int row0, int rows, int K) -> bool {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, name, &bytes, &dt, &nd, sh);
        if (!p) { std::fprintf(stderr, "MTP_UPLOAD missing %s\n", name); return false; }
        std::vector<unsigned short> src((size_t)rows * (size_t)K);
        std::memcpy(src.data(), (const unsigned short*)p + (size_t)row0 * K, src.size() * 2);
        std::vector<unsigned char> w; float wscale = 1.f;
        q27_quant_fp8(src.data(), rows, K, w, &wscale);
        unsigned char* dw = nullptr;
        CK(hipMalloc((void**)&dw, w.size()));
        CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        h->w = dw; h->wscale = wscale; h->in_scale = IN_SCALE; h->rows = rows; h->K = K;
        W->bytes += (long long)w.size();
        return true;
    };
    auto load_fp8_slice = [&](const char* name, q27_fp8_t* h, int rows, int Ktot, int koff, int Ksl) -> bool {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, name, &bytes, &dt, &nd, sh);
        if (!p) { std::fprintf(stderr, "MTP_UPLOAD missing %s\n", name); return false; }
        std::vector<unsigned short> sub((size_t)rows * (size_t)Ksl);
        for (int r = 0; r < rows; ++r) std::memcpy(&sub[(size_t)r * Ksl], (const unsigned short*)p + (size_t)r * Ktot + koff, (size_t)Ksl * 2);
        std::vector<unsigned char> w; float wscale = 1.f;
        q27_quant_fp8(sub.data(), rows, Ksl, w, &wscale);
        unsigned char* dw = nullptr;
        CK(hipMalloc((void**)&dw, w.size()));
        CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        h->w = dw; h->wscale = wscale; h->in_scale = IN_SCALE; h->rows = rows; h->K = Ksl;
        W->bytes += (long long)w.size();
        return true;
    };
    auto load_nv_rows = [&](const char* name, q27_nvfp4_t* h, int row0, int rows, int K) -> bool {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, name, &bytes, &dt, &nd, sh);
        if (!p) { std::fprintf(stderr, "MTP_UPLOAD missing %s\n", name); return false; }
        std::vector<unsigned short> src((size_t)rows * (size_t)K);
        std::memcpy(src.data(), (const unsigned short*)p + (size_t)row0 * K, src.size() * 2);
        std::vector<unsigned char> wpk, gs; float ws2 = 1.f;
        q27_quant_nvfp4(src.data(), rows, K, wpk, gs, &ws2);
        unsigned char *dw = nullptr, *dg = nullptr;
        CK(hipMalloc((void**)&dw, wpk.size()));
        CK(hipMalloc((void**)&dg, gs.size()));
        CK(hipMemcpy(dw, wpk.data(), wpk.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dg, gs.data(), gs.size(), hipMemcpyHostToDevice));
        h->w = dw; h->gs = dg; h->ws2 = ws2; h->in_scale = IN_SCALE; h->rows = rows; h->K = K;
        W->bytes += (long long)(wpk.size() + gs.size());
        return true;
    };
    auto grab = [&](const char* name, const unsigned short** dst, int n) -> bool {
        size_t bytes = 0; long long sh[Q27_MAX_DIMS]; int dt = 0, nd = 0;
        const void* p = q27_host_ptr(m, name, &bytes, &dt, &nd, sh);
        if (!p) { std::fprintf(stderr, "MTP_UPLOAD missing %s\n", name); return false; }
        unsigned short* d = nullptr;
        CK(hipMalloc((void**)&d, (size_t)n * 2));
        CK(hipMemcpy(d, p, (size_t)n * 2, hipMemcpyHostToDevice));
        *dst = d; W->bytes += (long long)n * 2;
        return true;
    };
    const char* B = "mtp.layers.0";
    char n[192];
    bool ok = true;
    if (g_mtp_tp) {   // Q27_MTP_TP: the draft layer sharded exactly like a trunk full-attention layer (plan of card `id`)
        const q27_tp_shard_t* T = q27_tp_shard(id);
        snprintf(n, sizeof n, "%s.self_attn.q_proj.weight", B);  ok &= load_fp8_rows(n, &W->q, T->q_h0 * 512, T->QL, 5120);
        snprintf(n, sizeof n, "%s.self_attn.k_proj.weight", B);  ok &= load_fp8_rows(n, &W->k, T->kv_h0 * Q27_HDIM, T->KVL, 5120); ok &= load_fp8(n, &W->k_full, 1024, 5120);
        snprintf(n, sizeof n, "%s.self_attn.v_proj.weight", B);  ok &= load_fp8_rows(n, &W->v, T->kv_h0 * Q27_HDIM, T->KVL, 5120); ok &= load_fp8(n, &W->v_full, 1024, 5120);
        snprintf(n, sizeof n, "%s.self_attn.o_proj.weight", B);  ok &= load_fp8_slice(n, &W->o, 5120, 6144, T->q_h0 * Q27_HDIM, T->OL);
        snprintf(n, sizeof n, "%s.mlp.gate_proj.weight", B);     ok &= load_nv_rows(n, &W->gate, T->mlp_off, T->mlp_len, 5120);
        snprintf(n, sizeof n, "%s.mlp.up_proj.weight", B);       ok &= load_nv_rows(n, &W->up,   T->mlp_off, T->mlp_len, 5120);
        snprintf(n, sizeof n, "%s.mlp.down_proj.weight", B);     ok &= load_nv_slice(n, &W->down[0], 5120, 17408, T->mlp_off, T->mlp_len);
        for (int j = 1; j < 4; ++j) W->down[j] = q27_nvfp4_t{};
        std::printf("Q27_MTP_TP card %d: q rows [%d,+%d) kv rows [%d,+%d) o K-slice [%d,+%d) mlp [%d,+%d)\n", id, T->q_h0 * 512, T->QL, T->kv_h0 * Q27_HDIM, T->KVL, T->q_h0 * Q27_HDIM, T->OL, T->mlp_off, T->mlp_len);
    } else {
    snprintf(n, sizeof n, "%s.self_attn.q_proj.weight", B);  ok &= load_fp8(n, &W->q, 12288, 5120);
    snprintf(n, sizeof n, "%s.self_attn.k_proj.weight", B);  ok &= load_fp8(n, &W->k,  1024, 5120);
    snprintf(n, sizeof n, "%s.self_attn.v_proj.weight", B);  ok &= load_fp8(n, &W->v,  1024, 5120);
    snprintf(n, sizeof n, "%s.self_attn.o_proj.weight", B);  ok &= load_fp8(n, &W->o,  5120, 6144);
    snprintf(n, sizeof n, "%s.mlp.gate_proj.weight", B);     ok &= load_nv(n, &W->gate, 17408, 5120);
    snprintf(n, sizeof n, "%s.mlp.up_proj.weight", B);       ok &= load_nv(n, &W->up,   17408, 5120);
    snprintf(n, sizeof n, "%s.mlp.down_proj.weight", B);
    for (int j = 0; j < 4; ++j) ok &= load_nv_slice(n, &W->down[j], 5120, 17408, j * 4352, 4352);
    }
    for (int j = 0; j < 2; ++j) ok &= load_nv_slice("mtp.fc.weight", &W->fc[j], 5120, 10240, j * 5120, 5120);
    snprintf(n, sizeof n, "%s.input_layernorm.weight", B);            ok &= grab(n, &W->in_norm, 5120);
    snprintf(n, sizeof n, "%s.post_attention_layernorm.weight", B);   ok &= grab(n, &W->post_norm, 5120);
    snprintf(n, sizeof n, "%s.self_attn.q_norm.weight", B);           ok &= grab(n, &W->q_norm, 256);
    snprintf(n, sizeof n, "%s.self_attn.k_norm.weight", B);           ok &= grab(n, &W->k_norm, 256);
    ok &= grab("mtp.norm.weight", &W->mtp_norm, 5120);
    ok &= grab("mtp.pre_fc_norm_embedding.weight", &W->pre_emb, 5120);
    ok &= grab("mtp.pre_fc_norm_hidden.weight", &W->pre_hid, 5120);
    return ok ? 0 : 1;
}

// probe: upload the draft head on card 0 and report device residency + a readback check
static int q27_mtp_upload_probe(q27_model_t* m) {
    MtpW W;
    const double t0 = tp_now();
    if (q27_mtp_upload(m, &W, 0)) { std::printf("  MTP_UPLOAD FAILED\n"); return 1; }
    CK(hipDeviceSynchronize());
    std::printf("  MTP_UPLOAD card 0: %.1f MB resident in %.1f ms\n", W.bytes / 1e6, tp_now() - t0);
    // readback check: the first 16 packed bytes of q_proj and the first 4 group scales
    unsigned char got[16];
    CK(hipMemcpy(got, W.q.w, 16, hipMemcpyDeviceToHost));
    std::printf("    q_proj first bytes:");
    for (int i = 0; i < 16; ++i) std::printf(" %02x", got[i]);
    std::printf("   wscale=%.6f in_scale=%.4f rows=%d K=%d\n", W.q.wscale, W.q.in_scale, W.q.rows, W.q.K);
    unsigned char g2[4];
    CK(hipMemcpy(g2, W.gate.gs, 4, hipMemcpyDeviceToHost));
    std::printf("    gate group scales:");
    for (int i = 0; i < 4; ++i) std::printf(" %02x", g2[i]);
    std::printf("   ws2=%.6e rows=%d K=%d\n", W.gate.ws2, W.gate.rows, W.gate.K);
    std::printf("    (replicated per card by design: the draft needs no collectives)\n");
    return 0;
}

// ---------------- MTP DRAFT STEP (deliverable 3) ----------------
// The draft head is a 17th FULL-ATTENTION layer, REPLICATED on every card: after each TP
// collective all four cards hold the identical hidden state and the draft's inputs are identical
// on all four, so a replicated draft needs no collectives at all. Only the lm_head stays
// column-parallel (62080 logits per card), so the draft's token is settled by the SAME 4-way max
// the target head already uses.
static int mtp_init(q27_model_t* m, int ndev) {
    for (int g = 0; g < ndev; ++g) {
        if (g_sr && g != g_sr_head) continue;   // Q27_LS_SR: the draft layer is not replicated; the head card runs it
        MtpState& S = g_mtp[g];
        CK(hipSetDevice(g));
        if (q27_mtp_upload(m, &S.W, g)) return 1;
        CK(hipMalloc((void**)&S.cat,  (size_t)2 * Q27_HID * 2));
        CK(hipMalloc((void**)&S.emb,  (size_t)Q27_HID * 2));
        CK(hipMalloc((void**)&S.qkva, (size_t)(Q27_QROWS + 2 * Q27_KVROWS) * 2));
        CK(hipMalloc((void**)&S.qh,   (size_t)Q27_OROWS * 2));
        CK(hipMalloc((void**)&S.mix6, (size_t)Q27_OROWS * 2));
        CK(hipMalloc((void**)&S.act,  (size_t)Q27_INTER * 2));
        CK(hipMalloc((void**)&S.nrm,  (size_t)Q27_HID * 2));
        CK(hipMalloc((void**)&S.hid,  (size_t)Q27_HID * 2));
        CK(hipMalloc((void**)&S.hid2, (size_t)Q27_HID * 2));
        CK(hipMalloc((void**)&S.xq,   (size_t)Q27_INTER));
        CK(hipMalloc((void**)&S.xs,   (size_t)(Q27_INTER / 16) * 4));
        CK(hipMalloc((void**)&S.dpa,  (size_t)Q27_VOCAB * 4));
        CK(hipMalloc((void**)&S.dpb,  (size_t)Q27_INTER * 4));
        // PAGEABLE, deliberately. A pinned buffer allocated on card 0's context was used as the
        // source of an H2D copy on cards 1-3 and the copy engine faulted reading it as if it were
        // device-accessible. A pageable H2D is always staged by the runtime and is correct on every
        // device; the payload is 10 KB per token, so the staging cost is noise.
        if (g == 0 || (g_sr && g == g_sr_head)) {   // Q27_LS_SR: the head card is the only draft card, so it owns the staging too
            // PINNED, not pageable. Round 29 switched this to std::malloc to work around a GPU fault
            // that turned out to be the NULL embed gather (0x54bf000 = 8678*10240), fixed separately.
            // The workaround was never needed and it is expensive: a pageable H2D/D2H pair per card
            // per draft step measured ~604 ms of host-side stall per step, turning a ~1 ms draft into
            // a 4.7x SLOWDOWN of the whole engine.
            CK(hipHostMalloc((void**)&S.host_emb, (size_t)Q27_HID * 2, hipHostMallocDefault));
            // The TP plan uploads NO embed_tokens to any card (all four report the same 4.1033 GiB
            // resident), so G->embed is null. q27_embed_gather(NULL, tok, ...) then read the table at
            // tok*10240 bytes -- and the GPU fault address 0x54bf000 is EXACTLY 8678*10240, with 8678
            // the first decoded token. The draft gathers its 10 KB row on the HOST instead: no table
            // is resident, the row is one memcpy, and the payload is noise against a token step.
            size_t eb = 0; long long esh[Q27_MAX_DIMS]; int edt = 0, end_ = 0;
            S.emb_tbl = q27_host_ptr(m, "model.language_model.embed_tokens.weight",
                                     &eb, &edt, &end_, esh);
            if (!S.emb_tbl) { std::fprintf(stderr, "Q27_MTP_EMB_MISSING\n"); return 1; }
        }
    }
    {
        // Handle-field A/B: the draft's handles are built by MY host quantizer, the lm_head's by the
        // ENGINE's own load-time converter. Same kernel consumes both, so any field that differs in
        // CONVENTION (not just in value) is the bug.
        MtpW& A = g_mtp[g_sr ? g_sr_head : 0].W;   // Q27_LS_SR: the draft lives on the head card only
        CK(hipSetDevice(g_sr ? g_sr_head : 0));
        std::printf("Q27_MTP_HANDLES\n");
        std::printf("  fc[0]  rows=%-6d K=%-6d ws2=%.6e in_scale=%.4f\n", A.fc[0].rows, A.fc[0].K, A.fc[0].ws2, A.fc[0].in_scale);
        std::printf("  fc[1]  rows=%-6d K=%-6d ws2=%.6e\n", A.fc[1].rows, A.fc[1].K, A.fc[1].ws2);
        std::printf("  gate   rows=%-6d K=%-6d ws2=%.6e in_scale=%.4f\n", A.gate.rows, A.gate.K, A.gate.ws2, A.gate.in_scale);
        std::printf("  up     rows=%-6d K=%-6d ws2=%.6e in_scale=%.4f\n", A.up.rows, A.up.K, A.up.ws2, A.up.in_scale);
        for (int j = 0; j < 4; ++j)
            std::printf("  down[%d] rows=%-6d K=%-6d ws2=%.6e in_scale=%.4f\n", j, A.down[j].rows, A.down[j].K, A.down[j].ws2, A.down[j].in_scale);
        std::printf("  lmhead rows=%-6d K=%-6d ws2=%.6e in_scale=%.4f   <-- engine-built reference\n",
                    g_lmref.rows, g_lmref.K, g_lmref.ws2, g_lmref.in_scale);
        std::printf("  fp8 q  rows=%-6d K=%-6d wscale=%.6e in_scale=%.4f\n", A.q.rows, A.q.K, A.q.wscale, A.q.in_scale);
        std::printf("  fp8 o  rows=%-6d K=%-6d wscale=%.6e in_scale=%.4f\n", A.o.rows, A.o.K, A.o.wscale, A.o.in_scale);
        // Reference magnitude: dequantize the ENGINE's lm_head with its own rule, so the draft's
        // raw BF16 magnitudes can be read against a tensor the working model actually uses.
        const q27_nvfp4_t* H = &g_lmref;
        if (H->w && H->gs) {
            const size_t WB = (size_t)H->rows * (size_t)(H->K >> 1), SB = (size_t)H->rows * (size_t)(H->K >> 4);
            std::vector<unsigned char> w(WB), g(SB);
            CK(hipMemcpy(w.data(), H->w, WB, hipMemcpyDeviceToHost));
            CK(hipMemcpy(g.data(), H->gs, SB, hipMemcpyDeviceToHost));
            double ss = 0; float amax = 0;
            for (int r = 0; r < H->rows; ++r)
                for (int k = 0; k < H->K; k += 2) {
                    const unsigned char by = w[(size_t)r * (H->K >> 1) + (k >> 1)];
                    const float s = q27_e4m3_val(g[(size_t)r * (H->K >> 4) + (k >> 4)]) * H->ws2;
                    const float v0 = q27_e2m1_val(by & 7) * s, v1 = q27_e2m1_val((by >> 4) & 7) * s;
                    ss += (double)v0 * v0 + (double)v1 * v1;
                    if (std::fabs(v0) > amax) amax = std::fabs(v0);
                    if (std::fabs(v1) > amax) amax = std::fabs(v1);
                }
            const size_t n2 = (size_t)H->rows * (size_t)H->K;
            std::printf("  engine lm_head dequantized: amax=%.6f rms=%.6f  (rows=%d K=%d ws2=%.3e)\n",
                        amax, std::sqrt(ss / n2), H->rows, H->K, H->ws2);
        }
        // THE DECISIVE COMPARISON: a trained MTP head must MIMIC the main model, so its weights
        // should be statistically similar to the main model's. Dequantize layer 3's own gate/up/down
        // shards by the same rule and put them beside the MTP's. A large mismatch means the MTP
        // tensors are being read from the wrong place -- plausible-looking data, right shapes, wrong
        // bytes -- which is exactly the failure that survives every magnitude-free test.
        const q27_layer_t* L3 = q27_layer_tp(m, 3, 0);
        const char* nm3[3] = {"L03.gate", "L03.up", "L03.down"};
        const q27_nvfp4_t* hs[3] = { L3 ? &L3->gate : nullptr, L3 ? &L3->up : nullptr,
                                     L3 ? &L3->down : nullptr };
        for (int k3 = 0; k3 < 3; ++k3) {
            const q27_nvfp4_t* Q = hs[k3];
            if (!Q || !Q->w || !Q->gs) { std::printf("  %-10s (not resident)\n", nm3[k3]); continue; }
            const size_t WB2 = (size_t)Q->rows * (size_t)(Q->K >> 1), SB2 = (size_t)Q->rows * (size_t)(Q->K >> 4);
            std::vector<unsigned char> w2(WB2), g2(SB2);
            CK(hipMemcpy(w2.data(), Q->w, WB2, hipMemcpyDeviceToHost));
            CK(hipMemcpy(g2.data(), Q->gs, SB2, hipMemcpyDeviceToHost));
            double s3 = 0; float a3 = 0;
            for (int r = 0; r < Q->rows; ++r)
                for (int kk = 0; kk < Q->K; kk += 2) {
                    const unsigned char by = w2[(size_t)r * (Q->K >> 1) + (kk >> 1)];
                    const float sc = q27_e4m3_val(g2[(size_t)r * (Q->K >> 4) + (kk >> 4)]) * Q->ws2;
                    const float v0 = q27_e2m1_val(by & 7) * sc, v1 = q27_e2m1_val((by >> 4) & 7) * sc;
                    s3 += (double)v0 * v0 + (double)v1 * v1;
                    if (std::fabs(v0) > a3) a3 = std::fabs(v0);
                    if (std::fabs(v1) > a3) a3 = std::fabs(v1);
                }
            const size_t n3 = (size_t)Q->rows * (size_t)Q->K;
            std::printf("  MAIN MODEL %-9s rows=%-6d K=%-6d amax=%.6f rms=%.6f\n",
                        nm3[k3], Q->rows, Q->K, a3, std::sqrt(s3 / n3));
        }
        std::fflush(stdout);
    }
    if (q27_env_flag("Q27_MTP_KERNBENCH", false)) {
        // Same kernel, same shape family, two handle origins: the ENGINE's lm_head (built by its own
        // converter, exercised every token) versus the DRAFT's handles (built by the host encoder).
        // If the engine's handle is fast and the draft's is not, the handles are the problem; if both
        // are slow, the kernel call is. Either way this is one number, not an inference.
        CK(hipSetDevice(0));
        auto bench = [&](const char* tag, const q27_nvfp4_t* h) {
            CK(hipStreamSynchronize(0));
            const double t0 = tp_now();
            for (int r = 0; r < 5; ++r) q27_proj_nvfp4(h, g_mtp[0].xq, g_mtp[0].xs, g_mtp[0].dpa, 0);
            CK(hipStreamSynchronize(0));
            const double dt = (tp_now() - t0) / 5.0;
            const double mb = (double)h->rows * (double)(h->K >> 1) / 1e6;
            std::printf("Q27_KB %-12s rows=%-6d K=%-6d  %8.3f ms/call  %7.1f MB  -> %8.1f GB/s\n",
                        tag, h->rows, h->K, dt, mb, mb / (dt > 0 ? dt : 1e-9));
            std::fflush(stdout);
        };
        // CARD 0's handle, not g_lmref (which is card ndev-1's shard): device pointers are not
        // valid in another device's page tables, and a kernel -- unlike the host-side dequantize
        // above -- faults instead of silently working.
        const q27_nvfp4_t lm0 = q27_globals(m, 0)->lm_head;
        bench("engine_lmhead", &lm0);
        bench("draft_gate",    &g_mtp[0].W.gate);
        bench("draft_fc0",     &g_mtp[0].W.fc[0]);
        bench("draft_down0",   &g_mtp[0].W.down[0]);
    }
    std::printf("Q27_MTP_INIT cards=%d %.1f MB/card resident  (replicated, no collectives)\n",
                ndev, g_mtp[0].W.bytes / 1e6);
    return 0;
}

// One draft step at FULL width on a single card. Every shape below is a shape the main model
// already runs, so no kernel is being asked for a geometry it has not been measured on.
// Order matters: it READS S.cat[0:5120] (norm of the PREVIOUS position's hidden state) and
// WRITES S.cat[0:5120] for the next one, so the caller must stash AFTER calling this.
static void mtp_dump_rms(const char* tag, const unsigned short* buf, int n) {
    std::vector<float> v = q27_d2h_bf16(buf, n);
    double ss = 0, amax = 0; int nz = 0;
    for (float f : v) { ss += (double)f * f; if (std::fabs(f) > amax) amax = std::fabs(f); if (f != 0.f) ++nz; }
    std::printf("Q27_MTP_STAGE %-12s rms=%.5f amax=%.4f nonzero=%d/%d\n", tag, std::sqrt(ss / n), amax, nz, n);
    std::fflush(stdout);
}

static void mtp_dump_rms_f(const char* tag, const float* buf, int n) {
    std::vector<float> v = q27_d2h_f32(buf, n);
    double ss = 0, amax = 0; int nz = 0;
    for (float f : v) { ss += (double)f * f; if (std::fabs(f) > amax) amax = std::fabs(f); if (f != 0.f) ++nz; }
    std::printf("Q27_MTP_STAGE %-12s rms=%.5f amax=%.4f nonzero=%d/%d\n", tag, std::sqrt(ss / n), amax, nz, n);
    std::fflush(stdout);
}

static void mtp_draft_step(Dev& d, MtpState& S, q27_globals_t* G, int dpos, int stash, const unsigned short* hprev, int head) {
    hipStream_t s = d.stream;
    const float IS = g_mtp_is;
    q27_rmsnorm(S.emb, S.W.pre_emb, S.cat + mtp_emb_off(), Q27_HID, g_mtp_po, s);
    // The stash MUST follow the quantize: it overwrites cat[0:5120], which the fc input above just
    // consumed. Both are on the same stream, so the ordering is the kernel queue's, not a guess.
    if (stash) q27_rmsnorm(hprev ? hprev : d.hidden_slots, S.W.pre_hid, S.cat + mtp_hid_off(), Q27_HID, g_mtp_po, s);
    // fc as 2 x K=5120: the two concat halves ARE the two slices, and each is permuted on its own
    // so every slice is a self-contained K=5120 problem of exactly the shape production runs.
    q27_quant_perm(S.cat,           S.xq,                     S.xs,                    Q27_HID, IS, s);
    q27_quant_perm(S.cat + Q27_HID, S.xq + Q27_HID,           S.xs + Q27_HID / 16,     Q27_HID, IS, s);
    q27_proj_nvfp4(&S.W.fc[0], S.xq,           S.xs,          S.dpa,               s);
    q27_proj_nvfp4(&S.W.fc[1], S.xq + Q27_HID, S.xs + Q27_HID / 16, S.dpa + Q27_HID, s);
    q27_f2bf_vec(S.dpa, S.nrm, Q27_HID, s);
    q27_f2bf_vec(S.dpa + Q27_HID, S.hid2, Q27_HID, s);
    q27_add_inplace_bf16(S.nrm, S.hid2, Q27_HID, s);              // S.nrm = residual stream
    const int dbgs = (g_mtp_dbg && d.id == 0 && !g_mtp_dbg_used);
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms("1_fc_out", S.nrm, Q27_HID); }
    if (q27_env_flag("Q27_MTP_NOFUSE", false) ||
        !q27_rmsnorm_quant_fp8(S.nrm, S.W.in_norm, S.hid, S.xq, S.xs, Q27_HID, g_mtp_po, IS, nullptr, 0, s)) {
        q27_rmsnorm(S.nrm, S.W.in_norm, S.hid, Q27_HID, g_mtp_po, s);
        q27_quant_fp8(S.hid, S.xq, S.xs, Q27_HID, IS, s);
    }
    q27_proj_fp8(&S.W.q, S.xq, S.xs, S.dpa, s); q27_f2bf_vec(S.dpa, S.qkva, Q27_QROWS, s);
    q27_proj_fp8(&S.W.k, S.xq, S.xs, S.dpa, s); q27_f2bf_vec(S.dpa, S.qkva + Q27_QROWS, Q27_KVROWS, s);
    q27_proj_fp8(&S.W.v, S.xq, S.xs, S.dpa, s); q27_f2bf_vec(S.dpa, S.qkva + Q27_QROWS + Q27_KVROWS, Q27_KVROWS, s);
    q27_attn_prep(S.qkva, S.W.q_norm, S.W.k_norm, d.mtp_kc, d.mtp_ks, d.mtp_vc, d.mtp_vs, S.qh, dpos, Q27_KVROWS, 0, s);
    q27_attn_decode(S.qh, d.mtp_kc, d.mtp_ks, d.mtp_vc, d.mtp_vs, S.qkva, S.mix6, dpos, Q27_KVROWS, 0, s);
    q27_quant_fp8(S.mix6, S.xq, S.xs, Q27_OROWS, IS, s);
    q27_proj_fp8(&S.W.o, S.xq, S.xs, S.dpa, s);
    q27_f2bf_vec(S.dpa, S.hid, Q27_HID, s);
    q27_add_inplace_bf16(S.nrm, S.hid, Q27_HID, s);
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms("2_post_attn", S.nrm, Q27_HID); }
    if (q27_env_flag("Q27_MTP_NOFUSE", false) ||
        !q27_rmsnorm_quant_perm(S.nrm, S.W.post_norm, S.hid, S.xq, S.xs, Q27_HID, g_mtp_po, IS, nullptr, 0, s)) {
        q27_rmsnorm(S.nrm, S.W.post_norm, S.hid, Q27_HID, g_mtp_po, s);
        q27_quant_perm(S.hid, S.xq, S.xs, Q27_HID, IS, s);
    }
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms("2b_mlp_in", S.hid, Q27_HID); }
    q27_proj_nvfp4(&S.W.gate, S.xq, S.xs, S.dpa, s);
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms_f("2c_gate", S.dpa, Q27_INTER); }
    q27_proj_nvfp4(&S.W.up,   S.xq, S.xs, S.dpb, s);
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms_f("2d_up", S.dpb, Q27_INTER); }
    q27_swiglu(S.dpa, S.dpb, S.act, Q27_INTER, s);
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms("2e_swiglu", S.act, Q27_INTER); }
    // down as 4 x K=4352 (17408/4, a multiple of 16 so the per-16 scales stay slice-aligned)
    for (int j = 0; j < 4; ++j)
        q27_quant_perm(S.act + (size_t)j * 4352, S.xq + (size_t)j * 4352, S.xs + (size_t)j * 272,
                       4352, IS, s);
    for (int j = 0; j < 4; ++j)
        q27_proj_nvfp4(&S.W.down[j], S.xq + (size_t)j * 4352, S.xs + (size_t)j * 272,
                       S.dpa + (size_t)j * Q27_HID, s);
    q27_f2bf_vec(S.dpa, S.hid, Q27_HID, s);
    for (int j = 1; j < 4; ++j) {
        q27_f2bf_vec(S.dpa + (size_t)j * Q27_HID, S.hid2, Q27_HID, s);
        q27_add_inplace_bf16(S.hid, S.hid2, Q27_HID, s);
    }
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms("2f_down_sum", S.hid, Q27_HID); }
    q27_add_inplace_bf16(S.nrm, S.hid, Q27_HID, s);
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms("3_post_mlp", S.nrm, Q27_HID); }
    if (!head) return;   // catch-up step: only the layer's KV entry mattered, the head is not needed
    q27_rmsnorm(S.nrm, S.W.mtp_norm, S.hid, Q27_HID, 1, s);
    if (dbgs) { hipStreamSynchronize(s); mtp_dump_rms("4_mtp_norm", S.hid, Q27_HID); g_mtp_dbg_used = 1; }
    q27_quant_perm(S.hid, S.xq, S.xs, Q27_HID, G->lm_head.in_scale, s);
    if (!exl3_head(d, G, S.hid, S.dpa, 1, s))
                        q27_proj_nvfp4(&G->lm_head, S.xq, S.xs, S.dpa, s);             // this card's logit shard
}

static void mtp_gold(q27_model_t* m, std::vector<Dev>& D, int ndev, const char* hfile, unsigned tok) {
    std::vector<unsigned short> h((size_t)Q27_HID);
    FILE* f = std::fopen(hfile, "rb");
    if (!f) { std::printf("Q27_MTP_GOLD cannot open %s\n", hfile); return; }
    const size_t got = std::fread(h.data(), 2, Q27_HID, f);
    std::fclose(f);
    if (got != Q27_HID) { std::printf("Q27_MTP_GOLD short read %zu\n", got); return; }
    unsigned best = 0; float bv = -1e30f;
    for (int g = 0; g < ndev; ++g) {
        Dev& d = D[g];
        MtpState& S = g_mtp[g];
        const q27_globals_t* G = q27_globals(m, d.id);
        CK(hipSetDevice(d.id));
        CK(hipMemcpy(d.hidden_slots, h.data(), (size_t)Q27_HID * 2, hipMemcpyHostToDevice));
        // emb(tok) straight from the host table -- no broadcast, no machinery
        std::memcpy(g_mtp[0].host_emb, (const char*)g_mtp[0].emb_tbl + (size_t)tok * Q27_HID * 2,
                    (size_t)Q27_HID * 2);
        CK(hipMemcpy(S.emb, g_mtp[0].host_emb, (size_t)Q27_HID * 2, hipMemcpyHostToDevice));
        if (g_gold_mode == 1) {
            // TAIL-ONLY: h -> mtp.norm -> lm_head -> argmax. Fed the oracle's final hidden this must
            // reproduce the oracle's OWN next token, so it isolates the tail from the layer.
            q27_rmsnorm(d.hidden_slots, S.W.mtp_norm, S.hid, Q27_HID, 1, d.stream);
            q27_quant_perm(S.hid, S.xq, S.xs, Q27_HID, G->lm_head.in_scale, d.stream);
            if (!exl3_head(d, G, S.hid, S.dpa, 1, d.stream))
                        q27_proj_nvfp4(&G->lm_head, S.xq, S.xs, S.dpa, d.stream);
        } else {
            const double g0 = tp_now();
            mtp_draft_step(d, S, (q27_globals_t*)G, 0, 1);       // dpos=0: attends to itself alone
            CK(hipStreamSynchronize(d.stream));
            std::printf("Q27_MTP_ISOLATED card %d draft_step=%.3f ms\n", d.id, (tp_now() - g0) * 1000);
            std::fflush(stdout);
            // Where does my layer output land relative to the oracle's NEXT hidden state? Both are
            // 'the hidden that predicts t_2', so a correct layer must land in the same basin.
            if (const char* rp = std::getenv("Q27_MTP_GOLD_REF")) {
                std::vector<unsigned short> ref((size_t)Q27_HID);
                FILE* rf = std::fopen(rp, "rb");
                if (rf) {
                    if (std::fread(ref.data(), 2, Q27_HID, rf) == Q27_HID) {
                        std::vector<float> mine = q27_d2h_bf16(S.nrm, Q27_HID);
                        double dot = 0, na = 0, nb = 0, d2 = 0;
                        for (int i2 = 0; i2 < Q27_HID; ++i2) {
                            const double a = mine[i2], b = q27_bf16_to_f(ref[i2]);
                            dot += a * b; na += a * a; nb += b * b; d2 += (a - b) * (a - b);
                        }
                        std::printf("Q27_MTP_LAYERCMP cos=%.4f relL2=%.4f |mine|=%.4f |ref|=%.4f\n",
                                    dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30),
                                    std::sqrt(d2 / (nb + 1e-30)), std::sqrt(na / Q27_HID), std::sqrt(nb / Q27_HID));
                    }
                    std::fclose(rf);
                }
            }
        }
        q27_argmax_val(S.dpa, d.dtok, d.dval, G->lm_head.rows, d.stream);
        CK(hipStreamSynchronize(d.stream));
        unsigned li; float lv;
        CK(hipMemcpy(&li, d.dtok, 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(&lv, d.dval, 4, hipMemcpyDeviceToHost));
        const unsigned gi = (unsigned)((long long)q27_tp_shard(g)->voc_off + (long long)li);
        std::printf("Q27_MTP_GOLD card %d local=%u val=%.4f global=%u\n", g, li, lv, gi);
        if (lv > bv) { bv = lv; best = gi; }
    }
    std::printf("Q27_MTP_GOLD h=%s tok=%u -> draft=%u (val=%.4f)\n", hfile, tok, best, bv);
    std::fflush(stdout);
}

// ---- CRASH BACKTRACE. 2026-09-18. ----------------------------------------------------------
// The layer-split sweep dies with a fault INSIDE libamdhip64 (+0x2E6218), i.e. HIP dereferencing
// something the engine handed it. dmesg gives the faulting ip but not the CALLER, and the caller is
// the only thing that identifies which handle was bad. -rdynamic is already on, so backtrace_symbols
// resolves engine frames by name. Async-signal-safety: backtrace_symbols_fd writes straight to the
// fd and allocates nothing, which is the one form that is legal here.
static void q27_crash_bt(int sig, siginfo_t* si, void*) {
    const char* nm = (sig == SIGSEGV) ? "SIGSEGV" : (sig == SIGBUS) ? "SIGBUS" : "SIGFPE";
    char hdr[160];
    int n = snprintf(hdr, sizeof hdr, "\nQ27_CRASH %s at addr %p -- backtrace (engine frames are named):\n",
                     nm, si ? si->si_addr : nullptr);
    ssize_t w = write(2, hdr, (size_t)(n > 0 ? n : 0)); (void)w;
    void* fr[64];
    int k = backtrace(fr, 64);
    backtrace_symbols_fd(fr, k, 2);
    _exit(139);
}
static void q27_install_crash_bt() {
    if (!q27_env_flag("Q27_CRASH_BT", true)) return;
    struct sigaction sa; std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = q27_crash_bt;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS,  &sa, nullptr);
}
int main(int argc, char** argv) {
    if (const char* error = q27_candidate_flag_error()) {
        std::fprintf(stderr, "FATAL Q27 candidate flags: %s\n", error);
        return 2;
    }
    q27_pfblas_log_config();
    g_t0_wall = tp_now();
    q27_install_crash_bt();
    MS("T0_start");
    if (q27_env_flag("Q27_FP8_CENSUS", false)) { CK(hipSetDevice(0)); q27_fp8_census(); return 0; }
    if (q27_env_flag("Q27_MK_BENCH", false)) { CK(hipSetDevice(0)); q27_mk_bench(); return 0; }
    if (q27_env_flag("Q27_SYNC_TAX", false)) { CK(hipSetDevice(0)); q27_sync_tax(); return 0; }
    if (q27_env_flag("Q27_MLP_BENCH", false)) { CK(hipSetDevice(0)); q27_mlp_bench(); return 0; }
    if (q27_env_flag("Q27_MLPB_BENCH", false)) { CK(hipSetDevice(0)); q27_mlpb_bench(); return 0; }
    if (q27_env_flag("Q27_WIDE_BENCH", false)) { CK(hipSetDevice(0)); q27_wide_bench(); return 0; }
    if (q27_env_flag("Q27_BW_PROBE", false)) { CK(hipSetDevice(0)); q27_bw_probe(); return 0; }
    if (argc < 2) {
        std::fprintf(stderr, "usage: q27_gen <model_dir> [ndev=4] [ctx=4096] [max_new=24] [oracle_dir|\"\"] [token ids...]\n"
                             "       Q27_TEXT=1 q27_gen <model_dir> 4 <ctx> <max_new> \"\" \"user text\"   (or ./q27_chat)\n");
        return 2;
    }
    const char* dir  = argv[1];
    const int   ndev = (argc>2)? atoi(argv[2]) : 4;
    const int   ctx  = (argc>3)? atoi(argv[3]) : 4096;
    const int   maxn = (argc>4)? atoi(argv[4]) : 24;
    // Q27_MLP_WSHARE must be resolved before the first dev_alloc sizes an MLP scratch buffer and
    // before tp_plan/q27_fp8_mlp_load slice a tensor, so it is parsed here and nowhere else.
    mlp_wshare_init(ndev);
    const int   slots = (getenv("Q27_SLOTS") ? atoi(getenv("Q27_SLOTS")) : 1);
    const char* orc  = (argc>5)? argv[5] : nullptr;   // oracle vector dir -> run the ladder
    if (orc && (orc[0]==0 || !strcmp(orc,"none") || !strcmp(orc,"-"))) orc = nullptr;
    std::vector<unsigned> prompt;
    g_text  = q27_env_flag("Q27_TEXT", false);
    g_think = q27_env_flag("Q27_THINK", false);
    // Read Q27_SERVE from the environment directly rather than from g_serve: the global is
    // assigned in the env-init pass, and this runs in main() where that ordering is not
    // guaranteed. On by default in serve mode; a reader that ignores Q27_TOK lines is unaffected.
    g_stream_ids = q27_env_flag("Q27_SERVE", false) && !g_text
                   && q27_env_flag("Q27_SERVE_STREAM", true);
    if (const char* sy = std::getenv("Q27_SYSTEM")) g_system = sy;
    if (g_text) {
        std::string text;
        for (int i = 6; i < argc; ++i) { if (i > 6) text += ' '; text += argv[i]; }
        const double t_tok0 = tp_now();
        try { g_tok = new hf::Tokenizer(std::filesystem::path(dir) / "tokenizer.json"); }
        catch (const std::exception& e) { std::fprintf(stderr, "Q27_TOKENIZER_FAIL %s\n", e.what()); return 1; }
        prompt = q27_encode(q27::chat_prompt(text, g_system, g_think));
        g_first_text = text;
        // Text goes to the real stdout; every diagnostic line (Q27_*, markers) goes to stderr.
        { const int fd = dup(1); if (fd >= 0) { g_text_out = fdopen(fd, "w"); std::fflush(stdout); dup2(2, 1); } }
        std::printf("Q27_TOKENIZER_MS %.0f\nQ27_TEXT_PROMPT tokens=%zu think=%d\n", tp_now() - t_tok0, prompt.size(), (int)g_think);
        std::fflush(stdout);
    } else {
        for (int i=6;i<argc;++i) prompt.push_back((unsigned)strtoul(argv[i],nullptr,10));
    }
    if (prompt.empty()) {  // the BASELINE.md prompt: "Define probability in one sentence."
        unsigned p[] = {248045,846,198,34167,18356,303,799,11316,13,248046,198,
                        248045,74455,198,248068,271,248069,271};
        prompt.assign(p, p+sizeof(p)/sizeof(p[0]));
    }

    if (const char* p = std::getenv("Q27_DFLASH2_PROBE")) {
        q27_df2_t W{}; char derr[256] = {0};
        if (q27_df2_open(p, &W, derr, sizeof derr)) { std::fprintf(stderr, "Q27_DFLASH2_OPEN_FAIL %s\n", derr); return 1; }
        const size_t bd = W.bytes_device;
        for (int g = 0; g < ndev; ++g)
            if (q27_df2_upload_sharded(&W, g, ndev, derr, sizeof derr)) { std::fprintf(stderr, "Q27_DFLASH2_UPLOAD_FAIL %s\n", derr); return 1; }
        std::printf("Q27_DFLASH2_PROBE_OK full-replicated=%.3f GiB, TP4-sharded uploads above (x %d)\n", (double)bd / 1073741824.0, ndev);
        q27_df2_free(&W);
        return 0;
    }
    char err[512] = {0};
    q27_model_t* m = q27_open(dir, err, sizeof err);
    if (!m) { std::fprintf(stderr,"Q27_OPEN_FAIL %s\n", err); return 1; }
    std::printf("Q27_CATALOG tensors=%d payload=%lld\n", q27_tensor_count(m), q27_payload_bytes(m));
    if (q27_env_flag("Q27_MTP_PROBE", false)) return q27_mtp_probe(m);
    if (q27_env_flag("Q27_MTP_QUANT", false)) return q27_mtp_quant_probe(m);
    if (q27_env_flag("Q27_MTP_UPLOAD", false)) return q27_mtp_upload_probe(m);
    if (q27_env_flag("Q27_MTP_WMAG", false)) return q27_mtp_wmag_probe(m);
    MS("T1_open_done");

    // Q27_TP_DEVMAP="0,0,1,2" -- LOGICAL SHARD -> PHYSICAL DEVICE. 2026-09-18.
    //
    // Three cards cannot be an ndev=3 TP group: Q27_NKV=4 and Q27_HID=5120 do not divide by 3, and
    // that gate is arithmetic, not caution. But the SHARD COUNT and the CARD COUNT do not have to be
    // the same number. This keeps the 4-way shard geometry every kernel was written for and simply
    // places two of those shards on one physical card -- the 32 GiB one, which has the capacity for
    // a double share where a 16 does not.
    //
    // "0,0,1,2" at ndev=4 => shards 0 and 1 both live on physical device 0; shards 2 and 3 on
    // devices 1 and 2. Three cards, four shards, no geometry change.
    //
    // COST, stated plainly: the card holding two shards does two shards of work, so it sets the
    // barrier round. This BUYS capacity and SPENDS balance. It is the right trade only when the
    // doubled card is also the fastest/biggest one.
    //
    // Collectives are host-staged (coll_p2p=0 on the shipped arm), so two shards sharing a device is
    // just two stream sets on one GPU -- no peer mapping is involved. Do NOT combine with P2P.
    std::vector<int> devs(ndev); for (int i=0;i<ndev;++i) devs[i]=i;
    if (const char* dm = getenv("Q27_TP_DEVMAP")) {
        std::vector<int> m; int v = 0, d = 0;
        for (const char* p = dm; ; ++p) {
            if (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); d = 1; }
            else { if (d) { m.push_back(v); v = 0; d = 0; } if (!*p) break; }
        }
        if ((int)m.size() != ndev) {
            std::fprintf(stderr, "FATAL Q27_TP_DEVMAP: %zu entries for ndev=%d\n", m.size(), ndev);
            return 1;
        }
        int ndevice = 0; CK(hipGetDeviceCount(&ndevice));
        for (int i = 0; i < ndev; ++i) {
            if (m[i] < 0 || m[i] >= ndevice) {
                std::fprintf(stderr, "FATAL Q27_TP_DEVMAP: shard %d -> device %d, only %d visible\n",
                             i, m[i], ndevice);
                return 1;
            }
            devs[i] = m[i];
        }
        std::fprintf(stderr, "Q27_TP_DEVMAP: %d shards on %d physical device(s):", ndev, ndevice);
        for (int i = 0; i < ndev; ++i) std::fprintf(stderr, " s%d->d%d", i, devs[i]);
        std::fprintf(stderr, "\n"); std::fflush(stderr);
    }
    // Q27_TP=1 selects the C1 tensor-parallel residency: every layer on every card, one shard
    // each. It is a different UPLOAD, so it is chosen here; the serial and pipeline paths below
    // are untouched.
    const int tp = q27_env_flag("Q27_TP", true) ? 1 : 0;   // shipped default: tensor-parallel residency
    // Q27_ALLOC_PAD=<MB>: reserve that much device memory on every card BEFORE the weight upload and keep it
    // (placement probe: the rung-3 binary's M=1 weight-streaming kernels ran ~9% slower with identical code;
    // sweeping the pad shows whether where the arenas land decides that). Default 0 = nothing allocated.
    if (const char* pe = getenv("Q27_ALLOC_PAD")) {
        const size_t mb = (size_t)strtoull(pe, nullptr, 10);
        for (int i = 0; mb && i < ndev; ++i) {
            void* pad = nullptr;
            if (hipSetDevice(devs[i]) == hipSuccess && hipMalloc(&pad, mb << 20) == hipSuccess)
                std::fprintf(stderr, "Q27_ALLOC_PAD dev %d: %zu MB at %p\n", devs[i], mb, pad);
        }
        hipSetDevice(devs[0]);
    }
    q27_memory_receipt("before_weights");
    const double t_up0 = tp_now();
    if (tp) {
        if (q27_env_flag("Q27_LAYER_SPLIT", true)) {
            // dual residency: the TP shards stay for the untouched decode path; the layer-split
            // FULL-layer layout rides alongside for the prefill sweep. Q27_LS_NOTP=1 skips the
            // TP residency entirely (prefill-only: frees ~8 GB/card for the w8 preshuffle).
            const bool ls_sr   = q27_env_flag("Q27_LS_SR", false);      // single residency: no TP upload at all (src/q27_sr_main.inc)
            const bool ls_notp = q27_env_flag("Q27_LS_NOTP", false) || ls_sr;
            if (ls_sr) q27_ls_mlp_i8k_set(0);   // no persistent int8 MLP mirrors: the prefill ring builds them per layer from the FP8 planes
            // Q27_LS_MLP_I8K=-1: window-aware residency. The prefill slots cost 30,720 B per
            // position on every card and scale with the prompt (g_pf_cap), so a long window buys
            // itself out of the fast MLP. One int8 MLP layer costs the same as ~3,960 positions, so
            // spend what this window leaves rather than a number tuned for one prompt length.
            {
                // A PER-CARD LIST ("16,10,10,10") is an explicit budget, so it bypasses the auto
                // path entirely -- and it must be tested BEFORE q27_env_int, which fatals on a
                // non-integer ("FATAL Q27_LS_MLP_I8K=... is not an integer"). q27_load's
                // ls_mlp_i8k(card) owns the parsing; here we only need to not interfere.
                const char* i8k_env = getenv("Q27_LS_MLP_I8K");
                const bool i8k_list = i8k_env && *i8k_env && strchr(i8k_env, ',');
                const int req = i8k_list ? 0 : q27_env_int("Q27_LS_MLP_I8K", 0);
                if (req < 0) {
                    // Mirror run_tp's slot sizing EXACTLY -- this is main(), where g_pf_cap is still
                    // the compile-time default, so the window has to be derived from the prompt that
                    // was actually handed to this engine.
                    int win = Q27_PF_CAP;
                    if ((int)prompt.size() > win) win = (int)prompt.size();
                    if (q27_env_flag("Q27_SERVE", false)) {
                        const int floor_ = q27_env_int("Q27_SERVE_SLOTS",
                                                       q27_env_int("Q27_SERVE_MAXPROMPT", 8192));
                        int cap = (int)prompt.size() + 256;
                        if (cap < floor_) cap = floor_;
                        if (cap > win) win = cap;
                    }
                    const int bonus = q27_env_int("Q27_V4_WIN_BONUS", Q27_V4_WIN_BONUS); const int over = win > (9216 + bonus) ? (win - (9216 + bonus)) : 0;
                    // Per-layer cost with Q27_DEC_MLP_I8 on: the int8 mirror's 117 MB net minus the
                    // 36 MB NVFP4 MLP shard the decode no longer needs = ~81 MB, i.e. ~2640 positions
                    // of slot. Measured: at the 9,216 floor all 16 layers fit with 0.61 GiB free.
                    const bool dec8 = q27_env_flag("Q27_DEC_MLP_I8", false) && !q27_env_flag("Q27_DEC_MLP_NV", false);
                    const int per = dec8 ? 2640 : 3960;
                    int k = (dec8 ? 16 : 13) + q27_env_int("Q27_V4_KBONUS", Q27_V4_KBONUS) - (over + per - 1) / per;   // V4 credit: 0.985 GiB|card / 117 MB per mirror = ~8 layers; take 3
                    if (k < 0) k = 0; if (k > Q27_LAYERS) k = Q27_LAYERS;
                    size_t fr = 0, tt = 0; hipSetDevice(devs[0]); hipMemGetInfo(&fr, &tt);
                    q27_ls_mlp_i8k_set(k);
                    std::fprintf(stderr, "Q27_LS_MLP_I8K auto: prompt=%zu window=%d -> %d of %d owned "
                                         "layers on int8/Tensile (free before layer-split %.2f GiB)\n",
                                 prompt.size(), win, k, Q27_LAYERS / ndev, (double)fr / 1073741824.0);
                    std::fflush(stderr);
                }
            }
            if (!ls_notp && q27_upload_tp(m, devs.data(), ndev, err, sizeof err) != Q27_OK) {
                std::fprintf(stderr,"Q27_UPLOAD_FAIL %s\n", err); return 1;
            }
            if (q27_upload_ls(m, devs.data(), ndev, err, sizeof err) != Q27_OK) {
                std::fprintf(stderr,"Q27_UPLOAD_FAIL %s\n", err); return 1;
            }
            if (ls_sr) {   // Q27_LS_SR: the MLP is the block-scaled FP8 checkpoint, full width on each layer's owner
                const char* f8p = getenv("Q27_FP8_MLP");
                if (!f8p || !*f8p) { std::fprintf(stderr, "FATAL Q27_LS_SR needs Q27_FP8_MLP=<dir of the FP8 checkpoint>\n"); return 1; }
                const int nf8 = q27_fp8_mlp_load_sr(m, ndev, f8p);
                std::printf("Q27_LS_SR: %d FP8 MLP projections resident on their owner cards\n", nf8);
            }
            if (!ls_sr) q27_resolve_tp_ls_mlp(m, ndev);   // Q27_ALIAS_MLP: TP MLP handles -> layer-split rows
            // Q27_E4M3_FAST: certify the decode gate/up scale planes for the 2-instruction group
            // decode. Default OFF so the A arm is the shipped path untouched; uncertified planes
            // fall back regardless of the flag. Q27_DOWN_FS also triggers the pass (it certifies
            // down_proj planes inside the same walk when set).
            if (q27_env_flag("Q27_E4M3_FAST", false) || q27_env_flag("Q27_DOWN_FS", false)) q27_nvfp4_validate_gu(m, ndev);
            // Q27_MXFP4_MLP: swap the decode gate/up shards to AMD Quark OCP-MXFP4 in place. Runs
            // AFTER the E4M3 validator on purpose -- it clears gs_fast on anything it converts.
            if (const char* mxp = getenv("Q27_MXFP4_MLP")) q27_mxfp4_sideload_mlp(m, ndev, mxp);
            // Q27_FP8_MLP: block-scaled FP8 MLP resident alongside; decode prefers it when present.
            if (const char* f8p = getenv("Q27_FP8_MLP")) { if (!ls_sr) q27_fp8_mlp_load(m, ndev, f8p); }
            if (!ls_sr && q27_env_flag("Q27_LS_FP8_PREFILL", false)) {
                const char* f8p = getenv("Q27_FP8_MLP");
                if (!f8p || q27_fp8_ls_prefill_mirrors(m, ndev, f8p) != Q27_LAYERS * 3) {
                    std::fprintf(stderr, "FATAL Q27_LS_FP8_PREFILL: incomplete owner mirrors\n"); return 1;
                }
            }
            // Q27_EXL3_DIR: trellis weights (4bpw / 6bpw) attach to each card's OWNED layers,
            // full width. Every projection dispatcher prefers a live EXL3 handle, so this
            // takes over the layer without the incumbent residency being disturbed -- which is
            // what lets the 8-bit arm stay the fallback authority while this is brought up.
            if (const char* exd = getenv("Q27_EXL3_DIR"))
                q27_exl3_sideload(m, devs.data(), ndev, exd);
        } else {
            if (q27_upload_tp(m, devs.data(), ndev, err, sizeof err) != Q27_OK) {
                std::fprintf(stderr,"Q27_UPLOAD_FAIL %s\n", err); return 1;
            }
            // Q27_TPSR (2026-09-17 night): the block-scaled FP8 MLP -- gate/up ROW shards and, with
            // Q27_FP8_MLP_DOWN=1, the down K-SLICE -- rides on the TP handles of the ONLY residency.
            // Until now this loader was reachable from the layer-split branch alone, so a
            // Q27_LAYER_SPLIT=0 boot under Q27_FP8_DROP_NV_GU dropped the NVFP4 gate/up and loaded
            // nothing in their place: that null plane (q27_proj_nvfp4_gu_b13 on w=NULL, first chunk
            // of layer 0) is the batched path's small-address fault after T6_embed_done.
            if (const char* f8p = getenv("Q27_FP8_MLP")) q27_fp8_mlp_load(m, ndev, f8p);
            // Q27_EXL3_DIR on the TP-only residency (Q27_LAYER_SPLIT=0). Refuses
            // unless ndev==1, where a "shard" is the full projection.
            if (const char* exd = getenv("Q27_EXL3_DIR"))
                q27_exl3_sideload(m, devs.data(), ndev, exd);
        }
    } else if (q27_upload_split(m, devs.data(), ndev, /*want_embed=*/1, err, sizeof err) != Q27_OK) {
        std::fprintf(stderr,"Q27_UPLOAD_FAIL %s\n", err); return 1;
    }
    q27_storage_finish();
    { char buf[4096]; q27_report(m, buf, sizeof buf); std::fputs(buf, stdout); }
    std::printf("Q27_UPLOAD_MS %.0f\n", tp_now() - t_up0);
    MS("T2_weights_resident");
    if (tp) {
        std::printf("Q27_TP_READY cards=%d ctx=%d\n", ndev, ctx);
    g_df2_cond = q27_env_flag("Q27_DFLASH2_COND", false) ? 1 : 0;
    g_df2_fwd  = q27_env_flag("Q27_DFLASH2_FWD", false) ? 1 : 0;
    g_df2_sel  = q27_env_flag("Q27_DFLASH2_SEL", false) ? 1 : 0;
    g_df2_draft = q27_env_flag("Q27_DFLASH2", false) ? 1 : 0;
    if (g_df2_draft) g_df2_nprompt = (int)prompt.size();
    if (g_df2_draft) {   // the integrated path: K=7 / CH=8 block speculation
        g_spec_k = 7;
        g_nr_ch = 8;
    }
    if (g_df2_cond || g_df2_fwd || g_df2_sel || g_df2_draft) {   // DFlash2 diagnostics + integrated path: load + shard + ctx per card
        const char* df2path = getenv("Q27_DFLASH2_CKPT");   // drafter checkpoint; override for any layout
        if (!df2path || !*df2path) df2path = "./models/dflash2/model.safetensors";
        q27_df2_t W{}; char derr[256] = {0};
        if (q27_df2_open(df2path, &W, derr, sizeof derr)) { std::fprintf(stderr, "Q27_DFLASH2_OPEN_FAIL %s\n", derr); return 1; }
        for (int g = 0; g < ndev; ++g) {
            if (q27_df2_upload_sharded(&W, g, ndev, derr, sizeof derr)) { std::fprintf(stderr, "Q27_DFLASH2_UPLOAD_FAIL %s\n", derr); return 1; }
            std::memcpy(&g_df2c[g].W, &W, sizeof W);   // per-card copy of the geometry/pointers (arena per card)
            if (q27_df2_ctx_alloc(&g_df2c[g], g, derr, sizeof derr)) { std::fprintf(stderr, "Q27_DFLASH2_CTX_FAIL %s\n", derr); return 1; }
            CK(hipSetDevice(g));
            { const double tp0 = tp_now(); q27_df2_prime(&g_df2c[g], 0);
              if (g == 0) std::printf("Q27_DFLASH2 primed cond_inject GEMM shapes rows=1..8 in %.1f ms/card\n", tp_now() - tp0); }
        }
    }
        const int rc = run_tp(m, ndev, ctx, maxn, prompt, orc);
        MS("T10_run_done");
        q27_close(m);
        MS("T11_exit");
        return rc;
    }

    const int per = Q27_LAYERS / ndev;
    std::vector<Dev> D(ndev);
    for (int g=0; g<ndev; ++g) {
        int nf=0, ng=0;
        for (int L=g*per; L<(g+1)*per; ++L) (Q27_IS_FULL(L)? nf : ng)++;
        dev_alloc(D[g], devs[g], ng, nf, ctx, slots);
    }
    std::printf("Q27_READY devices=%d layers/device=%d ctx=%d slots=%d\n", ndev, per, ctx, slots);

    // sub-layer bisect selection from the environment, so it costs nothing when off
    if (const char* e = std::getenv("Q27_PROFILE")) g_profile = atoi(e);
    if (const char* e = std::getenv("Q27_SUB_LAYER")) { g_orc = orc; g_sub_layer = atoi(e); }
    if (const char* e = std::getenv("Q27_SUB_POS"))   { g_sub_pos = atoi(e); } else g_sub_pos = 0;

    double t_prefill = 0, t_decode = 0; int n_decode = 0;
    auto ms_now = [](){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                        return t.tv_sec*1e3 + t.tv_nsec/1e6; };
    std::vector<unsigned> out;
    unsigned tok = prompt[0];
    size_t pi = 1;
    bool ladder = (orc != nullptr);
    int fails = 0;

    // ================= THREADED PIPELINE =================
    // One host thread per GPU, 2-deep mailboxes between stages. This is the mechanism the
    // single-threaded schedule was missing: hipSetDevice is per-thread, so four threads can have
    // four devices current at once and their kernels actually overlap.
    // Modelled on qf_hip4_pipeline.cpp in the estate (2-deep mailboxes, one pthread per GPU),
    // which production there never called.
    // ACCEPTANCE TEST, before any tok/s claim: the timeline below must show cards executing
    // concurrently in wall-clock. If it does not, nothing else here matters.
    if (q27_env_flag("Q27_THREADED", false)) {
        struct Msg { int slot; int pos; bool stop; };
        struct Mb {
            std::mutex m; std::condition_variable cvd, cvs; std::deque<Msg> q;
            size_t cap = 2; bool stopped = false;
            void push(Msg v){ std::unique_lock<std::mutex> l(m);
                cvs.wait(l,[&]{return q.size()<cap||stopped;}); if(stopped)return;
                q.push_back(v); cvd.notify_one(); }
            bool pop(Msg& v){ std::unique_lock<std::mutex> l(m);
                cvd.wait(l,[&]{return !q.empty()||stopped;});
                if(q.empty())return false; v=q.front(); q.pop_front(); cvs.notify_one(); return true; }
            void stop(){ std::unique_lock<std::mutex> l(m); stopped=true; cvd.notify_all(); cvs.notify_all(); }
        };
        std::vector<Mb> mb(ndev);
        struct Ev { int card, slot; double t0, t1; };
        std::vector<std::vector<Ev>> tl(ndev);
        for (auto& v : tl) v.reserve(8192);

        struct Seq { size_t pi; unsigned tok; int pos; std::vector<unsigned> out; std::atomic<bool> done; };
        std::vector<Seq> sq(slots);
        for (int i=0;i<slots;++i){ sq[i].pi=1; sq[i].tok=prompt[0]; sq[i].pos=0; sq[i].done=false; }
        Mb back;                                   // last stage -> driver
        std::atomic<int> produced{0};
        const double tp0 = ms_now();

        std::vector<std::thread> th;
        for (int g=0; g<ndev; ++g) th.emplace_back([&,g]{
            CK(hipSetDevice(D[g].id));             // per-thread current device: the whole point
            Msg msg;
            while (mb[g].pop(msg)) {
                if (msg.stop) break;
                const int i = msg.slot;
                const double t0 = ms_now();
                if (g > 0) CK(hipMemcpy(D[g].hidden_slots + (size_t)i*Q27_HID,
                                        (unsigned short*)D[g-1].pin_out + (size_t)i*Q27_HID,
                                        Q27_HID*2, hipMemcpyHostToDevice));
                int gs=0, fs=0;
                for (int L=g*per; L<(g+1)*per; ++L) {
                    const q27_layer_t* lay = q27_layer(m, L);
                    int isf = Q27_IS_FULL(L);
                    run_layer(D[g], lay, msg.pos, isf?0:gs, isf?fs:0, ctx, i);
                    if (isf) fs++; else gs++;
                }
                if (g == ndev-1) {
                    Dev& last = D[g];
                    const q27_globals_t* G = q27_globals(m, last.id);
                    q27_rmsnorm(last.hidden_slots + (size_t)i*Q27_HID, G->final_norm,
                                last.norm, Q27_HID, 1, last.stream);
                    q27_quant_perm(last.norm, last.xq, last.xs, Q27_HID, G->lm_head.in_scale, last.stream);
                    if (!exl3_head(last, G, last.norm, last.pa, 1, last.stream))
                        q27_proj_nvfp4(&G->lm_head, last.xq, last.xs, last.pa, last.stream);
                    q27_argmax(last.pa, last.dtok, Q27_VOCAB, last.stream);
                    CK(hipStreamSynchronize(last.stream));
                    unsigned nx; CK(hipMemcpy(&nx, last.dtok, 4, hipMemcpyDeviceToHost));
                    Seq& S = sq[i];
                    S.pos++;
                    if (S.pi < prompt.size()) S.tok = prompt[S.pi++];
                    else { S.tok = nx; S.out.push_back(nx); produced.fetch_add(1);
                           if (nx==248044u || (int)S.out.size()>=maxn) S.done = true; }
                    tl[g].push_back({g,i,t0,ms_now()});
                    back.push({i,S.pos,false});
                } else {
                    CK(hipStreamSynchronize(D[g].stream));
                    CK(hipMemcpy((unsigned short*)D[g].pin_out + (size_t)i*Q27_HID,
                                 D[g].hidden_slots + (size_t)i*Q27_HID,
                                 Q27_HID*2, hipMemcpyDeviceToHost));
                    tl[g].push_back({g,i,t0,ms_now()});
                    mb[g+1].push(msg);
                }
            }
        });

        for (int i=0;i<slots;++i) {               // prime the pipeline
            CK(hipSetDevice(D[0].id));
            q27_embed_gather(q27_globals(m,D[0].id)->embed, sq[i].tok,
                             D[0].hidden_slots + (size_t)i*Q27_HID, D[0].stream);
            CK(hipStreamSynchronize(D[0].stream));
            mb[0].push({i,0,false});
        }
        int live = slots;
        while (live > 0) {                        // driver: recycle finished slots
            Msg r; if (!back.pop(r)) break;
            Seq& S = sq[r.slot];
            if (S.done || S.pos >= ctx) { --live; continue; }
            CK(hipSetDevice(D[0].id));
            q27_embed_gather(q27_globals(m,D[0].id)->embed, S.tok,
                             D[0].hidden_slots + (size_t)r.slot*Q27_HID, D[0].stream);
            CK(hipStreamSynchronize(D[0].stream));
            mb[0].push({r.slot,S.pos,false});
        }
        for (int g=0;g<ndev;++g) mb[g].stop();
        back.stop();
        for (auto& t : th) t.join();
        const double tw = ms_now() - tp0;

        // ---- ACCEPTANCE TEST: do the cards actually overlap in wall-clock? ----
        double busy[16]={0}; double span_lo=1e18, span_hi=-1e18;
        for (int g=0;g<ndev;++g) for (auto&e:tl[g]) { busy[g]+=e.t1-e.t0;
            if(e.t0<span_lo)span_lo=e.t0; if(e.t1>span_hi)span_hi=e.t1; }
        const double span = span_hi-span_lo;
        double sumbusy=0; for(int g=0;g<ndev;++g) sumbusy+=busy[g];
        std::printf("Q27_OVERLAP wall_span=%.1f ms  sum_card_busy=%.1f ms  concurrency=%.2fx of 1 card\n",
                    span, sumbusy, span>0? sumbusy/span : 0.0);
        for (int g=0;g<ndev;++g)
            std::printf("  card %d: %5zu blocks, busy %8.1f ms (%.1f%% of span)\n",
                        g, tl[g].size(), busy[g], span>0?100*busy[g]/span:0.0);
        std::printf("Q27_PIPELINE_T slots=%d produced=%d wall_ms=%.1f  aggregate %.3f tok/s\n",
                    slots, produced.load(), tw, 1000.0*produced.load()/tw);
        for (int i=0;i<slots;++i){ std::printf("Q27_TOKENS[slot %d]",i);
            for(unsigned t:sq[i].out) std::printf(" %u",t); std::printf("\n"); }
        q27_close(m);
        return 0;
    }

    if (slots > 1) {
        // SOFTWARE PIPELINE. slots independent sequences, ndev cards. In round r, card g advances
        // sequence (r-g) mod slots through its own 16 layers. After ndev rounds every card is busy
        // on a different sequence, so the machine stops being 75% idle. This buys aggregate
        // throughput, NOT single-stream latency: one stream is causally serial and cannot be
        // pipelined. Every sequence here runs the same prompt, which is the throughput measurement,
        // not a correctness claim about batching different prompts.
        struct Seq { size_t pi; unsigned tok; int pos; std::vector<unsigned> out; bool done; };
        std::vector<Seq> sq(slots);
        for (int i=0;i<slots;++i) sq[i] = { 1, prompt[0], 0, {}, false };
        const double tp0 = ms_now();
        int rounds = 0, produced = 0;
        const int total_steps = (int)prompt.size() + maxn;
        // Each sequence is touched by card 0 only when r % slots == its index, so it advances one
        // token every `slots` rounds. The bound must be total_steps*slots, not total_steps: with
        // 70 rounds and 4 slots no sequence ever finished its 18-token prompt, which is why the
        // first run produced zero tokens.
        for (int r = 0; r < total_steps * slots + ndev; ++r) {
            for (int g = 0; g < ndev; ++g) {
                const int i = ((r - g) % slots + slots) % slots;
                if (r - g < 0) continue;
                Seq& S = sq[i];
                if (S.done || S.pos >= ctx) continue;
                CK(hipSetDevice(D[g].id));
                if (g == 0) {
                    q27_embed_gather(q27_globals(m, D[0].id)->embed, S.tok,
                                     D[0].hidden_slots + (size_t)i*Q27_HID, D[0].stream);
                } else {
                    CK(hipSetDevice(D[g-1].id));
                    CK(hipMemcpy(D[g-1].pinned, D[g-1].hidden_slots + (size_t)i*Q27_HID,
                                 Q27_HID*2, hipMemcpyDeviceToHost));
                    CK(hipSetDevice(D[g].id));
                    CK(hipMemcpy(D[g].hidden_slots + (size_t)i*Q27_HID, D[g-1].pinned,
                                 Q27_HID*2, hipMemcpyHostToDevice));
                }
                int gs = 0, fs = 0;
                for (int L = g*per; L < (g+1)*per; ++L) {
                    const q27_layer_t* lay = q27_layer(m, L);
                    int isf = Q27_IS_FULL(L);
                    run_layer(D[g], lay, S.pos, isf?0:gs, isf?fs:0, ctx, i);
                    if (isf) fs++; else gs++;
                }
                if (g == ndev-1) {
                    Dev& last = D[ndev-1];
                    const q27_globals_t* G = q27_globals(m, last.id);
                    q27_rmsnorm(last.hidden_slots + (size_t)i*Q27_HID, G->final_norm,
                                last.norm, Q27_HID, 1, last.stream);
                    q27_quant_perm(last.norm, last.xq, last.xs, Q27_HID, G->lm_head.in_scale, last.stream);
                    if (!exl3_head(last, G, last.norm, last.pa, 1, last.stream))
                        q27_proj_nvfp4(&G->lm_head, last.xq, last.xs, last.pa, last.stream);
                    unsigned* dt2; CK(hipMalloc(&dt2,4));
                    q27_argmax(last.pa, dt2, Q27_VOCAB, last.stream);
                    CK(hipStreamSynchronize(last.stream));
                    unsigned nx; CK(hipMemcpy(&nx,dt2,4,hipMemcpyDeviceToHost)); CK(hipFree(dt2));
                    S.pos++;
                    if (S.pi < prompt.size()) S.tok = prompt[S.pi++];
                    else { S.tok = nx; S.out.push_back(nx); ++produced;
                           if (nx == 248044u || (int)S.out.size() >= maxn) S.done = true; }
                }
            }
            ++rounds;
            bool all = true; for (auto& S : sq) if (!S.done) { all = false; break; }
            if (all) break;
        }
        const double tw = ms_now() - tp0;
        std::printf("Q27_PIPELINE slots=%d rounds=%d produced=%d wall_ms=%.1f  "
                    "aggregate %.3f tok/s  per-stream %.3f tok/s\n",
                    slots, rounds, produced, tw, 1000.0*produced/tw, 1000.0*produced/tw/slots);
        for (int i=0;i<slots;++i) {
            std::printf("Q27_TOKENS[slot %d]", i);
            for (unsigned t : sq[i].out) std::printf(" %u", t);
            std::printf("\n");
        }
        q27_close(m);
        return 0;
    }

    for (int pos=0; pos<ctx && (int)out.size()<maxn; ++pos) {
        const bool is_decode = (pi >= prompt.size());
        const double t0 = ms_now();
        // embedding on device 0
        CK(hipSetDevice(D[0].id));
        q27_embed_gather(q27_globals(m, D[0].id)->embed, tok, D[0].hidden_slots, D[0].stream);
        if (ladder && pos < 6) {
            CK(hipStreamSynchronize(D[0].stream));
            q27_gate("embed", q27_d2h_bf16(D[0].hidden_slots, Q27_HID),
                     q27_oracle(orc, "embed", -1, pos));
        }
        int gslot[16] = {0}, fslot[16] = {0};
        for (int g=0; g<ndev; ++g) {
            CK(hipSetDevice(D[g].id));
            if (g > 0) {   // pipeline crossing: 5120 BF16 through pinned host, measured 0.0147 ms
                CK(hipSetDevice(D[g-1].id));
                CK(hipMemcpy(D[g-1].pinned, D[g-1].hidden_slots, Q27_HID*2, hipMemcpyDeviceToHost));
                CK(hipSetDevice(D[g].id));
                CK(hipMemcpy(D[g].hidden_slots, D[g-1].pinned, Q27_HID*2, hipMemcpyHostToDevice));
            }
            for (int L=g*per; L<(g+1)*per; ++L) {
                const q27_layer_t* lay = q27_layer(m, L);
                if (!lay) { std::fprintf(stderr,"Q27_NOT_RESIDENT layer %d\n", L); return 1; }
                int isf = Q27_IS_FULL(L);
                run_layer(D[g], lay, pos, isf?0:gslot[g], isf?fslot[g]:0, ctx, 0);
                if (isf) fslot[g]++; else gslot[g]++;
                if (ladder && pos < 6) {   // gate EVERY layer boundary, at SIX sequential positions
                    CK(hipStreamSynchronize(D[g].stream));
                    char nm[64]; std::snprintf(nm,sizeof nm,"post_mlp L%02d P%d",L,pos);
                    // Tolerance note, deliberately not tuned-until-green: the oracle quantizes the
                    // MLP activation to NVFP4 (16 levels) where this engine uses int8 (256 levels),
                    // so a per-layer relative L2 around 1e-2 is the EXPECTED representation
                    // difference, not a defect - and it is the direction of MORE accuracy. What a
                    // structural defect looks like is a STEP: layer 3 jumped 10x over layers 0-2
                    // when v_proj was missing. Gate on the step, and on the token path.
                    if (!q27_gate(nm, q27_d2h_bf16(D[g].hidden_slots, Q27_HID),
                                  q27_oracle(orc,"post_mlp",L,pos), 2.5e-2)) { ++fails; }
                }
            }
        }
        Dev& last = D[ndev-1];
        CK(hipSetDevice(last.id));
        const q27_globals_t* G = q27_globals(m, last.id);
        q27_rmsnorm(last.hidden_slots, G->final_norm, last.norm, Q27_HID, 1, last.stream);
        q27_quant_perm(last.norm, last.xq, last.xs, Q27_HID, G->lm_head.in_scale, last.stream);
        if (!exl3_head(last, G, last.norm, last.pa, 1, last.stream))
                        q27_proj_nvfp4(&G->lm_head, last.xq, last.xs, last.pa, last.stream);
        if (const char* td = std::getenv("Q27_TOPK_DIR")) {          // first-token decision boundary
            static int dumped = 0;
            if (!dumped && pi >= prompt.size()) {
                CK(hipStreamSynchronize(last.stream));
                std::vector<float> lg = q27_d2h_f32(last.pa, Q27_VOCAB);
                char pth[512]; std::snprintf(pth,sizeof pth,"%s/amd_logits_P%06d.f32", td, pos);
                FILE* f = std::fopen(pth,"wb");
                if (f) { std::fwrite(lg.data(),4,lg.size(),f); std::fclose(f); }
                dumped = 1;
            }
        }
        unsigned* dtok; CK(hipMalloc(&dtok,4));
        q27_argmax(last.pa, dtok, Q27_VOCAB, last.stream);
        CK(hipStreamSynchronize(last.stream));
        unsigned next; CK(hipMemcpy(&next,dtok,4,hipMemcpyDeviceToHost)); CK(hipFree(dtok));

        const double dt = ms_now() - t0;
        if (is_decode) { t_decode += dt; ++n_decode; } else { t_prefill += dt;
            if (pi + 1 >= prompt.size()) for (int i=0;i<PF_N;++i) g_pf[i]=0; }   // steady state only

        if (pi < prompt.size()) tok = prompt[pi++];       // still ingesting the prompt
        else { tok = next; out.push_back(next); if (next==248044u) break; }   // eos_token_id
    }

    std::printf("Q27_TOKENS");
    for (unsigned t : out) std::printf(" %u", t);
    std::printf("\n");
    if (ladder) std::printf("Q27_LADDER fails=%d\n", fails);
    // Prefill here is the same single-token path run over the prompt, NOT a batched prefill, so
    // report the two separately and never blend them into one rate.
    std::printf("Q27_TIMING prompt_tokens=%zu prefill_ms=%.1f (%.1f ms/tok)  "
                "decode_tokens=%d decode_ms=%.1f (%.2f ms/tok, %.3f tok/s)\n",
                prompt.size(), t_prefill, prompt.size()? t_prefill/prompt.size() : 0.0,
                n_decode, t_decode, n_decode? t_decode/n_decode : 0.0,
                n_decode? 1000.0*n_decode/t_decode : 0.0);
    if (g_profile && n_decode > 0) {
        double tot = 0; for (int i=0;i<PF_N;++i) tot += g_pf[i];
        std::printf("Q27_PROFILE decode steady state, %d tokens, %.2f ms/token accounted\n",
                    n_decode, tot/n_decode);
        for (int i=0;i<PF_N;++i)
            std::printf("  %-12s %8.2f ms/token  %5.1f%%\n",
                        PF_NAME[i], g_pf[i]/n_decode, 100.0*g_pf[i]/tot);
    }
    q27_close(m);
    return fails ? 4 : 0;
}
