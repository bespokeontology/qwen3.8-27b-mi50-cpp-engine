// q27_dflash2.h — DFlash2 block-drafter weights: checkpoint ABI, load-time int8 repack, residency.
// Branch dflash2-port. The BF16 safetensors is STORAGE ONLY: every GEMM tensor is repacked once at
// load into int8 per-64-group rows + fp32 group scales (the proven q27_i8g mirror layout); norms and
// conv base kernels ride in fp32 (tiny, not in any hot loop). No BF16 execution anywhere.
#pragma once
#include <cstdint>
#include <cstddef>
#include <hip/hip_runtime.h>

struct q27_i8g_d_t {              // device-resident int8 per-64-group weight
    signed char* w = nullptr;     // [rows][K]  int8
    float*       s = nullptr;     // [rows][K/64] fp32 group scales
    float        alpha = 1.f;
    int          rows = 0, K = 0;
    size_t       bytes() const { return (size_t)rows * K + (size_t)rows * (K / 64) * 4; }
};

struct q27_df2_layer_t {
    q27_i8g_d_t q, k, v, o;            // 4096/1024/1024 x 5120, o 5120x4096
    q27_i8g_d_t gate, up, down;        // 17408x5120, 17408x5120, 5120x17408
    q27_i8g_d_t attn_conv_proj, mlp_conv_proj;   // 1280x5120
    float* attn_conv_base = nullptr;   // fp32 [2][2][5120]
    float* mlp_conv_base = nullptr;    // fp32 [2][2][5120]
    float* in_norm = nullptr;          // fp32 [5120]
    float* post_norm = nullptr;        // fp32 [5120]
    float* q_norm = nullptr;           // fp32 [128]
    float* k_norm = nullptr;           // fp32 [128]
};

struct q27_df2_t {
    q27_df2_layer_t L[5];
    q27_i8g_d_t fc;                    // 5120x25600 (5 taps x 5120)
    q27_i8g_d_t sel_hidden_proj;       // 256x5120
    q27_i8g_d_t pred_codebook;         // 248320x256, BF16 (kept bf16: the trained selector scoring)
    q27_i8g_d_t succ_codebook;         // 248320x256, BF16
    int codebook_bf16 = 1;             // 1: codebook .w holds raw bf16 bytes (rows*K*2), s unused
    float* hidden_norm = nullptr;      // fp32 [5120]
    float* final_norm = nullptr;       // fp32 [5120]
    size_t bytes_device = 0;          // int8 weights + scales + fp32 smalls, per card
    size_t bytes_host = 0;             // host-side staging (freed at q27_df2_free)
    void*  arena = nullptr;            // one hipMalloc arena per upload
    struct q27_df2_host_t;             // forward: internal host staging
    q27_df2_host_t* H = nullptr;
};

// Host build: parse + gate the checkpoint identity, convert to the int8 representation.
int q27_df2_open(const char* path, q27_df2_t* W, char* err, size_t errcap);
// Device upload for one card (replicated; one arena, one hipFree at q27_df2_free).
int q27_df2_upload(q27_df2_t* W, int dev, char* err, size_t errcap);
// TP4-sharded upload: GEMM output dims sharded by row range [dev*N/4,(dev+1)*N/4), down sharded on
// its input K-dim, convs/norms replicated, selector codebooks on card 0 only. W's geometry fields end
// up describing the LAST uploaded card (single-instance limitation; the executor gets per-card structs).
int q27_df2_upload_sharded(q27_df2_t* W, int dev, int ndev, char* err, size_t errcap);
void q27_df2_free(q27_df2_t* W);
