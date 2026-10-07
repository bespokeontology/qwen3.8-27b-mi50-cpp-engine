// q27_load.h — checkpoint loader ABI for the Qwen3.8-27B native gfx906 engine.
//
// Reads the three safetensors shards of the merged NVFP4 checkpoint directly (8-byte LE header
// length, JSON header, payload), builds a name->descriptor catalog, and uploads a chosen LAYER
// RANGE to a chosen GPU so the 16.40 GiB decode trunk can be split across cards.
//
// No JSON library, no Python, no allocation after q27_upload() returns.
//
// Storage ABI, group sizes and scale semantics are q27.h / the implementation contract §5.
// This header only says WHERE the bytes are and HOW the handles get populated.
//
// ---- what is in the checkpoint (measured from the bytes, 2026-09-08) ----
//   whole file set                 21,921,428,072 B = 20.4159 GiB   (2194 tensors)
//   model.visual.*  (vision tower)    921,460,192 B =    878.77 MiB  <- NOT loaded, see below
//   mtp.*           (draft head)      849,398,784 B =    810.05 MiB  <- NOT loaded, see below
//   64 text layers                 16,892,600,448 B =  15.7325 GiB
//     full-attention layer (L%4==3)    255,284,280 B each  x16
//     gated-delta  layer (L%4!=3)      266,834,416 B each  x48
//   lm_head.*                         715,161,608 B =    682.03 MiB
//   model.language_model.norm              10,240 B
//   embed_tokens                    2,542,796,800 B =   2425.00 MiB  (optional, see below)
//
//   decode trunk = layers + final norm + lm_head = 17,607,772,296 B = 16.3985 GiB.
//   A single MI50 is 15.98 GiB, so the trunk MUST span >= 2 cards. The intended first
//   configuration is 4 cards x 16 layers; the arenas q27_upload_split actually allocates are
//     card 0  layers [0,16)  + embed_tokens   6,765,949,952 B = 6.3013 GiB  (261 tensors)
//     card 1  layers [16,32)                  4,223,153,152 B = 3.9331 GiB  (260)
//     card 2  layers [32,48)                  4,223,153,152 B = 3.9331 GiB  (260)
//     card 3  layers [48,64) + norm + lm_head 4,938,324,992 B = 4.5992 GiB  (263)
//   (960 B of 256-byte alignment padding per card). Pass want_embed=0 and card 0 is 3.9331 GiB
//   too. Every card keeps >= 9.6 GiB for the BF16 KV cache, the 144 MiB fp32 recurrent state
//   and scratch.
//
//   The F32 scalars (input_scale, weight_scale, weight_scale_2) are NEVER uploaded: they are
//   read on the host and live inside q27_nvfp4_t / q27_fp8_t as plain floats, which is why the
//   arena is 800 B/card smaller than the on-disk byte count of the same tensor set.
//
// ---- what is deliberately skipped ----
//   The vision tower (model.visual.*, 878.77 MiB) and the MTP draft head (mtp.*, 810.05 MiB)
//   are NOT needed for raw autoregressive text decode: the vision tower only produces image
//   embeddings for multimodal prompts, and the MTP head is a speculative draft model that the
//   plain single-token decode path never calls. They are catalogued (so the 2194 / 21,921,428,072
//   fail-closed check is over the WHOLE file set) but never uploaded.
#pragma once

#include "q27.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- fail-closed checkpoint constants ----------------
#define Q27_CKPT_SHARDS         3
#define Q27_CKPT_TENSORS        2194
#define Q27_CKPT_PAYLOAD_BYTES  21921428072LL

#define Q27_MAX_DEVICES         16
#define Q27_MAX_DIMS            8

// ---------------- dtype codes ----------------
enum {
    Q27_DT_UNKNOWN = 0,
    Q27_DT_BF16, Q27_DT_F16, Q27_DT_F32, Q27_DT_F64,
    Q27_DT_F8_E4M3, Q27_DT_F8_E5M2,
    Q27_DT_U8, Q27_DT_I8, Q27_DT_U16, Q27_DT_I16,
    Q27_DT_U32, Q27_DT_I32, Q27_DT_U64, Q27_DT_I64, Q27_DT_BOOL
};

// ---------------- error codes ----------------
#define Q27_OK          0
#define Q27_E_IO       (-1)   // open/stat/mmap failed
#define Q27_E_FORMAT   (-2)   // header is not the JSON we expect, or the census check failed
#define Q27_E_MISSING  (-3)   // a required tensor name is absent
#define Q27_E_SHAPE    (-4)   // a tensor has a shape/dtype the ABI does not allow
#define Q27_E_HIP      (-5)   // hipMalloc / hipMemcpy / hipSetDevice failed
#define Q27_E_ARG      (-6)   // bad argument
#define Q27_E_STATE    (-7)   // e.g. a layer uploaded twice

// ---------------- opaque model ----------------
typedef struct q27_model q27_model_t;

// ---------------- per-layer device handles ----------------
// Every pointer is a DEVICE pointer on `device`. Fields for the mixer this layer does not run are
// zeroed: a full-attention layer has rows==0 / NULL in the gated-delta block and vice versa.
typedef struct {
    int layer;                          // global layer index [0,64)
    int is_full;                        // Q27_IS_FULL(layer): full attention, else gated delta
    int device;                         // GPU that owns these pointers, -1 if not resident

    // pre-norms, BF16 [5120], applied as (1 + W)  (contract §2)
    const unsigned short* input_norm;
    const unsigned short* post_norm;

    // MLP, NVFP4. gate and up SHARE input_scale (verified over all 64 layers) so one
    // q27_quant_perm call feeds both.  gate/up: rows 17408 K 5120.  down: rows 5120 K 17408.
    q27_nvfp4_t gate, up, down;

    // ---- full attention (16 layers) ----
    // q/k/v share one input_scale (verified; the oracle throws otherwise, contract §3).
    // q_proj is 12288 rows = 24 heads x (256 query + 256 OUTPUT GATE), see q27.h Q27_QROWS.
    q27_fp8_t q_proj, k_proj, v_proj, o_proj;
    const unsigned short* q_norm;       // BF16 [256], (1 + W)
    const unsigned short* k_norm;       // BF16 [256], (1 + W)

    // ---- gated delta (48 layers) ----
    q27_fp8_t in_qkv;                   // rows 10240 K 5120  (q 2048 | k 2048 | v 6144)
    q27_fp8_t in_z;                     // rows  6144 K 5120
    q27_fp8_t out_proj;                 // rows  5120 K 6144
    // ---- NVFP4 twins, filled by the Q27_FP8X4 load-time conversion ----
    q27_nvfp4_t q4, k4, v4, o4, iqkv4, iz4, op4;
    q27_nvfp4_t q8, k8, v8, o8, iqkv8, iz8, op8;   // Q27_LS_Q8: int8 dot4 execution mirrors of the seven fp8 projections
    q27_i8g_t gate8, up8, down8;                   // Q27_LS_Q8: int8 per-64-group execution mirrors of the NVFP4 MLP (int32 accumulation inside a group)
    q27_i8g_t q8g, k8g, v8g, o8g, iqkv8g, iz8g, op8g;   // Q27_LS_Q8 stage 2: int8 per-64 mirrors of the seven fp8 projections
    q27_i8r_t gate8r, up8r, down8r, q8r, k8r, v8r, o8r, iqkv8r, iz8r, op8r;   // Q27_LS_Q8 stage 3: per-row int8 views (same bytes, requantized in place) for the rocBLAS GEMMs
    q27_i8r_t ab8r;                                // Q27_LS_Q8 stage 3: GDN a/b projection as one int8 per-row tensor [96][5120] (rocBLAS)
                                                // (w = int8 [rows][K], gs = e4m3 per-16 group scale, ws2 = wscale, in_scale)
    const unsigned short* in_a;         // BF16 [48][5120]   dt projection, BF16 GEMV
    const unsigned short* in_b;         // BF16 [48][5120]   beta projection, BF16 GEMV
    const unsigned short* conv1d;       // BF16 [10240][4]   channel-major, taps contiguous
    const unsigned short* A_log;        // BF16 [48]
    const unsigned short* dt_bias;      // BF16 [48]
    const unsigned short* gdn_norm;     // BF16 [128]  RAW weight — NO +1 (contract §2)
} q27_layer_t;

// ---------------- non-layer device handles, per device ----------------
typedef struct {
    const unsigned short* embed;        // BF16 [248320][5120], NULL unless requested
    const unsigned short* final_norm;   // BF16 [5120], (1 + W), NULL unless requested
    q27_nvfp4_t lm_head;                // rows 248320 K 5120, .w == NULL unless requested
} q27_globals_t;

// ---------------- upload request ----------------
typedef struct {
    int device;                         // HIP device ordinal
    int layer_lo, layer_hi;             // upload layers [lo, hi); pass lo==hi for none
    int with_embed;                     // also upload embed_tokens          (2425.00 MiB)
    int with_final_norm;                // also upload model.language_model.norm  (10 KiB)
    int with_lm_head;                   // also upload lm_head (NVFP4)         (682.03 MiB)
} q27_upload_req_t;

// ---------------- catalog ----------------
// Opens the three shards read-only, mmaps them, parses the JSON headers and verifies the census
// (Q27_CKPT_TENSORS tensors, Q27_CKPT_PAYLOAD_BYTES payload bytes). Fails closed on any mismatch.
// No GPU is touched. `err` may be NULL.
q27_model_t* q27_open(const char* model_dir, char* err, size_t errcap);
void         q27_close(q27_model_t* m);

int          q27_tensor_count(const q27_model_t* m);
long long    q27_payload_bytes(const q27_model_t* m);

// Host pointer into the mmap for a named tensor, or NULL. Any out-param may be NULL.
// `shape` must have room for Q27_MAX_DIMS entries when non-NULL.
const void*  q27_host_ptr(const q27_model_t* m, const char* name, size_t* nbytes,
                          int* dtype, int* ndim, long long* shape);

// Catalog iteration (tests / census dumps).
const char*  q27_catalog_name(const q27_model_t* m, int i);
int          q27_catalog_info(const q27_model_t* m, int i, int* shard, int* dtype,
                              long long* offset, size_t* nbytes, int* ndim, long long* shape);

// ---------------- upload ----------------
// Allocates ONE hipMalloc arena per call on req->device and copies every requested tensor into it
// with 256-byte alignment, straight out of the mmap. Populates the layer/global handles.
// Re-uploading a layer that is already resident is Q27_E_STATE.
int q27_upload(q27_model_t* m, const q27_upload_req_t* req, char* err, size_t errcap);

// ---------------- tensor-parallel upload (C1) ----------------
// Uploads ALL 64 layers to ALL `ndev` devices, but each device receives only its SHARD of every
// large matrix. Shard map, card g of G:
//
//   COLUMN-PARALLEL (contiguous output-row range, no collective after):
//     q_proj      [12288,5120]  rows [g*12288/G, +12288/G)   6 of 24 heads, query AND output gate
//     k_proj/v_proj[1024,5120]  rows [g*1024/G , +1024/G )   1 of the 4 KV heads
//     in_proj_z   [ 6144,5120]  rows [g*6144/G , +6144/G )   12 of the 48 value heads
//     gate/up     [17408,5120]  rows [g*17408/G, +17408/G)
//     lm_head     [248320,5120] rows [g*248320/G, +248320/G) 62080 logits per card
//     in_proj_qkv [10240,5120]  THREE ranges, not one: the tensor is q 2048 | k 2048 | v 6144 and
//                               a single contiguous quarter would cut a head group in half. Card g
//                               takes q rows [g*2048/G,+2048/G), k rows [2048+g*2048/G,+2048/G),
//                               v rows [4096+g*6144/G,+6144/G) -> 4 key heads + 12 value heads,
//                               packed contiguously on the card as  q | k | v  of width 10240/G.
//   ROW-PARALLEL (strided input-column range, ALL-REDUCE after):
//     o_proj      [5120,6144]   cols [g*6144/G , +6144/G )
//     out_proj    [5120,6144]   cols [g*6144/G , +6144/G )
//     down_proj   [5120,17408]  cols [g*17408/G, +17408/G)
//   REPLICATED whole: input/post/q/k/gdn norms, final norm, A_log, dt_bias, conv1d, in_proj_a/b.
//   weight_scale_2 / weight_scale / input_scale are per-tensor F32 scalars, replicated unchanged:
//   a row-parallel partial sum is scaled identically on every card, so the sum of the scaled
//   partials equals the scaled sum.
//
// Slicing the stored formats:
//   NVFP4 is U8 [R][K/2] + e4m3 [R][K/16]. A column-parallel slice is a contiguous row range of
//   BOTH planes. A row-parallel slice is STRIDED: per row, bytes [g*KP/G, +KP/G) of the packed
//   plane and [g*NG/G, +NG/G) of the scale plane, copied row by row through a host staging buffer.
//   FP8 is [R][K] byte-granular, same two cases.
//
// FAIL CLOSED: every split must be exact. rows % G == 0 for column-parallel, K % G == 0 AND
// (K/G) % 16 == 0 for row-parallel so no group-of-16 scale straddles a shard boundary. Nothing
// is rounded; a non-dividing geometry is Q27_E_SHAPE.
//
// The handles are stamped with the SHARD's rows/K, so every existing kernel runs unmodified on the
// smaller shape. Retrieve them with q27_layer_tp(m, layer, device); q27_layer(m, layer) returns
// the devices[0] shard (also shard-shaped) so the residency API keeps working.
//
// Resident cost: ~4.4 GiB/card at G=4 (embed_tokens is NOT uploaded in this mode; the TP engine
// gathers the 10 KiB embedding row on the host, straight out of the mmap).
int q27_upload_tp(q27_model_t* m, const int* devices, int ndev, char* err, size_t errcap);
size_t q27_tp_free_fp8_proj(q27_model_t* m, int (*rekey)(const void* oldw, const void* neww));   // Q27_DEC_I8: free the TP shards' fp8 projections (their own arena), sentinel the handles (rekey moves each mirror key); returns bytes freed
// LAYER-SPLIT residency (Q27_LAYER_SPLIT): card g holds the FULL weights of layers
// [g*LPP, (g+1)*LPP), LPP = Q27_LAYERS/ndev. No matrix is sharded, so the intra-layer
// all-reduce collectives disappear entirely; the only cross-card traffic is the 5120-float
// activation handoff between layer groups. Card ndev-1 keeps the full final norm + lm_head.
int q27_upload_ls(q27_model_t* m, const int* devices, int ndev, char* err, size_t errcap);
// Per-(layer, device) FULL-layer handles of the layer-split layout (q27_upload_ls), and its
// per-device globals (the full final norm + lm_head live on devices[ndev-1]).
const q27_layer_t* q27_layer_ls(const q27_model_t* m, int layer, int device);
const q27_globals_t* q27_globals_ls(const q27_model_t* m, int device);

// Per-(layer, device) shard handles. NULL if that layer is not resident on that device, or if the
// model was not uploaded with q27_upload_tp.
const q27_layer_t* q27_layer_tp(const q27_model_t* m, int layer, int device);
int                q27_tp_ndev(const q27_model_t* m);   // 0 unless q27_upload_tp succeeded

// Convenience: the intended first configuration. Splits all 64 layers evenly over `ndev` devices
// (ndev must divide 64), puts embed_tokens on devices[0] (pass want_embed=0 to keep the 2.37 GiB
// table on the host and gather there), and the final norm + lm_head on devices[ndev-1].
int q27_upload_split(q27_model_t* m, const int* devices, int ndev, int want_embed,
                     char* err, size_t errcap);

// ---------------- residency ----------------
const q27_layer_t*   q27_layer(const q27_model_t* m, int layer);      // NULL if not resident
const q27_globals_t* q27_globals(const q27_model_t* m, int device);   // NULL if device unused
int                  q27_layer_device(const q27_model_t* m, int layer); // -1 if not resident
long long            q27_resident_bytes(const q27_model_t* m, int device); // -1 if device unused
long long            q27_resident_bytes_total(const q27_model_t* m);

// Human-readable residency report. Returns the number of bytes it wanted to write (snprintf
// semantics); truncates into `buf`.
int q27_report(const q27_model_t* m, char* buf, size_t cap);

// Releases every arena on `device` and clears the handles that pointed into it.
void q27_free_device(q27_model_t* m, int device);

#ifdef __cplusplus
}
#endif
