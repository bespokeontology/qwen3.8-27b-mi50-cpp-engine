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
#include "q27_exl3.h"

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
    // Q27_DEC_MLP_I8: the TP decode's MLP handles for a layer whose MLP is int8 on this card. They
    // are VIEWS of the same per-row mirror the prefill's rocBLAS path uses -- gate/up as a row range
    // of the full-width mirror, down as a column range of it -- so a layer whose MLP is int8 needs no
    // resident NVFP4 MLP shard at all. `gs` is always K: one scale per row.
    q27_i8r_t dc_gate8r, dc_up8r, dc_down8r;
    // Q27_MXFP4_MLP: DECODE-ONLY OCP-MXFP4 copies of the MLP projections. Separate handles rather
    // than a rewrite of gate/up/down because PREFILL shares those same handles for every layer the
    // layer-split does not own (48 of 64 per card), and it has no MXFP4 consumer yet -- pointing the
    // shared handle at an E8M0 K/32 plane produced token salad from the first token (2026-09-17).
    // gs_mx on these is the "use me" flag; when it is 0 the decode dispatcher takes gate/up/down.
    q27_nvfp4_t mx_gate, mx_up, mx_down;
    // Q27_FP8_MLP: block-scaled FP8 MLP projections (Qwen3.8-27B-FP8). w=[rows][K] e4m3,
    // bs=[rows/128][K/128] BF16 weight_scale_inv, both SHARD-LOCAL. Non-null w selects the FP8
    // MLP path in decode; null leaves the shipped NVFP4 path untouched.
    q27_fp8_t f8_gate, f8_up, f8_down;
    q27_i8g_t q8g, k8g, v8g, o8g, iqkv8g, iz8g, op8g;   // Q27_LS_Q8 stage 2: int8 per-64 mirrors of the seven fp8 projections
    q27_i8r_t gate8r, up8r, down8r, q8r, k8r, v8r, o8r, iqkv8r, iz8r, op8r;   // Q27_LS_Q8 stage 3: per-row int8 views (same bytes, requantized in place) for the rocBLAS GEMMs
    q27_i8r_t ab8r;                                // Q27_LS_Q8 stage 3: GDN a/b projection as one int8 per-row tensor [96][5120] (rocBLAS)
                                                // (w = int8 [rows][K], gs = e4m3 per-16 group scale, ws2 = wscale, in_scale)
    // ---- Q27_EXL3: trellis-quantized projections (4bpw / 6bpw) ----
    // ONE resident representation: the packed trellis plus suh/svh. A live
    // handle wins at the dispatch site ahead of every other format; a null one
    // leaves the incumbent path byte-for-byte untouched, which is what keeps
    // the 8-bit arm intact while this is brought up.
    // Unlike every handle above, these are NOT fed a pre-quantized activation:
    // EXL3 rotates its own input (suh then blockwise H128), so the dispatcher
    // hands them the bf16 norm vector directly and there is no in_scale.
    q27_exl3_t ex_gate, ex_up, ex_down;
    // in_proj_qkv is THREE handles, not one. It is the only projection the TP
    // plan shards as multiple NON-CONTIGUOUS output ranges (q 2048 | k 2048 |
    // v 6144, so each card gets whole heads), and concatenating those ranges
    // into a single EXL3 tensor is WRONG: measured 2.8 dB against the BF16
    // reference versus 22.15 dB for the same ranges kept separate, with the
    // trellis and svh bytes byte-identical either way. Contiguous slices are
    // fine (a 4-way contiguous split of the same tensor passes at 22.15), so
    // the output rotation does not survive arbitrary regrouping of its blocks.
    q27_exl3_t ex_iqkv[3], ex_iz, ex_op;     // gated delta  (48 layers)
    q27_exl3_t ex_q, ex_k, ex_v, ex_o;       // full attention (16 layers)
    int ex_live = 0;                          // all of this layer's projections are EXL3

    const unsigned short* in_a;         // BF16 [48][5120]   dt projection, BF16 GEMV
    const unsigned short* in_b;         // BF16 [48][5120]   beta projection, BF16 GEMV
    const unsigned short* conv1d;       // BF16 [10240][4]   channel-major, taps contiguous
    const unsigned short* A_log;        // BF16 [48]
    const unsigned short* dt_bias;      // BF16 [48]
    const unsigned short* gdn_norm;     // BF16 [128]  RAW weight — NO +1 (contract §2)
    // Q27_LS_SR (single residency): down_proj as its four K-slices [5120][4352] (+ compacted bs
    // [40][34] each), all on the OWNER card. FOUR slices rather than one full-width [5120][17408]
    // tensor because the block-scaled consumer (q27_proj_fp8b_nr -> q27_k_proj_fp8_snr<..,NSTEP=5,
    // ..,BS=1>) walks K in five 1024-wide steps and refuses K > 5120 per launch. The owner therefore
    // runs the same TP-slice consumer FOUR times, once per slice against the matching 4352-column
    // range of the swiglu activation, and sums the four fp32 partials -- exactly the arithmetic the
    // 4-card TP decode already does across cards (row-parallel down + all-reduce), collapsed onto one
    // card. Each slice is its own [rows][K] plane with a SLICE-LOCAL bs plane (bs_kblk = 34), so the
    // kernel indexes (row>>7, k>>7) inside the slice with no column offset, as it does for a TP shard.
    // w == nullptr outside Q27_LS_SR=1 and on non-owner cards; f8_down stays null in SR mode (the
    // slices are the only down representation). APPENDED LAST on purpose: nothing above moves, so
    // every offsetof()-keyed slot in q27_upload_ls stays where it was.
    q27_fp8_t f8_dn[4];
} q27_layer_t;

// Q27_EXL3 per-layer load (full width, CURRENT device). Defined in q27_exl3.cpp;
// declared here because it needs the completed layer type.
int  q27_exl3_load_layer(q27_exl3_store_t* S, q27_layer_t* H, int L,
                         size_t* bytes_out, char* err, size_t errcap);
// TP-sharded variant: card g of ndev, matching Tp::col / Tp::row exactly.
int  q27_exl3_load_layer_tp(q27_exl3_store_t* S, q27_layer_t* H, int L, int g, int ndev,
                            size_t* bytes_out, char* err, size_t errcap);
void q27_exl3_free_layer(q27_layer_t* H);
int  q27_exl3_load_mtp(const char* dir, q27_exl3_t* fc, q27_exl3_t* gate,
                       q27_exl3_t* up, q27_exl3_t* down, char* err, size_t errcap);
// Q27_EXL3_DIR sideload over the layer-split residency: attaches trellis
// weights to each card's owned layers. Returns the number of layers covered.
int  q27_exl3_sideload(q27_model_t* m, const int* devices, int ndev, const char* dir);

// ---------------- non-layer device handles, per device ----------------
typedef struct {
    const unsigned short* embed;        // BF16 [248320][5120], NULL unless requested
    const unsigned short* final_norm;   // BF16 [5120], (1 + W), NULL unless requested
    q27_nvfp4_t lm_head;                // rows 248320 K 5120, .w == NULL unless requested
    // Q27_EXL3: the trellis lm_head. Both published stores carry it at K=6
    // (even the "4bpw" one), and 248320 is a multiple of 128 so the blockwise
    // Hadamard needs no padding. Live => every head site takes it and the
    // NVFP4 head above is never allocated.
    q27_exl3_t  ex_head;
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
// Which card owns which layer under the layer-split residency (the same rule the sweep uses).
int q27_ls_owned(int layer, int card, int ndev);
// THE single layer-split ownership oracle: which card owns block b (b = layer / Q27_LS_BLK).
// Honours Q27_LS_OWN uneven quotas. Every site that needs ownership MUST call this, not
// re-derive the modulo -- five inline copies existed before 2026-09-18 and a partial change
// to any one of them makes a card address a layer it does not hold.
int q27_ls_block_owner(int b, int ndev);
// 1 = this layer's layer-split MLP is converted to the persistent int8 mirror, i.e. its full-width
// NVFP4 originals are gone and no other consumer may address them.
int q27_ls_mlp_i8(int layer, int card, int ndev, int k);
// Resolve the int8 MLP residency count before the layer-split upload (pass -1 for window-aware auto).
void q27_ls_mlp_i8k_set(int k);
// Q27_ALIAS_MLP: after both residencies exist, point the TP MLP row shards of layer-split-owned
// layers at the full-width layer-split tensors (no second copy). Returns the number aliased.
int q27_resolve_tp_ls_mlp(q27_model_t* m, int ndev);
// Q27_SERVE_GROW eviction: release one layer's int8 MLP residency on its OWNER card and restore that
// layer's NVFP4 authority (re-read from the still-mapped checkpoint). Returns 1 if a layer was freed.
int q27_ls_mlp_evict(q27_model_t* m, int layer, int card, int ndev);
// Q27_E4M3_FAST eligibility pass over the decode gate/up e4m3 scale planes. Returns how many were
// certified. Not calling it leaves every gs_fast at 0, i.e. the shipped decoder everywhere.
int q27_nvfp4_validate_gu(q27_model_t* m, int ndev);
// Q27_MXFP4_MLP=<safetensors>: overwrite the resident TP decode gate/up shards with AMD Quark
// OCP-MXFP4 tensors (E2M1 nibbles + E8M0 per-32 scales). Returns shards converted.
int q27_mxfp4_sideload_mlp(q27_model_t* m, int ndev, const char* path);
// Q27_FP8_MLP=<dir>: load the block-scaled FP8 MLP (Qwen3.8-27B-FP8, one shard per layer) into the
// per-layer f8_gate/f8_up/f8_down handles. Returns projections made resident.
int q27_fp8_mlp_load(q27_model_t* m, int ndev, const char* dir);
int q27_fp8_ls_prefill_mirrors(q27_model_t* m, int ndev, const char* dir);
void q27_storage_finish(void);
void q27_storage_owner_ready(int device);

// ---------------- Q27_LS_SR: single residency (2026-09-17) ----------------
// Q27_LS_SR=1 = Q27_LS_NOTP semantics (no TP upload at all) + the layer-split copy is the ONLY copy:
//   - q27_upload_ls uploads NO NVFP4 MLP blocks (gate/up/down w/gs stay null; rows/K/ws2/in_scale
//     are still stamped from the checkpoint scalars), puts the seven fp8 projections of every owned
//     layer into per-tensor TEMPORARIES (recorded in the model, handles left pointing at them) so
//     main can build the int8 per-64 side-table mirrors from them (q27_fp8_make_i8g, keyed by the
//     STABLE handle address inside tpl_ls), builds ab8r for owned GDN layers and no other mirror,
//     forces Q27_LS_HEAD_ALIAS off so the head card carries final_norm + the full NVFP4 lm_head in
//     its own arena, and mirrors glob_ls into glob so q27_globals() answers without a TP residency.
//   - q27_fp8_mlp_load_sr loads the block-scaled FP8 MLP at FULL WIDTH on the owner card
//     (f8_gate/f8_up [17408][5120] + bs [136][40]; f8_dn[0..3] = the four K-slices of down_proj).
//     in_scale comes from the NVFP4 checkpoint's F32 scalars
//     model.language_model.layers.<L>.mlp.{gate,up,down}_proj.input_scale (read via q27_host_ptr,
//     the same bytes Tp::nvfp4_col reads), cross-checked against the stamped handle.
//   - q27_ls_sr_release_fp8 frees the temporaries once the mirrors exist and sentinels the handles.
// Q27_LS_SR_DRYRUN=1 makes both loaders print the per-card byte plan and return before any hipMalloc
// (q27_upload_ls then returns Q27_E_STATE so a production boot cannot continue on null handles).
// With Q27_LS_SR unset every existing path is untouched.
int  q27_fp8_mlp_load_sr(q27_model_t* m, int ndev, const char* dir);   // projections loaded (6 per owned layer, 384 at 4 cards); FATAL + abort if any layer is not fully covered
size_t q27_ls_sr_release_fp8(q27_model_t* m, int (*rekey)(const void* oldw, const void* neww));   // mirrors q27_tp_free_fp8_proj: frees the fp8 temporaries, sentinels the seven handles' w, calls rekey(old, new) per handle; returns bytes freed over all cards (nullable rekey -- the side table is keyed by handle address)
int  q27_ls_sr_owner(const q27_model_t* m, int layer);                 // card INDEX g (position in the device list) owning `layer` under the layer-split rule, or -1
int  q27_ls_sr_head_card(const q27_model_t* m);                        // card INDEX holding final_norm + lm_head (the owner of layer 63), or -1

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
