// MTP DRAFT HEAD -- types and declarations, factored out of q27_main.cpp.
//
// WHY THIS IS A HEADER: the draft step is CALLED from run_tp and its per-card decode loop, which are
// defined long before the probe code at the bottom of the driver. Defining these next to the probes
// put them after their first use and the build failed with seven "undeclared identifier" errors --
// the same ordering defect recorded twice already in this campaign. Including this file right after
// Dev is complete makes the dependency order a property of the file layout instead of a property of
// where text happened to land. Only DECLARATIONS live here; the bodies stay next to the probes.
#ifndef Q27_MTP_DEFS_H
#define Q27_MTP_DEFS_H

// The draft layer is REPLICATED on every card rather than TP-sharded. Every card already holds the
// full hidden state after the collectives, and the draft's inputs (that state and the sampled
// token's embedding) are identical on all four, so a replicated draft produces identical drafts
// with NO communication at all -- the only cross-card op left is the existing 4-way head max.
// 285 MB quantized per card is nothing against 4.1 GiB of resident model.
//
// Weights arrive BF16 and are quantized at load (q27_mtp_quant.h): NVFP4 for fc and the MLP, FP8 for
// attention, BF16 passthrough for the six 5120-wide norms. in_scale for the fp8/nvfp4 handles is a
// fixed activation scale (0.02 => activations up to ~9 stay inside e4m3); the draft's quality only
// moves the acceptance rate, never correctness, so a fixed scale is acceptable and measurable.
struct MtpW {
    q27_fp8_t   q, k, v, o;
    // SLICED onto the EXACT K geometries the TP engine exercises (5120 and 4352). The full-width
    // forms (fc K=10240, down K=17408) are geometries only the non-TP serial path uses, and the
    // draft's MLP output measured 5445 where the oracle measures 0.026 for the same layer type --
    // so the draft moves onto the kernel shapes that are actually gated in production.
    q27_nvfp4_t fc[2], gate, up, down[4];
    const unsigned short *in_norm = nullptr, *post_norm = nullptr;
    const unsigned short *q_norm = nullptr, *k_norm = nullptr, *mtp_norm = nullptr;
    const unsigned short *pre_emb = nullptr, *pre_hid = nullptr;
    long long bytes = 0;
};

struct MtpState {
    MtpW W;
    unsigned short* cat = nullptr;      // [10240] bf16 = norm_hid(h_n) | norm_emb(emb)
    unsigned short* emb = nullptr;      // [5120]  bf16 embedding row of t_{n+1}
    unsigned short* host_emb = nullptr; // staging, allocated on card 0, read by all four
    const void* emb_tbl = nullptr;      // HOST base of embed_tokens; the TP plan keeps NO device copy
    // FULL-WIDTH scratch. This is NOT the Dev scratch and must never be: under TP every Dev
    // buffer is SHARD-sized (qkva 3584, act 4352, xq 4352 on a 4-way split) because each card owns
    // a slice of every projection -- but the draft layer is REPLICATED and runs at full width
    // (qkva 14336, act 17408). Reusing Dev scratch wrote 12288 bf16 into a 3584-element buffer and
    // the first draft took a GPU page fault. d.pa/d.pb ARE full width (they hold the 62080-logit
    // head shard) and are reused below.
    unsigned short *qkva = nullptr, *qh = nullptr, *mix6 = nullptr;
    unsigned short *act = nullptr, *nrm = nullptr, *hid = nullptr, *hid2 = nullptr;
    signed char* xq = nullptr;
    float* xs = nullptr;
    // The draft's OWN projection scratch. It previously borrowed d.pa/d.pb, which the MAIN model's
    // head also writes and which the pipelined decode path has in flight -- a write-after-read
    // hazard between the draft and the main stream. dpa must hold the lm_head logit shard (62080),
    // dpb only the 17408-wide gate/up output.
    float* dpa = nullptr;
    float* dpb = nullptr;
    int  ready = 0, n_draft = 0, n_hit = 0, last_draft = -1;
    MtpState() { for (int a = 0; a < 8; ++a) for (int b = 0; b < 16; ++b) q_iss[a][b] = -1; }
    // PER-CHAIN-POSITION acceptance. Draft at chain position j, issued at target step p, predicts
    // t_{p+1+j}; it is therefore resolved at step p+j. The scoreboard requires the rate AT EACH
    // POSITION, not just the aggregate, because later positions are strictly harder.
    int  ch_n[8] = {0}, ch_hit[8] = {0};
    // RING keyed by RESOLVING STEP, not by chain level. A per-level slot is overwritten on the very
    // next step (level j registered at step p with cmp p+j is clobbered at p+1 before it can resolve),
    // which is why chain levels 2 and 3 never accumulated a single sample. Exactly one draft is
    // scheduled to resolve at each future step, so step%8 addresses them without collision.
    // 32 slots, not 8: three levels are written per step, so an 8-slot ring reuses each slot every
    // ~2.7 steps -- shorter than the T-1=3 steps a level-3 entry must survive, which is why levels 2
    // and 3 never accumulated samples. 32 > 4x the max in-flight depth.
    // PER-LEVEL QUEUE keyed by issuing step. A step-indexed ring cannot work here: every step issues
    // all T-1 chain levels, so the slots for p+1, p+2, p+3 are overwritten by the very next step no
    // matter how large the ring is -- which is why levels 2 and 3 never resolved. Indexing by
    // (level, issuing_step mod 16) and validating with the stored issuing step removes the collision.
    int  q_tok[8][16];
    int  q_iss[8][16];
};
static MtpState g_mtp[Q27_MAX_DEVICES];
static int g_mtp_on = 0;
static int g_spec = 0;      // Q27_SPEC=1: speculative decoding (MTP draft + row-batched verify)
static int g_spec_k = 2;    // Q27_SPEC_K: drafts per round (chained through the MTP layer); verify rows = K+1
// fc is [5120, 10240]: the concat order of (norm_hidden, norm_embedding) is a property of the
// checkpoint's own module and is NOT recoverable from shapes. Q27_MTP_SWAP=1 tries the other
// order; the acceptance rate decides, which is the only honest way to settle it.
static int g_mtp_swap = 0;
// The MTP weights arrive BF16, so their activation scale is NOT in the checkpoint the way the
// quantized tensors' scales are: it has to be chosen. 0.02 was a guess and it governs BOTH the
// quantizer and the handle, so a wrong value scales every draft projection. Sweepable because a
// guess that gates the acceptance rate is not a parameter, it is a hypothesis.
static float g_mtp_is = 0.02f;
// Stage trace: RMS after each stage of the draft on card 0, first draft only. A wiring error shows
// as the FIRST stage being wrong; a scale error as a trajectory that diverges at one named stage.
static int g_mtp_dbg = 0;
static int g_mtp_dbg_used = 0;
static inline int mtp_hid_off() { return g_mtp_swap ? Q27_HID : 0; }
static inline int mtp_emb_off() { return g_mtp_swap ? 0 : Q27_HID; }

static int  q27_mtp_upload(q27_model_t* m, MtpW* W, int id);
static q27_nvfp4_t g_lmref;   // a copy of the engine-built lm_head handle, for field A/B
static int  mtp_init(q27_model_t* m, int ndev);
static void mtp_draft_step(Dev& d, MtpState& S, q27_globals_t* G, int dpos, int stash, const unsigned short* hprev = nullptr, int head = 1);
// Golden-input test of the draft head in isolation: feed it the ORACLE's own final hidden state and
// a specified token id, and see which token it predicts. This bypasses the stash/publish plumbing
// AND the main model's numerics, so it separates 'the head is wrong' from 'my plumbing is wrong'.
static void mtp_gold(q27_model_t* m, std::vector<Dev>& D, int ndev, const char* hfile, unsigned tok);
static int  g_gold_mode = 0;   // 1 = tail only (mtp.norm + lm_head), else the full head
// plus_one for the draft's internal norms (input/post_attention layernorms and the two pre_fc
// norms). mtp.norm is proven correct at plus_one=1 by the tail gold test, so if the layer is wrong
// this is the one structural parameter that has never been varied.
static int  g_mtp_po = 1;
static int  g_mtp_chain = 1;   // drafts per round (chain positions 0..T-1)

#endif
