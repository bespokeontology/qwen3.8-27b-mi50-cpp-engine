// =============================================================================================
// SINGLE RESIDENCY (Q27_LS_SR=1) -- included by q27_main.cpp after the speculative-round helpers.
//
// THE MODEL IS RESIDENT ONCE. Under Q27_LS_SR the layer-split residency (card g owns FULL layers,
// the map q27_ls_owned / Q27_LS_BLK=4 the prefill sweep already uses) is the ONLY residency: no
// q27_upload_tp, no 1/4-width shards, no head-quarter KV, no replicated draft layer. The MLP is the
// official block-scaled FP8 checkpoint (gate/up full width, down as its four K-slices so the shipped
// TP-slice consumer runs unchanged), the seven fp8 projections are the int8 per-64 side-table
// mirrors decode reads today (Q27_DEC_I8), built from the fp8 originals which are then released.
//
// DECODE RUNS SEQUENTIALLY THROUGH THE CARDS in 4-layer blocks, 0 -> 1 -> 2 -> 3 -> 0 ..., the same
// ring the sweep walks. Every kernel is the shipped one at the ndev=1 (full-width) geometry the
// attention/GDN dispatchers already instantiate; what disappears is the two host-staged collectives
// per layer, the 4-way head max and the x4 draft. The head (final norm, FULL lm_head, argmax, MTP
// draft, draft KV) lives on the card that owns layer 63.
//
// PREFILL keeps the sweep's kernel graph exactly, fed from a 2-slot TRANSIENT mirror ring per card:
// projections as per-row int8 copied out of the resident per-64 mirrors (today's numerics), the MLP
// as per-64 int8 converted from the resident FP8 and requantised in place to per-row for rocBLAS
// (the fast 1554-class path; from FP8 rather than NVFP4, which is a quality gain).
//
// Record: SINGLE_RESIDENCY_DESIGN_20260917.md (Build Law directory). Everything here is behind
// g_sr; with the flag unset no line of this file runs.
// =============================================================================================
#include "q27_sr.h"

extern "C" int q27_proj_fp8b_nrs(const q27_fp8_t* w, const signed char* xq, int xst, const float* xs,
                                 float* y, int ystride, const unsigned short* radd, int rstride,
                                 float rs, int NR, hipStream_t s);   // q27_fp8.hip: block-scaled consumer with an explicit activation row stride
extern "C" int q27_i8g64_rowify(q27_i8g_t* src, q27_i8r_t* dst, float* sgrp, hipStream_t st);
extern "C" int q27_fp8_freed(void);
extern "C" int q27_sr_proj_i8g_rows(const q27_i8g_t* m, const signed char* xq, int xst, const float* xs, void* y, int ystride,
                                    const unsigned short* radd, int rstride, float rs, int NR, int outbf, hipStream_t s);   // q27_fp8.hip: the int8 per-64 row consumer, exported

static Dev* g_sr_D = nullptr;      // the per-card table: hops address the peers' buffers
static int  g_sr_ndev = 0;
static int  g_sr_rb = 1;           // Q27_SR_MLP_RB=1 (default): prefill MLP on the rocBLAS per-row path (the fast configuration)
static int  g_sr_time = 0;         // Q27_SR_TIME=1: per-card GPU event timing of the sequential round (Q27_SRT line per request)

static inline int sr_head() { return g_sr_head; }

// ---- ring geometry (fail closed: the decode chain assumes owner(b) = b % ndev) --------------
static void sr_check_geometry(int ndev) {
    const int LPP = Q27_LAYERS / ndev;
    int blk = q27_env_int("Q27_LS_BLK", 2); if (blk < 1 || blk > LPP || (LPP % blk)) blk = LPP;
    if (blk != 4 || q27_env_int("Q27_LS_BAL", 0) != 0 || ndev != 4) {
        std::fprintf(stderr, "FATAL Q27_LS_SR: the sequential decode chain is derived for Q27_LS_BLK=4, "
                             "Q27_LS_BAL=0, ndev=4 (got blk=%d bal=%d ndev=%d)\n",
                     blk, q27_env_int("Q27_LS_BAL", 0), ndev);
        std::fflush(stderr); std::exit(1);
    }
    if (!q27_env_int("Q27_DEC_I8", 0)) {
        std::fprintf(stderr, "FATAL Q27_LS_SR: decode reads the int8 per-64 projection mirrors, which need Q27_DEC_I8=1\n");
        std::fflush(stderr); std::exit(1);
    }
    for (int L = 0; L < 4; ++L) if (!q27_ls_owned(L, 0, ndev)) { std::fprintf(stderr, "FATAL Q27_LS_SR: card 0 must own layer %d\n", L); std::exit(1); }
    if (!q27_ls_owned(Q27_LAYERS - 1, ndev - 1, ndev)) { std::fprintf(stderr, "FATAL Q27_LS_SR: the last card must own layer 63\n"); std::exit(1); }
}

// ---- per-card SR buffers (after dev_alloc_tp) ------------------------------------------------
static void sr_alloc_card(Dev& d, int ndev, int K) {
    CK(hipSetDevice(d.id));
    auto A = [&](void** p, size_t b) {
        const hipError_t e = hipMalloc(p, b);
        if (e != hipSuccess) { size_t fr = 0, tt = 0; hipMemGetInfo(&fr, &tt);
            std::fprintf(stderr, "Q27_LS_SR alloc FAILED card %d: %zu B with %.3f GiB free\n", d.id, b, (double)fr / 1073741824.0);
            std::fflush(stderr); CK(e); }
        CK(hipMemset(*p, 0, b));
    };
    const size_t NRM = 8, HID = Q27_HID, IN = Q27_INTER;
    // ownership tables: local GDN index <-> global gdn slot
    d.sr_ngdn = 0;
    for (int gs = 0; gs < Q27_LAYERS; ++gs) d.sr_gdn_local[gs] = -1;
    for (int L = 0; L < Q27_LAYERS; ++L) {
        if (Q27_IS_FULL(L) || !q27_ls_owned(L, d.id, ndev)) continue;
        const int gs = L - (L >> 2);
        d.sr_gdn_gs[d.sr_ngdn] = gs; d.sr_gdn_local[gs] = d.sr_ngdn; ++d.sr_ngdn;
    }
    for (int v = 0; v < 4; ++v) CK(hipEventCreateWithFlags(&d.sr_ev[v], hipEventDisableTiming));
    for (int i = 0; i < 12; ++i) CK(hipEventCreate(&d.sr_te[i]));   // timing-capable (Q27_SR_TIME)
    for (int i = 0; i < 16 * 7; ++i) CK(hipEventCreate(&d.sr_ste[i]));
    d.sr_round = 0; d.sr_lidx = 0;
    // decode scratch at full width
    A((void**)&d.sr_xqw, NRM * IN);
    A((void**)&d.sr_xsw, NRM * (IN / 16) * 4);
    for (int c = 0; c < 4; ++c) A((void**)&d.sr_pd[c], NRM * HID * 4);
    A((void**)&d.sr_part,  NRM * HID * 4);
    A((void**)&d.sr_part2, NRM * HID * 4);
    // GDN snapshot banks for the rollback, FULL width for this card's OWN gdn layers (the TP round
    // sizes them at the quarter; here the live S is full width and so is the bank).
    if (K > 0 && d.sr_ngdn > 0) {
        if (d.nr_Ssnap) { CK(hipFree(d.nr_Ssnap)); d.nr_Ssnap = nullptr; }
        if (d.nr_csnap) { CK(hipFree(d.nr_csnap)); d.nr_csnap = nullptr; }
        A((void**)&d.nr_Ssnap, (size_t)K * d.sr_ngdn * d.tp_s * 4);
        A((void**)&d.nr_csnap, (size_t)K * d.sr_ngdn * d.tp_conv * 2);
    }
    // prefill transient mirror ring: 2 slots
    const size_t proj_rows_max = (size_t)((Q27_GDN_QKV + Q27_GDN_Z) > (Q27_QROWS + 2 * Q27_KVROWS) ? (Q27_GDN_QKV + Q27_GDN_Z) : (Q27_QROWS + 2 * Q27_KVROWS));   // 16384
    const size_t pw_bytes = proj_rows_max * HID + (size_t)HID * Q27_GDN_Z;   // stacked in-projections + o/out_proj
    const size_t prs_n    = proj_rows_max + HID;
    const size_t mw_bytes = 2 * IN * HID + HID * IN;
    const size_t ms_n     = 2 * IN * (HID / 64) + HID * (IN / 64);
    const size_t mrs_n    = 2 * IN + HID;
    for (int sl = 0; sl < 2; ++sl) {
        A((void**)&d.sr_pw[sl], pw_bytes);  A((void**)&d.sr_prs[sl], prs_n * 4);  d.sr_player[sl] = -1;
        A((void**)&d.sr_mw[sl], mw_bytes);  A((void**)&d.sr_ms[sl], ms_n * 4);    A((void**)&d.sr_mrs[sl], mrs_n * 4);  d.sr_mlayer[sl] = -1;
    }
    size_t fr = 0, tt = 0; CK(hipMemGetInfo(&fr, &tt));
    std::fprintf(stderr, "Q27_LS_SR card %d: %d owned gdn layers, mirror ring 2 x (%.1f + %.1f) MiB, decode scratch %.1f MiB, free after %.2f GiB\n",
                 d.id, d.sr_ngdn, (double)(pw_bytes + prs_n * 4) / 1048576.0, (double)(mw_bytes + ms_n * 4 + mrs_n * 4) / 1048576.0,
                 (double)(NRM * IN + NRM * (IN / 16) * 4 + 6 * NRM * HID * 4) / 1048576.0, (double)fr / 1073741824.0);
    std::fflush(stderr);
}

// ---- build the per-64 side-table mirrors on the LS handles, then release the fp8 originals ----
static void sr_build_mirrors(q27_model_t* m, int ndev) {
    size_t bytes = 0; int nt = 0, bad = 0;
    for (int gg = 0; gg < ndev; ++gg) {
        CK(hipSetDevice(gg));
        for (int L = 0; L < Q27_LAYERS; ++L) {
            if (!q27_ls_owned(L, gg, ndev)) continue;
            const q27_layer_t* lay = q27_layer_ls(m, L, gg);
            if (!lay) { ++bad; continue; }
            const q27_fp8_t* hs[4] = { nullptr, nullptr, nullptr, nullptr }; int nh = 0;
            if (lay->is_full) { hs[0] = &lay->q_proj; hs[1] = &lay->k_proj; hs[2] = &lay->v_proj; hs[3] = &lay->o_proj; nh = 4; }
            else              { hs[0] = &lay->in_qkv; hs[1] = &lay->in_z; hs[2] = &lay->out_proj; nh = 3; }
            for (int i = 0; i < nh; ++i) { if (q27_fp8_make_i8g(hs[i], 0)) { ++nt; bytes += (size_t)hs[i]->rows * hs[i]->K; } else ++bad; }
        }
        CK(hipDeviceSynchronize());
    }
    std::printf("Q27_LS_SR: %d int8 per-64 projection mirrors on the layer-split handles, %.2f GiB over %d cards (%d failed)\n",
                nt, (double)bytes * 1.0625 / 1073741824.0, ndev, bad);
    if (bad) { std::fprintf(stderr, "FATAL Q27_LS_SR: %d projection mirrors failed; the fp8 originals would be the only copy and they are about to be released\n", bad); std::fflush(stderr); std::exit(1); }
    q27_ls_sr_release_fp8(m, q27_fp8_i8g_rekey);
    q27_fp8_freed_set(1);
    for (int gg = 0; gg < ndev; ++gg) { size_t fr = 0, tt = 0; CK(hipSetDevice(gg)); CK(hipMemGetInfo(&fr, &tt));
        std::printf("Q27_VRAM after_sr_mirrors card %d: free %.2f GiB of %.2f GiB\n", gg, (double)fr / 1073741824.0, (double)tt / 1073741824.0); }
    CK(hipSetDevice(0));
}

// ---- prefill: the layer view the sweep reads (resident fields + this layer's transient mirrors) --
static const q27_layer_t* sr_layer_view(Dev& d, q27_model_t* m, int L, hipStream_t sk) {
    const int sl = L & 1;
    if (d.sr_player[sl] == L && d.sr_mlayer[sl] == L) return &d.sr_view[sl];
    const q27_layer_t* src = q27_layer_ls(m, L, d.id);
    if (!src) { std::fprintf(stderr, "Q27_LS_SR card %d: layer %d is not owned here\n", d.id, L); std::fflush(stderr); std::exit(1); }
    q27_layer_t& V = d.sr_view[sl];
    V = *src;
    const size_t HID = Q27_HID, IN = Q27_INTER;
    // -- projections: per-64 resident -> per-row transient (today's rocBLAS numerics), stacked --
    signed char* pw = d.sr_pw[sl]; float* prs = d.sr_prs[sl];
    // wbyte: byte offset of the tensor's first row in the slot; srow: its first row-scale slot.
    // The stacked in-projections share one 5120-byte row pitch (q27_cat_rows needs them
    // contiguous); o/out_proj (K=6144) start after them at their own pitch.
    auto rowview = [&](const q27_fp8_t* h, q27_i8r_t* dst, size_t wbyte, size_t srow) -> bool {
        const q27_i8g_t* g8 = q27_fp8_i8g(h);
        if (!g8 || !g8->w || !g8->s) return false;
        if (!q27_sr_i8g64_to_i8row(g8, pw + wbyte, (size_t)g8->K, prs + srow, sk)) return false;
        *dst = q27_i8r_t{}; dst->w = pw + wbyte; dst->s = prs + srow; dst->alpha = g8->alpha;
        dst->rows = g8->rows; dst->K = g8->K; dst->ng = 1; dst->gs = g8->K;
        return true;
    };
    bool ok = true;
    if (V.is_full) {
        const size_t rq = Q27_QROWS, rk = Q27_KVROWS, in_rows = rq + 2 * rk;
        ok &= rowview(&src->q_proj, &V.q8r, 0, 0);
        ok &= rowview(&src->k_proj, &V.k8r, rq * HID, rq);
        ok &= rowview(&src->v_proj, &V.v8r, (rq + rk) * HID, rq + rk);
        ok &= rowview(&src->o_proj, &V.o8r, in_rows * HID, in_rows);
    } else {
        const size_t rqkv = Q27_GDN_QKV, rz = Q27_GDN_Z, in_rows = rqkv + rz;
        ok &= rowview(&src->in_qkv,   &V.iqkv8r, 0, 0);
        ok &= rowview(&src->in_z,     &V.iz8r,   rqkv * HID, rqkv);
        ok &= rowview(&src->out_proj, &V.op8r,   in_rows * HID, in_rows);
    }
    if (!ok) { std::fprintf(stderr, "Q27_LS_SR card %d: projection mirror view failed for layer %d\n", d.id, L); std::fflush(stderr); std::exit(1); }
    d.sr_player[sl] = L;
    // -- MLP: block-scaled FP8 resident -> per-64 transient -> per-row in place (rocBLAS) --
    if (!src->f8_gate.w || !src->f8_up.w || !src->f8_dn[0].w) { std::fprintf(stderr, "Q27_LS_SR card %d: layer %d has no FP8 MLP planes\n", d.id, L); std::fflush(stderr); std::exit(1); }
    signed char* mw = d.sr_mw[sl]; float* ms = d.sr_ms[sl]; float* mrs = d.sr_mrs[sl];
    const size_t gw = IN * HID, gs_ = IN * (HID / 64);
    q27_i8g_t g8{}, u8{}, dn8{};
    const q27_fp8_t* dn4[4] = { &src->f8_dn[0], &src->f8_dn[1], &src->f8_dn[2], &src->f8_dn[3] };
    if (!q27_sr_fp8bs_to_i8g64_into(&src->f8_gate, &g8, mw, ms, sk) ||
        !q27_sr_fp8bs_to_i8g64_into(&src->f8_up,   &u8, mw + gw, ms + gs_, sk) ||
        !q27_sr_fp8bs4_to_i8g64_into(dn4, &dn8, mw + 2 * gw, ms + 2 * gs_, sk)) {
        std::fprintf(stderr, "Q27_LS_SR card %d: FP8 -> int8/64 MLP expand failed for layer %d\n", d.id, L); std::fflush(stderr); std::exit(1);
    }
    V.gate8 = g8; V.up8 = u8; V.down8 = dn8;
    if (g_sr_rb) {
        if (!q27_i8g64_rowify(&V.gate8, &V.gate8r, mrs, sk) ||
            !q27_i8g64_rowify(&V.up8,   &V.up8r,   mrs + IN, sk) ||
            !q27_i8g64_rowify(&V.down8, &V.down8r, mrs + 2 * IN, sk)) {
            std::fprintf(stderr, "Q27_LS_SR card %d: MLP rowify failed for layer %d\n", d.id, L); std::fflush(stderr); std::exit(1);
        }
    } else { V.gate8r = q27_i8r_t{}; V.up8r = q27_i8r_t{}; V.down8r = q27_i8r_t{}; }
    // the NVFP4 MLP handles are not resident under SR; the sweep still reads their in_scale
    V.gate.in_scale = src->f8_gate.in_scale; V.up.in_scale = src->f8_up.in_scale; V.down.in_scale = src->f8_dn[0].in_scale;
    d.sr_mlayer[sl] = L;
    return &V;
}

// ---- decode: one FULL-WIDTH layer on its owner card for NR rows -----------------------------
// pend_in: fp32 [NR][HID] pending residual from the previous layer on THIS card (folded into the
// input norm), or nullptr when the rows arrived through the hop / the embedding. pend_out receives
// the MLP output (fp32 [NR][HID]); the caller folds it into the next layer or adds it at the block end.
static void run_layer_sr_nr(Dev& d, const q27_layer_t* L, int pos0, int NR, const float* pend_in, float** pend_out,
                            int fsl, int gsl_global, int gsl_local) {
    hipStream_t s = d.stream;
    const int HID = Q27_HID, QS = Q27_QROWS + 2 * Q27_KVROWS, OR = Q27_OROWS, ZL = Q27_GDN_Z, QKVL = Q27_GDN_QKV, VH = Q27_GDN_VH, IN = Q27_INTER;
    const int MIXL = (Q27_GDN_Z > Q27_OROWS ? Q27_GDN_Z : Q27_OROWS);
    unsigned short* const hid = d.hidden_slots;
    const float in_s = L->is_full ? L->q_proj.in_scale : L->in_qkv.in_scale;
    // Q27_SR_TIME=2: stage boundaries for this layer (local index within the round)
    hipEvent_t* const ste = (g_sr_time >= 2 && d.sr_lidx < 16) ? d.sr_ste + d.sr_lidx * 7 : nullptr;
    auto ST = [&](int k) { if (ste) CK(hipEventRecord(ste[k], s)); };
    ST(0);
    // input norm (+ pending residual): bf16 residual rows in, natural int8 activations + bf16 norm out
    for (int r = 0; r < NR; ++r) {
        if (!q27_rmsnorm_quant_fp8(hid + (size_t)r * HID, L->input_norm, d.nr_norm + (size_t)r * HID,
                                   d.nr_xq + (size_t)r * HID, d.nr_xs + (size_t)r * (HID / 16), HID, 1, in_s,
                                   pend_in ? (const void*)(pend_in + (size_t)r * HID) : nullptr, 0, s)) {
            if (pend_in) q27_add_inplace(hid + (size_t)r * HID, pend_in + (size_t)r * HID, HID, s);
            q27_rmsnorm(hid + (size_t)r * HID, L->input_norm, d.nr_norm + (size_t)r * HID, HID, 1, s);
            q27_quant_fp8(d.nr_norm + (size_t)r * HID, d.nr_xq + (size_t)r * HID, d.nr_xs + (size_t)r * (HID / 16), HID, in_s, s);
        }
    }
    float* const part = d.sr_part;
    if (L->is_full) {
        nr_chunks(NR, [&](int c0, int cn) {
            if (!q27_proj_fp8_bf16_nr(&L->q_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS, QS, cn, s) ||
                !q27_proj_fp8_bf16_nr(&L->k_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS + Q27_QROWS, QS, cn, s) ||
                !q27_proj_fp8_bf16_nr(&L->v_proj, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkva + (size_t)c0 * QS + Q27_QROWS + Q27_KVROWS, QS, cn, s)) {
                std::fprintf(stderr, "Q27_LS_SR: q/k/v declined (L=%d NR=%d)\n", L->layer, cn); std::abort(); }
        });
        signed char* kc = d.kc + d.kv_off[fsl];   q27_kvs_t* ks = d.ks + d.kvs_off[fsl];
        signed char* vc = d.vc + Q27_VOFF(d.kv_off[fsl]);   q27_kvs_t* vs = d.vs + Q27_VSOFF(d.kvs_off[fsl]);
        ST(1);
        nr_chunks(NR, [&](int c0, int cn) {
            q27_attn_prep_tp_nr(d.nr_qkva + (size_t)c0 * QS, QS, L->q_norm, L->k_norm, kc, ks, vc, vs,
                                d.nr_qh + (size_t)c0 * OR, OR, pos0 + c0, cn, 1, Q27_KVROWS, 0, s);
            q27_attn_decode_tp_nr(d.nr_qh + (size_t)c0 * OR, OR, kc, ks, vc, vs, d.nr_qkva + (size_t)c0 * QS, QS,
                                  d.nr_mix6 + (size_t)c0 * MIXL, MIXL, pos0 + c0, cn, 1, Q27_KVROWS, 0, s);
        });
        ST(2);
        nr_chunks(NR, [&](int c0, int cn) {
            q27_quant_fp8_b(d.nr_mix6 + (size_t)c0 * MIXL, (size_t)MIXL, d.sr_xqw + (size_t)c0 * OR, (size_t)OR,
                            d.sr_xsw + (size_t)c0 * (OR / 16), (size_t)(OR / 16), OR, L->o_proj.in_scale, cn, s);
            const q27_i8g_t* mo = q27_fp8_i8g(&L->o_proj);
            if (!mo || !q27_sr_proj_i8g_rows(mo, d.sr_xqw + (size_t)c0 * OR, OR, d.sr_xsw + (size_t)c0 * (OR / 16),
                                             (void*)(part + (size_t)c0 * HID), HID, nullptr, 0, 0.f, cn, 0, s)) {
                std::fprintf(stderr, "Q27_LS_SR: o_proj declined (L=%d NR=%d mirror=%p)\n", L->layer, cn, (const void*)mo); std::abort(); }
        });
    } else {
        nr_chunks(NR, [&](int c0, int cn) {
            if (!q27_proj_fp8_bf16_nr(&L->in_qkv, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_qkv + (size_t)c0 * QKVL, QKVL, cn, s) ||
                !q27_proj_fp8_bf16_nr(&L->in_z,   d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_zbuf + (size_t)c0 * ZL, ZL, cn, s)) {
                std::fprintf(stderr, "Q27_LS_SR: in_qkv/in_z declined (L=%d NR=%d)\n", L->layer, cn); std::abort(); }
            q27_bf16_gemv2_tile(L->in_a, L->in_b, d.nr_norm + (size_t)c0 * HID, HID,
                                d.nr_ab + (size_t)c0 * VH, d.nr_bb + (size_t)c0 * VH, VH, HID, cn, s);
        });
        unsigned short* convp = d.conv + d.conv_off[gsl_global];
        float* Sp = d.S + d.s_off[gsl_global];                        // full width on the owner: no quarter offset
        const int NGL = d.sr_ngdn;
        ST(1);
        nr_chunks(NR, [&](int c0, int cn) {
            q27_gdn_conv_tp_tile2(d.nr_qkv + (size_t)c0 * QKVL, QKVL, cn, convp, d.conv_wide[gsl_global], L->conv1d, 0, 1,
                                  d.nr_csnap + (size_t)(gsl_local + c0 * NGL) * d.tp_conv, (size_t)NGL * d.tp_conv, (c0 + cn < NR) ? 1 : 0, s);
            q27_gdn_scan_tp_snap(d.nr_qkv + (size_t)c0 * QKVL, QKVL, d.nr_zbuf + (size_t)c0 * ZL, ZL,
                                 d.nr_ab + (size_t)c0 * VH, d.nr_bb + (size_t)c0 * VH, VH,
                                 L->A_log, L->dt_bias, L->gdn_norm, Sp, d.nr_mix6 + (size_t)c0 * MIXL, MIXL, 0, 1, cn,
                                 nullptr, nullptr, 0.0f, d.nr_Ssnap + (size_t)(gsl_local + c0 * NGL) * d.tp_s, (size_t)NGL * d.tp_s, (c0 + cn < NR) ? 1 : 0, s);
        });
        ST(2);
        nr_chunks(NR, [&](int c0, int cn) {
            q27_quant_fp8_b(d.nr_mix6 + (size_t)c0 * MIXL, (size_t)MIXL, d.sr_xqw + (size_t)c0 * ZL, (size_t)ZL,
                            d.sr_xsw + (size_t)c0 * (ZL / 16), (size_t)(ZL / 16), ZL, L->out_proj.in_scale, cn, s);
            const q27_i8g_t* mo = q27_fp8_i8g(&L->out_proj);
            if (!mo || !q27_sr_proj_i8g_rows(mo, d.sr_xqw + (size_t)c0 * ZL, ZL, d.sr_xsw + (size_t)c0 * (ZL / 16),
                                             (void*)(part + (size_t)c0 * HID), HID, nullptr, 0, 0.f, cn, 0, s)) {
                std::fprintf(stderr, "Q27_LS_SR: out_proj declined (L=%d NR=%d mirror=%p)\n", L->layer, cn, (const void*)mo); std::abort(); }
        });
    }
    // post norm (+ the mixer output as the pending residual): natural int8 for the FP8 gate/up
    ST(3);
    for (int r = 0; r < NR; ++r) {
        if (!q27_rmsnorm_quant_fp8(hid + (size_t)r * HID, L->post_norm, d.nr_norm + (size_t)r * HID,
                                   d.nr_xq + (size_t)r * HID, d.nr_xs + (size_t)r * (HID / 16), HID, 1, L->f8_gate.in_scale,
                                   (const void*)(part + (size_t)r * HID), 0, s)) {
            q27_add_inplace(hid + (size_t)r * HID, part + (size_t)r * HID, HID, s);
            q27_rmsnorm(hid + (size_t)r * HID, L->post_norm, d.nr_norm + (size_t)r * HID, HID, 1, s);
            q27_quant_fp8(d.nr_norm + (size_t)r * HID, d.nr_xq + (size_t)r * HID, d.nr_xs + (size_t)r * (HID / 16), HID, L->f8_gate.in_scale, s);
        }
    }
    float* const part2 = d.sr_part2;
    nr_chunks(NR, [&](int c0, int cn) {
        if (!q27_proj_fp8b_nr(&L->f8_gate, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_pa + (size_t)c0 * IN, IN, nullptr, 0, 0.f, cn, s) ||
            !q27_proj_fp8b_nr(&L->f8_up,   d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16), d.nr_pb + (size_t)c0 * IN, IN, nullptr, 0, 0.f, cn, s)) {
            std::fprintf(stderr, "Q27_LS_SR: FP8 gate/up declined (L=%d rows=%d K=%d NR=%d)\n", L->layer, L->f8_gate.rows, L->f8_gate.K, cn); std::abort(); }
        if (c0 == 0) ST(4);
        if (!q27_sr_swiglu_quant_nat_b(d.nr_pa + (size_t)c0 * IN, d.nr_pb + (size_t)c0 * IN, d.sr_xqw + (size_t)c0 * IN,
                                       d.sr_xsw + (size_t)c0 * (IN / 16), IN, L->f8_dn[0].in_scale, cn, s)) {
            std::fprintf(stderr, "Q27_LS_SR: natural swiglu declined (L=%d)\n", L->layer); std::abort(); }
        for (int c = 0; c < 4; ++c) {
            if (!q27_proj_fp8b_nrs(&L->f8_dn[c], d.sr_xqw + (size_t)c0 * IN + (size_t)c * (IN / 4), IN,
                                   d.sr_xsw + (size_t)c0 * (IN / 16) + (size_t)c * (IN / 64),
                                   d.sr_pd[c] + (size_t)c0 * HID, HID, nullptr, 0, 0.f, cn, s)) {
                std::fprintf(stderr, "Q27_LS_SR: FP8 down slice %d declined (L=%d rows=%d K=%d NR=%d)\n", c, L->layer, L->f8_dn[c].rows, L->f8_dn[c].K, cn); std::abort(); }
        }
        if (c0 == 0) ST(5);
        if (!q27_sr_sum4_f32(part2 + (size_t)c0 * HID, d.sr_pd[0] + (size_t)c0 * HID, d.sr_pd[1] + (size_t)c0 * HID,
                             d.sr_pd[2] + (size_t)c0 * HID, d.sr_pd[3] + (size_t)c0 * HID, (size_t)cn * HID, s)) {
            std::fprintf(stderr, "Q27_LS_SR: down sum declined\n"); std::abort(); }
    });
    ST(6);
    if (ste) d.sr_lidx++;
    *pend_out = part2;
}

// ---- decode: one speculative round, sequential through the cards ----------------------------
// All four threads run this in lockstep: two barrier pairs per round (draft publish, token publish)
// plus the hop chain. The chain is the sweep's protocol: a monotonic per-card publish counter the
// consumer spins on, then a device event so the copy has landed. Events are recorded BEFORE the
// counter is published, so a consumer can never wait on a stale (previous-round) record.
static int spec_round_sr(Dev& d, q27_model_t* m, const unsigned short* emb, int nlayer, int pos, unsigned tok,
                         int g, int ndev, TpColl& C, double* mark, unsigned* outtok, int K) {
    hipStream_t s = d.stream;
    const double sp0 = tp_now(); double spa = sp0;
    d.nr_eslot = 0;
    auto lap = [&](int k) { const double t = tp_now(); d.sp_t[k] += t - spa; spa = t; };
    const int head = g_sr_head;
    const int NR = K + 1, HID = Q27_HID;
    const q27_globals_t* G = q27_globals(m, d.id);
    Dev* const D = g_sr_D;
    (void)nlayer;
    // ---- 1. drafts: the head card only; K-1 chained single-row drafts after the catch-up pass ----
    unsigned dt[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
    const bool tm = g_sr_time != 0;
    if (g == head && K > 0) {
        MtpState& S = g_mtp[g];
        const unsigned short* hchain = nullptr;
        if (tm) CK(hipEventRecord(d.sr_te[4], s));
        for (int j = 0; j < K; ++j) {
            if (j == 0) {
                const unsigned short* hrows[8]; unsigned toks[8]; int nrm = 0;
                for (int c = 0; c < d.nr_ncatch; ++c) { hrows[nrm] = d.nr_hcatch + (size_t)c * HID; toks[nrm] = d.nr_catch_tok[c]; ++nrm; }
                hrows[nrm] = d.nr_hprev; toks[nrm] = tok; ++nrm;
                const int dpos0 = (d.nr_ncatch ? d.nr_catch_pos[0] : pos - 1) - d.nr_mtp_base;
                mtp_pass_nr(d, S, G, nrm, hrows, toks, emb, dpos0, nrm - 1);
                d.nr_ncatch = 0;
                hchain = g_mtpnr[d.id].nrm + (size_t)(nrm - 1) * HID;
            } else {
                const unsigned short* hr[1] = { hchain }; const unsigned tk[1] = { dt[j - 1] };
                mtp_pass_nr(d, S, G, 1, hr, tk, emb, pos - 1 - d.nr_mtp_base + j, 0);
                hchain = g_mtpnr[d.id].nrm;
            }
            q27_argmax_val(S.dpa, d.nr_tok, d.nr_val, G->lm_head.rows, s);
            if (tm && j == K - 1) CK(hipEventRecord(d.sr_te[5], s));
            CK(hipMemcpyAsync(d.nr_mail_h, d.nr_tok, 64, hipMemcpyDeviceToHost, s));
            CK(hipStreamSynchronize(s));
            dt[j] = ((const unsigned*)d.nr_mail_h)[0];
        }
        for (int j = 0; j < K; ++j) C.sr_dt[j] = dt[j];
    }
    lap(0);
    C.b1.wait();
    C.b2.wait();
    for (int j = 0; j < K; ++j) dt[j] = C.sr_dt[j];
    lap(1);
    // ---- 2. verify: [tok @ pos, dt[0] @ pos+1, ...] through the ring ----
    if (g == 0) for (int r = 0; r < NR; ++r) spec_emb_h2d(d, d.hidden_slots + (size_t)r * HID, emb, (r == 0) ? tok : dt[r - 1], s);
    d.sr_lidx = 0;
    lap(2);
    const long base = d.sr_round * 4;
    double t_hop = 0.0;
    unsigned nx[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
    for (int v = 0; v < 4; ++v) {
        const int b = g + 4 * v;                        // this card's v-th block this round
        if (b > 0) {
            const int src = (b - 1) % ndev, vsrc = (b - 1) / ndev;
            const long expect = base + vsrc + 1;
            const double th0 = tp_now();
            while (C.sr_pub[src].load(std::memory_order_acquire) < expect) { }
            t_hop += tp_now() - th0;
            CK(hipStreamWaitEvent(s, D[src].sr_ev[vsrc], 0));
        }
        if (tm) CK(hipEventRecord(d.sr_te[v], s));   // block start: executes once the incoming hop has landed
        const float* pend = nullptr;
        for (int L = 4 * b; L < 4 * b + 4; ++L) {
            const q27_layer_t* lay = q27_layer_ls(m, L, d.id);
            if (!lay) { std::fprintf(stderr, "FATAL Q27_LS_SR: card %d does not hold layer %d\n", d.id, L); std::fflush(stderr); std::abort(); }
            float* pend_out = nullptr;
            const int fsl = L >> 2, gsg = L - (L >> 2);
            run_layer_sr_nr(d, lay, pos, NR, pend, &pend_out, fsl, gsg, Q27_IS_FULL(L) ? -1 : d.sr_gdn_local[gsg]);
            pend = pend_out;
        }
        for (int r = 0; r < NR; ++r) q27_add_inplace(d.hidden_slots + (size_t)r * HID, pend + (size_t)r * HID, HID, s);
        if (tm) CK(hipEventRecord(d.sr_te[8 + v], s));   // block end: this card's compute for the visit is done
        if (b + 1 < 4 * ndev) {
            const int next = (b + 1) % ndev;
            CK(hipMemcpyPeerAsync(D[next].hidden_slots, D[next].id, d.hidden_slots, d.id, (size_t)NR * HID * 2, s));
            CK(hipEventRecord(d.sr_ev[v], s));
            C.sr_pub[g].store(base + v + 1, std::memory_order_release);
        } else {
            // ---- head: final norm, FULL lm_head, per-row argmax ----
            if (tm) CK(hipEventRecord(d.sr_te[6], s));
            for (int r = 0; r < NR; ++r) q27_rmsnorm(d.hidden_slots + (size_t)r * HID, G->final_norm, d.nr_norm + (size_t)r * HID, HID, 1, s);
            for (int r = 0; r < NR; ++r) q27_quant_perm(d.nr_norm + (size_t)r * HID, d.nr_xq + (size_t)r * HID, d.nr_xs + (size_t)r * (HID / 16), HID, G->lm_head.in_scale, s);
            nr_chunks(NR, [&](int c0, int cn) {
                if (!q27_proj_nvfp4_nr(&G->lm_head, d.nr_xq + (size_t)c0 * HID, d.nr_xs + (size_t)c0 * (HID / 16),
                                       d.nr_pa + (size_t)c0 * G->lm_head.rows, G->lm_head.rows, cn, s)) {
                    std::fprintf(stderr, "Q27_LS_SR: lm_head declined\n"); std::abort(); }
            });
            for (int r = 0; r < NR; ++r) q27_argmax_val(d.nr_pa + (size_t)r * G->lm_head.rows, d.nr_tok + r, d.nr_val + r, G->lm_head.rows, s);
            if (tm) CK(hipEventRecord(d.sr_te[7], s));
            CK(hipMemcpyAsync(d.nr_mail_h, d.nr_tok, 64, hipMemcpyDeviceToHost, s));
            CK(hipStreamSynchronize(s));
            for (int r = 0; r < NR; ++r) nx[r] = ((const unsigned*)d.nr_mail_h)[r];
            int nacc = 0; while (nacc < K && nx[nacc] == dt[nacc]) ++nacc;
            for (int r = 0; r < NR; ++r) C.sr_nx[r] = nx[r];
            C.sr_nacc = nacc;
        }
    }
    g_coll_nr_eff[g] = NR;
    lap(3);
    d.sp_t[4] += t_hop;
    C.b1.wait();
    C.b2.wait();
    for (int r = 0; r < NR; ++r) nx[r] = C.sr_nx[r];
    const int nacc = C.sr_nacc;
    lap(5);
    // ---- 3. rollback: rows beyond nacc advanced this card's recurrent state with wrong tokens ----
    const int ncommit = nacc + 1;
    for (int j = 0; j < ncommit; ++j) outtok[j] = nx[j];
    d.nr_drafts++; d.nr_hits += nacc;
    for (int j = 1; j <= nacc; ++j) d.nr_acc_hist[j]++;
    if (nacc < K && d.sr_ngdn > 0) {
        const float* Sb = d.nr_Ssnap + (size_t)nacc * d.sr_ngdn * d.tp_s;
        const unsigned short* Cb = d.nr_csnap + (size_t)nacc * d.sr_ngdn * d.tp_conv;
        for (int jl = 0; jl < d.sr_ngdn; ++jl) {
            const int gs = d.sr_gdn_gs[jl];
            CK(hipMemcpyAsync(d.S + d.s_off[gs], Sb + (size_t)jl * d.tp_s, d.tp_s * 4, hipMemcpyDeviceToDevice, s));
            CK(hipMemcpyAsync(d.conv + d.conv_off[gs], Cb + (size_t)jl * d.tp_conv, d.conv_bytes[gs], hipMemcpyDeviceToDevice, s));
        }
    }
    // ---- 4. hand-off on the head: the draft's next hidden + the catch-up rows ----
    if (g == head) {
        for (int j = 0; j < nacc; ++j) {
            CK(hipMemcpyAsync(d.nr_hcatch + (size_t)j * HID, d.hidden_slots + (size_t)j * HID, (size_t)HID * 2, hipMemcpyDeviceToDevice, s));
            d.nr_catch_tok[j] = nx[j]; d.nr_catch_pos[j] = pos + j;
        }
        d.nr_ncatch = nacc;
        CK(hipMemcpyAsync(d.nr_hprev, d.hidden_slots + (size_t)nacc * HID, (size_t)HID * 2, hipMemcpyDeviceToDevice, s));
    }
    d.sr_round++;
    if (tm) {   // every block event of every card has completed by now: the head synced after the last block, which depended on all the others
        float ms = 0.f;
        for (int v = 0; v < 4; ++v) if (hipEventElapsedTime(&ms, d.sr_te[v], d.sr_te[8 + v]) == hipSuccess) d.sr_ms_blk[v] += ms;
        if (g == head && K > 0 && hipEventElapsedTime(&ms, d.sr_te[4], d.sr_te[5]) == hipSuccess) d.sr_ms_draft += ms;
        if (g == head && hipEventElapsedTime(&ms, d.sr_te[6], d.sr_te[7]) == hipSuccess) d.sr_ms_head += ms;
        if (g_sr_time >= 2) for (int li = 0; li < d.sr_lidx && li < 16; ++li) for (int k = 0; k < 6; ++k)
            if (hipEventElapsedTime(&ms, d.sr_ste[li * 7 + k], d.sr_ste[li * 7 + k + 1]) == hipSuccess) d.sr_ms_st[k] += ms;
        d.sr_ms_wall += tp_now() - sp0; d.sr_rounds_t++;
    }
    lap(6); d.sp_t[7] += tp_now() - sp0;
    *mark = tp_now();
    if (g == head && (d.nr_drafts % 16) == 0) {
        std::printf("Q27_SPEC rounds=%ld accepted=%ld per-round=%.3f", d.nr_drafts, d.nr_hits, (double)d.nr_hits / (double)d.nr_drafts);
        for (int j = 1; j <= K; ++j) std::printf("  P(>=%d)=%.3f", j, (double)d.nr_acc_hist[j] / d.nr_drafts);
        std::printf("\n");
    }
    return ncommit;
}
