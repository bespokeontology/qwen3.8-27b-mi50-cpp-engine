// q27_load.cpp — safetensors checkpoint loader for the Qwen3.8-27B native gfx906 engine.
//
// Reads <model_dir>/model-0000{0,1,2}-of-00003.safetensors directly:
//   [8 bytes little-endian header length N][N bytes JSON header][payload]
// The JSON header maps  name -> {dtype, shape, data_offsets:[begin,end]} with the offsets
// relative to the start of the payload. A hand-written scanner reads it; there is no JSON
// library dependency and no Python anywhere on this path.
//
// WHAT IS LOADED AND WHAT IS NOT
//   The census gate is over the WHOLE file set: exactly 2194 tensors and 21,921,428,072 payload
//   bytes, else q27_open fails closed. But only the text trunk is ever uploaded:
//
//     model.visual.*   921,460,192 B (878.77 MiB)  vision tower  — SKIPPED
//     mtp.*            849,398,784 B (810.05 MiB)  MTP draft head — SKIPPED
//
//   Neither is needed for raw autoregressive text decode. The vision tower only turns image
//   patches into embeddings for multimodal prompts, and the MTP head is a speculative draft
//   model that the plain one-token-at-a-time decode loop never invokes. Skipping both takes the
//   resident set from 20.4159 GiB to 16.3985 GiB (layers 15.7325 + lm_head 0.6660 + final norm),
//   plus 2.3682 GiB of embed_tokens if you choose to keep the embedding table on the GPU.
//
// SHARDING
//   A single MI50 has 15.98 GiB, so 16.3985 GiB cannot fit on one card. q27_upload takes a layer
//   range [lo,hi) and a device; q27_upload_split wires up the intended 4 cards x 16 layers. The
//   arenas it allocates measure 6.3013 / 3.9331 / 3.9331 / 4.5992 GiB (card 0 carries
//   embed_tokens, card 3 the final norm and lm_head), leaving >= 9.6 GiB per card for the BF16
//   KV cache, the 144 MiB fp32 recurrent state and scratch.
//
// ALLOCATION SHAPE
//   One hipMalloc arena per q27_upload call, 256-byte aligned sub-allocation, one hipFree at
//   close. No per-tensor allocations, nothing allocated after the call returns (invariant: no
//   alloc after READY). Copies come straight out of the mmap in (shard, file offset) order so the
//   page cache is read sequentially, with madvise(MADV_WILLNEED) issued two tensors ahead of the
//   copy — the same "issue the loads, then wait" shape the estate measured on the GPU side.

#include "q27_load.h"
#include "q27_sr.h"
#include "../storage/q27_storage.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// ============================================================================================
// small helpers
// ============================================================================================
namespace {

int fail(char* err, size_t cap, int code, const char* fmt, ...) {
    if (err && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, cap, fmt, ap);
        va_end(ap);
    }
    return code;
}

struct DtEnt { const char* s; int code; int esz; };
const DtEnt kDt[] = {
    {"BF16", Q27_DT_BF16, 2},   {"F16", Q27_DT_F16, 2},   {"F32", Q27_DT_F32, 4},
    {"F64",  Q27_DT_F64,  8},   {"F8_E4M3", Q27_DT_F8_E4M3, 1}, {"F8_E5M2", Q27_DT_F8_E5M2, 1},
    {"U8",   Q27_DT_U8,   1},   {"I8",  Q27_DT_I8,  1},   {"U16", Q27_DT_U16, 2},
    {"I16",  Q27_DT_I16,  2},   {"U32", Q27_DT_U32, 4},   {"I32", Q27_DT_I32, 4},
    {"U64",  Q27_DT_U64,  8},   {"I64", Q27_DT_I64, 8},   {"BOOL", Q27_DT_BOOL, 1},
};

int dt_code(const std::string& s, int* esz) {
    for (size_t i = 0; i < sizeof(kDt) / sizeof(kDt[0]); ++i)
        if (s == kDt[i].s) { *esz = kDt[i].esz; return kDt[i].code; }
    *esz = 0;
    return Q27_DT_UNKNOWN;
}

const char* dt_name(int code) {
    for (size_t i = 0; i < sizeof(kDt) / sizeof(kDt[0]); ++i)
        if (kDt[i].code == code) return kDt[i].s;
    return "UNKNOWN";
}

// -------------------------------------------------------------------------------------------
// minimal JSON scanner, sized for a safetensors header: a flat object of objects whose values
// are strings, integer arrays and (inside __metadata__ only) strings. skip() is fully general so
// an unexpected field type cannot derail the parse.
// -------------------------------------------------------------------------------------------
struct Jp {
    const char* p;
    const char* e;
    const char* why;

    Jp(const char* b, const char* en) : p(b), e(en), why("") {}

    void ws() {
        while (p < e) {
            const char c = *p;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p; else break;
        }
    }
    bool eat(char c)    { ws(); if (p < e && *p == c) { ++p; return true; } return false; }
    bool expect(char c) { if (eat(c)) return true; why = "expected delimiter"; return false; }

    bool str(std::string& out) {
        ws();
        if (p >= e || *p != '"') { why = "expected string"; return false; }
        ++p;
        out.clear();
        while (p < e) {
            const unsigned char c = (unsigned char)*p++;
            if (c == '"') return true;
            if (c != '\\') { out.push_back((char)c); continue; }
            if (p >= e) { why = "truncated escape"; return false; }
            const char x = *p++;
            switch (x) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    if (e - p < 4) { why = "truncated \\u"; return false; }
                    unsigned v = 0;
                    for (int i = 0; i < 4; ++i) {
                        const char h = *p++;
                        v <<= 4;
                        if      (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                        else { why = "bad \\u hex"; return false; }
                    }
                    // Surrogate pairs are not recombined: every name in this checkpoint is ASCII,
                    // and a lone surrogate would only ever corrupt a name we do not look up.
                    if (v < 0x80) {
                        out.push_back((char)v);
                    } else if (v < 0x800) {
                        out.push_back((char)(0xC0u | (v >> 6)));
                        out.push_back((char)(0x80u | (v & 0x3Fu)));
                    } else {
                        out.push_back((char)(0xE0u | (v >> 12)));
                        out.push_back((char)(0x80u | ((v >> 6) & 0x3Fu)));
                        out.push_back((char)(0x80u | (v & 0x3Fu)));
                    }
                    break;
                }
                default: why = "bad escape"; return false;
            }
        }
        why = "unterminated string";
        return false;
    }

    // safetensors shapes and data_offsets are non-negative integers; reject anything else so a
    // float where an offset belongs is a hard error rather than a silent truncation.
    bool u64(long long& out) {
        ws();
        const char* s = p;
        if (p >= e || *p < '0' || *p > '9') { p = s; why = "expected integer"; return false; }
        unsigned long long v = 0;
        while (p < e && *p >= '0' && *p <= '9') {
            const unsigned long long d = (unsigned long long)(*p - '0');
            if (v > (0x7FFFFFFFFFFFFFFFull - d) / 10ull) { why = "integer overflow"; return false; }
            v = v * 10ull + d;
            ++p;
        }
        if (p < e && (*p == '.' || *p == 'e' || *p == 'E')) { p = s; why = "non-integer"; return false; }
        out = (long long)v;
        return true;
    }

    static bool numch(char c) {
        return (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E';
    }

    bool skip() {
        ws();
        if (p >= e) { why = "eof"; return false; }
        const char c = *p;
        if (c == '"') { std::string s; return str(s); }
        if (c == '{') {
            ++p;
            if (eat('}')) return true;
            for (;;) {
                std::string k;
                if (!str(k) || !expect(':') || !skip()) return false;
                if (eat(',')) continue;
                return expect('}');
            }
        }
        if (c == '[') {
            ++p;
            if (eat(']')) return true;
            for (;;) {
                if (!skip()) return false;
                if (eat(',')) continue;
                return expect(']');
            }
        }
        if (c == 't' || c == 'f' || c == 'n') {           // true / false / null
            while (p < e && *p >= 'a' && *p <= 'z') ++p;
            return true;
        }
        {
            const char* s = p;
            while (p < e && numch(*p)) ++p;
            if (p == s) { why = "unrecognised value"; return false; }
            return true;
        }
    }
};

}  // namespace

// The int8 MLP residency count, resolved ONCE for the run. -1 means "auto": the prefill slots grow
// with the prompt window (Q27_SERVE_SLOTS / the prompt length) and every card pays 30,720 B per
// position, so a long window eats exactly the memory the fast MLP wants. One int8 MLP layer costs
// ~122 MB net, i.e. the same as ~3960 positions of slot -- measured, not assumed: at 9,216 slots
// k=14 fits and k=15 does not, and a 29,000-slot window runs at k=8. Auto therefore spends what the
// current window leaves: 13 layers at the 9,216 floor, falling to 0 by ~60,000 slots (where prefill
// is attention-bound anyway). Explicit non-negative values override it and never change.
static int s_ls_mlp_i8k = -2;
void q27_ls_mlp_i8k_set(int k) { s_ls_mlp_i8k = k; }
// Q27_LS_MLP_I8K takes a SCALAR (same budget on every card) or a PER-CARD COMMA LIST
// e.g. "16,10,10,10". Per-card exists because the int8 mirrors and the prefill-sweep slots compete
// for the same VRAM, and the cards are NOT the same size. Measured 2026-09-19 on 32+16+16+16 with
// Q27_DEC_MLP_NV=0: at the 9216 slot floor the auto budget reaches k=16 on every card, but then a
// 22K window is REFUSED by the sweep and the turn falls back to decaying arena appends (the exact
// cost the 24576 floor exists to remove) -- while at 24576 slots the auto budget drops to k=10, so
// six layers per card lose their 8-bit mirror. One global budget is therefore set by the TIGHTEST
// card and throws the 32 GiB card's ~17 GiB of headroom away for nothing. A list lets card 0 keep
// every layer on int8 while the 16s take what they can afford AND the slot floor stays high.
// An explicit list always wins over the auto budget; a scalar behaves exactly as before.
static int ls_mlp_i8k(int card) {
    static int per[Q27_MAX_DEVICES];
    static int have = -1;
    if (have < 0) {
        have = 0;
        const char* e = getenv("Q27_LS_MLP_I8K");
        if (e && *e && strchr(e, ',')) {
            int cnt = 0, v = 0; bool any = false;
            for (const char* p = e; ; ++p) {
                if (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); any = true; }
                else {
                    if (any && cnt < Q27_MAX_DEVICES) per[cnt++] = v;
                    v = 0; any = false;
                    if (!*p) break;
                }
            }
            if (cnt > 0) {
                for (int i = cnt; i < Q27_MAX_DEVICES; ++i) per[i] = per[cnt - 1];
                have = 1;
                std::fprintf(stderr, "Q27_LS_MLP_I8K per-card:");
                for (int i = 0; i < cnt; ++i) std::fprintf(stderr, " card%d=%d", i, per[i]);
                std::fprintf(stderr, "  (list overrides the auto budget)\n");
                std::fflush(stderr);
            }
        }
    }
    if (have == 1 && card >= 0 && card < Q27_MAX_DEVICES) return per[card];
    if (s_ls_mlp_i8k > -2) return s_ls_mlp_i8k;
    return q27_env_int("Q27_LS_MLP_I8K", 0);
}

// ============================================================================================
// catalog types
// ============================================================================================
namespace {

struct Tensor {
    std::string name;
    int         shard;
    int         dtype;
    int         ndim;
    long long   shape[Q27_MAX_DIMS];
    long long   off;      // byte offset inside the shard payload
    long long   nbytes;
};

struct Shard {
    int            fd;
    unsigned char* map;
    size_t         maplen;
    size_t         payload_off;   // 8 + header length
    long long      payload_bytes;
};

struct Arena {
    int    device;
    void*  base;
    size_t size;
    int    fp8proj = 0;   // 1 = the seven fp8 projections of the TP shard (freeable after int8 mirrors are built)
    int    sr_mlp  = 0;   // Q27_LS_SR: 1 = one layer's block-scaled FP8 MLP planes (gate, up, 4 down slices) on its owner card
    int    layer   = -1;  // Q27_LS_SR: that layer (sr_mlp arenas only)
};

long long prod(const long long* s, int n) {
    long long v = 1;
    for (int i = 0; i < n; ++i) v *= s[i];
    return v;
}

const size_t kAlign = 256;
size_t alignup(size_t x) { return (x + kAlign - 1) & ~(kAlign - 1); }

}  // namespace

struct q27_model {
    std::string dir;
    Shard       sh[Q27_CKPT_SHARDS];
    int         nshard;

    std::vector<Tensor>                    t;
    std::unordered_map<std::string, int>   idx;
    long long                              payload_total;

    std::vector<Arena> arenas;

    q27_layer_t   layer[Q27_LAYERS];
    int           layer_res[Q27_LAYERS];        // 1 if uploaded
    q27_globals_t glob[Q27_MAX_DEVICES];
    int           dev_used[Q27_MAX_DEVICES];

    int embed_dev, norm_dev, head_dev;          // -1 until uploaded

    // ---- tensor-parallel residency (q27_upload_tp) ----
    // In TP mode EVERY layer lives on EVERY device, so one q27_layer_t per layer is not enough:
    // the handles are [layer][device index] and carry that card's SHARD dimensions. layer[] above
    // mirrors the devices[0] column so the single-device residency API keeps answering.
    std::vector<q27_layer_t> tpl;               // [Q27_LAYERS * tp_ndev], empty unless TP
    int tp_ndev;
    int tp_dev[Q27_MAX_DEVICES];                // device ordinal of each TP column

    // ---- layer-split residency (q27_upload_ls), coexisting with the TP layout ----
    // Card g holds the FULL weights of layers [g*LPP, (g+1)*LPP) here; the TP shards above stay
    // for the decode path, which is untouched. Card ls_ndev-1 also holds the full final norm
    // and lm_head in glob_ls.
    std::vector<q27_layer_t> tpl_ls;            // [Q27_LAYERS * ls_ndev], empty unless layer-split
    q27_globals_t glob_ls[Q27_MAX_DEVICES];
    int ls_ndev;

    // ---- Q27_LS_SR single residency (2026-09-17) ----
    // The layer-split copy is the ONLY copy. The seven fp8 projections of every owned layer sit in
    // per-tensor temporaries (one hipMalloc each, handle left pointing at it) until main has built the
    // int8 per-64 side-table mirrors from them and calls q27_ls_sr_release_fp8. Recorded here, not in
    // `arenas`, because they are transient by contract and released as a set.
    struct SrTemp { int device; void* ptr; int layer; size_t bytes; };
    std::vector<SrTemp> sr_fp8_temps;
    int       ls_sr;                            // 1 once q27_upload_ls ran in SR mode (all TP-only entry points then refuse)
    long long sr_mlp_bytes[Q27_MAX_DEVICES];    // q27_fp8_mlp_load_sr: FP8 MLP arena bytes per CARD INDEX
    int       sr_mlp_layers[Q27_MAX_DEVICES];   //                      and layers covered per card index

    q27_model() : nshard(0), payload_total(0), embed_dev(-1), norm_dev(-1), head_dev(-1),
                  tp_ndev(0), ls_ndev(0), ls_sr(0) {
        memset(sr_mlp_bytes, 0, sizeof(sr_mlp_bytes));
        memset(sr_mlp_layers, 0, sizeof(sr_mlp_layers));
        for (int i = 0; i < Q27_CKPT_SHARDS; ++i) {
            sh[i].fd = -1; sh[i].map = 0; sh[i].maplen = 0;
            sh[i].payload_off = 0; sh[i].payload_bytes = 0;
        }
        memset(layer, 0, sizeof(layer));
        memset(glob, 0, sizeof(glob));
        memset(glob_ls, 0, sizeof(glob_ls));
        memset(dev_used, 0, sizeof(dev_used));
        memset(tp_dev, 0, sizeof(tp_dev));
        for (int L = 0; L < Q27_LAYERS; ++L) {
            layer[L].layer   = L;
            layer[L].is_full = Q27_IS_FULL(L) ? 1 : 0;
            layer[L].device  = -1;
            layer_res[L]     = 0;
        }
    }
};

// ============================================================================================
// header parse
// ============================================================================================
namespace {

int parse_header(const char* json, size_t n, int shard, long long payload_bytes,
                 q27_model* M, char* err, size_t cap) {
    Jp j(json, json + n);
    if (!j.expect('{')) return fail(err, cap, Q27_E_FORMAT, "shard %d: header is not a JSON object", shard);
    if (j.eat('}'))     return Q27_OK;

    std::string name, field, dts;
    for (;;) {
        if (!j.str(name) || !j.expect(':'))
            return fail(err, cap, Q27_E_FORMAT, "shard %d: bad header key (%s)", shard, j.why);

        if (name == "__metadata__") {
            if (!j.skip())
                return fail(err, cap, Q27_E_FORMAT, "shard %d: bad __metadata__ (%s)", shard, j.why);
        } else {
            if (!j.expect('{'))
                return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' is not an object", shard, name.c_str());
            Tensor T;
            T.name  = name;
            T.shard = shard;
            T.dtype = Q27_DT_UNKNOWN;
            T.ndim  = 0;
            T.off   = -1;
            T.nbytes = -1;
            long long a = -1, b = -1;
            int esz = 0;
            dts.clear();

            for (;;) {
                if (!j.str(field) || !j.expect(':'))
                    return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' bad field (%s)",
                                shard, name.c_str(), j.why);
                if (field == "dtype") {
                    if (!j.str(dts))
                        return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' bad dtype", shard, name.c_str());
                } else if (field == "shape") {
                    if (!j.expect('['))
                        return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' bad shape", shard, name.c_str());
                    if (!j.eat(']')) {
                        for (;;) {
                            long long v;
                            if (!j.u64(v))
                                return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' bad shape entry (%s)",
                                            shard, name.c_str(), j.why);
                            if (T.ndim >= Q27_MAX_DIMS)
                                return fail(err, cap, Q27_E_SHAPE, "shard %d: '%s' has >%d dims",
                                            shard, name.c_str(), Q27_MAX_DIMS);
                            T.shape[T.ndim++] = v;
                            if (j.eat(',')) continue;
                            if (!j.expect(']'))
                                return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' unterminated shape",
                                            shard, name.c_str());
                            break;
                        }
                    }
                } else if (field == "data_offsets") {
                    if (!j.expect('[') || !j.u64(a) || !j.expect(',') || !j.u64(b) || !j.expect(']'))
                        return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' bad data_offsets (%s)",
                                    shard, name.c_str(), j.why);
                } else {
                    if (!j.skip())
                        return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' bad value for '%s'",
                                    shard, name.c_str(), field.c_str());
                }
                if (j.eat(',')) continue;
                if (!j.expect('}'))
                    return fail(err, cap, Q27_E_FORMAT, "shard %d: '%s' unterminated", shard, name.c_str());
                break;
            }

            T.dtype = dt_code(dts, &esz);
            if (T.dtype == Q27_DT_UNKNOWN)
                return fail(err, cap, Q27_E_FORMAT, "'%s': unknown dtype '%s'", name.c_str(), dts.c_str());
            if (a < 0 || b < a || b > payload_bytes)
                return fail(err, cap, Q27_E_FORMAT, "'%s': data_offsets [%lld,%lld) outside payload %lld",
                            name.c_str(), a, b, payload_bytes);
            T.off    = a;
            T.nbytes = b - a;
            const long long want = prod(T.shape, T.ndim) * (long long)esz;
            if (want != T.nbytes)
                return fail(err, cap, Q27_E_SHAPE, "'%s': %s shape needs %lld B, header says %lld B",
                            name.c_str(), dts.c_str(), want, T.nbytes);

            if (M->idx.find(T.name) != M->idx.end())
                return fail(err, cap, Q27_E_FORMAT, "'%s': duplicate tensor name", T.name.c_str());
            M->idx[T.name] = (int)M->t.size();
            M->t.push_back(T);
        }

        if (j.eat(',')) continue;
        if (!j.expect('}'))
            return fail(err, cap, Q27_E_FORMAT, "shard %d: unterminated header object", shard);
        break;
    }
    return Q27_OK;
}

}  // namespace

// ============================================================================================
// open / close
// ============================================================================================
q27_model_t* q27_open(const char* model_dir, char* err, size_t errcap) {
    if (!model_dir) { fail(err, errcap, Q27_E_ARG, "q27_open: model_dir is NULL"); return 0; }

    q27_model* M = new q27_model();
    M->dir = model_dir;
    M->t.reserve(Q27_CKPT_TENSORS);

    char path[1024];
    // Shard numbering is NOT a constant of the format. Our own merged requant writes
    // model-00000-of-00003; NVIDIA's ModelOpt artifact and the Hugging Face convention write
    // model-00001-of-00003. Detect the base once instead of assuming it - assuming it cost a run.
    int base = 0;
    {
        snprintf(path, sizeof path, "%s/model-%05d-of-%05d.safetensors", model_dir, 0, Q27_CKPT_SHARDS);
        if (access(path, R_OK) != 0) {
            snprintf(path, sizeof path, "%s/model-%05d-of-%05d.safetensors", model_dir, 1, Q27_CKPT_SHARDS);
            if (access(path, R_OK) == 0) base = 1;
        }
    }
    for (int s = 0; s < Q27_CKPT_SHARDS; ++s) {
        snprintf(path, sizeof path, "%s/model-%05d-of-%05d.safetensors", model_dir, s + base, Q27_CKPT_SHARDS);

        const int fd = open(path, O_RDONLY);
        if (fd < 0) { fail(err, errcap, Q27_E_IO, "open('%s'): %s", path, strerror(errno)); q27_close(M); return 0; }
        M->sh[s].fd = fd;

        struct stat st;
        if (fstat(fd, &st) != 0) { fail(err, errcap, Q27_E_IO, "fstat('%s'): %s", path, strerror(errno)); q27_close(M); return 0; }
        const size_t len = (size_t)st.st_size;
        if (len < 8) { fail(err, errcap, Q27_E_FORMAT, "'%s': %zu bytes, too small", path, len); q27_close(M); return 0; }

        void* mp = mmap(0, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mp == MAP_FAILED) { fail(err, errcap, Q27_E_IO, "mmap('%s', %zu): %s", path, len, strerror(errno)); q27_close(M); return 0; }
        M->sh[s].map    = (unsigned char*)mp;
        M->sh[s].maplen = len;
        M->nshard       = s + 1;
        // We jump tensor to tensor, not front to back over 8 GiB; upload() issues WILLNEED per range.
        madvise(mp, len, MADV_RANDOM);

        // 8-byte little-endian header length, decoded byte by byte so host endianness is irrelevant.
        const unsigned char* b = M->sh[s].map;
        unsigned long long hl = 0;
        for (int k = 7; k >= 0; --k) hl = (hl << 8) | (unsigned long long)b[k];
        if (hl == 0 || hl > len - 8) {
            fail(err, errcap, Q27_E_FORMAT, "'%s': header length %llu does not fit in %zu bytes", path, hl, len);
            q27_close(M); return 0;
        }
        M->sh[s].payload_off   = (size_t)(8ull + hl);
        M->sh[s].payload_bytes = (long long)(len - M->sh[s].payload_off);
        M->payload_total      += M->sh[s].payload_bytes;

        const int rc = parse_header((const char*)(b + 8), (size_t)hl, s, M->sh[s].payload_bytes, M, err, errcap);
        if (rc != Q27_OK) { q27_close(M); return 0; }
    }

    // ---- fail closed on the census. These two numbers were read off the checkpoint bytes. ----
    if ((int)M->t.size() != Q27_CKPT_TENSORS) {
        fail(err, errcap, Q27_E_FORMAT, "census: %d tensors, expected %d", (int)M->t.size(), Q27_CKPT_TENSORS);
        q27_close(M); return 0;
    }
    if (M->payload_total != Q27_CKPT_PAYLOAD_BYTES) {
        fail(err, errcap, Q27_E_FORMAT, "census: %lld payload bytes, expected %lld",
             M->payload_total, (long long)Q27_CKPT_PAYLOAD_BYTES);
        q27_close(M); return 0;
    }
    return M;
}

void q27_close(q27_model_t* m) {
    if (!m) return;
    q27_model* M = m;
    for (size_t i = 0; i < M->arenas.size(); ++i) {
        if (M->arenas[i].base) {
            hipSetDevice(M->arenas[i].device);
            hipFree(M->arenas[i].base);
        }
    }
    M->arenas.clear();
    for (size_t i = 0; i < M->sr_fp8_temps.size(); ++i) {   // Q27_LS_SR temporaries still held (release never ran)
        if (M->sr_fp8_temps[i].ptr) { hipSetDevice(M->sr_fp8_temps[i].device); hipFree(M->sr_fp8_temps[i].ptr); }
    }
    M->sr_fp8_temps.clear();
    for (int s = 0; s < Q27_CKPT_SHARDS; ++s) {
        if (M->sh[s].map) munmap(M->sh[s].map, M->sh[s].maplen);
        if (M->sh[s].fd >= 0) close(M->sh[s].fd);
    }
    delete M;
}

// ============================================================================================
// catalog access
// ============================================================================================
int       q27_tensor_count(const q27_model_t* m) { return m ? (int)m->t.size() : 0; }
long long q27_payload_bytes(const q27_model_t* m) { return m ? m->payload_total : 0; }

namespace {
const unsigned char* host_of(const q27_model* M, const Tensor& T) {
    return M->sh[T.shard].map + M->sh[T.shard].payload_off + (size_t)T.off;
}
float host_f32(const q27_model* M, int ti) {
    float v = 0.0f;
    memcpy(&v, host_of(M, M->t[ti]), sizeof(float));   // offsets are 4-aligned; memcpy anyway
    return v;
}
}  // namespace

const void* q27_host_ptr(const q27_model_t* m, const char* name, size_t* nbytes,
                         int* dtype, int* ndim, long long* shape) {
    if (!m || !name) return 0;
    std::unordered_map<std::string, int>::const_iterator it = m->idx.find(name);
    if (it == m->idx.end()) return 0;
    const Tensor& T = m->t[it->second];
    if (nbytes) *nbytes = (size_t)T.nbytes;
    if (dtype)  *dtype  = T.dtype;
    if (ndim)   *ndim   = T.ndim;
    if (shape)  for (int i = 0; i < Q27_MAX_DIMS; ++i) shape[i] = (i < T.ndim) ? T.shape[i] : 0;
    return host_of(m, T);
}

const char* q27_catalog_name(const q27_model_t* m, int i) {
    if (!m || i < 0 || i >= (int)m->t.size()) return 0;
    return m->t[i].name.c_str();
}

int q27_catalog_info(const q27_model_t* m, int i, int* shard, int* dtype,
                     long long* offset, size_t* nbytes, int* ndim, long long* shape) {
    if (!m || i < 0 || i >= (int)m->t.size()) return Q27_E_ARG;
    const Tensor& T = m->t[i];
    if (shard)  *shard  = T.shard;
    if (dtype)  *dtype  = T.dtype;
    if (offset) *offset = T.off;
    if (nbytes) *nbytes = (size_t)T.nbytes;
    if (ndim)   *ndim   = T.ndim;
    if (shape)  for (int k = 0; k < Q27_MAX_DIMS; ++k) shape[k] = (k < T.ndim) ? T.shape[k] : 0;
    return Q27_OK;
}

// ============================================================================================
// upload
// ============================================================================================
namespace {

struct Item { int ti; void* slot; };   // slot receives the device pointer (memcpy, no aliasing)

void prefetch(const unsigned char* p, size_t n) {
    if (q27storage::replay()) return;
    if (!p || !n) return;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    const uintptr_t a   = (uintptr_t)p & ~(uintptr_t)(ps - 1);
    const size_t    len = n + (size_t)((uintptr_t)p - a);
    madvise((void*)a, len, MADV_WILLNEED);
}

int copy_in(void* dst, const unsigned char* src, size_t n, char* err, size_t cap) {
    if (q27storage::loader().submit(dst, src, n)) return Q27_OK;
    const size_t CH = (size_t)64 << 20;
    size_t o = 0;
    while (o < n) {
        const size_t c = (n - o < CH) ? (n - o) : CH;
        const hipError_t e = hipMemcpy((char*)dst + o, src + o, c, hipMemcpyHostToDevice);
        if (e != hipSuccess)
            return fail(err, cap, Q27_E_HIP, "hipMemcpy(%zu B): %s", c, hipGetErrorString(e));
        o += c;
    }
    return Q27_OK;
}

// Undo a failed q27_upload: clear the layer handles it was populating and put the device's
// globals back the way an earlier successful upload left them.
void q27_load_rollback(q27_model* M, int lo, int hi, int dev, const q27_globals_t& saved) {
    for (int L = lo; L < hi; ++L) {
        memset(&M->layer[L], 0, sizeof(q27_layer_t));
        M->layer[L].layer   = L;
        M->layer[L].is_full = Q27_IS_FULL(L) ? 1 : 0;
        M->layer[L].device  = -1;
    }
    M->glob[dev] = saved;
}

}  // namespace

int q27_upload(q27_model_t* m, const q27_upload_req_t* req, char* err, size_t errcap) {
    if (!m || !req) return fail(err, errcap, Q27_E_ARG, "q27_upload: NULL argument");
    q27_model* M = m;

    const int dev = req->device;
    if (dev < 0 || dev >= Q27_MAX_DEVICES)
        return fail(err, errcap, Q27_E_ARG, "q27_upload: device %d out of range [0,%d)", dev, Q27_MAX_DEVICES);
    const int lo = req->layer_lo, hi = req->layer_hi;
    if (lo < 0 || hi > Q27_LAYERS || lo > hi)
        return fail(err, errcap, Q27_E_ARG, "q27_upload: bad layer range [%d,%d)", lo, hi);
    for (int L = lo; L < hi; ++L)
        if (M->layer_res[L])
            return fail(err, errcap, Q27_E_STATE, "layer %d already resident on device %d", L, M->layer[L].device);
    if (req->with_embed      && M->embed_dev >= 0) return fail(err, errcap, Q27_E_STATE, "embed_tokens already on device %d", M->embed_dev);
    if (req->with_final_norm && M->norm_dev  >= 0) return fail(err, errcap, Q27_E_STATE, "final norm already on device %d", M->norm_dev);
    if (req->with_lm_head    && M->head_dev  >= 0) return fail(err, errcap, Q27_E_STATE, "lm_head already on device %d", M->head_dev);

    // A second upload to the same device (say lm_head after the layers) must not lose what the
    // first one put in glob[dev] if it fails half way. Snapshot, restore on every failure path.
    const q27_globals_t saved_glob = M->glob[dev];

    std::vector<Item> items;
    items.reserve(64 + 20 * (size_t)(hi - lo));
    int  rc = Q27_OK;
    char nb[256];

    // ---- planners. Each validates dtype + shape against the ABI and fails closed. ----
    // The shape checks are the point: a q_proj read as 6144 rows instead of 12288 silently drops
    // the per-head OUTPUT GATE (contract §3), and that is exactly the class of bug that survives
    // a smoke test.
    struct Ctx { q27_model* M; std::vector<Item>* items; int* rc; char* err; size_t cap; };
    Ctx cx; cx.M = M; cx.items = &items; cx.rc = &rc; cx.err = err; cx.cap = errcap;

    // find a tensor, or set rc = missing
    struct Fn {
        static int find(Ctx& c, const char* name) {
            std::unordered_map<std::string, int>::const_iterator it = c.M->idx.find(name);
            if (it == c.M->idx.end()) {
                if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_MISSING, "missing tensor '%s'", name);
                return -1;
            }
            return it->second;
        }
        // 2-D weight: dtype + exact [rows][cols]
        static int w2(Ctx& c, const char* name, int dt, long long rows, long long cols, void* slot) {
            const int ti = find(c, name);
            if (ti < 0) return -1;
            const Tensor& T = c.M->t[ti];
            if (T.dtype != dt) {
                if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': dtype %s, expected %s",
                                                  name, dt_name(T.dtype), dt_name(dt));
                return -1;
            }
            if (T.ndim != 2 || T.shape[0] != rows || T.shape[1] != cols) {
                if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                                                  "'%s': shape rank %d [%lld,%lld], expected [%lld,%lld]",
                                                  name, T.ndim, T.ndim > 0 ? T.shape[0] : -1,
                                                  T.ndim > 1 ? T.shape[1] : -1, rows, cols);
                return -1;
            }
            Item it; it.ti = ti; it.slot = slot;
            c.items->push_back(it);
            return ti;
        }
        // any rank: dtype + exact element count (conv1d is [10240,1,4], vectors are [n])
        static int wn(Ctx& c, const char* name, int dt, long long nelem, void* slot) {
            const int ti = find(c, name);
            if (ti < 0) return -1;
            const Tensor& T = c.M->t[ti];
            if (T.dtype != dt) {
                if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': dtype %s, expected %s",
                                                  name, dt_name(T.dtype), dt_name(dt));
                return -1;
            }
            long long n = 1;
            for (int i = 0; i < T.ndim; ++i) n *= T.shape[i];
            if (n != nelem) {
                if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': %lld elements, expected %lld",
                                                  name, n, nelem);
                return -1;
            }
            Item it; it.ti = ti; it.slot = slot;
            c.items->push_back(it);
            return ti;
        }
        // F32 scalar, read on the HOST. Never uploaded: these ride in the weight handles.
        static float sc(Ctx& c, const char* name) {
            const int ti = find(c, name);
            if (ti < 0) return 0.0f;
            const Tensor& T = c.M->t[ti];
            if (T.dtype != Q27_DT_F32 || T.nbytes != 4) {
                if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': not an F32 scalar", name);
                return 0.0f;
            }
            return host_f32(c.M, ti);
        }
    };

    // ---- NVFP4 handle: U8 [rows][K/2] + e4m3 group scales [rows][K/16] + two F32 multipliers ----
    struct Grp {
        static void nvfp4(Ctx& c, const char* base, q27_nvfp4_t* h, int rows, int K) {
            char n[256];
            snprintf(n, sizeof n, "%s.weight", base);
            Fn::w2(c, n, Q27_DT_U8, rows, K / 2, (void*)&h->w);
            snprintf(n, sizeof n, "%s.weight_scale", base);
            Fn::w2(c, n, Q27_DT_F8_E4M3, rows, K / 16, (void*)&h->gs);
            snprintf(n, sizeof n, "%s.weight_scale_2", base);
            h->ws2 = Fn::sc(c, n);
            snprintf(n, sizeof n, "%s.input_scale", base);
            h->in_scale = Fn::sc(c, n);
            h->rows = rows;
            h->K    = K;
        }
        static void fp8(Ctx& c, const char* base, q27_fp8_t* h, int rows, int K) {
            char n[256];
            snprintf(n, sizeof n, "%s.weight", base);
            Fn::w2(c, n, Q27_DT_F8_E4M3, rows, K, (void*)&h->w);
            snprintf(n, sizeof n, "%s.weight_scale", base);
            h->wscale = Fn::sc(c, n);
            snprintf(n, sizeof n, "%s.input_scale", base);
            h->in_scale = Fn::sc(c, n);
            h->rows = rows;
            h->K    = K;
        }
    };

    const char* LP = "model.language_model.layers";

    for (int L = lo; L < hi && rc == Q27_OK; ++L) {
        q27_layer_t& H = M->layer[L];
        // wipe any stale handle state; layer/is_full are re-stamped so the struct is self-describing
        memset(&H, 0, sizeof(H));
        H.layer   = L;
        H.is_full = Q27_IS_FULL(L) ? 1 : 0;
        H.device  = dev;

        snprintf(nb, sizeof nb, "%s.%d.input_layernorm.weight", LP, L);
        Fn::wn(cx, nb, Q27_DT_BF16, Q27_HID, (void*)&H.input_norm);
        snprintf(nb, sizeof nb, "%s.%d.post_attention_layernorm.weight", LP, L);
        Fn::wn(cx, nb, Q27_DT_BF16, Q27_HID, (void*)&H.post_norm);

        // ---- MLP, NVFP4 ----
        snprintf(nb, sizeof nb, "%s.%d.mlp.gate_proj", LP, L);
        Grp::nvfp4(cx, nb, &H.gate, Q27_INTER, Q27_HID);
        snprintf(nb, sizeof nb, "%s.%d.mlp.up_proj", LP, L);
        Grp::nvfp4(cx, nb, &H.up, Q27_INTER, Q27_HID);
        snprintf(nb, sizeof nb, "%s.%d.mlp.down_proj", LP, L);
        Grp::nvfp4(cx, nb, &H.down, Q27_HID, Q27_INTER);

        // contract §1: ONE nvfp4_quantize serves gate AND up. Verified equal on all 64 layers.
        if (rc == Q27_OK && H.gate.in_scale != H.up.in_scale)
            rc = fail(err, errcap, Q27_E_SHAPE,
                      "layer %d: mlp gate.input_scale %.9g != up.input_scale %.9g; the shared "
                      "activation quantization in contract 1 does not hold", L,
                      (double)H.gate.in_scale, (double)H.up.in_scale);

        if (H.is_full) {
            // q_proj is 24 heads x 512 = query 256 + OUTPUT GATE 256 per head (q27.h Q27_QROWS).
            snprintf(nb, sizeof nb, "%s.%d.self_attn.q_proj", LP, L);
            Grp::fp8(cx, nb, &H.q_proj, Q27_QROWS, Q27_HID);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.k_proj", LP, L);
            Grp::fp8(cx, nb, &H.k_proj, Q27_KVROWS, Q27_HID);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.v_proj", LP, L);
            Grp::fp8(cx, nb, &H.v_proj, Q27_KVROWS, Q27_HID);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.o_proj", LP, L);
            Grp::fp8(cx, nb, &H.o_proj, Q27_HID, Q27_OROWS);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.q_norm.weight", LP, L);
            Fn::wn(cx, nb, Q27_DT_BF16, Q27_HDIM, (void*)&H.q_norm);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.k_norm.weight", LP, L);
            Fn::wn(cx, nb, Q27_DT_BF16, Q27_HDIM, (void*)&H.k_norm);

            // contract §3: the oracle throws unless q/k/v share input_scale. Same gate here.
            if (rc == Q27_OK && !(H.q_proj.in_scale == H.k_proj.in_scale &&
                                  H.k_proj.in_scale == H.v_proj.in_scale))
                rc = fail(err, errcap, Q27_E_SHAPE,
                          "layer %d: q/k/v input_scale differ (%.9g / %.9g / %.9g); one shared "
                          "activation quantization is required", L, (double)H.q_proj.in_scale,
                          (double)H.k_proj.in_scale, (double)H.v_proj.in_scale);
        } else {
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_qkv", LP, L);
            Grp::fp8(cx, nb, &H.in_qkv, Q27_GDN_QKV, Q27_HID);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_z", LP, L);
            Grp::fp8(cx, nb, &H.in_z, Q27_GDN_Z, Q27_HID);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.out_proj", LP, L);
            Grp::fp8(cx, nb, &H.out_proj, Q27_HID, Q27_GDN_Z);

            // in_proj_a / in_proj_b are BF16 [48,5120] — the only BF16 GEMVs in the trunk.
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_a.weight", LP, L);
            Fn::w2(cx, nb, Q27_DT_BF16, Q27_GDN_VH, Q27_HID, (void*)&H.in_a);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_b.weight", LP, L);
            Fn::w2(cx, nb, Q27_DT_BF16, Q27_GDN_VH, Q27_HID, (void*)&H.in_b);
            // conv1d is stored [10240,1,4]; the ABI reads it flat as [channel][tap] (contract §4.1).
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.conv1d.weight", LP, L);
            Fn::wn(cx, nb, Q27_DT_BF16, (long long)Q27_GDN_QKV * Q27_GDN_CONV, (void*)&H.conv1d);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.A_log", LP, L);
            Fn::wn(cx, nb, Q27_DT_BF16, Q27_GDN_VH, (void*)&H.A_log);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.dt_bias", LP, L);
            Fn::wn(cx, nb, Q27_DT_BF16, Q27_GDN_VH, (void*)&H.dt_bias);
            // RAW weight, no +1 (contract §2). The loader hands it over untouched; the kernel
            // must not apply the (1+W) form here.
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.norm.weight", LP, L);
            Fn::wn(cx, nb, Q27_DT_BF16, Q27_GDN_D, (void*)&H.gdn_norm);
        }
    }

    q27_globals_t& G = M->glob[dev];
    if (rc == Q27_OK && req->with_embed)
        Fn::w2(cx, "model.language_model.embed_tokens.weight", Q27_DT_BF16, Q27_VOCAB, Q27_HID, (void*)&G.embed);
    if (rc == Q27_OK && req->with_final_norm)
        Fn::wn(cx, "model.language_model.norm.weight", Q27_DT_BF16, Q27_HID, (void*)&G.final_norm);
    if (rc == Q27_OK && req->with_lm_head)
        Grp::nvfp4(cx, "lm_head", &G.lm_head, Q27_VOCAB, Q27_HID);   // NOT under the layers prefix

    if (rc != Q27_OK) { q27_load_rollback(M, lo, hi, dev, saved_glob); return rc; }

    if (items.empty()) return Q27_OK;

    // Copy in (shard, file offset) order: the page cache then sees one forward sweep per shard
    // instead of 20 seeks per layer.
    std::sort(items.begin(), items.end(), [M](const Item& a, const Item& b) {
        const Tensor& x = M->t[a.ti];
        const Tensor& y = M->t[b.ti];
        if (x.shard != y.shard) return x.shard < y.shard;
        return x.off < y.off;
    });

    size_t total = 0;
    for (size_t i = 0; i < items.size(); ++i) total += alignup((size_t)M->t[items[i].ti].nbytes);

    hipError_t he = hipSetDevice(dev);
    if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d): %s", dev, hipGetErrorString(he));

    void* base = 0;
    he = hipMalloc(&base, total);
    if (he != hipSuccess)
        return fail(err, errcap, Q27_E_HIP, "hipMalloc(%zu B = %.3f GiB) on device %d: %s",
                    total, (double)total / 1073741824.0, dev, hipGetErrorString(he));

    // Issue the page-cache reads ahead of the copies, then wait — the same dependency-graph shape
    // the estate measured on the GPU: several independent loads in flight, one wait at the end.
    for (int k = 0; k < 3 && k < (int)items.size(); ++k) {
        const Tensor& T = M->t[items[k].ti];
        prefetch(host_of(M, T), (size_t)T.nbytes);
    }

    size_t used = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i + 3 < items.size()) {
            const Tensor& N = M->t[items[i + 3].ti];
            prefetch(host_of(M, N), (size_t)N.nbytes);
        }
        const Tensor& T = M->t[items[i].ti];
        void* dst = (char*)base + used;
        q27storage::tag(T.name, T.dtype);
        const int crc = copy_in(dst, host_of(M, T), (size_t)T.nbytes, err, errcap);
        if (crc != Q27_OK) {
            hipFree(base);
            q27_load_rollback(M, lo, hi, dev, saved_glob);
            return crc;
        }
        memcpy(items[i].slot, &dst, sizeof(void*));   // patch the handle, no type punning
        used += alignup((size_t)T.nbytes);
    }

    Arena A; A.device = dev; A.base = base; A.size = total;
    M->arenas.push_back(A);
    M->dev_used[dev] = 1;
    for (int L = lo; L < hi; ++L) M->layer_res[L] = 1;
    if (req->with_embed)      M->embed_dev = dev;
    if (req->with_final_norm) M->norm_dev  = dev;
    if (req->with_lm_head)    M->head_dev  = dev;
    return Q27_OK;
}

int q27_upload_split(q27_model_t* m, const int* devices, int ndev, int want_embed,
                     char* err, size_t errcap) {
    if (!m || !devices || ndev <= 0)
        return fail(err, errcap, Q27_E_ARG, "q27_upload_split: bad argument");
    if (Q27_LAYERS % ndev)
        return fail(err, errcap, Q27_E_ARG, "q27_upload_split: %d devices does not divide %d layers",
                    ndev, Q27_LAYERS);
    const int per = Q27_LAYERS / ndev;
    for (int i = 0; i < ndev; ++i) {
        q27_upload_req_t r;
        r.device          = devices[i];
        r.layer_lo        = i * per;
        r.layer_hi        = (i + 1) * per;
        r.with_embed      = (i == 0        && want_embed) ? 1 : 0;
        r.with_final_norm = (i == ndev - 1) ? 1 : 0;
        r.with_lm_head    = (i == ndev - 1) ? 1 : 0;
        const int rc = q27_upload(m, &r, err, errcap);
        if (rc != Q27_OK) return rc;
    }
    return Q27_OK;
}


// ============================================================================================
// TENSOR-PARALLEL UPLOAD  (q27_upload_tp)
// ============================================================================================
// Every layer on every card, each card holding one shard of every large matrix. The shard map,
// the fail-closed divisibility rules and the reason in_proj_qkv is three ranges rather than one
// are in q27_load.h; this file is the mechanics.
//
// A destination BLOCK is one handle field (one device pointer). It is filled by one to three
// SEGMENTS, each a strided rectangle of a source tensor: `rows` rows of `row_bytes` taken every
// `src_stride` bytes starting at `src_off`, landing at `dst_off` inside the block.
//   column-parallel : one segment, row_bytes == src_stride  -> one contiguous copy from the mmap
//   in_proj_qkv     : three such segments, packed q|k|v on the card
//   row-parallel    : one segment, row_bytes < src_stride    -> gathered through a host staging
//                     buffer, then ONE H2D. (5120 separate 2176-byte hipMemcpys per tensor would
//                     be ~1M launches per card.)
namespace {

struct TpSeg {
    int       ti;
    long long src_off;      // byte offset of the first row inside the tensor
    long long rows;
    long long row_bytes;
    long long src_stride;
    long long dst_off;      // offset inside the destination block
};
struct TpBlk {
    void*     slot;         // receives the device pointer
    long long bytes;
    int       nseg;
    TpSeg     seg[3];
};

struct TpCtx {
    q27_model*          M;
    std::vector<TpBlk>* blk;
    int                 g;      // this card's index in the device list
    int                 ndev;
    int*                rc;
    char*               err;
    size_t              cap;
};

int tp_esize(int dt) {
    switch (dt) {
        case Q27_DT_U8: case Q27_DT_I8: case Q27_DT_F8_E4M3: case Q27_DT_F8_E5M2:
        case Q27_DT_BOOL:                                            return 1;
        case Q27_DT_BF16: case Q27_DT_F16: case Q27_DT_U16: case Q27_DT_I16: return 2;
        case Q27_DT_F32:  case Q27_DT_U32: case Q27_DT_I32:           return 4;
        default:                                                      return 8;
    }
}

struct Tp {
    // locate + validate a 2-D tensor
    static int w2(TpCtx& c, const char* name, int dt, long long rows, long long cols) {
        std::unordered_map<std::string, int>::const_iterator it = c.M->idx.find(name);
        if (it == c.M->idx.end()) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_MISSING, "missing tensor '%s'", name);
            return -1;
        }
        const Tensor& T = c.M->t[it->second];
        if (T.dtype != dt) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': dtype %s, expected %s",
                                              name, dt_name(T.dtype), dt_name(dt));
            return -1;
        }
        if (T.ndim != 2 || T.shape[0] != rows || T.shape[1] != cols) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                                              "'%s': rank %d [%lld,%lld], expected [%lld,%lld]",
                                              name, T.ndim, T.ndim > 0 ? T.shape[0] : -1,
                                              T.ndim > 1 ? T.shape[1] : -1, rows, cols);
            return -1;
        }
        return it->second;
    }
    static int wn(TpCtx& c, const char* name, int dt, long long nelem) {
        std::unordered_map<std::string, int>::const_iterator it = c.M->idx.find(name);
        if (it == c.M->idx.end()) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_MISSING, "missing tensor '%s'", name);
            return -1;
        }
        const Tensor& T = c.M->t[it->second];
        if (T.dtype != dt) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': dtype %s, expected %s",
                                              name, dt_name(T.dtype), dt_name(dt));
            return -1;
        }
        long long n = 1;
        for (int i = 0; i < T.ndim; ++i) n *= T.shape[i];
        if (n != nelem) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                                              "'%s': %lld elements, expected %lld", name, n, nelem);
            return -1;
        }
        return it->second;
    }
    static float sc(TpCtx& c, const char* name) {
        std::unordered_map<std::string, int>::const_iterator it = c.M->idx.find(name);
        if (it == c.M->idx.end()) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_MISSING, "missing tensor '%s'", name);
            return 0.0f;
        }
        const Tensor& T = c.M->t[it->second];
        if (T.dtype != Q27_DT_F32 || T.nbytes != 4) {
            if (*c.rc == Q27_OK) *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': not an F32 scalar", name);
            return 0.0f;
        }
        return host_f32(c.M, it->second);
    }

    // ---- replicated whole tensor ----
    static void rep(TpCtx& c, const char* name, int dt, long long nelem, void* slot) {
        if (*c.rc != Q27_OK) return;
        const int ti = wn(c, name, dt, nelem);
        if (ti < 0) return;
        const Tensor& T = c.M->t[ti];
        TpBlk B; B.slot = slot; B.bytes = T.nbytes; B.nseg = 1;
        B.seg[0].ti = ti; B.seg[0].src_off = 0; B.seg[0].rows = 1;
        B.seg[0].row_bytes = T.nbytes; B.seg[0].src_stride = T.nbytes; B.seg[0].dst_off = 0;
        c.blk->push_back(B);
    }

    // ---- column-parallel: `nrange` contiguous row ranges, each split G ways ----
    // range r covers source rows [base[r], base[r]+len[r]); this card takes the g-th equal slice
    // of each. Every len[r] must divide by G or the split would cut a head group.
    static void col(TpCtx& c, const char* name, int dt, long long rows, long long cols,
                    const long long* base, const long long* len, int nrange, void* slot,
                    long long* out_rows) {
        if (*c.rc != Q27_OK) return;
        const int ti = w2(c, name, dt, rows, cols);
        if (ti < 0) return;
        const long long rb = cols * (long long)tp_esize(dt);   // bytes per source row
        TpBlk B; B.slot = slot; B.bytes = 0; B.nseg = 0;
        long long tot = 0;
        for (int r = 0; r < nrange; ++r) {
            if (len[r] % c.ndev) {
                *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                             "'%s': column-parallel range of %lld rows does not divide by %d cards",
                             name, len[r], c.ndev);
                return;
            }
            const long long nr = len[r] / c.ndev;
            const long long r0 = base[r] + (long long)c.g * nr;
            if (r0 + nr > rows) {
                *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': shard rows [%lld,%lld) exceed %lld",
                             name, r0, r0 + nr, rows);
                return;
            }
            if (B.nseg >= 3) { *c.rc = fail(c.err, c.cap, Q27_E_ARG, "'%s': too many ranges", name); return; }
            TpSeg& S = B.seg[B.nseg++];
            S.ti = ti; S.src_off = r0 * rb; S.rows = nr;
            S.row_bytes = rb; S.src_stride = rb; S.dst_off = tot;
            tot += nr * rb;
        }
        B.bytes = tot;
        c.blk->push_back(B);
        if (out_rows) { *out_rows = 0; for (int r = 0; r < nrange; ++r) *out_rows += len[r] / c.ndev; }
    }

    // ---- row-parallel: every row, byte range [g*rb/G, +rb/G) ----
    static void row(TpCtx& c, const char* name, int dt, long long rows, long long cols, void* slot,
                    long long byte_off = -1, long long byte_len = -1) {
        if (*c.rc != Q27_OK) return;
        const int ti = w2(c, name, dt, rows, cols);
        if (ti < 0) return;
        const long long rb = cols * (long long)tp_esize(dt);
        if (byte_len < 0 && (rb % c.ndev)) {   // equal-split rule only without an explicit slice (Q27_TP3)
            *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                         "'%s': row-parallel row of %lld B does not divide by %d cards", name, rb, c.ndev);
            return;
        }
        const long long sl = byte_len < 0 ? rb / c.ndev : byte_len;
        const long long off = byte_off < 0 ? (long long)c.g * sl : byte_off;
        if (sl <= 0 || off < 0 || off + sl > rb) {
            *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': invalid row slice", name); return;
        }
        TpBlk B; B.slot = slot; B.bytes = rows * sl; B.nseg = 1;
        B.seg[0].ti = ti; B.seg[0].src_off = off; B.seg[0].rows = rows;
        B.seg[0].row_bytes = sl; B.seg[0].src_stride = rb; B.seg[0].dst_off = 0;
        c.blk->push_back(B);
    }

    // ---- the two storage formats, both directions ----
    // Q27_EXL3_DIR: EXL3 replaces every NVFP4 projection, so the 4-bit weights
    // are not allocated at all -- weights that never exist cannot be read, which
    // is a stronger guarantee than patching 147 call sites. The SCALARS
    // (ws2 / in_scale) are kept: they cost nothing and the norm/quantize kernels
    // that feed the incumbent arms still read in_scale. Same shape as the
    // Q27_FP8_DROP_NV_GU / _DN drops above.
    static bool exl3_drop() {
        static const bool d = [] { const char* e = getenv("Q27_EXL3_DIR"); const char* nd = getenv("Q27_EXL3_NODROP");
                                   return e && *e && !(nd && nd[0] == (char)49); }();   // NODROP=1 keeps NVFP4 resident: discriminator
        return d;
    }
    static void nvfp4_col(TpCtx& c, const char* base, q27_nvfp4_t* h, int rows, int K,
                          const long long* rbase, const long long* rlen, int nrange) {
        char n[256];
        long long got = 0;
        if (exl3_drop()) {
            for (int i = 0; i < nrange; ++i) got += rlen[i] / c.ndev;
            snprintf(n, sizeof n, "%s.weight_scale_2", base);  h->ws2      = sc(c, n);
            snprintf(n, sizeof n, "%s.input_scale", base);     h->in_scale = sc(c, n);
            h->w = nullptr; h->gs = nullptr; h->rows = (int)got; h->K = K;
            return;
        }
        snprintf(n, sizeof n, "%s.weight", base);
        col(c, n, Q27_DT_U8, rows, K / 2, rbase, rlen, nrange, (void*)&h->w, &got);
        snprintf(n, sizeof n, "%s.weight_scale", base);
        col(c, n, Q27_DT_F8_E4M3, rows, K / 16, rbase, rlen, nrange, (void*)&h->gs, 0);
        snprintf(n, sizeof n, "%s.weight_scale_2", base);  h->ws2      = sc(c, n);
        snprintf(n, sizeof n, "%s.input_scale", base);     h->in_scale = sc(c, n);
        h->rows = (int)got;                 // THE SHARD's rows
        h->K    = K;
    }
    static void nvfp4_row(TpCtx& c, const char* base, q27_nvfp4_t* h, int rows, int K,
                          int koff = -1, int klen = -1) {
        if (*c.rc != Q27_OK) return;
        if (klen < 0 && (K % c.ndev || (K / c.ndev) % 16)) {   // the equal-split rule applies only without an explicit slice (Q27_TP3)
            *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                         "'%s': row-parallel K=%d over %d cards gives %d, which is not an exact "
                         "multiple of the 16-wide scale group", base, K, c.ndev, K / c.ndev);
            return;
        }
        if (klen < 0) klen = K / c.ndev;
        if (koff < 0) koff = c.g * klen;
        if (koff < 0 || klen <= 0 || koff + klen > K || (koff & 15) || (klen & 15)) {
            *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': invalid NVFP4 K slice", base); return;
        }
        char n[256];
        if (exl3_drop()) {
            snprintf(n, sizeof n, "%s.weight_scale_2", base);  h->ws2      = sc(c, n);
            snprintf(n, sizeof n, "%s.input_scale", base);     h->in_scale = sc(c, n);
            h->w = nullptr; h->gs = nullptr; h->rows = rows; h->K = klen;
            return;
        }
        snprintf(n, sizeof n, "%s.weight", base);
        row(c, n, Q27_DT_U8, rows, K / 2, (void*)&h->w, koff / 2, klen / 2);
        snprintf(n, sizeof n, "%s.weight_scale", base);
        row(c, n, Q27_DT_F8_E4M3, rows, K / 16, (void*)&h->gs, koff / 16, klen / 16);
        snprintf(n, sizeof n, "%s.weight_scale_2", base);  h->ws2      = sc(c, n);
        snprintf(n, sizeof n, "%s.input_scale", base);     h->in_scale = sc(c, n);
        h->rows = rows;
        h->K    = klen;
    }
    static void fp8_col(TpCtx& c, const char* base, q27_fp8_t* h, int rows, int K,
                        const long long* rbase, const long long* rlen, int nrange) {
        char n[256];
        long long got = 0;
        snprintf(n, sizeof n, "%s.weight", base);
        col(c, n, Q27_DT_F8_E4M3, rows, K, rbase, rlen, nrange, (void*)&h->w, &got);
        snprintf(n, sizeof n, "%s.weight_scale", base);   h->wscale   = sc(c, n);
        snprintf(n, sizeof n, "%s.input_scale", base);    h->in_scale = sc(c, n);
        h->rows = (int)got;
        h->K    = K;
    }
    static void fp8_row(TpCtx& c, const char* base, q27_fp8_t* h, int rows, int K, int koff = -1, int klen = -1) {
        if (*c.rc != Q27_OK) return;
        if (koff >= 0 && klen > 0) {   // explicit K slice (Q27_TP3 ragged plan); fp8 = 1 byte per element
            if (koff + klen > K || (klen & 15)) { *c.rc = fail(c.err, c.cap, Q27_E_SHAPE, "'%s': invalid FP8 K slice [%d,+%d) of %d", base, koff, klen, K); return; }
            char n2[256];
            snprintf(n2, sizeof n2, "%s.weight", base);
            row(c, n2, Q27_DT_F8_E4M3, rows, K, (void*)&h->w, koff, klen);
            snprintf(n2, sizeof n2, "%s.weight_scale", base);   h->wscale   = sc(c, n2);
            snprintf(n2, sizeof n2, "%s.input_scale", base);    h->in_scale = sc(c, n2);
            h->rows = rows; h->K = klen;
            return;
        }
        if (K % c.ndev || (K / c.ndev) % 16) {
            *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                         "'%s': row-parallel K=%d over %d cards gives %d, which is not an exact "
                         "multiple of the 16-wide activation group", base, K, c.ndev, K / c.ndev);
            return;
        }
        char n[256];
        snprintf(n, sizeof n, "%s.weight", base);
        row(c, n, Q27_DT_F8_E4M3, rows, K, (void*)&h->w);
        snprintf(n, sizeof n, "%s.weight_scale", base);   h->wscale   = sc(c, n);
        snprintf(n, sizeof n, "%s.input_scale", base);    h->in_scale = sc(c, n);
        h->rows = rows;
        h->K    = K / c.ndev;
    }
};

// One card's whole plan: 64 sharded layers + the replicated final norm + the lm_head shard.
int tp_plan(q27_model* M, int g, int ndev, std::vector<TpBlk>& blk, q27_layer_t* col,
            q27_globals_t* G, char* err, size_t cap) {
    int rc = Q27_OK;
    TpCtx c; c.M = M; c.blk = &blk; c.g = g; c.ndev = ndev; c.rc = &rc; c.err = err; c.cap = cap;
    const char* LP = "model.language_model.layers";
    char nb[256];

    // one full range, the ordinary column-parallel case
    const long long one_base[1] = {0};
    // Q27_TP3: this worker's shard geometry. ndev == 1 is the layer-split FULL plan (every layer at
    // full width on its owner), which must not read the TP table.
    q27_tp_shard_t full_plan[1];
    if (ndev == 1 && !q27_tp_plan(1, full_plan)) return fail(err, cap, Q27_E_ARG, "tp_plan: full-width plan failed");
    const q27_tp_shard_t* const TSH = (ndev == 1) ? &full_plan[0] : q27_tp_shard(g);

    for (int L = 0; L < Q27_LAYERS && rc == Q27_OK; ++L) {
        q27_layer_t& H = col[L];
        memset(&H, 0, sizeof(H));
        H.layer = L; H.is_full = Q27_IS_FULL(L) ? 1 : 0; H.device = M->tp_dev[g];

        snprintf(nb, sizeof nb, "%s.%d.input_layernorm.weight", LP, L);
        Tp::rep(c, nb, Q27_DT_BF16, Q27_HID, (void*)&H.input_norm);
        snprintf(nb, sizeof nb, "%s.%d.post_attention_layernorm.weight", LP, L);
        Tp::rep(c, nb, Q27_DT_BF16, Q27_HID, (void*)&H.post_norm);

        {   // MLP: gate/up column-parallel, down row-parallel + ALL-REDUCE
            int mo[Q27_MAX_DEVICES] = {0}, ml[Q27_MAX_DEVICES] = {0};
            if (ndev > 1) q27_mlp_wshare(mo, ml, ndev); else ml[0] = Q27_INTER;
            const long long base_i[1] = {mo[g]}, len_i[1] = {ml[g]};
            TpCtx mc = c; mc.g = 0; mc.ndev = 1; // explicit aligned interval, not another equal split
            snprintf(nb, sizeof nb, "%s.%d.mlp.gate_proj", LP, L);
            Tp::nvfp4_col(mc, nb, &H.gate, Q27_INTER, Q27_HID, base_i, len_i, 1);
            snprintf(nb, sizeof nb, "%s.%d.mlp.up_proj", LP, L);
            Tp::nvfp4_col(mc, nb, &H.up, Q27_INTER, Q27_HID, base_i, len_i, 1);
            snprintf(nb, sizeof nb, "%s.%d.mlp.down_proj", LP, L);
            Tp::nvfp4_row(c, nb, &H.down, Q27_HID, Q27_INTER, mo[g], ml[g]);
        }
        if (rc == Q27_OK && H.gate.in_scale != H.up.in_scale)
            rc = fail(err, cap, Q27_E_SHAPE, "layer %d: mlp gate/up input_scale differ", L);

        if (H.is_full) {
            // Q27_TP3: explicit head ranges from the shard plan (whole GQA groups per card).
            const q27_tp_shard_t* T = TSH;
            const long long base_q[1] = {(long long)(Q27_QROWS / Q27_NHEAD) * T->q_h0}, len_q[1] = {(long long)T->QL};
            const long long base_kv[1] = {(long long)Q27_HDIM * T->kv_h0}, len_kv[1] = {(long long)T->KVL};
            TpCtx ac = c; ac.g = 0; ac.ndev = 1;   // explicit interval, not another equal split
            snprintf(nb, sizeof nb, "%s.%d.self_attn.q_proj", LP, L);
            Tp::fp8_col(ac, nb, &H.q_proj, Q27_QROWS, Q27_HID, base_q, len_q, 1);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.k_proj", LP, L);
            Tp::fp8_col(ac, nb, &H.k_proj, Q27_KVROWS, Q27_HID, base_kv, len_kv, 1);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.v_proj", LP, L);
            Tp::fp8_col(ac, nb, &H.v_proj, Q27_KVROWS, Q27_HID, base_kv, len_kv, 1);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.o_proj", LP, L);
            Tp::fp8_row(c, nb, &H.o_proj, Q27_HID, Q27_OROWS, Q27_HDIM * T->q_h0, T->OL);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.q_norm.weight", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, Q27_HDIM, (void*)&H.q_norm);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.k_norm.weight", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, Q27_HDIM, (void*)&H.k_norm);
            if (rc == Q27_OK && !(H.q_proj.in_scale == H.k_proj.in_scale &&
                                  H.k_proj.in_scale == H.v_proj.in_scale))
                rc = fail(err, cap, Q27_E_SHAPE, "layer %d: q/k/v input_scale differ", L);
        } else {
            // in_proj_qkv is q 2048 | k 2048 | v 6144: THREE ranges, so the card gets whole heads.
            // Q27_TP3: the card's key-head range [gk_h0, +gk_hn) of each (value heads = 3x).
            const q27_tp_shard_t* T = TSH;
            const long long qkv_base[3] = {(long long)Q27_GDN_D * T->gk_h0, 2048 + (long long)Q27_GDN_D * T->gk_h0, 4096 + 3LL * Q27_GDN_D * T->gk_h0};
            const long long qkv_len [3] = {(long long)Q27_GDN_D * T->gk_hn, (long long)Q27_GDN_D * T->gk_hn, 3LL * Q27_GDN_D * T->gk_hn};
            const long long base_z [1]  = {(long long)Q27_GDN_D * T->gv_h0}, len_z[1] = {(long long)T->ZL};
            TpCtx gc = c; gc.g = 0; gc.ndev = 1;
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_qkv", LP, L);
            Tp::fp8_col(gc, nb, &H.in_qkv, Q27_GDN_QKV, Q27_HID, qkv_base, qkv_len, 3);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_z", LP, L);
            Tp::fp8_col(gc, nb, &H.in_z, Q27_GDN_Z, Q27_HID, base_z, len_z, 1);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.out_proj", LP, L);
            Tp::fp8_row(c, nb, &H.out_proj, Q27_HID, Q27_GDN_Z, Q27_GDN_D * T->gv_h0, T->ZL);
            // replicated: the two BF16 GEMVs are 0.5 MB and their outputs are indexed by GLOBAL
            // value head, which the TP kernels reach with a head base rather than a sliced tensor.
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_a.weight", LP, L);
            // in_proj_a / in_proj_b were REPLICATED at full 48-head width on every card while
            // the step kernel reads only its own 12 heads (abuf[hbase+h]) -- three quarters of the
            // GEMV was computed and discarded, on 96 calls/token. TP had changed OWNERSHIP and the
            // execution never followed. Column-parallel by value-head range needs NO collective:
            // the input activation is already replicated and each card's output feeds only its own
            // step kernel. 48 % 4 == 0, and the shard base 12g is exactly the kernel's hbase.
            { const long long vb[1] = {(long long)T->gv_h0}, vl[1] = {(long long)T->gv_hn};
              Tp::col(gc, nb, Q27_DT_BF16, Q27_GDN_VH, Q27_HID, vb, vl, 1, (void*)&H.in_a, nullptr); }
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_b.weight", LP, L);
            { const long long vb[1] = {(long long)T->gv_h0}, vl[1] = {(long long)T->gv_hn};
              Tp::col(gc, nb, Q27_DT_BF16, Q27_GDN_VH, Q27_HID, vb, vl, 1, (void*)&H.in_b, nullptr); }
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.conv1d.weight", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, (long long)Q27_GDN_QKV * Q27_GDN_CONV, (void*)&H.conv1d);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.A_log", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, Q27_GDN_VH, (void*)&H.A_log);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.dt_bias", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, Q27_GDN_VH, (void*)&H.dt_bias);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.norm.weight", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, Q27_GDN_D, (void*)&H.gdn_norm);
        }
    }

    // final norm replicated on every card; lm_head column-parallel (62080 logits per card at G=4)
    if (rc == Q27_OK)
        Tp::rep(c, "model.language_model.norm.weight", Q27_DT_BF16, Q27_HID, (void*)&G->final_norm);
    if (rc == Q27_OK) {   // Q27_TP3: this card's lm_head row range
        const q27_tp_shard_t* T = TSH;
        const long long base_v[1] = {(long long)T->voc_off}, len_v[1] = {(long long)T->voc_len};
        TpCtx vc = c; vc.g = 0; vc.ndev = 1;
        Tp::nvfp4_col(vc, "lm_head", &G->lm_head, Q27_VOCAB, Q27_HID, base_v, len_v, 1);
    }
    return rc;
}

}  // namespace

int q27_upload_tp(q27_model_t* m, const int* devices, int ndev, char* err, size_t errcap) {
    if (!m || !devices || ndev <= 0)
        return fail(err, errcap, Q27_E_ARG, "q27_upload_tp: bad argument");
    q27_model* M = m;
    if (M->tp_ndev) return fail(err, errcap, Q27_E_STATE, "q27_upload_tp: already uploaded");
    for (int L = 0; L < Q27_LAYERS; ++L)
        if (M->layer_res[L])
            return fail(err, errcap, Q27_E_STATE, "layer %d is already resident (device %d)",
                        L, M->layer[L].device);
    for (int i = 0; i < ndev; ++i) {
        if (devices[i] < 0 || devices[i] >= Q27_MAX_DEVICES)
            return fail(err, errcap, Q27_E_ARG, "device %d out of range", devices[i]);
        for (int j = 0; j < i; ++j)
            if (devices[i] == devices[j])
                return fail(err, errcap, Q27_E_ARG, "device %d listed twice", devices[i]);
    }
    // Geometry: the shard plan (uniform when every dimension divides; ragged for ndev == 3).
    if (!q27_tp_shard_init(ndev))
        return fail(err, errcap, Q27_E_ARG, "q27_upload_tp: no valid TP shard plan for %d cards (Q27_TP3_Q/GK/VOC, Q27_MLP_WSHARE)", ndev);
    if (0) {
        const struct { const char* what; long long n; } need[] = {
            {"q_proj rows",   Q27_QROWS}, {"k/v_proj rows", Q27_KVROWS},
            {"in_proj_qkv q", 2048},      {"in_proj_qkv k", 2048}, {"in_proj_qkv v", 6144},
            {"in_proj_z rows",Q27_GDN_Z}, {"mlp rows",      Q27_INTER},
            {"lm_head rows",  Q27_VOCAB}, {"q heads",       Q27_NHEAD},
            {"kv heads",      Q27_NKV},   {"gdn value heads", Q27_GDN_VH},
            {"gdn key heads", Q27_GDN_KH},
        };
        for (size_t i = 0; i < sizeof(need)/sizeof(need[0]); ++i)
            if (need[i].n % ndev)
                return fail(err, errcap, Q27_E_ARG, "%d cards does not divide %s (%lld)",
                            ndev, need[i].what, need[i].n);
        const long long rowpar[3] = {Q27_OROWS, Q27_GDN_Z, Q27_INTER};
        const char* rpn[3] = {"o_proj K", "out_proj K", "down_proj K"};
        for (int i = 0; i < 3; ++i)
            if (rowpar[i] % ndev || (rowpar[i] / ndev) % 16)
                return fail(err, errcap, Q27_E_ARG,
                            "%d cards splits %s (%lld) into %lld, not a multiple of the 16-wide "
                            "scale group", ndev, rpn[i], rowpar[i], rowpar[i] / ndev);
    }

    M->tp_ndev = ndev;
    for (int i = 0; i < ndev; ++i) M->tp_dev[i] = devices[i];
    M->tpl.assign((size_t)Q27_LAYERS * (size_t)ndev, q27_layer_t());

    void* stage = 0;
    size_t stage_cap = 0;

    for (int g = 0; g < ndev; ++g) {
        const int dev = devices[g];
        std::vector<TpBlk> blk;
        blk.reserve(64 * 24 + 8);
        // plan into a column-major view: handle for (layer L, card g) is tpl[L*ndev + g]
        std::vector<q27_layer_t> colh(Q27_LAYERS);
        q27_globals_t GB; memset(&GB, 0, sizeof GB);

        int rc = tp_plan(M, g, ndev, blk, colh.data(), &GB, err, errcap);
        if (rc != Q27_OK) { if (stage) hipHostFree(stage); M->tp_ndev = 0; M->tpl.clear(); return rc; }

        // ---- Q27_ALIAS_MLP: do NOT allocate the MLP row shards of layers this card already holds
        //      at FULL width in the layer-split residency. gate/up are column-parallel, so a TP shard
        //      is rows [g*INTER/ndev, +INTER/ndev) of the card's own full-width tensor: same bytes,
        //      same K, same per-16 groups, no stride change -- only a different first row. The
        //      resolver below points the TP handles into the layer-split copy once it exists; until
        //      then those handles are null (which is why the blocks are dropped rather than trimmed:
        //      the arena is one allocation per card and a dropped block is memory never taken).
        //      down_proj is a K-row shard and needs a strided consumer, so it is not dropped here. ----
        static const bool alias_mlp = q27_env_flag("Q27_ALIAS_MLP", false);
        // Q27_DEC_MLP_I8: for a layer whose MLP is the persistent int8 mirror, the TP MLP shard is
        // pure duplicate residency -- the decode will read the same mirror through the views built by
        // q27_resolve_tp_ls_mlp. Drop the shard (37.5 MB/layer) instead of merely aliasing gate/up.
        static const bool dec_mlp_i8 = q27_env_flag("Q27_DEC_MLP_I8", false) && !q27_env_flag("Q27_DEC_MLP_NV", false);   // _NV=1: keep the NVFP4 TP shard and let DECODE read it (4 bits) while prefill keeps the int8 mirror (8 bits)
        if (dec_mlp_i8) {
            const int k8 = ls_mlp_i8k(g);
            int dropped = 0; size_t dbytes = 0;
            std::vector<TpBlk> keep; keep.reserve(blk.size());
            for (size_t i = 0; i < blk.size(); ++i) {
                bool dropit = false;
                for (int L = 0; L < Q27_LAYERS && !dropit; ++L) {
                    if (!q27_ls_mlp_i8(L, g, ndev, k8)) continue;
                    const q27_layer_t& H = colh[L];
                    const void* slots[6] = { &H.gate.w, &H.gate.gs, &H.up.w, &H.up.gs,
                                             &H.down.w, &H.down.gs };
                    for (int s = 0; s < 6; ++s) if (blk[i].slot == slots[s]) { dropit = true; break; }
                }
                if (dropit) { dbytes += alignup((size_t)blk[i].bytes); ++dropped; }
                else keep.push_back(blk[i]);
            }
            if (dropped) {
                blk.swap(keep);
                std::fprintf(stderr, "Q27_DEC_MLP_I8: card %d: %d duplicate MLP shards (%.1f MB) not "
                                     "allocated; the decode reads the int8 mirror instead\n",
                             g, dropped, (double)dbytes / 1048576.0);
                std::fflush(stderr);
            }
        }
        if (alias_mlp) {
            int dropped = 0; size_t dbytes = 0;
            std::vector<TpBlk> keep; keep.reserve(blk.size());
            for (size_t i = 0; i < blk.size(); ++i) {
                bool dropit = false;
                for (int L = 0; L < Q27_LAYERS && !dropit; ++L) {
                    if (!q27_ls_owned(L, g, ndev)) continue;
                    // A layer whose MLP is running from the persistent int8 mirror has NO full-width
                    // NVFP4 MLP left to address, so its TP shard must stay allocated (it is what the
                    // decode MLP reads). Aliasing it too would leave the TP handle pointing at a
                    // freed arena. The two mechanisms are complements, not layers of each other.
                    if (q27_ls_mlp_i8(L, g, ndev, ls_mlp_i8k(g))) continue;
                    const q27_layer_t& H = colh[L];
                    const void* slots[8] = { &H.gate.w, &H.gate.gs, &H.up.w, &H.up.gs,
                                             &H.gate.ws2, &H.gate.in_scale, &H.up.ws2, &H.up.in_scale };
                    const void* no_slots[4] = { nullptr, nullptr, nullptr, nullptr };   // scalars are cheap: keep them
                    (void)no_slots;
                    for (int s = 0; s < 4; ++s) if (blk[i].slot == slots[s]) { dropit = true; break; }
                }
                if (dropit) { dbytes += alignup((size_t)blk[i].bytes); ++dropped; }
                else keep.push_back(blk[i]);
            }
            if (dropped) {
                int owned = 0; for (int L = 0; L < Q27_LAYERS; ++L) if (q27_ls_owned(L, g, ndev)) ++owned;
                blk.swap(keep);
                std::fprintf(stderr, "Q27_ALIAS_MLP: card %d owns %d layer(s) at full width; %d duplicated MLP row "
                                     "shards (%.1f MB) not allocated, they will address the layer-split copy\n",
                             g, owned, dropped, (double)dbytes / 1048576.0);
                std::fflush(stderr);
            }
        }
        // ---- Q27_FP8_DROP_NV_GU: with Q27_FP8_MLP resident the decode gate/up consumer is the
        // block-scaled FP8 plane (q27_main.cpp:3037 takes f8_gate/f8_up whenever both are present),
        // so the NVFP4 TP gate/up shards it replaces are read by NOBODY: prefill sweeps the
        // layer-split FULL-WIDTH copies, not these per-card row shards. 25.07 MB per layer per card
        // x 64 layers = 1.60 GiB/card, of which ALIAS_MLP has already taken the 16 owned layers,
        // leaving ~1.20 GiB/card actually recovered here. The scalars (ws2/in_scale) are KEPT --
        // q27_fp8_mlp_load copies in_scale out of them, and they cost nothing.
        // Opt-in, because the FP8 load runs AFTER this drop: q27_fp8_mlp_load aborts loudly if it
        // cannot then cover every layer, rather than leaving decode a null gate/up. ----
        static const bool fp8_gu_drop = getenv("Q27_FP8_MLP") && q27_env_flag("Q27_FP8_DROP_NV_GU", false);
        if (fp8_gu_drop) {
            int dropped = 0; size_t dbytes = 0;
            std::vector<TpBlk> keep; keep.reserve(blk.size());
            for (size_t i = 0; i < blk.size(); ++i) {
                bool dropit = false;
                for (int L = 0; L < Q27_LAYERS && !dropit; ++L) {
                    const q27_layer_t& H = colh[L];
                    const void* slots[4] = { &H.gate.w, &H.gate.gs, &H.up.w, &H.up.gs };
                    for (int s = 0; s < 4; ++s) if (blk[i].slot == slots[s]) { dropit = true; break; }
                }
                if (dropit) { dbytes += alignup((size_t)blk[i].bytes); ++dropped; }
                else keep.push_back(blk[i]);
            }
            if (dropped) {
                blk.swap(keep);
                std::fprintf(stderr, "Q27_FP8_DROP_NV_GU: card %d: %d NVFP4 gate/up shards (%.1f MB) not "
                                     "allocated; decode reads the block-scaled FP8 planes instead\n",
                             g, dropped, (double)dbytes / 1048576.0);
                std::fflush(stderr);
            }
        }
        // ---- Q27_FP8_DROP_NV_DN (Q27_TPSR, 2026-09-17 night): with the block-scaled FP8 down K-SLICE
        // resident (Q27_FP8_MLP_DOWN=1) and consumed by the natural-order swiglu + slice kernel, the
        // NVFP4 down K-shard is read by nobody: 12.5 MB per layer per card x 64 = 0.75 GiB/card. The
        // scalars (ws2/in_scale) are kept, as above. Opt-in and fail-closed: q27_fp8_mlp_load refuses
        // to serve unless the FP8 down then covers every layer (a dropped representation is safe only
        // when its replacement is proven complete at the seam). ----
        static const bool fp8_dn_drop = getenv("Q27_FP8_MLP") && q27_env_flag("Q27_FP8_DROP_NV_DN", false);
        if (fp8_dn_drop) {
            int dropped = 0; size_t dbytes = 0;
            std::vector<TpBlk> keep; keep.reserve(blk.size());
            for (size_t i = 0; i < blk.size(); ++i) {
                bool dropit = false;
                for (int L = 0; L < Q27_LAYERS && !dropit; ++L) {
                    const q27_layer_t& H = colh[L];
                    const void* slots[2] = { &H.down.w, &H.down.gs };
                    for (int s = 0; s < 2; ++s) if (blk[i].slot == slots[s]) { dropit = true; break; }
                }
                if (dropit) { dbytes += alignup((size_t)blk[i].bytes); ++dropped; }
                else keep.push_back(blk[i]);
            }
            if (dropped) {
                blk.swap(keep);
                std::fprintf(stderr, "Q27_FP8_DROP_NV_DN: card %d: %d NVFP4 down shards (%.1f MB) not "
                                     "allocated; decode reads the block-scaled FP8 down K-slice instead\n",
                             g, dropped, (double)dbytes / 1048576.0);
                std::fflush(stderr);
            }
        }

        // (shard, file offset) order: one forward sweep per shard instead of 20 seeks per layer
        std::sort(blk.begin(), blk.end(), [M](const TpBlk& a, const TpBlk& b) {
            const Tensor& x = M->t[a.seg[0].ti];
            const Tensor& y = M->t[b.seg[0].ti];
            if (x.shard != y.shard) return x.shard < y.shard;
            return x.off + a.seg[0].src_off < y.off + b.seg[0].src_off;
        });

        size_t total = 0, maxstage = 0;
        for (size_t i = 0; i < blk.size(); ++i) {
            total += alignup((size_t)blk[i].bytes);
            bool strided = false;
            for (int s = 0; s < blk[i].nseg; ++s)
                if (blk[i].seg[s].row_bytes != blk[i].seg[s].src_stride) strided = true;
            if (strided && (size_t)blk[i].bytes > maxstage) maxstage = (size_t)blk[i].bytes;
        }
        if (maxstage > stage_cap) {
            if (stage) hipHostFree(stage);
            stage = 0; stage_cap = 0;
            if (hipHostMalloc(&stage, maxstage, hipHostMallocDefault) != hipSuccess) {
                M->tp_ndev = 0; M->tpl.clear();
                return fail(err, errcap, Q27_E_HIP, "hipHostMalloc(%zu) for the shard staging buffer",
                            maxstage);
            }
            stage_cap = maxstage;
        }

        hipError_t he = hipSetDevice(dev);
        if (he != hipSuccess) { if (stage) hipHostFree(stage); M->tp_ndev = 0; M->tpl.clear();
            return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d): %s", dev, hipGetErrorString(he)); }
        // Two arenas per card: the seven fp8 projections (q/k/v/o, in_qkv/in_z/out_proj) go into their own so they can
        // be freed once int8 execution mirrors exist (Q27_DEC_I8); everything else into the main one.
        std::vector<char> isfp8(blk.size(), 0);
        size_t total_fp8 = 0;
        for (size_t i = 0; i < blk.size(); ++i) {
            for (int L = 0; L < Q27_LAYERS && !isfp8[i]; ++L) {
                const q27_layer_t& H = colh[L];
                const void* slots[7] = { &H.q_proj.w, &H.k_proj.w, &H.v_proj.w, &H.o_proj.w, &H.in_qkv.w, &H.in_z.w, &H.out_proj.w };
                for (int s = 0; s < 7; ++s) if (blk[i].slot == slots[s]) { isfp8[i] = 1; break; }
            }
            if (isfp8[i]) total_fp8 += alignup((size_t)blk[i].bytes);
        }
        const size_t total_main = total - total_fp8;
        void* base = 0; void* base_fp8 = 0;
        he = hipMalloc(&base, total_main);
        if (he != hipSuccess) { if (stage) hipHostFree(stage); M->tp_ndev = 0; M->tpl.clear();
            return fail(err, errcap, Q27_E_HIP, "hipMalloc(%zu B = %.3f GiB) on device %d: %s",
                        total_main, (double)total_main / 1073741824.0, dev, hipGetErrorString(he)); }
        if (total_fp8) { he = hipMalloc(&base_fp8, total_fp8);
            if (he != hipSuccess) { hipFree(base); if (stage) hipHostFree(stage); M->tp_ndev = 0; M->tpl.clear();
                return fail(err, errcap, Q27_E_HIP, "hipMalloc(%zu B = %.3f GiB, fp8 projections) on device %d: %s",
                            total_fp8, (double)total_fp8 / 1073741824.0, dev, hipGetErrorString(he)); } }

        size_t used = 0, used_fp8 = 0;
        for (size_t i = 0; i < blk.size(); ++i) {
            TpBlk& B = blk[i];
            char* dst = isfp8[i] ? (char*)base_fp8 + used_fp8 : (char*)base + used;
            if (i + 3 < blk.size()) {                       // issue the page-cache reads ahead
                const TpBlk& N = blk[i + 3];
                const Tensor& T = M->t[N.seg[0].ti];
                prefetch(host_of(M, T) + N.seg[0].src_off,
                         (size_t)(N.seg[0].rows * N.seg[0].src_stride));
            }
            for (int s = 0; s < B.nseg; ++s) {
                const TpSeg& S = B.seg[s];
                q27storage::tag("tp." + M->t[S.ti].name, M->t[S.ti].dtype);
                const unsigned char* src = host_of(M, M->t[S.ti]) + S.src_off;
                if (q27storage::replay() || S.row_bytes == S.src_stride) {
                    const int crc = copy_in(dst + S.dst_off, src, (size_t)(S.rows * S.row_bytes),
                                            err, errcap);
                    if (crc != Q27_OK) { hipFree(base); if (base_fp8) hipFree(base_fp8); if (stage) hipHostFree(stage);
                                         M->tp_ndev = 0; M->tpl.clear(); return crc; }
                } else {                                    // strided: gather, then ONE H2D
                    char* p = (char*)stage;
                    for (long long r = 0; r < S.rows; ++r)
                        memcpy(p + r * S.row_bytes, src + r * S.src_stride, (size_t)S.row_bytes);
                    const int crc = copy_in(dst + S.dst_off, (const unsigned char*)stage,
                                            (size_t)(S.rows * S.row_bytes), err, errcap);
                    if (crc != Q27_OK) { hipFree(base); if (base_fp8) hipFree(base_fp8); if (stage) hipHostFree(stage);
                                         M->tp_ndev = 0; M->tpl.clear(); return crc; }
                }
            }
            void* dp = (void*)dst;
            memcpy(B.slot, &dp, sizeof(void*));
            if (isfp8[i]) used_fp8 += alignup((size_t)B.bytes); else used += alignup((size_t)B.bytes);
        }

        q27storage::flush();
        Arena A; A.device = dev; A.base = base; A.size = total_main; A.fp8proj = 0;
        M->arenas.push_back(A);
        if (total_fp8) { Arena F; F.device = dev; F.base = base_fp8; F.size = total_fp8; F.fp8proj = 1; M->arenas.push_back(F); }
        M->dev_used[dev] = 1;
        M->glob[dev] = GB;
        for (int L = 0; L < Q27_LAYERS; ++L) M->tpl[(size_t)L * ndev + g] = colh[L];
        q27storage::loader().owner_ready(dev,"tp-native");
        if (getenv("Q27_ARENA_PRINT"))     // placement probe: where this card's first layer weights landed
            std::fprintf(stderr, "Q27_ARENA dev %d: layer0 gate.w %p up.w %p down.w %p in_qkv.w %p\n", dev,
                         (const void*)colh[0].gate.w, (const void*)colh[0].up.w, (const void*)colh[0].down.w, (const void*)colh[0].in_qkv.w);
    }
    if (stage) hipHostFree(stage);

    // Q27_FP8X4=1: re-quantize the seven fp8 projections (attention q/k/v/o, GDN in_qkv/in_z/
    // out_proj) into NVFP4 twins at load, so the same 4-bit consumer kernels stream half the
    // bytes. The conversion kernels run on the default stream; synchronise before the engine
    // starts issuing on its own streams.
    // Q27_PF_X4W also needs the twins (it routes the BATCHED prefill projections through them) but
    // must NOT set g_fp8x4, which is wired into the serial per-position path and disables attb, the
    // GDN tile path and o_proj batching.
    if (q27_env_flag("Q27_FP8X4", false) || q27_env_int("Q27_PF_X4W", 0) > 0) {
        for (int g = 0; g < ndev; ++g) {
            hipError_t he = hipSetDevice(devices[g]);
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d) for FP8X4: %s",
                                              devices[g], hipGetErrorString(he));
            for (int L = 0; L < Q27_LAYERS; ++L) {
                q27_layer_t& H = M->tpl[(size_t)L * ndev + g];
                if (H.is_full) {
                    q27_fp8_to_nvfp4_conv(&H.q_proj, &H.q4, 0);
                    q27_fp8_to_nvfp4_conv(&H.k_proj, &H.k4, 0);
                    q27_fp8_to_nvfp4_conv(&H.v_proj, &H.v4, 0);
                    q27_fp8_to_nvfp4_conv(&H.o_proj, &H.o4, 0);
                } else {
                    q27_fp8_to_nvfp4_conv(&H.in_qkv, &H.iqkv4, 0);
                    q27_fp8_to_nvfp4_conv(&H.in_z, &H.iz4, 0);
                    q27_fp8_to_nvfp4_conv(&H.out_proj, &H.op4, 0);
                }
            }
            he = hipDeviceSynchronize();
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "FP8X4 sync device %d: %s",
                                              devices[g], hipGetErrorString(he));
        }
    }

    // Q27_PF_W8=1: int8 mirrors of the MLP weights (q27_dq4 applied once) for the v14 prefill kernels.
    // 67 MB per layer per card; the nibble tensors stay for decode.
    if (q27_env_flag("Q27_PF_W8", false)) {
        for (int g = 0; g < ndev; ++g) {
            hipError_t he = hipSetDevice(devices[g]);
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d) for W8: %s",
                                              devices[g], hipGetErrorString(he));
            for (int L = 0; L < Q27_LAYERS; ++L) {
                q27_layer_t& H = M->tpl[(size_t)L * ndev + g];
                if (!q27_nvfp4_make_w8(&H.gate, 0) || !q27_nvfp4_make_w8(&H.up, 0) || !q27_nvfp4_make_w8(&H.down, 0))
                    return fail(err, errcap, Q27_E_HIP, "W8 mirror allocation failed (layer %d, device %d)", L, devices[g]);
            }
            he = hipDeviceSynchronize();
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "W8 sync device %d: %s",
                                              devices[g], hipGetErrorString(he));
        }
    }


    // devices[0]'s column also answers the single-device residency API, shard dimensions and all.
    for (int L = 0; L < Q27_LAYERS; ++L) {
        M->layer[L]     = M->tpl[(size_t)L * ndev + 0];
        M->layer_res[L] = 1;
    }
    M->norm_dev = devices[0];
    M->head_dev = devices[0];
    return Q27_OK;
}

// LAYER-SPLIT UPLOAD (Q27_LAYER_SPLIT). One tp_plan at ndev=1 yields FULL matrices; the card
// keeps the blocks whose destination slot lies inside its layer range's column, plus (on the
// last card) the final norm and the full lm_head. Every matrix handle therefore carries full
// rows/K and the whole resident engine runs with nd=1 semantics: no shard, no collective.
// LAYER-SPLIT OWNERSHIP, in one place. Card g owns rotating BLOCKS of Q27_LS_BLK layers
// (block index b = L/blk, owner = b mod ndev, optionally rotated every other round by LS_BAL).
// The prefill sweep, the layer-split residency and the TP-side alias planner must all agree
// exactly, so the rule lives here instead of in three copies of the same lambda.
//
// Q27_LS_OWN="28,12,12,12" -- UNEVEN OWNERSHIP for mixed-capacity cards (2026-09-18: one
// 32 GiB card + three 16 GiB). Pipeline parallelism does not require equal shards -- only the
// activation handoff crosses a card boundary, and its size does not depend on how many layers
// sit on either side. Entries are per-card LAYER counts, must sum to Q27_LAYERS, and each must
// be a whole number of blocks. Malformed or absent -> the balanced scheme, unchanged.
//
// THE RING IS PRESERVED. The Q8 sweep requires owner(b) to advance one card per block ("the
// ring needs owner(b) = (b + const) % ndev", q27_main.cpp) or pipeline stages collide (measured:
// BAL rotation 637.9 ms against the constant shift). Blocks are still dealt in ring order
// 0,1,..,ndev-1,0,..; a card whose quota is spent is SKIPPED. With equal quotas this is
// bit-identical to the old modulo, so the balanced arm cannot move.
//
// WHAT IT COSTS. Ring throughput is set by the SLOWEST stage, so a card holding c times the
// mean layer count has c times the stage time. Uneven ownership BUYS residency and SPENDS
// prefill pipeline balance. It pays only when the freed memory buys back more than c -- the
// intended use is making an int8 Tensile mirror affordable on EVERY layer (Q27_LS_MLP_I8K),
// which costs ~264 MB/layer and is what separates 847 tok/s prefill from 1554.
static int q27_ls_own_blocks(int ndev, int blk, int* q) {
    const char* s = getenv("Q27_LS_OWN");
    if (!s || !*s || blk < 1 || ndev > Q27_MAX_DEVICES) return 0;
    int n = 0, sum = 0, v = 0, digits = 0;
    for (const char* p = s; ; ++p) {
        if (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++digits; if (v > Q27_LAYERS) return 0; }
        else if (*p == ',' || *p == ' ' || *p == '\0') {
            if (digits) {
                if (n >= ndev || v <= 0 || (v % blk)) return 0;
                q[n++] = v / blk; sum += v; v = 0; digits = 0;
            }
            if (!*p) break;
        } else return 0;
    }
    return (n == ndev && sum == Q27_LAYERS) ? 1 : 0;
}

// Which card owns block b. THE single ownership oracle -- every other site calls this.
int q27_ls_block_owner(int b, int ndev) {
    if (ndev <= 0 || b < 0) return 0;
    const int blk = q27_ls_blk_eff(ndev);   // Q27_TP3: quotas may override the LAYERS/ndev rule
    int quota[Q27_MAX_DEVICES];
    if (q27_ls_own_blocks(ndev, blk, quota)) {
        // UNEVEN QUOTAS, DEALT SO NO TWO ADJACENT BLOCKS SHARE AN OWNER. 2026-09-19.
        //
        // The previous deal walked the ring and SKIPPED spent cards, so once the small
        // cards exhausted, the largest took its remaining blocks CONSECUTIVELY -- e.g.
        // 28,12,12,12 at blk=2 is 14,6,6,6 blocks, and after 24 round-robin blocks card 0
        // held b24..b31 back to back. That violates the ring property the layer-split
        // pipeline is built on (`owner(b) = (b + const) % ndev`, "or pipeline stages
        // collide"): adjacent blocks on one card have no peer to hand the activation to,
        // so the stage serializes instead of overlapping. It is the DEAL ORDER that was
        // broken, not the idea of unequal quotas.
        //
        // Greedy largest-remaining-first, excluding the previous owner (the standard
        // rearrange-k-apart construction). It yields a no-two-adjacent assignment exactly
        // when max(quota) <= ceil(total/2) -- satisfied by 14,6,6,6 (14 <= 16) and by
        // Flash's 48 layers at blk=4, quotas 4,4,3,1 over 12 blocks (4 <= 6). When it is
        // NOT satisfiable the surplus must double up somewhere; we place those as late as
        // possible rather than silently collapsing to card 0.
        //
        // Recomputed per call rather than cached: b < 64 and ndev <= 16, so this is a few
        // hundred integer ops on a path that runs at load time and once per block.
        int total = 0;
        for (int i = 0; i < ndev; ++i) total += quota[i];
        if (total <= 0 || b >= total) return 0;
        // SELF-CHECK, once per process. An ownership map is exactly the kind of thing that
        // looks right and serializes the pipeline anyway, so the deal prints itself and
        // counts its own adjacency collisions rather than being trusted.
        {
            static int printed = 0;
            if (!printed) {
                printed = 1;
                int r2[Q27_MAX_DEVICES], cnt[Q27_MAX_DEVICES] = {0}, prev2 = -1, coll = 0;
                for (int i = 0; i < ndev; ++i) r2[i] = quota[i];
                char line[512]; int n2 = 0;
                for (int i = 0; i < total && n2 < (int)sizeof line - 8; ++i) {
                    int best = -1;
                    for (int g2 = 0; g2 < ndev; ++g2) {
                        if (r2[g2] <= 0 || g2 == prev2) continue;
                        if (best < 0 || r2[g2] > r2[best]) best = g2;
                    }
                    if (best < 0) for (int g2 = 0; g2 < ndev; ++g2) if (r2[g2] > 0) { best = g2; break; }
                    if (best < 0) break;
                    if (best == prev2) ++coll;
                    n2 += snprintf(line + n2, sizeof line - n2, "%d", best);
                    --r2[best]; prev2 = best; cnt[best]++;
                }
                char qs[128]; int nq = 0;
                for (int i = 0; i < ndev; ++i)
                    nq += snprintf(qs + nq, sizeof qs - nq, "%s%d/%d", i ? "," : "", cnt[i], quota[i]);
                fprintf(stderr, "Q27_LSOWN_DEAL ndev=%d blk=%d blocks=%d got/quota=%s adjacent_collisions=%d\n"
                                "Q27_LSOWN_MAP  %s\n", ndev, blk, total, qs, coll, line);
                fflush(stderr);
            }
        }
        int rem[Q27_MAX_DEVICES];
        for (int i = 0; i < ndev; ++i) rem[i] = quota[i];
        int prev = -1, owner = 0;
        for (int i = 0; i <= b; ++i) {
            int best = -1;
            for (int g2 = 0; g2 < ndev; ++g2) {
                if (rem[g2] <= 0 || g2 == prev) continue;
                if (best < 0 || rem[g2] > rem[best]) best = g2;
            }
            if (best < 0)                          // only the previous owner has blocks left
                for (int g2 = 0; g2 < ndev; ++g2) if (rem[g2] > 0) { best = g2; break; }
            if (best < 0) return 0;                // exhausted -- malformed quota vector
            owner = best; --rem[best]; prev = best;
        }
        return owner;
    }
    const int bal = q27_env_int("Q27_LS_BAL", 0);
    const int r = b / ndev;
    return (b + (bal ? (r & 1) : 0)) % ndev;
}

int q27_ls_owned(int layer, int card, int ndev) {
    if (ndev <= 0 || layer < 0 || layer >= Q27_LAYERS) return 0;
    const int blk = q27_ls_blk_eff(ndev);
    return q27_ls_block_owner(layer / blk, ndev) == card;
}

// Which of a card's owned layers carry the PERSISTENT int8 MLP mirror (Q27_LS_MLP_I8K). Those
// layers' NVFP4 MLP originals are converted and then DROPPED, so any other consumer that wanted to
// address the full-width tensor -- the TP decode handle's row-view alias, above all -- must fall
// back to its own shard instead. Indexed by the card's ownership ORDINAL, so the choice is symmetric
// across cards and the pipeline stays balanced.
int q27_ls_mlp_i8(int layer, int card, int ndev, int k) {
    if (k <= 0 || !q27_ls_owned(layer, card, ndev)) return 0;
    int ord = 0;
    for (int L = 0; L < Q27_LAYERS; ++L) {
        if (!q27_ls_owned(L, card, ndev)) continue;
        if (L == layer) return ord < k;
        ++ord;
    }
    return 0;
}

int q27_upload_ls(q27_model_t* m, const int* devices, int ndev, char* err, size_t errcap) {
    if (!m || !devices || ndev <= 0)
        return fail(err, errcap, Q27_E_ARG, "q27_upload_ls: bad argument");
    q27_model* M = m;
    if (M->ls_ndev) return fail(err, errcap, Q27_E_STATE, "q27_upload_ls: already uploaded");
    for (int i = 0; i < ndev; ++i) {
        if (devices[i] < 0 || devices[i] >= Q27_MAX_DEVICES)
            return fail(err, errcap, Q27_E_ARG, "device %d out of range", devices[i]);
        for (int j = 0; j < i; ++j)
            if (devices[i] == devices[j])
                return fail(err, errcap, Q27_E_ARG, "device %d listed twice", devices[i]);
    }
    if ((Q27_LAYERS % ndev) && q27_ls_blk_eff(ndev) == ((Q27_LAYERS / ndev > 0) ? Q27_LAYERS / ndev : 1) && !getenv("Q27_LS_OWN"))
        return fail(err, errcap, Q27_E_ARG, "%d cards does not divide %d layers (set Q27_LS_OWN quotas)", ndev, Q27_LAYERS);
    const int LPP = Q27_LAYERS / ndev;
    // Q27_LS_BLK: layers are assigned to cards in rotating BLOCKS of this many layers (card g owns blocks
    // g, g+ndev, g+2ndev, ...). LPP (default) = contiguous ranges; 4 = 16 pipeline stages of 4 layers.
    // Q27_LS_SR (single residency, 2026-09-17), read ONCE here and threaded through the whole upload:
    // no TP residency exists, so this copy is the only copy. Everything SR-specific below hangs off
    // this one boolean; with it false every line of the existing path executes exactly as before.
    // Q27_LS_SR_DRYRUN=1 (SR only): plan, print the per-card byte totals, allocate nothing, and return
    // Q27_E_STATE so a boot that set it by accident cannot continue on null handles.
    const bool sr     = q27_env_flag("Q27_LS_SR", false);
    const bool sr_dry = sr && q27_env_flag("Q27_LS_SR_DRYRUN", false);
    int ls_blk = q27_ls_blk_eff(ndev);   // default 2 (with 256-wide chunks): p1k 612 ms; Q27_TP3 quotas may override
    const int ls_bal = q27_env_int("Q27_LS_BAL", 0);   // 1 = rotate ownership every other round (balanced work, but adjacent blocks on one card collide in pipeline stages: 638 vs 542 ms) -> off
    auto ls_owned = [&](int L, int g) { return q27_ls_owned(L, g, ndev); };
    int head_card = ndev - 1; for (int gg = 0; gg < ndev; ++gg) if (ls_owned(Q27_LAYERS - 1, gg)) head_card = gg;   // the final norm + lm_head live with the last layer
    const q27_globals_t& tp_head = M->glob[devices[head_card]];
    const bool head_alias = q27_env_flag("Q27_LS_HEAD_ALIAS", true) && !sr
        && tp_head.final_norm && tp_head.lm_head.w
        && tp_head.lm_head.rows == Q27_VOCAB && tp_head.lm_head.K == Q27_HID;
    std::fprintf(stderr,"Q27_LS_HEAD owner=%d TP_rows=%d required_rows=%d alias=%d\n",
                 head_card,tp_head.lm_head.rows,Q27_VOCAB,int(head_alias));

    M->ls_ndev = ndev;
    M->tpl_ls.assign((size_t)Q27_LAYERS * (size_t)ndev, q27_layer_t());

    void* stage = 0;
    size_t stage_cap = 0;

    for (int g = 0; g < ndev; ++g) {
        const int dev = devices[g];
        std::vector<TpBlk> blk, fblk;
        std::vector<q27_layer_t> colh(Q27_LAYERS);
        q27_globals_t GB; memset(&GB, 0, sizeof GB);

        int rc = tp_plan(M, 0, 1, blk, colh.data(), &GB, err, errcap);   // FULL plan, no shards
        if (rc != Q27_OK) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear(); return rc; }

        (void)LPP;
        for (size_t i = 0; i < blk.size(); ++i) {
            TpBlk& B = blk[i];
            const ptrdiff_t off = (const char*)B.slot - (const char*)colh.data();
            if (off >= 0 && off < (ptrdiff_t)(sizeof(q27_layer_t) * Q27_LAYERS)) {
                const int L = (int)(off / (ptrdiff_t)sizeof(q27_layer_t));
                if (ls_owned(L, g)) { colh[L].device = dev; fblk.push_back(B); }
            } else if (g == head_card) {
                // TP normally holds only a vocabulary shard. LS emits a global
                // token from one card, so it needs every vocabulary row, exactly
                // as in the four-card donor. Only a genuinely full head may alias.
                if (!head_alias) fblk.push_back(B);         // final norm + full lm_head, on the card owning the last layer
            }
        }
        if (g != head_card) { GB.final_norm = nullptr; GB.lm_head = q27_nvfp4_t(); }
        else if (head_alias) { GB.final_norm = M->glob[dev].final_norm; GB.lm_head = M->glob[dev].lm_head; }

        std::sort(fblk.begin(), fblk.end(), [M](const TpBlk& a, const TpBlk& b) {
            const Tensor& x = M->t[a.seg[0].ti];
            const Tensor& y = M->t[b.seg[0].ti];
            if (x.shard != y.shard) return x.shard < y.shard;
            return x.off + a.seg[0].src_off < y.off + b.seg[0].src_off;
        });

        // ---- Q27_LS_SR: the NVFP4 MLP blocks (gate/up/down, w AND gs) of every owned layer are NOT
        //      allocated and NOT uploaded. The MLP of this residency is the block-scaled FP8 checkpoint,
        //      loaded at full width on the owner card by q27_fp8_mlp_load_sr; nothing on the SR path reads
        //      NVFP4 MLP bytes, so the 150.5 MB/layer would be a copy with no reader. The handles keep the
        //      rows/K/ws2/in_scale tp_plan already stamped (the scalars are host-read, not uploaded); only
        //      the two device pointers stay null, which is what q27_fp8_mlp_load_sr's seam check relies on.
        //      Removed from fblk here, before classification, so every byte count below is the SR plan. ----
        size_t sr_drop_bytes = 0; int sr_drop_n = 0;
        if (sr) {
            const size_t mlpoff[6] = {
                offsetof(q27_layer_t, gate) + offsetof(q27_nvfp4_t, w),  offsetof(q27_layer_t, gate) + offsetof(q27_nvfp4_t, gs),
                offsetof(q27_layer_t, up)   + offsetof(q27_nvfp4_t, w),  offsetof(q27_layer_t, up)   + offsetof(q27_nvfp4_t, gs),
                offsetof(q27_layer_t, down) + offsetof(q27_nvfp4_t, w),  offsetof(q27_layer_t, down) + offsetof(q27_nvfp4_t, gs) };
            std::vector<TpBlk> keep; keep.reserve(fblk.size());
            for (size_t i = 0; i < fblk.size(); ++i) {
                const ptrdiff_t off = (const char*)fblk[i].slot - (const char*)colh.data();
                bool drop = false;
                if (off >= 0 && off < (ptrdiff_t)(sizeof(q27_layer_t) * Q27_LAYERS)) {
                    const size_t inl = (size_t)off % sizeof(q27_layer_t);
                    for (int k = 0; k < 6; ++k) if (inl == mlpoff[k]) { drop = true; break; }
                }
                if (drop) { sr_drop_bytes += alignup((size_t)fblk[i].bytes); ++sr_drop_n; }
                else keep.push_back(fblk[i]);
            }
            fblk.swap(keep);
        }

        // Q27_LS_Q8: the NVFP4 MLP nibbles/scales and the fp8 projection bytes of this card's layers are
        // NOT kept resident -- they are uploaded to temporaries, converted into the int8 execution mirrors
        // (q27_fp8_to_i8g_conv / q27_nvfp4_to_i8g64_conv) and freed, which keeps ~4 GB per card for the
        // decode residency and the context. Identified by the pointer slot's offset inside q27_layer_t.
        const bool q8 = q27_env_flag("Q27_LS_Q8", true);
        // Q27_LS_MLP_NV: do NOT convert/consume the NVFP4 MLP weights. Keeping them resident costs
        // 2.14 GB/card for 16 owned layers; the per-64 int8 MLP mirrors they replace cost 4.27 GB/card
        // -- so the layer-split residency drops ~2.1 GB/card, which is the margin that lets the
        // layer-split KV cache fit at ctx=262144. The MLP then runs on the nvfp4 kernels. The fp8
        // projections keep their int8 mirrors (0.42 GB) because the wide int8 GEMMs are the sweep's speed.
        const bool mlp_nv = q27_env_flag("Q27_LS_MLP_NV", false);
        // Q27_LS_MLP_I8K: how many of this card's owned layers carry the PERSISTENT int8 MLP mirror
        // (converted from the NVFP4 originals, which are then dropped) and therefore run the
        // rocBLAS/Tensile MLP. 0 = the shipped all-NVFP4 MLP; 16 = the full 1263-class mirror set,
        // which is 4.23 GiB/card and does not fit at ctx=262144. The NET cost of the mirror is
        // 133.5 MB/card per layer (267.4 MB int8 + 16.7 MB scales minus the 150.5 MB of NVFP4 it
        // replaces), so this is a dial that spends exactly the memory the residency aliasing freed.
        const int mlp_i8k = ls_mlp_i8k(g);
        std::vector<char> mlp_i8(Q27_LAYERS, 0);
        { int ord = 0; for (int L = 0; L < Q27_LAYERS; ++L) { if (!ls_owned(L, g)) continue;
              mlp_i8[L] = (ord < mlp_i8k) ? 1 : 0; ++ord; } }
        std::vector<char> is_conv(fblk.size(), 0);
        if (sr) {
            // ---- Q27_LS_SR: the seven fp8 projections go to per-tensor TEMPORARIES exactly as the Q8
            //      path's is_conv=1 blocks do, but are NOT converted here, NOT freed and NOT nulled: the
            //      handle is left pointing at the temporary so main can build the int8 per-64 side-table
            //      mirror from it (q27_fp8_make_i8g keys the mirror by the handle's STABLE address in
            //      tpl_ls). q27_ls_sr_release_fp8 frees them afterwards. Q27_LS_MLP_I8K / Q27_LS_MLP_NV
            //      have no meaning without a resident NVFP4 MLP and are ignored. ----
            const size_t projoff[7] = {
                offsetof(q27_layer_t, q_proj) + offsetof(q27_fp8_t, w),  offsetof(q27_layer_t, k_proj) + offsetof(q27_fp8_t, w),
                offsetof(q27_layer_t, v_proj) + offsetof(q27_fp8_t, w),  offsetof(q27_layer_t, o_proj) + offsetof(q27_fp8_t, w),
                offsetof(q27_layer_t, in_qkv) + offsetof(q27_fp8_t, w),  offsetof(q27_layer_t, in_z) + offsetof(q27_fp8_t, w),
                offsetof(q27_layer_t, out_proj) + offsetof(q27_fp8_t, w) };
            for (size_t i = 0; i < fblk.size(); ++i) {
                const ptrdiff_t off = (const char*)fblk[i].slot - (const char*)colh.data();
                if (off < 0 || off >= (ptrdiff_t)(sizeof(q27_layer_t) * Q27_LAYERS)) continue;
                const size_t inl = (size_t)off % sizeof(q27_layer_t);
                for (int k = 0; k < 7; ++k) if (inl == projoff[k]) { is_conv[i] = 1; break; }
            }
            if (g == 0 && mlp_i8k) {
                std::fprintf(stderr, "Q27_LS_SR: Q27_LS_MLP_I8K=%d ignored -- no NVFP4 MLP is resident to mirror; "
                                     "the MLP is the block-scaled FP8 checkpoint\n", mlp_i8k);
                std::fflush(stderr);
            }
        } else if (q8) {
            const size_t convoff[13] = {
                offsetof(q27_layer_t, gate) + offsetof(q27_nvfp4_t, w),  offsetof(q27_layer_t, gate) + offsetof(q27_nvfp4_t, gs),
                offsetof(q27_layer_t, up)   + offsetof(q27_nvfp4_t, w),  offsetof(q27_layer_t, up)   + offsetof(q27_nvfp4_t, gs),
                offsetof(q27_layer_t, down) + offsetof(q27_nvfp4_t, w),  offsetof(q27_layer_t, down) + offsetof(q27_nvfp4_t, gs),
                offsetof(q27_layer_t, q_proj) + offsetof(q27_fp8_t, w),  offsetof(q27_layer_t, k_proj) + offsetof(q27_fp8_t, w),
                offsetof(q27_layer_t, v_proj) + offsetof(q27_fp8_t, w),  offsetof(q27_layer_t, o_proj) + offsetof(q27_fp8_t, w),
                offsetof(q27_layer_t, in_qkv) + offsetof(q27_fp8_t, w),  offsetof(q27_layer_t, in_z) + offsetof(q27_fp8_t, w),
                offsetof(q27_layer_t, out_proj) + offsetof(q27_fp8_t, w) };
            for (size_t i = 0; i < fblk.size(); ++i) {
                const ptrdiff_t off = (const char*)fblk[i].slot - (const char*)colh.data();
                if (off < 0 || off >= (ptrdiff_t)(sizeof(q27_layer_t) * Q27_LAYERS)) continue;
                const int Lb = (int)(off / (ptrdiff_t)sizeof(q27_layer_t));
                const size_t inl = (size_t)off % sizeof(q27_layer_t);
                // The MLP originals (k = 0..5) are uploaded to a temporary, converted and freed only
                // for the layers that asked for the mirror; the rest keep them resident in the arena.
                const int k0 = (mlp_i8k ? (mlp_i8[Lb] ? 0 : 6) : (mlp_nv ? 6 : 0));
                for (int k = k0; k < 13; ++k) if (inl == convoff[k]) { is_conv[i] = 1; break; }
            }
        }
        // Each converted tensor's ORIGINAL is uploaded to a temporary, mirrored, and freed. They used
        // to be freed only after EVERY layer's mirrors were built, so all 16 layers' originals were
        // live at once -- ~1.7 GiB/card of pure transient on top of the arena, and at ctx=262144 that
        // peak, not the steady state, is what failed ("Q8: int8/64 projection mirror allocation
        // failed"). Remember which layer each temporary belongs to and release it as soon as that
        // layer's mirrors exist: the peak becomes one layer, not sixteen.
        std::vector<void*> temps;
        std::vector<int> temp_layer;
        size_t total = 0, maxstage = 0;
        for (size_t i = 0; i < fblk.size(); ++i) {
            if (!is_conv[i]) total += alignup((size_t)fblk[i].bytes);
            bool strided = false;
            for (int s = 0; s < fblk[i].nseg; ++s)
                if (fblk[i].seg[s].row_bytes != fblk[i].seg[s].src_stride) strided = true;
            if (strided && (size_t)fblk[i].bytes > maxstage) maxstage = (size_t)fblk[i].bytes;
        }
        // ---- Q27_LS_SR per-card residency plan (printed for the boot log; the dry run stops here) ----
        size_t sr_temp_bytes = 0, sr_head_bytes = 0; int sr_owned = 0;
        if (sr) {
            for (size_t i = 0; i < fblk.size(); ++i) {
                const ptrdiff_t off = (const char*)fblk[i].slot - (const char*)colh.data();
                const bool inlayer = (off >= 0 && off < (ptrdiff_t)(sizeof(q27_layer_t) * Q27_LAYERS));
                if (is_conv[i]) sr_temp_bytes += alignup((size_t)fblk[i].bytes);
                else if (!inlayer) sr_head_bytes += alignup((size_t)fblk[i].bytes);   // final_norm + full lm_head (head card only)
            }
            for (int L = 0; L < Q27_LAYERS; ++L) if (ls_owned(L, g)) ++sr_owned;
            char headtxt[96];
            if (g == head_card) std::snprintf(headtxt, sizeof headtxt, "full NVFP4 in LS arena (%.1f MiB incl. final_norm)", (double)sr_head_bytes / 1048576.0);
            else                std::snprintf(headtxt, sizeof headtxt, "none");
            std::fprintf(stderr, "Q27_LS_SR card %d: %d owned layers, arena %.1f MiB, fp8 proj temps %.1f MiB, "
                                 "MLP: block-scaled FP8 (loaded separately), lm_head %s%s\n",
                         g, sr_owned, (double)total / 1048576.0, (double)sr_temp_bytes / 1048576.0, headtxt,
                         sr_dry ? "   [DRYRUN: nothing allocated]" : "");
            if (sr_dry) {
                std::fprintf(stderr, "Q27_LS_SR_DRYRUN card %d (device %d): NVFP4 MLP blocks not allocated: %d (%.1f MiB); "
                                     "layer arena %.1f MiB (of which lm_head+final_norm %.1f MiB); fp8 proj temps %.1f MiB transient\n",
                             g, dev, sr_drop_n, (double)sr_drop_bytes / 1048576.0, (double)total / 1048576.0,
                             (double)sr_head_bytes / 1048576.0, (double)sr_temp_bytes / 1048576.0);
                std::fflush(stderr);
                continue;            // no hipHostMalloc, no hipSetDevice, no hipMalloc, no copy
            }
            std::fflush(stderr);
        }
        if (maxstage > stage_cap) {
            if (stage) hipHostFree(stage);
            stage = 0; stage_cap = 0;
            if (hipHostMalloc(&stage, maxstage, hipHostMallocDefault) != hipSuccess) {
                M->ls_ndev = 0; M->tpl_ls.clear();
                return fail(err, errcap, Q27_E_HIP, "hipHostMalloc(%zu) for the shard staging buffer", maxstage);
            }
            stage_cap = maxstage;
        }

        hipError_t he = hipSetDevice(dev);
        if (he != hipSuccess) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
            return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d): %s", dev, hipGetErrorString(he)); }
        void* base = 0;
        he = hipMalloc(&base, total);
        if (he != hipSuccess) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
            return fail(err, errcap, Q27_E_HIP, "hipMalloc(%zu B) on device %d: %s",
                        total, dev, hipGetErrorString(he)); }

        size_t used = 0;
        for (size_t i = 0; i < fblk.size(); ++i) {
            TpBlk& B = fblk[i];
            char* dst = (char*)base + used;
            if (is_conv[i]) {
                void* tp = nullptr;
                if (hipMalloc(&tp, (size_t)B.bytes) != hipSuccess) { hipFree(base); if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
                    return fail(err, errcap, Q27_E_HIP, "hipMalloc(%lld B) temporary for a mirrored tensor on device %d", B.bytes, dev); }
                { const ptrdiff_t toff = (const char*)B.slot - (const char*)colh.data();
                  temp_layer.push_back((toff >= 0 && toff < (ptrdiff_t)(sizeof(q27_layer_t) * Q27_LAYERS))
                                        ? (int)(toff / (ptrdiff_t)sizeof(q27_layer_t)) : -1); }
                temps.push_back(tp); dst = (char*)tp;
            }
            if (i + 3 < fblk.size()) {
                const TpBlk& N = fblk[i + 3];
                const Tensor& T = M->t[N.seg[0].ti];
                prefetch(host_of(M, T) + N.seg[0].src_off,
                         (size_t)(N.seg[0].rows * N.seg[0].src_stride));
            }
            for (int s = 0; s < B.nseg; ++s) {
                const TpSeg& S = B.seg[s];
                q27storage::tag("ls." + M->t[S.ti].name, M->t[S.ti].dtype);
                const unsigned char* src = host_of(M, M->t[S.ti]) + S.src_off;
                if (q27storage::replay() || S.row_bytes == S.src_stride) {
                    const int crc = copy_in(dst + S.dst_off, src, (size_t)(S.rows * S.row_bytes), err, errcap);
                    if (crc != Q27_OK) { hipFree(base); if (stage) hipHostFree(stage);
                                         M->ls_ndev = 0; M->tpl_ls.clear(); return crc; }
                } else {
                    char* p = (char*)stage;
                    for (long long r = 0; r < S.rows; ++r)
                        memcpy(p + r * S.row_bytes, src + r * S.src_stride, (size_t)S.row_bytes);
                    const int crc = copy_in(dst + S.dst_off, (const unsigned char*)stage,
                                            (size_t)(S.rows * S.row_bytes), err, errcap);
                    if (crc != Q27_OK) { hipFree(base); if (stage) hipHostFree(stage);
                                         M->ls_ndev = 0; M->tpl_ls.clear(); return crc; }
                }
            }
            void* dp = (void*)dst;
            memcpy(B.slot, &dp, sizeof(void*));
            if (!is_conv[i]) used += alignup((size_t)B.bytes);
        }

        q27storage::flush();
        Arena A; A.device = dev; A.base = base; A.size = total;
        M->arenas.push_back(A);
        M->glob_ls[dev] = GB;
        if (sr) {
            // ---- Q27_LS_SR: keep the fp8 temporaries (handles already point at them), record each one
            //      for q27_ls_sr_release_fp8, and build ONLY ab8r for the owned GDN layers -- it is 1 MB
            //      and the sweep reads it. No q8g/gate8/per-row mirrors: the fp8 mirrors are main's
            //      (side table, from these temporaries) and the MLP is the FP8 checkpoint's. ----
            for (size_t t = 0; t < temps.size(); ++t) {
                if (!temps[t]) continue;
                q27_model::SrTemp T; T.device = dev; T.ptr = temps[t]; T.layer = temp_layer[t]; T.bytes = 0;
                for (size_t i = 0; i < fblk.size(); ++i)   // the block that landed in this temporary
                    if (is_conv[i]) { void* p = nullptr; memcpy(&p, fblk[i].slot, sizeof p); if (p == temps[t]) { T.bytes = (size_t)fblk[i].bytes; break; } }
                M->sr_fp8_temps.push_back(T);
            }
            temps.clear(); temp_layer.clear();
            for (int L = 0; L < Q27_LAYERS; ++L) { if (!ls_owned(L, g)) continue;
                q27_layer_t& H = colh[L];
                if (H.is_full) continue;
                if (!q27_ab_to_i8r_conv(H.in_a, H.in_b, Q27_GDN_VH, Q27_HID, &H.ab8r, 0)) {
                    if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
                    return fail(err, errcap, Q27_E_HIP, "Q27_LS_SR: GDN a/b int8 view failed (layer %d, device %d)", L, dev);
                }
            }
            he = hipDeviceSynchronize();
            if (he != hipSuccess) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
                return fail(err, errcap, Q27_E_HIP, "Q27_LS_SR ab8r sync device %d: %s", dev, hipGetErrorString(he)); }
        } else if (q8) {   // build the int8 execution mirrors for this card's layers, then drop the originals
            for (int L = 0; L < Q27_LAYERS; ++L) { if (!ls_owned(L, g)) continue;
                q27_layer_t& H = colh[L];
                static const bool q8cat = q27_env_flag("Q27_Q8_CAT", true);   // stack the same-input projections into one buffer (one GEMM)
                if (H.is_full) {
                    if (q8cat) { const q27_fp8_t* s3[3] = { &H.q_proj, &H.k_proj, &H.v_proj }; q27_i8g_t d3[3]; if (!q27_fp8_to_i8g64_cat(s3, 3, d3, 0)) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear(); return fail(err, errcap, Q27_E_HIP, "Q8: q/k/v mirror (layer %d)", L); } H.q8g = d3[0]; H.k8g = d3[1]; H.v8g = d3[2]; }
                    else { q27_fp8_to_i8g64_conv(&H.q_proj, &H.q8g, 0);  q27_fp8_to_i8g64_conv(&H.k_proj, &H.k8g, 0); q27_fp8_to_i8g64_conv(&H.v_proj, &H.v8g, 0); }
                    q27_fp8_to_i8g64_conv(&H.o_proj, &H.o8g, 0);
                } else {
                    if (q8cat) { const q27_fp8_t* s2[2] = { &H.in_qkv, &H.in_z }; q27_i8g_t d2[2]; if (!q27_fp8_to_i8g64_cat(s2, 2, d2, 0)) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear(); return fail(err, errcap, Q27_E_HIP, "Q8: in_qkv/in_z mirror (layer %d)", L); } H.iqkv8g = d2[0]; H.iz8g = d2[1]; }
                    else { q27_fp8_to_i8g64_conv(&H.in_qkv, &H.iqkv8g, 0); q27_fp8_to_i8g64_conv(&H.in_z, &H.iz8g, 0); }
                    q27_fp8_to_i8g64_conv(&H.out_proj, &H.op8g, 0);
                }
                const bool want_i8 = !mlp_nv || mlp_i8[L];
                if (want_i8 && (!q27_nvfp4_to_i8g64_conv(&H.gate, &H.gate8, 0) || !q27_nvfp4_to_i8g64_conv(&H.up, &H.up8, 0) ||
                                !q27_nvfp4_to_i8g64_conv(&H.down, &H.down8, 0))) {
                    if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
                    return fail(err, errcap, Q27_E_HIP, "Q8: int8/64 MLP mirror failed (layer %d, device %d)", L, dev);
                }
                if (q27_env_flag("Q27_Q8_RB", true)) {   // stage 3: per-(row,1024-group) int8 views for the rocBLAS GEMMs (in place, by family mask)
                    const int rbm = q27_env_int("Q27_Q8_RBM", 7), rbgs = q27_env_int("Q27_Q8_GS", 0);   // group size: 0 = full K (default), or 1024
                    int ok = 1;
                    if (want_i8 && (rbm & 4)) ok &= q27_i8g64_to_row_conv(&H.gate8, &H.gate8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.up8, &H.up8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.down8, &H.down8r, rbgs, 0);
                    // The per-64 scale plane is UNREACHABLE once the per-row form exists: the row
                    // conversion requantizes w in place and installs its own per-row scales, so the
                    // 64-group plane can no longer describe w under any interpretation, and no
                    // consumer in the rocBLAS configuration reads it. It is 16.7 MB per layer --
                    // exactly the currency this campaign spends on residency, so hand it back.
                    if (want_i8 && (rbm & 4)) {
                        if (H.gate8.s) { hipFree((void*)H.gate8.s); H.gate8.s = nullptr; }
                        if (H.up8.s)   { hipFree((void*)H.up8.s);   H.up8.s   = nullptr; }
                        if (H.down8.s) { hipFree((void*)H.down8.s); H.down8.s = nullptr; }
                    }
                    if (H.is_full && (rbm & 1)) ok &= q27_i8g64_to_row_conv(&H.q8g, &H.q8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.k8g, &H.k8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.v8g, &H.v8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.o8g, &H.o8r, rbgs, 0);
                    else if (!H.is_full && (rbm & 2)) ok &= q27_i8g64_to_row_conv(&H.iqkv8g, &H.iqkv8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.iz8g, &H.iz8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.op8g, &H.op8r, rbgs, 0);
                    if (!H.is_full && (rbm & 2)) ok &= q27_ab_to_i8r_conv(H.in_a, H.in_b, Q27_GDN_VH, Q27_HID, &H.ab8r, 0);
                    if (!ok) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear(); return fail(err, errcap, Q27_E_HIP, "Q8: per-row int8 view failed (layer %d, device %d)", L, dev); }
                }
                for (size_t t = 0; t < temps.size(); ++t)   // this layer's mirrors exist: its originals can go
                    if (temps[t] && temp_layer[t] == L) { hipFree(temps[t]); temps[t] = nullptr; }
            }
            he = hipDeviceSynchronize();
            if (he != hipSuccess) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
                return fail(err, errcap, Q27_E_HIP, "Q8 mirror sync device %d: %s", dev, hipGetErrorString(he)); }
            for (size_t i = 0; i < temps.size(); ++i) if (temps[i]) hipFree(temps[i]);
            temps.clear(); temp_layer.clear();
            for (int L = 0; L < Q27_LAYERS; ++L) {   if (!ls_owned(L, g)) continue;   // the originals are gone: nothing on the Q8 path reads them
                q27_layer_t& H = colh[L];
                if (!mlp_nv) { H.gate.w = nullptr; H.gate.gs = nullptr; H.up.w = nullptr; H.up.gs = nullptr; H.down.w = nullptr; H.down.gs = nullptr; }
                H.q_proj.w = nullptr; H.k_proj.w = nullptr; H.v_proj.w = nullptr; H.o_proj.w = nullptr;
                H.in_qkv.w = nullptr; H.in_z.w = nullptr; H.out_proj.w = nullptr;
            }
        }
        for (int L = 0; L < Q27_LAYERS; ++L)
            M->tpl_ls[(size_t)L * ndev + g] = ls_owned(L, g) ? colh[L] : q27_layer_t();
        q27storage::loader().owner_ready(dev,"ls-mirrors");
    }
    if (stage) hipHostFree(stage);

    if (sr_dry) {
        // The plan is printed; nothing was allocated and no handle was patched. Leave the model exactly
        // as it was before the call and refuse, so a boot that carried this flag by accident stops here.
        M->ls_ndev = 0; M->tpl_ls.clear();
        return fail(err, errcap, Q27_E_STATE, "Q27_LS_SR_DRYRUN: residency plan printed, nothing allocated");
    }

    // NVFP4 twins for the wide fp8 projections: only the card's OWN layers. Not in SR mode: the fp8
    // handles point at temporaries that q27_ls_sr_release_fp8 will free, and the SR sweep runs on the
    // int8 side-table mirrors main builds from them, so a twin built here would outlive its source.
    if (!sr && (q27_env_flag("Q27_FP8X4", false) || q27_env_int("Q27_PF_X4W", 0) > 0)) {
        for (int g = 0; g < ndev; ++g) {
            hipError_t he = hipSetDevice(devices[g]);
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d) for FP8X4: %s",
                                              devices[g], hipGetErrorString(he));
            for (int L = 0; L < Q27_LAYERS; ++L) { if (!ls_owned(L, g)) continue;
                q27_layer_t& H = M->tpl_ls[(size_t)L * ndev + g];
                if (H.is_full) {
                    q27_fp8_to_nvfp4_conv(&H.q_proj, &H.q4, 0);
                    q27_fp8_to_nvfp4_conv(&H.k_proj, &H.k4, 0);
                    q27_fp8_to_nvfp4_conv(&H.v_proj, &H.v4, 0);
                    q27_fp8_to_nvfp4_conv(&H.o_proj, &H.o4, 0);
                } else {
                    q27_fp8_to_nvfp4_conv(&H.in_qkv, &H.iqkv4, 0);
                    q27_fp8_to_nvfp4_conv(&H.in_z, &H.iz4, 0);
                    q27_fp8_to_nvfp4_conv(&H.out_proj, &H.op4, 0);
                }
            }
            he = hipDeviceSynchronize();
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "FP8X4 sync device %d: %s",
                                              devices[g], hipGetErrorString(he));
        }
    }

    // upload_tp normally sets tp_dev[] to the device list; the per-g tp_plan(ndev=1) calls
    // above clobber tp_dev[0] each iteration. Restore the mapping so q27_layer_ls finds the
    // right column per device (Q27_LS_NOTP skips the TP upload, which would otherwise repair it).
    for (int i = 0; i < ndev; ++i) { M->tp_dev[i] = devices[i]; M->dev_used[devices[i]] = 1; }
    M->tp_ndev = ndev;

    // PRESHUFFLED int8 mirrors of the LS full-width MLP weights (Q27_LS_W8=1): 4.27 GB per
    // card; pair with Q27_LS_NOTP=1 (the TP residency is not needed prefill-only).
    if (q27_env_flag("Q27_LS_W8", false)) {
        for (int g = 0; g < ndev; ++g) {
            hipError_t he = hipSetDevice(devices[g]);
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d) for LS W8S: %s",
                                              devices[g], hipGetErrorString(he));
            for (int L = 0; L < Q27_LAYERS; ++L) {
                q27_layer_t& H = M->tpl_ls[(size_t)L * ndev + g];
                if (!H.gate.w) continue;
                if (!q27_nvfp4_make_w8s(&H.gate, 0) || !q27_nvfp4_make_w8s(&H.up, 0) || !q27_nvfp4_make_w8s(&H.down, 0))
                    return fail(err, errcap, Q27_E_HIP, "LS W8S mirror failed (layer %d, device %d)", L, devices[g]);
            }
            he = hipDeviceSynchronize();
            if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "LS W8S sync device %d: %s",
                                              devices[g], hipGetErrorString(he));
        }
    }

    if (sr) {
        // ---- Q27_LS_SR: this is the only residency, so it must also answer the TP-era queries. ----
        // q27_globals(m, dev) returns &glob[dev], which q27_upload_tp used to fill and which the decode
        // round, the MTP init and the head all read for final_norm / lm_head. No TP upload ran, so mirror
        // the layer-split globals into it: the head card (owner of layer 63) gets the FULL lm_head
        // (rows 248320) + final_norm, the other cards null lm_head / null final_norm.
        for (int i = 0; i < ndev; ++i) { M->glob[devices[i]] = M->glob_ls[devices[i]]; M->dev_used[devices[i]] = 1; }
        M->norm_dev = devices[head_card];
        M->head_dev = devices[head_card];
        M->ls_sr = 1;
        size_t temp_total = 0; for (size_t i = 0; i < M->sr_fp8_temps.size(); ++i) temp_total += M->sr_fp8_temps[i].bytes;
        std::fprintf(stderr, "Q27_LS_SR: single residency on %d cards; head card %d (device %d) holds final_norm + full lm_head; "
                             "%zu fp8 projection temporaries (%.1f MiB total) await q27_ls_sr_release_fp8; NVFP4 MLP: none resident\n",
                     ndev, head_card, devices[head_card], M->sr_fp8_temps.size(), (double)temp_total / 1048576.0);
        std::fflush(stderr);
        // Platform law: HIP validates handles against the caller's CURRENT device. This helper visited
        // every card; hand the caller back its own (devices[0]) on the success path too.
        hipError_t he = hipSetDevice(devices[0]);
        if (he != hipSuccess) return fail(err, errcap, Q27_E_HIP, "hipSetDevice(%d) restore: %s", devices[0], hipGetErrorString(he));
    }

    return Q27_OK;
}

// ---- Q27_LS_SR ownership queries, so main does not re-derive the rule --------------------------
int q27_ls_sr_owner(const q27_model_t* m, int layer) {
    if (!m || !m->ls_ndev || layer < 0 || layer >= Q27_LAYERS) return -1;
    for (int g = 0; g < m->ls_ndev; ++g) if (q27_ls_owned(layer, g, m->ls_ndev)) return g;
    return -1;
}
int q27_ls_sr_head_card(const q27_model_t* m) {
    if (!m || !m->ls_ndev) return -1;
    return q27_ls_sr_owner(m, Q27_LAYERS - 1);   // final_norm + lm_head live with the last layer (q27_upload_ls)
}

const q27_layer_t* q27_layer_ls(const q27_model_t* m, int layer, int device) {
    if (!m || !m->ls_ndev || layer < 0 || layer >= Q27_LAYERS) return 0;
    for (int i = 0; i < m->ls_ndev; ++i)
        if (m->tp_dev[i] == device) return &m->tpl_ls[(size_t)layer * m->ls_ndev + i];
    return 0;
}

const q27_layer_t* q27_layer_tp(const q27_model_t* m, int layer, int device) {
    if (!m || !m->tp_ndev || layer < 0 || layer >= Q27_LAYERS) return 0;
    // Q27_LS_SR / Q27_LS_NOTP: q27_upload_ls sets tp_ndev (the device map) without a TP upload, so
    // tpl is EMPTY and &tpl[i] would be a non-null garbage pointer (measured: segfault at 0x7160 in
    // mtp_init's layer-3 comparison). No TP table, no TP handle.
    if (m->tpl.empty()) return 0;
    for (int i = 0; i < m->tp_ndev; ++i)
        if (m->tp_dev[i] == device) return &m->tpl[(size_t)layer * m->tp_ndev + i];
    return 0;
}

int q27_tp_ndev(const q27_model_t* m) { return m ? m->tp_ndev : 0; }

// ---- Q27_ALIAS_MLP resolver ------------------------------------------------------------------
// Runs after BOTH residencies exist. For every layer this card holds at full width in the
// layer-split layout, point the TP handle's gate/up at the matching ROW RANGE of that full-width
// tensor instead of at a copy. Column-parallel sharding makes the shard a pure row block: the
// packed row stride (K/2 bytes) and the scale row stride (K/16) are unchanged, so the consumer
// kernels need no stride at all -- only the first row moves. The two per-tensor scalars
// (weight_scale_2, input_scale) were read from the checkpoint independently for both handles and
// must agree; if they do not, the alias is refused and the shard keeps whatever it has.
// ---- Q27_SERVE_GROW: hand one layer's int8 MLP residency back -----------------------------------
// Slot growth runs out of headroom before it runs out of window: every extra position costs 30,720 B
// on every card, so a bigger bucket eventually has to be paid for. The currency is the fast MLP
// residency: releasing ONE layer's int8 mirror (~284 MB) and putting that layer back on its NVFP4
// authority (~187 MB of restored data) returns ~96-120 MB. Nothing is lost but speed on that layer,
// and the NVFP4 representation is the one the model was shipped with, so the fallback is authoritative
// rather than approximate.
//
// The NVFP4 data of a converted layer was freed at load, but the checkpoint shards stay mmapped, so
// the restore is a re-read of ~150 MB, not a reload. OWNER-CARD ONLY: the loader drops the TP MLP
// shard solely on the card that owns the layer in the layer-split layout (q27_ls_mlp_i8 requires
// q27_ls_owned), so the other three cards still hold their own NVFP4 shards and are untouched.
//
// The decode side is rebuilt the same way the load-time alias works -- gate/up are ROW VIEWS of the
// restored full-width tensors (no allocation at all), and only the K-sharded down needs its own copy
// (rows x INTER/ndev, a strided H2D). Nulling gate8r/dc_* is what switches the prefill (bl_l keys off
// lay->gate8r.w) and the decode (keys off L->dc_gate8r.w) back to NVFP4.
// ---- Q27_MXFP4_MLP: side-load AMD Quark OCP-MXFP4 gate/up over the resident decode shards -------
//
// WHY IN PLACE. Quark's MXFP4 weight tensor is byte-for-byte the same SHAPE as ours -- U8
// [rows][K/2], K-contiguous, E2M1 nibbles -- and its scale plane is SMALLER than ours ([rows][K/32]
// E8M0 vs [rows][K/16] E4M3). So the Quark bytes fit the buffers the TP decode shard already owns.
// That means no second residency, nothing to free, and no chance of the "allocate under pressure"
// stall: at boot the tight card has ~1.43 GiB free and a parallel MXFP4 copy of gate+up would want
// ~1.5 GiB/card. Overwriting costs nothing.
//
// WHAT IT DOES NOT TOUCH. Only Tp->gate / Tp->up, i.e. the TENSOR-PARALLEL DECODE shards. Prefill
// reads the separate layer-split residency (Ls->gate / Ls->up) and is unaffected, which is what
// keeps this experiment scoped to the decode MLP consumer. Q27_ALIAS_MLP is refused on this build
// (0 applied / 64 refused in the boot log), so the two really are distinct allocations here; if that
// ever changes this function must copy instead of overwrite.
//
// ws2 = 1.0f is REQUIRED, not a simplification: MXFP4 has no weight_scale_2: the E8M0 block scale is
// the entire weight scale. Leaving our NVFP4 ws2 in place would rescale every output. in_scale stays
// ours because we keep our own int8 activation path rather than Quark's MXFP4 activations.
static int q27_mx_find(const std::string& h, const char* name,
                       long long* a, long long* b, long long* d0, long long* d1) {
    std::string key = std::string("\"") + name + "\":";
    size_t p = h.find(key); if (p == std::string::npos) return 0;
    size_t sh = h.find("\"shape\":[", p); size_t of = h.find("\"data_offsets\":[", p);
    if (sh == std::string::npos || of == std::string::npos) return 0;
    if (std::sscanf(h.c_str() + sh + 9, "%lld,%lld", d0, d1) != 2) return 0;
    if (std::sscanf(h.c_str() + of + 16, "%lld,%lld", a, b) != 2) return 0;
    return 1;
}
int q27_mxfp4_sideload_mlp(q27_model_t* m, int ndev, const char* path) {
    if (!m || ndev < 1 || !path || !*path) return 0;
    q27_model* M = m; if (!M->tp_ndev) return 0;
    if (M->ls_sr) return 0;   // Q27_LS_SR: no TP residency exists (tpl is empty); this is a TP-shard entry point
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "Q27_MXFP4_MLP: cannot open %s\n", path); return 0; }
    unsigned long long hlen = 0;
    if (std::fread(&hlen, 8, 1, f) != 1 || hlen == 0 || hlen > (1ull << 28)) {
        std::fprintf(stderr, "Q27_MXFP4_MLP: bad safetensors header in %s\n", path); std::fclose(f); return 0; }
    std::string hdr(hlen, '\0');
    if (std::fread(&hdr[0], 1, hlen, f) != hlen) { std::fclose(f); return 0; }
    const long long base = 8 + (long long)hlen;
    int done = 0, skipped = 0;

    // E8M0 -> E4M3 RE-EXPRESSION, DONE ONCE AT LOAD.
    //
    // The execution representation the shipped consumer eats is [rows][K/16] of E4M3 with a
    // per-tensor ws2. OCP MXFP4 ships [rows][K/32] of E8M0 and no second-level scale. Both are
    // exact powers of two when an E4M3 mantissa is 0: E8M0 byte b is 2^(b-127); E4M3 byte (e<<3)
    // is 2^(e-7). So choose ws2 = 2^(bmin-127) for the tensor and store e = b - bmin + 7. The
    // product is bit-for-bit the MXFP4 scale, and one E8M0 byte simply feeds BOTH of the two
    // 16-groups its 32-wide block covers.
    //
    // This is why there is no MXFP4 kernel path and no second residency: the earlier private-buffer
    // version needed ~2.27 GiB/card and died with "Q27_HIP out of memory at q27_main.cpp:8895",
    // free down to 0.03 GiB, because the side-load runs BEFORE the KV cache is allocated. Repacking
    // instead writes the same byte counts the tensor already owns, so prefill and decode both get
    // the Quark weights and nothing extra is allocated.
    //
    // CONSTRAINT: e must stay in 1..14 (e=0 is the subnormal arm, e=15,m=7 is NaN), so a tensor's
    // exponent span must be <= 7. Measured on this checkpoint: gate_proj L0 spans 117..124 -- 8
    // codes, span 7 -- which fits exactly. A tensor that does not fit is SKIPPED, never clamped:
    // clamping would silently rescale part of a layer.
    auto repack = [&](const unsigned char* e8, size_t nblk, std::vector<unsigned char>& out16,
                      size_t ngrp16, float* ws2_out) -> bool {
        unsigned bmin = 255, bmax = 0;
        for (size_t i = 0; i < nblk; ++i) { unsigned b = e8[i]; if (b < bmin) bmin = b; if (b > bmax) bmax = b; }
        if (bmin > bmax) return false;
        if (bmin == 0 || bmax == 255) return false;           // reserved / NaN in OCP E8M0
        if (bmax - bmin > 7) return false;                    // cannot be expressed with m=0 E4M3
        out16.resize(ngrp16);
        for (size_t g = 0; g < ngrp16; ++g) {
            const size_t blk = g >> 1;                        // two 16-groups share one 32-block
            const unsigned e = (blk < nblk ? e8[blk] : bmin) - bmin + 7u;
            out16[g] = (unsigned char)(e << 3);               // mantissa 0 => exact power of two
        }
        *ws2_out = std::ldexp(1.0f, (int)bmin - 127);
        return true;
    };

    const int nr = Q27_INTER / ndev, KH = Q27_HID / 2, KS32 = Q27_HID / 32, KS16 = Q27_HID / 16;
    std::vector<unsigned char> wbuf((size_t)nr * KH), sbuf((size_t)nr * KS32), s16;
    for (int L = 0; L < Q27_LAYERS; ++L) {
        for (int which = 0; which < 2; ++which) {
            char nw[256], ns[256];
            const char* fam = which ? "up_proj" : "gate_proj";
            std::snprintf(nw, sizeof nw, "model.language_model.layers.%d.mlp.%s.weight", L, fam);
            std::snprintf(ns, sizeof ns, "model.language_model.layers.%d.mlp.%s.weight_scale", L, fam);
            long long wa, wb, wd0, wd1, sa, sb, sd0, sd1;
            if (!q27_mx_find(hdr, nw, &wa, &wb, &wd0, &wd1) ||
                !q27_mx_find(hdr, ns, &sa, &sb, &sd0, &sd1)) { ++skipped; continue; }
            if (wd0 != Q27_INTER || wd1 != KH || sd0 != Q27_INTER || sd1 != KS32) {
                std::fprintf(stderr, "Q27_MXFP4_MLP: L%d %s shape w[%lld,%lld] s[%lld,%lld] unexpected -- skipping\n",
                             L, fam, wd0, wd1, sd0, sd1);
                ++skipped; continue;
            }
            for (int c = 0; c < M->tp_ndev; ++c) {
                q27_layer_t* Tp = &M->tpl[(size_t)L * M->tp_ndev + c];
                q27_nvfp4_t* dst = which ? &Tp->up : &Tp->gate;
                if (!dst->w || !dst->gs || dst->rows != nr || dst->K != Q27_HID) { ++skipped; continue; }
                const long long r0 = (long long)c * nr;
                if (fseeko(f, (off_t)(base + wa + r0 * KH), SEEK_SET) != 0 ||
                    std::fread(wbuf.data(), 1, wbuf.size(), f) != wbuf.size()) { ++skipped; continue; }
                if (fseeko(f, (off_t)(base + sa + r0 * KS32), SEEK_SET) != 0 ||
                    std::fread(sbuf.data(), 1, sbuf.size(), f) != sbuf.size()) { ++skipped; continue; }
                float ws2 = 1.0f;
                if (!repack(sbuf.data(), sbuf.size(), s16, (size_t)nr * KS16, &ws2)) {
                    std::fprintf(stderr, "Q27_MXFP4_MLP: L%d %s card %d scale span not expressible -- skipping\n",
                                 L, fam, c);
                    ++skipped; continue;
                }
                if (hipSetDevice(c) != hipSuccess) { ++skipped; continue; }
                if (hipMemcpy((void*)dst->w,  wbuf.data(), wbuf.size(), hipMemcpyHostToDevice) != hipSuccess ||
                    hipMemcpy((void*)dst->gs, s16.data(),  s16.size(),  hipMemcpyHostToDevice) != hipSuccess) {
                    std::fprintf(stderr, "Q27_MXFP4_MLP: L%d %s card %d upload failed\n", L, fam, c);
                    ++skipped; continue;
                }
                dst->ws2 = ws2;      // carries the per-tensor 2^(bmin-127)
                dst->gs_mx = 0;      // consumed by the EXISTING E4M3 path, by construction
                dst->gs_fast = 0;
                ++done;
            }
        }
    }

    // down_proj: COLUMN range of the full [HID][INTER/2]; read whole, slice per card.
    if (q27_env_flag("Q27_MXFP4_DOWN", true)) {
        const int dKH = Q27_INTER / 2, dK32 = Q27_INTER / 32;
        const int sKH = (Q27_INTER / ndev) / 2, sK32 = (Q27_INTER / ndev) / 32, sK16 = (Q27_INTER / ndev) / 16;
        std::vector<unsigned char> fw((size_t)Q27_HID * dKH), fs((size_t)Q27_HID * dK32);
        std::vector<unsigned char> cw((size_t)Q27_HID * sKH), cs((size_t)Q27_HID * sK32), cs16;
        for (int L = 0; L < Q27_LAYERS; ++L) {
            char nw[256], ns[256];
            std::snprintf(nw, sizeof nw, "model.language_model.layers.%d.mlp.down_proj.weight", L);
            std::snprintf(ns, sizeof ns, "model.language_model.layers.%d.mlp.down_proj.weight_scale", L);
            long long wa, wb, wd0, wd1, sa, sb, sd0, sd1;
            if (!q27_mx_find(hdr, nw, &wa, &wb, &wd0, &wd1) ||
                !q27_mx_find(hdr, ns, &sa, &sb, &sd0, &sd1)) { ++skipped; continue; }
            if (wd0 != Q27_HID || wd1 != dKH || sd0 != Q27_HID || sd1 != dK32) {
                std::fprintf(stderr, "Q27_MXFP4_MLP: L%d down shape unexpected -- skipping\n", L); ++skipped; continue; }
            if (fseeko(f, (off_t)(base + wa), SEEK_SET) != 0 ||
                std::fread(fw.data(), 1, fw.size(), f) != fw.size()) { ++skipped; continue; }
            if (fseeko(f, (off_t)(base + sa), SEEK_SET) != 0 ||
                std::fread(fs.data(), 1, fs.size(), f) != fs.size()) { ++skipped; continue; }
            for (int c = 0; c < M->tp_ndev; ++c) {
                q27_layer_t* Tp = &M->tpl[(size_t)L * M->tp_ndev + c];
                q27_nvfp4_t* dst = &Tp->down;
                if (!dst->w || !dst->gs || dst->rows != Q27_HID || dst->K != Q27_INTER / ndev) { ++skipped; continue; }
                for (int r = 0; r < Q27_HID; ++r) {
                    std::memcpy(&cw[(size_t)r * sKH],  &fw[(size_t)r * dKH  + (size_t)c * sKH],  sKH);
                    std::memcpy(&cs[(size_t)r * sK32], &fs[(size_t)r * dK32 + (size_t)c * sK32], sK32);
                }
                float ws2 = 1.0f;
                if (!repack(cs.data(), cs.size(), cs16, (size_t)Q27_HID * sK16, &ws2)) {
                    std::fprintf(stderr, "Q27_MXFP4_MLP: L%d down card %d scale span not expressible -- skipping\n", L, c);
                    ++skipped; continue;
                }
                if (hipSetDevice(c) != hipSuccess) { ++skipped; continue; }
                if (hipMemcpy((void*)dst->w,  cw.data(),   cw.size(),   hipMemcpyHostToDevice) != hipSuccess ||
                    hipMemcpy((void*)dst->gs, cs16.data(), cs16.size(), hipMemcpyHostToDevice) != hipSuccess) {
                    std::fprintf(stderr, "Q27_MXFP4_MLP: L%d down card %d upload failed\n", L, c); ++skipped; continue; }
                dst->ws2 = ws2; dst->gs_mx = 0; dst->gs_fast = 0;
                ++done;
            }
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "Q27_MXFP4_MLP: %d MLP shards repacked from OCP-MXFP4 (E2M1 + E8M0/32 -> "
                         "E4M3/16 x ws2), %d skipped, from %s\n", done, skipped, path);
    std::fflush(stderr);
    return done;
}


// ---- Q27_FP8_MLP: load the block-scaled FP8 MLP from Qwen/Qwen3.8-27B-FP8 ----------------------
//
// One shard per layer (layers-<L>.safetensors), tensors:
//   gate/up .weight            F8_E4M3 [INTER][HID]        .weight_scale_inv BF16 [INTER/128][HID/128]
//   down    .weight            F8_E4M3 [HID][INTER]        .weight_scale_inv BF16 [HID/128][INTER/128]
// gate/up are COLUMN-parallel here (each card owns a contiguous ROW range of INTER); down is
// ROW-parallel (each card owns a contiguous COLUMN range of INTER). Both scale planes are compacted
// to the card's own block range at load, so the kernel indexes (row>>7, k>>7) shard-locally and
// carries no card offset -- the same reason the weight bytes are sliced rather than strided at use.
static int q27_fp8b_find(const std::string& h, const char* name,
                         long long* a, long long* b, long long* d0, long long* d1, std::string* dt) {
    std::string key = std::string("\"") + name + "\":";
    size_t p = h.find(key); if (p == std::string::npos) return 0;
    size_t dq = h.find("\"dtype\":\"", p); size_t sh = h.find("\"shape\":[", p); size_t of = h.find("\"data_offsets\":[", p);
    if (dq == std::string::npos || sh == std::string::npos || of == std::string::npos) return 0;
    size_t de = h.find('"', dq + 9); if (de == std::string::npos) return 0;
    *dt = h.substr(dq + 9, de - (dq + 9));
    if (std::sscanf(h.c_str() + sh + 9, "%lld,%lld", d0, d1) != 2) return 0;
    if (std::sscanf(h.c_str() + of + 16, "%lld,%lld", a, b) != 2) return 0;
    return 1;
}
// Controlled format/schedule comparison: replace the existing owner-local INT8
// prefill mirrors with mirrors made from the same FP8 checkpoint used by decode.
// No new resident copy and no change to the large-M execution kernels.
void q27_storage_finish(void) { q27storage::loader().finish(); }
void q27_storage_owner_ready(int device) { q27storage::loader().owner_ready(device,"all-weights-with-decode-mirrors"); }

int q27_fp8_ls_prefill_mirrors(q27_model_t* m, int ndev, const char* dir) {
    if (!m || !dir || !*dir || m->ls_ndev != ndev) return -1;
    int done = 0;
    for (int L = 0; L < Q27_LAYERS; ++L) {
        int owner = -1;
        for (int g = 0; g < ndev; ++g) if (q27_ls_owned(L, g, ndev)) owner = g;
        if (owner < 0 || hipSetDevice(m->tp_dev[owner]) != hipSuccess) return -1;
        q27_layer_t& H = m->tpl_ls[(size_t)L * ndev + owner];
        char path[1024]; std::snprintf(path, sizeof path, "%s/layers-%d.safetensors", dir, L);
        std::FILE* f = std::fopen(path, "rb");
        if (!f) return -1;
        unsigned long long hlen = 0;
        if (std::fread(&hlen, 8, 1, f) != 1 || !hlen || hlen > (1u << 24)) { std::fclose(f); return -1; }
        std::string hdr(hlen, '\0');
        if (std::fread(&hdr[0], 1, hlen, f) != hlen) { std::fclose(f); return -1; }
        for (int which = 0; which < 3; ++which) {
            const char* fam = which == 0 ? "gate_proj" : which == 1 ? "up_proj" : "down_proj";
            auto* dst = which == 0 ? &H.gate8r : which == 1 ? &H.up8r : &H.down8r;
            const int rows = which == 2 ? Q27_HID : Q27_INTER;
            const int K = which == 2 ? Q27_INTER : Q27_HID;
            char nw[256], ns[256];
            std::snprintf(nw, sizeof nw, "model.language_model.layers.%d.mlp.%s.weight", L, fam);
            std::snprintf(ns, sizeof ns, "model.language_model.layers.%d.mlp.%s.weight_scale_inv", L, fam);
            long long wa, wb, r, k, sa, sb, sr, sk; std::string dtw, dts;
            if (!dst->w || !dst->s || dst->rows != rows || dst->K != K ||
                !q27_fp8b_find(hdr, nw, &wa, &wb, &r, &k, &dtw) ||
                !q27_fp8b_find(hdr, ns, &sa, &sb, &sr, &sk, &dts) ||
                dtw != "F8_E4M3" || dts != "BF16" || r != rows || k != K ||
                sr != rows / 128 || sk != K / 128 || wa < 0 || sa < 0 ||
                wb - wa != (long long)rows * K || sb - sa != sr * sk * 2) {
                std::fprintf(stderr, "Q27_LS_FP8_PREFILL: invalid layer %d %s mirror/source\n", L, fam);
                std::fclose(f); return -1;
            }
            std::vector<unsigned char> w((size_t)(wb-wa)), bs((size_t)(sb-sa));
            if (fseeko(f, (off_t)(8 + hlen + wa), SEEK_SET) || std::fread(w.data(), 1, w.size(), f) != w.size() ||
                fseeko(f, (off_t)(8 + hlen + sa), SEEK_SET) || std::fread(bs.data(), 1, bs.size(), f) != bs.size()) {
                std::fclose(f); return -1;
            }
            void *dw = nullptr, *ds = nullptr;
            bool ok = hipMalloc(&dw, w.size()) == hipSuccess && hipMalloc(&ds, bs.size()) == hipSuccess;
            if (ok) ok = hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice) == hipSuccess &&
                         hipMemcpy(ds, bs.data(), bs.size(), hipMemcpyHostToDevice) == hipSuccess;
            q27_fp8_t src = {};
            src.rows = rows; src.K = K; src.w = (const unsigned char*)dw;
            src.bs = (const unsigned short*)ds; src.bs_kblk = K / 128; src.wscale = 1.0f;
            src.in_scale = which == 0 ? H.gate.in_scale : which == 1 ? H.up.in_scale : H.down.in_scale;
            if (ok) ok = q27_sr_fp8bs_to_i8row(&src, const_cast<signed char*>(dst->w), K,
                                               const_cast<float*>(dst->s), nullptr) != 0;
            if (ok) ok = hipDeviceSynchronize() == hipSuccess;
            if (dw) hipFree(dw);
            if (ds) hipFree(ds);
            if (!ok) { std::fclose(f); return -1; }
            dst->alpha = src.in_scale;
            ++done;
        }
        std::fclose(f);
    }
    hipSetDevice(m->tp_dev[0]);
    std::fprintf(stderr, "Q27_LS_FP8_PREFILL: %d full-width owner mirrors from FP8 checkpoint\n", done);
    return done;
}

int q27_fp8_mlp_load(q27_model_t* m, int ndev, const char* dir) {
    // Q27_FP8_MLP_DOWN: the down_proj FP8 plane is LOADED HERE AND READ BY NOBODY. The decode
    // branch (q27_main.cpp:3037) consumes f8_gate/f8_up and then takes the NVFP4 L->down, because
    // the FP8 consumer wants NATURAL-order activations while q27_swiglu_quant_b emits the
    // even-then-odd order. That is 21.25 MiB x 64 layers = 1.33 GiB/card of residency nothing
    // dereferences, spent before devalloc on the card that binds first. Default OFF; set =1 only
    // once a natural-order swiglu quantiser makes down_proj actually consumable.
    const bool want_down = q27_env_flag("Q27_FP8_MLP_DOWN", false);
    if (!m || ndev < 1 || !dir || !*dir) return 0;
    q27_model* M = m; if (!M->tp_ndev) return 0;
    if (M->ls_sr) return 0;   // Q27_LS_SR: no TP residency exists (tpl is empty); this is a TP-shard entry point
    const int NB = 128;
    // ---- Q27_MLP_WSHARE: the gate/up rows (and with them down_proj's K slice) this card owns. ----
    // Uniform INTER/ndev unless the operator asks for an asymmetric share; validated in q27.h
    // (positive multiples of 128 summing to INTER). The buffers below are sized for the LARGEST
    // share so the same staging buffer serves every card.
    int mlp_off[Q27_MAX_DEVICES] = {0}, mlp_len[Q27_MAX_DEVICES] = {0};
    q27_mlp_wshare(mlp_off, mlp_len, ndev);
    int nr_max = 0;
    for (int c = 0; c < ndev; ++c) if (mlp_len[c] > nr_max) nr_max = mlp_len[c];
    const int gu_kblk = Q27_HID / NB;                         // 40
    const int dn_nblk = Q27_HID / NB;                         // 40
    const int dn_kblk_full = Q27_INTER / NB;                  // 136
    std::vector<unsigned char> wg((size_t)nr_max * Q27_HID), wd((size_t)Q27_HID * nr_max);
    std::vector<unsigned short> sg((size_t)(nr_max / NB) * gu_kblk),
                                sd((size_t)dn_nblk * (nr_max / NB));
    std::vector<unsigned char> fdw((size_t)Q27_HID * Q27_INTER);
    std::vector<unsigned short> fds((size_t)dn_nblk * dn_kblk_full);
    int done = 0, skipped = 0;
    for (int L = 0; L < Q27_LAYERS; ++L) {
        char path[512]; std::snprintf(path, sizeof path, "%s/layers-%d.safetensors", dir, L);
        std::FILE* f = std::fopen(path, "rb");
        if (!f) { ++skipped; continue; }
        unsigned long long hlen = 0;
        if (std::fread(&hlen, 8, 1, f) != 1 || hlen == 0 || hlen > (1u << 24)) { std::fclose(f); ++skipped; continue; }
        std::string hdr(hlen, '\0');
        if (std::fread(&hdr[0], 1, hlen, f) != hlen) { std::fclose(f); ++skipped; continue; }
        const long long base = 8 + (long long)hlen;
        for (int which = 0; which < 3; ++which) {
            if (which == 2 && !want_down) continue;
            const char* fam = which == 0 ? "gate_proj" : which == 1 ? "up_proj" : "down_proj";
            char nw[256], ns[256];
            std::snprintf(nw, sizeof nw, "model.language_model.layers.%d.mlp.%s.weight", L, fam);
            std::snprintf(ns, sizeof ns, "model.language_model.layers.%d.mlp.%s.weight_scale_inv", L, fam);
            long long wa, wb, wd0, wd1, sa, sb, sd0, sd1; std::string dtw, dts;
            if (!q27_fp8b_find(hdr, nw, &wa, &wb, &wd0, &wd1, &dtw) ||
                !q27_fp8b_find(hdr, ns, &sa, &sb, &sd0, &sd1, &dts)) { ++skipped; continue; }
            if (dtw != "F8_E4M3" || dts != "BF16") { ++skipped; continue; }
            const bool isdown = (which == 2);
            if (!isdown && (wd0 != Q27_INTER || wd1 != Q27_HID)) { ++skipped; continue; }
            if ( isdown && (wd0 != Q27_HID   || wd1 != Q27_INTER)) { ++skipped; continue; }
            if (isdown) {   // read the whole tensor once, slice columns per card
                if (fseeko(f, (off_t)(base + wa), SEEK_SET) != 0 ||
                    std::fread(fdw.data(), 1, fdw.size(), f) != fdw.size()) { ++skipped; continue; }
                if (fseeko(f, (off_t)(base + sa), SEEK_SET) != 0 ||
                    std::fread(fds.data(), 2, fds.size(), f) != fds.size()) { ++skipped; continue; }
            }
            for (int c = 0; c < M->tp_ndev; ++c) {
                q27_layer_t* Tp = &M->tpl[(size_t)L * M->tp_ndev + c];
                q27_fp8_t* dst = which == 0 ? &Tp->f8_gate : which == 1 ? &Tp->f8_up : &Tp->f8_down;
                const int nr = mlp_len[c];                    // gate/up rows == down K for this card
                const int ks = mlp_len[c];
                const int r0k = mlp_off[c];
                const int gu_nblk = nr / NB, dn_kblk = ks / NB;
                const size_t wbytes = isdown ? (size_t)Q27_HID * ks : (size_t)nr * Q27_HID;
                const size_t sbytes = (isdown ? (size_t)dn_nblk * dn_kblk
                                              : (size_t)gu_nblk * gu_kblk) * 2;
                if (!isdown) {
                    const long long r0 = (long long)r0k;
                    if (fseeko(f, (off_t)(base + wa + r0 * Q27_HID), SEEK_SET) != 0 ||
                        std::fread(wg.data(), 1, wbytes, f) != wbytes) { ++skipped; continue; }
                    if (fseeko(f, (off_t)(base + sa + (long long)(r0k / NB) * gu_kblk * 2), SEEK_SET) != 0 ||
                        std::fread(sg.data(), 2, sbytes / 2, f) != sbytes / 2) { ++skipped; continue; }
                } else {
                    for (int r = 0; r < Q27_HID; ++r)
                        std::memcpy(&wd[(size_t)r * ks], &fdw[(size_t)r * Q27_INTER + (size_t)r0k], ks);
                    for (int r = 0; r < dn_nblk; ++r)
                        std::memcpy(&sd[(size_t)r * dn_kblk], &fds[(size_t)r * dn_kblk_full + (size_t)(r0k / NB)],
                                    (size_t)dn_kblk * 2);
                }
                if (hipSetDevice(c) != hipSuccess) { ++skipped; continue; }
                void* dw = nullptr; void* ds = nullptr;
                if (hipMalloc(&dw, wbytes) != hipSuccess) { ++skipped; continue; }
                if (hipMalloc(&ds, sbytes) != hipSuccess) { hipFree(dw); ++skipped; continue; }
                if (hipMemcpy(dw, isdown ? (const void*)wd.data() : (const void*)wg.data(), wbytes,
                              hipMemcpyHostToDevice) != hipSuccess ||
                    hipMemcpy(ds, isdown ? (const void*)sd.data() : (const void*)sg.data(), sbytes,
                              hipMemcpyHostToDevice) != hipSuccess) {
                    hipFree(dw); hipFree(ds); ++skipped; continue; }
                dst->w = (const unsigned char*)dw;
                dst->bs = (const unsigned short*)ds;
                dst->bs_kblk = isdown ? dn_kblk : gu_kblk;
                dst->wscale = 1.0f;                       // the block plane carries the weight scale
                dst->in_scale = isdown ? Tp->down.in_scale : (which ? Tp->up.in_scale : Tp->gate.in_scale);
                dst->rows = isdown ? Q27_HID : nr;
                dst->K    = isdown ? ks : Q27_HID;
                ++done;
            }
        }
        std::fclose(f);
    }
    // The NVFP4 gate/up shards may already have been dropped on the promise that FP8 covers every
    // layer (Q27_FP8_DROP_NV_GU, q27_upload_tp). If that promise is not kept, decode would take the
    // fallback branch into a NULL plane and fault somewhere unrelated. Refuse here instead: a
    // dropped representation is only safe when its replacement is proven complete, at the seam.
    const bool dn_drop = q27_env_flag("Q27_FP8_DROP_NV_DN", false);
    if (dn_drop && !want_down) {
        std::fprintf(stderr, "FATAL Q27_FP8_DROP_NV_DN: the NVFP4 down shards were dropped but Q27_FP8_MLP_DOWN is not set, "
                             "so no down projection would be resident. Set Q27_FP8_MLP_DOWN=1.\n");
        std::fflush(stderr);
        std::abort();
    }
    if (q27_env_flag("Q27_FP8_DROP_NV_GU", false) || dn_drop) {
        const int expect = (want_down ? 3 : 2) * Q27_LAYERS * M->tp_ndev;
        if (done != expect) {
            std::fprintf(stderr, "FATAL Q27_FP8_DROP_NV_GU: FP8 covered %d of %d projections "
                                 "(%d skipped) but the NVFP4 gate/up shards were already dropped. "
                                 "Refusing to serve a null gate/up.\n", done, expect, skipped);
            std::fflush(stderr);
            std::abort();
        }
    }
    std::fprintf(stderr, "Q27_FP8_MLP: %d MLP projections resident as block-scaled FP8 "
                         "(e4m3 [rows][K] + BF16 [rows/128][K/128]), %d skipped, from %s\n",
                 done, skipped, dir);
    std::fflush(stderr);
    return done;
}

// ---- Q27_LS_SR: the block-scaled FP8 MLP at FULL WIDTH on the OWNER card (2026-09-17) -----------
//
// The single-residency counterpart of q27_fp8_mlp_load. There is no TP shard to fill: each layer's
// MLP lives ONCE, on the card that owns the layer under q27_ls_owned, and the handles land in
// M->tpl_ls[L*ndev + g] (the layer-split column) -- where the SR sweep and the SR decode read.
//
//   f8_gate, f8_up : the full [17408][5120] e4m3 tensor + the full bs plane [136][40] BF16
//                    (rows = INTER, K = HID, bs_kblk = 40, wscale = 1.0f)
//   f8_dn[c], c<4  : down_proj K-slice c = columns [c*4352, +4352) of every row, host-gathered
//                    exactly as q27_fp8_mlp_load gathers card c's shard, with the bs plane compacted
//                    to that column range: [40][34] (rows = HID, K = 4352, bs_kblk = 34). Four slices
//                    because the block-scaled consumer refuses K > 5120 per launch (q27_load.h, f8_dn).
//   f8_down        : left NULL in SR mode -- the slices are the only down representation.
//
// in_scale PROVENANCE. The FP8 checkpoint carries no activation scale (config.json: quant_method
// "fp8", activations quantised dynamically), and in SR mode the NVFP4 gate/up/down PLANES are not
// uploaded -- but the NVFP4 loader never uploaded the scalar either: Tp::nvfp4_col / Grp::nvfp4 read
// it on the host with sc("<base>.input_scale") and stamp it into the handle. So the value is read
// here from the SAME bytes, the mmap'd NVFP4 checkpoint's F32 [] tensor
//     model.language_model.layers.<L>.mlp.{gate,up,down}_proj.input_scale
// via q27_host_ptr, and cross-checked against the in_scale tp_plan stamped into the layer-split
// handle (that scalar read survives SR because only the device-pointer BLOCKS are dropped). gate
// and up must agree (contract 1: ONE activation quantization feeds both). Either check failing is a
// seam failure and is fatal. There is deliberately no config.json fallback: the scalar exists in
// every layer of this checkpoint, so its absence means the wrong checkpoint is mounted -- exactly
// the case that must not be guessed around.
//
// ALLOCATION: one hipMalloc arena per layer on the owner card (gate.w, gate.bs, up.w, up.bs, then
// the four down slices w+bs, 256-byte aligned) = 267,420,160 B = 255.03 MiB per layer, recorded in
// M->arenas (sr_mlp=1, layer) and summed per card index in M->sr_mlp_bytes / sr_mlp_layers.
// FAILS CLOSED: any layer not fully covered (file, tensor, shape, scalar, allocation or copy) is
// reported per layer and the function aborts after the sweep -- the NVFP4 MLP was never uploaded in
// SR mode, so there is no representation to fall back to. A dropped representation is only safe when
// its replacement is proven complete at the seam.
//
// DEVICE: hipSetDevice() is restored to M->tp_dev[0] (the caller's device 0) on every exit path.
// Q27_LS_SR_DRYRUN=1 prices the plan from the checkpoint headers and the ownership rule alone (no
// prior upload needed) and returns before any hipSetDevice/hipMalloc; an incomplete checkpoint then
// prints the FATAL line and returns -1 instead of aborting.
int q27_fp8_mlp_load_sr(q27_model_t* m, int ndev, const char* dir) {
    if (!m || ndev < 1 || ndev > Q27_MAX_DEVICES || !dir || !*dir) return 0;
    q27_model* M = m;
    const bool dry = q27_env_flag("Q27_LS_SR_DRYRUN", false);
    if (!dry && (!M->ls_sr || M->ls_ndev != ndev)) {
        std::fprintf(stderr, "FATAL q27_fp8_mlp_load_sr: needs q27_upload_ls in Q27_LS_SR mode first "
                             "(ls_sr=%d ls_ndev=%d ndev=%d)\n", M->ls_sr, M->ls_ndev, ndev);
        std::fflush(stderr);
        std::abort();
    }
    enum { NB = 128, NSL = 4, KSL = Q27_INTER / NSL };                          // 4352 per down slice
    static_assert(Q27_INTER % NSL == 0 && KSL % NB == 0 && KSL <= 5120,
                  "down_proj K-slices must be 128-blocked and fit one NSTEP=5 block-scaled launch");
    static_assert(Q27_INTER % NB == 0 && Q27_HID % NB == 0, "block-scaled planes need 128-multiple rows and K");
    const int gu_nblk = Q27_INTER / NB, gu_kblk = Q27_HID / NB;                      // 136, 40
    const int dn_nblk = Q27_HID / NB, dn_kblk_full = Q27_INTER / NB, dn_kblk = KSL / NB;   // 40, 136, 34
    const size_t gw_b   = (size_t)Q27_INTER * Q27_HID;          // 89,128,960  gate/up weight, full
    const size_t gbs_b  = (size_t)gu_nblk * gu_kblk * 2;        // 10,880      gate/up bs, full
    const size_t dwf_b  = (size_t)Q27_HID * Q27_INTER;          // 89,128,960  down weight, full (host only)
    const size_t dbsf_b = (size_t)dn_nblk * dn_kblk_full * 2;   // 10,880      down bs, full (host only)
    const size_t dws_b  = (size_t)Q27_HID * KSL;                // 22,282,240  down weight, one slice
    const size_t dbss_b = (size_t)dn_nblk * dn_kblk * 2;        // 2,720       down bs, one slice
    const size_t per_layer = 2 * (alignup(gw_b) + alignup(gbs_b)) + (size_t)NSL * (alignup(dws_b) + alignup(dbss_b));

    const char* LP = "model.language_model.layers";
    std::vector<unsigned char> wbuf, fdw, wd;
    std::vector<unsigned short> sg, fds, sd;
    if (!dry) { wbuf.resize(gw_b); fdw.resize(dwf_b); wd.resize(dws_b); sg.resize(gbs_b / 2); fds.resize(dbsf_b / 2); sd.resize(dbss_b / 2); }

    // the NVFP4 checkpoint's F32 [] input_scale for one MLP family of one layer (see PROVENANCE above)
    auto scalar = [&](int L, const char* fam, float* out) -> bool {
        char n[256]; std::snprintf(n, sizeof n, "%s.%d.mlp.%s.input_scale", LP, L, fam);
        size_t nb = 0; int dt = 0, nd = 0; long long sh[Q27_MAX_DIMS];
        const void* p = q27_host_ptr(m, n, &nb, &dt, &nd, sh);
        if (!p || dt != Q27_DT_F32 || nb != 4) return false;
        std::memcpy(out, p, 4);
        return true;
    };

    long long card_bytes[Q27_MAX_DEVICES]; int card_layers[Q27_MAX_DEVICES];
    memset(card_bytes, 0, sizeof card_bytes); memset(card_layers, 0, sizeof card_layers);
    int done = 0, uncovered = 0, cur_dev = -1;
    for (int L = 0; L < Q27_LAYERS; ++L) {
        int g = -1;
        for (int c = 0; c < ndev; ++c) if (q27_ls_owned(L, c, ndev)) { g = c; break; }
        if (g < 0) { std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d has NO owner (Q27_LS_BLK / ndev=%d)\n", L, ndev); ++uncovered; continue; }

        float is_g = 0.0f, is_u = 0.0f, is_d = 0.0f;
        if (!scalar(L, "gate_proj", &is_g) || !scalar(L, "up_proj", &is_u) || !scalar(L, "down_proj", &is_d)) {
            std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: mlp input_scale scalar missing or not F32 [] in the NVFP4 checkpoint\n", L);
            ++uncovered; continue;
        }
        if (is_g != is_u) {
            std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: gate.input_scale %.9g != up.input_scale %.9g (contract 1)\n",
                         L, (double)is_g, (double)is_u);
            ++uncovered; continue;
        }
        q27_layer_t* Ls = dry ? nullptr : &M->tpl_ls[(size_t)L * ndev + g];
        if (Ls) {
            if (Ls->layer != L || Ls->device < 0) {
                std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: layer-split handle on card %d is not resident\n", L, g);
                ++uncovered; continue;
            }
            if (Ls->gate.w || Ls->gate.gs || Ls->up.w || Ls->up.gs || Ls->down.w || Ls->down.gs) {
                std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: an NVFP4 MLP plane is resident on card %d -- not an SR upload\n", L, g);
                ++uncovered; continue;
            }
            if ((Ls->gate.in_scale != 0.0f && Ls->gate.in_scale != is_g) ||
                (Ls->up.in_scale   != 0.0f && Ls->up.in_scale   != is_u) ||
                (Ls->down.in_scale != 0.0f && Ls->down.in_scale != is_d)) {
                std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: stamped in_scale (%.9g/%.9g/%.9g) != checkpoint scalar (%.9g/%.9g/%.9g)\n",
                             L, (double)Ls->gate.in_scale, (double)Ls->up.in_scale, (double)Ls->down.in_scale,
                             (double)is_g, (double)is_u, (double)is_d);
                ++uncovered; continue;
            }
        }

        char path[512]; std::snprintf(path, sizeof path, "%s/layers-%d.safetensors", dir, L);
        std::FILE* f = std::fopen(path, "rb");
        if (!f) { std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: cannot open %s\n", L, path); ++uncovered; continue; }
        unsigned long long hlen = 0;
        if (std::fread(&hlen, 8, 1, f) != 1 || hlen == 0 || hlen > (1u << 24)) {
            std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: bad safetensors header in %s\n", L, path); std::fclose(f); ++uncovered; continue; }
        std::string hdr(hlen, '\0');
        if (std::fread(&hdr[0], 1, hlen, f) != hlen) { std::fclose(f); ++uncovered; continue; }
        const long long base = 8 + (long long)hlen;
        struct Fam { const char* fam; long long wa, wb, wd0, wd1, sa, sb, sd0, sd1; };
        Fam T[3] = { {"gate_proj", 0,0,0,0,0,0,0,0}, {"up_proj", 0,0,0,0,0,0,0,0}, {"down_proj", 0,0,0,0,0,0,0,0} };
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i) {
            char nw[256], ns[256]; std::string dtw, dts;
            std::snprintf(nw, sizeof nw, "%s.%d.mlp.%s.weight", LP, L, T[i].fam);
            std::snprintf(ns, sizeof ns, "%s.%d.mlp.%s.weight_scale_inv", LP, L, T[i].fam);
            if (!q27_fp8b_find(hdr, nw, &T[i].wa, &T[i].wb, &T[i].wd0, &T[i].wd1, &dtw) ||
                !q27_fp8b_find(hdr, ns, &T[i].sa, &T[i].sb, &T[i].sd0, &T[i].sd1, &dts)) {
                std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: %s weight/weight_scale_inv missing in %s\n", L, T[i].fam, path); ok = false; break; }
            if (dtw != "F8_E4M3" || dts != "BF16") {
                std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: %s dtype %s/%s, expected F8_E4M3/BF16\n", L, T[i].fam, dtw.c_str(), dts.c_str()); ok = false; break; }
            const bool isdown = (i == 2);
            const long long er0 = isdown ? Q27_HID : Q27_INTER, er1 = isdown ? Q27_INTER : Q27_HID;
            const long long es0 = isdown ? dn_nblk : gu_nblk,  es1 = isdown ? dn_kblk_full : gu_kblk;
            if (T[i].wd0 != er0 || T[i].wd1 != er1 || T[i].sd0 != es0 || T[i].sd1 != es1 ||
                T[i].wb - T[i].wa != er0 * er1 || T[i].sb - T[i].sa != es0 * es1 * 2) {
                std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: %s shape w[%lld,%lld] s[%lld,%lld] (bytes %lld/%lld), expected w[%lld,%lld] s[%lld,%lld]\n",
                             L, T[i].fam, T[i].wd0, T[i].wd1, T[i].sd0, T[i].sd1, T[i].wb - T[i].wa, T[i].sb - T[i].sa, er0, er1, es0, es1);
                ok = false; break; }
        }
        if (!ok) { std::fclose(f); ++uncovered; continue; }
        card_layers[g] += 1; card_bytes[g] += (long long)per_layer;
        if (dry) { std::fclose(f); done += 6; continue; }        // priced; nothing touched

        const int dev = M->tp_dev[g];
        if (dev != cur_dev) {
            if (hipSetDevice(dev) != hipSuccess) { std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: hipSetDevice(%d) failed\n", L, dev); std::fclose(f); ++uncovered; continue; }
            cur_dev = dev;
        }
        void* base_d = nullptr;
        if (hipMalloc(&base_d, per_layer) != hipSuccess) {
            std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: hipMalloc(%zu B = %.1f MiB) on device %d failed\n", L, per_layer, (double)per_layer / 1048576.0, dev);
            std::fclose(f); ++uncovered; continue;
        }
        size_t used = 0; bool cok = true;
        for (int i = 0; i < 2 && cok; ++i) {                     // gate, up: full tensors, straight in
            if (fseeko(f, (off_t)(base + T[i].wa), SEEK_SET) != 0 || std::fread(wbuf.data(), 1, gw_b, f) != gw_b) { cok = false; break; }
            char* dw = (char*)base_d + used; used += alignup(gw_b);
            if (copy_in(dw, wbuf.data(), gw_b, nullptr, 0) != Q27_OK) { cok = false; break; }
            if (fseeko(f, (off_t)(base + T[i].sa), SEEK_SET) != 0 || std::fread(sg.data(), 2, gbs_b / 2, f) != gbs_b / 2) { cok = false; break; }
            char* ds = (char*)base_d + used; used += alignup(gbs_b);
            if (copy_in(ds, (const unsigned char*)sg.data(), gbs_b, nullptr, 0) != Q27_OK) { cok = false; break; }
            q27_fp8_t* dst = (i == 0) ? &Ls->f8_gate : &Ls->f8_up;
            dst->w = (const unsigned char*)dw; dst->bs = (const unsigned short*)ds; dst->bs_kblk = gu_kblk;
            dst->wscale = 1.0f; dst->in_scale = (i == 0) ? is_g : is_u; dst->rows = Q27_INTER; dst->K = Q27_HID;
        }
        if (cok) {                                               // down: read whole once, gather four column slices
            if (fseeko(f, (off_t)(base + T[2].wa), SEEK_SET) != 0 || std::fread(fdw.data(), 1, dwf_b, f) != dwf_b ||
                fseeko(f, (off_t)(base + T[2].sa), SEEK_SET) != 0 || std::fread(fds.data(), 2, dbsf_b / 2, f) != dbsf_b / 2) cok = false;
            for (int c = 0; c < NSL && cok; ++c) {
                for (int r = 0; r < Q27_HID; ++r)
                    std::memcpy(&wd[(size_t)r * KSL], &fdw[(size_t)r * Q27_INTER + (size_t)c * KSL], (size_t)KSL);
                for (int r = 0; r < dn_nblk; ++r)
                    std::memcpy(&sd[(size_t)r * dn_kblk], &fds[(size_t)r * dn_kblk_full + (size_t)c * dn_kblk], (size_t)dn_kblk * 2);
                char* dw = (char*)base_d + used; used += alignup(dws_b);
                if (copy_in(dw, wd.data(), dws_b, nullptr, 0) != Q27_OK) { cok = false; break; }
                char* ds = (char*)base_d + used; used += alignup(dbss_b);
                if (copy_in(ds, (const unsigned char*)sd.data(), dbss_b, nullptr, 0) != Q27_OK) { cok = false; break; }
                q27_fp8_t* dst = &Ls->f8_dn[c];
                dst->w = (const unsigned char*)dw; dst->bs = (const unsigned short*)ds; dst->bs_kblk = dn_kblk;
                dst->wscale = 1.0f; dst->in_scale = is_d; dst->rows = Q27_HID; dst->K = KSL;
            }
        }
        std::fclose(f);
        if (!cok || used != per_layer) {
            std::fprintf(stderr, "Q27_LS_SR fp8 mlp: layer %d: read/upload failed on device %d (used %zu of %zu)\n", L, dev, used, per_layer);
            hipFree(base_d);
            Ls->f8_gate = q27_fp8_t(); Ls->f8_up = q27_fp8_t();
            for (int c = 0; c < NSL; ++c) Ls->f8_dn[c] = q27_fp8_t();
            ++uncovered; continue;
        }
        Arena A; A.device = dev; A.base = base_d; A.size = per_layer; A.fp8proj = 0; A.sr_mlp = 1; A.layer = L;
        M->arenas.push_back(A);
        M->sr_mlp_bytes[g] += (long long)per_layer; M->sr_mlp_layers[g] += 1;
        done += 6;
    }
    if (!dry) hipSetDevice(M->tp_dev[0]);                        // platform law: the caller's device, every exit path

    for (int g = 0; g < ndev; ++g)
        std::fprintf(stderr, "Q27_LS_SR FP8 MLP card %d (device %d): %d layers x %.2f MiB = %.1f MiB (%.3f GiB)%s\n",
                     g, dry ? g : M->tp_dev[g], card_layers[g], (double)per_layer / 1048576.0,
                     (double)card_bytes[g] / 1048576.0, (double)card_bytes[g] / 1073741824.0,
                     dry ? "   [DRYRUN: nothing allocated]" : "");
    const int expect = 6 * Q27_LAYERS;                           // gate + up + 4 down slices, every layer has one owner
    if (uncovered || done != expect) {
        std::fprintf(stderr, "FATAL Q27_LS_SR: block-scaled FP8 MLP covered %d of %d projections (%d layer(s) not covered) "
                             "from %s. The NVFP4 MLP was never uploaded in SR mode, so there is nothing to fall back to. "
                             "Refusing to serve a null MLP.\n", done, expect, uncovered, dir);
        std::fflush(stderr);
        if (dry) return -1;
        std::abort();
    }
    std::fprintf(stderr, "Q27_LS_SR: %d MLP projections resident ONCE as block-scaled FP8 at full width on their owner cards "
                         "(gate/up [%d][%d] + bs [%d][%d]; down as %d K-slices [%d][%d] + bs [%d][%d]), from %s\n",
                 done, Q27_INTER, Q27_HID, gu_nblk, gu_kblk, (int)NSL, Q27_HID, (int)KSL, dn_nblk, dn_kblk, dir);
    std::fflush(stderr);
    return done;
}

// ---- Q27_LS_SR: release the fp8 projection temporaries once main's int8 mirrors carry every consumer.
// Order: drain every device that holds a temporary (the mirror-build kernels may still be in flight on
// another stream; a free under a running reader is recycled VRAM), free each temporary on ITS device,
// then replace the seven fp8 weight pointers of every owned layer's layer-split handle by a unique
// non-canonical (x86-64) / unmapped (GPU VA) sentinel so a stray fp8 launch faults loudly instead of
// reading recycled memory -- the same discipline as q27_tp_free_fp8_proj. The sentinel range
// 0xDEAD2xxx_xxxx is disjoint from the TP path's 0xDEAD0... and the draft head's 0xDEAD1..., so
// every sentinel in the process is unique. `rekey` is called exactly as q27_tp_free_fp8_proj calls it
// (old device pointer -> sentinel); the side table is keyed by the HANDLE's address today, so this is
// a no-op that preserves the convention, and it may be nullptr. Restores device tp_dev[0] on exit. ----
size_t q27_ls_sr_release_fp8(q27_model_t* m, int (*rekey)(const void* oldw, const void* neww)) {
    if (!m) return 0;
    q27_model* M = m;
    if (!M->ls_sr) { std::fprintf(stderr, "q27_ls_sr_release_fp8: not in Q27_LS_SR mode, nothing to release\n"); std::fflush(stderr); return 0; }
    size_t per_dev[Q27_MAX_DEVICES]; int per_n[Q27_MAX_DEVICES], synced[Q27_MAX_DEVICES];
    memset(per_dev, 0, sizeof per_dev); memset(per_n, 0, sizeof per_n); memset(synced, 0, sizeof synced);
    for (size_t i = 0; i < M->sr_fp8_temps.size(); ++i) {
        const int d = M->sr_fp8_temps[i].device;
        if (d < 0 || d >= Q27_MAX_DEVICES || synced[d]) continue;
        if (hipSetDevice(d) == hipSuccess) hipDeviceSynchronize();
        synced[d] = 1;
    }
    for (size_t i = 0; i < M->sr_fp8_temps.size(); ++i) {
        q27_model::SrTemp& T = M->sr_fp8_temps[i];
        if (!T.ptr) continue;
        if (hipSetDevice(T.device) != hipSuccess) { std::fprintf(stderr, "q27_ls_sr_release_fp8: hipSetDevice(%d) failed; %p leaked\n", T.device, T.ptr); continue; }
        if (hipFree(T.ptr) == hipSuccess) { per_dev[T.device] += T.bytes; ++per_n[T.device]; }
        else std::fprintf(stderr, "q27_ls_sr_release_fp8: hipFree(%p) on device %d failed (layer %d)\n", T.ptr, T.device, T.layer);
        T.ptr = nullptr;
    }
    M->sr_fp8_temps.clear();
    uint64_t n = 0; int sentinels = 0;
    for (int g = 0; g < M->ls_ndev; ++g) {
        for (int L = 0; L < Q27_LAYERS; ++L) {
            if (!q27_ls_owned(L, g, M->ls_ndev)) continue;
            q27_layer_t& H = M->tpl_ls[(size_t)L * M->ls_ndev + g];
            const unsigned char** f[7] = { &H.q_proj.w, &H.k_proj.w, &H.v_proj.w, &H.o_proj.w, &H.in_qkv.w, &H.in_z.w, &H.out_proj.w };
            for (int k = 0; k < 7; ++k) {
                if (!*f[k]) continue;
                const unsigned char* sent = (const unsigned char*)(uintptr_t)(0xDEAD20000000ULL + (++n) * 4096ULL);
                if (rekey) rekey((const void*)*f[k], (const void*)sent);
                *f[k] = sent;
                ++sentinels;
            }
        }
    }
    for (int g = 0; g < M->ls_ndev; ++g) {
        const int d = M->tp_dev[g];
        std::fprintf(stderr, "Q27_LS_SR release card %d (device %d): %d fp8 projection temporaries freed, %.1f MiB released\n",
                     g, d, per_n[d], (double)per_dev[d] / 1048576.0);
    }
    std::fprintf(stderr, "Q27_LS_SR release: %d fp8 handles now carry unmapped sentinels; their consumers must be the int8 side-table mirrors\n", sentinels);
    std::fflush(stderr);
    hipSetDevice(M->tp_dev[0]);
    // A non-void function that falls off its end is undefined behaviour, and at -O3 clang emits no
    // epilogue for it: the first SR boot returned into a checkpoint mmap address (dmesg: "segfault
    // at 7540000000cf ip 00007540000000cf ... in model-00000-of-00003.safetensors"). Return the total.
    size_t total = 0;
    for (int d = 0; d < Q27_MAX_DEVICES; ++d) total += per_dev[d];
    return total;
}
// ---- Q27_E4M3_FAST: one-shot eligibility pass over the decode gate/up scale planes -------------
// Runs once after upload and only when Q27_E4M3_FAST=1. Walks the TP layers on each card and asks
// q27_nvfp4_gs_validate to look at the ACTUAL RESIDENT BYTES -- not the checkpoint, because a plane
// can be manufactured at runtime (q27_f32_to_e4m3 under Q27_FP8X4) and a file scan would wrongly
// certify it. Anything not certified keeps the full q27_e4m3 decoder, so with the flag off, or with
// a single ineligible plane, the shipped path is unchanged.
extern "C" int q27_nvfp4_gs_validate(q27_nvfp4_t* w, hipStream_t s);
int q27_nvfp4_validate_gu(q27_model_t* m, int ndev) {
    if (!m || ndev < 1) return 0;
    q27_model* M = m;
    if (!M->tp_ndev) return 0;
    if (M->ls_sr) return 0;   // Q27_LS_SR: no TP residency exists (tpl is empty); this is a TP-shard entry point
    int ok = 0, total = 0;
    const bool down_fs = q27_env_flag("Q27_DOWN_FS", false);
    for (int i = 0; i < M->tp_ndev; ++i) {
        if (hipSetDevice(i) != hipSuccess) continue;
        for (int L = 0; L < Q27_LAYERS; ++L) {
            q27_layer_t* Tp = &M->tpl[(size_t)L * M->tp_ndev + i];
            if (Tp->gate.gs) { ++total; ok += q27_nvfp4_gs_validate(&Tp->gate, 0); }
            if (Tp->up.gs)   { ++total; ok += q27_nvfp4_gs_validate(&Tp->up, 0); }
            if (down_fs && Tp->down.gs) { ++total; ok += q27_nvfp4_gs_validate(&Tp->down, 0); }
        }
    }
    std::fprintf(stderr, "Q27_E4M3_FAST: %d of %d decode gate/up scale planes eligible "
                         "(8 <= b <= 126); anything not certified keeps the full decoder\n", ok, total);
    std::fflush(stderr);
    return ok;
}

int q27_ls_mlp_evict(q27_model_t* m, int layer, int card, int ndev) {
    if (!m || layer < 0 || layer >= Q27_LAYERS || ndev < 1) return 0;
    q27_model* M = m;
    if (!M->ls_ndev || !M->tp_ndev) return 0;
    if (M->ls_sr) return 0;   // Q27_LS_SR: nothing to evict (no int8 MLP mirror, no TP shard); tpl is empty
    q27_layer_t* Ls = nullptr; int tpi = -1;
    for (int i = 0; i < M->ls_ndev; ++i) if (i == card) Ls = &M->tpl_ls[(size_t)layer * M->ls_ndev + i];
    for (int i = 0; i < M->tp_ndev; ++i) if (i == card) tpi = i;
    if (!Ls || tpi < 0) return 0;
    q27_layer_t* Tp = &M->tpl[(size_t)layer * M->tp_ndev + tpi];
    if (!Ls->gate8r.w || !Ls->up8r.w || !Ls->down8r.w) return 0;   // not converted: nothing to give back
    if (hipSetDevice(card) != hipSuccess) return 0;

    const char* LP = "model.language_model.layers";
    const char* fam[3] = { "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
    const int rows[3] = { Q27_INTER, Q27_INTER, Q27_HID };
    const int Kfull[3] = { Q27_HID, Q27_HID, Q27_INTER };
    const void* hw[3] = { nullptr, nullptr, nullptr };
    const void* hs[3] = { nullptr, nullptr, nullptr };
    q27_nvfp4_t nv[3];
    q27_nvfp4_t* lsh[3] = { &Ls->gate, &Ls->up, &Ls->down };
    // EVERY ALLOCATION THIS ATTEMPT MAKES, FREED TOGETHER ON ANY FAILURE. The rebuild below
    // allocates the NVFP4 authority one family at a time, and the failure path used to free only the
    // allocation that failed -- the two families already built stayed allocated and were never
    // referenced again. On the card with the least headroom that leaks ~50 MB per attempt, which
    // gets worse the more a growth is retried and is exactly what makes it fail again: measured
    // 2026-09-15, card 3 sat at 63 MB free after a growth attempt while cards 0-2 had 1.75 GiB,
    // and its eviction then failed on every round.
    size_t ev_frA = 0, ev_frT = 0;   // Q27_EVICT_DBG: free before the NVFP4 authority is built
    hipMemGetInfo(&ev_frA, &ev_frT);
    std::vector<void*> held;
    auto fail = [&]() { for (void* p : held) hipFree(p); held.clear(); return 0; };
    for (int i = 0; i < 3; ++i) {
        char nw[256], ns[256];
        std::snprintf(nw, sizeof nw, "%s.%d.%s.weight", LP, layer, fam[i]);
        std::snprintf(ns, sizeof ns, "%s.%d.%s.weight_scale", LP, layer, fam[i]);
        size_t nbw = 0, nbs = 0; int dt = 0, nd = 0; long long sh[8] = {0};
        hw[i] = q27_host_ptr(m, nw, &nbw, &dt, &nd, sh);
        hs[i] = q27_host_ptr(m, ns, &nbs, &dt, &nd, sh);
        const size_t want_w = (size_t)rows[i] * (size_t)(Kfull[i] / 2);
        const size_t want_s = (size_t)rows[i] * (size_t)(Kfull[i] / 16);
        if (!hw[i] || nbw < want_w || !hs[i] || nbs < want_s) return 0;
        nv[i] = *lsh[i];                                   // keeps ws2 / in_scale
        void* dw = nullptr; void* ds = nullptr;
        if (hipMalloc(&dw, want_w) != hipSuccess) return fail();
        held.push_back(dw);
        if (hipMalloc(&ds, want_s) != hipSuccess) return fail();
        held.push_back(ds);
        if (hipMemcpy(dw, hw[i], want_w, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(ds, hs[i], want_s, hipMemcpyHostToDevice) != hipSuccess) return fail();
        nv[i].w = (const unsigned char*)dw; nv[i].gs = (const unsigned char*)ds;
        nv[i].rows = rows[i]; nv[i].K = Kfull[i];
    }
    // decode handles: gate/up as row views of the full-width tensors (zero allocation), down as the
    // card's column shard of the packed row.
    const int nr = Q27_INTER / ndev, Ksh = Q27_INTER / ndev, r0 = card * nr;
    q27_nvfp4_t tp_g = Tp->gate, tp_u = Tp->up, tp_d = Tp->down;
    tp_g.w  = nv[0].w  + (size_t)r0 * (size_t)(Q27_HID / 2);
    tp_g.gs = nv[0].gs + (size_t)r0 * (size_t)(Q27_HID / 16);
    tp_g.rows = nr; tp_g.K = Q27_HID;
    tp_u.w  = nv[1].w  + (size_t)r0 * (size_t)(Q27_HID / 2);
    tp_u.gs = nv[1].gs + (size_t)r0 * (size_t)(Q27_HID / 16);
    tp_u.rows = nr; tp_u.K = Q27_HID;
    {
        const size_t rb = (size_t)(Q27_INTER / 2), sl = rb / (size_t)ndev;
        const size_t sb = (size_t)(Q27_INTER / 16), ss = sb / (size_t)ndev;
        void* dw = nullptr; void* ds = nullptr;
        if (hipMalloc(&dw, (size_t)Q27_HID * sl) != hipSuccess) return fail();
        held.push_back(dw);
        if (hipMalloc(&ds, (size_t)Q27_HID * ss) != hipSuccess) return fail();
        held.push_back(ds);
        if (hipMemcpy2D(dw, sl, (const char*)hw[2] + (size_t)card * sl, rb, sl, (size_t)Q27_HID,
                        hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy2D(ds, ss, (const char*)hs[2] + (size_t)card * ss, sb, ss, (size_t)Q27_HID,
                        hipMemcpyHostToDevice) != hipSuccess) return fail();
        tp_d.w = (const unsigned char*)dw; tp_d.gs = (const unsigned char*)ds;
        tp_d.rows = Q27_HID; tp_d.K = Ksh;
    }
    // MID-POINT SAMPLE. Without this the Q27_EVICT_DBG line below samples free memory exactly TWICE
    // -- once before the authority build and once after the release -- so its "authority build spent
    // X" and "release returned -X" are the SAME difference printed with opposite signs, and the line
    // cannot say whether the mirror gave anything back. Measured 2026-09-17 11:45:48: four cards each
    // reported "spent 164 MB / returned -164 MB" with rc=0 on every hipFree, which is unreadable --
    // a successful free that returns nothing and a build that costs everything look identical.
    // Sampling here separates the two terms, which is what a predict-before-evict guard has to price.
    size_t ev_frM = 0; hipMemGetInfo(&ev_frM, &ev_frT);
    // Commit: release the int8 mirror (w is shared with the per-row form, which requantised it in
    // place; the per-64 scale plane was already handed back at load), null every pointer that keys a
    // consumer to int8, then publish the NVFP4 handles.
    size_t freed = 0;
    void* dead[12] = { (void*)Ls->gate8.w, (void*)Ls->gate8.s, (void*)Ls->up8.w, (void*)Ls->up8.s,
                       (void*)Ls->down8.w, (void*)Ls->down8.s, (void*)Ls->gate8r.s, (void*)Ls->up8r.s,
                       (void*)Ls->down8r.s, nullptr, nullptr, nullptr };
    // Q27_EVICT_DBG: this whole repair exists because this function LOST memory on every card
    // (0.56 -> 0.48 GiB, 2026-09-16) while the latch in q27_evict_one_mlp_own_card reported it. The
    // release below frees handles that the per-row form shares ("w is shared with the per-row form"),
    // so they may be VIEWS rather than arena bases -- hipFree returns an error for those and the
    // return code was never read (`freed += 0`). Print what each free actually did.
    const void* dbg_g8w = (const void*)Ls->gate8.w, * dbg_g8rw = (const void*)Ls->gate8r.w;
    const int dbg_rows = Ls->gate8.rows, dbg_K = Ls->gate8.K;
    Ls->gate8 = q27_i8g_t(); Ls->up8 = q27_i8g_t(); Ls->down8 = q27_i8g_t();
    Ls->gate8r = q27_i8r_t(); Ls->up8r = q27_i8r_t(); Ls->down8r = q27_i8r_t();
    Tp->dc_gate8r = q27_i8r_t(); Tp->dc_up8r = q27_i8r_t(); Tp->dc_down8r = q27_i8r_t();
    int rc[9];
    for (int i = 0; i < 9; ++i) rc[i] = dead[i] ? (int)hipFree(dead[i]) : -1;
    {
        size_t ev_frB = 0; hipMemGetInfo(&ev_frB, &ev_frT);
        std::fprintf(stderr, "Q27_EVICT_DBG card %d L%d: free %.0f -> %.0f MB (authority build spent "
                             "%.0f MB) -> %.0f MB (release returned %.0f MB); net %+.0f MB; gate rows=%d K=%d "
                             "gate8.w=%p gate8r.w=%p shared=%d; rc w=%d/%d/%d s=%d/%d/%d\n",
                     card, layer, (double)ev_frA / 1048576.0, (double)ev_frM / 1048576.0,
                     (double)((long long)ev_frA - (long long)ev_frM) / 1048576.0,
                     (double)ev_frB / 1048576.0,
                     (double)((long long)ev_frB - (long long)ev_frM) / 1048576.0,
                     (double)((long long)ev_frB - (long long)ev_frA) / 1048576.0,
                     dbg_rows, dbg_K, dbg_g8w, dbg_g8rw, (int)(dbg_g8w == dbg_g8rw),
                     rc[0], rc[2], rc[4], rc[6], rc[7], rc[8]);
        std::fflush(stderr);
    }
    held.clear();     // the NVFP4 handles are the layer's now; nothing here is a leak
    *lsh[0] = nv[0]; *lsh[1] = nv[1]; *lsh[2] = nv[2];
    Tp->gate = tp_g; Tp->up = tp_u; Tp->down = tp_d;
    (void)freed;
    return 1;
}

int q27_resolve_tp_ls_mlp(q27_model_t* m, int ndev) {
    q27_model* M = m;
    if (!M || !M->tp_ndev || M->tp_ndev != ndev || !M->ls_ndev) return 0;
    if (M->ls_sr) return 0;   // Q27_LS_SR: no TP handles to alias (tpl is empty)
    const bool want_alias = q27_env_flag("Q27_ALIAS_MLP", false);
    // DEC_MLP_I8 populates the decode int8 views whenever the row mirrors exist. DEC_MLP_NV=1
    // keeps the NVFP4 shards resident (storage recipe) while the trunk still takes Tensile when
    // dc_* is set. Previously want_dec required !DEC_MLP_NV, which dropped shards and broke upload.
    const bool want_dec   = q27_env_flag("Q27_DEC_MLP_I8", false);
    if (!want_alias && !want_dec) return 0;
    int aliased = 0, refused = 0, dec_mlp = 0;
    for (int g = 0; g < ndev; ++g) {
        for (int L = 0; L < Q27_LAYERS; ++L) {
            if (!q27_ls_owned(L, g, ndev)) continue;
            q27_layer_t* T = &M->tpl[(size_t)L * ndev + g];
            const q27_layer_t* S = q27_layer_ls(m, L, M->tp_dev[g]);
            // A layer whose MLP is the int8 mirror has no full-width NVFP4 MLP to alias -- but it is
            // exactly the layer whose DECODE MLP can read that mirror, so build those views first and
            // only then skip the NVFP4 alias.
            const bool layer_i8 = q27_ls_mlp_i8(L, g, ndev, ls_mlp_i8k(g));
            if (!S) { ++refused; continue; }
            if (want_dec && layer_i8) {
                if (S->gate8r.w && S->up8r.w && S->down8r.w) {
                    int moff[8], mlen[8];
                    q27_mlp_wshare(moff, mlen, ndev);
                    const int nr = mlen[g];   // ragged TP3: 5888/5760/5760, not Q27_INTER/ndev
                    const int Ksh = mlen[g];  // down K-slice is the same share
                    q27_i8r_t* dg = &T->dc_gate8r; q27_i8r_t* du = &T->dc_up8r; q27_i8r_t* dd = &T->dc_down8r;
                    *dg = S->gate8r; dg->w = S->gate8r.w + (size_t)moff[g] * (size_t)S->gate8r.K;
                                     dg->s = S->gate8r.s + (size_t)moff[g];
                    *du = S->up8r;   du->w = S->up8r.w   + (size_t)moff[g] * (size_t)S->up8r.K;
                                     du->s = S->up8r.s   + (size_t)moff[g];
                    *dd = S->down8r; dd->w = S->down8r.w + (size_t)moff[g]; dd->K = Ksh;
                    dg->rows = nr; du->rows = nr;      // the VIEW holds this card's shard rows
                    dg->ng = du->ng = dd->ng = 1;
                    dg->gs = dg->K; du->gs = du->K; dd->gs = Ksh;
                    ++dec_mlp;
                } else { ++refused; }
            }
            if (layer_i8) { ++refused; continue; }
            if (!want_alias) continue;
            const q27_nvfp4_t* src[2] = { &S->gate, &S->up };
            q27_nvfp4_t* dst[2] = { &T->gate, &T->up };
            for (int i = 0; i < 2; ++i) {
                if (!src[i]->w || !src[i]->gs) { ++refused; continue; }
                if (src[i]->K != dst[i]->K) { ++refused; continue; }
                // dst[i]->rows is ALREADY the shard's row count (col() stores len/ndev), and the
                // shard is rows [g*nr, (g+1)*nr) of the full-width row -- so the full tensor must
                // hold exactly ndev shards and the row offset is g*nr, not g*nr/ndev.
                const int nr = dst[i]->rows;
                if (nr <= 0 || src[i]->rows != nr * ndev) { ++refused; continue; }
                if (src[i]->ws2 != dst[i]->ws2 || src[i]->in_scale != dst[i]->in_scale) { ++refused; continue; }
                // the shard is rows [g*nr, (g+1)*nr) of the full-width row: same layout, later start
                const size_t rowp = (size_t)src[i]->K / 2;           // packed nibble bytes per row
                const size_t rows = (size_t)src[i]->K / 16;          // per-16 scale bytes per row
                const size_t r0 = (size_t)g * (size_t)nr;
                q27_nvfp4_t v = *src[i];
                v.w  = src[i]->w  + r0 * rowp;
                v.gs = src[i]->gs + r0 * rows;
                v.rows = nr;
                *dst[i] = v;
                ++aliased;
            }
        }
    }
    std::fprintf(stderr, "Q27_ALIAS_MLP: %d TP MLP handles (gate/up) now address the layer-split copy "
                         "(%d refused)\n", aliased, refused);
    std::fprintf(stderr, "Q27_DEC_MLP_I8: %d TP decode MLPs now read the int8 mirror in place\n", dec_mlp);
    std::fflush(stderr);
    return aliased;
}

// ============================================================================================
// residency
// ============================================================================================
const q27_layer_t* q27_layer(const q27_model_t* m, int layer) {
    if (!m || layer < 0 || layer >= Q27_LAYERS || !m->layer_res[layer]) return 0;
    return &m->layer[layer];
}

const q27_globals_t* q27_globals(const q27_model_t* m, int device) {
    if (!m || device < 0 || device >= Q27_MAX_DEVICES || !m->dev_used[device]) return 0;
    return &m->glob[device];
}

const q27_globals_t* q27_globals_ls(const q27_model_t* m, int device) {
    if (!m || device < 0 || device >= Q27_MAX_DEVICES || !m->dev_used[device]) return 0;
    return &m->glob_ls[device];
}

int q27_layer_device(const q27_model_t* m, int layer) {
    if (!m || layer < 0 || layer >= Q27_LAYERS || !m->layer_res[layer]) return -1;
    return m->layer[layer].device;
}

long long q27_resident_bytes(const q27_model_t* m, int device) {
    if (!m || device < 0 || device >= Q27_MAX_DEVICES || !m->dev_used[device]) return -1;
    long long n = 0;
    for (size_t i = 0; i < m->arenas.size(); ++i)
        if (m->arenas[i].device == device) n += (long long)m->arenas[i].size;
    return n;
}

long long q27_resident_bytes_total(const q27_model_t* m) {
    if (!m) return 0;
    long long n = 0;
    for (size_t i = 0; i < m->arenas.size(); ++i) n += (long long)m->arenas[i].size;
    return n;
}

void q27_free_device(q27_model_t* m, int device) {
    if (!m || device < 0 || device >= Q27_MAX_DEVICES) return;
    q27_model* M = m;
    std::vector<Arena> keep;
    for (size_t i = 0; i < M->arenas.size(); ++i) {
        if (M->arenas[i].device == device) {
            hipSetDevice(device);
            hipFree(M->arenas[i].base);
        } else {
            keep.push_back(M->arenas[i]);
        }
    }
    M->arenas.swap(keep);
    for (int L = 0; L < Q27_LAYERS; ++L) {
        if (M->layer_res[L] && M->layer[L].device == device) {
            memset(&M->layer[L], 0, sizeof(q27_layer_t));
            M->layer[L].layer   = L;
            M->layer[L].is_full = Q27_IS_FULL(L) ? 1 : 0;
            M->layer[L].device  = -1;
            M->layer_res[L]     = 0;
        }
    }
    memset(&M->glob[device], 0, sizeof(q27_globals_t));
    if (M->embed_dev == device) M->embed_dev = -1;
    if (M->norm_dev  == device) M->norm_dev  = -1;
    if (M->head_dev  == device) M->head_dev  = -1;
    M->dev_used[device] = 0;
}

// ============================================================================================
// report
// ============================================================================================
namespace {
struct Buf { char* p; size_t cap; size_t n; };
void bap(Buf& b, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const size_t room = (b.n < b.cap) ? (b.cap - b.n) : 0;
    const int w = vsnprintf(room ? b.p + b.n : 0, room, fmt, ap);
    va_end(ap);
    if (w > 0) b.n += (size_t)w;
}
}  // namespace

int q27_report(const q27_model_t* m, char* buf, size_t cap) {
    Buf b; b.p = buf; b.cap = cap; b.n = 0;
    if (buf && cap) buf[0] = 0;
    if (!m) { bap(b, "q27: no model\n"); return (int)b.n; }

    bap(b, "q27 checkpoint %s\n", m->dir.c_str());
    bap(b, "  catalog: %d tensors, %lld payload bytes (%.4f GiB)\n",
        (int)m->t.size(), m->payload_total, (double)m->payload_total / 1073741824.0);
    bap(b, "  skipped: model.visual.* (vision tower) and mtp.* (draft head) - not used by decode\n");

    if (m->ls_sr) {                         // Q27_LS_SR: every layer resident ONCE, at full width, on its owner card
        long long t = 0; size_t pend = 0;
        for (size_t i = 0; i < m->sr_fp8_temps.size(); ++i) pend += m->sr_fp8_temps[i].bytes;
        bap(b, "  SINGLE RESIDENCY (Q27_LS_SR) over %d cards: no TP shards; each layer once, full width, "
               "on its owner card; MLP = block-scaled FP8 (gate/up full, down as 4 K-slices)\n", m->ls_ndev);
        for (int i = 0; i < m->ls_ndev; ++i) {
            const int d = m->tp_dev[i];
            const long long rb = q27_resident_bytes(m, d);
            int owned = 0; for (int L = 0; L < Q27_LAYERS; ++L) if (q27_ls_owned(L, i, m->ls_ndev)) ++owned;
            t += rb;
            bap(b, "  card %d (device %d): %lld B (%.4f GiB) resident  %d owned layers  FP8 MLP %d layers %.3f GiB"
                   "%s%s\n", i, d, rb, (double)rb / 1073741824.0, owned, m->sr_mlp_layers[i],
                (double)m->sr_mlp_bytes[i] / 1073741824.0,
                m->glob_ls[d].final_norm ? " +final_norm" : "", m->glob_ls[d].lm_head.w ? " +lm_head[248320 rows]" : "");
        }
        bap(b, "  total resident: %lld B (%.4f GiB)", t, (double)t / 1073741824.0);
        if (pend) bap(b, "  (+ %.1f MiB of fp8 projection temporaries awaiting q27_ls_sr_release_fp8)", (double)pend / 1048576.0);
        bap(b, "\n");
        return (int)b.n;
    }
    if (m->tp_ndev) {                       // tensor-parallel: every layer on every card
        long long t = 0;
        bap(b, "  TENSOR PARALLEL over %d cards: all %d layers resident on every card, each holding "
               "its shard\n", m->tp_ndev, Q27_LAYERS);
        for (int i = 0; i < m->tp_ndev; ++i) {
            const int d = m->tp_dev[i];
            const long long rb = q27_resident_bytes(m, d);
            t += rb;
            bap(b, "  card %d (device %d): %lld B (%.4f GiB) resident  layers [0,%d) sharded 1/%d"
                   " +final_norm +lm_head[%d rows]\n",
                i, d, rb, (double)rb / 1073741824.0, Q27_LAYERS, m->tp_ndev,
                m->glob[d].lm_head.rows);
        }
        bap(b, "  total resident: %lld B (%.4f GiB)\n", t, (double)t / 1073741824.0);
        return (int)b.n;
    }

    long long tot = 0;
    for (int d = 0; d < Q27_MAX_DEVICES; ++d) {
        if (!m->dev_used[d]) continue;
        const long long rb = q27_resident_bytes(m, d);
        tot += rb;
        bap(b, "  device %d: %lld B (%.4f GiB) resident", d, rb, (double)rb / 1073741824.0);
        // contiguous layer runs
        int printed = 0;
        int L = 0;
        while (L < Q27_LAYERS) {
            if (m->layer_res[L] && m->layer[L].device == d) {
                int e = L;
                while (e < Q27_LAYERS && m->layer_res[e] && m->layer[e].device == d) ++e;
                bap(b, "%s layers [%d,%d)", printed ? "," : "  ", L, e);
                printed = 1;
                L = e;
            } else {
                ++L;
            }
        }
        if (!printed) bap(b, "   no layers");
        if (m->glob[d].embed)      bap(b, " +embed");
        if (m->glob[d].final_norm) bap(b, " +final_norm");
        if (m->glob[d].lm_head.w)  bap(b, " +lm_head");
        bap(b, "\n");
    }
    bap(b, "  total resident: %lld B (%.4f GiB)\n", tot, (double)tot / 1073741824.0);

    int missing = 0;
    for (int L = 0; L < Q27_LAYERS; ++L) if (!m->layer_res[L]) ++missing;
    if (missing) bap(b, "  WARNING: %d of %d layers are not resident\n", missing, Q27_LAYERS);
    if (m->head_dev < 0) bap(b, "  WARNING: lm_head is not resident\n");
    if (m->norm_dev < 0) bap(b, "  WARNING: final norm is not resident\n");
    return (int)b.n;
}

// ---- Q27_DEC_I8: release the fp8 projections of the TP shards once int8 execution mirrors carry every consumer.
// Frees the tagged arenas and replaces the seven fp8 weight pointers of every (layer, card) handle by unique unmapped
// sentinels (a stray fp8 launch faults loudly instead of reading recycled VRAM); `rekey` moves each int8 mirror's
// side-table key to the sentinel so the routers still find it. Returns the bytes freed over all cards (0 if none). ----
extern "C" size_t q27_tp_free_fp8_proj(q27_model_t* m, int (*rekey)(const void* oldw, const void* neww)) {
    if (!m) return 0;
    q27_model* M = m;
    size_t freed = 0;
    for (size_t i = 0; i < M->arenas.size(); ++i) {
        if (M->arenas[i].fp8proj && M->arenas[i].base) {
            hipSetDevice(M->arenas[i].device);
            hipFree(M->arenas[i].base);
            freed += M->arenas[i].size;
            M->arenas[i].base = 0; M->arenas[i].size = 0;
        }
    }
    if (!freed) return 0;
    uint64_t n = 0;
    for (size_t i = 0; i < M->tpl.size(); ++i) {
        q27_layer_t& H = M->tpl[i];
        const unsigned char** f[7] = { &H.q_proj.w, &H.k_proj.w, &H.v_proj.w, &H.o_proj.w, &H.in_qkv.w, &H.in_z.w, &H.out_proj.w };
        for (int k = 0; k < 7; ++k) {
            if (!*f[k]) continue;
            // non-canonical (x86-64) and unmapped (GPU VA) => any dereference faults loudly; unique per handle
            const unsigned char* sent = (const unsigned char*)(uintptr_t)(0xDEAD00000000ULL + (++n) * 4096ULL);
            if (rekey) rekey((const void*)*f[k], (const void*)sent);
            *f[k] = sent;
        }
    }
    return freed;
}
