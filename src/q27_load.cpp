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

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

    q27_model() : nshard(0), payload_total(0), embed_dev(-1), norm_dev(-1), head_dev(-1),
                  tp_ndev(0), ls_ndev(0) {
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
    if (!p || !n) return;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    const uintptr_t a   = (uintptr_t)p & ~(uintptr_t)(ps - 1);
    const size_t    len = n + (size_t)((uintptr_t)p - a);
    madvise((void*)a, len, MADV_WILLNEED);
}

int copy_in(void* dst, const unsigned char* src, size_t n, char* err, size_t cap) {
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
    static void row(TpCtx& c, const char* name, int dt, long long rows, long long cols, void* slot) {
        if (*c.rc != Q27_OK) return;
        const int ti = w2(c, name, dt, rows, cols);
        if (ti < 0) return;
        const long long rb = cols * (long long)tp_esize(dt);
        if (rb % c.ndev) {
            *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                         "'%s': row-parallel row of %lld B does not divide by %d cards", name, rb, c.ndev);
            return;
        }
        const long long sl = rb / c.ndev;
        TpBlk B; B.slot = slot; B.bytes = rows * sl; B.nseg = 1;
        B.seg[0].ti = ti; B.seg[0].src_off = (long long)c.g * sl; B.seg[0].rows = rows;
        B.seg[0].row_bytes = sl; B.seg[0].src_stride = rb; B.seg[0].dst_off = 0;
        c.blk->push_back(B);
    }

    // ---- the two storage formats, both directions ----
    static void nvfp4_col(TpCtx& c, const char* base, q27_nvfp4_t* h, int rows, int K,
                          const long long* rbase, const long long* rlen, int nrange) {
        char n[256];
        long long got = 0;
        snprintf(n, sizeof n, "%s.weight", base);
        col(c, n, Q27_DT_U8, rows, K / 2, rbase, rlen, nrange, (void*)&h->w, &got);
        snprintf(n, sizeof n, "%s.weight_scale", base);
        col(c, n, Q27_DT_F8_E4M3, rows, K / 16, rbase, rlen, nrange, (void*)&h->gs, 0);
        snprintf(n, sizeof n, "%s.weight_scale_2", base);  h->ws2      = sc(c, n);
        snprintf(n, sizeof n, "%s.input_scale", base);     h->in_scale = sc(c, n);
        h->rows = (int)got;                 // THE SHARD's rows
        h->K    = K;
    }
    static void nvfp4_row(TpCtx& c, const char* base, q27_nvfp4_t* h, int rows, int K) {
        if (*c.rc != Q27_OK) return;
        if (K % c.ndev || (K / c.ndev) % 16) {
            *c.rc = fail(c.err, c.cap, Q27_E_SHAPE,
                         "'%s': row-parallel K=%d over %d cards gives %d, which is not an exact "
                         "multiple of the 16-wide scale group", base, K, c.ndev, K / c.ndev);
            return;
        }
        char n[256];
        snprintf(n, sizeof n, "%s.weight", base);
        row(c, n, Q27_DT_U8, rows, K / 2, (void*)&h->w);        // KP/G packed bytes per row
        snprintf(n, sizeof n, "%s.weight_scale", base);
        row(c, n, Q27_DT_F8_E4M3, rows, K / 16, (void*)&h->gs);  // NG/G scales per row
        snprintf(n, sizeof n, "%s.weight_scale_2", base);  h->ws2      = sc(c, n);
        snprintf(n, sizeof n, "%s.input_scale", base);     h->in_scale = sc(c, n);
        h->rows = rows;
        h->K    = K / c.ndev;               // THE SHARD's K
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
    static void fp8_row(TpCtx& c, const char* base, q27_fp8_t* h, int rows, int K) {
        if (*c.rc != Q27_OK) return;
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

    for (int L = 0; L < Q27_LAYERS && rc == Q27_OK; ++L) {
        q27_layer_t& H = col[L];
        memset(&H, 0, sizeof(H));
        H.layer = L; H.is_full = Q27_IS_FULL(L) ? 1 : 0; H.device = M->tp_dev[g];

        snprintf(nb, sizeof nb, "%s.%d.input_layernorm.weight", LP, L);
        Tp::rep(c, nb, Q27_DT_BF16, Q27_HID, (void*)&H.input_norm);
        snprintf(nb, sizeof nb, "%s.%d.post_attention_layernorm.weight", LP, L);
        Tp::rep(c, nb, Q27_DT_BF16, Q27_HID, (void*)&H.post_norm);

        {   // MLP: gate/up column-parallel, down row-parallel + ALL-REDUCE
            const long long len_i[1] = {Q27_INTER};
            snprintf(nb, sizeof nb, "%s.%d.mlp.gate_proj", LP, L);
            Tp::nvfp4_col(c, nb, &H.gate, Q27_INTER, Q27_HID, one_base, len_i, 1);
            snprintf(nb, sizeof nb, "%s.%d.mlp.up_proj", LP, L);
            Tp::nvfp4_col(c, nb, &H.up, Q27_INTER, Q27_HID, one_base, len_i, 1);
            snprintf(nb, sizeof nb, "%s.%d.mlp.down_proj", LP, L);
            Tp::nvfp4_row(c, nb, &H.down, Q27_HID, Q27_INTER);
        }
        if (rc == Q27_OK && H.gate.in_scale != H.up.in_scale)
            rc = fail(err, cap, Q27_E_SHAPE, "layer %d: mlp gate/up input_scale differ", L);

        if (H.is_full) {
            const long long len_q[1]  = {Q27_QROWS};
            const long long len_kv[1] = {Q27_KVROWS};
            snprintf(nb, sizeof nb, "%s.%d.self_attn.q_proj", LP, L);
            Tp::fp8_col(c, nb, &H.q_proj, Q27_QROWS, Q27_HID, one_base, len_q, 1);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.k_proj", LP, L);
            Tp::fp8_col(c, nb, &H.k_proj, Q27_KVROWS, Q27_HID, one_base, len_kv, 1);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.v_proj", LP, L);
            Tp::fp8_col(c, nb, &H.v_proj, Q27_KVROWS, Q27_HID, one_base, len_kv, 1);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.o_proj", LP, L);
            Tp::fp8_row(c, nb, &H.o_proj, Q27_HID, Q27_OROWS);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.q_norm.weight", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, Q27_HDIM, (void*)&H.q_norm);
            snprintf(nb, sizeof nb, "%s.%d.self_attn.k_norm.weight", LP, L);
            Tp::rep(c, nb, Q27_DT_BF16, Q27_HDIM, (void*)&H.k_norm);
            if (rc == Q27_OK && !(H.q_proj.in_scale == H.k_proj.in_scale &&
                                  H.k_proj.in_scale == H.v_proj.in_scale))
                rc = fail(err, cap, Q27_E_SHAPE, "layer %d: q/k/v input_scale differ", L);
        } else {
            // in_proj_qkv is q 2048 | k 2048 | v 6144: THREE ranges, so the card gets whole heads.
            const long long qkv_base[3] = {0, 2048, 4096};
            const long long qkv_len [3] = {2048, 2048, 6144};
            const long long len_z  [1]  = {Q27_GDN_Z};
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_qkv", LP, L);
            Tp::fp8_col(c, nb, &H.in_qkv, Q27_GDN_QKV, Q27_HID, qkv_base, qkv_len, 3);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_z", LP, L);
            Tp::fp8_col(c, nb, &H.in_z, Q27_GDN_Z, Q27_HID, one_base, len_z, 1);
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.out_proj", LP, L);
            Tp::fp8_row(c, nb, &H.out_proj, Q27_HID, Q27_GDN_Z);
            // replicated: the two BF16 GEMVs are 0.5 MB and their outputs are indexed by GLOBAL
            // value head, which the TP kernels reach with a head base rather than a sliced tensor.
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_a.weight", LP, L);
            // in_proj_a / in_proj_b were REPLICATED at full 48-head width on every card while
            // the step kernel reads only its own 12 heads (abuf[hbase+h]) -- three quarters of the
            // GEMV was computed and discarded, on 96 calls/token. TP had changed OWNERSHIP and the
            // execution never followed. Column-parallel by value-head range needs NO collective:
            // the input activation is already replicated and each card's output feeds only its own
            // step kernel. 48 % 4 == 0, and the shard base 12g is exactly the kernel's hbase.
            { static const long long vb[1] = {0}, vl[1] = {Q27_GDN_VH};
              Tp::col(c, nb, Q27_DT_BF16, Q27_GDN_VH, Q27_HID, vb, vl, 1, (void*)&H.in_a, nullptr); }
            snprintf(nb, sizeof nb, "%s.%d.linear_attn.in_proj_b.weight", LP, L);
            { static const long long vb[1] = {0}, vl[1] = {Q27_GDN_VH};
              Tp::col(c, nb, Q27_DT_BF16, Q27_GDN_VH, Q27_HID, vb, vl, 1, (void*)&H.in_b, nullptr); }
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
    if (rc == Q27_OK) {
        const long long len_v[1] = {Q27_VOCAB};
        Tp::nvfp4_col(c, "lm_head", &G->lm_head, Q27_VOCAB, Q27_HID, one_base, len_v, 1);
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
    // Geometry that must divide exactly, checked before a single byte moves.
    {
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
                const unsigned char* src = host_of(M, M->t[S.ti]) + S.src_off;
                if (S.row_bytes == S.src_stride) {          // contiguous: straight out of the mmap
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

        Arena A; A.device = dev; A.base = base; A.size = total_main; A.fp8proj = 0;
        M->arenas.push_back(A);
        if (total_fp8) { Arena F; F.device = dev; F.base = base_fp8; F.size = total_fp8; F.fp8proj = 1; M->arenas.push_back(F); }
        M->dev_used[dev] = 1;
        M->glob[dev] = GB;
        for (int L = 0; L < Q27_LAYERS; ++L) M->tpl[(size_t)L * ndev + g] = colh[L];
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
    if (Q27_LAYERS % ndev)
        return fail(err, errcap, Q27_E_ARG, "%d cards does not divide %d layers", ndev, Q27_LAYERS);
    const int LPP = Q27_LAYERS / ndev;
    // Q27_LS_BLK: layers are assigned to cards in rotating BLOCKS of this many layers (card g owns blocks
    // g, g+ndev, g+2ndev, ...). LPP (default) = contiguous ranges; 4 = 16 pipeline stages of 4 layers.
    int ls_blk = q27_env_int("Q27_LS_BLK", 2); if (ls_blk < 1 || ls_blk > LPP || (LPP % ls_blk)) ls_blk = LPP;   // default 2 (with 256-wide chunks): p1k 612 ms
    const int ls_bal = q27_env_int("Q27_LS_BAL", 0);   // 1 = rotate ownership every other round (balanced work, but adjacent blocks on one card collide in pipeline stages: 638 vs 542 ms) -> off
    auto ls_owned = [&](int L, int g) { const int b = L / ls_blk, r = b / ndev; return ((b + (ls_bal ? (r & 1) : 0)) % ndev) == g; };
    int head_card = ndev - 1; for (int gg = 0; gg < ndev; ++gg) if (ls_owned(Q27_LAYERS - 1, gg)) head_card = gg;   // the final norm + lm_head live with the last layer

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
                fblk.push_back(B);                          // final norm + full lm_head, on the card owning the last layer
            }
        }
        if (g != head_card) { GB.final_norm = nullptr; GB.lm_head = q27_nvfp4_t(); }

        std::sort(fblk.begin(), fblk.end(), [M](const TpBlk& a, const TpBlk& b) {
            const Tensor& x = M->t[a.seg[0].ti];
            const Tensor& y = M->t[b.seg[0].ti];
            if (x.shard != y.shard) return x.shard < y.shard;
            return x.off + a.seg[0].src_off < y.off + b.seg[0].src_off;
        });

        // Q27_LS_Q8: the NVFP4 MLP nibbles/scales and the fp8 projection bytes of this card's layers are
        // NOT kept resident -- they are uploaded to temporaries, converted into the int8 execution mirrors
        // (q27_fp8_to_i8g_conv / q27_nvfp4_to_i8g64_conv) and freed, which keeps ~4 GB per card for the
        // decode residency and the context. Identified by the pointer slot's offset inside q27_layer_t.
        const bool q8 = q27_env_flag("Q27_LS_Q8", true);
        std::vector<char> is_conv(fblk.size(), 0);
        if (q8) {
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
                const size_t inl = (size_t)off % sizeof(q27_layer_t);
                for (int k = 0; k < 13; ++k) if (inl == convoff[k]) { is_conv[i] = 1; break; }
            }
        }
        std::vector<void*> temps;
        size_t total = 0, maxstage = 0;
        for (size_t i = 0; i < fblk.size(); ++i) {
            if (!is_conv[i]) total += alignup((size_t)fblk[i].bytes);
            bool strided = false;
            for (int s = 0; s < fblk[i].nseg; ++s)
                if (fblk[i].seg[s].row_bytes != fblk[i].seg[s].src_stride) strided = true;
            if (strided && (size_t)fblk[i].bytes > maxstage) maxstage = (size_t)fblk[i].bytes;
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
                const unsigned char* src = host_of(M, M->t[S.ti]) + S.src_off;
                if (S.row_bytes == S.src_stride) {
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

        Arena A; A.device = dev; A.base = base; A.size = total;
        M->arenas.push_back(A);
        M->glob_ls[dev] = GB;
        if (q8) {   // build the int8 execution mirrors for this card's layers, then drop the originals
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
                if (!q27_nvfp4_to_i8g64_conv(&H.gate, &H.gate8, 0) || !q27_nvfp4_to_i8g64_conv(&H.up, &H.up8, 0) ||
                    !q27_nvfp4_to_i8g64_conv(&H.down, &H.down8, 0)) {
                    if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
                    return fail(err, errcap, Q27_E_HIP, "Q8: int8/64 MLP mirror failed (layer %d, device %d)", L, dev);
                }
                if (q27_env_flag("Q27_Q8_RB", true)) {   // stage 3: per-(row,1024-group) int8 views for the rocBLAS GEMMs (in place, by family mask)
                    const int rbm = q27_env_int("Q27_Q8_RBM", 7), rbgs = q27_env_int("Q27_Q8_GS", 0);   // group size: 0 = full K (default), or 1024
                    int ok = 1;
                    if (rbm & 4) ok &= q27_i8g64_to_row_conv(&H.gate8, &H.gate8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.up8, &H.up8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.down8, &H.down8r, rbgs, 0);
                    if (H.is_full && (rbm & 1)) ok &= q27_i8g64_to_row_conv(&H.q8g, &H.q8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.k8g, &H.k8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.v8g, &H.v8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.o8g, &H.o8r, rbgs, 0);
                    else if (!H.is_full && (rbm & 2)) ok &= q27_i8g64_to_row_conv(&H.iqkv8g, &H.iqkv8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.iz8g, &H.iz8r, rbgs, 0) & q27_i8g64_to_row_conv(&H.op8g, &H.op8r, rbgs, 0);
                    if (!H.is_full && (rbm & 2)) ok &= q27_ab_to_i8r_conv(H.in_a, H.in_b, Q27_GDN_VH, Q27_HID, &H.ab8r, 0);
                    if (!ok) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear(); return fail(err, errcap, Q27_E_HIP, "Q8: per-row int8 view failed (layer %d, device %d)", L, dev); }
                }
            }
            he = hipDeviceSynchronize();
            if (he != hipSuccess) { if (stage) hipHostFree(stage); M->ls_ndev = 0; M->tpl_ls.clear();
                return fail(err, errcap, Q27_E_HIP, "Q8 mirror sync device %d: %s", dev, hipGetErrorString(he)); }
            for (size_t i = 0; i < temps.size(); ++i) hipFree(temps[i]);
            temps.clear();
            for (int L = 0; L < Q27_LAYERS; ++L) {   if (!ls_owned(L, g)) continue;   // the originals are gone: nothing on the Q8 path reads them
                q27_layer_t& H = colh[L];
                H.gate.w = nullptr; H.gate.gs = nullptr; H.up.w = nullptr; H.up.gs = nullptr; H.down.w = nullptr; H.down.gs = nullptr;
                H.q_proj.w = nullptr; H.k_proj.w = nullptr; H.v_proj.w = nullptr; H.o_proj.w = nullptr;
                H.in_qkv.w = nullptr; H.in_z.w = nullptr; H.out_proj.w = nullptr;
            }
        }
        for (int L = 0; L < Q27_LAYERS; ++L)
            M->tpl_ls[(size_t)L * ndev + g] = ls_owned(L, g) ? colh[L] : q27_layer_t();
    }
    if (stage) hipHostFree(stage);

    // NVFP4 twins for the wide fp8 projections: only the card's OWN layers.
    if (q27_env_flag("Q27_FP8X4", false) || q27_env_int("Q27_PF_X4W", 0) > 0) {
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

    return Q27_OK;
}

const q27_layer_t* q27_layer_ls(const q27_model_t* m, int layer, int device) {
    if (!m || !m->ls_ndev || layer < 0 || layer >= Q27_LAYERS) return 0;
    for (int i = 0; i < m->ls_ndev; ++i)
        if (m->tp_dev[i] == device) return &m->tpl_ls[(size_t)layer * m->ls_ndev + i];
    return 0;
}

const q27_layer_t* q27_layer_tp(const q27_model_t* m, int layer, int device) {
    if (!m || !m->tp_ndev || layer < 0 || layer >= Q27_LAYERS) return 0;
    for (int i = 0; i < m->tp_ndev; ++i)
        if (m->tp_dev[i] == device) return &m->tpl[(size_t)layer * m->tp_ndev + i];
    return 0;
}

int q27_tp_ndev(const q27_model_t* m) { return m ? m->tp_ndev : 0; }

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
