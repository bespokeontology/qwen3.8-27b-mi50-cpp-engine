// q27_exl3.cpp — EXL3 store open + per-projection device upload.
// Hand-written safetensors scanner (same format contract as q27_load.cpp and
// q27_dflash2.cpp), no JSON library, no Python. Unlike q27_open this takes an
// arbitrary shard count: the EXL3 stores ship two shards, the NVFP4 checkpoint
// three, and exllamav3 repacks are free to ship any number.
#include "q27_exl3.h"

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// --- Q27_EXL3_SKIP: per-consumer bisect switch -----------------------------
// Every EXL3 call site is written as "try EXL3, else the incumbent", gated on
// the handle being live. So the cheapest way to ask WHICH consumer is wired
// wrong is to never upload that class: the handle stays dead and the site falls
// back on its own. Comma list of: mlp, qkv, z, o, head. Pair it with
// Q27_EXL3_NODROP=1 so the NVFP4 the fallback needs is still resident.
//
// Built 2026-09-19 after the 1-card/4-card bisect showed BOTH broken while
// every tensor passed the seam test vs BF16 -- i.e. the defect is dispatch,
// not format, and only a per-consumer discriminator can localize it.
#define Q27_EXL3_SKIP_MLP  1u
#define Q27_EXL3_SKIP_QKV  2u
#define Q27_EXL3_SKIP_Z    4u
#define Q27_EXL3_SKIP_O    8u
#define Q27_EXL3_SKIP_HEAD 16u

static unsigned q27_exl3_skip_mask() {
    static const unsigned m = [] {
        const char* e = getenv("Q27_EXL3_SKIP");
        if (!e || !*e) return 0u;
        auto tok = [e](const char* t) {
            const size_t n = strlen(t);
            for (const char* p = e; (p = strstr(p, t)) != nullptr; p += n) {
                const char before = (p == e) ? ',' : p[-1];
                const char after  = p[n];
                if ((before == ',' || before == ' ') &&
                    (after == ',' || after == ' ' || after == '\0')) return true;
            }
            return false;
        };
        unsigned v = 0;
        if (tok("mlp"))  v |= Q27_EXL3_SKIP_MLP;
        if (tok("qkv"))  v |= Q27_EXL3_SKIP_QKV;
        if (tok("z"))    v |= Q27_EXL3_SKIP_Z;
        if (tok("o"))    v |= Q27_EXL3_SKIP_O;
        if (tok("head")) v |= Q27_EXL3_SKIP_HEAD;
        std::fprintf(stderr, "Q27_EXL3_SKIP='%s' -> mask 0x%x (mlp=%d qkv=%d z=%d o=%d head=%d)\n",
                     e, v, !!(v & Q27_EXL3_SKIP_MLP), !!(v & Q27_EXL3_SKIP_QKV),
                     !!(v & Q27_EXL3_SKIP_Z), !!(v & Q27_EXL3_SKIP_O), !!(v & Q27_EXL3_SKIP_HEAD));
        return v;
    }();
    return m;
}

namespace {

int fail(char* err, size_t cap, int code, const char* fmt, ...) {
    if (err && cap) { va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap); }
    return code;
}

struct Ent {
    int shard = 0;
    long long off = 0, len = 0;          // offsets are relative to that shard's payload base
    std::vector<long long> shape;
    std::string dtype;
};

// --- minimal JSON scanner over a safetensors header -------------------------
struct Scanner {
    const char* P; const char* E;
    void ws() { while (P < E && (*P==' '||*P=='\t'||*P=='\n'||*P=='\r'||*P==','||*P==':')) ++P; }
    bool jstr(std::string& out) {
        ws(); if (P>=E||*P!='"') return false; ++P; const char* s=P;
        while (P<E && *P!='"') { if (*P=='\\') P+=2; else ++P; }
        if (P>=E) return false; out.assign(s,P-s); ++P; return true;
    }
    bool jnum(long long& out) {
        ws(); if (P>=E) return false; char* en=nullptr;
        long long v=strtoll(P,&en,10); if (en==P) return false; out=v; P=en; return true;
    }
};

struct Shard { int fd = -1; void* map = nullptr; size_t size = 0; size_t base = 0; };

}  // namespace

struct q27_exl3_store_t {
    std::vector<Shard> sh;
    std::unordered_map<std::string, Ent> t;
    std::string dir;
    size_t mapped = 0;
};

static int scan_shard(q27_exl3_store_t* S, int idx, char* err, size_t errcap) {
    Shard& s = S->sh[idx];
    if (s.size < 8) return fail(err, errcap, 2, "exl3 shard %d too small", idx);
    unsigned long long hlen = 0;
    std::memcpy(&hlen, s.map, 8);                       // 8-byte LE header length
    if (hlen == 0 || hlen + 8 > s.size)
        return fail(err, errcap, 2, "exl3 shard %d bad header length %llu", idx, hlen);
    s.base = 8 + (size_t)hlen;

    Scanner J{ (const char*)s.map + 8, (const char*)s.map + 8 + (size_t)hlen };
    J.ws(); if (J.P >= J.E || *J.P != '{') return fail(err, errcap, 2, "exl3 shard %d header not an object", idx);
    ++J.P;
    while (true) {
        J.ws(); if (J.P >= J.E) break;
        if (*J.P == '}') { ++J.P; break; }
        std::string name; if (!J.jstr(name)) break;
        J.ws(); if (J.P >= J.E || *J.P != '{') break;
        if (name == "__metadata__") {                    // skip the nested object wholesale
            int d = 1; ++J.P;
            while (d && J.P < J.E) { if (*J.P=='{') ++d; else if (*J.P=='}') --d; ++J.P; }
            continue;
        }
        Ent e{}; e.shard = idx;
        int depth = 1; ++J.P;
        while (depth && J.P < J.E) {
            if (*J.P=='{') { ++depth; ++J.P; continue; }
            if (*J.P=='}') { --depth; ++J.P; continue; }
            std::string k; const char* save = J.P;
            if (!J.jstr(k)) { ++J.P; continue; }
            if (k == "shape") {
                J.ws(); if (J.P<J.E && *J.P=='[') { ++J.P; J.ws();
                    while (J.P<J.E && *J.P!=']') { long long v=0; if (J.jnum(v)) e.shape.push_back(v); J.ws(); }
                    if (J.P<J.E) ++J.P; }
            } else if (k == "data_offsets") {
                J.ws(); if (J.P<J.E && *J.P=='[') { ++J.P; J.ws(); long long a=0,b=0;
                    if (J.jnum(a)) { J.ws(); J.jnum(b); } e.off=a; e.len=b-a;
                    while (J.P<J.E && *J.P!=']') ++J.P; if (J.P<J.E) ++J.P; }
            } else if (k == "dtype") {
                std::string dv; if (J.jstr(dv)) e.dtype = dv; else { J.P = save; ++J.P; }
            } else {
                std::string skip; if (!J.jstr(skip)) { J.P = save; ++J.P; }
            }
            J.ws();
        }
        S->t[name] = e;
    }
    return 0;
}

int q27_exl3_open(const char* dir, q27_exl3_store_t** out, char* err, size_t errcap) {
    if (!dir || !out) return fail(err, errcap, 1, "exl3 open: null arg");
    *out = nullptr;
    DIR* d = opendir(dir);
    if (!d) return fail(err, errcap, 1, "exl3 opendir %s: %s", dir, strerror(errno));
    std::vector<std::string> files;
    while (struct dirent* de = readdir(d)) {
        const char* n = de->d_name; size_t ln = std::strlen(n);
        if (ln > 12 && !std::strcmp(n + ln - 12, ".safetensors")) files.push_back(n);
    }
    closedir(d);
    if (files.empty()) return fail(err, errcap, 1, "exl3 %s: no .safetensors", dir);
    std::sort(files.begin(), files.end());               // shard order = lexical, as published

    q27_exl3_store_t* S = new q27_exl3_store_t();
    S->dir = dir;
    S->sh.resize(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        std::string p = std::string(dir) + "/" + files[i];
        struct stat st{};
        if (stat(p.c_str(), &st)) { q27_exl3_close(S); return fail(err, errcap, 1, "exl3 stat %s: %s", p.c_str(), strerror(errno)); }
        int fd = open(p.c_str(), O_RDONLY);
        if (fd < 0) { q27_exl3_close(S); return fail(err, errcap, 1, "exl3 open %s: %s", p.c_str(), strerror(errno)); }
        void* m = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { close(fd); q27_exl3_close(S); return fail(err, errcap, 1, "exl3 mmap %s failed", p.c_str()); }
        S->sh[i].fd = fd; S->sh[i].map = m; S->sh[i].size = (size_t)st.st_size;
        S->mapped += (size_t)st.st_size;
        if (int rc = scan_shard(S, (int)i, err, errcap)) { q27_exl3_close(S); return rc; }
    }
    *out = S;
    return 0;
}

void q27_exl3_close(q27_exl3_store_t* S) {
    if (!S) return;
    for (auto& s : S->sh) { if (s.map && s.map != MAP_FAILED) munmap(s.map, s.size); if (s.fd >= 0) close(s.fd); }
    delete S;
}

size_t q27_exl3_store_bytes(const q27_exl3_store_t* S) { return S ? S->mapped : 0; }

static const unsigned char* host_of(const q27_exl3_store_t* S, const Ent& e) {
    const Shard& s = S->sh[e.shard];
    return (const unsigned char*)s.map + s.base + (size_t)e.off;
}

static const Ent* look(const q27_exl3_store_t* S, const std::string& n) {
    auto it = S->t.find(n);
    return it == S->t.end() ? nullptr : &it->second;
}

// Upload a contiguous host range to the current device.
static int up(const void* host, size_t bytes, void** dev, char* err, size_t errcap, const char* what) {
    if (hipMalloc(dev, bytes) != hipSuccess)
        return fail(err, errcap, 3, "exl3 hipMalloc %zu for %s failed", bytes, what);
    if (hipMemcpy(*dev, host, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(*dev); *dev = nullptr;
        return fail(err, errcap, 3, "exl3 H2D %zu for %s failed", bytes, what);
    }
    return 0;
}

static int upload_impl(q27_exl3_store_t* S, const char* prefix, int n0, int ncols,
                       q27_exl3_t* w, char* err, size_t errcap) {
    if (!S || !prefix || !w) return fail(err, errcap, 1, "exl3 upload: null arg");
    const std::string p = prefix;
    const Ent* T = look(S, p + ".trellis");
    const Ent* U = look(S, p + ".suh");
    const Ent* V = look(S, p + ".svh");
    if (!T) return fail(err, errcap, 4, "exl3: %s.trellis absent", prefix);
    if (!U || !V) return fail(err, errcap, 4, "exl3: %s suh/svh absent", prefix);
    if (T->shape.size() != 3) return fail(err, errcap, 4, "exl3: %s.trellis rank %zu != 3", prefix, T->shape.size());
    if (T->dtype != "I16" && T->dtype != "U16")
        return fail(err, errcap, 4, "exl3: %s.trellis dtype %s (want I16)", prefix, T->dtype.c_str());

    const int rows_k   = (int)T->shape[0];               // in/16
    const int cells_n  = (int)T->shape[1];               // out/16
    const int cellw    = (int)T->shape[2];               // 256*K/16 uint16s
    const int K        = cellw * 16 / 256;
    if (K * 256 / 16 != cellw || K < 1 || K > 8)
        return fail(err, errcap, 4, "exl3: %s trailing dim %d is not 256*K/16", prefix, cellw);

    const int in  = rows_k * 16;
    const int out_full = cells_n * 16;
    if ((int)U->shape[0] != in || (int)V->shape[0] != out_full)
        return fail(err, errcap, 4, "exl3: %s scale lengths %lld/%lld vs in/out %d/%d",
                    prefix, U->shape.empty()?-1:U->shape[0], V->shape.empty()?-1:V->shape[0], in, out_full);

    // Column slice: the trellis's SECOND axis is the output dim in 16-wide
    // cells, and the cells of one k row are contiguous, so a column range is a
    // strided copy -- one contiguous run per k row.
    const int n_beg = n0, n_len = (ncols > 0 ? ncols : out_full);
    if (n_beg % 128 || n_len % 128 || n_beg + n_len > out_full)
        return fail(err, errcap, 4, "exl3: %s bad column slice [%d,%d) of %d (need 128-multiples)",
                    prefix, n_beg, n_beg + n_len, out_full);

    const size_t cell_bytes = (size_t)cellw * 2;
    const size_t row_bytes  = (size_t)(n_len / 16) * cell_bytes;         // per k row, sliced
    const size_t tre_bytes  = row_bytes * (size_t)rows_k;
    const unsigned char* th = host_of(S, *T);
    if ((size_t)T->len != (size_t)rows_k * cells_n * cell_bytes)
        return fail(err, errcap, 4, "exl3: %s.trellis len %lld vs shape %zu", prefix, T->len,
                    (size_t)rows_k * cells_n * cell_bytes);

    void* dtre = nullptr;
    if (hipMalloc(&dtre, tre_bytes) != hipSuccess)
        return fail(err, errcap, 3, "exl3 hipMalloc %zu for %s.trellis failed", tre_bytes, prefix);
    if (n_len == out_full) {
        if (hipMemcpy(dtre, th, tre_bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(dtre); return fail(err, errcap, 3, "exl3 H2D %s.trellis failed", prefix); }
    } else {
        const size_t src_stride = (size_t)cells_n * cell_bytes;
        if (hipMemcpy2D(dtre, row_bytes,
                        th + (size_t)(n_beg / 16) * cell_bytes, src_stride,
                        row_bytes, (size_t)rows_k, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(dtre); return fail(err, errcap, 3, "exl3 H2D2D %s.trellis failed", prefix); }
    }

    void* dsuh = nullptr; void* dsvh = nullptr;
    if (int rc = up(host_of(S, *U), (size_t)in * 2, &dsuh, err, errcap, "suh")) { hipFree(dtre); return rc; }
    if (int rc = up(host_of(S, *V) + (size_t)n_beg * 2, (size_t)n_len * 2, &dsvh, err, errcap, "svh")) {
        hipFree(dtre); hipFree(dsuh); return rc; }

    w->trellis = (const uint16_t*)dtre;
    w->suh = (const __half*)dsuh;
    w->svh = (const __half*)dsvh;
    w->in = in; w->out = n_len; w->K = K;
    // PRESENCE, not value, is the codebook selector -- that is how
    // exl3_reconstruct_gfx906 was validated bit-exact against this store.
    w->mcg  = look(S, p + ".mcg")  ? 1 : 0;
    w->mul1 = look(S, p + ".mul1") ? 1 : 0;
    return 0;
}

// ---------------------------------------------------------------------------
// TENSOR-PARALLEL SHARDS. The engine's TP residency splits every projection one
// of two ways (q27_load.cpp Tp::col / Tp::row), and EXL3 follows it exactly:
//
//   COL (output-sharded: q/k/v, gate/up, in_qkv, in_z, lm_head)
//       card g takes output range [rbase[i] + g*rlen[i]/ndev, +rlen[i]/ndev)
//       for each of nrange ranges -- in_proj_qkv really is three (q 2048 |
//       k 2048 | v 6144), so the card gets WHOLE HEADS. Slice the trellis's
//       second axis and svh; suh is whole.
//
//   ROW (input-sharded: o_proj, out_proj, down)
//       card g takes input range [g*in/ndev, +in/ndev). Slice the trellis's
//       first axis and suh; svh is whole, and each card produces a PARTIAL that
//       the existing fp32 all-reduce sums.
//
// WHY THE ROW CASE NEEDS NO COLLECTIVE SURGERY: the EXL3 epilogue is
// H128(y') (*) svh. H is LINEAR and svh is ELEMENTWISE, so
//     H(sum_g y'_g) (*) svh  ==  sum_g ( H(y'_g) (*) svh )
// -- every card may apply its whole epilogue locally and the incumbent reduce
// still sums to the right answer. Nothing about the collective changes.
//
// AND THE SLICES ARE EXACT: H128 is blockwise over 128 contiguous elements, and
// every Qwen3.8-27B shard boundary at ndev=4 is a multiple of 128
// (17408/4=4352, 12288/4=3072, 1024/4=256, 6144/4=1536, 10240/4=2560,
// 248320/4=62080), so no Hadamard block is ever split across cards.
enum { Q27_EXL3_COL = 0, Q27_EXL3_ROW = 1 };

static int upload_shard(q27_exl3_store_t* S, const char* prefix, int kind,
                        const long long* rbase, const long long* rlen, int nrange,
                        int g, int ndev, q27_exl3_t* w, char* err, size_t errcap)
{
    if (!S || !prefix || !w) return fail(err, errcap, 1, "exl3 shard: null arg");
    const std::string p = prefix;
    const Ent* T = look(S, p + ".trellis");
    const Ent* U = look(S, p + ".suh");
    const Ent* V = look(S, p + ".svh");
    if (!T || !U || !V) return fail(err, errcap, 4, "exl3: %s missing trellis/suh/svh", prefix);
    if (T->shape.size() != 3) return fail(err, errcap, 4, "exl3: %s.trellis rank %zu", prefix, T->shape.size());

    const int rows_k  = (int)T->shape[0];            // in/16
    const int cells_n = (int)T->shape[1];            // out/16
    const int cellw   = (int)T->shape[2];
    const int K       = cellw * 16 / 256;
    const int in_full = rows_k * 16, out_full = cells_n * 16;
    if (K < 1 || K > 8) return fail(err, errcap, 4, "exl3: %s bad K", prefix);

    const size_t cell_b = (size_t)cellw * 2;
    const size_t src_row_b = (size_t)cells_n * cell_b;     // one k row, full width
    const unsigned char* th = host_of(S, *T);
    const unsigned char* uh = host_of(S, *U);
    const unsigned char* vh = host_of(S, *V);

    int in_use = in_full, out_use = 0;
    std::vector<std::pair<int,int>> cols;                  // (begin, len) in OUTPUT features
    int k0 = 0;

    if (kind == Q27_EXL3_COL) {
        for (int i = 0; i < nrange; ++i) {
            const long long len = rlen[i] / ndev;
            const long long beg = rbase[i] + (long long)g * len;
            if (len % 128 || beg % 128 || beg + len > out_full)
                return fail(err, errcap, 4, "exl3: %s col range %d -> [%lld,%lld) not 128-aligned in %d",
                            prefix, i, beg, beg + len, out_full);
            cols.push_back({(int)beg, (int)len});
            out_use += (int)len;
        }
    } else {
        const int len = in_full / ndev;
        if (in_full % ndev || len % 128)
            return fail(err, errcap, 4, "exl3: %s row shard in=%d over %d not 128-aligned", prefix, in_full, ndev);
        k0 = g * len; in_use = len; out_use = out_full;
        cols.push_back({0, out_full});
    }

    const size_t dst_row_b = (size_t)(out_use / 16) * cell_b;
    const size_t tre_b     = dst_row_b * (size_t)(in_use / 16);
    void* dtre = nullptr;
    if (hipMalloc(&dtre, tre_b) != hipSuccess)
        return fail(err, errcap, 3, "exl3 hipMalloc %zu for %s shard", tre_b, prefix);

    // One strided copy per output range; ranges land back-to-back inside each k row.
    size_t dst_off = 0;
    for (size_t i = 0; i < cols.size(); ++i) {
        const size_t run_b = (size_t)(cols[i].second / 16) * cell_b;
        if (hipMemcpy2D((char*)dtre + dst_off, dst_row_b,
                        th + (size_t)(k0 / 16) * src_row_b + (size_t)(cols[i].first / 16) * cell_b,
                        src_row_b, run_b, (size_t)(in_use / 16), hipMemcpyHostToDevice) != hipSuccess) {
            (void)hipFree(dtre);
            return fail(err, errcap, 3, "exl3 H2D2D %s range %zu", prefix, i);
        }
        dst_off += run_b;
    }

    void* dsuh = nullptr; void* dsvh = nullptr;
    if (hipMalloc(&dsuh, (size_t)in_use * 2) != hipSuccess) { (void)hipFree(dtre); return fail(err, errcap, 3, "exl3 suh alloc"); }
    if (hipMemcpy(dsuh, uh + (size_t)k0 * 2, (size_t)in_use * 2, hipMemcpyHostToDevice) != hipSuccess) {
        (void)hipFree(dtre); (void)hipFree(dsuh); return fail(err, errcap, 3, "exl3 suh H2D"); }
    if (hipMalloc(&dsvh, (size_t)out_use * 2) != hipSuccess) { (void)hipFree(dtre); (void)hipFree(dsuh); return fail(err, errcap, 3, "exl3 svh alloc"); }
    { size_t o = 0;
      for (size_t i = 0; i < cols.size(); ++i) {
          if (hipMemcpy((char*)dsvh + o, vh + (size_t)cols[i].first * 2, (size_t)cols[i].second * 2,
                        hipMemcpyHostToDevice) != hipSuccess) {
              (void)hipFree(dtre); (void)hipFree(dsuh); (void)hipFree(dsvh);
              return fail(err, errcap, 3, "exl3 svh H2D"); }
          o += (size_t)cols[i].second * 2;
      } }

    w->trellis = (const uint16_t*)dtre;
    w->suh = (const __half*)dsuh;
    w->svh = (const __half*)dsvh;
    w->in = in_use; w->out = out_use; w->K = K;
    w->mcg  = look(S, p + ".mcg")  ? 1 : 0;
    w->mul1 = look(S, p + ".mul1") ? 1 : 0;
    return 0;
}

int q27_exl3_upload_shard_col(q27_exl3_store_t* S, const char* prefix, const long long* rbase,
                              const long long* rlen, int nrange, int g, int ndev,
                              q27_exl3_t* w, char* err, size_t errcap) {
    return upload_shard(S, prefix, Q27_EXL3_COL, rbase, rlen, nrange, g, ndev, w, err, errcap);
}
int q27_exl3_upload_shard_row(q27_exl3_store_t* S, const char* prefix, int g, int ndev,
                              q27_exl3_t* w, char* err, size_t errcap) {
    return upload_shard(S, prefix, Q27_EXL3_ROW, nullptr, nullptr, 0, g, ndev, w, err, errcap);
}

int q27_exl3_upload(q27_exl3_store_t* S, const char* prefix, q27_exl3_t* w, char* err, size_t errcap) {
    return upload_impl(S, prefix, 0, 0, w, err, errcap);
}
int q27_exl3_upload_cols(q27_exl3_store_t* S, const char* prefix, int n0, int n,
                         q27_exl3_t* w, char* err, size_t errcap) {
    return upload_impl(S, prefix, n0, n, w, err, errcap);
}

void q27_exl3_free(q27_exl3_t* w) {
    if (!w) return;
    if (w->trellis) hipFree((void*)w->trellis);
    if (w->suh) hipFree((void*)w->suh);
    if (w->svh) hipFree((void*)w->svh);
    *w = q27_exl3_t{};
}

// ---------------------------------------------------------------------------
// Per-layer load. The EXL3 tensor names map 1:1 onto the engine's handles --
// linear_attn.in_proj_qkv -> in_qkv, self_attn.q_proj -> q_proj, and so on --
// so this is a format substitution, not a re-architecture. Everything lands
// FULL WIDTH on the CURRENT device: the layer-split residency already gives
// each card whole layers, so no projection is sharded and no collective runs
// inside a layer.
//
// K is read per tensor. Both stores put the lm_head at K=6 while the "4bpw"
// store's layers are K=4, so a global K would be wrong for one tensor in
// every configuration.
// The per-layer loader and the sideload need the engine's layer type. The
// standalone kernel bench links only the EXL3 objects, so they are excluded
// there rather than dragging the whole loader in.
#ifndef Q27_EXL3_STANDALONE
#include "q27_load.h"

static int one(q27_exl3_store_t* S, const std::string& pfx, q27_exl3_t* w,
               char* err, size_t cap, size_t* tot) {
    int rc = q27_exl3_upload(S, pfx.c_str(), w, err, cap);
    if (!rc && tot) *tot += w->bytes();
    return rc;
}

// Per-layer load. `ndev == 1` takes the whole projection; ndev > 1 takes this
// card's TP shard, matching q27_load.cpp's Tp::col / Tp::row exactly -- the
// same ranges, the same order. in_proj_qkv is THREE column ranges
// (q 2048 | k 2048 | v 6144) so each card gets whole heads, which is the one
// place a single range would silently mis-shard.
int q27_exl3_load_layer_tp(q27_exl3_store_t* S, q27_layer_t* H, int L, int g, int ndev,
                           size_t* bytes_out, char* err, size_t errcap) {
    if (!S || !H) return 1;
    char pre[128];
    std::snprintf(pre, sizeof pre, "model.language_model.layers.%d.", L);
    const std::string p = pre;
    size_t tot = 0;
    int rc = 0;

    static const long long one_b[1] = {0};
    const long long inter[1] = {Q27_INTER}, qrows[1] = {Q27_QROWS}, kvrows[1] = {Q27_KVROWS},
                    zrows[1] = {Q27_GDN_Z};
    static const long long qkv_b[3] = {0, 2048, 4096};
    static const long long qkv_l[3] = {2048, 2048, 6144};

    const unsigned skip = q27_exl3_skip_mask();

    #define EXCOL(name, h, base, len, nr) \
        do { rc = upload_shard(S, (p + name).c_str(), Q27_EXL3_COL, base, len, nr, g, ndev, h, err, errcap); \
             if (rc) return rc; tot += (h)->bytes(); } while (0)
    #define EXROW(name, h) \
        do { rc = upload_shard(S, (p + name).c_str(), Q27_EXL3_ROW, nullptr, nullptr, 0, g, ndev, h, err, errcap); \
             if (rc) return rc; tot += (h)->bytes(); } while (0)

    if (!(skip & Q27_EXL3_SKIP_MLP)) {
        EXCOL("mlp.gate_proj", &H->ex_gate, one_b, inter, 1);
        EXCOL("mlp.up_proj",   &H->ex_up,   one_b, inter, 1);
        EXROW("mlp.down_proj", &H->ex_down);
    }

    if (H->is_full) {
        if (!(skip & Q27_EXL3_SKIP_QKV)) {
            EXCOL("self_attn.q_proj", &H->ex_q, one_b, qrows,  1);
            EXCOL("self_attn.k_proj", &H->ex_k, one_b, kvrows, 1);
            EXCOL("self_attn.v_proj", &H->ex_v, one_b, kvrows, 1);
        }
        if (!(skip & Q27_EXL3_SKIP_O)) EXROW("self_attn.o_proj", &H->ex_o);
        H->ex_live = (skip ? 1 : (H->ex_gate.live() && H->ex_up.live() && H->ex_down.live() &&
                     H->ex_q.live() && H->ex_k.live() && H->ex_v.live() && H->ex_o.live()));
    } else {
        // three SEPARATE handles -- see the note on q27_layer_t::ex_iqkv
        if (!(skip & Q27_EXL3_SKIP_QKV))
        for (int r = 0; r < 3; ++r) {
            const long long rb[1] = {qkv_b[r]}, rl[1] = {qkv_l[r]};
            rc = upload_shard(S, (p + "linear_attn.in_proj_qkv").c_str(), Q27_EXL3_COL,
                              rb, rl, 1, g, ndev, &H->ex_iqkv[r], err, errcap);
            if (rc) return rc;
            tot += H->ex_iqkv[r].bytes();
        }
        if (!(skip & Q27_EXL3_SKIP_Z)) EXCOL("linear_attn.in_proj_z",   &H->ex_iz,   one_b, zrows, 1);
        if (!(skip & Q27_EXL3_SKIP_O)) EXROW("linear_attn.out_proj",    &H->ex_op);
        H->ex_live = (skip ? 1 : (H->ex_gate.live() && H->ex_up.live() && H->ex_down.live() &&
                     H->ex_iqkv[0].live() && H->ex_iqkv[1].live() && H->ex_iqkv[2].live() &&
                     H->ex_iz.live() && H->ex_op.live()));
    }
    #undef EXCOL
    #undef EXROW
    if (bytes_out) *bytes_out += tot;
    return H->ex_live ? 0 : 5;
}

int q27_exl3_load_layer(q27_exl3_store_t* S, q27_layer_t* H, int L,
                        size_t* bytes_out, char* err, size_t errcap) {
    return q27_exl3_load_layer_tp(S, H, L, 0, 1, bytes_out, err, errcap);
}

// The MTP draft layer. The engine slices NVFP4 fc into 2 and down into 4
// K-ranges because its 4-bit kernels cap K; EXL3 has no such cap, so these load
// FULL WIDTH and the caller drops the slice loops and their partial sums.
int q27_exl3_load_mtp(const char* dir, q27_exl3_t* fc, q27_exl3_t* gate,
                      q27_exl3_t* up, q27_exl3_t* down, char* err, size_t errcap) {
    q27_exl3_store_t* S = nullptr;
    if (int rc = q27_exl3_open(dir, &S, err, errcap)) return rc;
    struct { const char* n; q27_exl3_t* h; } t[4] = {
        {"mtp.fc", fc}, {"mtp.layers.0.mlp.gate_proj", gate},
        {"mtp.layers.0.mlp.up_proj", up}, {"mtp.layers.0.mlp.down_proj", down} };
    int rc = 0;
    for (int i = 0; i < 4 && !rc; ++i) rc = q27_exl3_upload(S, t[i].n, t[i].h, err, errcap);
    q27_exl3_close(S);
    return rc;
}

void q27_exl3_free_layer(q27_layer_t* H) {
    if (!H) return;
    q27_exl3_t* all[12] = { &H->ex_gate, &H->ex_up, &H->ex_down, &H->ex_iqkv[0], &H->ex_iqkv[1],
                            &H->ex_iqkv[2], &H->ex_iz, &H->ex_op, &H->ex_q, &H->ex_k, &H->ex_v, &H->ex_o };
    for (int i = 0; i < 12; ++i) q27_exl3_free(all[i]);
    H->ex_live = 0;
}

// ---------------------------------------------------------------------------
// Q27_EXL3_DIR sideload. Same seam as q27_fp8_mlp_load / q27_mxfp4_sideload_mlp:
// the incumbent residency is already up, and this attaches a second, BETTER
// representation to the layer handles. Every dispatcher prefers a live EXL3
// handle; anything this does not cover keeps running exactly as before, so a
// partial load degrades to the incumbent arm rather than to a null pointer.
//
// Layers land FULL WIDTH on their layer-split owner card, which is why no
// projection is sharded and no collective runs inside an EXL3 layer.
int q27_exl3_sideload(q27_model_t* m, const int* devices, int ndev, const char* dir) {
    if (!m || !devices || ndev < 1 || !dir || !*dir) return 0;
    char err[512] = {0};
    q27_exl3_store_t* S = nullptr;
    if (q27_exl3_open(dir, &S, err, sizeof err)) {
        std::fprintf(stderr, "Q27_EXL3 open %s FAILED: %s\n", dir, err);
        return 0;
    }
    std::fprintf(stderr, "Q27_EXL3: %s opened, %zu tensors, %.2f GiB mapped\n",
                 dir, S->t.size(), (double)q27_exl3_store_bytes(S) / 1073741824.0);

    // WHICH RESIDENCY TO ATTACH TO. Under Q27_LAYER_SPLIT the owner card holds
    // whole layers and q27_layer_ls is what both decode and the sweep read.
    // Without it there is only the TP residency, and at ndev=1 its "shards" are
    // full width -- which is exactly the shape of a full-width trellis, so the
    // attach is valid. At ndev>1 a TP shard is a fraction of each projection and
    // a full-width trellis would be the wrong weights, so refuse rather than
    // silently serve garbage.
    // Layer-split handles are FULL WIDTH per owned layer, so ndev plays no part
    // there. The TP residency is sharded, and EXL3 now follows the same split.
    const bool use_ls = q27_env_flag("Q27_LAYER_SPLIT", true);
    std::fprintf(stderr, "Q27_EXL3: attaching to the %s residency\n", use_ls ? "layer-split" : "TP (ndev=1, full width)");

    int done = 0, failed = 0;
    size_t per_card[Q27_MAX_DEVICES] = {0};
    for (int g = 0; g < ndev; ++g) {
        if (hipSetDevice(devices[g]) != hipSuccess) continue;
        for (int L = 0; L < Q27_LAYERS; ++L) {
            if (use_ls && !q27_ls_owned(L, g, ndev)) continue;
            q27_layer_t* H = const_cast<q27_layer_t*>(use_ls ? q27_layer_ls(m, L, g)
                                                             : q27_layer_tp(m, L, g));
            if (!H) continue;
            size_t b = 0;
            if (int rc = q27_exl3_load_layer_tp(S, H, L, g, use_ls ? 1 : ndev, &b, err, sizeof err)) {
                std::fprintf(stderr, "Q27_EXL3 layer %d (card %d) FAILED rc=%d: %s\n", L, g, rc, err);
                q27_exl3_free_layer(H);
                ++failed;
                continue;
            }
            per_card[g] += b;
            ++done;
        }
    }
    // lm_head: column-parallel over ndev, exactly as Tp::nvfp4_col placed the
    // NVFP4 head (q27_load.cpp:1300). K=6 in BOTH stores, and 248320/4 = 62080
    // is a multiple of 128, so the blockwise Hadamard slices cleanly.
    if (!(q27_exl3_skip_mask() & Q27_EXL3_SKIP_HEAD)) {
        static const long long hb[1] = {0};
        const long long hl[1] = {Q27_VOCAB};
        for (int g = 0; g < ndev; ++g) {
            if (hipSetDevice(devices[g]) != hipSuccess) continue;
            q27_globals_t* G = const_cast<q27_globals_t*>(use_ls ? q27_globals_ls(m, g) : q27_globals(m, g));
            if (!G) continue;
            if (!use_ls && !G->lm_head.w && ndev > 1) { /* head not on this card */ }
            const int hn = use_ls ? 1 : ndev, hg = use_ls ? 0 : g;
            if (upload_shard(S, "lm_head", Q27_EXL3_COL, hb, hl, 1, hg, hn, &G->ex_head, err, sizeof err)) {
                std::fprintf(stderr, "Q27_EXL3 lm_head card %d FAILED: %s\n", g, err);
            } else {
                per_card[g] += G->ex_head.bytes();
                std::fprintf(stderr, "Q27_EXL3 lm_head card %d: K=%d in=%d out=%d (%.2f GiB)\n",
                             g, G->ex_head.K, G->ex_head.in, G->ex_head.out,
                             (double)G->ex_head.bytes() / 1073741824.0);
            }
        }
    }

    for (int g = 0; g < ndev; ++g)
        if (per_card[g]) {
            size_t fr = 0, tt = 0;
            if (hipSetDevice(devices[g]) == hipSuccess) (void)hipMemGetInfo(&fr, &tt);
            std::fprintf(stderr, "Q27_EXL3 card %d: %.2f GiB of trellis resident, %.2f GiB free\n",
                         g, (double)per_card[g] / 1073741824.0, (double)fr / 1073741824.0);
        }
    std::fprintf(stderr, "Q27_EXL3: %d layers EXL3-resident, %d failed\n", done, failed);
    std::fflush(stderr);
    // The store's mmap is only needed during upload.
    q27_exl3_close(S);
    return done;
}

#endif  // Q27_EXL3_STANDALONE
