// q27_dflash2.cpp — checkpoint open + load-time int8 repack + per-card upload for the DFlash2 drafter.
// Hand-written safetensors scanner (same format contract as q27_load.cpp), no JSON lib, no Python.
// Branch dflash2-port. Single-instance by design (one q27_df2_open per process).
#include "q27_dflash2.h"
#include "q27.h"
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>

namespace {
int fail(char* err, size_t cap, int code, const char* fmt, ...) {
    if (err && cap) { va_list ap; va_start(ap, fmt); vsnprintf(err, cap, fmt, ap); va_end(ap); }
    return code;
}
struct Ent { long long off = 0, len = 0; int rows = 0, K = 0; };
struct HdrScan { std::unordered_map<std::string, Ent> m; long long payload = 0; };
static const char* P; static const char* E;
static void ws() { while (P < E && (*P==' '||*P=='\t'||*P=='\n'||*P=='\r'||*P==','||*P==':')) ++P; }
static bool jstr(std::string& out) { ws(); if (P>=E||*P!='"') return false; ++P; const char* s=P; while (P<E && *P!='"') { if (*P=='\\') P+=2; else ++P; } if (P>=E) return false; out.assign(s,P-s); ++P; return true; }
static bool jnum(long long& out) { ws(); if (P>=E) return false; char* en=nullptr; long long v=strtoll(P,&en,10); if (en==P) return false; out=v; P=en; return true; }
static int scan(const std::string& hdr, HdrScan& H) {
    P = hdr.data(); E = hdr.data()+hdr.size();
    ws(); if (P>=E||*P!='{') return -1; ++P;
    bool first = true;
    while (true) {
        ws(); if (P>=E) break; if (*P=='}') { ++P; break; }
        if (!first) ws(); first = false;
        std::string name; if (!jstr(name)) break;
        ws(); if (P>=E||*P!='{') break;
        if (name == "__metadata__") { int d=1; ++P; while (d && P<E) { if (*P=='{')++d; else if (*P=='}')--d; ++P; } continue; }
        Ent e{}; int depth=1; ++P; std::vector<long long> sh;
        while (depth && P<E) {
            if (*P=='{') { ++depth; ++P; continue; }
            if (*P=='}') { --depth; ++P; continue; }
            std::string k; const char* save=P; if (!jstr(k)) { ++P; continue; }
            if (k=="shape") { ws(); if (P<E&&*P=='[') { ++P; ws(); sh.clear(); while (P<E&&*P!=']') { long long v=0; if (jnum(v)) sh.push_back(v); ws(); } if (P<E) ++P; } }
            else if (k=="data_offsets") { ws(); if (P<E&&*P=='[') { ++P; ws(); long long a=0,b=0; if (jnum(a)) { ws(); jnum(b); } e.off=a; e.len=b-a; while (P<E&&*P!=']') ++P; if (P<E) ++P; } }
            else { if (!jstr(k)) { P=save; ++P; } }
            ws();
        }
        if (sh.size()==2) { e.rows=(int)sh[0]; e.K=(int)sh[1]; }
        H.m[name]=e;
    }
    for (const auto& kv : H.m) H.payload += kv.second.len;
    return 0;
}
struct HostTen { std::vector<signed char> w; std::vector<float> s; int rows=0, K=0; };
static void conv_bf16_i8g(const unsigned short* src, int rows, int K, HostTen& out) {
    out.rows=rows; out.K=K; const int ng=K/64;
    out.w.assign((size_t)rows*K,0); out.s.assign((size_t)rows*ng,0.f);
    for (int r=0;r<rows;++r) {
        const unsigned short* row = src + (size_t)r*K;
        for (int g=0;g<ng;++g) {
            const unsigned short* grp = row + (size_t)g*64;
            float amax=1e-12f;
            for (int i=0;i<64;++i) {
                unsigned u=(unsigned)grp[i]<<16;   // bf16 pattern is already little-endian in the u16; shift to the f32 high half
                float f; std::memcpy(&f,&u,4); float a=std::fabs(f); if (a>amax) amax=a;
            }
            const float sc = amax/127.f, isc = 127.f/amax;
            for (int i=0;i<64;++i) {
                unsigned u=(unsigned)grp[i]<<16; float f; std::memcpy(&f,&u,4);
                int v=(int)std::lround(f*isc); if (v>127)v=127; if (v<-127)v=-127;
                out.w[(size_t)r*K+(size_t)g*64+i]=(signed char)v;
            }
            out.s[(size_t)r*ng+g]=sc;
        }
    }
}
static void conv_bf16_f32(const unsigned short* src, int n, float* dst) {
    for (int i=0;i<n;++i) { unsigned u=(unsigned)src[i]<<16; float f; std::memcpy(&f,&u,4); dst[i]=f; }
}
} // namespace

struct q27_df2_t::q27_df2_host_t {
    HostTen q[5], k[5], v[5], o[5], g[5], u[5], d[5], ac[5], mc[5];
    std::vector<HostTen> fc, shp, pb, sb;
    std::vector<float> in_norm[5], post_norm[5], q_norm[5], k_norm[5], acb[5], mcb[5];
    std::vector<float> hidden_norm, final_norm;
};

int q27_df2_open(const char* path, q27_df2_t* W, char* err, size_t errcap) {
    struct stat st{}; if (stat(path,&st)) return fail(err,errcap,1,"df2 stat %s: %s",path,strerror(errno));
    int fd=open(path,O_RDONLY); if (fd<0) return fail(err,errcap,1,"df2 open: %s",strerror(errno));
    void* map=mmap(nullptr,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if (map==MAP_FAILED) { close(fd); return fail(err,errcap,1,"df2 mmap failed"); }
    const char* base=(const char*)map;
    unsigned long long N=0; for (int i=0;i<8;++i) N |= ((unsigned long long)(unsigned char)base[i]) << (8*i);
    if (N > (unsigned long long)st.st_size-8) { munmap(map,(size_t)st.st_size); close(fd); return fail(err,errcap,1,"df2 header len bad"); }
    HdrScan H; if (scan(std::string(base+8,(size_t)N), H)) { munmap(map,(size_t)st.st_size); close(fd); return fail(err,errcap,1,"df2 header scan failed"); }
    if (H.m.size()!=81 || H.payload!=3848808960LL) { munmap(map,(size_t)st.st_size); close(fd); return fail(err,errcap,1,"df2 identity mismatch: %zu tensors %lld payload (want 81 / 3848808960)", H.m.size(), H.payload); }
    auto* HS = new q27_df2_t::q27_df2_host_t(); W->H = HS;
    auto load = [&](const std::string& nm, HostTen& t) -> bool {
        auto it=H.m.find(nm); if (it==H.m.end()) { fail(err,errcap,0,"df2 missing tensor %s",nm.c_str()); return false; }
        conv_bf16_i8g((const unsigned short*)(base+8+(size_t)N+it->second.off), it->second.rows, it->second.K, t);
        return true;
    };
    auto small = [&](const std::string& nm, int nv, std::vector<float>& dst) -> bool {
        auto it=H.m.find(nm); if (it==H.m.end()) { fail(err,errcap,0,"df2 missing tensor %s",nm.c_str()); return false; }
        dst.resize(nv); conv_bf16_f32((const unsigned short*)(base+8+(size_t)N+it->second.off), nv, dst.data());
        return true;
    };
#define DF2_BAD() do { munmap(map,(size_t)st.st_size); close(fd); delete HS; W->H=nullptr; return 1; } while(0)
    char n[256];
    for (int i=0;i<5;++i) {
        snprintf(n,sizeof n,"layers.%d.self_attn.q_proj.weight",i);    if (!load(n,HS->q[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.self_attn.k_proj.weight",i);    if (!load(n,HS->k[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.self_attn.v_proj.weight",i);    if (!load(n,HS->v[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.self_attn.o_proj.weight",i);    if (!load(n,HS->o[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.mlp.gate_proj.weight",i);       if (!load(n,HS->g[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.mlp.up_proj.weight",i);         if (!load(n,HS->u[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.mlp.down_proj.weight",i);       if (!load(n,HS->d[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.attention_conv.kernel_projection.weight",i); if (!load(n,HS->ac[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.mlp_conv.kernel_projection.weight",i);       if (!load(n,HS->mc[i])) DF2_BAD();
    }
    if (!load("fc.weight",HS->fc.emplace_back()) || !load("candidate_selector.hidden_projection.weight",HS->shp.emplace_back())) DF2_BAD();
    {   // codebooks stay BF16 (selector scoring fidelity): keep the raw payload
        auto rawload = [&](const std::string& nm, HostTen& t) -> bool {
            auto it=H.m.find(nm); if (it==H.m.end()) { fail(err,errcap,0,"df2 missing tensor %s",nm.c_str()); return false; }
            t.rows=it->second.rows; t.K=it->second.K;
            const size_t nb=(size_t)t.rows*t.K*2;
            t.w.resize(nb);
            std::memcpy(t.w.data(), base+8+(size_t)N+it->second.off, nb);
            return true;
        };
        if (!rawload("candidate_selector.predecessor_codebook",HS->pb.emplace_back()) ||
            !rawload("candidate_selector.successor_codebook",HS->sb.emplace_back())) DF2_BAD();
    }
    for (int i=0;i<5;++i) {
        snprintf(n,sizeof n,"layers.%d.input_layernorm.weight",i);         if (!small(n,5120,HS->in_norm[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.post_attention_layernorm.weight",i); if (!small(n,5120,HS->post_norm[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.self_attn.q_norm.weight",i);        if (!small(n,128,HS->q_norm[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.self_attn.k_norm.weight",i);        if (!small(n,128,HS->k_norm[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.attention_conv.base_kernel",i);     if (!small(n,2*2*5120,HS->acb[i])) DF2_BAD();
        snprintf(n,sizeof n,"layers.%d.mlp_conv.base_kernel",i);           if (!small(n,2*2*5120,HS->mcb[i])) DF2_BAD();
    }
    if (!small("hidden_norm.weight",5120,HS->hidden_norm) || !small("norm.weight",5120,HS->final_norm)) DF2_BAD();
    munmap(map,(size_t)st.st_size); close(fd);
    // sizes
    W->bytes_device=0; W->bytes_host=0;
    for (int i=0;i<5;++i) {
        for (const auto* t : {&HS->q[i],&HS->k[i],&HS->v[i],&HS->o[i],&HS->g[i],&HS->u[i],&HS->d[i],&HS->ac[i],&HS->mc[i]}) {
            W->bytes_device += (size_t)t->rows*t->K + (size_t)t->rows*(t->K/64)*4;
            W->bytes_host += t->w.size() + t->s.size()*4;
        }
        W->bytes_device += (size_t)2*2*5120*4*2;   // two fp32 base kernels
    }
    for (const auto* t : {&HS->fc[0],&HS->shp[0],&HS->pb[0],&HS->sb[0]}) {
        W->bytes_device += (size_t)t->rows*t->K + (size_t)t->rows*(t->K/64)*4;
        W->bytes_host += t->w.size() + t->s.size()*4;
    }
    {   // host-side sanity dump of the o weights conversion (bf16 decode correctness gate)
        const HostTen& t0 = HS->o[0];
        std::printf("Q27_DFLASH2_OWT rows=%d K=%d w[0..7]=%d %d %d %d %d %d %d %d s0=%.6f s1=%.6f\n",
                    t0.rows, t0.K, t0.w[0],t0.w[1],t0.w[2],t0.w[3],t0.w[4],t0.w[5],t0.w[6],t0.w[7], t0.s[0], t0.s[1]);
    }
    {
        auto set = [](q27_i8g_d_t& t, const HostTen& h) { t.rows=h.rows; t.K=h.K; };
        for (int i=0;i<5;++i) { set(W->L[i].q,HS->q[i]); set(W->L[i].k,HS->k[i]); set(W->L[i].v,HS->v[i]); set(W->L[i].o,HS->o[i]);
            set(W->L[i].gate,HS->g[i]); set(W->L[i].up,HS->u[i]); set(W->L[i].down,HS->d[i]);
            set(W->L[i].attn_conv_proj,HS->ac[i]); set(W->L[i].mlp_conv_proj,HS->mc[i]); }
        set(W->fc,HS->fc[0]); set(W->sel_hidden_proj,HS->shp[0]); set(W->pred_codebook,HS->pb[0]); set(W->succ_codebook,HS->sb[0]);
    }
    return 0;
}

int q27_df2_upload(q27_df2_t* W, int dev, char* err, size_t errcap) {
    auto* HS = W->H; if (!HS) return fail(err,errcap,1,"df2 upload: no host state");
    hipError_t e0 = hipSetDevice(dev); if (e0 != hipSuccess) return fail(err,errcap,1,"df2 hipSetDevice %d: %d", dev, (int)e0);
    size_t fr0=0, tt0=0; hipMemGetInfo(&fr0,&tt0);
    e0 = hipMalloc(&W->arena, W->bytes_device);
    if (e0 != hipSuccess) return fail(err,errcap,1,"df2 arena %zu B on dev %d: %d", W->bytes_device, dev, (int)e0);
    char* base=(char*)W->arena; size_t off=0;
    auto put = [&](q27_i8g_d_t& t, const HostTen& h) {
        const size_t wb=(size_t)h.rows*h.K, sb=(size_t)h.rows*(h.K/64)*4;
        t.w=(signed char*)(base+off); off+=wb; if (off&255) off=(off+255)&~255ull;
        t.s=(float*)(base+off); off+=sb; if (off&255) off=(off+255)&~255ull;
        hipMemcpy(t.w,h.w.data(),wb,hipMemcpyHostToDevice);
        hipMemcpy(t.s,h.s.data(),sb,hipMemcpyHostToDevice);
    };
    auto putf = [&](float*& t, const std::vector<float>& h) {
        t=(float*)(base+off); off+=h.size()*4; if (off&255) off=(off+255)&~255ull;
        hipMemcpy(t,h.data(),h.size()*4,hipMemcpyHostToDevice);
    };
    std::vector<unsigned short> bf;   // norms/base kernels go as bf16 (the kernels take bf16 weights)
    auto putbf = [&](float*& t, const std::vector<float>& h) {
        t=(float*)(base+off); off+=h.size()*2; if (off&255) off=(off+255)&~255ull;
        bf.resize(h.size()); for (size_t i=0;i<h.size();++i) { float f=h[i]; unsigned u; std::memcpy(&u,&f,4); bf[i]=(unsigned short)((u + 0x7fff + ((u>>16)&1)) >> 16); }
        hipMemcpy(t,bf.data(),h.size()*2,hipMemcpyHostToDevice);
    };
    for (int i=0;i<5;++i) {
        put(W->L[i].q,HS->q[i]); put(W->L[i].k,HS->k[i]); put(W->L[i].v,HS->v[i]); put(W->L[i].o,HS->o[i]);
        put(W->L[i].gate,HS->g[i]); put(W->L[i].up,HS->u[i]); put(W->L[i].down,HS->d[i]);
        put(W->L[i].attn_conv_proj,HS->ac[i]); put(W->L[i].mlp_conv_proj,HS->mc[i]);
        putf(W->L[i].attn_conv_base,HS->acb[i]); putf(W->L[i].mlp_conv_base,HS->mcb[i]);
        putbf(W->L[i].in_norm,HS->in_norm[i]); putbf(W->L[i].post_norm,HS->post_norm[i]);
        putbf(W->L[i].q_norm,HS->q_norm[i]); putbf(W->L[i].k_norm,HS->k_norm[i]);
    }
    put(W->fc,HS->fc[0]); put(W->sel_hidden_proj,HS->shp[0]); put(W->pred_codebook,HS->pb[0]); put(W->succ_codebook,HS->sb[0]);
    putf(W->hidden_norm,HS->hidden_norm); putf(W->final_norm,HS->final_norm);
    size_t fr1=0, tt1=0; hipMemGetInfo(&fr1,&tt1);
    std::printf("Q27_DFLASH2 upload card %d: %.3f GiB arena, free %.2f -> %.2f GiB of %.2f GiB\n",
                dev, (double)W->bytes_device/1073741824.0, (double)fr0/1073741824.0, (double)fr1/1073741824.0, (double)tt1/1073741824.0);
    return 0;
}

int q27_df2_upload_sharded(q27_df2_t* W, int dev, int ndev, char* err, size_t errcap) {
    auto* HS = W->H; if (!HS) return fail(err,errcap,1,"df2 upload: no host state");
    hipError_t e0 = hipSetDevice(dev); if (e0 != hipSuccess) return fail(err,errcap,1,"df2 hipSetDevice %d: %d", dev, (int)e0);
    const int g = dev;
    // per-dev staging vectors (host slices)
    struct Slice { std::vector<signed char> w; std::vector<float> s; int rows=0, K=0; };
    auto rowshard = [&](const HostTen& h, int rows) -> Slice {
        Slice o; o.rows=rows; o.K=h.K; const int ng=h.K/64;
        o.w.assign(h.w.begin(), h.w.begin()+(size_t)rows*h.K);
        o.s.assign(h.s.begin(), h.s.begin()+(size_t)rows*ng);
        return o;
    };
    auto rowshard_off = [&](const HostTen& h, int r0, int rows) -> Slice {   // card g owns intermediate rows [r0, r0+rows)
        Slice o; o.rows=rows; o.K=h.K; const int ng=h.K/64;
        o.w.assign(h.w.begin()+(size_t)r0*h.K, h.w.begin()+(size_t)(r0+rows)*h.K);
        o.s.assign(h.s.begin()+(size_t)r0*ng, h.s.begin()+(size_t)(r0+rows)*ng);
        return o;
    };
    auto colshard = [&](const HostTen& h, int k0, int K) -> Slice {
        Slice o; o.rows=h.rows; o.K=K; const int ng=K/64;
        o.w.resize((size_t)h.rows*K); o.s.resize((size_t)h.rows*ng);
        for (int r=0;r<h.rows;++r) {
            std::memcpy(o.w.data()+(size_t)r*K, h.w.data()+(size_t)r*h.K+k0, (size_t)K);
            std::memcpy(o.s.data()+(size_t)r*ng, h.s.data()+(size_t)r*(h.K/64)+k0/64, (size_t)ng*4);
        }
        return o;
    };
    const int Qr=4096/ndev, KVr=1024/ndev, Or=5120/ndev, GUr=17408/ndev, FCr=5120/ndev, SELr=256/ndev, DOWNk=17408/ndev;
    std::vector<Slice> sq(5),sk(5),sv(5),so(5),sg(5),su(5),sd(5),sac(5),smc(5),sfc(1),ssel(1),spb(1),ssb(1);
    for (int i=0;i<5;++i) {
        sq[i]=rowshard(HS->q[i],4096); sk[i]=rowshard(HS->k[i],1024); sv[i]=rowshard(HS->v[i],1024); so[i]=rowshard(HS->o[i],5120);   // q/k/v/o FULL per card: attention + o run replicated, zero draft-time reduces
        // MLP is REPLICATED per card: the drafter is tiny (+~1 GiB/card) and replicating it removes
        // FIVE cross-card reduces (each with a host barrier) from every block, and makes the fixed
        // five-layer forward capturable as one graph. Sharding it bought nothing but latency.
        sg[i]=rowshard_off(HS->g[i],g*GUr,GUr); su[i]=rowshard_off(HS->u[i],g*GUr,GUr); sd[i]=colshard(HS->d[i],g*DOWNk,DOWNk);
        sac[i]=rowshard(HS->ac[i],1280); smc[i]=rowshard(HS->mc[i],1280);
    }
    sfc[0]=rowshard(HS->fc[0],5120); ssel[0]=rowshard(HS->shp[0],256);   // fc + selector FULL per card: the conditioning runs replicated, no draft-time reduce
    if (g==0) {   // codebooks: full raw bf16 copies (card 0 only)
        auto rawcopy = [&](const HostTen& h, Slice& o) { o.rows=h.rows; o.K=h.K; o.w.assign(h.w.begin(), h.w.end()); };
        rawcopy(HS->pb[0], spb[0]); rawcopy(HS->sb[0], ssb[0]);
    }
    (void)0;   // codebooks handled above (raw bf16 copies)
    size_t bytes=0;
    auto add = [&](const Slice& s) { bytes += (size_t)s.rows*s.K + (size_t)s.rows*(s.K/64)*4; };
    for (int i=0;i<5;++i) { add(sq[i]); add(sk[i]); add(sv[i]); add(so[i]); add(sg[i]); add(su[i]); add(sd[i]); add(sac[i]); add(smc[i]);
        bytes += (size_t)2*2*5120*4*2;      // attn_conv_base + mlp_conv_base, fp32
        bytes += (size_t)2*5120*2;          // in_norm + post_norm, bf16  -- these four were NEVER counted, so the
        bytes += (size_t)2*128*2;           // q_norm + k_norm, bf16         arena overflowed and the tail uploads
        bytes += 256*6; }                   // per-put 256B alignment slack  (hidden_norm/final_norm) landed nowhere
    bytes += (size_t)2*5120*2 + 256*4;      // hidden_norm + final_norm, bf16
    add(sfc[0]); add(ssel[0]); if (g==0) { bytes += (size_t)spb[0].rows*spb[0].K*2 + (size_t)ssb[0].rows*ssb[0].K*2; }   // codebooks: raw bf16
    size_t fr0=0, tt0=0; hipMemGetInfo(&fr0,&tt0);
    e0 = hipMalloc(&W->arena, bytes); if (e0 != hipSuccess) return fail(err,errcap,1,"df2 shard arena %zu B on dev %d: %d", bytes, dev, (int)e0);
    char* base=(char*)W->arena; size_t off=0;
    auto put = [&](q27_i8g_d_t& t, const Slice& h) {
        const size_t wb=(size_t)h.rows*h.K, sb=(size_t)h.rows*(h.K/64)*4;
        t.rows=h.rows; t.K=h.K;
        t.w=(signed char*)(base+off); off+=wb; if (off&255) off=(off+255)&~255ull;
        t.s=(float*)(base+off); off+=sb; if (off&255) off=(off+255)&~255ull;
        hipMemcpy(t.w,h.w.data(),wb,hipMemcpyHostToDevice);
        hipMemcpy(t.s,h.s.data(),sb,hipMemcpyHostToDevice);
    };
    auto putf = [&](float*& t, const std::vector<float>& h) {
        t=(float*)(base+off); off+=h.size()*4; if (off&255) off=(off+255)&~255ull;
        hipMemcpy(t,h.data(),h.size()*4,hipMemcpyHostToDevice);
    };
    std::vector<unsigned short> bf;   // norms/base kernels go as bf16 (the kernels take bf16 weights)
    auto putbf = [&](float*& t, const std::vector<float>& h) {
        t=(float*)(base+off); off+=h.size()*2; if (off&255) off=(off+255)&~255ull;
        bf.resize(h.size()); for (size_t i=0;i<h.size();++i) { float f=h[i]; unsigned u; std::memcpy(&u,&f,4); bf[i]=(unsigned short)((u + 0x7fff + ((u>>16)&1)) >> 16); }
        hipMemcpy(t,bf.data(),h.size()*2,hipMemcpyHostToDevice);
    };
    for (int i=0;i<5;++i) {
        put(W->L[i].q,sq[i]); put(W->L[i].k,sk[i]); put(W->L[i].v,sv[i]); put(W->L[i].o,so[i]);
        put(W->L[i].gate,sg[i]); put(W->L[i].up,su[i]); put(W->L[i].down,sd[i]);
        put(W->L[i].attn_conv_proj,sac[i]); put(W->L[i].mlp_conv_proj,smc[i]);
        putf(W->L[i].attn_conv_base,HS->acb[i]); putf(W->L[i].mlp_conv_base,HS->mcb[i]);
        putbf(W->L[i].in_norm,HS->in_norm[i]); putbf(W->L[i].post_norm,HS->post_norm[i]);
        putbf(W->L[i].q_norm,HS->q_norm[i]); putbf(W->L[i].k_norm,HS->k_norm[i]);
    }
    put(W->fc,sfc[0]); put(W->sel_hidden_proj,ssel[0]);
    if (g==0) {   // codebooks: raw bf16 [rows][K], no scales
        for (int cb = 0; cb < 2; ++cb) {
            const Slice& h = cb ? ssb[0] : spb[0];
            q27_i8g_d_t& t = cb ? W->succ_codebook : W->pred_codebook;
            const size_t nb = (size_t)h.rows * h.K * 2;
            t.rows=h.rows; t.K=h.K; t.w=(signed char*)(base+off); off+=nb; if (off&255) off=(off+255)&~255ull; t.s=nullptr;
            hipMemcpy(t.w, h.w.data(), nb, hipMemcpyHostToDevice);
        }
    } else { W->pred_codebook.w=nullptr; W->pred_codebook.s=nullptr; W->pred_codebook.rows=0; W->succ_codebook.w=nullptr; W->succ_codebook.s=nullptr; W->succ_codebook.rows=0; }
    putbf(W->hidden_norm,HS->hidden_norm); putbf(W->final_norm,HS->final_norm);
    if (off > bytes) return fail(err,errcap,1,"df2 shard arena OVERFLOW on dev %d: used %zu B of %zu B - tail uploads (norms) were silently dropped", dev, off, bytes);
    if (getenv("Q27_DFLASH2_NORMW")) {   // does the device arena hold what the safetensors file holds?
        unsigned short rb[4]; float fr[4]; hipMemcpy(rb,(const unsigned short*)W->hidden_norm,8,hipMemcpyDeviceToHost);
        for (int i=0;i<4;++i){unsigned u=(unsigned)rb[i]<<16; std::memcpy(&fr[i],&u,4);} 
        std::printf("Q27_DFLASH2_NORMW dev%d host_hidden=%.4f %.4f %.4f %.4f  dev_hidden=%.4f %.4f %.4f %.4f  host_in0=%.4f %.4f\n", dev,
            HS->hidden_norm[0],HS->hidden_norm[1],HS->hidden_norm[2],HS->hidden_norm[3], fr[0],fr[1],fr[2],fr[3], HS->in_norm[0][0], HS->in_norm[0][1]);
    }
    size_t fr1=0, tt1=0; hipMemGetInfo(&fr1,&tt1);
    std::printf("Q27_DFLASH2 shard upload card %d/%d: %.3f GiB arena, free %.2f -> %.2f GiB of %.2f GiB\n",
                dev, ndev, (double)bytes/1073741824.0, (double)fr0/1073741824.0, (double)fr1/1073741824.0, (double)tt1/1073741824.0);
    return 0;
}

void q27_df2_free(q27_df2_t* W) {
    if (W->arena) { hipFree(W->arena); W->arena=nullptr; }
    delete W->H; W->H=nullptr;
}
