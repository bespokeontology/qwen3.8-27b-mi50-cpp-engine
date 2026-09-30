// Q27_BUILDSTAMP — the engine names its own binary at boot.
//
// WHY THIS EXISTS. On 2026-09-17 the box carried a whole session of A/B numbers whose arm
// identity could not be reconstructed afterwards: no boot line carried a build identity, so
// "which binary ran this boot" was inferred from file mtimes while `q27_gen` was being
// overwritten underneath the experiment. Every arm comparison from that session is therefore
// unattributable. This file makes the identity a property of the RUNNING IMAGE, not of the disk.
//
// It hashes /proc/self/exe, so it names the inode the process actually exec'd. Replacing
// q27_gen on disk while an engine is resident cannot change what a running boot reports, and a
// boot that reports a different md5 from the deployed file is a finding, not a puzzle.
//
// MD5 is implemented here rather than shelled out to `md5sum` because the stamp must cost one
// file read at boot and must not add a process to the serve path. The digest is the same
// algorithm the estate already quotes for q27_gen (`md5sum ./q27_gen`), so
// boot stamps compare directly with every hash in the record.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

// DISK-IDENTIFIABLE ARM. The boot stamp below names the arm only once the engine is RUNNING, which
// is too late to stop a wrong-arm binary being deployed -- on 2026-09-17 a plain `make` (V4 defaulted
// to 0 while the frozen engine is V4=1) produced a candidate that differed from the engine under test
// in the V datapath. Compiling the arm in as a string makes it readable off the disk image:
//     strings q27_gen | grep Q27_ARM
// so a binary can be identified before it is ever launched, and a preserved q27_gen.<name>-<hash>
// copy stays self-describing after its build command has been forgotten.
#define Q27_ARM_STR2(x) #x
#define Q27_ARM_STR(x)  Q27_ARM_STR2(x)
#ifdef Q27_EXPERIMENTS
#define Q27_ARM_EXP "1"
#else
#define Q27_ARM_EXP "0"
#endif
extern "C" __attribute__((used)) const char q27_build_arm[];
const char q27_build_arm[] =
    "Q27_ARM v4=" Q27_ARM_STR(Q27_V4)
    " ch="        Q27_ARM_STR(Q27_PF_CH)
    " tslot="     Q27_ARM_STR(Q27_PF_TSLOT)
    " exp="       Q27_ARM_EXP;

namespace {

struct Md5 {
    uint32_t h[4];
    uint64_t len;              // message length in BYTES
    unsigned char buf[64];
    size_t   n;                // bytes buffered
};

inline uint32_t rol(uint32_t x, int s) { return (x << s) | (x >> (32 - s)); }

const uint32_t K[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
    0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu, 0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
    0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
    0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu, 0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
    0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u, 0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u
};
const int S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

void md5_block(Md5& c, const unsigned char* p) {
    uint32_t m[16];
    for (int i = 0; i < 16; ++i)
        m[i] = (uint32_t)p[4 * i] | ((uint32_t)p[4 * i + 1] << 8)
             | ((uint32_t)p[4 * i + 2] << 16) | ((uint32_t)p[4 * i + 3] << 24);
    uint32_t a = c.h[0], b = c.h[1], d = c.h[2], e = c.h[3];
    for (int i = 0; i < 64; ++i) {
        uint32_t f; int g;
        if (i < 16)      { f = (b & d) | (~b & e);          g = i; }
        else if (i < 32) { f = (e & b) | (~e & d);          g = (5 * i + 1) & 15; }
        else if (i < 48) { f = b ^ d ^ e;                   g = (3 * i + 5) & 15; }
        else             { f = d ^ (b | ~e);                g = (7 * i) & 15; }
        const uint32_t tmp = e;
        e = d;
        d = b;
        b = b + rol(a + f + K[i] + m[g], S[i]);
        a = tmp;
    }
    c.h[0] += a; c.h[1] += b; c.h[2] += d; c.h[3] += e;
}

void md5_init(Md5& c) {
    c.h[0] = 0x67452301u; c.h[1] = 0xefcdab89u; c.h[2] = 0x98badcfeu; c.h[3] = 0x10325476u;
    c.len = 0; c.n = 0;
}

void md5_update(Md5& c, const unsigned char* p, size_t n) {
    c.len += n;
    if (c.n) {
        const size_t take = (n < 64 - c.n) ? n : (64 - c.n);
        std::memcpy(c.buf + c.n, p, take);
        c.n += take; p += take; n -= take;
        if (c.n == 64) { md5_block(c, c.buf); c.n = 0; }
    }
    while (n >= 64) { md5_block(c, p); p += 64; n -= 64; }
    if (n) { std::memcpy(c.buf, p, n); c.n = n; }
}

void md5_final(Md5& c, unsigned char out[16]) {
    const uint64_t bits = c.len * 8u;
    static const unsigned char pad[64] = { 0x80 };
    const size_t padlen = (c.n < 56) ? (56 - c.n) : (120 - c.n);
    md5_update(c, pad, padlen);
    unsigned char tail[8];
    for (int i = 0; i < 8; ++i) tail[i] = (unsigned char)((bits >> (8 * i)) & 0xff);
    const uint64_t saved = c.len;
    md5_update(c, tail, 8);
    c.len = saved;                                  // the length field is not part of the message
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out[4 * i + j] = (unsigned char)((c.h[i] >> (8 * j)) & 0xff);
}

}   // namespace

// md5 of the file the RUNNING process was exec'd from (or of `path` when given). Returns 1 on
// success and writes 32 lowercase hex chars + NUL into `hex`.
extern "C" int q27_md5_file(const char* path, char* hex, unsigned long long* bytes) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return 0;
    Md5 c; md5_init(c);
    static unsigned char buf[1 << 16];
    size_t got; unsigned long long total = 0;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) { md5_update(c, buf, got); total += got; }
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) return 0;
    unsigned char d[16]; md5_final(c, d);
    static const char* H = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) { hex[2 * i] = H[d[i] >> 4]; hex[2 * i + 1] = H[d[i] & 15]; }
    hex[32] = '\0';
    if (bytes) *bytes = total;
    return 1;
}

namespace { char g_bin8[9] = "????????"; }

// The Q27X_* override ledger (declared in q27.h, defined once here). Every native override that
// actually fired, with the value the frozen front end had supplied underneath it. A requested
// value is not evidence of an effective one; this is what the boot receipt prints.
char q27_ovr_log[768] = "";
int  q27_ovr_n = 0;

// First 8 hex chars of the RUNNING image's md5, cached at boot. Every request line carries it, so
// an A/B is attributable from the log alone even when the deployed file is swapped underneath.
extern "C" const char* q27_binstamp8(void) { return g_bin8; }

// The boot line. One line, printed once per engine process, carrying the identity of the image
// that is about to serve plus the configuration that decides the greedy path. `cfg` is a short
// caller-built string (compile-time geometry); everything else is read here.
//
// STDERR ON PURPOSE: the server's stdout pump forwards a WHITELIST of Q27_* prefixes and drops the
// rest, so a new line on stdout is a line nobody ever sees. stderr is logged verbatim (Q27_SPECCFG,
// Q27_KVPLACE and the alias reports all reach the log that way). The same line goes to stdout too,
// so a future whitelist entry would make it cheap for the front end to read.
extern "C" void q27_bootstamp(int spec, int spec_k, int ctx, int ndev, const char* cfg) {
    char hex[33] = "unavailable";
    unsigned long long bytes = 0;
    struct stat st;
    const bool have_stat = (stat("/proc/self/exe", &st) == 0);
    const char* prepared=getenv("Q27_PREPARED_EXE_MD5");
    if(prepared && std::strlen(prepared)==32 && std::strspn(prepared,"0123456789abcdef")==32 && have_stat) {
        std::memcpy(hex,prepared,33);bytes=st.st_size;
        std::fprintf(stderr,"Q27_BOOTSTAMP identity=prelaunch; live inode hash verified after timed readiness\n");
    } else if (!q27_md5_file("/proc/self/exe", hex, &bytes)) std::snprintf(hex, sizeof(hex), "unavailable");
    for (int i = 0; i < 8 && hex[i]; ++i) g_bin8[i] = hex[i];
    g_bin8[8] = '\0';
    // `ovr=` closes the phantom-arm hole: it names every Q27X_* override the engine ACTUALLY
    // applied and the front-end value it displaced, so "what was requested" and "what is running"
    // are the same receipt. `ovr=none` is itself evidence -- it is how a mistyped sweep variable
    // (Q27X_SPECK=3) announces that the arm did not exist.
    const char* fmt = "Q27_BOOTSTAMP exe_md5=%s size=%llu mtime=%lld spec=%d spec_k=%d ctx=%d ndev=%d %s ovr=%s\n";
    std::fprintf(stderr, fmt, hex, bytes, have_stat ? (long long)st.st_mtime : -1LL, spec, spec_k,
                 ctx, ndev, cfg ? cfg : "", q27_ovr_n ? q27_ovr_log : "none");
    std::fprintf(stdout, fmt, hex, bytes, have_stat ? (long long)st.st_mtime : -1LL, spec, spec_k,
                 ctx, ndev, cfg ? cfg : "", q27_ovr_n ? q27_ovr_log : "none");
    std::fflush(stderr);
    std::fflush(stdout);
}
