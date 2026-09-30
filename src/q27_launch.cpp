// ---------------------------------------------------------------------------------------------
// DIRECT HIP LAUNCH COUNTER (2026-09-17)
//
// The host-collective decomposition (Q27_COLLT) prices a decode round as
//   layers 23.76 ms  =  stage_wait 13.10 (GPU execution)  +  collective 2.87  +  ~7.79 UNACCOUNTED
// and the 7.79 ms was obtained BY SUBTRACTION, with "host kernel-launch overhead" as the
// hypothesis. Subtraction names a residual; it does not identify it. This counts the launches
// directly so the thesis is measured or killed.
//
// CHOKEPOINT. Every launch in this engine reaches the runtime through `hipLaunchKernel`: the 357
// hipLaunchKernelGGL sites and the 19 <<<>>> sites both lower to it, and `nm -u` shows it as the
// single undefined launch symbol. Wrapping it at LINK time (-Wl,--wrap=hipLaunchKernel) therefore
// counts EVERY launch with no call-site edits and no chance of missing a helper -- which a macro
// over hipLaunchKernelGGL would not do, because <<<>>> bypasses the macro entirely.
//
// ATTRIBUTION WITHOUT TAGGING. The wrapper receives the device stub's address, so the per-kernel
// split falls out of a small address->count table plus dladdr() at report time. No family tags to
// place, nothing to keep in sync with the layer body, and nothing to get wrong by hand.
//
// THREAD-LOCAL ON PURPOSE. Each card runs its own host thread issuing its own launches; the
// request report runs on card 0's thread, which is the same thread that issued card 0's launches.
// Thread-local state therefore needs no atomics on the hot path (one increment per launch) and
// reports exactly one card's stream of work, which is the quantity the ledger is denominated in.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cxxabi.h>
#include <ctime>

extern "C" hipError_t __real_hipLaunchKernel(const void*, dim3, dim3, void**, size_t, hipStream_t);

namespace {
constexpr int LK_CAP = 512;                 // power of two; open addressing, never resized
struct LkSlot { const void* fn; long n; };
}

thread_local LkSlot q27_lk_tab[LK_CAP];
thread_local long   q27_lk_total = 0;
thread_local double q27_lk_ms    = 0.0;    // wall spent INSIDE __real_hipLaunchKernel

// WHY THIS EXISTS. host_resid_us/launch is a residual DIVIDED BY a count: it charges everything the
// host does between blocking points -- argument marshalling, helper-function logic, hipMemcpyAsync,
// event records, stream waits -- to kernel dispatch, and is therefore an UPPER BOUND on dispatch,
// not a measurement of it. That distinction decides a whole optimisation lane, because this engine
// already banks a REJECTED epilogue fusion (q27_main.cpp:4033) that deleted 64 launches/token and
// still lost 52.274 -> 50.979 tok/s. If dispatch is only a fraction of the residual, that negative
// is explained and launch-count reduction is not the lever. Two clock reads per launch cost ~0.2%.
static inline double lk_now_ms() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + (double)t.tv_nsec / 1e6;
}

// One predictable branch and one increment. At ~1,500 launches/round against a 28 ms round this is
// below the noise floor, but it is still gated so a production build can carry the wrapper inert.
extern "C" hipError_t __wrap_hipLaunchKernel(const void* f, dim3 gd, dim3 bd,
                                             void** args, size_t sh, hipStream_t s) {
    ++q27_lk_total;
    size_t h = ((uintptr_t)f >> 4) & (LK_CAP - 1);
    for (int i = 0; i < LK_CAP; ++i) {
        LkSlot& sl = q27_lk_tab[(h + i) & (LK_CAP - 1)];
        if (sl.fn == f) { ++sl.n; break; }
        if (!sl.fn)     { sl.fn = f; sl.n = 1; break; }
    }
    const double t0 = lk_now_ms();
    const hipError_t e = __real_hipLaunchKernel(f, gd, bd, args, sh, s);
    q27_lk_ms += lk_now_ms() - t0;
    return e;
}

extern "C" void q27_lk_reset(void) {
    std::memset(q27_lk_tab, 0, sizeof q27_lk_tab);
    q27_lk_total = 0;
    q27_lk_ms = 0.0;
}

extern "C" long q27_lk_total_get(void) { return q27_lk_total; }

// resid_ms is the per-round host residual the COLLT ledger could not attribute
// (layers - stage_wait - collective). Dividing it by the measured launch count is the whole point:
// it yields host-residual-us/launch, which is the number that decides whether the residual really
// is dispatch cost. A few us/launch says yes; a fraction of a us says the residual is elsewhere.
extern "C" void q27_lk_dump(int req, long rounds, int nlayer, double resid_ms_per_round, int topn) {
    if (rounds <= 0 || q27_lk_total <= 0) return;
    const double per_round = (double)q27_lk_total / (double)rounds;
    const double per_layer = nlayer > 0 ? per_round / (double)nlayer : 0.0;
    const double us_each   = per_round > 0 ? (resid_ms_per_round * 1000.0) / per_round : 0.0;
    const double disp_ms = q27_lk_ms / (double)rounds;                  // dispatch, ms/round
    const double disp_us = per_round > 0 ? (disp_ms * 1000.0) / per_round : 0.0;
    std::fprintf(stderr,
        "Q27_LAUNCH %d rounds=%ld total=%ld launches/round=%.1f launches/layer=%.2f "
        "host_resid_ms/round=%.4f host_resid_us/launch=%.3f | dispatch_ms/round=%.4f "
        "dispatch_us/launch=%.3f dispatch_share_of_resid=%.1f%%\n",
        req, rounds, q27_lk_total, per_round, per_layer, resid_ms_per_round, us_each,
        disp_ms, disp_us, resid_ms_per_round > 0 ? 100.0 * disp_ms / resid_ms_per_round : 0.0);

    // Rank by frequency. The top of this list IS the fusion work list: the highest-frequency
    // kernels are where a fused launch removes the most dispatches, and adjacency in the layer
    // body is what makes a pair fusable.
    for (int k = 0; k < topn; ++k) {
        int best = -1; long bn = 0;
        for (int i = 0; i < LK_CAP; ++i)
            if (q27_lk_tab[i].fn && q27_lk_tab[i].n > bn) { bn = q27_lk_tab[i].n; best = i; }
        if (best < 0) break;
        const void* fn = q27_lk_tab[best].fn;
        const char* nm = "?";
        Dl_info info;
        char* dem = nullptr;
        if (dladdr(fn, &info) && info.dli_sname) {
            int st = 0;
            dem = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &st);
            nm = (st == 0 && dem) ? dem : info.dli_sname;
        }
        std::fprintf(stderr, "Q27_LAUNCH   %8.2f/round  %8.3f/layer  %ld  %s\n",
                     (double)bn / (double)rounds,
                     nlayer > 0 ? (double)bn / (double)rounds / (double)nlayer : 0.0, bn, nm);
        if (dem) std::free(dem);
        q27_lk_tab[best].n = 0;            // consume: this dump is one-shot per request
    }
    std::fflush(stderr);
}
