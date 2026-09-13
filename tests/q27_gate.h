// Gate a device buffer against a CUDA-oracle reference vector.
// POLICY, earned from negative controls that actually fired:
//   * relative L2 and max-abs are the DETECTORS. cosine is supporting evidence only - a control
//     that broke the output by 119% in relative L2 still read cosine 0.9998.
//   * recurrent ops must be gated over SEVERAL SEQUENTIAL POSITIONS. A defect in decay ordering is
//     exactly invisible at position 0, where the state is still zero.
//   * bit identity with the CUDA oracle is NOT required and must not be chased.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

// oracle vectors: raw little-endian BF16, 5120 elements, <tag>_L<layer>_P<pos>.bin
static inline std::vector<float> q27_oracle(const char* dir, const char* tag, int layer, int pos,
                                            int n = 5120) {
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%s_L%03d_P%06d.bin", dir, tag, layer, pos);
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("Q27_GATE_MISSING %s\n", path); std::exit(2); }
    std::vector<unsigned short> raw(n);
    if (std::fread(raw.data(), 2, n, f) != (size_t)n) { std::printf("Q27_GATE_SHORT %s\n", path); std::exit(2); }
    std::fclose(f);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) { unsigned u = ((unsigned)raw[i]) << 16; v[i] = *(float*)&u; }
    return v;
}

struct q27_verdict { double rel_l2, max_abs, cosine; long nonfinite; bool pass; };

static inline q27_verdict q27_compare(const std::vector<float>& got, const std::vector<float>& ref,
                                      double rel_l2_tol = 2e-3, double max_abs_tol = 1e30) {
    // COUNT NON-FINITE FIRST, SEPARATELY. Every reduction below is blind to NaN in the direction
    // that matters: `fabs(a-b) > mx` is FALSE for NaN, so max-abs reads exactly 0.0 on all-NaN
    // output, and `na > 0` is false so cosine reads 0.0 too. This header calls max-abs a DETECTOR,
    // so an all-NaN buffer could be scored as perfect by anything reading it. (Found by the
    // A prior gate reported "worst relative 0.000e+00" on all-NaN
    // attention output for exactly this reason.) A gate that cannot see inf/NaN is not a gate.
    long nonfinite = 0;
    for (size_t i = 0; i < got.size(); ++i) if (!std::isfinite(got[i])) ++nonfinite;

    double d = 0, s = 0, mx = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double a = got[i], b = ref[i];
        d += (a-b)*(a-b); s += b*b; if (fabs(a-b) > mx) mx = fabs(a-b);
        dot += a*b; na += a*a; nb += b*b;
    }
    q27_verdict v;
    v.rel_l2 = (s > 0) ? sqrt(d/s) : sqrt(d);
    v.max_abs = mx;
    v.cosine = (na > 0 && nb > 0) ? dot/sqrt(na*nb) : 0.0;
    v.nonfinite = nonfinite;
    v.pass = (nonfinite == 0) && (v.rel_l2 <= rel_l2_tol) && (v.max_abs <= max_abs_tol);
    return v;
}

static inline bool q27_gate(const char* name, const std::vector<float>& got,
                            const std::vector<float>& ref, double tol = 2e-3) {
    q27_verdict v = q27_compare(got, ref, tol);
    if (v.nonfinite)
        std::printf("  %-34s NON-FINITE %ld/%zu  **FAIL**   (relL2/maxabs/cosine below are "
                    "meaningless -- NaN compares false, so max-abs reads 0)\n",
                    name, v.nonfinite, got.size());
    std::printf("  %-34s relL2=%.6e  maxabs=%.6e  cosine=%.8f  %s\n",
                name, v.rel_l2, v.max_abs, v.cosine, v.pass ? "PASS" : "**FAIL**");
    return v.pass;
}

// read a device BF16 buffer into host floats
static inline std::vector<float> q27_d2h_bf16(const void* dev, int n) {
    std::vector<unsigned short> raw(n);
    hipMemcpy(raw.data(), dev, (size_t)n*2, hipMemcpyDeviceToHost);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) { unsigned u = ((unsigned)raw[i]) << 16; v[i] = *(float*)&u; }
    return v;
}
static inline std::vector<float> q27_d2h_f32(const void* dev, int n) {
    std::vector<float> v(n);
    hipMemcpy(v.data(), dev, (size_t)n*4, hipMemcpyDeviceToHost);
    return v;
}
