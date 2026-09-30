// q27_sr.h -- single-residency ("SR") layout helpers: transient per-ROW int8 mirrors for the prefill
// sweep's rocBLAS int8 GEMMs, a natural-order per-16 swiglu quantizer for decode, and a 4-way fp32 sum.
//
// WHY THESE EXIST. Under SR each card holds only its 16 owned layers: MLP as block-scaled FP8
// (q27_fp8_t with bs != nullptr, the Qwen3.8-27B-FP8 e4m3 + BF16 weight_scale_inv convention) and the
// seven fp8 projections as int8 per-64 mirrors (q27_i8g_t). The prefill sweep consumes per-ROW int8
// (q27_i8r_t, ng = 1, gs = K) and today manufactures it IN PLACE from a per-64 mirror
// (q27_i8g64_rowify). In-place is exactly what SR cannot afford -- the resident tensor is the only
// copy -- so every producer here writes OUT OF PLACE into a caller-owned transient.
//
// CONVENTIONS (all match the engine's existing quantizers; the test in tools/q27_sr_test.cpp holds them):
//   * e4m3 WEIGHT bytes are decoded with q27_e4m3s (signed; 0x7F/0xFF decode as +/-INF, the OCP NaN
//     codes never occur in the checkpoint) and the BF16 block scale is a MULTIPLIER:
//         v = q27_e4m3s(w[k]) * bf16_to_f32(bs[(row >> 7) * bs_kblk + (k >> 7)])
//     which is q27_k_proj_fp8_snr<..., BS=1>'s arithmetic verbatim.
//   * Row scale of an fp8 row = max|v| * (1/127) (the engine writes mx * (1.0f/127.0f), not mx/127);
//     inv = scale > 0 ? 1/scale : 0; q = clamp(rintf(v * inv), -127, 127).
//   * Row scale of a per-64 mirror row = MAX OF THE ROW'S PER-64 SCALES, not max|w|, and
//     q = clamp(rintf((float)w[k] * s[k >> 6] * inv)) -- bit-identical to q27_k_i8g64_to_grp at GS = K.
//   * Every kernel is 256 threads, __launch_bounds__(256), 16-byte loads/stores. Base pointers must be
//     16-byte aligned and every row stride a multiple of 16 bytes (K % 64 == 0 at the callers). The
//     wrappers FAIL CLOSED (return 0) on any violation rather than fall back to a slow path.
#ifndef Q27_SR_H
#define Q27_SR_H
#include <hip/hip_runtime.h>
#include <stddef.h>
#include "q27.h"

extern "C" {

// (a) Block-scaled fp8 [rows][K] -> per-ROW int8 [rows][K] (row stride ldw_out BYTES, >= K) + one fp32
//     scale per row. Requires src->bs != nullptr, src->bs_kblk == K/128, K % 128 == 0, K <= 17408.
//     One wave per row; the row is held in registers between the max pass and the quantize pass, so the
//     tensor is read once and written once (bandwidth-bound: ~1 ms budget for [17408][5120] on MI50).
int q27_sr_fp8bs_to_i8row(const q27_fp8_t* src, signed char* w_out, size_t ldw_out, float* rowscale_out, hipStream_t s);

// (b) Four K-slices of ONE logical tensor (the down_proj: rows = 5120, Ks = 4352): src4[c] is [rows][Ks]
//     with its own compacted bs plane [rows/128][Ks/128] (bs_kblk == Ks/128), slice c occupying logical
//     columns [c*Ks, (c+1)*Ks). Row scale = max over ALL FOUR slices; output is contiguous int8
//     [rows][4*Ks] (row stride 4*Ks bytes) + one scale per row. All four slices must agree on rows and K.
int q27_sr_fp8bs4_to_i8row(const q27_fp8_t* const src4[4], signed char* w_out, float* rowscale_out, hipStream_t s);

// (c) OUT-OF-PLACE q27_i8g64_rowify at GS = K: same bytes and scales as q27_k_i8g64_to_grp on the same
//     input. Honours src->ldw / src->lds (K-view strides, elements; 0 = dense). Writes at row stride
//     ldw_out bytes (>= K). Requires K % 64 == 0; ldw (if set) and ldw_out multiples of 16.
int q27_sr_i8g64_to_i8row(const q27_i8g_t* src, signed char* w_out, size_t ldw_out, float* rowscale_out, hipStream_t s);

// (d) q27_swiglu_quant_b with the codes stored in NATURAL order (code j at position j of its 16-group)
//     instead of the NVFP4 consumer's even-then-odd order. Same swiglu, same per-16 scales, same rounding,
//     same argument contract: C rows, row stride n for gate/up/xq and n/16 for xs; n % 16 == 0.
int q27_sr_swiglu_quant_nat_b(const float* gate, const float* up, signed char* xq, float* xs, int n, float in_scale, int C, hipStream_t s);

// (e) dst[i] = ((a[i] + b[i]) + c[i]) + d[i], fp32, in that order. dst may alias a.
int q27_sr_sum4_f32(float* dst, const float* a, const float* b, const float* c, const float* d, size_t n, hipStream_t s);

// (f) Block-scaled fp8 [rows][K] -> per-64 int8 MIRROR into caller-owned w8 [rows][K] int8 + s64
//     [rows][K/64] fp32, with the q27_nvfp4_to_i8g64_into contract: dst is stamped rows, K,
//     alpha = src->in_scale * src->wscale (wscale is 1.0 for a block-scaled tensor), ldw = lds = 0,
//     dst->w = w8, dst->s = s64. Per group: v = e4m3(w) * bf16(bs[...]) (64 | 128, so one block scale
//     per group), sc = max|v| * (1/127), q = clamp(rintf(v*inv)) -- q27_k_fp8_to_i8g64 with the block
//     scale folded in. Requires src->bs != nullptr, bs_kblk == K/128, K % 128 == 0.
int q27_sr_fp8bs_to_i8g64_into(const q27_fp8_t* src, q27_i8g_t* dst, signed char* w8, float* s64, hipStream_t s);

// (g) The four K-slices [rows][Ks] of one logical [rows][4*Ks] tensor (each with its own compacted bs
//     plane, bs_kblk == Ks/128) -> ONE per-64 mirror w8 [rows][4*Ks] + s64 [rows][4*Ks/64], slice c at
//     columns [c*Ks, (c+1)*Ks). Ks % 64 == 0 puts every 64-group inside one slice. dst stamped rows,
//     K = 4*Ks, alpha = src4[0]->in_scale * src4[0]->wscale, ldw = lds = 0.
int q27_sr_fp8bs4_to_i8g64_into(const q27_fp8_t* const src4[4], q27_i8g_t* dst, signed char* w8, float* s64, hipStream_t s);

}
#endif
