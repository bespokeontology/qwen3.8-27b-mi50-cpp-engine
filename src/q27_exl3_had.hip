// q27_exl3_had.hip — gfx906 (wave64) port of exllamav3 had_r_128, fp16 path.
//
// Ported from the intermediate CUDA had128.cu port, itself a
// verbatim port of exllamav3 pin c5d9c657:
//   dispatch + r_scale:  exllamav3_ext/quant/hadamard.cu:88-139
//   inner transform:     exllamav3_ext/quant/hadamard_inner.cuh:9-144
//
// THE INDEX ORDER OF H128 IS DEFINED BY THIS KERNEL (in-thread H4 over elements
// [4t..4t+3], then H32 across lanes via shfl_xor). The quantizer used the same
// kernel, so any deviation in order is a wrong answer even if it is still "a
// Hadamard transform". Do not "simplify" the butterfly.
//
// EXL3 applies the transform BLOCKWISE over each contiguous 128 elements, so
// every Qwen3.8-27B dim works with no padding: 5120=40*128, 6144=48*128,
// 10240=80*128, 17408=136*128.
//
// wave64 adaptation: the CUDA kernel is 32 threads = one CUDA warp per
// 128-element chunk. On gfx906 a wave is 64 lanes, so a 32-thread block would
// idle half of every wave. Here blockDim=256 (4 full wave64s) and each 32-lane
// GROUP owns one chunk -> 8 chunks per workgroup, with __shfl_xor(..., 32)
// giving the 32-lane segmentation for free. Numerics are unchanged: every lane
// still shuffles only inside its own 32-lane group, exactly as the warp did.
//
// The 64-bit packed shuffle of the CUDA original is split into four 32-bit
// shuffles; on gfx906 a 64-bit shuffle lowers to two ds_bpermute anyway, so
// this is the same instruction count and bit-identical (no arithmetic on the
// packed form -- it is only a transport).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace q27exl3
{

typedef struct { __half2 x, y; } h4;   // exllamav3 util.cuh half4 layout

// hadamard_inner.cuh:14-41 (sign via bit twiddle, fp32 butterfly)
static __device__ __forceinline__
void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        // width 32 => segmented inside each 32-lane half of the wave64
        float ph0 = __shfl_xor(h0, i, 32);
        float ph1 = __shfl_xor(h1, i, 32);
        float ph2 = __shfl_xor(h2, i, 32);
        float ph3 = __shfl_xor(h3, i, 32);
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

// hadamard_inner.cuh:90-144 (had_hf_r_128_inner) hoisted into the kernel.
// One 32-lane group per 128-element chunk; chunk index runs over rows*n/128 so
// a whole M-row activation matrix is one launch (hadamard.cu:9-22 grid mapping,
// rows folded into the linear chunk index since rows are contiguous).
// BF16_IN: the engine's activation currency is bf16 (unsigned short), so the
// pre stage widens on read rather than paying a separate conversion pass over
// the activation. bf16 -> fp32 is a 16-bit left shift; no table, no rounding.
static __device__ __forceinline__ float bf2f(unsigned short b)
{ unsigned u = (unsigned)b << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
static __device__ __forceinline__ unsigned short f2bf(float f)
{ unsigned u; __builtin_memcpy(&u, &f, 4); return (unsigned short)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(256)
void k_had_hf_r_128(const __half* __restrict__ input_ptr,
                    __half* __restrict__ output_ptr,
                    const __half* __restrict__ scale,
                    const float r_scale,
                    const int chunks_per_row,
                    const int chunks_total,
                    const int in_stride, const int out_stride)
{
    const int gid   = blockIdx.x * 256 + threadIdx.x;
    const int chunk = gid >> 5;                 // 32 lanes per chunk
    if (chunk >= chunks_total) return;
    const int t = threadIdx.x & 31;
    const int crow = chunk / chunks_per_row, cblk = chunk % chunks_per_row;
    const size_t inoff  = (size_t)crow * in_stride  + (size_t)cblk * 128;
    const size_t outoff = (size_t)crow * out_stride + (size_t)cblk * 128;

    h4 v = ((const h4*) (input_ptr + inoff))[t];

    // scale is per-FEATURE, length n: index inside the row, not inside the batch

    if constexpr (pre_scale)
    {
        h4 s = ((const h4*) scale)[cblk * 32 + t];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    if constexpr (post_scale)
    {
        h4 s = ((const h4*) scale)[cblk * 32 + t];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }

    ((h4*) (output_ptr + outoff))[t] = v;
}

// Fused fp32->fp16 entry for the GEMV output: the fused decode+GEMV accumulates
// in fp32 (atomicAdd), so the post stage reads fp32 and writes fp16. Saves a
// separate narrowing pass over out_features*M.
template <bool post_scale>
__global__ __launch_bounds__(256)
void k_had_hf_r_128_f32in(const float* __restrict__ input_ptr,
                          __half* __restrict__ output_ptr,
                          const __half* __restrict__ scale,
                          const float r_scale,
                          const int chunks_per_row,
                          const int chunks_total,
                          const int in_stride, const int out_stride)
{
    const int gid   = blockIdx.x * 256 + threadIdx.x;
    const int chunk = gid >> 5;
    if (chunk >= chunks_total) return;
    const int t = threadIdx.x & 31;
    const int crow = chunk / chunks_per_row, cblk = chunk % chunks_per_row;
    const size_t inoff  = (size_t)crow * in_stride  + (size_t)cblk * 128;
    const size_t outoff = (size_t)crow * out_stride + (size_t)cblk * 128;

    const float4 f = ((const float4*) (input_ptr + inoff))[t];
    const int srow = cblk;

    float v0 = f.x, v1 = f.y, v2 = f.z, v3 = f.w;
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, t);
    h4 v;
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    if constexpr (post_scale)
    {
        h4 s = ((const h4*) scale)[cblk * 32 + t];
        v.x = __hmul2(v.x, s.x);
        v.y = __hmul2(v.y, s.y);
    }

    ((h4*) (output_ptr + outoff))[t] = v;
}

// bf16 activation in, fp16 out: the EXL3 pre stage as the engine calls it.
template <bool pre_scale>
__global__ __launch_bounds__(256)
void k_had_pre_bf16in(const unsigned short* __restrict__ input_ptr,
                      __half* __restrict__ output_ptr,
                      const __half* __restrict__ scale,
                      const float r_scale,
                      const int chunks_per_row,
                      const int chunks_total,
                      const int in_stride, const int out_stride)
{
    const int gid   = blockIdx.x * 256 + threadIdx.x;
    const int chunk = gid >> 5;
    if (chunk >= chunks_total) return;
    const int t = threadIdx.x & 31;
    const int crow = chunk / chunks_per_row, cblk = chunk % chunks_per_row;
    const size_t inoff  = (size_t)crow * in_stride  + (size_t)cblk * 128;
    const size_t outoff = (size_t)crow * out_stride + (size_t)cblk * 128;

    const ushort4 b = ((const ushort4*)(input_ptr + inoff))[t];
    float v0 = bf2f(b.x), v1 = bf2f(b.y), v2 = bf2f(b.z), v3 = bf2f(b.w);

    if constexpr (pre_scale) {
        const int srow = cblk;
        h4 sc = ((const h4*) scale)[cblk * 32 + t];
        v0 *= __half2float(__low2half(sc.x));  v1 *= __half2float(__high2half(sc.x));
        v2 *= __half2float(__low2half(sc.y));  v3 *= __half2float(__high2half(sc.y));
    }

    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    h4 v;
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    ((h4*) (output_ptr + outoff))[t] = v;
}

// fp32 accumulator in, bf16 out: the EXL3 post stage as the engine consumes it.
template <bool post_scale>
__global__ __launch_bounds__(256)
void k_had_post_bf16out(const float* __restrict__ input_ptr,
                        unsigned short* __restrict__ output_ptr,
                        const __half* __restrict__ scale,
                        const float r_scale,
                        const int chunks_per_row,
                        const int chunks_total,
                        const int in_stride, const int out_stride)
{
    const int gid   = blockIdx.x * 256 + threadIdx.x;
    const int chunk = gid >> 5;
    if (chunk >= chunks_total) return;
    const int t = threadIdx.x & 31;
    const int crow = chunk / chunks_per_row, cblk = chunk % chunks_per_row;
    const size_t inoff  = (size_t)crow * in_stride  + (size_t)cblk * 128;
    const size_t outoff = (size_t)crow * out_stride + (size_t)cblk * 128;

    const float4 f = ((const float4*)(input_ptr + inoff))[t];
    float s0 = f.x + f.y, d0 = f.x - f.y, s1 = f.z + f.w, d1 = f.z - f.w;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    h0 *= r_scale; h1 *= r_scale; h2 *= r_scale; h3 *= r_scale;

    if constexpr (post_scale) {
        const int srow = cblk;
        h4 sc = ((const h4*) scale)[cblk * 32 + t];
        h0 *= __half2float(__low2half(sc.x));  h1 *= __half2float(__high2half(sc.x));
        h2 *= __half2float(__low2half(sc.y));  h3 *= __half2float(__high2half(sc.y));
    }
    ushort4 o; o.x = f2bf(h0); o.y = f2bf(h1); o.z = f2bf(h2); o.w = f2bf(h3);
    ((ushort4*)(output_ptr + outoff))[t] = o;
}

// fp32 accumulator in, fp32 out: for the projections whose consumer is the
// engine's fp32 partial (out_proj, down_proj -- the ones that get reduced).
template <bool post_scale>
__global__ __launch_bounds__(256)
void k_had_post_f32out(const float* __restrict__ input_ptr,
                       float* __restrict__ output_ptr,
                       const __half* __restrict__ scale,
                       const float r_scale,
                       const int chunks_per_row,
                       const int chunks_total,
                       const int in_stride, const int out_stride)
{
    const int gid   = blockIdx.x * 256 + threadIdx.x;
    const int chunk = gid >> 5;
    if (chunk >= chunks_total) return;
    const int t = threadIdx.x & 31;
    const int crow = chunk / chunks_per_row, cblk = chunk % chunks_per_row;
    const size_t inoff  = (size_t)crow * in_stride  + (size_t)cblk * 128;
    const size_t outoff = (size_t)crow * out_stride + (size_t)cblk * 128;

    const float4 f = ((const float4*)(input_ptr + inoff))[t];
    float s0 = f.x + f.y, d0 = f.x - f.y, s1 = f.z + f.w, d1 = f.z - f.w;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    h0 *= r_scale; h1 *= r_scale; h2 *= r_scale; h3 *= r_scale;
    if constexpr (post_scale) {
        const int srow = cblk;
        h4 sc = ((const h4*) scale)[cblk * 32 + t];
        h0 *= __half2float(__low2half(sc.x));  h1 *= __half2float(__high2half(sc.x));
        h2 *= __half2float(__low2half(sc.y));  h3 *= __half2float(__high2half(sc.y));
    }
    float4 o; o.x = h0; o.y = h1; o.z = h2; o.w = h3;
    ((float4*)(output_ptr + outoff))[t] = o;
}

}  // namespace q27exl3

static const float Q27_EXL3_RSCALE = 0.088388347648f;   // 1/sqrt(128), hadamard.cu:107

// y[M][n] = H128_blockwise( x[M][n] (* pre_scale) ) (* post_scale)
// Exactly one of pre_scale / post_scale, mirroring the authority's dispatch law
// (hadamard.cu:115-131): pre (suh) multiplies BEFORE the transform, post (svh)
// AFTER. n must be a multiple of 128.
hipError_t q27_exl3_had128_fp16(__half* y, const __half* x,
                                const __half* pre_scale, const __half* post_scale,
                                int n, int rows, hipStream_t stream, int lin, int lout)
{
    if (lin <= 0) lin = n; if (lout <= 0) lout = n;
    if (n <= 0 || (n & 127) || rows <= 0)      return hipErrorInvalidValue;
    if (pre_scale && post_scale)               return hipErrorInvalidValue;
    const int cpr = n / 128;
    const int tot = cpr * rows;
    const dim3 grid((tot * 32 + 255) / 256);
    if (pre_scale)
        hipLaunchKernelGGL((q27exl3::k_had_hf_r_128<true, false>), grid, dim3(256), 0, stream,
                           x, y, pre_scale, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    else if (post_scale)
        hipLaunchKernelGGL((q27exl3::k_had_hf_r_128<false, true>), grid, dim3(256), 0, stream,
                           x, y, post_scale, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    else
        hipLaunchKernelGGL((q27exl3::k_had_hf_r_128<false, false>), grid, dim3(256), 0, stream,
                           x, y, nullptr, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    return hipPeekAtLastError();
}

// fp32 input variant for the GEMV output stage (post scale only).
hipError_t q27_exl3_had128_f32in(__half* y, const float* x, const __half* post_scale,
                                 int n, int rows, hipStream_t stream, int lin, int lout)
{
    if (lin <= 0) lin = n; if (lout <= 0) lout = n;
    if (n <= 0 || (n & 127) || rows <= 0) return hipErrorInvalidValue;
    const int cpr = n / 128;
    const int tot = cpr * rows;
    const dim3 grid((tot * 32 + 255) / 256);
    if (post_scale)
        hipLaunchKernelGGL((q27exl3::k_had_hf_r_128_f32in<true>), grid, dim3(256), 0, stream,
                           x, y, post_scale, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    else
        hipLaunchKernelGGL((q27exl3::k_had_hf_r_128_f32in<false>), grid, dim3(256), 0, stream,
                           x, y, nullptr, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    return hipPeekAtLastError();
}

// bf16 activation in -> fp16 rotated out (EXL3 pre stage, engine-facing).
hipError_t q27_exl3_had128_pre_bf16(__half* y, const unsigned short* x, const __half* pre_scale,
                                    int n, int rows, hipStream_t stream, int lin, int lout)
{
    if (lin <= 0) lin = n; if (lout <= 0) lout = n;
    if (n <= 0 || (n & 127) || rows <= 0) return hipErrorInvalidValue;
    const int cpr = n / 128, tot = cpr * rows;
    const dim3 grid((tot * 32 + 255) / 256);
    if (pre_scale)
        hipLaunchKernelGGL((q27exl3::k_had_pre_bf16in<true>), grid, dim3(256), 0, stream,
                           x, y, pre_scale, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    else
        hipLaunchKernelGGL((q27exl3::k_had_pre_bf16in<false>), grid, dim3(256), 0, stream,
                           x, y, nullptr, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    return hipPeekAtLastError();
}

// fp32 accumulator -> bf16 out (EXL3 post stage, engine-facing).
hipError_t q27_exl3_had128_post_bf16(unsigned short* y, const float* x, const __half* post_scale,
                                     int n, int rows, hipStream_t stream, int lin, int lout)
{
    if (lin <= 0) lin = n; if (lout <= 0) lout = n;
    if (n <= 0 || (n & 127) || rows <= 0) return hipErrorInvalidValue;
    const int cpr = n / 128, tot = cpr * rows;
    const dim3 grid((tot * 32 + 255) / 256);
    if (post_scale)
        hipLaunchKernelGGL((q27exl3::k_had_post_bf16out<true>), grid, dim3(256), 0, stream,
                           x, y, post_scale, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    else
        hipLaunchKernelGGL((q27exl3::k_had_post_bf16out<false>), grid, dim3(256), 0, stream,
                           x, y, nullptr, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    return hipPeekAtLastError();
}

hipError_t q27_exl3_had128_post_f32(float* y, const float* x, const __half* post_scale,
                                    int n, int rows, hipStream_t stream, int lin, int lout)
{
    if (lin <= 0) lin = n; if (lout <= 0) lout = n;
    if (n <= 0 || (n & 127) || rows <= 0) return hipErrorInvalidValue;
    const int cpr = n / 128, tot = cpr * rows;
    const dim3 grid((tot * 32 + 255) / 256);
    if (post_scale)
        hipLaunchKernelGGL((q27exl3::k_had_post_f32out<true>), grid, dim3(256), 0, stream,
                           x, y, post_scale, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    else
        hipLaunchKernelGGL((q27exl3::k_had_post_f32out<false>), grid, dim3(256), 0, stream,
                           x, y, nullptr, Q27_EXL3_RSCALE, cpr, tot, lin, lout);
    return hipPeekAtLastError();
}
