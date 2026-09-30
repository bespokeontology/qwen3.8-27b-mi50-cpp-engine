// exl3_dequant_gfx906.hip — gfx906 (MI50, ROCm 5.7, wave64) HIP port of
// src/exl3_dequant.cu, the proven bit-exact EXL3 trellis decode
// (exllamav3 v0.0.43 pin c5d9c657 ext.reconstruct).
//
// All decode bit math lives in exl3_dequant_gfx906.h (shared with the
// roofline probe). This TU carries the reconstruct kernel + launch,
// mirroring reconstruct.cu:10-92 via exl3_dequant.cu:422-543.
//
// wave64 launch geometry: blockDim stays 256 threads = 4 full wave64
// wavefronts; one wavefront (64 lanes = the CUDA kernel's warp pair
// 2w/2w+1) cooperates on two of the tile's 8-column groups. All CUDA
// warp-level index math (lane_id = t % 32, warp_id = t / 32) is unchanged;
// the only wave64 adaptation is inside frags_to_tile (full 64-lane mask,
// width-32 segmented shuffle — see the header).
//
// VGPR budget / occupancy (gfx906: 65536 VGPRs per CU, 4 SIMDs, wave64;
// allocation granularity 4 VGPRs/lane, 256-lane wave groups):
//   target <= 128 VGPRs/thread. A 256-thread block = 4 waves; at 128
//   VGPRs/lane one wave occupies 128*64 = 8192 of a SIMD's 16384 VGPRs, so
//   2 waves/SIMD -> 8 waves/CU -> 2 resident blocks/CU. LDS per block is
//   packed staging (256*K/16*2*8 bytes; 1 KiB at K=4) + tile (16*8*8 half2
//   = 4 KiB) = 5 KiB at K=4, far under the 64 KiB/CU LDS limit, so VGPRs
//   set occupancy. No MFMA, no FP8 hardware, no --ffast-math (Makefile
//   law): fp16 adds/FMAs map to plain v_add_f16 / v_fma_f16 / packed
//   v_pk variants the compiler selects; every op is IEEE fp16, bit-exact
//   vs the CUDA reference by construction.
//
// Build: see Makefile.hip (hipcc --offload-arch=gfx906 -fno-gpu-rdc).
// Host-only syntax check: make -f src/hip/Makefile.hip check

#include "exl3_dequant_gfx906.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace exl3gfx
{

// ---------------------------------------------------------------------------
// ref: reconstruct.cu:10-83 (mirrored at exl3_dequant.cu:422-494)
template <int K, int cb>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    __half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset,
    int packed_blocks_k,
    int prepack
)
{
    constexpr int packed_size = 256 * K / 16;  // in uint16s   // reconstruct.cu:20

    int t = threadIdx.x;                        // reconstruct.cu:22
    int lane_id = t % 32;                       // reconstruct.cu:23 (see wave64 note, frags_to_tile)
    int warp_id = t / 32;                       // reconstruct.cu:24
    int k = blockIdx.y;                         // reconstruct.cu:25
    int n = blockIdx.x * 8;                     // reconstruct.cu:26
    int tiles_n = gridDim.x;                    // reconstruct.cu:27
    int out_blocks_n = tiles_n * 8;             // reconstruct.cu:28

    // Load packed 16*128 tile. GLM_PREPACK=0: checkpoint order [k][n][W]
    // (reconstruct.cu:32-34). GLM_PREPACK=1: load-time [n][k][W] — the same
    // eight (k, n+j) tiles, each at (n+j)*packed_blocks_k + k.
    __shared__ uint32_t s_packed[8][packed_size / 2];                    // reconstruct.cu:31
    if (!prepack) {
        g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;  // reconstruct.cu:32
        if (t < packed_size)                                                 // reconstruct.cu:33
            ((int4*) s_packed)[t] = ((int4*) g_packed)[t];                   // reconstruct.cu:34
    } else {
        constexpr int int4_per_tile = packed_size / 8;
        if (t < packed_size) {
            const int j = t / int4_per_tile;
            const int o = t % int4_per_tile;
            const uint16_t* src = g_packed
                + ((uint64_t)(n + packed_n_offset + j) * packed_blocks_k + k)
                  * packed_size;
            ((int4*) s_packed)[t] = ((const int4*) src)[o];
        }
    }
    __syncthreads();

    // Dequant
    FragB frag[2];                                                       // reconstruct.cu:38
    dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);  // reconstruct.cu:39

    // Shuffle from tensor core layout to row major tile
    __shared__ __half2 tile[16][8][8];                                   // reconstruct.cu:43
    frags_to_tile(frag, lane_id, warp_id, tile);                         // reconstruct.cu:45-73
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;                                     // reconstruct.cu:78
    int c = t % 16;                                     // reconstruct.cu:79
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));                          // reconstruct.cu:80
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;  // reconstruct.cu:81
    *out_int4 = tile_int4[t];                           // reconstruct.cu:82
}

// ref: reconstruct.cu:85-92
#define __(i, cb) reconstruct_kernel<i, cb>
constexpr auto reconstruct_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

}  // namespace exl3gfx

// ---------------------------------------------------------------------------
// Host entry point. Mirrors reconstruct()/reconstruct_slice() with
// n_offset == 0 (reconstruct.cu:141-152 -> 97-139), exactly as
// exl3_dequant.cu:511-544.
void exl3_reconstruct_gfx906(
    __half* out,
    const uint16_t* packed,
    int K,
    bool mcg,
    bool mul1,
    int in_features,
    int out_features,
    hipStream_t stream
)
{
    // packed logical shape: (rows, packed_cols, 256*K/16) uint16
    // ref: reconstruct.cu:110-111 (shape checks)
    int rows = in_features / 16;         // reconstruct.cu:114 (packed.size(0))
    int packed_cols = out_features / 16; // reconstruct.cu:115 (packed.size(1))

    int cols = out_features / 16;        // reconstruct.cu:121
    int packed_n_offset = 0;             // reconstruct.cu:122 with n_offset=0

    dim3 blockDim(256);                  // reconstruct.cu:124
    dim3 gridDim(cols / 8, rows);        // reconstruct.cu:125

    int cbi = K - 1;                     // reconstruct.cu:127
    if (mcg) cbi += 8;                   // reconstruct.cu:128
    else if (mul1) cbi += 16;            // reconstruct.cu:129

    int prepack = 0;
    if (const char* e = std::getenv("GLM_PREPACK"))
        prepack = (e[0] != '\0' && !(e[0] == '0' && e[1] == '\0')) ? 1 : 0;

    EXL3_LAUNCH(exl3gfx::reconstruct_kernel_instances[cbi], gridDim, blockDim, stream,  // reconstruct.cu:131
        out,
        packed,
        packed_cols,
        packed_n_offset,
        rows,
        prepack
    );
}
