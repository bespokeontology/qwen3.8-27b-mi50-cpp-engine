static __device__ __forceinline__ void cols_test_original_quant_body(
        const unsigned short* __restrict__ x, signed char* __restrict__ xq,
        float* __restrict__ xs, int n, float global) {
    const int g  = blockIdx.x * blockDim.x + threadIdx.x;
    const int k0 = g * 16;
    if (k0 >= n) return;

    float v[16];
    float mx = 0.0f;
    if (k0 + 16 <= n) {
        const uint4 a = *(const uint4*)(x + k0);      // 16 bf16 = 32 B, 16-B aligned
        const uint4 b = *(const uint4*)(x + k0 + 8);
        const unsigned p[8] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w };
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[2*i  ] = q27_bf2f((unsigned short)(p[i] & 0xFFFFu));
            v[2*i+1] = q27_bf2f((unsigned short)(p[i] >> 16));
            mx = fmaxf(mx, fmaxf(fabsf(v[2*i]), fabsf(v[2*i+1])));
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = (k0 + i < n) ? q27_bf2f(x[k0 + i]) : 0.0f;
            mx = fmaxf(mx, fabsf(v[i]));
        }
    }

    const float s   = mx * global * (1.0f / 127.0f);
    const float inv = (s > 0.0f) ? (global / s) : 0.0f;    // == 127/mx; global cancels by design
    xs[g] = s;

    // pack into four dwords directly: a `signed char[16]` local is only 1-byte aligned, so
    // type-punning it to uint4 would be an under-aligned access waiting to be a scratch spill.
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned d = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int t = (int)rintf(v[4*i + j] * inv);
            t = (t > 127) ? 127 : ((t < -127) ? -127 : t);
            d |= ((unsigned)t & 0xFFu) << (8 * j);
        }
        w[i] = d;
    }
    if (k0 + 16 <= n) {
        uint4 o; o.x = w[0]; o.y = w[1]; o.z = w[2]; o.w = w[3];
        *(uint4*)(xq + k0) = o;                       // xq+k0 is 16-B aligned (k0 % 16 == 0)
    } else {
        for (int i = 0; i < 16 && k0 + i < n; ++i)
            xq[k0 + i] = (signed char)((w[i >> 2] >> (8 * (i & 3))) & 0xFFu);
    }
}
