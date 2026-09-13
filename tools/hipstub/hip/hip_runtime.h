// Host-only HIP stub. NOT for building anything that runs. Its ONLY purpose is
// `g++ -fsyntax-only` on the host TUs from a workstation with no ROCm, so an edit to
// q27_main.cpp / q27_load.cpp can be syntax-verified BEFORE it consumes a GPU window.
#pragma once
#include <cstddef>
#include <cmath>
#include <cstring>
// Device qualifiers collapse to nothing so the __device__ helpers in q27.h merely PARSE.
// They are never called from a host TU; this checks syntax, not semantics.
#define __device__
#define __host__
#define __global__
#define __shared__
#define __forceinline__ inline
#define __restrict__
typedef int hipError_t;
enum { hipSuccess = 0 };
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t*  hipEvent_t;
typedef enum { hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
               hipMemcpyDeviceToDevice, hipMemcpyHostToHost, hipMemcpyDefault } hipMemcpyKind;
enum { hipHostMallocDefault = 0,
       hipHostMallocCoherent    = 0x40000000,
       hipHostMallocNonCoherent = 0x80000000u };
inline const char* hipGetErrorString(hipError_t)                        { return ""; }
inline hipError_t  hipSetDevice(int)                                    { return 0; }
// Real HIP provides templated hipMalloc/hipHostMalloc overloads, so typed T** args are legal.
template<class T> hipError_t hipMalloc(T**, size_t)                     { return 0; }
inline hipError_t  hipMalloc(void**, size_t)                            { return 0; }
inline hipError_t  hipFree(void*)                                       { return 0; }
template<class T> hipError_t hipHostMalloc(T**, size_t, unsigned = 0)   { return 0; }
inline hipError_t  hipHostMalloc(void**, size_t, unsigned = 0)          { return 0; }
inline hipError_t  hipDeviceGetPCIBusId(char*, int, int)                 { return 0; }
enum { hipDeviceAttributeCanUseStreamWaitValue = 1, hipEventDisableTiming = 2,
       hipStreamWaitValueEq = 0 };
inline hipError_t  hipDeviceGetAttribute(int*, int, int)                 { return 0; }
inline hipError_t  hipEventCreateWithFlags(hipEvent_t*, unsigned)        { return 0; }
inline hipError_t  hipEventSynchronize(hipEvent_t)                       { return 0; }
inline hipError_t  hipStreamWaitValue32(hipStream_t, void*, unsigned, unsigned, unsigned = 0xffffffff) { return 0; }
inline hipError_t  hipHostFree(void*)                                   { return 0; }
inline hipError_t  hipHostGetDevicePointer(void** d, void* h, unsigned) { *d = h; return 0; }
inline hipError_t  hipMemset(void*, int, size_t)                        { return 0; }
inline hipError_t  hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return 0; }
inline hipError_t  hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return 0; }
inline hipError_t  hipEventCreate(hipEvent_t*)                          { return 0; }
inline hipError_t  hipEventDestroy(hipEvent_t)                          { return 0; }
inline hipError_t  hipEventRecord(hipEvent_t, hipStream_t)              { return 0; }
inline hipError_t  hipEventElapsedTime(float* ms, hipEvent_t, hipEvent_t) { *ms = 0; return 0; }
inline hipError_t  hipStreamCreate(hipStream_t*)                        { return 0; }
inline hipError_t  hipStreamSynchronize(hipStream_t)                    { return 0; }

// gfx906 device builtins and bit-cast helpers referenced by q27.h's __device__ inlines.
// Bodies are deliberately WRONG-but-typed: this header must never produce a running binary.
inline unsigned __builtin_amdgcn_perm(unsigned, unsigned, unsigned) { return 0u; }
inline int      __builtin_amdgcn_sdot4(int, int, int c, bool)       { return c; }
inline float    __uint_as_float(unsigned u) { float f; std::memcpy(&f,&u,4); return f; }
inline unsigned __float_as_uint(float f)    { unsigned u; std::memcpy(&u,&f,4); return u; }

// Added 2026-09-10 (Qwen lane): symbols the engine now uses that the stub lacked. Without them
// `make syntax` failed on HEAD for reasons unrelated to any edit, which defeats the gate.
#ifndef hipStreamNonBlocking
#define hipStreamNonBlocking 1u
#endif
inline hipError_t  hipStreamCreateWithFlags(hipStream_t*, unsigned)          { return 0; }
inline hipError_t  hipStreamWaitEvent(hipStream_t, hipEvent_t, unsigned)       { return 0; }
inline hipError_t  hipDeviceSynchronize()                                      { return 0; }
// added 2026-09-11: symbols the host TUs now use (peer copies, peer queries, uint4 in q27.h)
struct uint4 { unsigned x, y, z, w; };
inline uint4 make_uint4(unsigned a, unsigned b, unsigned c, unsigned d) { uint4 r; r.x = a; r.y = b; r.z = c; r.w = d; return r; }
inline hipError_t hipMemcpyPeerAsync(void*, int, const void*, int, size_t, hipStream_t) { return 0; }
inline hipError_t hipDeviceCanAccessPeer(int*, int, int) { return 0; }
inline hipError_t hipDeviceEnablePeerAccess(int, unsigned) { return 0; }
inline hipError_t hipEventQuery(hipEvent_t) { return 0; }
inline hipError_t hipMemcpy2DAsync(void*, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind, hipStream_t) { return 0; }
inline hipError_t hipDeviceGetStreamPriorityRange(int* a, int* b) { if (a) *a = 0; if (b) *b = 0; return 0; }
inline hipError_t hipStreamCreateWithPriority(hipStream_t* s, unsigned, int) { if (s) *s = 0; return 0; }
enum { hipErrorPeerAccessAlreadyEnabled = 704 };
inline hipError_t hipGetLastError() { return 0; }
