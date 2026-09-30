#pragma once
#include "transfer.hpp"
#include <hip/hip_runtime.h>
namespace storage {
struct HipTransfer {
    static constexpr bool direct_reads = true;
    using Stream=hipStream_t;using Context=int;
    void* pinned=nullptr;Stream q=nullptr;hipEvent_t begin[2]{},end[2]{};size_t half;
    static void ck(hipError_t e) {check(e==hipSuccess,hipGetErrorString(e));}
    HipTransfer(int device,size_t size):half(size) {ck(hipSetDevice(device));try {ck(hipHostMalloc(&pinned,2*half));ck(hipStreamCreateWithFlags(&q,hipStreamNonBlocking));for(int i=0;i<2;++i){ck(hipEventCreate(&begin[i]));ck(hipEventCreate(&end[i]));}}catch(...){release();throw;}}
    void release() {if(q)hipStreamSynchronize(q);for(int i=0;i<2;++i){if(begin[i])hipEventDestroy(begin[i]);if(end[i])hipEventDestroy(end[i]);}if(q)hipStreamDestroy(q);if(pinned)hipHostFree(pinned);}
    ~HipTransfer(){release();}
    char* buffer(int i){return static_cast<char*>(pinned)+i*half;}
    Stream stream(){return q;}
    void begin_copy(int i){ck(hipEventRecord(begin[i],q));}
    void append_copy(void* destination,const void* source,size_t bytes){ck(hipMemcpyAsync(destination,source,bytes,hipMemcpyHostToDevice,q));}
    void end_copy(int i){ck(hipEventRecord(end[i],q));}
    void wait(int i){ck(hipEventSynchronize(end[i]));}
    double milliseconds(int i){float ms=0;ck(hipEventElapsedTime(&ms,begin[i],end[i]));return ms;}
    void sync(){ck(hipStreamSynchronize(q));}
};
}
