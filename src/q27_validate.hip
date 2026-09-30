// Optional post-request finite-state gate. It runs outside native timing.
#include "q27.h"
__global__ void q27_finite_kernel(const void* ptr, size_t count, int kind,
                                  unsigned* bad, int category) {
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(size_t)gridDim.x*blockDim.x) {
        bool invalid=false;
        if(kind==0) invalid=(__float_as_uint(((const float*)ptr)[i])&0x7f800000u)==0x7f800000u;
        else {const unsigned bits=((const unsigned short*)ptr)[i];
              const unsigned mask=kind==1?0x7f80u:0x7c00u;invalid=(bits&mask)==mask;}
        if(invalid)atomicAdd(bad+category,1u);
    }
}
extern "C" void q27_finite_check(const void* ptr,size_t count,int kind,unsigned* bad,int category,hipStream_t stream) {
    if(!ptr||!count)return;
    const size_t blocks=(count+255)/256;
    hipLaunchKernelGGL(q27_finite_kernel,dim3(blocks<1024?blocks:1024),dim3(256),0,stream,ptr,count,kind,bad,category);
}
