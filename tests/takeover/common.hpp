#pragma once
#include "q27.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "original_quant_body.inc"
char q27_ovr_log[768] = {}; int q27_ovr_n = 0;
#define HC(x) do { auto e=(x); if(e!=hipSuccess){std::fprintf(stderr,"HIP %s at %d\n",hipGetErrorString(e),__LINE__);std::exit(2);} }while(0)
inline unsigned short bf(float x) { unsigned u; std::memcpy(&u,&x,4); u+=0x7fff+((u>>16)&1); return u>>16; }
template<class T> struct Buf {
    T* d=nullptr; std::vector<T> h;
    Buf(size_t n,T v=T{}) : h(n,v) { HC(hipMalloc(&d,n*sizeof(T))); upload(); }
    ~Buf(){hipFree(d);}
    void upload(){HC(hipMemcpy(d,h.data(),h.size()*sizeof(T),hipMemcpyHostToDevice));}
    void download(){HC(hipMemcpy(h.data(),d,h.size()*sizeof(T),hipMemcpyDeviceToHost));}
};
template<class T> size_t compare(Buf<T>& a,Buf<T>& b,const char* what) {
    a.download(); b.download(); size_t n=0;
    for(size_t i=0;i<a.h.size();++i) if(std::memcmp(&a.h[i],&b.h[i],sizeof(T))) {
        if(n++==0) std::fprintf(stderr,"%s first=%zu reference=%g fused=%g\n",what,i,double(a.h[i]),double(b.h[i]));
    }
    return n;
}
__global__ void reference_quant(const unsigned short* x,int xst,signed char* q,int qst,float* sc,int sst,int n,float gs) {
    cols_test_original_quant_body(x+(size_t)blockIdx.y*xst,q+(size_t)blockIdx.y*qst,sc+(size_t)blockIdx.y*sst,n,gs);
}
