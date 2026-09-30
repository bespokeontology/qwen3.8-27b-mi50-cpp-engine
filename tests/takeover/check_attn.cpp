#include "../../src/q27_attn.hip"
#include "common.hpp"
// Exercise the actual product combine, including its NR row pitches. The oracle is
// the frozen separate quantizer applied to the actual BF16 producer output.
__global__ void seed_partials(int heads,int nr,int ns,int pattern) {
    const int d=threadIdx.x,h=blockIdx.x,r=blockIdx.z;
    for(int s=0;s<ns;++s) {
        if(d==0) {q27_attn_part_ml[0][r][s][h][0]=-float(s)*.07f; q27_attn_part_ml[0][r][s][h][1]=1.f+float((h+r+s)%7)*.03f;}
        const float v=pattern==0?0.f:float((d*31+h*17+r*13+s*7)%257-128)*.03125f;
        q27_attn_part_o[0][r][s][h][d]=q27_f2bf(v);
    }
}
int main() {
    int cases=0; size_t total=0;
    for(int dev=0;dev<3;++dev){HC(hipSetDevice(dev));
      for(int heads:{6,12})for(int nr:{1,3,4})for(int ns:{2,7,128})for(int pattern:{0,1}) {
        const int n=heads*256,ost=3072+256,qst=n,rawst=heads*512,guard=32;
        Buf<unsigned short> raw(nr*rawst),y(nr*ost+2*guard,0x5a5a),z(nr*ost+2*guard,0x5a5a);
        Buf<signed char> qa(nr*qst+2*guard,85),qb(nr*qst+2*guard,85);
        Buf<float> sa(nr*(qst/16)+2*guard,-12345.75f),sb(nr*(qst/16)+2*guard,-12345.75f);
        for(size_t i=0;i<raw.h.size();++i) raw.h[i]=bf(float(int((i*13)%53)-26)*.125f); raw.upload();
        const float scale=pattern==0?1.f:3.2f;
        hipLaunchKernelGGL(seed_partials,dim3(heads,1,nr),dim3(256),0,0,heads,nr,ns,pattern);
        hipLaunchKernelGGL(q27_attn_combine_k,dim3(heads,1,nr),dim3(256),0,0,raw.d,y.d+guard,ns,(signed char*)nullptr,(float*)nullptr,0.f,ost,rawst,0,0,0,0);
        hipLaunchKernelGGL(reference_quant,dim3((n/16+255)/256,nr),dim3(256),0,0,y.d+guard,ost,qa.d+guard,qst,sa.d+guard,qst/16,n,scale);
        hipLaunchKernelGGL(q27_attn_combine_k,dim3(heads,1,nr),dim3(256),0,0,raw.d,z.d+guard,ns,qb.d+guard,sb.d+guard,scale,ost,rawst,qst,0,0,1);
        HC(hipDeviceSynchronize());
        auto bad=compare(y,z,"BF16/padding")+compare(qa,qb,"Q/guards")+compare(sa,sb,"scales/guards");
        total+=bad;++cases;
        if(bad) std::fprintf(stderr,"ATTN dev=%d heads=%d nr=%d splits=%d pattern=%d bad=%zu\n",dev,heads,nr,ns,pattern,bad);
      }
    }
    std::printf("ATTN cases=%d mismatches=%zu\n",cases,total);return total?1:0;
}
