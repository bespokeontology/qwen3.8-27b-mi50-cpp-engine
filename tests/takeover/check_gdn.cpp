#include "../../src/q27_gdn.hip"
#include "common.hpp"
int main() {
    int cases=0;size_t total=0;
    for(int dev=0;dev<3;++dev){HC(hipSetDevice(dev));
      for(int kh0:{0,6,11})for(int nr:{1,3,4})for(int pattern:{0,1}) {
        const int khn=kh0==0?6:5,vh=khn*3,n=vh*128,qst=khn*128*5,ost=3072,abst=vh;
        const int guard=32; const size_t shard=size_t(vh)*128*128,snapst=shard*2;
        Buf<unsigned short> qkv(nr*qst),z(nr*n),a(nr*abst),b(nr*abst),al(48),dt(48),nw(128,bf(1.f));
        Buf<float> sr(shard+2*guard,-12345.75f),sc(shard+2*guard,-12345.75f);
        Buf<float> br(nr*snapst+2*guard,-12345.75f),bc(nr*snapst+2*guard,-12345.75f);
        Buf<unsigned short> yr(nr*ost+2*guard,0x5a5a),yc(nr*ost+2*guard,0x5a5a);
        Buf<signed char> qr(nr*n+2*guard,85),qc(nr*n+2*guard,85);
        Buf<float> xs(nr*(n/16)+2*guard,-12345.75f),ys(nr*(n/16)+2*guard,-12345.75f);
        for(size_t i=0;i<qkv.h.size();++i)qkv.h[i]=bf(pattern?float(int((i*17)%67)-33)/32.f:0.f);
        for(size_t i=0;i<z.h.size();++i)z.h[i]=bf(float(int((i*13)%31)-15)/8.f);
        for(size_t i=0;i<a.h.size();++i){a.h[i]=bf(float(int(i%11)-5)/2.f);b.h[i]=bf(float(int(i%13)-6)/4.f);}
        for(size_t i=0;i<shard;++i)sr.h[guard+i]=sc.h[guard+i]=pattern?float(int((i*17)%257)-128)/4096.f:0.f;
        qkv.upload();z.upload();a.upload();b.upload();sr.upload();sc.upload();
        auto run=[&](Buf<float>& state,Buf<unsigned short>& out,Buf<float>& banks,signed char* qq,float* ss){
          q27_gdn_scan_tp_snap_r(qkv.d,qst,z.d,n,a.d,b.d,abst,al.d,dt.d,nw.d,state.d+guard,out.d+guard,ost,kh0,khn,nr,qq,ss,.3125f,banks.d+guard,snapst,1,0);
        };
        run(sr,yr,br,nullptr,nullptr);
        hipLaunchKernelGGL(reference_quant,dim3((n/16+255)/256,nr),dim3(256),0,0,yr.d+guard,ost,qr.d+guard,n,xs.d+guard,n/16,n,3.2f);
        run(sc,yc,bc,qc.d+guard,ys.d+guard);HC(hipDeviceSynchronize());
        auto bad=compare(sr,sc,"state")+compare(br,bc,"banks")+compare(yr,yc,"BF16/padding")+compare(qr,qc,"Q/guards")+compare(xs,ys,"scales/guards");
        total+=bad;++cases;if(bad)std::fprintf(stderr,"GDN dev=%d kh0=%d nr=%d pattern=%d bad=%zu\n",dev,kh0,nr,pattern,bad);
      }
    }
    std::printf("GDN cases=%d mismatches=%zu\n",cases,total);return total?1:0;
}
