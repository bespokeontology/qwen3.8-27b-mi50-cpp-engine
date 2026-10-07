// DFlash2 BF16 EXECUTION ORACLE - host fp32 recomputation of the five-layer draft block
// from the checkpoint's raw BF16 weights. No Python, no GPU. Reads the engine's dump for the
// anchor/mask embeddings so both arms see byte-identical inputs, then reports per-layer
// cosine/norm against the engine's INT8 arm. NO-CONTEXT arm (matches Q27_DFLASH2_NOCTX=1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
static const int HID=5120, QH=32, KVH=8, HD=128, INTER=17408, DYN=1280, ROWS=8, NL=5;
static inline float b2f(unsigned short u){ unsigned x=(unsigned)u<<16; float f; std::memcpy(&f,&x,4); return f; }
struct Ten { std::vector<float> v; long long r=0,c=0; };
static std::vector<char> HDR; static FILE* FP=nullptr; static long long BASE=0;
static bool find_ten(const std::string& name, long long& o0, long long& o1, long long& d0, long long& d1){
    std::string key = "\"" + name + "\":";
    const char* p = (const char*)memmem(HDR.data(), HDR.size(), key.data(), key.size());
    if(!p) return false;
    const char* sh = strstr(p, "\"shape\":["); const char* du = strstr(p, "\"data_offsets\":[");
    if(!sh||!du) return false;
    d0 = atoll(sh+9); const char* c = strchr(sh+9, ','); d1 = c ? atoll(c+1) : 1;
    if(!c || c > du) { d1 = d0; d0 = 1; }
    o0 = atoll(du+16); const char* c2 = strchr(du+16, ','); o1 = atoll(c2+1);
    return true;
}
static Ten load(const std::string& name){
    Ten t; long long o0,o1,d0,d1;
    if(!find_ten(name,o0,o1,d0,d1)){ std::fprintf(stderr,"missing tensor %s\n",name.c_str()); exit(2); }
    long long n=(o1-o0)/2; t.r=d0; t.c=d1; if(t.r*t.c!=n){ t.r=n/ (d1?d1:1); t.c=d1?d1:n; }
    std::vector<unsigned short> raw(n);
    std::fseek(FP, BASE+o0, SEEK_SET); if(std::fread(raw.data(),2,n,FP)!=(size_t)n){ std::fprintf(stderr,"short read %s\n",name.c_str()); exit(2);}    
    t.v.resize(n); for(long long i=0;i<n;++i) t.v[i]=b2f(raw[i]);
    return t;
}
// y[rows][out] = x[rows][in] * W[out][in]^T
static int I8W=0, I8A=0;   // precision bisection arms, applied exactly like the engine (per-64 weights, per-16 activations)
static void q64(std::vector<float>& w,int rows,int K){ for(int r=0;r<rows;++r) for(int g=0;g<K/64;++g){ float* p=&w[(size_t)r*K+(size_t)g*64];
    float am=1e-12f; for(int i=0;i<64;++i) am=std::fmax(am,std::fabs(p[i])); float sc=am/127.f, isc=127.f/am;
    for(int i=0;i<64;++i){ int v=(int)std::lround(p[i]*isc); if(v>127)v=127; if(v<-127)v=-127; p[i]=(float)v*sc; } } }
static void q16(std::vector<float>& a,int rows,int K){ for(int r=0;r<rows;++r) for(int g=0;g<K/16;++g){ float* p=&a[(size_t)r*K+(size_t)g*16];
    float am=1e-12f; for(int i=0;i<16;++i) am=std::fmax(am,std::fabs(p[i])); float sc=am/127.f, isc=127.f/am;
    for(int i=0;i<16;++i){ int v=(int)std::lround(p[i]*isc); if(v>127)v=127; if(v<-127)v=-127; p[i]=(float)v*sc; } } }
static void gemm(const std::vector<float>& x,const Ten& W,std::vector<float>& y,int rows,int out,int in){
    std::vector<float> xq; const std::vector<float>* xp = &x;
    if(I8A){ xq = x; q16(xq,rows,in); xp = &xq; }
    Ten Wq; const Ten* Wp = &W;
    if(I8W){ Wq.v = W.v; q64(Wq.v,out,in); Wp = &Wq; }
    const std::vector<float>& xx = *xp; const Ten& WW = *Wp;
    y.assign((size_t)rows*out,0.f);
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;++r) for(int o=0;o<out;++o){ const float* w=&WW.v[(size_t)o*in]; const float* xr=&xx[(size_t)r*in];
        float a=0.f; for(int i=0;i<in;++i) a+=xr[i]*w[i]; y[(size_t)r*out+o]=a; }
}
static void rmsnorm(const std::vector<float>& x,const Ten& w,std::vector<float>& y,int rows,int n){
    y.assign((size_t)rows*n,0.f);
    for(int r=0;r<rows;++r){ const float* xr=&x[(size_t)r*n]; double s=0; for(int i=0;i<n;++i) s+=(double)xr[i]*xr[i];
        float inv=1.0f/std::sqrt((float)(s/n)+1e-6f);                    // w-only: reference passes correction_weight=false
        for(int i=0;i<n;++i) y[(size_t)r*n+i]=xr[i]*inv*w.v[i]; }
}
static void dconv(const std::vector<float>& in,const std::vector<float>& dyn,const Ten& base,std::vector<float>& out,int branch){
    out.assign((size_t)ROWS*HID,0.f); const int groups=HID/16;
    for(int row=0;row<ROWS;++row) for(int ch=0;ch<HID;++ch){ int grp=ch/16; float res=0.f;
        for(int off=0;off<2;++off){ if(row<off) continue;
            float v=in[(size_t)(row-off)*HID+ch];
            float b=base.v[(size_t)(branch*2+off)*HID+ch];
            float d=dyn[(size_t)row*(4*groups)+(size_t)branch*(2*groups)+(size_t)off*groups+grp];
            res+=(b+d)*v; }
        out[(size_t)row*HID+ch]=res; }
}
static void prepqk(std::vector<float>& x,const Ten& nw,int heads,int pos0){
    for(int row=0;row<ROWS;++row) for(int h=0;h<heads;++h){ float* p=&x[((size_t)row*heads+h)*HD];
        double s=0; for(int d=0;d<HD;++d) s+=(double)p[d]*p[d];
        float inv=1.0f/std::sqrt((float)(s/HD)+1e-6f);
        float tmp[HD]; for(int d=0;d<HD;++d) tmp[d]=p[d]*inv*nw.v[d];
        for(int d=0;d<HD;++d){ int half=64, pair=d<half?d+half:d-half, freq=d%half;
            float invf=std::pow(10000000.0f,-2.0f*(float)freq/(float)HD), ang=(float)(pos0+row)*invf;
            p[d]=tmp[d]*std::cos(ang)+(d<half?-tmp[pair]:tmp[pair])*std::sin(ang); } }
}
static double cosine(const float* a,const float* b,int n){ double d=0,na=0,nb=0;
    for(int i=0;i<n;++i){ d+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i]; }
    return (na>0&&nb>0)? d/(std::sqrt(na)*std::sqrt(nb)) : 0.0; }
static double l2(const float* a,int n){ double s=0; for(int i=0;i<n;++i) s+=(double)a[i]*a[i]; return std::sqrt(s); }
static FILE* DF=nullptr;
static void cmp(const char* tag,const std::vector<float>& ours,int width){
    if(!DF) return; std::vector<unsigned short> e((size_t)ROWS*width);
    if(std::fread(e.data(),2,(size_t)ROWS*width,DF)!=(size_t)ROWS*width) return;
    std::vector<float> ef((size_t)ROWS*width); for(size_t i=0;i<ef.size();++i) ef[i]=b2f(e[i]);
    for(int r=0;r<2;++r) std::printf("    %-8s row%d cos=%+.6f |oracle|=%.3f |engine|=%.3f ratio=%.3f\n",
        tag,r,cosine(&ours[(size_t)r*width],&ef[(size_t)r*width],width),
        l2(&ours[(size_t)r*width],width),l2(&ef[(size_t)r*width],width),
        l2(&ef[(size_t)r*width],width)>0?l2(&ours[(size_t)r*width],width)/l2(&ef[(size_t)r*width],width):-1.0);
    std::fflush(stdout);
}
static void cmpf(const char* tag,const std::vector<float>& ours,int width,int stride,int off){
    if(!DF) return; std::vector<float> ef((size_t)ROWS*width);
    if(std::fread(ef.data(),4,(size_t)ROWS*width,DF)!=(size_t)ROWS*width) return;
    for(int r=0;r<2;++r){ const float* o=&ours[(size_t)r*stride+off];
        std::printf("    %-8s row%d cos=%+.6f |oracle|=%.3f |engine|=%.3f ratio=%.3f\n",
            tag,r,cosine(o,&ef[(size_t)r*width],width),l2(o,width),l2(&ef[(size_t)r*width],width),
            l2(&ef[(size_t)r*width],width)>0?l2(o,width)/l2(&ef[(size_t)r*width],width):-1.0); }
    std::fflush(stdout);
}
int main(int argc,char** argv){
    const char* ck = argc>1?argv[1]:"/data/qwen38-27b/dflash-aligned-v5/model.safetensors";
    const char* dp = argc>2?argv[2]:"/tmp/df2_block_dump.bin";
    int pos = argc>3?atoi(argv[3]):0;
    for(int i=4;i<argc;++i){ if(!strcmp(argv[i],"--i8w")) I8W=1; else if(!strcmp(argv[i],"--i8a")) I8A=1; }
    FP=std::fopen(ck,"rb"); if(!FP){ std::fprintf(stderr,"open %s\n",ck); return 2; }
    unsigned long long hl=0; std::fread(&hl,8,1,FP); HDR.resize(hl); std::fread(HDR.data(),1,hl,FP); BASE=8+(long long)hl;
    FILE* D=std::fopen(dp,"rb"); DF=D; if(!D){ std::fprintf(stderr,"open %s (run with Q27_DFLASH2_DUMP=1 first)\n",dp); return 2; }
    std::vector<unsigned short> e(2*HID); std::fread(e.data(),2,2*HID,D);
    std::vector<float> hidden((size_t)ROWS*HID);
    for(int i=0;i<HID;++i) hidden[i]=b2f(e[i]);
    for(int r=1;r<ROWS;++r) for(int i=0;i<HID;++i) hidden[(size_t)r*HID+i]=b2f(e[HID+i]);
    std::printf("ORACLE ckpt=%s pos=%d rows=%d (NO CONTEXT arm)\n",ck,pos,ROWS);
    std::vector<float> nrm,dyn,cv,q,k,v,att,mix,gt,up;
    for(int l=0;l<NL;++l){
        char n[128];
        snprintf(n,sizeof n,"layers.%d.input_layernorm.weight",l);            Ten inw=load(n);
        snprintf(n,sizeof n,"layers.%d.attention_conv.kernel_projection.weight",l); Ten acp=load(n);
        snprintf(n,sizeof n,"layers.%d.attention_conv.base_kernel",l);         Ten acb=load(n);
        snprintf(n,sizeof n,"layers.%d.self_attn.q_proj.weight",l);            Ten wq=load(n);
        snprintf(n,sizeof n,"layers.%d.self_attn.k_proj.weight",l);            Ten wk=load(n);
        snprintf(n,sizeof n,"layers.%d.self_attn.v_proj.weight",l);            Ten wv=load(n);
        snprintf(n,sizeof n,"layers.%d.self_attn.q_norm.weight",l);            Ten qn=load(n);
        snprintf(n,sizeof n,"layers.%d.self_attn.k_norm.weight",l);            Ten kn=load(n);
        snprintf(n,sizeof n,"layers.%d.self_attn.o_proj.weight",l);            Ten wo=load(n);
        snprintf(n,sizeof n,"layers.%d.post_attention_layernorm.weight",l);    Ten pnw=load(n);
        snprintf(n,sizeof n,"layers.%d.mlp_conv.kernel_projection.weight",l);  Ten mcp=load(n);
        snprintf(n,sizeof n,"layers.%d.mlp_conv.base_kernel",l);               Ten mcb=load(n);
        snprintf(n,sizeof n,"layers.%d.mlp.gate_proj.weight",l);               Ten wg=load(n);
        snprintf(n,sizeof n,"layers.%d.mlp.up_proj.weight",l);                 Ten wu=load(n);
        snprintf(n,sizeof n,"layers.%d.mlp.down_proj.weight",l);               Ten wd=load(n);
        rmsnorm(hidden,inw,nrm,ROWS,HID);
        if(l==0) cmp("norm",nrm,HID);
        gemm(nrm,acp,dyn,ROWS,DYN,HID);
        dconv(nrm,dyn,acb,cv,0);
        if(l==0) cmp("conv",cv,HID);
        gemm(cv,wq,q,ROWS,QH*HD,HID); gemm(cv,wk,k,ROWS,KVH*HD,HID); gemm(cv,wv,v,ROWS,KVH*HD,HID);
        prepqk(q,qn,QH,pos); prepqk(k,kn,KVH,pos);
        att.assign((size_t)ROWS*QH*HD,0.f);
        for(int row=0;row<ROWS;++row) for(int h=0;h<QH;++h){ int kh=h/(QH/KVH);
            float mx=-1e30f,den=0.f; std::vector<float> acc(HD,0.f);
            for(int nr=0;nr<ROWS;++nr){ const float* kp=&k[((size_t)nr*KVH+kh)*HD]; const float* qp=&q[((size_t)row*QH+h)*HD];
                float sc=0.f; for(int d=0;d<HD;++d) sc+=qp[d]*kp[d]; sc*=0.08838834764831845f;
                float nm=std::fmax(mx,sc), pf=std::exp(mx-nm), cf=std::exp(sc-nm);
                const float* vp=&v[((size_t)nr*KVH+kh)*HD];
                for(int d=0;d<HD;++d) acc[d]=acc[d]*pf+cf*vp[d];
                den=den*pf+cf; mx=nm; }
            for(int d=0;d<HD;++d) att[((size_t)row*QH+h)*HD+d]=acc[d]/den; }
        if(l==0) cmp("attn",att,QH*HD);
        gemm(att,wo,mix,ROWS,HID,QH*HD);
        if(l==0) cmp("mixer",mix,HID);
        dconv(mix,dyn,acb,nrm,1);
        for(size_t i=0;i<hidden.size();++i) hidden[i]+=nrm[i];
        rmsnorm(hidden,pnw,nrm,ROWS,HID);
        gemm(nrm,mcp,dyn,ROWS,DYN,HID);
        dconv(nrm,dyn,mcb,cv,0);
        gemm(cv,wg,gt,ROWS,INTER,HID); gemm(cv,wu,up,ROWS,INTER,HID);
        if(l==0){ cmpf("gate",gt,4352,INTER,0); cmpf("up",up,4352,INTER,0); }
        for(size_t i=0;i<gt.size();++i){ float x=gt[i]; gt[i]=(x/(1.0f+std::exp(-x)))*up[i]; }
        gemm(gt,wd,mix,ROWS,HID,INTER);
        if(l==0) cmp("mlp_out",mix,HID);
        dconv(mix,dyn,mcb,nrm,1);
        for(size_t i=0;i<hidden.size();++i) hidden[i]+=nrm[i];
        // compare against the engine's residual after this layer
        std::vector<unsigned short> eng((size_t)ROWS*HID);
        if(std::fread(eng.data(),2,(size_t)ROWS*HID,D)==(size_t)ROWS*HID){
            std::vector<float> ef((size_t)ROWS*HID); for(size_t i=0;i<ef.size();++i) ef[i]=b2f(eng[i]);
            for(int r=0;r<2;++r) std::printf("  L%d row%d  cos=%+.6f  |oracle|=%.3f |engine|=%.3f  ratio=%.3f\n",
                l,r,cosine(&hidden[(size_t)r*HID],&ef[(size_t)r*HID],HID),
                l2(&hidden[(size_t)r*HID],HID),l2(&ef[(size_t)r*HID],HID),
                l2(&ef[(size_t)r*HID],HID)>0?l2(&hidden[(size_t)r*HID],HID)/l2(&ef[(size_t)r*HID],HID):-1.0);
        }
        std::fflush(stdout);
    }
    Ten fn=load("norm.weight");
    rmsnorm(hidden,fn,nrm,ROWS,HID);
    std::vector<unsigned short> eng((size_t)ROWS*HID);
    if(std::fread(eng.data(),2,(size_t)ROWS*HID,D)==(size_t)ROWS*HID){
        std::vector<float> ef((size_t)ROWS*HID); for(size_t i=0;i<ef.size();++i) ef[i]=b2f(eng[i]);
        for(int r=0;r<2;++r) std::printf("  FINAL row%d cos=%+.6f  |oracle|=%.3f |engine|=%.3f\n",
            r,cosine(&nrm[(size_t)r*HID],&ef[(size_t)r*HID],HID),l2(&nrm[(size_t)r*HID],HID),l2(&ef[(size_t)r*HID],HID));
    }
    std::fclose(D); std::fclose(FP);
    return 0;
}
