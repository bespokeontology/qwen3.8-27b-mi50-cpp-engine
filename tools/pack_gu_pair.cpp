// C++17 offline, lossless gate/up nibble-pair packer. Storage ABI: q27.h.
// One 16-weight group uses 16 payload bytes and TWO independent scale bytes.
// This does not reduce weight bytes, and is not a GPU performance result.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using Bytes=std::vector<uint8_t>;
struct Packed { Bytes weights,scales; };
static size_t groups(size_t rows,size_t k) {
    if(!rows || !k || k%16 || rows>std::numeric_limits<size_t>::max()/k)
        throw std::invalid_argument("positive rows and K divisible by 16 required, without overflow");
    return rows*(k/16);
}
static Packed pack(const Bytes& gate,const Bytes& up,const Bytes& gs,const Bytes& us,size_t rows,size_t k) {
    const size_t ng=groups(rows,k);
    if(gate.size()!=ng*8 || up.size()!=ng*8 || gs.size()!=ng || us.size()!=ng)
        throw std::invalid_argument("input sizes do not match rows and K");
    Packed out{Bytes(ng*16),Bytes(ng*2)};
    for(size_t i=0;i<ng*8;++i) {
        out.weights[2*i]=(gate[i]&15u)|((up[i]&15u)<<4);
        out.weights[2*i+1]=(gate[i]>>4)|(up[i]&240u);
    }
    for(size_t i=0;i<ng;++i){out.scales[2*i]=gs[i];out.scales[2*i+1]=us[i];}
    return out;
}
static std::array<Bytes,4> unpack(const Packed& in,size_t rows,size_t k) {
    const size_t ng=groups(rows,k);
    if(in.weights.size()!=ng*16 || in.scales.size()!=ng*2)throw std::invalid_argument("packed sizes mismatch");
    std::array<Bytes,4> out{Bytes(ng*8),Bytes(ng*8),Bytes(ng),Bytes(ng)};
    for(size_t i=0;i<ng*8;++i) {
        out[0][i]=(in.weights[2*i]&15u)|((in.weights[2*i+1]&15u)<<4);
        out[1][i]=(in.weights[2*i]>>4)|(in.weights[2*i+1]&240u);
    }
    for(size_t i=0;i<ng;++i){out[2][i]=in.scales[2*i];out[3][i]=in.scales[2*i+1];}
    return out;
}
static Bytes read(const std::string& path,size_t size) {
    if(std::filesystem::file_size(path)!=size)throw std::invalid_argument("wrong size: "+path);
    Bytes b(size);std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(b.data()),size))throw std::runtime_error("read failed: "+path);
    return b;
}
static void write(const std::string& path,const Bytes& b) {
    if(std::filesystem::exists(path))throw std::runtime_error("refusing to overwrite: "+path);
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(b.data()),b.size());
    if(!f)throw std::runtime_error("write failed: "+path);
}
static void selftest() {
    // Enumerate every pair of packed gate/up bytes, including both signs and -0.
    size_t cases=0;
    for(unsigned g=0;g<256;++g)for(unsigned u=0;u<256;++u) {
        Bytes a(8,g),b(8,u),sg(1,g),su(1,u);auto p=pack(a,b,sg,su,1,16);
        for(int k=0;k<16;++k) {
            const unsigned expected=((g>>(4*(k%2)))&15u)|(((u>>(4*(k%2)))&15u)<<4);
            if(p.weights[k]!=expected)throw std::runtime_error("element oracle failed");
        }
        if(unpack(p,1,16)!=std::array<Bytes,4>{a,b,sg,su})throw std::runtime_error("round trip failed");
        if(p.scales[0]!=g || p.scales[1]!=u)throw std::runtime_error("independent scales changed");
        ++cases;
    }
    for(size_t rows:{size_t(1),size_t(3),size_t(57)}) {
        const size_t ng=groups(rows,5120);Bytes a(ng*8),b(a.size()),sg(ng),su(ng);
        uint32_t seed=19;auto next=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return uint8_t(seed);};
        for(auto& x:a)x=next();
        for(auto& x:b)x=next();
        for(auto& x:sg)x=next();
        for(auto& x:su)x=next();
        if(unpack(pack(a,b,sg,su,rows,5120),rows,5120)!=std::array<Bytes,4>{a,b,sg,su})throw std::runtime_error("row boundary failed");
        ++cases;
    }
    int rejected=0;
    for(auto shape:std::array<std::array<size_t,2>,3>{{{0,16},{1,15},{std::numeric_limits<size_t>::max(),16}}}) {
        try{groups(shape[0],shape[1]);}catch(const std::invalid_argument&){++rejected;}
    }
    try{pack(Bytes(7),Bytes(8),Bytes(1),Bytes(1),1,16);}catch(const std::invalid_argument&){++rejected;}
    if(rejected!=4)throw std::runtime_error("invalid input accepted");
    std::printf("GU_PAIR C++ packer PASS cases=%zu refusal_cases=%d payload=16B scales=2B per group\n",cases,rejected);
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--selftest"){selftest();return 0;}
        if(argc!=9 || std::string(argv[1])!="--pack") {
            std::fprintf(stderr,"usage: %s --selftest | --pack ROWS K GATE UP GATE_SCALES UP_SCALES OUT_PREFIX\n",argv[0]);return 2;
        }
        const size_t rows=std::stoull(argv[2]),k=std::stoull(argv[3]),ng=groups(rows,k);
        const std::string wp=std::string(argv[8])+".weights",sp=std::string(argv[8])+".scales";
        if(std::filesystem::exists(wp)||std::filesystem::exists(sp))throw std::runtime_error("output exists");
        auto p=pack(read(argv[4],ng*8),read(argv[5],ng*8),read(argv[6],ng),read(argv[7],ng),rows,k);
        write(wp,p.weights);write(sp,p.scales);
        std::printf("packed %zu groups; %zu weight bytes and %zu scale bytes\n",ng,p.weights.size(),p.scales.size());
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
    return 0;
}
