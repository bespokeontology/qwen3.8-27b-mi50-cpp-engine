#pragma once
#include "hip_transfer.hpp"
#include <sstream>
#include <memory>
namespace q27storage {
struct Piece { size_t seq; int drive, device; uint64_t offset, bytes, dst; std::string tensor, precision; };
struct Recipe { size_t seq; int device; uint64_t bytes; std::string tensor, precision; std::vector<Piece> pieces; };
struct Pending { GfEntry entry; void* dst; int device; };
class Loader {
    std::string export_root, root, trace_root, tensor, precision;
    std::ofstream index;
    int output = -1, flushes = 0;
    uint64_t written = 0, begin = storage::now_ns();
    size_t sequence = 0;
    std::vector<Recipe> recipe;
    std::vector<Pending> pending;
    std::vector<std::string> files;
    GfStore store;
    void require(bool v,const std::string& msg) {storage::check(v,"Q27 storage: "+msg);}
public:
    Loader() {
        if(const char* p=getenv("Q27_STORAGE_EXPORT"))export_root=p;
        if(const char* p=getenv("Q27_STORAGE_LAYOUT"))root=p;
        require(root.empty()||export_root.empty(),"export and replay are exclusive");
        if(!root.empty()||!export_root.empty()) {
            std::fprintf(stderr,"Q27_STORAGE origin_monotonic_ns=%llu\n",(unsigned long long)begin);
            const char* fp8=getenv("Q27_FP8_MLP");
            require(!(fp8&&*fp8),"this recipe adapter currently requires native NVFP4 MLP; FP8 sideload is not covered");
            require(!getenv("Q27_EXL3_DIR"),"EXL3 sideload is not covered");
        }
        if(!export_root.empty()) {
            output=open((export_root+"/native.bin").c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
            require(output>=0,"new export payload");
            index.open(export_root+"/recipe.tsv");require(index.good(),"export recipe");
            index<<"sequence\tlogical_gpu\tphysical_gpu\ttensor\tprecision\toffset\tbytes\n";
        }
        if(root.empty())return;
        const char* tr=getenv("Q27_STORAGE_TRACE");require(tr&&*tr,"trace directory required");trace_root=tr;
        std::ifstream paths(root+"/files.txt");std::string line;
        while(std::getline(paths,line)) {require(!line.empty(),"empty file path");files.push_back(line);}
        require(!files.empty()&&files.size()<=2,"one or two physical drive files");
        store.n_shards=files.size();
        for(auto& f:files) {
            int fd=open(f.c_str(),O_RDONLY|O_CLOEXEC);require(fd>=0,"open payload "+f);
            struct stat st{};require(fstat(fd,&st)==0,"payload stat");
            store.shard_fds.push_back(fd);store.shard_sizes.push_back(st.st_size);store.shard_paths.push_back(f.c_str());
        }
        setenv("GLM_STORAGE_MAP",(root+"/drives.txt").c_str(),1);
        setenv("GLM_STORAGE_COALESCE","1",1);
        unsetenv("GLM_STORAGE_BARRIER");unsetenv("GLM_STORAGE_SERIAL");
        if(files.size()==1)setenv("GLM_STORAGE_SERIAL","1",1);
        std::ifstream in(root+"/manifest.tsv");require(in.good(),"manifest");std::getline(in,line);
        while(std::getline(in,line)) {
            std::istringstream row(line);Piece p{};int physical;std::string file;
            require(bool(row>>p.seq>>p.device>>physical>>p.tensor>>p.precision>>p.drive>>file>>p.offset>>p.bytes>>p.dst),"manifest row");
            require(p.seq<=recipe.size()&&p.drive>=0&&p.drive<store.n_shards&&p.bytes>0,"manifest index");
            require(p.device>=0&&p.device<4&&physical==(p.device<2?0:p.device-1),"manifest GPU ownership");
            require(file==files[p.drive],"manifest file binding");
            if(p.seq==recipe.size())recipe.push_back({p.seq,p.device,0,p.tensor,p.precision,{}});
            auto& r=recipe.back();require(r.seq==p.seq&&r.device==p.device&&r.tensor==p.tensor&&r.precision==p.precision&&r.bytes==p.dst,"manifest contiguous recipe");
            r.bytes+=p.bytes;r.pieces.push_back(p);
        }
        require(!recipe.empty(),"empty recipe");
        std::fprintf(stderr,"Q27_STORAGE native replay: %zu uploads, %zu physical drives\n",recipe.size(),files.size());
    }
    bool replay()const{return !root.empty();}
    bool enabled()const{return replay()||output>=0;}
    void tag(const std::string& name,const std::string& dtype) {tensor=name;precision=dtype;}
    bool submit(void* dst,const unsigned char* src,size_t bytes) {
        if(!enabled())return false;
        int device=-1;storage::HipTransfer::ck(hipGetDevice(&device));
        require(!tensor.empty()&&!precision.empty(),"native upload missing tensor metadata");
        if(output>=0) {
            const uint64_t off=(written+4095)&~uint64_t(4095);size_t n=0;
            while(n<bytes) {ssize_t k=pwrite(output,src+n,bytes-n,off+n);if(k<0&&errno==EINTR)continue;require(k>0,"export write");n+=k;}
            index<<sequence<<'\t'<<device<<'\t'<<(device<2?0:device-1)<<'\t'<<tensor<<'\t'<<precision<<'\t'<<off<<'\t'<<bytes<<'\n';
            written=off+bytes;require(ftruncate(output,(written+4095)&~uint64_t(4095))==0,"aligned export length");
        } else {
            require(sequence<recipe.size(),"unexpected upload");const auto& r=recipe[sequence];
            require(r.device==device&&r.bytes==bytes&&r.tensor==tensor&&r.precision==precision,
                    "recipe mismatch at "+std::to_string(sequence)+" "+tensor);
            for(const auto& p:r.pieces) pending.push_back({{{uint32_t(p.drive),p.offset,p.bytes}},static_cast<char*>(dst)+p.dst,device});
        }
        ++sequence;return replay();
    }
    void flush() {
        if(!replay()||pending.empty())return;
        const int dev=pending.front().device;
        std::vector<storage::Extent<hipStream_t>> extents;extents.reserve(pending.size());
        for(auto& p:pending) {require(p.device==dev,"mixed owners at flush");extents.push_back({&p.entry,p.dst,{}});}
        const std::string prefix=trace_root+"/transfer-"+std::to_string(flushes++)+"-logical"+std::to_string(dev);
        setenv("GLM_STORAGE_TRACE",prefix.c_str(),1);
        storage::transfer<storage::HipTransfer>(store,extents,dev);
        storage::HipTransfer::ck(hipSetDevice(dev));
        std::fprintf(stderr,"Q27_STORAGE payload_ready logical=%d physical=%d elapsed_ms=%.3f uploads=%zu\n",dev,dev<2?0:dev-1,(storage::now_ns()-begin)/1e6,sequence);
        pending.clear();
    }
    void owner_ready(int dev,const char* stage) {
        if(replay())std::fprintf(stderr,"Q27_STORAGE owner_ready logical=%d physical=%d elapsed_ms=%.3f stage=%s\n",dev,dev<2?0:dev-1,(storage::now_ns()-begin)/1e6,stage);
    }
    void finish() {
        flush();
        if(replay())require(sequence==recipe.size(),"incomplete upload recipe");
        if(output>=0) {require(fsync(output)==0,"export sync");close(output);output=-1;index.flush();require(index.good(),"export index");}
        if(enabled()||!export_root.empty())std::fprintf(stderr,"Q27_STORAGE native_load_complete uploads=%zu elapsed_ms=%.3f\n",sequence,(storage::now_ns()-begin)/1e6);
    }
    ~Loader(){if(output>=0)close(output);for(int fd:store.shard_fds)close(fd);}
};
inline Loader& loader(){static Loader value;return value;}
inline bool replay(){return loader().replay();}
inline void tag(const std::string& tensor,int dtype){
    const char* names[]={"UNKNOWN","BF16","F16","F32","F64","F8_E4M3","F8_E5M2","U8","I8","U16","I16","U32","I32","U64","I64","BOOL"};
    std::string precision=(dtype>=0&&dtype<16)?names[dtype]:"UNKNOWN";
    if(dtype==7&&tensor.size()>=7&&tensor.compare(tensor.size()-7,7,".weight")==0)precision="NVFP4-packed";
    loader().tag(tensor,precision);
}
inline void flush(){loader().flush();}
}
