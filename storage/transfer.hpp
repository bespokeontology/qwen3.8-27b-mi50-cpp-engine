#pragma once
// Native bounded streaming shared by the HIP and ACL loaders. No model math.
#include "glmflash_store.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
namespace storage {
inline uint64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline void check(bool ok,const std::string& msg) { if(!ok)throw std::runtime_error(msg); }
inline double cpu(const rusage& r) { return r.ru_utime.tv_sec+r.ru_utime.tv_usec/1e6+r.ru_stime.tv_sec+r.ru_stime.tv_usec/1e6; }
inline std::vector<int> drive_map(const GfStore& store) {
    const char* path=getenv("GLM_STORAGE_MAP");check(path,"GLM_STORAGE_MAP required");
    std::ifstream f(path);check(f.good(),"open drive map");std::vector<int> drives(store.n_shards);
    dev_t devices[2]{};bool known[2]{};
    for(int i=0;i<store.n_shards;++i) { int slot,d;check(bool(f>>slot>>d)&&slot==i&&d>=0&&d<2,"drive map entry");
        struct stat st{};check(fstat(store.shard_fds[i],&st)==0,"shard fstat");
        check(!known[d]||devices[d]==st.st_dev,"drive maps multiple filesystems");devices[d]=st.st_dev;known[d]=true;drives[i]=d;
    }
    check(!known[0]||!known[1]||devices[0]!=devices[1],"two drive IDs map to same filesystem");
    std::string extra;check(!(f>>extra),"extra drive map entries");return drives;
}
struct Read { int drive,shard;uint64_t offset,bytes,start,end,enqueue,complete_observed;double copy_ms; };
struct DiskSample { uint64_t at,sector[2]; };
inline DiskSample disks() { DiskSample s{};s.at=now_ns();for(int d=0;d<2;++d) {std::ifstream f("/sys/block/nvme"+std::to_string(d)+"n1/stat");uint64_t ignored;check(bool(f>>ignored>>ignored>>s.sector[d]),"diskstats");}return s; }
template<class Stream> struct Extent {
    const GfEntry* entry;void* destination;
    // Optional conversion is enqueued after all chunks, on this extent's stream.
    std::function<void(Stream)> after;
};
// Backend owns one stream, two pinned buffers and two timing-event pairs per reader.
// A completion event prevents a buffer being overwritten while DMA still uses it.
template<class Backend> void transfer(GfStore& store,const std::vector<Extent<typename Backend::Stream>>& extents,typename Backend::Context context) {
    const auto drives=drive_map(store);const bool serial=getenv("GLM_STORAGE_SERIAL")!=nullptr;
    const char* prefix=getenv("GLM_STORAGE_TRACE");check(prefix&&*prefix,"GLM_STORAGE_TRACE required");
    const bool coalesce=getenv("GLM_STORAGE_COALESCE")!=nullptr;
    const size_t half=(coalesce?16ull:4ull)<<20;const int readers=serial?1:2;
    std::vector<size_t> work[2];uint64_t expected[2]{};
    for(size_t i=0;i<extents.size();++i) {const auto& r=extents[i].entry->rec;check(r.file_idx<uint32_t(store.n_shards)&&r.data_off<=store.shard_sizes[r.file_idx]&&r.nbytes<=store.shard_sizes[r.file_idx]-r.data_off,"extent bounds");const int d=drives[r.file_idx];work[serial?0:d].push_back(i);expected[d]+=r.nbytes;}
    for(auto& w:work)std::sort(w.begin(),w.end(),[&](size_t a,size_t b){auto& x=extents[a].entry->rec;auto& y=extents[b].entry->rec;return x.file_idx!=y.file_idx?x.file_idx<y.file_idx:x.data_off<y.data_off;});
    uint64_t direct_bytes[2]{},copied[2]{};
    std::vector<Read> traces[2];std::thread threads[2];std::exception_ptr errors[2];
    std::mutex mutex;std::condition_variable cv;int ready=0;bool go=false;std::atomic<bool> stop{false};
    for(int t=0;t<readers;++t)threads[t]=std::thread([&,t]{bool announced=false;try {
        Backend backend(context,half);bool live[2]{};size_t prior[2]{};int slot=0;
        auto finish=[&](int h){if(live[h]) {backend.wait(h);auto& op=traces[t][prior[h]];op.complete_observed=now_ns();op.copy_ms=backend.milliseconds(h);live[h]=false;}};
        {std::unique_lock<std::mutex> lock(mutex);++ready;announced=true;cv.notify_all();cv.wait(lock,[&]{return go;});}
        struct Fds {std::vector<int> fd;~Fds(){for(int f:fd)if(f>=0)close(f);}} direct;
        direct.fd.assign(store.n_shards,-1);
        size_t current=0;uint64_t item_done=0;
        struct Fragment {size_t index;uint64_t done,bytes;};
        while(current<work[t].size()&&!stop) {
            finish(slot);
            const auto& first=extents[work[t][current]].entry->rec;
            const uint64_t start_offset=first.data_off+item_done,limit=start_offset+half;
            uint64_t end_offset=start_offset;std::vector<Fragment> fragments;
            size_t next=current;uint64_t next_done=item_done;
            while(next<work[t].size()) {
                const auto index=work[t][next];const auto& r=extents[index].entry->rec;
                const uint64_t from=r.data_off+next_done;
                if(r.file_idx!=first.file_idx||from>=limit||from<end_offset||from-end_offset>(64ull<<10))break;
                const uint64_t n=std::min<uint64_t>(r.nbytes-next_done,limit-from);
                fragments.push_back({index,next_done,n});end_offset=from+n;next_done+=n;
                if(next_done==r.nbytes){++next;next_done=0;}else break;
                if(!coalesce)break;
            }
            check(!fragments.empty(),"nonempty coalesced extent");
            const size_t n=end_offset-start_offset;char* pin=backend.buffer(slot);
            int fd=store.shard_fds[first.file_idx];bool use_direct=false;
            if(Backend::direct_reads&&coalesce&&!(start_offset%4096)&&!(n%4096)&&!(reinterpret_cast<uintptr_t>(pin)%4096)) {
                auto& dfd=direct.fd[first.file_idx];if(dfd<0)dfd=open(store.shard_paths[first.file_idx],O_RDONLY|O_DIRECT|O_CLOEXEC);
                check(dfd>=0,"open aligned O_DIRECT extent");fd=dfd;use_direct=true;
            }
            const uint64_t start=now_ns();size_t got=0;
            while(got<n) {ssize_t k=pread(fd,pin+got,n-got,start_offset+got);if(k<0&&errno==EINTR)continue;check(k>0,"checkpoint pread failed, errno="+std::to_string(errno));got+=k;}
            const uint64_t end=now_ns();backend.begin_copy(slot);
            for(const auto& part:fragments) {
                const auto& item=extents[part.index];const auto& r=item.entry->rec;
                backend.append_copy(static_cast<char*>(item.destination)+part.done,pin+r.data_off+part.done-start_offset,part.bytes);
                copied[t]+=part.bytes;
                if(part.done+part.bytes==r.nbytes&&item.after)item.after(backend.stream());
            }
            backend.end_copy(slot);
            if(use_direct)direct_bytes[t]+=n;
            traces[t].push_back({drives[first.file_idx],int(first.file_idx),start_offset,n,start,end,now_ns(),0,0});prior[slot]=traces[t].size()-1;live[slot]=true;slot^=1;
            gf_store_drop_range(&store,first.file_idx,start_offset,n);
            current=next;item_done=next_done;
        }
        finish(0);finish(1);backend.sync();
    }catch(...) {errors[t]=std::current_exception();stop=true;{std::lock_guard<std::mutex> lock(mutex);if(!announced)++ready;}cv.notify_all();}});
    {std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return ready==readers;});}
    std::string barrier_error;
    if(const char* root=getenv("GLM_STORAGE_BARRIER")) {
        try {
            const char* owner=getenv("GLM_STORAGE_OWNER");check(owner,"storage owner for barrier");
            const std::string who=owner;check(who=="amd"||who=="npu","barrier owner");
            const std::string ready=std::string(root)+"/"+who+".ready";
            int fd=open(ready.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);check(fd>=0,"new owner readiness file");close(fd);
            const auto peer=std::string(root)+"/"+(who=="amd"?"npu":"amd")+".ready";
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(120);
            while(access(peer.c_str(),F_OK)!=0) {check(std::chrono::steady_clock::now()<deadline,"storage peer readiness timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));}
        }catch(const std::exception& e){barrier_error=e.what();stop=true;}
    }
    std::vector<DiskSample> samples;std::atomic<bool> sampling{true};std::exception_ptr sampling_error;
    std::thread sampler([&]{try {while(sampling) {samples.push_back(disks());std::this_thread::sleep_for(std::chrono::milliseconds(10));}}catch(...) {sampling_error=std::current_exception();}});
    rusage before{},after{};getrusage(RUSAGE_SELF,&before);const uint64_t begin=now_ns();
    {std::lock_guard<std::mutex> lock(mutex);go=true;}cv.notify_all();for(int t=0;t<readers;++t)threads[t].join();
    const uint64_t end=now_ns();getrusage(RUSAGE_SELF,&after);sampling=false;sampler.join();
    for(auto& e:errors)if(e)std::rethrow_exception(e);
    if(sampling_error)std::rethrow_exception(sampling_error);
    check(barrier_error.empty(),barrier_error);
    const double seconds=(end-begin)/1e9;uint64_t bytes[2]{};double read_s[2]{},copy_ms[2]{};
    std::ofstream trace(std::string(prefix)+"-reads.tsv");trace<<"drive\tshard\toffset\tbytes\tread_start_ns\tread_end_ns\tenqueue_ns\tcompletion_observed_ns\tdevice_event_ms\n";
    for(auto& rows:traces)for(auto& r:rows) {bytes[r.drive]+=r.bytes;read_s[r.drive]+=(r.end-r.start)/1e9;copy_ms[r.drive]+=r.copy_ms;trace<<r.drive<<'\t'<<r.shard<<'\t'<<r.offset<<'\t'<<r.bytes<<'\t'<<r.start<<'\t'<<r.end<<'\t'<<r.enqueue<<'\t'<<r.complete_observed<<'\t'<<r.copy_ms<<'\n';}
    check(copied[0]+copied[1]==expected[0]+expected[1],"complete transfer bytes");
    std::ofstream diskfile(std::string(prefix)+"-disks.tsv");diskfile<<"time_ns\tnvme0_read_sectors\tnvme1_read_sectors\n";int simultaneous=0;
    for(size_t i=0;i<samples.size();++i) {auto& s=samples[i];diskfile<<s.at<<'\t'<<s.sector[0]<<'\t'<<s.sector[1]<<'\n';if(i&&s.sector[0]>samples[i-1].sector[0]&&s.sector[1]>samples[i-1].sector[1])++simultaneous;}
    std::ofstream result(std::string(prefix)+".json");result<<std::setprecision(12)<<"{\"reader_threads\":"<<readers<<",\"pinned_bytes\":"<<readers*2*half<<",\"bytes\":["<<bytes[0]<<','<<bytes[1]<<"],\"seconds\":"<<seconds<<",\"GB_per_second\":["<<bytes[0]/seconds/1e9<<','<<bytes[1]/seconds/1e9<<"],\"combined_GB_per_second\":"<<(bytes[0]+bytes[1])/seconds/1e9<<",\"payload_bytes\":"<<(expected[0]+expected[1])<<",\"O_DIRECT_bytes\":"<<(direct_bytes[0]+direct_bytes[1])<<",\"cpu_cores\":"<<(cpu(after)-cpu(before))/seconds<<",\"rss_bytes\":"<<uint64_t(after.ru_maxrss)*1024<<",\"pread_seconds\":["<<read_s[0]<<','<<read_s[1]<<"],\"device_event_ms\":["<<copy_ms[0]<<','<<copy_ms[1]<<"],\"both_disks_progress_samples\":"<<simultaneous<<",\"explicit_payload_memcpy_bytes\":0,\"timing_note\":\"copy events include queued conversion; host observations do not establish exact disk-DMA overlap\",\"passed\":true}\n";
    check(trace.good()&&diskfile.good()&&result.good(),"write storage receipts");
}
} // namespace storage
