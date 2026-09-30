#include <chrono>
// Preserve the frozen Q27 binary's four logical shards on three physical MI50s.
// The checkpoint loader and execution code see the SAME logical namespace.
// Map peer queries and copies to the physical devices; same-device copies use D2D.
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace {
// Q27_SHARD_MAP="0,0,1,2" (default, four logical workers on three cards) or "0,1,2" (Q27_TP3: one
// worker per card). Unlisted logical slots keep the default so three-worker and four-worker
// engines share this adapter.
static int mapping[4] = {0, 0, 1, 2};
static void mapping_from_env() {
    static bool done = false; if (done) return; done = true;
    const char* v = getenv("Q27_SHARD_MAP"); if (!v || !*v) return;
    int k = 0; const char* p = v;
    while (*p && k < 4) { char* e = 0; long x = strtol(p, &e, 10); if (e == p) break; if (x < 0 || x > 2) break; mapping[k++] = (int)x; p = e; while (*p == ',' || *p == ' ') ++p; }
}
thread_local int logical = 0;
std::once_flag once;
std::atomic<unsigned long long> copies{0}, copy_bytes{0};
template<class T> T sym(const char* name) {
    void* p = dlsym(RTLD_NEXT, name);
    if (!p) { std::fprintf(stderr, "Q27_SHARD_MAP missing %s\n", name); std::abort(); }
    return reinterpret_cast<T>(p);
}
void init() {
    std::call_once(once, [] {
        auto count = sym<hipError_t(*)(int*)>("hipGetDeviceCount");
        int n = 0;
        mapping_from_env();
        if (count(&n) != hipSuccess || n != 3) {
            std::fprintf(stderr, "Q27_SHARD_MAP requires exactly three visible physical GPUs; got %d\n", n);
            std::abort();
        }
        std::fprintf(stderr, "Q27_SHARD_MAP logical 0,1,2,3 -> physical %d,%d,%d,%d; mapped physical peer access\n", mapping[0], mapping[1], mapping[2], mapping[3]);
        std::atexit([] { std::fprintf(stderr, "Q27_SHARD_MAP peer_copies=%llu peer_copy_bytes=%llu\n", copies.load(), copy_bytes.load()); });
    });
}
bool valid(int d) { return d >= 0 && d < 4; }
}

namespace {
enum TransferKind { PEER_COPY, PEER_2D, SAME_DEVICE, HOST_TO_DEVICE, DEVICE_TO_HOST, HOST_TO_HOST, UNKNOWN_COPY, HOST_STAGED, STREAM_WAIT, XFER_COUNT };
std::atomic<bool> measuring{false};
std::atomic<unsigned long long> meter_bytes[XFER_COUNT],meter_calls[XFER_COUNT];
struct CopyEvent { hipEvent_t begin,end; int logical_device,kind; };
std::mutex meter_mu;std::vector<CopyEvent> meter_events;
struct Allocation { size_t bytes;int physical; };
std::mutex alloc_mu;std::unordered_map<void*,Allocation> allocations;
std::atomic<bool> meter_timing{false};
bool timed_copies(){return meter_timing.load(std::memory_order_relaxed);}
bool alloc_meter(){static bool v=getenv("Q27_MEM_LEDGER")&&atoi(getenv("Q27_MEM_LEDGER"));return v;}
struct CopyMeter {
 int kind;size_t bytes;bool active;hipStream_t stream;CopyEvent event{};
 CopyMeter(int k,size_t b,hipStream_t st):kind(k),bytes(b),active(measuring.load(std::memory_order_relaxed)),stream(st){
  if(active&&timed_copies()&&(kind==PEER_COPY||kind==PEER_2D||kind==SAME_DEVICE||kind==STREAM_WAIT)){
   event.logical_device=logical;event.kind=k;
   if(hipEventCreate(&event.begin)!=hipSuccess||hipEventCreate(&event.end)!=hipSuccess)std::abort();
   if(hipEventRecord(event.begin,stream)!=hipSuccess)std::abort();
  }
 }
 void finish(hipError_t status){
  if(!active)return;
  if(status!=hipSuccess)std::abort();
  meter_bytes[kind]+=bytes;++meter_calls[kind];
  if(event.begin){if(hipEventRecord(event.end,stream)!=hipSuccess)std::abort();std::lock_guard<std::mutex>lock(meter_mu);meter_events.push_back(event);}
 }
};
std::atomic<bool> stage_enabled{false};
bool host_staging(){return stage_enabled.load(std::memory_order_relaxed);}
bool control_flag(const char* setting,const char* path_setting){
 bool value=getenv(setting)&&atoi(getenv(setting));
 if(const char* path=getenv(path_setting)){FILE* f=std::fopen(path,"r");int n=-1;if(!f||std::fscanf(f,"%d",&n)!=1||(n!=0&&n!=1))std::abort();std::fclose(f);value=n;}
 return value;
}
std::atomic<unsigned long long> staged_wall_ns{0};
// Diagnostic whole-engine control. Producer waits already queued on the calling
// stream must finish first. Private nonblocking streams perform D2H then H2D;
// waiting only those streams keeps the co-resident worker's streams independent.
hipError_t stage_copy(void* dst,size_t dp,const void* src,size_t sp,size_t w,size_t h,
                     int source,int destination,hipStream_t consumer) {
 static auto set=sym<hipError_t(*)(int)>("hipSetDevice");
 static auto cp=sym<hipError_t(*)(void*,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DAsync");
 struct Stage {void* buffer=nullptr;size_t cap=0;hipStream_t stream[3]={};};
 thread_local Stage stage;
 const auto begin=std::chrono::steady_clock::now();
 auto check=[](hipError_t e){if(e!=hipSuccess){std::fprintf(stderr,"Q27_HOST_STAGE_AB error %d\n",int(e));std::abort();}};
 check(hipStreamSynchronize(consumer));
 if(stage.cap<w*h){if(stage.buffer)check(hipHostFree(stage.buffer));check(hipHostMalloc(&stage.buffer,w*h,hipHostMallocDefault));stage.cap=w*h;}
 check(set(source));if(!stage.stream[source])check(hipStreamCreateWithFlags(&stage.stream[source],hipStreamNonBlocking));
 check(cp(stage.buffer,w,src,sp,w,h,hipMemcpyDeviceToHost,stage.stream[source]));check(hipStreamSynchronize(stage.stream[source]));
 check(set(destination));if(!stage.stream[destination])check(hipStreamCreateWithFlags(&stage.stream[destination],hipStreamNonBlocking));
 check(cp(dst,dp,stage.buffer,w,w,h,hipMemcpyHostToDevice,stage.stream[destination]));check(hipStreamSynchronize(stage.stream[destination]));
 check(set(mapping[logical]));
 if(measuring.load()) {meter_bytes[DEVICE_TO_HOST]+=w*h;++meter_calls[DEVICE_TO_HOST];meter_bytes[HOST_TO_DEVICE]+=w*h;++meter_calls[HOST_TO_DEVICE];
  staged_wall_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();}
 return hipSuccess;
}
int copy_kind(const void* src,const void* dst,hipMemcpyKind kind){
 if(kind==hipMemcpyHostToDevice)return HOST_TO_DEVICE;
 if(kind==hipMemcpyDeviceToHost)return DEVICE_TO_HOST;
 if(kind==hipMemcpyHostToHost)return HOST_TO_HOST;
 hipPointerAttribute_t sa{},da{};
 static auto attr=sym<hipError_t(*)(hipPointerAttribute_t*,const void*)>("hipPointerGetAttributes");
 if(attr(&sa,src)!=hipSuccess||attr(&da,dst)!=hipSuccess)return UNKNOWN_COPY;
 return sa.device==da.device?SAME_DEVICE:PEER_2D;
}
}
extern "C" void q27_xfer_begin(int req){
 measuring.store(false);meter_timing=control_flag("Q27_XFER_TIME","Q27_XFER_CONTROL");stage_enabled=control_flag("Q27_HOST_STAGE_AB","Q27_STAGE_CONTROL");staged_wall_ns=0;for(int k=0;k<XFER_COUNT;++k){meter_bytes[k]=0;meter_calls[k]=0;}
 {std::lock_guard<std::mutex>lock(meter_mu);if(!meter_events.empty())std::abort();}
 measuring.store(true);std::fprintf(stderr,"Q27_XFER_BEGIN req=%d timing=%d\n",req,int(timed_copies()));
}
extern "C" void q27_xfer_report(int req){
 measuring.store(false);
 double elapsed[XFER_COUNT]={0};
 static auto set=sym<hipError_t(*)(int)>("hipSetDevice");
 const int caller=logical;
 {std::lock_guard<std::mutex>lock(meter_mu);for(const auto& e:meter_events){
   set(mapping[e.logical_device]);float ms=0;
   if(hipEventSynchronize(e.end)!=hipSuccess||hipEventElapsedTime(&ms,e.begin,e.end)!=hipSuccess)std::abort();
   elapsed[e.kind]+=ms;hipEventDestroy(e.begin);hipEventDestroy(e.end);
  }meter_events.clear();}
 set(mapping[caller]);
 std::fprintf(stderr,"Q27_HOST_STAGE_AB req=%d enabled=%d summed_host_wall_ms=%.6f\n",req,int(host_staging()),staged_wall_ns.load()/1e6);
 const char* names[XFER_COUNT]={"peer","peer2d","same_device","h2d","d2h","h2h","unknown","host_staged","stream_wait"};
 for(int k=0;k<XFER_COUNT;++k)std::fprintf(stderr,"Q27_XFER req=%d kind=%s bytes=%llu calls=%llu event_ms=%.6f timing=%d\n",req,names[k],meter_bytes[k].load(),meter_calls[k].load(),elapsed[k],int(timed_copies()));
}
extern "C" void q27_memory_report(const char* label){
 if(!alloc_meter())return;
 size_t sum[3]={0};size_t count[3]={0};
 {std::lock_guard<std::mutex>lock(alloc_mu);for(const auto& a:allocations){sum[a.second.physical]+=a.second.bytes;++count[a.second.physical];}}
 static auto set=sym<hipError_t(*)(int)>("hipSetDevice");const int caller=logical;
 for(int g=0;g<3;++g){set(g);size_t free=0,total=0;hipMemGetInfo(&free,&total);
  std::fprintf(stderr,"Q27_MEMORY label=%s physical=%d allocator_live=%zu allocations=%zu free=%zu total=%zu used=%zu\n",label,g,sum[g],count[g],free,total,total-free);}
 set(mapping[caller]);
}
extern "C" hipError_t hipMalloc(void** ptr,size_t bytes){
 static auto fn=sym<hipError_t(*)(void**,size_t)>("hipMalloc");const auto e=fn(ptr,bytes);
 if(e==hipSuccess&&alloc_meter()){std::lock_guard<std::mutex>lock(alloc_mu);allocations[*ptr]={bytes,mapping[logical]};}
 return e;
}
extern "C" hipError_t hipFree(void* ptr){
 static auto fn=sym<hipError_t(*)(void*)>("hipFree");const auto e=fn(ptr);
 if(e==hipSuccess&&alloc_meter()){std::lock_guard<std::mutex>lock(alloc_mu);allocations.erase(ptr);}return e;
}
extern "C" hipError_t hipMemcpyAsync(void* dst,const void* src,size_t bytes,hipMemcpyKind kind,hipStream_t stream){
 static auto fn=sym<hipError_t(*)(void*,const void*,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyAsync");
 CopyMeter m(measuring.load()?copy_kind(src,dst,kind):SAME_DEVICE,bytes,stream);const auto e=fn(dst,src,bytes,kind,stream);m.finish(e);return e;
}
extern "C" hipError_t hipMemcpy(void* dst,const void* src,size_t bytes,hipMemcpyKind kind){
 static auto fn=sym<hipError_t(*)(void*,const void*,size_t,hipMemcpyKind)>("hipMemcpy");
 CopyMeter m(measuring.load()?copy_kind(src,dst,kind):SAME_DEVICE,bytes,nullptr);const auto e=fn(dst,src,bytes,kind);m.finish(e);return e;
}
extern "C" hipError_t hipMemcpy2DAsync(void* dst,size_t dpitch,const void* src,size_t spitch,size_t width,size_t height,hipMemcpyKind kind,hipStream_t stream){
 static auto fn=sym<hipError_t(*)(void*,size_t,const void*,size_t,size_t,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpy2DAsync");
 const int k=(measuring.load()||host_staging())?copy_kind(src,dst,kind):SAME_DEVICE;
 if(host_staging()&&k==PEER_2D){hipPointerAttribute_t sa{},da{};
  static auto attr=sym<hipError_t(*)(hipPointerAttribute_t*,const void*)>("hipPointerGetAttributes");
  if(attr(&sa,src)!=hipSuccess||attr(&da,dst)!=hipSuccess)std::abort();
  CopyMeter m(HOST_STAGED,width*height,stream);auto e=stage_copy(dst,dpitch,src,spitch,width,height,sa.device,da.device,stream);m.finish(e);return e;}
 CopyMeter m(k,width*height,stream);const auto e=fn(dst,dpitch,src,spitch,width,height,kind,stream);m.finish(e);return e;
}

extern "C" hipError_t hipGetDeviceCount(int* n) {
    init(); if (!n) return hipErrorInvalidValue; *n = 4; return hipSuccess;
}
extern "C" hipError_t hipSetDevice(int d) {
    init(); if (!valid(d)) return hipErrorInvalidDevice;
    static auto fn = sym<hipError_t(*)(int)>("hipSetDevice");
    const auto e = fn(mapping[d]); if (e == hipSuccess) logical = d; return e;
}
extern "C" hipError_t hipGetDevice(int* d) {
    init(); if (!d) return hipErrorInvalidValue; *d = logical; return hipSuccess;
}
extern "C" hipError_t hipDeviceGetAttribute(int* p, hipDeviceAttribute_t attr, int d) {
    init(); if (!valid(d)) return hipErrorInvalidDevice;
    static auto fn = sym<hipError_t(*)(int*,hipDeviceAttribute_t,int)>("hipDeviceGetAttribute");
    return fn(p, attr, mapping[d]);
}
extern "C" hipError_t hipDeviceGetPCIBusId(char* p, int len, int d) {
    init(); if (!valid(d)) return hipErrorInvalidDevice;
    static auto fn = sym<hipError_t(*)(char*,int,int)>("hipDeviceGetPCIBusId");
    return fn(p, len, mapping[d]);
}
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int d) {
    init(); if (!valid(d)) return hipErrorInvalidDevice;
    static auto fn = sym<hipError_t(*)(hipDeviceProp_t*,int)>("hipGetDeviceProperties");
    return fn(p, mapping[d]);
}
extern "C" hipError_t hipDeviceCanAccessPeer(int* p, int d, int peer) {
    init(); if (!p) return hipErrorInvalidValue;
    if (!valid(d) || !valid(peer)) return hipErrorInvalidDevice;
    if(mapping[d]==mapping[peer]) { *p=1;return hipSuccess; }
    static auto fn = sym<hipError_t(*)(int*,int,int)>("hipDeviceCanAccessPeer");
    return fn(p,mapping[d],mapping[peer]);
}
extern "C" hipError_t hipDeviceEnablePeerAccess(int peer, unsigned int flags) {
    init();if(!valid(peer))return hipErrorInvalidDevice;
    if(mapping[logical]==mapping[peer])return hipErrorPeerAccessAlreadyEnabled;
    static auto fn=sym<hipError_t(*)(int,unsigned)>("hipDeviceEnablePeerAccess");
    return fn(mapping[peer],flags);
}
extern "C" hipError_t hipMemcpyPeerAsync(void* dst, int dd, const void* src, int sd, size_t bytes, hipStream_t stream) {
    init(); if (!valid(dd) || !valid(sd)) return hipErrorInvalidDevice;
    if (!bytes) return hipSuccess;
    if (!dst || !src) return hipErrorInvalidValue;
    static auto peer_copy=sym<hipError_t(*)(void*,int,const void*,int,size_t,hipStream_t)>("hipMemcpyPeerAsync");
    static auto local_copy=sym<hipError_t(*)(void*,const void*,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyAsync");
    if(host_staging()&&mapping[dd]!=mapping[sd]){
        CopyMeter meter(HOST_STAGED,bytes,stream);
        const auto result=stage_copy(dst,bytes,src,bytes,bytes,1,mapping[sd],mapping[dd],stream);
        meter.finish(result);return result;
    }
    CopyMeter meter(mapping[dd]==mapping[sd]?SAME_DEVICE:PEER_COPY,bytes,stream);
    const auto result = mapping[dd]==mapping[sd]
        ? local_copy(dst,src,bytes,hipMemcpyDeviceToDevice,stream)
        : peer_copy(dst,mapping[dd],src,mapping[sd],bytes,stream);
    meter.finish(result);
    if (result == hipSuccess) { ++copies; copy_bytes += bytes; }
    return result;
}

extern "C" hipError_t hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned flags){
 static auto fn=sym<hipError_t(*)(hipStream_t,hipEvent_t,unsigned)>("hipStreamWaitEvent");
 CopyMeter meter(STREAM_WAIT,0,stream);const auto result=fn(stream,event,flags);meter.finish(result);return result;
}
