#pragma once
// Minimal adapter for the frozen GLM transfer engine. Native payload only.
#include <cstdint>
#include <vector>
#include <fcntl.h>
struct GfEntry { struct { uint32_t file_idx; uint64_t data_off, nbytes; } rec; };
struct GfStore {
    int n_shards = 0;
    std::vector<int> shard_fds;
    std::vector<uint64_t> shard_sizes;
    std::vector<const char*> shard_paths;
};
inline void gf_store_drop_range(GfStore* s,int file,uint64_t off,uint64_t bytes) {
    posix_fadvise(s->shard_fds[file],off,bytes,POSIX_FADV_DONTNEED);
}
