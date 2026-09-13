// tools/st_dump.cpp — standalone safetensors header dumper (host tool, no deps, no Python).
// Prints name / dtype / shape / payload-bytes for every tensor in a .safetensors file.
// Hand-written scanner, same format contract as q27_load.cpp.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

struct Ten { std::string name, dtype; std::vector<long long> shape; long long begin, end; long long bytes() const { return end - begin; } };

static const char* p_; static const char* e_;
static void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r' || *p_ == ',' || *p_ == ':')) ++p_; }
static bool lit(const char* s) { size_t n = strlen(s); if ((size_t)(e_ - p_) < n || strncmp(p_, s, n)) return false; p_ += n; return true; }
static bool str(std::string& out) { ws(); if (p_ >= e_ || *p_ != '"') return false; ++p_; const char* s = p_; while (p_ < e_ && *p_ != '"') { if (*p_ == '\\') p_ += 2; else ++p_; } if (p_ >= e_) return false; out.assign(s, p_ - s); ++p_; return true; }
static bool num(long long& out) { ws(); if (p_ >= e_) return false; char* en = nullptr; long long v = strtoll(p_, &en, 10); if (en == p_) return false; out = v; p_ = en; return true; }

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: st_dump <file.safetensors>\n"); return 2; }
    FILE* f = std::fopen(argv[1], "rb"); if (!f) { std::perror("fopen"); return 2; }
    std::fseek(f, 0, SEEK_END); long long fsz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    unsigned char h8[8]; if (std::fread(h8, 1, 8, f) != 8) { std::fprintf(stderr, "short file\n"); return 2; }
    unsigned long long N = 0; for (int i = 0; i < 8; ++i) N |= ((unsigned long long)h8[i]) << (8 * i);
    if (N > (unsigned long long)fsz - 8) { std::fprintf(stderr, "bad header len %llu\n", N); return 2; }
    std::string hdr((size_t)N, 0); if (std::fread(&hdr[0], 1, (size_t)N, f) != (size_t)N) { std::fprintf(stderr, "short header\n"); return 2; }
    std::fclose(f);
    p_ = hdr.data(); e_ = hdr.data() + hdr.size();
    ws(); if (p_ >= e_ || *p_ != '{') { std::fprintf(stderr, "not an object\n"); return 2; } ++p_;
    std::vector<Ten> tens; bool first = true;
    while (true) {
        ws(); if (p_ >= e_) break; if (*p_ == '}') { ++p_; break; }
        if (!first) { ws(); } first = false;
        std::string name; if (!str(name)) break;
        ws(); if (p_ >= e_ || *p_ != '{') break;   // __metadata__ etc: skip to matching close
        int depth = 1; ++p_;
        if (name == "__metadata__") { while (depth && p_ < e_) { if (*p_ == '{') ++depth; else if (*p_ == '}') --depth; ++p_; } continue; }
        Ten t; t.name = name; t.begin = -1; t.end = -1;
        while (depth && p_ < e_) {
            if (*p_ == '{') { ++depth; ++p_; continue; }
            if (*p_ == '}') { --depth; ++p_; continue; }
            std::string k; const char* save = p_; if (!str(k)) { ++p_; continue; }
            if (k == "dtype") { if (!str(t.dtype)) { p_ = save; ++p_; continue; } }
            else if (k == "shape") { ws(); if (p_ < e_ && *p_ == '[') { ++p_; ws(); t.shape.clear(); while (p_ < e_ && *p_ != ']') { long long v = 0; if (num(v)) t.shape.push_back(v); ws(); } if (p_ < e_) ++p_; } }
            else if (k == "data_offsets") { ws(); if (p_ < e_ && *p_ == '[') { ++p_; ws(); long long a = 0, b = 0; if (num(a)) { ws(); num(b); } t.begin = a; t.end = b; while (p_ < e_ && *p_ != ']') ++p_; if (p_ < e_) ++p_; } }
            ws();
        }
        tens.push_back(t);
    }
    std::sort(tens.begin(), tens.end(), [](const Ten& a, const Ten& b) { return a.name < b.name; });
    long long tot = 0;
    for (const auto& t : tens) {
        std::printf("%-52s %-6s [", t.name.c_str(), t.dtype.c_str());
        for (size_t i = 0; i < t.shape.size(); ++i) std::printf("%s%lld", i ? ", " : "", t.shape[i]);
        std::printf("]  %lld B\n", t.bytes());
        tot += t.bytes();
    }
    std::printf("TOTAL_TENSORS %zu TOTAL_PAYLOAD %lld B (%.3f GiB)\n", tens.size(), tot, (double)tot / 1073741824.0);
    return 0;
}
