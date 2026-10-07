// q27_tok -- native tokenizer driver for the Qwen3.8-27B engine (no Python).
//   q27_tok <model_dir> encode  "<text>"            -> ids
//   q27_tok <model_dir> decode  <id> <id> ...       -> text
//   q27_tok <model_dir> chat    "<user text>" [--think]   -> ids of the chat-formatted prompt
//   q27_tok <model_dir> golden                      -> the 41-id Kolmogorov prompt must reproduce exactly
// Build: g++ -O2 -std=c++17 -Isrc/tok tools/q27_tok.cpp src/tok/hf_tokenizer.cpp -lpcre2-8 -o q27_tok
#include "hf_tokenizer.h"
#include "q27_chat.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: q27_tok <model_dir> encode|decode|chat|golden ...\n"); return 2; }
    const std::string dir = argv[1], mode = argv[2];
    hf::Tokenizer T(std::filesystem::path(dir) / "tokenizer.json");
    if (mode == "encode" && argc >= 4) {
        for (int id : T.encode(argv[3])) std::printf("%d ", id);
        std::printf("\n"); return 0;
    }
    if (mode == "decode") {
        std::vector<std::int32_t> ids; for (int i = 3; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        std::printf("%s\n", T.decode(ids).c_str()); return 0;
    }
    if (mode == "chat" && argc >= 4) {
        const bool think = (argc >= 5 && !std::strcmp(argv[4], "--think"));
        for (int id : T.encode(q27::chat_prompt(argv[3], "", think))) std::printf("%d ", id);
        std::printf("\n"); return 0;
    }
    if (mode == "golden") {
        const char* user = "Explain the Kolmogorov axioms of probability, and explain precisely why countable "
                           "additivity is a strictly stronger requirement than finite additivity.";
        static const int want[] = {248045,846,198,814,20139,279,43998,199956,269,833,3731,87069,314,18356,11,321,
            10033,22898,3069,1698,470,884,17516,369,264,24660,15766,15808,1056,33093,884,17516,13,248046,198,
            248045,74455,198,248068,271,248069,271};
        const int nw = (int)(sizeof(want) / sizeof(want[0]));
        const std::string prompt = q27::chat_prompt(user, "", false);
        std::vector<std::int32_t> got = T.encode(prompt);
        bool ok = ((int)got.size() == nw);
        for (int i = 0; ok && i < nw; ++i) ok = (got[i] == want[i]);
        std::printf("golden encode: %s (%zu ids, want %d)\n", ok ? "EXACT" : "MISMATCH", got.size(), nw);
        if (!ok) { for (int id : got) std::printf("%d ", id); std::printf("\n"); }
        const std::string back = T.decode(std::vector<std::int32_t>(want, want + nw));
        std::printf("golden decode: %s\n", back == prompt ? "EXACT" : "MISMATCH");
        if (back != prompt) std::printf("got:  %s\nwant: %s\n", back.c_str(), prompt.c_str());
        return ok && back == prompt ? 0 : 1;
    }
    std::fprintf(stderr, "bad mode\n"); return 2;
}
