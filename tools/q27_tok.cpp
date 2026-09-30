// q27_tok -- native tokenizer driver for the Qwen3.8-27B engine (no Python).
//   q27_tok <model_dir> encode  "<text>"            -> ids
//   q27_tok <model_dir> encode                      -> ids, text read from STDIN (no argv limit)
//   q27_tok <model_dir> decode  <id> <id> ...       -> text
//   q27_tok <model_dir> chat    "<user text>" [--think]   -> ids of the chat-formatted prompt
//   q27_tok <model_dir> golden                      -> the 41-id Kolmogorov prompt must reproduce exactly
// Build: g++ -O2 -std=c++17 -Isrc/tok tools/q27_tok.cpp src/tok/hf_tokenizer.cpp -lpcre2-8 -o q27_tok
#include "hf_tokenizer.h"
#include "q27_chat.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
    // STDIN FORM. A serving conversation grows without bound, and the argv form hits E2BIG at
    // ~100 KB of text -- which is reached at roughly 20 k tokens, i.e. exactly when a long session
    // starts to matter. Same tokenizer, same ids; only the transport changes. Reads to EOF, so the
    // caller must close stdin (subprocess.run(input=...) does).
    if (mode == "encode") {
        std::string all, chunk(1 << 16, '\0');
        for (;;) {
            std::cin.read(&chunk[0], (std::streamsize)chunk.size());
            const std::streamsize got = std::cin.gcount();
            if (got <= 0) break;
            all.append(chunk.data(), (size_t)got);
        }
        for (int id : T.encode(all)) std::printf("%d ", id);
        std::printf("\n"); return 0;
    }
    if (mode == "decode") {
        std::vector<std::int32_t> ids; for (int i = 3; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
        std::printf("%s\n", T.decode(ids).c_str()); return 0;
    }
    // decode-stream: ONE long-lived process for a whole serving session, so incremental decode
    // costs no process spawns. Protocol, both directions line-oriented but length-prefixed on the
    // way out so that newlines inside a piece (file contents, diffs, shell output) survive:
    //   in:  "<id>\n" append one token   |  "R\n" reset between turns  |  EOF ends
    //   out: "D <nbytes>\n" followed by exactly <nbytes> raw bytes
    // Pieces are decoded one token at a time (the same decode_token the engine's text mode uses)
    // and an incomplete trailing UTF-8 sequence is held back until the continuation bytes arrive,
    // so a delta is always valid UTF-8 and never splits a character across two events.
    if (mode == "decode-stream") {
        std::string pend;                       // bytes held back: an incomplete UTF-8 tail
        std::string line;
        auto utf8_complete_len = [](const std::string& b) -> size_t {
            size_t i = b.size();
            size_t scan = 0;                    // walk back over continuation bytes
            while (i > 0 && scan < 4 && ((unsigned char)b[i - 1] & 0xC0) == 0x80) { --i; ++scan; }
            if (i == 0) return b.size();
            const unsigned char lead = (unsigned char)b[i - 1];
            size_t need = 1;
            if      ((lead & 0x80) == 0x00) need = 1;
            else if ((lead & 0xE0) == 0xC0) need = 2;
            else if ((lead & 0xF0) == 0xE0) need = 3;
            else if ((lead & 0xF8) == 0xF0) need = 4;
            else return b.size();               // stray continuation byte: do not hold it
            const size_t have = b.size() - (i - 1);
            return (have >= need) ? b.size() : (i - 1);
        };
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
            if (line.empty()) continue;
            if (line == "R") { pend.clear(); std::printf("D 0\n"); std::fflush(stdout); continue; }
            pend += T.decode_token((std::int32_t)std::atoi(line.c_str()));
            const size_t cut = utf8_complete_len(pend);
            const std::string out = pend.substr(0, cut);
            pend.erase(0, cut);
            std::printf("D %zu\n", out.size());
            if (!out.empty()) std::fwrite(out.data(), 1, out.size(), stdout);
            std::fflush(stdout);
        }
        return 0;
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
