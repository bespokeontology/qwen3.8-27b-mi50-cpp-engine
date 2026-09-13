// Qwen3.5 chat template, the text-only subset the engine ships: an optional system turn, the user
// turn, and the assistant generation prompt. With thinking disabled the template emits an empty
// think block, which is exactly what the engine's reference 41-id prompt contains:
//   <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
// Multi-turn continuation appends "<|im_end|>\n" after the assistant's answer (the engine stops on
// <|im_end|> without feeding it back), then the next user turn in the same form.
#pragma once
#include <string>
#include <string_view>
namespace q27 {
inline std::string chat_prompt(std::string_view user, std::string_view system = "", bool thinking = false) {
    std::string s;
    if (!system.empty()) { s += "<|im_start|>system\n"; s += system; s += "<|im_end|>\n"; }
    s += "<|im_start|>user\n"; s += user; s += "<|im_end|>\n<|im_start|>assistant\n";
    if (!thinking) s += "<think>\n\n</think>\n\n";
    return s;
}
// The text that continues a conversation after an assistant answer that ended on <|im_end|>.
inline std::string chat_next_turn(std::string_view user, bool thinking = false) {
    std::string s = "<|im_end|>\n<|im_start|>user\n"; s += user; s += "<|im_end|>\n<|im_start|>assistant\n";
    if (!thinking) s += "<think>\n\n</think>\n\n";
    return s;
}
}  // namespace q27
