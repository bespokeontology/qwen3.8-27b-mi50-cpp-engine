#pragma once
#include <cstdlib>
// Fail before model loading when a requested benchmark arm cannot reach its
// consumer. A successful process exit is otherwise indistinguishable from A/A.
inline const char* q27_candidate_flag_error() {
    auto on=[](const char* k){const char* v=std::getenv(k);return v&&std::atoi(v)!=0;};
    if(on("Q27_QUANT_DOWN"))
        return "Q27_QUANT_DOWN is banked for an unsafe grid barrier; use the two-launch path";
    if(on("Q27_GU_INT4") && on("Q27_GU_E8M0"))
        return "Q27_GU_INT4 and Q27_GU_E8M0 are mutually exclusive experiments";
    if((on("Q27_GU_SWILU") || on("Q27_GU_SWILU_Q")) &&
       (on("Q27_GU_INT4") || on("Q27_GU_E8M0") || on("Q27_GU_WIDE") || on("Q27_GU_ILEAVE")))
        return "GU fusion bypasses the requested INT4/E8M0/WIDE/ILEAVE decode consumer";
    return nullptr;
}
