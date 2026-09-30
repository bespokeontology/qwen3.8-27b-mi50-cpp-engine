#include "q27_candidate_flags.h"
#include <cstdio>
#include <initializer_list>
int main() {
    const char* keys[]={"Q27_QUANT_DOWN","Q27_GU_INT4","Q27_GU_E8M0","Q27_GU_SWILU","Q27_GU_SWILU_Q","Q27_GU_WIDE","Q27_GU_ILEAVE"};
    for(auto k:keys)unsetenv(k);
    int count=0;
    if(q27_candidate_flag_error())return 1;
    for(auto f:{"Q27_GU_SWILU","Q27_GU_SWILU_Q"})for(auto e:{"Q27_GU_INT4","Q27_GU_E8M0","Q27_GU_WIDE","Q27_GU_ILEAVE"}) {
        setenv(f,"1",1);setenv(e,"1",1);if(!q27_candidate_flag_error())return 2;
        unsetenv(f);if(q27_candidate_flag_error())return 3;
        unsetenv(e);++count;
    }
    setenv("Q27_GU_INT4","1",1);setenv("Q27_GU_E8M0","1",1);if(!q27_candidate_flag_error())return 4;
    unsetenv("Q27_GU_INT4");unsetenv("Q27_GU_E8M0");++count;
    setenv("Q27_QUANT_DOWN","1",1);if(!q27_candidate_flag_error())return 5;
    unsetenv("Q27_QUANT_DOWN");++count;
    setenv("Q27_GU_SWILU","1",1);if(q27_candidate_flag_error())return 6;
    std::printf("FLAG_GUARDS PASS cases=%d plus valid controls\n",count);
}
