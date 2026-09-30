# Source attribution

The engine is native C++/HIP. It uses ROCm/rocBLAS and PCRE2 as separately installed dependencies.

The EXL3 reconstruction and Hadamard code in `src/exl3_dequant_gfx906.h`, `src/exl3_dequant_gfx906.hip` and `src/q27_exl3_had.hip` adapts algorithms and code from [exllamav3](https://github.com/turboderp-org/exllamav3), with source comments retaining the upstream `c5d9c657` pin and file references. Its MIT notice is included in [licenses/exllamav3-MIT.txt](licenses/exllamav3-MIT.txt). These experimental paths are not enabled in the corrected NVFP4 qualification configuration.

Model weights and tokenizer assets are not distributed. Obtain them separately under their upstream terms. Flash-Next is a separate engine and is not part of this Qwen3.8-27B release.
