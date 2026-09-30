# Qwen3.8 27B C++ and HIP engine for AMD MI50

Native inference on **3 × AMD MI50 32GB**, gfx906, ROCm 5.7.1. The corrected stack was qualified through **262K context** on September 29, 2026. The engine, tokenizer, HTTP frontend, tool-call parser, and component tests use C++/HIP. No Python is required for this build or serving path.

## Measured results

| Workload | Corrected result | Matched control | Measurement |
|---|---:|---:|---|
| Native decode, 1K fixture | **69.17 tok/s** | 65.87 tok/s | Sequential A/B/B/A; **+5.01%** |
| MTP K=2, 1K fixture | **110.66 committed tok/s** | 108.64 committed tok/s | Separate A/B/B/A; **+1.86%** |
| 262K initial prefill | **264.32 tok/s** | — | 261888 input tokens; one qualification run |
| 262K initial / repeat / continuation | **43.51 / 43.62 / 45.71 committed tok/s** | — | MTP K=2; exact-answer context probes |
| Corrected production-kernel tests | **162/162 exact passes** | — | Attention 108; GDN 54; three GPUs |
| Qualification receipt audit | **235/235 checks passed** | — | Ordered Harness, tool, 32K, 128K, 262K gates |

[Raw receipts](receipts/corrected-20260929/) include per-run timing, answers, state checks and transport records. Native **>70.17 tok/s** and a general short-workload **>120 committed tok/s** remain unmet. The old **68.59 tok/s** development result is superseded because two fused kernels omitted a required BF16 rounding boundary.

These measurements belong to the qualified frozen binary with SHA256 `4db2594b14a4f8488b323d407a401c8de6f150a3e1537cba7793166900fff4c0`. This repository exports its active source with portable paths, build/launch packaging, and a frontend reasoning/content parsing fix. See [publication validation](docs/PUBLICATION_VALIDATION.md) for checks performed on this checkout. No binary or model weights are included.

## Actual DeepSeek Harness rates

**Harness UI tok/s is the primary user-visible rate.** Engine committed throughput measures internal MTP generation. They have different timing and token-count boundaries and must be reported separately.

| Actual Harness turn | UI tok/s | Engine committed tok/s | TTFT | Total task wall | Input tokens by step | Visible final tokens |
|---|---:|---:|---:|---:|---|---:|
| Qualification Minimal | **251** | **121.00** | 1.999 s | 2.167 s | 1080 | 7 |
| Qualification bash/tool return | **277** | **135.27 / 125.67** | 0.625 s | 2.067 s | 1165 / 1264 | 4 |
| Frozen restart Minimal | **207** | **122.30** | 1.990 s | 2.136 s | 1060 | 5 |
| Frozen restart bash/tool return | **271** | **130.27 / 99.39** | 0.775 s | 2.126 s | 1132 / 1249 | 4 |

Sources: [qualification](receipts/corrected-20260929/qualification/) and [September 30 restart](receipts/corrected-20260929/restart/). The two engine rates in a tool turn belong to its two model steps. Visible final tokens exclude reasoning, tool arguments and boundary whitespace; the C++ tokenizer supplies the counts. The UI observation and raw native turn log accompany each result.

The native **69.17 tok/s** figure comes from a separate target-only benchmark, not these Harness turns. A Minimal observation above 120 does not establish that the general K2 fixture exceeds 120. These short exact-answer turns validate integration; they are not a broad coding or model-quality benchmark.

## Context qualification

These probes use the native HTTP API. **They have no Harness UI rate.** Generation columns report committed MTP throughput.

| Input tokens | Initial prefill tok/s | Initial | Repeat | Continuation | Rewind | Reset |
|---:|---:|---:|---:|---:|---:|---:|
| 32512 | **975.17** | 107.50 | 107.31 | 107.71 | 106.54 | 138.90 |
| 130816 | **458.81** | 67.42 | 67.57 | 68.39 | 68.11 | 138.59 |
| 261888 | **264.32** | 43.51 | 43.62 | 45.71 | 45.80 | 135.20 |

| Initial probe | TTFT | Total turn wall | Visible output tokens |
|---|---:|---:|---:|
| 32K | 33.433 s | 33.853 s | 25 |
| 128K | 285.704 s | 286.268 s | 25 |
| 262K | 992.250 s | 992.992 s | 25 |

[32K](receipts/corrected-20260929/ctx32/), [128K](receipts/corrected-20260929/ctx128/) and [262K](receipts/corrected-20260929/ctx262/) receipts contain the exact expected and returned answers, continuation, nonzero rewind, reset, finite-state scans, and transport checks. Tested requests had **zero non-finite state**, **zero host collective calls**, and **zero host-staged transfer bytes**. These finite-scan fields and transport counters have the scope recorded in the receipts; they do not imply that all host activity is eliminated.

The corrected stack's prefill qualification is not a matched A/B comparison against the prior freeze. No prefill improvement over that freeze is claimed.

## Build

Requirements: Linux, a compatible ROCm toolchain and runtime, a C++17 compiler, PCRE2 development headers/library, and Bash. The measured setup uses ROCm 5.7.1 and gfx906. Other architectures and software versions are unqualified.

```bash
make -j2 all CH=256 TSLOT=256 V4=1 EXP=0
make -j2 component-build CH=256 TSLOT=256 V4=1 EXP=0
```

If ROCm 5.7.1 needs an explicit GCC installation on the host:

```bash
make -j2 all component-build \
  HIPCC='/opt/rocm/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/11'
```

The build produces `q27_gen`, `q27_tok`, the C++ frontend `native/q27_fe`, and the native transport adapter. Object directories include layout-changing build flags to prevent stale-object reuse. See [component checks](tests/takeover/README.md).

## Run the corrected three card configuration

Obtain a compatible Qwen3.8-27B NVFP4 checkpoint separately. The directory must include `config.json`, `model.safetensors.index.json`, its referenced shards, and tokenizer assets. The corrected path uses native NVFP4 MLP weights and int8 projection mirrors. BF16/EXL3 experiments and DFlash2 are outside this release's qualification.

```bash
./scripts/serve-corrected.sh /path/to/model 8020
```

The launcher reads [configs/corrected-tp3.env](configs/corrected-tp3.env), sets the model and engine paths, and serves on **127.0.0.1:8020**. It starts with a clean set of Q27 flags. The process stays in the foreground; stop it with Ctrl-C. The three-card configuration assumes three suitable visible devices and the qualified memory layout; it is not qualified for three 16GB cards.

```bash
curl http://127.0.0.1:8020/v1/models
curl http://127.0.0.1:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with exactly: 2+2=4"}],"max_tokens":128}'
```

The public launcher uses ordinary checkpoint loading. The measured deployment used an external prepared storage layout for startup; that payload is not distributed and startup time is not a public performance claim. The engine retains optional native storage export/replay support. No local workstation layout or private Harness configuration is required by the public launcher.

## Changes in the corrected stack

- Three-card ragged sharding and GPU-resident LL collectives, including register receive and epilogue work.
- K16 attention and eligible long-prefill rocBLAS dispatch.
- Grouped SwiGLU and producer fusions for GDN a/b/convolution, GDN output quantization, gate/up SwiGLU, and attention output quantization.
- Required **FP32 → BF16 → INT8** boundary restored in the native attention and GDN fused epilogues. The production-kernel regression suite compares outputs, codes, scales, state, snapshots, padding and guards exactly.
- Startup rejection of conflicting experimental consumers. INT4/E8M0 results collected while `Q27_GU_SWILU` bypassed those consumers are invalid as candidate measurements.
- C++ nibble-pair packer with **65539 layout checks** and four invalid-input checks. It has no qualified GPU consumer or performance result.

Generated-token hashes are diagnostic. Correctness claims also require exact task answers and component/state checks. [Qualification audit](receipts/corrected-20260929/qualification-audit.json) and [measurement notes](docs/MEASUREMENTS.md) define the evidence boundaries.

## Historical release

The older **4 × MI50 16GB** release measured **103.7 committed tok/s** on its own 1K fixture. Its source, hardware, flags and KV path differ. Its description is preserved in [HISTORICAL_20260913.md](docs/HISTORICAL_20260913.md), with the original [authority matrix](docs/AUTHORITY_MATRIX.md). Those results are not a controlled comparison against this three-card stack.

Flash-Next is a separate engine and is not included in this Qwen3.8-27B source update.

## License

[MIT](LICENSE), with [third-party notices](NOTICE.md). Model weights and tokenizer assets remain subject to their upstream terms.
