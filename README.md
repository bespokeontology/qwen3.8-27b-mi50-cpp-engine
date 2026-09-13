# Qwen3.8-27B Native ROCm Inference on 4× AMD MI50

## Summary

A C/C++/HIP inference engine for Qwen3.8-27B, running four-way tensor parallel on four AMD
Instinct MI50 (gfx906) GPUs under ROCm 5.7.1. The execution path contains no Python: the
checkpoint is loaded by native code and decode runs in HIP kernels and rocBLAS calls issued
from C++.

Chained multi-token-prediction (MTP) speculative decoding at K=2 measured 103.7 committed
output tok/s on a 1,024-token prose prompt over a 256-token generation, against 68.6 tok/s for
non-speculative decode on the same binary. Prompt ingest is 1,771 tok/s at 1K on the
non-speculative path.

A block-parallel speculative decoder (DFlash2) is also implemented. Its measurements are in
*Speculative Decoding*.

All figures in this document are from a single frozen binary, re-run after the measurement
matrix completed.

## Hardware

| item | value |
|---|---|
| GPUs | 4 × AMD Instinct MI50, `gfx906`, 16 GB HBM2 each (also sold as Radeon Pro VII) |
| wavefront | 64 lanes, 60 CUs/card, no MFMA/matrix cores |
| interconnect | PCIe; the host performs the four-way collective reduction |
| GPU memory in use | 17.13 GiB total model residency across four cards, plus 6.72 GiB of int8 weight mirrors; 5.20 GiB free per card after the fp8 prefill arena is released |

gfx906 provides no matrix cores. GEMM throughput derives from integer dot-product paths and
rocBLAS Tensile kernels.

## Software

| item | value |
|---|---|
| ROCm | 5.7.1-98 (`/opt/rocm/bin/hipcc`) |
| kernel | Linux 6.8.0-138-generic |
| compiler | `hipcc -O3 -std=c++17 --offload-arch=gfx906` |
| BLAS | rocBLAS (int8→int32 `rocblas_gemm_ex` with `pack_int8x4`), plus hand-written HIP kernels |
| other | `libpcre2-8` for the tokenizer pre-tokenizer regex |
| runtime Python | none |

The tokenizer, checkpoint loader, quantizers, attention, MLP, collectives, speculative decoding
and chat loop are native. Python is used only for offline asset preparation.

## Build

Requires ROCm (5.7.1 or compatible) with `hipcc`, a C++17 host compiler, and `libpcre2-8`.

```
make -j12            # builds q27_gen (engine) and q27_tok (tokenizer CLI)
make -j12 q27_gen    # engine only
```

`make syntax` runs an offline host-side syntax check of the translation units against
`tools/hipstub`, requiring neither ROCm nor a GPU.

The default build targets `gfx906` (`--offload-arch=gfx906`). Other architectures are untested.

## Model weights

Weights are not distributed with this repository. The engine loads a Hugging Face-format
Qwen3.8-27B directory containing `config.json`, `model.safetensors.index.json` and the
`model-*.safetensors` shards, plus `chat_template.jinja` for the chat path. Point the engine at
that directory; no offline conversion step is required.

Quantization to the int8 execution format is performed at load time from the checkpoint's
weights. Approximately 24 GiB of aggregate GPU memory is required across the four cards.

The DFlash2 drafter uses a separate checkpoint (`dflash-aligned-v5/model.safetensors`) and is
needed only when `Q27_DFLASH2=1` is set.

## Running

```
./q27_gen <model_dir> [ndev=4] [ctx=4096] [max_new=24] [oracle_dir|""] [token ids...]
```

Free-text prompt instead of token ids:

```
Q27_TEXT=1 ./q27_gen <model_dir> 4 4096 256 "" "Explain gated delta rule attention."
```

Interactive chat, loading the model once and reading further turns from stdin:

```
./q27_chat <model_dir> "your first message"
```

`q27_chat` accepts `Q27_CTX` (default 16384), `Q27_MAXN`, `Q27_SERVE_MAXN`, `Q27_THINK=1`,
`Q27_SYSTEM="..."`, and `Q27_TEMP`/`Q27_TOPP`/`Q27_TOPK` for sampling. Sampling uses the plain
decode path.

Tokenizer CLI:

```
./q27_tok <model_dir> encode|decode|chat|golden ...
```

## Execution modes

| mode | flags | notes |
|---|---|---|
| chained MTP | `Q27_SPEC=1 Q27_SPEC_K=2` | highest measured throughput; K=2, 3 and 7 supported |
| plain decode | `Q27_SPEC=0` | no speculation; required for sampling |
| DFlash2 | `Q27_SPEC=1 Q27_SPEC_K=<K> Q27_DFLASH2=1` | block-parallel drafter; set `Q27_DFLASH2_CKPT` to the drafter checkpoint; see *Limitations* |

Baseline environment:

```
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-5.7.1/lib/rocblas/library
Q27_TP=1 Q27_LAYER_SPLIT=1 Q27_LS_Q8=1
Q27_DEC_I8=1 Q27_DEC_I8_PLAIN=1
Q27_COLL_NRBF16=1 Q27_NR_P2P=1 Q27_NR_P2P_MIN=4 Q27_NR_CH=8
```

`Q27_TP` enables four-way tensor parallel, `Q27_LAYER_SPLIT`/`Q27_LS_Q8` select the layer-split
int8 prefill, `Q27_DEC_I8` builds the int8 decode weight mirrors, `Q27_NR_P2P` selects the
GPU-resident push collective above `Q27_NR_P2P_MIN` rows, and `Q27_NR_CH` sets the decode
row-chunk width.

## Model

Qwen3.8-27B: 64 layers, hidden size 5,120, 32 query heads and 8 KV heads of dimension 128, MLP
intermediate 17,408, vocabulary 62,080. The layer stack comprises 48 gated-delta-rule
(linear-attention) layers and 16 full-attention layers.

Execution formats:

- **Decode weights**: int8, per-64-element group scales, 6.72 GiB across four cards. The fp8
  projections used during prefill are released once the int8 mirrors are built.
- **Activations**: int8, per-16-element group scales.
- **KV cache**: int8, per-group scales.
- **Norm weights, codebooks, convolution base kernels**: bf16 / fp32.
- **lm_head**: replicated on every card (62,080 rows); per-card top-k indices are global
  vocabulary ids.

Placement is four-way tensor parallel: all 64 layers are resident on every card, each holding
its shard of every layer's weights.

## Engine Architecture

**Four-GPU execution.** One host thread drives each card. Layers are sharded across the four
cards; each card computes a partial result and the four partials are reduced per layer. Two
reduction transports are implemented: a host-staged path, and a GPU-resident push collective
(`Q27_NR_P2P`) in which each card stores bf16 partials into its peers' receive slots, followed
by a local reduce fused with the subsequent norm. The push collective is selected for passes of
four or more rows; the host-staged path is selected below that width.

**Prefill.** The prompt sweep uses layer-split residency: each card owns whole layer blocks and
the prompt streams through the ring in 256-position chunks, one stream per card, forming a
pipeline rather than exchanging partial sums at every layer. Within a block the layer runs
int8: fp32 residual, device-side sum-of-squares norms, weights requantized in place from the
per-64 mirrors, `rocblas_gemm_ex` int8→int32 with per-shape Tensile solutions selected at
initialization, stacked same-input projections so that q/k/v is a single GEMM, and fused
epilogues (quantize, SwiGLU, output projection with residual and post-norm).

**Decode.** Decode is row-batched. A round evaluates NR rows through all 64 layers in one pass,
where NR is one plus the number of speculative draft tokens. The int8 projection kernels serve
NR = 1..8. Native 8-row kernels are used at eight-row width (`Q27_NR_CH=8`).

**KV and recurrent state.** Full-attention layers maintain an int8 KV cache with per-group
scales. Gated-delta-rule layers maintain recurrent state with convolution carries. Speculative
rounds snapshot the state invalidated by a rejected draft and restore the accepted prefix.

**Chained MTP.** The checkpoint includes a multi-token-prediction head: pre-fc norms of the
previous hidden state and of the next token's embedding are concatenated, passed through one
full-attention decoder layer and a norm, and projected by the shared lm_head. For K drafts the
head is evaluated K times in sequence, each consuming its own previous residual output. The
trunk verifies `[current token, draft_1 … draft_K]` in one row-batched decode step of NR = K+1
rows, and the longest matching prefix is committed.

**DFlash2.** A block-parallel speculative decoder. A five-layer draft model produces an
eight-row block in a single forward pass: one anchor row carrying the committed token's
embedding and seven rows carrying a mask-token embedding, yielding seven draft positions
without seven sequential draft steps. The block attends a persistent draft KV cache conditioned
on target-model residual taps captured at layers 5, 19, 33, 47 and 61, concatenated and
projected by a learned `fc`. Candidates are the top-16 lm_head logits per block row; the
proposal is selected by a learned predecessor/successor codebook walk across the block.

## Results

Full matrix: `docs/AUTHORITY_MATRIX.md`. Committed output tok/s:

| workload | prompt tokens | plain | chained K=2 | chained K=3 | chained K=7 | DFlash2 K=3 | DFlash2 K=7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prose p1k | 1,024 | 68.6 | 103.7 | 82.3 | 59.1 | 38.0 | 32.8 |
| coding pcode | 483 | 69.5 | 102.2 | — | 71.5 | 45.0 | 38.2 |
| prose p8k | 8,191 | 61.6 | 113.5 | 102.4 | 99.4 | — | — |
| coding pcode8k | 6,782 | — | 84.8 | — | 42.7 | 17.1 | — |

Chained MTP K=2 produced the highest committed-token throughput in all completed workload
comparisons.

Prefill (prompt ingest), same binary:

| prompt | tokens | path | prefill ms | tok/s |
|---|---:|---|---:|---:|
| p1k | 1,023 | plain | 577.6 | 1,771 |
| p1k | 1,023 | chained K=2 | 626.1 | 1,634 |
| pcode8k | 6,781 | chained K=2 | 5,088.3 | 1,333 |
| p8k | 8,190 | chained K=2 | 6,484.2 | 1,263 |

The 48 ms difference at 1K between the plain and chained paths corresponds to the MTP head's
prefill pass over the prompt (`Q27_MTP_PREFILL positions=1023 ms=43.7`), which is not executed
when speculation is disabled.

The prose p8k prompt contains 444 distinct token ids across 8,191 positions (5.4% lexical
diversity, against 37.7% for p1k). The chained drafter reaches P(accept ≥ 1) = 1.000 on it, and
those runs terminate on an end-of-sequence token after 106–108 committed tokens rather than the
requested 256. The p1k and coding rows are the representative measurements.

Repeat run from the frozen binary, chained K=2 on prose p1k: 102.894 tok/s against the matrix
value of 103.737, with 105 rounds, 2.448 committed tokens per round, P(≥1) = 0.800 and
P(≥2) = 0.648 in both runs.

## Speculative Decoding

### Chained MTP

| workload | K | rounds | committed tok/round | P(≥1) | P(≥2) | P(≥3) | round ms | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prose p1k | 2 | 105 | 2.448 | 0.800 | 0.648 | — | 23.6 | 103.7 |
| prose p1k | 3 | 94 | 2.723 | 0.766 | 0.596 | 0.362 | 33.0 | 82.3 |
| prose p1k | 7 | 87 | 2.966 | 0.701 | 0.494 | 0.322 | 50.2 | 59.1 |
| coding pcode | 2 | 105 | 2.438 | 0.819 | 0.619 | — | 23.8 | 102.2 |
| coding pcode | 7 | 71 | 3.606 | 0.732 | 0.577 | 0.451 | 50.4 | 71.5 |
| coding pcode8k | 2 | 120 | 2.133 | 0.700 | 0.433 | — | 25.1 | 84.8 |

Committed tokens per round increase with K, and round cost increases faster. Target
verification cost scales with round width: 21.2 ms at NR=3, 28.8 ms at NR=4, 48.6 ms at NR=8.
K=3 is within 10% of K=2 on the 8K prompt.

### DFlash2

The DFlash2 block executor was compared stage-by-stage against a host fp32 implementation of
the same algorithm, computed from the checkpoint's raw bf16 weights with identical inputs
(`tools/df2_oracle.cpp` in the source tree):

| stage | cosine vs host fp32 reference |
|---|---:|
| layer-0 MLP output | 0.99997 |
| layer-0 residual | 0.999992 |
| layer-4 residual | 0.999935 |
| final norm (row 1) | 0.99947 |

Applying the engine's int8 quantization to weights and to activations in the host reference
changed cosine similarity by less than 0.007.

Acceptance and throughput:

| workload | K | rounds | committed tok/round | P(≥1) | P(≥2) | P(≥3) | round ms | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prose p1k | 3 | 124 | 2.065 | 0.548 | 0.331 | 0.185 | 54.3 | 38.0 |
| prose p1k | 7 | 114 | 2.246 | 0.553 | 0.316 | 0.202 | 68.4 | 32.8 |
| coding pcode | 3 | 125 | 2.056 | 0.560 | 0.304 | 0.192 | 45.7 | 45.0 |
| coding pcode | 7 | 112 | 2.277 | 0.562 | 0.312 | 0.205 | 59.5 | 38.2 |

DFlash2 throughput is limited by lower draft acceptance and a fixed per-round block cost.
Measured P(≥1) is 0.548–0.562, against 0.700–0.819 for chained MTP on the same prompts under
the same target verification pass. The five-layer block costs approximately 17 ms per round at
every K (median 16.6 ms, mean 17.1 ms over 115 rounds), independent of the number of draft
positions used; chained MTP draft cost is 2.5–12.2 ms depending on K. The block reads
approximately 1.0 GiB of int8 weights per card per round, corresponding to approximately 1.4 ms
at achievable bandwidth.

With zero block-execution cost, the measured K=7 acceptance rate would correspond to
approximately 44 tok/s on prose p1k, below the 103.7 tok/s chained K=2 result.

DFlash2 is disabled by default because chained MTP K=2 produced higher throughput in the
completed benchmarks.

## Benchmark Methodology

- **Workload classes**: prose/reasoning and coding/structured.
- **Prompts**: `p1k` (1,024 tokens, prose), `p8k` (8,191 tokens, prose), `pcode` (483 tokens,
  coding), `pcode8k` (6,782 tokens, coding with a source-file context), published under
  `bench/prompts/`.
- **Generation length**: 256 new tokens requested per run. Runs emitting an end-of-sequence
  token terminate earlier; the committed token count is used in the rate, not the request.
- **Sampling**: greedy (temperature 0).
- **Timing**: `decode_ms` is generation-only wall clock from the first decode step to the last,
  excluding model load and prefill. `prefill_ms` is the prompt sweep alone.
- **Authority metric**: committed output tokens ÷ decode wall clock. Proposed-but-rejected
  draft tokens are excluded. Candidate generation rate is not reported as throughput.
- **Bootstrap treatment**: speculative paths perform one-time setup on the first round (drafter
  priming; for DFlash2, conditioning of the prompt into the draft KV cache). 256-token runs
  amortize this cost across approximately 90–160 rounds. Shorter runs inflate per-round
  averages and are not used.
- **Controls**: all matrix rows were produced by the same binary under the same environment,
  differing only in speculation flags.

## Reproducibility

| item | value |
|---|---|
| tag | `q27-complete-20260913` |
| commit | `5359c4db4a622e30f3922700b0b543786b3f17b0` |
| source archive | `q27-src-1a49908.tar.gz`, sha256 `58c518ccf6235711146e62fdc551284d6aa254aa5c07feca3a651b6527cd818d` |
| binary | `q27_gen`, sha256 `497227d4b0d6382747003e8807e8ea3d20ca1a74252e5d9ad39c83ef7d631edb`, md5 `bc0653af1dcdd14c9596dcd13bd7f5ad` |
| drafter checkpoint | `dflash-aligned-v5/model.safetensors`, sha256 `80b6f0f9d709d19394280d4c73ba4aae63550b1d211eca85fed7f7fad307922c` |
| build | `make -j12 q27_gen` (`-DQ27_PF_CH=256 -DQ27_PF_TSLOT=256`) |

The source archive is taken at commit `1a49908`. The tagged commit `5359c4d` differs from it
only by the addition of a receipt file under `receipts/` and contains no change to `src/`,
`include/` or the `Makefile`.

Environment for all measurements in this document:

```
ROCBLAS_TENSILE_LIBPATH=/opt/rocm-5.7.1/lib/rocblas/library
Q27_TP=1 Q27_LAYER_SPLIT=1 Q27_LS_Q8=1 Q27_RB_TUNE_LOG=0
Q27_DEC_I8=1 Q27_DEC_I8_PLAIN=1
Q27_COLL_NRBF16=1 Q27_NR_P2P=1 Q27_NR_P2P_MIN=4 Q27_NR_CH=8
```

Path selection:

```
chained MTP:      Q27_SPEC=1 Q27_SPEC_K=2
non-speculative:  Q27_SPEC=0
DFlash2:          Q27_SPEC=1 Q27_SPEC_K=<K> Q27_DFLASH2=1
```

Invocation:

```
q27_gen <model-dir> <ndev> <ctx> <new-tokens> "" $(cat <prompt>.ids)
```

`<ctx>` is an integer and must exceed prompt length plus generated tokens. The benchmark
harness is `bench/run_authority.sh`.

Repeat runs of the same configuration differ by under 1% in committed tok/s (103.737 and
102.894 on the identical configuration and binary) with identical round counts and acceptance
statistics.

## Limitations

- **DFlash2 at ≥8K context.** Prose p8k at K=3 and K=7, and coding pcode8k at K=7, abort. The
  draft block cost rises to 8,020 ms at position 8,204, after which the round exceeds the
  engine's staging done-flag timeout (`never reached generation 10 (saw 9) after 5000 ms`) and
  the process terminates. Chained and non-speculative paths complete normally at 8K.
- **32K prefill scratch.** A prefill-scratch limitation applies at 32K contexts and is
  unaddressed in this release.
- **Prose p8k prompt characteristics.** 5.4% distinct tokens, P(accept ≥ 1) = 1.000, and
  termination after 106–108 committed tokens. The 113.5 tok/s figure reflects those conditions.
- **Configuration scope.** Results apply to this checkpoint, this binary, ROCm 5.7.1, Linux
  6.8.0-138 and four MI50s.
- **Prefill rate.** An earlier prefill-only configuration in this project measured 1,892 tok/s
  at 1K (539–541 ms). This engine measures 1,771 tok/s on the plain path and 1,634 tok/s on the
  chained path. The MTP head's prefill pass accounts for part of the difference; the remainder
  is unaccounted for in this release.

## Implementation Notes

### CUDA warp-width assumptions
gfx906 uses 64-lane wavefronts. Code ported from CUDA that assumes a 32-lane shuffle width must
specify the subgroup width explicitly; otherwise a reduction intended to span one 32-lane group
spans the full wavefront and combines values from two groups.

### Activation-layout contracts
The projection and DFlash2 GEMM paths use different activation layouts: one natural order, one
an interleaved permutation. Quantizer output layout must match the consuming GEMM. A mismatch
within a 16-element group preserves output magnitude while altering direction, which is not
distinguishable from a modelling error by magnitude checks alone.

### Loader bounds checking
Device uploads should check copy return values and validate arena bounds after allocation and
upload. An unchecked overrun leaves trailing tensors zero-valued on the device. Under a
`(1 + w)` normalization convention, a zero weight is an identity operation, so the pipeline
produces well-formed output.

### Reference validation
The DFlash2 implementation was compared stage-by-stage with a host fp32 implementation using
checkpoint bf16 weights and identical inputs. Determinism and non-zero output do not establish
numerical correctness. The same harness, with quantization applied on the host, isolates the
contribution of the integer execution representation.

### GEMM initialization
Speculative decode issues conditioning GEMMs with a row count equal to accepted drafts plus
one, taking every value from 1 to K+1 at runtime. Each previously unseen shape incurs a
one-time rocBLAS solution selection of 10–46 ms. All runtime row-count variants should be
initialized before timed execution.

### Timing
One-time initialization and prompt-conditioning costs must be separated from steady-state
decode timing. A round-phase timer that includes first-round setup overstates per-round draft
cost by approximately 2.3× at 26 rounds.

### Collective elimination by replication
Replicating the draft model's MLP across cards to remove five cross-card reductions per block
requires approximately 1 GiB additional memory per card and, in this configuration, deadlocks
during device allocation. The sharded configuration is retained.

### Round-width selection
Native 8-row decode kernels (`Q27_NR_CH=8`) reduce round cost at NR=8 relative to 4-row
chunking: 75.1 ms to 68.2 ms, with layer time 19.96 ms to 15.08 ms and head-sync 28.70 ms to
26.65 ms.

## Conclusion

On four AMD MI50 GPUs the engine sustains 103.7 committed output tok/s on the 1,024-token prose
benchmark and 102.2 tok/s on the coding benchmark using chained MTP with K=2. A repeat run from
the frozen binary measured 102.9 tok/s on the prose benchmark. DFlash2 was implemented and
validated against a host fp32 reference but produced lower throughput under the tested
configurations.

## License

MIT. See [LICENSE](LICENSE).

The engine source in this repository is MIT licensed. Model weights are not distributed here
and remain subject to their own upstream license terms; the Qwen3.8-27B checkpoint and the
DFlash2 drafter checkpoint must be obtained separately.
