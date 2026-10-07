# Reference gate

`reference_prompt_ids.txt` — the 41 token ids of the chat-formatted prompt
"Explain the Kolmogorov axioms of probability, and explain precisely why countable additivity is a
strictly stronger requirement than finite additivity." (thinking off). `./q27_tok <model_dir> golden`
must reproduce them from the text and decode them back.

`reference_tokens.txt` — the 24 greedy tokens every build must produce from that prompt:

```bash
Q27_TP=1 ./q27_gen <model_dir> 4 4096 24 "" $(cat tests/reference_prompt_ids.txt) | grep ^Q27_TOKENS
# must equal: Q27_TOKENS <contents of tests/reference_tokens.txt>
```

The same 24 tokens must come out of the text path:

```bash
Q27_TP=1 Q27_TEXT=1 ./q27_gen <model_dir> 4 4096 24 "" "Explain the Kolmogorov axioms of probability, and explain precisely why countable additivity is a strictly stronger requirement than finite additivity." 2>&1 | grep ^Q27_TOKENS
```

`q27_gate.h` is the numerical ladder used against the CUDA oracle's layer vectors during the port.

## Prefill gates (added 2026-09-10 evening)

Every prefill change must reproduce the frozen release binary's tokens on: the 41-id reference prompt
(above), a 1026-id prompt (its last chunk holds one position — the case that once double-ran the
GDN recurrence), a 1024-id prompt, and the 3-turn serve transcript (`Q27_SERVE=1`, md5 of the
`Q27_TOKENS` lines against the release transcript). 8K and 32K are run when a change survives the
short gates. Before a kernel variant gets GPU time, its `hipcc -Rpass-analysis=kernel-resource-usage`
report must show no scratch and occupancy no lower than the kernel it replaces, unless the design
deliberately trades waves for bytes (the four-row tile projection does, and says so).

## Hardware-counter instruments (added 2026-09-11)

Two rocprof post-processors live in `tools/`:

- `tools/pmc_census.py <csv>` — counters over the kernel census (`Q27_MLPB_BENCH=1 ./q27_gen`), aggregated
  per kernel INSTANTIATION (template arguments kept; substring families mixed R=1 and R=2 before).
- `tools/pmc_map.py <csv>` — counters over a real run (e.g. a 128-token prefill on all four cards), ranked
  by share of GPU time: the execution map with VALU busy, waves per dispatch and per-wave instruction
  counts for every kernel.

Counter sets this rocprof (ROCm 5.7.1) can create on gfx906, at most one pass each:
```
pmc : SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_VMEM_RD SQ_INSTS_SALU SQ_INSTS_SMEM GRBM_GUI_ACTIVE
pmc : SQ_WAVES SQ_ACTIVE_INST_VALU SQ_THREAD_CYCLES_VALU SQ_WAIT_INST_LDS SQ_INST_CYCLES_SALU SQ_INSTS_VALU GRBM_GUI_ACTIVE
pmc : TCC_HIT_sum TCC_MISS_sum TCC_EA_RDREQ_sum TCC_EA_RDREQ_32B_sum
```
Any set containing SQ_BUSY_CYCLES, SQ_ACTIVE_INST_LDS, SQ_WAIT_INST_ANY, SQ_INSTS_VALU_INT32/CVT or the
TA/TCP sums fails whole ("Context Create failed"). VALU busy = SQ_ACTIVE_INST_VALU x 4 / (240 SIMDs x
GRBM_GUI_ACTIVE). Run as `rocprof -i set.txt --timestamp on -o out.csv <command>`.

What the map found on 2026-09-11 (rung 2 -> rung 3): every prefill kernel is stall-bound, not
bandwidth-bound (weights are read exactly once); the reduction epilogues were serialized permute+wait
chains (rebuilt exactly: `src/q27_wave.h`); the GDN tile scan runs 12 blocks per card at 8.8% VALU busy.
Static complement: `hipcc --save-temps`, count the main loop between a label and its backward branch,
and measure `ds_read`/`ds_bpermute` -> `s_waitcnt` distances.

## Rung-3 gate additions (2026-09-11)

- Every bank compares decode against the previous frozen binary with an interleaved 12-pair protocol
  (200 tokens each, alternating binaries), not only its own decode block: the first rung-3 build
  decoded 3.3% slower with byte-identical decode kernels because ~350 KB of closed experimental
  kernels sat in its code objects (receipt v93). Experiments compile only with `make EXP=1`.
- The bank window (`~/q27_window62.sh` / `67.sh` on the box) copies the binary as `.pending` right
  after the build and renames it into `/data/q27-freeze/` only when every gate passed, appending the
  MANIFEST line itself.
