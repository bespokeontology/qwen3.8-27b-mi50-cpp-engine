# DFlash2 vs chained MTP — authority-shaped runs, 2026-09-12

Binary: dflash2-port, same binary for every row. Env: ROCBLAS_TENSILE_LIBPATH=/opt/rocm-5.7.1/lib/rocblas/library
Q27_TP=1 Q27_LAYER_SPLIT=1 Q27_LS_Q8=1 Q27_SPEC=1 Q27_COLL_NRBF16=1 Q27_NR_P2P=1 Q27_NR_P2P_MIN=4 Q27_NR_CH=4
Q27_DEC_I8=1 Q27_DEC_I8_PLAIN=1 [Q27_DFLASH2=1 for the DFlash rows].
Workload: prose/reasoning p1k (1023-token prompt), 256 new tokens, greedy. Prefill ~621-633 ms.
Authority metric = committed output tokens / decode wall clock.

| path    | K | rounds | tok/round | P>=1  | P>=2  | P>=3  | round ms | committed tok/s |
|---------|---|-------:|----------:|------:|------:|------:|---------:|----------------:|
| chained | 2 |    105 |     2.448 | 0.800 | 0.648 |   -   |     23.5 |     **103.964** |
| chained | 3 |     94 |     2.723 | 0.766 | 0.596 | 0.362 |     33.0 |          82.449 |
| chained | 7 |     82 |     3.122 | 0.720 | 0.549 | 0.354 |     57.6 |          54.123 |
| DFlash2 | 2 |    130 |     1.962 | 0.569 | 0.392 |   -   |     46.2 |          42.362 |
| DFlash2 | 3 |    124 |     2.065 | 0.548 | 0.331 | 0.185 |     54.6 |          37.778 |
| DFlash2 | 7 |    114 |     2.237 | 0.535 | 0.325 | 0.211 |     75.1 |          29.747 |

Round decomposition (card 0, ms):

| path        | draft* | layers | head-sync | tail  | total |
|-------------|-------:|-------:|----------:|------:|------:|
| chained K=7 |   7.85 |  19.80 |     29.08 |  0.48 |  57.6 |
| DFlash2 K=7 |   8.45 |  19.96 |     28.70 | 17.86 |  75.1 |

*the "draft" bucket brackets the chained MTP path (lap(0)); under DFlash2 it holds only
bootstrap/prompt-conditioning amortization. The DFlash block lands in `tail`.

## What the numbers say

1. There is NO target-verifier anomaly. layers+head-sync are identical in both paths
   (~48.7 ms at NR=8). Target verify scales with NR: 21.2 / 28.8 / 48.6 ms at NR=3/4/8.
2. The DFlash block is a FIXED ~17 ms tax at every K (tail 0.48 -> 17.9 ms), measured
   directly: n=115 p50=16.6 mean=17.1 ms; forward ~13.1, selector ~1.6, cond sub-ms after priming.
3. DFlash2 loses on BOTH axes on this checkpoint: it costs ~17 ms more per round AND
   drafts less accurately than the checkpoint's own MTP head (P>=1 0.535 vs 0.720 at K=7;
   0.569 vs 0.800 at K=2).
4. Ceiling arithmetic: even with a FREE block, DFlash2 K=7 = 2.237 tok / 57.2 ms = 39 tok/s.
   To reach 104 tok/s at the measured NR=8 verify cost it would need ~5.0 committed
   tokens/round, i.e. P>=1..P>=4 all near 1.0. Our accepted-drafts/round is 1.237 against
   the reference's reported greedy 2.7-2.8, so there is still an acceptance gap to close,
   but closing it entirely still does not reach the bar without also removing the block tax.

## Negatives banked
- Replicating the drafter MLP per card (to delete 5 cross-card reduces/block): +1 GiB/card and
  reproducibly HANGS at T4_devalloc_done, all GPUs 0%, host-side. Reverted.
- Batching the 88 per-row RMSNorm launches into 11: block 17.4 -> 17.1 ms. Launch count is not
  the block's dominant cost.
- Selector __shfl_down width (64-lane gfx906 wavefront folding two candidates): correct fix,
  2.217 -> 2.237 tok/round. Not the acceptance limiter.

## K=1 large-sample correction
DFlash2 K=1, 256 tokens, 158 rounds: tok/round 1.614, P>=1 **0.614**, round 39.7 ms
(layers 16.7 | head-sync 0.35 | tail 16.6 = the block), 40.628 tok/s.
The earlier P>=1=0.703 was a 37-round sample. First-draft accuracy is 0.614 against the
chained MTP head's 0.800 - the DFlash2 drafter is simply less accurate on this checkpoint.

## Where the block's 13.1 ms goes (roofline)
Per block the drafter touches ~1.0 GiB of int8 weights per card (q/k/v/o are replicated,
MLP is 1/4-sharded). At ~700 GB/s that is ~1.4 ms of unavoidable traffic; measured 13.1 ms
is ~9x off roofline. q27_df2_gemm routes to q27_proj_i8g_rows, the SAME family the target's
DEC_I8 decode path uses, so the gap is occupancy/dispatch at NR=8 across many small GEMMs,
not a wrong kernel family. This is the next structural target for the block.

## Ceiling arithmetic against the 103.964 tok/s bar
- DFlash2 K=1 with a FREE block: 1.614 tok / 23.1 ms = 70 tok/s.
- DFlash2 K=7 with a FREE block: 2.237 tok / 57.2 ms = 39 tok/s.
- DFlash2 K=7 with a free block AND perfect acceptance (8 tok/round): 8 / 57.2 = 140 tok/s.
So the bar is reachable only if BOTH the block tax is removed AND acceptance rises far above
what this drafter currently produces. Acceptance is the binding constraint, and it is a
property of the DFlash2 head on this checkpoint, not of the port: the block is validated to
cosine 0.99947 against a host fp32 recomputation of the reference algorithm.

## Q27_NR_CH=8 (native 8-row kernels) — KEEP
The frozen K=7 authority ran Q27_NR_CH=8; the DFlash rows above ran CH=4 for an NR=8 verify.
Corrected: DFlash2 K=7 round 75.1 -> 68.2 ms (layers 19.96 -> 15.08, head-sync 28.70 -> 26.65),
2.246 tok/round, **32.873 tok/s** (was 29.747). CH=8 is now the DFlash default for K=7.
