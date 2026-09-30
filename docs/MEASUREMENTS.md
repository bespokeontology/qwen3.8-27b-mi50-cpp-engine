# Measurement scope and evidence

The current benchmark authority is the corrected September 29 freeze, engine SHA256 `4db2594b14a4f8488b323d407a401c8de6f150a3e1537cba7793166900fff4c0`. Active source provenance is recorded in [SOURCE_MAP.json](../receipts/corrected-20260929/SOURCE_MAP.json). [RECEIPT_MAP.json](../receipts/corrected-20260929/RECEIPT_MAP.json) records original and sanitized receipt hashes.

## Native and K2 fixtures

Each bracket runs A1, B1, B2, A2 sequentially, with the same fixture and binary. A keeps PF_BLAS, GROUP4 and LL register receive. B additionally enables corrected LL_EPI, GDN_AB_CONV, GDN_SCAN_Q, GU_SWILU and ATTN_Q. Native uses K=0; K2 uses K=2. This compares the group of changes, not each fusion individually.

| Run | Native ms/token | Native tok/s |
|---|---:|---:|
| A1 | 15.21765 | 65.713 |
| B1 | 14.47569 | 69.081 |
| B2 | 14.43765 | 69.263 |
| A2 | 15.14392 | 66.033 |

Average time gives **15.18078 → 14.45667 ms/token**, or **65.87275 → 69.17224 tok/s**. K2's separately measured committed rates are **108.64051 → 110.65900 tok/s**. Both candidate runs beat both control runs in each bracket. Rates are generation throughput, excluding startup and prompt prefill. The fixture has 1024 input IDs. Native timing records 1023 prefill positions and 255 timed decode tokens out of 256 generated tokens. The K2 B1 receipt records 1023 timed prefill positions and 257 timed decode tokens, with 256 returned token IDs. These legacy timing boundaries are retained; do not replace the actual denominator with the requested cap.

Raw [native](../receipts/corrected-20260929/native-b1/) and [K2](../receipts/corrected-20260929/k2-b1/) run directories include launch parameters, engine output and result JSON. Other bracket arms are siblings. The fixture is [corrected-p1k.ids](../bench/prompts/corrected-p1k.ids). The historical result field `timing.native_tok_s` is named generically by the fixture collector: in a K2 directory it means **committed speculative throughput**, not target-only throughput. Check the launch speculation flags.

## Harness and context probes

Harness UI rates are copied from the actual displayed turn rate, preserved in `ui-ax.txt`. Engine rates come from the linked native request logs. Harness TTFT is measured to its first reasoning/text delta and can differ from engine prefill time. Total task wall includes all model steps and tool execution. Visible final token counts use the C++ tokenizer and exclude reasoning and tool arguments; API output counts include additional generated material and are recorded separately.

The restart and original qualification prompts differ slightly, so their short-turn rates are not an A/B performance experiment. In particular, the tool response step at restart recorded 99.39 committed tok/s with lower acceptance. A short integration test does not establish broad throughput or model quality.

Context probes are native API requests, not Harness UI tests. The legacy JSON field `native_decode_tok_s` in these K2 probes means internal committed MTP throughput. Initial, repeat, continuation, rewind and reset each retain exact expected/actual answers and linked finite/transport records. A small exact-answer retrieval fixture is not a comprehensive 262K quality benchmark.

The receipt audit's **235/235** count includes source/config identity and evidence-linkage checks as well as functional checks. It is separate from the **162** production-kernel regression cases. No token hash is used as sole proof of numerical correctness.

## Publication boundaries

The repository includes sanitized raw benchmark exports. Paths become relative examples; no model payloads, private chat transcripts, screenshots, credentials or complete request captures are shipped. Harness session UUIDs and tool event objects are removed from exported result JSON. Numeric metrics, token arrays, expected/actual test answers and request linkage are retained. A sanitized receipt is not claimed to be byte-identical to its original; both hashes are recorded.

The public build differs from the frozen binary in path strings, packaging and a frontend fix separating prompt-open reasoning from non-streaming final content. Fresh checkout validation is documented separately; the long qualification remains attributed to the frozen source/binary.
