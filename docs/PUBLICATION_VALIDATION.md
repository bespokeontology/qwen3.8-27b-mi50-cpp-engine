# Public checkout validation

This update exports the corrected qualified C++/HIP source and native serving tools. It also adds portable launch/build packaging, removes private paths and session identifiers from published material, and preserves the earlier four-card release as historical documentation.

## Checks performed on September 30

| Check | Result |
|---|---|
| Engine, tokenizer, frontend and transport adapter build | **PASS** with ROCm 5.7.1 and GCC 11 |
| Production attention regression | **108/108**, zero mismatches |
| Production GDN regression | **54/54**, zero mismatches |
| Startup flag guard fixtures | **10 invalid combinations** rejected; valid controls pass |
| C++ nibble packer | **65539 layout checks**, four invalid inputs rejected |
| C++ frontend/parser selftest | **26/26** |
| Bash launcher syntax | **PASS** |
| Ordinary checkpoint startup and HTTP exact-answer smoke | **PASS**, reasoning separated; finite state on three GPUs; zero host collectives or host-staged bytes |
| Credential-pattern scan of publication files and existing Git history | No matches in the patterns checked |
| Publication file scan for private workstation paths and session IDs | No matches |

[Fresh raw test logs and build identities](../receipts/public-checkout-20260930/) accompany this document. Existing compiler warnings remain. The frontend test's example path and expected value were both made relative.

The public HTTP smoke check exposed a non-streaming frontend parsing defect: the prompt already opened `<think>`, but the completed-response splitter expected an opening tag in generated text. The public frontend now separates that reasoning from final content and preserves unfinished reasoning separately. Three regression cases cover the prompt-open, truncated and thinking-disabled cases. The before/after HTTP receipts retain the failure and correction. This frontend fix is additional to the frozen source; it does not change engine kernels or the original benchmark attribution.

The public launcher startup/HTTP result is recorded in the accompanying validation receipt. The published 32K/128K/262K performance figures retain their original frozen-binary attribution. They were not rerun for this publication update. The script's ordinary checkpoint-loading path also differs from the private deployment's optional prepared storage payload.

The sanitization scan is a bounded pattern check and a review of the exported file set, not a guarantee that all conceivable secret formats are detectable. No model weights, executables, private session transcripts, private Harness settings, complete request captures or Python files are part of the publication. Third-party source notices are included in `NOTICE.md` and `licenses/`.
