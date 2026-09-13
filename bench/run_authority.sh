#!/bin/bash
# Authority benchmark harness for the Qwen3.8-27B ROCm engine.
# Reproduces every row of docs/AUTHORITY_MATRIX.md from the frozen binary.
#
#   ./run_authority.sh <path-to-q27_gen> <path-to-model-dir> [output-dir]
#
# Each row: 256 new tokens requested, greedy, one binary, one environment.
# Authority metric = committed output tokens / decode wall clock.
set -u
BIN=${1:?usage: run_authority.sh <q27_gen> <model-dir> [outdir]}
MODEL=${2:?usage: run_authority.sh <q27_gen> <model-dir> [outdir]}
OUT=${3:-./authority-results}
PR="$(cd "$(dirname "$0")" && pwd)/prompts"
mkdir -p "$OUT"
TSV=$OUT/matrix.tsv
[ -s "$TSV" ] || printf "workload\tprompt_tok\tpath\tK\tprefill_ms\trounds\ttok_per_round\tP1\tP2\tP3\tround_ms\tdraft_ms\tlayers_ms\theadsync_ms\ttail_ms\tdecode_ms\tcommitted_tok_s\n" > "$TSV"

row() {  # name prompt-file ctx path K
  local NAME=$1 P=$2 CTX=$3 PATHNM=$4 K=$5 TAG="${1}_${4}${5}"
  local DF="" SP="Q27_SPEC=1 Q27_SPEC_K=$K"
  [ "$PATHNM" = "df2" ]   && DF="Q27_DFLASH2=1"
  [ "$PATHNM" = "plain" ] && SP="Q27_SPEC=0"
  env ROCBLAS_TENSILE_LIBPATH=/opt/rocm-5.7.1/lib/rocblas/library \
      Q27_TP=1 Q27_LAYER_SPLIT=1 Q27_LS_Q8=1 Q27_RB_TUNE_LOG=0 \
      $SP Q27_COLL_NRBF16=1 Q27_NR_P2P=1 Q27_NR_P2P_MIN=4 Q27_NR_CH=8 \
      Q27_DEC_I8=1 Q27_DEC_I8_PLAIN=1 $DF \
      "$BIN" "$MODEL" 4 "$CTX" 256 "" $(cat "$PR/$P") > "$OUT/$TAG.out" 2>&1
  local L T TM
  L=$(grep -a "Q27_SPEC K=" "$OUT/$TAG.out" | tail -1)
  T=$(grep -a "Q27_SPEC_T"  "$OUT/$TAG.out" | tail -1)
  TM=$(grep -a "Q27_TP_TIMING" "$OUT/$TAG.out" | tail -1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$NAME" "$(wc -w < "$PR/$P")" "$PATHNM" "$K" \
    "$(sed -n 's/.*prefill_ms=\([0-9.]*\).*/\1/p' <<<"$TM")" \
    "$(sed -n 's/.*rounds=\([0-9]*\).*/\1/p' <<<"$L")" \
    "$(sed -n 's/.*tokens_per_round=\([0-9.]*\).*/\1/p' <<<"$L")" \
    "$(sed -n 's/.*P(>=1)=\([0-9.]*\).*/\1/p' <<<"$L")" \
    "$(sed -n 's/.*P(>=2)=\([0-9.]*\).*/\1/p' <<<"$L")" \
    "$(sed -n 's/.*P(>=3)=\([0-9.]*\).*/\1/p' <<<"$L")" \
    "$(sed -n 's/.*total \([0-9.]*\).*/\1/p' <<<"$T")" \
    "$(sed -n 's/.*draft \([0-9.]*\).*/\1/p' <<<"$T")" \
    "$(sed -n 's/.*layers \([0-9.]*\).*/\1/p' <<<"$T")" \
    "$(sed -n 's/.*head-sync \([0-9.]*\).*/\1/p' <<<"$T")" \
    "$(sed -n 's/.*tail \([0-9.]*\).*/\1/p' <<<"$T")" \
    "$(sed -n 's/.*decode_ms=\([0-9.]*\).*/\1/p' <<<"$TM")" \
    "$(grep -aE '^    raw ' "$OUT/$TAG.out" | tail -1 | awk '{print $2}')" >> "$TSV"
  echo "done $TAG"
}

row prose1k  p1k.ids     4096 plain 0
row prose1k  p1k.ids     4096 chain 2
row prose1k  p1k.ids     4096 chain 3
row prose1k  p1k.ids     4096 chain 7
row prose1k  p1k.ids     4096 df2   3
row prose1k  p1k.ids     4096 df2   7
row code483  pcode.ids   4096 plain 0
row code483  pcode.ids   4096 chain 2
row code483  pcode.ids   4096 chain 7
row code483  pcode.ids   4096 df2   3
row code483  pcode.ids   4096 df2   7
row prose8k  p8k.ids     9216 plain 0
row prose8k  p8k.ids     9216 chain 2
row prose8k  p8k.ids     9216 chain 3
row prose8k  p8k.ids     9216 chain 7
row code6k8  pcode8k.ids 8192 chain 2
row code6k8  pcode8k.ids 8192 chain 7
row code6k8  pcode8k.ids 8192 df2   3
# DFlash2 at >=8K aborts (see README Limitations); those rows are omitted.
echo "matrix written to $TSV"
