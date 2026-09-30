#!/usr/bin/env bash
set -euo pipefail
q27_public_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
q27_public_model=$(realpath -- "${1:?usage: serve-corrected.sh MODEL_DIRECTORY [PORT]}")
q27_public_port=${2:-8020}
[[ $q27_public_port =~ ^[0-9]+$ ]] && (( q27_public_port > 0 && q27_public_port < 65536 )) || { echo 'Invalid port' >&2; exit 2; }
[[ -f "$q27_public_model/config.json" && -f "$q27_public_model/model.safetensors.index.json" ]] || { echo 'Expected a model checkpoint directory' >&2; exit 2; }
for q27_public_key in ${!Q27_@} ${!Q27X_@}; do unset "$q27_public_key"; done
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES LD_PRELOAD
source "$q27_public_root/configs/corrected-tp3.env"
export Q27_MODEL="$q27_public_model" Q27_ENGINE_DIR="$q27_public_root" Q27_FE_PORT="$q27_public_port"
export LD_PRELOAD="$q27_public_root/native/libhip_shard_map_p2p.so"
export Q27_PREPARED_EXE_MD5=$(md5sum "$q27_public_root/q27_gen" | cut -d' ' -f1)
export ROCBLAS_TENSILE_LIBPATH="${ROCBLAS_TENSILE_LIBPATH:-/opt/rocm/lib/rocblas/library}"
cd "$q27_public_root"
exec "$q27_public_root/native/q27_fe"
