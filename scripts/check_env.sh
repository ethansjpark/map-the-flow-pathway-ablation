#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/workspace/logs/check_env.log"
# 14.1 GB model + 1.56 GB archives = 15.66 GB; 16 GB is only a payload floor.
MIN_FREE_BYTES=16000000000

mkdir -p /workspace/logs
exec > >(tee -a "$LOG_FILE") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
failed=0
check() {
  local label="$1"
  shift
  if "$@"; then log "PASS $label"; else log "FAIL $label"; failed=1; fi
}
check_disk() {
  local available
  available="$(df -B1 --output=avail /workspace | tail -n 1 | tr -d ' ')" || return 1
  log "Free disk: $available bytes (required floor: $MIN_FREE_BYTES bytes)"
  [[ "$available" =~ ^[0-9]+$ ]] && (( available >= MIN_FREE_BYTES ))
}

log "Starting environment checks"
check "nvidia-smi" nvidia-smi -L
check "nvcc" nvcc --version
check "Python 3.10" python3 -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 10)'
check "torch 2.2.1 / CUDA 11.8" python3 -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.__version__.split("+")[0] == "2.2.1" and torch.version.cuda == "11.8"'
check "torch.cuda.is_available" python3 -c 'import torch; print(torch.cuda.is_available()); assert torch.cuda.is_available()'
for module in transformers flash_attn decord mmcv; do
  check "import $module" python3 -c "import $module"
done
check "free disk" check_disk
check "model folder" test -d /workspace/models/LLaVA-NeXT-7B-Video-FT
check "dataset folder" test -d /workspace/datasets/TVBench

if (( failed != 0 )); then
  log "FAIL check_env"
  exit 1
fi
log "PASS check_env"
