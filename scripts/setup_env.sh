#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/byminji/map-the-flow.git"
REPO_DIR="/workspace/map-the-flow"
PINNED_COMMIT="566f3ed064d324408fdf96f7a921ebf1e65f0908"
LOG_FILE="/workspace/logs/setup_env.log"

mkdir -p /workspace/logs
exec > >(tee -a "$LOG_FILE") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
trap 'status=$?; log "FAIL setup_env (exit $status)"; exit "$status"' ERR

log "Starting environment setup"
python_version="$(python3 --version 2>&1)"
log "$python_version"
if [[ "$python_version" != "Python 3.10."* ]]; then
  log "FAIL: Python 3.10 is required"
  exit 1
fi

if [[ -e "$REPO_DIR" && ! -d "$REPO_DIR/.git" ]]; then
  log "FAIL: $REPO_DIR exists but is not a Git checkout"
  exit 1
fi
if [[ ! -d "$REPO_DIR/.git" ]]; then
  log "Cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi
if [[ "$(git -C "$REPO_DIR" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
  if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    log "FAIL: instance checkout has local changes; cannot switch commit"
    exit 1
  fi
  git -C "$REPO_DIR" checkout --detach "$PINNED_COMMIT"
fi
[[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$PINNED_COMMIT" ]]
log "Pinned commit: $PINNED_COMMIT"

for spec in transformers==4.40.0 flash-attn==2.5.6 decord==0.6.0; do
  grep -Fxq "$spec" "$REPO_DIR/requirements.txt" || {
    log "FAIL: missing requirements pin $spec"
    exit 1
  }
done

mkdir -p /workspace/models/LLaVA-NeXT-7B-Video-FT /workspace/datasets/TVBench /workspace/outputs
# PyPI 'apex' is an unrelated package and fails to build; NVIDIA apex is optional in this code (guarded by try/except)
sed -i '/^apex==/d' "$REPO_DIR/requirements.txt"
log "Installing pinned requirements"
python3 -m pip install -r "$REPO_DIR/requirements.txt"
log "Installing mmcv-full==1.7.2"
python3 -m pip install mmcv-full==1.7.2 --no-build-isolation
log "PASS setup_env"
