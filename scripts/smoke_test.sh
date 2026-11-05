#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/map-the-flow"
MODEL_DIR="/workspace/models/LLaVA-NeXT-7B-Video-FT"
DATA_DIR="/workspace/datasets/TVBench"
OUTPUT_DIR="/workspace/outputs/smoke_test"
RESULT="$OUTPUT_DIR/tvbench/effective-pathway-7b/LLaVA-NeXT-7B-Video-FT/jsons/06_Object Count.json"
LOG_FILE="/workspace/logs/smoke_test.log"

mkdir -p /workspace/logs
exec > >(tee -a "$LOG_FILE") 2>&1
log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
trap 'status=$?; log "FAIL smoke_test (exit $status)"; exit "$status"' ERR

log "Starting one Object Count sample"
[[ -s "$MODEL_DIR/config.json" ]] || { log "FAIL: missing model config.json"; exit 1; }
[[ -s "$DATA_DIR/json/object_count.json" ]] || { log "FAIL: missing Object Count annotations"; exit 1; }
[[ -d "$DATA_DIR/video/object_count" ]] || { log "FAIL: missing Object Count video folder"; exit 1; }

# Change only the instance checkout; this script is never run in the local repo.
CONFIG="$REPO_DIR/tasks/eval/config_dataset.py"
sed -i "s#'tvbench': \"/your/path/to/datasets/TVBench\"#'tvbench': \"$DATA_DIR\"#" "$CONFIG"
grep -Fq "'tvbench': \"$DATA_DIR\"" "$CONFIG" || {
  log "FAIL: TVBench path was not set in $CONFIG"
  exit 1
}
log "TVBench path set in instance checkout"

if [[ -s "$RESULT" ]]; then
  log "PASS smoke_test (existing result): $RESULT"
  exit 0
fi

cd "$REPO_DIR"
python3 analysis/effective_pathway_analysis.py \
  --output_dir "$OUTPUT_DIR" \
  --model_path "$MODEL_DIR" \
  --pooling_shape 8-12-12 \
  --conv_mode eval_mvbench \
  --dataset_name tvbench \
  --target effective-pathway-7b \
  --task_id 6 \
  --test_ratio 1

[[ -s "$RESULT" ]] || { log "FAIL: missing result JSON: $RESULT"; exit 1; }
log "PASS smoke_test: $RESULT"
