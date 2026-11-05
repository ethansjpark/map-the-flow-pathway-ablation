#!/usr/bin/env bash
set -u
cd /workspace/map-the-flow
for T in 6 5 8 3; do
  for TARGET in effective-pathway-7b random; do
    EXTRA=""
    [ "$TARGET" = "random" ] && EXTRA="--random_block_ratio 0.58"
    echo "[$(date -u +%FT%T)] START task=$T target=$TARGET"
    python3 analysis/effective_pathway_analysis.py --output_dir /workspace/outputs/reproduction --model_path /workspace/models/LLaVA-NeXT-7B-Video-FT --pooling_shape 8-12-12 --conv_mode eval_mvbench --dataset_name tvbench --target $TARGET $EXTRA --task_id $T --test_ratio -1 || echo "FAIL task=$T target=$TARGET"
    echo "[$(date -u +%FT%T)] DONE task=$T target=$TARGET"
  done
done
echo "[$(date -u +%FT%T)] ALL DONE reproduction"
