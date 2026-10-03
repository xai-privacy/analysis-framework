#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-mlx-community/Llama-3.2-3B-Instruct-4bit}"
ITERS="${ITERS:-100}"
LR="${LR:-1e-4}"
LAYERS="${LAYERS:-16}"

mkdir -p adapters/direct adapters/rationale results

echo "=== Train DIRECT-answer adapter ==="
mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data ./data/direct \
  --mask-prompt \
  --iters "$ITERS" \
  --batch-size 1 \
  --num-layers "$LAYERS" \
  --learning-rate "$LR" \
  --steps-per-eval 20 \
  --save-every 20 \
  --adapter-path ./adapters/direct

echo "=== Train RATIONALE+answer adapter ==="
mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data ./data/rationale \
  --mask-prompt \
  --iters "$ITERS" \
  --batch-size 1 \
  --num-layers "$LAYERS" \
  --learning-rate "$LR" \
  --steps-per-eval 20 \
  --save-every 20 \
  --adapter-path ./adapters/rationale
