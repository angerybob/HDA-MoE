#!/usr/bin/env bash
# MT-Bench: run the first 6 questions per category (question.jsonl: 10 lines per category).
# --question-begin/--question-end are 0-based with end exclusive → 0,6 / 10,16 / … / 70,76.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_JUDGE_DIR="/data/home/haochenhuang/TCAD/fastchat/fastchat/llm_judge"
MODEL_PATH="/opt/models/Mixtral-8x7B-Instruct-v0.1"
TRACE_DIR="/data/home/haochenhuang/TCAD/expert_trace/mixtral/score"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

cd "$LLM_JUDGE_DIR"

# Format: category_name:begin:end (same line order as FastChat MT-Bench question.jsonl).
CATEGORIES=(
  "writing:1:6"
  "roleplay:11:16"
  "reasoning:21:26"
  "math:31:36"
  "coding:41:46"
  "extraction:51:56"
  "stem:61:66"
  "humanities:71:76"
)

for spec in "${CATEGORIES[@]}"; do
  IFS=: read -r name begin end <<< "$spec"
  model_id="mixtral_${name}"
  trace="${TRACE_DIR}/experts_${name}_score.json"

  echo "========== ${name} questions [${begin}, ${end}) model_id=${model_id} =========="
  python3 gen_model_answer.py \
    --model-path "$MODEL_PATH" \
    --model-id "$model_id" \
    --no-hd-gating \
    --question-begin "$begin" \
    --question-end "$end" \
    --num-gpus-per-model 2 \
    --num-gpus-total 2 \
    --trace "$trace" \
    --batch 1 \
    --trace-gating-softmax
done

echo "All categories finished."
