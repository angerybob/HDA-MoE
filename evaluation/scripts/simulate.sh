#!/usr/bin/env bash
# qwen / ds / mixtral hardware-aware gating simulations (configs driven by for-loops)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE="TCAD/expert_trace"

run_sim() {
  local scores_npz=$1
  local out_dir=$2
  local suffix=$3
  local reward_comp=$4
  local reward_comm=$5
  local top_k=$6
  local model_name=$7
  local hd_mesh_rows=$8
  local hd_mesh_cols=$9
  local hd_comp=${10}
  local hd_bw=${11}
  local max_tokens=${12}

  python3 simulate_hd_gating_from_scores.py \
    --scores-npz "$scores_npz" \
    --output-json "${out_dir}/experts_reasoning_hd_sim_${suffix}.json" \
    --reward-comp "$reward_comp" \
    --reward-comm "$reward_comm" \
    --top-k "$top_k" \
    --model-name "$model_name" \
    --hd-mesh-rows "$hd_mesh_rows" \
    --hd-mesh-cols "$hd_mesh_cols" \
    --hd-comp "$hd_comp" \
    --hd-bw "$hd_bw" \
    --max-tokens "$max_tokens"
}

# Each line: suffix rows cols hd_comp hd_bw
qwen_mesh_cfgs=(
  "2_5_75_0 4 8 2500000000000 75000000000"
  "5_0_50_0 4 8 5000000000000 50000000000"
  "10_0_25_0 4 8 10000000000000 25000000000"
  "4_4 4 4 2500000000000 75000000000"
  "8_8 8 8 2500000000000 75000000000"
)

# ds / mixtral share the same mesh sweep (4_4 / 8_8 use 5T / 50G)
ds_mixtral_mesh_cfgs=(
  "10_0_25_0 4 8 10000000000000 25000000000"
  "5_0_50_0 4 8 5000000000000 50000000000"
  "2_5_75_0 4 8 2500000000000 75000000000"
  "4_4 4 4 5000000000000 50000000000"
  "8_8 8 8 5000000000000 50000000000"
)

# ---------- qwen ----------
for cfg in "${qwen_mesh_cfgs[@]}"; do
  read -r suffix rows cols comp bw <<<"$cfg"
  run_sim \
    "${BASE}/qwen/score/reconstructed_softmax_reasoning.npz" \
    "${BASE}/qwen/hd_gating" \
    "$suffix" \
    -130000 -0.001 8 qwen \
    "$rows" "$cols" "$comp" "$bw" \
    512
done

# ---------- ds ----------
for cfg in "${ds_mixtral_mesh_cfgs[@]}"; do
  read -r suffix rows cols comp bw <<<"$cfg"
  run_sim \
    "${BASE}/ds/score/reconstructed_softmax_reasoning.npz" \
    "${BASE}/ds/hd_gating" \
    "$suffix" \
    -30000 -0.025 6 ds \
    "$rows" "$cols" "$comp" "$bw" \
    256
done

# ---------- mixtral ----------
for cfg in "${ds_mixtral_mesh_cfgs[@]}"; do
  read -r suffix rows cols comp bw <<<"$cfg"
  run_sim \
    "${BASE}/mixtral/score/reconstructed_softmax_reasoning.npz" \
    "${BASE}/mixtral/hd_gating" \
    "$suffix" \
    -1160000 -0.3 2 mixtral \
    "$rows" "$cols" "$comp" "$bw" \
    512
done
