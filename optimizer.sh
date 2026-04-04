#!/bin/bash

# Parallel layer jobs for simulator.py (run from TCAD repo root).

MAX_JOBS=6
comp=10.0
BW=25.0
batch=128
mesh_shapeX=4
mesh_shapeY=8
model="qwen"
LOG_DIR="logs/reasoning_${model}_${comp}TFLOPS_${BW}GBPS_for_${mesh_shapeX}*${mesh_shapeY}_mesh_${batch}_batches"
SCRIPT_PATH="simulator.py"
RESULT_DIR="results/reasoning_${model}_${comp}_TFLOPS_${BW}_GBPS_for_${mesh_shapeX}*${mesh_shapeY}_mesh_${batch}_batches"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULT_DIR"

for layer_id in {0..27}; do
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done

    echo "Starting layer_id = $layer_id"
    nohup python3 "$SCRIPT_PATH" \
        --layer-id "$layer_id" \
        --comp "$comp" \
        --comm "$BW" \
        --batch "$batch" \
        --mesh-shape "$mesh_shapeX" "$mesh_shapeY" \
        --model "$model" \
        > "${LOG_DIR}/log_layer_${layer_id}.txt" 2>&1 &
done

wait
echo "All jobs finished. Logs under $LOG_DIR/"
