# Evaluation scripts

Python and shell utilities under `TCAD/evaluation/scripts` for MoE placement, routing traces, and latency studies. Most Python entry points append `TCAD/` and `TCAD/evaluation/` to `sys.path`; run them from anywhere, but set `--cwd` / `--deployment-root` so `expert_trace/` and `results/` resolve correctly.

---

## Core workflows

### `sim.py`

**Purpose:** Quick **hybrid parallel** evaluation—load routing traces and NPZ placements, compare TP/EP vs node/link-balanced latency under static vs dynamic expert routing (first-pass sanity check for placement quality).

**Typical inputs:** Trace at `<deployment-root>/<trace-subdir>/experts_<dataset>_<model>.json` (defaults assume a separate `deployment` tree; override for your layout).

**Example (trace at `expert_trace/<model>/experts_<dataset>_<model>.json`):**

```bash
cd /data/home/haochenhuang/TCAD
python3 evaluation/scripts/sim.py \
  --deployment-root /data/home/haochenhuang/TCAD \
  --trace-subdir expert_trace/ds \
  --cwd . \
  --model ds \
  --dataset reasoning \
  --mesh 4 8 \
  --results-json evaluation/results/result.json
```

The default `--deployment-root` points at a separate `deployment` tree; override it whenever your traces live under the TCAD repo as above.

---



### `e2e_hda.py`

**Purpose:** End-to-end **time-between-tokens (TBT)**-style latency comparison: baseline trace vs **hardware-aware (HDA)** adaptive trace; appends/writes JSON (default `evaluation/results/result_hda_e2e4.json`).

**Note:** The HDA trace path depends on mesh and `(comp, bw)`; see the script’s `experts_*_hd_sim_*.json` resolution logic.

**Example:**

```bash
cd /data/home/haochenhuang/TCAD
python3 evaluation/scripts/e2e_hda.py \
  --cwd . \
  --model ds \
  --batch 32 \
  --mesh 4 8 \
  --comp 2.5 \
  --bw 75 \
  --results-json evaluation/results/result_hda_e2e4.json
```

---


### `ablation.py`

**Purpose:** Ablation study—quantify how individual design axes (e.g., TP/EP vs compute-only balance vs node/link balance) affect outcomes using NPZ placements and a **single** static routing trace.

**Typical inputs:** `expert_trace/<model>/experts_<dataset>_<model>.json`, placement NPZs under the results tree (see `--mesh-batch-label`).

**Example:**

```bash
cd /data/home/haochenhuang/TCAD
python3 evaluation/scripts/ablation.py \
  --cwd . \
  --model ds \
  --dataset reasoning \
  --mesh 8 8 \
  --batch 128 \
  --mesh-batch-label 128 \
  --results-json evaluation/results/result_ablation.json
```

---

### `dynamic.py`

**Purpose:** Evaluate **dynamic scheduling**—compare link-side behavior when using **selected** vs **predicted** experts (pre-broadcast chunk size `k`), using paired traces (`experts_<dataset>_<model>_pre.json` and a second dataset for sampling).

**Example:**

```bash
cd /data/home/haochenhuang/TCAD
python3 evaluation/scripts/dynamic.py \
  --cwd . \
  --model ds \
  --dataset reasoning \
  --compare-dataset roleplay \
  --batch 512 \
  --mesh 4 8 \
  --results-json evaluation/results/result2_dynamic.json
```

---

### `adaptive.py`

**Purpose:** Compare latency under **adaptive** routing traces for static vs dynamic vs pre-broadcast vs **hardware-aware** deployment; stresses how hardware-aware gating behaves when routing changes and supports **small batch** settings (`--batch`).

**Typical inputs:** Default trace `expert_trace/<model>/adaptive/experts_<dataset>_<model>_adaptive.json`, or `--trace-path`.

**Example:**

```bash
cd /data/home/haochenhuang/TCAD
python3 evaluation/scripts/adaptive.py \
  --cwd . \
  --model mixtral \
  --batch 32 \
  --mesh 4 8 \
  --results-json evaluation/results/result_adaptive.json
```

---



### `run_gen_model_answer_mtbench_categories.sh`

**Purpose:** **Collect real MoE gating traces** on MT-Bench: runs the first six questions per category via FastChat `gen_model_answer.py`, with `--trace-gating-softmax` and per-category score traces.

**You should edit** `MODEL_PATH`, `TRACE_DIR`, `LLM_JUDGE_DIR`, and GPU settings at the top of the script for your machine.

**Example:**

```bash
bash /data/home/haochenhuang/TCAD/evaluation/scripts/run_gen_model_answer_mtbench_categories.sh
```

---

### `simulate_hd_gating_from_scores.py`

**Purpose:** **Simulate hardware-aware MoE gating (HD-MoE)** on CPU/CUDA from reconstructed per-layer softmax scores (NPZ), matching FastChat’s `moe_gating_hd.apply_hd_moe_routing`. Outputs JSON aligned with `experts_*_score.json` (`original_selected_experts`, `selected_experts`, optional metadata).

**Requires:** Compatible P-matrix NPZ layout as expected by `moe_gating_hd` (see repo docs / `TCAD/results/...`).

**Example:**

```bash
cd /data/home/haochenhuang/TCAD
python3 evaluation/scripts/simulate_hd_gating_from_scores.py \
  --scores-npz expert_trace/qwen/score/reconstructed_softmax_reasoning.npz \
  --output-json expert_trace/qwen/hd_gating/experts_reasoning_hd_sim_custom.json \
  --model-name qwen \
  --top-k 8 \
  --reward-comp -130000 \
  --reward-comm -0.001 \
  --hd-mesh-rows 4 --hd-mesh-cols 8 \
  --hd-comp 2500000000000 \
  --hd-bw 75000000000 \
  --max-tokens 512 \
  --device cpu
```

---

### `simulate.sh`

**Purpose:** Batch driver for **hardware-aware gating simulation**: loops over mesh / `(hd_comp, hd_bw)` presets and calls `simulate_hd_gating_from_scores.py` for Qwen, DeepSeek, and Mixtral.

**Important:** The script `cd`s to its own directory and uses `BASE="TCAD/expert_trace"` as a **relative** path. Either run from a layout where `evaluation/scripts/TCAD/expert_trace` exists, or **edit `BASE`** to an absolute path such as `/data/home/haochenhuang/TCAD/expert_trace`.

**Example:**

```bash
bash /data/home/haochenhuang/TCAD/evaluation/scripts/simulate.sh
```

---
