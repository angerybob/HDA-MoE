# HDA-MoE: Hybrid Parallelism and Dynamic, Adaptive Scheduling for Mixture-of-Experts with 3D Near-Memory Processing

This repository contains the **source code** for the **TCAD** journal manuscript **HDA-MoE** (HDA-MoE: Hybrid Parallelism and Dynamic, Adaptive Scheduling for Mixture-of-Experts with 3D Near-Memory Processing). The work is a **journal extension** of **HD-MoE** presented at **IEEE/ACM ICCAD 2025**.

**Related links**

- ICCAD 2025 code release (conference version): [code](https://github.com/angerybob/HD-MoE)  
- ICCAD 2025 paper (IEEE Xplore): [paper](https://ieeexplore.ieee.org/abstract/document/11240984)

## Paper overview

Mixture-of-Experts (MoE) large language models improve efficiency but remain sensitive to **memory bandwidth** and **parallel mapping** on distributed near-memory systems. This line of work studies **MoE inference on 3D near-memory processing (3D NMP)** and proposes:

1. **Offline hybrid parallel mapping** — combining **tensor parallelism (TP)** and **expert parallelism (EP)** to balance compute load and communication.  
2. **Online dynamic scheduling** — adapting to **time-varying expert activation** (e.g., pre-broadcast and prediction-aware schedules).  
3. **Hardware-aware gating** — routing experts using a policy that reflects **compute capability** and **interconnect bandwidth**, so gating decisions align with the deployed hardware.

The TCAD manuscript extends the ICCAD HD-MoE study with the above **hardware-aware gating** angle and the corresponding **accuracy** and **latency** evaluations reported in the paper.

## What this repository implements

| Topic | Location / mechanism |
|--------|----------------------|
| **Hybrid parallel deployment** (node–link balance, placement generation) | `optimizer.sh` → `simulator.py`; core optimizer in `node_allocation.py` (`MoE3DPNMOptimizer`) |
| **Dynamic scheduling** (evaluation vs static / predicted routing) | `evaluation/scripts/dynamic.py`, and related drivers |
| **Hardware-aware gating** (simulation + trace pipeline) | `evaluation/scripts/simulate_hd_gating_from_scores.py`, `trace_gating_softmax_to_npz.py`, `fastchat/fastchat/llm_judge/moe_gating_hd.py`, plus `e2e_hda.py` / `e2e_hda.py`, `adaptive.py`, etc. |

**Hybrid baseline (comparison in experiments)**  
The directory [`hybrid_baseline/`](hybrid_baseline/) holds scripts used to generate **hybrid-baseline** deployment strategies for evaluation (e.g. compute-balance–only placement). **`hybrid_baseline/comp_bal.sh`** and **`hybrid_baseline/gen_comp_balance.py`** currently contain **machine-specific paths** (`sys.path`, log/result directories); **edit them** to point at your checkout (e.g. this repo’s root so `node_allocation` resolves, and your desired `logs/` / `results/` locations).

**HumanEval accuracy**  
[`human-eval/`](human-eval/) is a fork of the official [openai/human-eval](https://github.com/openai/human-eval) harness, extended with helpers for our answer format. See [`human-eval/EVAL_HUMANEVAL.md`](human-eval/EVAL_HUMANEVAL.md) for converting outputs and running `evaluate_functional_correctness`.

**Task accuracy with hardware-aware gating (MT-Bench, GSM8K, ARC, etc.)**  
[`fastchat/`](fastchat/) is based on [lm-sys/FastChat](https://github.com/lm-sys/FastChat) with changes for **MoE trace collection** and **hardware-aware gating** during generation. See [`fastchat/fastchat/llm_judge/README.md`](fastchat/fastchat/llm_judge/README.md) (section *Hardware-aware gating accuracy*).

**Other evaluations and figures**  
[`evaluation/`](evaluation/) contains placement utilities, plotting scripts under `evaluation/draw/`, and documented entry points under [`evaluation/scripts/`](evaluation/scripts/README.md).

## Quick start

### 1. Environment

```bash
conda create -n hda python=3.10
conda activate hda
```

Clone **this** repository and initialize submodules (`fastchat`, `human-eval`):

```bash
git clone --recursive git@github.com:angerybob/HDA-MoE.git
cd HDA-MoE
# If you already cloned without --recursive:
git submodule update --init --recursive
```

Install PyTorch (match your CUDA stack if needed), then root dependencies:

```bash
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
```

Install the FastChat subtree:

```bash
cd fastchat
git checkout adaptive_gating
pip install -e ".[model_worker,llm_judge]"
cd ..
```

**Note:** `node_allocation.py` uses **Gurobi** (`gurobipy` in `requirements.txt`). You need a valid **Gurobi license** for placement optimization.

### 2. Generate hybrid parallel deployment strategies

From the **repository root**, `optimizer.sh` launches **parallel** jobs that run `simulator.py` **per layer** (configurable `MAX_JOBS`, `comp`, `BW`, `batch`, `mesh_shapeX` / `mesh_shapeY`, `model`, and `layer_id` range). Example:

```bash
# Optional: run in background and log
nohup bash optimizer.sh > script.log 2>&1 &
```

**Outputs**

- Strategy artifacts: under `results/` (see `RESULT_DIR` in `optimizer.sh`).  
- Per-layer logs: under `logs/` (see `LOG_DIR` in `optimizer.sh`).

**Configuration**

Edit `optimizer.sh` for hardware (`comp`, `BW`), mesh (`mesh_shapeX`, `mesh_shapeY`), workload (`batch`), model name (`model`), and the **`for layer_id in {0..N}`** range so **N matches the MoE layer count** of your target model (e.g. Qwen-style configs use a different depth than Mixtral).

### 3. Evaluate deployments, dynamic scheduling, and HDA gating

Run scripts from the repo root (or as documented); many tools resolve `expert_trace/` and `results/` via flags such as `--cwd` and `--deployment-root`. **Authoritative examples** are in [`evaluation/scripts/README.md`](evaluation/scripts/README.md).


## Core modules (repo root)

- **`node_allocation.py`** — `MoE3DPNMOptimizer` and node–link balance optimization.  
- **`simulator.py`** — layer-wise simulation and optimization driver for 3D NMP MoE inference (compute + communication modeling).  
- **`baseline.py`** — baselines (TP, EP, hybrid TP–EP) for comparison.  
- **`expert_trace/`** — expert activation / gating statistics for supported models (e.g. Mixtral, DeepSeek, Qwen); add traces here to study new models.

## Supported models and datasets

- **Models:** MoE LLMs supported by the bundled traces and FastChat paths (e.g. Qwen, Mixtral, DeepSeek); extend by adding traces under `expert_trace/` and wiring models in the FastChat / eval scripts.  
- **Datasets:** MT-Bench-style multi-turn evaluation via FastChat; code benchmarks (HumanEval) via `human-eval/`; additional sets (GSM8K, ARC, etc.) as described in `fastchat/fastchat/llm_judge/README.md`.

## Citation

If you use this code, please cite the **TCAD** manuscript when available, and the **ICCAD 2025** HD-MoE paper:

```bibtex
@INPROCEEDINGS{11240984,
  author={Huang, Haochen and Zhong, Shuzhang and Zhang, Zhe and Li, Shuangchen and Niu, Dimin and Zheng, Hongzhong and Wang, Runsheng and Li, Meng},
  booktitle={2025 IEEE/ACM International Conference On Computer Aided Design (ICCAD)}, 
  title={HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing}, 
  year={2025},
  volume={},
  number={},
  pages={1-9},
  keywords={Costs;Three-dimensional displays;Tensors;Computational modeling;Memory management;Bandwidth;Parallel processing;Dynamic scheduling;Distance measurement;Computational efficiency;Automated Deployment;Mixture-of-Experts;3D Near-Memory Processing},
  doi={10.1109/ICCAD66269.2025.11240984}}
```


## License

This repository is licensed under the MIT License — see [`LICENSE`](LICENSE). Submodule trees retain their own terms: [`fastchat/LICENSE`](fastchat/LICENSE) (Apache-2.0) and [`human-eval/LICENSE`](human-eval/LICENSE) (MIT, OpenAI).
