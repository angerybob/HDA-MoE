"""End-to-end latency: baseline trace vs hardware-aware adaptive trace; writes result_hda_e2e4.json."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
from tqdm import tqdm

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_eval_dir = os.path.dirname(_scripts_dir)
_project_root = os.path.dirname(_eval_dir)
sys.path.append(_project_root)
sys.path.append(_eval_dir)

from node_allocation import MoE3DPNMOptimizer
from moe_placement_utils import EP_deployment, generate_random_placement

# Per-device HBM bandwidth (bytes/s); aggregate across mesh = D * BWMEM_PER_DEVICE.
BWMEM_PER_DEVICE = 625e9

# Qwen3.5-35B-A3B text_config (see models/Qwen3.5-35B-A3B/config.json).
QWEN35_NUM_HEADS = 16
QWEN35_NUM_KV_HEADS = 2
QWEN35_HEAD_DIM = 256
QWEN35_LINEAR_K_HEADS = 16
QWEN35_LINEAR_V_HEADS = 32
QWEN35_LINEAR_K_DIM = 128
QWEN35_LINEAR_V_DIM = 128
QWEN35_LINEAR_CONV_KERNEL = 4
QWEN35_FULL_ATTN_LAYERS = 10
QWEN35_LINEAR_ATTN_LAYERS = 30
QWEN35_PARTIAL_ROPE = 0.25


def _mem_bw_aggregate(D: int, bw_per_device: float = BWMEM_PER_DEVICE) -> float:
    """Aggregate memory bandwidth: 625 GB/s per device × D devices."""
    return D * bw_per_device


def _moe_expert_weight_mem_layer(
    f_row: np.ndarray, batch: int, intermediate: float, h: int, D: int, bw_per_device: float
) -> float:
    """MoE expert weight traffic for one layer; floor(f*B)>0 experts, D-way aggregate BW."""
    n_active = int(np.sum(np.floor(f_row * batch) > 0))
    nbytes = n_active * 2 * intermediate * h**2
    return nbytes / _mem_bw_aggregate(D, bw_per_device)


def _qwen35_full_attn_comp_layer(
    batch: int, h: int, decode_L: int, D: int, comp: float
) -> float:
    """Full-attention layer: GQA decode step with KV length decode_L."""
    kv_dim = QWEN35_NUM_KV_HEADS * QWEN35_HEAD_DIM
    attn_dim = QWEN35_NUM_HEADS * QWEN35_HEAD_DIM
    flops = (
        2 * batch * h * (2 * attn_dim)  # q_proj (query + gate)
        + 2 * batch * h * kv_dim  # k_proj
        + 2 * batch * h * kv_dim  # v_proj
        + 2 * batch * h * h  # o_proj
        + 2 * batch * attn_dim * decode_L  # QK^T
        + 2 * batch * attn_dim * decode_L  # attn @ V
    )
    return flops / (D * comp)


def _qwen35_linear_attn_comp_layer(batch: int, h: int, D: int, comp: float) -> float:
    """Linear-attention (GatedDeltaNet) layer: single-token decode with recurrent state."""
    key_dim = QWEN35_LINEAR_K_HEADS * QWEN35_LINEAR_K_DIM
    value_dim = QWEN35_LINEAR_V_HEADS * QWEN35_LINEAR_V_DIM
    conv_dim = 2 * key_dim + value_dim
    proj_flops = (
        2 * batch * h * (2 * key_dim + value_dim)  # in_proj_qkv
        + 2 * batch * h * value_dim  # in_proj_z
        + 4 * batch * h * QWEN35_LINEAR_V_HEADS  # in_proj_b, in_proj_a
        + 2 * batch * h * h  # out_proj
    )
    conv_flops = 2 * batch * conv_dim * QWEN35_LINEAR_CONV_KERNEL
    recurrent_flops = (
        2 * batch * QWEN35_LINEAR_V_HEADS * QWEN35_LINEAR_K_DIM * QWEN35_LINEAR_V_DIM * 6
    )
    return (proj_flops + conv_flops + recurrent_flops) / (D * comp)


def _qwen35_full_attn_mem_layer(
    batch: int, h: int, decode_L: int, D: int, bw_per_device: float
) -> float:
    kv_dim = QWEN35_NUM_KV_HEADS * QWEN35_HEAD_DIM
    nbytes = (
        batch * h * 2  # hidden activation R/W
        + batch * kv_dim * decode_L * 2  # K,V cache read
        + batch * kv_dim * decode_L * QWEN35_PARTIAL_ROPE  # partial-RoPE KV write
        + h * (batch + h)  # RMSNorm + small projection traffic
    )
    return nbytes / _mem_bw_aggregate(D, bw_per_device)


def _qwen35_linear_attn_mem_layer(batch: int, h: int, D: int, bw_per_device: float) -> float:
    key_dim = QWEN35_LINEAR_K_HEADS * QWEN35_LINEAR_K_DIM
    value_dim = QWEN35_LINEAR_V_HEADS * QWEN35_LINEAR_V_DIM
    conv_dim = 2 * key_dim + value_dim
    nbytes = (
        batch * h * 2
        + batch * QWEN35_LINEAR_V_HEADS * QWEN35_LINEAR_K_DIM * QWEN35_LINEAR_V_DIM * 2
        + batch * conv_dim * (QWEN35_LINEAR_CONV_KERNEL - 1)
        + h * batch
    )
    return nbytes / _mem_bw_aggregate(D, bw_per_device)


def _total_att_mem_qwen35(
    optimizer: MoE3DPNMOptimizer,
    batch: int,
    h: int,
    intermediate: float,
    decode_L: int,
    D: int,
    bw_per_device: float,
) -> float:
    full_comp = _qwen35_full_attn_comp_layer(batch, h, decode_L, D, optimizer.comp)
    linear_comp = _qwen35_linear_attn_comp_layer(batch, h, D, optimizer.comp)
    full_mem = _qwen35_full_attn_mem_layer(batch, h, decode_L, D, bw_per_device)
    linear_mem = _qwen35_linear_attn_mem_layer(batch, h, D, bw_per_device)
    attn = QWEN35_FULL_ATTN_LAYERS * (full_comp + full_mem) + QWEN35_LINEAR_ATTN_LAYERS * (
        linear_comp + linear_mem
    )
    moe_w = 0.0
    for layer_id in range(optimizer.layer):
        moe_w += _moe_expert_weight_mem_layer(
            optimizer.f[layer_id], batch, intermediate, h, D, bw_per_device
        )
    return attn + moe_w


def _layer_mem_without_moe_weights(
    model: str,
    batch: int,
    h: int,
    intermediate: float,
    decode_L: int,
    D: int,
    BWmem: float,
) -> float:
    """Per-layer memory traffic excluding MoE expert weight loads."""
    if model == "ds":
        nbytes = (
            2048 * (3072 + batch)
            + 2048 * (2048 + batch)
            + 512 * 4096
            + 512 * 2 * decode_L
            + batch * h * (intermediate + 1)
        )
    elif model == "mixtral":
        nbytes = (
            2 * batch * h * decode_L
            + 3 * h * (batch + h)
            + h * (batch + h)
            + batch * h * (intermediate + 1)
        )
    elif model in ("qwen", "qwen35"):
        nbytes = (
            2 * batch * h * 0.25 * decode_L
            + 1.5 * h * (batch + h)
            + 0.25 * h * (batch + h)
            + batch * h * (intermediate + 1)
        )
    else:
        raise ValueError(f"unknown model: {model}")
    return nbytes / _mem_bw_aggregate(D, BWmem)


def _total_att_mem(
    optimizer: MoE3DPNMOptimizer,
    model: str,
    batch: int,
    h: int,
    intermediate: float,
    decode_L: int,
    D: int,
    bw_per_device: float,
    att_comp_layer: float,
) -> float:
    base_mem = _layer_mem_without_moe_weights(model, batch, h, intermediate, decode_L, D, bw_per_device)
    total = 0.0
    for layer_id in range(optimizer.layer):
        moe_w = _moe_expert_weight_mem_layer(
            optimizer.f[layer_id], batch, intermediate, h, D, bw_per_device
        )
        total += base_mem + att_comp_layer + moe_w
    return total


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end evaluation with hardware-aware (HDA) routing traces."
    )
    p.add_argument("--batch", type=int, default=32, help="batch size")
    p.add_argument(
        "--model",
        type=str,
        default="ds",
        choices=["mixtral", "ds", "qwen", "qwen35"],
        help="model name",
    )
    p.add_argument(
        "--trace-path",
        default=None,
        help="Routing trace JSON (optimizer flat format); skips hd_gating file lookup",
    )
    p.add_argument(
        "--mesh",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=[4, 8],
        help="2D mesh shape (X Y)",
    )
    p.add_argument(
        "--comp",
        type=float,
        default=2.5,
        help="compute capability in TFLOPS (used in trace path and optimizer)",
    )
    p.add_argument(
        "--bw",
        type=float,
        default=75.0,
        help="interconnect bandwidth in GB/s",
    )
    p.add_argument(
        "--cwd",
        default=".",
        help="Working directory for relative expert_trace/ and results/ paths",
    )
    p.add_argument(
        "--results-json",
        default="evaluation/results/result_hda_e2e4.json",
        help="Output JSON path (relative to cwd unless absolute)",
    )
    p.add_argument(
        "--topology",
        choices=["mesh", "torus", "fat_tree"],
        default="mesh",
        help="Physical topology used by communication simulation.",
    )
    p.add_argument(
        "--fat-tree-oversubscription",
        type=float,
        default=1.0,
        help="Fat-tree agg-core oversubscription ratio; 1.0 is full-bisection.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling trace rows and random EP placement.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    root = os.path.abspath(args.cwd)
    batch = args.batch
    model = args.model
    dataset_list = ["reasoning", "math", "coding", "writing", "roleplay"]
    dataset = dataset_list[0]
    adaptive_trace_tag = dataset_list[0]
    mesh_shape = tuple(args.mesh)
    comp = args.comp
    BW = args.bw
    hd_sim_extra = 0
    if model == "mixtral":
        E, e, SE, h, IS, mlp_first, num_layers = 8, 2, 0, 4096, 14336, False, 32
    elif model == "ds":
        E, e, SE, h, IS, mlp_first, num_layers = 64, 6, 0, 2048, 1408, True, 26
    elif model == "qwen":
        E, e, SE, h, IS, mlp_first, num_layers = 64, 8, 0, 3584, 2560, False, 28
    elif model == "qwen35":
        E, e, SE, h, IS, mlp_first, num_layers = 256, 8, 0, 2048, 512, False, 40
    D = mesh_shape[0] * mesh_shape[1]

    if args.trace_path:
        data_path = args.trace_path if os.path.isabs(args.trace_path) else os.path.join(root, args.trace_path)
        print(data_path)
        try:
            with open(data_path, encoding="utf-8") as f:
                trace_payload = json.load(f)
                if (
                    isinstance(trace_payload, dict)
                    and "original_selected_experts" in trace_payload
                    and "selected_experts" in trace_payload
                ):
                    data = trace_payload["original_selected_experts"]
                    adaptive_sample = trace_payload["selected_experts"]
                else:
                    data = trace_payload
                    adaptive_sample = trace_payload
        except FileNotFoundError:
            print("File not found; check path and filename.")
            sys.exit(1)
    else:
        _mx, _my = mesh_shape[0], mesh_shape[1]
        if (_mx, _my) in ((4, 4), (8, 8)):
            rel = f"expert_trace/{model}/hd_gating/experts_{dataset}_hd_sim_{_mx}_{_my}.json"
        else:
            _comp_tag = str(float(comp)).replace(".", "_")
            _bw_f = float(BW)
            _bw_tag = str(int(_bw_f)) if _bw_f.is_integer() else str(BW).replace(".", "_")
            rel = f"expert_trace/{model}/hd_gating/experts_{dataset}_hd_sim_{_comp_tag}_{_bw_tag}_{hd_sim_extra}.json"
        data_path = os.path.join(root, rel)
        print(data_path)
        try:
            with open(data_path, encoding="utf-8") as f:
                data1 = json.load(f)
                data, adaptive_sample = (
                    data1["original_selected_experts"],
                    data1["selected_experts"],
                )
        except FileNotFoundError:
            print("File not found; check path and filename.")
            sys.exit(1)

    optimizer = MoE3DPNMOptimizer(
        E=E,
        e=e,
        h=h,
        IS=IS,
        B=batch,
        D=D,
        BW=BW * 1e9,
        comp=comp * 1e12,
        num_layers=num_layers,
        mlp_first=mlp_first,
        routing_trace=data,
        topology_type=args.topology,
        topology_config={"oversubscription": args.fat_tree_oversubscription} if args.topology == "fat_tree" else None,
    )
    optimizer_adaptive = MoE3DPNMOptimizer(
        E=E,
        e=e,
        h=h,
        IS=IS,
        B=batch,
        D=D,
        BW=BW * 1e9,
        comp=comp * 1e12,
        num_layers=num_layers,
        mlp_first=mlp_first,
        routing_trace=adaptive_sample,
        topology_type=args.topology,
        topology_config={"oversubscription": args.fat_tree_oversubscription} if args.topology == "fat_tree" else None,
    )
    P_tp = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
    P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)

    optimizer.X, optimizer.Y = mesh_shape
    decode_L = 64
    intermediate = optimizer.IS / optimizer.h
    BWmem = 625e9
    att_comp = {
        "ds": (2048 * 2048 + 512 * 4096 + 2048 * 3072 + 2048 * 2816 * 2 + 2048 * 2 * decode_L * 2)
        * batch
        / (D * optimizer.comp),
        "mixtral": (3 * batch * h**2 + h**2 * batch + batch * h * decode_L * 2) / (D * optimizer.comp),
        "qwen": (
            batch * h**2
            + 2 * batch * 0.25 * h**2
            + 0.25 * h**2 * batch
            + 20480 * batch * h * 2
            + 0.25 * batch * h * decode_L * 2
        )
        / (D * optimizer.comp),
    }

    if model == "qwen35":
        att_mem = _total_att_mem_qwen35(
            optimizer, batch, h, intermediate, decode_L, D, BWmem
        )
    else:
        att_mem = _total_att_mem(
            optimizer, model, batch, h, intermediate, decode_L, D, BWmem, att_comp[model]
        )

    t_inf = (2 * e * batch * intermediate * h**2) / (D * optimizer.comp) + att_mem
    k = 1
    while optimizer.optimal_broadcast_chunk(k=k) < t_inf:
        k += 1
    k = max(k - 1, 0)
    _n0 = len(data[str(1)])
    sample_id = random.sample(range(_n0), min(batch, _n0))

    tp_comp = 0
    tp_comm = 0
    ep_comp = 0
    ep_comm = 0
    comp_comp = 0
    comp_comm = 0
    comp = 0
    comm_link = 0
    adaptive_pre_comp = 0
    adaptive_pre_comm = 0

    res_dir = os.path.join(root, "results")
    base_subdir = (
        f"{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_"
        f"for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches"
    )
    subdir = base_subdir

    for layer_id in tqdm(range(optimizer.layer)):
        file_path = os.path.join(
            res_dir,
            subdir,
            f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz",
        )
        loaded_arrays = np.load(file_path)

        comp_path = os.path.join(
            res_dir,
            "comp_balance_only",
            base_subdir,
            f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz",
        )
        P = loaded_arrays["arr1"]
        M = loaded_arrays["arr2"]
        loaded_comp_arrays = np.load(comp_path)
        P_comp = loaded_comp_arrays["arr1"]
        M_comp = loaded_comp_arrays["arr2"]

        sample_key = str(layer_id + optimizer.mlp_first)
        indices = [sid % len(data[sample_key]) for sid in sample_id]
        raw_random_samples = [data[sample_key][i] for i in indices]
        adaptive_indices = [sid % len(adaptive_sample[sample_key]) for sid in sample_id]
        raw_adaptive_random_samples = [adaptive_sample[sample_key][i] for i in adaptive_indices]

        scaling = optimizer.B / len(raw_random_samples) if len(raw_random_samples) > 0 else 0
        adaptive_scaling = (
            optimizer.B / len(raw_adaptive_random_samples) if len(raw_adaptive_random_samples) > 0 else 0
        )

        random_samples = (
            raw_random_samples
            if (len(raw_random_samples) > 0 and isinstance(raw_random_samples[0], (list, tuple)))
            else [raw_random_samples]
        )
        adaptive_random_samples = (
            raw_adaptive_random_samples
            if (
                len(raw_adaptive_random_samples) > 0
                and isinstance(raw_adaptive_random_samples[0], (list, tuple))
            )
            else [raw_adaptive_random_samples]
        )

        comp_map = np.zeros((optimizer.E,))
        adaptive_comp_map = np.zeros((optimizer.E,))
        for sublist in random_samples:
            comp_map[sublist] += 2 * optimizer.h * optimizer.IS * scaling
        for sublist in adaptive_random_samples:
            adaptive_comp_map[sublist] += 2 * optimizer.h * optimizer.IS * adaptive_scaling

        M_rand = generate_random_placement(optimizer.D, mesh_shape)
        optimizer.M = M_rand

        tp_comp += optimizer.compute_time(P_tp)[layer_id]
        tp_comm += 4 * optimizer.comm_time(P_tp)[layer_id]

        ep_comp += optimizer.compute_time(P_ep)[layer_id]
        comm_temp, _link = optimizer.comm_time_acc(M_rand, P_ep, layer_id)
        ep_comm += comm_temp * 2

        comp_comp += optimizer.compute_time(P_comp)[layer_id]
        comm_temp, _link = optimizer.comm_time_acc(M_comp, P_comp, layer_id)
        comm_temp += 2 * optimizer.comm_time(P)[layer_id]
        comp_comm += comm_temp * 2

        comp += optimizer.compute_time(P)[layer_id]

        comm_temp, _link = optimizer.comm_time_acc(M, P, layer_id)
        comm_link += comm_temp * 2

        adaptive_pre_comp += optimizer_adaptive.compute_time(P)[layer_id]
        adaptive_comm_temp, _link = optimizer_adaptive.comm_time_acc(M, P, layer_id)
        adaptive_pre_comm += adaptive_comm_temp * 2

    print(f"TP_communication: {tp_comm*1e6:.2f} us")
    print(f"TP_computation: {tp_comp*1e6:.2f} us")
    print(f"TP_latency: {(tp_comp+tp_comm+att_mem)*1e6:.2f} us")

    print(f"EP_communication: {ep_comm*1e6:.2f} us")
    print(f"EP_computation: {ep_comp*1e6:.2f} us")
    print(f"EP_latency: {(ep_comp+ep_comm+att_mem)*1e6:.2f} us")

    print(f"compute_balancing_communication: {comp_comm*1e6:.2f} us")
    print(f"compute_balancing_computation: {comp_comp*1e6:.2f} us")
    print(f"compute_balancing_latency: {(comp_comp+comp_comm+att_mem)*1e6:.2f} us")

    print(f"node_link_balancing_communication: {comm_link*1e6:.2f} us")
    print(f"node_link_balancing_computation: {comp*1e6:.2f} us")
    print(f"node_link_balancing_latency: {(comp+comm_link+att_mem)*1e6:.2f} us")

    print(f"node_link_balancing_speedup_EP:{(ep_comp+ep_comm+att_mem)/(comp+comm_link+att_mem):.2f}")
    print(f"node_link_balancing_speedup_TP:{(tp_comp+tp_comm+att_mem)/(comp+comm_link+att_mem):.2f}")
    print(f"node_link_balancing_speedup_comp:{(comp_comp+comp_comm+att_mem)/(comp+comm_link+att_mem):.2f}")

    print(f"adaptive_pre_communication: {adaptive_pre_comm*1e6:.2f} us")
    print(f"adaptive_pre_computation: {adaptive_pre_comp*1e6:.2f} us")
    print(f"adaptive_pre_latency: {(adaptive_pre_comp+adaptive_pre_comm+att_mem)*1e6:.2f} us")
    print(f"hd_speedup: {(comp+comm_link+att_mem)/(adaptive_pre_comp+adaptive_pre_comm+att_mem):.2f}")

    config = {
        "mesh_shape": mesh_shape,
        "model": model,
        "dataset": dataset,
        "sample": adaptive_trace_tag,
        "comp_TFLOPS": optimizer.comp * 1e-12,
        "BW_GBPS": optimizer.BW * 1e-9,
        "batch": optimizer.B,
        "topology": args.topology,
        "fat_tree_oversubscription": args.fat_tree_oversubscription
        if args.topology == "fat_tree"
        else None,
    }

    result = {
        "config": config,
        "TP": {
            "communication_us": round(tp_comm * 1e6, 2),
            "computation_us": round(tp_comp * 1e6, 2),
            "latency_us": round((tp_comp + tp_comm + att_mem) * 1e6, 2),
        },
        "EP": {
            "communication_us": round(ep_comm * 1e6, 2),
            "computation_us": round(ep_comp * 1e6, 2),
            "latency_us": round((ep_comp + ep_comm + att_mem) * 1e6, 2),
        },
        "compute_balancing": {
            "static": {
                "communication_us": round(comp_comm * 1e6, 2),
                "computation_us": round(comp_comp * 1e6, 2),
                "latency_us": round((comp_comp + comp_comm + att_mem) * 1e6, 2),
            },
        },
        "node_link_balancing": {
            "static": {
                "communication_us": round(comm_link * 1e6, 2),
                "computation_us": round(comp * 1e6, 2),
                "latency_us": round((comp + comm_link + att_mem) * 1e6, 2),
            },
        },
        "adaptive_pre": {
            "communication_us": round(adaptive_pre_comm * 1e6, 2),
            "computation_us": round(adaptive_pre_comp * 1e6, 2),
            "latency_us": round((adaptive_pre_comp + adaptive_pre_comm + att_mem) * 1e6, 2),
            "speedup_EP": round(
                (ep_comp + ep_comm + att_mem) / (adaptive_pre_comp + adaptive_pre_comm + att_mem), 2
            ),
            "speedup_TP": round(
                (tp_comp + tp_comm + att_mem) / (adaptive_pre_comp + adaptive_pre_comm + att_mem), 2
            ),
            "speedup_comp": round(
                (comp_comp + comp_comm + att_mem) / (adaptive_pre_comp + adaptive_pre_comm + att_mem), 2
            ),
            "speedup_hd": round(
                (comp + comm_link + att_mem) / (adaptive_pre_comp + adaptive_pre_comm + att_mem), 2
            ),
        },
    }

    file_path = (
        args.results_json if os.path.isabs(args.results_json) else os.path.join(root, args.results_json)
    )
    new_data = result

    combined_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                old_raw = f.read().strip()
                if old_raw:
                    old_data = json.loads(old_raw)
                    if isinstance(old_data, list):
                        combined_data = old_data
                    else:
                        combined_data = [old_data]
        except (json.JSONDecodeError, OSError):
            combined_data = []

    combined_data.append(new_data)

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    main()
