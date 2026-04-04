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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end evaluation with hardware-aware (HDA) routing traces."
    )
    p.add_argument("--batch", type=int, default=32, help="batch size")
    p.add_argument(
        "--model",
        type=str,
        default="ds",
        choices=["mixtral", "ds", "qwen"],
        help="model name",
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
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
    D = mesh_shape[0] * mesh_shape[1]

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

    mem = {
        "ds": (
            2048 * (3072 + batch)
            + 2048 * (2048 + batch)
            + 512 * 4096
            + 512 * 2 * decode_L
            + 2 * (E + 2) * intermediate * h**2
            + batch * h * (intermediate + 1)
        )
        / (D * BWmem),
        "mixtral": (
            2 * batch * h * decode_L
            + 3 * h * (batch + h)
            + h * (batch + h)
            + 2 * E * intermediate * h**2
            + batch * h * (intermediate + 1)
        )
        / (D * BWmem),
        "qwen": (
            2 * batch * h * 0.25 * decode_L
            + 1.5 * h * (batch + h)
            + 0.25 * h * (batch + h)
            + 2 * (E + 8) * intermediate * h**2
            + batch * h * (intermediate + 1)
        )
        / (D * BWmem),
    }

    att_mem = (mem[model] + att_comp[model]) * optimizer.layer

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
    subdir = (
        f"{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_"
        f"for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches"
    )

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
            subdir,
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
