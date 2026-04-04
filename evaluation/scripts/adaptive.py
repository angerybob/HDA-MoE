"""Compare static vs dynamic vs pre-broadcast vs hardware-aware deployment latency under adaptive routing traces; append results to JSON."""

from __future__ import annotations

import argparse
import copy
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

# (E, e, SE, h, IS, mlp_first, num_layers) — aligned with e2e_hda / ablation
MODEL_SPECS = {
    "mixtral": (8, 2, 0, 4096, 14336, False, 32),
    "ds": (64, 6, 0, 2048, 1408, True, 26),
    "qwen": (64, 8, 0, 3584, 2560, False, 28),
}


def flatten_routing_trace_for_optimizer(trace_3d: dict) -> dict:
    """Per layer: 3D [batch][seq][expert ids] -> 2D [activations][expert ids] for MoE3DPNMOptimizer."""
    return {
        k: [activation for batch in v for activation in batch]
        for k, v in trace_3d.items()
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cwd",
        default=".",
        help="Working directory: expert_trace/, results/, and output JSON are resolved relative to this path unless absolute",
    )
    p.add_argument("--batch", type=int, default=32, help="Batch size")
    p.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="mixtral")
    p.add_argument(
        "--dataset",
        default="reasoning",
        help="Dataset name (used in results subdir and default trace path template)",
    )
    p.add_argument(
        "--trace-path",
        default=None,
        help="Expert routing JSON; default expert_trace/<model>/adaptive/experts_<dataset>_<model>_adaptive.json",
    )
    p.add_argument("--mesh", type=int, nargs=2, metavar=("X", "Y"), default=[4, 8])
    p.add_argument("--comp", type=float, default=2.5, help="Per-device compute (TFLOPS)")
    p.add_argument("--bw", type=float, default=75.0, help="Interconnect bandwidth (GB/s)")
    p.add_argument(
        "--mesh-batch-label",
        type=int,
        default=128,
        help="Integer N in NPZ path segment mesh_{N}_batches",
    )
    p.add_argument(
        "--results-json",
        default="evaluation/results/result_adaptive.json",
        help="Append-only results JSON (relative to cwd or absolute)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = os.path.abspath(args.cwd)
    batch = args.batch
    model = args.model
    dataset = args.dataset
    mesh_shape = tuple(args.mesh)
    D = mesh_shape[0] * mesh_shape[1]
    mb = args.mesh_batch_label

    E, e, SE, h, IS, mlp_first, num_layers = MODEL_SPECS[model]

    sample_path = (
        args.trace_path
        if args.trace_path
        else os.path.join(
            root,
            "expert_trace",
            model,
            "adaptive",
            f"experts_{dataset}_{model}_adaptive.json",
        )
    )
    if not os.path.isabs(sample_path):
        sample_path = os.path.join(root, sample_path)

    try:
        with open(sample_path, encoding="utf-8") as f1:
            sample2 = json.load(f1)
        sample = sample2["original_selected_experts"]
        pre_sample = sample2["original_predict_experts"]
        adaptive_sample = sample2["selected_experts"]
        adaptive_pre_sample = sample2["predict_experts"]
    except FileNotFoundError:
        print(f"File not found: {sample_path}")
        sys.exit(1)

    routing_trace_flat = flatten_routing_trace_for_optimizer(sample)
    optimizer = MoE3DPNMOptimizer(
        E=E,
        e=e,
        h=h,
        IS=IS,
        B=batch,
        D=D,
        BW=args.bw * 1e9,
        comp=args.comp * 1e12,
        num_layers=num_layers,
        mlp_first=mlp_first,
        routing_trace=routing_trace_flat,
    )
    P_tp = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
    P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)

    sample_id = random.randint(0, len(sample[str(1)]) - 2)
    s = sample[str(1 + optimizer.mlp_first)][sample_id]
    while len(s) != batch:
        sample_id = random.randint(0, len(sample[str(1)]) - 1)
        s = sample[str(1 + optimizer.mlp_first)][sample_id]

    BWmem = 625e9
    decode_L = 64
    L = decode_L
    intermediate = optimizer.IS / optimizer.h

    att_comp = {
        "ds": (2048 * 2048 + 512 * 4096 + 2048 * 3072 + 2048 * 2816 * 2 + 2048 * 2 * decode_L * 2)
        * batch
        / (D * optimizer.comp),
        "mixtral": (3 * batch * h**2 + h**2 * batch + batch * h * decode_L * 2) / (D * optimizer.comp),
        "qwen": (
            batch * h**2
            + 2 * batch * 0.25 * h**2
            + 0.25 * h**2 * batch
            + 2560 * batch * h * 2
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

    att_mem = mem[model] + att_comp[model]
    t_inf = (2 * e * batch * intermediate * h**2) / (D * optimizer.comp) + att_mem
    k = 1
    while optimizer.optimal_broadcast_chunk(k=k) < t_inf:
        k += 1
    k -= 1
    print(f"Number of experts to pre-broadcast: {k:.0f}")

    tp_comp_dynamic = 0.0
    tp_comm_dynamic = 0.0
    ep_comp_dynamic = 0.0
    ep_comm_dynamic = 0.0
    comp_dynamic = 0.0
    comm_link_dynamic = 0.0
    comp_adaptive = 0.0
    comm_adaptive = 0.0
    comp_pre = 0.0
    comm_pre = 0.0
    comp_pre_adaptive = 0.0
    comm_pre_adaptive = 0.0

    res_dir = os.path.join(root, "results")

    for layer_id in tqdm(range(optimizer.layer)):
        subdir = (
            f"{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_"
            f"for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{mb}_batches"
        )
        file_path = os.path.join(
            res_dir,
            subdir,
            f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz",
        )
        loaded_arrays = np.load(file_path)

        P = loaded_arrays["arr1"]
        comp_map = np.zeros((optimizer.E,))
        adaptive_comp_map = np.zeros((optimizer.E,))

        s = sample[str(layer_id + optimizer.mlp_first)][sample_id]
        if len(s) == batch:
            random_samples = s
            if str(layer_id + optimizer.mlp_first) in pre_sample:
                next_samples = pre_sample[str(layer_id + optimizer.mlp_first)][sample_id]
                adaptive_random_samples = adaptive_sample[str(layer_id + optimizer.mlp_first)][sample_id]
                adaptive_next_samples = adaptive_pre_sample[str(layer_id + optimizer.mlp_first)][sample_id]
            else:
                next_samples = []
                adaptive_random_samples = []
                adaptive_next_samples = []

        for sublist in random_samples:
            comp_map[sublist] += 2 * optimizer.h * optimizer.IS
        for sublist in adaptive_random_samples:
            adaptive_comp_map[sublist] += 2 * optimizer.h * optimizer.IS

        tp_comp_dynamic += optimizer.compute_time_dynamic(P_tp, comp_map)[layer_id]
        tp_comm_dynamic += 4 * optimizer.comm_time_dynamic(P_tp, random_samples)[layer_id]

        ep_comp_dynamic += optimizer.compute_time_dynamic(P_ep, comp_map)[layer_id]
        ep_comm_dynamic += 2 * optimizer.comm_time_acc_dynamic(
            generate_random_placement(optimizer.D, mesh_shape), P_ep, layer_id, random_samples
        )

        comp_dynamic += optimizer.compute_time_dynamic(P, comp_map)[layer_id]
        comp_adaptive += optimizer.compute_time_dynamic(P, adaptive_comp_map)[layer_id]

        M = generate_random_placement(optimizer.D, mesh_shape)

        comm_link_dynamic += 2 * optimizer.comm_time_acc_dynamic(M, P, layer_id, random_samples)
        comm_adaptive += 2 * optimizer.comm_time_acc_dynamic(M, P, layer_id, adaptive_random_samples)

        if layer_id == 0:
            comp_pre += optimizer.compute_time_dynamic(P, comp_map)[0]
            comm_pre += 2 * optimizer.comm_time_acc_dynamic(M, P, 0, random_samples)
            comp_pre_adaptive += optimizer.compute_time_dynamic(P, adaptive_comp_map)[0]
            comm_pre_adaptive += 2 * optimizer.comm_time_acc_dynamic(M, P, 0, adaptive_random_samples)

        if layer_id < optimizer.layer - 1:
            pre_path = os.path.join(
                res_dir,
                subdir,
                f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id + 1:.0f}.npz",
            )
            pre_arrays = np.load(pre_path)
            P_next = pre_arrays["arr1"]
            M_next = pre_arrays["arr2"]
            Z_next = P_next[layer_id + 1] > 0
            p_copy = copy.deepcopy(P_next)
            z_copy = copy.deepcopy(Z_next)
            adaptive_p_copy = copy.deepcopy(P_next)
            adaptive_z_copy = copy.deepcopy(Z_next)
            comp_map_next = np.zeros((optimizer.E,))
            adaptive_comp_map_next = np.zeros((optimizer.E,))

            random_samples_next = sample[str(layer_id + optimizer.mlp_first + 1)][sample_id]
            next_samples = pre_sample[str(layer_id + optimizer.mlp_first + 1)][sample_id]
            for sublist in random_samples_next:
                comp_map_next[sublist] += 2 * optimizer.h * optimizer.IS

            adaptive_random_samples_next = adaptive_sample[str(layer_id + optimizer.mlp_first + 1)][sample_id]
            adaptive_next_samples = adaptive_pre_sample[str(layer_id + optimizer.mlp_first + 1)][sample_id]

            for sublist in adaptive_random_samples_next:
                adaptive_comp_map_next[sublist] += 2 * optimizer.h * optimizer.IS

            compute_load_next = np.zeros((optimizer.layer, optimizer.D))
            adaptive_compute_load_next = np.zeros((optimizer.layer, optimizer.D))

            for _ in range(k):
                _ap, adaptive_e_sel = optimizer.priority_detection(
                    p_copy, layer_id + 1, adaptive_next_samples
                )
                adaptive_p_copy[layer_id + 1, adaptive_e_sel, :] = 0
                adaptive_z_copy[adaptive_e_sel, :] = 0
                for sublist in adaptive_random_samples_next:
                    if adaptive_e_sel in sublist:
                        activate_node = []
                        _j, d = np.nonzero(P_next[layer_id + 1][sublist])
                        for c in d:
                            activate_node.append(c)

                        adaptive_compute_load_next = np.sum(
                            adaptive_p_copy * adaptive_comp_map_next[None, :, None], axis=1
                        )
                        scatter_node = np.argmin(adaptive_compute_load_next[layer_id + 1, activate_node])

                        adaptive_compute_load_next[layer_id + 1, scatter_node] += 2 * optimizer.h * optimizer.IS

            for _ in range(k):
                _prio, e_sel = optimizer.priority_detection(p_copy, layer_id + 1, next_samples)
                p_copy[layer_id + 1, e_sel, :] = 0
                z_copy[e_sel, :] = 0
                for sublist in random_samples_next:
                    if e_sel in sublist:
                        activate_node = []
                        _j, d = np.nonzero(P_next[layer_id + 1][sublist])
                        for c in d:
                            activate_node.append(c)

                        compute_load_next = np.sum(p_copy * comp_map_next[None, :, None], axis=1)
                        scatter_node = np.argmin(compute_load_next[layer_id + 1, activate_node])
                        compute_load_next[layer_id + 1, scatter_node] += 2 * optimizer.h * optimizer.IS

            if k > 0:
                comp_pre += np.max(compute_load_next[layer_id + 1]) / optimizer.comp
                comm_pre += 2 * optimizer.comm_time_acc_dynamic(
                    M_next, p_copy, layer_id + 1, random_samples_next
                )
                comp_pre_adaptive += np.max(adaptive_compute_load_next[layer_id + 1]) / optimizer.comp
                comm_pre_adaptive += 2 * optimizer.comm_time_acc_dynamic(
                    M_next, adaptive_p_copy, layer_id + 1, adaptive_random_samples_next
                )
            else:
                compute_load_next = np.sum(p_copy * comp_map_next[None, :, None], axis=1)
                adaptive_compute_load_next = np.sum(
                    adaptive_p_copy * adaptive_comp_map_next[None, :, None], axis=1
                )
                comp_pre += np.max(compute_load_next[layer_id + 1]) / optimizer.comp
                comm_pre += 2 * optimizer.comm_time_acc_dynamic(
                    M_next, p_copy, layer_id + 1, random_samples_next
                )
                comp_pre_adaptive += np.max(adaptive_compute_load_next[layer_id + 1]) / optimizer.comp
                comm_pre_adaptive += 2 * optimizer.comm_time_acc_dynamic(
                    M_next, adaptive_p_copy, layer_id + 1, adaptive_random_samples_next
                )

    print(f"TP_communication_dynamic: {tp_comm_dynamic*1e6:.2f} us")
    print(f"TP_computation_dynamic: {tp_comp_dynamic*1e6:.2f} us")
    print(f"TP_latency_dynamic: {(tp_comp_dynamic+tp_comm_dynamic)*1e6:.2f} us")

    print(f"EP_communication_dynamic: {ep_comm_dynamic*1e6:.2f} us")
    print(f"EP_computation_dynamic: {ep_comp_dynamic*1e6:.2f} us")
    print(f"EP_latency_dynamic: {(ep_comp_dynamic+ep_comm_dynamic)*1e6:.2f} us")

    print(f"link_balancing_dynamic: {comm_link_dynamic*1e6:.2f} us")
    print(f"link_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"link_balancing_latency_dynamic: {(comp_dynamic+comm_link_dynamic)*1e6:.2f} us")

    print(
        f"node_link_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}"
    )
    print(
        f"node_link_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}"
    )

    print(f"preb_communication_dynamic: {comm_pre*1e6:.2f} us")
    print(f"preb_computation_dynamic: {comp_pre*1e6:.2f} us")
    print(f"preb_latency_dynamic: {(comp_pre+comm_pre)*1e6:.2f} us")

    print(f"preb_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_pre+comm_pre):.2f}")
    print(f"preb_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_pre+comm_pre):.2f}")

    print(f"preb_speedup_dynamic:{(comp_dynamic+comm_link_dynamic)/(comp_pre+comm_pre):.2f}")
    print(f"adaptive_speedup_dynamic:{(comp_dynamic+comm_link_dynamic)/(comp_adaptive+comm_adaptive):.2f}")
    print(
        f"adaptive_pre_speedup_dynamic:{(comp_dynamic+comm_link_dynamic)/(comp_pre_adaptive+comm_pre_adaptive):.2f}"
    )

    config = {
        "mesh_shape": mesh_shape,
        "model": model,
        "dataset": dataset,
        "sample": dataset,
        "comp_TFLOPS": optimizer.comp * 1e-12,
        "BW_GBPS": optimizer.BW * 1e-9,
        "batch": optimizer.B,
    }

    result = {
        "config": config,
        "static_deployment": {
            "communication_us": round(comm_link_dynamic * 1e6, 2),
            "computation_us": round(comp_dynamic * 1e6, 2),
            "latency_us": round((comp_dynamic + comm_link_dynamic) * 1e6, 2),
        },
        "dynamic_deployment": {
            "communication_us": round(comm_pre * 1e6, 2),
            "computation_us": round(comp_pre * 1e6, 2),
            "latency_us": round((comp_pre + comm_pre) * 1e6, 2),
            "speedup": round((comp_dynamic + comm_link_dynamic) / (comp_pre + comm_pre), 2),
        },
        "adaptive_deployment": {
            "communication_us": round(comm_adaptive * 1e6, 2),
            "computation_us": round(comp_adaptive * 1e6, 2),
            "latency_us": round((comp_adaptive + comm_adaptive) * 1e6, 2),
            "speedup": round((comp_dynamic + comm_link_dynamic) / (comp_adaptive + comm_adaptive), 2),
        },
        "adaptive_dynamic_deployment": {
            "communication_us": round(comm_pre_adaptive * 1e6, 2),
            "computation_us": round(comp_pre_adaptive * 1e6, 2),
            "latency_us": round((comp_pre_adaptive + comm_pre_adaptive) * 1e6, 2),
            "speedup": round(
                (comp_dynamic + comm_link_dynamic) / (comp_pre_adaptive + comm_pre_adaptive), 2
            ),
        },
    }

    out_path = (
        args.results_json
        if os.path.isabs(args.results_json)
        else os.path.join(root, args.results_json)
    )
    new_data = result
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            old_data = json.load(f)
        combined_data = old_data + [new_data]
    else:
        combined_data = [new_data]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    main()
