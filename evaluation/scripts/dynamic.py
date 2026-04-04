"""Dynamic routing: compare link latency under selected vs predicted experts (pre-broadcast chunk size k)."""

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
from moe_placement_utils import (
    EP_deployment,
    comp_overhead,
    generate_random_placement,
    mem_overhead,
)

MODEL_SPECS = {
    "ds": (64, 6, 0, 2048, 1408, True, 26),
    "mixtral": (8, 2, 0, 4096, 14336, False, 32),
    "qwen": (64, 8, 0, 3584, 2560, False, 28),
}

DATASET_CHOICES = ["reasoning", "math", "coding", "writing", "roleplay"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cwd", default=".", help="Base dir for expert_trace/ and results/")
    p.add_argument("--batch", type=int, default=512, help="Default matches legacy 128*4")
    p.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="ds")
    p.add_argument(
        "--dataset",
        default="reasoning",
        choices=DATASET_CHOICES,
        help="Primary trace name (experts_{dataset}_{model}_pre.json)",
    )
    p.add_argument(
        "--compare-dataset",
        default="roleplay",
        choices=DATASET_CHOICES,
        help="Second trace for sample/pre_sample (legacy: index 4 → roleplay)",
    )
    p.add_argument("--mesh", type=int, nargs=2, metavar=("X", "Y"), default=[4, 8])
    p.add_argument("--comp", type=float, default=2.5, help="TFLOPS (passed to optimizer)")
    p.add_argument("--bw", type=float, default=75.0, help="GB/s (passed to optimizer)")
    p.add_argument(
        "--mesh-batch-label",
        type=int,
        default=128,
        help="Folder segment mesh_{N}_batches in NPZ paths",
    )
    p.add_argument(
        "--results-json",
        default="evaluation/results/result2_dynamic.json",
        help="Output JSON (relative to cwd unless absolute)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = os.path.abspath(args.cwd)
    batch = args.batch
    model = args.model
    dataset = args.dataset
    sample1 = args.compare_dataset
    mesh_shape = tuple(args.mesh)
    E, e, SE, h, IS, mlp_first, num_layers = MODEL_SPECS[model]
    D = mesh_shape[0] * mesh_shape[1]

    data_path = os.path.join(
        root, "expert_trace", model, f"experts_{dataset}_{model}_pre.json"
    )
    sample_path = os.path.join(
        root, "expert_trace", model, f"experts_{sample1}_{model}_pre.json"
    )
    try:
        with open(data_path, encoding="utf-8") as f:
            data1 = json.load(f)
            data, _ = data1["selected_experts"], data1["predict_experts"]
        with open(sample_path, encoding="utf-8") as f1:
            sample2 = json.load(f1)
            sample, pre_sample = sample2["selected_experts"], sample2["predict_experts"]
    except FileNotFoundError:
        print("File not found; check path and filename.")
        sys.exit(1)

    optimizer = MoE3DPNMOptimizer(
        E=E,
        e=e,
        SE=SE,
        h=h,
        IS=IS,
        B=batch,
        D=D,
        BW=args.bw * 1e9,
        comp=args.comp * 1e12,
        num_layers=num_layers,
        mlp_first=mlp_first,
        routing_trace=data,
    )
    P_tp = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
    P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)

    sample_id = random.sample(range(0, len(sample[str(1)])), batch)
    BWmem = 625e9
    L = 1024
    t_inf = comp_overhead(
        optimizer.comp,
        optimizer.D,
        batch,
        optimizer.h,
        L,
        optimizer.IS / optimizer.h,
        optimizer.e,
        include_moe_ffn=True,
    ) + mem_overhead(
        BWmem, optimizer.D, batch, optimizer.h, L, optimizer.IS / optimizer.h, optimizer.E
    )
    k = 1
    while optimizer.optimal_broadcast_chunk(k=k) < t_inf:
        k += 1
    k -= 1
    print(f"Number of experts to pre-broadcast: {k:.0f}")

    tp_comp_dynamic = 0
    tp_comm_dynamic = 0
    ep_comp_dynamic = 0
    ep_comm_dynamic = 0
    comp_dynamic = 0
    comm_link = 0
    comm_link_dynamic = 0
    comp_pre = 0
    comm_pre = 0

    res_dir = os.path.join(root, "results")
    mb = args.mesh_batch_label
    subdir = (
        f"{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_"
        f"for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{mb}_batches"
    )

    for layer_id in tqdm(range(optimizer.layer)):
        file_path = os.path.join(
            res_dir,
            subdir,
            f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz",
        )
        loaded_arrays = np.load(file_path)
        P = loaded_arrays["arr1"]
        M = loaded_arrays["arr2"]
        comp_map = np.zeros((optimizer.E,))
        random_samples = []
        next_samples = []
        for i in sample_id:
            random_samples.append(sample[str(layer_id + optimizer.mlp_first)][i])
            next_samples.append(pre_sample[str(layer_id + optimizer.mlp_first)][i])
        for sublist in random_samples:
            comp_map[sublist] += 2 * optimizer.h * optimizer.IS

        M_rand = generate_random_placement(optimizer.D, mesh_shape)

        tp_comp_dynamic += optimizer.compute_time_dynamic(P_tp, comp_map)[layer_id]
        tp_comm_dynamic += 2 * optimizer.comm_time_dynamic(P_tp, random_samples)[layer_id]

        ep_comp_dynamic += optimizer.compute_time_dynamic(P_ep, comp_map)[layer_id]
        ep_comm_dynamic += 2 * optimizer.comm_time_acc_dynamic(M_rand, P_ep, layer_id, random_samples)
        comp_dynamic += optimizer.compute_time_dynamic(P, comp_map)[layer_id]

        comm_temp, _link = optimizer.comm_time_acc(M, P, layer_id)
        comm_link += comm_temp * 2
        comm_link_dynamic += 2 * optimizer.comm_time_acc_dynamic(M, P, layer_id, random_samples)

        if layer_id == 0:
            comp_pre += optimizer.compute_time_dynamic(P, comp_map)[0]
            comm_pre += 2 * optimizer.comm_time_acc_dynamic(M, P, 0, random_samples)

        if layer_id < optimizer.layer - 1:
            pre_path = os.path.join(
                res_dir,
                subdir,
                f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id+1:.0f}.npz",
            )
            pre_arrays = np.load(pre_path)
            P_next = pre_arrays["arr1"]
            M_next = pre_arrays["arr2"]
            Z_next = P_next[layer_id + 1] > 0
            p_copy = copy.deepcopy(P_next)
            z_copy = copy.deepcopy(Z_next)
            comp_map_next = np.zeros((optimizer.E,))
            random_samples_next = []
            next_samples = []
            for i in sample_id:
                random_samples_next.append(sample[str(layer_id + 1 + 1)][i])
                next_samples.append(pre_sample[str(layer_id + 1 + 1)][i])
            for sublist in random_samples_next:
                comp_map_next[sublist] += 2 * optimizer.h * optimizer.IS
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
            comp_pre += np.max(compute_load_next[layer_id + 1]) / optimizer.comp
            comm_pre += 2 * optimizer.comm_time_acc_dynamic(
                M_next, p_copy, layer_id + 1, random_samples_next
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

    config = {
        "mesh_shape": mesh_shape,
        "model": model,
        "dataset": dataset,
        "sample": sample1,
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
    }

    out_path = (
        args.results_json if os.path.isabs(args.results_json) else os.path.join(root, args.results_json)
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
