"""Ablation: TP/EP vs compute-only balance vs node/link balance using NPZ placements and a single routing trace."""

from __future__ import annotations

import argparse
import json
import os
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

# (E, e, SE, h, IS, mlp_first, num_layers) — Qwen IS aligned with e2e_hda / e2e_hda2 / sim.
MODEL_SPECS = {
    "mixtral": (8, 2, 0, 4096, 14336, False, 32),
    "ds": (64, 6, 0, 2048, 1408, True, 26),
    "qwen": (64, 8, 0, 3584, 2560, False, 28),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cwd",
        default=".",
        help="Working directory for relative paths expert_trace/ and results/",
    )
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="ds")
    p.add_argument("--dataset", default="reasoning")
    p.add_argument("--mesh", type=int, nargs=2, metavar=("X", "Y"), default=[8, 8])
    p.add_argument("--comp", type=float, default=5.0, help="Per-device compute (TFLOPS)")
    p.add_argument("--bw", type=float, default=50.0, help="Interconnect bandwidth (GB/s)")
    p.add_argument(
        "--mesh-batch-label",
        type=int,
        default=128,
        help="Integer in NPZ path segment mesh_{N}_batches for the main results tree",
    )
    p.add_argument(
        "--results-json",
        default="evaluation/results/result_ablation.json",
        help="Append-only JSON path (relative to cwd unless absolute)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = os.path.abspath(args.cwd)
    mesh_shape = tuple(args.mesh)
    E, e, SE, h, IS, mlp_first, num_layers = MODEL_SPECS[args.model]
    D = mesh_shape[0] * mesh_shape[1]

    data_path = os.path.join(
        root, "expert_trace", args.model, f"experts_{args.dataset}_{args.model}.json"
    )
    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        with open(data_path, encoding="utf-8") as f1:
            sample = json.load(f1)
    except FileNotFoundError:
        print("File not found; check path and filename.")
        sys.exit(1)

    optimizer = MoE3DPNMOptimizer(
        E=E,
        e=e,
        SE=SE,
        h=h,
        IS=IS,
        B=args.batch,
        D=D,
        BW=args.bw * 1e9,
        comp=args.comp * 1e12,
        num_layers=num_layers,
        mlp_first=mlp_first,
        routing_trace=data,
    )
    P_tp = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
    P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)

    tp_comp = 0
    tp_comm = 0
    ep_comp = 0
    ep_comm = 0
    comp_comp = 0
    comp_comm = 0
    comp = 0
    comm_node = 0
    comm_link = 0

    mb = args.mesh_batch_label
    res_dir = os.path.join(root, "results")

    for layer_id in tqdm(range(optimizer.layer)):
        subdir = (
            f"{args.dataset}_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_"
            f"for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{mb}_batches"
        )
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

        M_rand = generate_random_placement(optimizer.D, mesh_shape)
        optimizer.M = M_rand

        tp_comp += optimizer.compute_time(P_tp)[layer_id]
        tp_comm += 2 * optimizer.comm_time(P_tp)[layer_id]

        ep_comp += optimizer.compute_time(P_ep)[layer_id]
        comm_temp, _link = optimizer.comm_time_acc(M_rand, P_ep, layer_id)
        ep_comm += comm_temp * 2

        comp_comp += optimizer.compute_time(P_comp)[layer_id]
        comm_temp, _link = optimizer.comm_time_acc(M_comp, P_comp, layer_id)
        comm_temp += optimizer.comm_time(P)[layer_id]
        comp_comm += comm_temp * 2

        comp += optimizer.compute_time(P)[layer_id]
        comm_temp, _link = optimizer.comm_time_acc(M_rand, P, layer_id)
        comm_node += comm_temp * 2

        comm_temp, _link = optimizer.comm_time_acc(M, P, layer_id)
        comm_link += comm_temp * 2

    print(f"TP_communication: {tp_comm*1e6:.2f} us")
    print(f"TP_computation: {tp_comp*1e6:.2f} us")
    print(f"TP_latency: {(tp_comp+tp_comm)*1e6:.2f} us")

    print(f"EP_communication: {ep_comm*1e6:.2f} us")
    print(f"EP_computation: {ep_comp*1e6:.2f} us")
    print(f"EP_latency: {(ep_comp+ep_comm)*1e6:.2f} us")

    print(f"comp_communication: {comp_comm*1e6:.2f} us")
    print(f"comp_computation: {comp_comp*1e6:.2f} us")
    print(f"comp_latency: {(comp_comp+comp_comm)*1e6:.2f} us")

    print(f"node_balancing_communication: {comm_node*1e6:.2f} us")
    print(f"node_balancing_computation: {comp*1e6:.2f} us")
    print(f"node_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
    print(f"node_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")
    print(f"node_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_node):.2f}")
    print(f"node_balancing_speedup_comp:{(comp_comp+comp_comm)/(comp+comm_node):.2f}")

    print(f"link_balancing: {comm_link*1e6:.2f} us")
    print(f"link_balancing_computation: {comp*1e6:.2f} us")
    print(f"link_balancing_latency: {(comp+comm_link)*1e6:.2f} us")
    print(f"link_balancing_speedup:{(comm_node)/(comm_link):.2f}")
    print(f"node_link_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_link):.2f}")
    print(f"node_link_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_link):.2f}")
    print(f"node_link_balancing_speedup_comp:{(comp_comp+comp_comm)/(comp+comm_link):.2f}")

    result = {
        "config": {
            "mesh_shape": mesh_shape,
            "model": args.model,
            "dataset": args.dataset,
            "comp_TFLOPS": optimizer.comp * 1e-12,
            "BW_GBPS": optimizer.BW * 1e-9,
        },
        "TP": {
            "communication_us": round(tp_comm * 1e6, 2),
            "computation_us": round(tp_comp * 1e6, 2),
            "latency_us": round((tp_comp + tp_comm) * 1e6, 2),
        },
        "EP": {
            "communication_us": round(ep_comm * 1e6, 2),
            "computation_us": round(ep_comp * 1e6, 2),
            "latency_us": round((ep_comp + ep_comm) * 1e6, 2),
        },
        "compute_balancing": {
            "communication_us": round(comp_comm * 1e6, 2),
            "computation_us": round(comp_comp * 1e6, 2),
            "latency_us": round((comp_comp + comp_comm) * 1e6, 2),
        },
        "node_balancing": {
            "communication_us": round(comm_node * 1e6, 2),
            "computation_us": round(comp * 1e6, 2),
            "latency_us": round((comp + comm_node) * 1e6, 2),
            "speedup_EP": round((ep_comp + ep_comm) / (comp + comm_node), 2),
            "speedup_TP": round((tp_comp + tp_comm) / (comp + comm_node), 2),
            "speedup_comp": round((comp_comp + comp_comm) / (comp + comm_node), 2),
        },
        "link_balancing": {
            "communication_us": round(comm_link * 1e6, 2),
            "computation_us": round(comp * 1e6, 2),
            "latency_us": round((comp + comm_link) * 1e6, 2),
            "speedup": round(comm_node / comm_link, 2),
            "speedup_EP": round((ep_comp + ep_comm) / (comp + comm_link), 2),
            "speedup_TP": round((tp_comp + tp_comm) / (comp + comm_link), 2),
            "speedup_comp": round((comp_comp + comp_comm) / (comp + comm_link), 2),
        },
    }

    out_path = (
        args.results_json
        if os.path.isabs(args.results_json)
        else os.path.join(root, args.results_json)
    )
    new_data = result
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        combined_data = old_data + [new_data]
    else:
        combined_data = [new_data]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    main()
