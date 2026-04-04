"""Load routing traces + NPZ placements; compare TP/EP vs node/link-balanced latency (static vs dynamic expert routing)."""

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

# (E, e, SE, h, IS, mlp_first, num_layers) — same defaults as the pre-argparse script (DeepSeek was the effective model).
MODEL_SPECS = {
    "mixtral": (8, 2, 0, 4096, 14336, False, 32),
    "ds": (64, 6, 0, 2048, 1408, True, 26),
    "qwen": (64, 8, 0, 3584, 2560, False, 28),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--deployment-root",
        default="/data/home/haochenhuang/deployment",
        help="Root that contains evaluation/ and results/ trees used by this script",
    )
    p.add_argument(
        "--trace-subdir",
        default="evaluation",
        help="Subdirectory under deployment-root where experts_{dataset}_{model}.json lives",
    )
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="ds")
    p.add_argument("--dataset", default="reasoning")
    p.add_argument("--mesh", type=int, nargs=2, metavar=("X", "Y"), default=[4, 8])
    p.add_argument("--comp", type=float, default=2.5, help="Per-device compute (TFLOPS)")
    p.add_argument("--bw", type=float, default=75.0, help="Interconnect bandwidth (GB/s)")
    p.add_argument(
        "--ref-trace-layer",
        type=int,
        default=5,
        help="Which layer's trace rows to sample for batch indices (matches JSON key str(layer+1))",
    )
    p.add_argument(
        "--mesh-batch-label",
        type=int,
        default=128,
        help="Integer in NPZ path segment mesh_{N}_batches (historical folder naming)",
    )
    p.add_argument(
        "--cwd",
        default=".",
        help="Working directory for relative --results-json",
    )
    p.add_argument(
        "--results-json",
        default="evaluation/results/result.json",
        help="Append-only JSON path (relative to cwd unless absolute)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cwd = os.path.abspath(args.cwd)
    mesh_shape = tuple(args.mesh)
    E, e, SE, h, IS, mlp_first, num_layers = MODEL_SPECS[args.model]
    D = mesh_shape[0] * mesh_shape[1]

    data_path = os.path.join(
        args.deployment_root,
        args.trace_subdir,
        f"experts_{args.dataset}_{args.model}.json",
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

    optimizer.X, optimizer.Y = mesh_shape

    ref_layer = args.ref_trace_layer
    tp_comp = 0
    tp_comm = 0
    tp_comp_dynamic = 0
    tp_comm_dynamic = 0
    ep_comp = 0
    ep_comm = 0
    ep_comp_dynamic = 0
    ep_comm_dynamic = 0
    comp = 0
    comm_node = 0
    comp_dynamic = 0
    comm_node_dynamic = 0
    comm_link = 0
    comm_link_dynamic = 0

    sample_id = random.sample(
        range(0, len(sample[str(ref_layer + 1)])), args.batch
    )

    res_root = os.path.join(args.deployment_root, "results")
    mb = args.mesh_batch_label
    for layer_id in tqdm(range(optimizer.layer)):
        file_path = os.path.join(
            res_root,
            f"{args.dataset}_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{mb}_batches",
            f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz",
        )
        loaded_arrays = np.load(file_path)
        comp_map = np.zeros((optimizer.E,))
        P = loaded_arrays["arr1"]
        M = loaded_arrays["arr2"]
        random_samples = []
        for i in sample_id:
            random_samples.append(sample[str(layer_id + 1)][i])
        for sublist in random_samples:
            comp_map[sublist] += 2 * optimizer.h * optimizer.IS

        M_rand = generate_random_placement(optimizer.D, mesh_shape)
        optimizer.M = M_rand

        tp_comp += optimizer.compute_time(P_tp)[layer_id]
        tp_comm += 2 * optimizer.comm_time(P_tp)[layer_id]

        tp_comp_dynamic += optimizer.compute_time_dynamic(P_tp, comp_map)[layer_id]
        tp_comm_dynamic += 2 * optimizer.comm_time_dynamic(P_tp, random_samples)[layer_id]

        ep_comp += optimizer.compute_time(P_ep)[layer_id]
        comm_temp, _link = optimizer.comm_time_acc(M_rand, P_ep, layer_id)
        ep_comm += comm_temp * 2

        ep_comp_dynamic += optimizer.compute_time_dynamic(P_ep, comp_map)[layer_id]
        ep_comm_dynamic += 2 * optimizer.comm_time_acc_dynamic(
            M_rand, P_ep, layer_id, random_samples
        )

        comp += optimizer.compute_time(P)[layer_id]
        comm_temp, _link = optimizer.comm_time_acc(M_rand, P, layer_id)
        comm_node += comm_temp * 2

        comp_dynamic += optimizer.compute_time_dynamic(P, comp_map)[layer_id]
        comm_node_dynamic += 2 * optimizer.comm_time_acc_dynamic(
            M_rand, P, layer_id, random_samples
        )

        optimizer.M = M
        comm_temp, _link = optimizer.comm_time_acc(M, P, layer_id)
        comm_link += comm_temp * 2

        comm_link_dynamic += 2 * optimizer.comm_time_acc_dynamic(
            M, P, layer_id, random_samples
        )

    print(f"TP_communication: {tp_comm*1e6:.2f} us")
    print(f"TP_computation: {tp_comp*1e6:.2f} us")
    print(f"TP_latency: {(tp_comp+tp_comm)*1e6:.2f} us")
    print(f"TP_communication_dynamic: {tp_comm_dynamic*1e6:.2f} us")
    print(f"TP_computation_dynamic: {tp_comp_dynamic*1e6:.2f} us")
    print(f"TP_latency_dynamic: {(tp_comp_dynamic+tp_comm_dynamic)*1e6:.2f} us")
    print(f"EP_communication: {ep_comm*1e6:.2f} us")
    print(f"EP_computation: {ep_comp*1e6:.2f} us")
    print(f"EP_latency: {(ep_comp+ep_comm)*1e6:.2f} us")
    print(f"EP_communication_dynamic: {ep_comm_dynamic*1e6:.2f} us")
    print(f"EP_computation_dynamic: {ep_comp_dynamic*1e6:.2f} us")
    print(f"EP_latency_dynamic: {(ep_comp_dynamic+ep_comm_dynamic)*1e6:.2f} us")

    print(f"node_balancing_communication: {comm_node*1e6:.2f} us")
    print(f"node_balancing_computation: {comp*1e6:.2f} us")
    print(f"node_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
    print(f"node_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")
    print(f"node_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_node):.2f}")
    print(f"node_balancing_communication_dynamic: {comm_node_dynamic*1e6:.2f} us")
    print(f"node_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"node_balancing_latency_dynamic: {(comp_dynamic+comm_node_dynamic)*1e6:.2f} us")
    print(
        f"node_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}"
    )
    print(
        f"node_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}"
    )

    Z = P > 0
    dis = optimizer.evaluate_placement(M, Z, layer_id)
    print(f"Total communication distance is {dis:.2f} nodes")
    M_rand = generate_random_placement(optimizer.D, mesh_shape)
    dis_rand = optimizer.evaluate_placement(M_rand, Z, layer_id)
    print(f"Total communication distance of random mapping is {dis_rand:.2f} nodes")

    print(f"link_balancing: {comm_link*1e6:.2f} us")
    print(f"link_balancing_computation: {comp*1e6:.2f} us")
    print(f"link_balancing_latency: {(comp+comm_link)*1e6:.2f} us")
    print(f"link_balancing_speedup:{(comm_node)/(comm_link):.2f}")
    print(f"node_link_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_link):.2f}")
    print(f"node_link_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_link):.2f}")

    print(f"link_balancing_dynamic: {comm_link_dynamic*1e6:.2f} us")
    print(f"link_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"link_balancing_latency_dynamic: {(comp_dynamic+comm_link_dynamic)*1e6:.2f} us")
    print(f"link_balancing_speedup_dynamic:{(comm_node_dynamic)/(comm_link_dynamic):.2f}")
    print(
        f"node_link_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}"
    )
    print(
        f"node_link_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}"
    )

    result = {
        "config": {
            "mesh_shape": mesh_shape,
            "model": args.model,
            "dataset": args.dataset,
            "comp_TFLOPS": optimizer.comp * 1e-12,
            "BW_GBPS": optimizer.BW * 1e-9,
        },
        "TP": {
            "static": {
                "communication_us": round(tp_comm * 1e6, 2),
                "computation_us": round(tp_comp * 1e6, 2),
                "latency_us": round((tp_comp + tp_comm) * 1e6, 2),
            },
            "dynamic": {
                "communication_us": round(tp_comm_dynamic * 1e6, 2),
                "computation_us": round(tp_comp_dynamic * 1e6, 2),
                "latency_us": round((tp_comp_dynamic + tp_comm_dynamic) * 1e6, 2),
            },
        },
        "EP": {
            "static": {
                "communication_us": round(ep_comm * 1e6, 2),
                "computation_us": round(ep_comp * 1e6, 2),
                "latency_us": round((ep_comp + ep_comm) * 1e6, 2),
            },
            "dynamic": {
                "communication_us": round(ep_comm_dynamic * 1e6, 2),
                "computation_us": round(ep_comp_dynamic * 1e6, 2),
                "latency_us": round((ep_comp_dynamic + ep_comm_dynamic) * 1e6, 2),
            },
        },
        "node_balancing": {
            "static": {
                "communication_us": round(comm_node * 1e6, 2),
                "computation_us": round(comp * 1e6, 2),
                "latency_us": round((comp + comm_node) * 1e6, 2),
                "speedup_EP": round((ep_comp + ep_comm) / (comp + comm_node), 2),
                "speedup_TP": round((tp_comp + tp_comm) / (comp + comm_node), 2),
            },
            "dynamic": {
                "communication_us": round(comm_node_dynamic * 1e6, 2),
                "computation_us": round(comp_dynamic * 1e6, 2),
                "latency_us": round((comp_dynamic + comm_node_dynamic) * 1e6, 2),
                "speedup_EP": round(
                    (ep_comp_dynamic + ep_comm_dynamic) / (comp_dynamic + comm_node_dynamic),
                    2,
                ),
                "speedup_TP": round(
                    (tp_comp_dynamic + tp_comm_dynamic) / (comp_dynamic + comm_node_dynamic),
                    2,
                ),
            },
        },
        "link_balancing": {
            "static": {
                "communication_us": round(comm_link * 1e6, 2),
                "computation_us": round(comp * 1e6, 2),
                "latency_us": round((comp + comm_link) * 1e6, 2),
                "speedup": round(comm_node / comm_link, 2),
                "speedup_EP": round((ep_comp + ep_comm) / (comp + comm_link), 2),
                "speedup_TP": round((tp_comp + tp_comm) / (comp + comm_link), 2),
            },
            "dynamic": {
                "communication_us": round(comm_link_dynamic * 1e6, 2),
                "computation_us": round(comp_dynamic * 1e6, 2),
                "latency_us": round((comp_dynamic + comm_link_dynamic) * 1e6, 2),
                "speedup": round(comm_node_dynamic / comm_link_dynamic, 2),
                "speedup_EP": round(
                    (ep_comp_dynamic + ep_comm_dynamic) / (comp_dynamic + comm_link_dynamic),
                    2,
                ),
                "speedup_TP": round(
                    (tp_comp_dynamic + tp_comm_dynamic) / (comp_dynamic + comm_link_dynamic),
                    2,
                ),
            },
        },
        "communication_distance": {
            "optimized": round(dis, 2),
            "random": round(dis_rand, 2),
        },
    }

    out_path = (
        args.results_json
        if os.path.isabs(args.results_json)
        else os.path.join(cwd, args.results_json)
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
