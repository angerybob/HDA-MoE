"""One-layer TP/EP vs node-balance micro-benchmark (legacy DeepSeek-style NPZ path layout)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_ROOT, "evaluation")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EVAL)

from node_allocation import MoE3DPNMOptimizer
from moe_placement_utils import EP_deployment, generate_random_placement


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cwd", default=".", help="Working directory for trace and results paths")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--mesh", type=int, nargs=2, metavar=("X", "Y"), default=[8, 8])
    p.add_argument("--layer-id", type=int, default=11)
    p.add_argument(
        "--trace",
        default="expert_trace/ds/experts_reasoning_ds.json",
        help="Routing trace JSON (relative to cwd unless absolute)",
    )
    p.add_argument(
        "--comp",
        type=float,
        default=10.0,
        help="TFLOPS (used in default NPZ path)",
    )
    p.add_argument(
        "--bw",
        type=float,
        default=25.0,
        help="GB/s (used in default NPZ path)",
    )
    p.add_argument(
        "--npz",
        default=None,
        help="Override full path to layer npz; default results/{comp}_TFLOPS_{bw}_GBPS_for_{batch}_batches/arrays_...",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = os.path.abspath(args.cwd)
    batch = args.batch
    mesh_shape = tuple(args.mesh)
    D = mesh_shape[0] * mesh_shape[1]
    layer_id = args.layer_id

    E, e, SE, h, IS, mlp_first, num_layers = 64, 6, 0, 2048, 1408, True, 26

    trace_path = args.trace if os.path.isabs(args.trace) else os.path.join(root, args.trace)
    try:
        with open(trace_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {trace_path}")
        sys.exit(1)

    optimizer = MoE3DPNMOptimizer(
        E=E,
        e=e,
        SE=SE,
        h=h,
        IS=IS,
        B=batch,
        D=D,
        BW=75e9,
        comp=2.5e12,
        num_layers=num_layers,
        mlp_first=mlp_first,
        routing_trace=data,
    )
    P_tp = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
    P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)
    optimizer.X, optimizer.Y = mesh_shape

    if args.npz:
        file_path = args.npz if os.path.isabs(args.npz) else os.path.join(root, args.npz)
    else:
        subdir = f"{args.comp:.1f}_TFLOPS_{args.bw:.1f}_GBPS_for_{batch}_batches"
        file_path = os.path.join(
            root,
            "results",
            subdir,
            f"arrays_{args.comp:.1f}_TFLOPS_{args.bw:.1f}_GBPS_in_layer_{layer_id:.0f}.npz",
        )

    loaded_arrays = np.load(file_path)
    P = loaded_arrays["arr1"]
    _ = loaded_arrays["arr2"]

    comp_map = np.zeros((optimizer.E,))
    sample = data
    random_samples = random.sample(sample[str(layer_id + 1)], batch)
    for sublist in random_samples:
        comp_map[sublist] += 8 * optimizer.h**2

    M_rand = generate_random_placement(optimizer.D, mesh_shape)
    optimizer.M = M_rand

    tp_comp = optimizer.compute_time(P_tp)[layer_id]
    tp_comm = 2 * optimizer.comm_time(P_tp)[layer_id]

    tp_comp_dynamic = optimizer.compute_time_dynamic(P_tp, comp_map)[layer_id]
    tp_comm_dynamic = 2 * optimizer.comm_time_dynamic(P_tp, random_samples)[layer_id]

    ep_comp = optimizer.compute_time(P_ep)[layer_id]
    comm_temp, link = optimizer.comm_time_acc(M_rand, P_ep, layer_id)
    ep_comm = comm_temp * 2

    ep_comp_dynamic = optimizer.compute_time_dynamic(P_ep, comp_map)[layer_id]
    ep_comm_dynamic = 2 * optimizer.comm_time_acc_dynamic(M_rand, P_ep, layer_id, random_samples)
    ep_ideal = 2 * optimizer.comm_time(P_ep)[layer_id]

    print(f"TP_communication: {tp_comm*1e6:.2f} us")
    print(f"TP_computation: {tp_comp*1e6:.2f} us")
    print(f"TP_latency: {(tp_comp+tp_comm)*1e6:.2f} us")
    print(f"TP_communication_dynamic: {tp_comm_dynamic*1e6:.2f} us")
    print(f"TP_computation_dynamic: {tp_comp_dynamic*1e6:.2f} us")
    print(f"TP_latency_dynamic: {(tp_comp_dynamic+tp_comm_dynamic)*1e6:.2f} us")
    print(f"EP_communication: {ep_comm*1e6:.2f} us")
    print(f"EP_communication_ideal: {ep_ideal*1e6:.2f} us")
    print(f"EP_computation: {ep_comp*1e6:.2f} us")
    print(f"EP_latency: {(ep_comp+ep_comm)*1e6:.2f} us")
    print(f"EP_communication_dynamic: {ep_comm_dynamic*1e6:.2f} us")
    print(f"EP_computation_dynamic: {ep_comp_dynamic*1e6:.2f} us")
    print(f"EP_latency_dynamic: {(ep_comp_dynamic+ep_comm_dynamic)*1e6:.2f} us")

    comp = optimizer.compute_time(P)[layer_id]
    comm_node, link = optimizer.comm_time_acc(M_rand, P, layer_id)
    comm_node *= 2
    print(f"node_balancing_communication: {comm_node*1e6:.2f} us")
    print(f"node_balancing_computation: {comp*1e6:.2f} us")
    print(f"node_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
    print(f"node_balancing_speedup:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")


if __name__ == "__main__":
    main()
