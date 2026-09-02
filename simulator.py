"""Per-layer MoE placement: regress comm gamma, ILP node balance, Bayesian placement search; saves NPZ + BO plot."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(_ROOT, "evaluation")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EVAL)

from node_allocation import MoE3DPNMOptimizer
from moe_placement_utils import EP_deployment, generate_random_placement

MODEL_SPECS = {
    "mixtral": ("expert_trace/mixtral/experts_reasoning_mixtral.json", 8, 2, 0, 4096, 14336, False, 32),
    "ds": ("expert_trace/ds/experts_reasoning_ds.json", 64, 6, 0, 2048, 1408, True, 26),
    "qwen": ("expert_trace/qwen/experts_reasoning_qwen.json", 64, 8, 0, 3584, 2560, False, 28),
    "qwen35": (
        "expert_trace/qwen35/experts_reasoning_qwen35.json",
        256,
        8,
        0,
        2048,
        512,
        False,
        40,
    ),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cwd", default=".", help="Working directory for trace and results/")
    p.add_argument("--layer-id", type=int, default=1)
    p.add_argument("--comp", type=float, default=10.0, help="Per-device compute (TFLOPS)")
    p.add_argument("--comm", type=float, default=25.0, help="Interconnect bandwidth (GB/s)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--mesh-shape", type=int, nargs=2, metavar=("X", "Y"), default=[4, 8])
    p.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="mixtral")
    p.add_argument(
        "--trace-path",
        default=None,
        help="Optional routing trace override. The model default remains unchanged when omitted.",
    )
    p.add_argument(
        "--draw-regression",
        action="store_true",
        help="Only plot comm regression samples (skip ILP + BO)",
    )
    p.add_argument(
        "--ilp-time-limit",
        type=int,
        default=1800,
        help="Gurobi ILP time limit per layer (seconds)",
    )
    p.add_argument(
        "--ilp-mip-gap",
        type=float,
        default=0.05,
        help="Gurobi MIPGap (relative); lower = tighter ILP solution",
    )
    p.add_argument(
        "--bo-iter",
        type=int,
        default=70,
        help="Bayesian optimization iterations (gp_minimize n_calls)",
    )
    p.add_argument(
        "--bo-initial-points",
        type=int,
        default=10,
        help="Random initial points before GP search in BO",
    )
    p.add_argument(
        "--memory-factor",
        type=float,
        default=None,
        help="Optional per-node expert-weight storage cap as factor * (E/D).",
    )
    p.add_argument("--topology", choices=["mesh", "torus", "fat_tree"], default="mesh")
    p.add_argument("--fat-tree-oversubscription", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = os.path.abspath(args.cwd)
    layer_id = args.layer_id
    comp = args.comp
    BW = args.comm
    batch = args.batch
    mesh_shape = tuple(args.mesh_shape)

    data_path, E, e, SE, h, IS, mlp_first, num_layers = MODEL_SPECS[args.model]
    trace_path = args.trace_path or data_path
    trace_path = trace_path if os.path.isabs(trace_path) else os.path.join(root, trace_path)

    print(f"computation throughput (TFLOPS): {comp:.2f}")
    print(f"communication bandwidth (GB/s): {BW:.2f}")

    D = mesh_shape[0] * mesh_shape[1]

    try:
        with open(trace_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {trace_path}")
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
    folder_path = Path(root) / (
        f"results/reasoning_{args.model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_"
        f"for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_{optimizer.B:.0f}_batches"
    )

    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=True)

    P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)
    M_rand = generate_random_placement(optimizer.D, mesh_shape)
    P_init = P_ep
    x: list[float] = []
    y: list[float] = []
    print("Sampling communication latency for regression...")

    layer_key = str(layer_id + optimizer.mlp_first)
    for b in tqdm(range(256, 512 + 128, 2)):
        raw = random.sample(data[layer_key], b)
        extra = list(range(optimizer.E - SE, optimizer.E))
        random_samples = [list(row) + extra for row in raw]
        comm = optimizer.comm_time_dynamic(P_init, random_samples)[layer_id]
        comm_acc = optimizer.comm_time_acc_dynamic(M_rand, P_init, layer_id, random_samples)
        x.append(comm)
        y.append(comm_acc)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"The parameter gamma is {slope:.2f}")
    print(
        f"intercept is {intercept:.2f}, r_value is {r_value:.2f}, "
        f"p_value is {p_value:.2f}, std_err is {std_err:.2f}"
    )

    if args.draw_regression:
        plt.figure(figsize=(8, 3))
        plt.scatter(x, y, color="blue", label="Communication Samples", s=50, alpha=0.7)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, "r-", linewidth=2, label=f"Regressed Line: y = {slope:.2f}x + {intercept:.2f}")
        plt.legend(fontsize=12)
        plt.xlabel("Node Communication", fontsize=14)
        plt.ylabel("Schedule Communication", fontsize=14)
        plt.title("Regress Results", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_dir = Path(root) / "evaluation" / "figs"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / "communication2.png")
        plt.close()
        return

    file_path = (
        folder_path
        / f"arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz"
    )
    print(f"results will be saved at {file_path}")

    P = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
    optimizer.X, optimizer.Y = mesh_shape

    optimizer.M = M_rand
    ep_comp = optimizer.compute_time(P_ep)[layer_id]
    ep_comm, link = optimizer.comm_time_acc(M_rand, P_ep, layer_id)
    ep_comm *= 2
    tp_comp = optimizer.compute_time(P)[layer_id]
    tp_comm = 2 * optimizer.comm_time(P)[layer_id]
    print(f"TP_communication: {tp_comm*1e6:.2f} us")
    print(f"TP_computation: {tp_comp*1e6:.2f} us")
    print(f"TP_latency: {(tp_comp+tp_comm)*1e6:.2f} us")
    print(f"EP_communication: {ep_comm*1e6:.2f} us")
    print(f"EP_computation: {ep_comp*1e6:.2f} us")
    print(f"EP_latency: {(ep_comp+ep_comm)*1e6:.2f} us")

    comp_map = np.zeros((optimizer.E,))
    random_samples = random.sample(data[str(layer_id + optimizer.mlp_first)], batch)
    for sublist in random_samples:
        for idx in sublist:
            comp_map[idx] += 2 * optimizer.h * optimizer.IS

    tp_comp_dynamic = optimizer.compute_time_dynamic(P, comp_map)[layer_id]
    tp_comm_dynamic = 2 * optimizer.comm_time_dynamic(P, random_samples)[layer_id]
    ep_comp_dynamic = optimizer.compute_time_dynamic(P_ep, comp_map)[layer_id]
    ep_comm_dynamic = 2 * optimizer.comm_time_acc_dynamic(M_rand, P_ep, layer_id, random_samples)

    print(f"TP_communication_dynamic: {tp_comm_dynamic*1e6:.2f} us")
    print(f"TP_computation_dynamic: {tp_comp_dynamic*1e6:.2f} us")
    print(f"TP_latency_dynamic: {(tp_comp_dynamic+tp_comm_dynamic)*1e6:.2f} us")
    print(f"EP_communication_dynamic: {ep_comm_dynamic*1e6:.2f} us")
    print(f"EP_computation_dynamic: {ep_comp_dynamic*1e6:.2f} us")
    print(f"EP_latency_dynamic: {(ep_comp_dynamic+ep_comm_dynamic)*1e6:.2f} us")

    P = optimizer.ilp_solver_gurobi(
        l=layer_id,
        gamma=slope,
        time_limit=args.ilp_time_limit,
        mip_gap=args.ilp_mip_gap,
        memory_factor=args.memory_factor,
    )
    if P is None:
        print("ILP solver returned no solution; abort.")
        sys.exit(1)
    print("Node balance (ILP) finished.")

    comp = optimizer.compute_time(P)[layer_id]
    comm_node, link = optimizer.comm_time_acc(M_rand, P, layer_id)
    comm_node *= 2
    print(f"node_balancing_communication: {comm_node*1e6:.2f} us")
    print(f"node_balancing_computation: {comp*1e6:.2f} us")
    print(f"node_balancing_latency: {(comp+comm_node)*1e6:.2f} us")
    print(f"node_balancing_speedup_EP:{(ep_comp+ep_comm)/(comp+comm_node):.2f}")
    print(f"node_balancing_speedup_TP:{(tp_comp+tp_comm)/(comp+comm_node):.2f}")

    comp_dynamic = optimizer.compute_time_dynamic(P, comp_map)[layer_id]
    comm_node_dynamic = 2 * optimizer.comm_time_acc_dynamic(M_rand, P, layer_id, random_samples)
    print(f"node_balancing_communication_dynamic: {comm_node_dynamic*1e6:.2f} us")
    print(f"node_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"node_balancing_latency_dynamic: {(comp_dynamic+comm_node_dynamic)*1e6:.2f} us")
    print(f"node_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}")
    print(f"node_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_node_dynamic):.2f}")
    Z = P > 0
    M_init = M_rand

    max_iter = args.bo_iter
    M, cost_history = optimizer.optimize_placement_bo(
        M_init,
        Z,
        layer_id,
        max_iter=max_iter,
        random_state=layer_id,
        n_initial_points=args.bo_initial_points,
    )
    bo_best_s = float(np.min(cost_history)) if len(cost_history) else 0.0
    bo_init_s = float(cost_history[0]) if len(cost_history) else 0.0
    if bo_init_s > 0:
        bo_improve = (1 - bo_best_s / bo_init_s) * 100
    else:
        bo_improve = 0.0
    print(
        f"BO: {max_iter} iters, comm_acc best {bo_best_s*1e6:.4f} us "
        f"(init {bo_init_s*1e6:.4f} us, improve {bo_improve:.1f}%)"
    )

    np.savez_compressed(file_path, arr1=P, arr2=M)

    optimizer.M = M
    comm_link, link = optimizer.comm_time_acc(M, P, layer_id)
    comm_link *= 2
    print(f"link_balancing: {comm_link*1e6:.2f} us")
    print(f"link_balancing_computation: {comp*1e6:.2f} us")
    print(f"link_balancing_latency: {(comp+comm_link)*1e6:.2f} us")
    print(f"link_balancing_speedup:{(comm_node)/(comm_link):.2f}")
    print(f"node_link_balancing_speedup:{(ep_comp+ep_comm)/(comp+comm_link):.2f}")

    comm_link_dynamic = 2 * optimizer.comm_time_acc_dynamic(M, P, layer_id, random_samples)
    print(f"link_balancing_dynamic: {comm_link_dynamic*1e6:.2f} us")
    print(f"link_balancing_computation_dynamic: {comp_dynamic*1e6:.2f} us")
    print(f"link_balancing_latency_dynamic: {(comp_dynamic+comm_link_dynamic)*1e6:.2f} us")
    print(f"link_balancing_speedup_dynamic:{(comm_node_dynamic)/(comm_link_dynamic):.2f}")
    print(f"node_link_balancing_speedup_EP_dynamic:{(ep_comp_dynamic+ep_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}")
    print(f"node_link_balancing_speedup_TP_dynamic:{(tp_comp_dynamic+tp_comm_dynamic)/(comp_dynamic+comm_link_dynamic):.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, color="blue", linewidth=1)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Schedule Time", fontsize=12)
    plt.title("Bayesian Optimization", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    bo_png = folder_path / f"BO_{max_iter}_in_layer_{layer_id}.png"
    plt.savefig(bo_png)
    plt.close()


if __name__ == "__main__":
    main()
