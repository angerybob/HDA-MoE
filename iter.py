"""One-off: load a saved placement NPZ, compare comm vs random init, plot vertical link heatmaps."""

import sys

sys.path.append("/data/home/haochenhuang/deployment")
from node_allocation import MoE3DPNMOptimizer
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import ast


def EP_deployment(L, E, D):
    """
    Build per-layer expert deployment tensor P with shape (L, E, D).
    P[l, e, d] = a means expert e places fraction a of its weights on device d (0 <= a <= 1).
    """
    P = np.zeros((L, E, D))

    if D >= E:
        k, r = divmod(D, E)
        devices = np.arange(D)
        np.random.shuffle(devices)
        start = 0
        for e in range(E):
            num_devices = k + 1 if e < r else k
            end = start + num_devices
            assigned_devices = devices[start:end]
            P[:, e, assigned_devices] = 1.0 / num_devices
            start = end
    else:
        m, r = divmod(E, D)
        experts = np.arange(E)
        np.random.shuffle(experts)
        expert_idx = 0
        for d in range(D):
            num_experts = m + 1 if d < r else m
            assigned_experts = experts[expert_idx : expert_idx + num_experts]
            P[:, assigned_experts, d] = 1.0
            expert_idx += num_experts
    return P


def generate_random_placement(D, mesh_shape):
    """
    Random device placement on a 2D mesh.
    :return: array of shape D x X x Y (one-hot over mesh cells).
    """
    X, Y = mesh_shape
    all_positions = [(x, y) for x in range(X) for y in range(Y)]

    if len(all_positions) < D:
        raise ValueError(f"Mesh size {X}x{Y} cannot accommodate {D} devices")

    selected = random.sample(all_positions, D)
    placement = np.zeros((D, X, Y), dtype=int)
    for d, (x, y) in enumerate(selected):
        placement[d, x, y] = 1
    return placement


batch = 128

try:
    with open(
        "/data/home/haochenhuang/deployment/experts_reasoning_ds.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    with open(
        "/data/home/haochenhuang/deployment/experts_reasoning_ds.json",
        "r",
        encoding="utf-8",
    ) as f1:
        sample = json.load(f1)
except FileNotFoundError:
    print("File not found; check path and filename.")
    raise

optimizer = MoE3DPNMOptimizer(
    E=64, h=2048, B=batch, BW=25e9, comp=10e12, routing_trace=data
)
P_tp = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)

layer_id = 1
file_path = "/data/home/haochenhuang/deployment/results/reasoning_ds_10.0_TFLOPS_25.0_GBPS_for_8*8_mesh_128_batches/arrays_10.0_TFLOPS_25.0_GBPS_in_layer_1.npz"

loaded_arrays = np.load(file_path)

# NPZ: arr1 = expert placement matrix P; arr2 = device mapping M.
P = loaded_arrays["arr1"]
M1 = loaded_arrays["arr2"]
mesh_shape = (8, 8)
M_init = generate_random_placement(optimizer.D, mesh_shape)
comm, link = optimizer.comm_time_acc(M_init, P_ep, layer_id)
comm_ideal = optimizer.comm_time(P_ep)[layer_id]
print(comm_ideal / comm)

comm_m, link_m = optimizer.comm_time_acc(M1, P, layer_id)
speedup = comm / comm_m
print(comm_ideal * 0.3 / comm_m)
print(f"speedup:{speedup:.2f}")

# To plot SA/BO convergence, run optimize_placement_* above and pass cost_history into matplotlib.
out_npz = f"/data/home/haochenhuang/deployment/BO_node_arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz"
np.savez_compressed(out_npz, arr1=P, arr2=M1)


def load(P, optimizer, layer_id):
    single_comm = np.zeros((optimizer.layer, optimizer.D))
    for lid, layer_fg in optimizer.fg[optimizer.e].items():
        for list_key, freq in layer_fg.items():
            group = ast.literal_eval(list_key)
            devices = np.sum(P[lid][group] > 0, axis=0)
            redundant = freq * optimizer.B * optimizer.h * (devices > 0)
            single_comm[lid] += redundant
    comm_load = 4 * single_comm[layer_id] / (optimizer.BW)
    load = comm_load
    return load.reshape(8, 8)


def link(optimizer, M, P, layer_id, mesh_size):
    """Map per-link traffic from comm_time_acc onto horizontal/vertical edges of the 2D mesh."""
    _comm_time, link_load = optimizer.comm_time_acc(M, P, layer_id)

    horizontal_links = np.zeros((mesh_size[0], mesh_size[1] - 1))
    vertical_links = np.zeros((mesh_size[0] - 1, mesh_size[1]))

    for (src, dst), load in link_load.items():
        x1, y1 = src
        x2, y2 = dst
        if x1 == x2:
            min_y = min(y1, y2)
            horizontal_links[x1, min_y] += load
        elif y1 == y2:
            min_x = min(x1, x2)
            vertical_links[min_x, y1] += load
    return horizontal_links, vertical_links


load_EP = load(P_ep, optimizer, layer_id)
load_ours = load(P, optimizer, layer_id)

horizontal_links_init, vertical_links_init = link(
    optimizer, M_init, P, layer_id, mesh_shape
)
horizontal_links, vertical_links = link(optimizer, M1, P, layer_id, mesh_shape)

vmin = min(np.min(horizontal_links), np.min(vertical_links))
vmax = max(np.max(horizontal_links), np.max(vertical_links))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(
    vertical_links_init, ax=axes[0], cmap="YlGnBu", vmin=vmin, vmax=vmax, annot=False
)
axes[0].set_title("Initial Vertical Link Congestion (Total Occupation Time)")
axes[0].set_xlabel("X Coordinate")
axes[0].set_ylabel("Y Coordinate")

sns.heatmap(vertical_links, ax=axes[1], cmap="YlGnBu", vmin=vmin, vmax=vmax, annot=False)
axes[1].set_title("Vertical Link Congestion (Total Occupation Time)")
axes[1].set_xlabel("X Coordinate")
axes[1].set_ylabel("Y Coordinate")

plt.tight_layout()
plt.savefig(
    f"/data/home/haochenhuang/deployment/evaluation/BO_vertical_for_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.png"
)
plt.close()
