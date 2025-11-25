import sys
import os

# 获取当前脚本的绝对路径，并向上追溯到项目根目录（HD-MoE/）
current_dir = os.path.dirname(os.path.abspath(__file__))  # balance2.py 的目录
project_root = os.path.dirname(os.path.dirname(current_dir))  # HD-MoE/ 目录

# 将项目根目录添加到 Python 路径
sys.path.append(project_root)
from node_allocation import MoE3DPNMOptimizer
import numpy as np
import json
import pdb
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy


def comp_overhead(comp, D, batch, h, L, intermediate, e):
    generation = 3 * batch * h**2
    score_context = 2 * batch * L * h
    projection = batch * h**2
    FFN = 2 * e * batch * intermediate * h**2
    return (generation + score_context + projection) / (D * comp)


def comm_overhead(BW, D, batch, h, alpha, e):
    all_reduce = (5 * batch * h * (D - 1)) / (BW * D) + 4 * alpha * D**0.5
    all2all = (4 * batch * h * (e - 1)) / (BW * D) + (batch * h * (D - 1)) / (BW * D) + 4 * alpha * D**0.5  # An estimate
    return all_reduce + all2all  # Consider Attention All-Reduce and MoE All2all


def mem_overhead(BWmem, D, batch, h, L, intermediate, E):
    KV_cache = 2 * batch * h * L
    generation = 3 * h * (batch + h)
    projection = h * (batch + h)
    FFN = 2 * E * intermediate * h**2 + batch * h * (intermediate + 1)
    return (KV_cache + generation + projection + FFN) / (D * BWmem)


def EP_deployment(L, E, D):
    """
    生成专家部署策略矩阵 P，维度为 E×D。
    P[e][d] = a 表示第 e 个专家在第 d 个设备上部署了 a 的权重（0 ≤ a ≤ 1）。
    """
    P = np.zeros((L, E, D))

    if D >= E:
        # D >= E 时，每个专家分配到多个设备，设备权重均匀分布
        k, r = divmod(D, E)  # 每个专家至少分到 k 个设备，前 r 个专家多分 1 个设备
        devices = np.arange(D)  # 设备索引
        np.random.shuffle(devices)  # 随机打乱设备顺序
        start = 0
        for e in range(E):
            num_devices = k + 1 if e < r else k  # 当前专家分到的设备数
            end = start + num_devices
            assigned_devices = devices[start:end]  # 随机分配到设备
            P[:, e, assigned_devices] = 1.0 / num_devices  # 权重均匀分布
            start = end  # 更新下一个起始位置
    else:
        # D < E 时，专家尽可能均衡分配到设备，每个专家只在一个设备
        m, r = divmod(E, D)  # 每个设备至少分到 m 个专家，前 r 个设备多分 1 个专家
        experts = np.arange(E)  # 专家索引
        np.random.shuffle(experts)  # 随机打乱专家顺序
        expert_idx = 0
        for d in range(D):
            num_experts = m + 1 if d < r else m  # 当前设备分到的专家数
            assigned_experts = experts[expert_idx : expert_idx + num_experts]  # 随机分配到专家
            P[:, assigned_experts, d] = 1.0  # 权重为 1
            expert_idx += num_experts
    return P


def generate_random_placement(D, mesh_shape):
    """
    生成随机的设备布局
    :param D: 设备数量
    :param mesh_shape: (X, Y)网格尺寸
    :return: D x X x Y的放置矩阵
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
model = "ds"
dataset_list = ["reasoning", "math", "coding", "writing", "roleplay"]
dataset = dataset_list[0]
sample1 = dataset_list[1]  # adaptive 使用单独的数据集 trace，baseline 用 dataset
mesh_shape = (8, 8)
comp = 5
BW = 50
if model == "mixtral":
    E, e, SE, h, IS, mlp_first, num_layers = 8, 2, 0, 4096, 14336, False, 32  # Mixtral
elif model == "ds":
    E, e, SE, h, IS, mlp_first, num_layers = 64, 6, 0, 2048, 1408, True, 26  # DeepSeekMoE
elif model == "qwen":
    E, e, SE, h, IS, mlp_first, num_layers = 64, 8, 0, 3584, 18944, False, 28  # Qwen2
D = mesh_shape[0] * mesh_shape[1]

data_path = f'expert_trace/{model}/predict/experts_{dataset}_{model}_pre.json'
adaptive_path = f'expert_trace/{model}/new_adaptive/experts_{sample1}_64_-3e4_-2e-2_{model}_adaptive.json'
try:
    with open(data_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)
        data, pre = data1["selected_experts"], data1["predict_experts"]
    sample, pre_sample = data, pre  # baseline 使用 predict trace
    with open(adaptive_path, "r", encoding="utf-8") as f2:
        sample_adap = json.load(f2)
        adaptive_sample, adaptive_pre_sample = sample_adap["selected_experts"], sample_adap["predict_experts"]
except FileNotFoundError:
    print("文件未找到，请检查文件路径和文件名。")
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
P_tp = np.ones((optimizer.layer, optimizer.E, optimizer.D)) / optimizer.D
P_ep = EP_deployment(optimizer.layer, optimizer.E, optimizer.D)

optimizer.X, optimizer.Y = mesh_shape
L = 1024
intermediate = optimizer.IS / optimizer.h
BWmem = 625e9
att_mem = optimizer.layer * (
    comp_overhead(optimizer.comp, optimizer.D, optimizer.B, optimizer.h, L, intermediate, e)
    + mem_overhead(BWmem, optimizer.D, optimizer.B, optimizer.h, L, intermediate, optimizer.E)
)

t_inf = comp_overhead(
    optimizer.comp, optimizer.D, optimizer.B, optimizer.h, L, intermediate, optimizer.e
) + mem_overhead(BWmem, optimizer.D, optimizer.B, optimizer.h, L, intermediate, optimizer.E)
k = 1
while optimizer.optimal_broadcast_chunk(k=k) < t_inf:
    k += 1
k = max(k - 1, 0)
sample_id = random.randint(0, len(sample[str(1)]) - 1)

tp_comp = 0
tp_comm = 0

ep_comp = 0
ep_comm = 0

comp_comp = 0
comp_comm = 0
comp_pre_comp = 0
comp_pre_comm = 0
comp = 0

node_pre_comp = 0
node_pre_comm = 0
comm_link = 0

adaptive_pre_comp = 0
adaptive_pre_comm = 0


for layer_id in tqdm(range(optimizer.layer)):
    file_path = f'results/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id:.0f}.npz'
    loaded_arrays = np.load(file_path)

    comp_path = f'results/comp_balance_only/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id}.npz'
    P = loaded_arrays["arr1"]
    M = loaded_arrays["arr2"]
    loaded_comp_arrays = np.load(comp_path)
    P_comp = loaded_comp_arrays["arr1"]
    M_comp = loaded_comp_arrays["arr2"]

    sample_key = str(layer_id + optimizer.mlp_first)
    index = sample_id % len(sample[sample_key])
    raw_random_samples = sample[sample_key][index]
    adaptive_index = sample_id % len(adaptive_sample[sample_key])
    raw_adaptive_random_samples = adaptive_sample[sample_key][adaptive_index]

    scaling = optimizer.B / len(raw_random_samples) if len(raw_random_samples) > 0 else 0
    adaptive_scaling = optimizer.B / len(raw_adaptive_random_samples) if len(raw_adaptive_random_samples) > 0 else 0

    random_samples = (
        raw_random_samples
        if (len(raw_random_samples) > 0 and isinstance(raw_random_samples[0], (list, tuple)))
        else [raw_random_samples]
    )
    adaptive_random_samples = (
        raw_adaptive_random_samples
        if (len(raw_adaptive_random_samples) > 0 and isinstance(raw_adaptive_random_samples[0], (list, tuple)))
        else [raw_adaptive_random_samples]
    )

    comp_map = np.zeros((optimizer.E))
    adaptive_comp_map = np.zeros((optimizer.E))
    for sublist in random_samples:
        comp_map[sublist] += 2 * optimizer.h * optimizer.IS * scaling
    for sublist in adaptive_random_samples:
        adaptive_comp_map[sublist] += 2 * optimizer.h * optimizer.IS * adaptive_scaling

    M_rand = generate_random_placement(optimizer.D, mesh_shape)
    optimizer.M = M_rand

    tp_comp += optimizer.compute_time(P_tp)[layer_id]
    tp_comm += 2 * optimizer.comm_time(P_tp)[layer_id]

    ep_comp += optimizer.compute_time(P_ep)[layer_id]
    comm_temp, link = optimizer.comm_time_acc(M_rand, P_ep, layer_id)
    ep_comm += comm_temp * 2

    comp_comp += optimizer.compute_time(P_comp)[layer_id]
    comm_temp, link = optimizer.comm_time_acc(M_comp, P_comp, layer_id)
    comm_temp += optimizer.comm_time(P)[layer_id]
    comp_comm += comm_temp * 2

    comp += optimizer.compute_time(P)[layer_id]

    comm_temp, link = optimizer.comm_time_acc(M, P, layer_id)
    comm_link += comm_temp * 2

    if layer_id == 0:
        comp_pre_comp += optimizer.compute_time_dynamic(P_comp, comp_map)[0]
        comp_pre_comm += 2 * optimizer.comm_time_acc_dynamic(M_comp, P_comp, 0, random_samples)

        node_pre_comp += optimizer.compute_time_dynamic(P, comp_map)[0]
        node_pre_comm += 2 * optimizer.comm_time_acc_dynamic(M, P, 0, random_samples)

        adaptive_pre_comp += optimizer.compute_time_dynamic(P, adaptive_comp_map)[0]
        adaptive_pre_comm += 2 * optimizer.comm_time_acc_dynamic(M, P, 0, adaptive_random_samples)

    if layer_id < optimizer.layer - 1:
        pre_path = f'results/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id+1:.0f}.npz'
        pre_arrays = np.load(pre_path)
        P_next = pre_arrays["arr1"]
        M_next = pre_arrays["arr2"]

        pre_comp_path = f'results/comp_balance_only/{dataset}_{model}_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_for_{mesh_shape[0]:.0f}*{mesh_shape[1]:.0f}_mesh_128_batches/arrays_{optimizer.comp*1e-12:.1f}_TFLOPS_{optimizer.BW*1e-9:.1f}_GBPS_in_layer_{layer_id+1}.npz'
        pre_comp_arrays = np.load(pre_comp_path)
        P_comp_next = pre_comp_arrays["arr1"]
        M_comp_next = pre_comp_arrays["arr2"]

        next_key = str(layer_id + 1 + optimizer.mlp_first)
        next_index = sample_id % len(sample[next_key])
        raw_random_samples_next = sample[next_key][next_index]
        next_samples = pre_sample[next_key][next_index]

        adaptive_next_index = sample_id % len(adaptive_sample[next_key])
        raw_adaptive_random_samples_next = adaptive_sample[next_key][adaptive_next_index]
        adaptive_next_samples = adaptive_pre_sample[next_key][adaptive_next_index]

        next_scaling = optimizer.B / len(raw_random_samples_next) if len(raw_random_samples_next) > 0 else 0
        adaptive_next_scaling = (
            optimizer.B / len(raw_adaptive_random_samples_next) if len(raw_adaptive_random_samples_next) > 0 else 0
        )

        random_samples_next = (
            raw_random_samples_next
            if (len(raw_random_samples_next) > 0 and isinstance(raw_random_samples_next[0], (list, tuple)))
            else [raw_random_samples_next]
        )
        adaptive_random_samples_next = (
            raw_adaptive_random_samples_next
            if (len(raw_adaptive_random_samples_next) > 0 and isinstance(raw_adaptive_random_samples_next[0], (list, tuple)))
            else [raw_adaptive_random_samples_next]
        )

        comp_map_next = np.zeros((optimizer.E))
        adaptive_comp_map_next = np.zeros((optimizer.E))
        for sublist in random_samples_next:
            comp_map_next[sublist] += 2 * optimizer.h * optimizer.IS * next_scaling
        for sublist in adaptive_random_samples_next:
            adaptive_comp_map_next[sublist] += 2 * optimizer.h * optimizer.IS * adaptive_next_scaling

        p_copy_comp = copy.deepcopy(P_comp_next)
        p_copy_node = copy.deepcopy(P_next)
        adaptive_p_copy = copy.deepcopy(P_next)

        compute_load_next = np.sum(p_copy_comp * comp_map_next[None, :, None], axis=1)
        for _ in range(k):
            prio, e_idx = optimizer.priority_detection(p_copy_comp, layer_id + 1, next_samples)
            p_copy_comp[layer_id + 1, e_idx, :] = 0
            for sublist in random_samples_next:
                if e_idx in sublist:
                    _, d = np.nonzero(P_comp_next[layer_id + 1][sublist])
                    activate_node = list(d)
                    compute_load_next = np.sum(p_copy_comp * comp_map_next[None, :, None], axis=1)
                    scatter_node = np.argmin(compute_load_next[layer_id + 1, activate_node])
                    compute_load_next[layer_id + 1, scatter_node] += 2 * optimizer.h * optimizer.IS * next_scaling
        comp_pre_comp += np.max(compute_load_next[layer_id + 1]) / optimizer.comp
        comp_pre_comm += 2 * optimizer.comm_time_acc_dynamic(M_comp_next, p_copy_comp, layer_id + 1, random_samples_next)

        node_compute_load_next = np.sum(p_copy_node * comp_map_next[None, :, None], axis=1)
        for _ in range(k):
            prio, e_idx = optimizer.priority_detection(p_copy_node, layer_id + 1, next_samples)
            p_copy_node[layer_id + 1, e_idx, :] = 0
            for sublist in random_samples_next:
                if e_idx in sublist:
                    _, d = np.nonzero(P_next[layer_id + 1][sublist])
                    activate_node = list(d)
                    node_compute_load_next = np.sum(p_copy_node * comp_map_next[None, :, None], axis=1)
                    scatter_node = np.argmin(node_compute_load_next[layer_id + 1, activate_node])
                    node_compute_load_next[layer_id + 1, scatter_node] += (
                        2 * optimizer.h * optimizer.IS * next_scaling
                    )
        node_pre_comp += np.max(node_compute_load_next[layer_id + 1]) / optimizer.comp
        node_pre_comm += 2 * optimizer.comm_time_acc_dynamic(M_next, p_copy_node, layer_id + 1, random_samples_next)

        adaptive_compute_load_next = np.sum(adaptive_p_copy * adaptive_comp_map_next[None, :, None], axis=1)
        for _ in range(k):
            adaptive_prio, adaptive_e_idx = optimizer.priority_detection(
                adaptive_p_copy, layer_id + 1, adaptive_next_samples
            )
            adaptive_p_copy[layer_id + 1, adaptive_e_idx, :] = 0
            for sublist in adaptive_random_samples_next:
                if adaptive_e_idx in sublist:
                    _, d = np.nonzero(P_next[layer_id + 1][sublist])
                    activate_node = list(d)
                    adaptive_compute_load_next = np.sum(adaptive_p_copy * adaptive_comp_map_next[None, :, None], axis=1)
                    scatter_node = np.argmin(adaptive_compute_load_next[layer_id + 1, activate_node])
                    adaptive_compute_load_next[layer_id + 1, scatter_node] += (
                        2 * optimizer.h * optimizer.IS * adaptive_next_scaling
                    )
        adaptive_pre_comp += np.max(adaptive_compute_load_next[layer_id + 1]) / (optimizer.comp)
        adaptive_pre_comm += 2*optimizer.comm_time_acc_dynamic(
            M_next, adaptive_p_copy, layer_id + 1, adaptive_random_samples_next
        )

print(f"TP_communication: {tp_comm*1e6:.2f} us")
print(f"TP_computation: {tp_comp*1e6:.2f} us")
print(f"TP_latency: {(tp_comp+tp_comm+att_mem)*1e6:.2f} us")

print(f"EP_communication: {ep_comm*1e6:.2f} us")
print(f"EP_computation: {ep_comp*1e6:.2f} us")
print(f"EP_latency: {(ep_comp+ep_comm+att_mem)*1e6:.2f} us")

print(f"node_link_balancing_communication: {comm_link*1e6:.2f} us")
print(f"node_link_balancing_computation: {comp*1e6:.2f} us")
print(f"node_link_balancing_latency: {(comp+comm_link+att_mem)*1e6:.2f} us")

print(f"node_link_balancing_speedup_EP:{(ep_comp+ep_comm+att_mem)/(comp+comm_link+att_mem):.2f}")
print(f"node_link_balancing_speedup_TP:{(tp_comp+tp_comm+att_mem)/(comp+comm_link+att_mem):.2f}")
print(f"node_link_balancing_speedup_comp:{(comp_comp+comp_comm+att_mem)/(comp+comm_link+att_mem):.2f}")

print(f"compute_prebroadcast_communication: {comp_pre_comm*1e6:.2f} us")
print(f"compute_prebroadcast_computation: {comp_pre_comp*1e6:.2f} us")
print(f"compute_prebroadcast_latency: {(comp_pre_comp+comp_pre_comm+att_mem)*1e6:.2f} us")

print(f"node_link_prebroadcast_communication: {node_pre_comm*1e6:.2f} us")
print(f"node_link_prebroadcast_computation: {node_pre_comp*1e6:.2f} us")
print(f"node_link_prebroadcast_latency: {(node_pre_comp+node_pre_comm+att_mem)*1e6:.2f} us")

print(f"adaptive_pre_communication: {adaptive_pre_comm*1e6:.2f} us")
print(f"adaptive_pre_computation: {adaptive_pre_comp*1e6:.2f} us")
print(f"adaptive_pre_latency: {(adaptive_pre_comp+adaptive_pre_comm+att_mem)*1e6:.2f} us")

import os
import json

# 配置参数
config = {
    "mesh_shape": mesh_shape,  # 从变量 mesh_shape 获取
    "model": model,  # DeepSeekMoE 模型
    "dataset": dataset,
    "sample": sample1,
    "comp_TFLOPS": optimizer.comp * 1e-12,  # 计算能力 (TFLOPS)
    "BW_GBPS": optimizer.BW * 1e-9,  # 带宽 (GBPS)
    "batch": optimizer.B,  # 从变量 batch 获取
}

# 构建结果字典
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
            "speedup_EP": round((ep_comp + ep_comm + att_mem) / (comp + comm_link + att_mem), 2),
            "speedup_TP": round((tp_comp + tp_comm + att_mem) / (comp + comm_link + att_mem), 2),
            "speedup_comp": round((comp_comp + comp_comm + att_mem) / (comp + comm_link + att_mem), 2),
        },
        "prebroadcast": {
            "communication_us": round(node_pre_comm * 1e6, 2),
            "computation_us": round(node_pre_comp * 1e6, 2),
            "latency_us": round((node_pre_comp + node_pre_comm + att_mem) * 1e6, 2),
        },
    },
    "adaptive_pre": {
        "communication_us": round(adaptive_pre_comm * 1e6, 2),
        "computation_us": round(adaptive_pre_comp * 1e6, 2),
        "latency_us": round((adaptive_pre_comp + adaptive_pre_comm + att_mem) * 1e6, 2),
    },
}

file_path = "evaluation/results/result_hda_e2e.json"
new_data = result  # 你的新数据

# 如果文件存在，尝试读取旧数据；若为空或格式异常则回退为空列表
combined_data = []
if os.path.exists(file_path):
    try:
        with open(file_path, "r") as f:
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

# 写回文件
with open(file_path, "w") as f:
    json.dump(combined_data, f, indent=4)
