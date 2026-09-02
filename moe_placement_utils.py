"""Utilities for MoE expert placement (EP deployment and random mesh layout)."""

import random

import numpy as np


def EP_deployment(L, E, D):
    """
    Generate expert deployment matrix P with shape (L, E, D).
    P[l, e, d] = weight of expert e on device d (0 <= weight <= 1).
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
    """Random device layout: D x X x Y placement matrix."""
    X, Y = mesh_shape
    all_positions = [(x, y) for x in range(X) for y in range(Y)]

    if len(all_positions) < D:
        raise ValueError(f"Mesh size {X}x{Y} cannot accommodate {D} devices")

    selected = random.sample(all_positions, D)
    placement = np.zeros((D, X, Y), dtype=int)
    for d, (x, y) in enumerate(selected):
        placement[d, x, y] = 1
    return placement
