#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay fastchat hardware-aware MoE gating (HD-MoE) on CPU/CUDA from per-layer expert softmax
(e.g. npz from reconstruct_expert_scores_from_topk.py), calling moe_gating_hd.apply_hd_moe_routing per layer.
By default **all tokens in a layer share one batch** (production needs HD_MOE_BATCH_SIZE >= rows; this script
calls set_hd_moe_overrides(batch_size=...) each forward to match the current batch size).

Outputs:
  - **JSON** (aligned with experts_*_score.json):
    - **original_selected_experts**: top-k expert ids before reward (flat per layer, same token order as scores npz)
    - **selected_experts**: top-k after reward (HD routing)
    - **simulation_meta** (optional): rewards, token stats, etc.
  - Optional **NPZ**: per-layer topk before/after and softmax values for analysis

Requires P-matrix npz paths consistent with moe_gating_hd._get_npz_path (see TCAD/results/...).

Example:
  python3 simulate_hd_gating_from_scores.py \\
    --scores-npz /path/to/gating_score_reasoning.npz \\
    --output-json /path/to/hd_sim_experts.json \\
    --reward-comp 8000 --reward-comm -0.05 \\
    --top-k 8 --model-name qwen
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
# fastchat package: evaluation/scripts -> TCAD/fastchat
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TCAD_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_FASTCHAT_PKG = os.path.join(_TCAD_ROOT, "fastchat")
if _FASTCHAT_PKG not in sys.path:
    sys.path.insert(0, _FASTCHAT_PKG)

from fastchat.llm_judge import moe_gating_hd  # noqa: E402


def _pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _simulate_chunk(
    original_scores: torch.Tensor,
    layer_id: int,
    device: torch.device,
    model_name: str,
    top_k: int,
    reward_comp: float,
    reward_comm: float,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    bool,
]:
    """
    Returns (topk_before, softmax_at_before, topk_after, softmax_at_after, hd_applied).
    If hd_applied is False, after equals before; softmax_at_after gathered from before.
    """
    tb = torch.topk(original_scores, top_k, dim=-1, sorted=False)
    topk_before = tb.indices
    vals_before = tb.values

    res = moe_gating_hd.apply_hd_moe_routing(
        original_scores,
        layer_id,
        device,
        model_name,
        top_k,
        reward_comp=reward_comp,
        reward_comm=reward_comm,
    )
    if res is None:
        ta = topk_before
        # Non-HD path: after equals before
        va = vals_before
        return topk_before, vals_before, ta, va, False

    topk_after, topk_weight_after = res
    # topk_weight_after = original_scores.gather(1, topk_after)
    va = topk_weight_after
    vb = vals_before
    return topk_before, vb, topk_after, va, True


def run_layer(
    mat: np.ndarray,
    layer_id: int,
    device: torch.device,
    model_name: str,
    top_k: int,
    reward_comp: float,
    reward_comm: float,
    chunk_size: int,
    mesh: Optional[Tuple[int, int]] = None,
    hd_comp: Optional[float] = None,
    hd_bw: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """mat: (n_tokens, E). chunk_size<=0 means entire layer in one HD batch."""
    n, E = mat.shape
    topk_before = np.zeros((n, top_k), dtype=np.int32)
    topk_after = np.zeros((n, top_k), dtype=np.int32)
    sm_before = np.zeros((n, top_k), dtype=np.float32)
    sm_after = np.zeros((n, top_k), dtype=np.float32)
    hd_ok = 0
    hd_fail = 0

    step = n if chunk_size <= 0 else chunk_size
    for start in range(0, n, step):
        end = min(n, start + step)
        bs = end - start
        print(
            f"[cmd] run_layer layer_id={layer_id} token_range=[{start},{end}) "
            f"batch_size={bs} chunk_step={step}",
            file=sys.stderr,
        )
        # Match apply_hd_moe_routing: HD_MOE_BATCH_SIZE >= batch rows
        moe_gating_hd.set_hd_moe_overrides(
            batch_size=bs,
            mesh_shape=mesh,
            comp=hd_comp,
            bw=hd_bw,
            reward_comp=reward_comp,
            reward_comm=reward_comm,
            use_original_gating=False,
            record_gating_softmax=False,
        )
        chunk = torch.from_numpy(mat[start:end].astype(np.float64)).to(device)

        tpb, vb, tpa, va, applied = _simulate_chunk(
            chunk,
            layer_id,
            device,
            model_name,
            top_k,
            reward_comp,
            reward_comm,
        )
        assert tpb is not None and tpa is not None and vb is not None and va is not None
        topk_before[start:end] = tpb.cpu().numpy().astype(np.int32)
        topk_after[start:end] = tpa.cpu().numpy().astype(np.int32)
        sm_before[start:end] = vb.cpu().numpy().astype(np.float32)
        sm_after[start:end] = va.cpu().numpy().astype(np.float32)
        if applied:
            hd_ok += end - start
        else:
            hd_fail += end - start

    return topk_before, topk_after, sm_before, sm_after, hd_ok, hd_fail


def _topk_arrays_to_trace_dict(
    layer_keys: List[str],
    topk_before: Dict[str, np.ndarray],
    topk_after: Dict[str, np.ndarray],
) -> Tuple[Dict[str, List[List[int]]], Dict[str, List[List[int]]]]:
    """Layer id string -> token rows as trace (each row k ints)."""
    orig: Dict[str, List[List[int]]] = {}
    sel: Dict[str, List[List[int]]] = {}
    for lk in layer_keys:
        lid = lk.replace("layer_", "")
        tb = topk_before[lk]
        ta = topk_after[lk]
        orig[lid] = [row.astype(int).tolist() for row in tb]
        sel[lid] = [row.astype(int).tolist() for row in ta]
    return orig, sel


def load_scores_npz(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    z = np.load(path, allow_pickle=True)
    meta = {}
    if "meta" in z.files:
        try:
            meta = json.loads(str(z["meta"]))
        except Exception:
            meta = {}
    layers: Dict[str, np.ndarray] = {}
    for k in z.files:
        if k == "meta":
            continue
        if k.startswith("layer_"):
            layers[k] = np.asarray(z[k])
    if not layers:
        raise ValueError(f"no layer_* arrays in npz: {path}")
    return layers, meta


def load_layer_reward_scale(path: Optional[str]) -> Dict[int, float]:
    if not path:
        return {}
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "scale" in data:
        data = data["scale"]
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON dict or {{'scale': dict}} in {path}")
    return {int(k): float(v) for k, v in data.items()}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="HD-MoE simulation on reconstructed softmax; top-k before/after reward"
    )
    p.add_argument(
        "--scores-npz",
        type=str,
        required=True,
        help="npz from reconstruct_expert_scores_from_topk.py; qwen/mixtral often layer_0.., ds often layer_1.. (trace JSON keys)",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="JSON aligned with experts_*_score.json (original_selected_experts + selected_experts)",
    )
    p.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="Optional: extra npz with per-layer numpy arrays",
    )
    p.add_argument("--top-k", type=int, default=8, help="top-k; must match model")
    p.add_argument(
        "--model-name",
        type=str,
        default="qwen",
        choices=["qwen", "qwen35", "mixtral", "ds"],
        help="HD_MOE_CONFIGS key (E, P paths, ...)",
    )
    p.add_argument("--reward-comp", type=float, default=0.0)
    p.add_argument("--reward-comm", type=float, default=0.0)
    p.add_argument(
        "--layer-reward-scale-json",
        type=str,
        default=None,
        help="Optional layer_id -> scalar map. HDA rewards become reward_* * scale[layer_id].",
    )
    p.add_argument("--hd-mesh-rows", type=int, default=None)
    p.add_argument("--hd-mesh-cols", type=int, default=None)
    p.add_argument(
        "--hd-comp",
        type=float,
        default=None,
        help="Compute throughput in TFLOPS (or FLOP/s when >= 1e6).",
    )
    p.add_argument(
        "--hd-bw",
        type=float,
        default=None,
        help="Link bandwidth in GB/s (or B/s when >= 1e6).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Tokens per batch. <=0: whole layer one batch (recommended). "
        ">0: stride; each forward sets HD_MOE_BATCH_SIZE to current rows.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for P matrix and compute",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer ids, e.g. 0,1,2; default all layer_* in npz",
    )
    args = p.parse_args(argv)

    if args.hd_comp is not None and args.hd_comp < 1.0e6:
        args.hd_comp *= 1.0e12
    if args.hd_bw is not None and args.hd_bw < 1.0e6:
        args.hd_bw *= 1.0e9

    _cmd_argv = argv if argv is not None else sys.argv
    print(f"[cmd] {shlex.join([sys.executable] + _cmd_argv)}", file=sys.stderr)

    if not args.output_json and not args.output_npz:
        p.error("specify at least one of --output-json or --output-npz")

    mesh = None
    if args.hd_mesh_rows is not None and args.hd_mesh_cols is not None:
        mesh = (args.hd_mesh_rows, args.hd_mesh_cols)

    device = _pick_device(args.device)
    layers_mat, scores_meta = load_scores_npz(os.path.expanduser(args.scores_npz))
    reward_scale_by_layer = load_layer_reward_scale(args.layer_reward_scale_json)

    layer_keys = sorted(layers_mat.keys(), key=lambda x: int(x.replace("layer_", "")))
    if args.layers:
        want = {f"layer_{x.strip()}" for x in args.layers.split(",") if x.strip()}
        layer_keys = [k for k in layer_keys if k in want]

    npz_out: Dict[str, Any] = {}
    topk_before_dict: Dict[str, np.ndarray] = {}
    topk_after_dict: Dict[str, np.ndarray] = {}
    total_ok = total_fail = 0

    for lk in layer_keys:
        mat = layers_mat[lk]
        lid = int(lk.replace("layer_", ""))
        print(f"[layer] {lk} shape={mat.shape} ...", file=sys.stderr)
        # ds: scores npz layer_1..L matches JSON keys "1".."L"; P npz uses in_layer_0..L-1 (e2e layer_id loop).
        # qwen/mixtral npz starts at layer_0; no offset.
        hd_layer_id = lid - 1 if args.model_name == "ds" else lid
        if hd_layer_id < 0:
            raise ValueError(
                f"cannot map {lk} to HD layer index for model {args.model_name!r} (hd_layer_id={hd_layer_id}); "
                "ds score npz should start at layer_1."
            )
        reward_scale = reward_scale_by_layer.get(hd_layer_id, reward_scale_by_layer.get(lid, 1.0))
        tb, ta, sb, sa, ok, fail = run_layer(
            mat,
            hd_layer_id,
            device,
            args.model_name,
            args.top_k,
            args.reward_comp * reward_scale,
            args.reward_comm * reward_scale,
            args.chunk_size,
            mesh=mesh,
            hd_comp=args.hd_comp,
            hd_bw=args.hd_bw,
        )
        topk_before_dict[lk] = tb
        topk_after_dict[lk] = ta
        npz_out[f"{lk}_topk_before"] = tb
        npz_out[f"{lk}_topk_after"] = ta
        npz_out[f"{lk}_softmax_at_topk_before"] = sb
        npz_out[f"{lk}_softmax_at_topk_after"] = sa
        total_ok += ok
        total_fail += fail

    meta = {
        "source_scores_npz": os.path.abspath(os.path.expanduser(args.scores_npz)),
        "scores_meta": scores_meta,
        "model_name": args.model_name,
        "top_k": args.top_k,
        "reward_comp": args.reward_comp,
        "reward_comm": args.reward_comm,
        "layer_reward_scale_json": os.path.abspath(os.path.expanduser(args.layer_reward_scale_json)) if args.layer_reward_scale_json else None,
        "layer_reward_scale_layers": len(reward_scale_by_layer),
        "chunk_size": args.chunk_size,
        "chunk_mode": "full_layer" if args.chunk_size <= 0 else f"stride_{args.chunk_size}",
        "device": str(device),
        "hd_tokens_applied": total_ok,
        "hd_tokens_fallback": total_fail,
        "layers": layer_keys,
    }

    if args.output_json:
        orig_json, sel_json = _topk_arrays_to_trace_dict(
            layer_keys, topk_before_dict, topk_after_dict
        )
        json_payload = {
            "original_selected_experts": orig_json,
            "selected_experts": sel_json,
            "simulation_meta": meta,
        }
        jpath = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(jpath) or ".", exist_ok=True)
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, separators=(",", ":"))
        print(f"[write] JSON -> {jpath}", file=sys.stderr)

    if args.output_npz:
        npz_out["meta"] = json.dumps(meta, ensure_ascii=False)
        outp = os.path.expanduser(args.output_npz)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        np.savez_compressed(outp, **npz_out)
        print(f"[write] NPZ -> {outp}", file=sys.stderr)

    print(
        f"[done] HD applied tokens: {total_ok}, fallback (no HD): {total_fail}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
