#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用重构得到的各层专家 softmax（如 reconstruct_expert_scores_from_topk.py 输出的 npz）
在 CPU/CUDA 上复现 fastchat 的 hardware-aware MoE gating（HD-MoE）：对每层调用 moe_gating_hd.apply_hd_moe_routing。
默认 **整层所有 token 在同一 batch**（与线上一致需令 `HD_MOE_BATCH_SIZE` ≥ batch 行数；本脚本在每次前向按当前 batch 大小同步 `set_hd_moe_overrides(batch_size=...)`）。

输出：
  - 推荐 **JSON**（与 experts_*_score.json 对齐）：仅含
    - **original_selected_experts**：加 reward **前** 的 top-k 专家索引（每层与 scores npz 相同 token 顺序的扁平列表）
    - **selected_experts**：加 reward **后** 的 top-k（hardware-aware gating 路由结果）
    - **simulation_meta**（可选）：reward、token 统计等
  - 可选 **NPZ**：保留各层 `layer_k_topk_before/after` 与 softmax 取值，便于数值分析

依赖：需存在与 moe_gating_hd._get_npz_path 一致的 P 矩阵 npz（见 TCAD/results/...）。

用法示例：
  python3 simulate_hd_gating_from_scores.py \\
    --scores-npz /path/to/reconstructed_softmax.npz \\
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
import pdb
# fastchat：evaluation/scripts -> TCAD/fastchat
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
    返回 (topk_before, softmax_at_before, topk_after, softmax_at_after, hd_applied)。
    hd_applied 为 False 时 after 与 before 相同，softmax_at_after 在 before 上 gather。
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
        # 与「未走 HD」一致：after 即 before
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
    """mat: (n_tokens, E). chunk_size<=0 表示本层所有 token 一次送入 HD（同一 batch）。"""
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
        # 与 apply_hd_moe_routing 内判断一致：须 HD_MOE_BATCH_SIZE >= batch 行数
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
    """层 id 字符串 -> 与 trace 一致的 token 行列表（每行为 k 个 int）。"""
    orig: Dict[str, List[List[int]]] = {}
    sel: Dict[str, List[List[int]]] = {}
    for lk in layer_keys:
        lid = lk.replace("layer_", "")
        # pdb.set_trace()
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
        raise ValueError(f"npz 中未找到 layer_* 数组: {path}")
    return layers, meta


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="对重构 softmax 做 hardware-aware gating 仿真，输出 reward 前后 top-k")
    p.add_argument(
        "--scores-npz",
        type=str,
        required=True,
        help="reconstruct_expert_scores_from_topk.py 输出的 npz；qwen/mixtral 多为 layer_0..，ds 多为 layer_1..（与 trace JSON 键对齐）",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="写出与 experts_*_score.json 对齐的 JSON（original_selected_experts + selected_experts）",
    )
    p.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="可选：额外写出含逐层 numpy 数组的 npz",
    )
    p.add_argument("--top-k", type=int, default=8, help="top-k，须与模型一致")
    p.add_argument(
        "--model-name",
        type=str,
        default="qwen",
        choices=["qwen", "mixtral", "ds"],
        help="HD_MOE_CONFIGS 键，决定 E、P 路径等",
    )
    p.add_argument("--reward-comp", type=float, default=0.0)
    p.add_argument("--reward-comm", type=float, default=0.0)
    p.add_argument("--hd-mesh-rows", type=int, default=None)
    p.add_argument("--hd-mesh-cols", type=int, default=None)
    p.add_argument("--hd-comp", type=float, default=None, help="TFLOPS，覆盖默认 comp")
    p.add_argument("--hd-bw", type=float, default=None, help="BPS，覆盖默认 BW")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="每批 token 数。0 或负数表示本层**全部 token 同一 batch**（推荐，与「所有 score 一起算 HD」一致）；"
        "为正整数则按该步长切分；每次前向会同步 HD_MOE_BATCH_SIZE=当前 batch 行数。",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="每层只处理前 N 个 token（调试用，显著加速）；默认处理该层全部 token",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="P 矩阵与计算所在设备",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="只处理这些层，逗号分隔，如 0,1,2；默认 npz 中全部 layer_*",
    )
    args = p.parse_args(argv)

    _cmd_argv = argv if argv is not None else sys.argv
    print(f"[cmd] {shlex.join([sys.executable] + _cmd_argv)}", file=sys.stderr)

    if not args.output_json and not args.output_npz:
        p.error("请至少指定 --output-json 或 --output-npz")

    if args.max_tokens is not None and args.max_tokens <= 0:
        p.error("--max-tokens 须为正整数或省略")

    mesh = None
    if args.hd_mesh_rows is not None and args.hd_mesh_cols is not None:
        mesh = (args.hd_mesh_rows, args.hd_mesh_cols)

    device = _pick_device(args.device)
    layers_mat, scores_meta = load_scores_npz(os.path.expanduser(args.scores_npz))

    layer_keys = sorted(layers_mat.keys(), key=lambda x: int(x.replace("layer_", "")))
    if args.layers:
        want = {f"layer_{x.strip()}" for x in args.layers.split(",") if x.strip()}
        layer_keys = [k for k in layer_keys if k in want]

    npz_out: Dict[str, Any] = {}
    topk_before_dict: Dict[str, np.ndarray] = {}
    topk_after_dict: Dict[str, np.ndarray] = {}
    total_ok = total_fail = 0

    for lk in layer_keys:
        mat_full = layers_mat[lk]
        n_full = int(mat_full.shape[0])
        mat = (
            mat_full[: args.max_tokens]
            if args.max_tokens is not None
            else mat_full
        )
        lid = int(lk.replace("layer_", ""))
        if args.max_tokens is not None and n_full > args.max_tokens:
            print(
                f"[layer] {lk} using first {args.max_tokens}/{n_full} tokens, shape={mat.shape} ...",
                file=sys.stderr,
            )
        else:
            print(f"[layer] {lk} shape={mat.shape} ...", file=sys.stderr)
        # ds：expert_trace / npz 为 layer_1..L，与 JSON 键 "1".."L" 一致；TCAD/results 下 P 矩阵文件名为
        # in_layer_0..L-1（与 e2e 中 layer_id 循环一致）。qwen/mixtral 的 npz 为 layer_0 起，无需偏移。
        hd_layer_id = lid - 1 if args.model_name == "ds" else lid
        if hd_layer_id < 0:
            raise ValueError(
                f"模型 {args.model_name!r} 下 {lk} 无法映射到 HD 层索引（hd_layer_id={hd_layer_id}）；"
                "ds 的分数 npz 应从 layer_1 开始。"
            )
        tb, ta, sb, sa, ok, fail = run_layer(
            mat,
            hd_layer_id,
            device,
            args.model_name,
            args.top_k,
            args.reward_comp,
            args.reward_comm,
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
        "chunk_size": args.chunk_size,
        "chunk_mode": "full_layer" if args.chunk_size <= 0 else f"stride_{args.chunk_size}",
        "max_tokens_per_layer": args.max_tokens,
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
        f"[done] HD 生效 token 累计: {total_ok}, 回退(未进 HD): {total_fail}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
