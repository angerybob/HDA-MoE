#!/usr/bin/env python3
"""Measure expert redundancy and HDA output perturbation on sampled hidden states.

The script loads a MoE language model, captures hidden states entering selected
MoE layers, applies original vs. HDA top-k routing on the same states, and
reports:

  * expert-output cosine similarity between original top-k experts and the
    experts immediately after top-k;
  * normalized routed-MoE output difference;
  * RMS delta-p and next-token top-1 agreement, when logit metrics are enabled;
  * top-1 preservation and top-k overlap;
  * optional logit KL and perplexity change from a second HDA-patched forward.

Expert-output similarity is measured from the original MoE only. HDA routing is
used only for perturbation metrics. By default the HDA decision is computed live
with the same apply_hd_moe_routing implementation used by inference. A trace JSON
from an actual HD run can also be provided when it is aligned with the current
prompt/token order.
The script intentionally keeps the sample size small by default so the
experiment can be run per model and then aggregated for the TCAD revision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
FASTCHAT = ROOT / "fastchat"
if str(FASTCHAT) not in sys.path:
    sys.path.insert(0, str(FASTCHAT))

from fastchat.llm_judge import moe_gating_hd  # noqa: E402


MODEL_KEYS = ("ds", "mixtral", "qwen", "qwen35")

DEFAULT_PROMPTS = [
    "Explain why mixture-of-experts models can reduce inference cost.",
    "Solve the problem step by step: if a matrix has 8 rows and 12 columns, how many entries does it have?",
    "Write a short Python function that returns the greatest common divisor of two integers.",
    "Summarize the tradeoff between computation balance and communication overhead in distributed inference.",
]


class StopAfterCapture(RuntimeError):
    pass


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": torch.float16 if torch.cuda.is_available() else torch.float32,
    }[name]


def _load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    # DeepSeek-V2-Lite-Chat ships older custom modeling code that imports this
    # helper from transformers. Recent transformers removed it; restoring the
    # symbol is enough for inference-only loading.
    if args.model == "ds":
        try:
            import transformers.utils.import_utils as import_utils
            if not hasattr(import_utils, "is_torch_fx_available"):
                import_utils.is_torch_fx_available = lambda: False
        except Exception:
            pass
        try:
            from transformers.cache_utils import DynamicCache

            if not hasattr(DynamicCache, "from_legacy_cache"):
                DynamicCache.from_legacy_cache = classmethod(lambda cls, past_key_values=None: cls(past_key_values))
            if not hasattr(DynamicCache, "to_legacy_cache"):
                DynamicCache.to_legacy_cache = lambda self: tuple((layer[0], layer[1]) for layer in self)
            if not hasattr(DynamicCache, "get_usable_length"):
                DynamicCache.get_usable_length = lambda self, new_seq_length=None, layer_idx=0: self.get_seq_length(layer_idx)
            if not hasattr(DynamicCache, "seen_tokens"):
                DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
            if not hasattr(DynamicCache, "get_max_length"):
                DynamicCache.get_max_length = lambda self: self.get_max_cache_shape()
        except Exception:
            pass

    model_path = args.model_path
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.device_map:
        if args.device_map in ("cuda", "cuda:0", "0"):
            kwargs["device_map"] = {"": 0}
        elif args.device_map in ("cuda:1", "1"):
            kwargs["device_map"] = {"": 1}
        else:
            kwargs["device_map"] = args.device_map
    if args.dtype != "auto":
        kwargs["torch_dtype"] = _dtype(args.dtype)
    else:
        kwargs["torch_dtype"] = "auto"

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)
    if args.model == "ds":
        _force_load_safetensors_weights(model, Path(model_path))
    model.eval()
    return model, tokenizer, model_path


def _force_load_safetensors_weights(model: torch.nn.Module, model_path: Path) -> None:
    """Work around DeepSeek remote-code loading with newer transformers.

    In the current environment, DeepSeek-V2-Lite-Chat's older modeling code
    builds correctly, but several 2D weights can remain at initializer scale
    after from_pretrained. Copying tensors from the safetensors index restores
    the actual checkpoint values while preserving the already-created device map.
    """

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return
    from safetensors import safe_open

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not weight_map:
        return

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    by_file: Dict[str, List[str]] = {}
    for name, filename in weight_map.items():
        if name in params or name in buffers:
            by_file.setdefault(filename, []).append(name)

    with torch.no_grad():
        for filename, names in sorted(by_file.items()):
            with safe_open(model_path / filename, framework="pt", device="cpu") as f:
                for name in names:
                    target = params.get(name, buffers.get(name))
                    if target is None:
                        continue
                    tensor = f.get_tensor(name)
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    cur = model
    for attr in ("model", "language_model", "decoder", "transformer"):
        nxt = getattr(cur, attr, None)
        if nxt is not None and nxt is not cur:
            cur = nxt
    return cur


def _find_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = [
        getattr(_base_model(model), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise RuntimeError("could not find decoder layers on the loaded model")


def _find_moe(layer: torch.nn.Module) -> Optional[torch.nn.Module]:
    for name in ("block_sparse_moe", "mlp", "moe", "feed_forward"):
        mod = getattr(layer, name, None)
        if mod is not None and hasattr(mod, "experts"):
            return mod
    for _, mod in layer.named_modules():
        if mod is not layer and hasattr(mod, "experts") and (
            hasattr(mod, "gate") or hasattr(mod, "router")
        ):
            return mod
    return None


def _selected_layer_ids(layers: Sequence[torch.nn.Module], arg: Optional[str]) -> List[int]:
    ids = [i for i, layer in enumerate(layers) if _find_moe(layer) is not None]
    if not ids:
        raise RuntimeError("no MoE layers were found")
    if arg:
        if arg.strip().lower() == "all":
            return ids
        return sorted({int(x.strip()) for x in arg.split(",") if x.strip()})
    if len(ids) <= 6:
        return ids
    picks = {ids[0], ids[len(ids) // 4], ids[len(ids) // 2], ids[(3 * len(ids)) // 4], ids[-1]}
    return sorted(picks)


def _read_prompts(path: Optional[str], max_prompts: int) -> List[str]:
    if path is None:
        return DEFAULT_PROMPTS[:max_prompts]
    prompts: List[str] = []
    is_jsonl = str(path).endswith(".jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                item = json.loads(line)
                if isinstance(item, dict):
                    turns = item.get("turns")
                    if isinstance(turns, list) and turns:
                        prompts.append(str(turns[0]))
                    elif "prompt" in item:
                        prompts.append(str(item["prompt"]))
                    elif "question" in item:
                        prompts.append(str(item["question"]))
                    else:
                        prompts.append(line)
                else:
                    prompts.append(str(item))
            else:
                prompts.append(line)
            if len(prompts) >= max_prompts:
                break
    return prompts


def _load_routing_trace(path: Optional[Path]) -> Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    orig = data.get("original_selected_experts")
    sel = data.get("selected_experts")
    if not isinstance(orig, dict) or not isinstance(sel, dict):
        raise ValueError(f"routing trace must contain original_selected_experts and selected_experts: {path}")
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, orig_rows in orig.items():
        if key not in sel:
            continue
        layer_id = int(key)
        out[layer_id] = (
            torch.as_tensor(orig_rows, dtype=torch.long),
            torch.as_tensor(sel[key], dtype=torch.long),
        )
    if not out:
        raise ValueError(f"no common layer keys in routing trace: {path}")
    return out


def _load_float_sequence(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {int(k): float(v) for k, v in data.items()}
    if isinstance(data, list):
        return [float(v) for v in data]
    raise ValueError(f"expected JSON list or dict in {path}")


def _load_layer_reward_scale(path: Optional[Path]) -> Optional[Dict[int, float]]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "scale" in data:
        data = data["scale"]
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON dict or {{'scale': dict}} in {path}")
    return {int(k): float(v) for k, v in data.items()}


def _load_resident_experts(path: Optional[Path]) -> Optional[Dict[int, List[int]]]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON dict layer_id -> expert ids in {path}")
    out: Dict[int, List[int]] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            raise ValueError(f"resident experts for layer {key} must be a list")
        out[int(key)] = [int(v) for v in values]
    return out


def _derive_cache_prior_residents(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    max_tokens_per_layer: int,
    resident_top_n: int,
) -> Dict[int, List[int]]:
    layers = _find_layers(model)
    layer_ids = [i for i, layer in enumerate(layers) if _find_moe(layer) is not None]
    hidden = _capture_hidden_states(
        model,
        tokenizer,
        prompts,
        layer_ids,
        max_tokens_per_layer,
        max_length,
        stop_after_last_layer=True,
    )
    residents: Dict[int, List[int]] = {}
    for layer_id in layer_ids:
        moe = _find_moe(layers[layer_id])
        if moe is None:
            continue
        top_k = int(
            getattr(moe, "top_k", getattr(getattr(moe, "gate", None), "top_k", moe_gating_hd.HD_MOE_CONFIGS.get("mixtral", {}).get("e", 2)))
        )
        scores = _router_scores(moe, hidden[layer_id])
        idx, _ = _topk_original(scores, moe, min(top_k, scores.shape[-1]))
        counts = torch.bincount(idx.reshape(-1).cpu(), minlength=scores.shape[-1]).float()
        keep = max(1, min(int(resident_top_n), scores.shape[-1]))
        residents[layer_id] = torch.topk(counts, k=keep, sorted=True).indices.tolist()
    return residents


def _capture_hidden_states(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    layer_ids: Sequence[int],
    max_tokens_per_layer: int,
    max_length: int,
    stop_after_last_layer: bool,
) -> Dict[int, torch.Tensor]:
    layers = _find_layers(model)
    captures: Dict[int, List[torch.Tensor]] = defaultdict(list)
    handles = []
    last_layer = max(layer_ids)

    def make_hook(layer_id: int):
        def hook(_module, inputs):
            hidden = inputs[0].detach()
            hidden = hidden.reshape(-1, hidden.shape[-1]).cpu()
            remaining = max_tokens_per_layer - sum(x.shape[0] for x in captures[layer_id])
            if remaining > 0:
                captures[layer_id].append(hidden[:remaining])
            if stop_after_last_layer and layer_id == last_layer:
                raise StopAfterCapture()
        return hook

    for layer_id in layer_ids:
        moe = _find_moe(layers[layer_id])
        if moe is None:
            raise RuntimeError(f"layer {layer_id} is not an MoE layer")
        handles.append(moe.register_forward_pre_hook(make_hook(layer_id)))

    try:
        with torch.inference_mode():
            for prompt in prompts:
                toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                first_param = next(model.parameters())
                toks = {k: v.to(first_param.device) for k, v in toks.items()}
                try:
                    model(**toks, use_cache=False)
                except StopAfterCapture:
                    pass
                if all(sum(x.shape[0] for x in captures[i]) >= max_tokens_per_layer for i in layer_ids):
                    break
    finally:
        for h in handles:
            h.remove()

    out = {}
    for layer_id in layer_ids:
        if not captures[layer_id]:
            raise RuntimeError(f"no hidden states captured for layer {layer_id}")
        out[layer_id] = torch.cat(captures[layer_id], dim=0)[:max_tokens_per_layer]
    return out


def _router_scores(moe: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = _router_logits(moe, x)
    return logits.softmax(dim=-1, dtype=torch.float32)


def _router_logits(moe: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate = getattr(moe, "gate", None) or getattr(moe, "router", None)
    if gate is None:
        raise RuntimeError("MoE module has no gate/router")

    dev = next(gate.parameters(), next(moe.parameters())).device
    x_dev = x.to(dev)

    # DeepSeek-style gate stores expert weights directly and returns top-k rather
    # than logits. Reconstruct the pre-top-k softmax from its weight matrix.
    if hasattr(gate, "weight") and not isinstance(gate, torch.nn.Linear):
        weight = getattr(gate, "weight")
        logits = F.linear(x_dev.float(), weight.to(x_dev.device).float(), None)
        scoring = getattr(gate, "scoring_func", "softmax")
        if scoring != "softmax":
            raise RuntimeError(f"unsupported gate scoring_func={scoring!r}")
        return logits.float()

    raw = gate(x_dev)
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    return raw.float()


def _topk_original(scores: torch.Tensor, moe: torch.nn.Module, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gate = getattr(moe, "gate", None)
    if gate is not None and getattr(gate, "topk_method", None) == "group_limited_greedy":
        n_group = int(getattr(gate, "n_group"))
        topk_group = int(getattr(gate, "topk_group"))
        group_scores = scores.view(scores.shape[0], n_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(scores.shape[0], n_group, scores.shape[1] // n_group)
        masked = scores.masked_fill(~score_mask.reshape(scores.shape).bool(), 0.0)
        vals, idx = torch.topk(masked, k=top_k, dim=-1, sorted=True)
        return idx, vals
    vals, idx = torch.topk(scores, k=top_k, dim=-1, sorted=True)
    return idx, vals


def _ranked_experts_original(scores: torch.Tensor, moe: torch.nn.Module, k_total: int) -> torch.Tensor:
    gate = getattr(moe, "gate", None)
    k_total = min(k_total, scores.shape[-1])
    if gate is not None and getattr(gate, "topk_method", None) == "group_limited_greedy":
        n_group = int(getattr(gate, "n_group"))
        topk_group = int(getattr(gate, "topk_group"))
        group_scores = scores.view(scores.shape[0], n_group, -1).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(scores.shape[0], n_group, scores.shape[1] // n_group)
        masked = scores.masked_fill(~score_mask.reshape(scores.shape).bool(), float("-inf"))
        return torch.topk(masked, k=k_total, dim=-1, sorted=True).indices
    return torch.topk(scores, k=k_total, dim=-1, sorted=True).indices


def _topk_hda(
    scores: torch.Tensor,
    layer_id: int,
    model_name: str,
    top_k: int,
    reward_comp: float,
    reward_comm: float,
    mesh: Optional[Tuple[int, int]],
    hd_comp: Optional[float],
    hd_bw: Optional[float],
    include_future_original: bool,
    reward_scale_by_layer: Optional[Dict[int, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    moe_gating_hd.set_hd_moe_overrides(
        batch_size=int(scores.shape[0]),
        mesh_shape=mesh,
        comp=hd_comp,
        bw=hd_bw,
        reward_comp=reward_comp,
        reward_comm=reward_comm,
        use_original_gating=False,
        record_gating_softmax=False,
        routing_policy="hda",
        include_future_original_in_comp_map=include_future_original,
        reward_scale_by_layer=reward_scale_by_layer,
    )
    hd_layer = layer_id - 1 if model_name == "ds" else layer_id
    result = moe_gating_hd.apply_hd_moe_routing(
        scores,
        hd_layer,
        scores.device,
        model_name,
        top_k,
        reward_comp=reward_comp,
        reward_comm=reward_comm,
    )
    if result is None:
        vals, idx = torch.topk(scores, k=top_k, dim=-1, sorted=True)
        return idx, vals, False
    idx, vals = result
    order = torch.argsort(vals, dim=-1, descending=True)
    idx = torch.gather(idx, 1, order)
    vals = torch.gather(vals, 1, order)
    return idx, vals, True


def _topk_cache_prior(
    logits: torch.Tensor,
    scores: torch.Tensor,
    layer_id: int,
    top_k: int,
    cache_prior_strength: float,
    cache_prior_top_j: int,
    cache_prior_ranges: Optional[Any],
    cache_prior_resident_experts: Dict[int, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    moe_gating_hd.set_hd_moe_overrides(
        batch_size=int(scores.shape[0]),
        use_original_gating=False,
        record_gating_softmax=False,
        routing_policy="cache_prior",
        cache_prior_strength=cache_prior_strength,
        cache_prior_top_j=cache_prior_top_j,
        cache_prior_ranges=cache_prior_ranges,
        cache_prior_resident_experts=cache_prior_resident_experts,
    )
    result = moe_gating_hd.apply_cache_prior_routing(logits, scores, layer_id, top_k)
    if result is None:
        vals, idx = torch.topk(scores, k=top_k, dim=-1, sorted=True)
        return idx, vals, False
    idx, vals = result
    order = torch.argsort(vals, dim=-1, descending=True)
    idx = torch.gather(idx, 1, order)
    vals = torch.gather(vals, 1, order)
    return idx, vals, True


def _normalize_weights(weights: torch.Tensor, moe: torch.nn.Module) -> torch.Tensor:
    norm_topk = bool(getattr(moe, "norm_topk_prob", getattr(getattr(moe, "gate", None), "norm_topk_prob", True)))
    if weights.shape[-1] > 1 and norm_topk:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        scale = float(getattr(getattr(moe, "gate", None), "routed_scaling_factor", 1.0))
        weights = weights * scale
    return weights


def _expert_count(moe: torch.nn.Module) -> int:
    for attr in ("num_experts", "n_routed_experts"):
        val = getattr(moe, attr, None) or getattr(getattr(moe, "gate", None), attr, None)
        if val is not None:
            return int(val)
    return len(getattr(moe, "experts"))


def _expert_output(moe: torch.nn.Module, expert_id: int, x: torch.Tensor) -> torch.Tensor:
    experts = getattr(moe, "experts")
    dev = next(moe.parameters()).device
    x_dev = x.to(dev)
    try:
        expert = experts[int(expert_id)]
        return expert(x_dev)
    except Exception:
        ids = torch.full((x_dev.shape[0], 1), int(expert_id), dtype=torch.long, device=dev)
        weights = torch.ones((x_dev.shape[0], 1), dtype=x_dev.dtype, device=dev)
        return experts(x_dev, ids, weights)


def _weighted_moe_output(
    moe: torch.nn.Module,
    x: torch.Tensor,
    selected: torch.Tensor,
    weights: torch.Tensor,
    include_shared: bool,
) -> torch.Tensor:
    dev = next(moe.parameters()).device
    x_dev = x.to(dev)
    selected = selected.to(dev)
    weights = _normalize_weights(weights.to(dev).float(), moe).to(x_dev.dtype)
    out = torch.zeros_like(x_dev)

    for expert_id in torch.unique(selected).tolist():
        eid = int(expert_id)
        positions, rank = torch.where(selected == eid)
        if positions.numel() == 0:
            continue
        y = _expert_output(moe, eid, x_dev[positions])
        out[positions] += y.to(out.device, out.dtype) * weights[positions, rank].unsqueeze(-1)

    if include_shared and hasattr(moe, "shared_expert"):
        shared = moe.shared_expert(x_dev)
        gate = getattr(moe, "shared_expert_gate", None)
        if gate is not None:
            shared = torch.sigmoid(gate(x_dev)) * shared
        out = out + shared.to(out.device, out.dtype)
    elif include_shared and hasattr(moe, "shared_experts"):
        out = out + moe.shared_experts(x_dev).to(out.device, out.dtype)
    return out


def _pairwise_cosine(outputs: torch.Tensor) -> Optional[float]:
    if outputs.shape[0] < 2:
        return None
    normed = F.normalize(outputs.float(), p=2, dim=-1)
    sim = normed @ normed.T
    tri = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1, device=sim.device)
    return float(sim[tri[0], tri[1]].mean().item())


def _analyze_layer(
    moe: torch.nn.Module,
    x: torch.Tensor,
    layer_id: int,
    model_name: str,
    top_k: int,
    reward_comp: float,
    reward_comm: float,
    mesh: Optional[Tuple[int, int]],
    hd_comp: Optional[float],
    hd_bw: Optional[float],
    include_future_original: bool,
    include_shared: bool,
    max_similarity_pairs: int,
    low_impact_candidates: int,
    routing_trace: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    routing_policy: str = "hda",
    cache_prior_strength: float = 0.0,
    cache_prior_top_j: int = 1,
    cache_prior_ranges: Optional[Any] = None,
    cache_prior_resident_experts: Optional[Dict[int, List[int]]] = None,
    reward_scale_by_layer: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    logits = _router_logits(moe, x)
    scores = logits.softmax(dim=-1, dtype=torch.float32)
    orig_idx, orig_w = _topk_original(scores, moe, top_k)
    ranked_idx = _ranked_experts_original(
        scores,
        moe,
        top_k + max(1, low_impact_candidates),
    )
    trace_pair = routing_trace.get(layer_id) if routing_trace is not None else None
    # DeepSeek trace keys are 1-based, while the layer list is 0-based sparse layers.
    if trace_pair is None and routing_trace is not None and model_name == "ds":
        trace_pair = routing_trace.get(layer_id + 1)
    if trace_pair is not None:
        trace_orig, trace_hda = trace_pair
        n = min(x.shape[0], scores.shape[0], trace_orig.shape[0], trace_hda.shape[0])
        x = x[:n]
        scores = scores[:n]
        orig_idx = trace_orig[:n].to(scores.device)
        hda_idx = trace_hda[:n].to(scores.device)
        orig_w = scores.gather(1, orig_idx)
        hda_w = scores.gather(1, hda_idx)
        hd_applied = True
        trace_used = True
    elif routing_policy == "cache_prior":
        hda_idx, hda_w, hd_applied = _topk_cache_prior(
            logits,
            scores,
            layer_id,
            top_k,
            cache_prior_strength,
            cache_prior_top_j,
            cache_prior_ranges,
            cache_prior_resident_experts or {},
        )
        trace_used = False
    else:
        hda_idx, hda_w, hd_applied = _topk_hda(
            scores,
            layer_id,
            model_name,
            top_k,
            reward_comp,
            reward_comm,
            mesh,
            hd_comp,
            hd_bw,
            include_future_original,
            reward_scale_by_layer,
        )
        trace_used = False

    x_dev = x.to(next(moe.parameters()).device)
    with torch.inference_mode():
        y_orig = _weighted_moe_output(moe, x_dev, orig_idx, orig_w, include_shared)
        y_hda = _weighted_moe_output(moe, x_dev, hda_idx, hda_w, include_shared)

    diff = torch.linalg.vector_norm((y_orig - y_hda).float(), dim=-1)
    denom = torch.linalg.vector_norm(y_orig.float(), dim=-1).clamp_min(1e-12)
    perturb = 2.0 * diff / denom

    orig_ranked_topk = ranked_idx[:, :top_k]
    top1_preserved = (orig_ranked_topk[:, 0].to(hda_idx.device) == hda_idx[:, 0]).float()
    overlap = []
    changed_rows = []
    topk_rank_replaced = torch.zeros(top_k, dtype=torch.float64)
    expert_selected_count = torch.zeros(scores.shape[1], dtype=torch.float64)
    expert_replaced_count = torch.zeros(scores.shape[1], dtype=torch.float64)
    for i in range(orig_idx.shape[0]):
        o = set(int(v) for v in orig_idx[i].tolist())
        h = set(int(v) for v in hda_idx[i].tolist())
        overlap.append(len(o & h) / float(top_k))
        if o != h:
            changed_rows.append(i)
        for rank, expert_id in enumerate(orig_ranked_topk[i].tolist()):
            expert_id = int(expert_id)
            expert_selected_count[expert_id] += 1.0
            if expert_id not in h:
                topk_rank_replaced[rank] += 1.0
                expert_replaced_count[expert_id] += 1.0
    topk_rank_replaced = topk_rank_replaced / max(1, orig_idx.shape[0])
    topk_expert_replaced_ratio = torch.where(
        expert_selected_count > 0,
        expert_replaced_count / expert_selected_count.clamp_min(1.0),
        torch.full_like(expert_selected_count, float("nan")),
    )

    # Expert-output similarity is a property of the original MoE. Compare every
    # original top-k expert with each candidate just below the top-k boundary.
    # This directly measures whether experts HDA may swap in are functionally
    # close to the experts selected by the model-side gate.
    sim_values: List[float] = []
    sim_sums = torch.zeros((low_impact_candidates, top_k), dtype=torch.float64)
    diff_sums = torch.zeros((low_impact_candidates, top_k), dtype=torch.float64)
    score_gap_sums = torch.zeros((low_impact_candidates, top_k), dtype=torch.float64)
    replacement_sim_sums = torch.zeros((low_impact_candidates, top_k), dtype=torch.float64)
    replacement_diff_sums = torch.zeros((low_impact_candidates, top_k), dtype=torch.float64)
    replacement_kl_sums = torch.zeros((low_impact_candidates, top_k), dtype=torch.float64)
    sim_counts = torch.zeros((low_impact_candidates, top_k), dtype=torch.float64)
    random_k_sim_sum = 0.0
    random_k_sim_count = 0
    random_single_replacement_sim_sum = 0.0
    random_single_replacement_sim_count = 0
    for row in range(min(orig_idx.shape[0], max_similarity_pairs)):
        top_experts = [int(v) for v in ranked_idx[row, :top_k].tolist()]
        below_experts = [int(v) for v in ranked_idx[row, top_k : top_k + low_impact_candidates].tolist()]
        if not top_experts or not below_experts:
            continue
        top_outs = []
        below_outs = []
        for eid in top_experts:
            weight = scores[row, eid].to(x_dev.device, x_dev.dtype)
            top_outs.append((weight * _expert_output(moe, eid, x_dev[row : row + 1])).reshape(-1))
        for eid in below_experts:
            weight = scores[row, eid].to(x_dev.device, x_dev.dtype)
            below_outs.append((weight * _expert_output(moe, eid, x_dev[row : row + 1])).reshape(-1))
        top_contrib = torch.stack(top_outs, dim=0).float()
        below_contrib = torch.stack(below_outs, dim=0).float()
        orig_topk_sum = top_contrib.sum(dim=0)
        random_pool = [int(v) for v in ranked_idx[row, top_k:].tolist()]
        if len(random_pool) >= top_k:
            rng = random.Random(12345 + int(layer_id) * 1000003 + row)
            for _ in range(4):
                random_experts = rng.sample(random_pool, top_k)
                random_outs = []
                for eid in random_experts:
                    weight = scores[row, eid].to(x_dev.device, x_dev.dtype)
                    random_outs.append((weight * _expert_output(moe, eid, x_dev[row : row + 1])).reshape(-1))
                random_sum = torch.stack(random_outs, dim=0).float().sum(dim=0)
                random_sim = F.cosine_similarity(
                    orig_topk_sum.unsqueeze(0),
                    random_sum.unsqueeze(0),
                    dim=-1,
                )
                random_k_sim_sum += float(random_sim.item())
                random_k_sim_count += 1
            for top_rank in range(top_k):
                eid = rng.choice(random_pool)
                weight = scores[row, eid].to(x_dev.device, x_dev.dtype)
                random_contrib = (weight * _expert_output(moe, eid, x_dev[row : row + 1])).reshape(-1).float()
                replaced_sum = orig_topk_sum - top_contrib[top_rank] + random_contrib
                random_repl_sim = F.cosine_similarity(
                    orig_topk_sum.unsqueeze(0),
                    replaced_sum.unsqueeze(0),
                    dim=-1,
                )
                random_single_replacement_sim_sum += float(random_repl_sim.item())
                random_single_replacement_sim_count += 1
        top_mat = F.normalize(top_contrib, p=2, dim=-1)
        below_mat = F.normalize(below_contrib, p=2, dim=-1)
        pair_sim = top_mat @ below_mat.T
        for top_rank in range(pair_sim.shape[0]):
            for below_rank in range(pair_sim.shape[1]):
                val = float(pair_sim[top_rank, below_rank].item())
                if math.isfinite(val):
                    sim_values.append(val)
                    sim_sums[below_rank, top_rank] += val
                    diff = torch.linalg.vector_norm(top_contrib[top_rank] - below_contrib[below_rank])
                    denom = (
                        torch.linalg.vector_norm(top_contrib[top_rank])
                        + torch.linalg.vector_norm(below_contrib[below_rank])
                    ).clamp_min(1e-12)
                    diff_sums[below_rank, top_rank] += float((2.0 * diff / denom).item())
                    score_gap_sums[below_rank, top_rank] += float(
                        abs(scores[row, top_experts[top_rank]].item() - scores[row, below_experts[below_rank]].item())
                    )
                    replaced_sum = orig_topk_sum - top_contrib[top_rank] + below_contrib[below_rank]
                    repl_sim = F.cosine_similarity(
                        orig_topk_sum.unsqueeze(0),
                        replaced_sum.unsqueeze(0),
                        dim=-1,
                    )
                    replacement_sim_sums[below_rank, top_rank] += float(repl_sim.item())
                    repl_diff = torch.linalg.vector_norm(orig_topk_sum - replaced_sum)
                    repl_denom = (
                        torch.linalg.vector_norm(orig_topk_sum)
                        + torch.linalg.vector_norm(replaced_sum)
                    ).clamp_min(1e-12)
                    replacement_diff_sums[below_rank, top_rank] += float((2.0 * repl_diff / repl_denom).item())
                    orig_logp = F.log_softmax(orig_topk_sum.float(), dim=-1)
                    repl_logp = F.log_softmax(replaced_sum.float(), dim=-1)
                    orig_prob = orig_logp.exp()
                    replacement_kl_sums[below_rank, top_rank] += float(
                        (orig_prob * (orig_logp - repl_logp)).sum().item()
                    )
                    sim_counts[below_rank, top_rank] += 1.0
    sim_matrix = torch.where(sim_counts > 0, sim_sums / sim_counts.clamp_min(1.0), torch.full_like(sim_sums, float("nan")))
    diff_matrix = torch.where(
        sim_counts > 0,
        diff_sums / sim_counts.clamp_min(1.0),
        torch.full_like(diff_sums, float("nan")),
    )
    score_gap_matrix = torch.where(
        sim_counts > 0,
        score_gap_sums / sim_counts.clamp_min(1.0),
        torch.full_like(score_gap_sums, float("nan")),
    )
    replacement_sim_matrix = torch.where(
        sim_counts > 0,
        replacement_sim_sums / sim_counts.clamp_min(1.0),
        torch.full_like(replacement_sim_sums, float("nan")),
    )
    replacement_diff_matrix = torch.where(
        sim_counts > 0,
        replacement_diff_sums / sim_counts.clamp_min(1.0),
        torch.full_like(replacement_diff_sums, float("nan")),
    )
    replacement_kl_matrix = torch.where(
        sim_counts > 0,
        replacement_kl_sums / sim_counts.clamp_min(1.0),
        torch.full_like(replacement_kl_sums, float("nan")),
    )

    return {
        "layer": layer_id,
        "tokens": int(x.shape[0]),
        "hd_applied": bool(hd_applied),
        "routing_trace_used": bool(trace_used),
        "changed_ratio": float(len(changed_rows) / max(1, x.shape[0])),
        "top1_preserved": float(top1_preserved.mean().item()),
        "topk_overlap": float(sum(overlap) / max(1, len(overlap))),
        "topk_rank_replaced_ratio": topk_rank_replaced.tolist(),
        "topk_expert_replaced_ratio": topk_expert_replaced_ratio.tolist(),
        "topk_expert_selected_count": expert_selected_count.tolist(),
        "topk_expert_replaced_count": expert_replaced_count.tolist(),
        "expert_output_similarity": float(sum(sim_values) / len(sim_values)) if sim_values else None,
        "expert_output_similarity_matrix": sim_matrix.tolist(),
        "weighted_contribution_similarity_matrix": sim_matrix.tolist(),
        "weighted_contribution_normdiff_matrix": diff_matrix.tolist(),
        "single_replacement_moe_output_similarity_matrix": replacement_sim_matrix.tolist(),
        "single_replacement_moe_output_normdiff_matrix": replacement_diff_matrix.tolist(),
        "single_replacement_moe_output_kl_matrix": replacement_kl_matrix.tolist(),
        "random_k_moe_output_similarity": (
            float(random_k_sim_sum / random_k_sim_count) if random_k_sim_count > 0 else None
        ),
        "random_single_replacement_moe_output_similarity": (
            float(random_single_replacement_sim_sum / random_single_replacement_sim_count)
            if random_single_replacement_sim_count > 0
            else None
        ),
        "router_score_abs_gap_matrix": score_gap_matrix.tolist(),
        "expert_output_similarity_matrix_counts": sim_counts.tolist(),
        "similarity_samples": len(sim_values),
        "normalized_moe_output_diff_mean": float(perturb.mean().item()),
        "normalized_moe_output_diff_p95": float(torch.quantile(perturb.cpu(), 0.95).item()),
        "normalized_moe_output_diff_max": float(perturb.max().item()),
    }


def _masked_sequence_metrics(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels: torch.Tensor,
    score_mask: Optional[torch.Tensor] = None,
    prefix: str = "",
) -> Dict[str, float]:
    a = logits_a[:, :-1, :].float()
    b = logits_b[:, :-1, :].float()
    y = labels[:, 1:].to(a.device)
    mask = y.ne(-100)
    if score_mask is not None:
        mask = mask & score_mask.to(mask.device)
    logp_a = F.log_softmax(a, dim=-1)
    logp_b = F.log_softmax(b, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    kl = (p_a * (logp_a - logp_b)).sum(dim=-1)
    ce_a = F.cross_entropy(a.reshape(-1, a.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="none").reshape_as(y)
    ce_b = F.cross_entropy(b.reshape(-1, b.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="none").reshape_as(y)
    denom = mask.float().sum().clamp_min(1.0)
    nll_a = (ce_a * mask).sum() / denom
    nll_b = (ce_b * mask).sum() / denom
    target = y.clamp_min(0).unsqueeze(-1)
    p_obs_a = p_a.gather(-1, target).squeeze(-1)
    p_obs_b = p_b.gather(-1, target).squeeze(-1)
    delta_p2 = ((p_obs_b - p_obs_a) ** 2) * mask
    top1_same = (a.argmax(dim=-1) == b.argmax(dim=-1)).to(mask.dtype) * mask
    return {
        f"{prefix}scored_tokens": int(mask.sum().item()),
        f"{prefix}logit_kl_mean": float((kl.to(mask.device) * mask).sum().item() / denom.item()),
        f"{prefix}rms_delta_p_pct": float(100.0 * torch.sqrt(delta_p2.sum() / denom).item()),
        f"{prefix}top1_agreement_pct": float(100.0 * top1_same.sum().item() / denom.item()),
        f"{prefix}ppl_orig": float(torch.exp(nll_a).item()),
        f"{prefix}ppl_hda": float(torch.exp(nll_b).item()),
        f"{prefix}ppl_delta": float((torch.exp(nll_b) - torch.exp(nll_a)).item()),
        f"{prefix}log_ppl_ratio": float((nll_b - nll_a).item()),
        f"{prefix}ppl_ratio": float(torch.exp(nll_b - nll_a).item()),
    }


def _sequence_metrics(logits_a: torch.Tensor, logits_b: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    metrics = _masked_sequence_metrics(logits_a, logits_b, labels)

    # llama.cpp's perplexity/KL mode scores only the second half of each context
    # window, so every evaluated token has a non-trivial left context. Keep this
    # alongside the full-sequence metric instead of replacing it.
    seq_len = labels.shape[1]
    shifted_len = max(0, seq_len - 1)
    positions = torch.arange(shifted_len, device=labels.device).unsqueeze(0)
    score_mask = positions >= (seq_len // 2)
    metrics.update(
        _masked_sequence_metrics(
            logits_a,
            logits_b,
            labels,
            score_mask=score_mask,
            prefix="llamacpp_",
        )
    )
    return metrics


def _decode_style_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    score_start: int,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect suffix next-token logits with KV cache and small decode chunks."""
    seq_len = input_ids.shape[1]
    if seq_len < 2 or score_start >= seq_len - 1:
        raise ValueError(f"sequence too short for decode-style scoring: seq_len={seq_len}, score_start={score_start}")
    chunk_size = max(1, int(chunk_size))
    device = input_ids.device
    past = None
    if score_start > 0:
        prefix_pos = torch.arange(0, score_start, device=device).unsqueeze(0)
        prefix_mask = torch.ones((input_ids.shape[0], score_start), dtype=torch.long, device=device)
        out = model(
            input_ids=input_ids[:, :score_start],
            attention_mask=prefix_mask,
            position_ids=prefix_pos,
            use_cache=True,
        )
        past = out.past_key_values

    logits_parts: List[torch.Tensor] = []
    labels_parts: List[torch.Tensor] = []
    for start in range(score_start, seq_len - 1, chunk_size):
        end = min(seq_len - 1, start + chunk_size)
        pos = torch.arange(start, end, device=device).unsqueeze(0)
        mask = torch.ones((input_ids.shape[0], end), dtype=torch.long, device=device)
        out = model(
            input_ids=input_ids[:, start:end],
            attention_mask=mask,
            position_ids=pos,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits_parts.append(out.logits.detach())
        labels_parts.append(input_ids[:, start + 1 : end + 1])
    return torch.cat(logits_parts, dim=1), torch.cat(labels_parts, dim=1)


def _decode_style_sequence_metrics(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    a = logits_a.float()
    b = logits_b.float()
    y = labels.to(a.device)
    mask = y.ne(-100)
    logp_a = F.log_softmax(a, dim=-1)
    logp_b = F.log_softmax(b, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    kl = (p_a * (logp_a - logp_b)).sum(dim=-1)
    ce_a = F.cross_entropy(a.reshape(-1, a.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="none").reshape_as(y)
    ce_b = F.cross_entropy(b.reshape(-1, b.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="none").reshape_as(y)
    denom = mask.float().sum().clamp_min(1.0)
    nll_a = (ce_a * mask).sum() / denom
    nll_b = (ce_b * mask).sum() / denom
    target = y.clamp_min(0).unsqueeze(-1)
    p_obs_a = p_a.gather(-1, target).squeeze(-1)
    p_obs_b = p_b.gather(-1, target).squeeze(-1)
    delta_p2 = ((p_obs_b - p_obs_a) ** 2) * mask
    top1_same = (a.argmax(dim=-1) == b.argmax(dim=-1)).to(mask.dtype) * mask
    return {
        "decode_scored_tokens": int(mask.sum().item()),
        "decode_logit_kl_mean": float((kl.to(mask.device) * mask).sum().item() / denom.item()),
        "decode_rms_delta_p_pct": float(100.0 * torch.sqrt(delta_p2.sum() / denom).item()),
        "decode_top1_agreement_pct": float(100.0 * top1_same.sum().item() / denom.item()),
        "decode_ppl_orig": float(torch.exp(nll_a).item()),
        "decode_ppl_hda": float(torch.exp(nll_b).item()),
        "decode_ppl_delta": float((torch.exp(nll_b) - torch.exp(nll_a)).item()),
        "decode_log_ppl_ratio": float((nll_b - nll_a).item()),
        "decode_ppl_ratio": float(torch.exp(nll_b - nll_a).item()),
    }


def _aggregate_sequence_metric_dicts(metric_dicts: Sequence[Dict[str, float]], prefix: str) -> Dict[str, float]:
    if not metric_dicts:
        raise ValueError("no sequence metric dictionaries to aggregate")
    count_key = f"{prefix}scored_tokens"
    total = sum(int(m.get(count_key, 0)) for m in metric_dicts)
    if total <= 0:
        raise ValueError("no scored tokens in sequence metric dictionaries")

    def wavg(key: str) -> float:
        return float(sum(float(m[key]) * int(m[count_key]) for m in metric_dicts) / total)

    log_ppl_orig = float(
        sum(math.log(float(m[f"{prefix}ppl_orig"])) * int(m[count_key]) for m in metric_dicts) / total
    )
    log_ppl_hda = float(
        sum(math.log(float(m[f"{prefix}ppl_hda"])) * int(m[count_key]) for m in metric_dicts) / total
    )
    log_ratio = wavg(f"{prefix}log_ppl_ratio")
    rms = math.sqrt(
        sum(((float(m[f"{prefix}rms_delta_p_pct"]) / 100.0) ** 2) * int(m[count_key]) for m in metric_dicts)
        / total
    )
    return {
        count_key: int(total),
        f"{prefix}logit_kl_mean": wavg(f"{prefix}logit_kl_mean"),
        f"{prefix}rms_delta_p_pct": float(100.0 * rms),
        f"{prefix}top1_agreement_pct": wavg(f"{prefix}top1_agreement_pct"),
        f"{prefix}ppl_orig": float(math.exp(log_ppl_orig)),
        f"{prefix}ppl_hda": float(math.exp(log_ppl_hda)),
        f"{prefix}ppl_delta": float(math.exp(log_ppl_hda) - math.exp(log_ppl_orig)),
        f"{prefix}log_ppl_ratio": float(log_ratio),
        f"{prefix}ppl_ratio": float(math.exp(log_ratio)),
    }


def _optional_logit_metrics(
    model: torch.nn.Module,
    tokenizer,
    model_path: str,
    prompts: Sequence[str],
    args: argparse.Namespace,
) -> Optional[Dict[str, float]]:
    if not args.compute_logits:
        return None
    first_param = next(model.parameters())
    text = "\n\n".join(prompts)
    sliding_stride = int(args.logit_window_stride or 0)
    toks = tokenizer(
        text,
        return_tensors="pt",
        truncation=(sliding_stride <= 0),
        max_length=args.max_length if sliding_stride <= 0 else None,
    )
    input_ids_full = toks["input_ids"].to(first_param.device)
    attention_full = toks.get("attention_mask")
    if attention_full is not None:
        attention_full = attention_full.to(first_param.device)
    labels = input_ids_full.clone()
    hd_logit_batch_size = args.hd_logit_batch_size or int(toks["input_ids"].numel())
    windows: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    if sliding_stride > 0:
        seq_len = int(input_ids_full.shape[1])
        window_len = min(int(args.max_length), seq_len)
        if window_len < 2:
            raise ValueError(f"sequence too short for logit metrics: seq_len={seq_len}")
        starts = list(range(0, max(1, seq_len - window_len + 1), sliding_stride))
        if starts[-1] != seq_len - window_len:
            starts.append(seq_len - window_len)
        if args.logit_max_windows is not None:
            starts = starts[: int(args.logit_max_windows)]
        for start in starts:
            end = start + window_len
            windows.append((input_ids_full[:, start:end], attention_full[:, start:end] if attention_full is not None else None))
    else:
        windows.append((input_ids_full, attention_full))

    orig_decode_windows: List[torch.Tensor] = []
    decode_label_windows: List[torch.Tensor] = []
    orig_windows: List[torch.Tensor] = []
    label_windows: List[torch.Tensor] = []
    for input_ids_win, attention_win in windows:
        labels_win = input_ids_win.clone()
        score_start = labels_win.shape[1] // 2 if args.logit_score_suffix_only else 0
        if args.logit_eval_mode == "decode":
            with torch.inference_mode():
                orig_decode, decode_labels = _decode_style_logits(
                    model,
                    input_ids_win,
                    score_start=score_start,
                    chunk_size=hd_logit_batch_size,
                )
            orig_decode_windows.append(orig_decode)
            decode_label_windows.append(decode_labels)
        else:
            model_inputs = {"input_ids": input_ids_win}
            if attention_win is not None:
                model_inputs["attention_mask"] = attention_win
            with torch.inference_mode():
                orig = model(**model_inputs, use_cache=False).logits.detach()
            orig_windows.append(orig)
            label_windows.append(labels_win)

    mesh = (args.hd_mesh_rows, args.hd_mesh_cols) if args.hd_mesh_rows and args.hd_mesh_cols else None
    zero_reward_control = (
        args.routing_policy == "hda"
        and float(args.reward_comp) == 0.0
        and float(args.reward_comm) == 0.0
    )
    if args.routing_policy == "cache_prior":
        if not args.cache_prior_resident_experts:
            raise RuntimeError("cache_prior logit metrics require resident experts")
        moe_gating_hd.set_hd_moe_overrides(
            batch_size=hd_logit_batch_size,
            use_original_gating=False,
            record_gating_softmax=False,
            routing_policy="cache_prior",
            cache_prior_strength=args.cache_prior_strength,
            cache_prior_top_j=args.cache_prior_top_j,
            cache_prior_ranges=args.cache_prior_ranges_data,
            cache_prior_resident_experts=args.cache_prior_resident_experts,
        )
        moe_gating_hd.patch_moe_model_for_hd_gating(model, model_path, trace_only=False)
    elif not zero_reward_control:
        moe_gating_hd.set_hd_moe_overrides(
            batch_size=hd_logit_batch_size,
            mesh_shape=mesh,
            comp=args.hd_comp,
            bw=args.hd_bw,
            reward_comp=args.reward_comp,
            reward_comm=args.reward_comm,
            use_original_gating=False,
            record_gating_softmax=False,
            routing_policy="hda",
            include_future_original_in_comp_map=args.hd_comp_map_include_future_original,
            reward_scale_by_layer=args.layer_reward_scale,
        )
        moe_gating_hd.patch_moe_model_for_hd_gating(model, model_path, trace_only=False)
    if args.logit_eval_mode == "decode":
        metric_parts: List[Dict[str, float]] = []
        for (input_ids_win, _), orig_decode, decode_labels in zip(windows, orig_decode_windows, decode_label_windows):
            labels_win = input_ids_win.clone()
            score_start = labels_win.shape[1] // 2 if args.logit_score_suffix_only else 0
            with torch.inference_mode():
                hda_decode, hda_decode_labels = _decode_style_logits(
                    model,
                    input_ids_win,
                    score_start=score_start,
                    chunk_size=hd_logit_batch_size,
                )
            if not torch.equal(decode_labels, hda_decode_labels):
                raise RuntimeError("decode-style labels are not aligned between original and HDA runs")
            metric_parts.append(_decode_style_sequence_metrics(orig_decode, hda_decode, decode_labels))
        metrics = metric_parts[0] if len(metric_parts) == 1 else _aggregate_sequence_metric_dicts(metric_parts, "decode_")
    else:
        metric_parts = []
        for (input_ids_win, attention_win), orig, labels_win in zip(windows, orig_windows, label_windows):
            model_inputs = {"input_ids": input_ids_win}
            if attention_win is not None:
                model_inputs["attention_mask"] = attention_win
            with torch.inference_mode():
                hda = model(**model_inputs, use_cache=False).logits.detach()
            metric_parts.append(_sequence_metrics(orig, hda, labels_win))
        metrics = metric_parts[0] if len(metric_parts) == 1 else _aggregate_sequence_metric_dicts(metric_parts, "")
    metrics["hd_logit_batch_size"] = int(hd_logit_batch_size)
    metrics["logit_eval_mode"] = args.logit_eval_mode
    metrics["logit_score_start"] = int((windows[0][0].shape[1] // 2) if args.logit_score_suffix_only else 0)
    metrics["logit_windows"] = int(len(windows))
    metrics["logit_window_stride"] = int(sliding_stride)
    metrics["zero_reward_control"] = bool(zero_reward_control)
    metrics["routing_policy"] = args.routing_policy
    if args.routing_policy == "cache_prior":
        metrics["cache_prior_strength"] = float(args.cache_prior_strength)
        metrics["cache_prior_top_j"] = int(args.cache_prior_top_j)
        metrics["cache_prior_resident_layers"] = int(len(args.cache_prior_resident_experts or {}))
    return metrics


def _write_outputs(rows: List[Dict[str, Any]], summary: Dict[str, Any], out_json: Path, out_csv: Optional[Path]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "layers": rows}, indent=2), encoding="utf-8")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys()) if rows else []
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODEL_KEYS, required=True)
    p.add_argument(
        "--model-path",
        required=True,
        help="Local checkpoint directory or Hugging Face model identifier.",
    )
    p.add_argument("--prompts", default=None, help="Optional text file, one prompt per line, or JSONL question file with turns[0].")
    p.add_argument("--max-prompts", type=int, default=4)
    p.add_argument("--layers", default=None, help="Comma-separated layer ids. Default: five MoE layers spread through the model.")
    p.add_argument("--max-tokens-per-layer", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--reward-comp", type=float, default=-50000000.0)
    p.add_argument("--reward-comm", type=float, default=-0.25)
    p.add_argument(
        "--layer-reward-scale-json",
        type=Path,
        default=None,
        help="Optional layer_id -> scalar map. HDA rewards become reward_* * scale[layer_id].",
    )
    p.add_argument(
        "--routing-policy",
        choices=["hda", "cache_prior"],
        default="hda",
        help="Routing perturbation backend. cache_prior is the HF port of llama.cpp Cache-Prior.",
    )
    p.add_argument("--cache-prior-strength", type=float, default=0.5)
    p.add_argument("--cache-prior-top-j", type=int, default=1)
    p.add_argument("--cache-prior-ranges-json", type=Path, default=None)
    p.add_argument("--cache-prior-resident-json", type=Path, default=None)
    p.add_argument(
        "--cache-prior-resident-top-n",
        type=int,
        default=None,
        help="If no resident JSON is supplied, derive this many resident experts per layer from original routing frequency. Default: top-k.",
    )
    p.add_argument(
        "--cache-prior-resident-tokens-per-layer",
        type=int,
        default=None,
        help="Tokens per layer used when deriving cache-prior resident experts. Default: --max-tokens-per-layer.",
    )
    p.add_argument("--hd-mesh-rows", type=int, default=4)
    p.add_argument("--hd-mesh-cols", type=int, default=8)
    p.add_argument("--hd-comp", type=float, default=10.0e12, help="FLOP/s, or TFLOPS if a small value such as 5.0 is passed.")
    p.add_argument("--hd-bw", type=float, default=25.0e9, help="B/s, or GB/s if a small value such as 50.0 is passed.")
    p.add_argument("--hd-comp-map-include-future-original", action="store_true")
    p.add_argument("--include-shared", action="store_true", help="Include shared experts in the perturbation y.")
    p.add_argument("--max-similarity-pairs", type=int, default=64)
    p.add_argument(
        "--low-impact-candidates",
        type=int,
        default=None,
        help="Number of experts immediately after top-k to compare against top-k. Default: top-k.",
    )
    p.add_argument("--compute-logits", action="store_true", help="Run an additional HDA-patched forward for logit KL and PPL delta.")
    p.add_argument(
        "--hd-logit-batch-size",
        type=int,
        default=None,
        help=(
            "Override the HD gating batch size used during the optional logit forward. "
            "By default the full tokenized sequence length is used; set this to e.g. 32 "
            "to match the e2e decode-batch setting."
        ),
    )
    p.add_argument(
        "--logit-eval-mode",
        choices=["full", "decode"],
        default="full",
        help="full: one teacher-forced forward; decode: prefill prefix then score suffix in KV-cache chunks.",
    )
    p.add_argument(
        "--logit-score-suffix-only",
        action="store_true",
        help="Score only the second half of the context, matching llama.cpp perplexity/KL convention.",
    )
    p.add_argument(
        "--logit-window-stride",
        type=int,
        default=0,
        help=(
            "If positive, compute logit metrics over sliding windows of length --max-length "
            "with this stride, aggregating token-level statistics across windows."
        ),
    )
    p.add_argument(
        "--logit-max-windows",
        type=int,
        default=None,
        help="Optional cap on the number of sliding windows used for logit metrics.",
    )
    p.add_argument(
        "--routing-json",
        type=Path,
        default=None,
        help=(
            "Actual HD run trace JSON with original_selected_experts and selected_experts. "
            "Use only when the trace token order is aligned with the current prompts; "
            "otherwise omit it and compute HDA routing live on the captured states."
        ),
    )
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    p.add_argument("--no-stop-after-last-layer", action="store_true")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def _normalize_hardware_units(args: argparse.Namespace) -> None:
    if args.hd_comp is not None and args.hd_comp < 1.0e6:
        args.hd_comp *= 1.0e12
    if args.hd_bw is not None and args.hd_bw < 1.0e6:
        args.hd_bw *= 1.0e9


def main() -> int:
    args = parse_args()
    _normalize_hardware_units(args)
    model, tokenizer, model_path = _load_model_and_tokenizer(args)
    layers = _find_layers(model)
    layer_ids = _selected_layer_ids(layers, args.layers)
    prompts = _read_prompts(args.prompts, args.max_prompts)
    routing_trace = _load_routing_trace(args.routing_json)
    args.layer_reward_scale = _load_layer_reward_scale(args.layer_reward_scale_json)
    args.cache_prior_ranges_data = _load_float_sequence(args.cache_prior_ranges_json)
    args.cache_prior_resident_experts = _load_resident_experts(args.cache_prior_resident_json)
    if args.routing_policy == "cache_prior" and args.cache_prior_resident_experts is None:
        first_moe = next((_find_moe(layers[i]) for i in layer_ids if _find_moe(layers[i]) is not None), None)
        if first_moe is None:
            raise RuntimeError("no MoE layer available for cache-prior resident derivation")
        default_top_k = int(
            args.top_k
            or getattr(first_moe, "top_k", getattr(getattr(first_moe, "gate", None), "top_k", moe_gating_hd.HD_MOE_CONFIGS[args.model]["e"]))
        )
        resident_top_n = args.cache_prior_resident_top_n if args.cache_prior_resident_top_n and args.cache_prior_resident_top_n > 0 else default_top_k
        args.cache_prior_resident_experts = _derive_cache_prior_residents(
            model,
            tokenizer,
            prompts,
            args.max_length,
            args.cache_prior_resident_tokens_per_layer or args.max_tokens_per_layer,
            resident_top_n,
        )
    hidden = _capture_hidden_states(
        model,
        tokenizer,
        prompts,
        layer_ids,
        args.max_tokens_per_layer,
        args.max_length,
        stop_after_last_layer=not args.no_stop_after_last_layer,
    )
    mesh = (args.hd_mesh_rows, args.hd_mesh_cols) if args.hd_mesh_rows and args.hd_mesh_cols else None
    rows = []
    for layer_id in layer_ids:
        moe = _find_moe(layers[layer_id])
        if moe is None:
            continue
        top_k = args.top_k or int(
            getattr(moe, "top_k", getattr(getattr(moe, "gate", None), "top_k", moe_gating_hd.HD_MOE_CONFIGS[args.model]["e"]))
        )
        low_impact_candidates = args.low_impact_candidates or top_k
        rows.append(
            _analyze_layer(
                moe,
                hidden[layer_id],
                layer_id,
                args.model,
                top_k,
                args.reward_comp,
                args.reward_comm,
                mesh,
                args.hd_comp,
                args.hd_bw,
                args.hd_comp_map_include_future_original,
                args.include_shared,
                args.max_similarity_pairs,
                low_impact_candidates,
                routing_trace,
                args.routing_policy,
                args.cache_prior_strength,
                args.cache_prior_top_j,
                args.cache_prior_ranges_data,
                args.cache_prior_resident_experts,
                args.layer_reward_scale,
            )
        )

    numeric_keys = [
        "changed_ratio",
        "top1_preserved",
        "topk_overlap",
        "expert_output_similarity",
        "normalized_moe_output_diff_mean",
        "normalized_moe_output_diff_p95",
        "normalized_moe_output_diff_max",
    ]
    summary: Dict[str, Any] = {
        "model": args.model,
        "model_path": model_path,
        "layers": layer_ids,
        "top_k": args.top_k,
        "low_impact_candidates": args.low_impact_candidates,
        "reward_comp": args.reward_comp,
        "reward_comm": args.reward_comm,
        "layer_reward_scale_json": str(args.layer_reward_scale_json) if args.layer_reward_scale_json else None,
        "layer_reward_scale_layers": len(args.layer_reward_scale or {}),
        "routing_policy": args.routing_policy,
        "routing_json": str(args.routing_json) if args.routing_json else None,
        "tokens_per_layer": args.max_tokens_per_layer,
    }
    if args.routing_policy == "cache_prior":
        summary["cache_prior_strength"] = float(args.cache_prior_strength)
        summary["cache_prior_top_j"] = int(args.cache_prior_top_j)
        summary["cache_prior_resident_json"] = str(args.cache_prior_resident_json) if args.cache_prior_resident_json else None
        summary["cache_prior_ranges_json"] = str(args.cache_prior_ranges_json) if args.cache_prior_ranges_json else None
        summary["cache_prior_resident_top_n"] = args.cache_prior_resident_top_n
        summary["cache_prior_resident_layers"] = len(args.cache_prior_resident_experts or {})
        summary["cache_prior_resident_experts"] = args.cache_prior_resident_experts
    for key in numeric_keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        summary[key] = float(sum(vals) / len(vals)) if vals else None
    vector_keys = ["topk_rank_replaced_ratio"]
    for key in vector_keys:
        vecs = [torch.as_tensor(r[key], dtype=torch.float64) for r in rows if r.get(key) is not None]
        if vecs:
            summary[key] = torch.stack(vecs, dim=0).mean(dim=0).tolist()
    expert_selected = [
        torch.as_tensor(r["topk_expert_selected_count"], dtype=torch.float64)
        for r in rows
        if r.get("topk_expert_selected_count") is not None
    ]
    expert_replaced = [
        torch.as_tensor(r["topk_expert_replaced_count"], dtype=torch.float64)
        for r in rows
        if r.get("topk_expert_replaced_count") is not None
    ]
    if expert_selected and expert_replaced:
        selected_sum = torch.stack(expert_selected, dim=0).sum(dim=0)
        replaced_sum = torch.stack(expert_replaced, dim=0).sum(dim=0)
        expert_ratio = torch.where(
            selected_sum > 0,
            replaced_sum / selected_sum.clamp_min(1.0),
            torch.full_like(selected_sum, float("nan")),
        )
        summary["topk_expert_replaced_ratio"] = expert_ratio.tolist()
        summary["topk_expert_selected_count"] = selected_sum.tolist()
        summary["topk_expert_replaced_count"] = replaced_sum.tolist()
    matrix_keys = [
        "expert_output_similarity_matrix",
        "weighted_contribution_similarity_matrix",
        "weighted_contribution_normdiff_matrix",
        "single_replacement_moe_output_similarity_matrix",
        "single_replacement_moe_output_normdiff_matrix",
        "single_replacement_moe_output_kl_matrix",
        "router_score_abs_gap_matrix",
    ]
    for key in matrix_keys:
        mats = [torch.as_tensor(r[key], dtype=torch.float64) for r in rows if r.get(key) is not None]
        if mats:
            stacked = torch.stack(mats, dim=0)
            summary[key] = torch.nanmean(stacked, dim=0).tolist()
    count_mats = [
        torch.as_tensor(r["expert_output_similarity_matrix_counts"], dtype=torch.float64)
        for r in rows
        if r.get("expert_output_similarity_matrix_counts") is not None
    ]
    if count_mats:
        summary["expert_output_similarity_matrix_counts"] = torch.stack(count_mats, dim=0).sum(dim=0).tolist()
    logit_metrics = _optional_logit_metrics(model, tokenizer, model_path, prompts, args)
    if logit_metrics:
        summary.update(logit_metrics)

    _write_outputs(rows, summary, args.output_json, args.output_csv)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
