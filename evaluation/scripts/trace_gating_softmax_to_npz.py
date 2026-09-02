#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read JSON produced by gen_model_answer with --trace-gating-softmax (must contain
selected_gating_softmax) and write an npz with the same layout as
reconstruct_expert_scores_from_topk.py --output-npz, for use with
simulate_hd_gating_from_scores.py.

The input JSON must have a top-level selected_gating_softmax key whose structure
matches selected_experts (chunk nesting or flat token rows).

Example:
  python3 trace_gating_softmax_to_npz.py \\
    --trace /path/to/experts_reasoning_score.json \\
    --output-npz /path/to/gating_full_softmax.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from reconstruct_expert_scores_from_topk import layers_to_npz_arrays  # noqa: E402
except ModuleNotFoundError:
    def _flatten_rows(obj: Any) -> list[list[float]]:
        rows: list[list[float]] = []
        if isinstance(obj, list):
            if obj and all(isinstance(x, (int, float)) for x in obj):
                rows.append([float(x) for x in obj])
            else:
                for item in obj:
                    rows.extend(_flatten_rows(item))
        return rows

    def layers_to_npz_arrays(layers: Dict[str, Any]) -> Dict[str, np.ndarray]:
        arrs: Dict[str, np.ndarray] = {}
        for key, value in layers.items():
            rows = _flatten_rows(value)
            if not rows:
                continue
            width = len(rows[0])
            rows = [r for r in rows if len(r) == width]
            arrs[f"layer_{key}"] = np.asarray(rows, dtype=np.float32)
        return arrs


def _load_gating_softmax_tree(path: str) -> Dict[str, Any]:
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    gsm = raw.get("selected_gating_softmax")
    if not gsm:
        raise ValueError(
            f"{path} has no selected_gating_softmax; "
            "ensure trace was collected with --trace-gating-softmax (mt_bench)."
        )
    if not isinstance(gsm, dict):
        raise ValueError("selected_gating_softmax must be a dict keyed by layer id")
    return gsm


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Export (n_token, E) per-layer arrays from trace JSON with selected_gating_softmax"
    )
    p.add_argument(
        "--trace",
        type=str,
        required=True,
        help="Path to experts_*_score.json (must contain selected_gating_softmax)",
    )
    p.add_argument(
        "--output-npz",
        type=str,
        required=True,
        help="Output npz: each layer_<id> is float32 with shape (n_token, E)",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer ids to export only, e.g. 0,1,2; default: all layers",
    )
    p.add_argument(
        "--meta-json",
        type=str,
        default=None,
        help="Optional path to write meta as a standalone JSON file",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    trace_path = os.path.abspath(os.path.expanduser(args.trace))
    gsm = _load_gating_softmax_tree(trace_path)

    if args.layers:
        want = {x.strip() for x in args.layers.split(",") if x.strip()}
        gsm = {k: v for k, v in gsm.items() if str(k) in want}
        if not gsm:
            print("[error] --layers filter left no layers", file=sys.stderr)
            return 1

    arrs = layers_to_npz_arrays(gsm)
    meta = {
        "source": "trace_selected_gating_softmax",
        "trace_json": trace_path,
        "layer_keys": list(arrs.keys()),
        "shapes": {k: list(v.shape) for k, v in arrs.items()},
    }
    npz_path = os.path.abspath(os.path.expanduser(args.output_npz))
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    save_dict: Dict[str, Any] = {**arrs, "meta": json.dumps(meta, ensure_ascii=False)}
    np.savez_compressed(npz_path, **save_dict)
    print(
        f"[write] NPZ -> {npz_path}  layers={len(arrs)}  "
        f"example_shape={next(iter(arrs.values())).shape if arrs else ()}",
        file=sys.stderr,
    )

    if args.meta_json:
        mp = os.path.abspath(os.path.expanduser(args.meta_json))
        os.makedirs(os.path.dirname(mp) or ".", exist_ok=True)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[write] meta -> {mp}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
