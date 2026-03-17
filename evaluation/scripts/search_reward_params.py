#!/usr/bin/env python3
"""
搜索 gen_model_answer.py 的 --reward-comp 和 --reward-comm 最优参数。
目标：humaneval 准确率不低于 baseline 3 个百分点，mt-bench 上 adaptive_pre_speedup_dynamic 最大。
顺序：先 mixtral、qwen，最后 ds。
"""
import os
import re
import json
import subprocess
import argparse
from pathlib import Path

# 路径常量
TCAD_ROOT = Path(__file__).resolve().parents[2]
FASTCHAT_LLM_JUDGE = TCAD_ROOT / "fastchat" / "fastchat" / "llm_judge"
GEN_SCRIPT = FASTCHAT_LLM_JUDGE / "gen_model_answer.py"
ADAPTIVE_SCRIPT = TCAD_ROOT / "evaluation" / "scripts" / "adaptive.py"
HUMANEVAL_ROOT = Path("/data/home/haochenhuang/human-eval")
DATA_DIR = "/data1/datasets"
HUMANEVAL_ANSWER_DIR = Path("/data1/datasets/humaneval/model_answer")
EXPERT_TRACE_DIR = TCAD_ROOT / "expert_trace"

# 模型配置: key, model_path, need_2gpu, model_id_prefix, humaneval_max_questions, acc_margin
# humaneval_max_questions=10 时只生成/评估前 10 题；None 为全量。acc_margin 为允许相对 baseline 的下降（如 0.20 表示可低 20 个百分点）。
MODELS = [
    {
        "key": "mixtral",
        "model_path": "/data1/pretrained_models/Mixtral-8x7B-v0.1",
        "need_2gpu": True,
        "model_id_prefix": "humaneval-mixtral-8x7b-v0.1",
        "humaneval_max_questions": 10,
        "acc_margin": 0.20,
    },
    {
        "key": "qwen",
        "model_path": "/data1/pretrained_models/Qwen2-57B-A14B-Instruct",
        "need_2gpu": True,
        "model_id_prefix": "humaneval-qwen2-57b-a14b-instruct",
        "humaneval_max_questions": 10,
        "acc_margin": 0.20,
    },
    {
        "key": "ds",
        "model_path": "/data1/pretrained_models/DeepSeek-V2-Lite-Chat",
        "need_2gpu": False,
        "model_id_prefix": "humaneval-deepseek-v2-lite-chat",
        "humaneval_max_questions": None,
        "acc_margin": 0.03,
    },
]

# 公共 gen 参数（与 prompt 一致）
GEN_COMMON = [
    "--batch", "32",
    "--hd-mesh-rows", "4",
    "--hd-mesh-cols", "8",
    "--hd-comp", "10000000000000",
    "--hd-bw", "25000000000",
]


def run_gen_humaneval(
    model_key: str,
    model_path: str,
    model_id: str,
    reward_comp: float,
    reward_comm: float,
    need_2gpu: bool,
    max_questions: int | None = None,
) -> bool:
    """在 humaneval 上跑 gen_model_answer。max_questions=10 时只跑前 10 题（用于 mixtral/qwen 加速）。"""
    trace_path = EXPERT_TRACE_DIR / model_key / "adaptive" / "humaneval_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", str(GEN_SCRIPT),
        "--model-path", model_path,
        "--model-id", model_id,
        *GEN_COMMON,
        "--reward-comp", str(reward_comp),
        "--reward-comm", str(reward_comm),
        "--data-dir", DATA_DIR,
        "--datasets", "humaneval",
        "--trace", str(trace_path),
    ]
    if max_questions is not None:
        cmd.extend(["--question-begin", "0", "--question-end", str(max_questions)])
    else:
        cmd.append("--all-questions")
    if need_2gpu:
        cmd.extend(["--num-gpus-per-model", "2", "--num-gpus-total", "2"])
    env = os.environ.copy()
    if need_2gpu:
        env["CUDA_VISIBLE_DEVICES"] = "2,3"
    env["PYTHONPATH"] = str(TCAD_ROOT / "fastchat")
    cwd = str(TCAD_ROOT)
    print(f"[run_gen_humaneval] model={model_key} model_id={model_id} reward_comp={reward_comp} reward_comm={reward_comm}")
    ret = subprocess.run(cmd, cwd=cwd, env=env)
    return ret.returncode == 0


def run_gen_mtbench_trace(model_key: str, model_path: str, reward_comp: float, reward_comm: float, need_2gpu: bool) -> bool:
    """跑 mt-bench reasoning (q21–22)，写 trace 到 expert_trace/{model}/adaptive/experts_reasoning_{model}_adaptive.json"""
    trace_path = EXPERT_TRACE_DIR / model_key / "adaptive" / f"experts_reasoning_{model_key}_adaptive.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", str(GEN_SCRIPT),
        "--model-path", model_path,
        "--model-id", f"mtbench-trace-{model_key}",
        *GEN_COMMON,
        "--reward-comp", str(reward_comp),
        "--reward-comm", str(reward_comm),
        "--question-begin", "21",
        "--question-end", "23",
        "--trace", str(trace_path),
    ]
    if need_2gpu:
        cmd.extend(["--num-gpus-per-model", "2", "--num-gpus-total", "2"])
    env = os.environ.copy()
    if need_2gpu:
        env["CUDA_VISIBLE_DEVICES"] = "2,3"
    env["PYTHONPATH"] = str(TCAD_ROOT / "fastchat")
    cwd = str(TCAD_ROOT)
    print(f"[run_gen_mtbench_trace] model={model_key} reward_comp={reward_comp} reward_comm={reward_comm}")
    ret = subprocess.run(cmd, cwd=cwd, env=env)
    return ret.returncode == 0


def convert_and_eval_humaneval(
    model_id: str,
    samples_path: Path,
    problem_file: str | None = None,
    use_subset10: bool = False,
) -> float | None:
    """转格式并评估，返回 pass@1（0~1）。use_subset10 时先 make_subset_problems 再按 data/HumanEval_subset10.jsonl 评估。"""
    answer_file = HUMANEVAL_ANSWER_DIR / f"{model_id}.jsonl"
    if not answer_file.exists():
        print(f"[convert_and_eval] 答案文件不存在: {answer_file}")
        return None
    # convert
    convert_cmd = [
        "python", str(HUMANEVAL_ROOT / "convert_huamaneval_to_samples.py"),
        str(answer_file),
        "-o", str(samples_path),
        "--normalize",
    ]
    ret = subprocess.run(convert_cmd, cwd=str(HUMANEVAL_ROOT))
    if ret.returncode != 0:
        return None
    if use_subset10:
        subset_problems = HUMANEVAL_ROOT / "data" / "HumanEval_subset10.jsonl"
        subset_cmd = [
            "python", str(HUMANEVAL_ROOT / "make_subset_problems.py"),
            str(samples_path), str(subset_problems),
        ]
        ret = subprocess.run(subset_cmd, cwd=str(HUMANEVAL_ROOT))
        if ret.returncode != 0:
            return None
        problem_file = "data/HumanEval_subset10.jsonl"
    # evaluate
    eval_cmd = ["python", "-m", "human_eval.evaluate_functional_correctness", str(samples_path)]
    if problem_file:
        eval_cmd.extend(["--problem_file", problem_file])
    out = subprocess.run(eval_cmd, cwd=str(HUMANEVAL_ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        print(f"[convert_and_eval] evaluate 失败: {out.stderr}")
        return None
    # 解析 pass@1
    m = re.search(r"'pass@1':\s*[\w.]*\(?([\d.]+)\)?", out.stdout)
    if m:
        return float(m.group(1))
    return None


def run_adaptive(model_key: str) -> float | None:
    """跑 adaptive.py，返回 adaptive_pre_speedup_dynamic"""
    cmd = ["python", str(ADAPTIVE_SCRIPT), "--model", model_key]
    out = subprocess.run(cmd, cwd=str(TCAD_ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        print(f"[run_adaptive] 失败: {out.stderr}")
        return None
    m = re.search(r"adaptive_pre_speedup_dynamic:([\d.]+)", out.stdout)
    if m:
        return float(m.group(1))
    return None


def search_one_model(
    model: dict,
    baseline_acc: float,
    param_grid: list[tuple[float, float]],
    results_dir: Path,
    skip_humaneval_if_exists: bool,
    skip_mtbench_if_trace_exists: bool,
) -> dict:
    """对单个模型做参数搜索，返回 best 与所有记录."""
    model_key = model["key"]
    model_path = model["model_path"]
    need_2gpu = model["need_2gpu"]
    prefix = model["model_id_prefix"]
    acc_margin = model.get("acc_margin", 0.03)
    records = []
    best = None

    for reward_comp, reward_comm in param_grid:
        # 文件名友好：负号和小数点用字母或下划线替代
        rc_str = str(reward_comp).replace(".", "_").replace("-", "n")
        rcomm_str = str(reward_comm).replace(".", "_").replace("-", "n")
        model_id = f"{prefix}-reward-comp-{rc_str}-reward-comm-{rcomm_str}"

        # 1) Humaneval gen
        answer_file = HUMANEVAL_ANSWER_DIR / f"{model_id}.jsonl"
        max_questions = model.get("humaneval_max_questions")
        if skip_humaneval_if_exists and answer_file.exists():
            print(f"[search] 跳过已有答案: {model_id}")
        else:
            ok = run_gen_humaneval(model_key, model_path, model_id, reward_comp, reward_comm, need_2gpu, max_questions=max_questions)
            if not ok:
                print(f"[search] humaneval gen 失败: {model_id}")
                continue

        # 2) Eval acc
        samples_path = results_dir / f"{model_id}.samples.jsonl"
        use_subset10 = (max_questions == 10)
        acc = convert_and_eval_humaneval(model_id, samples_path, use_subset10=use_subset10)
        if acc is None:
            continue
        if acc < baseline_acc - acc_margin:
            print(f"[search] acc {acc:.4f} < baseline-{acc_margin} = {baseline_acc - acc_margin:.4f}，跳过 speedup")
            records.append({"reward_comp": reward_comp, "reward_comm": reward_comm, "acc": acc, "speedup": None, "skip_reason": "acc_low"})
            continue

        # 3) Mt-bench trace（用同一组参数写 trace）
        trace_path = EXPERT_TRACE_DIR / model_key / "adaptive" / f"experts_reasoning_{model_key}_adaptive.json"
        if skip_mtbench_if_trace_exists and trace_path.exists():
            print(f"[search] 使用已有 trace 跑 adaptive: {trace_path}")
        else:
            ok = run_gen_mtbench_trace(model_key, model_path, reward_comp, reward_comm, need_2gpu)
            if not ok:
                print(f"[search] mt-bench trace 失败: reward_comp={reward_comp} reward_comm={reward_comm}")
                records.append({"reward_comp": reward_comp, "reward_comm": reward_comm, "acc": acc, "speedup": None})
                continue

        # 4) Adaptive speedup
        speedup = run_adaptive(model_key)
        records.append({"reward_comp": reward_comp, "reward_comm": reward_comm, "acc": acc, "speedup": speedup})
        if speedup is not None and (best is None or speedup > best["speedup"]):
            best = {"reward_comp": reward_comp, "reward_comm": reward_comm, "acc": acc, "speedup": speedup}
            print(f"[search] 当前最佳: reward_comp={reward_comp} reward_comm={reward_comm} acc={acc:.4f} speedup={speedup:.2f}")

    return {"baseline_acc": baseline_acc, "records": records, "best": best}


def get_baseline_acc(model: dict, results_dir: Path, samples_name: str = "baseline.samples.jsonl") -> float | None:
    """测 baseline（reward_comp=0, reward_comm=0）的 pass@1"""
    model_key = model["key"]
    model_path = model["model_path"]
    need_2gpu = model["need_2gpu"]
    prefix = model["model_id_prefix"]
    max_questions = model.get("humaneval_max_questions")
    model_id = f"{prefix}-baseline"
    ok = run_gen_humaneval(model_key, model_path, model_id, 0.0, 0.0, need_2gpu, max_questions=max_questions)
    if not ok:
        return None
    samples_path = results_dir / samples_name
    return convert_and_eval_humaneval(model_id, samples_path, use_subset10=(max_questions == 10))


def build_param_grid(strategy: str, model_key: str) -> list[tuple[float, float]]:
    """策略化参数网格。reward_comp 在 -1e4~-1e5，reward_comm 在 -1e-2~-1e-1。"""
    if strategy == "coarse":
        # 粗搜：少点，先找方向
        reward_comp_vals = [-100000, -50000, -30000, -10000]
        reward_comm_vals = [-0.1, -0.05, -0.01]
        grid = [(rc, rcomm) for rc in reward_comp_vals for rcomm in reward_comm_vals]
    elif strategy == "fine":
        # 在粗搜最佳附近细搜
        reward_comp_vals = [-40000, -30000, -20000]
        reward_comm_vals = [-0.08, -0.05, -0.03, -0.01]
        grid = [(rc, rcomm) for rc in reward_comp_vals for rcomm in reward_comm_vals]
    else:
        # default: 有方向、少次数。reward-comp 单调性强，取几档；reward-comm 取 2 档
        # mixtral/qwen 可更保守（更接近 0）
        if model_key in ("mixtral", "qwen"):
            reward_comp_vals = [-50000, -30000, -20000, -10000]
            reward_comm_vals = [-0.05, -0.01]
        else:
            reward_comp_vals = [-100000, -50000, -30000, -10000]
            reward_comm_vals = [-0.05, -0.01]
        grid = [(rc, rcomm) for rc in reward_comp_vals for rcomm in reward_comm_vals]
    return grid


def main():
    parser = argparse.ArgumentParser(description="搜索 reward-comp / reward-comm 最优参数")
    parser.add_argument("--models", nargs="*", default=["mixtral", "qwen", "ds"], help="要跑的模型 key")
    parser.add_argument("--acc-margin", type=float, default=0.03, help="允许相对 baseline 的准确率下降")
    parser.add_argument("--strategy", choices=["default", "coarse", "fine"], default="default")
    parser.add_argument("--skip-baseline", action="store_true", help="不重跑 baseline，用已有 baseline 答案评估")
    parser.add_argument("--skip-humaneval-if-exists", action="store_true", help="若答案文件已存在则跳过 humaneval gen")
    parser.add_argument("--skip-mtbench-if-trace-exists", action="store_true", help="若 trace 已存在则只跑 adaptive")
    parser.add_argument("--results-dir", type=Path, default=None, help="存放 samples 与 result json 的目录")
    parser.add_argument("--baseline-only", action="store_true", help="只测各模型 baseline acc 后退出")
    args = parser.parse_args()

    results_dir = args.results_dir or (Path(__file__).resolve().parent / "search_reward_results")
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for m in MODELS:
        if m["key"] not in args.models:
            continue
        model_key = m["key"]
        print(f"\n========== 模型: {model_key} ==========")

        # Baseline
        baseline_acc = None
        if not args.skip_baseline:
            baseline_acc = get_baseline_acc(m, results_dir, samples_name=f"{model_key}_baseline.samples.jsonl")
            if baseline_acc is None:
                print(f"[{model_key}] baseline 评估失败，跳过该模型")
                continue
            print(f"[{model_key}] baseline pass@1 = {baseline_acc:.4f}")
        else:
            # 从已有 baseline 答案评估
            model_id = f"{m['model_id_prefix']}-baseline"
            samples_path = results_dir / f"{model_key}_baseline.samples.jsonl"
            use_subset10 = (m.get("humaneval_max_questions") == 10)
            baseline_acc = convert_and_eval_humaneval(model_id, samples_path, use_subset10=use_subset10)
            if baseline_acc is None:
                print(f"[{model_key}] 无法从已有答案得到 baseline，跳过")
                continue
            print(f"[{model_key}] baseline pass@1 (from existing) = {baseline_acc:.4f}")

        if args.baseline_only:
            all_results[model_key] = {"baseline_acc": baseline_acc}
            continue

        param_grid = build_param_grid(args.strategy, model_key)
        print(f"[{model_key}] 参数网格点数: {len(param_grid)}")
        res = search_one_model(
            m,
            baseline_acc=baseline_acc,
            param_grid=param_grid,
            results_dir=results_dir,
            skip_humaneval_if_exists=args.skip_humaneval_if_exists,
            skip_mtbench_if_trace_exists=args.skip_mtbench_if_trace_exists,
        )
        all_results[model_key] = res

    out_path = results_dir / "search_reward_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
