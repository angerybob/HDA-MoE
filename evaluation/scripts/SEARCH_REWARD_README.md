# reward-comp / reward-comm 参数搜索

## 目标

- **准确率**：humaneval 上 pass@1 不低于 baseline 3 个百分点（`--acc-margin 0.03`）。
- **加速比**：mt-bench reasoning 上 `adaptive_pre_speedup_dynamic` 尽量大。

为三个模型各找一组最优 `--reward-comp`、`--reward-comm`（在 `-1e4`~`-1e5` 与 `-1e-2`~`-1e-1` 范围内）。

## 运行顺序

脚本默认按 **mixtral → qwen → ds** 顺序跑；mixtral/qwen 使用 2 卡（`CUDA_VISIBLE_DEVICES=2,3`）。

## 后台运行

```bash
cd /data/home/haochenhuang/TCAD/evaluation/scripts

# 三个模型都搜（mixtral, qwen, ds）
nohup bash run_search_reward.sh > search_reward.log 2>&1 &

# 只跑 mixtral 和 qwen（按 prompt 要求先做这两个）
nohup bash run_search_reward.sh mixtral qwen > search_reward.log 2>&1 &
```

查看进度：

```bash
tail -f search_reward.log
```

## 输出

- 结果 JSON：`evaluation/scripts/search_reward_results/search_reward_results.json`
  - 每个模型有 `baseline_acc`、`records`（每组参数的 acc/speedup）、`best`（满足准确率约束下加速比最优的一组参数）。
- 答案与 samples：humaneval 答案在 `/data1/datasets/humaneval/model_answer/{model_id}.jsonl`；脚本会在 `search_reward_results/` 下生成对应 `.samples.jsonl` 用于评估。

## 常用参数

| 参数 | 说明 |
|------|------|
| `--models mixtral qwen` | 只跑指定模型 |
| `--acc-margin 0.03` | 允许相对 baseline 的准确率下降（默认 0.03） |
| `--strategy default` | 参数网格：default / coarse / fine |
| `--skip-baseline` | 不重跑 baseline，用已有答案文件评估 baseline acc |
| `--skip-humaneval-if-exists` | 若某 model_id 的答案已存在则跳过该组 humaneval gen（省时间） |
| `--skip-mtbench-if-trace-exists` | 若 reasoning trace 已存在则只跑 adaptive（调试用） |
| `--baseline-only` | 只测各模型 baseline 准确率后退出 |
| `--results-dir DIR` | 指定结果与 samples 目录 |

## 策略说明

- **default**：有方向、少次数。reward-comp 取几档（mixtral/qwen 更保守），reward-comm 取 2 档。
- **coarse**：更粗的网格，先找大致方向。
- **fine**：在粗搜最佳附近细搜。

单次 humaneval 约 1 小时，请尽量用 `default` 或 `coarse` 减少总时间；需要时再用 `--skip-humaneval-if-exists` 断点续跑。
