#!/usr/bin/env bash
# 在 tmux 里跑 reward 参数搜索，日志同时写文件。只跑 mixtral 和 qwen，10 题 + 允许低于 baseline 20%。
# 用法: 先激活含 torch 的 conda/venv，再 bash run_search_tmux.sh
# 若需在 tmux 内自动激活 conda，可设置: export CONDA_ACTIVATE="source /path/to/conda.sh && conda activate your_env"

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TCAD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SESSION_NAME="${SESSION_NAME:-search_reward}"
LOG="${LOG:-$SCRIPT_DIR/search_reward.log}"

export PYTHONPATH="${TCAD_ROOT}/fastchat:${PYTHONPATH:-}"

cd "$SCRIPT_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
  echo "Or kill it first: tmux kill-session -t $SESSION_NAME"
  exit 1
fi

# 在 tmux 里执行的命令：可选先激活 conda，再跑搜索（tmux 新 session 默认无 conda，需激活才有 torch）
RUN_CMD="cd '$SCRIPT_DIR' && python3 search_reward_params.py --models mixtral qwen 2>&1 | tee -a '$LOG'"
if [ -n "${CONDA_ACTIVATE}" ]; then
  RUN_CMD="${CONDA_ACTIVATE} && $RUN_CMD"
elif [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
  # 若 base 无 torch，请先: export CONDA_ACTIVATE="source /opt/anaconda3/etc/profile.d/conda.sh && conda activate 你的环境名"
  RUN_CMD="source /opt/anaconda3/etc/profile.d/conda.sh && conda activate base && $RUN_CMD"
fi

tmux new-session -d -s "$SESSION_NAME" "$RUN_CMD"
echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Tail log: tail -f $LOG"
echo ""
echo "若日志出现 ModuleNotFoundError: No module named 'torch'，请先激活含 torch 的 conda 环境再启动，例如："
echo "  export CONDA_ACTIVATE=\"source /opt/anaconda3/etc/profile.d/conda.sh && conda activate 你的环境名\""
echo "  bash run_search_tmux.sh"
