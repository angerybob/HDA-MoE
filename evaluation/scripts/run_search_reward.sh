#!/usr/bin/env bash
# 后台运行 reward 参数搜索。用法：
#   cd /data/home/haochenhuang/TCAD/evaluation/scripts
#   nohup bash run_search_reward.sh > search_reward.log 2>&1 &
# 或仅跑 mixtral + qwen：nohup bash run_search_reward.sh mixtral qwen > search_reward.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 可选：只跑指定模型，默认 mixtral qwen ds
MODELS="${*:-qwen mixtral ds}"
export PYTHONPATH="${TCAD_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}/fastchat:${PYTHONPATH:-}"

echo "========== 开始时间: $(date) =========="
echo "模型顺序: $MODELS"

python3 search_reward_params.py --models $MODELS

echo "========== 结束时间: $(date) =========="
