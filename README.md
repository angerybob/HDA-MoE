## Quick Start

### 1. 环境搭建（Environment Setup）

#### 创建并激活conda环境
```bash
# 创建环境
conda create -n tcad python=3.10
# 激活环境
conda activate tcad
```

#### 克隆仓库并安装依赖
```bash
# 克隆仓库
git clone --recursive git@github.com:angerybob/TCAD.git
# 进入仓库目录
cd TCAD
# 安装依赖
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
cd fastchat
pip install -e ".[model_worker,llm_judge]"
```

### 2. 获得专家激活数据与adaptive gating

```bash
cd fastchat/fastchat/llm_judge
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 gen_model_answer.py --model-path /opt/pretrained_models/DeepSeek-V2-Lite-Chat --model-id 66666 --num-gpus-per-model 8 --num-gpus-total 8
# or
CUDA_VISIBLE_DEVICES=2 python3 gen_model_answer.py --model-path /opt/pretrained_models/DeepSeek-V2-Lite-Chat --model-id 66666 --num-gpus-per-model 1 --num-gpus-total 1
```
`--num-gpus-per-model`：一个模型放在几张卡上（张量并行维度）

`--num-gpus-total`：一共几张卡（对应`CUDA_VISIBLE_DEVICES`一共几个编号）

`--num-gpus-total/--num-gpus-per-model`：数据并行维度

`fastchat/fastchat/llm_judge/modeling_deepseek.py`中是具体的推理流程

跑出来的trace放在`expert_trace/ds/adaptive`目录下，具体命名可以在`fastchat/fastchat/llm_judge/gen_model_answer.py`修改相关代码

修改不同batch需要在两个地方同时修改

`fastchat/fastchat/llm_judge/gen_model_answer.py`中：
```python
prompt = conv.get_prompt()
batch = 32
inputs = tokenizer([prompt]*batch,return_tensors="pt",padding=True)
```

`fastchat/fastchat/llm_judge/modeling_deepseek.py`中：
```python
if scores.shape[0]==32:
```

adaptive gating 的两个超参数：
`fastchat/fastchat/llm_judge/modeling_deepseek.py`
```python
reward_comp = -1e3
reward_comm = -5e-5
```

adaptive gating 的评估可以在`evaluation/scripts/adaptive.py`的基础上修改

### 3. 生成部署策略（Generate Deployment Strategy）

通过优化脚本生成针对特定硬件和模型的部署策略，后台运行并输出日志：
```bash
nohup optimizer.sh > script.log 2>&1 &
```

- **输出位置**：
  - 部署策略结果：`results/` 文件夹
  - 每层输出日志：`logs/` 文件夹

- **参数配置**：
  可在 `optimizer.sh` 中修改以下配置以适配不同场景：
  ```bash
  # 硬件配置
  comp=10.0          # 算力（TFLOPS）
  BW=25.0            # 带宽（GB/s）
  mesh_shapeX=4      # 2D mesh X维度尺寸
  mesh_shapeY=8      # 2D mesh Y维度尺寸
  # 任务配置
  batch=128          # 批次大小
  model="qwen"       # 模型类型（如"qwen"、"mixtral"等）
  ```
  脚本中for循环的层数也要根据模型具体配置修改 

### 4. 评估部署策略（Evaluate Deployment Strategy）

使用评估脚本验证部署策略的性能，支持端到端 latency、消融实验和动态调度评估：

#### 评估命令
```bash
# 评估端到端TBT
python evaluation/scripts/e2e.py

# 消融实验（验证各模块作用）
python evaluation/scripts/ablation.py

# 动态调度策略评估
python evaluation/scripts/dynamic.py
```

- **评估结果位置**：`evaluation/results/` 文件夹
- **评估前配置**：需在对应脚本中修改硬件配置（算力、带宽等）、模型类型及数据集，确保与生成的部署策略匹配。


### 5. 结果可视化（Visualization）

通过绘图脚本将评估结果可视化，生成与论文对应的关键图表：

```bash
# 绘制不同硬件配置下的端到端加速比（对应Fig. 8）
python evaluation/draw/draw.py

# 绘制不同mesh尺寸下的性能（对应Fig. 9）
python evaluation/draw/draw_mesh.py

# 绘制节点平衡优化的加速比（对应Fig. 10）
python evaluation/draw/ablation_draw.py

# 绘制节点平衡对计算延迟的优化（对应Fig. 11）
python evaluation/draw/ablation2_draw.py

# 绘制节点级资源利用平衡（对应Fig. 12）
python evaluation/draw/balance.py

# 绘制链路平衡优化的加速比（对应Fig. 13）
python evaluation/draw/ablation3_draw.py

# 绘制链路级资源利用平衡（对应Fig. 14）
python evaluation/draw/balance2.py

# 绘制动态调度策略性能（对应Fig. 15 (a)）
python evaluation/draw/dynamic_draw.py

# 绘制不同预广播专家数量下的动态调度性能（对应Fig. 15 (b)）
python evaluation/draw/dynamic_draw2.py
```

- **图表输出位置**：`evaluation/figs/` 文件夹


## 核心模块说明（Core Modules）

- **`node_allocation.py`**：实现 `MoE3DPNMOptimizer` 类，封装了文章中提出的Node-Link Balance优化算法。

- **`simulator.py`**：主要优化流程实现，模拟3D NMP架构下的MoE推理过程，包含计算与通信开销建模。

- **`baseline.py`**：提供基线策略（TP、EP、混合TP-EP）的实现，用于快速对比优化结果。

- **`expert_trace/`**：存储不同模型（如Mixtral、DeepSeek等）的专家激活统计数据，用于部署策略的生成与优化。


## 支持的模型与数据集（Supported Models & Datasets）

- **模型**：支持MoE架构模型（如Qwen、Mixtral、DeepSeek等），可通过 `expert_trace/` 中的专家激活数据扩展新模型。
- **数据集**：默认使用MT Bench数据集（广泛用于LLM性能评估），可在评估脚本中替换为其他数据集。

