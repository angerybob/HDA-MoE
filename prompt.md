​帮我搜一下python3 gen_model_answer.py \
  --model-path /path/to/ds \
  --model-id myrun \
  --batch 32 \
  --hd-mesh-rows 4 \
  --hd-mesh-cols 8 \
  --hd-comp 10000000000000 \
  --hd-bw 25000000000 \
  --reward-comp -30000 \
  --reward-comm -0.01 \
--data-dir /data1/datasets --datasets humaneval --all-questions --trace /data/home/haochenhuang/TCAD/expert_trace/ds/adaptive/arc-e.json
这个命令关于--reward-comp和--reward-comm的最优参数是多少，
--reward-comp应该是在-1e4到-1e5之间，--reward-comm应该是在-1e-2到-1e-1之间（ds大概是这样，mixtral可能是粗粒度moe超参数需要更保守一些），
--model-path在/data1/pretrained_models/Mixtral-8x7B-v0.1/和/data1/pretrained_models/DeepSeek-V2-Lite-Chat/和/data1/pretrained_models/Qwen2-57B-A14B-Instruct/,一共三个模型都要搜一搜
--model-id是输出答案的文件名，比如humaneval-mixtral-8x7b-v0.1-reward-comp-0.1-reward-comm-0.01.jsonl
最优参数的标准是，在humaneval数据集上，准确率不低于baseline 3个百分点，然后在mt-bench数据集上，加速比最快。

准确率提取方式：
跑gen_model_answer.py时要指定数据集为humaneval，--model-id是输出答案的文件名，比如humaneval-mixtral-8x7b-v0.1-reward-comp-0.1-reward-comm-0.01.jsonl就可以写--model-id humaneval-mixtral-8x7b-v0.1-reward-comp-0.1-reward-comm-0.01，可以去/data1/datasets/humaneval/model_answer/humaneval-mixtral-8x7b-v0.1-reward-comp-0.1-reward-comm-0.01.jsonl找对应的答案，
然后，cd /data/home/haochenhuang/human-eval
python convert_huamaneval_to_samples.py /data1/datasets/humaneval/model_answer/test.jsonl  -o huamaneval.samples.jsonl --normalize转换成可以兼容的评估格式，
然后，用python -m human_eval.evaluate_functional_correctness huamaneval.samples.jsonl评估准确率，他会输出类似{'pass@1': np.float64(0.0)}的东西，里面的浮点数就是准确率。
（可以参考/data/home/haochenhuang/human-eval/EVAL_HUAMANEVAL.md）

加速比提取方式：
你需要先用mt-bench上的reasoning任务（对应的是TCAD/fastchat/fastchat/llm_judge/data/mt_bench/question.jsonl，可以直接指定命令行参数--question-begin 21 --question-end 23就行了）跑一些专家激活的trace，然后把这些trace以TCAD/expert_trace/ds/experts_reasoning_ds.json的命名格式（就是deepseek就是ds，mixtral就是mixtral，qwen2就是qwen），--trace这个参数能控制输出json的路径
注意这里跑gen_model_answer.py时不要指定--data-dir和--datasets，默认就是mt-bench
然后跑/data/home/haochenhuang/TCAD/evaluation/scripts/adaptive.py来计算加速比。
注意adaptive.py中模型也要对应，比如python adaptive.py --model ds就是deepseek，python adaptive.py --model mixtral就是mixtral，python adaptive.py --model qwen就是qwen2那个模型。
跑adaptive.py的时候，会print(f"adaptive_pre_speedup_dynamic:{(comp_dynamic+comm_link_dynamic)/(comp_pre_adaptive+comm_pre_adaptive):.2f}")，提取这个输出的值作为加速比。

请你完成这项任务，为这三个模型都搜出一套最优参数来。可以写一个脚本来丢后台跑来自动完成这些工作。
注意：
- 准确率需要和baseline比，所以应该先测出来两个参数都=0的时候的基准acc
此外，评估mixtral和qwen需要两张卡，跑python3 gen_model_answer.py的时候要加上两个命令行参数--num-gpus-per-model 2 --num-gpus-total 2，可以设置CUDA_VISIBLE_DEVICES=2,3 
- 跑一次数据集要1个小时，所以尽量不要暴力穷举搜索，你想想是不是可以有方向有策略地搜索参数，我的经验是--reward-comp对准确率和加速比的影响比较有单调性，--reward-comm可能单调性稍微弱一些
- 先完成mixtral和qwen2，ds可以最后来