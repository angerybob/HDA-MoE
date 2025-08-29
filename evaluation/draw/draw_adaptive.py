import argparse
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np

def plot_adaptive_results(x_axis: str = "reward_comp",
                          json_file_path: str = "/data/home/shengxuanqiu/TCAD/evaluation/results/result_adaptive.json",
                          filter_comp: float | None = 3e-4):
    """
    读取 result_adaptive.json 文件，并绘制
    Speedup 和 Accuracy 关于 Reward Compensation/Communication 的关系图。
    x_axis: "reward_comp" 或 "reward_comm" 或 "batch"
    """
    os.makedirs('/data/home/shengxuanqiu/TCAD/evaluation/figs/adaptive', exist_ok=True)

    if x_axis == "batch":
        # 批处理模式 - 绘制不同批大小的性能比较
        batches = [1, 4, 8, 32, 64, 128]
        
        # 存储数据
        speedup_data = {}
        accuracy_values = [0.75, 0.7188, 0.7188, 0.73, 0.7014, 0.7188]  # 准确率数据
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    config = entry['config']
                    
                    # 筛选指定的reward配置
                    if (config['reward_comp'] == "-3e4" and 
                        config['reward_comm'] == "-2e-2" and
                        config['batch'] in batches):
                        
                        batch = config['batch']
                        
                        speedup_data[batch] = {
                            'pre-broadcast': entry['dynamic_deployment']['speedup'],
                            'adap': entry['adaptive_deployment']['speedup'],
                            'pre+adap': entry['adaptive_dynamic_deployment']['speedup']
                        }
                        
        except FileNotFoundError:
            print(f"错误: 文件未找到 {json_file_path}")
            return
        except (json.JSONDecodeError, KeyError) as e:
            print(f"错误: 解析JSON文件失败或缺少键: {e}")
            return

        if not speedup_data:
            print("错误: 未能从文件中加载有效数据进行绘图。")
            return

        # 绘图设置
        plt.rcParams.update({
            "font.size": 25,
            "axes.labelweight": "normal",
            "axes.labelsize": 40,
            "legend.frameon": True,
            "lines.linewidth": 3
        })
        size = 25
        
        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(18, 8))
        
        # 设置颜色（绿，蓝，橙，红）
        colors = ["#59A14F", "#4E79A7", "#F28E2B", "#E15759"]
        
        # 准备数据
        x = np.arange(len(batches))
        width = 0.2
        
        # 绘制速度比的柱状图（从左到右：baseline, pre-broadcast, adap, pre+adap）
        ax.bar(x - 1.5*width, [1.0] * len(batches),  # baseline固定为1
               width, label='baseline', color=colors[0], edgecolor="black")
        ax.bar(x - 0.5*width, [speedup_data[batch]['pre-broadcast'] for batch in batches], 
               width, label='pre-broadcast', color=colors[1], edgecolor="black")
        ax.bar(x + 0.5*width, [speedup_data[batch]['adap'] for batch in batches], 
               width, label='adap', color=colors[2], edgecolor="black")
        ax.bar(x + 1.5*width, [speedup_data[batch]['pre+adap'] for batch in batches], 
               width, label='pre+adap', color=colors[3], edgecolor="black")
        
        # 绘制准确率的折线图
        ax2 = ax.twinx()
        ax2.plot(x, accuracy_values, 
                color="green", linestyle="--", marker="o", label="Accuracy", markersize=8, linewidth=3)
        
        # 设置坐标轴
        ax.set_xlabel("Batch Size", fontsize=size)
        ax.set_ylabel("Speedup", fontsize=size)
        ax.set_xticks(x)
        ax.set_xticklabels([str(batch) for batch in batches])
        ax.legend(loc="upper left", fontsize=20)
        
        # 设置准确率的y轴
        ax2.set_ylabel("Accuracy", fontsize=size)
        ax2.set_ylim([0.2, 1.0])
        
        plt.tight_layout()
        out_png = '/data/home/shengxuanqiu/TCAD/evaluation/figs/adaptive/adaptive_batch_comparison.png'
        fig.savefig(out_png, dpi=300)

    elif x_axis == "reward_comp":
        accuracy_data = {
            0.5: 0.7500,
            1.0: 0.6875,
            2.0: 0.7188,
            3.0: 0.7500,
            4.0: 0.6562,
            5.0: 0.6250,
            6.0: 0.5938,
            7.0: 0.5625,
            8.0: 0.3750,
            9.0: 0.2500
        }

        plot_data = []
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    comp_val = float(entry['config']['reward_comm'])
                    if filter_comp is not None and not math.isclose(abs(comp_val), 0, rel_tol=0, abs_tol=1e-12):
                        continue

                    reward_str = entry['config']['reward_comp']
                    reward_abs_val = abs(float(reward_str)) / 10000.0  # 单位：1e4
                    speedup = entry['adaptive_deployment']['speedup']
                    accuracy = accuracy_data.get(reward_abs_val)
                    if accuracy is not None:
                        plot_data.append({
                            'reward': reward_abs_val,
                            'speedup': speedup,
                            'accuracy': accuracy
                        })
        except FileNotFoundError:
            print(f"错误: 文件未找到 {json_file_path}")
            return
        except (json.JSONDecodeError, KeyError) as e:
            print(f"错误: 解析JSON文件失败或缺少键: {e}")
            return

        if not plot_data:
            print("错误: 未能从文件中加载有效数据进行绘图。")
            return

        plot_data.sort(key=lambda x: x['reward'])
        rewards = [item['reward'] for item in plot_data]
        speedups = [item['speedup'] for item in plot_data]
        accuracies = [item['accuracy'] for item in plot_data]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('reward comp (-1e4)')
        ax1.set_xticks(rewards)
        size = 15

        color1 = 'tab:blue'
        ax1.set_ylabel('Speedup', fontsize=size)
        ax1.plot(rewards, speedups, color=color1, marker='o', linestyle='-', label='Speedup', linewidth=2.5)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(1.0, 2.4)
        ax1.grid(True, which='both', linestyle='--', linewidth=1)

        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Accuracy', fontsize=size)
        ax2.plot(rewards, accuracies, color=color2, marker='s', linestyle='-', label='Accuracy', linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 1.0)

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        fig.tight_layout()
        out_png = '/data/home/shengxuanqiu/TCAD/evaluation/figs/adaptive/adaptive_performance.png'
        fig.savefig(out_png, dpi=300)

    else:  # x_axis == "reward_comm"
        accuracy_data_comm = {
            0:0.6875,
            1.0:0.7188,
            2.0:0.6973,
            3.0:0.7188,
            4.0:0.7500,
            5.0:0.7188,
            6.0:0.6875,
            7.0:0.6342,
            8.0:0.5938,
            9.0:0.5564
        }

        plot_data = []
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    
                    comp_val = float(entry['config']['reward_comp'])
                    if filter_comp is not None and not math.isclose(abs(comp_val), float(filter_comp), rel_tol=0, abs_tol=1e-12):
                        continue

                    reward_str = entry['config']['reward_comm']
                    reward_abs_val = abs(float(reward_str)) / 1e-2
                    speedup = entry['adaptive_deployment']['speedup']

                    accuracy = accuracy_data_comm.get(reward_abs_val)
                    plot_data.append({
                        'reward': reward_abs_val,
                        'speedup': speedup,
                        'accuracy': accuracy  
                    })
        except FileNotFoundError:
            print(f"错误: 文件未找到 {json_file_path}")
            return
        except (json.JSONDecodeError, KeyError) as e:
            print(f"错误: 解析JSON文件失败或缺少键: {e}")
            return

        if not plot_data:
            print("错误: 未能从文件中加载有效数据进行绘图。")
            return

        plot_data.sort(key=lambda x: x['reward'])
        rewards = [item['reward'] for item in plot_data]
        speedups = [item['speedup'] for item in plot_data]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('reward comm (-1e-2)')
        ax1.set_xticks(rewards)
        size = 15

        color1 = 'tab:blue'
        ax1.set_ylabel('Speedup', fontsize=size)
        ax1.plot(rewards, speedups, color=color1, marker='o', linestyle='-', label='Speedup', linewidth=2.5)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(1.4, 1.9)
        ax1.grid(True, which='both', linestyle='--', linewidth=1)

        if any(item['accuracy'] is not None for item in plot_data):
            accuracies = [item['accuracy'] for item in plot_data if item['accuracy'] is not None]
            rewards_acc = [item['reward'] for item in plot_data if item['accuracy'] is not None]
            ax2 = ax1.twinx()
            color2 = 'tab:green'
            ax2.set_ylabel('Accuracy', fontsize=size)
            ax2.plot(rewards_acc, accuracies, color=color2, marker='s', linestyle='-', label='Accuracy', linewidth=2.5)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_ylim(0, 1.0)
            fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

        fig.tight_layout()
        out_png = '/data/home/shengxuanqiu/TCAD/evaluation/figs/adaptive/adaptive_performance_comm.png'
        fig.savefig(out_png, dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-axis', choices=['reward_comp', 'reward_comm', 'batch'], default='batch',
                        help='选择横坐标')
    parser.add_argument('--json', type=str, default='/data/home/shengxuanqiu/TCAD/evaluation/results/result_adaptive.json',
                        help='结果 JSON 文件路径')
    parser.add_argument('--filter-comp', type=str, default='3e4',
                        help='仅保留 reward_comp为该值的条目')
    args = parser.parse_args()

    fc_arg = args.filter_comp.strip().lower()
    if fc_arg in ('none', 'all', ''):
        fc = None
    else:
        fc = float(args.filter_comp)

    plot_adaptive_results(x_axis=args.x_axis, json_file_path=args.json, filter_comp=fc)