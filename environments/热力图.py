#!/usr/bin/env python
# coding:utf-8
"""
雷达频谱热力图生成器
基于RfEnvironment真实频谱扫描数据生成时间-频率功率谱密度热力图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib import cm
import copy
import os
import sys


# 导入您的环境类
from RF_spectrum import RfEnvironment


class RealRadarSpectrumVisualizer:
    def __init__(self, jammers=['constant'], jammer_dist=20, jamming_power=10,
                 channels=None, length=32, stride=1, csc=0.05):
        """
        基于真实RfEnvironment数据的频谱可视化器
        """
        if channels is None:
            channels = list(range(5180, 5340, 20))  # 默认频道

        self.channels = channels
        self.jammers = jammers
        self.jammer_dist = jammer_dist
        self.jamming_power = jamming_power
        self.length = length
        self.stride = stride
        self.csc = csc
        self.jammer_bandwidth = 20  # MHz

        # 创建环境实例
        self.env = RfEnvironment(
            jammers=jammers,
            jammer_dist=jammer_dist,
            jamming_power=jamming_power,
            csc=csc,
            channels=channels,
            length=length,
            stride=stride
        )

        print(f"频道范围: {self.channels[0]} - {self.channels[-1]} MHz")
        print(f"频道数量: {len(self.channels)}")
        print(f"干扰带宽: {self.jammer_bandwidth} MHz")
        print(f"环境ID: {self.env.id}")

    def collect_episode_data(self, n_episodes=5, episode_length=100):
        """
        收集多个episode的频谱数据和agent动作
        """
        all_spectral_data = []
        all_actions = []
        all_rewards = []
        all_jammer_info = []

        for episode in range(n_episodes):
            print(f"收集第 {episode + 1}/{n_episodes} 个episode的数据...")

            # 重置环境
            state = self.env.reset()
            episode_spectral = []
            episode_actions = []
            episode_rewards = []
            episode_jammer = []

            for step in range(episode_length):
                # 记录当前状态的频谱数据
                current_spectral = self.get_current_spectral_data()
                episode_spectral.append(current_spectral)

                # 记录干扰器信息
                jammer_info = {
                    'jammer_type': self.env.jammer,
                    'jammed_freq': self.env.jammed_freq,
                    'time_step': step
                }
                episode_jammer.append(jammer_info)

                # 选择动作（这里使用随机策略，您可以替换为训练好的agent）
                action = self.simple_anti_jam_policy(state, step)
                episode_actions.append({
                    'step': step,
                    'action': action,
                    'frequency': self.channels[action]
                })

                # 执行动作
                state, reward, done, _ = self.env.step(action)
                episode_rewards.append(reward)

                if done:
                    break

            all_spectral_data.append(episode_spectral)
            all_actions.append(episode_actions)
            all_rewards.append(episode_rewards)
            all_jammer_info.append(episode_jammer)

        return all_spectral_data, all_actions, all_rewards, all_jammer_info

    def get_current_spectral_data(self):
        """
        直接使用环境中已经加载的频谱数据
        """
        spectral_data = {}

        # 直接使用环境中的spectral_data和处理后的state
        current_state = self.env.state

        # 为每个频道构建数据
        for i, channel in enumerate(self.channels):
            # 使用环境中当前的state值作为SNR
            if current_state is not None and i < len(current_state):
                snr_value = float(current_state[i])
            else:
                snr_value = -70  # 默认值

            spectral_data[channel] = {
                'snr_mean': snr_value,
                'snr_std': 5,  # 假设标准差
                'snr_values': np.array([snr_value]),
                'raw_data': []
            }

        return spectral_data

    def simple_anti_jam_policy(self, state, step):
        """
        简单的抗干扰策略
        """
        # 找到SNR最高的频道
        try:
            state_reshaped = state.reshape(-1)
            best_channel = np.argmax(state_reshaped)

            # 添加一些随机性避免预测性太强
            if np.random.random() < 0.1:  # 10%的探索概率
                best_channel = np.random.choice(len(self.channels))

            return best_channel
        except:
            return np.random.choice(len(self.channels))

    def create_spectrum_matrix(self, spectral_data_list):
        """
        将频谱数据转换为热力图矩阵
        """
        n_steps = len(spectral_data_list)
        n_channels = len(self.channels)

        # 创建矩阵 (频道 x 时间)
        spectrum_matrix = np.zeros((n_channels, n_steps))

        for step_idx, spectral_data in enumerate(spectral_data_list):
            for channel_idx, channel in enumerate(self.channels):
                if channel in spectral_data:
                    spectrum_matrix[channel_idx, step_idx] = spectral_data[channel]['snr_mean']
                else:
                    spectrum_matrix[channel_idx, step_idx] = -70  # 默认噪声水平

        return spectrum_matrix

    def plot_episode_heatmap(self, episode_idx=0, n_episodes=1, save_path=None):
        """
        绘制单个episode的频谱热力图
        """
        # 收集数据
        all_spectral, all_actions, all_rewards, all_jammer_info = self.collect_episode_data(
            n_episodes=n_episodes, episode_length=100
        )

        if episode_idx >= len(all_spectral):
            episode_idx = 0

        spectral_data = all_spectral[episode_idx]
        actions = all_actions[episode_idx]
        jammer_info = all_jammer_info[episode_idx]

        # 创建频谱矩阵
        spectrum_matrix = self.create_spectrum_matrix(spectral_data)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10),
                                       gridspec_kw={'height_ratios': [4, 1]})

        # 主热力图
        im = ax1.imshow(spectrum_matrix,
                        aspect='auto',
                        origin='lower',
                        extent=[0, len(spectral_data),
                                0, len(self.channels) - 1],
                        cmap='viridis',
                        vmin=-80, vmax=-20)

        # 设置频道标签
        ax1.set_yticks(range(len(self.channels)))
        ax1.set_yticklabels([f'{freq}\nMHz' for freq in self.channels])

        # 标记干扰区域
        for step_info in jammer_info:
            if step_info['jammer_type'] != 'none':
                jammed_freq = step_info['jammed_freq']
                if jammed_freq in self.channels:
                    jammed_idx = self.channels.index(jammed_freq)
                    time_step = step_info['time_step']

                    # 标记干扰区域
                    rect = patches.Rectangle((time_step - 0.5, jammed_idx - 0.4),
                                             1, 0.8,
                                             linewidth=1, edgecolor='red',
                                             facecolor='red', alpha=0.3)
                    ax1.add_patch(rect)

        # 标记第一次出现的干扰
        first_jam_freq = None
        first_jam_type = None
        for step_info in jammer_info:
            if step_info['jammer_type'] != 'none':
                first_jam_freq = step_info['jammed_freq']
                first_jam_type = step_info['jammer_type']
                break

        if first_jam_freq and first_jam_freq in self.channels:
            jammed_idx = self.channels.index(first_jam_freq)
            ax1.text(5, jammed_idx, f'干扰: {first_jam_freq} MHz ({first_jam_type})',
                     color='red', fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax1.set_xlabel('时间步', fontsize=12)
        ax1.set_ylabel('频道', fontsize=12)
        ax1.set_title(f'真实雷达频谱热力图 - Episode {episode_idx + 1}\n'
                      f'干扰类型: {first_jam_type if first_jam_type else "未知"}, '
                      f'功率: {self.jamming_power} dBm, 带宽: {self.jammer_bandwidth} MHz',
                      fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)

        # Agent选频时间线
        action_timeline = [action['action'] for action in actions]
        time_steps = [action['step'] for action in actions]

        # 创建颜色映射
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.channels)))

        for i in range(len(time_steps) - 1):
            start_time = time_steps[i]
            end_time = time_steps[i + 1]
            channel_idx = action_timeline[i]

            ax2.barh(0, end_time - start_time, left=start_time,
                     color=colors[channel_idx], alpha=0.8, height=0.8)

            # 添加频道标签
            if end_time - start_time > 2:  # 只在较宽的区间显示标签
                ax2.text((start_time + end_time) / 2, 0,
                         f'Ch{channel_idx}\n{self.channels[channel_idx]}',
                         ha='center', va='center', fontsize=8, fontweight='bold')

        ax2.set_xlim(0, len(spectral_data))
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('时间步', fontsize=12)
        ax2.set_ylabel('选择频道', fontsize=12)
        ax2.set_title('Agent选频时间线', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks([])

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('SNR (dB)', fontsize=12)

        # 创建频道图例
        legend_elements = [patches.Patch(color=colors[i],
                                         label=f'Ch{i}: {self.channels[i]}MHz')
                           for i in range(len(self.channels))]
        ax2.legend(handles=legend_elements,
                   bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存至: {save_path}")

        plt.show()

        # 统计信息
        total_switches = sum(1 for i in range(1, len(action_timeline))
                             if action_timeline[i] != action_timeline[i - 1])
        channels_used = len(set(action_timeline))

        stats = {
            'episode': episode_idx + 1,
            'jammer_type': first_jam_type,
            'jammed_frequency': first_jam_freq,
            'total_channel_switches': total_switches,
            'channels_used': channels_used,
            'average_snr': np.mean(spectrum_matrix),
            'snr_range': (np.min(spectrum_matrix), np.max(spectrum_matrix))
        }

        return stats


# 使用示例
if __name__ == "__main__":
    # 创建基于真实数据的可视化器
    channels = list(range(5180, 5340, 20))  # 您代码中的频道设置

    print("正在创建真实数据可视化器...")
    visualizer = RealRadarSpectrumVisualizer(
        jammers=['constant'],
        jammer_dist=20,
        jamming_power=10,
        channels=channels,
        length=32,
        stride=1,
        csc=0.05
    )

    print("正在生成恒定干扰热力图...")
    stats1 = visualizer.plot_episode_heatmap(episode_idx=0, n_episodes=1)
    print(f"统计信息: {stats1}")

    # 生成扫频干扰的热力图
    print("\n正在生成扫频干扰热力图...")
    visualizer_sweep = RealRadarSpectrumVisualizer(
        jammers=['sweeping'],
        jammer_dist=20,
        jamming_power=10,
        channels=channels,
        length=32,
        stride=1,
        csc=0.05
    )
    stats2 = visualizer_sweep.plot_episode_heatmap(episode_idx=0, n_episodes=1)
    print(f"统计信息: {stats2}")

    # 生成随机干扰的热力图
    print("\n正在生成随机干扰热力图...")
    visualizer_random = RealRadarSpectrumVisualizer(
        jammers=['random'],
        jammer_dist=20,
        jamming_power=10,
        channels=channels,
        length=32,
        stride=1,
        csc=0.05
    )
    stats3 = visualizer_random.plot_episode_heatmap(episode_idx=0, n_episodes=1)
    print(f"统计信息: {stats3}")