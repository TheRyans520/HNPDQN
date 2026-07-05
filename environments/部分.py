#!/usr/bin/env python
# coding:utf-8
"""
论文示意图：动态Jammer频率切换与RL智能体部分观测状态
Paper Figure: Dynamic Jammer Frequency Switching with RL Agent Partial Observation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# 设置论文级别的图形参数
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class PaperFigureGenerator:
    """论文示意图生成器"""

    def __init__(self):
        # 参数设置
        self.channels = np.arange(2412, 2468, 5)  # 12个频道
        self.time_steps = 80  # 减少时间步以适应论文图片
        self.observation_window = 8  # RL智能体观测窗口
        self.current_time = 50  # 当前时间点

        # 颜色设置
        self.interference_colors = ['#E8F5E8', '#90EE90', '#FFD700', '#FF6347', '#8B0000']
        self.cmap = LinearSegmentedColormap.from_list('interference', self.interference_colors)

    def generate_jammer_scenario(self):
        """生成动态jammer场景"""
        # 初始化频谱数据
        rssi_matrix = np.random.normal(-75, 3, (self.time_steps, len(self.channels)))

        # 生成jammer动态切换序列
        jammer_positions = []

        # 创建更清晰的切换模式
        for t in range(self.time_steps):
            if t < 20:
                # 阶段1：低频段攻击
                pos = 2 + (t % 3)
            elif t < 40:
                # 阶段2：跳跃攻击
                pos = [1, 5, 9, 3, 7][t % 5]
            elif t < 60:
                # 阶段3：高频段扫描
                pos = 8 + (t % 3)
            else:
                # 阶段4：快速随机切换
                pos = np.random.choice([0, 4, 8, 11])

            jammer_positions.append(pos)

            # 在被攻击频道添加干扰
            jammer_freq = self.channels[pos]
            for i, freq in enumerate(self.channels):
                if abs(freq - jammer_freq) <= 10:  # 20MHz带宽的一半
                    interference = 15 * max(0, 1 - abs(freq - jammer_freq) / 10)
                    rssi_matrix[t, i] += interference

        return rssi_matrix, jammer_positions

    def create_paper_figure(self, save_path='paper_jammer_figure.png'):
        """创建论文示意图"""

        # 生成数据
        rssi_matrix, jammer_positions = self.generate_jammer_scenario()

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # === 上图：完整的jammer动态切换视图 ===
        im1 = ax1.imshow(rssi_matrix.T,
                         aspect='auto',
                         cmap=self.cmap,
                         vmin=-80, vmax=-30,
                         origin='lower',
                         interpolation='gaussian')

        # 绘制jammer切换轨迹
        ax1.plot(range(self.time_steps), jammer_positions,
                 color='red', linewidth=3, alpha=0.9)

        # 标记jammer位置
        ax1.scatter(range(self.time_steps), jammer_positions,
                    color='darkred', s=25, alpha=0.8, zorder=5)

        # 标记当前时间
        ax1.axvline(x=self.current_time, color='blue', linestyle='--',
                    linewidth=3, alpha=0.8, label='Current Time')

        # 设置标题和标签 - 增加title的pad避免重叠
        ax1.set_title('(a) Full Spectrum View: Dynamic Jammer Frequency Switching',
                      fontsize=14, fontweight='bold', pad=25)
        ax1.set_ylabel('Channel Index (MHz)', fontsize=12)
        ax1.set_xlabel('Time Steps', fontsize=12)

        # 设置坐标轴
        channel_ticks = range(0, len(self.channels), 2)
        ax1.set_yticks(channel_ticks)
        ax1.set_yticklabels([f'{int(self.channels[i])}' for i in channel_ticks])

        time_ticks = range(0, self.time_steps, 10)
        ax1.set_xticks(time_ticks)


        # === 下图：RL智能体的部分观测视图 ===
        obs_start = max(0, self.current_time - self.observation_window + 1)
        obs_end = self.current_time + 1
        obs_rssi = rssi_matrix[obs_start:obs_end, :]
        obs_jammer_pos = jammer_positions[obs_start:obs_end]

        im2 = ax2.imshow(obs_rssi.T,
                         aspect='auto',
                         cmap=self.cmap,
                         vmin=-80, vmax=-30,
                         origin='lower',
                         interpolation='gaussian')

        # 绘制智能体观测到的jammer轨迹
        local_time_steps = range(len(obs_jammer_pos))
        ax2.plot(local_time_steps, obs_jammer_pos,
                 color='red', linewidth=3, alpha=0.9, label='Observable Jammer')
        ax2.scatter(local_time_steps, obs_jammer_pos,
                    color='darkred', s=25, alpha=0.8, zorder=5)

        # 标记当前时间（在局部坐标系中）
        current_local = len(obs_jammer_pos) - 1
        ax2.axvline(x=current_local, color='blue', linestyle='--',
                    linewidth=3, alpha=0.8)

        # 添加观测窗口边框强调
        rect = patches.Rectangle((0, -0.5), len(obs_jammer_pos) - 1, len(self.channels),
                                 linewidth=3, edgecolor='blue', facecolor='none',
                                 linestyle='-', alpha=0.7)
        ax2.add_patch(rect)

        # 设置标题和标签 - 将标题移到图下方
        ax2.set_ylabel('Channel Index (MHz)', fontsize=12)
        ax2.set_xlabel('Relative Time Steps', fontsize=12, labelpad=10)

        # 将(b)标题放在图下方但不挡住xlabel
        ax2.text(0.5, -0.25, f'(b) RL Agent Partial Observation (Window Size: {self.observation_window})',
                 transform=ax2.transAxes, ha='center', va='top', fontsize=14, fontweight='bold')

        # 设置坐标轴
        ax2.set_yticks(channel_ticks)
        ax2.set_yticklabels([f'{int(self.channels[i])}' for i in channel_ticks])
        ax2.set_xticks(range(len(obs_jammer_pos)))

        # 添加图例和说明文字 - 移动图例到图外下方
        # 不在图内添加图例，避免遮挡线条

        # === 添加统一的颜色条 - 进一步增加pad ===
        cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical',
                            fraction=0.03, pad=0.15, aspect=40)
        cbar.set_label('Signal Strength (dBm)', fontsize=12)

        # === 添加连接线和注释 ===
        # 添加箭头连接完整视图和部分观测视图
        con = patches.ConnectionPatch((obs_start, -0.5), (0, -0.5),
                                      "data", "data",
                                      axesA=ax1, axesB=ax2,
                                      color="blue", linewidth=2, alpha=0.6,
                                      arrowstyle="->", shrinkB=5)
        ax2.add_artist(con)

        con2 = patches.ConnectionPatch((obs_end - 1, -0.5), (len(obs_jammer_pos) - 1, -0.5),
                                       "data", "data",
                                       axesA=ax1, axesB=ax2,
                                       color="blue", linewidth=2, alpha=0.6,
                                       arrowstyle="->", shrinkB=5)
        ax2.add_artist(con2)

        # === 在完整视图中标记观测窗口区域 ===
        obs_rect = patches.Rectangle((obs_start, -0.5), self.observation_window, len(self.channels),
                                     linewidth=2, edgecolor='blue', facecolor='lightblue',
                                     alpha=0.2, linestyle='--')
        ax1.add_patch(obs_rect)


        # === 最终布局调整 - 为下方标题和更远的颜色条留出空间 ===
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5, top=0.92, bottom=0.18, right=0.82, left=0.08)

        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"论文示意图已保存至: {save_path}")

        return fig


def main():
    """生成论文示意图"""
    print("正在生成论文示意图...")
    generator = PaperFigureGenerator()
    fig = generator.create_paper_figure('paper_jammer_rl_figure.png')
    plt.show()
    print("完成！")


if __name__ == "__main__":
    main()