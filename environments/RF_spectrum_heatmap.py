#!/usr/bin/env python
# coding:utf-8
"""
RF Spectrum Interference Pattern Heatmap Generator
严格基于RfEnvironment环境代码的频谱干扰模式可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from typing import List, Dict, Tuple, Optional

# 设置中文字体支持和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")


class RFSpectrumVisualizer:
    """RF频谱干扰模式可视化器 - 基于RfEnvironment"""

    def __init__(self):
        """根据环境代码初始化参数"""
        # 严格按照要求设置参数
        self.channels = np.arange(2412, 2468, 5)  # 2412MHz到2467MHz，间隔5MHz
        self.jammer_freq = 2442  # MHz
        self.jammer_dist = 20  # cm
        self.jamming_power = 10  # dBm
        self.jammer_bandwidth = 20  # MHz (来自环境代码)
        self.time_steps = 100

        # 环境中的干扰类型 - 4种模式
        self.jammer_types = ['constant', 'sweeping', 'random', 'dynamic']  # 对应你要求的4种模式

        print(f"频道设置: {self.channels} MHz")
        print(f"干扰频率: {self.jammer_freq} MHz")
        print(f"干扰距离: {self.jammer_dist} cm")
        print(f"干扰功率: {self.jamming_power} dBm")

        # 创建颜色映射 - 从绿色(无干扰)到红色(强干扰)
        colors = ['#006400', '#32CD32', '#FFFF00', '#FF8C00', '#FF0000']
        self.cmap = LinearSegmentedColormap.from_list('rf_spectrum', colors)

    def simulate_spectral_data(self, jammer_type: str) -> np.ndarray:
        """
        模拟频谱数据 - 基于环境代码的逻辑
        返回: [time_steps, channels] 的RSSI矩阵
        """
        # 初始化RSSI矩阵 (基础噪声水平)
        rssi_matrix = np.random.normal(-80, 5, (self.time_steps, len(self.channels)))

        if jammer_type == 'constant':
            # 恒定干扰 - 固定在2442MHz
            jammed_indices = self._get_affected_channels(self.jammer_freq)
            for idx in jammed_indices:
                # 在被干扰的频道上叠加干扰信号
                interference = self._calculate_interference_power(self.channels[idx])
                rssi_matrix[:, idx] += interference

        elif jammer_type == 'sweeping':
            # 扫频干扰 - 按照环境代码的sweep逻辑
            for t in range(self.time_steps):
                # 模拟sweep_counter逻辑
                sweep_slot = t % len(self.channels)
                current_jammed_freq = self.channels[sweep_slot]

                jammed_indices = self._get_affected_channels(current_jammed_freq)
                for idx in jammed_indices:
                    interference = self._calculate_interference_power(self.channels[idx])
                    rssi_matrix[t, idx] += interference

        elif jammer_type == 'random':
            # 随机干扰 - 完全随机选择频道
            np.random.seed(42)  # 保证可重现
            for t in range(self.time_steps):
                # 每个时间步随机选择干扰频道
                random_freq = np.random.choice(self.channels)
                jammed_indices = self._get_affected_channels(random_freq)
                for idx in jammed_indices:
                    interference = self._calculate_interference_power(self.channels[idx])
                    rssi_matrix[t, idx] += interference

        elif jammer_type == 'dynamic':
            # 动态干扰 - 更复杂的自适应模式
            np.random.seed(123)  # 不同的随机种子
            switch_interval = 10  # 每10步切换策略
            for t in range(self.time_steps):
                # 动态策略：前半段集中攻击高频，后半段攻击低频
                if (t // switch_interval) % 2 == 0:
                    # 攻击高频段
                    target_freq = np.random.choice(self.channels[len(self.channels) // 2:])
                else:
                    # 攻击低频段  
                    target_freq = np.random.choice(self.channels[:len(self.channels) // 2])

                jammed_indices = self._get_affected_channels(target_freq)
                for idx in jammed_indices:
                    interference = self._calculate_interference_power(self.channels[idx])
                    rssi_matrix[t, idx] += interference

        return rssi_matrix

    def _get_affected_channels(self, jammed_freq: float) -> List[int]:
        """获取受干扰影响的频道索引 - 基于环境代码的带宽逻辑"""
        affected_indices = []
        for i, freq in enumerate(self.channels):
            # 基于环境代码: freq > (jammed_freq + bandwidth/2) or freq < (jammed_freq - bandwidth/2)
            if not (freq > (jammed_freq + self.jammer_bandwidth / 2) or
                    freq < (jammed_freq - self.jammer_bandwidth / 2)):
                affected_indices.append(i)
        return affected_indices

    def _calculate_interference_power(self, freq: float) -> float:
        """计算干扰功率 - 考虑距离和频率的影响"""
        # 基础干扰功率
        base_interference = self.jamming_power

        # 距离衰减 (简化的路径损耗模型)
        # 实际环境中会更复杂，这里简化处理
        distance_loss = 20 * np.log10(self.jammer_dist / 10)  # 相对于10cm的损耗

        # 频率响应 (在干扰带宽内)
        freq_response = max(0, 1 - abs(freq - self.jammer_freq) / (self.jammer_bandwidth / 2))

        return (base_interference - distance_loss) * freq_response

    def plot_interference_heatmaps(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制四种干扰模式的对比热力图"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 改为2x2布局
        fig.suptitle('RF频谱干扰模式对比分析\n(频段: 2412-2467MHz, 干扰: 2442MHz@10dBm, 距离: 20cm)',
                     fontsize=14, fontweight='bold', y=0.95)

        # 干扰模式对应的中文名称
        jammer_names = {
            'constant': 'Constant (恒定干扰)',
            'sweeping': 'Sweeping (扫频干扰)',
            'random': 'Random (随机干扰)',
            'dynamic': 'Dynamic (动态干扰)'
        }

        # 将axes展平以便于索引
        axes_flat = axes.flatten()

        # 为每种干扰模式生成热力图
        im = None
        for i, jammer_type in enumerate(self.jammer_types):
            ax = axes_flat[i]

            # 生成频谱数据
            rssi_data = self.simulate_spectral_data(jammer_type)

            # 绘制热力图
            im = ax.imshow(rssi_data.T,
                           aspect='auto',
                           cmap=self.cmap,
                           vmin=-85, vmax=-20,  # RSSI范围
                           origin='lower',
                           interpolation='bilinear')

            # 设置标题
            ax.set_title(jammer_names[jammer_type], fontsize=12, fontweight='bold', pad=15)

            # 设置Y轴 (频道)
            ax.set_ylabel('频道 (MHz)', fontsize=10)
            channel_ticks = np.arange(0, len(self.channels), 2)  # 每隔2个频道显示一个刻度
            ax.set_yticks(channel_ticks)
            ax.set_yticklabels([f'{int(self.channels[i])}' for i in channel_ticks])

            # 设置X轴 (时间)
            ax.set_xlabel('时间步', fontsize=10)
            time_ticks = np.arange(0, self.time_steps, 20)
            ax.set_xticks(time_ticks)
            ax.set_xticklabels([f'{t}' for t in time_ticks])

            # 标注干扰频率位置
            jammer_freq_idx = np.where(self.channels == self.jammer_freq)[0]
            if len(jammer_freq_idx) > 0:
                ax.axhline(y=jammer_freq_idx[0], color='white', linestyle='--',
                           linewidth=2, alpha=0.8, label=f'干扰频率 {self.jammer_freq}MHz')
                ax.legend(loc='upper right', fontsize=8)

        # 添加颜色条
        cbar = fig.colorbar(im, ax=axes_flat, orientation='vertical',
                            fraction=0.05, pad=0.04, aspect=30)
        cbar.set_label('RSSI (dBm)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.92, hspace=0.3, wspace=0.3)

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"图片已保存至: {save_path}")

        return fig

    def plot_frequency_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """绘制频率域分析图"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('频谱干扰模式详细分析', fontsize=14, fontweight='bold')

        # 1. 平均功率谱密度
        ax1 = axes[0, 0]
        for jammer_type in self.jammer_types:
            rssi_data = self.simulate_spectral_data(jammer_type)
            avg_power = np.mean(rssi_data, axis=0)
            ax1.plot(self.channels, avg_power, 'o-', linewidth=2,
                     label=f'{jammer_type.capitalize()}', markersize=4)

        ax1.axvline(x=self.jammer_freq, color='red', linestyle='--', alpha=0.7,
                    label=f'干扰频率 {self.jammer_freq}MHz')
        ax1.set_xlabel('频率 (MHz)')
        ax1.set_ylabel('平均RSSI (dBm)')
        ax1.set_title('平均功率谱密度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 干扰强度统计
        ax2 = axes[0, 1]
        interference_stats = []
        labels = []
        for jammer_type in self.jammer_types:
            rssi_data = self.simulate_spectral_data(jammer_type)
            # 计算高于阈值的点数比例
            threshold = -60  # dBm
            interference_ratio = np.sum(rssi_data > threshold) / rssi_data.size
            interference_stats.append(interference_ratio * 100)
            labels.append(jammer_type.capitalize())

        bars = ax2.bar(labels, interference_stats, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_ylabel('干扰占比 (%)')
        ax2.set_title(f'干扰强度统计 (>{threshold}dBm)')
        ax2.grid(True, alpha=0.3)

        # 在柱状图上添加数值
        for bar, stat in zip(bars, interference_stats):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{stat:.1f}%', ha='center', va='bottom')

        # 3. 时间序列 - 关键频道
        ax3 = axes[1, 0]
        key_channel_idx = np.where(self.channels == self.jammer_freq)[0][0]  # 干扰频道

        for jammer_type in self.jammer_types:
            rssi_data = self.simulate_spectral_data(jammer_type)
            time_series = rssi_data[:, key_channel_idx]
            ax3.plot(range(self.time_steps), time_series,
                     linewidth=1.5, label=f'{jammer_type.capitalize()}', alpha=0.8)

        ax3.set_xlabel('时间步')
        ax3.set_ylabel('RSSI (dBm)')
        ax3.set_title(f'关键频道 {self.jammer_freq}MHz 时间序列')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 参数信息表
        ax4 = axes[1, 1]
        ax4.axis('off')

        # 创建参数表格
        param_data = [
            ['参数', '设置值'],
            ['频段范围', f'{self.channels[0]}-{self.channels[-1]} MHz'],
            ['频道间隔', '5 MHz'],
            ['干扰频率', f'{self.jammer_freq} MHz'],
            ['干扰功率', f'{self.jamming_power} dBm'],
            ['干扰源距离', f'{self.jammer_dist} cm'],
            ['干扰带宽', f'{self.jammer_bandwidth} MHz'],
            ['时间步数', f'{self.time_steps}'],
            ['频道总数', f'{len(self.channels)}']
        ]

        table = ax4.table(cellText=param_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # 设置表格样式
        for i in range(len(param_data)):
            if i == 0:  # 表头
                table[(i, 0)].set_facecolor('#4ECDC4')
                table[(i, 1)].set_facecolor('#4ECDC4')
                table[(i, 0)].set_text_props(weight='bold')
                table[(i, 1)].set_text_props(weight='bold')
            else:
                table[(i, 0)].set_facecolor('#F0F0F0')
                table[(i, 1)].set_facecolor('#FFFFFF')

        ax4.set_title('仿真参数设置', fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            analysis_path = save_path.replace('.png', '_analysis.png')
            plt.savefig(analysis_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"分析图已保存至: {analysis_path}")

        return fig

    def generate_report(self):
        """生成文字报告"""
        print("\n" + "=" * 60)
        print("RF频谱干扰模式分析报告")
        print("=" * 60)

        for jammer_type in self.jammer_types:
            rssi_data = self.simulate_spectral_data(jammer_type)

            print(f"\n【{jammer_type.upper()}干扰模式】")
            print(f"平均RSSI: {np.mean(rssi_data):.2f} dBm")
            print(f"RSSI标准差: {np.std(rssi_data):.2f} dBm")
            print(f"最大RSSI: {np.max(rssi_data):.2f} dBm")
            print(f"最小RSSI: {np.min(rssi_data):.2f} dBm")

            # 计算受影响频道
            avg_power = np.mean(rssi_data, axis=0)
            affected_channels = []
            for i, power in enumerate(avg_power):
                if power > -60:  # 阈值
                    affected_channels.append(self.channels[i])

            print(f"主要受影响频道: {affected_channels} MHz")
            print(f"受影响频道数量: {len(affected_channels)}/{len(self.channels)}")


def main():
    """主函数 - 生成所有可视化结果"""
    print("RF频谱干扰模式可视化工具")
    print("基于RfEnvironment环境代码")
    print("-" * 40)

    # 创建可视化器
    visualizer = RFSpectrumVisualizer()

    # 生成主要的热力图对比
    print("\n正在生成干扰模式热力图...")
    fig1 = visualizer.plot_interference_heatmaps(save_path='rf_interference_heatmap.png')
    plt.show()

    # 生成详细分析图
    print("\n正在生成频率域分析图...")
    fig2 = visualizer.plot_frequency_analysis(save_path='rf_interference_analysis.png')
    plt.show()

    # 生成文字报告
    visualizer.generate_report()

    print(f"\n可视化完成！")
    print("生成的图片:")
    print("1. rf_interference_heatmap.png - 干扰模式对比热力图")
    print("2. rf_interference_analysis.png - 详细频率分析图")


if __name__ == "__main__":
    main()