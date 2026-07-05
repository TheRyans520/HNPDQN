import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def generate_optimized_jamming_visuals():
    # --- 参数设置 ---
    time_steps = 100
    channels = np.array([2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 2452, 2457, 2462, 2467])
    num_channels = len(channels)

    # 正常通信时的SNR（假设无干扰时SNR = 25 dB）
    normal_snr = 25  # dB
    # 干扰造成的SNR下降（干扰时SNR降到 0-5 dB）
    jammed_snr = 3  # dB

    # 模拟基础SNR（正常通信）
    base_snr = np.random.normal(normal_snr, 2, (time_steps, num_channels))

    # 干扰颜色映射（深蓝表示差SNR，亮色表示好SNR）
    # 注意：SNR越高越好，所以颜色映射要反转
    colors = ['#FF0000', '#FFFF00', '#006400', '#000033']  # 红(差) -> 绿 -> 蓝(好)
    my_cmap = LinearSegmentedColormap.from_list('snr_jamming', colors)

    # --- 1. 生成数据（SNR模式）---
    # Sweeping Jammer
    sweeping_snr = base_snr.copy()
    for t in range(time_steps):
        jammed_idx = t % num_channels
        sweeping_snr[t, jammed_idx] = jammed_snr  # 干扰时SNR骤降

    # Random Jammer
    random_snr = base_snr.copy()
    np.random.seed(42)
    for t in range(time_steps):
        jammed_idx = np.random.randint(0, num_channels)
        random_snr[t, jammed_idx] = jammed_snr  # 干扰时SNR骤降

    # --- 绘图函数（SNR版本）---
    def plot_snr_heatmap(data, title, filename_base):
        plt.figure(figsize=(10, 6))

        # SNR范围：0-30 dB（0=完全被干扰，30=通信质量极佳）
        ax = sns.heatmap(data.T, cmap=my_cmap, vmin=0, vmax=30,
                         xticklabels=10,
                         cbar_kws={'label': 'SNR (dB)', 'ticks': [0, 5, 10, 15, 20, 25, 30]})

        plt.title(title, fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Time Slots', fontsize=11)
        plt.ylabel('Frequency (MHz)', fontsize=11)

        # 设置纵坐标
        plt.yticks(np.arange(num_channels) + 0.5, channels, rotation=0)

        plt.tight_layout()

        # 保存为PDF矢量图
        pdf_filename = f"{filename_base}.pdf"
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', dpi=300)
        print(f"已保存矢量图: {pdf_filename}")

        # 可选：同时保存高分辨率PNG
        png_filename = f"{filename_base}.png"
        plt.savefig(png_filename, dpi=600, bbox_inches='tight')
        print(f"已保存位图: {png_filename}")

        plt.show()

    # --- 输出图像 ---
    plot_snr_heatmap(sweeping_snr, "Sweeping Jammer", "jammer_sweeping_snr")
    plot_snr_heatmap(random_snr, "Random Jammer", "jammer_random_snr")


if __name__ == "__main__":
    sns.set_theme(style="white")
    generate_optimized_jamming_visuals()