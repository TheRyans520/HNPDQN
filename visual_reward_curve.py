import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


class FinalReproductionTrainer:
    def __init__(self):
        # 1. 顶刊配色方案 (Key 保持与 pkl 一致)
        self.agent_to_color_group = {
            "HNP_DQN": "#E31A1C",  # 顶刊正红
            "DQN": "#1F78B4",  # 科技蓝
            "DDQN": "#33A02C",  # 森林绿
            "DQN with Fixed Q Targets": "#6A3D9A"  # 皇家紫
        }
        # 对应训练配置中的标准差倍数
        self.standard_deviation_results = 1.0

    def get_mean_and_standard_deviation_difference_results(self, results):
        """完全还原 Trainer 类中的均值与标准差计算逻辑"""

        def get_results_at_a_time_step(results, timestep):
            return [result[timestep] for result in results]

        def get_standard_deviation_at_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return np.std(results_at_a_time_step)

        min_len = min(len(r) for r in results)
        mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(min_len)]

        mean_minus_x_std = [
            mean_val - self.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep)
            for timestep, mean_val in enumerate(mean_results)]

        mean_plus_x_std = [
            mean_val + self.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep)
            for timestep, mean_val in enumerate(mean_results)]

        return mean_minus_x_std, mean_results, mean_plus_x_std

    def plot_and_save(self, pkl_path, output_basename="test_results"):
        with open(pkl_path, 'rb') as f:
            preexisting_results = pickle.load(f)

        # 稍微增加高度 (5.5) 给下方的方框图例留出空间
        fig, ax = plt.subplots(figsize=(7, 5.8))

        max_episodes = 0

        # 遍历数据绘图
        for agent_name, agent_data in preexisting_results.items():
            # 【修改点 1】: 将图例显示的名称从下划线改为连字符
            display_name = agent_name.replace("HNP_DQN", "HNP-DQN")

            # 提取前两个实验的 rolling_scores (索引 1)
            agent_rolling_score_results = [run[1] for run in agent_data[:2]]
            color = self.agent_to_color_group.get(agent_name, "blue")

            low, mid, high = self.get_mean_and_standard_deviation_difference_results(agent_rolling_score_results)

            current_len = len(mid)
            if current_len > max_episodes:
                max_episodes = current_len

            x_vals = np.arange(current_len)

            # 核心算法 HNP-DQN 加粗显示
            lw = 2.5 if "HNP" in display_name else 1.8
            ax.plot(x_vals, mid, label=display_name, color=color, linewidth=lw,
                    zorder=3 if "HNP" in display_name else 2)
            ax.fill_between(x_vals, low, high, alpha=0.15, color=color, zorder=1)

        # --- 样式优化 ---

        # 【修改点 2】: 强制横坐标范围，消除两端溢出
        ax.set_xlim(0, max_episodes)

        # 【修改点 3】: 强制显示两端刻度 (0 和 200)
        tick_values = np.linspace(0, max_episodes, 6, dtype=int)
        ax.set_xticks(tick_values)

        # 封闭边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor('black')

        # 坐标轴格式化
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

        # 【修改点 4】: 图例加上黑色方框，并挪到横坐标下方
        # ncol=2 采用 2x2 布局，bbox_to_anchor 控制垂直偏移
        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5, -0.20),
                  ncol=2,
                  frameon=True,
                  edgecolor='black',
                  fancybox=False,  # 直角方框更显正式
                  fontsize=10)

        plt.tight_layout()

        # --- 高保真保存 ---
        pdf_path = f"{output_basename}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

        png_path = f"{output_basename}.png"
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

        print(f"图像保存成功！")
        print(f"识别到数据长度: {max_episodes} 轮")
        print(f"输出文件: {pdf_path} / {png_path}")

        plt.show()


# --- 运行执行 ---
viz = FinalReproductionTrainer()
# 请替换为您实际的 pkl 路径
viz.plot_and_save('/content/Anti_Jam_testing_0.1_random.pkl', 'Random_Testing_Curve')