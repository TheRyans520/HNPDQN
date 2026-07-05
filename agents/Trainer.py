import copy
import random
import pickle
import os
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt


class Trainer(object):
    """Runs games for given agents. Optionally will visualise and save the results"""

    def __init__(self, config, agents):
        self.config = config
        self.agents = agents
        self.agent_to_agent_group = self.create_agent_to_agent_group_dictionary()
        self.agent_to_color_group = self.create_agent_to_color_dictionary()
        self.results = None
        self.signals_result = None
        self.colors = ["red", "blue", "green", "orange", "yellow", "purple"]
        self.colour_ix = 0
        self.y_limits = None
        self.trained_agents = []

    def create_agent_to_agent_group_dictionary(self):
        """Creates a dictionary that maps an agent to their wider agent group"""
        agent_to_agent_group_dictionary = {
            "HNP_DQN": "DQN_Agents",
            "Poly_DQN": "DQN_Agents",  # Add this line to resolve the KeyError
            "MTN_DQN": "DQN_Agents",
            "MTN_DQN_Attn": "DQN_Agents",
            "MTN_DQN_Contrastive": "DQN_Agents",
            "MTN_DQN_Res": "DQN_Agents",
            "DQN": "DQN_Agents",
            "DQN-HER": "DQN_Agents",
            "DDQN": "DQN_Agents",
            "DDQN with Prioritised Replay": "DQN_Agents",
            "DQN with Fixed Q Targets": "DQN_Agents",
            "Duelling DQN": "DQN_Agents",
            "Dueling DDQN": "DQN_Agents"
        }
        return agent_to_agent_group_dictionary

    def create_agent_to_color_dictionary(self):
        """Creates a dictionary that maps an agent to a hex color (for plotting purposes)"""
        agent_to_color_dictionary = {
            "HNP_DQN": "#08BF5B",
            "Poly_DQN": "#D5F826",
            "MTN_DQN": "#F82652",
            "MTN_DQN_Attn": "#10A946",  # 深绿
            "MTN_DQN_Contrastive": "#A227B6",  # 深紫
            "MTN_DQN_Res": "#ABB627",  # 黑黄
            "DQN": "#0000FF",
            "DQN with Fixed Q Targets": "#1F618D",
            "DDQN": "#2980B9",
            "DDQN with Prioritised Replay": "#7FB3D5",
            "Dueling DDQN": "#22DAF3",
        }
        return agent_to_color_dictionary

    # =========================================================================================
    # 函数1：run_games_for_agents (修改：去掉了标题，并注释了loss图的调用)
    # =========================================================================================
    def run_games_for_agents(self):
        """Run a set of games for each agent and plot results together"""
        self.trained_agents = []  # <--- 添加这一行，确保每次调用该方法时重置已记录的智能体
        self.results = self.create_object_to_store_results()
        self.signals_result = self.create_object_to_store_results()

        for agent_number, agent_class in enumerate(self.agents):
            agent_name = agent_class.agent_name
            self.run_games_for_agent(agent_number + 1, agent_class)

        all_scores = []
        for agent_data in self.results.values():
            for run in agent_data:
                all_scores.extend(run[1])  # rolling_scores
        self.y_limits = [min(all_scores) - 1, max(all_scores) + 1]

        if self.config.visualise_overall_agent_results:
            fig, ax = plt.subplots(figsize=(7, 5))
            for agent_class in self.agents:
                agent_name = agent_class.agent_name
                agent_rolling_score_results = [results[1] for results in self.results[agent_name]]
                self.visualise_overall_agent_results(
                    agent_rolling_score_results,
                    agent_name,
                    show_mean_and_std_range=True,
                    y_limits=self.y_limits,
                    ax=ax
                )

            # 【要求4】: 去掉图顶部的标题
            # ax.set_title(self.environment_name, fontsize=14, weight="bold")
            ax.set_xlabel("Episode", fontsize=12)
            ax.set_ylabel("Average Return", fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(frameon=False, fontsize=10, loc="lower right")
            plt.tight_layout()

        if self.config.file_to_save_data_results:
            self.save_obj(self.results, self.config.file_to_save_data_results)
        if self.config.file_to_save_results_graph:
            print(f"Saving reward curve to: {self.config.file_to_save_results_graph}")
            plt.savefig(self.config.file_to_save_results_graph, bbox_inches="tight")
        # 显示奖励曲线图
        plt.show()

        return self.results

    def create_object_to_store_results(self):
        """Creates a dictionary that we will store the results in if it doesn't exist, otherwise it loads it up"""
        if self.config.overwrite_existing_results_file or not self.config.file_to_save_data_results or not os.path.isfile(
                self.config.file_to_save_data_results):
            results = {}
        else:
            results = self.load_obj(self.config.file_to_save_data_results)
        return results

    def run_games_for_agent(self, agent_number, agent_class):
        """Runs a set of games for a given agent, saving the results in self.results"""
        agent_results = []
        agent_name = agent_class.agent_name
        agent_group = self.agent_to_agent_group[agent_name]
        agent_round = 1
        for run in range(self.config.runs_per_agent):
            agent_config = copy.deepcopy(self.config)

            if self.environment_has_changeable_goals(
                    agent_config.environment) and self.agent_cant_handle_changeable_goals_without_flattening(
                    agent_name):
                print("Flattening changeable-goal environment for agent {}".format(agent_name))
                agent_config.environment = gym.wrappers.FlattenDictWrapper(agent_config.environment,
                                                                           dict_keys=["observation", "desired_goal"])

            base_seed = self.config.seed + run
            agent_config.seed = base_seed

            agent_config.hyperparameters = agent_config.hyperparameters[agent_group]
            print("AGENT NAME: {}".format(agent_name))
            print("\033[1m" + "{}.{}: {}".format(agent_number, agent_round, agent_name) + "\033[0m", flush=True)
            agent = agent_class(agent_config)
            self.trained_agents.append(agent)
            self.environment_name = agent.environment_title
            print(agent.hyperparameters)
            print("RANDOM SEED ", agent_config.seed)
            game_scores, rolling_scores, time_taken, game_signals = agent.run_n_episodes()

            # 修改这里：将 loss_history 放入结果列表中保存
            loss_history = getattr(agent, "loss_history", [])

            agent_results.append([
                game_scores,
                rolling_scores,
                len(rolling_scores),
                -1 * max(rolling_scores),
                time_taken,
                game_signals,
                loss_history  # 新增项：索引为 6
            ])

            print("Time taken: {}".format(time_taken), flush=True)
            self.print_two_empty_lines()

            # 这里是否重复添加，是bug吗？
            #agent_results.append(
                #[game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken, game_signals])
            if self.config.visualise_individual_results:
                self.visualise_overall_agent_results([rolling_scores], agent_name, show_each_run=True,
                                                     y_limits=self.y_limits)
                plt.show()
            agent_round += 1
        self.results[agent_name] = agent_results

    def environment_has_changeable_goals(self, env):
        return False

    def visualise_overall_agent_results(
            self, agent_results, agent_name,
            show_mean_and_std_range=False, show_each_run=False,
            color=None, ax=None, title=None, y_limits=None
    ):
        import matplotlib.ticker as ticker  # 必须导入定位器
        """
        学术风格绘图：均值 ± 标准差，半透明区间 (已优化横坐标与边框)
        """
        assert isinstance(agent_results, list), "agent_results must be a list of lists"

        if not ax:
            fig, ax = plt.subplots(figsize=(6, 4))
        if not color:
            color = self.agent_to_color_group.get(agent_name, "blue")

        if show_mean_and_std_range:
            mean_minus_x_std, mean_results, mean_plus_x_std = \
                self.get_mean_and_standard_deviation_difference_results(agent_results)
            x_vals = np.arange(len(mean_results))
            ax.plot(x_vals, mean_results, label=agent_name, color=color, linewidth=2)
            ax.fill_between(x_vals, mean_minus_x_std, mean_plus_x_std, alpha=0.2, color=color)
        else:
            for ix, result in enumerate(agent_results):
                x_vals = np.arange(len(result))
                ax.plot(x_vals, result, label=f"{agent_name}_{ix + 1}",
                        color=color, linewidth=1.5, alpha=0.7)

        # --- 核心优化 1: 强制封闭坐标轴 (与Loss图样式对齐) ---
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_edgecolor('black')

        # --- 核心优化 2: 自适应整数刻度 (解决Episode过多变拥挤的问题) ---
        # 这行替换掉你原来的 set_xticks 逻辑
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', integer=True))

        # --- 核心优化 3: 纵坐标去科学计数法 ---
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.get_yaxis().get_major_formatter().set_scientific(False)

        # 标签与网格
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Average Return", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)  # 降低网格透明度，更显高级

        if y_limits:
            ax.set_ylim(y_limits)

        ax.legend(frameon=False, fontsize=10, loc="lower right")  # 奖励图通常建议放右下角，避免遮挡上升曲线

        plt.tight_layout()

    def get_y_limits(self, results):
        """Extracts the minimum and maximum seen y_values from a set of results"""
        min_result = float("inf")
        max_result = float("-inf")
        for result in results:
            temp_max = np.max(result)
            temp_min = np.min(result)
            if temp_max > max_result:
                max_result = temp_max
            if temp_min < min_result:
                min_result = temp_min
        return min_result, max_result

    def get_next_color(self):
        """Gets the next color in list self.colors. If it gets to the end then it starts from beginning"""
        self.colour_ix += 1
        if self.colour_ix >= len(self.colors): self.colour_ix = 0
        color = self.colors[self.colour_ix]
        return color

    def get_mean_and_standard_deviation_difference_results(self, results):
        """From a list of lists of agent results it extracts the mean results and the mean results plus or minus
         some multiple of the standard deviation"""

        def get_results_at_a_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return results_at_a_time_step

        def get_standard_deviation_at_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return np.std(results_at_a_time_step)

        mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
        mean_minus_x_std = [
            mean_val - self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep)
            for
            timestep, mean_val in enumerate(mean_results)]
        mean_plus_x_std = [
            mean_val + self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep)
            for
            timestep, mean_val in enumerate(mean_results)]
        return mean_minus_x_std, mean_results, mean_plus_x_std

    def hide_spines(self, ax, spines_to_hide):
        """Hides splines on a matplotlib image"""
        for spine in spines_to_hide:
            ax.spines[spine].set_visible(False)

    def ignore_points_after_game_solved(self, mean_minus_x_std, mean_results, mean_plus_x_std):
        """Removes the datapoints after the mean result achieves the score required to solve the game"""
        for ix in range(len(mean_results)):
            if mean_results[ix] >= self.config.environment.get_score_to_win():
                break
        return mean_minus_x_std[:ix], mean_results[:ix], mean_plus_x_std[:ix]

    def draw_horizontal_line_with_label(self, ax, y_value, x_min, x_max, label):
        """Draws a dotted horizontal line on the given image at the given point and with the given label"""
        ax.hlines(y=y_value, xmin=x_min, xmax=x_max,
                  linewidth=2, color='k', linestyles='dotted', alpha=0.5)
        ax.text(x_max, y_value * 0.965, label)

    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")

    def save_obj(self, obj, name):
        """Saves given object as a pickle file"""
        if name[-4:] != ".pkl":
            name += ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        """Loads a pickle file object"""
        with open(name, 'rb') as f:
            return pickle.load(f)

    def visualise_preexisting_results(self, save_image_path=None, data_path=None, colors=None, show_image=True, ax=None,
                                      title=None, y_limits=None):
        """Visualises saved data results and then optionally saves the image"""
        if not data_path:
            preexisting_results = self.create_object_to_store_results()
        else:
            preexisting_results = self.load_obj(data_path)
        for ix, agent in enumerate(list(preexisting_results.keys())):
            agent_rolling_score_results = [results[1] for results in preexisting_results[agent]]
            if colors:
                color = colors[ix]
            else:
                color = None
            self.visualise_overall_agent_results(agent_rolling_score_results, agent, show_mean_and_std_range=True,
                                                 color=color, ax=ax, title=title, y_limits=y_limits)
        if save_image_path: plt.savefig(save_image_path, bbox_inches="tight")
        if show_image: plt.show()

    def visualise_set_of_preexisting_results(self, results_data_paths, save_image_path=None, show_image=True,
                                             plot_titles=None,
                                             y_limits=[None, None]):
        """Visualises a set of preexisting results on 1 plot by making subplots"""
        assert isinstance(results_data_paths, list), "all_results must be a list of data paths"

        num_figures = len(results_data_paths)
        col_width = 15
        row_height = 6

        if num_figures <= 2:
            fig, axes = plt.subplots(1, num_figures, figsize=(col_width, row_height))
        elif num_figures <= 4:
            fig, axes = plt.subplots(2, num_figures, figsize=(row_height, col_width))
        else:
            raise ValueError("Need to tell this method how to deal with more than 4 plots")
        for ax_ix in range(len(results_data_paths)):
            self.visualise_preexisting_results(show_image=False, data_path=results_data_paths[ax_ix], ax=axes[ax_ix],
                                               title=plot_titles[ax_ix], y_limits=y_limits[ax_ix])
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)

        if save_image_path: plt.savefig(save_image_path)  # , bbox_inches="tight")
        if show_image: plt.show()

    # ✅ 【新功能】已更新此函数
    # =========================================================================================
    # 函数3：visualise_loss_curves (修改：去掉了标题，修改了图例位置)
    # 如果您完全不需要loss图，可以不替换此函数。但为了完整性，这里提供了修改后的版本。
    # =========================================================================================
    def visualise_loss_curves(self, agents, save_path=None):
        import matplotlib.ticker as ticker
        import numpy as np

        # 1. 算法去重
        seen_names = set()
        unique_agents = []
        for a in agents:
            if hasattr(a, "loss_history") and len(a.loss_history) > 0:
                if a.agent_name not in seen_names:
                    unique_agents.append(a)
                    seen_names.add(a.agent_name)

        if len(unique_agents) == 0: return

        # 2. 统一纵坐标
        all_loss_values = [loss for a in unique_agents for loss in a.loss_history]
        global_max_loss = np.max(all_loss_values) if all_loss_values else 1.0

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, agent in enumerate(unique_agents):
            if i >= 4: break
            ax = axes[i]

            losses = np.array(agent.loss_history)
            episodes_axis = np.arange(1, len(losses) + 1)

            # 绘图
            ax.plot(episodes_axis, losses, linewidth=1.5,
                    color=self.agent_to_color_group.get(agent.agent_name, "blue"))

            # --- 风格：封闭坐标图 ---
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_edgecolor('black')

            # --- 纵坐标：取消 1e1 科学计数法 ---
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            y_formatter = ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)

            # --- 横坐标：核心优化！自适应整数刻度 ---
            # MaxNLocator(integer=True) 确保只显示整数
            # 它会自动根据 5 或 200 这种量级，自动挑选合适的间隔（如隔 1 显示或隔 50 显示）
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', integer=True))

            # 布局美化
            ax.set_title(f"Algorithm: {agent.agent_name}", fontsize=13, fontweight='bold', pad=20)
            ax.set_ylim([0, global_max_loss * 1.1])
            ax.set_xlabel("Episode", fontsize=11)
            ax.set_ylabel("Loss Value", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.3)

        # 删除多余子图
        for j in range(len(unique_agents), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=4.0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()