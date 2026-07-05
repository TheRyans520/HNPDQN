import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import Counter

from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer


class PolynomialLayer(nn.Module):
    """多项式展开层 + LayerNorm，稳定不同阶项之间的数值尺度"""
    def __init__(self, hidden_dim, poly_power=2):
        super().__init__()
        self.poly_power = poly_power
        self.norm = nn.LayerNorm(hidden_dim * poly_power)

    def forward(self, h):
        expanded = [h]
        for p in range(2, self.poly_power + 1):
            expanded.append(h ** p)
        poly_features = torch.cat(expanded, dim=1)
        poly_features = self.norm(poly_features)
        return poly_features


class HNP_DQN_Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=[128, 64], poly_power=3):
        super().__init__()
        # 特征提取层（加入 BatchNorm 稳定训练）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.ReLU(),
        )
        # 多项式展开层 + LayerNorm
        self.poly_layer = PolynomialLayer(hidden_units[1], poly_power)
        # 多项式输出后进一步线性层
        self.post_poly = nn.Sequential(
            nn.Linear(hidden_units[1] * poly_power, 128),
            nn.ReLU(),
        )
        # Dueling 分支
        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, output_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        poly_h = self.poly_layer(h)
        feat = self.post_poly(poly_h)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class HNP_DQN(Base_Agent):
    agent_name = "HNP_DQN"

    def __init__(self, config):
        super().__init__(config)
        hp = self.hyperparameters

        self.memory = Replay_Buffer(
            hp["buffer_size"], hp["batch_size"], config.seed, self.device
        )

        # 从 config 读取 poly_degree（默认为 2，即二阶多项式展开：h, h²）
        poly_degree = hp.get("poly_degree", 2)
        # 从 config 读取 hidden_units（与 DQN_Agents 的 linear_hidden_units 保持一致）
        hidden_units = hp.get("linear_hidden_units", [128, 64])

        self.q_network_local = HNP_DQN_Network(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_units=hidden_units,
            poly_power=poly_degree,
        ).to(self.device)

        self.q_network_target = HNP_DQN_Network(
            input_dim=self.state_size,
            output_dim=self.action_size,
            hidden_units=hidden_units,
            poly_power=poly_degree,
        ).to(self.device)

        # 初始化目标网络权重为本地网络权重
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target, tau=1.0)

        self.q_network_optimizer = optim.Adam(
            self.q_network_local.parameters(), lr=hp["learning_rate"], eps=1e-4
        )
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

        self.loss_history = []

    def reset_game(self):
        super().reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self):
        # 每个 episode 单独记录 loss
        episode_losses = []

        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)

            if self.config.training:
                if self.time_for_q_network_to_learn():
                    for _ in range(self.hyperparameters.get("learning_iterations", 1)):
                        loss_value = self.learn()  # 接收返回值
                        if loss_value is not None:
                            episode_losses.append(loss_value)

                self.save_experience()

            self.state = self.next_state
            self.global_step_number += 1

        # 一个 episode 结束后，只记录一次平均 loss
        if len(episode_losses) > 0:
            self.loss_history.append(np.mean(episode_losses))
        else:
            self.loss_history.append(0)

        self.episode_number += 1

    def pick_action(self, state=None):
        if state is None:
            state = self.state
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        if not self.config.training:
            self.q_network_local = self.locally_load_policy()
        self.q_network_local.eval()
        with torch.no_grad():
            q_values = self.q_network_local(state)
        if self.config.training:
            self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes(
            {
                "action_values": q_values,
                "turn_off_exploration": self.turn_off_exploration,
                "episode_number": self.episode_number,
            }
        )
        self.logger.info(f"Q values {q_values} -- Action chosen {action}")
        return action

    def learn(self, experiences=None):
        if experiences is None:
            states, actions, rewards, next_states, dones = self.sample_experiences()
        else:
            states, actions, rewards, next_states, dones = experiences

        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions]
        self.logger.info(f"Action counts {Counter(actions_list)}")

        self.q_network_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network_local.parameters(),
            max_norm=self.hyperparameters.get("gradient_clipping_norm", 0.7)
        )
        self.q_network_optimizer.step()

        self.soft_update_of_target_network(
            self.q_network_local,
            self.q_network_target,
            self.hyperparameters.get("tau", 0.005)
        )

        # 返回 loss 数值，而不是直接记录
        return loss.item()

    def compute_loss(self, states, next_states, rewards, actions, dones):
        # 1️⃣ 计算 Q_expected
        Q_expected = self.q_network_local(states).gather(1, actions.long())

        # 2️⃣ Double DQN：本地网络选动作，目标网络估值
        with torch.no_grad():
            Q_targets_next = self.compute_q_values_for_next_states(next_states)
            Q_targets = rewards + self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones)

        # 3️⃣ MSE Loss（所有模型统一使用）
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_values_for_next_states(self, next_states):
        """Double DQN: 本地网络选择最优动作，目标网络评估该动作的 Q 值"""
        max_action_indexes = self.q_network_local(next_states).detach().argmax(1)
        Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def locally_save_policy(self):
        torch.save(
            self.q_network_local.state_dict(),
            f"{self.config.models_dir}/{self.agent_name}_network.pt",
        )

    def locally_load_policy(self):
        self.q_network_local.load_state_dict(
            torch.load(f"{self.config.models_dir}/{self.agent_name}_network.pt")
        )
        return self.q_network_local

    def time_for_q_network_to_learn(self):
        return (
            self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0
            and self.enough_experiences_to_learn_from()
        )

    def sample_experiences(self):
        return self.memory.sample()
