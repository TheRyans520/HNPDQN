# 修改后的 DQN，增加了 loss_history 支持
from collections import Counter
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer


class DQN(Base_Agent):
    agent_name = "DQN"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                                    config.seed, self.device)
        self.q_network_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size)
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(config)

        # ✅ 新增：loss 记录
        self.loss_history = []

    def reset_game(self):
        super(DQN, self).reset_game()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.q_network_optimizer)

    def step(self):
        episode_losses = []

        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)

            if self.config.training:
                if self.time_for_q_network_to_learn():
                    for _ in range(self.hyperparameters["learning_iterations"]):
                        loss_value = self.learn()
                        if loss_value is not None:
                            episode_losses.append(loss_value)

                self.save_experience()

            self.state = self.next_state
            self.global_step_number += 1

        if len(episode_losses) > 0:
            self.loss_history.append(np.mean(episode_losses))
        else:
            self.loss_history.append(0)

        self.episode_number += 1

    def pick_action(self, state=None):
        if state is None: state = self.state
        if isinstance(state, np.int64) or isinstance(state, int): state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2: state = state.unsqueeze(0)
        if not self.config.training:
            self.q_network_local = self.locally_load_policy()
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        if self.config.training:
            self.q_network_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action_values": action_values,
                                                                                    "turn_off_exploration": self.turn_off_exploration,
                                                                                    "episode_number": self.episode_number})
        self.logger.info("Q values {} -- Action chosen {}".format(action_values, action))
        return action

    def learn(self, experiences=None):
        if experiences is None:
            states, actions, rewards, next_states, dones = self.sample_experiences()
        else:
            states, actions, rewards, next_states, dones = experiences
        loss = self.compute_loss(states, next_states, rewards, actions, dones)

        actions_list = [action_X.item() for action_X in actions]
        self.logger.info("Action counts {}".format(Counter(actions_list)))
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss,
                                    self.hyperparameters["gradient_clipping_norm"])
        return loss.item()

    def compute_loss(self, states, next_states, rewards, actions, dones):
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        Q_targets_next = self.q_network_local(next_states).detach().max(1)[0].unsqueeze(1)
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        Q_targets_current = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        Q_expected = self.q_network_local(states).gather(1, actions.long())
        return Q_expected

    def locally_save_policy(self):
        torch.save(self.q_network_local.state_dict(),
                   "{}/{}_network.pt".format(self.config.models_dir, self.agent_name))

    def locally_load_policy(self):
        filename = f'{self.config.models_dir}/{self.agent_name}_network.pt'
        saved_q_network_local = self.q_network_local
        saved_q_network_local.load_state_dict(torch.load(filename))
        return saved_q_network_local

    def time_for_q_network_to_learn(self):
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones
