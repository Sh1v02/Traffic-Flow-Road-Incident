import random
from copy import deepcopy

import numpy as np
import torch

from src.Agents.Agent import Agent
from src.Models import MultiLayerPerceptron
from src.Buffers import UniformExperienceReplayBuffer
from src.Wrappers.GPUSupport import optimise


class DDQNAgent(Agent):
    def __init__(self, state_dims, action_dims, optimiser=torch.optim.Adam, loss=torch.nn.HuberLoss(),
                 lr=0.003, gamma=0.99, epsilon=1.0, min_epsilon=0.01, epsilon_decay=0.99,
                 update_target_network_frequency=1000, batch_size=32):
        self.action_dims = action_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.replay_buffer = UniformExperienceReplayBuffer(state_dims, 1, 10000, actions_type=torch.int32)

        self.online_network = MultiLayerPerceptron(optimiser, loss, state_dims, action_dims,
                                                   optimiser_args={"lr": lr}, hidden_layer_dims=[256, 256])
        self.target_network = optimise(deepcopy(self.online_network))

        self.update_target_network_frequency = update_target_network_frequency
        self.steps = 0

    # epsilon-greedy action selection
    def get_action(self, state: torch.Tensor, training=True):
        random_action = training and random.uniform(0, 1) <= self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        action = torch.max(self.online_network(state), dim=0)[1].item() if not random_action \
            else random.randint(0, self.action_dims - 1)

        return action

    def store_experience_in_replay_buffer(self, *args):
        self.replay_buffer.add_experience(*args)

    def update_target_network_parameters(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        self.steps += 1
        if self.replay_buffer.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_experience(self.batch_size)
        predicted_q_values = self.online_network(states)[np.arange(self.batch_size), actions.view(-1)]
        temp_target_q_values = self.target_network(next_states)[
            np.arange(self.batch_size), self.online_network(next_states).max(dim=1)[1]]
        target_q_values = rewards.view(-1) + (self.gamma * temp_target_q_values * (1 - dones.view(-1)))

        self.online_network.update(predicted_q_values, target_q_values)

        if self.steps % self.update_target_network_frequency == 0:
            self.update_target_network_parameters()

    def get_agent_specific_config(self):
        return {}
