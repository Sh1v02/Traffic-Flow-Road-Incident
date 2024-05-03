import random

import numpy as np
import torch

from src.Agents.Agent import Agent
from src.Buffers import UniformExperienceReplayBuffer
from src.Models import MultiLayerPerceptron
from src.Utilities import settings


class DDQNAgent(Agent):
    def __init__(self, state_dims, action_dims, optimiser=torch.optim.Adam, loss=torch.nn.HuberLoss(),
                 replay_buffer=None):
        self.action_dims = action_dims
        self.gamma = settings.DDQN_DISCOUNT_FACTOR
        self.epsilon = settings.DDQN_EPSILON
        self.epsilon_decay = settings.DDQN_EPSILON_DECAY
        self.min_epsilon = settings.DDQN_MIN_EPSILON
        self.batch_size = settings.DDQN_BATCH_SIZE
        self.update_target_network_frequency = settings.DDQN_UPDATE_TARGET_NETWORK_FREQUENCY
        self.replay_buffer = UniformExperienceReplayBuffer(state_dims, actions_type=torch.int32,
                                                           max_size=settings.DDQN_REPLAY_BUFFER_SIZE) \
            if not replay_buffer else replay_buffer

        self.online_network = MultiLayerPerceptron(optimiser, loss, state_dims, action_dims,
                                                   optimiser_args={"lr": settings.DDQN_LR},
                                                   hidden_layer_dims=settings.DDQN_NETWORK_DIMS)
        self.target_network = self.online_network.deep_copy_network()

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
        return {
            "DDQN_NETWORK_DIMS": str(settings.DDQN_NETWORK_DIMS),
            "DDQN_DISCOUNT_FACTOR/GAMMA": str(settings.DDQN_DISCOUNT_FACTOR),
            "DDQN_LR": str(settings.DDQN_LR),
            "DDQN_EPSILON": str(settings.DDQN_EPSILON),
            "DDQN_EPSILON_DECAY": str(settings.DDQN_EPSILON_DECAY),
            "DDQN_MIN_EPSILON": str(settings.DDQN_MIN_EPSILON),
            "DDQN_BATCH_SIZE": str(settings.DDQN_BATCH_SIZE),
            "DDQN_REPLAY_BUFFER_SIZE": str(settings.DDQN_REPLAY_BUFFER_SIZE),
            "DDQN_UPDATE_TARGET_NETWORK_FREQUENCY": str(settings.DDQN_UPDATE_TARGET_NETWORK_FREQUENCY)
        }
