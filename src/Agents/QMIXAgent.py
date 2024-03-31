import random

import numpy as np
import torch

from src.Agents.Agent import Agent
from src.Buffers.QMIXExperienceReplayBuffer import QMIXExperienceReplayBuffer
from src.Models.QMIX.QMIX import QMIX
from src.Utilities import multi_agent_settings, settings
from src.Wrappers.GPUSupport import optimise


class QMIXAgent(Agent):
    def __init__(self, optimiser, local_state_dims, global_state_dims, action_dims,
                 hidden_layer_dims=64, optimiser_args=None):
        self.qmix = QMIX(local_state_dims, global_state_dims, action_dims, hidden_layer_dims=hidden_layer_dims)
        self.action_dims = action_dims

        self.gamma = settings.QMIX_DISCOUNT_FACTOR
        self.batch_size = settings.QMIX_BATCH_SIZE
        self.epsilon = settings.QMIX_EPSILON
        self.epsilon_decay = settings.QMIX_EPSILON_DECAY
        self.min_epsilon = settings.QMIX_MIN_EPSILON

        self.replay_buffer = QMIXExperienceReplayBuffer(local_state_dims, global_state_dims)

        optimiser_args = optimiser_args if optimiser_args else {}
        self.optimiser = optimiser(self.qmix.parameters(), **optimiser_args)

        self.steps = 0

    def store_experience_in_replay_buffer(self, *args):
        self.replay_buffer.add_experience(*args)

    def get_action(self, local_states: torch.Tensor, training=True):
        random_action = training and random.uniform(0, 1) <= self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        if random_action:
            actions = tuple(random.randint(0, self.action_dims - 1) for _ in range(multi_agent_settings.AGENT_COUNT))
            return actions

        agent_q_values = self.qmix.get_q_values(local_states)
        actions = tuple(torch.argmax(q_value).item() for q_value in agent_q_values)

        return actions

    def learn(self):
        self.steps += 1
        if self.steps < self.batch_size:
            return

        # TODO: Change the batch size
        local_states, global_states, actions, rewards, next_local_states, next_global_states, dones = \
            self.replay_buffer.sample_experience(batch_size=self.batch_size)

        # Get q-values based on the actions sampled
        current_q_values = torch.zeros(self.batch_size, multi_agent_settings.AGENT_COUNT, self.action_dims)
        for batch_index in range(len(local_states)):
            current_q_values[batch_index] = torch.stack(
                [
                    self.qmix.online_agent_networks[i](local_states[batch_index]) for i in
                    range(len(self.qmix.target_agent_networks))
                ]
            )

        current_single_q_values = torch.empty(0)
        for i in range(multi_agent_settings.AGENT_COUNT):
            if current_single_q_values.numel() == 0:
                current_single_q_values = current_q_values[:, i, :][np.arange(self.batch_size), actions].unsqueeze(1)
            else:
                current_single_q_values = torch.cat((current_single_q_values, current_q_values[:, i, :][
                    np.arange(self.batch_size), actions].unsqueeze(1)), dim=1)
        # Get the current_q_totals
        current_q_totals = self.qmix.online_mixer_network(current_single_q_values, global_states)

        # Change target_q_totals shape: (self.batch_size, 1, 1) -> (self.batch_size)
        current_q_totals = current_q_totals.squeeze(1).squeeze(1)

        # Get target q_values based on the max action from the online_agent networks
        with torch.no_grad():
            target_q_values = optimise(torch.zeros(self.batch_size, multi_agent_settings.AGENT_COUNT, self.action_dims))
            for batch_index in range(len(local_states)):
                target_q_values[batch_index] = torch.stack(
                    [
                        self.qmix.target_agent_networks[i](local_states[batch_index]) for i in
                        range(len(self.qmix.target_agent_networks))
                    ]
                )

            actions_for_next_state_max_q_values = [
                self.qmix.online_agent_networks[i](next_local_states).max(dim=1)[1] for i in
                range(multi_agent_settings.AGENT_COUNT)
            ]

            target_single_q_values = optimise(torch.empty(0))
            for i in range(len(actions_for_next_state_max_q_values)):
                if target_single_q_values.numel() == 0:
                    target_single_q_values = target_q_values[:, i, :][
                        np.arange(self.batch_size), actions_for_next_state_max_q_values[i]].unsqueeze(1)
                else:
                    target_single_q_values = torch.cat((target_single_q_values, target_q_values[:, i, :][
                        np.arange(self.batch_size), actions_for_next_state_max_q_values[i]].unsqueeze(1)), dim=1)

        # Now take the q values that match the online networks best action in the next_local_state
        # Pass all of these into the target_mixer_network to get the target_q_totals
        target_q_totals = self.qmix.target_mixer_network(target_single_q_values, global_states)

        # Change target_q_totals shape: (self.batch_size, 1, 1) -> (self.batch_size)
        target_q_totals = target_q_totals.squeeze(1).squeeze(1)
        targets = rewards + (self.gamma * (1 - dones) * target_q_totals)

        self.optimiser.zero_grad()
        loss = self.qmix.loss(current_q_totals, targets)
        loss.backward()
        self.optimiser.step()

        if settings.QMIX_SOFT_UPDATE:
            self.qmix.update_target_networks(tau=settings.QMIX_SOFT_UPDATE_TAU)
        elif self.steps % settings.QMIX_HARD_UPDATE_NETWORKS_FREQUENCY == 0:
            self.qmix.update_target_networks()

    def get_agent_specific_config(self):
        return {
            "QMIX_DISCOUNT_FACTOR": str(settings.QMIX_DISCOUNT_FACTOR),
            "QMIX_NETWORK_DIMS": str(settings.QMIX_NETWORK_DIMS),
            "QMIX_SOFT_UPDATE": str(settings.QMIX_SOFT_UPDATE),
            "QMIX_SOFT_UPDATE_TAU": str(settings.QMIX_SOFT_UPDATE_TAU),
            "QMIX_HARD_UPDATE_NETWORKS_FREQUENCY": str(settings.QMIX_HARD_UPDATE_NETWORKS_FREQUENCY),
            "QMIX_BATCH_SIZE": str(settings.QMIX_BATCH_SIZE),
            "QMIX_EPSILON": str(settings.QMIX_EPSILON),
            "QMIX_EPSILON_DECAY": str(settings.QMIX_EPSILON_DECAY),
            "QMIX_MIN_EPSILON": str(settings.QMIX_MIN_EPSILON),
            "QMIX_LR": str(settings.QMIX_LR),
        }
