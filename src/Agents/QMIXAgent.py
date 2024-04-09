import random

import numpy as np
import torch

from src.Agents.Agent import Agent
from src.Buffers.QMIXExperienceReplayBuffer import QMIXExperienceReplayBuffer
from src.Buffers.QMIXPrioritisedExperienceReplayBuffer import QMIXPrioritisedExperienceReplayBuffer
from src.Models.QMIX.QMIX import QMIX
from src.Utilities import multi_agent_settings, settings
from src.Wrappers.GPUSupport import optimise, tensor


class QMIXAgent(Agent):
    def __init__(self, optimiser, local_state_dims, global_state_dims, action_dims, optimiser_args=None):
        self.qmix = QMIX(local_state_dims, global_state_dims, action_dims)
        self.action_dims = action_dims

        self.gamma = settings.QMIX_DISCOUNT_FACTOR
        self.batch_size = settings.QMIX_BATCH_SIZE
        self.epsilon = settings.QMIX_EPSILON
        self.epsilon_decay = settings.QMIX_EPSILON_DECAY
        self.min_epsilon = settings.QMIX_MIN_EPSILON

        self.replay_buffer = QMIXExperienceReplayBuffer(local_state_dims, global_state_dims,
                                                        max_size=settings.QMIX_REPLAY_BUFFER_SIZE) \
            if not settings.QMIX_PER else (
            QMIXPrioritisedExperienceReplayBuffer(local_state_dims, global_state_dims,
                                                  max_size=settings.QMIX_REPLAY_BUFFER_SIZE))

        self.eval_parameters = (list(self.qmix.online_agent_networks.parameters()) +
                                list(self.qmix.online_mixer_network.parameters()))
        optimiser_args = optimiser_args if optimiser_args else {}
        self.optimiser = optimiser(self.eval_parameters, **optimiser_args)

        # LR decay setup
        if settings.QMIX_LR[3]:
            self.lr_decay_steps = settings.QMIX_LR[2]
            self.lr_starting_value = settings.QMIX_LR[0]
            self.lr_decay_factor = settings.QMIX_LR[1] / settings.QMIX_LR[0]

        self.steps = 0

    def store_experience_in_replay_buffer(self, *args):
        self.replay_buffer.add_experience(*args)

    def get_action(self, local_states: torch.Tensor, training=True):
        random_action = training and random.uniform(0, 1) <= self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon) if training else self.epsilon

        if random_action:
            actions = tuple(random.randint(0, self.action_dims - 1) for _ in range(multi_agent_settings.AGENT_COUNT))
            return actions

        agent_q_values = self.qmix.get_q_values(local_states)
        actions = tuple(torch.argmax(q_value).item() for q_value in agent_q_values)

        return actions

    def learn(self, steps=None):
        self.steps = steps if steps else self.steps + 1
        if self.replay_buffer.size < self.batch_size:
            return

        (local_states, global_states, actions, rewards, next_local_states, next_global_states, dones,
         experience_indexes, imp_weights) = self.replay_buffer.sample_experience(batch_size=self.batch_size)

        if self.qmix.shared_agent_net:
            current_single_q_values, target_single_q_values = self._learn_shared_agents(local_states, actions,
                                                                                        next_local_states)
        else:
            current_single_q_values, target_single_q_values = self._learn_unshared_agents(local_states, actions,
                                                                                          next_local_states)

        # Get the current_q_totals and target_q_totals
        if settings.AGENT_TYPE.lower() == "qmix":
            current_q_totals = self.qmix.online_mixer_network(current_single_q_values, global_states)
            target_q_totals = self.qmix.target_mixer_network(target_single_q_values, next_global_states)
        else:
            current_q_totals = self.qmix.online_mixer_network(current_single_q_values)
            target_q_totals = self.qmix.target_mixer_network(target_single_q_values)

        targets = rewards + (self.gamma * (1 - dones) * target_q_totals)

        if settings.QMIX_PER:
            self.replay_buffer.update_priorities(experience_indexes, targets - current_q_totals)
            self.replay_buffer.linear_beta_anneal(self.steps)

        self.optimiser.zero_grad()
        loss = self.qmix.loss(current_q_totals, targets)
        # Apply any importance weights to our loss (prioritised experience replay)
        if imp_weights is not None:
            loss *= tensor(imp_weights)
        loss = torch.mean(loss)
        loss.backward()
        if settings.QMIX_GRADIENT_CLIP:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimiser.step()

        if settings.QMIX_SOFT_UPDATE:
            self.qmix.update_target_networks(tau=settings.QMIX_SOFT_UPDATE_TAU)
        elif self.steps % settings.QMIX_HARD_UPDATE_NETWORKS_FREQUENCY == 0:
            self.qmix.update_target_networks()

        if settings.QMIX_LR[3]:
            self.decay_lr()

    def _learn_shared_agents(self, local_states, actions, next_local_states):

        current_q_values = self.qmix.online_agent_networks(local_states).unsqueeze(1)

        # Get q-values based on the actions sampled
        current_single_q_values = current_q_values[torch.arange(self.batch_size), :, actions]

        # Get target q_values based on the max action from the online_agent networks, for the next local states
        with torch.no_grad():
            target_q_values = self.qmix.target_agent_networks(next_local_states).unsqueeze(1)

            actions_for_next_state_max_q_values = self.qmix.online_agent_networks(next_local_states).max(dim=1)[1]

            target_single_q_values = target_q_values[
                                     torch.arange(self.batch_size), :, actions_for_next_state_max_q_values]

        return current_single_q_values, target_single_q_values

    def _learn_unshared_agents(self, local_states, actions, next_local_states):
        current_q_values = torch.concat(tuple(self.qmix.online_agent_networks[i](local_states).unsqueeze(1) for i in
                                              range(multi_agent_settings.AGENT_COUNT)), dim=1)

        # Get q-values based on the actions sampled
        current_single_q_values = current_q_values[torch.arange(self.batch_size), :, actions]

        # Get target q_values based on the max action from the online_agent networks, for the next local states
        with torch.no_grad():
            target_q_values = torch.concat(
                tuple(self.qmix.target_agent_networks[i](next_local_states).unsqueeze(1) for i in
                      range(multi_agent_settings.AGENT_COUNT)), dim=1)

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

        return current_single_q_values, target_single_q_values

    def decay_lr(self):
        # Don't decay past the max number of steps to decay
        if self.steps > self.lr_decay_steps:
            return

        new_lr = self.lr_starting_value * (self.lr_decay_factor ** (self.steps / self.lr_decay_steps))
        for p in self.optimiser.param_groups:
            p["lr"] = new_lr

    def get_agent_specific_config(self):
        if settings.AGENT_TYPE.lower() == "vdn":
            config = {
                "VDN_AGENT_NETWORKS_SHARED": str(settings.QMIX_AGENT_NETWORKS_SHARED),
                "VDN_DISCOUNT_FACTOR": str(settings.QMIX_DISCOUNT_FACTOR),
                "VDN_LR": str(settings.QMIX_LR),
                "VDN_GRADIENT_CLIP": str(settings.QMIX_GRADIENT_CLIP),
                "VDN_AGENT_NETWORK_DIMS": str(settings.QMIX_AGENT_NETWORK_DIMS),
                "VDN_SOFT_UPDATE": str(settings.QMIX_SOFT_UPDATE),
                "VDN_SOFT_UPDATE_TAU": str(settings.QMIX_SOFT_UPDATE_TAU),
                "VDN_HARD_UPDATE_NETWORKS_FREQUENCY": str(settings.QMIX_HARD_UPDATE_NETWORKS_FREQUENCY),
                "VDN_REPLAY_BUFFER_SIZE": str(settings.QMIX_REPLAY_BUFFER_SIZE),
                "VDN_BATCH_SIZE": str(settings.QMIX_BATCH_SIZE),
                "VDN_EPSILON": str(settings.QMIX_EPSILON),
                "VDN_EPSILON_DECAY": str(settings.QMIX_EPSILON_DECAY),
                "VDN_MIN_EPSILON": str(settings.QMIX_MIN_EPSILON),
                "VDN_LEARN_PER_EPISODE": str(settings.QMIX_LEARN_PER_EPISODE),
                "PRIORITISED_EXPERIENCE_REPLAY": str(settings.QMIX_PER)
            }
        else:
            config = {
                "QMIX_AGENT_NETWORKS_SHARED": str(settings.QMIX_AGENT_NETWORKS_SHARED),
                "QMIX_DISCOUNT_FACTOR": str(settings.QMIX_DISCOUNT_FACTOR),
                "QMIX_LR": str(settings.QMIX_LR),
                "QMIX_GRADIENT_CLIP": str(settings.QMIX_GRADIENT_CLIP),
                "QMIX_AGENT_NETWORK_DIMS": str(settings.QMIX_AGENT_NETWORK_DIMS),
                "QMIX_HYPER_NETWORK_LAYERS": str(settings.QMIX_HYPER_NETWORK_LAYERS),
                "QMIX_HYPER_NETWORK_DIMS": str(settings.QMIX_HYPER_NETWORK_DIMS),
                "QMIX_MIXER_NETWORK_DIMS": str(settings.QMIX_MIXER_NETWORK_DIMS),
                "QMIX_SOFT_UPDATE": str(settings.QMIX_SOFT_UPDATE),
                "QMIX_SOFT_UPDATE_TAU": str(settings.QMIX_SOFT_UPDATE_TAU),
                "QMIX_HARD_UPDATE_NETWORKS_FREQUENCY": str(settings.QMIX_HARD_UPDATE_NETWORKS_FREQUENCY),
                "QMIX_REPLAY_BUFFER_SIZE": str(settings.QMIX_REPLAY_BUFFER_SIZE),
                "QMIX_BATCH_SIZE": str(settings.QMIX_BATCH_SIZE),
                "QMIX_EPSILON": str(settings.QMIX_EPSILON),
                "QMIX_EPSILON_DECAY": str(settings.QMIX_EPSILON_DECAY),
                "QMIX_MIN_EPSILON": str(settings.QMIX_MIN_EPSILON),
                "QMIX_LEARN_PER_EPISODE": str(settings.QMIX_LEARN_PER_EPISODE),
                "PRIORITISED_EXPERIENCE_REPLAY": str(settings.QMIX_PER)
            }

        if settings.QMIX_PER:
            config.update(
                {
                    "PRIORITISED_EXPERIENCE_REPLAY_ALPHA": settings.QMIX_PER_ALPHA,
                    "PRIORITISED_EXPERIENCE_REPLAY_BETA": settings.QMIX_PER_BETA,
                    "PRIORITISED_EXPERIENCE_REPLAY_EPSILON": settings.QMIX_PER_EPSILON
                }
            )
        return config
