import numpy as np
import torch

from src.Buffers.ReplayBuffer import ReplayBuffer
from src.Utilities import settings, multi_agent_settings
from src.Wrappers.GPUSupport import optimise, tensor


class QMIXPrioritisedExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, local_state_dims, global_state_dims, action_dims=1, max_size=10000):
        super().__init__(max_size=max_size)

        # Memory tensors
        self._local_states_memory = optimise(torch.zeros((max_size, local_state_dims)))
        self._global_states_memory = optimise(torch.zeros((max_size, global_state_dims)))
        self._actions_memory = optimise(torch.zeros((max_size, action_dims), dtype=torch.int32))
        self._next_states_memory = optimise(torch.zeros((max_size, local_state_dims)))
        self._next_global_states_memory = optimise(torch.zeros((max_size, global_state_dims)))
        self._rewards_memory = optimise(torch.zeros((max_size, 1)))
        self._dones_memory = optimise(torch.zeros((max_size, 1)))

        self._priorities = np.empty(0)
        self.alpha = settings.QMIX_PER_ALPHA
        self.initial_beta = self.beta = settings.QMIX_PER_BETA
        self.total_beta_steps = settings.TRAINING_STEPS
        self.end_beta = 1.0
        self.epsilon = settings.QMIX_PER_EPSILON

        self.size = 0

    def add_experience(self, *args):
        local_states, global_state, actions, rewards, next_states, next_global_state, dones = args
        # Regardless of the number of agents, the priority if being set to the max will be the same for them all
        priority = np.max(self._priorities, initial=1)

        if self.size < self._max_size:
            self._priorities = np.append(self._priorities, priority)
            for agent_index in range(multi_agent_settings.AGENT_COUNT):
                self._local_states_memory[self.size] = local_states[agent_index]
                self._global_states_memory[self.size] = global_state
                self._actions_memory[self.size] = tensor(actions[agent_index])
                self._rewards_memory[self.size] = tensor(rewards[agent_index])
                self._next_states_memory[self.size] = next_states[agent_index]
                self._next_global_states_memory[self.size] = next_global_state
                self._dones_memory[self.size] = tensor(dones[agent_index])
            self.size += 1
            return

        for agent_index in range(multi_agent_settings.AGENT_COUNT):
            # If full, replace the lowest prioritised experience with this new one
            lowest_priority_index = np.argmin(self._priorities)
            self._priorities[lowest_priority_index] = priority

            self._local_states_memory[lowest_priority_index] = local_states[agent_index]
            self._global_states_memory[lowest_priority_index] = global_state
            self._actions_memory[lowest_priority_index] = tensor(actions[agent_index])
            self._rewards_memory[lowest_priority_index] = tensor(rewards[agent_index])
            self._next_states_memory[lowest_priority_index] = next_states[agent_index]
            self._next_global_states_memory[lowest_priority_index] = next_global_state
            self._dones_memory[lowest_priority_index] = tensor(dones[agent_index])

    def get_probability_distribution_of_priorities(self):
        unscaled_priority_probs = np.power(self._priorities, self.alpha)
        return unscaled_priority_probs / np.sum(unscaled_priority_probs)

    # Normalised importance sampling weights that we scale the updates by, relative to beta
    def calculate_importance_sampling_weights(self, experience_probs):
        weights_before_normalising = np.power(self.size * experience_probs, -self.beta)
        return weights_before_normalising / np.max(weights_before_normalising)

    def sample_experience(self, batch_size=32):
        # Distribution of sampling each experience
        experience_probs = self.get_probability_distribution_of_priorities()
        # Sample indexes from the replay buffer based on the experience_probs distribution
        experience_indexes = np.random.choice(np.arange(self.size), size=batch_size, p=experience_probs, replace=False)
        weights = self.calculate_importance_sampling_weights(experience_probs[experience_indexes])

        return (
            self._local_states_memory[experience_indexes],
            self._global_states_memory[experience_indexes],
            self._actions_memory[experience_indexes].squeeze(1),
            self._rewards_memory[experience_indexes].squeeze(1),
            self._next_states_memory[experience_indexes],
            self._next_global_states_memory[experience_indexes],
            self._dones_memory[experience_indexes].squeeze(1),
            experience_indexes,
            weights
        )

    # New priority for sampled experience is td error + epsilon (to prevent 0)
    def update_priorities(self, indexes, td_errors):
        self._priorities[indexes] = np.abs(td_errors.detach().cpu().numpy()) + self.epsilon

    def linear_beta_anneal(self, step):
        self.beta = min(
            self.end_beta,
            self.initial_beta + (self.end_beta - self.initial_beta) * (step / self.total_beta_steps)
        )
