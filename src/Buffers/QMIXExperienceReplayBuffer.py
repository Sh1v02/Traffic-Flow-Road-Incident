import torch

from src.Buffers.ReplayBuffer import ReplayBuffer
from src.Utilities import multi_agent_settings
from src.Wrappers.GPUSupport import optimise, tensor


class QMIXExperienceReplayBuffer(ReplayBuffer):
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

        self._new_element_pointer = 0
        self.size = 0

    # Stores the experience as tensors in their respective memory
    def add_experience(self, *args):
        local_states, global_state, actions, rewards, next_states, next_global_state, dones = args
        for agent_index in range(multi_agent_settings.AGENT_COUNT):
            self._local_states_memory[self._new_element_pointer] = local_states[agent_index]
            self._global_states_memory[self._new_element_pointer] = global_state
            self._actions_memory[self._new_element_pointer] = tensor(actions[agent_index])
            self._rewards_memory[self._new_element_pointer] = tensor(rewards[agent_index])
            self._next_states_memory[self._new_element_pointer] = next_states[agent_index]
            self._next_global_states_memory[self._new_element_pointer] = next_global_state
            self._dones_memory[self._new_element_pointer] = tensor(dones[agent_index])

            # Cyclic buffer -> replace the oldest element by overwriting its position
            self._new_element_pointer = (self._new_element_pointer + 1) % self._max_size
            if self.size < self._max_size:
                self.size += 1

    # Returns a batch of experiences as tensors
    def sample_experience(self, batch_size=32):
        batch_indexes = torch.randint(0, self.size, (batch_size,))
        return (
            self._local_states_memory[batch_indexes],
            self._global_states_memory[batch_indexes],
            self._actions_memory[batch_indexes].squeeze(1),
            self._rewards_memory[batch_indexes].squeeze(1),
            self._next_states_memory[batch_indexes],
            self._next_global_states_memory[batch_indexes],
            self._dones_memory[batch_indexes].squeeze(1),
            None,
            None
        )

    def update_priorities(self, *args):
        return

    def linear_beta_anneal(self, *args):
        return
