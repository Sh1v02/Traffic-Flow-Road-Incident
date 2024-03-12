from src.Buffers.ReplayBuffer import ReplayBuffer

import torch


# TODO: GPU support
# A cyclic buffer allowing constant time complexity O(1)
class UniformExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, state_dims, action_dims, max_size=10000, actions_type=torch.float32):
        super().__init__(max_size)

        # Memory tensors
        self._states_memory = torch.zeros((max_size, state_dims))
        self._actions_memory = torch.zeros((max_size, action_dims), dtype=actions_type)
        self._next_states_memory = torch.zeros((max_size, state_dims))
        self._rewards_memory = torch.zeros((max_size, 1))
        self._dones_memory = torch.zeros((max_size, 1))

        self._new_element_pointer = 0
        self.size = 0

    # Stores the experience as tensors in their respective memory
    def add_experience(self, *args):
        state, action, reward, next_state, done = args
        self._states_memory[self._new_element_pointer] = state
        self._actions_memory[self._new_element_pointer] = torch.tensor(action)
        self._next_states_memory[self._new_element_pointer] = next_state
        self._rewards_memory[self._new_element_pointer] = torch.tensor(reward)
        self._dones_memory[self._new_element_pointer] = torch.tensor(done)

        # Cyclic buffer -> replace the oldest element by overwriting its position
        self._new_element_pointer = (self._new_element_pointer + 1) % self._max_size
        if self.size < self._max_size:
            self.size += 1

    # Returns a batch of experiences as tensors
    def sample_experience(self, batch_size=32):
        batch_indexes = torch.randint(0, self.size, (batch_size,))
        return (
            self._states_memory[batch_indexes],
            self._actions_memory[batch_indexes],
            self._rewards_memory[batch_indexes],
            self._next_states_memory[batch_indexes],
            self._dones_memory[batch_indexes]
        )
