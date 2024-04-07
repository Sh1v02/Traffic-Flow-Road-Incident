import torch

from src.Buffers import UniformExperienceReplayBuffer
from src.Wrappers.GPUSupport import tensor


class SharedUniformExperienceReplayBuffer(UniformExperienceReplayBuffer):
    def __init__(self, state_dims, action_dims=1, max_size=10000, actions_type=torch.float32):
        super().__init__(state_dims, action_dims=action_dims, max_size=max_size, actions_type=actions_type)

    def add_experience(self, *args):
        states, actions, rewards, next_states, dones = args

        for agent_index in range(len(rewards)):
            self._states_memory[self._new_element_pointer] = states[agent_index]
            self._actions_memory[self._new_element_pointer] = tensor(actions[agent_index])
            self._next_states_memory[self._new_element_pointer] = next_states[agent_index]
            self._rewards_memory[self._new_element_pointer] = tensor(rewards[agent_index])
            self._dones_memory[self._new_element_pointer] = tensor(dones[agent_index])

            # Cyclic buffer -> replace the oldest element by overwriting its position
            self._new_element_pointer = (self._new_element_pointer + 1) % self._max_size
            if self.size < self._max_size:
                self.size += 1
