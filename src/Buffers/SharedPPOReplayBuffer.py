import numpy as np

from src.Buffers.ReplayBuffer import ReplayBuffer
from src.Utilities import multi_agent_settings


class SharedPPOReplayBuffer(ReplayBuffer):
    def __init__(self, global_states_buffer_required=False):
        super().__init__(multi_agent_settings.AGENT_COUNT)
        self.global_states_buffer_required = global_states_buffer_required

        self.local_states = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.actions = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.values = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.rewards = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.dones = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.probabilities = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        if self.global_states_buffer_required:
            self.global_states = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]

    def add_experience(self, *args):
        global_state = None
        if self.global_states_buffer_required:
            states, actions, values, rewards, dones, probabilities, global_state = args
        else:
            states, actions, values, rewards, dones, probabilities = args

        for agent_index in range(len(states)):
            self.local_states[agent_index].append(states[agent_index])
            self.actions[agent_index].append(actions[agent_index])
            self.rewards[agent_index].append(rewards[agent_index])
            self.values[agent_index].append(values[agent_index])
            self.dones[agent_index].append(dones[agent_index])
            self.probabilities[agent_index].append(probabilities[agent_index])
            if self.global_states_buffer_required:
                self.global_states[agent_index].append(global_state)

    def clear(self):
        self.local_states = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.actions = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.values = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.rewards = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.dones = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.probabilities = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        if self.global_states_buffer_required:
            self.global_states = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]

    # Return batches of size self.batch_size with random elements in each batch
    def sample_experience(self, batch_size=64):
        flattened_states = self.flatten_list(self.rewards)
        num_of_states = len(flattened_states)
        memory_indexes = np.arange(num_of_states)
        np.random.shuffle(memory_indexes)
        batches = [memory_indexes[i:i + batch_size] for i in range(0, num_of_states, batch_size)]

        return batches

    def get_buffer_contents(self):
        if self.global_states_buffer_required:
            return (self.local_states, self.actions, self.values, self.rewards, self.dones, self.probabilities,
                    self.global_states)

        return self.local_states, self.actions, self.values, self.rewards, self.dones, self.probabilities

    @staticmethod
    def flatten_list(list_to_flatten):
        return [value for sublist in list_to_flatten for value in sublist]
