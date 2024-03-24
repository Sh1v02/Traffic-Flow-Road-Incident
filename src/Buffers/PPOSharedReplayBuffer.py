import numpy as np
from src.Buffers.ReplayBuffer import ReplayBuffer
from src.Utilities import multi_agent_settings


class PPOSharedReplayBuffer(ReplayBuffer):
    def __init__(self):
        super().__init__(multi_agent_settings.AGENT_COUNT)

        self.states = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.actions = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.values = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.rewards = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.dones = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.probabilities = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]

    def add_experience(self, *args):
        states, actions, values, rewards, dones, probabilities = args
        for agent_index in range(len(states)):
            self.states[agent_index].append(states[agent_index])
            self.actions[agent_index].append(actions[agent_index])
            self.rewards[agent_index].append(rewards[agent_index])
            self.values[agent_index].append(values[agent_index])
            self.dones[agent_index].append(dones[agent_index])
            self.probabilities[agent_index].append(probabilities[agent_index])

    def clear(self):
        self.states = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.actions = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.values = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.rewards = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.dones = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]
        self.probabilities = [[] for _ in range(multi_agent_settings.AGENT_COUNT)]

    # Return batches of size self.batch_size with random elements in each batch
    def sample_experience(self, batch_size=20):
        flattened_states = self.flatten_list(self.states)
        num_of_states = len(flattened_states)
        memory_indexes = np.arange(num_of_states)
        np.random.shuffle(memory_indexes)
        batches = [memory_indexes[i:i + batch_size] for i in range(0, num_of_states, batch_size)]

        return (flattened_states, self.flatten_list(self.actions), self.flatten_list(self.values),
                self.flatten_list(self.rewards), self.flatten_list(self.dones), self.flatten_list(self.probabilities),
                batches)

    @staticmethod
    def flatten_list(list_to_flatten):
        return [value for sublist in list_to_flatten for value in sublist]
