import numpy as np
from src.Buffers.ReplayBuffer import ReplayBuffer


class PPOReplayBuffer(ReplayBuffer):
    def __init__(self):
        super().__init__()

        self.states = np.empty(0)
        self.actions = np.empty(0)
        self.values = np.empty(0)
        self.rewards = np.empty(0)
        self.dones = np.empty(0)
        self.probabilities = np.empty(0)

    def add_experience(self, experience):
        state, action, value, reward, done, probability = experience
        np.append(self.states, state)
        np.append(self.actions, action)
        np.append(self.values, value)
        np.append(self.rewards, reward)
        np.append(self.dones, done)
        np.append(self.probabilities, probability)

    def clear(self):
        self.states = np.empty(0)
        self.actions = np.empty(0)
        self.values = np.empty(0)
        self.rewards = np.empty(0)
        self.dones = np.empty(0)
        self.probabilities = np.empty(0)

    # Return batches of size self.batch_size with random elements in each batch
    def sample_experience(self, batch_size=20):
        num_of_states = len(self.states)
        memory_indexes = np.arange(num_of_states)
        np.random.shuffle(memory_indexes)
        batches = [memory_indexes[i:i + batch_size] for i in range(0, num_of_states, batch_size)]

        return self.states, self.actions, self.values, self.rewards, self.dones, self.probabilities, batches
