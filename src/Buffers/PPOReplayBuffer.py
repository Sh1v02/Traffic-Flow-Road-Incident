import numpy as np
from src.Buffers.ReplayBuffer import ReplayBuffer


class PPOReplayBuffer(ReplayBuffer):
    def __init__(self):
        super().__init__()

        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.probabilities = []

    def add_experience(self, *args):
        state, action, value, reward, done, probability = args
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.probabilities.append(probability)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.probabilities = []

    # Return batches of size self.batch_size with random elements in each batch
    def sample_experience(self, batch_size=20):
        num_of_states = len(self.states)
        memory_indexes = np.arange(num_of_states)
        np.random.shuffle(memory_indexes)
        batches = [memory_indexes[i:i + batch_size] for i in range(0, num_of_states, batch_size)]

        return self.states, self.actions, self.values, self.rewards, self.dones, self.probabilities, batches
