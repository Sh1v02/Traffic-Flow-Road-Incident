import numpy as np
from src.Buffers.ReplayBuffer import ReplayBuffer


class PPOReplayBuffer(ReplayBuffer):
    def __init__(self, num_agents_using_buffer=1, global_states_buffer_required=False):
        super().__init__(num_agents_using_buffer)
        self.global_states_buffer_required = global_states_buffer_required

        self.local_states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.probabilities = []
        if self.global_states_buffer_required:
            self.global_states = []

    def add_experience(self, *args):
        global_state = None
        if self.global_states_buffer_required:
            state, action, value, reward, done, probability, global_state = args
        else:
            state, action, value, reward, done, probability = args
        self.local_states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.probabilities.append(probability)
        if self.global_states_buffer_required:
            self.global_states.append(global_state)

    def clear(self):
        self.local_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.probabilities = []
        if self.global_states_buffer_required:
            self.global_states = []

    # Return batches of size self.batch_size with random elements in each batch
    def sample_experience(self, batch_size=64):
        num_of_states = len(self.local_states)
        memory_indexes = np.arange(num_of_states)
        np.random.shuffle(memory_indexes)
        batches = [memory_indexes[i:i + batch_size] for i in range(0, num_of_states, batch_size)]

        return batches

    def get_buffer_contents(self):
        if self.global_states_buffer_required:
            return (self.local_states, self.actions, self.values, self.rewards, self.dones, self.probabilities,
                    self.global_states)

        return self.local_states, self.actions, self.values, self.rewards, self.dones, self.probabilities
