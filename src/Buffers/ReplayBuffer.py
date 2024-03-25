from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    def __init__(self, num_agents_using_buffer=1, max_size=10_000):
        self._max_size = max_size
        self.num_agents = num_agents_using_buffer
        # Controls how many agents are using the buffer, and so whenever there is an update, represents how many
        #   agents are left to update using the buffer's contents
        self.num_agents_left_to_update = num_agents_using_buffer

    @abstractmethod
    def add_experience(self, *args):
        raise NotImplementedError(
            "add_experience() has no base implementation, and must be implemented in the child class"
        )

    @abstractmethod
    def sample_experience(self, batch_size=32):
        raise NotImplementedError(
            "sample_experience() has no base implementation, and must be implemented in the child class"
        )
