from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    def __init__(self, max_size=10_000):
        self._max_size = max_size

    @abstractmethod
    def add_experience(self, experience):
        raise NotImplementedError(
            "add_experience() has no base implementation, and must be implemented in the child class"
        )

    @abstractmethod
    def sample_experience(self, batch_size=32):
        raise NotImplementedError(
            "sample_experience() has no base implementation, and must be implemented in the child class"
        )
