from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def get_action(self, action):
        raise NotImplementedError("Agent must implement get_action()")

    @abstractmethod
    def store_experience_in_replay_buffer(self, *args):
        raise NotImplementedError("Agent must implement store_experience_in_replay_buffer()")

    @abstractmethod
    def learn(self):
        raise NotImplementedError("Agent must implement learn()")

    @abstractmethod
    def get_agent_specific_config(self):
        raise NotImplementedError("Agent must implement get_agent_specific_config")
