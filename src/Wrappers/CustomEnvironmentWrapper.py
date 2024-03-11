import gym
import torch

from src.Settings import settings


class CustomEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agent_type = settings.AGENT_TYPE.lower()

    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        if trunc:
            done = True

        if not settings.RECORD_EPISODES[0]:
            self.env.render()

        return self._transform_state(next_state), reward, done, trunc, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        return self._transform_state(state), info

    def _transform_state(self, state):
        # All networks take in a flattened state
        state = state.flatten()

        # Custom processing to the state based on the type of agent (which have different networks)
        if self.agent_type == 'ddqn':
            state = torch.Tensor(state)

        return state
