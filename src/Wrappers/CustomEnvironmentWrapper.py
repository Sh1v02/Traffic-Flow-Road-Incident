import gym
import numpy as np
import torch

from src.Utilities import settings
from src.Wrappers.GPUSupport import tensor


class CustomEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agent_type = settings.AGENT_TYPE.lower()
        self.is_multi_agent = settings.RUN_TYPE.lower() == "multiagent"

    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        if trunc:
            done = True
            if self.is_multi_agent:
                info["agents_dones"] = tuple(True for _ in info["agents_dones"])

        if settings.RENDER_ENVIRONMENT and not settings.RECORD_EPISODES[0]:
            self.env.render()

        return self._transform_state(next_state), reward, done, trunc, info

    def reset(self, **kwargs):
        states, infos = self.env.reset(seed=settings.ENVIRONMENT_SEED, **kwargs)

        return self._transform_state(states), infos

    def get_global_state(self, local_states):
        if self.agent_type == 'mappo':
            if settings.MAPPO_VALUE_FUNCTION_INPUT_REPRESENTATION.lower() == "cl":
                global_state = np.concatenate([local_state for local_state in local_states], axis=0)
            else:
                global_state = self.env.get_global_state().flatten()
            return global_state

        if self.agent_type == 'qmix':
            if settings.QMIX_VALUE_FUNCTION_INPUT_REPRESENTATION.lower() == "cl":
                global_state = torch.cat(local_states, dim=0)
            else:
                global_state = tensor(self.env.get_global_state().flatten())
            return global_state

        global_state = self.env.get_global_state().flatten()

        if self.agent_type == "ddqn" or self.agent_type == 'vdn':
            global_state = tensor(global_state)

        return global_state

    def _transform_state(self, states):
        # All networks take in a flattened state
        states = [state.flatten() for state in states] if self.is_multi_agent else states.flatten()

        # Custom processing to the state based on the type of agent (which have different networks)
        if self.agent_type == 'ddqn' or self.agent_type == 'qmix' or self.agent_type == 'vdn':
            states = [tensor(state) for state in states] if self.is_multi_agent else tensor(states)

        return states
