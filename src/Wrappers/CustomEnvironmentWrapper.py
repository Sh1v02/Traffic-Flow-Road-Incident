import gym

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
        states, infos = self.env.reset(**kwargs)

        return self._transform_state(states), infos

    def get_global_state(self):
        return self.env.get_global_state().flatten()

    def _transform_state(self, states):
        # All networks take in a flattened state
        states = [state.flatten() for state in states] if self.is_multi_agent else states.flatten()

        # Custom processing to the state based on the type of agent (which have different networks)
        if self.agent_type == 'ddqn':
            states = [tensor(state) for state in states] if self.is_multi_agent else tensor(states)

        return states
