from src.Agents.DDPGAgent import DDPGAgent
from src.Agents.DDQNAgent import DDQNAgent
from src.Agents.PPOAgent import PPOAgent
from src.Buffers import PPOReplayBuffer
from src.Utilities import settings, multi_agent_settings


class AgentFactory:
    @staticmethod
    def create_new_agent(env, replay_buffer=None):
        states, _ = env.reset(seed=settings.SEED)

        is_multi_agent = settings.RUN_TYPE.lower() == "multiagent"
        state_dims = len(states[0]) if is_multi_agent else len(states)
        action_dims = env.action_space[0].n if is_multi_agent else env.action_space.n

        replay_buffer = replay_buffer if replay_buffer else None

        if settings.AGENT_TYPE == "ddqn":
            return DDQNAgent(state_dims, action_dims)
        if settings.AGENT_TYPE == "ppo":
            return PPOAgent(state_dims, action_dims, replay_buffer=replay_buffer)
        if settings.AGENT_TYPE == "ddpg":
            return DDPGAgent(state_dims, action_dims, env, noise=0.8)

        raise Exception("No agent found with type: ", settings.AGENT_TYPE)

    @staticmethod
    def create_shared_replay_buffer():
        if settings.AGENT_TYPE.lower() == "ppo":
            return PPOReplayBuffer(num_agents_using_buffer=multi_agent_settings.AGENT_COUNT)
