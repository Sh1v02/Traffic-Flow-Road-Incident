
from src.Agents.DDPGAgent import DDPGAgent
from src.Agents.PPOAgent import PPOAgent
from src.Agents.DDQNAgent import DDQNAgent
from src.Utilities import settings


class AgentFactory:
    @staticmethod
    def create_new_agent(env, agent_type=settings.AGENT_TYPE):
        agent_type = agent_type.lower()
        states, _ = env.reset(seed=settings.SEED)

        is_multi_agent = settings.RUN_TYPE.lower() == "multiagent"
        state_dims = len(states[0]) if is_multi_agent else len(states)
        action_dims = env.action_space[0].n if is_multi_agent else env.action_space.n

        if agent_type == "ddqn":
            return DDQNAgent(state_dims, action_dims, batch_size=32,
                             update_target_network_frequency=50, lr=0.0005, gamma=0.8)
        if agent_type == "ppo":
            return PPOAgent(state_dims, action_dims)
        if agent_type == "ddpg":
            return DDPGAgent(state_dims, action_dims, env, noise=0.8)

        raise Exception("No agent found with type: ", agent_type)
