
from src.Agents.DDPGAgent import DDPGAgent
from src.Agents.PPOAgent import PPOAgent
from src.Agents.DDQNAgent import DDQNAgent
from src.Settings import settings


class AgentFactory:
    @staticmethod
    def create_new_agent(env, agent_type=settings.AGENT_TYPE):
        agent_type = agent_type.lower()
        state, _ = env.reset(seed=settings.SEED)

        if agent_type == "ddqn":
            return DDQNAgent(len(state), env.action_space.n, batch_size=32,
                             update_target_network_frequency=50, lr=0.0005, gamma=0.8)
        if agent_type == "ppo":
            return PPOAgent(len(state), env.action_space.n)
        if agent_type == "ddpg":
            return DDPGAgent(len(state), env.action_space.shape[0], env, noise=0.8)

        raise Exception("No agent found with type: ", agent_type)
