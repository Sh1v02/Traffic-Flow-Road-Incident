import torch

from src.Agents.DDPGAgent import DDPGAgent
from src.Agents.DDQNAgent import DDQNAgent
from src.Agents.PPOAgent import PPOAgent
from src.Buffers import SharedPPOReplayBuffer
from src.Buffers.SharedUniformExperienceReplayBuffer import SharedUniformExperienceReplayBuffer
from src.Models import PPOActorNetwork, PPOCriticNetwork
from src.Utilities import settings


class AgentFactory:
    @staticmethod
    def create_new_agent(state_dims, action_dims, env, replay_buffer=None, networks=None):
        if settings.AGENT_TYPE.lower() == "ddqn":
            return DDQNAgent(state_dims, action_dims, replay_buffer=replay_buffer)
        if settings.AGENT_TYPE.lower() == "ppo":
            return PPOAgent(state_dims, action_dims, replay_buffer=replay_buffer, networks=networks)
        if settings.AGENT_TYPE.lower() == "ddpg":
            return DDPGAgent(state_dims, action_dims, env, noise=0.8)

        raise Exception("No agent found with type: ", settings.AGENT_TYPE)

    @staticmethod
    def create_shared_replay_buffer(state_dims, action_dims):
        if settings.AGENT_TYPE.lower() == "ppo":
            return SharedPPOReplayBuffer()
        if settings.AGENT_TYPE.lower() == "ddqn":
            return SharedUniformExperienceReplayBuffer(state_dims, max_size=settings.DDQN_REPLAY_BUFFER_SIZE,
                                                       actions_type=torch.int32)

    @staticmethod
    def create_fully_shared_networks(state_dims, action_dims, optimiser=torch.optim.Adam, loss=torch.nn.MSELoss()):
        if settings.AGENT_TYPE.lower() == "ppo":
            return {
                "actor": PPOActorNetwork(optimiser, loss, state_dims, action_dims,
                                         optimiser_args={"lr": settings.PPO_LR[0]},
                                         hidden_layer_dims=settings.PPO_NETWORK_DIMS),
                "critic": PPOCriticNetwork(optimiser, loss, state_dims,
                                           optimiser_args={"lr": settings.PPO_LR[1]},
                                           hidden_layer_dims=settings.PPO_NETWORK_DIMS)
            }
