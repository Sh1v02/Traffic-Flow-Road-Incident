from copy import deepcopy

import torch
from torch import nn

from src.Models import QMIXMixerNetwork
from src.Models.QMIX.QMIXAgentNetwork import QMIXAgentNetwork
from src.Utilities import multi_agent_settings, settings
from src.Wrappers.GPUSupport import optimise


class QMIX(nn.Module):
    def __init__(self, local_state_dims, global_state_dims, action_dims, hidden_layer_dims=64, loss=torch.nn.MSELoss()):
        super().__init__()
        self.online_agent_networks = nn.ModuleList(
            [
                QMIXAgentNetwork(None, None, local_state_dims, action_dims) for _ in
                range(multi_agent_settings.AGENT_COUNT)
            ]
        )
        self.online_mixer_network = QMIXMixerNetwork(global_state_dims,
                                                     hyper_network_hidden_layer_dims=hidden_layer_dims,
                                                     hidden_layer_dims=settings.QMIX_MIXER_NETWORK_DIMS)

        # Create target networks
        self.target_agent_networks = nn.ModuleList(
            [
                optimise(deepcopy(self.online_agent_networks[i])) for i in
                range(multi_agent_settings.AGENT_COUNT)
            ]
        )
        self.target_mixer_network = optimise(deepcopy(self.online_mixer_network))

        self.loss = loss

        optimise(self)

    def get_q_values(self, local_states, target_networks=False):
        if not target_networks:
            agent_q_values = [self.online_agent_networks[i](local_states[i]) for i in
                              range(len(self.online_agent_networks))]
            return agent_q_values

        agent_q_values = [self.target_agent_networks[i](local_states[i]) for i in
                          range(len(self.target_agent_networks))]
        return agent_q_values

    def update_target_networks(self, tau=1):
        # Hard update -> we could just remove this bit, but even for the smallest efficiency keep it here
        if tau == 1:
            for i in range(multi_agent_settings.AGENT_COUNT):
                self.target_agent_networks[i].load_state_dict(self.online_agent_networks[i].state_dict())
            self.target_mixer_network.load_state_dict(self.online_mixer_network.state_dict())
            return

        # Soft update
        for param, target_param in zip(self.online_mixer_network.parameters(), self.target_mixer_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for i in range(multi_agent_settings.AGENT_COUNT):
            for param, target_param in zip(self.online_agent_networks[i].parameters(),
                                           self.target_agent_networks[i].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
