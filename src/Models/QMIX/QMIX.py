from copy import deepcopy

import torch
from torch import nn

from src.Models import QMIXMixerNetwork
from src.Models.QMIX.QMIXAgentNetwork import QMIXAgentNetwork
from src.Models.QMIX.MixerNetwork import VDNMixerNetwork
from src.Utilities import multi_agent_settings, settings
from src.Wrappers.GPUSupport import optimise


class QMIX(nn.Module):
    def __init__(self, local_state_dims, global_state_dims, action_dims, loss=torch.nn.MSELoss(reduction='none')):
        super().__init__()
        self.shared_agent_net = settings.QMIX_AGENT_NETWORKS_SHARED
        if self.shared_agent_net:
            self.online_agent_networks = QMIXAgentNetwork(None, None, local_state_dims, action_dims,
                                                          hidden_layer_dims=settings.QMIX_AGENT_NETWORK_DIMS)
            self.target_agent_networks = optimise(deepcopy(self.online_agent_networks))
        else:
            self.online_agent_networks = nn.ModuleList(
                [
                    QMIXAgentNetwork(None, None, local_state_dims, action_dims,
                                     hidden_layer_dims=settings.QMIX_AGENT_NETWORK_DIMS)
                    for _ in range(multi_agent_settings.AGENT_COUNT)
                ]
            )

            # Create target networks
            self.target_agent_networks = nn.ModuleList(
                [
                    optimise(deepcopy(self.online_agent_networks[i])) for i in
                    range(multi_agent_settings.AGENT_COUNT)
                ]
            )

        self.online_mixer_network = QMIXMixerNetwork(global_state_dims) if settings.AGENT_TYPE.lower() == "qmix" \
            else (VDNMixerNetwork())

        self.target_mixer_network = optimise(deepcopy(self.online_mixer_network))

        self.loss = loss

        optimise(self)

    def get_q_values(self, local_states):
        if self.shared_agent_net:
            agent_q_values = [self.online_agent_networks(local_states[i]) for i in
                              range(multi_agent_settings.AGENT_COUNT)]
        else:
            agent_q_values = [self.online_agent_networks[i](local_states[i]) for i in
                              range(len(self.online_agent_networks))]
        return agent_q_values

    def update_target_networks(self, tau=1):
        # Hard update -> we could just remove this bit, but even for the smallest efficiency keep it here
        if tau == 1:
            if self.shared_agent_net:
                self.target_agent_networks.load_state_dict(self.online_agent_networks.state_dict())
            else:
                for i in range(multi_agent_settings.AGENT_COUNT):
                    self.target_agent_networks[i].load_state_dict(self.online_agent_networks[i].state_dict())
            self.target_mixer_network.load_state_dict(self.online_mixer_network.state_dict())
            return

        # Soft update
        for param, target_param in zip(self.online_mixer_network.parameters(), self.target_mixer_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if self.shared_agent_net:
            for param, target_param in zip(self.online_agent_networks.parameters(),
                                           self.target_agent_networks.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        else:
            for i in range(multi_agent_settings.AGENT_COUNT):
                for param, target_param in zip(self.online_agent_networks[i].parameters(),
                                               self.target_agent_networks[i].parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
