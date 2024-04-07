import torch
from torch import nn

from src.Utilities import multi_agent_settings, settings
from src.Wrappers.GPUSupport import optimise


class VDNMixerNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, agent_q_values):
        # Change shapes: (self.batch_size, 1, 1) -> (self.batch_size)
        return torch.sum(agent_q_values, dim=-1, keepdim=True).squeeze(1)


class QMIXMixerNetwork(nn.Module):
    def __init__(self, global_state_dims):
        super().__init__()
        self.shared_agent_net = settings.QMIX_AGENT_NETWORKS_SHARED
        self.hidden_layer_dims = settings.QMIX_MIXER_NETWORK_DIMS

        if settings.QMIX_HYPER_NETWORK_LAYERS == 1:
            self.hyper_network_for_input_layer_weights = optimise(
                nn.Linear(global_state_dims, settings.QMIX_MIXER_NETWORK_DIMS *
                          (1 if self.shared_agent_net else multi_agent_settings.AGENT_COUNT))
            )
            self.hyper_network_for_output_layer_weights = optimise(
                nn.Linear(global_state_dims, settings.QMIX_MIXER_NETWORK_DIMS)
            )
        else:
            self.hyper_network_for_input_layer_weights = optimise(
                nn.Sequential(
                    nn.Linear(global_state_dims, settings.QMIX_HYPER_NETWORK_DIMS),
                    nn.ReLU(),
                    nn.Linear(settings.QMIX_HYPER_NETWORK_DIMS, settings.QMIX_MIXER_NETWORK_DIMS *
                              (1 if self.shared_agent_net else multi_agent_settings.AGENT_COUNT))
                )
            )

            self.hyper_network_for_output_layer_weights = optimise(
                nn.Sequential(
                    nn.Linear(global_state_dims, settings.QMIX_HYPER_NETWORK_DIMS),
                    nn.ReLU(),
                    nn.Linear(settings.QMIX_HYPER_NETWORK_DIMS, settings.QMIX_MIXER_NETWORK_DIMS)
                )
            )

        self.hyper_network_for_input_layer_bias = optimise(
            nn.Linear(global_state_dims, settings.QMIX_MIXER_NETWORK_DIMS)

        )
        self.hyper_network_for_output_layer_bias = optimise(
            nn.Sequential(
                nn.Linear(global_state_dims, settings.QMIX_MIXER_NETWORK_DIMS),
                nn.ReLU(),
                nn.Linear(settings.QMIX_MIXER_NETWORK_DIMS, 1)
            )
        )

    # Takes in q_values and the global states and produces q_total
    def forward(self, agent_q_values: torch.Tensor, global_states: torch.Tensor):
        agent_q_values = agent_q_values.unsqueeze(1) if self.shared_agent_net else \
            agent_q_values.view(-1, 1, multi_agent_settings.AGENT_COUNT)

        # TODO: Make sure the output of this matches agent_q_values dims so we can matrix multiply
        weights_1 = torch.abs(self.hyper_network_for_input_layer_weights(global_states))
        weights_1 = weights_1.unsqueeze(1) if self.shared_agent_net else \
            weights_1.view(-1, multi_agent_settings.AGENT_COUNT, self.hidden_layer_dims)

        bias_1 = self.hyper_network_for_input_layer_bias(global_states)
        bias_1 = bias_1.unsqueeze(1) if self.shared_agent_net else bias_1.view(-1, 1, self.hidden_layer_dims)

        # Weighted sum of weights_1 with agent_qs, then add bias_1 -> this acts as the first layer of the MixerNetwork
        # Passed through ELU to allow negatives
        mixer_layer_1_output = torch.nn.functional.elu(torch.bmm(agent_q_values, weights_1) + bias_1)

        # Weighted sum of weights_2 with first mixing network layer output, then add bias_1
        # No activation function
        weights_final = torch.abs(self.hyper_network_for_output_layer_weights(global_states))
        weights_final = weights_final.unsqueeze(2) if self.shared_agent_net else\
            weights_final.view(-1, self.hidden_layer_dims, 1)
        bias_final = self.hyper_network_for_output_layer_bias(global_states)
        bias_final = bias_final.unsqueeze(1) if self.shared_agent_net else bias_final.view(-1, 1, 1)

        q_total = torch.bmm(mixer_layer_1_output, weights_final) + bias_final
        q_total = q_total.view(32, -1, 1)

        # Change shapes: (self.batch_size, 1, 1) -> (self.batch_size)
        return q_total.squeeze(1).squeeze(1)
