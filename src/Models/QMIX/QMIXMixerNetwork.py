import torch
from torch import nn

from src.Utilities import multi_agent_settings
from src.Wrappers.GPUSupport import optimise


class QMIXMixerNetwork(nn.Module):
    def __init__(self, global_state_dims, hyper_network_hidden_layer_dims=64, hidden_layer_dims=32):
        super().__init__()

        self.hidden_layer_dims = hidden_layer_dims

        self.hyper_network_for_input_layer_weights = optimise(
            nn.Sequential(
                nn.Linear(global_state_dims, hyper_network_hidden_layer_dims),
                nn.ReLU(),
                nn.Linear(hyper_network_hidden_layer_dims, hidden_layer_dims * multi_agent_settings.AGENT_COUNT)
            )
        )

        self.hyper_network_for_output_layer_weights = optimise(
            nn.Sequential(
                nn.Linear(global_state_dims, hyper_network_hidden_layer_dims),
                nn.ReLU(),
                nn.Linear(hyper_network_hidden_layer_dims, hidden_layer_dims)
            )
        )

        self.hyper_network_for_input_layer_bias = optimise(
            nn.Linear(global_state_dims, hidden_layer_dims)

        )
        self.hyper_network_for_output_layer_bias = optimise(
            nn.Sequential(
                nn.Linear(global_state_dims, hidden_layer_dims),
                nn.ReLU(),
                nn.Linear(hidden_layer_dims, 1)
            )
        )

    # Takes in q_values and the global states and produces q_total
    def forward(self, agent_q_values: torch.Tensor, global_states: torch.Tensor):
        agent_q_values = agent_q_values.view(-1, 1, multi_agent_settings.AGENT_COUNT)

        # TODO: Make sure the output of this matches agent_q_values dims so we can matrix multiply
        weights_1 = torch.abs(self.hyper_network_for_input_layer_weights(global_states))
        weights_1 = weights_1.view(-1, multi_agent_settings.AGENT_COUNT, self.hidden_layer_dims)

        bias_1 = self.hyper_network_for_input_layer_bias(global_states)
        bias_1 = bias_1.view(-1, 1, self.hidden_layer_dims)

        # Weighted sum of weights_1 with agent_qs, then add bias_1 -> this acts as the first layer of the MixerNetwork
        # Passed through ELU to allow negatives
        mixer_layer_1_output = torch.nn.functional.elu(torch.bmm(agent_q_values, weights_1) + bias_1)

        # Weighted sum of weights_2 with first mixing network layer output, then add bias_1
        # No activation function
        weights_final = torch.abs(self.hyper_network_for_output_layer_weights(global_states))
        weights_final = weights_final.view(-1, self.hidden_layer_dims, 1)
        bias_final = self.hyper_network_for_output_layer_bias(global_states)
        bias_final = bias_final.view(-1, 1, 1)

        q_total = torch.bmm(mixer_layer_1_output, weights_final) + bias_final
        q_total = q_total.view(32, -1, 1)
        return q_total
