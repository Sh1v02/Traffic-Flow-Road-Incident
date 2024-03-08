from src.Models import NetworkCreator

import torch
from torch import nn


class CriticNetwork(NetworkCreator):
    # input_dims = state_dims + action_dims
    def __init__(self, optimiser, loss, input_dims, hidden_layer_dims=None, optimiser_args=None):
        super().__init__(optimiser, loss, input_dims, 1, hidden_layer_dims, optimiser_args)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        input_tensor = torch.cat([state, action], dim=1)
        x = self.relu(self.input_layer(input_tensor))
        x = self.relu(self.hidden_layer_1(x))
        q_value = self.output_layer(x)
        return q_value

    def update(self, predicted, target):
        self.optimiser.zero_grad()
        loss = self.loss(predicted, target)
        loss.backward()
        self.optimiser.step()
