import torch
from torch import nn


class CriticNetwork(nn.Module):
    # input_dims = state_dims + action_dims
    def __init__(self, optimiser, loss, input_dims, hidden_layer_dims=512, optimiser_args=None):
        super().__init__()

        self.loss = loss

        self.input_dims = input_dims
        self.hidden_layer_dims = hidden_layer_dims
        self.input_layer = nn.Linear(self.input_dims, self.hidden_layer_dims)
        self.hidden_layer_1 = nn.Linear(self.hidden_layer_dims, self.hidden_layer_dims)
        self.output_layer = nn.Linear(self.hidden_layer_dims, 1)

        self.relu = nn.ReLU()

        optimiser_args = optimiser_args if optimiser_args else {}
        self.optimiser = optimiser(self.parameters(), **optimiser_args)

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
