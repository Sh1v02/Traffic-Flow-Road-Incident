import torch
from torch import nn


class CriticNetwork(nn.Module):
    # input_dims = state_dims + action_dims
    def __init__(self, input_dims, hidden_layer_dims=512):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_layer_dims = hidden_layer_dims
        self.input_layer = nn.Linear(self.input_dims, self.hidden_layer_dims)
        self.hidden_layer_1 = nn.Linear(self.hidden_layer_dims, self.hidden_layer_dims)
        self.output_layer = nn.Linear(self.hidden_layer_dims, 1)

        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        input_tensor = torch.cat([state, action])
        x = self.relu(self.input_layer(input_tensor))
        x = self.relu(self.hidden_layer_1(x))
        q_value = self.output_layer(x)
        return q_value
