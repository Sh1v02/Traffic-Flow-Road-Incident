import torch

from src.Models.CriticNetwork import CriticNetwork


class DDPGCriticNetwork(CriticNetwork):
    def __init__(self, optimiser, loss, input_dims, hidden_layer_dims=None, optimiser_args=None):
        super().__init__(optimiser, loss, input_dims, hidden_layer_dims, optimiser_args)

    def forward(self, state: torch.Tensor, action: torch.Tensor = None):
        if action is None:
            raise ValueError("DDPG CriticNetwork must take in an action tensor")

        input_tensor = torch.cat([state, action], dim=1)
        value = self.network(input_tensor)
        return value
