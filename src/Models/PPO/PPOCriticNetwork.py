import torch

from src.Models.CriticNetwork import CriticNetwork


class PPOCriticNetwork(CriticNetwork):
    def __init__(self, optimiser, loss, input_dims, hidden_layer_dims=None, optimiser_args=None):
        hidden_layer_dims = hidden_layer_dims if hidden_layer_dims else [256, 256]
        super().__init__(optimiser, loss, input_dims, hidden_layer_dims, optimiser_args)

    def forward(self, state: torch.Tensor):
        value = self.network(state)
        return value
