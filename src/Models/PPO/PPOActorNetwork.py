import torch

from src.Models.ActorNetwork import ActorNetwork


class PPOActorNetwork(ActorNetwork):
    def __init__(self, optimiser, loss, input_dims, action_dims, hidden_layer_dims=None, optimiser_args=None):
        hidden_layer_dims = hidden_layer_dims if hidden_layer_dims else [256, 256]
        super().__init__(optimiser, loss, input_dims, action_dims, hidden_layer_dims, optimiser_args)

    def forward(self, state: torch.Tensor):
        action = self.softmax(self.network(state))
        return action
