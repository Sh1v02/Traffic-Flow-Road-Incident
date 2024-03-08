from src.Models import ActorNetwork
import torch


class DDPGActorNetwork(ActorNetwork):
    def __init__(self, optimiser, loss, input_dims, action_dims, hidden_layer_dims=None, optimiser_args=None):
        super().__init__(optimiser, loss, input_dims, action_dims, hidden_layer_dims, optimiser_args)

    def forward(self, state: torch.Tensor):
        action = self.tanh(self.network(state))
        return action
