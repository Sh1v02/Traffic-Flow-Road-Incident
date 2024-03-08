import torch
from torch import nn

from src.Models import NetworkCreator


class ActorNetwork(NetworkCreator):
    def __init__(self, optimiser, loss, input_dims, action_dims, hidden_layer_dims=None, optimiser_args=None):
        super().__init__(optimiser, loss, input_dims, action_dims, hidden_layer_dims, optimiser_args)

    def forward(self, state: torch.Tensor):
        action = self.tanh(self.network(state))
        return action

    def update(self, actor_loss):
        self.optimiser.zero_grad()
        loss = torch.mean(actor_loss, dim=0)
        loss.backward()
        self.optimiser.step()
