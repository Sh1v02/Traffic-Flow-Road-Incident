from src.Models import NetworkCreator
import torch


class CriticNetwork(NetworkCreator):
    def __init__(self, optimiser, loss, input_dims, hidden_layer_dims=None, optimiser_args=None):
        super().__init__(optimiser, loss, input_dims, 1, hidden_layer_dims, optimiser_args)

    def forward(self, state: torch.Tensor):
        value = self.network(state)
        return value

    def update(self, predicted, target):
        self.optimiser.zero_grad()
        loss = self.loss(predicted, target)
        loss.backward()
        self.optimiser.step()
