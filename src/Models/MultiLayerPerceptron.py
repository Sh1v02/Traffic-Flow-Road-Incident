from torch import nn

from src.Models.NetworkCreator import NetworkCreator


class MultiLayerPerceptron(NetworkCreator):
    def __init__(self, optimiser, loss, input_dims, action_dims, hidden_layer_dims=None, optimiser_args=None):
        super().__init__(optimiser, loss, input_dims, action_dims, hidden_layer_dims, optimiser_args)

    def forward(self, state):
        q_values = self.network(state)
        return q_values

    def update(self, predicted, target):
        self.optimiser.zero_grad()
        loss = self.loss(predicted, target)
        loss.backward()
        self.optimiser.step()
