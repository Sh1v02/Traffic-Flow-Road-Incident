import torch
from torch import nn


class ActorNetwork(nn.Module):
    def __init__(self, optimiser, loss, input_dims, action_dims, hidden_layer_dims=512, optimiser_args=None):
        super().__init__()

        self.loss = loss

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.hidden_layer_dims = hidden_layer_dims
        self.input_layer = nn.Linear(self.input_dims, self.hidden_layer_dims)
        self.hidden_layer_1 = nn.Linear(self.hidden_layer_dims, self.hidden_layer_dims)
        self.output_layer = nn.Linear(self.hidden_layer_dims, self.action_dims)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


        optimiser_args = optimiser_args if optimiser_args else {}
        self.optimiser = optimiser(self.parameters(), **optimiser_args)

    def forward(self, state: torch.Tensor):
        x = self.relu(self.input_layer(state))
        x = self.relu(self.hidden_layer_1(x))
        action = self.tanh(self.output_layer(x))

        return action

    def update(self, actor_loss):
        self.optimiser.zero_grad()
        loss = torch.mean(actor_loss, dim=0)
        loss.backward()
        self.optimiser.step()
