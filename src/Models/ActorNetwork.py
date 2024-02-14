from torch import nn


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_layer_dims=512):
        super().__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims
        self.hidden_layer_dims = hidden_layer_dims
        self.input_layer = nn.Linear(self.input_dims, self.hidden_layer_dims)
        self.hidden_layer_1 = nn.Linear(self.hidden_layer_dims, self.hidden_layer_dims)
        self.output_layer = nn.Linear(self.hidden_layer_dims, self.action_dims)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.input_layer(state))
        x = self.relu(self.hidden_layer_1(x))
        action = self.tanh(self.output_layer(x))

        return action
