import torch

from src.Models.NetworkCreator import NetworkCreator


class QMIXAgentNetwork(NetworkCreator):
    def __init__(self, optimiser, loss, local_state_dims, action_dims,
                 hidden_layer_dims=None, optimiser_args=None):
        hidden_layer_dims = hidden_layer_dims if hidden_layer_dims else [64]
        super().__init__(optimiser, loss, local_state_dims, action_dims,
                         hidden_layer_dims=hidden_layer_dims, optimiser_args=optimiser_args)

    # Takes in the partially observed state (local_state) and passes it through its network to get
    #   the q values for all actions
    def forward(self, local_state: torch.Tensor):
        # Pass through the network (no activation after final layer)
        q_values = self.network(local_state)

        return q_values
