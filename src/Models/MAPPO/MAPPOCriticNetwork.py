from src.Models import PPOCriticNetwork
from src.Utilities import multi_agent_settings


class MAPPOCriticNetwork(PPOCriticNetwork):
    def __init__(self, optimiser, loss, global_state_dims,
                 hidden_layer_dims=None, optimiser_args=None):
        super().__init__(optimiser, loss, global_state_dims, hidden_layer_dims=hidden_layer_dims,
                         optimiser_args=optimiser_args)
