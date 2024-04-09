from src.Models import PPOCriticNetwork
from src.Utilities import multi_agent_settings


class MAPPOCriticNetwork(PPOCriticNetwork):
    def __init__(self, optimiser, loss, local_state_dims, global_state_dims, value_function_input_type,
                 hidden_layer_dims=None, optimiser_args=None):
        if value_function_input_type == "cl":
            input_dims = local_state_dims * multi_agent_settings.AGENT_COUNT
        elif value_function_input_type == "as":
            input_dims = global_state_dims + local_state_dims
        else:
            input_dims = global_state_dims
        super().__init__(optimiser, loss, input_dims, hidden_layer_dims=hidden_layer_dims,
                         optimiser_args=optimiser_args)
