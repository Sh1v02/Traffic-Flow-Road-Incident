from src.Models import PPOCriticNetwork


class MAPPOCriticNetwork(PPOCriticNetwork):
    def __init__(self, optimiser, loss, local_state_dims, global_state_dims, value_function_input_type,
                 hidden_layer_dims=None, optimiser_args=None):
        input_dims = global_state_dims if value_function_input_type == "ep" else global_state_dims + local_state_dims
        super().__init__(optimiser, loss, input_dims, hidden_layer_dims=hidden_layer_dims,
                         optimiser_args=optimiser_args)
