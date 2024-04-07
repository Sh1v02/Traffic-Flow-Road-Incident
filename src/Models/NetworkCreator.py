from copy import deepcopy

from torch import nn

from src.Wrappers.GPUSupport import optimise


class NetworkCreator(nn.Module):
    def __init__(self, optimiser, loss, input_dims, output_dims, hidden_layer_dims=None, optimiser_args=None):
        super().__init__()

        # List of all layer dimensions in network
        self.layer_dims = [input_dims] + (hidden_layer_dims if hidden_layer_dims else [512, 512]) + [output_dims]
        self.network = self.create_network()

        self.loss = loss
        optimiser_args = optimiser_args if optimiser_args else {}
        if optimiser:
            self.optimiser = optimiser(self.parameters(), **optimiser_args)

        # Initialise activations
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        optimise(self)

    # Creates a dynamically sized network based on the number of hidden layers specified, + input and output layers
    # Returns a neural network, without a final activation function -> this is added in each network's forward()
    #   implementation
    def create_network(self):
        layers = []
        for layer_index in range(len(self.layer_dims) - 1):
            layers.append(nn.Linear(self.layer_dims[layer_index], self.layer_dims[layer_index + 1]))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers[:-1])

    def deep_copy_network(self):
        return optimise(deepcopy(self))

