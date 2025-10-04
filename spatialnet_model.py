import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialBlock(nn.Module):
    def __init__(self, num_neurons, input_dim, output_dim):
        super().__init__()
        '''
        This is a special kind of network in which neurons have spatial positions.
        The output of a given point in the input space is a weighted sum of the outputs of the neurons,
        where the weights are determined by the distance between the input point and the neuron positions.

        The neurons project a gaussian field.
        '''

        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim


        vecs = torch.randn(num_neurons, input_dim)
        directions = vecs / (vecs.norm(dim=1, keepdim=True) + 1e-8)
        radii = torch.rand(num_neurons, 1) ** (1.0 / input_dim)
        self.positions = nn.Parameter(directions * radii)

        self.weights = nn.Parameter(torch.randn(num_neurons, output_dim))
        self.log_sigma = nn.Parameter(torch.full((num_neurons,), 0.1))

        self.target = nn.Parameter(torch.zeros(num_neurons, output_dim)) #what each neuron wants to 'say'


        
    def gaussian_weights(self, x):
        dists = torch.cdist(x, self.positions)  # (batch_size, num_neurons)
        sigma = F.softplus(self.log_sigma) + 1e-6  # keep σ > 0
        return torch.exp(-dists**2 / (2 * sigma**2))

    def forward(self, x):
        '''
        The inputs are of shape (batch_size, input_dim)
        We need to copute the distance between each input and each neuron position.
        '''
        # distance computation
        gweights = self.gaussian_weights(x)  # (batch_size, num_neurons)
        gweights = gweights / (gweights.sum(dim=1, keepdim=True) + 1e-8)  # normalize weights
        #gweights = gweights / self.num_neurons
        output = gweights @ self.weights  # (batch_size, output_dim)
        return output

    
class SpatialNet(nn.Module):
    def __init__(self, num_neurons = [10,10,10], hidden_dims = [10,10], input_dim = 10, output_dim = 10):
        super().__init__()
        self.num_neurons = num_neurons
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = []


        
        if hidden_dims:
            assert len(hidden_dims) == len(num_neurons) -1, 'Must have a hidden dim for each layer'

            self.layers.append(SpatialBlock(num_neurons[0], input_dim, hidden_dims[0]))
            for n, h in zip(num_neurons[1:-1], hidden_dims[:-1]):
                self.layers.append(SpatialBlock(n, h, h))
            self.layers.append(SpatialBlock(num_neurons[-1], hidden_dims[-1], output_dim))
        else:
            self.layers.append(SpatialBlock(num_neurons[0], input_dim, output_dim))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
