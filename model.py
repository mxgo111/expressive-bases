import abc
import copy
from collections import OrderedDict
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import matplotlib; matplotlib.use('Agg')  # Allows to create charts with undefined $DISPLAY
import matplotlib.pyplot as plt
import seaborn as sns

from util import (
    ftens_cuda,
    to_np,
    cuda_available,
    add_output_noise,
)


LR = 0.01
EPOCHS = 10000
DEFAULT_ACTIVATION_MODULE = nn.ReLU


class FullyConnected(nn.Module):
    '''
    Fully connected neural network
    '''
    
    def __init__(
            self,
            layers,
            output_activation=False,
            activation_module=DEFAULT_ACTIVATION_MODULE,
            bias=True,
    ):
        super(FullyConnected, self).__init__()

        assert(len(layers) >= 2)
        self.layers = layers
        self.output_activation = output_activation
        self.activation_module = activation_module
        self.bias = bias

        # seq stores layers
        seq = OrderedDict()
        seq['layer_0'] = nn.Linear(self.layers[0], self.layers[1], bias=bias)

        idx = 0
        for i in range(len(self.layers[1:-1])):
            seq['layer_{}_activation'.format(i)] = activation_module()

            idx = i + 1
            seq['layer_{}'.format(idx)] = nn.Linear(
                self.layers[idx], self.layers[idx + 1], bias=bias,
            )

        if output_activation:
            seq['layer_{}_activation'.format(idx + 1)] = activation_module()

        self.network = nn.Sequential(seq)

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        return torch.cat([p.view(-1) for p in self.parameters()])

    def rand_init(self, std=1.0):
        def init_weights(l):
            if isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, mean=0.0, std=std)                
                if self.bias:
                    torch.nn.init.normal_(l.bias, mean=0.0, std=std)

        self.network.apply(init_weights)            


def bayesian_linear_regression_posterior_1d(X, y, weights_var, output_var):
    assert(len(X.shape) == 2)
    assert(len(y.shape) == 2)
    assert(y.shape[-1] == 1)
    
    eye = torch.eye(X.shape[-1])
    if cuda_available():
        eye = eye.cuda()
        
    posterior_precision = eye / weights_var + torch.mm(X.t(), X) / output_var
    posterior_cov = torch.pinverse(posterior_precision)
    posterior_mu = torch.mm(posterior_cov, torch.mm(X.t(), y)).squeeze() / output_var
    
    return dists.MultivariateNormal(posterior_mu, precision_matrix=posterior_precision)

    
class BayesianRegression(nn.Module):
    def __init__(self, weights_var, output_var):
        super(BayesianRegression, self).__init__()        

        self.weights_var = weights_var
        self.output_var = output_var
        
        self.posterior = None
        
    def data_to_features(self, x):
        '''
        Concatenate features x with a column of 1s (for bias)
        '''
        
        ones = torch.ones(x.shape[0], 1)
        if cuda_available():
            ones = ones.cuda()

        return torch.cat([x, ones], -1)

    def get_posterior(self):
        return self.posterior
    
    def infer_posterior(self, x, y):
        '''
        Infers posterior and stores it within the class instance
        '''
        
        phi = self.data_to_features(x)
        assert(len(phi.shape) == 2)
        
        self.posterior = bayesian_linear_regression_posterior_1d(
            phi, y, self.weights_var, self.output_var,
        )
        
        return self.posterior
        
    def sample_posterior_predictive(self, x, num_samples):
        assert(self.posterior is not None)
        
        phi = self.data_to_features(x)
        
        weights = self.posterior.rsample(torch.Size([num_samples]))
        assert(weights.shape == (num_samples, phi.shape[-1]))

        r = torch.mm(phi, weights.t())
        assert(r.shape == torch.Size([x.shape[0], num_samples]))
        
        return add_output_noise(r, self.output_var)

    def sample_prior_predictive(self, x, num_samples, output_noise=True):
        phi = self.data_to_features(x)

        weights = dists.Normal(0.0, math.sqrt(self.weights_var)).sample((num_samples, phi.shape[-1]))
        assert(len(weights.shape) == 2)
        
        r = torch.mm(phi, weights.t())
        assert(r.shape == torch.Size([x.shape[0], num_samples]))

        if output_noise:
            return add_output_noise(r, self.output_var)    
        return r

