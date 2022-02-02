import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
import matplotlib; matplotlib.use('Agg')  # Allows to create charts with undefined $DISPLAY
import matplotlib.pyplot as plt

from model import FullyConnected, BayesianRegression
from viz import plot_1d_posterior_predictive
from util import (
    ftens_cuda,
    to_np,
    cuda_available,
    add_output_noise,
    train_objective,
)


OUTPUT_DIR = 'output'


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # define model
    w_prior_var = 1.0 # variance of prior over weights
    output_var = 0.01 # variance of observation noise
    model = BayesianRegression(w_prior_var, output_var)
    
    # set training data
    N = 100
    x_train = dists.Uniform(-1.0, 1.0).sample((N, 1))
    y_train = add_output_noise(torch.pow(x_train, 3.0), output_var)
    
    # parameters of optimizer
    LEARNING_RATE = 0.001 
    EPOCHS = 20000

    # architecture and activation
    ACTIVATION = nn.LeakyReLU
    layers = [1, 50, 50, 1]
    
    # define a neural network feature basis and final layer
    basis = FullyConnected(layers[:-1], activation_module=ACTIVATION, output_activation=True)
    final_layer = FullyConnected(layers[-2:], activation_module=ACTIVATION)

    # define MLE loss
    def mle_loss():
        y_pred = final_layer(basis(x_train))
        loss = torch.mean(torch.sum(torch.pow(final_layer(basis(x_train)) - y_train, 2.0), -1))
        
        return loss, (basis, final_layer)

    # randomly initialize basis and last layer
    basis.rand_init(math.sqrt(w_prior_var))
    final_layer.rand_init(math.sqrt(w_prior_var))

    # optimize loss to learn network
    (basis, final_layer), loss = train_objective(
        list(basis.parameters()) + list(final_layer.parameters()),
        mle_loss,
        lr=LEARNING_RATE,
    )

    # infer posterior over the last layer weights given the basis
    model.infer_posterior(basis(x_train), y_train)

    # sample from posterior predictive
    x_viz = ftens_cuda(np.linspace(-2.0, 2.0, 500)).unsqueeze(-1)
    y_pred = model.sample_posterior_predictive(basis(x_viz), 500)

    # visualize posterior predictive
    plot_1d_posterior_predictive(x_train, y_train, x_viz, y_pred)
    

if __name__ == '__main__':
    main()

