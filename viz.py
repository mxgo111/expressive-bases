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
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from util import *


def get_coverage_bounds(posterior_pred_samples, percentile):
    '''
    Assumes N x samples
    '''
    assert(not (percentile < 0. or percentile > 100.))

    lower_percentile = (100.0 - percentile) / 2.0
    upper_percentile = 100.0 - lower_percentile

    upper_bounds = np.percentile(posterior_pred_samples, upper_percentile, axis=-1)
    lower_bounds = np.percentile(posterior_pred_samples, lower_percentile, axis=-1)

    return lower_bounds, upper_bounds


def plot_1d_posterior_predictive(x_train, y_train, x_viz, y_pred):
    assert(len(x_train.shape) == 2 and x_train.shape[-1] == 1)
    assert(len(y_train.shape) == 2 and y_train.shape[-1] == 1)
    assert(len(x_viz.shape) == 2 and x_viz.shape[-1] == 1)    
    assert(len(y_pred.shape) == 2 and y_pred.shape[0] == x_viz.shape[0])

    # make sure x_viz is sorted in ascending order
    x_viz = to_np(x_viz.squeeze())
    assert(np.all(x_viz[:-1] <= x_viz[1:]))
        
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # plot predictive intervals
    for picp, alpha in zip([50.0, 68.0, 95.0], [0.4, 0.3, 0.2]):
        lower, upper = get_coverage_bounds(to_np(y_pred), picp)
        
        ax.fill_between(
            x_viz, lower, upper, label='{}%-PICP'.format(picp), color='steelblue', alpha=alpha,
        )

    # plot predictive mean
    pred_mean = to_np(torch.mean(y_pred, -1))        
    ax.plot(x_viz, pred_mean, color='blue', lw=3, label='Predictive Mean')

    # plot training data
    ax.scatter(x_train, y_train, color='red', s=10.0, zorder=10, label='Training Data')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')    
    ax.set_title('Posterior Predictive')
    ax.legend()
    
    #plt.tight_layout()
    plt.show()
    #plt.savefig('output/pred.png')
    #plt.close()
    

def get_uncertainty_in_gap(model, basis, x_train, y_train, n_points, picp=95.0):
    assert(len(x_train.shape) == 2 and x_train.shape[-1] == 1)
    assert(len(y_train.shape) == 2 and y_train.shape[-1] == 1)

    # make sure x_train is sorted in ascending order
    x_train_sorted = np.sort(to_np(x_train.squeeze()))
    assert(np.all(x_train_sorted[:-1] <= x_train_sorted[1:]))

    # find gap
    N = len(x_train)
    gap = np.linspace(x_train_sorted.squeeze()[N//2-1], x_train_sorted.squeeze()[N//2], n_points)
    h = gap[1] - gap[0]
    gap = ftens_cuda(gap).unsqueeze(-1)
    
    # sample from inside gap
    y_pred = model.sample_posterior_predictive(basis(gap), n_points)
    lower, upper = get_coverage_bounds(to_np(y_pred), picp)
    
    area = uncertainty_area(upper, lower, h)

    return area


def plot_basis_functions_1d(num_final_layers, x_vals, basis, x_train, posterior_mean, numcols=12):
    basis_vals = basis(torch.tensor(x_vals.reshape(-1, 1)))

    # sort functions
    def argsort(seq):
        # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
        return sorted(range(len(seq)), key=lambda x: abs(max(seq[x]) - min(seq[x])))

    functions = [basis_vals[:, i].detach().cpu().numpy() for i in range(num_final_layers)]
    argsorted_basis = argsort(functions)

    # training data
    x_train_np = x_train.detach().cpu().numpy().squeeze()
    basis_train_np = basis(x_train).detach().cpu().numpy()

    fig, axs = plt.subplots(num_final_layers//numcols + 1, numcols, figsize=(40, 15))
    for j in range(num_final_layers):
        i = argsorted_basis[j]
        row, col = j//numcols, j % numcols
        axs[row,col].plot(x_vals, functions[i])
        axs[row,col].scatter(x_train_np, basis_train_np[:,i], c="red") # scatterplot training data
        axs[row,col].set_title(f"w_posterior_mean={np.round(posterior_mean.detach().cpu().numpy()[i], 3)}")
    plt.savefig("visualize-bases-02-07-22-v3")
    plt.tight_layout()
    plt.show()
    
    return basis_vals

def eff_dim(evals, z):
        assert z > 0
        return np.sum(np.divide(evals, evals+z))

def compute_eff_dim(basis_vals, z=1, visual=False):
    # each column is a basis function
    basis_vals_np = basis_vals.detach().cpu().numpy()
    basis_vals_df = pd.DataFrame(basis_vals_np)
    
    # calculate correlations
    corr = basis_vals_df.corr()
    
    # drop irrelevant rows/columns
    corr.dropna(axis=0, how='all', inplace=True)
    corr.dropna(axis=1, how='all', inplace=True)
    
    # eigenvals
    evals, evecs = np.linalg.eig(corr)
    
    if visual:
        plt.figure(figsize=(10,10))
        # plot the heatmap
        sns.heatmap(corr, 
                xticklabels=corr.columns,
                yticklabels=corr.columns)
        
        # Scree plot
        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(len(evals)), evals)
        plt.plot(evals)
        plt.title("Eigenvalues of correlation matrix")
        plt.show()
    
    return eff_dim(evals, z)
    
    
