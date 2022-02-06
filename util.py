import math
import copy
import numpy as np
import torch
import torch.optim as optim

torch.set_default_tensor_type(torch.DoubleTensor)


def train_objective(params, loss_fn, k = 0, lr=0.01, l2=0.0, epochs=5000, print_freq=100):
    '''
    Optimizes 'loss_fn' with respect to 'params'
    'loss_fn' must return a tuple of two:
    the value of the loss, and the model. 
    'k' is the regularization term in MAP 
    '''
    
    best_model = None
    min_loss = float('inf')

    optimizer = optim.Adam(params, lr=lr, weight_decay=l2)
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            # save loss and model if loss is the smallest observed so far
            if loss_fn == "mle_loss":
                loss, model = loss_fn()
            else:
                loss, model = loss_fn(k)
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_model = copy.deepcopy(model)
        
            loss.backward()
            optimizer.step()
        
            if epoch % print_freq == 0:
                print('Epoch {}: loss = {}'.format(epoch, loss.item()))
    except KeyboardInterrupt:
        print('Interrupted...')

    print('Final Loss = {}'.format(min_loss))
    return best_model, min_loss
            

def add_output_noise(r, output_var):
    '''
    Adds Gaussian noise to a tensor
    '''
    
    eps = torch.nn.init.normal_(torch.zeros_like(r), std=math.sqrt(output_var))
    assert(eps.size() == r.size())
    return r + eps


def cuda_available():
    return torch.cuda.is_available()


def ftens_cuda(*args, **kwargs):
    if cuda_available():
        t = torch.cuda.DoubleTensor(*args, **kwargs)
    else:
        t = torch.DoubleTensor(*args, **kwargs)

    return t


def to_np(v):
    if type(v) == float:
        return v

    if v.is_cuda:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = torch.mean(X)
    if std is None:
        std = torch.std(X)
    
    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean
