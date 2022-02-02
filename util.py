import math
import copy

import torch
import torch.optim as optim

torch.set_default_tensor_type(torch.DoubleTensor)


def train_objective(params, loss_fn, lr=0.01, l2=0.0, epochs=5000, print_freq=100):
    '''
    Optimizes 'loss_fn' with respect to 'params'
    'loss_fn' must take no arguments, and must return a tuple of two:
    the value of the loss, and the model. 
    '''
    
    best_model = None
    min_loss = float('inf')

    optimizer = optim.Adam(params, lr=lr, weight_decay=l2)
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            # save loss and model if loss is the smallest observed so far
            loss, model = loss_fn()
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


