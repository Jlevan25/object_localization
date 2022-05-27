import time

import numpy as np
import torch


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} in {end - start:0.8f} seconds")
        return result

    return wrapper


def one_hot_argmax(tensor):
    batch_size, classes, *dims = tensor.shape
    preds = torch.zeros(tensor.nelement() // classes, classes)
    preds[torch.arange(len(preds)), tensor.argmax(dim=1).reshape(-1)] = 1
    return preds.reshape(batch_size, *dims, classes).permute(0, -1, *(torch.arange(len(dims)) + 1))


def sum_except_dim(x, dim):
    return x.transpose(dim, 0).reshape(x.shape[dim], -1).sum(1)


def check_zero_divide(x, y, val=-1):
    if any(y == 0):
        out = np.ones(x.shape, dtype='float32') * val
        indexes = y > 0
        out[indexes] = x[indexes] / y[indexes]
    else:
        out = x / y

    return out


def split_params4weight_decay(model):
    wd_params, no_wd = [], []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            wd_params.append(param)
        else:
            no_wd.append(param)
    return wd_params, no_wd

