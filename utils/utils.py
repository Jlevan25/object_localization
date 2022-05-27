import time
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


def check_zero_divide(x, y):
    if any(y == 0):
        out = torch.zeros_like(x)
        indexes = y > 0
        out[indexes] = x[indexes] / y[indexes]
    else:
        out = x / y

    return out
