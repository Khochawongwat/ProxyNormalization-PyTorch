from scipy.special import erfinv
import numpy as np
import torch
from torch import nn

def normalize_tensor(y: torch.Tensor, eps: float):
    """
    Function to normalize a tensor.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")
    if not isinstance(eps, float):
        raise TypeError("eps must be a float.")
    
    return (y - torch.mean(y)) / torch.std(y)

def tensor_is_normalized(y: torch.Tensor, eps: float):
    """
    Function to check if a tensor is normalized.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")
    if not isinstance(eps, float):
        raise TypeError("eps must be a float.")
    
    return torch.allclose(torch.mean(y), torch.zeros(1, dtype=y.dtype), atol=eps) and \
        torch.allclose(torch.var(y), torch.ones(1, dtype=y.dtype), atol=eps)

def uniformly_sampled_gaussian(num_rand: int):
    """
    Function to generate uniformly sampled gaussian values.
    """
    if not isinstance(num_rand, int):
        raise TypeError("num_rand must be an integer.")
    
    rand = 2 * (np.arange(num_rand) + 0.5) / float(num_rand) - 1
    return np.sqrt(2) * erfinv(rand)


def create_channelwise_variable(y: torch.Tensor, init: float):
    """
    Function to create a channel-wise variable.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")
    if not isinstance(init, float):
        raise TypeError("init must be a float.")
    
    num_channels = int(y.shape[-1])
    return nn.Parameter(init * torch.ones((1, 1, 1, num_channels), dtype=y.dtype))
