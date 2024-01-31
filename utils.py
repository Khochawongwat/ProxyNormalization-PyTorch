from scipy.special import erfinv
import numpy as np
import torch
from torch import nn

def uniformly_sampled_gaussian(num_rand: int) -> np.ndarray:
    """
    Function to generate uniformly sampled gaussian values.
    """
    if not isinstance(num_rand, int):
        raise TypeError("num_rand must be an integer.")
    
    rand = 2 * (np.arange(num_rand) + 0.5) / float(num_rand) - 1
    return np.sqrt(2) * erfinv(rand)


def create_channelwise_variable(y: torch.Tensor, init: float) -> nn.Parameter:
    """
    Function to create a channel-wise variable.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")
    if not isinstance(init, float):
        raise TypeError("init must be a float.")
    
    num_channels = y.shape[-1]
    return nn.Parameter(init * torch.ones((1, 1, 1, num_channels), dtype=y.dtype))
