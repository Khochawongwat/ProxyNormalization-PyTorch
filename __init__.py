import numpy as np
import torch
from torch import nn

from utils import create_channelwise_variable, uniformly_sampled_gaussian

class ProxyNormalization(nn.Module): 
    """
    Proxy Normalization class.
    """
    def __init__(self, y: torch.Tensor, activation_fn: callable, eps: float, n_samples: int):
        super().__init__()
        self.y = y
        self.activation_fn = activation_fn
        self.eps = eps
        self.n_samples = n_samples

    def forward(self) -> torch.Tensor:
        beta = create_channelwise_variable(self.y, 0)
        gamma = create_channelwise_variable(self.y, 1)

        z = self.activation_fn(gamma * self.y.add_(beta))
        proxy_y = torch.tensor(uniformly_sampled_gaussian(self.n_samples), dtype=self.y.dtype)
        proxy_y = proxy_y.view(self.n_samples, 1, 1, 1)
        proxy_z = self.activation_fn(gamma * proxy_y.add_(beta))

        proxy_mean = torch.mean(proxy_z, dim=0, keepdim=True)
        proxy_var = torch.var(proxy_z, dim=0, unbiased=False, keepdim=True)
        inv_proxy_std = torch.rsqrt(proxy_var + self.eps)

        tilde_z = (z.sub_(proxy_mean)).mul_(inv_proxy_std)

        return tilde_z