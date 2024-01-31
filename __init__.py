import numpy as np
import torch
from torch import nn

from utils import create_channelwise_variable, uniformly_sampled_gaussian, tensor_is_normalized

class ProxyNormalization(nn.Module): 
    """
    Proxy Normalization class.
    """
    def __init__(self, y: torch.Tensor, activation_fn: callable, eps: float, n_samples: int, apply_activation: bool = True):
        super().__init__()

        if(y.ndim != 4):
            raise ValueError("y must be a 4-dimensional tensor.")
        
        if y.shape == (1, 1, 1, 1):
            raise ValueError("y is a scalar. Proxy Normalization will not be applied.")
            
        if(not tensor_is_normalized(y, eps)):
            raise ValueError("y must be normalized.")
        
        self.y = y
        self.activation_fn = activation_fn
        self.eps = eps
        self.n_samples = n_samples
        self.apply_activation = apply_activation

    def forward(self) -> torch.Tensor:
        beta = create_channelwise_variable(self.y, 0.0)
        gamma = create_channelwise_variable(self.y, 1.0)
        
        #Affine transform equation =  gamma * n + beta

        #affine transform on y and apply activation function
        z = self.activation_fn(gamma * self.y.add_(beta))

        #Create a proxy distribution of y
        proxy_y = torch.tensor(uniformly_sampled_gaussian(self.n_samples), dtype=self.y.dtype)

        proxy_y = torch.reshape(proxy_y, (self.n_samples, 1, 1, 1))
        
        #Affine transform on proxy of y and apply activation function
        proxy_z = self.activation_fn((gamma * proxy_y).add_(beta))
        proxy_z = proxy_z.type(self.y.dtype)
        
        proxy_mean = torch.mean(proxy_z, dim=0, keepdim=True)
        proxy_var = torch.var(proxy_z, dim=0, unbiased=False, keepdim=True)
        
        proxy_mean = proxy_mean.type(self.y.dtype)
        
        inv_proxy_std = torch.rsqrt(proxy_var + self.eps)
        inv_proxy_std = inv_proxy_std.type(self.y.dtype)

        tilde_z = (z - proxy_mean)  * inv_proxy_std

        #Not sure if I should apply activation function here
        if self.apply_activation:
            tilde_z = self.activation_fn(tilde_z)

        return tilde_z