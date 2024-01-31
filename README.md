Proxy Normalization in PyTorch
This repository contains an unofficial PyTorch implementation of the paper "Proxy-Normalizing Activations to Match Batch Normalization while Removing Batch Dependence" by Labatie, A., Masters, D., Eaton-Rosen, Z., & Luschi, C. (2022, April 3).

About the Implementation
The core of this implementation is the ProxyNormalization class. This class applies the Proxy Normalization technique to a given 4-dimensional tensor.

Key Parameters
y: This is a 4-dimensional torch.Tensor that will undergo proxy normalization. It's crucial to note that this tensor should already be normalized before being passed to the ProxyNormalization class.

activation_fn: This is the activation function to be applied. It can be any callable function that applies an activation function to a tensor.

eps: A small floating-point value used to ensure numerical stability during the normalization process. As per the original paper, this value should not be set too low. The default value is 0.03.

n_samples: The number of samples to be used in the proxy distribution. The default value is 256.

Added: apply_activation: A boolean flag that determines whether the activation function should be applied to the final output tensor, tilde_z. If set to True, the activation function specified by activation_fn is applied. The default value is False as per the original paper.
