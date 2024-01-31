**Proxy Normalization in PyTorch**

*Welcome to this repository, which provides an unofficial PyTorch implementation of the paper "Proxy-Normalizing Activations to Match Batch Normalization while Removing Batch Dependence" by Labatie, A., Masters, D., Eaton-Rosen, Z., & Luschi, C. (2022, April 3).*

**Implementation Highlights**

*At the heart of this implementation is the ProxyNormalization class. This class embodies the Proxy Normalization technique, applying it to a specified 4-dimensional tensor.*

**Key Parameters**
	
***y**: This is a 4-dimensional torch.Tensor that is set to undergo proxy normalization. It's essential to note that this tensor should be pre-normalized before being passed to the ProxyNormalization class.*

***activation_fn**: This represents the activation function to be applied. It can be any callable function that applies an activation function to a tensor.*

***eps**: A minuscule floating-point value used to ensure numerical stability during the normalization process. As suggested by the original paper, this value should not be set too low. The default value is 0.03.*

***n_samples**: This represents the number of samples to be used in the proxy distribution. The default value is 256.*

***apply_activation**: A boolean flag that determines whether the activation function should be applied to the final output tensor, tilde_z. If set to True, the activation function specified by activation_fn is applied. As per the original paper, the default value is False.*

This implementation offers a flexible and efficient way to apply Proxy Normalization in PyTorch, providing control over the activation function, the number of samples in the proxy distribution, and the application of the activation function to the final output.
