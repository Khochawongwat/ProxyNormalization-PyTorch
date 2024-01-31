import unittest
import numpy as np
import torch
from utils import uniformly_sampled_gaussian, create_channelwise_variable
from __init__ import ProxyNormalization

class TestUniformlySampledGaussian(unittest.TestCase):
    def test_input_type(self):
        with self.assertRaises(TypeError):
            uniformly_sampled_gaussian('not an integer')

    def test_output_shape(self):
        num_rand = 10
        result = uniformly_sampled_gaussian(num_rand)
        self.assertEqual(result.shape, (num_rand,))

class TestCreateChannelwiseVariable(unittest.TestCase):
    def test_input_type(self):
        with self.assertRaises(TypeError):
            create_channelwise_variable('not a tensor', 0.1)
        with self.assertRaises(TypeError):
            create_channelwise_variable(torch.tensor([1, 2, 3]), 'not a float')

    def test_output_shape(self):
        y = torch.rand((1, 1, 1, 5))
        init = 0.1
        result = create_channelwise_variable(y, init)
        self.assertEqual(result.shape, (1, 1, 1, y.shape[-1]))

class TestProxyNormalization(unittest.TestCase):
    def setUp(self):
        self.y = torch.randn(1, 1, 1, 1)
        self.activation_fn = torch.relu
        self.eps = 0.03
        self.n_samples = 256
        self.proxy_norm = ProxyNormalization(self.y, self.activation_fn, self.eps, self.n_samples)

    def test_initialization(self):
        self.assertEqual(self.proxy_norm.y.shape, self.y.shape)
        self.assertEqual(self.proxy_norm.activation_fn, self.activation_fn)
        self.assertEqual(self.proxy_norm.eps, self.eps)
        self.assertEqual(self.proxy_norm.n_samples, self.n_samples)

    def test_forward_output_shape(self):
        output = self.proxy_norm.forward()
        self.assertEqual(output.shape, self.y.shape)

    #Relu tests
    def test_forward_output_dtype(self):
        output = self.proxy_norm.forward()
        self.assertEqual(output.dtype, self.y.dtype)

    #Relu must not output negative values
    def test_forward_output_values(self):
        output = self.proxy_norm.forward()
        self.assertTrue(torch.all(output >= 0))

    def test_forward_output_not_nan(self):
        output = self.proxy_norm.forward()
        self.assertFalse(torch.isnan(output).any())

if __name__ == "__main__":
    unittest.main()