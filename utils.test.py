import unittest
import numpy as np
import torch
from utils import normalize_tensor, tensor_is_normalized, uniformly_sampled_gaussian, create_channelwise_variable
from __init__ import ProxyNormalization

class TestNormalizeTensor(unittest.TestCase):
    def setUp(self):
        self.eps = 0.03

    def test_input_type(self):
        with self.assertRaises(TypeError):
            normalize_tensor('not a tensor', self.eps)
        with self.assertRaises(TypeError):
            normalize_tensor(torch.tensor([1, 2, 3]), 'not a float')
    
    def test_output_shape(self):
        y = torch.rand((1, 1, 1, 5))
        y = normalize_tensor(y, self.eps)
        self.assertEqual(y.shape, y.shape)

    def test_output_dtype(self):
        y = torch.rand((1, 1, 1, 5))
        y = normalize_tensor(y, self.eps)
        self.assertEqual(y.dtype, y.dtype)

    def test_tensor_is_normalized(self):
        y = torch.randn((1, 1, 1, 5))
        y = normalize_tensor(y, self.eps)
        self.assertTrue(tensor_is_normalized(y, self.eps))

    def test_tensor_is_not_normalized(self):
        y = torch.rand((1, 1, 1, 5))
        self.assertFalse(tensor_is_normalized(y, self.eps))

class TestTensorIsNormalized(unittest.TestCase):
    def setUp(self):
        self.eps = 0.03

    def test_tensor_is_normalized(self):
        # Create a tensor that is normalized
        y = torch.randn((1, 1, 1, 5))
        y = (y - torch.mean(y)) / torch.std(y)
        self.assertTrue(tensor_is_normalized(y, self.eps))

    def test_tensor_is_not_normalized(self):
        # Create a tensor that is not normalized
        y = torch.rand((1, 1, 1, 5))
        self.assertFalse(tensor_is_normalized(y, self.eps))

    def test_raises_type_error(self):
        # Test that a TypeError is raised if the first argument is not a tensor
        with self.assertRaises(TypeError):
            tensor_is_normalized(1.0, self.eps)
        # Test that a TypeError is raised if the second argument is not a float
        with self.assertRaises(TypeError):
            tensor_is_normalized(torch.tensor(1.0), "not a float")

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
        self.eps = 0.03
        self.y = normalize_tensor(torch.randn((1, 1, 1, 5)), self.eps)
        self.activation_fn = torch.relu
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