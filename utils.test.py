import unittest
import numpy as np
import torch
from utils import uniformly_sampled_gaussian, create_channelwise_variable

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

if __name__ == '__main__':
    unittest.main()