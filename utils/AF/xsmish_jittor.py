"""
Script based on:
Wang, Xueliang, Honge Ren, and Achuan Wang.
 "Smish: A Novel Activation Function for Deep Learning Methods.
 " Electronics 11.4 (2022): 540.
smish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + sigmoid(x)))
"""

import jittor
from jittor import nn
import utils.AF.fsmish_jittor as func

jittor.flags.use_cuda = 1


class Smish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def execute(self, input):
        """
        Forward pass of the function.
        """
        return func.smish(input)
