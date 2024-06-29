"""
Applies the mish function element-wise:
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
"""

from jittor import nn
import jittor

jittor.flags.use_cuda = 1


class Mish(nn.Module):
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
        return jittor.nn.mish(input)
