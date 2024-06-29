"""
Script provides functional interface for Mish activation function.
"""

import jittor

jittor.flags.use_cuda = 1


def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return input * jittor.tanh(jittor.nn.softplus(input))
