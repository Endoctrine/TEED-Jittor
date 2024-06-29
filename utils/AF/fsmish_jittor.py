"""
Script based on:
Wang, Xueliang, Honge Ren, and Achuan Wang.
 "Smish: A Novel Activation Function for Deep Learning Methods.
 " Electronics 11.4 (2022): 540.
"""

import jittor

jittor.flags.use_cuda = 1


def smish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    return input * jittor.tanh(jittor.log(jittor.sigmoid(input).add(1)))
