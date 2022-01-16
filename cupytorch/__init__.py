try:
    import cupy as np

    np.array(0)
    print('use cupy backend')
except:
    import numpy as np

    print('use numpy backend')

from .tensor import Tensor
from .autograd import no_grad, enable_grad
from . import autograd, nn, optim, utils

# vars().update(getmembers(Tensor, ismethod))
tensor = Tensor.tensor
ones = Tensor.ones
ones_like = Tensor.ones_like
zeros = Tensor.zeros
zeros_like = Tensor.zeros_like
eye = Tensor.eye
rand = Tensor.rand
rand_like = Tensor.rand_like
randn = Tensor.randn
randn_like = Tensor.randn_like
uniform = Tensor.uniform
arange = Tensor.arange
randperm = Tensor.randperm
stack = Tensor.stack
