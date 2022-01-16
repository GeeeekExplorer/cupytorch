from math import sqrt, pi

from cupytorch import Tensor
from .module import Module


class ReLU(Module):

    def forward(self, input: Tensor) -> Tensor:
        return input.maximum(0.)


class GeLU(Module):

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1 + Tanh()(sqrt(2 / pi) * (input + 0.044715 * input ** 3)))


class Sigmoid(Module):

    def forward(self, input: Tensor) -> Tensor:
        return 1 / (1 + (-input).exp())


class Tanh(Module):

    def forward(self, input: Tensor) -> Tensor:
        e1, e2 = input.exp(), (-input).exp()
        return (e1 - e2) / (e1 + e2)


class Softmax(Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        # output = input - input.amax(self.dim, True)
        output = input.exp()
        return output / (output.sum(self.dim, True))


class LogSoftmax(Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return Softmax(self.dim)(input).log()
