from math import sqrt

import cupytorch as ct
from cupytorch import Tensor
from .module import Module
from ..parameter import Parameter


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            ct.uniform(-1 / sqrt(in_features), 1 / sqrt(in_features), out_features, in_features))
        self.bias = Parameter(
            ct.uniform(-1 / sqrt(in_features), 1 / sqrt(in_features), out_features)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weight.T
        if self.bias:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features,
                                                                 self.bias is not None)
