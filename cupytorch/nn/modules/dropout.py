import cupytorch as ct

from cupytorch import Tensor
from .module import Module


class Dropout(Module):

    def __init__(self, p=0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            mask = ct.rand_like(input) > self.p
            return mask * input / (1 - self.p)
        else:
            return input

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)
