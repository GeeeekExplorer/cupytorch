from cupytorch import Tensor
from .activation import LogSoftmax
from .module import Module


class L1Loss(Module):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        output = (input - target).abs()
        return output.mean()


class MSELoss(Module):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        output = (input - target) ** 2
        return output.mean()


class NLLLoss(Module):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        output = input[range(input.shape[0]), target]
        return -output.mean()


class CrossEntropyLoss(Module):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return NLLLoss()(LogSoftmax(1)(input), target)
