from math import sqrt
from typing import Tuple, Optional

import cupytorch as ct
from cupytorch import Tensor, np
from ..parameter import Parameter
from .module import Module
from .activation import Sigmoid, Tanh


class LSTMCell(Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        std = -1 / sqrt(hidden_size)
        self.W_ii = Parameter(ct.uniform(-std, std, hidden_size, input_size))
        self.W_hi = Parameter(ct.uniform(-std, std, hidden_size, hidden_size))
        self.b_hi = Parameter(ct.uniform(-std, std, hidden_size, 1))
        self.W_if = Parameter(ct.uniform(-std, std, hidden_size, input_size))
        self.W_hf = Parameter(ct.uniform(-std, std, hidden_size, hidden_size))
        self.b_hf = Parameter(ct.uniform(-std, std, hidden_size, 1))
        self.W_ig = Parameter(ct.uniform(-std, std, hidden_size, input_size))
        self.W_hg = Parameter(ct.uniform(-std, std, hidden_size, hidden_size))
        self.b_hg = Parameter(ct.uniform(-std, std, hidden_size, 1))
        self.W_io = Parameter(ct.uniform(-std, std, hidden_size, input_size))
        self.W_ho = Parameter(ct.uniform(-std, std, hidden_size, hidden_size))
        self.b_ho = Parameter(ct.uniform(-std, std, hidden_size, 1))

    def forward(self, input: Tensor, hc: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, h, c = input.T, hc[0].T, hc[1].T
        i = Sigmoid()(self.W_ii @ x + self.W_hi @ h + self.b_hi)
        f = Sigmoid()(self.W_if @ x + self.W_hf @ h + self.b_hf)
        g = Tanh()(self.W_ig @ x + self.W_hg @ h + self.b_hg)
        o = Sigmoid()(self.W_io @ x + self.W_ho @ h + self.b_ho)
        c_ = f * c + i * g
        h_ = o * Tanh()(c_)
        return h_.T, c_.T

    def extra_repr(self) -> str:
        return 'input_size={}, hidden_size={}'.format(self.input_size, self.hidden_size)


class LSTM(Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, input: Tensor, hc: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if hc is None:
            h = c = ct.zeros(input.shape[1], self.hidden_size)
        else:
            h, c = hc
        output = []
        for i in range(input.shape[0]):
            h, c = self.cell(input[i], (h, c))
            output.append(h)
        output = ct.stack(output)
        return output, (h, c)

    def extra_repr(self) -> str:
        return 'input_size={}, hidden_size={}'.format(self.input_size, self.hidden_size)
