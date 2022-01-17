from math import prod
from typing import Union, Tuple, Sequence, Dict, Any, Optional

import cupytorch as ct
from cupytorch import np
from .grad_mode import get_grad_enabled
from ..tensor import Tensor


# copy from https://github.com/zhouzaida/minitorch/blob/master/minitorch/autograd/node.py
def unbroadcast(grad_input: Tensor, input: Tensor) -> Tensor:
    if grad_input.shape == input.shape:
        return grad_input
    data = grad_input.data
    for _ in range(len(grad_input.shape) - len(input.shape)):
        data = data.sum(axis=0)
    for i, dim in enumerate(input.shape):
        if dim == 1:
            data = data.sum(axis=i, keepdims=True)
    return ct.tensor(data)


class Function:

    def __init__(self):
        self.next_functions: Tuple[Optional[Function], ...] = ()
        self.ctx: Dict[str, Union[Tuple[Tensor, ...], Tensor, Any]] = {}

    def __call__(self, *inputs: Any) -> Tensor:
        self.ctx['output'] = self.forward(*inputs)
        return self.ctx['output']

    def forward(self, *inputs: Any) -> Union[Tensor, Tuple[Tensor, ...]]:
        raise NotImplementedError

    def backward(self, grad_output: Tensor) -> Union[Tensor, Tuple[Optional[Tensor], ...]]:
        raise NotImplementedError

    def single_helper(self, data: np.ndarray, t: Tensor) -> Tensor:
        requires_grad = get_grad_enabled() and t.requires_grad
        if requires_grad:
            self.ctx['input'] = t
            self.next_functions = ((AccumulateGrad(t) if t.is_leaf and t.requires_grad else t.grad_fn, 0),)
        return Tensor(data, requires_grad=requires_grad, grad_fn=self if requires_grad else None)

    def multiple_helper(self, data: np.ndarray, *tensors: Tensor) -> Tensor:
        requires_grad = get_grad_enabled() and any(t.requires_grad for t in tensors)
        if requires_grad:
            self.ctx['inputs'] = tensors
            self.next_functions = tuple((AccumulateGrad(t) if t.is_leaf and t.requires_grad else t.grad_fn, 0)
                                        for t in tensors)
        return Tensor(data, requires_grad=requires_grad, grad_fn=self if requires_grad else None)


class AccumulateGrad(Function):

    def __init__(self, t: Tensor):
        super().__init__()
        self.variable = t

    def forward(self, t: Tensor) -> None:
        pass

    def backward(self, grad_output: Tensor) -> None:
        pass


class ViewBackward(Function):

    def forward(self, t: Tensor, *shape) -> Tensor:
        return self.single_helper(t.data.reshape(*shape), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output.reshape(*self.ctx['input'].shape)


class SliceBackward(Function):

    def forward(self, t: Tensor, key) -> Tensor:
        self.ctx['key'] = key
        return self.single_helper(t.data[key], t)

    def backward(self, grad_output: Tensor) -> Tensor:
        g = ct.zeros_like(self.ctx['input'])
        g[self.ctx['key']] = grad_output
        return g


class AbsBackward(Function):

    def forward(self, t: Tensor) -> Tensor:
        return self.single_helper(np.abs(t.data), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return ct.tensor(np.sign(self.ctx['input'].data) * grad_output.data)


class NegBackward(Function):

    def forward(self, t: Tensor) -> Tensor:
        return self.single_helper(-t.data, t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return -grad_output


class TBackward(Function):

    def forward(self, t: Tensor) -> Tensor:
        return self.single_helper(t.data.T, t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output.T


class ExpBackward(Function):

    def forward(self, t: Tensor) -> Tensor:
        return self.single_helper(np.exp(t.data), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return self.ctx['output'] * grad_output


class LogBackward(Function):

    def forward(self, t: Tensor) -> Tensor:
        return self.single_helper(np.log(t.data), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return grad_output / self.ctx['input']


class SqrtBackward(Function):

    def forward(self, t: Tensor) -> Tensor:
        return self.single_helper(np.sqrt(t.data), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return 0.5 / self.ctx['output']


class AddBackward(Function):

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return self.multiple_helper(t1.data + t2.data, t1, t2)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        t1, t2 = self.ctx['inputs']
        g1 = unbroadcast(grad_output, t1) if t1.requires_grad else None
        g2 = unbroadcast(grad_output, t2) if t2.requires_grad else None
        return g1, g2


class SubBackward(Function):

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return self.multiple_helper(t1.data - t2.data, t1, t2)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        t1, t2 = self.ctx['inputs']
        g1 = unbroadcast(grad_output, t1) if t1.requires_grad else None
        g2 = unbroadcast(-grad_output, t2) if t2.requires_grad else None
        return g1, g2


class MulBackward(Function):

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return self.multiple_helper(t1.data * t2.data, t1, t2)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        t1, t2 = self.ctx['inputs']
        g1 = unbroadcast(t2 * grad_output, t1) if t1.requires_grad else None
        g2 = unbroadcast(t1 * grad_output, t2) if t2.requires_grad else None
        return g1, g2


class DivBackward(Function):

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return self.multiple_helper(t1.data / t2.data, t1, t2)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        t1, t2 = self.ctx['inputs']
        g1 = unbroadcast(grad_output / t2, t1) if t1.requires_grad else None
        g2 = unbroadcast(-t1 * grad_output / t2 ** 2, t2) if t2.requires_grad else None
        return g1, g2


class MatmulBackward(Function):

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return self.multiple_helper(t1.data @ t2.data, t1, t2)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        t1, t2 = self.ctx['inputs']
        g1 = grad_output @ t2.T if t1.requires_grad else None
        g2 = t1.T @ grad_output if t2.requires_grad else None
        return g1, g2


class PowBackward(Function):

    def forward(self, t: Tensor, exp: Union[int, float]) -> Tensor:
        self.ctx['exp'] = exp
        return self.single_helper(t.data ** exp, t)

    def backward(self, grad_output: Tensor) -> Tensor:
        t, exp = self.ctx['input'], self.ctx['exp']
        return grad_output * exp * t ** (exp - 1)


class SumBackward(Function):

    def forward(self, t: Tensor, dim: Optional[int] = None, keepdim=False) -> Tensor:
        return self.single_helper(t.data.sum(dim, keepdims=keepdim), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        return ct.tensor(np.broadcast_to(grad_output.data, self.ctx['input'].shape))


class MeanBackward(Function):

    def forward(self, t: Tensor, dim: Optional[int] = None, keepdim=False) -> Tensor:
        return self.single_helper(t.data.mean(dim, keepdims=keepdim), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        shape = self.ctx['input'].shape
        return ct.tensor(prod(grad_output.shape) / prod(shape) * np.broadcast_to(grad_output.data, shape))


class MaxBackward(Function):

    def forward(self, t: Tensor, dim: Optional[int] = None, keepdim=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        values = t.data.max(dim, keepdims=keepdim)
        if dim is None:
            return self.single_helper(values, t)
        self.ctx['dim'] = dim
        indices = t.data.argmax(dim)
        self.ctx['indices'] = np.expand_dims(indices, dim)
        if keepdim:
            indices = self.ctx['indices']
        return self.single_helper(values, t), ct.tensor(indices)

    def backward(self, grad_output: Tensor) -> Tensor:
        if isinstance(self.ctx['output'], Tensor):
            mask = self.ctx['input'] == self.ctx['output']
            return mask / mask.sum() * grad_output
        elif isinstance(self.ctx['output'], tuple):
            t = self.ctx['input']
            dim = self.ctx['dim']
            mask = np.indices(t.shape)[dim] == self.ctx['indices']
            g = ct.zeros_like(t)
            g[mask] = grad_output.view(-1)
            return g


class MinBackward(Function):

    def forward(self, t: Tensor, dim: Optional[int] = None, keepdim=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        values = t.data.min(dim, keepdims=keepdim)
        if dim is None:
            return self.single_helper(values, t)
        self.ctx['dim'] = dim
        indices = t.data.argmin(dim)
        self.ctx['indices'] = np.expand_dims(indices, dim)
        if keepdim:
            indices = self.ctx['indices']
        return self.single_helper(values, t), ct.tensor(indices)

    def backward(self, grad_output: Tensor) -> Tensor:
        if isinstance(self.ctx['output'], Tensor):
            mask = self.ctx['input'] == self.ctx['output']
            return mask / mask.sum() * grad_output
        elif isinstance(self.ctx['output'], tuple):
            t = self.ctx['input']
            dim = self.ctx['dim']
            mask = np.indices(t.shape)[dim] == self.ctx['indices']
            g = ct.zeros_like(t)
            g[mask] = grad_output.view(-1)
            return g


class AmaxBackward(Function):

    def forward(self, t: Tensor, dim: Optional[int] = None, keepdim=False) -> Tensor:
        self.ctx['dim'] = dim
        return self.single_helper(t.data.max(dim, keepdims=keepdim), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        mask = self.ctx['output'] == self.ctx['input']
        return mask / mask.sum(self.ctx['dim'], True) * grad_output


class AminBackward(Function):

    def forward(self, t: Tensor, dim: Optional[int] = None, keepdim=False) -> Tensor:
        self.ctx['dim'] = dim
        return self.single_helper(t.data.min(dim, keepdims=keepdim), t)

    def backward(self, grad_output: Tensor) -> Tensor:
        mask = (self.ctx['output'] == self.ctx['input'])
        return mask / mask.sum(self.ctx['dim'], True) * grad_output


class MaximumBackward(Function):

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return self.multiple_helper(np.maximum(t1.data, t2.data), t1, t2)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        t1, t2 = self.ctx['inputs']
        mask = (self.ctx['output'] == t1).data
        g1 = unbroadcast(ct.tensor(np.where(mask, grad_output.data, 0)), t1) if t1.requires_grad else None
        g2 = unbroadcast(ct.tensor(np.where(~mask, grad_output.data, 0)), t2) if t2.requires_grad else None
        return g1, g2


class MinimumBackward(Function):

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return self.multiple_helper(np.minimum(t1.data, t2.data), t1, t2)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        t1, t2 = self.ctx['inputs']
        mask = (self.ctx['output'] == t1).data
        g1 = unbroadcast(ct.tensor(np.where(mask, grad_output.data, 0)), t1) if t1.requires_grad else None
        g2 = unbroadcast(ct.tensor(np.where(~mask, grad_output.data, 0)), t2) if t2.requires_grad else None
        return g1, g2


class StackBackward(Function):

    def forward(self, tensors: Sequence[Tensor], dim=0) -> Tensor:
        self.ctx['dim'] = dim
        return self.multiple_helper(np.stack([t.data for t in tensors]), *tensors)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        dim = self.ctx['dim']
        return tuple(grad_output[tuple(slice(d) if i != dim else j for i, d in enumerate(grad_output.shape))]
                     for j in range(len(self.ctx['inputs'])))
