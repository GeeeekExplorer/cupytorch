from ..tensor import Tensor
from .function import Function, AccumulateGrad, ViewBackward, SliceBackward, AbsBackward, NegBackward, TBackward, \
    ExpBackward, LogBackward, SqrtBackward, AddBackward, SubBackward, MulBackward, DivBackward, MatmulBackward, \
    PowBackward, SumBackward, MeanBackward, MaxBackward, MinBackward, AmaxBackward, AminBackward, MaximumBackward, \
    MinimumBackward, StackBackward
from .grad_mode import no_grad, enable_grad


@no_grad()
def backward(t: Tensor, grad=None) -> None:
    from math import prod
    from collections import defaultdict, deque
    assert t.requires_grad, "tensor does not require grad and does not have a grad_fn"
    assert prod(t.shape) == 1 or grad is not None, "grad can be implicitly created only for scalar outputs"

    # modified from https://github.com/geohot/tinygrad/blob/master/tinygrad/tensor.py
    # can't free intermediate variable
    # order = []
    # visit = set()
    # def traverse(x: Tensor):
    #     visit.add(x)
    #     if x.grad_fn:
    #         inputs = (x.grad_fn.ctx['input'],) if 'input' in x.grad_fn.ctx else x.grad_fn.ctx['inputs']
    #         [traverse(y) for y in inputs if y not in visit]
    #         order.append(x)
    # traverse(t)
    # t.grad = Tensor.ones_like(t)
    # for x in reversed(order):
    #     grad_inputs = x.grad_fn.backward(x.grad)
    #     if not isinstance(grad_inputs, tuple):
    #         grad_inputs = (grad_inputs,)
    #     inputs = (x.grad_fn.ctx['input'],) if 'input' in x.grad_fn.ctx else x.grad_fn.ctx['inputs']
    #     for y, g in zip(inputs, grad_inputs):
    #         if y.requires_grad:
    #             y.grad = g if y.grad is None else y.grad + g

    # standard toposort implemented by myself, also works well
    # and can free intermediate variable to reduce memory usage
    d = defaultdict(int)
    q = deque([t])
    while q:
        x = q.popleft()
        inputs = (x.grad_fn.ctx['input'],) if 'input' in x.grad_fn.ctx else x.grad_fn.ctx['inputs']
        for y in inputs:
            if y.grad_fn:  # exclude leaf nodes
                if d[y] == 0:  # have not visited
                    q.append(y)
                d[y] += 1
    t.grad = Tensor.ones_like(t) if grad is None else grad
    q = deque([t])
    while q:
        x = q.popleft()
        grad_inputs = x.grad_fn.backward(x.grad)
        del x.grad
        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        inputs = (x.grad_fn.ctx['input'],) if 'input' in x.grad_fn.ctx else x.grad_fn.ctx['inputs']
        for y, g in zip(inputs, grad_inputs):
            if y.requires_grad:  # include leaf nodes
                assert y.shape == g.shape, "shapes of tensor and grad mismatch"
                y.grad = g if y.grad is None else y.grad + g
                if d[y] == 1:  # exclude leaf nodes
                    q.append(y)
                d[y] -= 1
