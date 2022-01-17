from math import prod
from typing import Union, Tuple, Sequence, Optional

from cupytorch import np


class Tensor:
    from numpy.typing import ArrayLike
    TensorLike = Union[ArrayLike, 'Tensor']

    def __init__(self, data: ArrayLike, *, dtype=None, requires_grad=False, grad_fn=None):
        if dtype is None:
            dtype = str(np.array(data).dtype)
            if 'float' in dtype:
                dtype = 'float32'
        assert dtype in ('float32', 'int64', 'bool'), f"cupytorch only supports float32, int64 and bool"
        self.data: np.ndarray = np.asarray(data, dtype)
        assert not requires_grad or 'float' in self.dtype, "only tensors of floating point dtype can require gradients"
        self.grad: Optional[Tensor] = None
        self.grad_fn: Optional[Function] = grad_fn if requires_grad else None
        self.is_leaf = False
        self.requires_grad = requires_grad

    def __repr__(self):
        s = '\n'.join(' ' * 7 + s if i else s for i, s in enumerate(str(self.data).split('\n')))
        return f"tensor({s}" + \
               (f", grad_fn=<{type(self.grad_fn).__name__}>" if self.grad_fn else "") + \
               (f", requires_grad={self.requires_grad}" if self.grad_fn is None and self.requires_grad else "") + ")"

    def __hash__(self):  # because __eq__ definition
        return id(self)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> str:
        return str(self.data.dtype)

    @property
    def T(self) -> 'Tensor':
        return self.t()

    @classmethod
    def tensor(cls, data: TensorLike, *, dtype=None, requires_grad=False) -> 'Tensor':
        if isinstance(data, Tensor):
            return data
        t = Tensor(data, dtype=dtype, requires_grad=requires_grad)
        t.is_leaf = True
        t.requires_grad = requires_grad
        return t

    @classmethod
    def ones(cls, *size, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.tensor(np.ones(size), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, input: 'Tensor', *, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.ones(*input.shape, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros(cls, *size, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.tensor(np.zeros(size), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, input: 'Tensor', *, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.zeros(*input.shape, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def eye(cls, n: int, m=None, *, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.tensor(np.eye(n, m), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def rand(cls, *size, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.tensor(np.random.rand(*size), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def rand_like(cls, input: 'Tensor', *, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.rand(*input.shape, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def randn(cls, *size, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.tensor(np.random.randn(*size), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def randn_like(cls, input: 'Tensor', *, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.randn(*input.shape, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def uniform(cls, low=0, high=1, *size, dtype=None, requires_grad=True) -> 'Tensor':
        return cls.tensor(np.random.uniform(low, high, size), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def arange(cls, start=0, end=None, step=1, *, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.tensor(np.arange(start, end, step), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def randperm(cls, n, *, dtype=None, requires_grad=False) -> 'Tensor':
        return cls.tensor(np.random.permutation(n), dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def stack(cls, tensors: Sequence['Tensor'], dim=0):
        return StackBackward()(tensors, dim)

    def __len__(self):
        return len(self.data)

    def backward(self) -> None:
        backward(self)

    def t(self) -> 'Tensor':
        return TBackward()(self)

    def reshape(self, *shape) -> 'Tensor':
        return ViewBackward()(self, *shape)

    view = reshape

    def item(self):
        assert prod(self.shape) == 1
        return self.data.item()

    def __getitem__(self, key) -> 'Tensor':
        if isinstance(key, Tensor):
            key = key.data
        if isinstance(key, tuple):
            key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
        return SliceBackward()(self, key)

    def __setitem__(self, key, value: TensorLike) -> None:
        assert not self.is_leaf or not self.requires_grad
        if isinstance(key, Tensor):
            key = key.data
        if isinstance(key, tuple):
            key = tuple(k.data if isinstance(k, Tensor) else k for k in key)
        self.data[key] = self.tensor(value).data

    def abs(self) -> 'Tensor':
        return AbsBackward()(self)

    def neg(self) -> 'Tensor':
        return NegBackward()(self)

    def exp(self) -> 'Tensor':
        return ExpBackward()(self)

    def log(self) -> 'Tensor':
        return LogBackward()(self)

    def sqrt(self) -> 'Tensor':
        return SqrtBackward()(self)

    def add(self, other: TensorLike) -> 'Tensor':
        return AddBackward()(self, self.tensor(other))

    def sub(self, other: TensorLike) -> 'Tensor':
        return SubBackward()(self, self.tensor(other))

    def mul(self, other: TensorLike) -> 'Tensor':
        return MulBackward()(self, self.tensor(other))

    def div(self, other: TensorLike) -> 'Tensor':
        return DivBackward()(self, self.tensor(other))

    def matmul(self, other: TensorLike) -> 'Tensor':
        return MatmulBackward()(self, self.tensor(other))

    def pow(self, exp: Union[int, float]) -> 'Tensor':
        return PowBackward()(self, exp)

    def sum(self, dim: Optional[int] = None, keepdim=False) -> 'Tensor':
        return SumBackward()(self, dim, keepdim)

    def mean(self, dim: Optional[int] = None, keepdim=False) -> 'Tensor':
        return MeanBackward()(self, dim, keepdim)

    def max(self, dim: Optional[int] = None, keepdim=False) -> Union['Tensor', Tuple['Tensor', 'Tensor']]:
        return MaxBackward()(self, dim, keepdim)

    def min(self, dim: Optional[int] = None, keepdim=False) -> Union['Tensor', Tuple['Tensor', 'Tensor']]:
        return MinBackward()(self, dim, keepdim)

    def amax(self, dim: Optional[int] = None, keepdim=False) -> 'Tensor':
        return AmaxBackward()(self, dim, keepdim)

    def amin(self, dim: Optional[int] = None, keepdim=False) -> 'Tensor':
        return AminBackward()(self, dim, keepdim)

    def argmax(self, dim: Optional[int] = None, keepdim=False) -> 'Tensor':
        return self.tensor(np.argmax(self.data, dim, keepdims=keepdim))

    def argmin(self, dim: Optional[int] = None, keepdim=False) -> 'Tensor':
        return self.tensor(np.argmin(self.data, dim, keepdims=keepdim))

    def maximum(self, other: TensorLike) -> 'Tensor':
        return MaximumBackward()(self, self.tensor(other))

    def minimum(self, other: TensorLike) -> 'Tensor':
        return MinimumBackward()(self, self.tensor(other))

    def gt(self, other: TensorLike) -> 'Tensor':
        return self.tensor(self.data > self.tensor(other).data)

    def lt(self, other: TensorLike) -> 'Tensor':
        return self.tensor(self.data < self.tensor(other).data)

    def eq(self, other: TensorLike) -> 'Tensor':
        return self.tensor(self.data == self.tensor(other).data)

    def ge(self, other: TensorLike) -> 'Tensor':
        return self.tensor(self.data >= self.tensor(other).data)

    def le(self, other: TensorLike) -> 'Tensor':
        return self.tensor(self.data <= self.tensor(other).data)

    def ne(self, other: TensorLike) -> 'Tensor':
        return self.tensor(self.data != self.tensor(other).data)

    __abs__ = abs
    __neg__ = neg
    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __truediv__ = div
    __matmul__ = matmul
    __pow__ = pow
    __gt__ = gt
    __lt__ = lt
    __eq__ = eq
    __ge__ = ge
    __le__ = le
    __ne__ = ne

    def __radd__(self, other: TensorLike) -> 'Tensor':
        return self.tensor(other).add(self)

    def __rsub__(self, other: TensorLike) -> 'Tensor':
        return self.tensor(other).sub(self)

    def __rmul__(self, other: TensorLike) -> 'Tensor':
        return self.tensor(other).mul(self)

    def __rtruediv__(self, other: TensorLike) -> 'Tensor':
        return self.tensor(other).div(self)

    def __rmatmul__(self, other: TensorLike) -> 'Tensor':
        return self.tensor(other).matmul(self)

    def __iadd__(self, other: TensorLike):
        self.data += self.tensor(other).data
        return self

    def __isub__(self, other: TensorLike):
        self.data -= self.tensor(other).data
        return self

    def __imul__(self, other: TensorLike):
        self.data *= self.tensor(other).data
        return self

    def __itruediv__(self, other: TensorLike):
        self.data /= self.tensor(other).data
        return self

    def __imatmul__(self, other: TensorLike):
        self.data @= self.tensor(other).data
        return self

    def __and__(self, other: TensorLike) -> 'Tensor':
        other = self.tensor(other)
        assert 'float' not in self.dtype + other.dtype
        return self.tensor(self.data & other.data)

    def __or__(self, other: TensorLike) -> 'Tensor':
        other = self.tensor(other)
        assert 'float' not in self.dtype + other.dtype
        return self.tensor(self.data | other.data)

    def __xor__(self, other: TensorLike) -> 'Tensor':
        other = self.tensor(other)
        assert 'float' not in self.dtype + other.dtype
        return self.tensor(self.data ^ other.data)

    def __invert__(self) -> 'Tensor':
        assert 'float' not in self.dtype
        return self.tensor(~self.data)


from .autograd import *  # avoid circular import
