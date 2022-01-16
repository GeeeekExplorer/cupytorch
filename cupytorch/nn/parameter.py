from cupytorch import Tensor


class Parameter(Tensor):

    def __init__(self, tensor: Tensor, requires_grad=True):
        super().__init__(tensor.data, requires_grad=requires_grad)
        self.is_leaf = True
