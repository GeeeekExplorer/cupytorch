import cupytorch as ct
from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, params, lr, momentum=0, weight_decay=0) -> None:
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @ct.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    if weight_decay:
                        p.grad += weight_decay * p

                    if momentum:
                        state = self.state[p]
                        if 'momentum_buffer' in state:
                            p.grad += momentum * state['momentum_buffer']
                        state['momentum_buffer'] = p.grad

                    p -= lr * p.grad
