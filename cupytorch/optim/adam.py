import cupytorch as ct
from .optimizer import Optimizer


class Adam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    @ct.no_grad()
    def step(self):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    if weight_decay:
                        p.grad += weight_decay * p

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = ct.zeros_like(p)
                        state['exp_avg_sq'] = ct.zeros_like(p)

                    state['step'] += 1
                    state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * p.grad
                    state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * p.grad ** 2
                    exp_avg = state['exp_avg'] / (1 - beta1 ** state['step'])
                    exp_avg_sq = state['exp_avg_sq'] / (1 - beta2 ** state['step'])

                    p -= lr * exp_avg / (exp_avg_sq.sqrt() + eps)
