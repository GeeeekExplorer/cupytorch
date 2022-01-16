from .optimizer import Optimizer


class _LRScheduler(object):

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.last_epoch = -1

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr


class LambdaLR(_LRScheduler):

    def __init__(self, optimizer, lr_lambda):
        super().__init__(optimizer)
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            assert len(lr_lambda) == len(optimizer.param_groups)
            self.lr_lambdas = list(lr_lambda)

    def get_lr(self):
        return [base_lr * func(self.last_epoch) for func, base_lr in zip(self.lr_lambdas, self.base_lrs)]


class StepLR(_LRScheduler):

    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
