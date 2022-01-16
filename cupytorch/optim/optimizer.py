from collections import defaultdict

from cupytorch.nn import Parameter


# simplify from https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
class Optimizer:

    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None

    def step(self):
        raise NotImplementedError

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param group must be a dict"
        param_group['params'] = list(param_group['params'])

        for param in param_group['params']:
            if not isinstance(param, Parameter):
                raise TypeError("optimizer can only optimize Parameters")
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        self.param_groups.append(param_group)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string
