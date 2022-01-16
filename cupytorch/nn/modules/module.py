from collections import OrderedDict
from typing import Union, Tuple, Any, Iterator

from cupytorch import Tensor
from ..parameter import Parameter


# simplify from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
class Module:

    def __init__(self, *args, **kwargs):
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def train(self):
        self.training = True
        for _, module in self._modules.items():
            module.train()

    def eval(self):
        self.training = True
        for _, module in self._modules.items():
            module.eval()

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *inputs: Tensor, **kwargs) -> Any:
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        _parameters = self.__dict__['_parameters']
        if name in _parameters:
            return _parameters[name]
        _modules = self.__dict__['_modules']
        if name in _modules:
            return _modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name: str):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def parameters(self, recurse=True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix='', recurse=True) -> Iterator[Tuple[str, Parameter]]:
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            for name, param in module._parameters.items():
                name = module_prefix + ('.' if module_prefix else '') + name
                yield name, param

    def modules(self) -> Iterator['Module']:
        for name, module in self.named_modules():
            yield module

    def named_modules(self, prefix=''):
        yield prefix, self
        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            yield from module.named_modules(submodule_prefix)

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        extra_repr = self.extra_repr()
        extra_lines = extra_repr.split('\n') if extra_repr else []

        child_lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            if '\n' in mod_str:
                mod_lines = mod_str.split('\n')
                mod_str = '\n'.join([mod_lines[0]] + ['  ' + line for line in mod_lines[1:]])
            child_lines.append('(' + name + '): ' + mod_str)

        lines = extra_lines + child_lines
        main_str = self.__class__.__name__ + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str
