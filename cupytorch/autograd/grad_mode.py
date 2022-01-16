from functools import wraps
from typing import Any

grad_enabled = True


def set_grad_enabled(val: bool) -> None:
    global grad_enabled
    grad_enabled = val


def get_grad_enabled() -> bool:
    return grad_enabled


class no_grad:

    def __enter__(self) -> None:
        self.prev = grad_enabled
        set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_grad_enabled(self.prev)

    def __call__(self, func):
        @wraps(func)
        def no_grad_func(*args, **kwargs):
            self.prev = grad_enabled
            set_grad_enabled(False)
            ret = func(*args, **kwargs)
            set_grad_enabled(self.prev)
            return ret

        return no_grad_func


class enable_grad:

    def __enter__(self) -> None:
        self.prev = grad_enabled
        set_grad_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        set_grad_enabled(self.prev)

    def __call__(self, func):
        @wraps(func)
        def no_grad_func(*args, **kwargs):
            self.prev = grad_enabled
            set_grad_enabled(False)
            ret = func(*args, **kwargs)
            set_grad_enabled(self.prev)
            return ret

        return no_grad_func
