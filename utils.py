import torch
import numpy as np
import random
from collections.abc import Callable
from typing import Type, TypeVar

T = TypeVar("T")


def abstract_class_property(*names: str) -> Callable[[Type[T], ...], Type[T]]:
    def _func(cls: Type[T]) -> Type[T]:
        original_init_subclass = cls.__init_subclass__

        def _init_subclass(_cls, **kwargs):
            # The default implementation of __init_subclass__ takes no
            # positional arguments, but a custom implementation does.
            # If the user has not reimplemented __init_subclass__ then
            # the first signature will fail and we try the second.
            try:
                original_init_subclass(_cls, **kwargs)
            except TypeError:
                original_init_subclass(**kwargs)

            # Check that each attribute is defined.
            for name in names:
                if not hasattr(_cls, name):
                    raise NotImplementedError(
                        f"{name} has not been defined for {_cls.__name__}"
                    )
                if getattr(_cls, name, NotImplemented) is NotImplemented:
                    raise NotImplementedError(
                        f"dataset_list has not been defined for {_cls.__name__}"
                    )

        cls.__init_subclass__ = classmethod(_init_subclass)
        return cls

    return _func


def safe_div(
    numer: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    return numer / torch.where(
        denom == 0,
        1.0,
        denom,
    )


def move_dict_to_device(batch_data: dict, device) -> dict:
    for k, v in batch_data.items():
        if isinstance(v, torch.Tensor):
            batch_data[k] = v.to(device)
        elif isinstance(v, dict):
            batch_data[k] = move_dict_to_device(v, device)
    return batch_data


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def cumsum_backward(x, dim):
    x_reversed = torch.flip(x, dims=[dim])
    cumsum_rev = torch.cumsum(x_reversed, dim=dim)
    return torch.flip(cumsum_rev, dims=[dim])
