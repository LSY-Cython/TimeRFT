from collections.abc import Callable
from functools import partial
from typing import cast, Union

import numpy as np

Sampler = Callable[[Union[int, np.ndarray]], Union[int, np.ndarray]]


def uniform_sampler(n: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    return np.random.randint(1, n + 1)


def binomial_sampler(n: Union[int, np.ndarray], p: float = 0.5) -> Union[int, np.ndarray]:
    return np.random.binomial(n - 1, p) + 1


def beta_binomial_sampler(
    n: Union[int, np.ndarray], a: float = 1, b: float = 1
) -> Union[int, np.ndarray]:
    # equivalent to uniform_sampler when a = b = 1
    if isinstance(n, np.ndarray):
        p = np.random.beta(a, b, size=n.shape)
    else:
        p = np.random.beta(a, b)
    return np.random.binomial(n - 1, p) + 1


def get_sampler(distribution: str, **kwargs) -> Sampler:
    if distribution == "uniform":
        return uniform_sampler
    elif distribution == "binomial":
        p = kwargs.get("p", 0.5)
        return cast(Sampler, partial(binomial_sampler, p=p))
    elif distribution == "beta_binomial":
        a = kwargs.get("a", 1)
        b = kwargs.get("b", 1)
        return cast(Sampler, partial(beta_binomial_sampler, a=a, b=b))
    else:
        raise NotImplementedError(f"distribution {distribution} not implemented")